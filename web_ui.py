#!/usr/bin/env python3
"""OneBit AI web chat: CUDA, MLX/Metal and CPU share one streaming interface."""
import argparse
import json
import math
import queue
import socket
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from inference_runtime import create_runtime

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OneBit AI - Qwen3-4B LATTICE</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%236366f1'/><text x='50' y='68' font-size='50' font-weight='bold' fill='white' text-anchor='middle'>1B</text></svg>">
<style>
  :root {
    --bg-main: #0c0e14;
    --bg-card: #151821;
    --bg-user: #1f2536;
    --bg-bot: #151821;
    --border: #262b3d;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.25);
    --accent-hover: #4f46e5;
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --sidebar-bg: #11131c;
    --badge-bg: #1e1b4b;
    --badge-text: #a5b4fc;
    --code-bg: #090a0f;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-main); color: var(--text-main); height: 100vh; display: flex; overflow: hidden; }

  /* Sidebar */
  aside {
    width: 290px;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 18px;
    gap: 16px;
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .brand-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: white;
    font-size: 16px;
    box-shadow: 0 0 12px var(--accent-glow);
  }
  .brand-title { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
  .brand-sub { font-size: 11px; color: var(--text-muted); }

  .meta-cards { display: flex; flex-direction: column; gap: 8px; }
  .meta-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
  }
  .meta-card .label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .meta-card .val { color: #fff; font-weight: 600; margin-top: 2px; display: flex; align-items: center; justify-content: space-between; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; }

  .settings-group { display: flex; flex-direction: column; gap: 12px; margin-top: 4px; }
  .setting-label { font-size: 12px; color: var(--text-muted); display: flex; justify-content: space-between; }
  input[type=range] { width: 100%; accent-color: var(--accent); margin-top: 4px; }
  textarea.sys-prompt {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-main);
    padding: 8px;
    font-size: 12px;
    resize: none;
    height: 70px;
    outline: none;
  }
  textarea.sys-prompt:focus { border-color: var(--accent); }

  .btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-main);
    padding: 10px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn:hover { background: var(--border); }
  .btn-new {
    background: var(--accent);
    color: white;
    border: none;
    box-shadow: 0 0 14px var(--accent-glow);
  }
  .btn-new:hover { background: var(--accent-hover); }

  /* Main Chat Area */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    position: relative;
  }

  header {
    height: 56px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    background: rgba(12, 14, 20, 0.85);
    backdrop-filter: blur(8px);
  }
  .header-badges { display: flex; gap: 8px; }
  .badge {
    background: var(--badge-bg);
    color: var(--badge-text);
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(99, 102, 241, 0.3);
  }

  .chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 24px 15% 30px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .msg {
    display: flex;
    gap: 14px;
    max-width: 90%;
    line-height: 1.6;
    animation: fadeIn 0.2s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  .msg.user {
    align-self: flex-end;
    flex-direction: row-reverse;
  }
  .msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    flex-shrink: 0;
  }
  .msg.user .msg-avatar { background: #374151; color: #fff; }
  .msg.bot .msg-avatar { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; }

  .msg-content {
    background: var(--bg-bot);
    border: 1px solid var(--border);
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 14px;
    word-break: break-word;
  }
  .msg.user .msg-content {
    background: var(--bg-user);
    border-color: rgba(99, 102, 241, 0.4);
  }
  .msg-stats {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
    display: flex;
    gap: 12px;
  }

  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 10px 0;
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
  }
  code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  p { margin-bottom: 8px; }
  p:last-child { margin-bottom: 0; }

  /* Input area */
  .input-container {
    padding: 16px 15% 24px;
    background: var(--bg-main);
  }
  .input-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 8px 12px 8px 16px;
    display: flex;
    align-items: flex-end;
    gap: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: border-color 0.2s;
  }
  .input-bar:focus-within { border-color: var(--accent); }
  textarea#prompt-input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-main);
    font-size: 14px;
    resize: none;
    max-height: 180px;
    height: 24px;
    outline: none;
    line-height: 24px;
  }
  .send-btn {
    width: 36px;
    height: 36px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: 0.2s;
    flex-shrink: 0;
  }
  .send-btn:hover { background: var(--accent-hover); }
  .send-btn:disabled { background: #374151; cursor: not-allowed; opacity: 0.6; }

  .stop-btn {
    background: #ef4444;
  }
  .stop-btn:hover { background: #dc2626; }

  /* Welcome card */
  .welcome-card {
    text-align: center;
    margin: auto 0;
    padding: 30px;
  }
  .welcome-card h2 { font-size: 26px; font-weight: 700; margin-bottom: 8px; }
  .welcome-card p { color: var(--text-muted); font-size: 14px; max-width: 480px; margin: 0 auto 24px; }
  .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 560px; margin: 0 auto; }
  .suggestion-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 12px 14px;
    border-radius: 8px;
    text-align: left;
    color: var(--text-main);
    font-size: 13px;
    cursor: pointer;
    transition: 0.2s;
  }
  .suggestion-btn:hover { border-color: var(--accent); background: var(--sidebar-bg); }

  .cursor {
    display: inline-block;
    width: 7px;
    height: 15px;
    background: var(--accent);
    margin-left: 2px;
    vertical-align: middle;
    animation: blink 0.8s infinite;
  }
  @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
</style>
</head>
<body>

<aside>
  <div class="brand">
    <div class="brand-logo">1B</div>
    <div>
      <div class="brand-title">OneBit AI</div>
      <div class="brand-sub">LATTICE Ternary Inference</div>
    </div>
  </div>

  <button class="btn btn-new" onclick="resetChat()">
    <span>+</span> New Conversation
  </button>

  <div class="meta-cards">
    <div class="meta-card">
      <div class="label">Model Architecture</div>
      <div class="val">Qwen3-4B <span class="status-dot"></span></div>
    </div>
    <div class="meta-card">
      <div class="label">Quantization</div>
      <div class="val">LATTICE W1.58 (2-bit packed)</div>
    </div>
    <div class="meta-card">
      <div class="label">Hardware Accelerator</div>
      <div class="val" id="hardware-display">Loading...</div>
    </div>
    <div class="meta-card">
      <div class="label" id="memory-label">Runtime memory</div>
      <div class="val" id="vram-display">Loading...</div>
    </div>
  </div>

  <div class="settings-group">
    <div>
      <div class="setting-label"><span>Temperature</span><span id="temp-val">0.7</span></div>
      <input type="range" id="temp-input" min="0" max="1.5" step="0.05" value="0.7" oninput="document.getElementById('temp-val').innerText=this.value">
    </div>
    <div>
      <div class="setting-label"><span>Max Tokens</span><span id="max-val">128</span></div>
      <input type="range" id="max-input" min="32" max="1024" step="32" value="128" oninput="document.getElementById('max-val').innerText=this.value">
    </div>
    <div>
      <div class="setting-label"><span>System Prompt</span></div>
      <textarea class="sys-prompt" id="sys-prompt">You are a helpful assistant.</textarea>
    </div>
  </div>
</aside>

<main>
  <header>
    <div style="font-weight:600; font-size:14px; color:#fff;">Qwen3-4B-LATTICE_pd0.01</div>
    <div class="header-badges">
      <span class="badge">W1.58A16</span>
      <span class="badge" id="mode-badge">Loading...</span>
      <span class="badge" id="speed-badge" style="background:#064e3b; color:#6ee7b7; border-color:#059669;">Ready</span>
    </div>
  </header>

  <div class="chat-box" id="chat-box">
    <div class="welcome-card" id="welcome-card">
      <h2>Welcome to OneBit AI</h2>
      <p id="welcome-desc">Running packed ternary weights directly in memory.</p>
      <div class="suggestions">
        <button class="suggestion-btn" onclick="sendSuggestion('Explain how ternary lattice quantization works in simple terms.')">
          💡 <strong>Quantization:</strong> How ternary lattice works
        </button>
        <button class="suggestion-btn" onclick="sendSuggestion('Write an optimized Python binary search implementation with comments.')">
          💻 <strong>Code:</strong> Optimized binary search
        </button>
        <button class="suggestion-btn" onclick="sendSuggestion('What are the advantages of 1-bit / 1.58-bit LLMs over 4-bit?')">
          ⚡ <strong>Performance:</strong> 1.58-bit vs 4-bit
        </button>
        <button class="suggestion-btn" onclick="sendSuggestion('Write a short, engaging sci-fi story about an AI running on a pocket device.')">
          📖 <strong>Creative:</strong> Sci-fi pocket AI story
        </button>
      </div>
    </div>
  </div>

  <div class="input-container">
    <div class="input-bar">
      <textarea id="prompt-input" placeholder="Type your message... (Shift+Enter for new line)" rows="1" oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
      <button class="send-btn" id="action-btn" onclick="sendMessage()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
      </button>
    </div>
  </div>
</main>

<script>
let chatHistory = [];
let isGenerating = false;
let currentRequestId = null;
let currentAbortController = null;

async function updateStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ready) {
      const modeBadge = document.getElementById('mode-badge');
      if (modeBadge) modeBadge.innerText = data.quant;
      const speedBadge = document.getElementById('speed-badge');
      if (speedBadge) speedBadge.innerText = 'Ready';
      document.getElementById('hardware-display').innerText = data.hardware;
      document.getElementById('vram-display').innerText = data.vram_gb.toFixed(2) + ' GiB';
      document.getElementById('memory-label').innerText = data.device === 'mlx' ? 'MLX unified memory' : data.device === 'cuda' ? 'Resident VRAM' : 'Process RAM';
      const welcomeP = document.getElementById('welcome-desc');
      if (welcomeP) welcomeP.innerText = 'Running locally with ' + data.quant + '.';
    }
  } catch(e) {}
}
updateStatus();

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isGenerating) sendMessage();
  }
}

function sendSuggestion(text) {
  document.getElementById('prompt-input').value = text;
  autoResize(document.getElementById('prompt-input'));
  sendMessage();
}

function resetChat() {
  if (isGenerating) stopGeneration();
  chatHistory = [];
  const box = document.getElementById('chat-box');
  box.innerHTML = `
    <div class="welcome-card" id="welcome-card">
      <h2>Welcome to OneBit AI</h2>
      <p>Running packed ternary weights directly against custom Triton GEMV kernels in GPU VRAM.</p>
      <div class="suggestions">
        <button class="suggestion-btn" onclick="sendSuggestion('Explain how ternary lattice quantization works in simple terms.')">
          💡 <strong>Quantization:</strong> How ternary lattice works
        </button>
        <button class="suggestion-btn" onclick="sendSuggestion('Write an optimized Python binary search implementation with comments.')">
          💻 <strong>Code:</strong> Optimized binary search
        </button>
        <button class="suggestion-btn" onclick="sendSuggestion('What are the advantages of 1-bit / 1.58-bit LLMs over 4-bit?')">
          ⚡ <strong>Performance:</strong> 1.58-bit vs 4-bit
        </button>
        <button class="suggestion-btn" onclick="sendSuggestion('Write a short, engaging sci-fi story about an AI running on a pocket device.')">
          📖 <strong>Creative:</strong> Sci-fi pocket AI story
        </button>
      </div>
    </div>
  `;
}

function formatMarkdown(text) {
  // Clean markdown parser for code blocks, inline code, bold, italic and paragraphs
  let escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  escaped = escaped.replace(/```([a-z0-9_-]*)\\n([\\s\\S]*?)```/g, function(match, lang, code) {
    return '<pre><code>' + code + '</code></pre>';
  });
  escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");
  escaped = escaped.replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>");
  escaped = escaped.replace(/\\*(.*?)\\*/g, "<em>$1</em>");
  escaped = escaped.replace(/\\n\\n/g, "</p><p>");
  escaped = escaped.replace(/\\n/g, "<br>");
  return '<p>' + escaped + '</p>';
}

async function sendMessage() {
  const input = document.getElementById('prompt-input');
  const text = input.value.trim();
  if (!text || isGenerating) return;

  const welcome = document.getElementById('welcome-card');
  if (welcome) welcome.remove();

  // Add User Message
  appendMessage('user', text);
  chatHistory.push({ role: 'user', content: text });
  input.value = '';
  input.style.height = '24px';

  // Prepare Bot Message
  const botMsgEl = appendMessage('bot', '');
  const contentEl = botMsgEl.querySelector('.msg-content');
  contentEl.innerHTML = '<div style="display:flex; align-items:center; gap:8px; color:#a5b4fc; font-size:13px; padding:4px 0;"><span class="status-dot" style="background:#6366f1; box-shadow:0 0 8px #6366f1;"></span> <em>Processing prompt; output will stream when ready...</em></div>';

  const temp = parseFloat(document.getElementById('temp-input').value) || 0.7;
  const maxTokens = parseInt(document.getElementById('max-input').value) || 128;
  const sysPrompt = document.getElementById('sys-prompt').value || '';

  currentRequestId = 'req_' + Date.now();
  currentAbortController = new AbortController();

  isGenerating = true;
  setButtonToStop();

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: chatHistory,
        system: sysPrompt,
        temperature: temp,
        max_tokens: maxTokens,
        request_id: currentRequestId
      }),
      signal: currentAbortController.signal
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Server returned ${response.status}: ${errText || response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedText = "";
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop(); // keep last incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data: ')) continue;
        const jsonStr = trimmed.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const payload = JSON.parse(jsonStr);
          if (payload.token) {
            if (!accumulatedText) contentEl.innerHTML = '';
            accumulatedText += payload.token;
            contentEl.innerHTML = formatMarkdown(accumulatedText) + '<span class="cursor"></span>';
            scrollBottom();
          }
          if (payload.done) {
            if (payload.error) {
              contentEl.textContent = 'Generation failed: ' + payload.error;
              isGenerating = false;
              setButtonToSend();
              currentAbortController = null;
              return;
            }
            contentEl.innerHTML = formatMarkdown(accumulatedText);
            if (payload.stats) {
              const statsDiv = document.createElement('div');
              statsDiv.className = 'msg-stats';
              statsDiv.innerHTML = `<span>⚡ ${payload.stats.tok_per_sec.toFixed(1)} tok/s</span><span>📊 ${payload.stats.tokens} tokens</span><span>⏱️ ${payload.stats.time.toFixed(1)}s</span>`;
              botMsgEl.appendChild(statsDiv);
              document.getElementById('speed-badge').innerText = payload.stats.tok_per_sec.toFixed(1) + ' tok/s';
              document.getElementById('vram-display').innerText = payload.stats.vram_gb.toFixed(2) + ' GiB';
            }
            chatHistory.push({ role: 'assistant', content: accumulatedText });
            isGenerating = false;
            setButtonToSend();
            currentAbortController = null;
            return;
          }
        } catch (err) {
          console.error("Parse error:", err, jsonStr);
        }
      }
    }

    throw new Error('Generation connection closed before completion. Check the server or watchdog log.');
  } catch (err) {
    if (err.name !== 'AbortError') {
      contentEl.innerHTML = `<span style="color:#ef4444; font-weight: 500;">⚠️ Error: ${err.message}</span>`;
    }
  } finally {
    isGenerating = false;
    currentAbortController = null;
    setButtonToSend();
    scrollBottom();
    const inp = document.getElementById('prompt-input');
    if (inp) inp.focus();
  }
}

function stopGeneration() {
  if (currentRequestId) {
    fetch('/api/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: currentRequestId })
    }).catch(() => {});
  }
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
  isGenerating = false;
  setButtonToSend();
}

function setButtonToStop() {
  const btn = document.getElementById('action-btn');
  btn.className = 'send-btn stop-btn';
  btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg>';
  btn.onclick = stopGeneration;
}

function setButtonToSend() {
  const btn = document.getElementById('action-btn');
  btn.className = 'send-btn';
  btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
  btn.onclick = sendMessage;
}

function appendMessage(role, text) {
  const box = document.getElementById('chat-box');
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  msg.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'U' : '1B'}</div>
    <div class="msg-content">${formatMarkdown(text)}</div>
  `;
  box.appendChild(msg);
  scrollBottom();
  return msg;
}

function scrollBottom() {
  const box = document.getElementById('chat-box');
  box.scrollTop = box.scrollHeight;
}
</script>
</body>
</html>
"""


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime):
        super().__init__(address, WebUIHandler)
        self.runtime = runtime
        self.model_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.requests = {}


class WebUIHandler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _json(self, code, payload):
        self._send(code, json.dumps(payload).encode("utf-8"))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= 1024 * 1024:
            raise ValueError("Request body must be between 1 byte and 1 MiB")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, self.server.runtime.status())
        elif path == "/favicon.ico":
            self._send(204, b"")
        else:
            self._json(404, {"error": "Not found"})

    def do_HEAD(self):
        self._send(200, b"", "text/html; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/chat", "/api/stop"):
            self._json(404, {"error": "Not found"})
            return
        try:
            data = self._read_json()
            request_id = data.get("request_id", "default")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
                raise ValueError("request_id must be a nonempty string of at most 256 characters")
            if path == "/api/stop":
                with self.server.request_lock:
                    event = self.server.requests.get(request_id)
                    if event:
                        event.set()
                self._json(200, {"status": "stopped"})
                return
            messages = data.get("messages")
            system = data.get("system", "You are a helpful assistant.")
            if not isinstance(messages, list) or not messages or not isinstance(system, str):
                raise ValueError("Expected messages and a text system prompt")
            for message in messages:
                if (not isinstance(message, dict) or message.get("role") not in ("user", "assistant")
                        or not isinstance(message.get("content"), str)):
                    raise ValueError("Messages must have user/assistant roles and text content")
            if messages[-1]["role"] != "user":
                raise ValueError("The final message must be from the user")
            temperature = float(data.get("temperature", 0.7))
            max_tokens = int(data.get("max_tokens", 128))
            if not math.isfinite(temperature) or not 0 <= temperature <= 2 or not 1 <= max_tokens <= 1024:
                raise ValueError("temperature must be 0–2 and max_tokens must be 1–1024")
            runtime = self.server.runtime
            prompt = runtime.tokenizer.format(system, messages, runtime.max_context - max_tokens)
            ids = runtime.tokenizer.encode(prompt)
        except (ValueError, TypeError, OverflowError) as exc:
            self._json(400, {"error": str(exc)})
            return

        stop = threading.Event()
        with self.server.request_lock:
            if request_id in self.server.requests:
                self._json(409, {"error": "request_id is already active"})
                return
            self.server.requests[request_id] = stop
        disconnected, stream_done = threading.Event(), threading.Event()
        chunks = queue.Queue(maxsize=256)

        def cancelled():
            return stop.is_set() or disconnected.is_set()

        def publish(payload):
            chunk = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            while not disconnected.is_set():
                try:
                    chunks.put(chunk, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def writer():
            try:
                while not stream_done.is_set() or not chunks.empty():
                    try:
                        chunk = chunks.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                disconnected.set()
                stop.set()

        writer_thread = None
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(10)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            writer_thread = threading.Thread(target=writer, daemon=True)
            writer_thread.start()
            with self.server.model_lock:
                for payload in runtime.stream(ids, max_tokens, temperature, cancelled):
                    publish(payload)
        except Exception as exc:
            if writer_thread:
                publish({"done": True, "error": str(exc)})
        finally:
            stream_done.set()
            if writer_thread:
                writer_thread.join(timeout=11)
            with self.server.request_lock:
                self.server.requests.pop(request_id, None)
            self.close_connection = True


def run_server(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("auto", "cuda", "mlx", "cpu"), default="auto")
    parser.add_argument("--model", help="Packed LATTICE checkpoint (default: model.lat.pt or upstream filename)")
    parser.add_argument("--tokenizer", help="Local Qwen3 tokenizer.json")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    print(f"Loading runtime: {args.runtime}...", flush=True)
    runtime = create_runtime(
        args.runtime, args.model, args.tokenizer, cpu_threads=args.cpu_threads
    )
    server = InferenceServer((args.host, args.port), runtime)
    print(f"{runtime.description} on {runtime.hardware}", flush=True)
    print(f"Web UI ready at http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()

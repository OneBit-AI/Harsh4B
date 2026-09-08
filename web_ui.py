#!/usr/bin/env python3
"""
OneBit AI - Qwen3-4B LATTICE Ternary Web UI Server
Hosts a local, responsive web chat interface powered by the packed ternary Triton kernel.
"""

import os
import sys
import json
import time
import queue
import threading
import socket
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Ensure local kernel modules can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import torch
from build_packed_model import build
from packed_linear import set_kernel
from transformers import AutoTokenizer
try:
    from transformers import StaticCache
except ImportError:
    StaticCache = None

# Configuration
MODEL_ID = "Qwen/Qwen3-4B"
PACKED_PATH = os.path.join(PARENT_DIR, "save_models", "Qwen3-4B-LATTICE_pd0.01.lat.pt")
ROT_PATH = os.path.join(SCRIPT_DIR, "rot_4b_LR.pt")
KERNEL_VERSION = "dense"
HOST = "0.0.0.0"
PORT = 7860

print("=" * 70)
print("  OneBit AI: Loading Qwen3-4B LATTICE (Packed Ternary W1.58)...")
print("=" * 70)

set_kernel(KERNEL_VERSION)
t_start = time.time()
model, _ = build(MODEL_ID, PACKED_PATH, ROT_PATH, device="cuda", verbose=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"[Model Ready in {time.time() - t_start:.1f}s | {vram_gb:.2f} GiB VRAM]")

GRAPH_BUCKETS = [512, 1024, 2048]
persistent_graphs = {}
static_in = None
static_pos = None
_token_decode_cache = {}

def fast_decode_token(token_id):
    s = _token_decode_cache.get(token_id)
    if s is None:
        s = tokenizer.decode([token_id], skip_special_tokens=True)
        _token_decode_cache[token_id] = s
    return s

def sample_next_token(logits, generated_tokens=None, temperature=0.7, top_p=0.9, top_k=50, rep_penalty=1.15):
    l = logits.reshape(1, -1).clone().float()
    if rep_penalty != 1.0 and generated_tokens:
        # High-performance vectorized repetition penalty directly on GPU
        unique_toks = torch.tensor(list(set(generated_tokens)), device=l.device, dtype=torch.long)
        token_logits = l[0, unique_toks]
        l[0, unique_toks] = torch.where(token_logits > 0, token_logits / rep_penalty, token_logits * rep_penalty)

    if temperature <= 0.01:
        return l.argmax(-1, keepdim=True)
    l = l / max(temperature, 1e-5)

    # Top-K (val is already sorted in descending order)
    K = min(top_k if top_k > 0 else 50, l.shape[-1])
    val, idx = torch.topk(l, K, sorted=True)

    probs = torch.softmax(val, dim=-1)
    if top_p < 1.0:
        cum_probs = torch.cumsum(probs, dim=-1)
        mask = (cum_probs - probs) >= top_p
        probs[mask] = 0.0
        p_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(p_sum > 0, probs / p_sum, torch.zeros_like(probs))

    sampled_idx = torch.multinomial(probs, 1)
    return idx.gather(-1, sampled_idx)

def format_chat_prompt(system_prompt, messages, max_prompt_budget=1536):
    """
    Ensures the chat prompt remains safely within the CUDA Graph static cache budget.
    Always preserves system prompt and latest user prompt, sliding conversation history
    to retain the most recent context turns.
    """
    sys_msg = [{"role": "system", "content": system_prompt}]
    if not messages:
        try:
            return tokenizer.apply_chat_template(sys_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(sys_msg, tokenize=False, add_generation_prompt=True)

    latest_msg = [messages[-1]]
    prev_msgs = messages[:-1]

    if len(prev_msgs) <= 2:
        msgs = sys_msg + messages
    else:
        msgs = sys_msg + messages
        try:
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(txt)) > max_prompt_budget:
            kept = []
            for m in reversed(prev_msgs):
                test_msgs = sys_msg + [m] + kept + latest_msg
                try:
                    p_txt = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    p_txt = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
                if len(tokenizer.encode(p_txt)) <= max_prompt_budget:
                    kept.insert(0, m)
                else:
                    break
            msgs = sys_msg + kept + latest_msg

    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

if StaticCache is not None:
    print(f"[CUDA Graph] Pre-capturing multi-bucket decode graphs {GRAPH_BUCKETS}...", flush=True)
    static_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")
    for cap in GRAPH_BUCKETS:
        cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=cap,
                            device="cuda", dtype=torch.float16)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                model(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = model(input_ids=static_in, past_key_values=cache, use_cache=True,
                        cache_position=static_pos).logits
        persistent_graphs[cap] = {"cache": cache, "graph": g, "out": out}
    print(f"[CUDA Graph] Multi-bucket decode graphs {GRAPH_BUCKETS} ready! (Zero startup latency, 35-44 tok/s)", flush=True)

model_lock = threading.Lock()
generation_stop_flags = {}

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
      <div class="val">NVIDIA RTX 5070 (Triton)</div>
    </div>
    <div class="meta-card">
      <div class="label">Resident VRAM</div>
      <div class="val" id="vram-display">""" + f"{vram_gb:.2f} GiB" + """</div>
    </div>
  </div>

  <div class="settings-group">
    <div>
      <div class="setting-label"><span>Temperature</span><span id="temp-val">0.7</span></div>
      <input type="range" id="temp-input" min="0" max="1.5" step="0.05" value="0.7" oninput="document.getElementById('temp-val').innerText=this.value">
    </div>
    <div>
      <div class="setting-label"><span>Max Tokens</span><span id="max-val">512</span></div>
      <input type="range" id="max-input" min="64" max="2048" step="64" value="512" oninput="document.getElementById('max-val').innerText=this.value">
    </div>
    <div>
      <div class="setting-label"><span>System Prompt</span></div>
      <textarea class="sys-prompt" id="sys-prompt">You are a helpful and concise AI assistant.</textarea>
    </div>
  </div>
</aside>

<main>
  <header>
    <div style="font-weight:600; font-size:14px; color:#fff;">Qwen3-4B-LATTICE_pd0.01</div>
    <div class="header-badges">
      <span class="badge">W1.58A16</span>
      <span class="badge">Triton v4_w1</span>
      <span class="badge" style="background:#064e3b; color:#6ee7b7; border-color:#059669;">~20 tok/s</span>
    </div>
  </header>

  <div class="chat-box" id="chat-box">
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
  contentEl.innerHTML = '<span class="cursor"></span>';

  const temp = parseFloat(document.getElementById('temp-input').value) || 0.7;
  const maxTokens = parseInt(document.getElementById('max-input').value) || 512;
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
            accumulatedText += payload.token;
            contentEl.innerHTML = formatMarkdown(accumulatedText) + '<span class="cursor"></span>';
            scrollBottom();
          }
          if (payload.done) {
            contentEl.innerHTML = formatMarkdown(accumulatedText);
            if (payload.stats) {
              const statsDiv = document.createElement('div');
              statsDiv.className = 'msg-stats';
              statsDiv.innerHTML = `<span>⚡ ${payload.stats.tok_per_sec.toFixed(1)} tok/s</span><span>📊 ${payload.stats.tokens} tokens</span><span>⏱️ ${payload.stats.time.toFixed(1)}s</span>`;
              botMsgEl.appendChild(statsDiv);
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

    contentEl.innerHTML = formatMarkdown(accumulatedText);
    if (accumulatedText && (!chatHistory.length || chatHistory[chatHistory.length - 1].content !== accumulatedText)) {
      chatHistory.push({ role: 'assistant', content: accumulatedText });
    }
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


class WebUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[HTTP] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    def do_OPTIONS(self):
        print(f"[HTTP] OPTIONS {self.path} (CORS preflight)", flush=True)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, *")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            content = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(content)
            self.close_connection = True
        elif parsed.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
        elif parsed.path == "/api/status":
            vram = torch.cuda.memory_allocated() / (1024 ** 3)
            payload = json.dumps({
                "model": MODEL_ID,
                "quant": "LATTICE Ternary W1.58 (Dense-T1 + CUDA Graph)",
                "vram_gb": round(vram, 2),
                "ready": True
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/stop":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            req_id = data.get("request_id")
            if req_id:
                generation_stop_flags[req_id] = True
            payload = b'{"status":"stopped"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return

        if parsed.path == "/api/chat":
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))

            messages = data.get("messages", [])
            system_prompt = data.get("system", "You are a helpful assistant.")
            temperature = float(data.get("temperature", 0.7))
            max_tokens = int(data.get("max_tokens", 512))
            req_id = data.get("request_id", "default")
            generation_stop_flags[req_id] = False

            last_user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")[:60]
                    break
            print(f"[HTTP] /api/chat received prompt: \"{last_user_msg}...\" (history len: {len(messages)})", flush=True)

            # Format chat prompt with sliding context window to fit StaticCache budget
            prompt_text = format_chat_prompt(system_prompt, messages, max_prompt_budget=3072)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            token_queue = queue.Queue(maxsize=256)
            stream_done = threading.Event()
            client_disconnected = threading.Event()

            def stream_worker():
                while not stream_done.is_set() or not token_queue.empty():
                    try:
                        chunk = token_queue.get(timeout=0.02)
                        try:
                            self.wfile.write(chunk.encode("utf-8"))
                            if token_queue.empty():
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            client_disconnected.set()
                            break
                        finally:
                            token_queue.task_done()
                    except queue.Empty:
                        pass

            writer_thread = threading.Thread(target=stream_worker, daemon=True)
            writer_thread.start()

            # Execute generation inside lock
            with model_lock:
                input_ids = tokenizer(prompt_text, return_tensors="pt").to("cuda")
                plen = input_ids["input_ids"].shape[1]

                stop_ids = {tokenizer.eos_token_id}
                for tok_s in ["<|im_end|>", "<|endoftext|>"]:
                    try:
                        tid = tokenizer.convert_tokens_to_ids(tok_s)
                        if tid is not None and tid != tokenizer.unk_token_id:
                            stop_ids.add(tid)
                    except Exception:
                        pass

                MAX_CAP = max(GRAPH_BUCKETS)
                available_tokens = MAX_CAP - plen - 4
                actual_max_tokens = min(max_tokens, max(1, available_tokens))
                needed_cap = plen + actual_max_tokens + 4

                chosen_cap = None
                for cap in sorted(GRAPH_BUCKETS):
                    if needed_cap <= cap:
                        chosen_cap = cap
                        break
                if chosen_cap is None:
                    chosen_cap = MAX_CAP

                bucket = persistent_graphs.get(chosen_cap)
                can_use_graph = (bucket is not None and available_tokens >= 16)
                if can_use_graph:
                    persistent_cache = bucket["cache"]
                    persistent_graph = bucket["graph"]
                    static_out = bucket["out"]

                t_gen_start = time.time()
                t_decode_start = 0
                generated_count = 0
                generated_tok_list = []

                with torch.no_grad():
                    if can_use_graph:
                        persistent_cache.reset()
                        outputs = model(**input_ids, past_key_values=persistent_cache, use_cache=True,
                                        cache_position=torch.arange(plen, device="cuda"))
                        logits = outputs.logits[:, -1:]

                        nxt = sample_next_token(logits, generated_tok_list, temperature=temperature)
                        token_id = int(nxt)
                        t_decode_start = time.time()
                        if token_id not in stop_ids:
                            token_str = fast_decode_token(token_id)
                            generated_count += 1
                            generated_tok_list.append(token_id)
                            msg = f"data: {json.dumps({'token': token_str, 'done': False})}\n\n"
                            token_queue.put(msg)

                            cur = nxt.clone()
                            for i in range(actual_max_tokens - 1):
                                if generation_stop_flags.get(req_id, False) or client_disconnected.is_set():
                                    break

                                static_in.copy_(cur)
                                static_pos.fill_(plen + i)
                                persistent_graph.replay()
                                cur_logits = static_out[:, -1:]

                                cur = sample_next_token(cur_logits, generated_tok_list, temperature=temperature)
                                token_id = int(cur)
                                if token_id in stop_ids:
                                    break

                                token_str = fast_decode_token(token_id)
                                generated_count += 1
                                generated_tok_list.append(token_id)

                                msg = f"data: {json.dumps({'token': token_str, 'done': False})}\n\n"
                                token_queue.put(msg)
                    else:
                        outputs = model(**input_ids, use_cache=True)
                        past_key_values = outputs.past_key_values
                        logits = outputs.logits[:, -1:]
                        t_decode_start = time.time()

                        for i in range(max_tokens):
                            if generation_stop_flags.get(req_id, False) or client_disconnected.is_set():
                                break

                            next_token = sample_next_token(logits, generated_tok_list, temperature=temperature)
                            token_id = int(next_token)
                            if token_id in stop_ids:
                                break

                            token_str = fast_decode_token(token_id)
                            generated_count += 1
                            generated_tok_list.append(token_id)

                            msg = f"data: {json.dumps({'token': token_str, 'done': False})}\n\n"
                            token_queue.put(msg)

                            pos = torch.tensor([plen + i], device="cuda")
                            outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True, cache_position=pos)
                            past_key_values = outputs.past_key_values
                            logits = outputs.logits[:, -1:]

                t_total = time.time() - (t_decode_start if t_decode_start > 0 else t_gen_start)
                tok_per_sec = generated_count / max(t_total, 1e-5)
                vram_now = torch.cuda.memory_allocated() / (1024 ** 3)

                done_msg = f"data: {json.dumps({'done': True, 'stats': {'tokens': generated_count, 'time': round(t_total, 2), 'tok_per_sec': round(tok_per_sec, 1), 'vram_gb': round(vram_now, 2)}})}\n\n"
                token_queue.put(done_msg)
                stream_done.set()
                writer_thread.join(timeout=5.0)
                print(f"[HTTP] /api/chat done: {generated_count} tokens in {t_total:.2f}s ({tok_per_sec:.1f} tok/s)", flush=True)

            if req_id in generation_stop_flags:
                del generation_stop_flags[req_id]

            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_WR)
            except Exception:
                pass
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


def run_server():
    server = ThreadingHTTPServer((HOST, PORT), WebUIHandler)
    print(f"\n=======================================================")
    print(f"  OneBit AI Web UI Server running successfully!")
    print(f"  Local / Tailscale URL: http://{HOST}:{PORT}")
    print(f"  From your Mac browser: http://100.87.108.82:{PORT}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server()

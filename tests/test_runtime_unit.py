import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

import inference_runtime as rt
import test as cli
from web_ui import InferenceServer


def test_cli_defaults_to_cpu_and_explains_mlx_switch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["test.py", "--prompt", "hello"])
    args = cli.parse_args()
    assert args.runtime == "cpu"

    cli.print_runtime_banner(args)
    output = capsys.readouterr().out
    assert "RUNTIME: CPU (default)" in output
    assert "--runtime mlx" in output


def test_auto_apple_silicon_does_not_import_torch(monkeypatch):
    monkeypatch.setattr(rt.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(rt.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "mlx" else None)
    assert rt.select_runtime() == "mlx"
    assert rt.select_runtime("cpu") == "cpu"


@pytest.fixture
def tokenizer():
    return rt.ChatTokenizer(rt.ROOT / "tokenizer.json")


def test_chat_format_and_context_budget(tokenizer):
    prompt = tokenizer.format("help", [{"role": "user", "content": "Hi"}], 100)
    assert prompt == "<|im_start|>system\nhelp<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    history = [{"role": "user", "content": "old " * 100}, {"role": "assistant", "content": "answer"},
               {"role": "user", "content": "Hi"}]
    assert tokenizer.format("help", history, 100) == prompt
    with pytest.raises(ValueError, match="exceed"):
        tokenizer.format("help", [{"role": "user", "content": "big " * 300}], 100)


class FakeRuntime(rt.Runtime):
    device = "cpu"
    hardware = "test CPU"
    description = "Test runtime"
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = tokenizer.encode("Hello नमस्ते 世界!")
        self.decode_calls = 0
        self.fail = False
        self.delay = 0
    def prefill(self, ids, max_tokens):
        if self.fail:
            raise RuntimeError("injected generation failure")
        return self.ids[0], {"i": 0}
    def decode(self, token, session):
        self.decode_calls += 1
        time.sleep(self.delay)
        session["i"] += 1
        return self.ids[session["i"] % len(self.ids)]
    def sample(self, logits, generated, temperature):
        return logits


def test_stream_unicode_and_no_unused_decode(tokenizer):
    runtime = FakeRuntime(tokenizer)
    events = list(runtime.stream([1], len(runtime.ids), 0))
    assert "".join(e.get("token", "") for e in events) == tokenizer.decode(runtime.ids)
    assert runtime.decode_calls == len(runtime.ids) - 1
    assert events[-1]["stats"]["tokens"] == len(runtime.ids)


@pytest.fixture
def server(tokenizer):
    runtime = FakeRuntime(tokenizer)
    server = InferenceServer(("127.0.0.1", 0), runtime)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def request(server, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", body,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def test_http_status_stream_validation_and_error_cleanup(server):
    with request(server, "/api/status") as response:
        assert json.load(response)["device"] == "cpu"
    body = dict(messages=[dict(role="user", content="Hi")], max_tokens=4, temperature=0)
    for failing in (False, True):
        server.runtime.fail = failing
        with request(server, "/api/chat", body) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith("data: ")]
        assert events[-1]["done"]
        assert ("error" in events[-1]) == failing
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(server, "/api/chat", dict(body, max_tokens=-1))
    assert exc.value.code == 400
    assert not server.requests


def test_http_stop_cancels_generation(server):
    server.runtime.delay = 0.03
    body = dict(messages=[dict(role="user", content="Hi")], max_tokens=100, request_id="cancel")
    with request(server, "/api/chat", body) as response:
        assert response.readline().startswith(b"data:")
        with request(server, "/api/stop", dict(request_id="cancel")) as stopped:
            assert json.load(stopped)["status"] == "stopped"
        response.read()
    assert server.runtime.decode_calls < 100

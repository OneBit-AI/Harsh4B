"""Lazy CUDA, MLX/Metal and CPU backends for the shared chat server."""
from __future__ import annotations

import importlib.util
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-4B"
GRAPH_BUCKETS = (512, 1024, 2048)


def select_runtime(requested="auto"):
    if requested not in ("auto", "cuda", "mlx", "cpu"):
        raise ValueError(f"Unknown runtime: {requested}")
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx") is not None:
            return "mlx"
    if importlib.util.find_spec("torch") is not None:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


def checkpoint_path(path=None):
    candidates = [Path(path)] if path else [
        ROOT / "model.lat.pt", ROOT / "Qwen3-4B-LATTICE_pd0.01.lat.pt",
        ROOT.parent / "save_models/Qwen3-4B-LATTICE_pd0.01.lat.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"LATTICE checkpoint missing: {candidates[0]}. Use --model PATH.")


class ChatTokenizer:
    """Qwen3's text-only, thinking-disabled chat format without PyTorch imports."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(str(path))
        if self.backend.token_to_id("<|im_end|>") != 151645:
            raise ValueError("Expected a Qwen3 tokenizer.json")
        self.stop_ids = {self.backend.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>")}

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=True)

    def format(self, system, messages, budget):
        def render(history):
            turns = ([{"role": "system", "content": system}] if system else []) + history
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in turns)
            return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        kept = list(messages)
        while True:
            text = render(kept)
            if len(self.encode(text)) <= budget:
                return text
            if len(kept) <= 1:
                raise ValueError("The system and latest message exceed the context budget; shorten them.")
            kept.pop(0)
            # Drop complete old turns, never leave an orphaned assistant reply.
            while len(kept) > 1 and kept[0]["role"] != "user":
                kept.pop(0)


class Runtime:
    max_context = 2048

    def status(self):
        return dict(model=MODEL_ID, device=self.device, runtime=self.device,
                    hardware=self.hardware, quant=self.description, ready=True,
                    vram_gb=round(self.memory_gib(), 2), max_context=self.max_context)

    def memory_gib(self):
        return 0.0

    def stream(self, prompt_ids, max_tokens, temperature, stopped=lambda: False):
        """One fresh session per request; no unused forward after the last token."""
        from tokenizers.decoders import DecodeStream
        if not prompt_ids or len(prompt_ids) + max_tokens > self.max_context:
            raise ValueError("Prompt and output exceed the runtime context budget")
        generated = []
        decoder = DecodeStream(skip_special_tokens=True)
        prefill_seconds = decode_seconds = 0.0
        decode_steps = 0
        if not stopped():
            t0 = time.perf_counter()
            logits, session = self.prefill(prompt_ids, max_tokens)
            token = self.sample(logits, generated, temperature)
            prefill_seconds = time.perf_counter() - t0
            while not stopped() and token not in self.tokenizer.stop_ids:
                generated.append(token)
                text = decoder.step(self.tokenizer.backend, token)
                if text:
                    yield {"token": text, "done": False}
                if len(generated) >= max_tokens or stopped():
                    break
                t0 = time.perf_counter()
                logits = self.decode(token, session)
                token = self.sample(logits, generated, temperature)
                decode_seconds += time.perf_counter() - t0
                decode_steps += 1
        yield {"done": True, "stats": {
            "tokens": len(generated), "time": round(prefill_seconds + decode_seconds, 3),
            "prefill_time": round(prefill_seconds, 3), "decode_time": round(decode_seconds, 3),
            "decode_steps": decode_steps,
            "tok_per_sec": round(decode_steps / decode_seconds, 2) if decode_seconds else 0.0,
            "vram_gb": round(self.memory_gib(), 2), "runtime": self.device,
        }}


def sample_mlx(mx, logits, generated, temperature=0.7, top_p=0.9, top_k=50, rep_penalty=1.15):
    values = logits.reshape(-1).astype(mx.float32)
    if generated and rep_penalty != 1.0:
        ids = mx.array(sorted(set(generated)), dtype=mx.int32)
        repeated = values[ids]
        values[ids] = mx.where(repeated > 0, repeated / rep_penalty, repeated * rep_penalty)
    if temperature <= 0.01:
        return int(mx.argmax(values))
    values = values / temperature
    k = min(top_k if top_k > 0 else 50, values.size)
    ids = mx.argpartition(-values, kth=k-1)[:k]
    ids = ids[mx.argsort(-values[ids])]
    selected = values[ids]
    probs = mx.softmax(selected)
    if top_p < 1.0:
        selected = mx.where(mx.cumsum(probs) - probs >= top_p, -mx.inf, selected)
    return int(ids[mx.random.categorical(selected)])


class MLXRuntime(Runtime):
    device = "mlx"
    description = "MLX / Metal · packed LATTICE W1.58"

    def __init__(self, path, tokenizer, **_):
        import mlx.core as mx
        from MLX.kernels import LatticeLinear
        from MLX.lattice_model import Qwen3Lattice, load_checkpoint
        if not mx.metal.is_available():
            raise RuntimeError("MLX/Metal requires an Apple Silicon Mac; use --runtime cpu.")
        self.mx, self.tokenizer = mx, tokenizer
        self.hardware = mx.device_info().get("device_name", "Apple Silicon GPU")
        mx.set_cache_limit(64 * 2**20)
        state, archive = load_checkpoint(path)
        try:
            self.model = Qwen3Lattice(mx, LatticeLinear, state)
        finally:
            archive.close()

    def prefill(self, ids, max_tokens):
        from MLX.lattice_model import KVCache
        cache = KVCache(self.mx, len(self.model.layers))
        # Bound prompt activations and release each chunk before the next.
        for start in range(0, len(ids), 128):
            logits = self.model(self.mx.array([ids[start:start+128]]), cache, start, last_only=True)
            self.mx.eval(logits)
        return logits, {"cache": cache, "position": len(ids)}

    def decode(self, token, session):
        logits = self.model(self.mx.array([[token]]), session["cache"], session["position"], last_only=True)
        session["position"] += 1
        return logits

    def sample(self, logits, generated, temperature):
        return sample_mlx(self.mx, logits, generated, temperature)

    def memory_gib(self):
        return self.mx.get_active_memory() / 2**30


def sample_torch(torch, logits, generated, temperature=0.7, top_p=0.9, top_k=50, rep_penalty=1.15):
    values = logits.reshape(1, -1).clone().float()
    if generated and rep_penalty != 1.0:
        ids = torch.tensor(sorted(set(generated)), device=values.device, dtype=torch.long)
        repeated = values[0, ids]
        values[0, ids] = torch.where(repeated > 0, repeated / rep_penalty, repeated * rep_penalty)
    if temperature <= 0.01:
        return int(values.argmax(-1).item())
    values /= temperature
    k = min(top_k if top_k > 0 else 50, values.shape[-1])
    selected, ids = torch.topk(values, k, sorted=True)
    probs = torch.softmax(selected, dim=-1)
    if top_p < 1.0:
        probs[(torch.cumsum(probs, dim=-1) - probs) >= top_p] = 0
        probs /= probs.sum(dim=-1, keepdim=True)
    return int(ids.gather(-1, torch.multinomial(probs, 1)).item())


class TorchRuntime(Runtime):
    def __init__(self, path, tokenizer, device="cpu", cpu_cache=None, cpu_threads=4, cpu_dtype="bfloat16", cpu_layout="auto"):
        import torch
        self.torch, self.tokenizer, self.device = torch, tokenizer, device
        self.graphs = {}
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; choose --runtime mlx or cpu.")
            try:
                from build_packed_model import build
                from packed_linear import set_kernel
            except ImportError as exc:
                raise RuntimeError("CUDA needs the upstream build_packed_model, gemv_triton and gemm_triton modules on PYTHONPATH.") from exc
            set_kernel("dense")
            self.model, _ = build(MODEL_ID, str(path), str(ROOT / "rot_4b_LR.pt"), device="cuda", verbose=True)
            self.hardware = torch.cuda.get_device_name()
            self.description = "CUDA · Triton Dense-T1 / bucketed graphs"
            self._capture_graphs()
        else:
            packed = cpu_layout == "packed" or (cpu_layout == "auto" and platform.system() == "Darwin" and not cpu_cache)
            if packed:
                from packed_cpu import build_packed_cpu
                self.model = build_packed_cpu(path, dtype=getattr(torch, cpu_dtype), threads=cpu_threads)
            else:
                from build_cpu_model import build_cpu
                self.model = build_cpu(packed_path=str(path), rot_path=str(ROOT / "rot_4b_LR.pt"),
                                       cache_path=cpu_cache, dtype=getattr(torch, cpu_dtype), threads=cpu_threads)
            self.hardware = f"{platform.machine()} CPU ({cpu_threads} threads)"
            self.description = f"CPU · PyTorch {cpu_dtype} / " + ("packed LATTICE" if packed else "absorbed KOTMS")

    def _capture_graphs(self):
        torch = self.torch
        try:
            from transformers import StaticCache
        except ImportError:
            return
        self.static_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")
        for cap in GRAPH_BUCKETS:
            cache = StaticCache(config=self.model.config, max_batch_size=1, max_cache_len=cap,
                                device="cuda", dtype=torch.float16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(stream):
                for _ in range(3):
                    self.model(input_ids=self.static_in, past_key_values=cache, use_cache=True, cache_position=self.static_pos)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out = self.model(input_ids=self.static_in, past_key_values=cache, use_cache=True, cache_position=self.static_pos).logits
            self.graphs[cap] = dict(cache=cache, graph=graph, out=out)

    def prefill(self, ids, max_tokens):
        torch = self.torch
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        bucket = next((self.graphs[cap] for cap in sorted(self.graphs) if len(ids) + max_tokens <= cap), None)
        with torch.inference_mode():
            if bucket:
                bucket["cache"].reset()
                out = self.model(input_ids=inputs, past_key_values=bucket["cache"], use_cache=True,
                                 cache_position=torch.arange(len(ids), device=self.device))
            else:
                past = None
                chunk = 8 if getattr(self.model, "_lattice_packed_cpu", False) else len(ids)
                for start in range(0, len(ids), chunk):
                    out = self.model(input_ids=inputs[:, start:start+chunk], past_key_values=past,
                                     use_cache=True, logits_to_keep=1)
                    past = out.past_key_values
        return out.logits[:, -1:], dict(cache=out.past_key_values, position=len(ids), bucket=bucket)

    def decode(self, token, session):
        torch = self.torch
        bucket = session["bucket"]
        with torch.inference_mode():
            if bucket:
                self.static_in.fill_(token)
                self.static_pos.fill_(session["position"])
                bucket["graph"].replay()
                logits = bucket["out"][:, -1:]
            else:
                out = self.model(input_ids=torch.tensor([[token]], dtype=torch.long, device=self.device),
                                 past_key_values=session["cache"], use_cache=True, logits_to_keep=1)
                session["cache"] = out.past_key_values
                logits = out.logits[:, -1:]
        session["position"] += 1
        return logits

    def sample(self, logits, generated, temperature):
        return sample_torch(self.torch, logits, generated, temperature)

    def memory_gib(self):
        if self.device == "cuda":
            return self.torch.cuda.memory_allocated() / 2**30
        import psutil
        return psutil.Process().memory_info().rss / 2**30


def create_runtime(name="auto", model_path=None, tokenizer_path=None, **options):
    selected = select_runtime(name)
    tokenizer_path = Path(tokenizer_path) if tokenizer_path else ROOT / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}; use --tokenizer PATH.")
    tokenizer = ChatTokenizer(tokenizer_path)
    cls = MLXRuntime if selected == "mlx" else TorchRuntime
    return cls(checkpoint_path(model_path), tokenizer, device=selected, **options)

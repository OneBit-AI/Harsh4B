#!/usr/bin/env python3
"""Run and benchmark model.lat.pt with LiteSpark/CPU or MLX/Metal.

The MLX path reads the PyTorch checkpoint's tensor storages directly, so
PyTorch is not a dependency for that path.  ``model.lat.pt`` is the only
model file opened.

Examples:
    python test.py --prompt "Explain ternary quantization."  # CPU is the default
    python test.py --token-ids 151643,198 --max-new-tokens 64
    .venv-mlx/bin/python test.py --runtime mlx --prompt "Explain ternary quantization."
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.lat.pt"
TOKENIZER_PATH = HERE / "tokenizer.json"
CPU_CACHE_PATH = None  # Source/dtype-specific, memory-mapped cache chosen by build_cpu.
CPU_ROT_PATH = HERE / "rot_4b_LR.pt"


from MLX.lattice_model import (
    StorageRef, TensorRef, load_checkpoint, unpack_codes, pack_codes, expand_t1,
    compact_t1_row_starts, rms_norm, attention, mlx_fp16, rope_factors,
    apply_rope, KVCache, LatticeProjection, Qwen3Lattice,
)

def parse_token_ids(value: str) -> list[int]:
    try:
        tokens = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--token-ids must be comma-separated integers") from exc
    if not tokens or any(token < 0 or token >= 151_936 for token in tokens):
        raise argparse.ArgumentTypeError("--token-ids must contain Qwen3 vocabulary IDs")
    return tokens


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--token-ids", type=parse_token_ids,
                       help="Comma-separated Qwen3 tokenizer IDs; requires no tokenizer file")
    group.add_argument("--prompt", help="Text prompt, encoded with tokenizer.json")
    parser.add_argument("--raw-prompt", action="store_true",
                        help="Treat --prompt as a raw completion instead of a Qwen chat turn")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH,
                        help="Qwen3 tokenizer.json (default: %(default)s)")
    parser.add_argument("--runtime", choices=("mlx", "cpu"), default="cpu",
                        help="Inference runtime: LiteSpark CPU (default) or MLX/Metal")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature; 0 selects greedy decoding (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed for repeatable benchmark output (default: %(default)s)")
    parser.add_argument("--profile-token", action="store_true",
                        help="Profile one warmed MLX decode token by operation category")
    parser.add_argument("--compact-t1", action="store_true",
                        help="Use checkpoint-native compact T1 to reduce memory at lower decode speed")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cpu-layout", choices=("auto", "packed", "absorbed"), default="auto",
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def print_runtime_banner(args):
    """Make the active backend and the command for switching unmistakable."""
    print("=" * 72, flush=True)
    if args.runtime == "cpu":
        print("RUNTIME: CPU (default) — LiteSpark packed int4 SIMD", flush=True)
        print("MLX/Metal is NOT used in this run.", flush=True)
        print("To switch to MLX:", flush=True)
        print("  .venv-mlx/bin/python test.py --runtime mlx --prompt \"your prompt\"", flush=True)
    else:
        print("RUNTIME: MLX — Apple Metal acceleration", flush=True)
        print("To switch back to the default CPU runtime:", flush=True)
        print("  python test.py --runtime cpu --prompt \"your prompt\"", flush=True)
    print("=" * 72, flush=True)


def input_tokens(args):
    if args.token_ids is not None:
        return args.token_ids, None
    if not args.tokenizer.is_file():
        raise SystemExit(f"Missing Qwen3 tokenizer: {args.tokenizer}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() > 151_936:
        raise SystemExit(f"{args.tokenizer} has more entries than Qwen3-4B's 151936 output IDs")
    text = args.prompt
    if not args.raw_prompt:
        text = ("<|im_start|>system\nYou are a helpful assistant. Answer directly in one sentence of at most 30 words.<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tokenizer.encode(text, add_special_tokens=False).ids, tokenizer


def generation_stop_ids(tokenizer):
    if tokenizer is None:
        return {151_643, 151_645}
    return {token_id for token_id in (
        tokenizer.token_to_id("<|endoftext|>"),
        tokenizer.token_to_id("<|im_end|>"),
    ) if token_id is not None}


def sample_torch(torch, logits, generated, args):
    values = logits.reshape(1, -1).clone().float()
    if generated and args.repetition_penalty != 1.0:
        ids = torch.tensor(sorted(set(generated)), device=values.device, dtype=torch.long)
        repeated = values[0, ids]
        values[0, ids] = torch.where(
            repeated > 0, repeated / args.repetition_penalty,
            repeated * args.repetition_penalty,
        )
    if args.temperature <= 0.01:
        return int(values.argmax(-1).item())
    values /= args.temperature
    selected, ids = torch.topk(values, min(args.top_k, values.shape[-1]), sorted=True)
    probabilities = torch.softmax(selected, dim=-1)
    if args.top_p < 1.0:
        probabilities[(torch.cumsum(probabilities, dim=-1) - probabilities) >= args.top_p] = 0
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
    return int(ids.gather(-1, torch.multinomial(probabilities, 1)).item())


def sample_mlx(mx, logits, generated, args):
    values = logits.reshape(-1).astype(mx.float32)
    if generated and args.repetition_penalty != 1.0:
        ids = mx.array(sorted(set(generated)), dtype=mx.int32)
        repeated = values[ids]
        values[ids] = mx.where(
            repeated > 0, repeated / args.repetition_penalty,
            repeated * args.repetition_penalty,
        )
    if args.temperature <= 0.01:
        return int(mx.argmax(values))
    values /= args.temperature
    count = min(args.top_k, values.size)
    ids = mx.argpartition(-values, kth=count - 1)[:count]
    ids = ids[mx.argsort(-values[ids])]
    selected = values[ids]
    probabilities = mx.softmax(selected)
    if args.top_p < 1.0:
        selected = mx.where(mx.cumsum(probabilities) - probabilities >= args.top_p, -mx.inf, selected)
    return int(ids[mx.random.categorical(selected)])


def sample_numpy(logits, generated, args, generator):
    values = np.array(logits, dtype=np.float32, copy=True).reshape(-1)
    if generated and args.repetition_penalty != 1.0:
        ids = np.array(sorted(set(generated)), dtype=np.int64)
        repeated = values[ids]
        values[ids] = np.where(
            repeated > 0, repeated / args.repetition_penalty,
            repeated * args.repetition_penalty,
        )
    if args.temperature <= 0.01:
        return int(values.argmax())
    values /= np.float32(args.temperature)
    count = min(args.top_k, values.size)
    ids = np.argpartition(-values, count - 1)[:count]
    ids = ids[np.argsort(-values[ids])]
    selected = values[ids]
    selected -= selected.max()
    probabilities = np.exp(selected, dtype=np.float32)
    probabilities /= probabilities.sum(dtype=np.float32)
    if args.top_p < 1.0:
        probabilities[(np.cumsum(probabilities) - probabilities) >= args.top_p] = 0
        probabilities /= probabilities.sum(dtype=np.float32)
    return int(generator.choice(ids, p=probabilities))


def profile_mlx_token(mx, model, cache, token, position):
    """Replay one token's real inputs to time packed GEMV and attention.

    The recording pass materializes realistic inputs without charging its many
    synchronization points to the result. Each category is then replayed with
    one synchronization. Dense/reference methods are fail-fast guarded.
    """
    projection_jobs = []
    attention_jobs = []
    projection_count = 0

    def materialize(operation):
        result = operation()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        return result

    def forbidden_dense_path(*_args, **_kwargs):
        raise RuntimeError("decode attempted separate dequantization/dense matmul")

    guarded = []
    for layer in model.layers:
        for name in ("q", "k", "v", "o", "gate", "up", "down"):
            linear = layer[name].linear
            guarded.append((linear, linear.reference, linear.dense_weight))
            linear.reference = forbidden_dense_path
            linear.dense_weight = forbidden_dense_path

    def project(projection, value):
        nonlocal projection_count
        shape = value.shape
        rotated = materialize(
            lambda: (
                projection.left
                @ value.reshape(-1, projection.left.shape[0], projection.right.shape[0]).astype(mx.float16)
                @ projection.right
            ).reshape(-1, projection.in_features),
        )
        if rotated.size != projection.in_features:
            raise RuntimeError("decode projection is not a single activation vector")
        projection_jobs.append((projection.linear, rotated))
        result = materialize(lambda: projection.linear(rotated))
        projection_count += 1
        return result.reshape(*shape[:-1], projection.out_features)

    try:
        x = materialize(lambda: model.embed[mx.array([[token]], dtype=mx.int32)])
        rope_cos, rope_sin = materialize(lambda: rope_factors(mx, mx.array([position]), x.dtype))
        for layer_index, layer in enumerate(model.layers):
            residual = x
            x = materialize(lambda: rms_norm(mx, x, layer["input_norm"]))
            q = project(layer["q"], x).reshape(1, 1, 32, 128).transpose(0, 2, 1, 3)
            k = project(layer["k"], x).reshape(1, 1, 8, 128).transpose(0, 2, 1, 3)
            v = project(layer["v"], x).reshape(1, 1, 8, 128).transpose(0, 2, 1, 3)
            q = materialize(lambda: rms_norm(mx, q, layer["q_norm"]))
            k = materialize(lambda: rms_norm(mx, k, layer["k_norm"]))
            q = materialize(lambda: apply_rope(mx, q, rope_cos, rope_sin))
            k = materialize(lambda: apply_rope(mx, k, rope_cos, rope_sin))
            k, v = materialize(lambda: cache.append(layer_index, k, v))
            attention_jobs.append((q, k, v))
            attended = materialize(lambda: attention(mx, q, k, v))
            attended = attended.transpose(0, 2, 1, 3).reshape(1, 1, layer["o"].in_features)
            projected = project(layer["o"], attended)
            x = materialize(lambda: residual + projected)
            residual = x
            normalized = materialize(lambda: rms_norm(mx, x, layer["post_norm"]))
            gate = project(layer["gate"], normalized)
            up = project(layer["up"], normalized)
            activated = materialize(lambda: (gate * mx.sigmoid(gate)) * up)
            down = project(layer["down"], activated)
            x = materialize(lambda: residual + down)
        materialize(lambda: rms_norm(mx, x, model.final_norm) @ model.lm_head.T)

        started = time.perf_counter()
        projection_outputs = [linear(value) for linear, value in projection_jobs]
        mx.eval(*projection_outputs)
        gemv_seconds = time.perf_counter() - started

        started = time.perf_counter()
        attention_outputs = [attention(mx, q, k, v) for q, k, v in attention_jobs]
        mx.eval(*attention_outputs)
        attention_seconds = time.perf_counter() - started
    finally:
        for linear, reference, dense_weight in guarded:
            linear.reference = reference
            linear.dense_weight = dense_weight

    if projection_count != 252:
        raise RuntimeError(f"expected 252 packed decode projections, observed {projection_count}")
    return gemv_seconds, attention_seconds, projection_count


def run_cpu(args, tokens, tokenizer):
    """Run generation through the torch-free LiteSpark CPU model."""
    try:
        from cpu_lightspark import build_litespark_cpu
    except ImportError as exc:
        raise SystemExit(
            "The CPU runtime requires litespark-inference; install the CPU dependencies first."
        ) from exc

    print("[CPU] Loading this repository's LATTICE model with LiteSpark...", flush=True)
    started = time.perf_counter()
    print(f"[CPU] Layout: lattice-unpacked int4; threads={args.cpu_threads}", flush=True)
    model = build_litespark_cpu(MODEL_PATH, threads=args.cpu_threads)
    print(f"[CPU] Model ready in {time.perf_counter() - started:.1f}s", flush=True)

    generated, decode_times = [], []
    stop_ids = generation_stop_ids(tokenizer)
    from tokenizers.decoders import DecodeStream
    decoder = DecodeStream(skip_special_tokens=True) if tokenizer else None
    generator = np.random.default_rng(args.seed)
    state = model.new_state(len(tokens) + args.max_new_tokens)
    print(f"[CPU] Prefilling {len(tokens)} prompt tokens; generation has started...", flush=True)
    started = time.perf_counter()
    for token_id in tokens:
        logits = model.forward_token(token_id, state)
    prefill_seconds = time.perf_counter() - started
    token = sample_numpy(logits, generated, args, generator)
    print(f"[CPU] First token ready in {prefill_seconds:.3f}s\n\n--- OUTPUT (streaming) ---", flush=True)

    for _ in range(args.max_new_tokens):
        if token in stop_ids:
            break
        generated.append(token)
        piece = decoder.step(tokenizer, token) if decoder else f"{token},"
        if piece:
            print(piece, end="", flush=True)
        if len(generated) >= args.max_new_tokens:
            break
        started = time.perf_counter()
        logits = model.forward_token(token, state)
        decode_times.append(time.perf_counter() - started)
        token = sample_numpy(logits, generated, args, generator)

    measured = decode_times[min(args.warmup_tokens, len(decode_times)):]
    print("\n--------------")
    print(f"model: {MODEL_PATH.name}")
    print("runtime: cpu (LiteSpark packed int4 SIMD)")
    print(f"stop: {'end-of-turn token' if token in stop_ids else f'{args.max_new_tokens}-token limit (output may be truncated)'}")
    print(f"prefill: {len(tokens)} tokens in {prefill_seconds:.3f}s ({len(tokens) / prefill_seconds:.2f} tok/s)")
    if measured:
        print(f"decode: {len(measured) / sum(measured):.2f} tok/s")
    else:
        print("decode: no non-EOS tokens were available to benchmark")


def main():
    args = parse_args()
    if (args.max_new_tokens < 1 or args.warmup_tokens < 0 or args.cpu_threads < 1
            or not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1
            or args.top_k < 1 or not math.isfinite(args.repetition_penalty)
            or args.repetition_penalty < 1):
        raise SystemExit(
            "token/thread counts must be positive, warmup must be nonnegative, "
            "temperature must be 0-2, top-p must be in (0,1], and repetition-penalty must be >=1"
        )
    print_runtime_banner(args)
    if args.profile_token and args.runtime != "mlx":
        raise SystemExit("--profile-token currently profiles the MLX/Metal decode path")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Missing required checkpoint: {MODEL_PATH}")
    tokens, tokenizer = input_tokens(args)
    if args.runtime == "cpu":
        run_cpu(args, tokens, tokenizer)
        return

    cpu_started = time.perf_counter()
    state, archive = load_checkpoint()
    print(f"CPU checkpoint map ready in {time.perf_counter() - cpu_started:.2f}s", flush=True)
    try:
        try:
            import mlx.core as mx
            from MLX.kernels import CompactLatticeLinear, LatticeLinear
        except ImportError as exc:
            raise SystemExit("Run with the MLX environment: .venv-mlx/bin/python test.py ...") from exc

        # Model construction creates short-lived cast/concatenation buffers.
        # Keep MLX from retaining those buffers after CPU-to-device transfer.
        mx.set_cache_limit(64 * 2**20)
        print(f"Loading {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 2**30:.2f} GiB) into MLX...", flush=True)
        started = time.perf_counter()
        linear_type = CompactLatticeLinear if args.compact_t1 else LatticeLinear
        model = Qwen3Lattice(mx, linear_type, state)
        mx.eval(model.embed, model.lm_head)
    finally:
        # All mapped tensors have been copied into MLX arrays by Qwen3Lattice.
        archive.close()
    print(f"Ready in {time.perf_counter() - started:.1f}s", flush=True)

    cache = KVCache(mx, 36)
    prompt = mx.array(tokens, dtype=mx.int32)[None]
    started = time.perf_counter()
    logits = model(prompt, cache, 0)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - started
    generated, decode_times = [], []
    stop_ids = generation_stop_ids(tokenizer)
    mx.random.seed(args.seed)
    token = sample_mlx(mx, logits[0, -1], generated, args)
    if args.profile_token:
        # Compile and warm every single-token kernel first.
        generated.append(token)
        logits = model(mx.array([[token]], dtype=mx.int32), cache, len(tokens))
        mx.eval(logits)
        token = sample_mlx(mx, logits[0, -1], generated, args)

        # Time one ordinary production decode without internal barriers.
        generated.append(token)
        started = time.perf_counter()
        logits = model(mx.array([[token]], dtype=mx.int32), cache, len(tokens) + 1)
        mx.eval(logits)
        model_seconds = time.perf_counter() - started
        started = time.perf_counter()
        token = sample_mlx(mx, logits[0, -1], generated, args)
        sampling_seconds = time.perf_counter() - started
        generated.append(token)

        gemv_seconds, attention_seconds, projection_count = profile_mlx_token(
            mx, model, cache, token, len(tokens) + 2
        )
        total = model_seconds + sampling_seconds
        timings = {
            "attention": attention_seconds,
            "packed linear / GEMV": gemv_seconds,
            "dequant/unpack": 0.0,
            "sampling": sampling_seconds,
            "everything else": total - gemv_seconds - attention_seconds - sampling_seconds,
        }
        print("\n--- ONE GENERATED TOKEN ---")
        for category, seconds in timings.items():
            percent = 100 * seconds / total if total else 0.0
            print(f"{category}: {seconds * 1000:.3f} ms ({percent:.1f}%)")
        print(f"total: {total * 1000:.3f} ms ({1 / total:.2f} tok/s)")
        print(f"decode path: {projection_count}/252 packed custom Metal GEMVs")
        print("dense/reference decode calls: 0 (fail-fast guard verified)")
        return
    for _ in range(args.max_new_tokens):
        if token in stop_ids:
            break
        generated.append(token)
        if len(generated) >= args.max_new_tokens:
            break
        started = time.perf_counter()
        logits = model(mx.array([[token]], dtype=mx.int32), cache, len(tokens) + len(generated) - 1)
        mx.eval(logits)
        decode_times.append(time.perf_counter() - started)
        token = sample_mlx(mx, logits[0, -1], generated, args)

    measured = decode_times[min(args.warmup_tokens, len(decode_times)):]
    print("\n--- OUTPUT ---")
    print(tokenizer.decode(generated) if tokenizer else ",".join(map(str, generated)))
    print("--------------")
    print(f"model: {MODEL_PATH.name}")
    print("runtime: mlx (Metal)")
    print(f"stop: {'end-of-turn token' if token in stop_ids else f'{args.max_new_tokens}-token limit (output may be truncated)'}")
    print(f"prefill: {len(tokens)} tokens in {prefill_seconds:.3f}s ({len(tokens) / prefill_seconds:.2f} tok/s)")
    if measured:
        print(f"decode: {len(measured) / sum(measured):.2f} tok/s")
    else:
        print("decode: no non-EOS tokens were available to benchmark")
    print(f"peak MLX memory: {mx.get_peak_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

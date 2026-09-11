#!/usr/bin/env python3
"""Run and benchmark model.lat.pt with MLX/Metal.

The checkpoint is a PyTorch zip archive, but this script reads its tensor
storages directly; PyTorch is not a runtime dependency.  ``model.lat.pt`` is
the only model file opened.

Examples:
    .venv-mlx/bin/python test.py --token-ids 151643,198 --max-new-tokens 64
    .venv-mlx/bin/python test.py --prompt "Explain ternary quantization."
"""
from __future__ import annotations

import argparse
import collections
import pickle
import statistics
import struct
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.lat.pt"
TOKENIZER_PATH = HERE / "tokenizer.json"


@dataclass(frozen=True)
class StorageRef:
    archive: zipfile.ZipFile
    prefix: str
    key: str
    dtype: np.dtype
    numel: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def numpy(self) -> np.ndarray:
        """Map one raw PyTorch storage entry without importing PyTorch."""
        if self.offset < 0 or any(s < 0 for s in self.shape):
            raise ValueError("invalid negative tensor offset or shape")
        member = f"{self.storage.prefix}/data/{self.storage.key}"
        info = self.storage.archive.getinfo(member)
        if info.compress_type == zipfile.ZIP_STORED and self.storage.archive.filename:
            # torch.save stores tensor payloads as uncompressed ZIP members.
            # Map them in place so a large storage (the 1.45 GiB embedding) is
            # not duplicated into an anonymous Python bytes allocation.
            with open(self.storage.archive.filename, "rb") as checkpoint:
                checkpoint.seek(info.header_offset)
                header = checkpoint.read(30)
            fields = struct.unpack("<IHHHHHIIIHH", header)
            if fields[0] != 0x04034B50:
                raise ValueError("invalid ZIP local-file header")
            data_offset = info.header_offset + 30 + fields[-2] + fields[-1]
            data = np.memmap(
                self.storage.archive.filename, mode="r", dtype=self.storage.dtype,
                offset=data_offset, shape=(self.storage.numel,),
            )
        else:
            raw = self.storage.archive.read(member)
            data = np.frombuffer(raw, dtype=self.storage.dtype, count=self.storage.numel)
        required = self.offset
        for size, stride in zip(self.shape, self.stride):
            if size:
                required += (size - 1) * stride
        if required >= data.size:
            raise ValueError("tensor view exceeds its checkpoint storage")
        return np.ndarray(
            shape=self.shape,
            dtype=self.storage.dtype,
            buffer=data,
            offset=self.offset * self.storage.dtype.itemsize,
            strides=tuple(s * self.storage.dtype.itemsize for s in self.stride),
        )


class _TorchArchiveUnpickler(pickle.Unpickler):
    """Restricted reader for standard torch.save tensor dictionaries."""
    _storage_types = {
        "ByteStorage": np.dtype(np.uint8),
        "HalfStorage": np.dtype(np.float16),
        "FloatStorage": np.dtype(np.float32),
    }

    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module, name):
        if module == "torch" and name in self._storage_types:
            return self._storage_types[name]
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}")

    @staticmethod
    def _rebuild_tensor(storage, offset, shape, stride, _requires_grad, _hooks):
        return TensorRef(storage, int(offset), tuple(shape), tuple(stride))

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent ID: {pid!r}")
        _tag, dtype, key, location, numel = pid
        if location != "cpu" or not isinstance(dtype, np.dtype):
            raise pickle.UnpicklingError("only CPU Byte/Half/Float tensor storages are supported")
        return StorageRef(self.archive, self.prefix, str(key), dtype, int(numel))


def load_checkpoint():
    """Return the mapped LATTICE state and its open zip archive.

    The caller must retain the archive until every TensorRef has been converted.
    """
    archive = zipfile.ZipFile(MODEL_PATH)
    try:
        data_name = next(info.filename for info in archive.infolist() if info.filename.endswith("/data.pkl"))
        prefix = data_name.removesuffix("/data.pkl")
        with archive.open(data_name) as data:
            state = _TorchArchiveUnpickler(data, archive, prefix).load()
    except Exception:
        archive.close()
        raise
    if not isinstance(state, dict) or state.get("__latmeta__", {}).get("n_linears") != 252:
        archive.close()
        raise ValueError("model.lat.pt is not the expected Qwen3-4B LATTICE checkpoint")
    return state, archive


def unpack_codes(codes: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(0, 8, 2, dtype=np.uint8)
    return ((codes[..., None] >> shifts) & 3).reshape(codes.shape[0], -1)[:, :width]


def pack_codes(codes: np.ndarray) -> np.ndarray:
    pad = (-codes.shape[1]) % 4
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)))
    return (codes[:, 0::4] | (codes[:, 1::4] << 2) |
            (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)).astype(np.uint8, copy=False)


def expand_t1(packed: dict) -> np.ndarray:
    """Restore compact global T1 codes to LATTICE's dense per-row 2-bit layout."""
    rows, width = map(int, packed["shape"])
    maskid = packed["maskid"].numpy()
    modes = unpack_codes(maskid, width)
    second_order = modes < 2
    offsets = np.cumsum(second_order.reshape(-1), dtype=np.int64) - 1
    if int(second_order.sum()) != int(packed["n_T1"]):
        raise ValueError("T1 count does not match mask IDs")
    source = packed["T1c"].numpy()
    flat = np.zeros(rows * width, dtype=np.uint8)
    selected = offsets[second_order.reshape(-1)]
    flat[second_order.reshape(-1)] = (source[selected >> 2] >> ((selected & 3) * 2)) & 3
    return pack_codes(flat.reshape(rows, width))


def rms_norm(mx, x, weight, eps=1e-6):
    x32 = x.astype(mx.float32)
    return (x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps) * weight).astype(x.dtype)


def mlx_fp16(mx, tensor: TensorRef):
    """Cast on the host so MLX never allocates a transient FP32 device copy."""
    return mx.array(tensor.numpy().astype(np.float16, copy=False))


def rope_factors(mx, positions, dtype, head_dim=128, theta=1_000_000.0):
    """Build the Qwen3 RoPE factors once per decode step, not per layer/head."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = positions[:, None].astype(mx.float32) * inv_freq[None, :]
    angles = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(angles)[None, None].astype(dtype), mx.sin(angles)[None, None].astype(dtype)


def apply_rope(mx, x, cos, sin):
    """Apply precomputed factors to [batch, heads, tokens, head_dim]."""
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


class KVCache:
    def __init__(self, mx, layers):
        self.mx = mx
        self.entries = [None] * layers

    def append(self, index, key, value):
        old = self.entries[index]
        self.entries[index] = (key, value) if old is None else (
            self.mx.concatenate([old[0], key], axis=2),
            self.mx.concatenate([old[1], value], axis=2),
        )
        return self.entries[index]


class LatticeProjection:
    def __init__(self, mx, lattice_linear, packed, left: TensorRef, right: TensorRef):
        self.mx = mx
        self.out_features, self.in_features = map(int, packed["shape"])
        self.left = mlx_fp16(mx, left)
        self.right = mlx_fp16(mx, right)
        self.linear = lattice_linear(
            mx.array(packed["maskid"].numpy()),
            mx.array(packed["T0"].numpy()),
            mx.array(expand_t1(packed)),
            mx.array(packed["mu"].numpy()),
            mx.array(packed["a0"].numpy()),
            mx.array(packed["a1"].numpy()),
            self.in_features,
            int(packed["blocksize"]),
        )
        # Force host-to-MLX copies before the checkpoint zip is closed.
        mx.eval(self.left, self.right, self.linear.maskid, self.linear.t0,
                self.linear.t1, self.linear.scales, self.linear.bias)

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1, self.left.shape[0], self.right.shape[0]).astype(self.mx.float16)
        x = (self.left @ x @ self.right).reshape(-1, self.in_features)
        return self.linear(x).reshape(*shape[:-1], self.out_features)


class Qwen3Lattice:
    def __init__(self, mx, lattice_linear, state):
        self.mx = mx
        self.layers = []
        embed_ref = state["model.embed_tokens.weight"]
        lm_head_ref = state["lm_head.weight"]
        self.embed = mlx_fp16(mx, embed_ref)
        self.final_norm = mlx_fp16(mx, state["model.norm.weight"])
        # Qwen3 ties these tensors to the same checkpoint storage. Preserve that
        # alias instead of copying the 742 MiB matrix into unified memory twice.
        tied_head = (
            embed_ref.storage.key == lm_head_ref.storage.key
            and embed_ref.offset == lm_head_ref.offset
            and embed_ref.shape == lm_head_ref.shape
            and embed_ref.stride == lm_head_ref.stride
        )
        self.lm_head = self.embed if tied_head else mlx_fp16(mx, lm_head_ref)
        for index in range(36):
            prefix = f"model.layers.{index}"
            attn = f"{prefix}.self_attn"
            mlp = f"{prefix}.mlp"
            project = lambda path: LatticeProjection(
                mx, lattice_linear, state[f"{path}.weight"], state[f"{path}.L"], state[f"{path}.R"])
            self.layers.append({
                "q": project(f"{attn}.q_proj"), "k": project(f"{attn}.k_proj"),
                "v": project(f"{attn}.v_proj"), "o": project(f"{attn}.o_proj"),
                "gate": project(f"{mlp}.gate_proj"), "up": project(f"{mlp}.up_proj"),
                "down": project(f"{mlp}.down_proj"),
                "input_norm": mlx_fp16(mx, state[f"{prefix}.input_layernorm.weight"]),
                "post_norm": mlx_fp16(mx, state[f"{prefix}.post_attention_layernorm.weight"]),
                "q_norm": mlx_fp16(mx, state[f"{attn}.q_norm.weight"]),
                "k_norm": mlx_fp16(mx, state[f"{attn}.k_norm.weight"]),
            })
        mx.eval(
            self.embed, self.final_norm, self.lm_head,
            *[value for layer in self.layers for key, value in layer.items() if not callable(value)],
        )

    def __call__(self, token_ids, cache, start_pos):
        mx = self.mx
        x = self.embed[token_ids]
        q_len = x.shape[1]
        positions = mx.arange(start_pos, start_pos + q_len)
        rope_cos, rope_sin = rope_factors(mx, positions, x.dtype)
        for layer_index, layer in enumerate(self.layers):
            residual = x
            x = rms_norm(mx, x, layer["input_norm"])
            q = layer["q"](x).reshape(1, q_len, 32, 128).transpose(0, 2, 1, 3)
            k = layer["k"](x).reshape(1, q_len, 8, 128).transpose(0, 2, 1, 3)
            v = layer["v"](x).reshape(1, q_len, 8, 128).transpose(0, 2, 1, 3)
            q = rms_norm(mx, q, layer["q_norm"])
            k = rms_norm(mx, k, layer["k_norm"])
            q = apply_rope(mx, q, rope_cos, rope_sin)
            k = apply_rope(mx, k, rope_cos, rope_sin)
            k, v = cache.append(layer_index, k, v)
            # Group four query heads per KV head and broadcast the KV tensors.
            # This is equivalent to repeat(..., 4, axis=1) without materializing
            # four copies of every layer's growing cache.
            grouped_q = q.reshape(1, 8, 4, q_len, 128)
            scores = (grouped_q @ k[:, :, None].transpose(0, 1, 2, 4, 3)) / np.sqrt(128)
            if q_len > 1:
                total = k.shape[2]
                mask = mx.triu(mx.full((q_len, total), -mx.inf), k=1 + total - q_len)
                scores = scores + mask
            attention = mx.softmax(scores.astype(mx.float32), axis=-1).astype(mx.float16) @ v[:, :, None]
            attention = attention.transpose(0, 3, 1, 2, 4).reshape(1, q_len, layer["o"].in_features)
            x = residual + layer["o"](attention)
            residual = x
            normalized = rms_norm(mx, x, layer["post_norm"])
            gate = layer["gate"](normalized)
            x = (gate * mx.sigmoid(gate)) * layer["up"](normalized)
            x = residual + layer["down"](x)
            # Materialize one layer at a time during prefill so its reconstructed
            # LATTICE matrix cannot remain live through the next layer.
            if q_len > 1:
                mx.eval(x)
        return rms_norm(mx, x, self.final_norm) @ self.lm_head.T


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
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH,
                        help="Qwen3 tokenizer.json (default: %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    return parser.parse_args()


def input_tokens(args):
    if args.token_ids is not None:
        return args.token_ids, None
    if not args.tokenizer.is_file():
        raise SystemExit(f"Missing Qwen3 tokenizer: {args.tokenizer}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() > 151_936:
        raise SystemExit(f"{args.tokenizer} has more entries than Qwen3-4B's 151936 output IDs")
    return tokenizer.encode(args.prompt).ids, tokenizer


def main():
    args = parse_args()
    if args.max_new_tokens < 1 or args.warmup_tokens < 0:
        raise SystemExit("--max-new-tokens must be positive and --warmup-tokens cannot be negative")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Missing required checkpoint: {MODEL_PATH}")
    try:
        import mlx.core as mx
        from MLX.kernels import LatticeLinear
    except ImportError as exc:
        raise SystemExit("Run with the MLX environment: .venv-mlx/bin/python test.py ...") from exc

    # Model construction creates short-lived cast/concatenation buffers. Keep
    # MLX from retaining gigabytes of those buffers while the 4B model loads.
    mx.set_cache_limit(64 * 2**20)

    tokens, tokenizer = input_tokens(args)
    state, archive = load_checkpoint()
    try:
        print(f"Loading {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 2**30:.2f} GiB) into MLX...", flush=True)
        started = time.perf_counter()
        model = Qwen3Lattice(mx, LatticeLinear, state)
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
    token = int(mx.argmax(logits[0, -1]))
    for _ in range(args.max_new_tokens):
        if token == 151_645:
            break
        generated.append(token)
        started = time.perf_counter()
        logits = model(mx.array([[token]], dtype=mx.int32), cache, len(tokens) + len(generated) - 1)
        mx.eval(logits)
        decode_times.append(time.perf_counter() - started)
        token = int(mx.argmax(logits[0, -1]))

    measured = decode_times[min(args.warmup_tokens, len(decode_times)):]
    print("\n--- OUTPUT ---")
    print(tokenizer.decode(generated) if tokenizer else ",".join(map(str, generated)))
    print("--------------")
    print(f"model: {MODEL_PATH.name}")
    print(f"prefill: {len(tokens)} tokens in {prefill_seconds:.3f}s ({len(tokens) / prefill_seconds:.2f} tok/s)")
    if measured:
        median = statistics.median(measured)
        print(f"decode: {len(measured)} measured tokens | {1 / median:.2f} tok/s median | "
              f"{statistics.mean(measured):.3f}s/token mean")
    else:
        print("decode: no non-EOS tokens were available to benchmark")
    print(f"peak MLX memory: {mx.get_peak_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

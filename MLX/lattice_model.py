"""Shared Qwen3 LATTICE checkpoint reader and MLX model (no GPU work on import)."""
from __future__ import annotations

import collections
import mmap
import os
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
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
            # numpy.memmap keeps the checkpoint FD open for the lifetime of
            # the array.  A packed model has hundreds of tensor views, which
            # can exhaust macOS's relatively small per-process FD limit.  Map
            # an aligned region ourselves and close the FD immediately; the
            # mmap object remains reachable through the ndarray's buffer.
            granularity = mmap.ALLOCATIONGRANULARITY
            map_offset = (data_offset // granularity) * granularity
            delta = data_offset - map_offset
            map_length = delta + self.storage.numel * self.storage.dtype.itemsize
            descriptor = os.open(self.storage.archive.filename, os.O_RDONLY)
            try:
                mapped = mmap.mmap(
                    descriptor, map_length, access=mmap.ACCESS_READ,
                    offset=map_offset,
                )
            finally:
                os.close(descriptor)
            data = np.ndarray(
                shape=(self.storage.numel,), dtype=self.storage.dtype,
                buffer=mapped, offset=delta,
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


def load_checkpoint(path=MODEL_PATH):
    """Return the mapped LATTICE state and its open zip archive.

    The caller must retain the archive until every TensorRef has been converted.
    """
    archive = zipfile.ZipFile(path)
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
    source = packed["T1c"].numpy()
    dense = np.empty((rows, (width + 3) // 4), dtype=np.uint8)
    # A whole-projection int64 prefix sum alone took ~190 MiB for each
    # Qwen MLP matrix, plus several equally sized indexing temporaries.
    # Bound scratch space independently of matrix size and retain the global
    # compact offset across chunks (row boundaries need not be byte-aligned).
    chunk_rows = max(1, 65536 // width)
    offset = 0
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        second_order = (unpack_codes(maskid[start:stop], width) < 2).reshape(-1)
        count = int(second_order.sum())
        indices = np.arange(offset, offset + count, dtype=np.int64)
        if offset + count > int(packed["n_T1"]) or (offset + count + 3) // 4 > source.size:
            raise ValueError("T1 count does not match mask IDs")
        flat = np.zeros(second_order.size, dtype=np.uint8)
        flat[second_order] = (source[indices >> 2] >> ((indices & 3) * 2)) & 3
        dense[start:stop] = pack_codes(flat.reshape(stop - start, width))
        offset += count
    if offset != int(packed["n_T1"]):
        raise ValueError("T1 count does not match mask IDs")
    return dense


def compact_t1_row_starts(maskid: np.ndarray, expected_codes: int) -> np.ndarray:
    """Compute each row's T1c offset without expanding any weight codes."""
    values = np.arange(256, dtype=np.uint8)
    counts = sum(((values >> shift) & 3) < 2 for shift in range(0, 8, 2)).astype(np.uint8)
    per_row = counts[maskid].sum(axis=1, dtype=np.uint64)
    starts = np.empty(maskid.shape[0], dtype=np.uint32)
    starts[0] = 0
    if starts.size > 1:
        starts[1:] = np.cumsum(per_row[:-1], dtype=np.uint64).astype(np.uint32)
    if int(per_row.sum(dtype=np.uint64)) != int(expected_codes):
        raise ValueError("compact T1 count does not match mask IDs")
    return starts


def rms_norm(mx, x, weight, eps=1e-6):
    return mx.fast.rms_norm(x, weight, eps)


def attention(mx, q, k, v):
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5,
        mask="causal" if q.shape[2] > 1 else None,
    )


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
    def __init__(self, mx, layers, step=256):
        self.mx = mx
        self.entries = [None] * layers
        self.lengths = [0] * layers
        self.step = step

    def append(self, index, key, value):
        old = self.entries[index]
        start = self.lengths[index]
        end = start + key.shape[2]
        if old is None or end > old[0].shape[2]:
            capacity = ((end + self.step - 1) // self.step) * self.step
            previous = 0 if old is None else old[0].shape[2]
            shape = (*key.shape[:2], capacity - previous, key.shape[3])
            extra = (self.mx.zeros(shape, dtype=key.dtype), self.mx.zeros(shape, dtype=value.dtype))
            old = extra if old is None else tuple(self.mx.concatenate([a, b], axis=2) for a, b in zip(old, extra))
        # MLX's update primitive can reuse materialized buffers between decode
        # steps. Grow by capacity buckets, not a pair of full copies per token.
        old[0][:, :, start:end] = key
        old[1][:, :, start:end] = value
        self.entries[index] = old
        self.lengths[index] = end
        return old[0][:, :, :end], old[1][:, :, :end]


class LatticeProjection:
    def __init__(self, mx, lattice_linear, packed, left: TensorRef, right: TensorRef):
        self.mx = mx
        self.out_features, self.in_features = map(int, packed["shape"])
        self.left = mlx_fp16(mx, left)
        self.right = mlx_fp16(mx, right)
        maskid = packed["maskid"].numpy()
        common = (
            mx.array(maskid), mx.array(packed["T0"].numpy()),
            mx.array(packed["mu"].numpy()), mx.array(packed["a0"].numpy()),
            mx.array(packed["a1"].numpy()), self.in_features, int(packed["blocksize"]),
        )
        if getattr(lattice_linear, "compact_t1", False):
            row_starts = compact_t1_row_starts(maskid, int(packed["n_T1"]))
            self.linear = lattice_linear(
                common[0], common[1], mx.array(packed["T1c"].numpy()), mx.array(row_starts),
                *common[2:],
            )
            code_buffers = (self.linear.t1_compact, self.linear.row_starts)
        else:
            self.linear = lattice_linear(common[0], common[1], mx.array(expand_t1(packed)), *common[2:])
            code_buffers = (self.linear.t1,)
        # Force host-to-MLX copies before the checkpoint zip is closed.
        mx.eval(self.left, self.right, self.linear.maskid, self.linear.t0,
                *code_buffers, self.linear.scales, self.linear.bias)

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

    def __call__(self, token_ids, cache, start_pos, last_only=False):
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
            attended = attention(mx, q, k, v)
            attended = attended.transpose(0, 2, 1, 3).reshape(1, q_len, layer["o"].in_features)
            x = residual + layer["o"](attended)
            residual = x
            normalized = rms_norm(mx, x, layer["post_norm"])
            gate = layer["gate"](normalized)
            x = (gate * mx.sigmoid(gate)) * layer["up"](normalized)
            x = residual + layer["down"](x)
            # Materialize one layer at a time during prefill so its reconstructed
            # LATTICE matrix cannot remain live through the next layer.
            if q_len > 1:
                mx.eval(x)
        if last_only:
            x = x[:, -1:]
        return rms_norm(mx, x, self.final_norm) @ self.lm_head.T

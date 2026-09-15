#!/usr/bin/env python3
"""Run this repository's Qwen3-4B LATTICE checkpoint with LiteSpark CPU kernels.

LATTICE's affine weights are reconstructed once into row-scaled signed int4,
packed two per byte, and cached. Decode dispatches LiteSpark's native int4
NEON/AVX GEMV directly. KOTMS rotations are absorbed during conversion, so
neither lattice unpacking nor rotation is repeated for every token.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.lat.pt"
TOKENIZER_PATH = ROOT / "tokenizer.json"
CONFIG_PATH = ROOT / "configs/qwen3-4b.json"
CACHE_VERSION = 2
DEFAULT_SYSTEM = "You are a helpful assistant. Answer directly in one sentence of at most 30 words."
DEFAULT_CPU_THREADS = 6
DEFAULT_GEMV_KERNEL = "prefetch512"


def configure_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")


@lru_cache(None)
def _native_kernel_library():
    """Build/load the local Apple-ARM SDOT experiment library."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    source = ROOT / "cpu_kernels/litespark_int4.cpp"
    compiler = shutil.which("clang++")
    libomp = Path("/opt/homebrew/opt/libomp")
    if compiler is None or not libomp.exists():
        return None
    digest = hashlib.sha256(source.read_bytes() + platform.machine().encode()).hexdigest()[:16]
    output = ROOT / ".cache/cpu-kernels" / f"litespark-int4-{digest}.dylib"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".{os.getpid()}.tmp.dylib")
        subprocess.run([
            compiler, "-O3", "-ffast-math", "-std=c++17", "-dynamiclib",
            "-mcpu=native", "-Xpreprocessor", "-fopenmp",
            f"-I{libomp / 'include'}", f"-L{libomp / 'lib'}",
            f"-Wl,-rpath,{libomp / 'lib'}", "-lomp",
            str(source), "-o", str(temporary),
        ], check=True)
        os.replace(temporary, output)
    return ctypes.CDLL(str(output))


def _native_kernel_function(symbol: str):
    library = _native_kernel_library()
    if library is None:
        return None
    function = getattr(library, symbol)
    function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_float, ctypes.c_void_p]
    function.argtypes += [ctypes.c_int, ctypes.c_int]
    function.restype = None
    return function


@lru_cache(None)
def _native_int4_kernel(variant=DEFAULT_GEMV_KERNEL):
    """Packed signed-int4 x int8 SDOT GEMV."""
    symbols = {
        "none": "litespark_int4_i8_gemv_nopf",
        "prefetch256": "litespark_int4_i8_gemv_pf256",
        "prefetch512": "litespark_int4_i8_gemv_pf512",
        "prefetch1024": "litespark_int4_i8_gemv",
    }
    if variant not in symbols:
        raise ValueError(f"unknown int4 GEMV kernel variant: {variant}")
    return _native_kernel_function(symbols[variant])


@lru_cache(None)
def _native_int8_kernel():
    """Expanded signed-int8 x int8 SDOT GEMV used for the paper-inspired A/B."""
    return _native_kernel_function("litespark_int8_i8_gemv")


@lru_cache(None)
def _native_attention_kernel():
    """Fused single-token grouped-query attention for Apple ARM64."""
    library = _native_kernel_library()
    if library is None:
        return None
    function = library.litespark_gqa_decode
    function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5
    function.restype = None
    return function


def _array(value) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _pack_ternary(weights: np.ndarray) -> np.ndarray:
    """LiteSpark's {-1,0,+1}->{0,1,2}, four-values-per-byte layout."""
    if weights.dtype != np.int8 or weights.ndim != 2 or weights.shape[1] % 4:
        raise ValueError("ternary matrix must be int8 with width divisible by four")
    unsigned = weights.view(np.uint8)
    packed = np.empty((weights.shape[0], weights.shape[1] // 4), dtype=np.uint8)
    temporary = np.empty_like(packed)
    np.add(unsigned[:, 0::4], np.uint8(1), out=packed, casting="unsafe")
    for shift, column in ((2, 1), (4, 2), (6, 3)):
        np.add(unsigned[:, column::4], np.uint8(1), out=temporary, casting="unsafe")
        temporary <<= shift
        packed |= temporary
    return packed


def _pack_int4(weights: np.ndarray) -> np.ndarray:
    """Pack signed [-7,+7] values as low/high nibbles for LiteSpark."""
    if weights.dtype != np.int8 or weights.ndim != 2 or weights.shape[1] % 2:
        raise ValueError("int4 matrix must be int8 with even width")
    if np.any(weights < -7) or np.any(weights > 7):
        raise ValueError("int4 values must be in [-7,+7]")
    return ((weights[:, 0::2].view(np.uint8) & 15)
            | ((weights[:, 1::2].view(np.uint8) & 15) << 4))


def _convert_projection(packed: dict, left_ref, right_ref, weights_path: Path, scale_path: Path) -> None:
    """Reconstruct, rotate, row-int4-quantize, and atomically cache a projection."""
    from MLX.lattice_model import expand_t1, unpack_codes

    rows, width = map(int, packed["shape"])
    blocksize = int(packed["blocksize"])
    if width % 2:
        raise ValueError("LiteSpark int4 projections require even width")
    maskid = _array(packed["maskid"])
    t0_packed = _array(packed["T0"])
    t1_packed = expand_t1(packed)
    mu = _array(packed["mu"])
    a0 = _array(packed["a0"])
    a1 = _array(packed["a1"])
    modes_o2 = tuple(int(x) for x in _array(packed["o2_idx"]).reshape(-1))
    left = _array(left_ref).astype(np.float32, copy=False)
    right = _array(right_ref).astype(np.float32, copy=False)
    dim_l, dim_r = left.shape[0], right.shape[0]
    if left.shape != (dim_l, dim_l) or right.shape != (dim_r, dim_r) or dim_l * dim_r != width:
        raise ValueError("KOTMS rotation dimensions do not match projection width")

    temporary_weights = weights_path.with_suffix(".npy.tmp")
    temporary_scales = scale_path.with_suffix(".npy.tmp")
    packed_out = np.lib.format.open_memmap(
        temporary_weights, mode="w+", dtype=np.uint8, shape=(rows, width // 2)
    )
    scales_out = np.lib.format.open_memmap(
        temporary_scales, mode="w+", dtype=np.float32, shape=(rows,)
    )
    blocks = np.arange(width, dtype=np.int64)[None, :] // blocksize
    trit = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    chunk_rows = max(8, min(64, 262144 // width))
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        mode = unpack_codes(maskid[start:stop], width).astype(np.int64, copy=False)
        code0 = unpack_codes(t0_packed[start:stop], width)
        code1 = unpack_codes(t1_packed[start:stop], width)
        row_index = np.arange(start, stop, dtype=np.int64)[:, None]
        weight = mu[mode, row_index, blocks].astype(np.float32, copy=False)
        weight = weight + a0[mode, row_index, blocks] * trit[code0]
        for scale_index, mode_id in enumerate(modes_o2):
            weight += (mode == mode_id) * a1[scale_index, row_index, blocks] * trit[code1]
        weight = (left.T @ weight.reshape(-1, dim_l, dim_r) @ right.T).reshape(stop - start, width)
        row_scale = np.maximum(np.max(np.abs(weight), axis=1), np.float32(1e-8)) / 7.0
        int4 = np.rint(weight / row_scale[:, None]).clip(-7, 7).astype(np.int8)
        packed_out[start:stop] = _pack_int4(int4)
        scales_out[start:stop] = row_scale

    packed_out.flush()
    scales_out.flush()
    del packed_out, scales_out
    os.replace(temporary_weights, weights_path)
    os.replace(temporary_scales, scale_path)


def _convert_embedding(source: np.ndarray, weights_path: Path, scale_path: Path) -> None:
    """Per-row symmetric int4 embedding/head conversion used by LiteSpark."""
    rows, width = source.shape
    if width % 2:
        raise ValueError("int4 embedding width must be even")
    temporary_weights = weights_path.with_suffix(".npy.tmp")
    temporary_scales = scale_path.with_suffix(".npy.tmp")
    packed = np.lib.format.open_memmap(
        temporary_weights, mode="w+", dtype=np.uint8, shape=(rows, width // 2)
    )
    scales = np.lib.format.open_memmap(
        temporary_scales, mode="w+", dtype=np.float32, shape=(rows,)
    )
    for start in range(0, rows, 2048):
        stop = min(rows, start + 2048)
        block = source[start:stop].astype(np.float32, copy=False)
        scale = np.maximum(np.max(np.abs(block), axis=1), np.float32(1e-8)) / 7.0
        quantized = np.rint(block / scale[:, None]).clip(-7, 7).astype(np.int8)
        packed[start:stop] = ((quantized[:, 0::2].view(np.uint8) & 15)
                              | ((quantized[:, 1::2].view(np.uint8) & 15) << 4))
        scales[start:stop] = scale
    packed.flush()
    scales.flush()
    del packed, scales
    os.replace(temporary_weights, weights_path)
    os.replace(temporary_scales, scale_path)


def _cache_identity(model_path: Path) -> tuple[str, dict]:
    stat = model_path.stat()
    identity = {
        "version": CACHE_VERSION,
        "source": str(model_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "format": "lattice-to-row-int4-litespark",
        "embedding": "symmetric-int4-per-row",
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return digest, identity


@dataclass(frozen=True)
class LiteSparkProjection:
    """One row-scaled matrix in the selected int4 or int8 experiment layout."""

    weights: np.ndarray
    scales: np.ndarray
    layout: str = "int4"

    @property
    def input_size(self) -> int:
        return self.weights.shape[1] * (2 if self.layout == "int4" else 1)

    @property
    def output_size(self) -> int:
        return self.weights.shape[0]


@dataclass(frozen=True)
class QwenLayer:
    qkv: LiteSparkProjection
    o: LiteSparkProjection
    gate_up: LiteSparkProjection
    down: LiteSparkProjection
    input_norm: np.ndarray
    post_norm: np.ndarray
    q_norm: np.ndarray
    k_norm: np.ndarray


class QwenDecodeState:
    """Caller-owned KV cache and scratch buffers; nothing allocates per GEMV."""

    def __init__(self, model: "LiteSparkQwen", capacity: int):
        if capacity < 1:
            raise ValueError("decode capacity must be positive")
        c = model.config
        layers = int(c["num_hidden_layers"])
        hidden = int(c["hidden_size"])
        intermediate = int(c["intermediate_size"])
        heads = int(c["num_attention_heads"])
        kv_heads = int(c["num_key_value_heads"])
        head_dim = int(c["head_dim"])
        self.capacity = capacity
        self.position = 0
        self.keys = np.empty((layers, capacity, kv_heads, head_dim), dtype=np.float32)
        self.values = np.empty_like(self.keys)
        self.x = np.empty(hidden, dtype=np.float32)
        self.normalized = np.empty(hidden, dtype=np.float32)
        self.quantized = np.empty(intermediate, dtype=np.int8)
        q_size = heads * head_dim
        kv_size = kv_heads * head_dim
        self.qkv = np.empty(q_size + 2 * kv_size, dtype=np.float32)
        self.q = self.qkv[:q_size].reshape(heads, head_dim)
        self.k = self.qkv[q_size:q_size + kv_size].reshape(kv_heads, head_dim)
        self.v = self.qkv[q_size + kv_size:].reshape(kv_heads, head_dim)
        self.attended = np.empty(heads * head_dim, dtype=np.float32)
        self.projected = np.empty(hidden, dtype=np.float32)
        self.gate_up = np.empty(intermediate * 2, dtype=np.float32)
        self.gate = self.gate_up[:intermediate]
        self.up = self.gate_up[intermediate:]
        self.activated = np.empty(intermediate, dtype=np.float32)
        groups = heads // kv_heads
        self.scores = np.empty((kv_heads, groups, capacity), dtype=np.float32)
        self.logits = np.empty(int(c["vocab_size"]), dtype=np.float32)


class LiteSparkQwen:
    """Torch-free Qwen3 decode runtime for this repository's LATTICE model."""

    def __init__(
        self, config, layers, embedding, embedding_scales, final_norm, kernel,
        gemv_kernel=DEFAULT_GEMV_KERNEL,
    ):
        self.config = config
        self.layers = layers
        self.embedding = embedding
        self.embedding_scales = embedding_scales
        self.lm_head = LiteSparkProjection(embedding, embedding_scales)
        self.final_norm = final_norm
        self.kernel = kernel
        self.int4_kernel = _native_int4_kernel(gemv_kernel)
        self.int8_kernel = _native_int8_kernel()
        self.attention_kernel = _native_attention_kernel()
        head_dim = int(config["head_dim"])
        positions = np.arange(int(config["max_position_embeddings"]), dtype=np.float32)[:, None]
        inv_freq = np.float32(1.0) / np.float32(config["rope_theta"]) ** (
            np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
        )
        angles = positions * inv_freq[None, :]
        self.rope_cos = np.cos(angles).astype(np.float32)
        self.rope_sin = np.sin(angles).astype(np.float32)

    def new_state(self, capacity: int) -> QwenDecodeState:
        return QwenDecodeState(self, capacity)

    def _embedding_into(self, token_id: int, out: np.ndarray) -> None:
        if token_id < 0 or token_id >= self.embedding.shape[0]:
            raise ValueError(f"token ID {token_id} is outside the vocabulary")
        packed = self.embedding[token_id]
        low = (packed.astype(np.int8) << 4) >> 4
        high = packed.astype(np.int8) >> 4
        out[0::2] = low
        out[1::2] = high
        out *= self.embedding_scales[token_id]

    def _projection(self, projection, values, quantized, activation_scale, out) -> None:
        if projection.layout == "int8":
            if self.int8_kernel is None:
                raise RuntimeError("expanded-int8 GEMV requires Apple ARM64 SDOT")
            self.int8_kernel(
                quantized.ctypes.data, projection.weights.ctypes.data,
                projection.scales.ctypes.data, activation_scale, out.ctypes.data,
                projection.output_size, projection.input_size,
            )
            return
        if self.int4_kernel is None:
            self.kernel.lm_head_int4(
                projection.weights, projection.scales, values, out,
                projection.input_size,
            )
            return
        self.int4_kernel(
            quantized.ctypes.data, projection.weights.ctypes.data,
            projection.scales.ctypes.data, activation_scale, out.ctypes.data,
            projection.output_size, projection.input_size,
        )

    def _qk_norm(self, values: np.ndarray, gamma: np.ndarray) -> None:
        squared_norm = np.einsum("ij,ij->i", values, values, dtype=np.float32)
        squared_norm /= np.float32(values.shape[1])
        squared_norm += np.float32(self.config["rms_norm_eps"])
        np.sqrt(squared_norm, out=squared_norm)
        np.divide(values, squared_norm[:, None], out=values)
        np.multiply(values, gamma[None, :], out=values)

    def _rope(self, values: np.ndarray, position: int) -> None:
        half = values.shape[1] // 2
        first = values[:, :half].copy()
        cosine = self.rope_cos[position]
        sine = self.rope_sin[position]
        values[:, :half] = first * cosine - values[:, half:] * sine
        values[:, half:] = values[:, half:] * cosine + first * sine

    def _attention(self, state: QwenDecodeState, layer_index: int) -> None:
        c = self.config
        kv_heads = int(c["num_key_value_heads"])
        groups = int(c["num_attention_heads"]) // kv_heads
        end = state.position + 1
        if self.attention_kernel is not None:
            self.attention_kernel(
                state.q.ctypes.data,
                state.keys[layer_index].ctypes.data,
                state.values[layer_index].ctypes.data,
                state.scores.ctypes.data,
                state.attended.ctypes.data,
                end,
                kv_heads,
                groups,
                int(c["head_dim"]),
                state.capacity,
            )
            return
        q = state.q.reshape(kv_heads, groups, int(c["head_dim"]))
        scores = state.scores[:, :, :end]
        np.einsum(
            "hgd,thd->hgt", q, state.keys[layer_index, :end],
            out=scores, dtype=np.float32, optimize=True,
        )
        scores *= np.float32(int(c["head_dim"]) ** -0.5)
        scores -= np.max(scores, axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=-1, keepdims=True, dtype=np.float32)
        np.einsum(
            "hgt,thd->hgd", scores, state.values[layer_index, :end],
            out=state.attended.reshape(kv_heads, groups, int(c["head_dim"])),
            dtype=np.float32, optimize=True,
        )

    def forward_token(self, token_id: int, state: QwenDecodeState) -> np.ndarray:
        """Consume one token and return its next-token logits."""
        if state.position >= state.capacity:
            raise ValueError("decode state capacity exhausted")
        eps = float(self.config["rms_norm_eps"])
        self._embedding_into(token_id, state.x)
        for layer_index, layer in enumerate(self.layers):
            self.kernel.rmsnorm_into(state.x, layer.input_norm, state.normalized, eps)
            quantized = state.quantized[:state.normalized.size]
            activation_scale = self.kernel.quantize_activation(state.normalized, quantized)
            self._projection(layer.qkv, state.normalized, quantized, activation_scale, state.qkv)
            self._qk_norm(state.q, layer.q_norm)
            self._qk_norm(state.k, layer.k_norm)
            self._rope(state.q, state.position)
            self._rope(state.k, state.position)
            state.keys[layer_index, state.position] = state.k
            state.values[layer_index, state.position] = state.v
            self._attention(state, layer_index)
            quantized = state.quantized[:state.attended.size]
            activation_scale = self.kernel.quantize_activation(state.attended, quantized)
            self._projection(layer.o, state.attended, quantized, activation_scale, state.projected)
            self.kernel.add_inplace(state.x, state.projected)

            self.kernel.rmsnorm_into(state.x, layer.post_norm, state.normalized, eps)
            quantized = state.quantized[:state.normalized.size]
            activation_scale = self.kernel.quantize_activation(state.normalized, quantized)
            self._projection(layer.gate_up, state.normalized, quantized, activation_scale, state.gate_up)
            np.negative(state.gate, out=state.activated)
            np.exp(state.activated, out=state.activated)
            state.activated += np.float32(1.0)
            np.divide(state.gate, state.activated, out=state.activated)
            state.activated *= state.up
            quantized = state.quantized[:state.activated.size]
            activation_scale = self.kernel.quantize_activation(state.activated, quantized)
            self._projection(layer.down, state.activated, quantized, activation_scale, state.projected)
            self.kernel.add_inplace(state.x, state.projected)

        self.kernel.rmsnorm_into(state.x, self.final_norm, state.normalized, eps)
        quantized = state.quantized[:state.normalized.size]
        activation_scale = self.kernel.quantize_activation(state.normalized, quantized)
        self._projection(
            self.lm_head, state.normalized, quantized, activation_scale, state.logits
        )
        state.position += 1
        return state.logits

    def profile_forward_token(
        self, token_id: int, state: QwenDecodeState,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, int], float]:
        """Time one real ``forward_token`` call without changing its hot path.

        The temporary wrappers keep the production implementation above as the
        code under test. Timings are exclusive: the tied vocabulary projection
        is charged to embedding/LM-head rather than to transformer-body GEMV.
        Packed int4 weights are unpacked inside the GEMV kernel, so there is no
        separate weight depacking pass to time.
        """
        categories = (
            "ternary/int4 GEMV",
            "activation quantization",
            "embedding / LM head",
            "attention",
            "RMSNorm",
            "packing/depacking",
        )
        timings = {name: 0.0 for name in categories}
        counts = {name: 0 for name in categories}

        def measured(name, operation, *args, **kwargs):
            started = time.perf_counter()
            try:
                return operation(*args, **kwargs)
            finally:
                timings[name] += time.perf_counter() - started
                counts[name] += 1

        original_embedding = self._embedding_into
        original_projection = self._projection
        original_qk_norm = self._qk_norm
        original_attention = self._attention
        original_kernel = self.kernel

        def embedding(token, out):
            return measured("embedding / LM head", original_embedding, token, out)

        def projection(projection_, values, quantized, activation_scale, out):
            name = "embedding / LM head" if projection_ is self.lm_head else "ternary/int4 GEMV"
            return measured(
                name, original_projection, projection_, values, quantized,
                activation_scale, out,
            )

        def qk_norm(values, gamma):
            return measured("RMSNorm", original_qk_norm, values, gamma)

        def attention_(decode_state, layer_index):
            return measured("attention", original_attention, decode_state, layer_index)

        class ProfiledKernel:
            def __getattr__(_self, name):
                return getattr(original_kernel, name)

            def rmsnorm_into(_self, *args, **kwargs):
                return measured(
                    "RMSNorm", original_kernel.rmsnorm_into, *args, **kwargs,
                )

            def quantize_activation(_self, *args, **kwargs):
                return measured(
                    "activation quantization",
                    original_kernel.quantize_activation,
                    *args, **kwargs,
                )

        self._embedding_into = embedding
        self._projection = projection
        self._qk_norm = qk_norm
        self._attention = attention_
        self.kernel = ProfiledKernel()
        try:
            started = time.perf_counter()
            logits = self.forward_token(token_id, state)
            total = time.perf_counter() - started
        finally:
            self._embedding_into = original_embedding
            self._projection = original_projection
            self._qk_norm = original_qk_norm
            self._attention = original_attention
            self.kernel = original_kernel
        return logits, timings, counts, total


def build_litespark_cpu(
    model_path=MODEL_PATH, *, threads=DEFAULT_CPU_THREADS, cache_root=None, verbose=True,
    gemv_layout="int4", gemv_kernel=DEFAULT_GEMV_KERNEL,
):
    """Load this Qwen3 LATTICE checkpoint into a torch-free LiteSpark runtime."""
    if gemv_layout not in ("int4", "int8"):
        raise ValueError("gemv_layout must be int4 or int8")
    configure_threads(threads)
    from MLX.lattice_model import load_checkpoint
    from litespark_inference.torchless import kernel

    kernel.ensure_built()
    if not kernel.has_omp():
        raise RuntimeError("LiteSpark has no OpenMP support; install libomp and reinstall it")
    model_path = Path(model_path)
    digest, identity = _cache_identity(model_path)
    directory = Path(cache_root) if cache_root else ROOT / ".cache/litespark-lattice" / digest
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "identity": identity, "complete": False, "projections": {}
    }
    if manifest["identity"] != identity:
        raise ValueError(f"LiteSpark cache does not match checkpoint: {directory}")
    _atomic_json(manifest_path, manifest)

    state, archive = load_checkpoint(model_path)
    try:
        config = json.loads(CONFIG_PATH.read_text())
        projection_names = []
        for index in range(int(config["num_hidden_layers"])):
            attn = f"model.layers.{index}.self_attn"
            mlp = f"model.layers.{index}.mlp"
            projection_names.extend([
                f"{attn}.q_proj", f"{attn}.k_proj", f"{attn}.v_proj",
                f"{attn}.o_proj", f"{mlp}.gate_proj", f"{mlp}.up_proj",
                f"{mlp}.down_proj",
            ])
        if len(projection_names) != 252:
            raise ValueError(f"expected 252 Qwen projections, found {len(projection_names)}")

        for index, name in enumerate(projection_names, 1):
            slug = name.replace(".", "_")
            weights_path = directory / f"{slug}.int4.npy"
            scales_path = directory / f"{slug}.scales.npy"
            if not weights_path.exists() or not scales_path.exists():
                if verbose:
                    print(f"  Converting projection {index}/252: {name}", flush=True)
                _convert_projection(state[name + ".weight"], state[name + ".L"], state[name + ".R"],
                                    weights_path, scales_path)
            manifest["projections"][name] = {"weights": weights_path.name, "scales": scales_path.name}
            _atomic_json(manifest_path, manifest)

        embed_path = directory / "embedding.int4.npy"
        embed_scale_path = directory / "embedding.scales.npy"
        if not embed_path.exists() or not embed_scale_path.exists():
            if verbose:
                print("  Converting tied embedding/lm_head to int4", flush=True)
            _convert_embedding(_array(state["model.embed_tokens.weight"]), embed_path, embed_scale_path)

        def expand_int4(weights: np.ndarray) -> np.ndarray:
            expanded = np.empty(
                (weights.shape[0], weights.shape[1] * 2), dtype=np.int8,
            )
            for start in range(0, weights.shape[0], 256):
                stop = min(weights.shape[0], start + 256)
                block = weights[start:stop].astype(np.int8)
                expanded[start:stop, 0::2] = (block << 4) >> 4
                expanded[start:stop, 1::2] = block >> 4
            return expanded

        def projection(name: str) -> LiteSparkProjection:
            files = manifest["projections"][name]
            weights = np.load(directory / files["weights"], allow_pickle=False)
            scales = np.load(directory / files["scales"], allow_pickle=False)
            expected = tuple(map(int, state[name + ".weight"]["shape"]))
            if weights.shape != (expected[0], expected[1] // 2) or scales.shape != (expected[0],):
                raise ValueError(f"invalid LiteSpark cache shape for {name}")
            if gemv_layout == "int8":
                weights = expand_int4(weights)
            return LiteSparkProjection(weights, scales, gemv_layout)

        def combine(*projections: LiteSparkProjection) -> LiteSparkProjection:
            if len({item.input_size for item in projections}) != 1:
                raise ValueError("cannot fuse LiteSpark projections with different input widths")
            return LiteSparkProjection(
                np.concatenate([item.weights for item in projections], axis=0),
                np.concatenate([item.scales for item in projections]),
                projections[0].layout,
            )

        layers = []
        for index in range(int(config["num_hidden_layers"])):
            prefix = f"model.layers.{index}"
            attn = f"{prefix}.self_attn"
            mlp = f"{prefix}.mlp"
            fp32 = lambda key: np.array(_array(state[key]), dtype=np.float32, copy=True)
            q = projection(f"{attn}.q_proj")
            k = projection(f"{attn}.k_proj")
            v = projection(f"{attn}.v_proj")
            gate = projection(f"{mlp}.gate_proj")
            up = projection(f"{mlp}.up_proj")
            layers.append(QwenLayer(
                qkv=combine(q, k, v),
                o=projection(f"{attn}.o_proj"),
                gate_up=combine(gate, up),
                down=projection(f"{mlp}.down_proj"),
                input_norm=fp32(f"{prefix}.input_layernorm.weight"),
                post_norm=fp32(f"{prefix}.post_attention_layernorm.weight"),
                q_norm=fp32(f"{attn}.q_norm.weight"),
                k_norm=fp32(f"{attn}.k_norm.weight"),
            ))
        embedding_packed = np.load(embed_path, allow_pickle=False)
        embedding_scales = np.load(embed_scale_path, allow_pickle=False)
        final_norm = np.array(_array(state["model.norm.weight"]), dtype=np.float32, copy=True)
        model = LiteSparkQwen(
            config, layers, embedding_packed, embedding_scales, final_norm,
            kernel, gemv_kernel,
        )
        manifest["complete"] = True
        _atomic_json(manifest_path, manifest)
        if verbose:
            print(
                f"LiteSpark Qwen ready: {platform.machine()}, "
                f"{kernel.max_threads()} OpenMP threads, "
                f"{gemv_layout} body GEMV ({gemv_kernel}), no PyTorch",
                flush=True,
            )
        return model
    finally:
        archive.close()


def _input_tokens(prompt: str, raw: bool) -> tuple[list[int], object]:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    text = prompt if raw else (
        f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return tokenizer.encode(text, add_special_tokens=False).ids, tokenizer


def sample_logits(
    logits: np.ndarray, generated: list[int], args, generator: np.random.Generator,
) -> int:
    """Sample one token with the standalone CPU runner's configured policy."""
    if args.argmax:
        return int(np.asarray(logits).argmax())
    values = np.array(logits, dtype=np.float32, copy=True).reshape(-1)
    if generated and args.repetition_penalty != 1.0:
        ids = np.array(sorted(set(generated)), dtype=np.int64)
        repeated = values[ids]
        values[ids] = np.where(
            repeated > 0,
            repeated / np.float32(args.repetition_penalty),
            repeated * np.float32(args.repetition_penalty),
        )
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--argmax", action="store_true",
        help="use deterministic greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--gemv-layout", choices=("int4", "int8"), default="int4",
        help="body-weight layout for the Apple SDOT A/B (default: int4)",
    )
    parser.add_argument(
        "--gemv-kernel",
        choices=("none", "prefetch256", "prefetch512", "prefetch1024"),
        default=DEFAULT_GEMV_KERNEL,
        help=f"packed-int4 prefetch variant (default: {DEFAULT_GEMV_KERNEL})",
    )
    parser.add_argument(
        "--profile-token", action="store_true",
        help="profile one CPU decode token after --warmup-tokens warmup steps",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.max_new_tokens < 1 or args.warmup_tokens < 0
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or args.top_k < 1 or not math.isfinite(args.top_p)
            or not 0 < args.top_p <= 1
            or not math.isfinite(args.repetition_penalty)
            or args.repetition_penalty < 1):
        parser.error(
            "threads, token counts, temperature, and top-k must be positive; "
            "warmup must be nonnegative; top-p must be in (0,1]; "
            "repetition-penalty must be >=1"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    from tokenizers.decoders import DecodeStream

    print("=" * 72)
    print(
        "RUNTIME: Qwen3-4B LATTICE -> LiteSpark CPU "
        f"({args.gemv_layout} SIMD)"
    )
    policy = "argmax" if args.argmax else (
        f"temperature={args.temperature:g}, top_k={args.top_k}, "
        f"top_p={args.top_p:g}, repetition_penalty={args.repetition_penalty:g}"
    )
    print(
        f"threads={args.threads}; model={args.model.name}; "
        f"body_gemv={args.gemv_layout}/{args.gemv_kernel}; sampling={policy}"
    )
    print("=" * 72, flush=True)
    started = time.perf_counter()
    model = build_litespark_cpu(
        args.model, threads=args.threads, cache_root=args.cache,
        gemv_layout=args.gemv_layout, gemv_kernel=args.gemv_kernel,
    )
    print(f"Ready in {time.perf_counter() - started:.1f}s", flush=True)

    tokens, tokenizer = _input_tokens(args.prompt, args.raw_prompt)
    stop_ids = {tokenizer.token_to_id("<|endoftext|>"), tokenizer.token_to_id("<|im_end|>")}
    decoder = DecodeStream(skip_special_tokens=True)
    generated, decode_times = [], []
    generator = np.random.default_rng(args.seed)
    profile_steps = args.warmup_tokens + 1 if args.profile_token else 0
    state = model.new_state(len(tokens) + max(args.max_new_tokens, profile_steps))
    started = time.perf_counter()
    for token_id in tokens:
        logits = model.forward_token(token_id, state)
    prefill_seconds = time.perf_counter() - started
    token = sample_logits(logits, generated, args, generator)
    if args.profile_token:
        for _ in range(args.warmup_tokens):
            generated.append(token)
            logits = model.forward_token(token, state)
            token = sample_logits(logits, generated, args, generator)

        profiled_position = state.position
        generated.append(token)
        logits, timings, counts, model_seconds = model.profile_forward_token(
            token, state,
        )
        started = time.perf_counter()
        token = sample_logits(logits, generated, args, generator)
        sampling_seconds = time.perf_counter() - started
        timings["sampling"] = sampling_seconds
        counts["sampling"] = 1
        total = model_seconds + sampling_seconds
        accounted = sum(timings.values())
        timings["other"] = max(0.0, total - accounted)
        counts["other"] = 0

        print("\n--- ONE WARMED CPU DECODE TOKEN ---")
        print(f"context position: {profiled_position} tokens")
        for name in (
            "ternary/int4 GEMV",
            "activation quantization",
            "embedding / LM head",
            "attention",
            "RMSNorm",
            "sampling",
            "packing/depacking",
            "other",
        ):
            seconds = timings[name]
            percent = 100.0 * seconds / total if total else 0.0
            print(f"{name:26s} {seconds * 1000:8.3f} ms  {percent:5.1f}%")
        print(f"{'total':26s} {total * 1000:8.3f} ms  100.0%")
        print(f"instrumented throughput: {1.0 / total:.2f} tok/s")
        print(
            "calls: "
            f"body GEMV={counts['ternary/int4 GEMV']}, "
            f"activation quant={counts['activation quantization']}, "
            f"attention={counts['attention']}, RMSNorm={counts['RMSNorm']}, "
            f"embedding/head={counts['embedding / LM head']}"
        )
        print(
            "packing/depacking: no standalone hot-path pass; int4 nibbles are "
            "unpacked inside GEMV, and embedding-row unpack is included in "
            "embedding / LM head."
        )
        print(
            "other: RoPE, KV-cache writes, residuals, SiLU, Python dispatch, "
            "and profiling overhead. Use a normal run for production tok/s."
        )
        return 0
    print("\n--- OUTPUT (streaming) ---", flush=True)
    for _ in range(args.max_new_tokens):
        if token in stop_ids:
            break
        generated.append(token)
        piece = decoder.step(tokenizer, token)
        if piece:
            print(piece, end="", flush=True)
        if len(generated) >= args.max_new_tokens:
            break
        started = time.perf_counter()
        logits = model.forward_token(token, state)
        token = sample_logits(logits, generated, args, generator)
        elapsed = time.perf_counter() - started
        if len(generated) > args.warmup_tokens:
            decode_times.append(elapsed)

    print("\n--------------")
    print(f"runtime: litespark cpu ({platform.machine()})")
    print(f"stop: {'end-of-turn token' if token in stop_ids else f'{args.max_new_tokens}-token limit'}")
    print(f"prefill: {len(tokens)} tokens in {prefill_seconds:.3f}s ({len(tokens)/prefill_seconds:.2f} tok/s)")
    if decode_times:
        print(f"decode: {len(decode_times)/sum(decode_times):.2f} tok/s")
    else:
        print("decode: no post-warmup tokens were available to benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

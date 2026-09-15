import os
import platform
import subprocess
import sys

import numpy as np

import pytest

from cpu_lightspark import (
    DEFAULT_CPU_THREADS,
    DEFAULT_GEMV_KERNEL,
    MODEL_PATH,
    _pack_int4,
    _pack_ternary,
    configure_threads,
    parse_args,
    sample_logits,
)


def test_lightspark_cli_defaults_to_local_lattice_configuration():
    args = parse_args(["--prompt", "hello"])
    assert args.model == MODEL_PATH
    assert args.threads == DEFAULT_CPU_THREADS
    assert args.warmup_tokens == 8
    assert args.temperature == 0.7
    assert args.top_k == 40
    assert args.top_p == 0.9
    assert args.repetition_penalty == 1.08
    assert not args.argmax
    assert args.gemv_layout == "int4"
    assert args.gemv_kernel == DEFAULT_GEMV_KERNEL


def test_lightspark_argmax_is_an_explicit_sampling_override():
    args = parse_args(["--prompt", "hello", "--argmax"])
    logits = np.array([-1.0, 3.0, 2.0], dtype=np.float32)
    assert sample_logits(logits, [], args, np.random.default_rng(0)) == 1


def test_lightspark_thread_configuration(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    configure_threads(6)
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"
    with pytest.raises(ValueError, match="positive"):
        configure_threads(0)


def test_lightspark_ternary_packing_matches_native_layout():
    weights = np.array([[-1, 0, 1, -1], [1, 1, 0, -1]], dtype=np.int8)
    expected = np.array([[0 | 1 << 2 | 2 << 4 | 0 << 6],
                         [2 | 2 << 2 | 1 << 4 | 0 << 6]], dtype=np.uint8)
    np.testing.assert_array_equal(_pack_ternary(weights), expected)


def test_lightspark_int4_packing_matches_native_layout():
    weights = np.array([[-7, -1, 0, 7], [6, -6, 3, -3]], dtype=np.int8)
    expected = np.array([[0xF9, 0x70], [0xA6, 0xD3]], dtype=np.uint8)
    np.testing.assert_array_equal(_pack_int4(weights), expected)
    with pytest.raises(ValueError, match=r"\[-7,\+7\]"):
        _pack_int4(np.array([[8, 0]], dtype=np.int8))


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="SDOT fixture is specific to the Apple ARM64 fast path",
)
def test_native_int4_sdot_matches_integer_reference():
    # Keep Homebrew libomp out of pytest's process: test_cpu_runtime imports
    # PyTorch, whose bundled libomp cannot safely coexist with LiteSpark's.
    script = r'''
import numpy as np
from cpu_lightspark import (
    _native_attention_kernel, _native_int4_kernel, _native_int8_kernel,
    _pack_int4,
)
rng = np.random.default_rng(41)
weights = rng.integers(-7, 8, size=(67, 128), dtype=np.int8)
activation = rng.integers(-127, 128, size=128, dtype=np.int8)
scales = rng.random(67, dtype=np.float32)
activation_scale = 0.007
output = np.empty(67, dtype=np.float32)
packed = _pack_int4(weights)
expected = (weights.astype(np.int32) @ activation.astype(np.int32)).astype(np.float32)
expected *= scales * activation_scale
for variant in ("none", "prefetch256", "prefetch512", "prefetch1024"):
    kernel = _native_int4_kernel(variant)
    assert kernel is not None
    kernel(activation.ctypes.data, packed.ctypes.data, scales.ctypes.data,
           activation_scale, output.ctypes.data, weights.shape[0], weights.shape[1])
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-5)
int8_kernel = _native_int8_kernel()
assert int8_kernel is not None
int8_kernel(activation.ctypes.data, weights.ctypes.data, scales.ctypes.data,
            activation_scale, output.ctypes.data, weights.shape[0], weights.shape[1])
np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-5)

attention = _native_attention_kernel()
assert attention is not None
kv_heads, groups, context, capacity, head_dim = 2, 3, 7, 11, 16
query = rng.standard_normal((kv_heads * groups, head_dim), dtype=np.float32)
keys = rng.standard_normal((capacity, kv_heads, head_dim), dtype=np.float32)
values = rng.standard_normal((capacity, kv_heads, head_dim), dtype=np.float32)
scores = np.empty((kv_heads, groups, capacity), dtype=np.float32)
attended = np.empty_like(query)
attention(query.ctypes.data, keys.ctypes.data, values.ctypes.data,
          scores.ctypes.data, attended.ctypes.data, context, kv_heads, groups,
          head_dim, capacity)
reference_scores = np.einsum(
    "hgd,thd->hgt", query.reshape(kv_heads, groups, head_dim), keys[:context],
    dtype=np.float32,
) / np.sqrt(np.float32(head_dim))
reference_scores -= reference_scores.max(axis=-1, keepdims=True)
reference_scores = np.exp(reference_scores)
reference_scores /= reference_scores.sum(axis=-1, keepdims=True)
reference = np.einsum(
    "hgt,thd->hgd", reference_scores, values[:context], dtype=np.float32,
).reshape(kv_heads * groups, head_dim)
np.testing.assert_allclose(attended, reference, rtol=2e-5, atol=2e-5)
'''
    environment = dict(os.environ, OMP_NUM_THREADS="2")
    subprocess.run([sys.executable, "-B", "-c", script], check=True, env=environment)

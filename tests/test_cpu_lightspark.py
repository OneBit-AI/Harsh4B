import os
import platform
import subprocess
import sys

import numpy as np

import pytest

from cpu_lightspark import MODEL_PATH, _pack_int4, _pack_ternary, configure_threads, parse_args


def test_lightspark_cli_defaults_to_local_lattice_configuration():
    args = parse_args(["--prompt", "hello"])
    assert args.model == MODEL_PATH
    assert args.threads == 4
    assert args.warmup_tokens == 8


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
from cpu_lightspark import _native_int4_kernel, _pack_int4
rng = np.random.default_rng(41)
weights = rng.integers(-7, 8, size=(67, 128), dtype=np.int8)
activation = rng.integers(-127, 128, size=128, dtype=np.int8)
scales = rng.random(67, dtype=np.float32)
activation_scale = 0.007
output = np.empty(67, dtype=np.float32)
kernel = _native_int4_kernel()
assert kernel is not None
packed = _pack_int4(weights)
kernel(activation.ctypes.data, packed.ctypes.data, scales.ctypes.data,
       activation_scale, output.ctypes.data, weights.shape[0], weights.shape[1])
expected = (weights.astype(np.int32) @ activation.astype(np.int32)).astype(np.float32)
expected *= scales * activation_scale
np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-5)
'''
    environment = dict(os.environ, OMP_NUM_THREADS="2")
    subprocess.run([sys.executable, "-B", "-c", script], check=True, env=environment)

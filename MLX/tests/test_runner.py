"""Regression checks for the Qwen LATTICE runner; GPU checks need the watchdog."""
import os

import numpy as np
import pytest

import test as runner


class Tensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


@pytest.mark.parametrize("rows,width", [(7, 73), (131, 1024), (25, 9728)])
@pytest.mark.parametrize("mode", [None, 0, 3])
def test_expand_t1_chunk_boundaries(rows, width, mode):
    rng = np.random.default_rng(71)
    modes = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    if mode is not None:
        modes.fill(mode)
    codes = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    selected = codes[modes < 2]
    packed = {
        "shape": (rows, width),
        "maskid": Tensor(runner.pack_codes(modes)),
        "T1c": Tensor(runner.pack_codes(selected.reshape(1, -1)).reshape(-1)),
        "n_T1": selected.size,
    }
    expected = runner.pack_codes(np.where(modes < 2, codes, 0))
    np.testing.assert_array_equal(runner.expand_t1(packed), expected)
    packed["n_T1"] += 1
    with pytest.raises(ValueError, match="T1 count"):
        runner.expand_t1(packed)


@pytest.fixture
def mx():
    if os.environ.get("MLX_GUARDED") != "1":
        pytest.skip("Run GPU checks through MLX.safety")
    import mlx.core as mx
    return mx


@pytest.mark.parametrize("shape", [(1, 1, 2560), (1, 32, 1, 128), (1, 8, 5, 128)])
def test_fused_rms_norm(mx, shape):
    rng = np.random.default_rng(73)
    x = rng.normal(size=shape).astype(np.float16)
    weight = rng.normal(size=(shape[-1],)).astype(np.float16)
    x32 = x.astype(np.float32)
    expected = x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6) * weight
    actual = runner.rms_norm(mx, mx.array(x), mx.array(weight))
    np.testing.assert_allclose(np.array(actual), expected, rtol=0.001, atol=0.002)


@pytest.mark.parametrize("q_len,total", [(1, 1), (1, 129), (5, 5), (3, 11)])
def test_fused_gqa_causal_alignment(mx, q_len, total):
    rng = np.random.default_rng(79)
    q = rng.normal(size=(1, 32, q_len, 128)).astype(np.float16)
    k = rng.normal(size=(1, 8, total, 128)).astype(np.float16)
    v = rng.normal(size=k.shape).astype(np.float16)
    keys = np.repeat(k.astype(np.float32), 4, axis=1)
    values = np.repeat(v.astype(np.float32), 4, axis=1)
    scores = (q.astype(np.float32) @ keys.swapaxes(-1, -2)) / np.sqrt(128)
    if q_len > 1:
        allowed = np.arange(total)[None, :] <= np.arange(q_len)[:, None] + total - q_len
        scores = np.where(allowed, scores, -np.inf)
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = probs @ values
    actual = runner.attention(mx, mx.array(q), mx.array(k), mx.array(v))
    np.testing.assert_allclose(np.array(actual), expected, rtol=0.003, atol=0.003)

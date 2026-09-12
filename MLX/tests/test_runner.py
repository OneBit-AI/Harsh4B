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


def test_kv_cache_capacity_growth_and_layer_isolation(mx):
    cache = runner.KVCache(mx, 2, step=4)
    expected = []
    for size in (3, 1, 2, 5):
        start = len(expected)
        expected.extend(range(start, start + size))
        key = mx.array(np.arange(start, start + size, dtype=np.float16).reshape(1, 1, size, 1))
        k, v = cache.append(0, key, -key)
        mx.eval(k, v)
        np.testing.assert_array_equal(np.array(k).reshape(-1), expected)
        np.testing.assert_array_equal(np.array(v).reshape(-1), -np.array(expected))
        assert cache.entries[0][0].shape[2] == ((len(expected)+3)//4)*4
    assert cache.lengths[1] == 0
    assert cache.entries[1] is None


def test_mlx_sampling_greedy_penalty_and_top_k(mx):
    from inference_runtime import sample_mlx
    logits = mx.array([1.0, 4.0, 3.5, -2.0])
    assert sample_mlx(mx, logits, [], 0) == 1
    assert sample_mlx(mx, logits, [1, 1], 0, rep_penalty=2) == 2
    for _ in range(10):
        assert sample_mlx(mx, logits, [], 1, top_k=1) == 1
        assert sample_mlx(mx, logits, [], 1, top_p=0.01) == 1


def test_qwen_forward_cached_prefill_matches_full(mx):
    # Exercise the real attention/RoPE/cache forward path with inexpensive
    # deterministic projections; no 4B checkpoint allocation is needed.
    class Projection:
        def __init__(self, width, rows):
            self.in_features, self.rows = width, rows
        def __call__(self, x):
            return x[..., mx.arange(self.rows) % x.shape[-1]] * 0.05
    model = runner.Qwen3Lattice.__new__(runner.Qwen3Lattice)
    model.mx = mx
    model.embed = mx.array(np.random.default_rng(91).normal(size=(32, 2560)).astype(np.float16))
    model.lm_head = model.embed
    model.final_norm = mx.ones((2560,), dtype=mx.float16)
    layer = {name: Projection(width, rows) for name, width, rows in (
        ("q",2560,4096), ("k",2560,1024), ("v",2560,1024), ("o",4096,2560),
        ("gate",2560,9728), ("up",2560,9728), ("down",9728,2560))}
    for name in ("input_norm", "post_norm", "q_norm", "k_norm"):
        layer[name] = mx.ones((128 if name in ("q_norm", "k_norm") else 2560,), dtype=mx.float16)
    model.layers = [layer]
    full = model(mx.array([[1, 2, 3, 4]]), runner.KVCache(mx, 1), 0, last_only=True)
    cache = runner.KVCache(mx, 1, step=2)
    mx.eval(model(mx.array([[1, 2]]), cache, 0, last_only=True))
    mx.eval(model(mx.array([[3]]), cache, 2, last_only=True))
    actual = model(mx.array([[4]]), cache, 3, last_only=True)
    mx.eval(actual, full)
    np.testing.assert_allclose(np.array(actual), np.array(full), atol=0.1, rtol=0.005)

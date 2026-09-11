import os
import numpy as np
import pytest

if os.environ.get('MLX_GUARDED') != '1':
    pytest.skip('Run GPU checks through MLX.safety', allow_module_level=True)
import mlx.core as mx
mx.set_memory_limit(1024 ** 3)
mx.set_cache_limit(32 * 1024 ** 2)

from MLX.kernels import CompactLatticeLinear, LatticeLinear, TernaryLinear, bitnet_gemv, lattice_gemv, pack_codes, pack_ternary, unpack_ternary


@pytest.mark.parametrize('width,rows', [(1, 1), (73, 7), (128, 8), (1536, 16)])
@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
def test_bitnet_against_dense(width, rows, dtype):
    rng = np.random.default_rng(7)
    w = rng.integers(-1, 2, (rows, width)).astype(np.float32)
    x = mx.array(rng.normal(size=(1, width)).astype(np.float32)).astype(dtype)
    scale = mx.array([0.037], dtype=mx.float32)
    packed = mx.array(pack_ternary(w))
    actual = bitnet_gemv(x, packed, scale, width)
    reference = x @ (mx.array(w) * scale).astype(dtype).T
    mx.eval(actual, reference)
    np.testing.assert_allclose(np.array(actual), np.array(reference), atol=0.004, rtol=0.003)


@pytest.mark.parametrize('width,bs', [(73, 64), (128, 64), (7, 3)])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_lattice_all_masks_tail_bias(width, bs, dtype):
    rows, rng = 5, np.random.default_rng(19)
    nb = (width + bs - 1) // bs
    mid = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    c0 = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    c1 = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    mu = rng.normal(size=(4, rows, nb)).astype(np.float32)
    a0 = rng.normal(size=(4, rows, nb)).astype(np.float32)
    a1 = rng.normal(size=(2, rows, nb)).astype(np.float32)
    x, bias = rng.normal(size=(1, width)).astype(dtype), rng.normal(size=(rows,)).astype(np.float32)
    w = np.empty((rows, width), np.float32)
    for r in range(rows):
        for c in range(width):
            m, b = mid[r,c], c // bs
            t0 = 1 if c0[r,c] == 1 else -1 if c0[r,c] == 2 else 0
            t1 = 1 if c1[r,c] == 1 else -1 if c1[r,c] == 2 else 0
            w[r,c] = mu[m,r,b] + a0[m,r,b]*t0 + (a1[m,r,b]*t1 if m < 2 else 0)
    actual = lattice_gemv(mx.array(x), *[mx.array(pack_codes(c)) for c in (mid,c0,c1)],
                         mx.array(mu), mx.array(a0), mx.array(a1), bs, mx.array(bias))
    mx.eval(actual)
    expected = (x.astype(np.float32) @ w.T + bias).astype(dtype)
    tolerance = 0.004 if dtype == np.float16 else 1e-4
    np.testing.assert_allclose(np.array(actual), expected, atol=tolerance, rtol=tolerance)
    layer = LatticeLinear(*[mx.array(pack_codes(c)) for c in (mid,c0,c1)],
                          mx.array(mu),mx.array(a0),mx.array(a1),width,bs,mx.array(bias))
    np.testing.assert_allclose(np.array(layer(mx.array(x))), expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(np.array(layer.dense_weight()), w, atol=1e-6, rtol=1e-6)
    prefill = mx.array(np.repeat(x[None,:,:],3,axis=1))
    expected_prefill = (np.array(prefill).astype(np.float32) @ w.T + bias).astype(dtype)
    np.testing.assert_allclose(np.array(layer(prefill)), expected_prefill, atol=tolerance, rtol=tolerance)


def test_compact_lattice_against_dense_t1():
    rows, width, bs = 7, 128, 128
    rng = np.random.default_rng(29)
    mid = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    c0 = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    c1 = rng.integers(0, 4, (rows, width), dtype=np.uint8)
    selected = c1[mid < 2]
    selected = np.pad(selected, (0, (-selected.size) % 4))
    t1c = (selected[0::4] | (selected[1::4] << 2) |
           (selected[2::4] << 4) | (selected[3::4] << 6)).astype(np.uint8)
    row_counts = (mid < 2).sum(axis=1, dtype=np.uint32)
    row_starts = np.zeros(rows, dtype=np.uint32)
    row_starts[1:] = np.cumsum(row_counts[:-1], dtype=np.uint32)
    mu = rng.normal(scale=0.1, size=(4, rows, 1)).astype(np.float16)
    a0 = rng.normal(scale=0.1, size=(4, rows, 1)).astype(np.float16)
    a1 = rng.normal(scale=0.1, size=(2, rows, 1)).astype(np.float16)
    x = mx.array(rng.normal(size=(1, width)).astype(np.float16))
    dense = LatticeLinear(
        mx.array(pack_codes(mid)), mx.array(pack_codes(c0)), mx.array(pack_codes(c1)),
        mx.array(mu), mx.array(a0), mx.array(a1), width, bs,
    )
    compact = CompactLatticeLinear(
        mx.array(pack_codes(mid)), mx.array(pack_codes(c0)), mx.array(t1c),
        mx.array(row_starts), mx.array(mu), mx.array(a0), mx.array(a1), width, bs,
    )
    actual, expected = compact(x), dense(x)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=0.002, rtol=0.002)
    np.testing.assert_allclose(np.array(compact.dense_weight()), np.array(dense.dense_weight()), atol=1e-6)


def test_packing_is_lossless():
    signs = np.random.default_rng(8).integers(-1, 2, (13, 73), dtype=np.int8)
    np.testing.assert_array_equal(np.array(unpack_ternary(mx.array(pack_ternary(signs)), 73)), signs)


def test_common_linear_does_not_change_input_or_quantize():
    signs = np.array([[1,0,-1], [0,1,1]], np.int8)
    layer = TernaryLinear(mx.array(pack_ternary(signs)), mx.array([0.5]), 3)
    # Non-grid inputs would expose an unsolicited activation quantizer.
    x = mx.array([[0.12345, 0.56789, -0.33333]], dtype=mx.float32)
    before = np.array(x)
    np.testing.assert_allclose(np.array(layer(x)), before @ (signs*0.5).T, atol=1e-7)
    np.testing.assert_array_equal(np.array(x), before)


def test_prefill_uses_same_linear_operation():
    rng = np.random.default_rng(3)
    signs = rng.integers(-1,2,(5,7),dtype=np.int8)
    layer = TernaryLinear(mx.array(pack_ternary(signs)), mx.array([0.25]), 7)
    x = mx.array(rng.normal(size=(1,4,7)).astype(np.float32))
    np.testing.assert_allclose(np.array(layer(x)), np.array(x) @ (signs*0.25).T, atol=1e-6)


def test_rejects_nonternary():
    with pytest.raises(ValueError):
        pack_ternary(np.array([[0, 0.5]]))

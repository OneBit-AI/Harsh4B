"""Two-bit packed Metal kernels, preserving the repository's code mapping.

One SIMD group computes one output row. Adjacent lanes load adjacent packed
bytes; each byte contains four components. Accumulate in FP32, reduce once.
LATTICE keeps mu+a0*T0+a1*T1; BitNet has its own single-component scale.
"""
from functools import lru_cache

import mlx.core as mx
import numpy as np


def pack_ternary(values):
    values = np.asarray(values)
    if values.ndim != 2 or not np.isin(values, [-1, 0, 1]).all():
        raise ValueError('expected a 2D ternary {-1,0,+1} matrix')
    return pack_codes(np.where(values == 1, 1, np.where(values == -1, 2, 0)).astype(np.uint8))


def pack_codes(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or not np.isin(codes, [0, 1, 2, 3]).all():
        raise ValueError('expected a 2D matrix of two-bit codes')
    codes = np.pad(codes.astype(np.uint8), ((0, 0), (0, (-codes.shape[1]) % 4)))
    return codes[:, 0::4] | (codes[:, 1::4] << 2) | (codes[:, 2::4] << 4) | (codes[:, 3::4] << 6)


def unpack_codes(packed, width):
    return ((packed[..., None] >> (mx.arange(4, dtype=mx.uint8) * 2)) & 3).reshape(packed.shape[0], -1)[:, :width]


def unpack_ternary(packed, width):
    codes = unpack_codes(packed, width)
    return mx.where(codes == 1, 1.0, mx.where(codes == 2, -1.0, 0.0))


@lru_cache(None)
def _bitnet_kernel():
    return mx.fast.metal_kernel(name='bitnet_packed_gemv', input_names=['x', 'codes', 'scale'], output_names=['out'], source=r'''
        uint lane = thread_index_in_simdgroup;
        uint row = thread_position_in_grid.x / 32;
        float acc = 0.0f;
        if (row < OC) {
            for (uint byte = lane; byte < (IC + 3) / 4; byte += 32) {
                uint packed = codes[row * ((IC + 3) / 4) + byte];
                for (uint j = 0; j < 4; ++j) {
                    uint col = byte * 4 + j;
                    uint code = (packed >> (2 * j)) & 3;
                    if (col < IC) {
                        // Round the scaled weight to the reference activation dtype.
                        T w = T(code == 1 ? scale[0] : (code == 2 ? -scale[0] : 0.0f));
                        acc += float(w) * float(x[col]);
                    }
                }
            }
        }
        float total = simd_sum(acc);
        if (lane == 0 && row < OC) out[row] = T(total);
    ''')


@lru_cache(None)
def _bitnet_kernel_v2():
    """Vectorized decode kernel: one uint32 load per lane per step (16 codes),
    the deployed scale applied once after accumulation. Requires IC % 16 == 0."""
    return mx.fast.metal_kernel(name='bitnet_packed_gemv_v2', input_names=['x', 'codes', 'scale'], output_names=['out'], source=r'''
        uint lane = thread_index_in_simdgroup;
        uint row = thread_position_in_grid.x / 32;
        float acc = 0.0f;
        if (row < OC) {
            device const uchar* wrow = codes + row * (IC / 4);
            float s = fabs(scale[0]);
            for (uint v = lane; v < IC / 16; v += 32) {
                uint p = *(
                    device const uint*)(wrow + v * 4);
                uint col = v * 16;
                for (uint j = 0; j < 16; ++j) {
                    uint code = (p >> (2 * j)) & 3;
                    float sign = float(code == 1) - float(code == 2);
                    acc += sign * float(x[col + j]);
                }
            }
            float total = simd_sum(acc);
            if (lane == 0 && row < OC) out[row] = T(total * s);
        }
    ''')


def bitnet_gemv(x, packed, scale, width):
    if width < 1 or x.ndim < 1 or x.shape[-1] != width or x.size != width or packed.dtype != mx.uint8 or packed.ndim != 2 or packed.shape[0] < 1 or packed.shape[1] != (width + 3) // 4:
        raise ValueError('single activation vector and correctly shaped uint8 packed weights required')
    if x.dtype not in (mx.float16, mx.float32) or scale.size != 1:
        raise ValueError('float16/float32 activations and one scale required')
    rows = packed.shape[0]
    kernel = _bitnet_kernel_v2() if width % 16 == 0 else _bitnet_kernel()
    return kernel(inputs=[x, packed, scale.astype(mx.float32)], template=[('T', x.dtype), ('IC', width), ('OC', rows)],
        grid=(((rows + 3) // 4) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows,)], output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1], rows)


@lru_cache(None)
def _lattice_kernel():
    return mx.fast.metal_kernel(name='lattice_dense_t1_gemv', input_names=['x', 'mid', 't0', 't1', 'scales', 'bias'], output_names=['out'], source=r'''
        uint lane = thread_index_in_simdgroup;
        uint row = thread_position_in_grid.x / 32;
        float acc = 0.0f;
        if (row < OC) {
            for (uint byte = lane; byte < (IC + 3) / 4; byte += 32) {
                uint addr = row * ((IC + 3) / 4) + byte;
                uint mb = mid[addr], c0b = t0[addr], c1b = t1[addr];
                for (uint j = 0; j < 4; ++j) {
                    uint col = byte * 4 + j;
                    if (col < IC) {
                        uint m = (mb >> (j * 2)) & 3;
                        uint c0 = (c0b >> (j * 2)) & 3, c1 = (c1b >> (j * 2)) & 3;
                        float v0 = c0 == 1 ? 1.0f : (c0 == 2 ? -1.0f : 0.0f);
                        float v1 = c1 == 1 ? 1.0f : (c1 == 2 ? -1.0f : 0.0f);
                        uint s = (row * NB + col / BS) * 10;
                        float w = float(scales[s + m]) + float(scales[s + 4 + m]) * v0;
                        if (m < 2) w += float(scales[s + 8 + m]) * v1;
                        acc += w * float(x[col]);
                    }
                }
            }
        }
        float total = simd_sum(acc);
        if (lane == 0 && row < OC) out[row] = T(total + float(bias[row]));
    ''')


def lattice_gemv(x, maskid, t0, t1_dense, mu, a0, a1, blocksize, bias=None):
    width = x.shape[-1]
    rows = maskid.shape[0]
    nb = (width + blocksize - 1) // blocksize if blocksize > 0 else 0
    if blocksize <= 0 or x.size != width or x.dtype not in (mx.float16, mx.float32):
        raise ValueError('positive blocksize and one float activation vector required')
    if any(c.dtype != mx.uint8 or c.shape != (rows, (width + 3) // 4) for c in (maskid, t0, t1_dense)):
        raise ValueError('invalid packed buffer shape/dtype')
    if mu.shape != (4, rows, nb) or a0.shape != mu.shape or a1.shape != (2, rows, nb):
        raise ValueError('invalid per-mask scale shapes')
    if bias is not None and bias.shape != (rows,):
        raise ValueError('invalid bias shape')
    scales = mx.concatenate([mu, a0, a1], axis=0).transpose(1, 2, 0)
    bias = mx.zeros((rows,), dtype=x.dtype) if bias is None else bias
    return _lattice_prepared(x, maskid, t0, t1_dense, scales, bias, blocksize)


def _lattice_prepared(x, maskid, t0, t1_dense, scales, bias, blocksize):
    width, rows, nb = x.shape[-1], maskid.shape[0], scales.shape[1]
    return _lattice_kernel()(inputs=[x, maskid, t0, t1_dense, scales, bias],
        template=[('T', x.dtype), ('IC', width), ('OC', rows), ('BS', blocksize), ('NB', nb)],
        grid=(((rows + 3) // 4) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(rows,)], output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1], rows)


class LatticeLinear:
    """Cache the dense-T1 layout and interleaved scales outside the hot loop.

    Inputs use dense TWO-BIT T1 (not compact T1c). mu/a0: [4,OC,NB];
    a1: [2,OC,NB]. No rotations or activation transformations are introduced.
    """
    def __init__(self, maskid, t0, t1_dense, mu, a0, a1, width, blocksize, bias=None):
        if width < 1 or blocksize < 1 or maskid.ndim != 2:
            raise ValueError('positive dimensions and matrix codes required')
        rows, nb = maskid.shape[0], (width + blocksize - 1) // blocksize
        if rows < 1 or any(c.dtype != mx.uint8 or c.shape != (rows, (width+3)//4) for c in (maskid,t0,t1_dense)):
            raise ValueError('invalid packed buffers')
        if mu.shape != (4,rows,nb) or a0.shape != mu.shape or a1.shape != (2,rows,nb):
            raise ValueError('invalid scale shapes')
        if bias is not None and bias.shape != (rows,):
            raise ValueError('invalid bias shape')
        self.width, self.blocksize = width, blocksize
        self.maskid, self.t0, self.t1 = maskid, t0, t1_dense
        # Match CUDA's FP32 reconstruction even when source scales are FP16.
        self.scales = mx.contiguous(mx.concatenate([mu,a0,a1], axis=0).transpose(1,2,0)).astype(mx.float32)
        self.bias = mx.zeros((rows,), dtype=mx.float32) if bias is None else bias.astype(mx.float32)
        mx.eval(self.scales, self.bias)

    def dense_weight(self):
        mid = unpack_codes(self.maskid, self.width).astype(mx.int32)
        t0, t1 = unpack_ternary(self.t0, self.width), unpack_ternary(self.t1, self.width)
        row = mx.arange(mid.shape[0])[:,None]
        block = (mx.arange(self.width)//self.blocksize)[None,:]
        mu = self.scales[row,block,mid]
        a0 = self.scales[row,block,4+mid]
        a1 = mx.where(mid < 2, self.scales[row,block,8+mx.minimum(mid,1)], 0)
        return mu + a0*t0 + a1*t1

    def reference(self, x):
        return (x.astype(mx.float32) @ self.dense_weight().T + self.bias).astype(x.dtype)

    def __call__(self, x):
        if x.shape[-1] != self.width or x.dtype not in (mx.float16,mx.float32):
            raise ValueError('input width/dtype mismatch')
        if x.size == self.width:
            return _lattice_prepared(x, self.maskid, self.t0, self.t1, self.scales, self.bias, self.blocksize)
        return self.reference(x)


class TernaryLinear:
    """Shared linear primitive. Accepts deployed codes; never quantizes a model.

    The caller retains its original activation quantization, normalization,
    rotations, cache and attention. Scale is the actual deployed magnitude.
    This uniform-scale primitive is distinct from LATTICE's affine format.
    """
    def __init__(self, packed, scale, width):
        if width < 1 or packed.ndim != 2 or packed.shape != (packed.shape[0], (width + 3) // 4) or packed.shape[0] < 1 or packed.dtype != mx.uint8:
            raise ValueError('invalid packed shape/dtype')
        if scale.size != 1 or not bool(mx.all(mx.isfinite(scale))):
            raise ValueError('one finite deployed scale required')
        self.packed, self.scale, self.width = packed, scale.astype(mx.float32), width

    def dense_weight(self, dtype):
        return (unpack_ternary(self.packed, self.width) * self.scale).astype(dtype)

    def reference(self, x):
        return x @ self.dense_weight(x.dtype).T

    def __call__(self, x):
        if x.shape[-1] != self.width or x.dtype not in (mx.float16, mx.float32):
            raise ValueError('input width/dtype mismatch')
        if x.size == self.width:
            return bitnet_gemv(x, self.packed, self.scale, self.width)
        # Multi-token prefill retains the framework's existing GEMM arithmetic.
        return self.reference(x)

// CPU-only packed LATTICE kernels. No Metal, MPS, CUDA, or weight requantization.
#include <algorithm>
#include <cstdint>
#include <dispatch/dispatch.h>

#if defined(__aarch64__)
#include <arm_neon.h>

// Expand sixteen consecutive 2-bit codes from four packed bytes. Keeping this
// in registers avoids constructing a dequantized weight buffer before GEMV.
static inline uint8x16_t unpack_16(const uint8_t *packed) {
    uint32_t word;
    __builtin_memcpy(&word, packed, sizeof(word));
    const uint8x16_t bytes = vreinterpretq_u8_u32(vdupq_n_u32(word));
    const uint8x16_t indices = {0, 0, 0, 0, 1, 1, 1, 1,
                               2, 2, 2, 2, 3, 3, 3, 3};
    const int8x16_t shifts = {0, -2, -4, -6, 0, -2, -4, -6,
                             0, -2, -4, -6, 0, -2, -4, -6};
    return vandq_u8(vshlq_u8(vqtbl1q_u8(bytes, indices), shifts), vdupq_n_u8(3));
}

static inline float32x4_t ternary_4(uint32x4_t codes) {
    const uint32x4_t one = vshrq_n_u32(vceqq_u32(codes, vdupq_n_u32(1)), 31);
    const uint32x4_t minus_one = vshrq_n_u32(vceqq_u32(codes, vdupq_n_u32(2)), 31);
    return vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(one),
                                   vreinterpretq_s32_u32(minus_one)));
}

static inline float32x4_t select_4(uint32x4_t modes, float s0, float s1,
                                   float s2, float s3) {
    float32x4_t result = vdupq_n_f32(s3);
    result = vbslq_f32(vceqq_u32(modes, vdupq_n_u32(2)), vdupq_n_f32(s2), result);
    result = vbslq_f32(vceqq_u32(modes, vdupq_n_u32(1)), vdupq_n_f32(s1), result);
    return vbslq_f32(vceqq_u32(modes, vdupq_n_u32(0)), vdupq_n_f32(s0), result);
}
#endif

struct Work {
    const float *x;
    const uint8_t *mid, *t0, *t1;
    const _Float16 *scales;
    float *out;
    int rows, width, bs, workers;
    bool dense;
    const uint32_t *row_starts;
};

static void run_rows(void *opaque, size_t worker) {
    const Work &w = *static_cast<Work *>(opaque);
    const int begin = w.rows * worker / w.workers;
    const int end = w.rows * (worker + 1) / w.workers;
    const int nb = (w.width + w.bs - 1) / w.bs;
    const int packed_width = (w.width + 3) / 4;
    for (int row = begin; row < end; ++row) {
        float sum = 0;
#if defined(__aarch64__)
        float32x4_t vector_sum = vdupq_n_f32(0);
#endif
        uint32_t compact_offset = w.row_starts ? w.row_starts[row] : 0;
        const int base = row * packed_width;
        for (int block = 0; block < nb; ++block) {
            const _Float16 *s = w.scales + (row * nb + block) * 10;
            const float mu0 = s[0], mu1 = s[1], mu2 = s[2], mu3 = s[3];
            const float a00 = s[4], a01 = s[5], a02 = s[6], a03 = s[7];
            const float a10 = s[8], a11 = s[9];
            const int stop = std::min(w.width, (block + 1) * w.bs);
#if defined(__aarch64__)
            int col = block * w.bs;
            if (!w.dense) {
                // Model blocks are 128-wide, so the hot path is naturally
                // aligned. The condition keeps arbitrary test shapes safe.
                for (; col + 16 <= stop && (col & 3) == 0; col += 16) {
                    const int p = base + col / 4;
                    const uint8x16_t mode_codes = unpack_16(w.mid + p);
                    const uint8x16_t t0_codes = unpack_16(w.t0 + p);
                    uint8x16_t t1_codes;
                    if (w.row_starts) {
                        alignas(16) uint8_t modes[16];
                        alignas(16) uint8_t sparse_t1[16] = {};
                        vst1q_u8(modes, mode_codes);
                        for (int lane = 0; lane < 16; ++lane) {
                            if (modes[lane] < 2) {
                                sparse_t1[lane] = (w.t1[compact_offset / 4]
                                    >> ((compact_offset & 3) * 2)) & 3;
                                ++compact_offset;
                            }
                        }
                        t1_codes = vld1q_u8(sparse_t1);
                    } else {
                        t1_codes = unpack_16(w.t1 + p);
                    }

                    const uint16x8_t mode_lo = vmovl_u8(vget_low_u8(mode_codes));
                    const uint16x8_t mode_hi = vmovl_u8(vget_high_u8(mode_codes));
                    const uint16x8_t t0_lo = vmovl_u8(vget_low_u8(t0_codes));
                    const uint16x8_t t0_hi = vmovl_u8(vget_high_u8(t0_codes));
                    const uint16x8_t t1_lo = vmovl_u8(vget_low_u8(t1_codes));
                    const uint16x8_t t1_hi = vmovl_u8(vget_high_u8(t1_codes));
                    const uint32x4_t modes[4] = {
                        vmovl_u16(vget_low_u16(mode_lo)), vmovl_u16(vget_high_u16(mode_lo)),
                        vmovl_u16(vget_low_u16(mode_hi)), vmovl_u16(vget_high_u16(mode_hi))};
                    const uint32x4_t codes0[4] = {
                        vmovl_u16(vget_low_u16(t0_lo)), vmovl_u16(vget_high_u16(t0_lo)),
                        vmovl_u16(vget_low_u16(t0_hi)), vmovl_u16(vget_high_u16(t0_hi))};
                    const uint32x4_t codes1[4] = {
                        vmovl_u16(vget_low_u16(t1_lo)), vmovl_u16(vget_high_u16(t1_lo)),
                        vmovl_u16(vget_low_u16(t1_hi)), vmovl_u16(vget_high_u16(t1_hi))};

                    for (int group = 0; group < 4; ++group) {
                        const float32x4_t mu = select_4(modes[group], mu0, mu1, mu2, mu3);
                        const float32x4_t a0 = select_4(modes[group], a00, a01, a02, a03);
                        const float32x4_t a1 = select_4(modes[group], a10, a11, 0, 0);
                        float32x4_t value = vfmaq_f32(mu, a0, ternary_4(codes0[group]));
                        value = vfmaq_f32(value, a1, ternary_4(codes1[group]));
                        vector_sum = vfmaq_f32(vector_sum, value, vld1q_f32(w.x + col + group * 4));
                    }
                }
            }
#else
            int col = block * w.bs;
#endif
            #pragma clang loop vectorize(enable)
            for (; col < stop; ++col) {
                const int shift = (col & 3) * 2;
                const int p = base + col / 4;
                const int m = (w.mid[p] >> shift) & 3;
                const int c0 = (w.t0[p] >> shift) & 3;
                int c1 = 0;
                if (w.row_starts) {
                    if (m < 2) {
                        c1 = (w.t1[compact_offset / 4] >> ((compact_offset & 3) * 2)) & 3;
                        ++compact_offset;
                    }
                } else c1 = (w.t1[p] >> shift) & 3;
                const float mu = m == 0 ? mu0 : m == 1 ? mu1 : m == 2 ? mu2 : mu3;
                const float a0 = m == 0 ? a00 : m == 1 ? a01 : m == 2 ? a02 : a03;
                const float a1 = m == 0 ? a10 : m == 1 ? a11 : 0;
                const float v0 = float(c0 == 1) - float(c0 == 2);
                const float v1 = float(c1 == 1) - float(c1 == 2);
                const float value = mu + a0 * v0 + a1 * v1;
                if (w.dense) w.out[row * w.width + col] = value;
                else sum += value * w.x[col];
            }
        }
        if (!w.dense) {
#if defined(__aarch64__)
            sum += vaddvq_f32(vector_sum);
#endif
            w.out[row] = sum;
        }
    }
}

extern "C" void lattice_cpu(const float *x, const uint8_t *mid, const uint8_t *t0,
                            const uint8_t *t1, const _Float16 *scales, float *out,
                            int rows, int width, int blocksize, int workers, int dense,
                            const uint32_t *row_starts) {
    Work w{x, mid, t0, t1, scales, out, rows, width, blocksize,
           std::max(1, std::min(workers, rows)), bool(dense), row_starts};
    dispatch_apply_f(w.workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &w, run_rows);
}

// LiteSpark-compatible signed-int4 x int8 GEMV for Apple/ARM NEON SDOT.
// Weights stay packed as two signed nibbles per byte and are unpacked only
// in registers inside the dot-product loop.

#include <arm_neon.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

static inline void unpack_int4_16(
    uint8x16_t packed, int8x16_t* low_half, int8x16_t* high_half
) {
    const int8x16_t values = vreinterpretq_s8_u8(packed);
    const int8x16_t low = vshrq_n_s8(vshlq_n_s8(values, 4), 4);
    const int8x16_t high = vshrq_n_s8(values, 4);
    *low_half = vzip1q_s8(low, high);
    *high_half = vzip2q_s8(low, high);
}

template <int PREFETCH_DISTANCE>
static void litespark_int4_i8_gemv_impl(
    const int8_t* __restrict__ x,
    const uint8_t* __restrict__ packed_weights,
    const float* __restrict__ weight_scales,
    float activation_scale,
    float* __restrict__ output,
    int rows,
    int width
) {
    const int packed_width = width >> 1;

#pragma omp parallel for if(rows >= 64) schedule(static)
    for (int row_index = 0; row_index < rows; ++row_index) {
        const uint8_t* row = packed_weights
            + static_cast<ptrdiff_t>(row_index) * packed_width;
        // Independent accumulators hide SDOT latency; a single dependency
        // chain leaves Apple's four integer/NEON issue paths under-filled.
        int32x4_t accumulator0 = vdupq_n_s32(0);
        int32x4_t accumulator1 = vdupq_n_s32(0);
        int32x4_t accumulator2 = vdupq_n_s32(0);
        int32x4_t accumulator3 = vdupq_n_s32(0);
        int packed_index = 0;

        // 32 packed bytes -> 64 signed int4 values -> four SDOT operations.
        for (; packed_index + 32 <= packed_width; packed_index += 32) {
            if constexpr (PREFETCH_DISTANCE > 0) {
                __builtin_prefetch(
                    row + packed_index + PREFETCH_DISTANCE, 0, 0);
            }
            int8x16_t w0, w1, w2, w3;
            unpack_int4_16(vld1q_u8(row + packed_index), &w0, &w1);
            unpack_int4_16(vld1q_u8(row + packed_index + 16), &w2, &w3);
            const int input_index = packed_index << 1;
            accumulator0 = vdotq_s32(accumulator0, w0, vld1q_s8(x + input_index));
            accumulator1 = vdotq_s32(accumulator1, w1, vld1q_s8(x + input_index + 16));
            accumulator2 = vdotq_s32(accumulator2, w2, vld1q_s8(x + input_index + 32));
            accumulator3 = vdotq_s32(accumulator3, w3, vld1q_s8(x + input_index + 48));
        }

        accumulator0 = vaddq_s32(accumulator0, accumulator1);
        accumulator2 = vaddq_s32(accumulator2, accumulator3);
        int32_t scalar = vaddvq_s32(vaddq_s32(accumulator0, accumulator2));
        for (; packed_index < packed_width; ++packed_index) {
            const uint8_t byte = row[packed_index];
            const int8_t low = static_cast<int8_t>(byte << 4) >> 4;
            const int8_t high = static_cast<int8_t>(byte) >> 4;
            const int input_index = packed_index << 1;
            scalar += static_cast<int32_t>(low) * x[input_index];
            scalar += static_cast<int32_t>(high) * x[input_index + 1];
        }
        output[row_index] = static_cast<float>(scalar)
            * activation_scale * weight_scales[row_index];
    }
}

extern "C" void litespark_int4_i8_gemv(
    const int8_t* x, const uint8_t* weights, const float* scales,
    float activation_scale, float* output, int rows, int width
) {
    litespark_int4_i8_gemv_impl<1024>(
        x, weights, scales, activation_scale, output, rows, width);
}

extern "C" void litespark_int4_i8_gemv_nopf(
    const int8_t* x, const uint8_t* weights, const float* scales,
    float activation_scale, float* output, int rows, int width
) {
    litespark_int4_i8_gemv_impl<0>(
        x, weights, scales, activation_scale, output, rows, width);
}

extern "C" void litespark_int4_i8_gemv_pf256(
    const int8_t* x, const uint8_t* weights, const float* scales,
    float activation_scale, float* output, int rows, int width
) {
    litespark_int4_i8_gemv_impl<256>(
        x, weights, scales, activation_scale, output, rows, width);
}

extern "C" void litespark_int4_i8_gemv_pf512(
    const int8_t* x, const uint8_t* weights, const float* scales,
    float activation_scale, float* output, int rows, int width
) {
    litespark_int4_i8_gemv_impl<512>(
        x, weights, scales, activation_scale, output, rows, width);
}

// Paper-inspired control: expand the same row-quantized weights to signed
// bytes once at load time, then feed them directly to SDOT. This doubles body
// weight traffic relative to int4 but removes every nibble-unpack instruction.
extern "C" void litespark_int8_i8_gemv(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ weights,
    const float* __restrict__ weight_scales,
    float activation_scale,
    float* __restrict__ output,
    int rows,
    int width
) {
#pragma omp parallel for if(rows >= 64) schedule(static)
    for (int row_index = 0; row_index < rows; ++row_index) {
        const int8_t* row = weights
            + static_cast<ptrdiff_t>(row_index) * width;
        int32x4_t accumulator0 = vdupq_n_s32(0);
        int32x4_t accumulator1 = vdupq_n_s32(0);
        int32x4_t accumulator2 = vdupq_n_s32(0);
        int32x4_t accumulator3 = vdupq_n_s32(0);
        int index = 0;

        for (; index + 64 <= width; index += 64) {
            __builtin_prefetch(row + index + 1024, 0, 0);
            accumulator0 = vdotq_s32(
                accumulator0, vld1q_s8(row + index), vld1q_s8(x + index));
            accumulator1 = vdotq_s32(
                accumulator1, vld1q_s8(row + index + 16), vld1q_s8(x + index + 16));
            accumulator2 = vdotq_s32(
                accumulator2, vld1q_s8(row + index + 32), vld1q_s8(x + index + 32));
            accumulator3 = vdotq_s32(
                accumulator3, vld1q_s8(row + index + 48), vld1q_s8(x + index + 48));
        }

        accumulator0 = vaddq_s32(accumulator0, accumulator1);
        accumulator2 = vaddq_s32(accumulator2, accumulator3);
        int32_t scalar = vaddvq_s32(vaddq_s32(accumulator0, accumulator2));
        for (; index < width; ++index) {
            scalar += static_cast<int32_t>(row[index]) * x[index];
        }
        output[row_index] = static_cast<float>(scalar)
            * activation_scale * weight_scales[row_index];
    }
}

// Single-token grouped-query attention.  NumPy's two einsum dispatches and
// several temporary ufunc passes cost more than the arithmetic at decode
// lengths, so keep the complete score/softmax/value operation in one native
// call.  Keys and values use [time, kv_head, head_dim] layout; query/output
// heads are ordered [kv_head, group, head_dim].
extern "C" void litespark_gqa_decode(
    const float* __restrict__ query,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ scores,
    float* __restrict__ output,
    int context,
    int kv_heads,
    int groups,
    int head_dim,
    int score_stride
) {
    const int query_heads = kv_heads * groups;
    const int kv_time_stride = kv_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#pragma omp parallel for if(query_heads >= 8) schedule(static)
    for (int query_head = 0; query_head < query_heads; ++query_head) {
        const int kv_head = query_head / groups;
        const float* q = query
            + static_cast<ptrdiff_t>(query_head) * head_dim;
        float* score = scores
            + static_cast<ptrdiff_t>(query_head) * score_stride;
        float maximum = -FLT_MAX;

        for (int token = 0; token < context; ++token) {
            const float* key = keys
                + static_cast<ptrdiff_t>(token) * kv_time_stride
                + static_cast<ptrdiff_t>(kv_head) * head_dim;
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            int dim = 0;
            for (; dim + 16 <= head_dim; dim += 16) {
                sum0 = vfmaq_f32(sum0, vld1q_f32(q + dim),
                                 vld1q_f32(key + dim));
                sum1 = vfmaq_f32(sum1, vld1q_f32(q + dim + 4),
                                 vld1q_f32(key + dim + 4));
                sum2 = vfmaq_f32(sum2, vld1q_f32(q + dim + 8),
                                 vld1q_f32(key + dim + 8));
                sum3 = vfmaq_f32(sum3, vld1q_f32(q + dim + 12),
                                 vld1q_f32(key + dim + 12));
            }
            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            float dot = vaddvq_f32(vaddq_f32(sum0, sum2));
            for (; dim < head_dim; ++dim) {
                dot += q[dim] * key[dim];
            }
            score[token] = dot * scale;
            maximum = std::max(maximum, score[token]);
        }

        float denominator = 0.0f;
        for (int token = 0; token < context; ++token) {
            const float probability = std::exp(score[token] - maximum);
            score[token] = probability;
            denominator += probability;
        }
        const float inverse_denominator = 1.0f / denominator;
        for (int token = 0; token < context; ++token) {
            score[token] *= inverse_denominator;
        }

        float* attended = output
            + static_cast<ptrdiff_t>(query_head) * head_dim;
        int dim = 0;
        for (; dim + 16 <= head_dim; dim += 16) {
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            for (int token = 0; token < context; ++token) {
                const float* value = values
                    + static_cast<ptrdiff_t>(token) * kv_time_stride
                    + static_cast<ptrdiff_t>(kv_head) * head_dim + dim;
                const float probability = score[token];
                sum0 = vfmaq_n_f32(sum0, vld1q_f32(value), probability);
                sum1 = vfmaq_n_f32(sum1, vld1q_f32(value + 4), probability);
                sum2 = vfmaq_n_f32(sum2, vld1q_f32(value + 8), probability);
                sum3 = vfmaq_n_f32(sum3, vld1q_f32(value + 12), probability);
            }
            vst1q_f32(attended + dim, sum0);
            vst1q_f32(attended + dim + 4, sum1);
            vst1q_f32(attended + dim + 8, sum2);
            vst1q_f32(attended + dim + 12, sum3);
        }
        for (; dim < head_dim; ++dim) {
            float sum = 0.0f;
            for (int token = 0; token < context; ++token) {
                const float* value = values
                    + static_cast<ptrdiff_t>(token) * kv_time_stride
                    + static_cast<ptrdiff_t>(kv_head) * head_dim;
                sum += score[token] * value[dim];
            }
            attended[dim] = sum;
        }
    }
}

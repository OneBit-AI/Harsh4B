// LiteSpark-compatible signed-int4 x int8 GEMV for Apple/ARM NEON SDOT.
// Weights stay packed as two signed nibbles per byte and are unpacked only
// in registers inside the dot-product loop.

#include <arm_neon.h>
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

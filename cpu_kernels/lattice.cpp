// CPU-only packed LATTICE kernels. No Metal, MPS, CUDA, or weight requantization.
#include <algorithm>
#include <cstdint>
#include <dispatch/dispatch.h>

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
        uint32_t compact_offset = w.row_starts ? w.row_starts[row] : 0;
        const int base = row * packed_width;
        for (int block = 0; block < nb; ++block) {
            const _Float16 *s = w.scales + (row * nb + block) * 10;
            const float mu0 = s[0], mu1 = s[1], mu2 = s[2], mu3 = s[3];
            const float a00 = s[4], a01 = s[5], a02 = s[6], a03 = s[7];
            const float a10 = s[8], a11 = s[9];
            const int stop = std::min(w.width, (block + 1) * w.bs);
            #pragma clang loop vectorize(enable)
            for (int col = block * w.bs; col < stop; ++col) {
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
        if (!w.dense) w.out[row] = sum;
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

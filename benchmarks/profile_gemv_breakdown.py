import torch
import triton
import triton.language as tl
import time, statistics, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import to_device
from gemv_triton_v4 import packed_gemv_v4, _gemv_v4

@triton.jit
def _gemv_no_scalars(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1c_ptr, base_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_base_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr, DIVISIBLE: tl.constexpr,
):
    r = tl.program_id(0)
    if r >= OC: return
    NBYTE: tl.constexpr = BS // 4
    acc = tl.zeros((), dtype=tl.float32)
    bi = tl.arange(0, NBYTE)
    sh = (tl.arange(0, 4) * 2).to(tl.uint8)
    off = tl.arange(0, BS)

    for b in range(NB):
        mb = tl.load(maskid_ptr + r * s_mid_r + b * NBYTE + bi)
        tb = tl.load(t0_ptr + r * s_t0_r + b * NBYTE + bi)
        mid = tl.reshape((mb[:, None] >> sh[None, :]) & 3, (BS,))
        t0c = tl.reshape((tb[:, None] >> sh[None, :]) & 3, (BS,))
        T0 = tl.where(t0c == 1, 1.0, tl.where(t0c == 2, -1.0, 0.0))

        c = b * BS + off
        is_o2 = mid < 2
        io = is_o2.to(tl.int32)
        local = tl.cumsum(io, axis=0) - io
        base = tl.load(base_ptr + r * s_base_r + b)
        t1i = base + local
        t1b = tl.load(t1c_ptr + (t1i >> 2), mask=is_o2, other=0)
        t1c_ = (t1b >> ((t1i & 3) * 2).to(tl.uint8)) & 3
        T1 = tl.where(is_o2, tl.where(t1c_ == 1, 1.0, tl.where(t1c_ == 2, -1.0, 0.0)), 0.0)

        # Skip the 10 scalar global loads:
        W = T0 + 0.5 * T1
        xv = tl.load(x_ptr + c).to(tl.float32)
        acc += tl.sum(W * xv, axis=0)

    if HAS_BIAS: acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))


@triton.jit
def _gemv_no_gather(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1c_ptr, base_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_base_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr, DIVISIBLE: tl.constexpr,
):
    r = tl.program_id(0)
    if r >= OC: return
    NBYTE: tl.constexpr = BS // 4
    acc = tl.zeros((), dtype=tl.float32)
    bi = tl.arange(0, NBYTE)
    sh = (tl.arange(0, 4) * 2).to(tl.uint8)
    off = tl.arange(0, BS)

    for b in range(NB):
        mb = tl.load(maskid_ptr + r * s_mid_r + b * NBYTE + bi)
        tb = tl.load(t0_ptr + r * s_t0_r + b * NBYTE + bi)
        mid = tl.reshape((mb[:, None] >> sh[None, :]) & 3, (BS,))
        t0c = tl.reshape((tb[:, None] >> sh[None, :]) & 3, (BS,))
        T0 = tl.where(t0c == 1, 1.0, tl.where(t0c == 2, -1.0, 0.0))

        c = b * BS + off
        # Skip the cumsum and gather:
        mu0 = tl.load(mu_ptr + 0*s_mu_m + r*s_mu_r + b).to(tl.float32)
        mu1 = tl.load(mu_ptr + 1*s_mu_m + r*s_mu_r + b).to(tl.float32)
        p0 = tl.load(a0_ptr + 0*s_mu_m + r*s_mu_r + b).to(tl.float32)
        p1 = tl.load(a0_ptr + 1*s_mu_m + r*s_mu_r + b).to(tl.float32)
        MU = tl.where(mid == 0, mu0, mu1)
        A0 = tl.where(mid == 0, p0, p1)

        W = MU + A0 * T0
        xv = tl.load(x_ptr + c).to(tl.float32)
        acc += tl.sum(W * xv, axis=0)

    if HAS_BIAS: acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))


@triton.jit
def _gemv_pure_stream(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1c_ptr, base_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_base_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr, DIVISIBLE: tl.constexpr,
):
    r = tl.program_id(0)
    if r >= OC: return
    NBYTE: tl.constexpr = BS // 4
    acc = tl.zeros((), dtype=tl.float32)
    bi = tl.arange(0, NBYTE)
    off = tl.arange(0, BS)

    for b in range(NB):
        mb = tl.load(maskid_ptr + r * s_mid_r + b * NBYTE + bi)
        tb = tl.load(t0_ptr + r * s_t0_r + b * NBYTE + bi)
        c = b * BS + off
        xv = tl.load(x_ptr + c).to(tl.float32)
        acc += tl.sum(xv, axis=0)

    if HAS_BIAS: acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))


def bench(f, iters=80):
    for _ in range(15): f()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); f(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(sorted(times))

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
pk = torch.load(rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt", map_location="cpu", weights_only=False)
pk.pop("__latmeta__", None)

key = [k for k, v in pk.items() if isinstance(v, dict) and tuple(v.get("shape", ())) == (2560, 9728)][0]
p = to_device(pk[key])
OC, IC = p["shape"]
NB, BS = int(p["nblocks"]), int(p["blocksize"])
x = torch.randn(1, IC, device="cuda", dtype=torch.float16) * 0.1
out = torch.empty(OC, device="cuda", dtype=torch.float16)
xf = x.reshape(-1)

def run_kernel(k_fn, nw=1, ns=3):
    k_fn[(OC,)](
        xf, out, xf,
        p["maskid"], p["T0"], p["T1c"], p["base"], p["mu"], p["a0"], p["a1"],
        OC, IC, NB,
        p["maskid"].stride(0), p["T0"].stride(0), p["base"].stride(0),
        p["mu"].stride(0), p["mu"].stride(1), p["a1"].stride(0), p["a1"].stride(1),
        HAS_BIAS=False, BS=BS, DIVISIBLE=True, num_warps=nw, num_stages=ns
    )

print("Profiling breakdown on down_proj [2560, 9728]:")
t_full = bench(lambda: run_kernel(_gemv_v4))
t_no_scalars = bench(lambda: run_kernel(_gemv_no_scalars))
t_no_gather = bench(lambda: run_kernel(_gemv_no_gather))
t_pure_stream = bench(lambda: run_kernel(_gemv_pure_stream))

print(f"  Full v4 kernel:       {t_full*1000:6.1f} us (100.0%)")
print(f"  Without 10 scalars:   {t_no_scalars*1000:6.1f} us ({t_no_scalars/t_full*100:5.1f}%)")
print(f"  Without T1 gather:    {t_no_gather*1000:6.1f} us ({t_no_gather/t_full*100:5.1f}%)")
print(f"  Pure load stream:     {t_pure_stream*1000:6.1f} us ({t_pure_stream/t_full*100:5.1f}%)")

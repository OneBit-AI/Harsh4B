import torch
import triton
import triton.language as tl
import time, statistics, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import to_device
from gemv_triton_v4 import packed_gemv_v4

@triton.jit
def _gemv_v4_nomask(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1c_ptr, base_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_base_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr,
):
    r = tl.program_id(0)
    if r >= OC:
        return
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

        mu0 = tl.load(mu_ptr + 0*s_mu_m + r*s_mu_r + b).to(tl.float32)
        mu1 = tl.load(mu_ptr + 1*s_mu_m + r*s_mu_r + b).to(tl.float32)
        mu2 = tl.load(mu_ptr + 2*s_mu_m + r*s_mu_r + b).to(tl.float32)
        mu3 = tl.load(mu_ptr + 3*s_mu_m + r*s_mu_r + b).to(tl.float32)
        p0 = tl.load(a0_ptr + 0*s_mu_m + r*s_mu_r + b).to(tl.float32)
        p1 = tl.load(a0_ptr + 1*s_mu_m + r*s_mu_r + b).to(tl.float32)
        p2 = tl.load(a0_ptr + 2*s_mu_m + r*s_mu_r + b).to(tl.float32)
        p3 = tl.load(a0_ptr + 3*s_mu_m + r*s_mu_r + b).to(tl.float32)
        q0 = tl.load(a1_ptr + 0*s_a1_m + r*s_a1_r + b).to(tl.float32)
        q1 = tl.load(a1_ptr + 1*s_a1_m + r*s_a1_r + b).to(tl.float32)
        MU = tl.where(mid == 0, mu0, tl.where(mid == 1, mu1, tl.where(mid == 2, mu2, mu3)))
        A0 = tl.where(mid == 0, p0, tl.where(mid == 1, p1, tl.where(mid == 2, p2, p3)))
        A1 = tl.where(mid == 0, q0, tl.where(mid == 1, q1, 0.0))

        W = MU + A0 * T0 + A1 * T1
        xv = tl.load(x_ptr + c).to(tl.float32)
        acc += tl.sum(W * xv, axis=0)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))

def packed_gemv_nomask(x, p, bias=None, num_warps=1, num_stages=3):
    OC, IC = p["shape"]
    NB, BS = int(p["nblocks"]), int(p["blocksize"])
    xf = x.reshape(-1)
    out = torch.empty(OC, device=xf.device, dtype=torch.float16)
    _gemv_v4_nomask[(OC,)](
        xf, out, bias if bias is not None else xf,
        p["maskid"], p["T0"], p["T1c"], p["base"], p["mu"], p["a0"], p["a1"],
        OC, IC, NB,
        p["maskid"].stride(0), p["T0"].stride(0), p["base"].stride(0),
        p["mu"].stride(0), p["mu"].stride(1), p["a1"].stride(0), p["a1"].stride(1),
        HAS_BIAS=bias is not None, BS=BS, num_warps=num_warps, num_stages=num_stages,
    )
    return out.reshape(1, OC)

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

sink = torch.zeros(1, device="cuda")
for name, sh in [("down_proj", (2560, 9728)), ("gate_proj", (9728, 2560)), ("k_proj", (1024, 2560)), ("o_proj", (2560, 4096))]:
    key = [k for k, v in pk.items() if isinstance(v, dict) and tuple(v.get("shape", ())) == sh][0]
    p = to_device(pk[key])
    x = torch.randn(1, sh[1], device="cuda", dtype=torch.float16) * 0.1
    ref = packed_gemv_v4(x, p, num_warps=1)
    nm = packed_gemv_nomask(x, p, num_warps=1)
    print(f"\n{name} {sh} diff: {(ref - nm).abs().max().item():.1e}")
    t_v4 = bench(lambda: sink.add_(packed_gemv_v4(x, p, num_warps=1)[0, 0]))
    t_nm = bench(lambda: sink.add_(packed_gemv_nomask(x, p, num_warps=1)[0, 0]))
    print(f"  v4:     {t_v4*1000:6.1f} us ({t_v4:.4f} ms)")
    print(f"  nomask: {t_nm*1000:6.1f} us ({t_nm:.4f} ms) -> {t_v4/t_nm:.2f}x speedup")

import torch
import triton
import triton.language as tl
import time, statistics, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import to_device
from gemv_triton_v4 import packed_gemv_v4, _gemv_v4

# Let's write a tuned kernel: v5 with BR=2 (two rows per program)
@triton.jit
def _gemv_v5_br2(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1c_ptr, base_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_base_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr,
):
    pid = tl.program_id(0)
    r0 = pid * 2
    r1 = r0 + 1
    if r0 >= OC:
        return
    r1_valid = r1 < OC

    NBYTE: tl.constexpr = BS // 4
    acc0 = tl.zeros((), dtype=tl.float32)
    acc1 = tl.zeros((), dtype=tl.float32)
    bi = tl.arange(0, NBYTE)
    sh = (tl.arange(0, 4) * 2).to(tl.uint8)
    off = tl.arange(0, BS)

    for b in range(NB):
        c = b * BS + off
        valid = c < IC
        xv = tl.load(x_ptr + c, mask=valid, other=0.0).to(tl.float32)

        # --- Row 0 ---
        mb0 = tl.load(maskid_ptr + r0 * s_mid_r + b * NBYTE + bi)
        tb0 = tl.load(t0_ptr + r0 * s_t0_r + b * NBYTE + bi)
        mid0 = tl.reshape((mb0[:, None] >> sh[None, :]) & 3, (BS,))
        t0c0 = tl.reshape((tb0[:, None] >> sh[None, :]) & 3, (BS,))
        T0_0 = tl.where(t0c0 == 1, 1.0, tl.where(t0c0 == 2, -1.0, 0.0))

        is_o2_0 = (mid0 < 2) & valid
        io0 = is_o2_0.to(tl.int32)
        local0 = tl.cumsum(io0, axis=0) - io0
        base0 = tl.load(base_ptr + r0 * s_base_r + b)
        t1i0 = base0 + local0
        t1b0 = tl.load(t1c_ptr + (t1i0 >> 2), mask=is_o2_0, other=0)
        t1c_0 = (t1b0 >> ((t1i0 & 3) * 2).to(tl.uint8)) & 3
        T1_0 = tl.where(is_o2_0, tl.where(t1c_0 == 1, 1.0, tl.where(t1c_0 == 2, -1.0, 0.0)), 0.0)

        mu0_0 = tl.load(mu_ptr + 0*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        mu1_0 = tl.load(mu_ptr + 1*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        mu2_0 = tl.load(mu_ptr + 2*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        mu3_0 = tl.load(mu_ptr + 3*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        p0_0 = tl.load(a0_ptr + 0*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        p1_0 = tl.load(a0_ptr + 1*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        p2_0 = tl.load(a0_ptr + 2*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        p3_0 = tl.load(a0_ptr + 3*s_mu_m + r0*s_mu_r + b).to(tl.float32)
        q0_0 = tl.load(a1_ptr + 0*s_a1_m + r0*s_a1_r + b).to(tl.float32)
        q1_0 = tl.load(a1_ptr + 1*s_a1_m + r0*s_a1_r + b).to(tl.float32)
        MU_0 = tl.where(mid0 == 0, mu0_0, tl.where(mid0 == 1, mu1_0, tl.where(mid0 == 2, mu2_0, mu3_0)))
        A0_0 = tl.where(mid0 == 0, p0_0, tl.where(mid0 == 1, p1_0, tl.where(mid0 == 2, p2_0, p3_0)))
        A1_0 = tl.where(mid0 == 0, q0_0, tl.where(mid0 == 1, q1_0, 0.0))

        W0 = MU_0 + A0_0 * T0_0 + A1_0 * T1_0
        acc0 += tl.sum(tl.where(valid, W0 * xv, 0.0), axis=0)

        # --- Row 1 ---
        if r1_valid:
            mb1 = tl.load(maskid_ptr + r1 * s_mid_r + b * NBYTE + bi)
            tb1 = tl.load(t0_ptr + r1 * s_t0_r + b * NBYTE + bi)
            mid1 = tl.reshape((mb1[:, None] >> sh[None, :]) & 3, (BS,))
            t0c1 = tl.reshape((tb1[:, None] >> sh[None, :]) & 3, (BS,))
            T0_1 = tl.where(t0c1 == 1, 1.0, tl.where(t0c1 == 2, -1.0, 0.0))

            is_o2_1 = (mid1 < 2) & valid
            io1 = is_o2_1.to(tl.int32)
            local1 = tl.cumsum(io1, axis=0) - io1
            base1 = tl.load(base_ptr + r1 * s_base_r + b)
            t1i1 = base1 + local1
            t1b1 = tl.load(t1c_ptr + (t1i1 >> 2), mask=is_o2_1, other=0)
            t1c_1 = (t1b1 >> ((t1i1 & 3) * 2).to(tl.uint8)) & 3
            T1_1 = tl.where(is_o2_1, tl.where(t1c_1 == 1, 1.0, tl.where(t1c_1 == 2, -1.0, 0.0)), 0.0)

            mu0_1 = tl.load(mu_ptr + 0*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            mu1_1 = tl.load(mu_ptr + 1*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            mu2_1 = tl.load(mu_ptr + 2*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            mu3_1 = tl.load(mu_ptr + 3*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            p0_1 = tl.load(a0_ptr + 0*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            p1_1 = tl.load(a0_ptr + 1*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            p2_1 = tl.load(a0_ptr + 2*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            p3_1 = tl.load(a0_ptr + 3*s_mu_m + r1*s_mu_r + b).to(tl.float32)
            q0_1 = tl.load(a1_ptr + 0*s_a1_m + r1*s_a1_r + b).to(tl.float32)
            q1_1 = tl.load(a1_ptr + 1*s_a1_m + r1*s_a1_r + b).to(tl.float32)
            MU_1 = tl.where(mid1 == 0, mu0_1, tl.where(mid1 == 1, mu1_1, tl.where(mid1 == 2, mu2_1, mu3_1)))
            A0_1 = tl.where(mid1 == 0, p0_1, tl.where(mid1 == 1, p1_1, tl.where(mid1 == 2, p2_1, p3_1)))
            A1_1 = tl.where(mid1 == 0, q0_1, tl.where(mid1 == 1, q1_1, 0.0))

            W1 = MU_1 + A0_1 * T0_1 + A1_1 * T1_1
            acc1 += tl.sum(tl.where(valid, W1 * xv, 0.0), axis=0)

    if HAS_BIAS:
        acc0 += tl.load(bias_ptr + r0).to(tl.float32)
        if r1_valid:
            acc1 += tl.load(bias_ptr + r1).to(tl.float32)
    tl.store(out_ptr + r0, acc0.to(tl.float16))
    if r1_valid:
        tl.store(out_ptr + r1, acc1.to(tl.float16))

def packed_gemv_v5(x, p, bias=None, num_warps=1, num_stages=2):
    OC, IC = p["shape"]
    NB, BS = int(p["nblocks"]), int(p["blocksize"])
    xf = x.reshape(-1)
    out = torch.empty(OC, device=xf.device, dtype=torch.float16)
    grid = (triton.cdiv(OC, 2),)
    _gemv_v5_br2[grid](
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

print("Comparing v4_w1 vs v5_br2 on down_proj [2560, 9728]:")
key = [k for k, v in pk.items() if isinstance(v, dict) and tuple(v.get("shape", ())) == (2560, 9728)][0]
p = to_device(pk[key])
x = torch.randn(1, 9728, device="cuda", dtype=torch.float16) * 0.1

ref = packed_gemv_v4(x, p, num_warps=1, num_stages=3)
out_v5 = packed_gemv_v5(x, p, num_warps=1, num_stages=2)
err = (out_v5 - ref).abs().max().item()
print(f"  v5_br2 numerical error vs v4_w1: {err:.1e}")

sink = torch.zeros(1, device="cuda")
ms_v4 = bench(lambda: sink.add_(packed_gemv_v4(x, p, num_warps=1, num_stages=3)[0, 0]))
print(f"  v4_w1 (stages=3): {ms_v4*1000:.1f} us ({ms_v4:.4f} ms)")

for nw in [1, 2]:
    for ns in [1, 2, 3]:
        ms_v5 = bench(lambda nw=nw, ns=ns: sink.add_(packed_gemv_v5(x, p, num_warps=nw, num_stages=ns)[0, 0]))
        print(f"  v5_br2 (warps={nw}, stages={ns}): {ms_v5*1000:.1f} us ({ms_v5:.4f} ms)")

print("\nComparing on gate_proj [9728, 2560]:")
key_g = [k for k, v in pk.items() if isinstance(v, dict) and tuple(v.get("shape", ())) == (9728, 2560)][0]
p_g = to_device(pk[key_g])
x_g = torch.randn(1, 2560, device="cuda", dtype=torch.float16) * 0.1

ms_v4_g = bench(lambda: sink.add_(packed_gemv_v4(x_g, p_g, num_warps=1, num_stages=3)[0, 0]))
print(f"  v4_w1 (stages=3): {ms_v4_g*1000:.1f} us ({ms_v4_g:.4f} ms)")

for nw in [1, 2]:
    for ns in [1, 2, 3]:
        ms_v5_g = bench(lambda nw=nw, ns=ns: sink.add_(packed_gemv_v5(x_g, p_g, num_warps=nw, num_stages=ns)[0, 0]))
        print(f"  v5_br2 (warps={nw}, stages={ns}): {ms_v5_g*1000:.1f} us ({ms_v5_g:.4f} ms)")

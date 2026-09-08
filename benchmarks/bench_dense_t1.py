import torch
import triton
import triton.language as tl
import time, statistics, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import to_device
from gemv_triton_v4 import packed_gemv_v4

# Let's write the dense-T1 kernel:
@triton.jit
def _gemv_dense_t1(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_t1_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr,
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
        t1b = tl.load(t1_ptr + r * s_t1_r + b * NBYTE + bi)

        mid = tl.reshape((mb[:, None] >> sh[None, :]) & 3, (BS,))
        t0c = tl.reshape((tb[:, None] >> sh[None, :]) & 3, (BS,))
        t1c = tl.reshape((t1b[:, None] >> sh[None, :]) & 3, (BS,))

        T0 = tl.where(t0c == 1, 1.0, tl.where(t0c == 2, -1.0, 0.0))
        T1 = tl.where(t1c == 1, 1.0, tl.where(t1c == 2, -1.0, 0.0))

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
        xv = tl.load(x_ptr + b * BS + off).to(tl.float32)
        acc += tl.sum(W * xv, axis=0)

    if HAS_BIAS: acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))


def unpack_t1_dense(p):
    """Unpack compacted T1c into dense [OC, IC // 4] uint8."""
    OC, IC = p["shape"]
    mid_u = torch.stack([p["maskid"] & 3, (p["maskid"] >> 2) & 3, (p["maskid"] >> 4) & 3, (p["maskid"] >> 6) & 3], -1).reshape(OC, -1)[:, :IC]
    is_o2 = mid_u < 2
    pad = p["nblocks"] * p["blocksize"] - IC
    o2p = torch.nn.functional.pad(is_o2, (0, pad), value=False) if pad else is_o2
    blk = o2p.view(OC, p["nblocks"], p["blocksize"]).to(torch.int64)
    local = torch.cumsum(blk, -1) - blk
    idx = (p["base"].to(torch.int64)[:, :, None] + local).view(OC, -1)[:, :IC]
    byte = p["T1c"][(idx >> 2).clamp_(0, p["T1c"].numel() - 1)]
    code = (byte >> ((idx & 3) * 2).to(torch.uint8)) & 3
    t1_codes = torch.where(is_o2, code, torch.zeros_like(code))  # [OC, IC] uint8
    # Pack 4 codes per byte into [OC, IC // 4] uint8
    t1_pad = torch.nn.functional.pad(t1_codes, (0, pad), value=0) if pad else t1_codes
    b0 = t1_pad[:, 0::4]; b1 = t1_pad[:, 1::4]; b2 = t1_pad[:, 2::4]; b3 = t1_pad[:, 3::4]
    t1_packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
    return t1_packed.contiguous()


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

print("=== Dense-T1 Benchmark vs v4_w1 ===")
for name, sh in [("down_proj", (2560, 9728)), ("gate_proj", (9728, 2560)), ("o_proj", (2560, 4096)), ("k_proj", (1024, 2560))]:
    key = [k for k, v in pk.items() if isinstance(v, dict) and tuple(v.get("shape", ())) == sh][0]
    p = to_device(pk[key])
    OC, IC = p["shape"]
    NB, BS = int(p["nblocks"]), int(p["blocksize"])
    x = torch.randn(1, IC, device="cuda", dtype=torch.float16) * 0.1
    xf = x.reshape(-1)
    
    t1_dense = unpack_t1_dense(p)
    out_dense = torch.empty(OC, device="cuda", dtype=torch.float16)
    
    # Check numerical accuracy
    ref = packed_gemv_v4(x, p, num_warps=1)
    _gemv_dense_t1[(OC,)](
        xf, out_dense, xf,
        p["maskid"], p["T0"], t1_dense,
        p["mu"], p["a0"], p["a1"],
        OC, IC, NB,
        p["maskid"].stride(0), p["T0"].stride(0), t1_dense.stride(0),
        p["mu"].stride(0), p["mu"].stride(1), p["a1"].stride(0), p["a1"].stride(1),
        HAS_BIAS=False, BS=BS, num_warps=1, num_stages=3
    )
    diff = (out_dense.reshape(1, OC) - ref).abs().max().item()
    
    t_v4 = bench(lambda: packed_gemv_v4(x, p, num_warps=1))
    t_dense = bench(lambda: _gemv_dense_t1[(OC,)](
        xf, out_dense, xf,
        p["maskid"], p["T0"], t1_dense,
        p["mu"], p["a0"], p["a1"],
        OC, IC, NB,
        p["maskid"].stride(0), p["T0"].stride(0), t1_dense.stride(0),
        p["mu"].stride(0), p["mu"].stride(1), p["a1"].stride(0), p["a1"].stride(1),
        HAS_BIAS=False, BS=BS, num_warps=1, num_stages=3
    ))
    
    print(f"\n{name} {sh}: numerical diff = {diff:.1e}")
    print(f"  v4_w1 (gather T1):   {t_v4*1000:6.1f} us ({t_v4:.4f} ms)")
    print(f"  dense_t1 (coalesced): {t_dense*1000:6.1f} us ({t_dense:.4f} ms) -> {t_v4/t_dense:.2f}x speedup!")

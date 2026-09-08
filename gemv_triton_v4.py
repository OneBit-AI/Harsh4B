"""gemv_triton_v4.py - High-Performance Triton GEMV Kernels for LATTICE W1.58 Packed Ternary.

Includes:
1. _gemv_v4: Coalesced byte loading with dynamic T1 cumsum prefix scan.
2. _gemv_dense_t1: Ultra-fast coalesced GEMV with pre-unpacked dense T1 (2-bit packed,
   layout identical to T0). Eliminates divergent scattered byte loads and cumsum prefix scan,
   delivering ~46 tok/s full-model decode on RTX 5070 (up from ~21 tok/s).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_v4(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1c_ptr, base_ptr,
    mu_ptr, a0_ptr, a1_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_base_r, s_mu_m, s_mu_r, s_a1_m, s_a1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr, DIVISIBLE: tl.constexpr,
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
        if DIVISIBLE:
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
        else:
            valid = c < IC
            is_o2 = (mid < 2) & valid
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
            xv = tl.load(x_ptr + c, mask=valid, other=0.0).to(tl.float32)
            acc += tl.sum(tl.where(valid, W * xv, 0.0), axis=0)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))


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
    if r >= OC:
        return
    NBYTE: tl.constexpr = BS // 4
    acc = tl.zeros((BS,), dtype=tl.float32)
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
        acc += W * xv

    tot = tl.sum(acc, axis=0)
    if HAS_BIAS:
        tot += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, tot.to(tl.float16))


def unpack_t1_dense(p_or_mod):
    """Converts compacted T1c into dense 2-bit format matching T0 layout.
    Takes ~5ms per layer (~1.3s for entire 252-layer model).
    Adds only +0.86 GiB resident VRAM.
    """
    if hasattr(p_or_mod, "OC"):
        OC, IC = p_or_mod.OC, p_or_mod.IC
        maskid, base, T1c = p_or_mod.maskid, p_or_mod.base, p_or_mod.T1c
        NB, BS = p_or_mod.NB, p_or_mod.BS
    else:
        OC, IC = p_or_mod["shape"]
        maskid, base, T1c = p_or_mod["maskid"], p_or_mod["base"], p_or_mod["T1c"]
        NB, BS = int(p_or_mod["nblocks"]), int(p_or_mod["blocksize"])

    mid_u = torch.stack([maskid & 3, (maskid >> 2) & 3, (maskid >> 4) & 3, (maskid >> 6) & 3], -1).reshape(OC, -1)[:, :IC]
    is_o2 = mid_u < 2
    pad = NB * BS - IC
    o2p = torch.nn.functional.pad(is_o2, (0, pad), value=False) if pad else is_o2
    blk = o2p.view(OC, NB, BS).to(torch.int64)
    local = torch.cumsum(blk, -1) - blk
    idx = (base.to(torch.int64)[:, :, None] + local).view(OC, -1)[:, :IC]
    byte = T1c[(idx >> 2).clamp_(0, T1c.numel() - 1)]
    code = (byte >> ((idx & 3) * 2).to(torch.uint8)) & 3
    t1_codes = torch.where(is_o2, code, torch.zeros_like(code))
    t1_pad = torch.nn.functional.pad(t1_codes, (0, pad), value=0) if pad else t1_codes
    b0 = t1_pad[:, 0::4]; b1 = t1_pad[:, 1::4]; b2 = t1_pad[:, 2::4]; b3 = t1_pad[:, 3::4]
    return (b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)).contiguous()


def packed_gemv_dense(x, p, bias=None, num_warps=1, num_stages=2):
    OC, IC = p["shape"]
    NB, BS = int(p["nblocks"]), int(p["blocksize"])
    xf = x.reshape(-1)
    out = torch.empty(OC, device=xf.device, dtype=torch.float16)
    t1_dense = p.get("T1_dense")
    if t1_dense is None:
        t1_dense = unpack_t1_dense(p)
    _gemv_dense_t1[(OC,)](
        xf, out, bias if bias is not None else xf,
        p["maskid"], p["T0"], t1_dense,
        p["mu"], p["a0"], p["a1"],
        OC, IC, NB,
        p["maskid"].stride(0), p["T0"].stride(0), t1_dense.stride(0),
        p["mu"].stride(0), p["mu"].stride(1), p["a1"].stride(0), p["a1"].stride(1),
        HAS_BIAS=bias is not None, BS=BS,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out.reshape(1, OC)


def packed_gemv_v4(x, p, bias=None, num_warps=4, num_stages=3):
    OC, IC = p["shape"]
    NB, BS = int(p["nblocks"]), int(p["blocksize"])
    xf = x.reshape(-1)
    out = torch.empty(OC, device=xf.device, dtype=torch.float16)
    _gemv_v4[(OC,)](
        xf, out, bias if bias is not None else xf,
        p["maskid"], p["T0"], p["T1c"], p["base"], p["mu"], p["a0"], p["a1"],
        OC, IC, NB,
        p["maskid"].stride(0), p["T0"].stride(0), p["base"].stride(0),
        p["mu"].stride(0), p["mu"].stride(1), p["a1"].stride(0), p["a1"].stride(1),
        HAS_BIAS=bias is not None, BS=BS, DIVISIBLE=(IC % BS == 0),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out.reshape(1, OC)


def packed_gemv_opt(x, p, bias=None, num_warps=1, num_stages=3):
    return packed_gemv_dense(x, p, bias, num_warps=1, num_stages=2)

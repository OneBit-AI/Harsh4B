import torch, time, statistics
import triton, triton.language as tl
from build_packed_model import build
from packed_linear import PackedLinear

print("Loading model layer...")
R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

m, _ = build(mid, pack, rot, verbose=False)

# Get down_proj (the largest layer [2560, 9728])
layer = None
for name, mod in m.named_modules():
    if isinstance(mod, PackedLinear) and mod.OC == 2560 and mod.IC == 9728:
        layer = mod
        print(f"Testing on {name} [{layer.OC}, {layer.IC}], NB={layer.NB}, BS={layer.BS}")
        break

# Unpack dense T1
def unpack_t1_dense(mod):
    OC, IC = mod.OC, mod.IC
    mid_u = torch.stack([mod.maskid & 3, (mod.maskid >> 2) & 3, (mod.maskid >> 4) & 3, (mod.maskid >> 6) & 3], -1).reshape(OC, -1)[:, :IC]
    is_o2 = mid_u < 2
    pad = mod.NB * mod.BS - IC
    o2p = torch.nn.functional.pad(is_o2, (0, pad), value=False) if pad else is_o2
    blk = o2p.view(OC, mod.NB, mod.BS).to(torch.int64)
    local = torch.cumsum(blk, -1) - blk
    idx = (mod.base.to(torch.int64)[:, :, None] + local).view(OC, -1)[:, :IC]
    byte = mod.T1c[(idx >> 2).clamp_(0, mod.T1c.numel() - 1)]
    code = (byte >> ((idx & 3) * 2).to(torch.uint8)) & 3
    t1_codes = torch.where(is_o2, code, torch.zeros_like(code))
    t1_pad = torch.nn.functional.pad(t1_codes, (0, pad), value=0) if pad else t1_codes
    b0 = t1_pad[:, 0::4]; b1 = t1_pad[:, 1::4]; b2 = t1_pad[:, 2::4]; b3 = t1_pad[:, 3::4]
    return (b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)).contiguous()

t1_dense = unpack_t1_dense(layer)

# Baseline dense-T1 kernel (current 42.7 TPS version)
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


# Packed Scales: Combine mu (4), a0 (4), a1 (2) padded to 16 halfs per (r, b)
# Shape: [OC, NB, 16] float16 -> 32 bytes per (r, b)
pad6 = torch.zeros(6, layer.OC, layer.NB, device=layer.mu.device, dtype=layer.mu.dtype)
scales_16 = torch.cat([layer.mu, layer.a0, layer.a1, pad6], dim=0).permute(1, 2, 0).contiguous()
print(f"Packed scales shape: {scales_16.shape}, stride: {scales_16.stride()}")

@triton.jit
def _gemv_packed_scales(
    x_ptr, out_ptr, bias_ptr,
    maskid_ptr, t0_ptr, t1_ptr,
    scales_ptr,
    OC, IC, NB,
    s_mid_r, s_t0_r, s_t1_r,
    HAS_BIAS: tl.constexpr, BS: tl.constexpr,
):
    r = tl.program_id(0)
    if r >= OC: return
    NBYTE: tl.constexpr = BS // 4
    acc = tl.zeros((), dtype=tl.float32)
    bi = tl.arange(0, NBYTE)
    sh = (tl.arange(0, 4) * 2).to(tl.uint8)
    off = tl.arange(0, BS)
    si = tl.arange(0, 16)

    for b in range(NB):
        mb = tl.load(maskid_ptr + r * s_mid_r + b * NBYTE + bi)
        tb = tl.load(t0_ptr + r * s_t0_r + b * NBYTE + bi)
        t1b = tl.load(t1_ptr + r * s_t1_r + b * NBYTE + bi)

        mid = tl.reshape((mb[:, None] >> sh[None, :]) & 3, (BS,))
        t0c = tl.reshape((tb[:, None] >> sh[None, :]) & 3, (BS,))
        t1c = tl.reshape((t1b[:, None] >> sh[None, :]) & 3, (BS,))

        T0 = tl.where(t0c == 1, 1.0, tl.where(t0c == 2, -1.0, 0.0))
        T1 = tl.where(t1c == 1, 1.0, tl.where(t1c == 2, -1.0, 0.0))

        # Vector load all 16 scale elements in 1 instruction
        s = tl.load(scales_ptr + (r * NB + b) * 16 + si).to(tl.float32)
        mu0 = tl.load(scales_ptr + (r * NB + b) * 16 + 0).to(tl.float32) # or from s
        # In Triton, s[0] indexing:
        mu0 = tl.sum(tl.where(si == 0, s, 0.0))
        mu1 = tl.sum(tl.where(si == 1, s, 0.0))
        mu2 = tl.sum(tl.where(si == 2, s, 0.0))
        mu3 = tl.sum(tl.where(si == 3, s, 0.0))
        p0 = tl.sum(tl.where(si == 4, s, 0.0))
        p1 = tl.sum(tl.where(si == 5, s, 0.0))
        p2 = tl.sum(tl.where(si == 6, s, 0.0))
        p3 = tl.sum(tl.where(si == 7, s, 0.0))
        q0 = tl.sum(tl.where(si == 8, s, 0.0))
        q1 = tl.sum(tl.where(si == 9, s, 0.0))

        MU = tl.where(mid == 0, mu0, tl.where(mid == 1, mu1, tl.where(mid == 2, mu2, mu3)))
        A0 = tl.where(mid == 0, p0, tl.where(mid == 1, p1, tl.where(mid == 2, p2, p3)))
        A1 = tl.where(mid == 0, q0, tl.where(mid == 1, q1, 0.0))

        W = MU + A0 * T0 + A1 * T1
        xv = tl.load(x_ptr + b * BS + off).to(tl.float32)
        acc += tl.sum(W * xv, axis=0)

    if HAS_BIAS: acc += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, acc.to(tl.float16))


x = torch.randn(layer.IC, device="cuda", dtype=torch.float16)
out_ref = torch.empty(layer.OC, device="cuda", dtype=torch.float16)
out_test = torch.empty(layer.OC, device="cuda", dtype=torch.float16)

# Test stage tuning on current dense_t1
for nw in [1, 2, 4]:
    for ns in [1, 2, 3, 4]:
        try:
            # Warmup
            for _ in range(10):
                _gemv_dense_t1[(layer.OC,)](
                    x, out_ref, layer.bias if layer.bias is not None else x,
                    layer.maskid, layer.T0, t1_dense,
                    layer.mu, layer.a0, layer.a1,
                    layer.OC, layer.IC, layer.NB,
                    layer.maskid.stride(0), layer.T0.stride(0), t1_dense.stride(0),
                    layer.mu.stride(0), layer.mu.stride(1), layer.a1.stride(0), layer.a1.stride(1),
                    HAS_BIAS=layer.bias is not None, BS=layer.BS,
                    num_warps=nw, num_stages=ns
                )
            torch.cuda.synchronize()
            t0 = time.time()
            N = 200
            for _ in range(N):
                _gemv_dense_t1[(layer.OC,)](
                    x, out_ref, layer.bias if layer.bias is not None else x,
                    layer.maskid, layer.T0, t1_dense,
                    layer.mu, layer.a0, layer.a1,
                    layer.OC, layer.IC, layer.NB,
                    layer.maskid.stride(0), layer.T0.stride(0), t1_dense.stride(0),
                    layer.mu.stride(0), layer.mu.stride(1), layer.a1.stride(0), layer.a1.stride(1),
                    HAS_BIAS=layer.bias is not None, BS=layer.BS,
                    num_warps=nw, num_stages=ns
                )
            torch.cuda.synchronize()
            us = (time.time() - t0) / N * 1e6
            print(f"dense_t1: num_warps={nw}, num_stages={ns} -> {us:.1f} us")
        except Exception as e:
            print(f"dense_t1: num_warps={nw}, num_stages={ns} -> failed: {e}")

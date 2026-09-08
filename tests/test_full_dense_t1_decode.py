import sys, os, time, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_packed_model import build
from packed_linear import PackedLinear
from gemm_triton import packed_gemm
from transformers import AutoTokenizer, StaticCache
import triton, triton.language as tl

# Define the dense-T1 kernel
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
    if HAS_BIAS: tot += tl.load(bias_ptr + r).to(tl.float32)
    tl.store(out_ptr + r, tot.to(tl.float16))


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


R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

print("1. Loading packed Qwen3-4B...", flush=True)
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

print("2. Converting T1c -> T1_dense across all 252 layers...", flush=True)
t0 = time.time()
with torch.no_grad():
    for mod in m.modules():
        if isinstance(mod, PackedLinear):
            t1_dense = unpack_t1_dense(mod)
            mod.register_buffer("T1_dense", t1_dense, persistent=False)
            
            # Patch forward to use dense-T1 kernel for decode
            def make_fwd(layer):
                def fwd(x):
                    in_dtype = x.dtype
                    init_shape = x.shape
                    xh = x.half() if in_dtype != torch.float16 else x
                    if layer.L is not None:
                        xh = xh.reshape(-1, layer.dim_l, layer.dim_r)
                        xh = layer.L @ xh @ layer.R
                        xh = xh.reshape(init_shape)
                    ntok = xh.numel() // layer.IC
                    if ntok == 1:
                        xf = xh.reshape(-1)
                        out = torch.empty(layer.OC, device=xf.device, dtype=torch.float16)
                        _gemv_dense_t1[(layer.OC,)](
                            xf, out, layer.bias if layer.bias is not None else xf,
                            layer.maskid, layer.T0, layer.T1_dense,
                            layer.mu, layer.a0, layer.a1,
                            layer.OC, layer.IC, layer.NB,
                            layer.maskid.stride(0), layer.T0.stride(0), layer.T1_dense.stride(0),
                            layer.mu.stride(0), layer.mu.stride(1), layer.a1.stride(0), layer.a1.stride(1),
                            HAS_BIAS=layer.bias is not None, BS=layer.BS,
                            num_warps=1, num_stages=2
                        )
                        out = out.reshape(1, layer.OC)
                    else:
                        out = packed_gemm(xh, layer._p(), layer.bias, NT=16)
                    out = out.reshape(*init_shape[:-1], layer.OC)
                    return out.to(in_dtype) if in_dtype != torch.float16 else out
                return fwd
            mod.forward = make_fwd(mod)

torch.cuda.empty_cache()
vram = torch.cuda.memory_allocated() / (1024**3)
print(f"   Done in {time.time()-t0:.2f}s | Resident VRAM: {vram:.2f} GiB / 12.0 GiB", flush=True)

# Test decode speed with StaticCache + CUDA Graph
print("3. Capturing CUDA Graph with Dense-T1 kernel...", flush=True)
ids = tok("The capital of France is", return_tensors="pt").to("cuda")
plen = ids["input_ids"].shape[1]
maxlen = 256
n_new = 64

cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=maxlen,
                    device="cuda", dtype=torch.float16)

with torch.no_grad():
    out = m(**ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(plen, device="cuda"))
    nxt = out.logits[:, -1:].argmax(-1)
    static_in = nxt.clone()
    static_pos = torch.tensor([plen], device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(input_ids=static_in, past_key_values=cache, use_cache=True,
              cache_position=static_pos)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = m(input_ids=static_in, past_key_values=cache, use_cache=True,
                       cache_position=static_pos).logits

    print("   CUDA Graph ready. Running 5 decode benchmark rounds (64 tokens each)...", flush=True)
    def run_bench():
        torch.cuda.synchronize()
        t0 = time.time()
        cur = nxt.clone()
        toks = [int(nxt)]
        for i in range(n_new - 1):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur = static_out[:, -1:].argmax(-1)
            toks.append(int(cur))
        torch.cuda.synchronize()
        return (n_new - 1) / (time.time() - t0), toks

    # Warmup
    run_bench()

    runs = []
    for r in range(5):
        tps, toks = run_bench()
        runs.append(tps)

    print(f"\n=======================================================", flush=True)
    print(f"  DENSE-T1 COALESCED KERNEL + CUDA GRAPH DECODE:")
    print(f"  MEDIAN TPS: {statistics.median(runs):.2f} tok/s")
    print(f"  ALL RUNS:   {', '.join(f'{q:.2f}' for q in sorted(runs))}")
    print(f"  LATENCY:    {1000/statistics.median(runs):.2f} ms/token")
    print(f"  VRAM:       {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    print(f"  OUTPUT:     {tok.decode(toks, skip_special_tokens=True)[:100]!r}")
    print(f"=======================================================", flush=True)

import sys, os, time, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_packed_model import build
from packed_linear import PackedLinear
from transformers import AutoTokenizer, StaticCache
import torch.nn.functional as F

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

print("1. Building packed model...", flush=True)
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

print("2. Dequantizing layers...", flush=True)
with torch.no_grad():
    for mod in m.modules():
        if isinstance(mod, PackedLinear):
            mod.register_buffer("W_fp16", mod.dequant(), persistent=False)
            def make_fwd(layer):
                def fwd(x):
                    in_dtype = x.dtype
                    init_shape = x.shape
                    xh = x.half() if in_dtype != torch.float16 else x
                    if layer.L is not None:
                        xh = xh.reshape(-1, layer.dim_l, layer.dim_r)
                        xh = layer.L @ xh @ layer.R
                        xh = xh.reshape(init_shape)
                    out = F.linear(xh.reshape(-1, layer.IC), layer.W_fp16, layer.bias)
                    out = out.reshape(*init_shape[:-1], layer.OC)
                    return out.to(in_dtype) if in_dtype != torch.float16 else out
                return fwd
            mod.forward = make_fwd(mod)
            del mod.maskid, mod.T0, mod.T1c, mod.base, mod.mu, mod.a0, mod.a1

torch.cuda.empty_cache()
print("3. Pre-allocating StaticCache...", flush=True)
ids = tok("The capital of France is", return_tensors="pt").to("cuda")
plen = ids["input_ids"].shape[1]
maxlen = 256
cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=maxlen,
                    device="cuda", dtype=torch.float16)

print("4. Running prefill forward...", flush=True)
with torch.no_grad():
    out = m(**ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(plen, device="cuda"))
    nxt = out.logits[:, -1:].argmax(-1)
    print(f"   Prefill done. First token: {tok.decode([int(nxt)])!r}", flush=True)

    print("5. Capturing CUDA Graph...", flush=True)
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
    print("   CUDA Graph captured successfully!", flush=True)

    print("6. Benchmarking 64 decode tokens...", flush=True)
    def run_bench():
        torch.cuda.synchronize()
        t0 = time.time()
        cur = nxt.clone()
        toks = [int(nxt)]
        for i in range(63):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur = static_out[:, -1:].argmax(-1)
            toks.append(int(cur))
        torch.cuda.synchronize()
        dt = time.time() - t0
        return 63 / dt, toks

    tps, toks = run_bench()
    print(f"   Warmup: {tps:.2f} tok/s")
    runs = [run_bench()[0] for _ in range(3)]
    print(f"\n==========================================")
    print(f"  UNPACKED CUBLAS + CUDA GRAPH DECODE:")
    print(f"  MEDIAN TPS: {statistics.median(runs):.2f} tok/s")
    print(f"  RUNS: {runs}")
    print(f"  TEXT: {tok.decode(toks, skip_special_tokens=True)!r}")
    print(f"==========================================")

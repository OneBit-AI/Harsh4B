import sys, os, time, statistics, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_packed_model import build
from packed_linear import PackedLinear
from transformers import AutoTokenizer, StaticCache
import torch.nn.functional as F

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

print("Building packed model...", flush=True)
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

vram_packed = torch.cuda.memory_allocated() / (1024**3)
print(f"Packed VRAM: {vram_packed:.2f} GiB", flush=True)

print("Pre-dequantizing all 252 PackedLinear layers to fp16 weights in VRAM...", flush=True)
t0 = time.time()
n_unpacked = 0
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
            n_unpacked += 1

torch.cuda.empty_cache()
vram_unpacked = torch.cuda.memory_allocated() / (1024**3)
print(f"Dequantized {n_unpacked} layers in {time.time()-t0:.2f}s | Unpacked resident VRAM: {vram_unpacked:.2f} GiB / 12.0 GiB", flush=True)

@torch.no_grad()
def main():
    ids = tok("The capital of France is", return_tensors="pt").to("cuda")
    plen = ids["input_ids"].shape[1]
    maxlen = 256
    n_new = 64

    cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=maxlen,
                        device="cuda", dtype=torch.float16)

    out = m(**ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(plen, device="cuda"))
    nxt = out.logits[:, -1:].argmax(-1)
    static_in = nxt.clone()
    static_pos = torch.tensor([plen], device="cuda")

    # Warmup & capture graph
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

    def run_bench():
        torch.cuda.synchronize()
        t_start = time.time()
        toks = [int(nxt)]
        cur = nxt.clone()
        for i in range(n_new - 1):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur = static_out[:, -1:].argmax(-1)
            toks.append(int(cur))
        torch.cuda.synchronize()
        return (n_new - 1) / (time.time() - t_start), toks

    # Warmup
    run_bench()

    runs = []
    for _ in range(5):
        tps, toks = run_bench()
        runs.append(tps)

    print(f"\n=======================================================", flush=True)
    print(f"  Unpacked 4B Decode under CUDA Graph (5 runs):", flush=True)
    print(f"  TPS: {statistics.median(runs):.2f} tok/s  (runs: {', '.join(f'{q:.2f}' for q in sorted(runs))})", flush=True)
    print(f"  Latency: {1000/statistics.median(runs):.2f} ms/token", flush=True)
    print(f"  Output: {tok.decode(toks, skip_special_tokens=True)[:100]!r}", flush=True)
    print(f"=======================================================", flush=True)

if __name__ == "__main__":
    main()

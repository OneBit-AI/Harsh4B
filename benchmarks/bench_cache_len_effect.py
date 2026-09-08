import time
import torch
from transformers import AutoTokenizer, StaticCache
from build_packed_model import build
from packed_linear import set_kernel

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

set_kernel("dense")
m, _ = build(mid, pack, rot, verbose=False)

for maxlen in [512, 1024, 2048, 4096]:
    cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=maxlen, device="cuda", dtype=torch.float16)
    static_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(3):
            m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos)
    torch.cuda.current_stream().wait_stream(s)
    
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        static_out = m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos).logits
        
    # Benchmark 50 steps
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(50):
        static_pos.fill_(i)
        g.replay()
    torch.cuda.synchronize()
    t_tot = time.time() - t0
    tok_s = 50 / t_tot
    ms_tok = (t_tot / 50) * 1000
    print(f"MAX_CACHE_LEN={maxlen:4d}: {tok_s:.2f} tok/s ({ms_tok:.2f} ms/tok)")

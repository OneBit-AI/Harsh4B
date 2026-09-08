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
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

MAX_CACHE_LEN = 4096
cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN, device="cuda", dtype=torch.float16)

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

# Test generation loop
def sample(logits, gen_tokens):
    l = logits.reshape(1, -1).clone().float()
    if gen_tokens:
        ut = torch.tensor(list(set(gen_tokens)), device=l.device, dtype=torch.long)
        tl = l[0, ut]
        l[0, ut] = torch.where(tl > 0, tl / 1.15, tl * 1.15)
    val, idx = torch.topk(l, 50, sorted=True)
    probs = torch.softmax(val, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = (cum - probs) >= 0.9
    probs[mask] = 0.0
    p_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(p_sum > 0, probs / p_sum, torch.zeros_like(probs))
    s_idx = torch.multinomial(probs, 1)
    return idx.gather(-1, s_idx)

cache.reset()
inp = tok("What is government?", return_tensors="pt").to("cuda")
plen = inp["input_ids"].shape[1]
with torch.no_grad():
    out = m(**inp, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
    cur = sample(out.logits[:, -1:], [])
    
    # Measure 60 tokens pure decode
    N = 60
    t0 = time.time()
    gen = []
    for i in range(N):
        static_in.copy_(cur)
        static_pos.fill_(plen + i)
        g.replay()
        cur = sample(static_out[:, -1:], gen)
        gen.append(int(cur))
    torch.cuda.synchronize()
    t_tot = time.time() - t0
    print(f"Pure in-process decode with ultra-sample: {N} tokens in {t_tot:.3f}s -> {N/t_tot:.2f} tok/s")

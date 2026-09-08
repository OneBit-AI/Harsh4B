import torch, time, statistics
from transformers import AutoTokenizer, StaticCache
from build_packed_model import build
from packed_linear import PackedLinear

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

print("1. Loading model...")
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

MAX_CACHE_LEN = 1024
cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
                    device="cuda", dtype=torch.float16)

static_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")
static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")

print("2. Warmup and capturing persistent CUDA Graph...")
# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(s):
    for _ in range(3):
        m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    static_out = m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos).logits

print("3. Testing 3 consecutive requests reusing the SAME graph...")

def generate(prompt, n_tokens=64):
    cache.reset()
    ids = tok(prompt, return_tensors="pt").to("cuda")
    plen = ids["input_ids"].shape[1]
    with torch.no_grad():
        out = m(**ids, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(plen, device="cuda"))
        nxt = out.logits[:, -1:].argmax(-1)
        
        torch.cuda.synchronize()
        t_decode_start = time.time()
        cur = nxt.clone()
        toks = [int(nxt)]
        for i in range(n_tokens - 1):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur = static_out[:, -1:].argmax(-1)
            toks.append(int(cur))
        torch.cuda.synchronize()
        t_decode = time.time() - t_decode_start
        tps = (n_tokens - 1) / t_decode
        text = tok.decode(toks, skip_special_tokens=True)
        return tps, text

for prompt in ["The capital of France is", "Once upon a time", "In computer science, a binary search tree"]:
    tps, text = generate(prompt, 64)
    print(f"\nPrompt: {prompt!r}")
    print(f"Decode TPS: {tps:.2f} tok/s | Latency: {1000/tps:.2f} ms/tok")
    print(f"Generated text: {text[:80]!r}")

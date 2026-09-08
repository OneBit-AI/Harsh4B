import torch
import time
from transformers import AutoTokenizer, StaticCache
from build_packed_model import build

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

print("Loading model...")
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

MAX_CACHE_LEN = 4096
print(f"Allocating StaticCache with MAX_CACHE_LEN={MAX_CACHE_LEN}...")
cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
                    device="cuda", dtype=torch.float16)

static_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")
static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")

print("Warming up for CUDA graph capture...")
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(s):
    for _ in range(3):
        m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos)
torch.cuda.current_stream().wait_stream(s)

print("Capturing CUDA Graph...")
g = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    static_out = m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos).logits
print("CUDA Graph captured successfully!")

# Now test prefill with a long prompt (1200 tokens)
long_text = "The quick brown fox jumps over the lazy dog. " * 120
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": long_text}
]
try:
    p_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    p_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

ids = tok(p_text, return_tensors="pt").to("cuda")
plen = ids["input_ids"].shape[1]
print(f"Testing with prompt length: {plen} tokens...")

cache.reset()
with torch.no_grad():
    t_prefill_start = time.time()
    out = m(**ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
    t_prefill = time.time() - t_prefill_start
    print(f"Prefill took: {t_prefill:.3f}s")
    
    nxt = out.logits[:, -1:].argmax(-1)
    cur = nxt.clone()
    
    t_dec_start = time.time()
    N_GEN = 100
    for i in range(N_GEN):
        static_in.copy_(cur)
        static_pos.fill_(plen + i)
        g.replay()
        cur = static_out[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t_dec = time.time() - t_dec_start
    
    tok_s = N_GEN / t_dec
    print(f"Generated {N_GEN} tokens in {t_dec:.3f}s -> {tok_s:.2f} tok/s with plen={plen}!")

vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"Total VRAM allocated: {vram_gb:.2f} GiB")

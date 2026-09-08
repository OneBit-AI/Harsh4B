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
print("Loading model...")
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

# Build buckets: 1024 and 2048
BUCKETS = [1024, 2048]
graphs = {}

static_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")
static_pos = torch.zeros((1,), dtype=torch.long, device="cuda")

for cap in BUCKETS:
    print(f"Capturing bucket {cap}...")
    cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=cap, device="cuda", dtype=torch.float16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(3):
            m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos)
    torch.cuda.current_stream().wait_stream(s)
    
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out = m(input_ids=static_in, past_key_values=cache, use_cache=True, cache_position=static_pos).logits
    graphs[cap] = {"cache": cache, "graph": g, "out": out}

vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"All graph buckets captured! Total VRAM: {vram_gb:.2f} GiB")

# Test decode speed on each bucket
for cap in BUCKETS:
    g = graphs[cap]["graph"]
    c = graphs[cap]["cache"]
    c.reset()
    
    # Simulate a prompt of 50 tokens
    prompt = tok("What is government?", return_tensors="pt").to("cuda")
    plen = prompt["input_ids"].shape[1]
    
    with torch.no_grad():
        out = m(**prompt, past_key_values=c, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
        cur = out.logits[:, -1:].argmax(-1)
        
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(60):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur = graphs[cap]["out"][:, -1:].argmax(-1)
        torch.cuda.synchronize()
        tok_s = 60 / (time.time() - t0)
        print(f"Bucket {cap} decode test (60 tokens): {tok_s:.2f} tok/s")

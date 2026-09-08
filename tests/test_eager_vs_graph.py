import torch
from transformers import AutoTokenizer, StaticCache
from build_packed_model import build

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

print("Loading model...")
m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

prompt = "What is the capital of France?"
ids = tok(prompt, return_tensors="pt").to("cuda")
plen = ids["input_ids"].shape[1]

# 1. Eager StaticCache
c_eager = StaticCache(m.config, 1, 256, device="cuda", dtype=torch.float16)
toks_eager = []
with torch.no_grad():
    out = m(**ids, past_key_values=c_eager, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
    cur = out.logits[:, -1:].argmax(-1)
    toks_eager.append(int(cur))
    for i in range(25):
        pos = torch.tensor([plen + i], device="cuda")
        out = m(input_ids=cur, past_key_values=c_eager, use_cache=True, cache_position=pos)
        cur = out.logits[:, -1:].argmax(-1)
        toks_eager.append(int(cur))
text_eager = tok.decode(toks_eager)
print(f"1. EAGER STATICCACHE:\n   {text_eager!r}\n")

# 2. Per-prompt captured CUDA Graph (like test_full_dense_t1_decode.py)
c_graph1 = StaticCache(m.config, 1, 256, device="cuda", dtype=torch.float16)
toks_graph1 = []
with torch.no_grad():
    out = m(**ids, past_key_values=c_graph1, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
    cur = out.logits[:, -1:].argmax(-1)
    toks_graph1.append(int(cur))
    
    static_in1 = cur.clone()
    static_pos1 = torch.tensor([plen], device="cuda")
    s1 = torch.cuda.Stream()
    s1.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s1):
        for _ in range(3):
            m(input_ids=static_in1, past_key_values=c_graph1, use_cache=True, cache_position=static_pos1)
    torch.cuda.current_stream().wait_stream(s1)
    
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        static_out1 = m(input_ids=static_in1, past_key_values=c_graph1, use_cache=True, cache_position=static_pos1).logits

    for i in range(25):
        static_in1.copy_(cur)
        static_pos1.fill_(plen + i)
        g1.replay()
        cur = static_out1[:, -1:].argmax(-1)
        toks_graph1.append(int(cur))
text_graph1 = tok.decode(toks_graph1)
print(f"2. PER-PROMPT CAPTURED GRAPH:\n   {text_graph1!r}\n")

# 3. Startup pre-captured CUDA Graph (like what web_ui.py had)
c_startup = StaticCache(m.config, 1, 1024, device="cuda", dtype=torch.float16)
static_in_s = torch.zeros((1, 1), dtype=torch.long, device="cuda")
static_pos_s = torch.zeros((1,), dtype=torch.long, device="cuda")
s_s = torch.cuda.Stream()
s_s.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(s_s):
    for _ in range(3):
        m(input_ids=static_in_s, past_key_values=c_startup, use_cache=True, cache_position=static_pos_s)
torch.cuda.current_stream().wait_stream(s_s)

g_startup = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g_startup):
    static_out_s = m(input_ids=static_in_s, past_key_values=c_startup, use_cache=True, cache_position=static_pos_s).logits

c_startup.reset()
toks_startup = []
with torch.no_grad():
    out = m(**ids, past_key_values=c_startup, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
    cur = out.logits[:, -1:].argmax(-1)
    toks_startup.append(int(cur))
    for i in range(25):
        static_in_s.copy_(cur)
        static_pos_s.fill_(plen + i)
        g_startup.replay()
        cur = static_out_s[:, -1:].argmax(-1)
        toks_startup.append(int(cur))
text_startup = tok.decode(toks_startup)
print(f"3. STARTUP PRE-CAPTURED GRAPH:\n   {text_startup!r}\n")

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

MAX_CACHE_LEN = 1024
cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
                    device="cuda", dtype=torch.float16)

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

# Simulate the user's chat session:
# Turn 1:
chat = [
    {"role": "user", "content": "hi"}
]
def turn(chat_history):
    cache.reset()
    prompt_msgs = [{"role": "system", "content": "You are a helpful assistant."}] + chat_history
    try:
        p_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p_text, return_tensors="pt").to("cuda")
    plen = ids["input_ids"].shape[1]
    with torch.no_grad():
        out = m(**ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
        nxt = out.logits[:, -1:].argmax(-1)
        toks = [int(nxt)]
        cur = nxt.clone()
        for i in range(40):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur = static_out[:, -1:].argmax(-1)
            tid = int(cur)
            if tid in [tok.eos_token_id, 151645, 151643]:
                break
            toks.append(tid)
    return tok.decode(toks)

ans1 = turn(chat)
print(f"Turn 1 response: {ans1!r}")

chat.append({"role": "assistant", "content": ans1})
chat.append({"role": "user", "content": "what is government"})

ans2 = turn(chat)
print(f"\nTurn 2 response: {ans2!r}")

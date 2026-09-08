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

cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=1024,
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


def sample_with_penalty(logits, generated_tokens, temperature=0.7, top_p=0.9, top_k=50, rep_penalty=1.15):
    # Apply repetition penalty
    l = logits.clone().float()
    if rep_penalty != 1.0 and generated_tokens:
        toks = list(set(generated_tokens))
        for t in toks:
            if l[0, t] > 0:
                l[0, t] /= rep_penalty
            else:
                l[0, t] *= rep_penalty
    
    # Temperature
    if temperature <= 0.01:
        return l.argmax(-1, keepdim=True)
    l = l / temperature
    
    # Top-K
    if top_k > 0:
        val, _ = torch.topk(l, min(top_k, l.shape[-1]))
        l[l < val[:, [-1]]] = -float('Inf')
        
    # Top-P
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(l, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        l[indices_to_remove] = -float('Inf')
        
    probs = torch.softmax(l, dim=-1)
    return torch.multinomial(probs, 1)


def generate(prompt, thinking=False, rep_pen=1.15):
    cache.reset()
    msgs = [{"role": "user", "content": prompt}]
    if not thinking:
        try:
            p_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        p_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
    ids = tok(p_text, return_tensors="pt").to("cuda")
    plen = ids["input_ids"].shape[1]
    
    stop_ids = {tok.eos_token_id, 151645, 151643}
    
    toks = []
    with torch.no_grad():
        out = m(**ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device="cuda"))
        cur_logits = out.logits[:, -1]
        nxt = sample_with_penalty(cur_logits, toks, temperature=0.7, rep_penalty=rep_pen)
        tid = int(nxt)
        if tid in stop_ids:
            return tok.decode(toks)
        toks.append(tid)
        
        cur = nxt.clone()
        for i in range(120):
            static_in.copy_(cur)
            static_pos.fill_(plen + i)
            g.replay()
            cur_logits = static_out[:, -1]
            cur = sample_with_penalty(cur_logits, toks, temperature=0.7, rep_penalty=rep_pen)
            tid = int(cur)
            if tid in stop_ids:
                break
            toks.append(tid)
            
    return tok.decode(toks)

print("\n--- Test A: What is government? (with rep_penalty=1.15, top_k=50, top_p=0.9) ---")
res_a = generate("What is government?", rep_pen=1.15)
print(res_a)

print("\n--- Test B: Write a short story about a clockmaker. (rep_penalty=1.15) ---")
res_b = generate("Write a short story about a clockmaker who discovers a magic gear.", rep_pen=1.15)
print(res_b)

import sys, os, time, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_packed_model import build
from packed_linear import set_kernel
from transformers import AutoTokenizer, StaticCache

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

set_kernel("v4_opt")
m, _ = build(mid, pack, rot)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

prompt = "Explain quantum computing in three short sentences."
messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tok(prompt_text, return_tensors="pt").to("cuda")
plen = input_ids["input_ids"].shape[1]

stop_ids = {tok.eos_token_id}
for s in ["<|im_end|>", "<|endoftext|>"]:
    tid = tok.convert_tokens_to_ids(s)
    if tid is not None: stop_ids.add(tid)

MAX_CACHE_LEN = 2048
max_tokens = 64
temperature = 0.7

print(f"Prompt length: {plen} tokens. Testing Graph decode with temperature {temperature}...")

cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
                    device="cuda", dtype=torch.float16)

# Prefill
with torch.no_grad():
    t_start = time.time()
    out = m(**input_ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(plen, device="cuda"))
    logits = out.logits[:, -1]
    
    if temperature > 0:
        probs = torch.softmax(logits.float() / max(temperature, 1e-5), -1)
        nxt = torch.multinomial(probs, 1)
    else:
        nxt = logits.argmax(-1, keepdim=True)
        
    static_in = nxt.clone()
    static_pos = torch.tensor([plen], device="cuda")
    
    # Warmup and graph capture
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
                       
    tokens = []
    token_id = int(nxt)
    if token_id not in stop_ids:
        tokens.append(token_id)
        print(f"First token: {tok.decode([token_id])!r}")
        
        t_decode_start = time.time()
        for i in range(max_tokens - 1):
            static_in.copy_(nxt)
            static_pos.fill_(plen + i)
            g.replay()
            cur_logits = static_out[:, -1]
            
            if temperature > 0:
                probs = torch.softmax(cur_logits.float() / max(temperature, 1e-5), -1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = cur_logits.argmax(-1, keepdim=True)
                
            token_id = int(nxt)
            if token_id in stop_ids:
                break
            tokens.append(token_id)
            print(tok.decode([token_id]), end="", flush=True)
            
        t_decode = time.time() - t_decode_start
        print(f"\n\nDecode time: {t_decode:.3f}s for {len(tokens)-1} tokens ({ (len(tokens)-1)/t_decode:.2f} tok/s)")

print(f"Total tokens generated: {len(tokens)}")
print(f"Decoded response:\n{tok.decode(tokens)}")

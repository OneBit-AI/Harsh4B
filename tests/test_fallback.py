import torch
from transformers import AutoTokenizer, DynamicCache
from build_packed_model import build

R2 = r"C:\Users\OneBit AI\Desktop\onebit ai\Poojith\1-ptq\ternary-v2"
mid = "Qwen/Qwen3-4B"
pack = rf"{R2}\save_models\Qwen3-4B-LATTICE_pd0.01.lat.pt"
rot = rf"{R2}\kernel\rot_4b_LR.pt"

m, _ = build(mid, pack, rot, verbose=False)
tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)

# Test the exact fallback logic from web_ui.py
prompt = "what is government"
msgs = [{"role": "user", "content": prompt}]
prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
input_ids = tok(prompt_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = m(**input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1]
    
    toks = []
    for _ in range(50):
        next_token = logits.argmax(-1, keepdim=True)
        token_id = int(next_token)
        toks.append(token_id)
        outputs = m(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1]

print("FALLBACK OUTPUT:")
print(repr(tok.decode(toks)))

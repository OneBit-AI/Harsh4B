from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
msgs = [{"role": "user", "content": "What is government?"}]

print("--- Default chat template ---")
t1 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(repr(t1))

print("--- With enable_thinking=False ---")
try:
    t2 = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(repr(t2))
except Exception as e:
    print(e)

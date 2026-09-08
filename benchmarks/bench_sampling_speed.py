import torch
import time

logits = torch.randn(1, 152064, device="cuda", dtype=torch.float16)
gen_tokens = list(range(100, 200)) # 100 generated tokens

def old_sample(logits, generated_tokens, rep_penalty=1.15):
    l = logits.reshape(1, -1).clone().float()
    toks = list(set(generated_tokens))
    for t in toks:
        if l[0, t] > 0:
            l[0, t] /= rep_penalty
        else:
            l[0, t] *= rep_penalty
    val, _ = torch.topk(l, 50)
    l[l < val[:, -1:]] = -float('Inf')
    probs = torch.softmax(l, dim=-1)
    return torch.multinomial(probs, 1)

def fast_sample(logits, generated_tokens, rep_penalty=1.15):
    l = logits.reshape(1, -1).clone().float()
    if generated_tokens:
        ut = torch.tensor(list(set(generated_tokens)), device=l.device, dtype=torch.long)
        tl = l[0, ut]
        l[0, ut] = torch.where(tl > 0, tl / rep_penalty, tl * rep_penalty)
    val, _ = torch.topk(l, 50)
    l[l < val[:, -1:]] = -float('Inf')
    probs = torch.softmax(l, dim=-1)
    return torch.multinomial(probs, 1)

# Warmup
for _ in range(10):
    old_sample(logits, gen_tokens)
    fast_sample(logits, gen_tokens)
torch.cuda.synchronize()

N = 100
t0 = time.time()
for _ in range(N):
    old_sample(logits, gen_tokens)
torch.cuda.synchronize()
t_old = (time.time() - t0) / N

t0 = time.time()
for _ in range(N):
    fast_sample(logits, gen_tokens)
torch.cuda.synchronize()
t_fast = (time.time() - t0) / N

print(f"Old sampling time per token: {t_old*1000:.3f} ms")
print(f"Fast sampling time per token: {t_fast*1000:.3f} ms")
print(f"Speedup: {t_old / t_fast:.2f}x")

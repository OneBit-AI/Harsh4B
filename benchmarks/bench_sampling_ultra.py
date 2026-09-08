import torch
import time

logits = torch.randn(1, 152064, device="cuda", dtype=torch.float16)
gen_tokens = list(range(100, 200))

def standard_sample(logits, generated_tokens, temperature=0.7, top_p=0.9, top_k=50, rep_penalty=1.15):
    l = logits.reshape(1, -1).clone().float()
    if generated_tokens:
        ut = torch.tensor(list(set(generated_tokens)), device=l.device, dtype=torch.long)
        tl = l[0, ut]
        l[0, ut] = torch.where(tl > 0, tl / rep_penalty, tl * rep_penalty)
    l = l / max(temperature, 1e-5)
    if top_k > 0:
        val, _ = torch.topk(l, min(top_k, l.shape[-1]))
        l[l < val[:, -1:]] = -float('Inf')
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

def ultra_sample(logits, generated_tokens, temperature=0.7, top_p=0.9, top_k=50, rep_penalty=1.15):
    l = logits.reshape(1, -1).clone().float()
    if generated_tokens:
        ut = torch.tensor(list(set(generated_tokens)), device=l.device, dtype=torch.long)
        tl = l[0, ut]
        l[0, ut] = torch.where(tl > 0, tl / rep_penalty, tl * rep_penalty)
    if temperature <= 0.01:
        return l.argmax(-1, keepdim=True)
    l = l / max(temperature, 1e-5)
    
    # Top-K (val is already sorted in descending order)
    K = min(top_k if top_k > 0 else 50, l.shape[-1])
    val, idx = torch.topk(l, K, sorted=True)
    
    probs = torch.softmax(val, dim=-1)
    if top_p < 1.0:
        cum_probs = torch.cumsum(probs, dim=-1)
        # remove tokens where cum_probs - probs > top_p
        mask = (cum_probs - probs) >= top_p
        probs[mask] = 0.0
        p_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(p_sum > 0, probs / p_sum, torch.zeros_like(probs))
    
    sampled_idx = torch.multinomial(probs, 1)
    return idx.gather(-1, sampled_idx)

# Test numerical equivalence / behavior
torch.manual_seed(42)
print("Testing sampling...")
for _ in range(5):
    t_std = standard_sample(logits, gen_tokens)
    t_ult = ultra_sample(logits, gen_tokens)

# Benchmark
for _ in range(10):
    standard_sample(logits, gen_tokens)
    ultra_sample(logits, gen_tokens)
torch.cuda.synchronize()

N = 200
t0 = time.time()
for _ in range(N):
    standard_sample(logits, gen_tokens)
torch.cuda.synchronize()
t_std = (time.time() - t0) / N

t0 = time.time()
for _ in range(N):
    ultra_sample(logits, gen_tokens)
torch.cuda.synchronize()
t_ult = (time.time() - t0) / N

print(f"Standard sample time: {t_std*1000:.3f} ms")
print(f"Ultra sample time:    {t_ult*1000:.3f} ms")
print(f"Speedup: {t_std / t_ult:.2f}x")

import torch
import triton
import triton.language as tl
import time, statistics

# Shapes from 4B model:
# 1. DL=5, DR=512 (72 linears)
# 2. DL=8, DR=320 (36 linears)
# 3. DL=16, DR=160 (72 linears)
# 4. DL=16, DR=256 (36 linears)
# 5. DL=16, DR=608 (36 linears)

shapes = [
    (5, 512, 72),
    (8, 320, 36),
    (16, 160, 72),
    (16, 256, 36),
    (16, 608, 36),
]

def bench(f, warmup=20, iters=100):
    for _ in range(warmup): f()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); f(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(sorted(times))

print("Benchmarking cuBLAS eager L @ x @ R for each shape:")
total_cublas_ms = 0.0
for dl, dr, count in shapes:
    L = torch.randn(dl, dl, device="cuda", dtype=torch.float16)
    R = torch.randn(dr, dr, device="cuda", dtype=torch.float16)
    x = torch.randn(1, dl, dr, device="cuda", dtype=torch.float16)
    
    # Eager PyTorch
    fn_cublas = lambda: (L @ x @ R)
    ms = bench(fn_cublas)
    total_shape_ms = ms * count
    total_cublas_ms += total_shape_ms
    print(f"  DL={dl:2d}, DR={dr:3d} (count {count:2d}): {ms*1000:6.1f} us each | {total_shape_ms:5.2f} ms total")

print(f"\nTotal PyTorch rotation time across all 252 linears: {total_cublas_ms:.2f} ms/token")

import torch
import time, statistics

shapes = [
    (5, 512, 72),
    (8, 320, 36),
    (16, 160, 72),
    (16, 256, 36),
    (16, 608, 36),
]

def bench(f, iters=100):
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); f(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(sorted(times))

print("Testing CUDA Graph on rotation shapes:")
total_graph_ms = 0.0
for dl, dr, count in shapes:
    L = torch.randn(dl, dl, device="cuda", dtype=torch.float16)
    R = torch.randn(dr, dr, device="cuda", dtype=torch.float16)
    x = torch.randn(1, dl, dr, device="cuda", dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        y = L @ x @ R
    
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = L @ x @ R
    
    ms = bench(lambda: g.replay())
    total_shape_ms = ms * count
    total_graph_ms += total_shape_ms
    print(f"  DL={dl:2d}, DR={dr:3d} (count {count:2d}): {ms*1000:6.1f} us each | {total_shape_ms:5.2f} ms total")

print(f"\nTotal Rotation under CUDA Graph across all 252 linears: {total_graph_ms:.2f} ms/token")

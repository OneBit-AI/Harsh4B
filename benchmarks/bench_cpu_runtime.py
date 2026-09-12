"""Small CPU GEMV/GEMM dtype/thread sweep; run under MLX.safety on macOS."""
import json
import statistics
import time

import torch

torch.manual_seed(19)
weight = torch.randn(9728, 2560) * 0.02
results = []
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    w = weight.to(dtype)
    for threads in (1, 4, 8):
        torch.set_num_threads(threads)
        for batch in (1, 8):
            x = torch.randn(batch, 2560).to(dtype)
            samples = []
            with torch.inference_mode():
                for i in range(12):
                    started = time.perf_counter()
                    y = torch.nn.functional.linear(x, w)
                    seconds = time.perf_counter() - started
                    if i >= 3:
                        samples.append(seconds)
            row = dict(dtype=str(dtype), threads=threads, tokens=batch,
                       median_ms=round(statistics.median(samples)*1000, 3))
            results.append(row)
            print(json.dumps(row), flush=True)

# CPU status — Apple M4

Current CPU path: packed int4 LiteSpark, six threads, 512-byte GEMV prefetch, and fused native grouped-query attention. Figures below are medians from fresh CPU-only benchmarks.

| Benchmark | Decode tok/s | TTFT | TPOT median / p95 | J/token |
| --- | ---: | ---: | ---: | ---: |
| 32-token prompt, 32-token output | 32.5 | 0.976 s | 30.4 / 33.4 ms | — |
| 128-token prompt, 32-token output | 30.5 | 4.057 s | 32.7 / 34.6 ms | — |
| 512-token prompt, 32-token output | 24.1 | 18.816 s | 41.1 / 45.7 ms | — |
| 128-token prompt, 64-token output, battery | 22.5 | 5.430 s | 43.6 / 51.9 ms | 2.834 |

The packed layout cuts weight traffic; the ARM kernel unpacks nibbles in registers. Fused attention avoids repeated NumPy dispatch and score/softmax passes. Six workers and 512-byte prefetch were fastest in the M4 sweep.

Short-context decode clears 30 tok/s, but longer context and the energy run do not yet hold that rate. Battery J/token is **whole-system energy including prefill**, not isolated CPU or decode energy; its idle-adjusted estimate was unusable (0 J/token).

Sources: `results/status_latency_cpu.json` (3 repeats, 1 warmup per cell) and `results/status_energy_cpu.json` (3 repeats, 1 warmup). Benchmark artifacts are local and ignored by Git.

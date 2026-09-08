# MLX / Apple Silicon benchmarks

All runs on Apple M4 (16 GiB unified memory, `applegpu_g16g`), MLX 0.32.2,
macOS 26, every workload supervised by `MLX.safety` (fail-closed watchdog:
thermal/memory-pressure/RSS/available-RAM kill switch, stop-file latch).

## End-to-end BitNet-b1.58-large (729M params)

`python -m MLX.generate` — 194-token prompt, 128 generated tokens, greedy.
Architecture is a faithful port of the HF checkpoint (`modeling_bitnet.py` +
`utils_quant.py`); weights are ternarized at load with the published absmean
rule and packed to 2-bit codes; only single-token decode linear arithmetic is
served by the packed Metal GEMV kernel.

| Arm | Decode tok/s | Prefill | Peak MLX memory |
| --- | --- | --- | --- |
| Packed GEMV (kernel v1) | 80.5 | 858 ms (~226 tok/s) | ~0.5 GiB |
| Packed GEMV (kernel v2) | **83.5** | 682 ms | ~0.5 GiB |
| Resident dense MLX (baseline) | 36.4 | 264 ms | ~1.9 GiB |

- Packed decode is **2.3x faster than resident-dense MLX** at ~4x lower memory.
- Prefill unpacks weights transiently per layer (bounded memory); a packed
  prefill GEMM is the obvious next step and is expected to beat the dense arm.
- Generated text is coherent English (base model, greedy decoding is
  repetitive — that is the checkpoint, not the kernel). Packed and dense arms
  diverge token-wise after the first tokens, as expected from different FP
  reduction orders; this is not a semantic regression gate.

## Layer microbenchmarks (decode GEMV, real checkpoint matrices)

`python -m MLX.bench_layers` — see `MLX/runs/layers-2.json` for full JSON
(v1 kernel). Amortized = 16 independent dispatches / batch, 5 batches.

| Shape (OC x IC) | v1 us/disp | v2 us/disp | speedup |
| --- | --- | --- | --- |
| 1536 x 1536 (q/o/k/v) | 75.1 | 63.0 | 1.19x |
| 4096 x 1536 (gate/up) | 123.1 | 63.7 | 1.93x |
| 1536 x 4096 (down) | 125.3 | 97.4 | 1.29x |

Kernel v2 (vectorized `uint32` packed loads, 16 codes/lane, scale applied once
after accumulation) is used automatically when `IC % 16 == 0`; the scalar tail
kernel remains the fallback. Weight packing is tested bit-exact; output differs
from a dense FP16 GEMM by at most a few FP16 ULPs (different reduction order).

## Known limitations

- Single-call kernel latency is dispatch-overhead dominated (~200-300 us floor
  on this machine); amortized numbers are the meaningful kernel metric.
- Not bit-exact vs dense GEMM; no perplexity/quality claim is made.
- Decode step still pays ~5 ms/token in small-op dispatch (norms, activation
  quant, attention bookkeeping). `mx.compile` + static KV cache is the next
  structural optimization, blocked by the growing-shape KV concatenate.
- MLX memory limits are guidelines, not hard caps; the supervisor is the
  actual safety boundary.

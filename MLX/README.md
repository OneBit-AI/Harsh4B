# Shared ternary kernels for MLX

Experimental inference primitives for Apple Silicon **GPU/Metal**, not the Apple Neural Engine. This directory adds no model architecture, tokenizer, attention, cache, normalization, activation quantizer, or generation loop. Existing CUDA files are unchanged.

## What is accelerated

`kernels.py` provides two formats:

- `TernaryLinear`: deployed ternary codes and one uniform scale. Four two-bit codes per byte; codes `0,1,2` decode to `0,+1,-1`, and code `3` follows the existing kernel's zero interpretation. The single-vector path reconstructs weights inside the Metal kernel.
- `LatticeLinear`: the repository's `mu + a0*T0 + a1*T1` equation, including all four mask IDs and the second component only for masks 0/1. Its constructor interleaves scales once; the hot loop reuses that layout and dense two-bit T1.

Adjacent SIMD lanes load adjacent packed bytes. A SIMD group computes an output row, accumulates in FP32, and reduces once at the end. Input tails are masked. Multi-token inputs use a reference MLX matrix multiplication. This is a decode-kernel optimization, not an optimized prefill GEMM implementation.

The caller supplies its existing, already-preprocessed activation tensor and deployed codes/scales. The library never changes the model's activation quantization or converts floating master weights to a different quantization scheme. Keep rotations, biases not represented by the chosen primitive, normalization and all surrounding model operations in their original locations. The uniform-scale and affine LATTICE formats are not interchangeable.

These primitives consume MLX arrays. They do not transparently replace modules in a PyTorch/CUDA model; per-token transfers between frameworks would need separate measurement.

## Correctness boundary

Packing and unpacking the deployed ternary weights is tested for **exact equality**. Tests cover all mask IDs, partial blocks, bias, FP16/FP32 input, prefill and the absence of unwanted activation transformations.

Floating-point reductions can round differently from a dense GEMM. The real-layer fixture tests found a maximum output difference of `0.0009765625` in FP16. That means this implementation is **not guaranteed bit-for-bit output equivalent**. Kernel tolerances do not establish unchanged perplexity or language quality. No model has been rewired to use the candidate kernels, and no end-to-end token/s or semantic-performance claim is made.

## Environment

From the repository root:

```sh
python3.12 -m venv MLX/.venv
MLX/.venv/bin/python -m pip install -r MLX/requirements.txt
```

The measured setup used Python 3.12, MLX 0.32.2, and an Apple M4 with 16 GiB memory. The pinned MLX wheel used here requires a compatible macOS version; this run used macOS 26.6.2.

## Kill switch first

Run GPU work through the supervisor:

```sh
MLX/.venv/bin/python -B -m MLX.safety \
  --log MLX/runs/tests.jsonl --stop-file MLX/STOP --max-seconds 45 -- \
  MLX/.venv/bin/python -B -m pytest -q -p no:cacheprovider MLX/tests
```

The workload waits behind a pipe gate until the supervisor and secondary watchdog arm. A separate sensor process reports approximately every 100 ms. The supervisor immediately sends `SIGKILL` to its own workload process group when any of these occur:

- macOS thermal state is anything except nominal, including the first `fair` level;
- system memory pressure is anything except normal;
- available system memory falls below 4 GiB;
- workload process-tree RSS exceeds 3 GiB, or swap grows at all;
- a sensor fails, becomes stale or stops reporting, or the secondary watchdog dies;
- the launcher disappears, a termination signal arrives, the deadline expires, or the stop latch appears.

The secondary watchdog kills the workload and sets the stop latch if the main supervisor itself disappears, including `SIGKILL`. It is exercised against a real sleeping subprocess in tests. Monitoring failure blocks launch or kills the workload rather than allowing it to proceed.

Manual emergency stop, using the **same stop-file path** as the running command:

```sh
MLX/.venv/bin/python -B -m MLX.safety --stop-file MLX/STOP --kill
```

The latch prevents another launch. Inspect the reason in the JSONL log and the current machine state before manually clearing it. There is no automatic restart. The nominal stop-check interval is approximately 100 ms; a missing sensor can take up to the 1-second heartbeat timeout to trigger a kill. OS scheduling and in-flight GPU work prevent a hard real-time guarantee.

MLX allocator limits are guidelines, not hard RAM caps. The independent watchdog and small workload limits are therefore essential. `btop` is the visual monitor; automatic decisions use the OS sensors, not parsing its screen. OS thermal pressure is not a direct per-core temperature measurement. This setup reduces risk but cannot guarantee that a laptop will never overheat or crash.

## Benchmark real layers without changing a model

The fixture source is `1bitLLM/bitnet_b1_58-large`, pinned in `models/bitnet-700m/source.json`. Its stored checkpoint tensors contain 728,843,904 scalar entries, below 2B. The download is about 2.9 GB because it contains floating master weights. The fixture builder reads only one tensor at a time and applies the published BitNet quantization equation to construct a frozen FP16 evaluation fixture. Both comparison paths receive identical weights and activations. Source files and checkpoint weights are not executed or modified.

```sh
MLX/.venv/bin/python -B -m MLX.fetch
MLX/.venv/bin/python -B -m MLX.safety \
  --log MLX/runs/download.jsonl --stop-file MLX/STOP --max-seconds 300 -- \
  MLX/.venv/bin/python -B -m MLX.fetch --weights

btop --update 1000
# In another terminal, from the same repository root:
MLX/.venv/bin/python -B -m MLX.safety \
  --log MLX/runs/layers-guard.jsonl --stop-file MLX/STOP --max-seconds 60 -- \
  MLX/.venv/bin/python -B -m MLX.bench_layers --output MLX/runs/layers.json
```

Use new log/output filenames on every run. `--model /absolute/path/to/downloaded/model` reuses an existing download. Results distinguish single-call latency from throughput amortized across 16 independent dispatches; neither is model token/s. The two baselines are resident dense MLX weights and unpacking then calling MLX matmul on every invocation.

See `BENCHMARK.md` for measured results and their limitations.

## Sources

- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX memory-limit semantics](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)
- [700M BitNet model card](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
- [Pinned source quantizer](https://huggingface.co/1bitLLM/bitnet_b1_58-large/blob/85d047191dcb224f0e04f20d26110caaf8dc1a47/utils_quant.py)

## End-to-end inference (this repo)

`MLX/bitnet_mlx.py` + `MLX/generate.py` run the real BitNet-b1.58-large
checkpoint end-to-end on Metal under the kill switch:

```sh
MLX/.venv/bin/python -B -m MLX.safety \
  --log MLX/runs/gen.jsonl --stop-file MLX/STOP --max-seconds 300 --max-rss-gib 6 -- \
  MLX/.venv/bin/python -B -m MLX.generate --output MLX/runs/gen.json \
  --prompt "Your prompt here" --max-tokens 128
```

Measured results and limitations: see `MLX/BENCHMARK.md`.

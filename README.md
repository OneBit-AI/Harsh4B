# OneBit AI: Qwen3-4B LATTICE

Run the Qwen3-4B LATTICE checkpoint locally on CPU, Apple Metal, or NVIDIA
CUDA. The fastest path on an Apple M4 is currently the CPU runner in
`cpu_lightspark.py`.

## Set up once

Use Python 3.12 and one virtual environment for the whole repository. All
dependencies live in `pyproject.toml`; do not make separate CPU and MLX
environments.

On Apple Silicon, install Python and OpenMP first:

```bash
brew install python@3.12 libomp
```

Then, from this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

The `all` extra is portable. Platform markers install MLX only on Apple
Silicon and Triton only where it is supported. The same `.venv` can run the
CPU runner, Metal runner, web UI, benchmarks, and tests available on that
machine.

Put these files in the repository root:

- `model.lat.pt` — the Qwen3-4B LATTICE checkpoint.
- `tokenizer.json` — the matching Qwen3 tokenizer.
- `rot_4b_LR.pt` — needed only by CUDA paths that do not embed rotations.

The large model files, generated caches, virtual environment, and benchmark
results are ignored by Git.

## Run the fast CPU version

The easiest command is:

```bash
./run.sh "Explain why water is cohesive"
```

The first CPU run converts the checkpoint into a roughly 2 GiB packed cache
under `.cache/litespark-lattice/`. This can take about two minutes. Later
starts normally take less than a second because the converted model is reused.

To control the run directly:

```bash
.venv/bin/python -B cpu_lightspark.py \
  --prompt "Explain why water is cohesive" \
  --threads 8 \
  --max-new-tokens 64
```

CPU sampling defaults are:

```text
temperature=0.7
top_k=40
top_p=0.9
repetition_penalty=1.08
```

Override any value with its command-line flag. For deterministic greedy
decoding, use `--argmax`:

```bash
.venv/bin/python -B cpu_lightspark.py \
  --prompt "Explain ternary inference" \
  --threads 8 \
  --argmax
```

Text prompts use Qwen's thinking-disabled chat format. Add `--raw-prompt` to
skip that wrapper. `--max-new-tokens` is a hard output limit, so a short limit
can cut off a sentence.

## Profile one CPU token

This profiles one real decode token after warmup:

```bash
.venv/bin/python -B cpu_lightspark.py \
  --prompt "Explain why water is cohesive" \
  --threads 8 \
  --warmup-tokens 8 \
  --profile-token
```

The profiler reports time spent in body GEMV, activation quantization,
embedding/LM head, attention, RMSNorm, sampling, packing, and other work. It
also checks the expected call counts for all 36 layers.

One Apple M4 run at context position 49 measured:

| Work | Time | Decode share |
| --- | ---: | ---: |
| Packed int4 GEMV | 23.814 ms | 68.6% |
| Attention | 3.428 ms | 9.9% |
| Embedding and LM head | 2.297 ms | 6.6% |
| RMSNorm | 1.255 ms | 3.6% |
| Activation quantization | 0.887 ms | 2.6% |
| Sampling | 0.413 ms | 1.2% |
| Standalone packing/depacking | 0 ms | 0% |
| Other and profiler overhead | 2.600 ms | 7.5% |

Packing is not a separate weight pass. The kernel unpacks signed int4 nibbles
inside NEON registers immediately before `SDOT`; embedding-row unpacking is
counted with embedding/LM head.

## CPU GEMV experiments

The Apple ARM64 kernel is in `cpu_kernels/litespark_int4.cpp`. It uses four
independent accumulators to hide `SDOT` latency, OpenMP across output rows, and
software prefetching.

The [LiteSpark paper](https://arxiv.org/abs/2605.06485) chose byte-aligned int8
ternary weights to avoid repeated unpacking. We tested that idea against this
model's row-quantized `[-7,+7]` weights. Expanded int8 was bit-exact with the
packed kernel but doubled body-weight traffic and was much slower:

| Body GEMV layout | Decode speed |
| --- | ---: |
| Expanded int8 | 17.95 tok/s |
| Packed int4, 512-byte prefetch | 27.28–27.55 tok/s |
| Packed int4, 1024-byte prefetch | **28.61–28.84 tok/s** |

Packed int4 was 1.71× faster than expanded int8. Moving prefetch from 512 to
1024 packed bytes improved full-model decode by 3.8–5.7%, so packed int4 with
1024-byte prefetch is the default.

The slower controls remain available for repeatable A/B tests:

```bash
# Prefetch-distance test
.venv/bin/python -B cpu_lightspark.py \
  --prompt "Explain ternary inference" --threads 8 --argmax \
  --gemv-kernel prefetch512

# Packed int4 versus expanded int8
.venv/bin/python -B cpu_lightspark.py \
  --prompt "Explain ternary inference" --threads 8 --argmax \
  --gemv-layout int8
```

Available packed-kernel choices are `none`, `prefetch256`, `prefetch512`, and
`prefetch1024`. Use `--argmax` for kernel comparisons so sampling cannot change
the token path.

## Other runtimes

Use the shared runner when comparing backends:

```bash
# CPU
.venv/bin/python -B test.py --runtime cpu \
  --prompt "Explain ternary inference"

# Apple Metal
.venv/bin/python -B test.py --runtime mlx \
  --prompt "Explain ternary inference"

# NVIDIA CUDA
.venv/bin/python -B test.py --runtime cuda \
  --prompt "Explain ternary inference"
```

`web_ui.py --runtime auto` prefers LiteSpark CPU on Apple Silicon, CUDA on a
supported NVIDIA machine, and then another available local backend:

```bash
.venv/bin/python -B web_ui.py --runtime auto
```

Open <http://127.0.0.1:7860>. The server binds only to localhost unless you
pass a different `--host`.

The CUDA path also expects upstream `build_packed_model.py`, `gemv_triton.py`,
`gemm_triton.py`, and loader modules on `PYTHONPATH`. They are not included in
this repository.

## Benchmarks

Run the latency and energy suite with:

```bash
./bench_all.sh
```

Useful smaller runs:

```bash
# CPU TTFT and per-token latency
.venv/bin/python -B benchmarks/bench_latency.py \
  --runtime cpu --cpu-threads 8

# CPU timing plus energy when a power source is available
.venv/bin/python -B benchmarks/bench_energy.py \
  --runtime cpu --cpu-threads 8 \
  --prompt-tokens 128 --max-new-tokens 64
```

`bench_all.sh` accepts environment overrides such as:

```bash
RUNTIMES="cpu" CPU_THREADS=8 REPEATS=5 ./bench_all.sh
```

Results go under `results/<timestamp>/` and are intentionally not committed.
Energy measurement uses `powermetrics` on macOS when passwordless sudo is
already available, battery telemetry when usable, NVML/RAPL on Linux, or a
time-only fallback.

## How the CPU path works

`cpu_lightspark.py` reads the local checkpoint without importing PyTorch. The
one-time converter:

1. reconstructs every affine LATTICE projection;
2. absorbs the KOTMS rotations into those weights;
3. quantizes each output row to signed int4;
4. packs two weights per byte;
5. saves a resumable memory-mapped cache.

During each token, activations are quantized to int8. Q/K/V and gate/up are
concatenated to reduce OpenMP launches. The native ARM kernel reads packed
weights, unpacks only in registers, performs NEON `SDOT`, and applies the
activation and per-row weight scales directly to the output. It never creates
a full floating-point weight matrix during decode.

The tied embedding and vocabulary head are also stored as row-scaled int4.
This CPU cache is a second quantization of the deployed LATTICE values, so its
logits are not bit-for-bit identical to the MLX affine kernel. Model structure,
tokenizer, norms, and checkpoint-derived weights remain the same.

## Tests

Run CPU and shared-runtime tests:

```bash
.venv/bin/python -B -m pytest -q \
  tests/test_cpu_lightspark.py \
  tests/test_runtime_unit.py \
  tests/test_cpu_runtime.py
```

Run MLX tests through the safety supervisor:

```bash
.venv/bin/python -B -m MLX.safety \
  --log MLX/runs/tests.jsonl \
  --stop-file MLX/STOP \
  --max-seconds 180 -- \
  .venv/bin/python -B -m pytest -q MLX/tests
```

CUDA-only tests require NVIDIA hardware plus the upstream modules listed
above.

## MLX safety supervisor

For a longer Metal run:

```bash
.venv/bin/python -B -m MLX.safety \
  --log MLX/runs/validation.jsonl \
  --stop-file MLX/STOP \
  --max-seconds 300 \
  --max-rss-gib 6 \
  --reserve-gib 2.5 -- \
  .venv/bin/python -B test.py --runtime mlx \
    --prompt "Explain ternary inference"
```

Emergency stop:

```bash
.venv/bin/python -B -m MLX.safety --stop-file MLX/STOP --kill
```

The supervisor watches thermal state, memory pressure, available memory,
process RSS, swap growth, sensor freshness, termination signals, and its stop
latch.

## License

MIT License.

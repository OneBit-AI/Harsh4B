# OneBit AI: Qwen3-4B LATTICE

Cross-platform inference for the Qwen3-4B-LATTICE W1.58A16 post-training ternary checkpoint. The repository provides a shared local chat UI and benchmark runner for CUDA/Triton, MLX/Metal on Apple Silicon, and torch-free LiteSpark CPU execution.

## Runtime overview

| Runtime | Hardware | Weight path | Selection |
| --- | --- | --- | --- |
| MLX | Apple Silicon GPU via Metal | Packed LATTICE kernels | `--runtime mlx` |
| CUDA | NVIDIA GPU | Triton packed kernels and bucketed CUDA graphs | `--runtime cuda` |
| CPU | macOS or Linux CPU | LATTICE converted once to packed per-row int4; NEON/AVX GEMV | `--runtime cpu` |

`web_ui.py --runtime auto` prefers LiteSpark CPU on Apple Silicon, CUDA on supported NVIDIA hosts, and then the remaining available local backend. MLX remains available explicitly with `--runtime mlx`. `test.py` defaults to CPU so benchmark results never silently use the GPU.

## Installation

Python dependencies are defined only in `pyproject.toml`. Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

Install a smaller runtime-specific environment with one of:

```bash
python -m pip install -e '.[cpu,test]'
python -m pip install -e '.[mlx,test]'
python -m pip install -e '.[cuda,test]'
```

Platform markers skip MLX outside Apple Silicon macOS and Triton outside Linux. The `all` extra therefore remains usable from one cross-platform dependency definition.

Place these model assets in the repository root:

- `model.lat.pt`: packed Qwen3-4B-LATTICE checkpoint.
- `tokenizer.json`: matching Qwen3 tokenizer.
- `rot_4b_LR.pt`: required by the upstream CUDA path when rotations are not embedded in the checkpoint.

The bundled `configs/qwen3-4b.json` allows CPU construction without downloading model configuration.

## Command-line inference

CPU with LiteSpark packed kernels:

```bash
python test.py \
  --runtime cpu \
  --cpu-threads 8 \
  --prompt "explain why water is cohesive" \
  --max-new-tokens 64 \
  --warmup-tokens 8
```

Apple Silicon Metal:

```bash
python test.py \
  --runtime mlx \
  --prompt "explain why water is cohesive" \
  --max-new-tokens 64 \
  --warmup-tokens 8
```

Text passed with `--prompt` is wrapped in Qwen3's thinking-disabled chat format. Generation uses seeded temperature, top-p/top-k sampling, and a repetition penalty. Use `--raw-prompt` for an unformatted completion benchmark or `--temperature 0` for greedy decoding.

`--max-new-tokens` is a hard output ceiling. If it is reached before Qwen emits an end-of-turn token, the runner reports that the output may be truncated. Increase it when testing longer answers.

The runner prints model startup time, prefill throughput, average decode
throughput, stop reason, and peak MLX memory where applicable.

Profile one warmed single-token Metal decode with fail-fast guards against the
dense/reference projection path:

```bash
python test.py --runtime mlx --profile-token --prompt "explain why water is cohesive"
```

The profile splits attention, packed custom GEMV, sampling, and remaining work.
It also reports the number of packed projections and fails if decode invokes a
separate full-weight dequantization or dense projection.

## Local Web UI

```bash
python web_ui.py --runtime auto
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The status badges identify the active runtime, hardware, memory use, context limit, and measured generation speed. The server binds to loopback by default; pass `--host` explicitly to expose it elsewhere.

The server provides bounded chat history, asynchronous Server-Sent Events, cancellation, backpressure handling, incremental Unicode decoding, and `TCP_NODELAY`. Only the selected backend is loaded, so CPU and GPU copies are not retained together.

## Implementation notes

### MLX and Metal

`MLX/kernels.py` implements packed two-bit ternary and affine LATTICE projections. SIMD lanes load adjacent packed bytes, accumulate in FP32, and reduce once per output row. Decode uses packed custom Metal kernels; multi-token prefill uses the reference matrix path. Growing KV-cache capacity buckets avoid reallocating the complete cache on every token.

Single-token decode reconstructs each affine weight from its packed codes and
FP16 scale table inside the custom GEMV kernel. The default loader expands the
checkpoint's compact T1 stream once into a dense *two-bit packed* layout; it
does not create dense floating-point weights. Multi-token prefill is different:
it currently launches a dequantization kernel and then a dense matrix multiply.
KOTMS activation rotations and the tied dense vocabulary head also remain
outside the packed projection kernel.

The checkpoint reader maps uncompressed `torch.save` ZIP members directly and closes each backing file descriptor immediately after creating its mapping. This keeps the 3.96 GiB checkpoint file-backed without exhausting macOS's per-process open-file limit.

Floating-point reduction order differs from dense matrix multiplication. Fixture tests observed small FP16-level differences; the implementation does not claim bit-for-bit equivalence or unchanged perplexity.

### CPU

`cpu_lightspark.py` reads this repository's `model.lat.pt` directly without importing PyTorch. On the first run it reconstructs every affine LATTICE projection, absorbs the KOTMS rotations, requantizes each output row to signed int4, and writes a resumable cache under `.cache/litespark-lattice/`. This one-time conversion takes roughly two minutes and produces about 2 GiB of packed projection and tied-embedding data; later starts take well under a second when the files are cached by the OS.

Apple ARM64 decode uses `cpu_kernels/litespark_int4.cpp`: packed nibbles are unpacked only into NEON registers, multiplied by LiteSpark-quantized int8 activations with SDOT, and scaled directly into the output. Q/K/V and gate/up are concatenated to reduce OpenMP launch overhead. No dense weight or separate dequantization buffer is created per token. Other architectures fall back to LiteSpark's portable packed-int4 routine.

The int4 cache is a second quantization of the deployed LATTICE values, so it is not bit-identical to the MLX affine kernel. It retains the repository's Qwen architecture, checkpoint weights, tokenizer, norms, and tied head; fixture and end-to-end tests should be rerun when changing the conversion rule.

### CUDA

The CUDA path uses packed Triton GEMV kernels, vectorized sampling, and static cache buckets at 512, 1024, and 2048 tokens. Some CUDA scripts also require the upstream `build_packed_model.py`, `gemv_triton.py`, `gemm_triton.py`, and loader modules on `PYTHONPATH`; those modules are not distributed in this repository.

## Measured results

Recent end-to-end runs on an Apple M4 with 16 GiB unified memory produced:

| Runtime | Configuration | Decode throughput |
| --- | --- | ---: |
| MLX/Metal | 64-token chat-formatted run | 15.73 tok/s |
| LiteSpark CPU | 8 threads, packed int4×int8 SDOT | 32.34 tok/s |
| Absorbed CPU | dense mmap baseline | approximately 0.10 tok/s |

The corrected MLX run generated a concise answer and stopped on Qwen's end-of-turn token. Performance varies with prompt length, thermals, memory pressure, model state, and OS scheduling; these numbers are measurements, not guarantees.

The existing CUDA benchmark reports 40–43 tok/s and approximately 5.5 GiB total VRAM on its tested NVIDIA configuration. CUDA hardware was not available during the Apple Silicon validation and those figures were not independently rerun here.

## Safety supervisor

Long-running MLX experiments can be launched through the fail-closed supervisor:

```bash
python -B -m MLX.safety \
  --log MLX/runs/validation.jsonl \
  --stop-file MLX/STOP \
  --max-seconds 300 \
  --max-rss-gib 6 \
  --reserve-gib 2.5 -- \
  python -B test.py --runtime mlx --prompt "Explain ternary inference"
```

The supervisor monitors thermal state, memory pressure, available memory, process-tree RSS, swap growth, sensor freshness, its secondary watchdog, termination signals, deadline, and the stop latch. A failed or stale sensor blocks launch or terminates the workload.

Emergency stop using the same latch path:

```bash
python -B -m MLX.safety --stop-file MLX/STOP --kill
```

Inspect the JSONL reason and machine state before manually clearing a latch. MLX allocator limits are guidelines rather than hard RAM caps, and OS scheduling prevents a hard real-time shutdown guarantee.

## Validation

CPU and shared runtime tests:

```bash
python -m pytest -q tests/test_runtime_unit.py tests/test_cpu_lightspark.py tests/test_cpu_runtime.py
```

MLX kernel, runner, and safety tests under supervision:

```bash
python -B -m MLX.safety \
  --log MLX/runs/tests.jsonl \
  --stop-file MLX/STOP \
  --max-seconds 180 -- \
  python -B -m pytest -q MLX/tests
```

CUDA-oriented scripts in `tests/` and `benchmarks/` require NVIDIA hardware and the upstream modules described above; they are not portable CPU pytest cases.

## License

MIT License.

# LiteSpark CPU Report

## What changed

CPU inference now runs this repository's own `model.lat.pt` checkpoint through a new torch-free Qwen runtime in `cpu_lightspark.py`.

On the first run, the loader:

1. Opens the LATTICE checkpoint directly.
2. Reconstructs each model projection.
3. Absorbs the KOTMS rotations into the weights.
4. Quantizes each output row to signed int4.
5. Saves the packed result under `.cache/litespark-lattice/`.

The first conversion takes about two minutes and uses about 2 GiB of cache storage. Later model starts take roughly one second or less when the operating-system file cache is warm.

## What runs for every token

The Apple ARM64 kernel in `cpu_kernels/litespark_int4.cpp` performs this path:

```text
packed int4 weights -> unpack nibbles in NEON registers -> int8 SDOT -> scaled output
```

There is no full floating-point weight matrix and no separate per-token weight dequantization buffer. LiteSpark quantizes the activation to int8, and the custom kernel unpacks the weight values only inside CPU registers while doing the dot product.

Q, K, and V are combined into one kernel launch. Gate and up projections are also combined. The tied vocabulary head uses the same packed SDOT path.

## Proof that it uses our model

- The only model input is the local `model.lat.pt` file.
- The local Qwen configuration is read from `configs/qwen3-4b.json`.
- The local tokenizer is read from `tokenizer.json`.
- No Hugging Face model is downloaded or loaded.
- The CPU runtime does not import PyTorch.

The tested water prompt produced coherent output beginning with:

```text
I will explain why water is cohesive. Water is cohesive because it is made of atoms that are polarized...
```

## Measured speed

Machine: Apple M4, 16 GiB unified memory. CPU threads: 8.

Exact 10-token command:

```bash
.venv-mlx/bin/python -B test.py --runtime cpu --cpu-threads 8 --prompt "explain why water is cohesive" --max-new-tokens 10 --warmup-tokens 2
```

Result:

```text
prefill: 25.17 tok/s
decode: 29.21 tok/s
```

A longer 30-token run, which gives the benchmark more work after warmup, measured:

```text
prefill: 31.29 tok/s
decode: 32.34 tok/s
```

These are plain total-tokens-divided-by-total-time measurements. They are not median figures. Speed changes with temperature, memory pressure, CPU temperature, prompt length, and macOS scheduling.

## Output stopping

`--max-new-tokens` is a hard limit. A 10-token run stops after exactly 10 generated tokens even if the sentence is unfinished. The simple runner uses 64 tokens so the model has room to finish. It can still stop earlier when the model emits its end-of-turn token.

## Verification

The relevant test command passed all 17 tests:

```bash
.venv-mlx/bin/python -B -m pytest -p no:cacheprovider tests/test_cpu_lightspark.py tests/test_runtime_unit.py tests/test_cpu_runtime.py -q --tb=short
```

The tests include a numerical comparison between the native signed-int4 SDOT result and a normal integer matrix multiplication reference.

The complete `tests/` directory is not portable on this Mac. Some existing tests require CUDA-only upstream files such as `build_packed_model`, some require Triton, and two expect a live local web server. Those collection failures are separate from the LiteSpark CPU changes.

## Important accuracy note

This is still our Qwen model and our LATTICE checkpoint, but the cached CPU weights receive a second per-row int4 quantization. Therefore CPU logits are not bit-for-bit identical to the original affine LATTICE kernel. The int4 version was chosen because ternary conversion produced broken text, while int4 produced coherent answers and remained fast.

## Main files

- `cpu_lightspark.py`: checkpoint conversion, cache loading, Qwen forward pass, KV cache, attention, sampling inputs, and LiteSpark dispatch.
- `cpu_kernels/litespark_int4.cpp`: packed int4 × int8 Apple NEON SDOT kernel.
- `test.py`: CPU defaults to the new LiteSpark runtime.
- `inference_runtime.py`: web/shared CPU runtime uses LiteSpark.
- `run.sh`: one-command prompt runner.
- `tests/test_cpu_lightspark.py`: packing, thread, CLI, and native-kernel tests.


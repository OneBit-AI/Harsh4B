# OneBit AI: Qwen3-4B LATTICE (W1.58 Ternary PTQ)

High-throughput post-training ternary quantization inference engine and interactive Web UI for **Qwen3-4B-LATTICE (W1.58A16)**, optimized with custom Triton GEMV kernels and CUDA Graph execution on modern NVIDIA GPUs (RTX 50-series Blackwell / Ada / Ampere).

---

## Performance Highlights

- **Sustained Throughput:** **40 – 43 tok/s** (23.4 ms/token) sustained across all conversation turns.
- **Memory Footprint:** Only **~5.5 GB VRAM** total (model weights + multi-bucket CUDA graphs + KV caches), easily fitting inside 12 GB GPUs.
- **Kernel Speedup:** **2.54x faster** Triton GEMV kernel via coalesced 2-bit dense memory layout and deferred vector accumulator.
- **Zero Startup Graph Latency:** Multi-bucket persistent CUDA graphs (`[512, 1024, 2048]`) captured at startup.
- **Zero Degradation in Multi-Turn Chat:** Smart sliding context manager ensures prompts never overflow the static graph cache, permanently eliminating eager-mode CPU dispatch fallbacks.

---

## Key Architecture & Optimizations

### 1. Triton GEMV Dense-T1 Kernel (`gemv_triton_v4.py`)
- **Dense 2-bit Unpacking (`unpack_t1_dense`):** Converts 32 divergent scattered byte gathers into a single coalesced vector load (`tl.load`), completely eliminating prefix sums (`tl.cumsum`) and branch divergence.
- **Deferred Vector Accumulation (`_gemv_dense_t1`):** Maintains thread-local vector accumulators across tiles inside the block loop, replacing per-iteration warp shuffle reductions (`__shfl_down_sync`) with exactly one reduction at the end of each row.

### 2. Multi-Bucketed CUDA Graphs
- Pre-captures optimized CUDA Graphs across three token capacities:
  - **Bucket 512:** Attends over up to 512 tokens $\rightarrow$ **44.0 tok/s**.
  - **Bucket 1024:** Attends over up to 1024 tokens $\rightarrow$ **40.2 tok/s**.
  - **Bucket 2048:** Attends over up to 2048 tokens $\rightarrow$ **34.8 tok/s**.
- Dynamically selects the best bucket per query.

### 3. GPU-Vectorized Ultra-Fast Sampling (`sample_next_token`)
- Replaces Python-level scalar repetition penalty loops with a single vectorized GPU tensor operation (`torch.where` + tensor indexing).
- Reduces sampling latency from **6.38 ms/token down to 0.53 ms/token (14x speedup)** while preserving 100% bit-exact sampling semantics (`temperature=0.7`, `top_p=0.9`, `top_k=50`, `rep_penalty=1.15`).

### 4. Decoupled Asynchronous Streaming & `TCP_NODELAY`
- A dedicated worker thread decouples network socket transmissions from the GPU generation loop.
- `TCP_NODELAY` bypasses Nagle's algorithm, streaming tokens instantaneously to the browser with zero socket stalls.

---

## Project Structure

```
├── web_ui.py             # Web UI server with multi-bucket graphs, decoupled streaming & chat UI
├── gemv_triton_v4.py     # Custom Triton GEMV kernels (Dense-T1 + Vector Acc)
├── packed_linear.py      # PackedLinear PyTorch module (E2M-ATQ buffers + KOTMS rotation)
├── benchmarks/           # Microbenchmarks & kernel breakdown suites
│   ├── bench_dense_t1.py
│   ├── bench_vector_acc.py
│   ├── bench_sampling_ultra.py
│   └── bench_cache_len_effect.py
├── tests/                # Verification tests & live API test client
│   ├── test_bucketed_graph.py
│   ├── test_live_server.py
│   └── test_sampling_quality.py
└── .gitignore
```

---

## Getting Started

### 1. Requirements
- Python 3.10+
- PyTorch 2.4+ with CUDA support
- Triton 3.0+
- Transformers, Accelerate

### 2. Launching the Web UI
```bash
python web_ui.py
```
Open your browser at `http://localhost:7860`.

---

## License
MIT License.

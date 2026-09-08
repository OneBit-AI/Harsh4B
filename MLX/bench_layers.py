"""Guarded linear-kernel microbenchmark using real sub-2B checkpoint matrices.

No model is instantiated or modified. The source BitNet quantization equation
is used only to construct frozen test fixtures; all backends receive the same
deployed weights and exactly the same activations. These are NOT model tok/s.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    if os.environ.get('MLX_GUARDED') != '1':
        raise SystemExit('Run via python -m MLX.safety')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='MLX/models/bitnet-700m')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 50 or Path(args.output).exists():
        raise SystemExit('Use 1..50 repetitions and a new output filename')
    import mlx.core as mx
    import numpy as np
    from safetensors import safe_open
    from .kernels import TernaryLinear, pack_ternary

    mx.set_memory_limit(512 * 1024 ** 2)  # allocator guideline, not a hard cap
    mx.set_cache_limit(16 * 1024 ** 2)
    mx.set_wired_limit(256 * 1024 ** 2)
    directory = Path(args.model)
    path = directory / 'model.safetensors'
    with safe_open(path, framework='np') as f:
        parameter_count = sum(int(np.prod(f.get_slice(k).get_shape())) for k in f.keys())
    if parameter_count >= 2_000_000_000:
        raise SystemExit('Checkpoint is not under 2B parameters')
    names = ['model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.down_proj.weight']
    result = dict(scope='linear microbenchmarks, not end-to-end inference', parameters=parameter_count,
                  source=json.loads((directory/'source.json').read_text()), mlx=mx.__version__,
                  device=mx.device_info(), dtype='float16', activation_policy='frozen input; no kernel-side quantization',
                  repeats=args.repeats, layers=[])
    rng = np.random.default_rng(27)

    def bench(fn):
        for _ in range(3):
            mx.eval(fn())
        mx.synchronize()
        durations = []
        for _ in range(args.repeats):
            begin = time.perf_counter_ns()
            output = fn()
            mx.eval(output)
            mx.synchronize()
            durations.append((time.perf_counter_ns() - begin) / 1000)
        return dict(median_us=statistics.median(durations), min_us=min(durations), max_us=max(durations))

    def amortized(fn, width):
        # Distinct inputs prevent reusing a previously evaluated result. One
        # evaluation covers 16 independent dispatches; report throughput, not
        # single-request latency. Same protocol and inputs for every backend.
        inputs = mx.array(np.random.default_rng(42).normal(size=(16,width)).astype(np.float16))
        mx.eval(inputs)
        batches = []
        for _ in range(5):
            start = time.perf_counter_ns()
            outputs = [fn(inputs[i:i+1]) for i in range(16)]
            mx.eval(*outputs)
            mx.synchronize()
            batches.append((time.perf_counter_ns()-start)/1000/16)
        return dict(median_us_per_dispatch=statistics.median(batches), dispatches_per_batch=16, batches=5)

    for name in names:
        with safe_open(path, framework='np') as f:
            # Source evaluation in FP16: preserve the published quantization rule.
            master = f.get_tensor(name).astype(np.float16).astype(np.float32)
        inv = np.float32(1) / np.maximum(np.abs(master).mean(dtype=np.float32), np.float32(1e-5))
        signs = np.clip(np.rint(master * inv), -1, 1).astype(np.int8)
        reference = (signs.astype(np.float32) / inv).astype(np.float16)
        deployed_scale = float(np.float16(np.float32(1) / inv))
        del master
        rows, width = signs.shape
        layer = TernaryLinear(mx.array(pack_ternary(signs)), mx.array([deployed_scale]), width)
        dense = mx.array(reference)
        mx.eval(layer.packed, layer.scale, dense)
        # Exact weight equality is required, not a loose numerical tolerance.
        np.testing.assert_array_equal(np.array(layer.dense_weight(mx.float16)), reference)
        max_error, equal_count, total_count = 0.0, 0, 0
        for kind in ('normal', 'small', 'zero', 'ones'):
            x_np = rng.normal(size=(1,width)).astype(np.float16)
            if kind == 'small': x_np *= np.float16(0.001)
            if kind == 'zero': x_np.fill(0)
            if kind == 'ones': x_np.fill(1)
            # Fixture follows the source per-token activation quantizer once.
            xf = x_np.astype(np.float32)
            inv_x = 127 / np.maximum(np.abs(xf).max(axis=-1, keepdims=True), 1e-5)
            x_np = (np.clip(np.rint(xf*inv_x), -128, 127)/inv_x).astype(np.float16)
            x = mx.array(x_np)
            actual, expected = np.array(layer(x)), np.array(x @ dense.T)
            np.testing.assert_allclose(actual, expected, rtol=0.002, atol=0.001)
            max_error = max(max_error, float(np.abs(actual.astype(np.float32)-expected.astype(np.float32)).max()))
            equal_count += int(np.count_nonzero(actual == expected))
            total_count += actual.size
        x = mx.array(rng.normal(size=(1,width)).astype(np.float16))
        mx.eval(x)
        timings = {'packed_metal': bench(lambda: layer(x)),
                   'resident_dense_mlx': bench(lambda: x @ dense.T),
                   'unpack_then_matmul_mlx': bench(lambda: layer.reference(x))}
        throughput = {'packed_metal': amortized(layer,width),
                      'resident_dense_mlx': amortized(lambda x: x @ dense.T,width),
                      'unpack_then_matmul_mlx': amortized(layer.reference,width)}
        row = dict(name=name, shape=[rows,width], weights_bit_exact=True,
                   max_abs_output_error=max_error, identical_output_fraction=equal_count/total_count,
                   packed_bytes=layer.packed.nbytes + layer.scale.nbytes, dense_bytes=dense.nbytes,
                   timings=timings,
                   amortized_throughput=throughput,
                   speedup_vs_resident_dense=timings['resident_dense_mlx']['median_us']/timings['packed_metal']['median_us'],
                   speedup_vs_unpack_each_call=timings['unpack_then_matmul_mlx']['median_us']/timings['packed_metal']['median_us'])
        result['layers'].append(row)
        print(json.dumps(row, indent=2), flush=True)
        del layer, dense, reference, signs, x
        mx.clear_cache()
    result['peak_mlx_bytes'] = mx.get_peak_memory()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()

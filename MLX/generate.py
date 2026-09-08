"""Guarded end-to-end BitNet-700M inference on MLX/Metal.

Reports prefill and decode tok/s and generated text. Run via python -m MLX.safety.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main():
    if os.environ.get('MLX_GUARDED') != '1':
        raise SystemExit('Run via python -m MLX.safety')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='MLX/models/bitnet-700m')
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--dense-decode', action='store_true',
                        help='baseline arm: resident dense weights, dense GEMV decode')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise SystemExit('Use a new output filename')
    import mlx.core as mx
    from tokenizers import Tokenizer

    from .bitnet_mlx import BitnetForCausalLM

    mx.set_memory_limit(3 * 1024 ** 3)  # allocator guideline, not a hard cap
    mx.set_cache_limit(32 * 1024 ** 2)
    mx.set_wired_limit(512 * 1024 ** 2)

    directory = Path(args.model)
    cfg = json.loads((directory / 'config.json').read_text())
    tok = Tokenizer.from_file(str(directory / 'tokenizer.json'))

    from safetensors import safe_open
    import numpy as np

    needed = {'model.embed_tokens.weight', 'model.norm.weight'}
    for i in range(cfg['num_hidden_layers']):
        p = f'model.layers.{i}'
        for n in ('input_layernorm', 'post_attention_layernorm'):
            needed.add(f'{p}.{n}.weight')
        for n in ('inner_attn_ln',):
            needed.add(f'{p}.self_attn.{n}.weight')
        for n in ('ffn_layernorm',):
            needed.add(f'{p}.mlp.{n}.weight')
        for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            needed.add(f'{p}.self_attn.{n}.weight')
        for n in ('gate_proj', 'up_proj', 'down_proj'):
            needed.add(f'{p}.mlp.{n}.weight')
    class LazyParams:
        """Reads one tensor at a time, releasing the file mapping after each read
        so checkpoint pages never stay resident for the whole run."""

        def __init__(self, path):
            self._path = path
            with safe_open(path, framework='np') as f:
                self._keys = set(f.keys())

        def __getitem__(self, name):
            with safe_open(self._path, framework='np') as f:
                return f.get_tensor(name)

    params = LazyParams(directory / 'model.safetensors')
    missing = needed - params._keys
    if missing:
        raise SystemExit(f'checkpoint missing tensors: {sorted(missing)[:5]}')
    print('Checkpoint indexed', flush=True)

    load_start = time.perf_counter()
    model = BitnetForCausalLM(cfg, params, dense_decode=args.dense_decode)
    mx.eval(model.embed)
    print(f'Model built + packed in {time.perf_counter() - load_start:.1f}s', flush=True)

    prompt_ids = tok.encode(args.prompt).ids
    print(f'Prompt: {args.prompt!r} ({len(prompt_ids)} tokens)', flush=True)

    gen_start = time.perf_counter()
    out_ids, prefill_time, decode_time = model.generate(
        mx.array(prompt_ids), args.max_tokens, temperature=args.temperature)
    total = time.perf_counter() - gen_start
    text = tok.decode(out_ids)
    decode_tok_s = (len(out_ids) - 1) / decode_time if len(out_ids) > 1 else 0.0
    print(f'Generated {len(out_ids)} tokens in {total:.2f}s '
          f'(prefill {prefill_time*1000:.1f}ms, decode {decode_time:.2f}s = {decode_tok_s:.1f} tok/s)', flush=True)
    print('--- OUTPUT ---')
    print(text, flush=True)

    result = dict(scope='end-to-end generation, MLX Metal', model=cfg.get('_name_or_path'),
                  parameters=int(model.embed.size) + sum(
                      int(l.packed.size) + int(l.scale.size)
                      for l in [m for layer in model.layers for m in
                                (layer.attn.q_proj.linear, layer.attn.k_proj.linear, layer.attn.v_proj.linear,
                                 layer.attn.o_proj.linear, layer.mlp.gate_proj.linear, layer.mlp.up_proj.linear,
                                 layer.mlp.down_proj.linear)]),
                  mlx=mx.__version__, device=mx.device_info(), prompt=args.prompt,
                  prompt_tokens=len(prompt_ids), generated_tokens=len(out_ids),
                  total_seconds=total, tok_per_second=len(out_ids) / total,
                  prefill_seconds=prefill_time, decode_seconds=decode_time,
                  decode_tok_per_second=decode_tok_s, dense_decode=args.dense_decode,
                  temperature=args.temperature, output=text, peak_mlx_bytes=mx.get_peak_memory())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()

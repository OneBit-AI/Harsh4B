#!/usr/bin/env python
"""Run BitNet-700M inference for a single query under the kill switch.

usage: python test.py "your query here"
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE / '.venv-mlx' / 'bin' / 'python') if (HERE / '.venv-mlx' / 'bin' / 'python').exists() else sys.executable


def supervised(query):
    if os.environ.get('MLX_GUARDED') == '1':
        return None
    log = HERE / 'MLX' / 'runs' / f"test-{time.strftime('%H%M%S')}.jsonl"
    cmd = [PY, '-B', '-m', 'MLX.safety', '--log', str(log), '--stop-file', str(HERE / 'MLX' / 'STOP'),
           '--max-seconds', '300', '--max-rss-gib', '6', '--reserve-gib', '2.5', '--', PY, '-B', __file__, query]
    code = subprocess.call(cmd)
    if code != 0 and not log.exists():
        print('watchdog refused to launch; if a stop latch exists, inspect the last run log, then clear it:')
        print(f'  {PY} -B -m MLX.safety --stop-file MLX/STOP --clear')
    return code


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: python test.py "<query>"')
    query = sys.argv[1]
    code = supervised(query)
    if code is not None:
        sys.exit(code)

    import mlx.core as mx
    import json
    from tokenizers import Tokenizer

    from MLX.bitnet_mlx import BitnetForCausalLM

    mx.set_memory_limit(3 * 1024 ** 3)
    mx.set_cache_limit(32 * 1024 ** 2)
    mx.set_wired_limit(512 * 1024 ** 2)

    directory = HERE / 'MLX' / 'models' / 'bitnet-700m'
    cfg = json.loads((directory / 'config.json').read_text())
    tok = Tokenizer.from_file(str(directory / 'tokenizer.json'))

    needed = {'model.embed_tokens.weight', 'model.norm.weight'}
    for i in range(cfg['num_hidden_layers']):
        p = f'model.layers.{i}'
        needed |= {f'{p}.input_layernorm.weight', f'{p}.post_attention_layernorm.weight',
                   f'{p}.self_attn.inner_attn_ln.weight', f'{p}.mlp.ffn_layernorm.weight'}
        for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            needed.add(f'{p}.self_attn.{n}.weight')
        for n in ('gate_proj', 'up_proj', 'down_proj'):
            needed.add(f'{p}.mlp.{n}.weight')

    from safetensors import safe_open

    class LazyParams:
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

    print('Building model (ternarize + pack)...', flush=True)
    t0 = time.perf_counter()
    model = BitnetForCausalLM(cfg, params)
    mx.eval(model.embed)
    print(f'Ready in {time.perf_counter() - t0:.1f}s\n', flush=True)

    ids = tok.encode(query).ids
    out, prefill_t, decode_t = model.generate(mx.array(ids), 200)
    text = tok.decode(out)
    tok_s = (len(out) - 1) / decode_t if len(out) > 1 else 0.0
    print('--- OUTPUT ---')
    print(text)
    print('--------------')
    print(f'{len(out)} tokens | prefill {len(ids)} tok in {prefill_t * 1000:.0f}ms | '
          f'decode {tok_s:.1f} tok/s')


if __name__ == '__main__':
    main()

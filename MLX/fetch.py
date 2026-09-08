"""Download a pinned, reviewed 700M BitNet checkpoint. Never execute remote code."""
import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download

MODEL_ID = '1bitLLM/bitnet_b1_58-large'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='MLX/models/bitnet-700m')
    parser.add_argument('--weights', action='store_true')
    args = parser.parse_args()
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'source.json'
    if manifest.exists():
        revision = json.loads(manifest.read_text())['revision']
    else:
        revision = HfApi().model_info(MODEL_ID).sha
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'README.md', 'utils_quant.py', 'modeling_bitnet.py']
    if args.weights:
        if shutil.disk_usage(directory).free < 6 * 1024 ** 3:
            raise RuntimeError('Need at least 6 GiB free disk space')
        names += ['model.safetensors']
    for name in names:
        path = hf_hub_download(MODEL_ID, name, revision=revision, local_dir=directory)
        print(path, flush=True)
    manifest.write_text(json.dumps({'model_id': MODEL_ID, 'revision': revision}, indent=2) + '\n')
    print(manifest.read_text())


if __name__ == '__main__':
    main()

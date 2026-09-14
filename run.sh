#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'Usage: ./run.sh "your prompt"' >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")" && pwd)"
python="$repo_dir/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo 'Missing .venv. Follow the setup commands in README.md first.' >&2
  exit 2
fi

exec "$python" -B "$repo_dir/cpu_lightspark.py" \
  --threads 8 \
  --prompt "$*" \
  --max-new-tokens 256 \
  --warmup-tokens 6

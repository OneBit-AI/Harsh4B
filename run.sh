#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'Usage: ./run.sh "your prompt"' >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$0")" && pwd)"
exec "$repo_dir/.venv-mlx/bin/python" -B "$repo_dir/test.py" \
  --runtime cpu \
  --cpu-threads 8 \
  --prompt "$*" \
  --max-new-tokens 64 \
  --warmup-tokens 8

#!/usr/bin/env bash
# Full serving + energy benchmark suite: TTFT/TPOT sweep + Joules/token.
#
# Runs benchmarks/bench_latency.py (TTFT/TPOT grid) and
# benchmarks/bench_energy.py (TTFT/TPOT + energy) for each runtime and
# collects timestamped JSON/CSV results under results/.
#
# Usage:
#   ./bench_all.sh                          # cpu + mlx, defaults
#   RUNTIMES="cpu" ./bench_all.sh           # cpu only
#   RUNTIMES="mlx" PROMPT_LENS="32,128,512,1024" REPEATS=5 ./bench_all.sh
#
# Env overrides:
#   PYTHON           python binary (default: .venv/bin/python)
#   RUNTIMES         space-separated runtimes (default: "cpu mlx")
#   PROMPT_LENS      e.g. "32,128,512"          (default, latency sweep)
#   DECODE_LENS      e.g. "32"                  (default, latency sweep)
#   REPEATS          repeats per cell/run       (default: 3)
#   WARMUP           untimed warmup runs        (default: 1)
#   ENERGY_TOKENS    --max-new-tokens for energy runs (default: 64)
#   CPU_THREADS      threads for the cpu runtime (default: 4)
#   RESULTS_DIR      output dir (default: results)
set -uo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"

PYTHON="${PYTHON:-$repo_dir/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Missing .venv. Follow the setup commands in README.md first.' >&2
  exit 2
fi
RUNTIMES="${RUNTIMES:-cpu mlx}"
PROMPT_LENS="${PROMPT_LENS:-32,128,512}"
DECODE_LENS="${DECODE_LENS:-32}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
ENERGY_TOKENS="${ENERGY_TOKENS:-64}"
CPU_THREADS="${CPU_THREADS:-4}"
RESULTS_DIR="${RESULTS_DIR:-results}"

echo "== benchmark suite =="
echo "python:   $PYTHON"
echo "runtimes: $RUNTIMES"
echo "latency:  prompts [$PROMPT_LENS] decodes [$DECODE_LENS] x$REPEATS (+$WARMUP warmup)"
echo "energy:   $ENERGY_TOKENS tokens x$REPEATS"

for f in model.lat.pt tokenizer.json benchmarks/bench_latency.py benchmarks/bench_energy.py; do
  [[ -f "$f" ]] || { echo "missing required file: $f" >&2; exit 2; }
done

ts="$(date +%Y%m%d_%H%M%S)"
out_dir="$RESULTS_DIR/$ts"
mkdir -p "$out_dir"
echo "results:  $out_dir"

failures=0
for rt in $RUNTIMES; do
  echo ""
  echo "----- runtime: $rt (latency sweep) -----"
  if "$PYTHON" -B benchmarks/bench_latency.py --runtime "$rt" \
      --prompt-lens "$PROMPT_LENS" --decode-lens "$DECODE_LENS" \
      --repeats "$REPEATS" --warmup "$WARMUP" --cpu-threads "$CPU_THREADS" \
      --json-out "$out_dir/latency_${rt}.json" \
      --csv-out "$out_dir/latency_${rt}.csv" \
      2>&1 | tee "$out_dir/latency_${rt}.log"; then
    echo "latency $rt: OK"
  else
    echo "latency $rt: FAILED (see $out_dir/latency_${rt}.log)" >&2
    failures=$((failures + 1))
    continue
  fi

  echo ""
  echo "----- runtime: $rt (energy) -----"
  if "$PYTHON" -B benchmarks/bench_energy.py --runtime "$rt" \
      --prompt-tokens 128 --max-new-tokens "$ENERGY_TOKENS" \
      --repeats "$REPEATS" --warmup "$WARMUP" --cpu-threads "$CPU_THREADS" \
      --json-out "$out_dir/energy_${rt}.json" \
      2>&1 | tee "$out_dir/energy_${rt}.log"; then
    echo "energy $rt: OK"
  else
    echo "energy $rt: FAILED (see $out_dir/energy_${rt}.log)" >&2
    failures=$((failures + 1))
  fi
done

echo ""
echo "== combined summary =="
OUT_DIR="$out_dir" "$PYTHON" - <<'EOF'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
for lat_path in sorted(out.glob("latency_*.json")):
    rt = lat_path.stem.split("_", 1)[1]
    try:
        lat = json.loads(lat_path.read_text())
    except Exception as exc:
        print(f"{rt}: unreadable {lat_path.name} ({exc})")
        continue
    print(f"--- {rt} ({lat.get('hardware', '?')}) ---")
    for c in lat["cells"]:
        print(f"  p={c['prompt_tokens']:5d} d={c['decode_tokens']:4d} | "
              f"TTFT {c['ttft_med_s']:.3f}s prefill {c['prefill_med_tok_s']:.0f}t/s | "
              f"TPOT med/p95 {c['tpot_med_ms']:.1f}/{c['tpot_p95_ms']:.1f}ms "
              f"dec {c['decode_med_tok_s']:.1f}t/s")
    en_path = out / f"energy_{rt}.json"
    if en_path.is_file():
        try:
            en = json.loads(en_path.read_text())
            print(f"  energy [{en.get('power_backend', '?')}]: "
                  f"{en.get('median_joules_per_token')} J/tok wall, "
                  f"{en.get('median_joules_per_token_incr')} J/tok incr, "
                  f"{en.get('median_tokens_per_joule')} tok/J")
        except Exception as exc:
            print(f"  energy: unreadable ({exc})")
EOF

if [[ "$failures" -gt 0 ]]; then
  echo "$failures run(s) FAILED; logs in $out_dir" >&2
  exit 1
fi
echo "all runs passed; results in $out_dir"

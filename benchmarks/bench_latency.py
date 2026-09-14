#!/usr/bin/env python3
"""TTFT/TPOT sweep across prompt lengths and decode lengths.

Single-point timing lives in bench_energy.py (which also measures Joules);
this script answers "how do TTFT and TPOT scale?" by sweeping a grid of
(prompt_len x decode_len) and reporting medians + p95 per cell.

Reuses the measurement core (timed_generate) from bench_energy so the two
scripts stay consistent. Runtime-agnostic via inference_runtime (cpu/mlx/cuda).

Definitions:
  TTFT        request start -> first generated token ready (prefill + sample)
  TPOT        per-token decode latency incl. sampling (median and p95 of ITLs)
  prefill_t/s prompt tokens / TTFT
  decode_t/s  decode tokens / decode wall time

Examples:
    .venv/bin/python benchmarks/bench_latency.py --runtime cpu
    .venv/bin/python benchmarks/bench_latency.py --runtime mlx \
        --prompt-lens 32,128,512,1024 --decode-lens 32,128 --repeats 3 \
        --json-out results/latency_mlx.json --csv-out results/latency_mlx.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # bench_energy
sys.path.insert(0, str(HERE.parent))  # inference_runtime

from bench_energy import (  # noqa: E402
    build_prompt_ids,
    percentile,
    timed_generate,
)


def parse_int_list(value: str) -> list[int]:
    try:
        items = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated ints, got {value!r}") from exc
    if not items or any(n < 1 for n in items):
        raise argparse.ArgumentTypeError("lengths must be positive ints")
    return items


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="auto",
                        help="auto/cpu/mlx/cuda (default: auto)")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--prompt-lens", type=parse_int_list,
                        default=[32, 128, 512],
                        help="prompt lengths in tokens (default: 32,128,512)")
    parser.add_argument("--decode-lens", type=parse_int_list, default=[32],
                        help="decode lengths in tokens (default: 32)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed runs per grid cell (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy/deterministic (default)")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def med(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def main():
    args = parse_args()
    from inference_runtime import create_runtime

    options: dict = {}
    if args.tokenizer:
        options["tokenizer_path"] = args.tokenizer
    if args.runtime in ("cpu", "auto"):
        options["cpu_threads"] = args.cpu_threads
    runtime = create_runtime(args.runtime, model_path=args.model, **options)
    print(f"runtime: {runtime.device} ({runtime.description})", flush=True)
    print(f"hardware: {runtime.hardware}", flush=True)
    print(f"grid: prompts {args.prompt_lens} x decodes {args.decode_lens} "
          f"x{args.repeats} repeats (+{args.warmup} warmup/cell)", flush=True)

    cells = []
    for prompt_len in args.prompt_lens:
        prompt_ids = build_prompt_ids(runtime.tokenizer, "", False, prompt_len)
        assert len(prompt_ids) == prompt_len, (
            len(prompt_ids), prompt_len)
        for decode_len in args.decode_lens:
            for _ in range(args.warmup):
                timed_generate(runtime, prompt_ids, decode_len,
                               args.temperature)
            ttfts, prefills, tpot_med, tpot_p95, dec_tps, e2e_tps = (
                [], [], [], [], [], [])
            for rep in range(args.repeats):
                t0 = time.perf_counter()
                result = timed_generate(runtime, prompt_ids, decode_len,
                                        args.temperature)
                wall_s = time.perf_counter() - t0
                n_gen = len(result["generated"])
                itls = result["itls"]
                ttfts.append(result["ttft"])
                prefills.append(len(prompt_ids) / result["prefill_s"]
                                if result["prefill_s"] else 0.0)
                tpot_med.append(med(itls) * 1000 if itls else 0.0)
                tpot_p95.append(percentile(itls, 95) * 1000 if itls else 0.0)
                dec_tps.append(len(itls) / result["decode_s"]
                               if result["decode_s"] else 0.0)
                e2e_tps.append(n_gen / wall_s if wall_s else 0.0)
            cell = dict(
                prompt_tokens=prompt_len, decode_tokens=decode_len,
                generated_tokens=n_gen,
                ttft_med_s=round(med(ttfts), 4),
                ttft_min_s=round(min(ttfts), 4),
                ttft_max_s=round(max(ttfts), 4),
                prefill_med_tok_s=round(med(prefills), 2),
                tpot_med_ms=round(med(tpot_med), 2),
                tpot_p95_ms=round(med(tpot_p95), 2),
                decode_med_tok_s=round(med(dec_tps), 2),
                e2e_med_tok_s=round(med(e2e_tps), 2),
            )
            cells.append(cell)
            print(f"[p={prompt_len:5d} d={decode_len:4d}] "
                  f"TTFT med {cell['ttft_med_s']:.3f}s "
                  f"({cell['ttft_min_s']:.3f}-{cell['ttft_max_s']:.3f}) "
                  f"prefill {cell['prefill_med_tok_s']:.0f}tok/s | "
                  f"TPOT med/p95 {cell['tpot_med_ms']:.1f}/"
                  f"{cell['tpot_p95_ms']:.1f}ms "
                  f"decode {cell['decode_med_tok_s']:.1f}tok/s", flush=True)

    print("\n--- TTFT/TPOT summary (medians) ---")
    print(f"{'prompt':>6} {'decode':>6} | {'TTFT(s)':>7} {'pref_t/s':>8} | "
          f"{'TPOTmed':>7} {'TPOTp95':>7} {'dec_t/s':>7} {'e2e_t/s':>7}")
    for cell in cells:
        print(f"{cell['prompt_tokens']:>6} {cell['decode_tokens']:>6} | "
              f"{cell['ttft_med_s']:>7.3f} {cell['prefill_med_tok_s']:>8.0f} | "
              f"{cell['tpot_med_ms']:>7.1f} {cell['tpot_p95_ms']:>7.1f} "
              f"{cell['decode_med_tok_s']:>7.1f} {cell['e2e_med_tok_s']:>7.1f}")

    summary = dict(runtime=runtime.device, hardware=runtime.hardware,
                   repeats=args.repeats, cells=cells)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
            writer.writeheader()
            writer.writerows(cells)
        print(f"wrote {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()

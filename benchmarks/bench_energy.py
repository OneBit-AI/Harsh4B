#!/usr/bin/env python3
"""Serving + energy benchmark: TTFT, TPOT, throughput and Joules/token.

Covers the gap in benchmarks/: existing scripts only report decode-only
tok/s (bench_pure_decode, bench_unpacked_graph/step, bench_cache_len_effect)
or micro-kernel us (bench_rot*, bench_dense_t1, bench_sampling_*). None
measures TTFT, TPOT percentiles, or energy. This script does, on any runtime
in inference_runtime (cpu / mlx / cuda).

Power backends (auto-selected, no new dependencies):
  powermetrics  macOS, `sudo -n powermetrics -s cpu_power,gpu_power` (best)
  nvml+rapl     Linux NVIDIA / Intel RAPL
  battery       macOS without sudo: ioreg Voltage*Amperage while discharging
  none          time-only fallback (Joules reported as null)

Method: poll power in a background thread, integrate P over the measured
window (E = sum P*dt). An idle baseline (--idle-seconds) is sampled before
the runs and subtracted: E_incr = E_total - P_idle * t. Joules/token uses
incremental energy over actually generated tokens.

Examples:
    .venv/bin/python benchmarks/bench_energy.py --runtime mlx --max-new-tokens 64
    .venv/bin/python benchmarks/bench_energy.py --runtime cpu --prompt-tokens 128 --max-new-tokens 64 --repeats 3
    sudo -n true && .venv/bin/python benchmarks/bench_energy.py --runtime mlx  # unlock powermetrics
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

IDLE_DEFAULT_S = 3.0
POLL_INTERVAL_S = 0.1


# ---------------------------------------------------------------- power ---
class PowerSampler:
    """Background power poller. Samples are (timestamp, watts)."""

    name = "none"

    def __init__(self, interval=POLL_INTERVAL_S):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def read_watts(self) -> float | None:
        return None

    def start(self):
        self.samples = []
        self._stop.clear()
        first = self.read_watts()
        if first is not None:
            self.samples.append((time.perf_counter(), first))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            watts = self.read_watts()
            if watts is not None:
                self.samples.append((time.perf_counter(), watts))

    def stop(self) -> tuple[float, float]:
        """Return (joules, avg_watts) integrated over the sampled window."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if len(self.samples) < 2:
            return 0.0, 0.0
        joules = 0.0
        for (t0, p0), (t1, _) in zip(self.samples, self.samples[1:]):
            joules += p0 * (t1 - t0)
        span = self.samples[-1][0] - self.samples[0][0]
        return joules, (joules / span if span > 0 else 0.0)


class PowermetricsSampler(PowerSampler):
    """macOS powermetrics CPU+GPU power (requires passwordless sudo)."""

    name = "powermetrics"

    def __init__(self, interval_ms=100):
        super().__init__(interval_ms / 1000)
        self._proc: subprocess.Popen | None = None
        self._latest: float | None = None
        self._lock = threading.Lock()

    def _available(self) -> bool:
        try:
            probe = subprocess.run(
                ["sudo", "-n", "powermetrics", "-s", "cpu_power",
                 "-i", "200", "-n", "1"],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return "Power" in probe.stdout

    def read_watts(self):
        with self._lock:
            return self._latest

    def start(self):
        self._proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "-s", "cpu_power,gpu_power",
             "-i", "100", "-n", "1000000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._reader = threading.Thread(target=self._parse, daemon=True)
        self._reader.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    break
            time.sleep(0.05)
        super().start()

    def _parse(self):
        cpu = gpu = 0.0
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            m = re.match(r"(CPU|GPU) Power:\s+([\d.]+)\s*mW", line)
            if m:
                if m.group(1) == "CPU":
                    cpu = float(m.group(2)) / 1000.0
                else:
                    gpu = float(m.group(2)) / 1000.0
                    with self._lock:
                        self._latest = cpu + gpu

    def stop(self):
        result = super().stop()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        return result


class BatterySampler(PowerSampler):
    """macOS without sudo: battery Voltage*Amperage while discharging."""

    name = "battery"

    def read_watts(self):
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-c", "AppleSmartBattery", "-l"],
                capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        v = re.search(r'"Voltage"\s*=\s*(\d+)', out)
        a = re.search(r'"Amperage"\s*=\s*(\d+)', out)
        ext = re.search(r'"ExternalConnected"\s*=\s*(Yes|No)', out)
        if not v or not a:
            return None
        amps = int(a.group(1))
        if amps > 2**63:  # unsigned 64-bit two's complement
            amps -= 2**64
        if ext and ext.group(1) == "Yes":
            return None  # on AC: battery current does not reflect load
        return abs(int(v.group(1)) * amps) / 1e6 if amps != 0 else None


class NvmlSampler(PowerSampler):
    """Linux NVIDIA GPU board power via pynvml (optional dependency)."""

    name = "nvml"

    def __init__(self, interval=POLL_INTERVAL_S):
        super().__init__(interval)
        import pynvml
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def read_watts(self):
        try:
            return self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            return None


class RaplSampler:
    """Linux Intel/AMD CPU energy counters (no thread needed)."""

    name = "rapl"

    def __init__(self):
        import glob
        self.paths = sorted(glob.glob(
            "/sys/class/powercap/intel-rapl:*/energy_uj"))
        if not self.paths:
            raise RuntimeError("no RAPL energy_uj counters")
        self._t0 = self._e0 = 0.0

    def _read_uj(self) -> int:
        total = 0
        for path in self.paths:
            with open(path) as fh:
                total += int(fh.read().strip())
        return total

    def start(self):
        self._t0 = time.perf_counter()
        self._e0 = self._read_uj()

    def stop(self) -> tuple[float, float]:
        joules = (self._read_uj() - self._e0) / 1e6
        span = time.perf_counter() - self._t0
        return joules, (joules / span if span > 0 else 0.0)


def select_sampler(preferred: str):
    """Return an energy sampler instance or None for time-only mode."""
    if preferred == "none":
        return None
    candidates: list = []
    if preferred in ("auto", "powermetrics") and sys.platform == "darwin":
        candidates.append(PowermetricsSampler)
    if preferred in ("auto", "nvml", "rapl"):
        candidates.append(NvmlSampler)
        candidates.append(RaplSampler)
    if preferred in ("auto", "battery") and sys.platform == "darwin":
        candidates.append(BatterySampler)
    if preferred not in ("auto", "none"):
        candidates = [c for c in candidates
                      if c.name == preferred] or candidates
    for cls in candidates:
        try:
            sampler = cls() if cls is not RaplSampler else cls()
            if cls is PowermetricsSampler and not sampler._available():
                continue
            if isinstance(sampler, BatterySampler) and sampler.read_watts() is None:
                continue  # on AC power or no battery: unusable
            return sampler
        except Exception:
            continue
    return None


# --------------------------------------------------------------- bench ---
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def sync_logits(runtime, logits):
    if runtime.device == "mlx":
        runtime.mx.eval(logits)
    elif runtime.device == "cuda":
        runtime.torch.cuda.synchronize()


def timed_generate(runtime, prompt_ids: list[int], max_new_tokens: int,
                   temperature: float):
    """One measured generation. Returns dict with TTFT/TPOT/throughput."""
    itls: list[float] = []  # per-token decode latencies incl. sampling
    generated: list[int] = []

    t_start = time.perf_counter()
    logits, session = runtime.prefill(prompt_ids, max_new_tokens)
    sync_logits(runtime, logits)
    token = runtime.sample(logits, generated, temperature)
    ttft = time.perf_counter() - t_start
    prefill_s = ttft  # prefill dominates; sample cost is included in TTFT
    if token in runtime.tokenizer.stop_ids:
        return dict(ttft=ttft, itls=[], generated=generated,
                    prefill_s=prefill_s, decode_s=0.0)

    generated.append(token)
    t_dec0 = time.perf_counter()
    while len(generated) < max_new_tokens:
        t0 = time.perf_counter()
        logits = runtime.decode(token, session)
        sync_logits(runtime, logits)
        token = runtime.sample(logits, generated, temperature)
        itls.append(time.perf_counter() - t0)
        if token in runtime.tokenizer.stop_ids:
            break
        generated.append(token)
    decode_s = time.perf_counter() - t_dec0
    return dict(ttft=ttft, itls=itls, generated=generated,
                prefill_s=prefill_s, decode_s=decode_s)


def build_prompt_ids(tokenizer, prompt: str, raw: bool, prompt_tokens: int | None):
    if prompt_tokens:
        base = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
        if not base:
            raise SystemExit("tokenizer returned no ids for the filler text")
        tiled = (base * ((prompt_tokens // len(base)) + 1))[:prompt_tokens]
        return tiled
    text = prompt if raw else tokenizer.format(
        "", [{"role": "user", "content": prompt}], budget=2048)
    ids = tokenizer.encode(text)
    if not ids:
        raise SystemExit("prompt encoded to zero tokens")
    return ids


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="auto",
                        help="auto/cpu/mlx/cuda (default: auto)")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--prompt", default="Explain ternary quantization in one sentence.",
                        help="chat prompt (ignored with --prompt-tokens)")
    parser.add_argument("--raw-prompt", action="store_true",
                        help="skip the Qwen chat template")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="synthetic prompt of exactly N tokens (sweep prefill)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy/deterministic (default)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--power", default="auto",
                        choices=["auto", "powermetrics", "nvml", "rapl",
                                 "battery", "none"],
                        help="energy backend (default: auto)")
    parser.add_argument("--idle-seconds", type=float, default=IDLE_DEFAULT_S)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


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

    prompt_ids = build_prompt_ids(runtime.tokenizer, args.prompt,
                                  args.raw_prompt, args.prompt_tokens)
    print(f"prompt: {len(prompt_ids)} tokens, "
          f"generating up to {args.max_new_tokens} tokens x{args.repeats}",
          flush=True)

    sampler = select_sampler(args.power)
    if sampler is None:
        print("power: none available -> time-only mode (Joules=null). "
              "On macOS run `sudo -n true` first for powermetrics, or run on "
              "battery for the ioreg fallback.", flush=True)
    else:
        print(f"power: {sampler.name}", flush=True)

    for _ in range(args.warmup):
        timed_generate(runtime, prompt_ids, args.max_new_tokens,
                       args.temperature)
    print(f"warmup: {args.warmup} run(s) done", flush=True)

    idle_watts = 0.0
    if sampler is not None and args.idle_seconds > 0:
        sampler.start()
        time.sleep(args.idle_seconds)
        idle_j, idle_watts = sampler.stop()
        span = max(idle_j / idle_watts, 0) if idle_watts else 0
        print(f"idle baseline: {idle_watts:.2f} W over {span:.1f}s", flush=True)

    rows = []
    for rep in range(args.repeats):
        sampler.start() if sampler else None
        t0 = time.perf_counter()
        result = timed_generate(runtime, prompt_ids, args.max_new_tokens,
                                args.temperature)
        wall_s = time.perf_counter() - t0
        energy_j, avg_w = sampler.stop() if sampler else (0.0, 0.0)

        n_gen = len(result["generated"])
        n_decode_steps = len(result["itls"])
        incr_j = max(energy_j - idle_watts * wall_s, 0.0) if sampler else 0.0
        itls = result["itls"]
        # Total J/token is the robust wall number; incremental subtracts the
        # idle baseline but is noisy on short runs (battery ≈ whole system).
        jpt_total = (energy_j / n_gen) if sampler and n_gen else None
        jpt_incr = (incr_j / n_gen) if sampler and n_gen else None
        row = dict(
            rep=rep,
            prompt_tokens=len(prompt_ids),
            generated_tokens=n_gen,
            ttft_s=round(result["ttft"], 4),
            prefill_tok_s=round(len(prompt_ids) / result["prefill_s"], 2)
            if result["prefill_s"] else 0.0,
            tpot_mean_ms=round(statistics.mean(itls) * 1000, 2) if itls else 0.0,
            tpot_median_ms=round(statistics.median(itls) * 1000, 2) if itls else 0.0,
            tpot_p95_ms=round(percentile(itls, 95) * 1000, 2) if itls else 0.0,
            decode_tok_s=round(n_decode_steps / result["decode_s"], 2)
            if result["decode_s"] else 0.0,
            e2e_tok_s=round(n_gen / wall_s, 2) if wall_s else 0.0,
            total_joules=round(energy_j, 3) if sampler else None,
            incremental_joules=round(incr_j, 3) if sampler else None,
            avg_watts=round(avg_w, 2) if sampler else None,
            joules_per_token=round(jpt_total, 4) if jpt_total is not None else None,
            joules_per_token_incr=round(jpt_incr, 4) if jpt_incr is not None else None,
            tokens_per_joule=round(n_gen / energy_j, 3)
            if sampler and energy_j > 0 else None,
        )
        rows.append(row)
        jpt = f"{row['joules_per_token']:.4f} J/tok (incr {row['joules_per_token_incr']:.4f})" if row['joules_per_token'] is not None else "J/tok=n/a"
        print(f"[rep {rep}] TTFT={row['ttft_s']:.3f}s "
              f"TPOT median/p95={row['tpot_median_ms']:.1f}/{row['tpot_p95_ms']:.1f}ms "
              f"decode={row['decode_tok_s']:.1f}tok/s {jpt} "
              f"({row['generated_tokens']} tok in {wall_s:.1f}s)", flush=True)

    def med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(vals), 4) if vals else None

    summary = dict(
        runtime=runtime.device, hardware=runtime.hardware,
        power_backend=sampler.name if sampler else "none",
        idle_watts=round(idle_watts, 2),
        median_ttft_s=med("ttft_s"),
        median_tpot_median_ms=med("tpot_median_ms"),
        median_tpot_p95_ms=med("tpot_p95_ms"),
        median_decode_tok_s=med("decode_tok_s"),
        median_joules_per_token=med("joules_per_token"),
        median_joules_per_token_incr=med("joules_per_token_incr"),
        median_tokens_per_joule=med("tokens_per_joule"),
        repeats=rows,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "repeats"},
                     indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()

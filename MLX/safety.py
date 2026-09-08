"""Independent, fail-closed macOS watchdog. Never imports MLX or a model.

Run: python -m MLX.safety --log run.jsonl --stop-file STOP -- command ...
Kill: python -m MLX.safety --stop-file STOP --kill
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import psutil

GIB = 1024 ** 3


@dataclass(frozen=True)
class Limits:
    reserve_bytes: int = 4 * GIB
    max_rss_bytes: int = 3 * GIB
    max_swap_growth: int = 0
    max_seconds: float = 120
    sample_seconds: float = 0.1
    sensor_timeout: float = 1.0


def violation(sample, baseline_swap, limits):
    """Pure decision function: missing/unknown sensors are failures, too."""
    required = ('thermal', 'pressure', 'available', 'swap', 'rss')
    if any(k not in sample or sample[k] is None for k in required):
        return 'missing sensor'
    if sample['thermal'] != 0:
        return 'thermal state is not nominal'
    if sample['pressure'] != 1:
        return 'memory pressure is not normal'
    if sample['available'] < limits.reserve_bytes:
        return 'system memory reserve breached'
    if sample['swap'] - baseline_swap > limits.max_swap_growth:
        return 'swap grew'
    if sample['rss'] > limits.max_rss_bytes:
        return 'inference RSS limit exceeded'
    return None


class Sensors:
    def __init__(self):
        if sys.platform != 'darwin':
            raise RuntimeError('macOS thermal sensors required')
        self.foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        self.objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        self.send_obj = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', self.objc))
        self.send_int = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', self.objc))
        self.info = self.send_obj(self.objc.objc_getClass(b'NSProcessInfo'), self.objc.sel_registerName(b'processInfo'))
        if not self.info:
            raise RuntimeError('NSProcessInfo unavailable')
        self.thermal_sel = self.objc.sel_registerName(b'thermalState')
        self.libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        self.libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]

    def read(self):
        pressure, size = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        if self.libc.sysctlbyname(b'kern.memorystatus_vm_pressure_level', ctypes.byref(pressure), ctypes.byref(size), None, 0) != 0:
            raise OSError(ctypes.get_errno(), 'memory pressure sensor failed')
        return dict(thermal=self.send_int(self.info, self.thermal_sel), pressure=pressure.value,
                    available=psutil.virtual_memory().available, swap=psutil.swap_memory().used,
                    rss=0, sample_time=time.monotonic())


def sensor_worker(out, interval):
    try:
        sensors = Sensors()
        while True:
            out.put(sensors.read(), timeout=1)
            time.sleep(interval)
    except BaseException as exc:
        try:
            out.put({'error': repr(exc)}, timeout=0.1)
        except Exception:
            pass


def kill_group(pid):
    """The pid must belong to the session created by this supervisor."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def tree_rss(pid):
    try:
        process = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [process] + process.children(recursive=True) if p.is_running())
    except psutil.NoSuchProcess:
        return 0


def lifeline_worker(owner_pid, workload_pid, ready, stop_file):
    """Survives a SIGKILL of the main supervisor and kills its gated workload."""
    try:
        owner = psutil.Process(owner_pid)
        ready.set()
        while owner.is_running() and owner.status() != psutil.STATUS_ZOMBIE:
            time.sleep(0.1)
    finally:
        kill_group(workload_pid)
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).touch(exist_ok=True)


def supervise(command, log, stop_file, limits=Limits(), *, _sensor_target=sensor_worker):
    """Checks sensors before launch, then watches a separately-sessioned worker.

    A separate sensor process isolates stalled sensor calls. The supervisor also
    watches its caller: loss of the launching terminal/parent kills inference.
    """
    log, stop_file = Path(log), Path(stop_file)
    log.parent.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        raise RuntimeError(f'Stop latch exists: {stop_file}; inspect it before a new run')
    if not command or limits.max_seconds <= 0 or limits.sample_seconds <= 0:
        raise ValueError('command and positive time limits required')
    owner = psutil.Process(os.getppid())
    ctx = mp.get_context('spawn')
    samples = ctx.Queue(maxsize=4)
    sensor = ctx.Process(target=_sensor_target, args=(samples, limits.sample_seconds), daemon=True)
    child, lifeline, gate_write = None, None, None
    reason, code = None, 125
    started = time.monotonic()
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[sig] = signal.signal(sig, interrupted)
    with log.open('x') as output:
        def emit(event, **fields):
            output.write(json.dumps(dict(event=event, elapsed=time.monotonic() - started, **fields)) + '\n')
            output.flush()
        try:
            sensor.start()
            first = samples.get(timeout=5)
            if 'error' in first:
                raise RuntimeError(first['error'])
            baseline_swap = first.get('swap', 0)
            reason = violation(first, baseline_swap, limits)
            if reason:
                raise RuntimeError('preflight: ' + reason)
            if stop_file.exists():
                raise RuntimeError('manual stop before launch')
            env = dict(os.environ, MLX_GUARDED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            gate_read, gate_write = os.pipe()
            try:
                child = subprocess.Popen([sys.executable, '-B', '-m', 'MLX.gated_worker', str(gate_read), *command],
                                         pass_fds=(gate_read,), start_new_session=True, env=env)
            finally:
                os.close(gate_read)
            ready = ctx.Event()
            lifeline = ctx.Process(target=lifeline_worker, args=(os.getpid(), child.pid, ready, str(stop_file)), daemon=True)
            lifeline.start()
            if not ready.wait(timeout=3) or not lifeline.is_alive():
                raise RuntimeError('secondary watchdog failed to arm')
            os.write(gate_write, b'G')
            os.close(gate_write)
            gate_write = None
            emit('armed', pid=child.pid, limits=asdict(limits), baseline=first, command=command)
            print(f'WATCHDOG ARMED pid={child.pid}; stop latch: {stop_file}', flush=True)
            while child.poll() is None:
                if not lifeline.is_alive():
                    reason = 'secondary watchdog lost'
                    break
                if stop_file.exists():
                    reason = 'manual stop'
                    break
                if not owner.is_running() or owner.status() == psutil.STATUS_ZOMBIE:
                    reason = 'launching parent disappeared'
                    break
                if time.monotonic() - started >= limits.max_seconds:
                    reason = 'wall time limit'
                    break
                try:
                    sample = samples.get(timeout=limits.sensor_timeout)
                except queue.Empty:
                    reason = 'sensor heartbeat lost'
                    break
                if 'error' in sample:
                    reason = 'sensor error: ' + sample['error']
                    break
                if time.monotonic() - sample['sample_time'] > limits.sensor_timeout:
                    reason = 'stale sensor sample'
                    break
                sample['rss'] = tree_rss(child.pid)
                reason = violation(sample, baseline_swap, limits)
                emit('sample', **sample)
                if reason:
                    break
            if reason:
                kill_group(child.pid)
                code = 137
            else:
                code = child.wait()
        except BaseException as exc:
            reason = f'{type(exc).__name__}: {exc}'
            if child is not None:
                kill_group(child.pid)
                code = 137
        finally:
            if gate_write is not None:
                os.close(gate_write)
            if child is not None:
                # Kill any remaining descendants in the owned group on all exits.
                kill_group(child.pid)
                child.wait(timeout=5)
            if lifeline is not None:
                if lifeline.is_alive():
                    lifeline.kill()
                lifeline.join(timeout=2)
            if sensor.is_alive():
                sensor.kill()
            if sensor.pid is not None:
                sensor.join(timeout=2)
            samples.close()
            if reason:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch(exist_ok=True)
            emit('finished', reason=reason, returncode=code)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    print(f'WATCHDOG FINISHED code={code} reason={reason}', flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', default='MLX/runs/watchdog.jsonl')
    parser.add_argument('--stop-file', default='MLX/STOP')
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--max-rss-gib', type=float, default=3.0,
                        help='process-tree RSS cap in GiB; raise for workloads whose '
                             'mmap\'d checkpoint file pages count toward RSS')
    parser.add_argument('--reserve-gib', type=float, default=4.0,
                        help='minimum free system memory to maintain, in GiB')
    parser.add_argument('--kill', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.kill:
        Path(args.stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stop_file).touch(exist_ok=True)
        print('Stop latch set; the watchdog kills the owned inference group on its next check.')
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    return supervise(command, args.log, args.stop_file,
                     Limits(max_seconds=args.max_seconds,
                            max_rss_bytes=int(args.max_rss_gib * GIB),
                            reserve_bytes=int(args.reserve_gib * GIB)))


if __name__ == '__main__':
    sys.exit(main())

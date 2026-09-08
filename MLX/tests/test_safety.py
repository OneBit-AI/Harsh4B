import json
from pathlib import Path
import sys
import time
import os
import signal
import subprocess
import threading

import psutil
import pytest

from MLX.safety import GIB, Limits, supervise, violation


def healthy():
    return dict(thermal=0, pressure=1, available=8*GIB, swap=0, rss=0, sample_time=time.monotonic())


def thermal_fault(out, interval):
    out.put(healthy())
    time.sleep(0.15)
    out.put(dict(healthy(), thermal=1))


def missing_sensor(out, interval):
    out.put({'error': 'injected sensor unavailable'})


def lost_sensor(out, interval):
    out.put(healthy())
    time.sleep(10)


def healthy_stream(out, interval):
    while True:
        out.put(healthy())
        time.sleep(interval)


@pytest.mark.parametrize('field,value', [('thermal', 1), ('thermal', 2), ('thermal', 3), ('pressure', 2), ('available', GIB), ('swap', 1), ('rss', 4*GIB), ('thermal', None)])
def test_trip_conditions(field, value):
    assert violation(dict(healthy(), **{field: value}), 0, Limits())


def test_healthy():
    assert violation(healthy(), 0, Limits()) is None


@pytest.mark.parametrize('sensor,expected', [(thermal_fault, 'thermal'), (lost_sensor, 'heartbeat')])
def test_supervisor_really_kills_child(tmp_path, sensor, expected):
    log = tmp_path / 'guard.jsonl'
    start = time.monotonic()
    rc = supervise([sys.executable, '-c', 'import time; time.sleep(60)'], log, tmp_path / 'STOP',
                   Limits(sensor_timeout=0.3), _sensor_target=sensor)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    pid = next(e['pid'] for e in events if e['event'] == 'armed')
    assert rc == 137
    assert expected in events[-1]['reason']
    assert not psutil.pid_exists(pid)
    assert (tmp_path / 'STOP').exists()
    assert time.monotonic() - start < 5


def test_preflight_failure_never_launches(tmp_path):
    log = tmp_path / 'guard.jsonl'
    rc = supervise([sys.executable, '-c', 'raise SystemExit(99)'], log, tmp_path / 'STOP', _sensor_target=missing_sensor)
    assert rc == 125
    assert 'armed' not in log.read_text()


def test_stop_latch_prevents_restart(tmp_path):
    stop = tmp_path / 'STOP'
    stop.touch()
    with pytest.raises(RuntimeError, match='Stop latch'):
        supervise(['never-run-this'], tmp_path / 'guard.jsonl', stop)


def test_manual_stop_kills_workload(tmp_path):
    log, stop = tmp_path/'guard.jsonl', tmp_path/'STOP'
    def stop_after_launch():
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            if log.exists() and 'armed' in log.read_text():
                stop.touch()
                return
            time.sleep(0.01)
    setter = threading.Thread(target=stop_after_launch)
    setter.start()
    try:
        rc = supervise([sys.executable,'-c','import time; time.sleep(60)'], log,stop,
                       Limits(max_seconds=5),_sensor_target=healthy_stream)
        assert rc == 137
        assert json.loads(log.read_text().splitlines()[-1])['reason'] == 'manual stop'
    finally:
        setter.join(timeout=5)


def test_deadline_kills_workload(tmp_path):
    log = tmp_path/'guard.jsonl'
    rc = supervise([sys.executable,'-c','import time; time.sleep(60)'],log,tmp_path/'STOP',
                   Limits(max_seconds=0.3),_sensor_target=healthy_stream)
    assert rc == 137
    assert json.loads(log.read_text().splitlines()[-1])['reason'] == 'wall time limit'


def test_secondary_watchdog_survives_supervisor_sigkill(tmp_path):
    log, stop = tmp_path/'guard.jsonl',tmp_path/'STOP'
    script = ('import sys; from MLX.safety import supervise; '
              'from MLX.tests.test_safety import healthy_stream; '
              'supervise([sys.executable,"-c","import time; time.sleep(60)"], '
              'sys.argv[1],sys.argv[2],_sensor_target=healthy_stream)')
    parent = subprocess.Popen([sys.executable,'-B','-c',script,str(log),str(stop)],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    child_pid = None
    try:
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            if log.exists():
                for line in log.read_text().splitlines():
                    event = json.loads(line)
                    if event['event'] == 'armed': child_pid = event['pid']
            if child_pid: break
            time.sleep(0.01)
        assert child_pid, 'supervisor did not arm'
        parent.kill()
        parent.wait(timeout=2)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            gone = not psutil.pid_exists(child_pid)
            if not gone:
                gone = psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
            if gone and stop.exists(): return
            time.sleep(0.01)
        pytest.fail('workload survived supervisor SIGKILL')
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if child_pid is not None:
            try: os.killpg(child_pid,signal.SIGKILL)
            except ProcessLookupError: pass

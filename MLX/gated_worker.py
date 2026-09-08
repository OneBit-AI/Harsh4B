"""Do not exec a workload until both watchdog processes have armed."""
import os
import sys

fd = int(sys.argv[1])
with os.fdopen(fd, 'rb', buffering=0) as gate:
    if gate.read(1) != b'G':
        raise SystemExit('Supervisor disappeared before arming')
command = sys.argv[2:]
os.execvpe(command[0], command, os.environ)

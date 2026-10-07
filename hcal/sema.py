"""Shared heavy-process semaphore of the held-out calibration study (hcal). A copy of lra/sema.py at 9762025 (itself
qpc/sema.py incl. AMENDMENT_A1 signal handling); only the store path and this docstring differ. Standalone (no hcal
imports), so the independent verifier may run it by path with python -P.

At most TWO heavy processes in total (study workers, attacker fitting, verification, restore checks, heavy synthetic
timing). Every heavy command runs as a child of this wrapper, which holds one of two flock slots for the child's whole
lifetime (released by the kernel if anything dies; a dead owner's flock is released automatically, which is the
stale-owner recovery):

    OMP_NUM_THREADS=1 <python> -m hcal.sema --label <role:what> -- <command ...>
    <python> -P <WORKTREE>/hcal/sema.py --label E:verify -- <command ...>
    <python> -m hcal.sema status

The wrapper blocks (polling every 5 s) until a slot is free. Every acquire/release is appended to
<PRIVATE_CACHE>/hcal_v1/run/SEMA_LOG.jsonl with the label, slot, owner pids, wall time and the child's CPU time.
"""
from __future__ import annotations

import fcntl
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(os.environ.get("PCRL_HCAL_PRIVATE_CACHE") or (Path.home() / "PCRL_eval_cache_private" / "hcal_v1")) / "run"
SLOTS = 2


def _log(rec):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "SEMA_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}) + "\n")


def acquire(label, poll=5.0):
    (RUN / "sema").mkdir(parents=True, exist_ok=True)
    waited = 0.0
    while True:
        for s in range(SLOTS):
            fh = open(RUN / "sema" / f"slot{s}.lock", "a+")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({"label": label, "pid": os.getpid()}))
            fh.flush()
            return s, fh, waited
        time.sleep(poll)
        waited += poll


def status():
    out = {}
    for s in range(SLOTS):
        p = RUN / "sema" / f"slot{s}.lock"
        if not p.exists():
            out[s] = "free"
            continue
        fh = open(p, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
            out[s] = "free"
        except BlockingIOError:
            fh.seek(0)
            out[s] = "held " + fh.read()
        fh.close()
    return out


def run(label, cmd):
    slot, fh, waited = acquire(label)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    _log({"event": "acquire", "label": label, "slot": slot, "wrapper_pid": os.getpid(), "waited_s": waited,
          "cmd0": os.path.basename(cmd[0]) if cmd else None})
    rc = None
    child = None
    got = []

    def _stop(signum, frame):                    # AMENDMENT_A1: a signalled wrapper never orphans its child; the
        got.append(int(signum))                  # handler only forwards the signal (no wait: Popen.wait's lock is
        if child is not None and child.poll() is None:   # held by the interrupted main frame) and the main loop
            child.terminate()                    # reaps the child, logs the signal and releases the slot
    for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sg, _stop)
    try:
        child = subprocess.Popen(cmd)
        while True:
            try:
                rc = child.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                continue
        if got:
            _log({"event": "signal", "label": label, "slot": slot, "wrapper_pid": os.getpid(), "signals": got,
                  "child_rc": rc})
            rc = 128 + got[0]
    finally:
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        _log({"event": "release", "label": label, "slot": slot, "wrapper_pid": os.getpid(), "rc": rc,
              "wall_s": round(time.time() - t0, 1),
              "cpu_s": round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 1),
              "child_maxrss_bytes": r1.ru_maxrss})
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
    return rc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "status":
        print(json.dumps(status(), indent=1))
        return 0
    if "--" not in argv or "--label" not in argv:
        raise SystemExit("usage: python -m hcal.sema --label <role:what> -- <command ...>")
    cut = argv.index("--")
    label = argv[argv.index("--label") + 1]
    return run(label, argv[cut + 1:])


if __name__ == "__main__":
    sys.exit(main())

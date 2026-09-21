"""Resumable unit ledger, run lock, atomic promotion and the Terminal-2 handoff writer.

Design constraints this file exists to satisfy:

* A reconnect must never start a duplicate worker. Ownership is a lock file holding pid, host and
  boot-unique start time; a stale lock is only broken when the recorded pid is genuinely gone.
* Unit reuse is decided by a **content hash of the configuration**, never by a filename or a
  timestamp, so an edited configuration cannot silently reuse a stale result.
* Artifacts are promoted atomically: write to a temp sibling, fsync, then `os.replace`. A partially
  written unit can therefore never be observed as complete.
* An old status file is *not* live process state. `read_status` returns what was written; liveness is
  a separate question answered by `lock_holder_alive`.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path

STATES = ('planned', 'fitting', 'fitted', 'auditing', 'verified', 'failed', 'duplicate',
          'untriggered')

STUDY = 'pcrl_stochastic_replacement_overnight_v1'


# ------------------------------------------------------------------ atomic io
def write_json_atomic(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1, default=str, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as fh:
        return json.load(fh)


def config_id(config: dict) -> str:
    """Content hash of a unit configuration. Determines reuse; order-insensitive."""
    blob = json.dumps(config, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ run lock
class RunLock:
    """Single-owner advisory lock. Refuses to start a second worker for the same job."""

    def __init__(self, path, job: str):
        self.path = Path(path)
        self.job = job
        self.record = None

    def acquire(self, steal_stale: bool = True) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mine = {'job': self.job, 'pid': os.getpid(), 'host': socket.gethostname(),
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'monotonic': time.monotonic()}
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as fh:
                json.dump(mine, fh, indent=1)
            self.record = mine
            return mine
        except FileExistsError:
            held = read_json(self.path, {})
            if held.get('host') == mine['host'] and not pid_alive(held.get('pid')):
                if not steal_stale:
                    raise RuntimeError(f'stale lock held by dead pid {held.get("pid")}')
                # The recorded pid is gone on this host, so the lock is genuinely stale.
                held['broken_utc'] = mine['started_utc']
                write_json_atomic(self.path.with_name(self.path.name + '.stale'), held)
                write_json_atomic(self.path, mine)
                self.record = mine
                return mine
            raise RuntimeError(
                f'job {self.job} already owned by pid {held.get("pid")} on {held.get("host")} '
                f'since {held.get("started_utc")}; refusing to start a duplicate worker')

    def release(self):
        if self.record and self.path.exists():
            current = read_json(self.path, {})
            if current.get('pid') == self.record['pid']:
                self.path.unlink()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def lock_holder_alive(path) -> bool:
    """Liveness of the recorded holder. A status file alone never answers this."""
    held = read_json(path, {})
    return bool(held) and held.get('host') == socket.gethostname() and pid_alive(held.get('pid'))


# ------------------------------------------------------------------ unit ledger
class Ledger:
    """One JSON file per unit, so concurrent workers never contend on a single index."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, cid: str) -> Path:
        return self.root / f'{cid}.json'

    def get(self, cid: str) -> dict | None:
        return read_json(self.path(cid))

    def put(self, cid: str, config: dict, state: str, **extra) -> dict:
        if state not in STATES:
            raise ValueError(f'unknown state {state!r}; allowed {STATES}')
        rec = self.get(cid) or {'config_id': cid, 'config': config, 'history': []}
        rec['config'] = config
        rec['state'] = state
        rec['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        rec['history'] = (rec.get('history') or []) + [{'state': state, 'utc': rec['updated_utc']}]
        rec.update(extra)
        write_json_atomic(self.path(cid), rec)
        return rec

    def claim(self, cid: str, config: dict) -> bool:
        """True if this worker should compute the unit; False if it is done or owned elsewhere."""
        rec = self.get(cid)
        if rec and rec.get('state') in ('fitted', 'verified', 'duplicate', 'untriggered'):
            return False
        if rec and rec.get('state') == 'fitting' and pid_alive(rec.get('pid')):
            return False
        self.put(cid, config, 'fitting', pid=os.getpid(), host=socket.gethostname())
        return True

    def counts(self) -> dict:
        out = {s: 0 for s in STATES}
        for p in self.root.glob('*.json'):
            rec = read_json(p, {})
            s = rec.get('state')
            if s in out:
                out[s] += 1
        return out

    def all(self) -> list:
        return [read_json(p, {}) for p in sorted(self.root.glob('*.json'))]


# ------------------------------------------------------------------ handoff
def handoff_dir(git_common_dir) -> Path:
    """Shared handoff root, normalised to an absolute path.

    A linked worktree's `.git` may be a FILE, so the caller passes the resolved
    `git rev-parse --git-common-dir` and this normalises it.
    """
    gcd = Path(git_common_dir).resolve()
    return (gcd / 'pcrl_overnight_replacement_v1').resolve()


def write_status(git_common_dir, payload: dict) -> Path:
    """Atomically write OUR status under terminal_1/. Never touches Terminal 2's files."""
    root = handoff_dir(git_common_dir) / 'terminal_1'
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload['written_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    payload['writer'] = {'pid': os.getpid(), 'host': socket.gethostname()}
    payload.setdefault('study', STUDY)
    payload['liveness_note'] = ('this file records what was written at written_utc; it is NOT proof '
                               'that a worker is currently running - check the run lock holder')
    return write_json_atomic(root / 'STATUS.json', payload)


def read_terminal_2(git_common_dir) -> dict:
    """Read-only view of Terminal 2's directory, if it exists. Never written to."""
    root = handoff_dir(git_common_dir) / 'terminal_2'
    if not root.exists():
        return {'present': False, 'files': []}
    files = sorted(p.name for p in root.iterdir() if p.is_file())
    out = {'present': True, 'files': files}
    for name in ('REVIEW.json', 'FINDINGS.json', 'STATUS.json'):
        if (root / name).exists():
            out[name] = read_json(root / name)
    return out

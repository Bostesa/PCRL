"""Shared plumbing for the competitive-method study: lock, threads, status, handoff.

Nothing scientific lives here. The point of this module is that exactly one
orchestrator can run at a time, that every worker is single-threaded, and that status
and handoff records are written **atomically** so a reader never sees a half-written
file.

The previous study measured the machine and declined concurrency. That measurement is
repeated here at import of `resource_snapshot()` and recorded rather than assumed.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'pcrl_competitive_method_v1'
OUT = ROOT / 'results' / STUDY
LOGS = OUT / 'logs'

# The other worktrees are read **read-only**. They are never written, switched,
# reset, stashed, cleaned or rebased.
ADVERSARIAL_ROOT = Path('/Users/nathansamson/PCRL-terminal-1-adversarial')
INVARIANT_ROOT = Path('/Users/nathansamson/PCRL-terminal-1-invariant')
ADVERSARIAL_RESULTS = ADVERSARIAL_ROOT / 'results/pcrl_direct_adversarial_v1'

PREDECESSORS = {
    'direct_adversarial_head': '106de9afa58cebbc26e34fb782e539e2a0881108',
    'direct_adversarial_evidence': '69e790af36c5ca53203dab17b757a8e3415ee934',
    'manuscript_integrated_v3': 'ce5ba24a0a0ed1848c4029a922ac66fa629d71fc',
    'invariant_baselines': '73903b7f28df68284285f0610a4036beb32b208f',
    'nonlinear_rank': 'c37807e4f568ef38e5528fc09c1506083278bf4d',
    'transport_2017_locked': '349efa454afd907389760fd1f59fd8806a215efd',
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ threads
def limit_threads():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(name, '1')
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                            # pragma: no cover
        pass


# ------------------------------------------------------------------ atomic IO
def write_json_atomic(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    with os.fdopen(handle, 'w') as fh:
        json.dump(payload, fh, indent=1, sort_keys=False, default=str)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def write_text_atomic(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    with os.fdopen(handle, 'w') as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def sha_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------ orchestrator lock
class OrchestratorLock:
    """One orchestrator at a time. A stale lock whose PID is gone is reclaimed."""

    def __init__(self, name: str = 'orchestrator'):
        self.path = OUT / f'{name}.lock'

    def acquire(self, stage: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({'pid': os.getpid(), 'stage': stage, 'started': utcnow()})
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w') as fh:
                    fh.write(payload)
                return self
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                held = json.loads(self.path.read_text() or '{}')
                pid = int(held.get('pid', -1))
                if pid > 0 and _pid_alive(pid):
                    raise RuntimeError(
                        f'orchestrator lock held by live pid {pid} for stage '
                        f'{held.get("stage")!r} since {held.get("started")}')
                self.path.unlink(missing_ok=True)
        raise RuntimeError('could not acquire the orchestrator lock')

    def release(self):
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------------------ machine measurement
def resource_snapshot() -> dict:
    """Measure memory NOW. Yesterday's condition is not evidence about today's."""
    snap = {'utc': utcnow(), 'cpu_count': os.cpu_count(),
            'load_average': list(os.getloadavg()),
            'platform': platform.platform(), 'python': platform.python_version()}
    try:
        swap = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        snap['vm_swapusage'] = swap
        found = re.findall(r'(\w+)\s*=\s*([\d.]+)M', swap)
        snap['swap_MB'] = {k: float(v) for k, v in found}
    except Exception as exc:                                     # pragma: no cover
        snap['vm_swapusage_error'] = str(exc)
    try:
        text = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=20).stdout
        page = int(re.search(r'page size of (\d+) bytes', text).group(1))
        counts = {k.strip(): int(v) for k, v in re.findall(r'^(.+?):\s+(\d+)\.', text, re.M)}
        gb = lambda key: round(counts.get(key, 0) * page / 2 ** 30, 3)
        snap['memory_GB'] = {
            'free': gb('Pages free'), 'inactive': gb('Pages inactive'),
            'wired': gb('Pages wired down'), 'active': gb('Pages active'),
            'compressor_occupied': gb('Pages occupied by compressor')}
    except Exception as exc:                                     # pragma: no cover
        snap['vm_stat_error'] = str(exc)
    try:
        total = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True,
                               text=True, timeout=20).stdout.strip()
        snap['physical_memory_GB'] = round(int(total) / 2 ** 30, 2)
    except Exception:                                            # pragma: no cover
        pass
    return snap


# ------------------------------------------------------------------ handoff
def git_common_dir() -> Path:
    out = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--path-format=absolute',
                          '--git-common-dir'], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def handoff_dir() -> Path:
    path = git_common_dir() / 'pcrl_parallel_handoff_v4' / 'terminal_1'
    path.mkdir(parents=True, exist_ok=True)
    return path


def head_sha() -> str:
    out = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def publish_status(milestone: str, *, evidence_commit=None, completed=(), outstanding=(),
                   notes=(), protocol=None, counts=None, extra=None) -> dict:
    """Write the private STATUS.json and append to HISTORY.jsonl, atomically.

    `evidence_commit` is the commit that CONTAINS the evidence for this milestone, and
    is `null` until that commit exists. `internal_head_at_write_time` is recorded
    separately and is never a substitute for it.
    """
    record = {
        'terminal': 1,
        'role': 'experiments',
        'study': STUDY,
        'branch': 'research/pcrl-competitive-method-v1',
        'milestone': milestone,
        'updated_utc': utcnow(),
        'evidence_commit': evidence_commit,
        'internal_head_at_write_time': head_sha(),
        'evidence_commit_note': ('evidence_commit is the COMMIT THAT CONTAINS the published '
                                 'evidence. internal_head_at_write_time is whatever HEAD '
                                 'happened to be when this record was written and is NOT a '
                                 'substitute for it. When evidence_commit is null the evidence '
                                 'for this milestone is not yet committed.'),
        'predecessor_commits': PREDECESSORS,
        'completed_units': list(completed),
        'outstanding_work': list(outstanding),
        'notes': list(notes),
        'counts': counts or {},
        'protocol': protocol or {},
        'boundaries': [
            '2016 remains SEALED and UNUSED. Nothing in this study touches it.',
            'Every number here is DEVELOPMENT on 2018 and 2017 pools that have been used '
            'repeatedly. No computation on them can undo that.',
            'The historical first locked 2017 result keeps its original status and is neither '
            'restated nor overwritten by this study.',
            'Residence and commute are reserved FROM TRAINING and selection, not from the '
            'research process: this project has inspected them repeatedly.',
        ],
        'private_paths': {
            'worktree': str(ROOT),
            'results': str(OUT),
            'adversarial_read_only': str(ADVERSARIAL_ROOT),
            'invariant_read_only': str(INVARIANT_ROOT),
        },
    }
    if extra:
        record.update(extra)
    directory = handoff_dir()
    write_json_atomic(directory / 'STATUS.json', record)
    line = json.dumps({'milestone': milestone, 'updated_utc': record['updated_utc'],
                       'evidence_commit': evidence_commit,
                       'completed_units': list(completed)}, default=str)
    with open(directory / 'HISTORY.jsonl', 'a') as fh:
        fh.write(line + '\n')
        fh.flush()
        os.fsync(fh.fileno())

    # The public copy carries no local absolute paths.
    public = {k: v for k, v in record.items() if k != 'private_paths'}
    public['path_policy'] = ('local absolute paths are deliberately omitted from the published '
                             'handoff; they live in the private status record only')
    write_json_atomic(OUT / 'HANDOFF.json', public)
    return record


class Stopwatch:
    def __init__(self, label: str):
        self.label = label
        self.tick = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.tick


__all__ = ['ROOT', 'STUDY', 'OUT', 'LOGS', 'PREDECESSORS', 'ADVERSARIAL_ROOT',
           'ADVERSARIAL_RESULTS', 'INVARIANT_ROOT', 'utcnow', 'limit_threads',
           'write_json_atomic', 'write_text_atomic', 'read_json', 'sha_file', 'sha_text',
           'OrchestratorLock', 'resource_snapshot', 'handoff_dir', 'publish_status',
           'head_sha', 'git_common_dir', 'Stopwatch']

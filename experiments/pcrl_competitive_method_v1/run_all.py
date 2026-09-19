"""Global orchestrator (RUN_STATUS amendment 1): fits, then ONE priority-ordered audit queue.

The audit queue is global across anchors and ordered by (priority, anchor, track,
unit), so a time cutoff removes a registered reduction slice uniformly from every
anchor rather than from whichever anchor happened to be slowest. Units are claimed
atomically (`mkdir`), so `--workers` > 1 is possible later without duplicate work; the
per-anchor identity table is updated under an exclusive file lock.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from .common import OUT, OrchestratorLock, limit_threads, read_json, utcnow, write_json_atomic
from .run_anchor import CUTOFF

SEEDS = (0, 1, 2)


def global_queue() -> list:
    matrix = read_json(OUT / 'MATRIX.json')
    slots = matrix['track_E']['slots'] + matrix['track_N']['slots']
    order = [(1, seed, '0', unit) for seed in SEEDS for unit in ('ref_A0', 'ref_J')]
    for s in slots:
        for seed in SEEDS:
            order.append((s['priority'], seed, s['track'], s['unit']))
    rank = {'0': 0, 'E': 1, 'N': 2}
    return sorted(order, key=lambda t: (t[0], t[1], rank[t[2]], t[3]))


@contextmanager
def seed_lock(seed: int):
    path = OUT / f'seed_{seed}' / '.identity.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def claim(seed: int, unit: str) -> bool:
    path = OUT / 'claims' / f'{seed}__{unit}'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path)
        return True
    except FileExistsError:
        return False


def audit_worker(worker: int):
    from . import run_dev
    for seed in SEEDS:
        with seed_lock(seed):
            run_dev.write_references(OUT, seed)
    log_path = OUT / 'logs' / f'audit_worker_{worker}.json'
    log = read_json(log_path) if log_path.exists() else []
    for priority, seed, _track, unit in global_queue():
        release = OUT / f'seed_{seed}' / 'releases' / unit / 'releases.npz'
        if not release.exists():
            continue
        if (os.environ.get('PCRL_NO_CUTOFF') != '1' and priority in CUTOFF
                and datetime.now(timezone.utc) > CUTOFF[priority]):
            log.append({'seed': seed, 'unit': unit, 'status': f'PENDING priority {priority} cutoff'})
            write_json_atomic(log_path, log)
            continue
        if not claim(seed, unit):
            continue
        identity_path = OUT / f'seed_{seed}' / 'audit_identity.json'
        with seed_lock(seed):
            key = run_dev.release_identity(release)
            table = read_json(identity_path) if identity_path.exists() else {}
            canonical = next((c for c, v in table.items()
                              if v['identity'] == key and v.get('audited_as') == c), None)
            if canonical is not None and canonical != unit:
                table[unit] = {'identity': key, 'audited_as': canonical, 'duplicate': True}
                write_json_atomic(identity_path, table)
                log.append({'seed': seed, 'unit': unit, 'status': 'duplicate',
                            'duplicate_of': canonical, 'utc': utcnow()})
                write_json_atomic(log_path, log)
                print('DEV_DUPLICATE', seed, unit, '->', canonical, flush=True)
                continue
        result = run_dev.audit_seed_unit(OUT, seed, unit)
        with seed_lock(seed):
            table = read_json(identity_path) if identity_path.exists() else {}
            if not result.get('failed'):
                table[unit] = {'identity': key, 'audited_as': unit, 'duplicate': False}
                write_json_atomic(identity_path, table)
        log.append({'seed': seed, 'unit': unit, 'priority': priority, 'utc': utcnow(),
                    'status': 'failed' if result.get('failed') else 'audited'})
        write_json_atomic(log_path, log)
    print('AUDIT_WORKER_DONE', worker, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', nargs='+', default=['fit_e', 'fit_n', 'audit'])
    parser.add_argument('--worker', type=int, default=0)
    args = parser.parse_args()
    cache = OUT / 'logs' / f'cache_global_{args.worker}'
    cache.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'JOBLIB_TEMP_FOLDER', 'MPLCONFIGDIR'):
        os.environ[key] = str(cache)
    limit_threads()
    from threadpoolctl import threadpool_limits
    lock = 'orchestrator' if args.worker == 0 else f'audit_worker_{args.worker}'
    with OrchestratorLock(lock).acquire(f'global {args.stages}'), threadpool_limits(limits=1):
        if 'fit_e' in args.stages:
            from .run_fit_e import run as fit_e
            fit_e(OUT, SEEDS)
        if 'fit_n' in args.stages:
            from .run_fit_n import run as fit_n
            fit_n(OUT, SEEDS)
        if 'audit' in args.stages:
            audit_worker(args.worker)


if __name__ == '__main__':
    main()

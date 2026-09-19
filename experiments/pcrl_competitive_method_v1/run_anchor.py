"""One anchor's full 2018 pipeline: Track E fit, Track N fit, audits in priority order.

One process per anchor, each holding its own lock, single-threaded, with its own
temporary/cache directories and a disjoint output tree. The audit queue follows the
priority of MATRIX.json; the prospective cutoffs of PROTOCOL §9 stop *launching*
reduction-slice units after their deadlines, uniformly for every matched method.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from .common import OUT, OrchestratorLock, limit_threads, read_json, utcnow, write_json_atomic

CUTOFF = {3: datetime(2026, 9, 19, 6, 45, tzinfo=timezone.utc),
          2: datetime(2026, 9, 19, 7, 0, tzinfo=timezone.utc)}


def queue(seed: int) -> list:
    matrix = read_json(OUT / 'MATRIX.json')
    slots = matrix['track_E']['slots'] + matrix['track_N']['slots']
    order = [('ref_A0', 1), ('ref_J', 1)]
    for priority in (1, 2, 3):
        for track in ('E', 'N'):
            order += [(s['unit'], priority) for s in slots
                      if s['priority'] == priority and s['track'] == track]
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stages', nargs='+', default=['fit_e', 'fit_n', 'audit'])
    args = parser.parse_args()
    seed = args.seed
    cache = OUT / 'logs' / f'cache_seed_{seed}'
    cache.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'JOBLIB_TEMP_FOLDER', 'MPLCONFIGDIR'):
        os.environ[key] = str(cache)
    limit_threads()
    from threadpoolctl import threadpool_limits
    with OrchestratorLock(f'anchor_{seed}').acquire(f'anchor {seed}: {args.stages}'), \
            threadpool_limits(limits=1):
        if 'fit_e' in args.stages:
            from .run_fit_e import run as fit_e
            fit_e(OUT, (seed,))
        if 'fit_n' in args.stages:
            from .run_fit_n import run as fit_n
            fit_n(OUT, (seed,))
        if 'audit' in args.stages:
            from .run_dev import audit_seed, write_references
            write_references(OUT, seed)
            releases = OUT / f'seed_{seed}' / 'releases'
            log = []
            for unit, priority in queue(seed):
                if not (releases / unit / 'releases.npz').exists():
                    log.append({'unit': unit, 'status': 'no release (infeasible or unfitted)'})
                    continue
                if priority in CUTOFF and datetime.now(timezone.utc) > CUTOFF[priority]:
                    log.append({'unit': unit, 'status': f'PENDING: priority {priority} cutoff'})
                    print('AUDIT_DEFERRED', seed, unit, flush=True)
                    continue
                result = audit_seed(OUT, seed, [unit])[unit]
                log.append({'unit': unit, 'priority': priority, 'utc': utcnow(),
                            'status': 'duplicate' if 'duplicate_of' in result else
                            ('failed' if result.get('failed') else 'audited'),
                            **({'duplicate_of': result['duplicate_of']}
                               if 'duplicate_of' in result else {})})
                write_json_atomic(OUT / f'seed_{seed}' / 'audit_queue_log.json', log)
            print('ANCHOR_AUDITS_DONE', seed, flush=True)


if __name__ == '__main__':
    main()

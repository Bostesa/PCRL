"""Locks of the final stage (only for a promoted finalist). No lock reads any label; none overwrites an existing lock.
    python -m amd.locks reporting      # REPORTING_LOCK.json: finalist and analysis locks, bank manifest, reader outputs, code hashes
                                       # + public READERS_SUMMARY.json (winners, coverage, controls, replay; aggregates only)
FINALIST_LOCK.json and ANALYSIS_LOCK.json are written by the coordinator from the promotion record and the statistical
reviewer's frozen design (amd.promote) and committed before any final reader fit.
"""
from __future__ import annotations
import json
import sys

from . import common as C


def reporting():
    lock = C.RESULTS / 'REPORTING_LOCK.json'
    if lock.exists():
        raise PermissionError('REPORTING_LOCK exists')
    for name in ('FINALIST_LOCK.json', 'ANALYSIS_LOCK.json', 'SERVICE_BANK_MANIFEST.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'the reporting lock requires the committed {name}')
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    rec = {'schema': 'AMD1-reporting-lock-v1', 'utc': C.now(), 'finalist_lock_sha256': C.sha(C.RESULTS / 'FINALIST_LOCK.json'),
           'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'service_bank_manifest_sha256': C.sha(C.RESULTS / 'SERVICE_BANK_MANIFEST.json'),
           'bank_files_sha256': man['bank_files_sha256'], 'readers': {},
           'code_sha256': {str(p.relative_to(C.ROOT)): C.sha(p) for p in sorted((C.ROOT / 'amd').glob('*.py'))},
           'note': 'finalist, final bank, readers, endpoint roster, claim conjunctions and analysis code frozen before the one logged coordinator DEV_REPORT evaluation'}
    summary = {'schema': 'AMD1-readers-summary-v1', 'per_seed': {}}
    for s in C.SEEDS:
        d = C.PRIV / 'runs' / f'READERS_{s}'
        rr = json.loads((d / 'READERS_SEED.json').read_text())
        rec['readers'][f'S{s}'] = {'dev_report_scores_sha256': C.sha(d / 'DEV_REPORT_SCORES.npz'), 'readers_seed_sha256': C.sha(d / 'READERS_SEED.json')}
        summary['per_seed'][f'S{s}'] = {'winners': {svc: {v: w['candidate'] for v, w in sv['winner'].items()} for svc, sv in rr['services'].items()},
                                        'attack_validation_auc': {svc: {v: w['attack_validation_auc'] for v, w in sv['winner'].items()} for svc, sv in rr['services'].items()},
                                        'coverage': rr['coverage'], 'controls': rr['controls'], 'fit_counts': rr['fit_counts'],
                                        'replay_vs_source': {k: rr['replay_vs_source'][k] for k in ('compared', 'bitwise', 'max_abs_diff')}}
    lock.write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    (C.RESULTS / 'READERS_SUMMARY.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'code_sha256'}, indent=1))


if __name__ == '__main__':
    {'reporting': reporting}[sys.argv[1]]()

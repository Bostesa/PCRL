"""Intermediate locks: TRAINING_COMPLETE_LOCK (after the fit units) and REPORTING_LOCK (after the final readers).

    python -m scm_bench.locks training     # hashes of each seed's fit outputs + a public FIT_SUMMARY.json (aggregates only)
    python -m scm_bench.locks reporting    # hashes of the frozen bank and reader scores + a public READERS_SUMMARY.json
Neither reads any label. Both refuse to overwrite an existing lock.
"""
from __future__ import annotations
import json
import sys

from . import common as C


def training():
    lock = C.RESULTS / 'TRAINING_COMPLETE_LOCK.json'
    if lock.exists():
        raise PermissionError('TRAINING_COMPLETE_LOCK exists')
    rec = {'schema': 'SCM1-training-complete-lock-v1', 'utc': C.now(), 'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'),
           'arm_outputs_sha256': {}, 'fit_seed_sha256': {}}
    summary = {'schema': 'SCM1-fit-summary-v1', 'per_seed': {}}
    for s in C.SEEDS:
        d = C.PRIV / 'runs' / f'FIT_{s}'
        rec['arm_outputs_sha256'][f'S{s}'] = C.sha(d / 'ARM_OUTPUTS.npz'); rec['fit_seed_sha256'][f'S{s}'] = C.sha(d / 'FIT_SEED.json')
        fs = json.loads((d / 'FIT_SEED.json').read_text())
        summary['per_seed'][f'S{s}'] = {k: fs[k] for k in ('partition', 'arms', 'capacity', 'diagnostics', 'cpu_s_total')}
        summary['per_seed'][f'S{s}']['teacher_tree_counts'] = {k: v['tree_count'] for k, v in fs['teachers'].items()}
        summary['per_seed'][f'S{s}']['propensity_tree_counts'] = {k: v['tree_count'] for k, v in fs['propensity'].items()}
    lock.write_text(json.dumps(rec, indent=1, sort_keys=True))
    (C.RESULTS / 'FIT_SUMMARY.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=float))
    print(json.dumps(rec, indent=1))


def reporting():
    lock = C.RESULTS / 'REPORTING_LOCK.json'
    if lock.exists():
        raise PermissionError('REPORTING_LOCK exists')
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    rec = {'schema': 'SCM1-reporting-lock-v1', 'utc': C.now(), 'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'),
           'selection_lock_sha256': C.sha(C.RESULTS / 'SELECTION_LOCK.json'), 'service_bank_manifest_sha256': C.sha(C.RESULTS / 'SERVICE_BANK_MANIFEST.json'),
           'bank_files_sha256': man['bank_files_sha256'], 'readers': {}, 'note': 'final readers frozen before the one logged DEV_REPORT truth re-read'}
    summary = {'schema': 'SCM1-readers-summary-v1', 'per_seed': {}}
    for s in C.SEEDS:
        d = C.PRIV / 'runs' / f'READERS_{s}'
        rr = json.loads((d / 'READERS_SEED.json').read_text())
        rec['readers'][f'S{s}'] = {'dev_report_scores_sha256': C.sha(d / 'DEV_REPORT_SCORES.npz'), 'readers_seed_sha256': C.sha(d / 'READERS_SEED.json')}
        summary['per_seed'][f'S{s}'] = {'winners': {svc: {v: w['candidate'] for v, w in sv['winner'].items()} for svc, sv in rr['services'].items()},
                                        'winners_slate7': {svc: {v: w['candidate'] for v, w in sv['winner_slate7'].items()} for svc, sv in rr['services'].items()},
                                        'attack_validation_auc': {svc: {v: w['attack_validation_auc'] for v, w in sv['winner'].items()} for svc, sv in rr['services'].items()},
                                        'coverage': rr['coverage'], 'controls': rr['controls'], 'fit_counts': rr['fit_counts'],
                                        'replay_vs_frozen_hybrids': {k: rr['replay_vs_frozen_hybrids'][k] for k in ('compared', 'bitwise', 'max_abs_diff')}}
    lock.write_text(json.dumps(rec, indent=1, sort_keys=True))
    (C.RESULTS / 'READERS_SUMMARY.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=float))
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    {'training': training, 'reporting': reporting}[sys.argv[1]]()

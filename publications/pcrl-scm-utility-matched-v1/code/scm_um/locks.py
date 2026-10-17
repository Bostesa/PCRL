"""Intermediate locks of the SCM utility-matched follow-up.

    python -m scm_um.locks trajectory   # TRAJECTORY_COMPLETE_LOCK.json (hashes of both seeds' trajectory files) + public
                                        # TRAJECTORY_SUMMARY.json (gate, checkpoints, coverage, capacity; aggregates only)
    python -m scm_um.locks reporting    # REPORTING_LOCK.json: selection lock, bank manifest, reader outputs, frozen report code
Neither reads any label. Both refuse to overwrite an existing lock.
"""
from __future__ import annotations
import json
import sys

from . import common as C


def trajectory():
    lock = C.RESULTS / 'TRAJECTORY_COMPLETE_LOCK.json'
    if lock.exists():
        raise PermissionError('TRAJECTORY_COMPLETE_LOCK exists')
    rec = {'schema': 'SCMUM1-trajectory-complete-lock-v1', 'utc': C.now(), 'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'files_sha256': {}}
    summ = {'schema': 'SCMUM1-trajectory-summary-v1', 'per_seed': {}}
    for s in C.SEEDS:
        d = C.PRIV / 'runs' / f'TRAJ_{s}'
        rec['files_sha256'][f'S{s}'] = {p.name: C.sha(p) for p in sorted(d.iterdir()) if p.is_file()}
        tr = json.loads((d / 'TRAJ_SEED.json').read_text())
        if not tr['gate']['ALL_PASS']:
            raise SystemExit('trajectory gate not passed')
        summ['per_seed'][f'S{s}'] = {k: tr[k] for k in ('gate', 'trajectories', 'checkpoints', 'cpu_s', 'cpu_s_total', 'n_effective_releases', 'initial_cells')}
        summ['per_seed'][f'S{s}']['releases'] = {rid: {k: v for k, v in r.items() if k in ('aliases', 'capacity', 'live_cells', 'plugin_costs_vs_C0', 'fitted_occupation_nll_F', 'first_label')}
                                                 for rid, r in tr['releases'].items()}
    lock.write_text(json.dumps(rec, indent=1, sort_keys=True))
    (C.RESULTS / 'TRAJECTORY_SUMMARY.json').write_text(json.dumps(summ, indent=1, sort_keys=True, default=float))
    print(json.dumps(rec, indent=1))


def reporting():
    lock = C.RESULTS / 'REPORTING_LOCK.json'
    if lock.exists():
        raise PermissionError('REPORTING_LOCK exists')
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    rec = {'schema': 'SCMUM1-reporting-lock-v1', 'utc': C.now(), 'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'),
           'trajectory_lock_sha256': C.sha(C.RESULTS / 'TRAJECTORY_COMPLETE_LOCK.json'), 'selection_lock_sha256': C.sha(C.RESULTS / 'SELECTION_LOCK.json'),
           'service_bank_manifest_sha256': C.sha(C.RESULTS / 'SERVICE_BANK_MANIFEST.json'), 'bank_files_sha256': man['bank_files_sha256'],
           'readers': {}, 'code_sha256': {str(p.relative_to(C.ROOT)): C.sha(p) for d in ('scm_um', 'scm_bench') for p in sorted((C.ROOT / d).glob('*.py'))},
           'note': 'selected releases, final bank, readers, comparison roster, matching checks and analysis code frozen before the one logged coordinator DEV_REPORT evaluation'}
    summary = {'schema': 'SCMUM1-readers-summary-v1', 'per_seed': {}}
    for s in C.SEEDS:
        d = C.PRIV / 'runs' / f'READERS_{s}'
        rr = json.loads((d / 'READERS_SEED.json').read_text())
        rec['readers'][f'S{s}'] = {'dev_report_scores_sha256': C.sha(d / 'DEV_REPORT_SCORES.npz'), 'readers_seed_sha256': C.sha(d / 'READERS_SEED.json')}
        summary['per_seed'][f'S{s}'] = {'winners': {svc: {v: w['candidate'] for v, w in sv['winner'].items()} for svc, sv in rr['services'].items()},
                                        'winners_slate7': {svc: {v: w['candidate'] for v, w in sv['winner_slate7'].items()} for svc, sv in rr['services'].items()},
                                        'attack_validation_auc': {svc: {v: w['attack_validation_auc'] for v, w in sv['winner'].items()} for svc, sv in rr['services'].items()},
                                        'coverage': rr['coverage'], 'controls': rr['controls'], 'fit_counts': rr['fit_counts'],
                                        'replay_vs_source': {k: rr['replay_vs_source'][k] for k in ('compared', 'bitwise', 'max_abs_diff')}}
    lock.write_text(json.dumps(rec, indent=1, sort_keys=True))
    (C.RESULTS / 'READERS_SUMMARY.json').write_text(json.dumps(summary, indent=1, sort_keys=True, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != 'code_sha256'}, indent=1))


if __name__ == '__main__':
    {'trajectory': trajectory, 'reporting': reporting}[sys.argv[1]]()

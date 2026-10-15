"""Queued admission of the FROZEN candidate and every frozen reference (no fitting, no INNER labels).

Verifies against the original committed locks (STAGE_B_LOCK, TRAINING_COMPLETE_LOCK, SELECTION_LOCK,
REPORTING_LOCK, DEPLOYMENT_RESULTS): teacher payloads and CatBoost files, L_TREE rho1 E20 and T_REFIT E5/E20
corrections, N_EQ models, deployment packages, frozen reader banks, U0/Ucal references and historical
code nominee releases. Recomputes L_TREE's INNER/ATT/FIT2 releases from its deployment package with the
frozen runtime and compares them bitwise with the stored releases. Writes the frozen-candidate manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(spec_path):
    from .evaluation_queue import queue_config
    from task_headroom.data import load_role, arr_sha
    from task_headroom import deploy, services
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    L = {k: json.loads(verify(v['path'], v['sha256']).read_text()) for k, v in cfg['old_locks'].items()}
    checks, facts = {}, {}
    # Teachers and N_EQ against STAGE_B_LOCK.
    for s in (0, 1):
        for t in ('income', 'occupation'):
            exp = L['STAGE_B_LOCK']['selected_teachers'][f'{s}|{t}']
            it = cfg['teachers'][str(s)][t]
            checks[f'teacher_payload_s{s}_{t}'] = _sha(verify(it['payload']['path'], it['payload']['sha256'])) == exp['teacher_payload_sha256']
            checks[f'teacher_cbm_s{s}_{t}'] = _sha(verify(it['cbm']['path'], it['cbm']['sha256'])) == exp['catboost_model_sha256']
        n = cfg['n_eq'][str(s)]
        checks[f'neq_model_s{s}'] = _sha(verify(n['path'], n['sha256'])) == L['STAGE_B_LOCK']['selected_teachers'][f'{s}|N_EQ']['model_sha256']
    # Corrections against TRAINING_COMPLETE_LOCK.
    for key, it in cfg['thetas'].items():
        checks[f'theta_{key}'] = _sha(verify(it['path'], it['sha256'])) == L['TRAINING_COMPLETE_LOCK']['checkpoints'][key]
    # Reader banks against REPORTING_LOCK.
    for s in (0, 1):
        b = cfg['banks'][str(s)]
        checks[f'bank_receipt_s{s}'] = _sha(verify(b['receipt']['path'], b['receipt']['sha256'])) == L['REPORTING_LOCK']['banks'][str(s)]['receipt_sha256']
        checks[f'bank_scores_s{s}'] = _sha(verify(b['scores']['path'], b['scores']['sha256'])) == L['REPORTING_LOCK']['banks'][str(s)]['scores_sha256']
    # Deployment packages against DEPLOYMENT_RESULTS.
    for name, it in cfg['packages'].items():
        man = verify(it['manifest']['path'], it['manifest']['sha256'])
        checks[f'package_manifest_{name}'] = _sha(man) == L['DEPLOYMENT_RESULTS']['packages'][name]['manifest_sha256']
    # Which services already have frozen readers (T_REFIT E20 must not).
    roster = L['SELECTION_LOCK']['attack_roster']
    facts['attacked_roster_keys'] = roster
    checks['T_REFIT_E20_not_previously_attacked'] = 'T_REFIT|none|E20' not in roster.values()
    checks['L_TREE_frozen_key'] = roster['L_TREE'] == 'L_TREE|1.0|E20'
    # Recompute L_TREE releases from the frozen deployment packages and compare with stored releases.
    R = cfg['roles']
    inner = load_role('INNER', R['INNER'], verify, ('features',))
    att = load_role('ATT', R['ATT'], verify, ('features',))
    fit2 = load_role('FIT2', R['FIT2'], verify, ('features',))
    for s in (0, 1):
        guard()
        pk = cfg['packages'][f'L_TREE_S{s}']
        pdir = Path(pk['manifest']['path']).parent
        for f, digest in json.loads(Path(pk['manifest']['path']).read_text())['files'].items():
            verify(str(pdir / f), cfg['package_files'][f'L_TREE_S{s}'][f])
        m, teachers, theta = deploy.load_package(pdir, pk['manifest']['sha256'])
        out = deploy.predict(m, teachers, theta, inner['X'], inner['feature_names'])
        st = np.load(verify(cfg['stored_releases'][str(s)]['L_TREE']['path'], cfg['stored_releases'][str(s)]['L_TREE']['sha256']))
        checks[f'L_TREE_s{s}_INNER_package_bitwise_stored'] = bool(np.array_equal(out['p_income'], st['p1']) and np.array_equal(out['p_occupation'], st['p2'])
                                                                    and np.array_equal(out['decision_income'], st['d1']) and np.array_equal(out['decision_occupation'], st['d2']))
        svc = cfg['services'][str(s)]['L_TREE']
        for role, d in (('ATT', att), ('FIT2', fit2), ('INNER', inner)):
            w = services.outputs(svc, d, verify, s)
            o = deploy.predict(m, teachers, theta, d['X'], d['feature_names'])
            checks[f'L_TREE_s{s}_{role}_service_equals_package'] = bool(np.array_equal(w[0]['probability'], o['p_income']) and np.array_equal(w[1]['probability'], o['p_occupation']))
            facts[f'L_TREE_s{s}_{role}_release_sha256'] = {'income': arr_sha(o['p_income']), 'occupation': arr_sha(o['p_occupation']),
                                                          'decisions': arr_sha(np.stack((o['decision_income'], o['decision_occupation'])))}
    # References and historical releases: hash only.
    for key, it in cfg['references'].items():
        verify(it['path'], it['sha256']); facts[f'reference_{key}_sha256'] = it['sha256']
    for key, it in cfg['historical_files'].items():
        verify(it['path'], it['sha256']); facts[f'historical_{key}_sha256'] = it['sha256']
    for key, it in cfg['runtime_code'].items():
        checks[f'runtime_code_{key}_matches_publication'] = _sha(verify(it['worktree']['path'], it['worktree']['sha256'])) == _sha(verify(it['published']['path'], it['published']['sha256']))
    res = {'schema': 'TC1-frozen-admission-v1', 'queue_unit_id': active['unit_id'], 'status': 'PASS' if all(checks.values()) else 'FAIL',
           'checks': checks, 'facts': facts, 'INNER_labels_opened': False, 'fitting_performed': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(res, f, indent=2, sort_keys=True)
    if res['status'] != 'PASS':
        raise AssertionError('frozen admission failed: ' + ','.join(k for k, v in checks.items() if not v))
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)

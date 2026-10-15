"""Queued source admission, role audit and the newly authorized FIT2 SEX export.

Checks: role hashes/schema/counts, cross-role row and group disjointness,
duplicate-group feature identity, UPDATE/GUARD/FIT2 re-derivation from the
OSF_DEFENSE_FIT manifest (FIT2 = inherited fold 2), RAW_PROJ live forward vs
the stored two-recipient outputs (narrowed to probabilities+decision), and the
fidelity of every historical comparator release under the narrow contract.
Writes FIT2 SEX only (for the predeclared internal recovery proxy). No task
model is fitted and no performance is computed. INNER labels are not opened.
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
    from .data import load_role, load_npz, arr_sha
    from .admission import load_manifest, load_role as manifest_role
    from . import services
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    R = cfg['roles']; checks = {}; facts = {}
    roles = {'UPDATE': load_role('UPDATE', R['UPDATE'], verify, ('features', 'income', 'occupation', 'sex')),
             'GUARD': load_role('GUARD', R['GUARD'], verify, ('features', 'income', 'occupation')),
             'FIT2': load_role('FIT2', R['FIT2'], verify, ('features', 'income', 'occupation', 'references_s0', 'references_s1')),
             'ATT': load_role('ATT', R['ATT'], verify, ('features', 'sex')),
             'INNER': load_role('INNER', R['INNER'], verify, ('features', 'references_s0', 'references_s1'))}
    names = tuple(map(str, roles['UPDATE']['feature_names']))
    checks['feature_schema_identical_all_roles'] = all(tuple(map(str, r['feature_names'])) == names for r in roles.values())
    keys = list(roles)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            checks[f'rows_disjoint_{a}_{b}'] = not np.intersect1d(roles[a]['row_id'], roles[b]['row_id']).size
            checks[f'groups_disjoint_{a}_{b}'] = not np.intersect1d(roles[a]['unit'], roles[b]['unit']).size
    for k, r in roles.items():
        ok = True
        for g in np.unique(r['unit']):
            ix = np.flatnonzero(r['unit'] == g)
            if len(ix) > 1 and not (r['X'][ix] == r['X'][ix[0]]).all():
                ok = False
        checks[f'duplicate_groups_identical_features_{k}'] = ok
        facts[k] = {'rows': int(len(r['row_id'])), 'groups': int(len(np.unique(r['unit']))),
                    'row_id_sha256': arr_sha(r['row_id']), 'X_sha256': arr_sha(r['X'])}
        for lab in ('income', 'occupation', 'sex'):
            if lab in r:
                facts[k][f'{lab}_counts'] = np.bincount(r[lab], minlength=2 if lab != 'occupation' else 6).tolist()
    guard()
    # Re-derive the fitting/selection roles from the inherited FIT manifest.
    m = load_manifest(verify(cfg['manifest']['path'], cfg['manifest']['sha256']), cfg['manifest']['sha256'])
    fit = manifest_role(m, 0, 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation', 'sex'})
    private = json.loads(verify(cfg['private_split']['path'], cfg['private_split']['sha256']).read_text())
    pos = {int(r): j for j, r in enumerate(fit['row_id'])}
    for role in ('UPDATE', 'GUARD', 'FIT2'):
        r = roles[role]; ix = np.asarray([pos[int(x)] for x in r['row_id']])
        same = (np.array_equal(fit['unit'][ix], r['unit']) and np.array_equal(fit['X'][ix], r['X'])
                and np.array_equal(fit['income'][ix], r['income']) and np.array_equal(fit['occupation'][ix], r['occupation']))
        checks[f'{role}_identical_to_FIT_manifest'] = bool(same)
        folds = np.unique(fit['fold'][ix]).tolist(); facts[role]['inherited_folds'] = folds
        if role in ('UPDATE', 'GUARD'):
            checks[f'{role}_matches_private_split_indices'] = bool(np.array_equal(np.sort(ix), np.sort(np.asarray(private['fit_row_indices'][role]))))
    checks['FIT2_is_exactly_fold_2'] = bool(set(roles['FIT2']['row_id'].tolist()) == set(fit['row_id'][fit['fold'] == 2].tolist()))
    checks['UPDATE_GUARD_FIT2_partition_FIT'] = bool(sum(len(roles[r]['row_id']) for r in ('UPDATE', 'GUARD', 'FIT2')) == len(fit['row_id']))
    uix = np.asarray([pos[int(x)] for x in roles['UPDATE']['row_id']])
    checks['UPDATE_sex_identical_to_manifest'] = bool(np.array_equal(fit['sex'][uix], roles['UPDATE']['sex']))
    att_m = manifest_role(m, 0, 'ATTACK_FIT_NEW', {'features', 'sex'})
    checks['ATT_identical_to_manifest'] = bool(np.array_equal(att_m['row_id'], roles['ATT']['row_id']) and np.array_equal(att_m['X'], roles['ATT']['X'])
                                               and np.array_equal(att_m['sex'], roles['ATT']['sex']))
    for s in (0, 1):
        ref = fit['row_id']
        mref = manifest_role(m, s, 'OSF_DEFENSE_FIT', {'references'})['references']
        fix = np.asarray([pos[int(x)] for x in roles['FIT2']['row_id']])
        checks[f'FIT2_U0_Ucal_identical_to_manifest_s{s}'] = all(np.array_equal(mref[k][fix], roles['FIT2'][f'references_s{s}'][k]) for k in ('U0_1', 'U0_2', 'Ucal_1', 'Ucal_2'))
    # New authorized export: FIT2 SEX (internal recovery proxy only).
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private export directory required')
    (out / 'FIT2').mkdir(parents=True, mode=0o700)
    fix = np.asarray([pos[int(x)] for x in roles['FIT2']['row_id']])
    sp = out / 'FIT2' / 'sex.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=roles['FIT2']['row_id'], unit=roles['FIT2']['unit'], sex=fit['sex'][fix].astype(np.int64))
    sp.chmod(0o400)
    facts['FIT2']['sex_counts'] = np.bincount(fit['sex'][fix], minlength=2).tolist()
    guard()
    # RAW_PROJ live forward vs stored two-recipient deployment outputs.
    raw = {}
    for s in (0, 1):
        svc = cfg['raw_proj'][str(s)]
        for role in ('FIT2', 'ATT', 'INNER'):
            w = services.outputs(svc, roles[role], verify, s)
            stored = np.load(verify(cfg['raw_proj_stored'][str(s)][role]['path'], cfg['raw_proj_stored'][str(s)][role]['sha256']), allow_pickle=False)
            r1, r2 = stored['r1'], stored['r2']
            same = (np.array_equal(w[0]['probability'], r1[:, 18:20]) and np.array_equal(w[0]['decision'], r1[:, 20].astype(np.int64))
                    and np.array_equal(w[1]['probability'], r2[:, 22:28]) and np.array_equal(w[1]['decision'], r2[:, 28].astype(np.int64)))
            checks[f'RAW_PROJ_s{s}_{role}_bitwise_equal_stored_narrowed'] = bool(same)
            raw[f's{s}_{role}'] = {'income_sha256': arr_sha(w[0]['probability']), 'occupation_sha256': arr_sha(w[1]['probability'])}
    # Historical releases: coverage and narrow-contract fidelity.
    hist = {}
    for label, per_seed in cfg['historical'].items():
        for s in (0, 1):
            for role in ('FIT2', 'ATT', 'INNER'):
                w = services.outputs(per_seed[str(s)], roles[role], verify, s)
                hist[f'{label}|s{s}|{role}'] = {'income_sha256': arr_sha(w[0]['probability']), 'occupation_sha256': arr_sha(w[1]['probability']),
                                               'income_distinct_vectors': int(len(np.unique(w[0]['probability'], axis=0))),
                                               'occupation_distinct_vectors': int(len(np.unique(w[1]['probability'], axis=0)))}
            checks[f'HIST_{label}_s{s}_faithful'] = True
    result = {'schema': 'THP1-source-admission-v1', 'queue_unit_id': active['unit_id'], 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'facts': facts, 'raw_proj': raw, 'historical': hist,
              'fit2_sex_export': {'sha256': _sha(sp), 'fields': ['row_id', 'unit', 'sex'], 'purpose': 'internal recovery proxy only'},
              'publication_pin': cfg['publication_pin'], 'INNER_labels_opened': False, 'performance_measured': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    if result['status'] != 'PASS':
        raise AssertionError('admission failed: ' + ','.join(k for k, v in checks.items() if not v))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)

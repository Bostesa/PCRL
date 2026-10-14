"""Queued role preparation: frozen label-independent UPDATE/GUARD split and
separate physical exports. Logs the actual inherited label materialization.

The split depends only on SPLIT_SALT and the canonical inherited group id.
No performance of any role is measured here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

from .recipe import SPLIT_SALT, PROBE_SALT, CAMPAIGN

EXPORTS = {'UPDATE': ('features', 'occupation', 'sex'),
           'GUARD': ('features', 'occupation'),
           'VALIDATION': ('features', 'income', 'occupation', 'references_s0', 'references_s1')}


def split_key(unit):
    return hashlib.sha256((SPLIT_SALT + str(unit)).encode()).hexdigest()


def probe_key(row_id):
    return hashlib.sha256((PROBE_SALT + str(row_id)).encode()).hexdigest()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _save(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    path.chmod(0o400)
    return {'path': str(path), 'sha256': _sha(path), 'fields': sorted(arrays)}


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    from .admission import load_manifest, load_role
    manifest_path = verify(spec['manifest']['path'], spec['manifest']['sha256'])
    manifest = load_manifest(manifest_path, spec['manifest']['sha256'])
    guard()
    # Actual provenance: the export worker materializes inherited FIT task and
    # SEX label arrays in order to subset them. No metric is computed here.
    fit = load_role(manifest, 0, 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation', 'sex'})
    refs = {s: load_role(manifest, s, 'OSF_DEFENSE_FIT', {'features', 'references'})['references'] for s in (0, 1)}
    for s in (0, 1):
        if not (np.array_equal(refs[s]['row_id'], fit['row_id']) and np.array_equal(refs[s]['unit'], fit['unit'])):
            raise PermissionError('reference pairing differs')
    fold = fit['fold']; unit = fit['unit']
    # Duplicate groups cannot cross inherited folds: the fold is a hash of the unit.
    crossing = [str(g) for g in np.unique(unit) if len(np.unique(fold[unit == g])) > 1]
    pool = np.flatnonzero(np.isin(fold, [0, 1])); valid = np.flatnonzero(fold == 2)
    groups = sorted({str(g) for g in unit[pool]}, key=lambda g: (split_key(g), g))
    n_guard = int(0.2 * len(groups))
    guard_groups = set(groups[:n_guard])
    unit_str = np.array([str(g) for g in unit])
    in_guard = np.isin(unit_str, list(guard_groups))
    upd = pool[~in_guard[pool]]; grd = pool[in_guard[pool]]
    sets = {'UPDATE': upd, 'GUARD': grd, 'VALIDATION': valid}
    g = {k: set(unit_str[v]) for k, v in sets.items()}
    overlap = {f'{a}__{b}': len(g[a] & g[b]) for a in g for b in g if a < b}
    if any(overlap.values()) or crossing:
        raise PermissionError('role groups overlap; resolve exposure before any fit')
    if len(upd) + len(grd) != len(pool) or len(pool) + len(valid) != len(fit['row_id']):
        raise PermissionError('role rows do not partition FIT')
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private role directory required')
    out.mkdir(parents=True, mode=0o700)
    guard()
    files = {}
    base = {k: fit[k] for k in ('row_id', 'unit')}
    for role, idx in sets.items():
        files[role] = {}
        files[role]['features'] = _save(out / role / 'features.npz', row_id=fit['row_id'][idx], unit=fit['unit'][idx],
                                        X=fit['X'][idx], feature_names=fit['feature_names'])
        for name in EXPORTS[role][1:]:
            if name.startswith('references'):
                r = refs[int(name[-1])]
                files[role][name] = _save(out / role / f'{name}.npz', **{k: r[k][idx] for k in r})
            else:
                files[role][name] = _save(out / role / f'{name}.npz', row_id=fit['row_id'][idx],
                                          unit=fit['unit'][idx], **{name: fit[name][idx]})
    probe_order = sorted(range(len(upd)), key=lambda i: (probe_key(fit['row_id'][upd[i]]), i))[:512]
    support = {}
    for role, idx in sets.items():
        support[role] = {'rows': int(len(idx)), 'groups': int(len(np.unique(unit[idx]))),
                         'occupation_class_counts': np.bincount(fit['occupation'][idx], minlength=6).tolist(),
                         'income_class_counts': np.bincount(fit['income'][idx], minlength=2).tolist() if role == 'VALIDATION' else 'not_exported',
                         'sex_class_counts': np.bincount(fit['sex'][idx], minlength=2).tolist() if role == 'UPDATE' else 'not_exported'}
    private = {'schema': 'SPG1-role-split-private-v1', 'campaign': CAMPAIGN,
               'fit_row_indices': {k: v.tolist() for k, v in sets.items()},
               'guard_groups': sorted(guard_groups), 'update_probe_positions': probe_order}
    with (out / 'ROLE_SPLIT_PRIVATE.json').open('x') as stream:
        json.dump(private, stream, sort_keys=True)
    os.chmod(out / 'ROLE_SPLIT_PRIVATE.json', 0o400)
    public = {'schema': 'SPG1-role-split-v1', 'campaign': CAMPAIGN, 'queue_unit_id': active['unit_id'],
              'split_salt': SPLIT_SALT, 'probe_salt': PROBE_SALT,
              'rule': 'GUARD = first floor(0.2*eligible FIT fold0/1 groups) in SHA256(salt+str(unit)) order; UPDATE = remainder; VALIDATION = FIT fold 2',
              'eligible_groups': len(groups), 'guard_groups': n_guard,
              'support': support, 'group_overlap': overlap, 'duplicate_groups_crossing_inherited_folds': len(crossing),
              'inherited_fit_rows': int(len(fit['row_id'])), 'inherited_fit_groups': int(len(np.unique(unit))),
              'inherited_fold_rows': {str(k): int((fold == k).sum()) for k in range(3)},
              'role_index_sha256': {k: hashlib.sha256(json.dumps(v.tolist()).encode()).hexdigest() for k, v in sets.items()},
              'role_group_sha256': {k: hashlib.sha256(json.dumps(sorted(g[k])).encode()).hexdigest() for k in g},
              'probe_rows': len(probe_order),
              'probe_position_sha256': hashlib.sha256(json.dumps(probe_order).encode()).hexdigest(),
              'exports': {role: {k: {'sha256': v['sha256'], 'fields': v['fields'], 'file': f'{role}/{k}.npz'}
                                 for k, v in d.items()} for role, d in files.items()},
              'private_split_sha256': _sha(out / 'ROLE_SPLIT_PRIVATE.json'),
              'label_provenance': {'worker_materialized': ['FIT income', 'FIT occupation', 'FIT SEX', 'FIT U0/Ucal references s0/s1'],
                                   'purpose': 'subset copy only; no metric, loss or performance computed',
                                   'GUARD_SEX_exported': False, 'UPDATE_income_exported': False, 'GUARD_income_exported': False},
              'INNER_opened': False, 'ATT_opened': False, 'performance_measured': False}
    for path in (out / 'ROLE_SPLIT.json', Path(spec['output'])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(public, stream, indent=2, sort_keys=True); stream.write('\n')
    paths = {role: {k: v['path'] for k, v in d.items()} for role, d in files.items()}
    with (out / 'ROLE_PATHS.json').open('x') as stream:
        json.dump({'paths': paths, 'sha256': {r: {k: v['sha256'] for k, v in d.items()} for r, d in files.items()}},
                  stream, indent=2, sort_keys=True)
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)

"""Queued, newly authorized income-label export for the inherited UPDATE and GUARD roles.

Re-derives the inherited role membership from the private split indices,
verifies that the re-derived features, units and occupation labels are
byte-identical to the inherited exports, and writes income-only exports for
UPDATE and GUARD. No metric, loss or performance is computed. FIT2, ATT and
INNER are not sources. SEX is not exported for GUARD.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _load(item, verify, fields):
    import numpy as np
    from zipfile import ZipFile
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as archive:
        if {x.filename for x in archive.infolist()} != {f + '.npy' for f in fields}:
            raise PermissionError('unexpected inherited export fields')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in fields}


def run(spec_path):
    from .evaluation_queue import queue_config
    active, spec, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    from .admission import load_manifest, load_role
    manifest = load_manifest(verify(spec['manifest']['path'], spec['manifest']['sha256']), spec['manifest']['sha256'])
    private = json.loads(verify(spec['private_split']['path'], spec['private_split']['sha256']).read_text())
    public = json.loads(verify(spec['role_split']['path'], spec['role_split']['sha256']).read_text())
    if hashlib.sha256(Path(spec['private_split']['path']).read_bytes()).hexdigest() != public['private_split_sha256']:
        raise PermissionError('private split differs from public split receipt')
    guard()
    # Actual provenance: the worker materializes FIT income and occupation to subset them.
    fit = load_role(manifest, 0, 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation'})
    out = Path(spec['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private export directory required')
    out.mkdir(parents=True, mode=0o700)
    files = {}; checks = {}; support = {}
    for role, n in (('UPDATE', 8180), ('GUARD', 2043)):
        idx = np.asarray(private['fit_row_indices'][role], dtype=np.int64)
        if hashlib.sha256(json.dumps(idx.tolist()).encode()).hexdigest() != public['role_index_sha256'][role] or len(idx) != n:
            raise PermissionError('role index differs from inherited split')
        feat = _load(spec['inherited'][role]['features'], verify, ('row_id', 'unit', 'X', 'feature_names'))
        occ = _load(spec['inherited'][role]['occupation'], verify, ('row_id', 'unit', 'occupation'))
        checks[f'{role}_features_identical'] = bool(np.array_equal(feat['row_id'], fit['row_id'][idx]) and np.array_equal(feat['unit'], fit['unit'][idx])
                                                    and np.array_equal(feat['X'], fit['X'][idx]) and np.array_equal(feat['feature_names'], fit['feature_names']))
        checks[f'{role}_occupation_identical'] = bool(np.array_equal(occ['occupation'], fit['occupation'][idx]) and np.array_equal(occ['row_id'], feat['row_id']))
        checks[f'{role}_fold_in_0_1'] = bool(np.isin(fit['fold'][idx], [0, 1]).all())
        if not all(v for k, v in checks.items() if k.startswith(role)):
            raise PermissionError(f'{role} re-derivation differs from inherited export')
        path = out / role / 'income.npz'
        path.parent.mkdir(parents=True, mode=0o700)
        with path.open('xb') as stream:
            np.savez(stream, row_id=fit['row_id'][idx], unit=fit['unit'][idx], income=fit['income'][idx])
        path.chmod(0o400)
        files[role] = {'path': str(path), 'sha256': _sha(path)}
        support[role] = {'rows': n, 'income_class_counts': np.bincount(fit['income'][idx], minlength=2).tolist()}
    paths = {'paths': {r: {'income': v['path']} for r, v in files.items()}, 'sha256': {r: {'income': v['sha256']} for r, v in files.items()}}
    with (out / 'INCOME_EXPORT_PATHS.json').open('x') as stream:
        json.dump(paths, stream, indent=2, sort_keys=True)
    result = {'schema': 'TRA1-income-export-v1', 'queue_unit_id': active['unit_id'], 'status': 'PASS' if all(checks.values()) else 'FAIL',
              'checks': checks, 'support': support, 'exports': {r: {'sha256': v['sha256'], 'file': f'{r}/income.npz', 'fields': ['income', 'row_id', 'unit']} for r, v in files.items()},
              'label_provenance': {'worker_materialized': ['FIT income', 'FIT occupation'], 'purpose': 'subset copy only; no metric, loss or performance computed',
                                   'UPDATE_income_exported': True, 'GUARD_income_exported': True, 'GUARD_SEX_exported': False},
              'FIT2_opened': False, 'ATT_opened': False, 'INNER_opened': False, 'performance_measured': False}
    with Path(spec['output']).open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    if result['status'] != 'PASS':
        raise AssertionError('income export failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)

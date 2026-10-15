"""Queue-only versioned same-host snapshot and backup-alone representative restore.

snapshot: hash-verified copy of the campaign's private artifacts plus pinned
inherited inputs (start checkpoints, geometry, prior checkpoints, role exports).
restore: rebuilds only from the snapshot, re-verifies every hash, re-executes
deployment packages against stored releases, reloads representative readers
and re-predicts their frozen INNER scores, and replays one adapter checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot(cfg, verify):
    target = Path(cfg['snapshot'])
    if '.private' not in target.resolve().parts or target.exists():
        raise PermissionError('fresh private snapshot directory required')
    target.mkdir(parents=True, mode=0o700)
    files = {}
    for rel_root, src in cfg['trees'].items():
        src = Path(src)
        for p in sorted(src.rglob('*')):
            if p.is_symlink():
                raise PermissionError('symlink refused')
            if p.is_file():
                rel = f'{rel_root}/{p.relative_to(src).as_posix()}'
                dst = target / rel; dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copy2(p, dst)
                h = _sha(dst)
                if h != _sha(p):
                    raise AssertionError('snapshot copy differs')
                files[rel] = {'sha256': h, 'bytes': dst.stat().st_size}
    for rel, item in cfg['pinned_files'].items():
        p = verify(item['path'], item['sha256'])
        dst = target / rel; dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(p, dst)
        if _sha(dst) != item['sha256']:
            raise AssertionError('pinned snapshot copy differs')
        files[rel] = {'sha256': item['sha256'], 'bytes': dst.stat().st_size}
    manifest = {'schema': 'TRA1-custody-snapshot-v1', 'version': cfg['version'], 'files': files, 'file_count': len(files),
                'total_bytes': sum(v['bytes'] for v in files.values()), 'same_device_copy': True, 'off_device': False}
    (target / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {'snapshot_manifest_sha256': _sha(target / 'SNAPSHOT_MANIFEST.json'), 'file_count': len(files), 'total_bytes': manifest['total_bytes']}


def restore(cfg, verify):
    import numpy as np
    import torch
    import joblib
    torch.set_num_threads(1)
    from .deployment import load_package
    man = verify(cfg['snapshot_manifest']['path'], cfg['snapshot_manifest']['sha256'])
    manifest = json.loads(man.read_text()); snap = man.parent
    target = Path(cfg['restore'])
    if '.private' not in target.resolve().parts or target.exists():
        raise PermissionError('fresh private restore directory required')
    shutil.copytree(snap, target)
    bad = [rel for rel, v in manifest['files'].items() if _sha(target / rel) != v['sha256']]
    if bad:
        raise AssertionError(f'restored hash mismatch: {bad[:5]}')
    checks = {'all_file_hashes_match': True, 'files': len(manifest['files']), 'packages': {}, 'readers': {}, 'adapter_replay': {}}
    inner = np.load(target / 'inputs/INNER_features.npz', allow_pickle=False)
    X, names = inner['X'], inner['feature_names'].tolist()
    for label, msha in cfg['packages'].items():
        rt, meta = load_package(target / 'deployment' / label, msha)
        out = rt.release(X, names)
        with np.load(target / 'deployment' / label / 'INNER_expected_wire.npz', allow_pickle=False) as e:
            d = [float(np.max(np.abs(out[i]['full'] - e[k]))) for i, k in enumerate(('r1', 'r2'))]
        if any(d):
            raise AssertionError('restored package output differs from stored release')
        checks['packages'][label] = {'restored_vs_stored_max_abs': d, 'rows': int(len(X)), 'scientific_status': meta['scientific_status']}
    from task_conditioned_representation.attacks import predict_reader, orient
    from compressed_income_safeguard.readers import tables, kind_for
    for unit_dir, view, reader in cfg['readers']:
        d = target / 'readers' / unit_dir
        rec = json.loads((d / 'FROZEN_COMPLETE.json').read_text())
        model = joblib.load(d / f'MODEL_{view}__{reader}.joblib')
        with np.load(d / 'INNER_RELEASE.npz', allow_pickle=False) as z:
            tab = tables(z['r1'], z['r2'])
        with np.load(d / 'INNER_SCORES.npz', allow_pickle=False) as z:
            frozen = z[f'{view}__{reader}'].copy()
        again = orient(predict_reader(model, tab[view][kind_for(reader)]), rec['ATT_validation'][view][reader]['orientation'])
        checks['readers'][f'{unit_dir}|{view}|{reader}'] = {'max_abs_vs_frozen_scores': float(np.max(np.abs(again - frozen)))}
        if not np.array_equal(again, frozen):
            raise AssertionError('restored reader does not reproduce frozen scores')
    for label, (ckpt_rel, pkg) in cfg['adapter_replay'].items():
        payload = torch.load(target / ckpt_rel, map_location='cpu', weights_only=False)
        ad = torch.load(target / 'deployment' / pkg / 'adapter.pt', map_location='cpu', weights_only=True)
        same = all(torch.equal(payload['states'][i]['R'], ad[f'R_{n}']) and torch.equal(payload['states'][i]['b'], ad[f'b_{n}'])
                   for i, n in enumerate(('income', 'occupation')))
        checks['adapter_replay'][label] = {'checkpoint_residuals_equal_package': bool(same)}
        if not same:
            raise AssertionError('restored adapter checkpoint differs from package')
    shutil.rmtree(target)
    checks['restore_directory_removed_after_check'] = True
    return checks


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'custody')
    result = snapshot(cfg, verify) if cfg['operation'] == 'snapshot' else restore(cfg, verify)
    result.update(queue_unit_id=active['unit_id'], operation=cfg['operation'])
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)

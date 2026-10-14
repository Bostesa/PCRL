"""Queue-only versioned snapshot and backup-alone representative restore.

snapshot: hash-verified copy of the campaign's private artifacts plus the
inherited start checkpoints and income geometry. restore: rebuilds from the
snapshot alone, re-verifies every hash and re-executes restored packages.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot(cfg, verify, guard):
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
                if _sha(dst) != _sha(p):
                    raise AssertionError('snapshot copy differs')
                files[rel] = {'sha256': _sha(dst), 'bytes': dst.stat().st_size}
    for rel, item in cfg['pinned_files'].items():
        p = verify(item['path'], item['sha256'])
        dst = target / rel; dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(p, dst)
        if _sha(dst) != item['sha256']:
            raise AssertionError('pinned snapshot copy differs')
        files[rel] = {'sha256': item['sha256'], 'bytes': dst.stat().st_size}
    manifest = {'schema': 'SPG1-custody-snapshot-v1', 'files': files, 'file_count': len(files),
                'total_bytes': sum(v['bytes'] for v in files.values()), 'same_device_copy': True, 'off_device': False}
    (target / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {'snapshot_manifest_sha256': _sha(target / 'SNAPSHOT_MANIFEST.json'), 'file_count': len(files),
            'total_bytes': manifest['total_bytes']}


def restore(cfg, verify, guard):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .deployment import load_package
    snap = Path(cfg['snapshot']); man = verify(cfg['snapshot_manifest']['path'], cfg['snapshot_manifest']['sha256'])
    manifest = json.loads(man.read_text())
    target = Path(cfg['restore'])
    if '.private' not in target.resolve().parts or target.exists():
        raise PermissionError('fresh private restore directory required')
    shutil.copytree(snap, target)
    bad = [rel for rel, v in manifest['files'].items() if _sha(target / rel) != v['sha256']]
    if bad:
        raise AssertionError(f'restored hash mismatch: {bad[:5]}')
    checks = {'all_file_hashes_match': True, 'files': len(manifest['files']), 'packages': {}}
    for label, mfile_sha in cfg['packages'].items():
        root = target / 'deployment' / label
        runtime, meta = load_package(root, mfile_sha)
        with np.load(root / 'VALIDATION_input.npz', allow_pickle=False) as z:
            out = runtime.release(z['X'], z['feature_names'].tolist())
        with np.load(root / 'VALIDATION_expected_wire.npz', allow_pickle=False) as e:
            d = [float(np.max(np.abs(out[i]['full'] - e[f'r{i+1}']))) for i in range(2)]
        if any(d):
            raise AssertionError('restored package output differs from stored wire')
        checks['packages'][label] = {'restored_vs_stored_wire_max_abs': d, 'rows': int(len(out[0]['full']))}
    split = json.loads((target / 'roles/ROLE_SPLIT.json').read_text())
    checks['role_map_split_sha256_matches_public'] = _sha(target / 'roles/ROLE_SPLIT.json') == cfg['public_role_split_sha256']
    checks['role_exports_match_split'] = all(_sha(target / 'roles' / v['file']) == v['sha256']
                                             for d in split['exports'].values() for v in d.values())
    for s in ('0', '1'):
        g = json.loads((target / f'inherited/geometry_S{s}.json').read_text())
        checks[f'income_basis_S{s}_binding_matches'] = g['binding_sha256'] == cfg['geometry_binding_sha256'][s]
        ck = torch.load(target / f'training/S{s}/SEPARATE_GUARD/epoch_5.pt', map_location='cpu', weights_only=False)
        src = torch.load(target / f'inherited/start_S{s}.pt', map_location='cpu', weights_only=False)
        checks[f'restored_candidate_S{s}_income_frozen'] = all(torch.equal(ck['model'][k], src['model'][k]) for k in ck['model'] if k.startswith(('enc.0.', 'head.0.')))
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise AssertionError(f'restore check failed: {checks}')
    shutil.rmtree(target)
    checks['restore_directory_removed_after_check'] = True
    return checks


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'custody')
    result = snapshot(cfg, verify, guard) if cfg['operation'] == 'snapshot' else restore(cfg, verify, guard)
    result.update(queue_unit_id=active['unit_id'], operation=cfg['operation'])
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)

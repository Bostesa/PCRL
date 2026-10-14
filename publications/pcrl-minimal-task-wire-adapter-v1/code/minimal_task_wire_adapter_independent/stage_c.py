"""Independent Stage C: restore from the custody snapshot alone, then raw-input deployment replay. Queue-only."""
import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import core as C
from .stage_b import wires_for

PACKAGES = ('J_TRUST_E10', 'TASK_TRUST_E10', 'RAW_PROJ', 'L_TRUST_E10', 'PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main(spec_path):
    from minimal_task_wire_adapter.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    torch.set_num_threads(1)
    t0 = time.time()
    res = {'unit_id': active['unit_id']}
    out = Path(cfg['output'])

    def dump():
        res['elapsed_s'] = time.time() - t0
        out.write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    restore = Path(cfg['restore_dir'])
    try:
        mpath = verify(cfg['manifest']['path'], cfg['manifest']['sha256'])
        manifest = json.loads(mpath.read_text())
        snap = mpath.parent
        if restore.exists():
            raise FileExistsError('fresh restore directory required')
        restore.mkdir(parents=True, mode=0o700)
        bad_src, bad_dst, nbytes = [], [], 0
        entries = {rel: (v['sha256'] if isinstance(v, dict) else v) for rel, v in manifest['files'].items()}
        sizes_bad = [rel for rel, v in manifest['files'].items() if isinstance(v, dict) and (snap / rel).is_file() and (snap / rel).stat().st_size != v['bytes']]
        for rel, digest in sorted(entries.items()):
            src = (snap / rel)
            if src.is_symlink() or '..' in Path(rel).parts:
                bad_src.append(rel); continue
            data = src.read_bytes()
            if sha_bytes(data) != digest:
                bad_src.append(rel); continue
            dst = restore / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data); nbytes += len(data)
        for rel, digest in entries.items():
            dst = restore / rel
            if not dst.is_file() or sha_bytes(dst.read_bytes()) != digest:
                bad_dst.append(rel)
        res['restore'] = {'files_in_manifest': len(manifest['files']), 'manifest_file_count': manifest.get('file_count'),
                          'bytes_restored': nbytes, 'manifest_total_bytes': manifest.get('total_bytes'), 'size_mismatches': sizes_bad, 'source_hash_failures': bad_src, 'restored_hash_failures': bad_dst}
        dump()
        guard()
        inner = C.npz(restore / 'inputs/INNER_features.npz')
        X = inner['X']
        reps = {}
        for seed in (0, 1):
            for pk in PACKAGES:
                sid = f'{pk}_S{seed}'; d = restore / 'deployment' / sid
                dep = json.loads((d / 'DEPLOYMENT.json').read_text())
                rec = {'kind': dep['kind']}
                rec['package_hashes_ok'] = all(sha_bytes((d / x['file']).read_bytes()) == x['sha256']
                                               for x in (dep['geometry'], dep['adapter'], dep['ownproj']) if x) and \
                    sha_bytes((d / 'model_state.pt').read_bytes()) == dep['model_sha256']
                rec['feature_names_ok'] = list(map(str, dep['feature_names'])) == list(map(str, inner['feature_names']))
                sd = C.tload(d / 'model_state.pt')
                if isinstance(sd, dict) and 'model' in sd and not any(k.startswith('enc.') for k in sd):
                    sd = sd['model']
                src = C.Source(sd)
                geom = json.loads((d / 'geometry.json').read_text())
                if dep['kind'] == 'ADAPTER':
                    r1, r2 = wires_for('ADAPTER', X, src, geom, ckpt=C.tload(d / 'adapter.pt'))
                elif dep['kind'] == 'RAW_PROJ':
                    r1, r2 = wires_for('RAW_PROJ', X, src, geom)
                    ad = C.tload(d / 'adapter.pt')
                    rz1, rz2 = wires_for('ADAPTER', X, src, geom, ckpt=ad)
                    rec['raw_proj_zero_adapter_bitwise'] = bool(np.array_equal(rz2, r2)) and not ad['R'].any() and not ad['b_res'].any()
                elif dep['kind'] == 'OWNPROJ':
                    g = json.loads((d / 'ownproj.json').read_text())['occupation_geometry']
                    own = torch.from_numpy(np.asarray(g['B'], np.float64).reshape(int(g['rank']), int(g['source_dimension'])).copy())
                    r1, r2 = wires_for('OWNPROJ', X, src, geom, prior_sd=sd, own_B=own)
                else:
                    raise ValueError(dep['kind'])
                exp = C.npz(d / 'INNER_expected_wire.npz')
                rec['expected_keys'] = sorted(exp)
                svc = pk.replace('_E10', '')
                rel = C.npz(restore / f'readers/MTA_READ_S{seed}_{svc}_V1/INNER_RELEASE.npz')
                for name, ref in (('expected', exp), ('reader_release', rel)):
                    rec[name] = {'r1_maxabs': float(np.max(np.abs(r1 - ref['r1']))), 'r2_maxabs': float(np.max(np.abs(r2 - ref['r2']))),
                                 'r1_bitwise': bool(np.array_equal(r1, ref['r1'])), 'r2_bitwise': bool(np.array_equal(r2, ref['r2'])),
                                 'decisions_equal': bool(np.array_equal(r1[:, -1], ref['r1'][:, -1]) and np.array_equal(r2[:, -1], ref['r2'][:, -1]))}
                rec['reader_release_row_ids_match_input'] = bool(np.array_equal(rel['row_id'], inner['row_id']))
                inp = C.npz(d / 'INNER_input.npz')
                rec['package_input_equals_snapshot_input'] = bool(np.array_equal(inp.get('X'), X)) if 'X' in inp else sorted(inp)
                reps[sid] = rec
                res['replay'] = reps; dump()
    except Exception:
        res['error'] = traceback.format_exc()
    finally:
        if restore.exists():
            shutil.rmtree(restore)
        res['restore_copy_deleted'] = not restore.exists()
    dump()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    main(ap.parse_args().spec)

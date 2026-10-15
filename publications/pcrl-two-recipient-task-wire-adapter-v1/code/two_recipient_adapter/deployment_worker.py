"""Queue-only two-recipient package export, raw83 execution, stored-release parity and refusal checks."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path


def _refused(fn):
    try:
        fn()
    except (ValueError, PermissionError, SystemExit, KeyError):
        return True
    return False


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from . import deployment as D
    out = Path(cfg['private_output'])
    if '.private' not in out.resolve().parts or out.exists():
        raise PermissionError('fresh private deployment directory required')
    out.mkdir(parents=True, mode=0o700)
    roles = {}
    for name, item in cfg['feature_roles'].items():
        with np.load(verify(item['path'], item['sha256']), allow_pickle=False) as z:
            roles[name] = {'X': z['X'].copy(), 'feature_names': z['feature_names'].copy()}
    results = {'schema': 'TRA1-deployment-v1', 'queue_unit_id': active['unit_id'], 'packages': {}}
    for pkg in cfg['packages']:
        guard()
        seed = pkg['seed']
        src = torch.load(verify(pkg['source']['path'], pkg['source']['sha256']), map_location='cpu', weights_only=False)
        geometry = json.loads(verify(pkg['geometry']['path'], pkg['geometry']['sha256']).read_text())
        binding = json.loads(verify(pkg['binding']['path'], pkg['binding']['sha256']).read_text())
        from .services import resolve
        states, scales, prov = resolve(pkg['service'], verify=verify, seed=seed, source_checkpoint_sha256=pkg['source']['sha256'],
                                       adapter_binding_sha256=pkg['binding']['sha256'], binding_scales=binding['scales'])
        states = [s if s is not None else {'R': torch.zeros(r, 64, dtype=torch.float64), 'b': torch.zeros(r, dtype=torch.float64)}
                  for s, r in zip(states, binding['ranks'])]
        root = out / pkg['label']
        msha = D.export(root, service_id=pkg['label'], kind=pkg['kind'], seed=seed, source_payload=src,
                        source_checkpoint_sha256=pkg['source']['sha256'], geometry_binding=geometry, states=states, scales=scales,
                        scientific_status=pkg['scientific_status'], provenance=pkg['provenance'])
        rt, meta = D.load_package(root, msha)
        rec = {'service_id': pkg['label'], 'seed': seed, 'kind': pkg['kind'], 'manifest_sha256': msha,
               'scientific_status': pkg['scientific_status'], 'roles': {}}
        stored = None
        with np.load(verify(pkg['stored_inner_release']['path'], pkg['stored_inner_release']['sha256']), allow_pickle=False) as z:
            stored = {'r1': z['r1'].copy(), 'r2': z['r2'].copy()}
        for rname, data in roles.items():
            live = rt.release(data['X'], data['feature_names'].tolist())
            inp = root / f'{rname}_input.npz'; cli = root / f'{rname}_cli_output.npz'
            with inp.open('xb') as f:
                np.savez(f, X=data['X'], feature_names=data['feature_names'])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                D.main(['--package', str(root), '--manifest-sha256', msha, '--input', str(inp), '--input-sha256', D.sha_file(inp), '--output', str(cli)])
            with np.load(cli, allow_pickle=False) as z:
                cli_out = {'r1': z['r1'].copy(), 'r2': z['r2'].copy()}
            entry = {'rows': int(len(data['X'])), 'cli_equals_live': all(np.array_equal(cli_out[k], live[i]['full']) for i, k in enumerate(('r1', 'r2'))),
                     'cli_output_sha256': D.sha_file(cli), 'dimensions': [int(live[0]['full'].shape[1]), int(live[1]['full'].shape[1])]}
            if rname == 'INNER':
                entry['live_vs_stored_release_max_abs'] = [float(np.max(np.abs(live[i]['full'] - stored[k]))) for i, k in enumerate(('r1', 'r2'))]
                entry['live_decisions_equal_stored'] = bool(np.array_equal(live[1]['full'][:, -1], stored['r2'][:, -1]) and np.array_equal(live[0]['full'][:, -1], stored['r1'][:, -1]))
                entry['live_probabilities_equal_stored'] = bool(np.array_equal(live[1]['full'][:, 22:28], stored['r2'][:, 22:28]) and np.array_equal(live[0]['full'][:, 18:20], stored['r1'][:, 18:20]))
                with (root / 'INNER_expected_wire.npz').open('xb') as f:
                    np.savez(f, r1=stored['r1'], r2=stored['r2'])
                if any(entry['live_vs_stored_release_max_abs']):
                    raise AssertionError('deployment output differs from stored release')
            rec['roles'][rname] = entry
        X = roles['INNER']['X'][:8]; names = roles['INNER']['feature_names'].tolist()
        bad = X.copy(); bad[0, 0] = np.nan
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            t = Path(tmp); extra = t / 'extra.npz'; teacher = t / 'teacher.npz'; ids = t / 'ids.npz'
            np.savez(extra, X=X, feature_names=np.array(names), labels=np.zeros(8))
            np.savez(teacher, X=X, feature_names=np.array(names), teacher_score=np.zeros((8, 6)))
            np.savez(ids, X=X, feature_names=np.array(names), row_id=np.arange(8))
            def cli(path, extra_args=()):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    D.main(['--package', str(root), '--manifest-sha256', msha, '--input', str(path), '--input-sha256', D.sha_file(path),
                            '--output', str(t / f'o{np.random.randint(1e9)}.npz'), *extra_args])
            pk = t / 'pkg'; shutil.copytree(root, pk)
            target = pk / 'adapter.pt'
            b = bytearray(target.read_bytes()); b[-9] ^= 1; target.write_bytes(bytes(b))
            mism = t / 'mism'; shutil.copytree(root, mism)
            mm = json.loads((mism / 'DEPLOYMENT.json').read_text()); mm['source_checkpoint_sha256'] = '0' * 64; mm['seed'] = 1 - seed
            (mism / 'DEPLOYMENT.json').write_text(json.dumps(mm))
            checks = {'reordered_columns': _refused(lambda: rt.release(X[:, ::-1], names[::-1])),
                      'missing_column': _refused(lambda: rt.release(X[:, :82], names[:82])),
                      'extra_column': _refused(lambda: rt.release(np.column_stack((X, X[:, :1])), names + ['extra'])),
                      'nonfinite_input': _refused(lambda: rt.release(bad, names)),
                      'integer_input': _refused(lambda: rt.release(X.astype(np.int64), names)),
                      'forbidden_original_feature_name': _refused(lambda: rt.release(X, ['sex'] + names[1:])),
                      'extra_output_channel_request': _refused(lambda: rt.release(X, names, teacher=True)),
                      'unprotected_output_bypass_request': _refused(lambda: rt.release(X, names, return_z16=True)),
                      'unknown_cli_flag': _refused(lambda: cli(root / 'INNER_input.npz', ['--emit-teacher'])),
                      'label_field_in_input': _refused(lambda: cli(extra)),
                      'teacher_score_field_in_input': _refused(lambda: cli(teacher)),
                      'fine_id_field_in_input': _refused(lambda: cli(ids)),
                      'wrong_manifest_hash': _refused(lambda: D.load_package(root, '0' * 64)),
                      'mismatched_checkpoint_manifest': _refused(lambda: D.load_package(mism, D.sha_file(mism / 'DEPLOYMENT.json'))),
                      'tampered_weights': _refused(lambda: D.load_package(pk, msha))}
        if not all(checks.values()):
            raise AssertionError(f'refusal check failed: {checks}')
        rec['refusal_checks'] = checks
        results['packages'][pkg['label']] = rec
    with Path(cfg['output']).open('x') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)

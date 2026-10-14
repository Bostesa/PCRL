"""Queue-only package export, label-free raw83 execution, wire parity and rejection checks."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path


def _expect_error(fn):
    try:
        fn()
    except (ValueError, PermissionError):
        return True
    return False


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from task_conditioned_representation.models import load_checkpoint
    from .deployment import load_package, export_checkpoint, sha_file, main as runtime_main
    from .wires import forward_role, prepare_geometry
    roles = {}
    for name, item in cfg['feature_roles'].items():
        with np.load(verify(item['path'], item['sha256']), allow_pickle=False) as z:
            if set(z.files) != {'row_id', 'unit', 'X', 'feature_names'}:
                raise PermissionError('feature-only export required for deployment')
            roles[name] = {'X': z['X'].copy(), 'feature_names': z['feature_names'].copy()}
    out = Path(cfg['private_output'])
    if '.private' not in out.resolve().parts or out.exists():
        raise PermissionError('fresh private deployment directory required')
    out.mkdir(parents=True, mode=0o700)
    results = {'schema': 'SPG1-deployment-v1', 'queue_unit_id': active['unit_id'], 'packages': {}, 'aliases': {}}
    for pkg in cfg['packages']:
        guard()
        seed = pkg['seed']; g = cfg['geometry'][str(seed)]
        binding = json.loads(verify(g['path'], g['sha256']).read_text())
        geometry = prepare_geometry(binding, expected_sha256=g['binding_sha256'], source_seed=seed)
        ck = verify(pkg['checkpoint']['path'], pkg['checkpoint']['sha256'])
        root = out / pkg['label']
        manifest_sha = export_checkpoint(ck, pkg['checkpoint']['sha256'], root, service_id=pkg['service_id'],
                                         geometry_binding=binding, geometry_binding_sha256=g['binding_sha256'], seed=seed)
        runtime, meta = load_package(root, manifest_sha)
        model, _ = load_checkpoint(ck, pkg['checkpoint']['sha256'])
        rec = {'service_id': pkg['service_id'], 'seed': seed, 'epoch': pkg['epoch'], 'status_note': pkg['note'],
               'manifest_sha256': manifest_sha, 'model_sha256': meta['model_sha256'],
               'source_checkpoint_sha256': pkg['checkpoint']['sha256'], 'roles': {}}
        for rname, data in roles.items():
            names = data['feature_names'].tolist()
            actual = runtime.release(data['X'], names)
            canonical = forward_role(model, data['X'], data['feature_names'], geometry, variant=pkg['service_id'])
            diffs = [float(np.max(np.abs(actual[i]['full'] - canonical[i]['wire']))) for i in range(2)]
            if any(diffs):
                raise AssertionError('standalone release differs from canonical wire')
            inp = root / f'{rname}_input.npz'; exp = root / f'{rname}_expected_wire.npz'; cli = root / f'{rname}_cli_output.npz'
            with inp.open('xb') as f:
                np.savez(f, X=data['X'], feature_names=data['feature_names'])
            with exp.open('xb') as f:
                np.savez(f, r1=canonical[0]['wire'], r2=canonical[1]['wire'])
            argv = sys.argv
            buf = io.StringIO()
            try:
                sys.argv = ['deployment', '--package', str(root), '--manifest-sha256', manifest_sha, '--input', str(inp),
                            '--input-sha256', sha_file(inp), '--output', str(cli)]
                with contextlib.redirect_stdout(buf):
                    runtime_main()
            finally:
                sys.argv = argv
            with np.load(cli, allow_pickle=False) as z:
                cli_diffs = [float(np.max(np.abs(z[f'r{i+1}'] - canonical[i]['wire']))) for i in range(2)]
            if any(cli_diffs):
                raise AssertionError('CLI output differs from stored canonical wire')
            rec['roles'][rname] = {'rows': int(len(data['X'])), 'dimensions': [int(actual[0]['full'].shape[1]), int(actual[1]['full'].shape[1])],
                                   'runtime_vs_canonical_max_abs': diffs, 'cli_vs_canonical_max_abs': cli_diffs,
                                   'expected_wire_sha256': sha_file(exp), 'input_sha256': sha_file(inp)}
        X = roles['VALIDATION']['X'][:8]; names = roles['VALIDATION']['feature_names'].tolist()
        bad = X.copy(); bad[0, 0] = np.nan
        checks = {'reordered_columns': _expect_error(lambda: runtime.release(X[:, ::-1], names[::-1])),
                  'missing_column': _expect_error(lambda: runtime.release(X[:, :82], names[:82])),
                  'nonfinite_input': _expect_error(lambda: runtime.release(bad, names)),
                  'integer_input': _expect_error(lambda: runtime.release(X.astype(np.int64), names)),
                  'extra_output_channel_request': _expect_error(lambda: runtime.release(X, names, labels=True)),
                  'forbidden_column_name': _expect_error(lambda: runtime.release(X, ['sex'] + names[1:])),
                  'wrong_manifest_hash': _expect_error(lambda: load_package(root, '0' * 64))}
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            t = Path(tmp) / 'pkg'; shutil.copytree(root, t)
            w = t / 'model_state.pt'; b = bytearray(w.read_bytes()); b[-9] ^= 1; w.write_bytes(bytes(b))
            checks['tampered_weights_hash'] = _expect_error(lambda: load_package(t, manifest_sha))
        if not all(checks.values()):
            raise AssertionError(f'rejection check failed: {checks}')
        rec['rejection_checks'] = checks
        results['packages'][pkg['label']] = rec
    for alias in cfg['aliases']:
        a = torch.load(verify(alias['checkpoint']['path'], alias['checkpoint']['sha256']), map_location='cpu', weights_only=False)['model']
        b = torch.load(verify(alias['source']['path'], alias['source']['sha256']), map_location='cpu', weights_only=False)['model']
        same = set(a) == set(b) and all(torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)
        if not same:
            raise AssertionError('selected epoch-0 checkpoint is not a literal alias of the start')
        results['aliases'][alias['label']] = {'literal_tensor_alias_of': alias['alias_of'], 'bitwise_equal': True,
                                              'checkpoint_sha256': alias['checkpoint']['sha256']}
    with Path(cfg['output']).open('x') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)

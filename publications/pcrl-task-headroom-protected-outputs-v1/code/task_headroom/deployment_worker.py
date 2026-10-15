"""Build hash-bound deployment packages, run the CLI on real admitted inputs, test refusals.

Parity: CLI outputs on INNER must equal the frozen INNER releases attacked by
the readers (probabilities bitwise, decisions exact); ATT and FIT2 are also run.
No labels are inputs. Refusal checks exercise every refusal path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build(dest, pkg, verify, feature_names):
    dest.mkdir(parents=True, mode=0o700)
    files, teachers = {}, {}
    for t in ('income', 'occupation'):
        src = verify(pkg['teachers'][t]['path'], pkg['teachers'][t]['sha256'])
        shutil.copyfile(src, dest / f'teacher_{t}.json'); teachers[t] = f'teacher_{t}.json'; files[f'teacher_{t}.json'] = _sha(dest / f'teacher_{t}.json')
    theta = None
    if pkg.get('theta'):
        shutil.copyfile(verify(pkg['theta']['path'], pkg['theta']['sha256']), dest / 'theta.npz'); theta = 'theta.npz'; files['theta.npz'] = _sha(dest / 'theta.npz')
    m = {'schema': 'THP1-tree-service-package-v1', 'service': pkg['label'], 'seed': pkg['seed'], 'nominee_key': pkg['key'],
         'feature_names': list(map(str, feature_names)), 'teachers': teachers, 'theta': theta, 'files': files,
         'outputs': ['p_income (n,2)', 'decision_income (n,)', 'p_occupation (n,6)', 'decision_occupation (n,)'],
         'not_transmitted': ['leaf ids', 'raw scores', 'teacher outputs', 'theta', 'routing', 'SEX', 'task labels'],
         'runtime': 'python -m task_headroom.deploy (numpy only; task_headroom/trees.py + deploy.py)'}
    (dest / 'MANIFEST.json').write_text(json.dumps(m, indent=2, sort_keys=True))
    return _sha(dest / 'MANIFEST.json')


def cli(args, cwd, env):
    r = subprocess.run([sys.executable, '-m', 'task_headroom.deploy', *args], cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, (json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {'stderr': r.stderr[-300:]})


def run(spec_path):
    import os
    from .evaluation_queue import queue_config
    from .data import load_role
    active, cfg, verify, guard = queue_config(spec_path, 'custody')
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private deployment directory required')
    out.mkdir(parents=True, mode=0o700)
    roles = {r: load_role(r, cfg['roles'][r], verify, ('features',)) for r in ('FIT2', 'ATT', 'INNER')}
    names = roles['INNER']['feature_names']
    inputs = {}
    for r, d in roles.items():
        p = out / f'input_{r}.npz'
        with p.open('xb') as f:
            np.savez(f, X=d['X'], feature_names=d['feature_names'])
        inputs[r] = (p, _sha(p))
    env = {'PYTHONPATH': cfg['pythonpath'], 'PATH': '/usr/bin:/bin', **{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}}
    results = {}
    for pkg in cfg['packages']:
        guard()
        name = f"{pkg['label']}_S{pkg['seed']}"; dest = out / name
        msha = build(dest, pkg, verify, names)
        rec = {'manifest_sha256': msha, 'runs': {}, 'refusals': {}}
        for r, (p, s) in inputs.items():
            o = out / f'{name}__{r}.npz'
            code, info = cli(['--package', str(dest), '--manifest-sha256', msha, '--input', str(p), '--input-sha256', s, '--output', str(o)], cfg['repo'], env)
            rec['runs'][r] = {'returncode': code, **info}
        with np.load(verify(pkg['inner_release']['path'], pkg['inner_release']['sha256'])) as z, np.load(out / f'{name}__INNER.npz') as c:
            rec['inner_parity'] = {'p_income_bitwise': bool(np.array_equal(z['p1'], c['p_income'])), 'p_occupation_bitwise': bool(np.array_equal(z['p2'], c['p_occupation'])),
                                   'decisions_identical': bool(np.array_equal(z['d1'], c['decision_income']) and np.array_equal(z['d2'], c['decision_occupation'])),
                                   'max_abs_diff': float(max(np.abs(z['p1'] - c['p_income']).max(), np.abs(z['p2'] - c['p_occupation']).max())),
                                   'output_keys': sorted(c.files)}
        # Refusal checks.
        p, s = inputs['INNER']
        X = roles['INNER']['X']; tmp = out / f'{name}__tmp'
        tmp.mkdir()
        def bad_input(fname, **arrays):
            q = tmp / fname
            with q.open('xb') as f:
                np.savez(f, **arrays)
            return q, _sha(q)
        cases = {
            'wrong_manifest_sha': ['--package', str(dest), '--manifest-sha256', '0' * 64, '--input', str(p), '--input-sha256', s, '--output', str(tmp / 'o1.npz')],
            'wrong_input_sha': ['--package', str(dest), '--manifest-sha256', msha, '--input', str(p), '--input-sha256', '0' * 64, '--output', str(tmp / 'o2.npz')],
            'hidden_leaves_request': ['--package', str(dest), '--manifest-sha256', msha, '--input', str(p), '--input-sha256', s, '--output', str(tmp / 'o3.npz'), '--emit', 'leaves'],
            'teacher_output_request': ['--package', str(dest), '--manifest-sha256', msha, '--input', str(p), '--input-sha256', s, '--output', str(tmp / 'o4.npz'), '--emit', 'teacher'],
            'existing_output': ['--package', str(dest), '--manifest-sha256', msha, '--input', str(p), '--input-sha256', s, '--output', str(out / f'{name}__INNER.npz')],
        }
        for label, (q, qs) in {
            'reordered_columns': bad_input('a.npz', X=X[:, ::-1], feature_names=names[::-1]),
            'missing_column': bad_input('b.npz', X=X[:, :82], feature_names=names[:82]),
            'extra_sex_column': bad_input('c.npz', X=np.column_stack((X, np.zeros(len(X)))).astype(np.float32), feature_names=np.append(names, 'sex')),
            'label_array_present': bad_input('d.npz', X=X, feature_names=names, sex=np.zeros(len(X), dtype=np.int64)),
            'nonfinite': bad_input('e.npz', X=np.where(np.arange(X.size).reshape(X.shape) == 0, np.nan, X).astype(np.float32), feature_names=names),
            'renamed_column': bad_input('f.npz', X=X, feature_names=np.where(np.arange(83) == 0, 'AGE', names)),
            'integer_matrix': bad_input('g.npz', X=X.astype(np.int64), feature_names=names),
        }.items():
            cases[label] = ['--package', str(dest), '--manifest-sha256', msha, '--input', str(q), '--input-sha256', qs, '--output', str(tmp / f'o_{label}.npz')]
        # Tampered package file.
        tam = out / f'{name}__tampered'; shutil.copytree(dest, tam)
        t = json.loads((tam / 'teacher_income.json').read_text()); t['bias'] = [t['bias'][0] + 1e-9]; (tam / 'teacher_income.json').write_text(json.dumps(t))
        cases['tampered_package_file'] = ['--package', str(tam), '--manifest-sha256', msha, '--input', str(p), '--input-sha256', s, '--output', str(tmp / 'o5.npz')]
        for label, args in cases.items():
            code, info = cli(args, cfg['repo'], env)
            rec['refusals'][label] = {'returncode': code, 'refused': code == 2 and info.get('status') == 'REFUSED', 'reason': info.get('reason')}
        shutil.rmtree(tam); shutil.rmtree(tmp)
        rec['all_refusals_ok'] = all(v['refused'] for v in rec['refusals'].values())
        rec['parity_ok'] = all(rec['inner_parity'][k] for k in ('p_income_bitwise', 'p_occupation_bitwise', 'decisions_identical')) and all(v['returncode'] == 0 for v in rec['runs'].values())
        results[name] = rec
    summary = {'schema': 'THP1-deployment-v1', 'queue_unit_id': active['unit_id'], 'packages': results,
               'all_parity_ok': all(r['parity_ok'] for r in results.values()), 'all_refusals_ok': all(r['all_refusals_ok'] for r in results.values()),
               'refusal_checks_per_package': len(next(iter(results.values()))['refusals'])}
    with Path(cfg['output']).open('x') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)

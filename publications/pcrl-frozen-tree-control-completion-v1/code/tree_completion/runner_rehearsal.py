"""Rehearsals of the confirmation runner: synthetic fixtures and already-used development roles (never fresh data)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


def validate_bindings(protocol):
    from .confirmation_runner import _sha
    bad = []
    def walk(obj, path=''):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                if _sha(obj['path']) != obj['sha256']:
                    bad.append(path)
            for k, v in obj.items():
                walk(v, f'{path}/{k}')
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f'{path}[{i}]')
    walk({'candidate': protocol['candidate'], 'comparators': protocol['comparators']})
    for svc in [protocol['candidate']] + list(protocol['comparators'].values()):
        if svc['kind'] == 'tree_package':
            for s, b in svc['per_seed'].items():
                if _sha(Path(b['package_dir']) / 'MANIFEST.json') != b['manifest_sha256']:
                    bad.append(f'package manifest seed {s}')
    if bad:
        raise PermissionError(f'binding hash mismatch: {bad}')
    return True


def synthetic_roles(names, n_att=4064, n_ass=2235, seed=0):
    rng = np.random.default_rng(seed)
    def make(n, offset):
        X = np.zeros((n, 83), dtype=np.float32)
        X[:, :5] = rng.normal(size=(n, 5))
        blocks = {}
        for j, nm in enumerate(names):
            if '=' in nm:
                blocks.setdefault(nm.split('=')[0], []).append(j)
        for cols in blocks.values():
            X[np.arange(n), rng.choice(cols, n)] = 1.0
        sex = (X[:, 0] + rng.normal(size=n) > -0.4).astype(np.int64)
        unit = np.arange(n) + offset; rid = np.arange(n) + offset
        return {'X': X, 'feature_names': np.asarray(names), 'row_id': rid, 'unit': unit, 'sex': sex}
    att = make(n_att, 0); ass = make(n_ass, 10 ** 6)
    yi = (ass['X'][:, 1] + rng.normal(size=n_ass) > 0.8).astype(np.int64)
    yo = np.clip((ass['X'][:, 2] * 1.3 + 2.5 + rng.normal(size=n_ass)).astype(int), 0, 5).astype(np.int64)
    refs = {s: {'U0': [rng.dirichlet([3, 1], n_ass), rng.dirichlet([1] * 6, n_ass)], 'Ucal': [rng.dirichlet([3, 1], n_ass), rng.dirichlet([1] * 6, n_ass)]} for s in ('0', '1')}
    return att, ass, {'income': yi, 'occupation': yo, 'sex': ass['sex']}, refs


def rehearse(mode, protocol, spec_path):
    from .confirmation_runner import run_pipeline, Immutable
    validate_bindings(protocol)
    if mode == 'synthetic-rehearsal':
        cfg = json.loads(Path(spec_path).read_text()) if spec_path else {}
        names = protocol['feature_names']
        att, ass, y, refs = synthetic_roles(names)
        proto = dict(protocol, mode_note='SYNTHETIC REHEARSAL: synthetic features/labels/SEX/references; frozen packages real; no evidence')
        res, msha = run_pipeline(proto, att, ass, lambda: (y, ass['unit']), lambda: refs, cfg['report_dir'], cfg['workdir'], adult_split=False)
        print(json.dumps({'mode': mode, 'iut_global_claim_confirmed': res['iut_global_claim_confirmed'], 'manifest_sha256': msha, 'clauses': len(res['rows'])}))
        return 0
    if mode == 'development-rehearsal':
        from .evaluation_queue import queue_config
        from task_headroom.data import load_role, load_npz
        active, cfg, verify, guard = queue_config(spec_path, 'verification', allow_inner_truth=True)
        R = cfg['roles']
        att = load_role('ATT', R['ATT'], verify, ('features', 'sex'))
        ass = load_role('INNER', R['INNER'], verify, ('features',))
        # Stored-release fidelity of the runnable comparators (must reproduce the attacked releases bitwise).
        from .confirmation_runner import release
        fidelity = {}
        for label, svc in protocol['comparators'].items():
            for s in protocol['seeds']:
                w = release(svc, ass['X'], ass['feature_names'], int(s), cfg['workdir'])
                st = np.load(verify(cfg['stored_inner_releases'][label][s]['path'], cfg['stored_inner_releases'][label][s]['sha256']))
                fidelity[f'{label}|s{s}'] = bool(np.array_equal(w[0]['probability'], st['p1']) and np.array_equal(w[1]['probability'], st['p2'])
                                                 and np.array_equal(w[0]['decision'], st['d1']) and np.array_equal(w[1]['decision'], st['d2']))
        def labels():
            t = {k: load_npz(cfg['inner_truth'][k], verify, ('row_id', 'unit', k)) for k in ('income', 'occupation', 'sex')}
            if not np.array_equal(t['sex']['row_id'], ass['row_id']):
                raise PermissionError('INNER truth pairing differs')
            return {k: t[k][k].astype(np.int64) for k in t}, t['sex']['unit']
        def refs():
            out = {}
            for s in protocol['seeds']:
                r = load_npz(cfg['inner_references'][s], verify, ('row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2'))
                out[s] = {'U0': [r['U0_1'], r['U0_2']], 'Ucal': [r['Ucal_1'], r['Ucal_2']]}
            return out
        proto = dict(protocol, mode_note='DEVELOPMENT REHEARSAL on already-used ATT/INNER people: reproduces frozen statistics; not evidence')
        res, msha = run_pipeline(proto, att, ass, labels, refs, cfg['report_dir'], cfg['workdir'], adult_split=True)
        expected = cfg['expected']
        got = {r['clause']: r['point'] for r in res['rows']}
        repro = {k: {'expected': v, 'got': got.get(k), 'match_1e-12': got.get(k) is not None and abs(got[k] - v) <= 1e-12} for k, v in expected.items()}
        out = {'schema': 'TC1-runner-development-rehearsal-v1', 'queue_unit_id': active['unit_id'], 'comparator_release_fidelity': fidelity,
               'reproduction_of_frozen_statistics': repro, 'iut_global_claim_confirmed_on_exposed_INNER': res['iut_global_claim_confirmed'],
               'sim_all_pass': res['sim_all_pass'], 'report_manifest_sha256': msha, 'not_evidence': True}
        with Path(cfg['output']).open('x') as f:
            json.dump(out, f, indent=2)
        return 0
    raise ValueError(mode)

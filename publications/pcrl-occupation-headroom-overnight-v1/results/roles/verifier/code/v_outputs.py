"""VER check 5: actual transmitted outputs (label-free).

contract  every stored occupation release (runs/<F>/<C>/S<s>/OCC.npz) and complete release (RELEASE.npz): all six roles present in
          the admitted row order (row counts per role), (n, 5) / (n, 2) float64, finite, non-negative, rows sum to 1 within 1e-9,
          exact argmax ties counted (a tie makes the decision depend on the argmax convention), the file hash against the record and
          every wire hash (float64 vector + int64 argmax decision) recomputed with own code against the record and the evaluator.
predict   for sampled configurations: the family's own `predict(seed, config, X, names)` (the deployed function) recomputed from each
          role's admitted features alone and compared bitwise with the stored release (run with the release's recorded interpreter).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
SIZES = {'MODEL_FIT': 20000, 'CALIBRATION': 2982, 'MODEL_SELECT': 8001, 'ATTACK_FIT': 8001, 'ATTACK_VALIDATION': 3000, 'DEV_REPORT': 5943}


def run_contract():
    prev = V.RECORDS / 'VER_OUTPUTS.json'
    db = json.loads(prev.read_text()) if prev.is_file() else {'check': 'VER_OUTPUTS', 'contract': {}, 'predict': {}}
    new = []
    for p in sorted(V.RUNS.glob('*/*/S*/OCC.npz')) + sorted(V.RUNS.glob('*/*/S*/RELEASE.npz')):
        key = str(p.relative_to(V.RUNS))
        h = V.file_sha256(p)
        if db['contract'].get(key, {}).get('file_sha256') == h:
            continue
        rec_p = p.with_suffix('.json').with_name('OCC.json' if p.name == 'OCC.npz' else 'RELEASE.json')
        r = json.loads(rec_p.read_text())
        rec_hash = r.get('occ_sha256') if p.name == 'OCC.npz' else r.get('release_sha256')
        res = {'file_sha256': h, 'file_hash_equals_record': h == rec_hash, 'roles': {}, 'kind': r.get('kind')}
        with np.load(p) as z:
            keys = set(z.files)
            tasks = ('occupation',) if p.name == 'OCC.npz' else ('income', 'occupation')
            res['keys_exact'] = keys == {f'{role}__{t}' for role in ROLES6 for t in tasks}
            for role in ROLES6:
                for t in tasks:
                    a = z[f'{role}__{t}']
                    c = V.check_prob(a, V.NCLASS[t])
                    c['rows_ok'] = a.shape[0] == SIZES[role]
                    c['wire_equals_record'] = V.wire_hash(a) == r.get('wire_sha256', {}).get(f'{role}__{t}')
                    res['roles'][f'{role}__{t}'] = c
        evs = {}
        for ev_p in sorted(p.parent.glob('EVAL_MS_*.json')):
            ev = json.loads(ev_p.read_text())
            t_ok = ev['wire_sha256']['occupation'] == r.get('wire_sha256', {}).get('MODEL_SELECT__occupation')
            evs[ev_p.name] = t_ok
        res['eval_occupation_wire_equals_release'] = evs
        res['all_ok'] = (res['file_hash_equals_record'] and res['keys_exact'] and all(c['contract_ok'] and c['rows_ok'] and c['wire_equals_record']
                                                                                  for c in res['roles'].values()) and all(evs.values()))
        res['argmax_ties_total'] = sum(c.get('argmax_exact_ties', 0) for c in res['roles'].values())
        res['verified_utc'] = V.utc()
        db['contract'][key] = res
        new.append(key)
    db['n_contract'] = len(db['contract'])
    db['n_contract_ok'] = sum(v['all_ok'] for v in db['contract'].values())
    V.write_record('VER_OUTPUTS', db)
    print(json.dumps({'new': new, 'n': db['n_contract'], 'ok': db['n_contract_ok'],
                      'not_ok': [k for k, v in db['contract'].items() if not v['all_ok']],
                      'ties': {k: v['argmax_ties_total'] for k, v in db['contract'].items() if v['argmax_ties_total']}}, indent=1))


def run_predict(entries):
    """entries: F/CONFIG/S<s>; family module ohn.fam_<f> (A, B, C) or ohn.protect (PROT)."""
    from ohn import common as C
    from ohn import devrun as D
    prev = V.RECORDS / 'VER_OUTPUTS_PREDICT.json'
    db = json.loads(prev.read_text()) if prev.is_file() else {'check': 'VER_OUTPUTS_PREDICT', 'predict': {}}
    feats = {role: C.load_role(role, ('features',)) for role in ROLES6}
    for ent in entries:
        fam, cfg, s = ent.split('/')
        seed = int(s[1:])
        mod = importlib.import_module({'A': 'ohn.fam_a', 'B': 'ohn.fam_b', 'C': 'ohn.fam_c', 'PROT': 'ohn.protect'}[fam])
        occ, rec, _ = D.load_occupation(fam, cfg, seed)
        res = {'interpreter_now': sys.executable, 'interpreter_recorded': rec.get('interpreter'), 'roles': {}}
        for role in ROLES6:
            X, names = feats[role]['X'], feats[role]['feature_names']
            p = mod.predict(seed, cfg, X, names)
            res['roles'][role] = {'bitwise': bool(np.array_equal(p, occ[role])), 'max_abs_diff': float(np.max(np.abs(p - occ[role]))),
                                  'decisions_equal': bool(np.array_equal(np.argmax(p, 1), np.argmax(occ[role], 1))),
                                  'wire_equal': V.wire_hash(p) == rec['wire_sha256'][f'{role}__occupation']}
        res['all_bitwise'] = all(v['bitwise'] for v in res['roles'].values())
        res['unit_id'] = V.unit_id()
        res['utc'] = V.utc()
        db['predict'][ent] = res
        print(ent, 'all_bitwise', res['all_bitwise'], {r: v['max_abs_diff'] for r, v in res['roles'].items()}, flush=True)
    V.write_record('VER_OUTPUTS_PREDICT', db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('contract', 'predict'))
    ap.add_argument('--entries', nargs='*', default=[])
    a = ap.parse_args()
    if a.mode == 'contract':
        run_contract()
    else:
        run_predict(a.entries)


if __name__ == '__main__':
    main()

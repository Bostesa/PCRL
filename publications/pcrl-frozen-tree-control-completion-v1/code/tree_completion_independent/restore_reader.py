"""Subprocess helper for Stage RESTORE (run by file path with PYTHONPATH = <restore>/code only).

Loads restored reader models (joblib records hold a scikit-learn model plus the restored scaler class, which is
why the restored code directory must be importable), rebuilds the reader input table from the restored INNER
release with own code, scores, orients with the restored receipt's frozen sign, and compares bitwise with the
restored INNER_SCORES. Prints one JSON line. Self-contained: imports nothing from the worktree.
"""
import json
import sys

import joblib
import numpy as np


def tables(z):
    out = {}
    for tag, p, d, k in (('income', z['p1'], z['d1'], 2), ('occupation', z['p2'], z['d2'], 6)):
        p = np.asarray(p, dtype=np.float64); d = np.asarray(d, dtype=np.int64); one = np.eye(k)[d]
        out[tag] = {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
                    'dec': one, 'conf': p.max(1, keepdims=True)}
    out['pair'] = {k: np.column_stack((out['income'][k], out['occupation'][k])) for k in out['income']}
    return out


KIND = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}


def main():
    root, svc, jobs = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    rec = json.load(open(f'{root}/FROZEN_COMPLETE.json'))
    with np.load(f'{root}/INNER_RELEASE__{svc}.npz', allow_pickle=False) as z:
        rel = {k: z[k].copy() for k in z.files}
    with np.load(f'{root}/INNER_SCORES.npz', allow_pickle=False) as z:
        stored = {k: z[k].copy() for k in z.files if k.startswith(svc + '__')}
    tab = tables(rel); out = {}
    for job in jobs:
        view, reader = job.split(':')
        model = joblib.load(f'{root}/MODEL__{svc}__{view}__{reader}.joblib')
        m = model['model']
        if list(m.classes_) != [0, 1]:
            raise SystemExit('class order differs')
        raw = m.predict_proba(model['scaler'].transform(tab[view][KIND[reader]]))[:, 1].astype(np.float64)
        sign = rec['services'][svc]['validation'][view][reader]['orientation']
        s = raw if sign == 1 else 1 - raw
        ref = stored[f'{svc}__{view}__{reader}']
        out[job] = {'bitwise': bool(s.dtype == ref.dtype and s.tobytes() == ref.tobytes()), 'max_abs': float(np.abs(s - ref).max()),
                    'orientation': sign, 'winner': rec['services'][svc]['winner'][view]['reader']}
    print(json.dumps(out))


if __name__ == '__main__':
    main()

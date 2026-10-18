"""Independent verifier, check 7 (mechanism reading): recompute the family-H DEV_REPORT releases (successor H3_STRONG_LJDIR_PB,
twin HC_STRONG, component control HC_TTREE_LJDIR_PB; both seeds) from the six-field encoding with this script's own arithmetic and
the parameters recorded in each STATE.json, without calling ihs.fam_h:
    income log-odds d = a * (s_B + off) + b,  s_B = w0 + w . (base columns),  off = sum(fixed * D) + beta * D_J,
    D_L = s_LOCAL - s_T, D_J = s_JOINT - s_T (stored Stage-B E20 income leaf corrections on the T_TREE routing),
    base columns: STRONG = (HGB score on the compact re-coding, s_T); TTREE = (s_T,).
Occupation: JOINT_TREE E20 occupation output. Label-free (DEV_REPORT features only); SEX is never read.
Public-use ACS 2019 Florida person records; the service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW,
SCHL, MAR, POBP.
"""
from __future__ import annotations
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402

from acs_replication import trees                     # noqa: E402  (the replication's probability convention)
from ihs import common as C                           # noqa: E402  (admitted loaders: load_role, teacher, stage_b_theta)

BLOCKS = ('COW', 'SCHL', 'MAR', 'POBP')


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def my_compact(X, names):
    names = [str(n) for n in names]; X = np.asarray(X, dtype=np.float64)
    assert names[0] == 'AGEP' and names[1] == 'WKHP'
    cols = [X[:, 0], X[:, 1]]
    for b in BLOCKS:
        idx = [i for i, n in enumerate(names) if n.startswith(b + '=')]
        Z = X[:, idx]
        assert np.all(Z.sum(1) == 1) and np.all((Z == 0) | (Z == 1))
        pos = Z.argmax(1)
        if b == 'SCHL':
            vals = np.array([3.0 if names[i].endswith('__OTHER__') else float(names[i].split('=')[1]) for i in idx])
            cols.append(vals[pos])
        else:
            cols.append(pos.astype(np.float64))
    return np.column_stack(cols)


def main():
    R = C.load_role('DEV_REPORT', ('features',))
    X, names = R['X'], R['feature_names']
    res = {'labels_read': False, 'utc': C.now(), 'configs': {}}
    for s in (0, 1):
        te_i, te_o = C.teacher(s, 'income'), C.teacher(s, 'occupation')
        Li, Lo = te_i.route(X, names), te_o.route(X, names)
        thL, thJ = C.stage_b_theta(s, 'LOCAL_TREE', 'E20'), C.stage_b_theta(s, 'JOINT_TREE', 'E20')
        sT = np.asarray(te_i.raw_from_leaves(Li)).reshape(-1)
        sL = np.asarray(te_i.raw_from_leaves(Li, thL[0])).reshape(-1)
        sJ = np.asarray(te_i.raw_from_leaves(Li, thJ[0])).reshape(-1)
        occ = trees.probabilities('occupation', te_o.raw_from_leaves(Lo, thJ[1]))
        step1 = C.RUNS / 'H' / 'STEP1' / f'S{s}'
        s1 = json.loads((step1 / 'STEP1.json').read_text())
        pk_sha = fsha(step1 / 'HGB.pkl')
        with open(step1 / 'HGB.pkl', 'rb') as f:
            hgb = pickle.load(f)
        for cfg in ('H3_STRONG_LJDIR_PB', 'HC_STRONG', 'HC_TTREE_LJDIR_PB'):
            d0 = C.RUNS / 'H' / cfg / f'S{s}'
            st = json.loads((d0 / 'STATE.json').read_text())
            c = st['config']
            if c['base'] == 'STRONG':
                hs = np.asarray(hgb.decision_function(my_compact(X, names))).reshape(-1)
                Z = np.column_stack((hs, sT))
            else:
                Z = sT.reshape(-1, 1)
            wb, wp = np.asarray(st['stack_w']), np.asarray(st['platt_w'])
            sB = wb[0] + Z @ wb[1:]
            comp = {'D_L': sL - sT, 'D_J': sJ - sT, 'T_TREE': sT}
            off = np.zeros(len(sT))
            for k, v in (c.get('fixed') or {}).items():
                off = off + float(v) * comp[k]
            if c.get('direction') == 'JOINT':
                off = off + float(st['beta']) * comp['D_J']
            assert not c.get('preserve_decision')
            d = wp[0] + (sB + off).reshape(-1, 1) @ wp[1:]
            inc = trees.probabilities('income', np.asarray(d, dtype=np.float64).reshape(-1))
            p1_own = 1.0 / (1.0 + np.exp(-np.asarray(d).reshape(-1)))
            with np.load(d0 / 'RELEASE.npz') as z:
                st_inc, st_occ = z['DEV_REPORT__income'].copy(), z['DEV_REPORT__occupation'].copy()
            res['configs'][f'{cfg}|S{s}'] = {
                'base': c['base'], 'fixed': c.get('fixed'), 'direction': c.get('direction'), 'beta': st['beta'],
                'stack_w': st['stack_w'], 'platt_w': st['platt_w'],
                'hgb_pickle_sha256_matches_STEP1_and_STATE': bool(pk_sha == s1['hgb_pickle_sha256'] and (st.get('hgb_pickle_sha256') in (None, pk_sha))),
                'income_bitwise_equal_stored': bool(np.array_equal(inc, st_inc)),
                'income_max_abs_diff_stored': float(np.max(np.abs(inc - st_inc))),
                'income_p1_own_sigmoid_max_abs_diff': float(np.max(np.abs(p1_own - st_inc[:, 1]))),
                'income_decisions_equal': bool(np.array_equal(inc.argmax(1), st_inc.argmax(1))),
                'occupation_bitwise_equal_stored': bool(np.array_equal(occ, st_occ))}
    res['all_income_bitwise'] = all(v['income_bitwise_equal_stored'] for v in res['configs'].values())
    res['all_income_within_1e-12'] = all(v['income_max_abs_diff_stored'] <= 1e-12 for v in res['configs'].values())
    res['all_occupation_bitwise'] = all(v['occupation_bitwise_equal_stored'] for v in res['configs'].values())
    out = C.PRIV / 'verifier'; out.mkdir(parents=True, exist_ok=True, mode=0o700)
    (out / 'MECHANISM_RECOMPUTE.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps(res, indent=1, sort_keys=True, default=float))


if __name__ == '__main__':
    main()

"""Designer B, campaign pcrl_input_and_pair_release_v1: two extra LABEL-FREE plug-in diagnostics for PROPOSAL.md.

Same inputs, plug-in world and readers as feasibility.py (imported from it).
  REM_TOP_OCC   CTL_REM-style occupation (decision and top probability kept, the remainder replaced by the decision cell's fixed
                shares) with the twin's income: does the plug-in world reproduce the measured super-additivity of the real
                CTL_REM_1p00 (occupation view 0.560, pair 0.646)?
  CSF_CELL_m15  the R1 revision's upper bound: coordinated flattening with rates allocated over the 10 joint decision cells by the
                ORACLE within-cell variance of q per unit flattening cost (the registered rule would use MODEL_FIT SEX labels
                through a cross-fitted release-level propensity, which can only do worse), at CSF_0.15's total plug-in KL.
    python3 results/pcrl_input_and_pair_release_v1/roles/designer_b/feasibility_extra.py [--versions 0,1] [--draws 2]
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feasibility as FE  # noqa: E402
import mech_sketch as MS  # noqa: E402


def rem_top(pO):
    """Keep the decision and the top probability; spread the remainder by the decision cell's mean remainder shares."""
    pO = MS.clipn(pO)
    k = pO.argmax(1)
    top = pO[np.arange(len(pO)), k]
    out = np.zeros_like(pO)
    for c in range(pO.shape[1]):
        idx = k == c
        if not np.any(idx):
            continue
        rest = pO[idx].copy()
        rest[:, c] = 0.0
        share = rest.sum(0) / rest.sum()
        out[idx] = (1.0 - top[idx])[:, None] * share[None, :]
        out[idx, c] = top[idx]
    return MS.clipn(out)


def run_version(v, draws):
    d = FE.load(v)
    n = len(d['q'])
    splits = FE.make_splits(n, d['q'], draws, seed=[2026, 10, 22, v])
    Z, pI, pO, q = d['Z'], d['p_inc'], d['p_occ'], d['q']
    rows = {'REM_TOP_OCC': FE.evaluate(d, pI, rem_top(pO), splits)}
    st = MS.fit(Z, None, None, None, None, v, 0.0, twin_inc=pI, twin_occ=pO, mode='coord')
    fI, fO = MS.release_from_twin(dict(st, pi=1.0), Z, pI, pO)            # everyone flattened (per-person costs)
    kl_person = np.sum(pI * (np.log(pI) - np.log(fI)), 1) + np.sum(pO * (np.log(pO) - np.log(fO)), 1)
    cell = pI.argmax(1) * 5 + pO.argmax(1)
    target = FE.spend(d, *MS.release_from_twin(dict(st, pi=0.15), Z, pI, pO))['kl_total']
    ratio = {c: float(np.var(q[cell == c]) / max(kl_person[cell == c].mean(), 1e-9)) for c in np.unique(cell)}
    order = sorted(ratio, key=ratio.get, reverse=True)
    u = MS.keyed_uniform(Z, st['tag'])
    budget = target * n
    rate = {c: 0.0 for c in ratio}
    for c in order:                                                         # water-filling by oracle benefit / cost
        cost = kl_person[cell == c].sum()
        rate[c] = min(1.0, budget / cost) if cost > 0 else 1.0
        budget -= rate[c] * cost
        if budget <= 0:
            break
    sel = u < np.array([rate[c] for c in cell])
    rI, rO = pI.copy(), pO.copy()
    rI[sel], rO[sel] = st['T_inc'][pI[sel].argmax(1)], st['T_occ'][pO[sel].argmax(1)]
    rows['CSF_CELL_m15'] = FE.evaluate(d, MS.clipn(rI), MS.clipn(rO), splits)
    rows['CSF_CELL_m15']['rates'] = {f'inc{c // 5}_occ{c % 5}': round(r, 3) for c, r in rate.items()}
    rows['CSF_CELL_m15']['selected_share'] = float(sel.mean())
    return {'version': v, 'rows': rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', default='0,1')
    ap.add_argument('--draws', type=int, default=2)
    a = ap.parse_args(argv)
    t0 = time.process_time()
    res = {'schema': 'IPR-designer-B-feasibility-extra-v1', 'label_free': True, 'versions': {}}
    for v in [int(x) for x in a.versions.split(',')]:
        res['versions'][str(v)] = r = run_version(v, a.draws)
        for name, row in r['rows'].items():
            A, S = row['auc'], row['spend']
            print(f'V{v} {name:14s} pair {A["pair"]:.4f} inc {A["income"]:.4f} occ {A["occupation"]:.4f} | KL inc {S["kl_inc"]:.4f} '
                  f'occ {S["kl_occ"]:.4f} men {S["men_occ_brier"]:.4f} S2x {S["s2_ratio"]:.2f}', flush=True)
        if 'rates' in r['rows']['CSF_CELL_m15']:
            print('   cell rates', r['rows']['CSF_CELL_m15']['rates'], 'share', round(r['rows']['CSF_CELL_m15']['selected_share'], 3))
    res['cpu_seconds'] = time.process_time() - t0
    print(f'cpu_seconds {res["cpu_seconds"]:.1f}')
    (HERE / 'FEASIBILITY_EXTRA.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

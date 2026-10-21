"""Designer B: LABEL-FREE probe run BEFORE the proposal was finalised (plug-in world as feasibility.py: cached MODEL_FIT
out-of-fold statistics only; q plays the truth). Version 0 only. Output REVISION_PROBE.json.
  R1  recipient-split strengths (alpha_I, alpha_O) on the GLOBAL tilt (stratify=False; the design at the time of the probe).
  R2  decision-stratified occupation tilt (one occupation weight vector per twin occupation decision, zero within-stratum
      covariance at alpha = 1; income tilt global). Its SHUF control here centres the donor amplitude on the GLOBAL mean (as run);
      the proposal's SHUF centres on the stratum mean (mech_sketch.release_from_stats). This probe made R2 the primary design.
Each with its spend-matched SHUF control (same law, keyed donor propensities).

    python3 results/pcrl_dual_design_release_v1/roles/designer_b/revision_probe.py
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json
import time
from pathlib import Path

import numpy as np

import feasibility as FE
import mech_sketch as MS


def split_release(st, stats, a_inc, a_occ, qd=None):
    r_i, _ = MS.release_from_stats(st, stats, q_override=qd, scale=a_inc)
    _, r_o = MS.release_from_stats(st, stats, q_override=qd, scale=a_occ)
    return r_i, r_o


def dec_fit(stats, s):
    """R2: per-decision-class occupation weights (zero within-class covariance at alpha = 1)."""
    dec = stats['p_occ'].argmax(1)
    q = stats['q']
    W, qb = np.zeros((5, 5)), np.zeros(5)
    for k in range(5):
        m = dec == k
        sc = s[m] - s[m].mean()
        qb[k] = q[m].mean()
        W[k] = (MS.clog(stats['p_occ'][m]) * sc[:, None]).mean(0) / float(np.mean((q[m] - qb[k]) * sc))
    return {'W': W, 'qb': qb}


def dec_release(st_g, st_d, stats, alpha, qd=None):
    dec = stats['p_occ'].argmax(1)
    a = (stats['q'] - st_d['qb'][dec]) if qd is None else (qd - st_g['qbar'])
    r_i, _ = MS.release_from_stats(st_g, stats, q_override=qd, scale=alpha)
    r_o = MS.softmax(np.log(np.maximum(stats['p_occ'], MS.FLOOR)) - alpha * a[:, None] * st_d['W'][dec])
    return r_i, r_o


def main():
    t0 = time.process_time()
    v = 0
    d = FE.load(v)
    splits = FE.make_splits(len(d['q']), d['q'], 2, seed=[2026, 10, 21, v])
    stats = {'p_inc': d['p_inc'], 'p_occ': d['p_occ'], 'q': d['q']}
    st = MS.fit_from_stats(stats, sex_expect=d['q'], strength=1.0, stratify=False)
    qd = MS.donor_q(d['q'], key=f'DDR_B_SHUF|V{v}')
    rows = {}
    for a_i, a_o in ((0.35, 0.65), (0.25, 0.8), (0.65, 0.35)):
        name = f'R1_PTC_i{a_i}_o{a_o}'
        rows[name] = FE.evaluate(d, *split_release(st, stats, a_i, a_o), splits)
        tgt = FE.tot_kl(rows[name]['spend'])
        c = FE.match_scale(lambda x: FE.tot_kl(FE.spend(d, *split_release(st, stats, a_i * x, a_o * x, qd=qd))), tgt, hi=3.0)
        rows[f'R1_SHUF_i{a_i}_o{a_o}'] = FE.evaluate(d, *split_release(st, stats, a_i * c, a_o * c, qd=qd), splits)
        rows[f'R1_SHUF_i{a_i}_o{a_o}']['scale_factor'] = c
    st_d = dec_fit(stats, d['q'])
    for alpha in (0.5,):
        rows[f'R2_PTCDEC_a{alpha}'] = FE.evaluate(d, *dec_release(st, st_d, stats, alpha), splits)
        tgt = FE.tot_kl(rows[f'R2_PTCDEC_a{alpha}']['spend'])
        c = FE.match_scale(lambda x: FE.tot_kl(FE.spend(d, *dec_release(st, st_d, stats, x, qd=qd))), tgt)
        rows[f'R2_SHUFDEC_a{alpha}'] = FE.evaluate(d, *dec_release(st, st_d, stats, c, qd=qd), splits)
        rows[f'R2_SHUFDEC_a{alpha}']['scale'] = c
    res = {'schema': 'DDR-designer-B-revision-probe-v1', 'label_free': True, 'version': v, 'rows': rows,
           'cpu_seconds': time.process_time() - t0}
    for name, row in rows.items():
        A, S = row['auc'], row['spend']
        print(f'{name:22s} pair {A["pair"]:.4f} inc {A["income"]:.4f} occ {A["occupation"]:.4f} | KLinc {S["kl_inc"]:.4f} '
              f'KLocc {S["kl_occ"]:.4f} BrI {S["brier_inc"]:.4f} accI {S["acc_inc"]:.4f} accO {S["acc_occ"]:.4f} '
              f'menB {S["men_occ_brier"]:.4f} womB {S["women_occ_brier"]:.4f}')
    print(f'cpu_seconds {res["cpu_seconds"]:.1f}')
    Path(__file__).resolve().parent.joinpath('REVISION_PROBE.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

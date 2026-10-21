"""Designer B: LABEL-FREE learnability check for PTC (the RCCS 'obfuscation' objection, S6). Plug-in world as in feasibility.py
(cached MODEL_FIT out-of-fold statistics only; q plays the truth). Pair view only. For reader-FIT sizes 2k / 4k / 8k / 16k drawn
from 16,000 people (a fixed 4,000-person SCORE set), pseudo-SEX labels ~ Bern(q): HGB15 and LR on the rich features (max), plus a
data-rich DENOISED reader (HGB regression of q itself, 31 leaves, 400 iterations; a smoothed-Bayes proxy, not a real reader).
If PTC's advantage over SHUF / BASE shrinks as reader data grow, the advantage is learnability, not removal.

    python3 results/pcrl_dual_design_release_v1/roles/designer_b/learning_curve.py [--version 0] [--global-tilt]
                                                                                   [--releases BASE,PTC,SHUF,ZTILT]
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

import feasibility as FE
import mech_sketch as MS

SIZES = (2000, 4000, 8000, 16000)


def pair_feats(r_inc, r_occ):
    return np.column_stack([FE.rich(r_inc), FE.rich(r_occ)])


def curve(d, X, rng_seed, draws=2):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    n = len(d['q'])
    out = {str(m): [] for m in SIZES}
    out['denoised_16000'] = []
    for k in range(draws):
        rng = np.random.default_rng([rng_seed, k])
        perm = rng.permutation(n)
        score, pool = perm[:4000], perm[4000:]
        s_pool = (rng.random(len(pool)) < d['q'][pool]).astype(int)
        mu, sd = X[pool].mean(0), X[pool].std(0)
        sd[sd < 1e-12] = 1.0
        Xs = (X - mu) / sd
        for m in SIZES:
            fi, sf = pool[:m], s_pool[:m]
            hgb = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=30,
                                                 l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xs[fi], sf)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[fi], sf)
            out[str(m)].append(max(FE.auc_soft(hgb.predict_proba(Xs[score])[:, 1], d['q'][score]),
                                   FE.auc_soft(lr.predict_proba(Xs[score])[:, 1], d['q'][score])))
        rich = HistGradientBoostingRegressor(max_leaf_nodes=31, max_iter=400, learning_rate=0.05, min_samples_leaf=20,
                                             early_stopping=False, random_state=0).fit(Xs[pool], d['q'][pool])
        out['denoised_16000'].append(FE.auc_soft(rich.predict(Xs[score]), d['q'][score]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(argv=None):
    """--global-tilt reproduces LEARNING_CURVE_GLOBAL_TILT.json (the unstratified occupation tilt, run first); the default is the
    proposal (occupation tilt stratified by the twin's decision). BASE does not depend on the mechanism."""
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--global-tilt', action='store_true')
    ap.add_argument('--releases', default='BASE,PTC,SHUF,ZTILT')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    t0 = time.process_time()
    v, alpha, strat = a.version, a.alpha, not a.global_tilt
    out = a.out or str(here / ('LEARNING_CURVE_GLOBAL_TILT.json' if a.global_tilt else 'LEARNING_CURVE.json'))
    d = FE.load(v)
    stats = {'p_inc': d['p_inc'], 'p_occ': d['p_occ'], 'q': d['q']}
    st = MS.fit_from_stats(stats, sex_expect=d['q'], strength=alpha, stratify=strat)
    rel = {'BASE': (d['p_inc'], d['p_occ']), f'PTC_a{alpha}': MS.release_from_stats(st, stats)}
    target = FE.tot_kl(FE.spend(d, *rel[f'PTC_a{alpha}']))
    qd = MS.donor_q(d['q'], key=f'DDR_B_SHUF|V{v}')
    c = FE.match_scale(lambda x: FE.tot_kl(FE.spend(d, *MS.release_from_stats(st, stats, q_override=qd, scale=x))), target)
    rel[f'SHUF_a{alpha}'] = MS.release_from_stats(st, stats, q_override=qd, scale=c)
    if 'ZTILT' in a.releases:
        qz = FE.pseudo_propensity(d, np.column_stack([MS.clog(d['p_inc']), MS.clog(d['p_occ'])]), [2026, 10, 21, v, 77])
        stats_z = dict(stats, q=qz)
        stz = MS.fit_from_stats(stats_z, sex_expect=d['q'], strength=1.0, stratify=strat)
        cz = FE.match_scale(lambda x: FE.tot_kl(FE.spend(d, *MS.release_from_stats(stz, stats_z, scale=x))), target)
        rel[f'ZTILT_m{alpha}'] = MS.release_from_stats(stz, stats_z, scale=cz)
    keep = {r.strip() for r in a.releases.split(',')}
    rel = {k: r for k, r in rel.items() if k.split('_')[0] in keep}
    res = {'schema': 'DDR-designer-B-learning-curve-v2', 'label_free': True, 'version': v, 'alpha': alpha, 'sizes': list(SIZES),
           'occupation_tilt': 'stratified by the twin decision' if strat else 'global',
           'curves': {name: curve(d, pair_feats(*r), rng_seed=20261021 + v) for name, r in rel.items()}}
    res['cpu_seconds'] = time.process_time() - t0
    print(f'V{v} pair-view AUC by reader-FIT size (max of HGB15, LR) and the denoised data-rich reader')
    print(f'{"release":14s} ' + ' '.join(f'{m:>7d}' for m in SIZES) + '  denoised16k')
    for name, cv in res['curves'].items():
        print(f'{name:14s} ' + ' '.join(f'{cv[str(m)]:7.4f}' for m in SIZES) + f'  {cv["denoised_16000"]:.4f}')
    print(f'cpu_seconds {res["cpu_seconds"]:.1f}')
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

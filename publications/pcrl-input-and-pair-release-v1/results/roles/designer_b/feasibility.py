"""Designer B, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in feasibility of CSF (PROPOSAL.md, mech_sketch.py).

Reads ONLY what the design brief names:
  .private/pcrl_new_release_mechanisms_v1/fa_cache/V<v>/TASK.pkl          twin out-of-fold vectors p_inc (n,2), p_occ (n,5)
  .private/pcrl_finite_routing_comparison_v1/fa_cache/V<v>/SEXSIDE.pkl    q = P(SEX=1|x), occ_s = P(occ|x,SEX=s)
  ddr.common.features('MODEL_FIT')                                         label-free admitted features (for the six-field keys)
No label file, nothing of MODEL_SELECT, no reader cache. One numerical thread.

Plug-in world (a simulation, NOT evidence about the benchmark readers; the previous designer B's construction): the cached
propensity q plays the truth, SEX ~ Bern(q). Spend is measured against the twin posterior taken as the truth (KL = expected NLL
excess, summed Brier excess, plug-in accuracy loss, men's / women's occupation summed-Brier excess with occ_s[0] / occ_s[1] as the
within-SEX truth). Readers: HGB15 and LR on (p, log p, one-hot argmax), trained on 4,000 pseudo-labelled people, expected AUC on
the other 16,000 with q as soft labels; the larger of the two per view; averaged over draws.

    python3 results/pcrl_input_and_pair_release_v1/roles/designer_b/feasibility.py [--versions 0,1] [--draws 2]
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))
import mech_sketch as MS  # noqa: E402

TASK = ROOT / '.private' / 'pcrl_new_release_mechanisms_v1' / 'fa_cache' / 'V{v}' / 'TASK.pkl'
SEXSIDE = ROOT / '.private' / 'pcrl_finite_routing_comparison_v1' / 'fa_cache' / 'V{v}' / 'SEXSIDE.pkl'
N_FIT = 4000
BUDGET_S2 = {'kl_occ': 0.008, 'kl_inc': 0.004, 'brier_occ': 0.0030, 'brier_inc': 0.0030, 'men_occ_brier': 0.0030,
             'women_occ_brier': 0.0030, 'acc_occ': 0.006, 'acc_inc': 0.004}


def load(v, Z_cache={}):
    with open(str(TASK).format(v=v), 'rb') as f:
        T = pickle.load(f)
    with open(str(SEXSIDE).format(v=v), 'rb') as f:
        S = pickle.load(f)
    if T['row_id_sha256'] != S['row_id_sha256']:
        raise RuntimeError('TASK and SEXSIDE rows differ')
    if 'Z' not in Z_cache:
        from ddr import common as C
        from nrm.fa import prc
        from ohn import fam_a as FA
        F = C.features('MODEL_FIT')
        Z_cache['Z'] = FA.compact_po(F['X'], tuple(map(str, F['feature_names'])))
        Z_cache['sha'] = prc.array_sha(np.asarray(F['row_id']))
    if Z_cache['sha'] != T['row_id_sha256']:
        raise RuntimeError('feature rows differ from the cached teacher rows')
    return {'p_inc': MS.clipn(T['oof']['inc']), 'p_occ': MS.clipn(T['oof']['occ']), 'Z': Z_cache['Z'],
            'q': np.clip(np.asarray(S['oof']['sex'], np.float64)[:, 1], 0.01, 0.99),
            'a': np.asarray(S['oof']['occ_s'][1], np.float64), 'b': np.asarray(S['oof']['occ_s'][0], np.float64)}


def auc_soft(score, w):
    score, w = np.asarray(score, np.float64), np.asarray(w, np.float64)
    order = np.argsort(score, kind='mergesort')
    s, ww = score[order], w[order]
    m = 1.0 - ww
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    gm, gw = np.add.reduceat(m, starts), np.add.reduceat(ww, starts)
    below_m = np.r_[0.0, np.cumsum(gm)[:-1]]
    num = float(np.sum(gw * below_m) + 0.5 * np.sum(gw * gm) - 0.5 * np.sum(ww * m))
    return num / float(ww.sum() * m.sum() - np.sum(ww * m))


def spend(d, r_inc, r_occ):
    p_i, p_o, q, a, b = d['p_inc'], d['p_occ'], d['q'], d['a'], d['b']
    kl = lambda p, r: float(np.mean(np.sum(p * (np.log(p) - np.log(r)), 1)))
    acc = lambda p, r: float(np.mean(p.max(1) - p[np.arange(len(p)), r.argmax(1)]))

    def within(truth, w):
        ex = np.sum(r_occ ** 2, 1) - np.sum(p_o ** 2, 1) - 2.0 * np.sum((r_occ - p_o) * truth, 1)
        return float(np.sum(w * ex) / np.sum(w))
    out = {'kl_inc': kl(p_i, r_inc), 'kl_occ': kl(p_o, r_occ), 'acc_inc': acc(p_i, r_inc), 'acc_occ': acc(p_o, r_occ),
           'brier_inc': float(np.mean(np.sum((r_inc - p_i) ** 2, 1))), 'brier_occ': float(np.mean(np.sum((r_occ - p_o) ** 2, 1))),
           'men_occ_brier': within(b, 1.0 - q), 'women_occ_brier': within(a, q)}
    out['kl_total'] = out['kl_inc'] + out['kl_occ']
    out['s2_ratio'] = max(out[k] / x for k, x in BUDGET_S2.items())       # worst component / its S2 budget (plug-in, no SE)
    return out


def rich(p):
    return np.column_stack([p, np.log(p), np.eye(p.shape[1])[p.argmax(1)]])


def readers_auc(feats, q, fit_idx, score_idx, s_fit):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    out = {}
    for view, X in feats.items():
        mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0)
        sd[sd < 1e-12] = 1.0
        Xs = (X - mu) / sd
        hgb = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=30,
                                             l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xs[fit_idx], s_fit)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[fit_idx], s_fit)
        res = {'HGB15': auc_soft(hgb.predict_proba(Xs[score_idx])[:, 1], q[score_idx]),
               'LR': auc_soft(lr.predict_proba(Xs[score_idx])[:, 1], q[score_idx])}
        res['max'] = max(res.values())
        out[view] = res
    return out


def evaluate(d, r_inc, r_occ, splits, views=('income', 'occupation', 'pair')):
    fi, fo = rich(r_inc), rich(r_occ)
    allf = {'income': fi, 'occupation': fo, 'pair': np.column_stack([fi, fo])}
    feats = {k: allf[k] for k in views}
    per = [readers_auc(feats, d['q'], a, b, c) for a, b, c in splits]
    return {'auc': {v: float(np.mean([p[v]['max'] for p in per])) for v in feats},
            'auc_draws': {v: [float(p[v]['max']) for p in per] for v in feats},
            'spend': spend(d, r_inc, r_occ)}


def make_splits(n, q, draws, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        perm = rng.permutation(n)
        out.append((perm[:N_FIT], perm[N_FIT:], (rng.random(N_FIT) < q[perm[:N_FIT]]).astype(int)))
    return out


def bisect(fn, target, lo, hi, it=30):
    """x in [lo, hi] with the increasing function fn(x) = target."""
    if fn(hi) < target:
        return hi
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def run_version(v, draws):
    from ddr import ctlkit as CK
    d = load(v)
    n = len(d['q'])
    splits = make_splits(n, d['q'], draws, seed=[2026, 10, 22, v])
    Z, pI, pO = d['Z'], d['p_inc'], d['p_occ']
    rows = {'TWIN': evaluate(d, pI, pO, splits)}
    st = {m: MS.fit(Z, None, None, None, None, v, 0.0, twin_inc=pI, twin_occ=pO, mode=m) for m in MS.MODES}
    rel = lambda mode, pi: MS.release_from_twin(dict(st[mode], pi=pi), Z, pI, pO)
    for mode, name in (('coord', 'FLAT_BOTH_ALL'), ('occ_only', 'FLAT_OCC_ALL'), ('inc_only', 'FLAT_INC_ALL')):
        rows[name] = evaluate(d, *rel(mode, 1.0), splits)
    for pi in (0.15, 0.30):
        rows[f'CSF_{pi:.2f}'] = evaluate(d, *rel('coord', pi), splits)
        rows[f'IND_{pi:.2f}'] = evaluate(d, *rel('indep', pi), splits)
    basis = CK.dq_fit(pI, pO)
    tag = f'IPR_B_FEAS_DQ|V{v}'
    for pi in (0.15, 0.30):
        target = rows[f'CSF_{pi:.2f}']['spend']['kl_total']
        delta = bisect(lambda x: spend(d, *CK.dq_release(basis, pI, pO, Z, x, tag))['kl_total'], target, 0.0, 6.0)
        rows[f'DQ_m{pi:.2f}'] = evaluate(d, *CK.dq_release(basis, pI, pO, Z, delta, tag), splits)
        rows[f'DQ_m{pi:.2f}']['delta'] = delta
    # all of CSF_0.30's total spend on the occupation recipient alone (single-recipient design)
    target = rows['CSF_0.30']['spend']['kl_total']
    pio = bisect(lambda x: spend(d, *rel('occ_only', x))['kl_total'], target, 0.0, 1.0)
    rows['OCC_ONLY_m0.30'] = evaluate(d, *rel('occ_only', pio), splits)
    rows['OCC_ONLY_m0.30']['pi'] = pio
    grid = {}
    for pi in MS.STRENGTHS:                           # spend-only preview over the registered grid (no reader fits)
        grid[f'{pi:.2f}'] = spend(d, *rel('coord', pi))
    tables = {'T_inc': st['coord']['T_inc'].tolist(), 'T_occ': st['coord']['T_occ'].tolist(),
              'occ_decision_shares': np.bincount(pO.argmax(1), minlength=5).tolist(),
              'inc_decision_shares': np.bincount(pI.argmax(1), minlength=2).tolist()}
    sel = MS.selection(dict(st['coord'], pi=0.30), Z)[0]
    return {'version': v, 'n': n, 'auc_q_itself': auc_soft(d['q'], d['q']), 'qbar': float(d['q'].mean()),
            'distinct_rows_Z': int(len(np.unique(Z, axis=0))), 'csf030_selected_share': float(sel.mean()),
            'tables': tables, 'spend_grid_coord': grid, 'rows': rows}


def report(res):
    for v, r in res['versions'].items():
        print(f'--- V{v}  AUC of q itself {r["auc_q_itself"]:.4f}; distinct six-field rows {r["distinct_rows_Z"]}; '
              f'CSF 0.30 selected share {r["csf030_selected_share"]:.3f}')
        print(f'{"release":16s} {"pair":>7s} {"inc":>7s} {"occ":>7s} | {"KLinc":>7s} {"KLocc":>7s} {"BrI":>7s} {"BrO":>7s} '
              f'{"menB":>7s} {"womB":>7s} {"S2x":>5s}')
        for name, row in r['rows'].items():
            A, S = row['auc'], row['spend']
            print(f'{name:16s} {A["pair"]:7.4f} {A["income"]:7.4f} {A["occupation"]:7.4f} | {S["kl_inc"]:7.4f} {S["kl_occ"]:7.4f} '
                  f'{S["brier_inc"]:7.4f} {S["brier_occ"]:7.4f} {S["men_occ_brier"]:7.4f} {S["women_occ_brier"]:7.4f} '
                  f'{S["s2_ratio"]:5.2f}')
        print('coord spend grid (pi: KL inc/occ, men Brier, S2 ratio):',
              {k: (round(g['kl_inc'], 4), round(g['kl_occ'], 4), round(g['men_occ_brier'], 4), round(g['s2_ratio'], 2))
               for k, g in r['spend_grid_coord'].items()})
    print(f'cpu_seconds {res["cpu_seconds"]:.1f}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', default='0,1')
    ap.add_argument('--draws', type=int, default=2)
    ap.add_argument('--out', default=str(HERE / 'FEASIBILITY.json'))
    a = ap.parse_args(argv)
    t0 = time.process_time()
    res = {'schema': 'IPR-designer-B-feasibility-v1', 'label_free': True,
           'inputs': [str(TASK.relative_to(ROOT)), str(SEXSIDE.relative_to(ROOT)), "ddr.common.features('MODEL_FIT')"],
           'reader_fit_n': N_FIT, 'draws': a.draws, 'strengths': list(MS.STRENGTHS), 'versions': {}}
    for v in [int(x) for x in a.versions.split(',')]:
        res['versions'][str(v)] = run_version(v, a.draws)
        print(f'V{v} done, cpu so far {time.process_time() - t0:.1f}s', flush=True)
    res['cpu_seconds'] = time.process_time() - t0
    report(res)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

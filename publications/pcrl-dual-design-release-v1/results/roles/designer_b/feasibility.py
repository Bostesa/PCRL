"""Designer B, campaign pcrl_dual_design_release_v1: LABEL-FREE feasibility arithmetic for PTC (PROPOSAL.md, mech_sketch.py).

Reads ONLY the cached MODEL_FIT out-of-fold statistics named in the design brief:
  .private/pcrl_new_release_mechanisms_v1/fa_cache/V<v>/TASK.pkl          task posteriors p_inc (n,2), p_occ (n,5)
  .private/pcrl_finite_routing_comparison_v1/fa_cache/V<v>/SEXSIDE.pkl    q = P(SEX=1|x) (n,), occ_s = P(occ|x,SEX=s)
No label file, nothing of MODEL_SELECT, no reader cache, no raw features. One numerical thread.

Plug-in world (a simulation, NOT evidence about the benchmark readers): the cached propensity q plays the truth, SEX ~ Bern(q).
  spend    E KL(p || r) per task with p as the true posterior; plug-in accuracy loss E[max p - p[argmax r]]; men's / women's
           occupation summed-Brier excess with occ_s[0] / occ_s[1] as the within-SEX truth, weights (1 - q) / q; decision flips.
  readers  pseudo-SEX labels ~ Bern(q) on a 4,000-person reader-FIT sample (the benchmark's reader budget); HGB15 and LR on the
           benchmark's 'rich' features (p, log p, one-hot argmax); expected AUC on the other 16,000 people with q as soft labels;
           the larger of the two readers per view. Two split / label draws, averaged.
  EST      the mechanism uses a propensity RE-ESTIMATED from an independent pseudo-label draw (HGB on the task and SEX-
           conditional occupation posteriors, cross-fitted; no q input), while the truth stays q: a check on the optimism of
           letting the mechanism use the truth.
  ZTILT    the relabeling ablation: amplitude from P(SEX | twin's released posteriors) only (fitted on pseudo-labels).

    python3 results/pcrl_dual_design_release_v1/roles/designer_b/feasibility.py [--versions 0,1] [--draws 2]
            [--record final | global-v2]       final -> FEASIBILITY.json (the proposal: stratified occupation tilt);
                                               global-v2 -> FEASIBILITY_GLOBAL_TILT.json (the earlier unstratified record)
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
import mech_sketch as MS  # noqa: E402  (the reference implementation under test)

TASK = ROOT / '.private' / 'pcrl_new_release_mechanisms_v1' / 'fa_cache' / 'V{v}' / 'TASK.pkl'
SEXSIDE = ROOT / '.private' / 'pcrl_finite_routing_comparison_v1' / 'fa_cache' / 'V{v}' / 'SEXSIDE.pkl'
N_FIT = 4000
BUDGETS = {  # the proposal's spend budgets over the twin (nrm.fa.prc.BUDGETS plus summed-Brier components); plug-in preview
    #            only: the registered rule uses the MODEL_FIT nested household estimate + 2 SE with labels
    'S2': {'kl_occ': 0.008, 'kl_inc': 0.004, 'brier_occ': 0.0030, 'brier_inc': 0.0030, 'men_occ_brier': 0.0030,
           'women_occ_brier': 0.0030, 'acc_occ': 0.006, 'acc_inc': 0.004},
    'S3': {'kl_occ': 0.012, 'kl_inc': 0.006, 'brier_occ': 0.0045, 'brier_inc': 0.0045, 'men_occ_brier': 0.0045,
           'women_occ_brier': 0.0045, 'acc_occ': 0.008, 'acc_inc': 0.006}}


def load(v):
    with open(str(TASK).format(v=v), 'rb') as f:
        T = pickle.load(f)
    with open(str(SEXSIDE).format(v=v), 'rb') as f:
        S = pickle.load(f)
    if T['row_id_sha256'] != S['row_id_sha256']:
        raise RuntimeError('TASK and SEXSIDE rows differ')
    return {'p_inc': np.asarray(T['oof']['inc'], np.float64), 'p_occ': np.asarray(T['oof']['occ'], np.float64),
            'q': np.clip(np.asarray(S['oof']['sex'], np.float64)[:, 1], 0.01, 0.99),
            'a': np.asarray(S['oof']['occ_s'][1], np.float64), 'b': np.asarray(S['oof']['occ_s'][0], np.float64)}


def auc_soft(score, w):
    """Expected AUC when person i is a woman with probability w_i (ties count one half; self-pairs removed)."""
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
    kl = lambda p, r: float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(r, 1e-300))), 1)))
    acc = lambda p, r: float(np.mean(p.max(1) - p[np.arange(len(p)), r.argmax(1)]))

    def within(truth, w):
        ex = np.sum(r_occ ** 2, 1) - np.sum(p_o ** 2, 1) - 2.0 * np.sum((r_occ - p_o) * truth, 1)
        return float(np.sum(w * ex) / np.sum(w))
    return {'kl_inc': kl(p_i, r_inc), 'kl_occ': kl(p_o, r_occ), 'acc_inc': acc(p_i, r_inc), 'acc_occ': acc(p_o, r_occ),
            'brier_inc': float(np.mean(np.sum((r_inc - p_i) ** 2, 1))), 'brier_occ': float(np.mean(np.sum((r_occ - p_o) ** 2, 1))),
            'men_occ_brier': within(b, 1.0 - q), 'women_occ_brier': within(a, q),
            'flip_inc': float(np.mean(p_i.argmax(1) != r_inc.argmax(1))), 'flip_occ': float(np.mean(p_o.argmax(1) != r_occ.argmax(1)))}


def rich(p):
    return np.column_stack([p, np.log(np.maximum(p, 1e-300)), np.eye(p.shape[1])[p.argmax(1)]])


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


def evaluate(d, r_inc, r_occ, splits):
    fi, fo = rich(r_inc), rich(r_occ)
    feats = {'income': fi, 'occupation': fo, 'pair': np.column_stack([fi, fo])}
    per = [readers_auc(feats, d['q'], a, b, c) for a, b, c in splits]
    return {'auc': {v: float(np.mean([p[v]['max'] for p in per])) for v in feats},
            'auc_hgb': {v: float(np.mean([p[v]['HGB15'] for p in per])) for v in feats},
            'spend': spend(d, r_inc, r_occ)}


def group_lr(d, r_inc, r_occ, splits):
    """Pair-view LR on single feature groups: where the remaining linear signal sits."""
    from sklearn.linear_model import LogisticRegression
    groups = {'logratio': np.column_stack([MS.clog(r_inc), MS.clog(r_occ)]), 'probs': np.column_stack([r_inc, r_occ]),
              'onehot': np.column_stack([np.eye(2)[r_inc.argmax(1)], np.eye(5)[r_occ.argmax(1)]])}
    out = {}
    for g, X in groups.items():
        vals = []
        for fi, si, sf in splits:
            mu, sd = X[fi].mean(0), X[fi].std(0)
            sd[sd < 1e-12] = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = LogisticRegression(C=1.0, max_iter=2000).fit((X[fi] - mu) / sd, sf)
            vals.append(auc_soft(m.predict_proba((X[si] - mu) / sd)[:, 1], d['q'][si]))
        out[g] = float(np.mean(vals))
    return out


def make_splits(n, q, draws, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        perm = rng.permutation(n)
        out.append((perm[:N_FIT], perm[N_FIT:], (rng.random(N_FIT) < q[perm[:N_FIT]]).astype(int)))
    return out


def match_scale(fn, target, lo=0.0, hi=4.0, it=40):
    """Scale c in [lo, hi] with total plug-in KL fn(c) = target (monotone bisection)."""
    if fn(hi) < target:
        return hi
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
    return hi


def pseudo_propensity(d, X, seed):
    """Cross-fitted HGB propensity from an independent pseudo-SEX draw ~ Bern(q) on features X (no q input)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    n = len(d['q'])
    s2 = (rng.random(n) < d['q']).astype(int)
    fold = rng.permutation(n) % 5
    out = np.zeros(n)
    for f in range(5):
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, min_samples_leaf=50,
                                           early_stopping=False, random_state=0).fit(X[fold != f], s2[fold != f])
        out[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    return np.clip(out, 0.01, 0.99)


def tot_kl(sp):
    return sp['kl_inc'] + sp['kl_occ']


def candidate_and_shuf(d, stats, alpha, splits, v, tag, rows, stratify=True, shuf=True):
    st = MS.fit_from_stats(stats, sex_expect=d['q'], strength=alpha, stratify=stratify)
    rows[f'PTC{tag}_a{alpha}'] = row = evaluate(d, *MS.release_from_stats(st, stats), splits)
    row['weights'] = {'inc': st['w_inc'].tolist(), 'occ': st['W_occ'].tolist(), 'qbar_occ': st['qbar_occ'].tolist(),
                      'fallback_strata': st['fallback_strata'], 'stratum_sizes': st['stratum_sizes']}
    if shuf:
        qd = MS.donor_q(stats['q'], key=f'DDR_B_SHUF|V{v}')
        c = match_scale(lambda x: tot_kl(spend(d, *MS.release_from_stats(st, stats, q_override=qd, scale=x))),
                        tot_kl(row['spend']))
        rows[f'SHUF{tag}_a{alpha}'] = evaluate(d, *MS.release_from_stats(st, stats, q_override=qd, scale=c), splits)
        rows[f'SHUF{tag}_a{alpha}']['scale'] = c
    return st


def run_version_global_v2(v, draws):
    """The earlier record FEASIBILITY_GLOBAL_TILT.json (schema v2): the GLOBAL occupation tilt over the whole strength grid, with
    decisions-only and MMP references (stratify=False reproduces the arithmetic of the code that produced it)."""
    d = load(v)
    n = len(d['q'])
    splits = make_splits(n, d['q'], draws, seed=[2026, 10, 21, v])
    rows = {'BASE': evaluate(d, d['p_inc'], d['p_occ'], splits)}
    soft1 = lambda p: np.eye(p.shape[1])[p.argmax(1)] * 0.98 + 0.02 / p.shape[1]
    rows['DECISIONS_ONLY'] = evaluate(d, soft1(d['p_inc']), soft1(d['p_occ']), splits)
    qbar = float(d['q'].mean())
    rows['MMP_OCC_REF'] = evaluate(d, d['p_inc'], qbar * d['a'] + (1 - qbar) * d['b'], splits)
    stats = {'p_inc': d['p_inc'], 'p_occ': d['p_occ'], 'q': d['q']}
    for alpha in MS.STRENGTHS:
        candidate_and_shuf(d, stats, alpha, splits, v, '', rows, stratify=False)
    groups = {'BASE': group_lr(d, d['p_inc'], d['p_occ'], splits)}
    st5 = MS.fit_from_stats(stats, sex_expect=d['q'], strength=0.5, stratify=False)
    groups['PTC_a0.5'] = group_lr(d, *MS.release_from_stats(st5, stats), splits)
    qe = pseudo_propensity(d, np.column_stack([MS.clog(d['p_inc']), MS.clog(d['p_occ']), d['a'], d['b']]), [2026, 10, 21, v, 99])
    for alpha in (0.5, 0.65):
        candidate_and_shuf(d, dict(stats, q=qe), alpha, splits, v, '_EST', rows, stratify=False)
    qz = pseudo_propensity(d, np.column_stack([MS.clog(d['p_inc']), MS.clog(d['p_occ'])]), [2026, 10, 21, v, 77])
    stats_z = dict(stats, q=qz)
    stz = MS.fit_from_stats(stats_z, sex_expect=d['q'], strength=1.0, stratify=False)
    for alpha in (0.5, 0.65):
        c = match_scale(lambda x: tot_kl(spend(d, *MS.release_from_stats(stz, stats_z, scale=x))),
                        tot_kl(rows[f'PTC_a{alpha}']['spend']))
        rows[f'ZTILT_m{alpha}'] = evaluate(d, *MS.release_from_stats(stz, stats_z, scale=c), splits)
        rows[f'ZTILT_m{alpha}']['scale'] = c
    return {'version': v, 'n': n, 'auc_q_itself': auc_soft(d['q'], d['q']), 'qbar': qbar,
            'est_quality': {'corr_q': float(np.corrcoef(qe, d['q'])[0, 1]), 'auc_soft': auc_soft(qe, d['q'])},
            'ztilt_quality': {'corr_q': float(np.corrcoef(qz, d['q'])[0, 1]), 'auc_soft': auc_soft(qz, d['q'])},
            'groups_pair_lr': groups, 'plugin_selection_preview_subset': {}, 'rows': rows}


FINAL_STRENGTHS = (0.35, 0.5, 0.65)


def run_version(v, draws):
    """Final label-free record of the proposal: stratified PTC on a subset of the strength grid with spend-matched SHUF, the
    global-occupation ablation, the estimated-propensity check and the relabeling ablation (ZTILT)."""
    d = load(v)
    n = len(d['q'])
    splits = make_splits(n, d['q'], draws, seed=[2026, 10, 21, v])
    rows = {'BASE': evaluate(d, d['p_inc'], d['p_occ'], splits)}
    stats = {'p_inc': d['p_inc'], 'p_occ': d['p_occ'], 'q': d['q']}
    for alpha in FINAL_STRENGTHS:
        candidate_and_shuf(d, stats, alpha, splits, v, '', rows)
    candidate_and_shuf(d, stats, 0.5, splits, v, '_GLOBAL', rows, stratify=False, shuf=False)
    # optimism check: the mechanism's propensity re-estimated from an independent pseudo-label draw (truth stays q)
    qe = pseudo_propensity(d, np.column_stack([MS.clog(d['p_inc']), MS.clog(d['p_occ']), d['a'], d['b']]), [2026, 10, 21, v, 99])
    candidate_and_shuf(d, dict(stats, q=qe), 0.5, splits, v, '_EST', rows)
    # relabeling ablation: propensity from the twin's posteriors only, its own weights, spend matched to PTC at alpha = 0.5
    qz = pseudo_propensity(d, np.column_stack([MS.clog(d['p_inc']), MS.clog(d['p_occ'])]), [2026, 10, 21, v, 77])
    stats_z = dict(stats, q=qz)
    stz = MS.fit_from_stats(stats_z, sex_expect=d['q'], strength=1.0)
    c = match_scale(lambda x: tot_kl(spend(d, *MS.release_from_stats(stz, stats_z, scale=x))), tot_kl(rows['PTC_a0.5']['spend']))
    rows['ZTILT_m0.5'] = evaluate(d, *MS.release_from_stats(stz, stats_z, scale=c), splits)
    rows['ZTILT_m0.5']['scale'] = c
    preview = {}
    for b, lim in BUDGETS.items():
        ok = [a for a in FINAL_STRENGTHS if all(rows[f'PTC_a{a}']['spend'][k] <= x for k, x in lim.items())]
        preview[b] = max(ok) if ok else None
    return {'version': v, 'n': n, 'auc_q_itself': auc_soft(d['q'], d['q']), 'qbar': float(d['q'].mean()),
            'est_quality': {'corr_q': float(np.corrcoef(qe, d['q'])[0, 1]), 'auc_soft': auc_soft(qe, d['q'])},
            'ztilt_quality': {'corr_q': float(np.corrcoef(qz, d['q'])[0, 1]), 'auc_soft': auc_soft(qz, d['q'])},
            'groups_pair_lr': {'PTC_a0.5': group_lr(d, *MS.release_from_stats(MS.fit_from_stats(stats, sex_expect=d['q'],
                                                                                                  strength=0.5), stats), splits)},
            'plugin_selection_preview_subset': preview, 'rows': rows}


def report(res):
    for v, r in res['versions'].items():
        print(f'--- V{v}  plug-in AUC of q itself {r["auc_q_itself"]:.4f}; EST propensity corr {r["est_quality"]["corr_q"]:.3f} '
              f'AUC {r["est_quality"]["auc_soft"]:.4f}; ZTILT propensity corr {r["ztilt_quality"]["corr_q"]:.3f} '
              f'AUC {r["ztilt_quality"]["auc_soft"]:.4f}')
        print(f'{"release":16s} {"pair":>7s} {"inc":>7s} {"occ":>7s} {"pairHGB":>7s} | {"KLinc":>7s} {"KLocc":>7s} {"accI":>6s} '
              f'{"accO":>6s} {"menB":>7s} {"womB":>7s} {"flipI":>6s} {"flipO":>6s}')
        for name, row in r['rows'].items():
            A, S = row['auc'], row['spend']
            print(f'{name:16s} {A["pair"]:7.4f} {A["income"]:7.4f} {A["occupation"]:7.4f} {row["auc_hgb"]["pair"]:7.4f} | '
                  f'{S["kl_inc"]:7.4f} {S["kl_occ"]:7.4f} {S["acc_inc"]:6.4f} {S["acc_occ"]:6.4f} {S["men_occ_brier"]:7.4f} '
                  f'{S["women_occ_brier"]:7.4f} {S["flip_inc"]:6.4f} {S["flip_occ"]:6.4f}')
        print('pair-LR by feature group:', json.dumps(r['groups_pair_lr']))
        print('plug-in selection preview on the evaluated subset (no SE):', r['plugin_selection_preview_subset'])
    print(f'cpu_seconds {res["cpu_seconds"]:.1f}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', default='0,1')
    ap.add_argument('--draws', type=int, default=2)
    ap.add_argument('--record', choices=('final', 'global-v2'), default='final')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    t0 = time.process_time()
    out = a.out or str(HERE / ('FEASIBILITY.json' if a.record == 'final' else 'FEASIBILITY_GLOBAL_TILT.json'))
    fn = run_version if a.record == 'final' else run_version_global_v2
    res = {'schema': 'DDR-designer-B-feasibility-v3' if a.record == 'final' else 'DDR-designer-B-feasibility-v2',
           'label_free': True, 'record': a.record,
           'inputs': [str(TASK.relative_to(ROOT)), str(SEXSIDE.relative_to(ROOT))], 'reader_fit_n': N_FIT, 'draws': a.draws,
           'strengths': list(MS.STRENGTHS), 'versions': {}}
    for v in [int(x) for x in a.versions.split(',')]:
        res['versions'][str(v)] = fn(v, a.draws)
        print(f'V{v} done, cpu so far {time.process_time() - t0:.1f}s', flush=True)
    res['cpu_seconds'] = time.process_time() - t0
    report(res)
    Path(out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

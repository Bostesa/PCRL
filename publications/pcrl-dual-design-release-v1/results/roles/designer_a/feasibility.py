"""Designer A (campaign pcrl_dual_design_release_v1): label-free feasibility of SEX-standardized task outputs (STD).

Inputs: ONLY the cached MODEL_FIT out-of-fold predictions named in the design brief
  TASK.pkl    (nrm fa_cache)  oof['inc'] (n,2), oof['occ'] (n,5)                       task-only HGB teachers (A1_PO_16 settings)
  SEXSIDE.pkl (frc fa_cache)  oof['sex'] (n,2) = P(SEX|x), oof['occ_s'][s] = P(occ|x, SEX=s)   SEX-aware HGB (same settings)
No label file, nothing of MODEL_SELECT, no reader cache is opened. SEX labels are never read: privacy numbers are PLUG-IN numbers
in which a pseudo-SEX is drawn from the cached propensity pi(x) (the 'plug-in world'); utility numbers are plug-in expectations
under two stated 'truths' (T1: the task-only posterior qO is correct; T2: the SEX-aware composite sum_s pi_s q_s is correct).

Releases of the occupation recipient compared (income recipient = the task-only twin throughout; no P(income | x, SEX) is cached):
  TWIN        qO                                              (task-only, lambda = 0)
  STD_l       (1 - l) qO + l * r1,  r1 = (1 - w) q0 + w q1     (w = mean pi, the fitted SEX share; l in 0.25 .. 1)
  STDi_l      (1 - om) q0 + om q1,  om = (1 - l) pi + l w      (propensity-shrinkage form; l = 0 is the composite)
  FLAT_a      (1 - a) qO + a pbar                              (SEX-blind flattening toward the fitted class shares; a matched)
  TEMP_T      softmax(log qO / T)                              (SEX-blind temperature; T matched)
  DONOR_l     (1 - l) qO + l * ((1 - pi_d) q0 + pi_d q1)        (targeting control: a keyed donor's propensity; l matched)
Spend matching: the SEX-blind control's strength is set by bisection so that its plug-in T1 NLL spend equals the STD point's.

Readers (plug-in): pseudo-SEX ~ Bernoulli(pi) on a FIT block of 4,000 people (about the reader-FIT size of the campaign grid);
LOG_LR (logistic regression on log-probabilities) and HGB15 (HistGradientBoosting, 15 leaves) analogues; scored on the other
16,000 people with the SOFT AUC against pi (the expected AUC over pseudo-SEX draws, no Monte Carlo noise in scoring).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = '.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v}/TASK.pkl'
SIDE = '.private/pcrl_finite_routing_comparison_v1/fa_cache/V{v}/SEXSIDE.pkl'
EPS = 1e-6
N_FIT = 4000


def load(v):
    with open(ROOT / TASK.format(v=v), 'rb') as f:
        t = pickle.load(f)
    with open(ROOT / SIDE.format(v=v), 'rb') as f:
        s = pickle.load(f)
    if t['row_id_sha256'] != s['row_id_sha256']:
        raise SystemExit('row order differs between the two caches')
    clipn = lambda P: (lambda Q: Q / Q.sum(1, keepdims=True))(np.clip(np.asarray(P, float), EPS, 1.0))
    return {'qI': clipn(t['oof']['inc']), 'qO': clipn(t['oof']['occ']), 'pi': np.clip(s['oof']['sex'][:, 1], 0.01, 0.99),
            'q0': clipn(s['oof']['occ_s'][0]), 'q1': clipn(s['oof']['occ_s'][1])}


def norm(P):
    P = np.clip(P, EPS, 1.0)
    return P / P.sum(1, keepdims=True)


def ce(T, R):
    return -(T * np.log(R)).sum(1)


def util(r, ref, t):
    """Plug-in expected excess of release r over the reference release ref when t is the true conditional."""
    return {'nll': float(np.mean(ce(t, r) - ce(t, ref))),
            'brier': float(np.mean(((r - t) ** 2).sum(1) - ((ref - t) ** 2).sum(1))),
            'acc_loss': float(np.mean(t[np.arange(len(t)), ref.argmax(1)] - t[np.arange(len(t)), r.argmax(1)])),
            'flips': float(np.mean(r.argmax(1) != ref.argmax(1)))}


def guards(r, ref, d):
    """Plug-in within-SEX occupation Brier excess over ref (SEX-aware model taken as the within-SEX truth)."""
    pi, q0, q1 = d['pi'], d['q0'], d['q1']
    men = np.sum((1 - pi) * (((r - q0) ** 2).sum(1) - ((ref - q0) ** 2).sum(1))) / np.sum(1 - pi)
    wom = np.sum(pi * (((r - q1) ** 2).sum(1) - ((ref - q1) ** 2).sum(1))) / np.sum(pi)
    return {'men_brier': float(men), 'women_brier': float(wom)}


def soft_auc(score, p):
    """Expected AUC of `score` for S ~ Bernoulli(p) independently per person (ties count one half; self-pairs excluded)."""
    o = np.argsort(score, kind='mergesort')
    s, a = score[o], p[o]
    b = 1.0 - a
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    A, B = np.add.reduceat(a, start), np.add.reduceat(b, start)
    AB = np.add.reduceat(a * b, start)
    below = np.r_[0.0, np.cumsum(B)[:-1]]
    num = np.sum(A * below) + 0.5 * np.sum(A * B - AB)
    den = a.sum() * b.sum() - np.sum(a * b)
    return float(num / den)


def feats(view):
    return np.column_stack([np.log(x) for x in view])


def reader_auc(F, pi, blocks, ys, kinds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    out = {}
    for kind in kinds:
        vals = []
        for b, (idx, y) in enumerate(zip(blocks, ys)):
            if kind == 'HGB15' and b >= 3:
                break
            sc = np.ones(len(pi), bool)
            sc[idx] = False
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if kind == 'LOG_LR':
                    mu, sd = F[idx].mean(0), F[idx].std(0) + 1e-9
                    m = LogisticRegression(C=1.0, max_iter=500).fit((F[idx] - mu) / sd, y)
                    z = m.decision_function((F[sc] - mu) / sd)
                else:
                    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                                       l2_regularization=1.0, early_stopping=False, random_state=b).fit(F[idx], y)
                    z = m.predict_proba(F[sc])[:, 1]
            vals.append(soft_auc(z, pi[sc]))
        out[kind] = float(np.mean(vals))
    return out


def match(fn, target, lo, hi, iters=40):
    """Bisection for the strength at which the spend fn(strength) equals target (fn increasing)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def donor_perm(n, v):
    """Keyed pseudo-random donor of every row (no information about the row's SEX); deterministic."""
    keys = [hashlib.sha256(f'DDR_A_donor_V{v}|{i}'.encode()).hexdigest() for i in range(n)]
    return np.argsort(keys)


def run(v, full=True):
    d = load(v)
    n = len(d['pi'])
    pi, qO, qI, q0, q1 = d['pi'], d['qO'], d['qI'], d['q0'], d['q1']
    w = float(pi.mean())
    pbar = qO.mean(0)
    comp = norm((1 - pi)[:, None] * q0 + pi[:, None] * q1)
    r1 = norm((1 - w) * q0 + w * q1)
    Delta = q1 - q0
    rel = {}
    rel['TWIN'] = qO
    rel['COMP'] = comp
    lams = (0.25, 0.5, 0.75, 1.0) if full else (0.5, 1.0)
    for l in lams:
        rel[f'STD_{l}'] = norm((1 - l) * qO + l * r1)
    if full:
        om = (1 - 0.5) * pi + 0.5 * w
        rel['STDi_0.5'] = norm((1 - om)[:, None] * q0 + om[:, None] * q1)
    spend = lambda r: float(np.mean(ce(qO, r) - ce(qO, qO)))
    # SEX-blind controls matched to each STD point's plug-in T1 NLL spend
    ctl = {}
    for l in lams:
        tgt = spend(rel[f'STD_{l}'])
        a = match(lambda a: spend(norm((1 - a) * qO + a * pbar)), tgt, 0.0, 1.0)
        rel[f'FLAT@{l}'] = norm((1 - a) * qO + a * pbar)
        ctl[f'FLAT@{l}'] = {'alpha': a}
        if l in (0.5, 1.0):
            T = match(lambda T: spend(norm(np.exp(np.log(qO) / T))), tgt, 1.0, 10.0)
            rel[f'TEMP@{l}'] = norm(np.exp(np.log(qO) / T))
            ctl[f'TEMP@{l}'] = {'T': T}
            pd = pi[donor_perm(n, v)]
            rd = norm((1 - pd)[:, None] * q0 + pd[:, None] * q1)
            ld = match(lambda x: spend(norm((1 - x) * qO + x * rd)), tgt, 0.0, 1.0)
            rel[f'DONOR@{l}'] = norm((1 - ld) * qO + ld * rd)
            ctl[f'DONOR@{l}'] = {'lambda': ld}
    # decomposition diagnostics (Fisher metric at qO)
    fis = lambda D: (D ** 2 / qO).sum(1)
    diag = {'w_fitted_sex_share': w, 'pi_sd': float(pi.std()),
            'KL_qO_comp': float(np.mean((qO * np.log(qO / comp)).sum(1))),
            'I_occ_sex_given_x_plugin': float(np.mean((1 - pi) * (q0 * np.log(q0 / comp)).sum(1) + pi * (q1 * np.log(q1 / comp)).sum(1))),
            'fisher_share_propensity_term': float(np.mean((pi - w) ** 2 * fis(Delta)) / np.mean(fis(qO - pbar))),
            'mean_abs_Delta_by_class': np.abs(Delta).mean(0).round(4).tolist(),
            'mean_Delta_by_class(women-men)': Delta.mean(0).round(4).tolist(),
            'class_shares_pbar': pbar.round(4).tolist()}
    # utility
    U = {}
    for k, r in rel.items():
        U[k] = {'T1': util(r, qO, qO), 'T2': util(r, qO, comp), 'guards': guards(r, qO, d)}
    # plug-in readers: identical FIT blocks and pseudo-SEX draws for every release (paired)
    rng = np.random.default_rng(20261021 + v)
    perm = rng.permutation(n)
    blocks = [perm[i * N_FIT:(i + 1) * N_FIT] for i in range(5)]
    ys = [(rng.random(N_FIT) < pi[idx]).astype(int) for idx in blocks]
    R = {'BAYES_pi': soft_auc(pi, pi), 'income_view_twin': reader_auc(feats([qI]), pi, blocks, ys, ('LOG_LR', 'HGB15'))}
    for k, r in rel.items():
        R[k] = {'occ': reader_auc(feats([r]), pi, blocks, ys, ('LOG_LR', 'HGB15')),
                'pair': reader_auc(feats([qI, r]), pi, blocks, ys, ('LOG_LR', 'HGB15'))}
    return {'version': v, 'diag': diag, 'controls': ctl, 'utility': U, 'readers': R}


def rich_readers(v):
    """Data-rich check: HGB15 fitted on 16,000 pseudo-labelled people, scored on the remaining 4,000 (TWIN, STD_1.0, FLAT@1.0)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = load(v)
    pi, qO, qI, q0, q1 = d['pi'], d['qO'], d['qI'], d['q0'], d['q1']
    w = float(pi.mean())
    pbar = qO.mean(0)
    r1 = norm((1 - w) * q0 + w * q1)
    spend = lambda r: float(np.mean(ce(qO, r) - ce(qO, qO)))
    a = match(lambda a: spend(norm((1 - a) * qO + a * pbar)), spend(r1), 0.0, 1.0)
    rel = {'TWIN': qO, 'STD_1.0': r1, 'FLAT@1.0': norm((1 - a) * qO + a * pbar)}
    rng = np.random.default_rng(777 + v)
    perm = rng.permutation(len(pi))
    fit, sc = perm[:16000], perm[16000:]
    y = (rng.random(16000) < pi[fit]).astype(int)
    out = {}
    for k, r in rel.items():
        out[k] = {}
        for view, F in (('occ', feats([r])), ('pair', feats([qI, r]))):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                                   l2_regularization=1.0, early_stopping=False, random_state=0).fit(F[fit], y)
            out[k][view] = soft_auc(m.predict_proba(F[sc])[:, 1], pi[sc])
    return out


if __name__ == '__main__':
    t0 = time.process_time()
    res = {'cpu_note': 'process_time of this script', 'runs': [run(0, full=True), run(1, full=False)]}
    t1 = time.process_time()
    if '--rich' in sys.argv:
        res['rich_v0'] = rich_readers(0)
    res['cpu_s'] = round(time.process_time() - t0, 2)
    res['cpu_s_main'] = round(t1 - t0, 2)
    (OUT / 'FEASIBILITY.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:200])
    print('CPU_SECONDS', res['cpu_s'])

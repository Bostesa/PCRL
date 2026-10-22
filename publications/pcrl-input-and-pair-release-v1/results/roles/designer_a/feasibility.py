"""Designer A, campaign pcrl_input_and_pair_release_v1: LABEL-FREE plug-in feasibility of shared field generalization (SFG).

Inputs: admitted MODEL_FIT features (ddr.common.features) and the cached out-of-fold predictions named in the design brief:
TASK.pkl (eta_inc, eta_occ: task posteriors) and SEXSIDE.pkl (pi = P(SEX = 1 | x), fitted earlier with MODEL_FIT SEX labels).
No label file, nothing of MODEL_SELECT, no reader cache is opened.

Plug-in world (pi and eta treated as the truth):
  * a generalization h maps the six fields to coarser levels; both task models see only h(x);
  * r_h = E[eta | h(X)] is estimated by household 2-fold cross-fitted HGB regressions of the cached eta on h(Z)
    (the identity h gives r_id, the smoother's own baseline);
  * task spend of h = mean KL(eta || r_h) - mean KL(eta || r_id) per task (nats); plug-in Brier and accuracy changes likewise;
    men's / women's occupation summed-Brier excess weighted by 1 - pi / pi;
  * SEX leakage = soft-label AUC (labels S ~ Bernoulli(pi)) of the cross-fitted regression E[pi | R]:
      R = h(Z) (BOUND: the data-processing ceiling for every reader of the pair), and R = the released logits (PAIR, INC, OCC);
  * SEX information removed by h: increase of the cross-fitted plug-in SEX log-loss of E[pi | h(Z)].
Ladders: SEX-aware (order steps by SEX information removed per nat of task spend) vs SEX-blind (cheapest task spend first).
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import sys, time, pickle, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, ROOT)
T0 = time.process_time()
from sklearn.ensemble import HistGradientBoostingRegressor
from ddr import common as C
from ohn import fam_a as FA
from nrm.fa import prc as PRC

F = C.features('MODEL_FIT')
NAMES = tuple(map(str, F['feature_names']))
Z = FA.compact_po(F['X'], NAMES)
UNIT = np.asarray(F['unit'])
FOLD = PRC.household_folds(UNIT, 2, 'IPR_DA_feas')
N = len(Z)


def load(v):
    with open(os.path.join(ROOT, f'.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v}/TASK.pkl'), 'rb') as f:
        T = pickle.load(f)
    with open(os.path.join(ROOT, f'.private/pcrl_finite_routing_comparison_v1/fa_cache/V{v}/SEXSIDE.pkl'), 'rb') as f:
        S = pickle.load(f)
    return np.clip(T['oof']['inc'], 1e-6, 1), np.clip(T['oof']['occ'], 1e-6, 1), np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)


# ------------------------------------------------------------------ the generalization steps (label-free field semantics)
age_lv = np.unique(Z[:, 0]); AGE_IDX = np.searchsorted(age_lv, Z[:, 0])          # 40 levels (ages 25..64 by rank)
wk_lv, wk_ct = np.unique(Z[:, 1], return_counts=True)
z40 = wk_lv[np.argmax(wk_ct)]; s_wk = 39.0 / (z40 - wk_lv.min()); HOURS = np.round(1 + (Z[:, 1] - wk_lv.min()) * s_wk)


def coarsen(Zin, steps):
    Zh = Zin.copy()
    for st in steps:
        if st == 'AGE5':
            Zh[:, 0] = AGE_IDX // 5
        elif st == 'AGE10':
            Zh[:, 0] = AGE_IDX // 10
        elif st == 'WK6':
            Zh[:, 1] = np.digitize(HOURS, [20, 35, 40, 41, 50])
        elif st == 'WK3':
            Zh[:, 1] = np.digitize(HOURS, [35, 45])
        elif st == 'SCH5':
            Zh[:, 3] = np.digitize(Zin[:, 3], [16, 18, 21, 22])
        elif st == 'SCH3':
            Zh[:, 3] = np.digitize(Zin[:, 3], [18, 21])
        elif st == 'COW5':                                  # for-profit | nonprofit | government | self-employed | unpaid
            Zh[:, 2] = np.array([0, 1, 2, 2, 2, 3, 3, 4])[Zin[:, 2].astype(int)]
        elif st == 'COW3':                                  # private | government | self-employed
            Zh[:, 2] = np.array([0, 0, 1, 1, 1, 2, 2, 0])[Zin[:, 2].astype(int)]
        elif st == 'MAR3':                                  # married | previously married | never married
            Zh[:, 4] = np.array([0, 1, 1, 1, 2])[Zin[:, 4].astype(int)]
        elif st == 'MAR1':                                  # suppressed
            Zh[:, 4] = 0
        elif st == 'POB4':                                  # Florida | other US | territories | foreign or other
            p = Zin[:, 5]
            Zh[:, 5] = np.where(p == 12, 0, np.where(p < 60, 1, np.where(p < 100, 2, 3)))
        elif st == 'POB1':                                  # suppressed
            Zh[:, 5] = 0
        else:
            raise KeyError(st)
    return Zh


FIELD_OF = {'AGE5': 'AGEP', 'AGE10': 'AGEP', 'WK6': 'WKHP', 'WK3': 'WKHP', 'SCH5': 'SCHL', 'SCH3': 'SCHL', 'COW5': 'COW',
            'COW3': 'COW', 'MAR3': 'MAR', 'MAR1': 'MAR', 'POB4': 'POBP', 'POB1': 'POBP'}
STRONG_OF = {'AGE5': 'AGE10', 'WK6': 'WK3', 'SCH5': 'SCH3', 'COW5': 'COW3', 'MAR3': 'MAR1', 'POB4': 'POB1'}
CAT = np.array([False, False, True, False, True, False])


def reg(R, y, cat=None):
    """Household 2-fold cross-fitted HGB regression of y on R."""
    out = np.zeros(len(y))
    for f in (0, 1):
        tr, te = FOLD != f, FOLD == f
        m = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=40,
                                          categorical_features=cat, random_state=0).fit(R[tr], y[tr])
        out[te] = m.predict(R[te])
    return out


def release(Zh, ei, eo):
    ri = np.clip(reg(Zh, ei[:, 1], CAT), 1e-4, 1 - 1e-4)
    ro = np.column_stack([np.clip(reg(Zh, eo[:, k], CAT), 1e-4, 1) for k in range(5)])
    return np.column_stack([1 - ri, ri]), ro / ro.sum(1, keepdims=True)


def soft_auc(score, p):
    o = np.argsort(score, kind='mergesort'); s, pp = score[o], p[o]; nn = 1 - pp
    _, st = np.unique(s, return_index=True)
    P, Nn = np.add.reduceat(pp, st), np.add.reduceat(nn, st)
    num = np.sum(P * (np.cumsum(Nn) - Nn)) + 0.5 * np.sum(P * Nn) - 0.5 * np.sum(pp * nn)
    return float(num / (pp.sum() * nn.sum() - np.sum(pp * nn)))


def ce(p, q):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return float(np.mean(-p * np.log(q) - (1 - p) * np.log(1 - q)))


def kl(e, r):
    return float(np.mean(np.sum(e * (np.log(e) - np.log(r)), 1)))


def logits(ri, ro):
    return np.column_stack([np.log(ri[:, 1] / ri[:, 0]), np.log(ro[:, 1:] / ro[:, :1])])


def leak(R, pi):
    return soft_auc(reg(R, pi), pi)


def evaluate(v, steps, ei, eo, pi, base=None, views=False):
    Zh = coarsen(Z, steps)
    pih = np.clip(reg(Zh, pi, CAT), 0.01, 0.99)
    ri, ro = release(Zh, ei, eo)
    L = logits(ri, ro)
    rec = {'steps': list(steps), 'bound_auc': soft_auc(pih, pi), 'sex_logloss': ce(pi, pih), 'pair_auc': leak(L, pi),
           'kl_inc': kl(ei, ri), 'kl_occ': kl(eo, ro),
           'brier_occ_men': float(np.sum((1 - pi) * np.sum((ro - eo) ** 2, 1)) / np.sum(1 - pi)),
           'brier_occ_women': float(np.sum(pi * np.sum((ro - eo) ** 2, 1)) / np.sum(pi)),
           'brier_inc': float(np.mean(np.sum((ri - ei) ** 2, 1))),
           'acc_loss_inc': float(np.mean(ei.max(1) - ei[np.arange(N), ri.argmax(1)])),
           'acc_loss_occ': float(np.mean(eo.max(1) - eo[np.arange(N), ro.argmax(1)]))}
    if views:
        rec['inc_auc'] = leak(L[:, :1], pi); rec['occ_auc'] = leak(L[:, 1:], pi)
    if base is not None:
        for k in ('kl_inc', 'kl_occ', 'brier_occ_men', 'brier_occ_women', 'brier_inc'):
            rec['d_' + k] = rec[k] - base[k]
        rec['d_spend_total'] = rec['d_kl_inc'] + rec['d_kl_occ']
        rec['d_sex_info'] = rec['sex_logloss'] - base['sex_logloss']
        rec['d_pair'] = rec['pair_auc'] - base['pair_auc']; rec['d_bound'] = rec['bound_auc'] - base['bound_auc']
    rec['_release'] = (ri, ro)
    return rec


def strip(r):
    return {k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items() if not k.startswith('_')}


def ladder(order, ei, eo, pi, base, v, ks):
    out = []
    for k in ks:
        cur = []
        for st in order[:k]:
            cur = [s for s in cur if FIELD_OF[s] != FIELD_OF[st]] + [st]
        out.append(strip(evaluate(v, cur, ei, eo, pi, base, views=True)))
    return out


def main():
    results = {'what': __doc__.strip().splitlines()[0], 'hours_check': {'z40': float(z40), 'hours_max': float(HOURS.max()),
                                                                        'hours_min': float(HOURS.min())}}
    for v in (0, 1):
        t = time.process_time()
        ei, eo, pi = load(v)
        R = {'pi_auc_max': soft_auc(pi, pi), 'twin_pair_auc': leak(logits(ei, eo), pi),
             'twin_inc_auc': leak(logits(ei, eo)[:, :1], pi), 'twin_occ_auc': leak(logits(ei, eo)[:, 1:], pi)}
        base = evaluate(v, [], ei, eo, pi, views=True)
        R['identity'] = strip(base)
        if v == 1:                      # version 1: only the ladders chosen in version 0 (cost)
            R['sex_aware_ladder'] = ladder(results['V0']['sex_aware_order'], ei, eo, pi, base, v, (2, 4))
            R['sex_blind_ladder'] = ladder(results['V0']['sex_blind_order'], ei, eo, pi, base, v, (2, 4))
            R['cpu_s'] = round(time.process_time() - t, 1); results['V1'] = R
            break
        singles = {}
        for st in FIELD_OF:
            singles[st] = strip(evaluate(v, [st], ei, eo, pi, base))
        R['singles'] = singles
        moderate = list(STRONG_OF)
        ratio = {st: singles[st]['d_sex_info'] / max(singles[st]['d_spend_total'], 1e-5) for st in FIELD_OF}
        R['ratio_sex_info_per_nat'] = {k: round(x, 4) for k, x in ratio.items()}
        # orders over the 12 steps; a strong step supersedes its field's moderate step
        aware = sorted(FIELD_OF, key=lambda s: -ratio[s])
        blind = sorted(FIELD_OF, key=lambda s: singles[s]['d_spend_total'])
        R['sex_aware_order'], R['sex_blind_order'] = aware, blind
        R['sex_aware_ladder'] = ladder(aware, ei, eo, pi, base, v, (1, 2, 3, 4, 6))
        R['sex_blind_ladder'] = ladder(blind, ei, eo, pi, base, v, (2, 4, 6, 8))
        R['cpu_s'] = round(time.process_time() - t, 1)
        results['V0'] = R
    results['cpu_s_total'] = round(time.process_time() - T0, 1)
    with open(os.path.join(HERE, 'FEASIBILITY.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: results[k] for k in ('hours_check', 'cpu_s_total')}))


if __name__ == '__main__':
    main()

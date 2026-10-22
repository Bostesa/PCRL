"""IB: coordinated, cell-targeted selective flattening (CSF-T; Designer B, family IB). PCRL campaign `pcrl_input_and_pair_release_v1`;
EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not
confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is a
training label on MODEL_FIT only (through a release-level propensity that orders 10 decision cells), never a runtime input or output.

Operation (card IB_CSFT_v1; Designer B's PROPOSAL.md and mech_sketch.py, integrated):
  twin      p_I(x), p_O(x): the shared cached task teachers (CTL_TWIN), unchanged
  tables    m_I[d], m_O[k]: means of the twin's household out-of-fold vectors over MODEL_FIT people with income decision d /
            occupation decision k (teacher-mean tables; a cell mean keeps its decision as argmax)
  key       u(x) = SHA-256('IPR_B_CSF|V<v>|' + the six compact fields) / 16^13 (one draw per six-field record, SEX-independent)
  rates     r_c over the 10 joint decision cells c = (income decision, occupation decision): cells fully flattened in decreasing order
            of Var_c(z) / mean_c(kappa) until sum kappa x rate = pi x sum kappa; z = P(SEX | the twin's centred log-probability pair)
            cross-fitted on household folds (MODEL_FIT SEX, training only); kappa = KL(p_I || m_I[d]) / 0.004 + KL(p_O || m_O[k]) / 0.008
  release   if u(x) < r_c(x): r_I = m_I[argmax p_I(x)], r_O = m_O[argmax p_O(x)] (the same people in both recipients); else the twin
  strength  pi* = the largest pi on STRENGTHS such that it and every smaller pi meet the card's budget over the twin (nested MODEL_FIT
            estimate: tables, z and rates refitted on four outer household folds and applied to the fifth; estimate + 2 SE)
Modes: 'cell' (CSF-T), 'coord' (uniform CSF, targeting off), 'indep' (uniform, independent per-recipient keys), 'cell_indep' (each
recipient allocated over its own decision cells with an own-view propensity, independent keys), 'occ_only', 'inc_only'.

Commands (real data only through the campaign queue after the receipt and the committed card):
    python3 -m ipr.ib.csf selftest
    python3 -m ipr.ib.csf fit --version V --card results/pcrl_input_and_pair_release_v1/cards/IB_CSFT_v1.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK

FAMILY = 'IB'
K_INC, K_OCC = 2, 5
FLOOR = 1e-6
STRENGTHS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
COST_NORM = (0.004, 0.008)
N_OUTER = 5
MODES = ('cell', 'coord', 'indep', 'cell_indep', 'occ_only', 'inc_only')
CODE_MODULES = ('ipr.ib.csf', 'ipr.famkit', 'ipr.paths', 'ipr.ctlkit', 'ipr.ctl.twin', 'nrm.fa.prc', 'ohn.fam_a')


# ---------------------------------------------------------------- the operation (label-free algebra)
def clipn(P):
    P = np.clip(np.asarray(P, np.float64), FLOOR, None)
    return P / P.sum(-1, keepdims=True)


def clog(P):
    L = np.log(clipn(P))
    return L - L.mean(1, keepdims=True)


def keyed_uniform(Z, tag):
    """Uniform in [0, 1) per row: SHA-256 of tag | the row's float64 bytes (identical six-field records get identical draws)."""
    Z = np.ascontiguousarray(np.asarray(Z, np.float64))
    return np.array([int(hashlib.sha256(f'{tag}|{r.tobytes().hex()}'.encode()).hexdigest()[:13], 16) / float(16 ** 13) for r in Z])


def decision_tables(P, K):
    """Teacher-mean row per decision cell (each row's maximum is its decision, so the mean keeps it); an empty cell falls back to the
    one-hot-smoothed global mean."""
    P = clipn(P)
    dec = P.argmax(1)
    glob = P.mean(0)
    T = np.zeros((K, K))
    for k in range(K):
        m = P[dec == k].mean(0) if np.any(dec == k) else 0.5 * glob + 0.5 * np.eye(K)[k]
        if m.argmax() != k:
            gap = np.max(np.delete(m, k)) - m[k]
            lam = min(1.0, (gap + 1e-9) / (1.0 + gap) + 1e-6)
            m = (1 - lam) * m + lam * np.eye(K)[k]
        T[k] = m / m.sum()
    return T


def flatten_cost(pI, pO, TI, TO):
    pI, pO = clipn(pI), clipn(pO)
    kI = np.sum(pI * (np.log(pI) - np.log(TI[pI.argmax(1)])), 1)
    kO = np.sum(pO * (np.log(pO) - np.log(TO[pO.argmax(1)])), 1)
    return kI, kO


def waterfill(cell, z, cost, budget):
    """Rates per cell: full flattening in decreasing Var_c(z) / mean_c(cost) order until the summed cost reaches `budget`."""
    cells = np.unique(cell)
    prio = {int(c): float(np.var(z[cell == c]) / max(cost[cell == c].mean(), 1e-12)) for c in cells}
    rate = {int(c): 0.0 for c in cells}
    for c in sorted(prio, key=lambda c_: (-prio[c_], c_)):
        cc = float(cost[cell == c].sum())
        rate[c] = min(1.0, budget / cc) if cc > 0 else 1.0
        budget -= rate[c] * cc
        if budget <= 1e-15:
            break
    return rate


def release_propensity(F, s, unit, version, tag):
    """Household cross-fitted P(s = 1 | F) on MODEL_FIT (F = a function of the twin's released vectors only)."""
    import warnings
    from sklearn.ensemble import HistGradientBoostingClassifier
    fold = FK.household_folds(unit, 5, tag)
    z = np.zeros(len(F))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for f in range(5):
            mdl = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                                 random_state=int(version)).fit(F[fold != f], s[fold != f])
            z[fold == f] = mdl.predict_proba(F[fold == f])[:, 1]
    return z


def fit_state(pI, pO, s, unit, version, pi, mode, tag_key):
    """State of one configuration from the twin's out-of-fold vectors on the fitting rows (s: SEX or a SEX-independent target; read
    only by the targeted modes)."""
    TI, TO = decision_tables(pI, K_INC), decision_tables(pO, K_OCC)
    st = {'pi': float(pi), 'mode': mode, 'tag': tag_key, 'T_inc': TI, 'T_occ': TO}
    if mode in ('cell', 'cell_indep'):
        kI, kO = flatten_cost(pI, pO, TI, TO)
        s = np.asarray(s).astype(int)
        if mode == 'cell':
            z = release_propensity(np.column_stack([clog(pI), clog(pO)]), s, unit, version, f'IPR_IB_z_S{int(version)}')
            cost = kI / COST_NORM[0] + kO / COST_NORM[1]
            st['rate'] = waterfill(pI.argmax(1) * K_OCC + pO.argmax(1), z, cost, float(pi) * float(np.sum(cost)))
        else:
            zI = release_propensity(clog(pI), s, unit, version, f'IPR_IB_zI_S{int(version)}')
            zO = release_propensity(clog(pO), s, unit, version, f'IPR_IB_zO_S{int(version)}')
            st['rate_I'] = waterfill(pI.argmax(1), zI, kI, float(pi) * float(np.sum(kI)))
            st['rate_O'] = waterfill(pO.argmax(1), zO, kO, float(pi) * float(np.sum(kO)))
    return st


def selection(st, Z, pI, pO):
    pi, mode, tag = st['pi'], st['mode'], st['tag']
    u = keyed_uniform(Z, tag)
    if mode == 'coord':
        return u < pi, u < pi
    if mode == 'indep':
        return u < pi, keyed_uniform(Z, tag + '|O') < pi
    if mode == 'occ_only':
        return np.zeros(len(u), bool), u < pi
    if mode == 'inc_only':
        return u < pi, np.zeros(len(u), bool)
    dI, dO = clipn(pI).argmax(1), clipn(pO).argmax(1)
    if mode == 'cell':
        r = np.array([st['rate'].get(int(c), 0.0) for c in dI * K_OCC + dO])
        return u < r, u < r
    rI = np.array([st['rate_I'].get(int(c), 0.0) for c in dI])
    rO = np.array([st['rate_O'].get(int(c), 0.0) for c in dO])
    return u < rI, keyed_uniform(Z, tag + '|O') < rO


def release_from_twin(st, Z, pI, pO):
    pI, pO = clipn(pI), clipn(pO)
    fI, fO = selection(st, Z, pI, pO)
    rI, rO = pI.copy(), pO.copy()
    rI[fI] = st['T_inc'][pI[fI].argmax(1)]
    rO[fO] = st['T_occ'][pO[fO].argmax(1)]
    return FK.clipn(rI), FK.clipn(rO)


def nested(pI, pO, s, unit, version, Z, pis, mode, tag_key):
    """{pi: (rI, rO)} on MODEL_FIT: tables, propensity and rates refitted on four outer household folds, applied to the fifth."""
    of = FK.household_folds(unit, N_OUTER, f'IPR_IB_ncv_S{int(version)}')
    n = len(unit)
    out = {p: (np.zeros((n, K_INC)), np.zeros((n, K_OCC))) for p in pis}
    for j in range(N_OUTER):
        tr, te = of != j, of == j
        for p in pis:
            st = fit_state(pI[tr], pO[tr], None if s is None else s[tr], unit[tr], version, p, mode, tag_key)
            rI, rO = release_from_twin(st, Z[te], pI[te], pO[te])
            out[p][0][te], out[p][1][te] = rI, rO
    return out


def selftest():
    rng = np.random.default_rng(7)
    n = 3000
    sex = rng.integers(0, 2, n)
    unit = np.arange(n) // 2
    Z = np.column_stack([rng.integers(25, 65, n), rng.integers(1, 99, n), rng.integers(0, 8, n), rng.integers(1, 25, n),
                         rng.integers(0, 5, n), rng.integers(1, 500, n)]).astype(float)
    pO = clipn(np.exp(rng.normal(size=(n, 5)) + 0.8 * (sex - 0.5)[:, None] * np.array([1.0, -0.5, 0.0, 0.3, -0.8])))
    pI = clipn(np.exp(rng.normal(size=(n, 2)) * 0.8 + 0.5 * (sex - 0.5)[:, None] * np.array([-1.0, 1.0])))
    ok = {}
    st = fit_state(pI, pO, sex, unit, 0, 0.2, 'cell', 'IPR_B_CSF|V0')
    fI, fO = selection(st, Z, pI, pO)
    ok['coordinated'] = bool(np.array_equal(fI, fO))
    rI, rO = release_from_twin(st, Z, pI, pO)
    ok['decisions_kept'] = bool(np.array_equal(rI.argmax(1), pI.argmax(1)) and np.array_equal(rO.argmax(1), pO.argmax(1)))
    ok['unselected_twin'] = bool(np.allclose(rO[~fO], clipn(pO)[~fO]))
    stc = fit_state(pI, pO, None, unit, 0, 0.2, 'coord', 'IPR_B_CSF|V0')
    gI, gO = selection(stc, Z, pI, pO)
    ok['uniform_rate'] = bool(abs(gI.mean() - 0.2) < 0.03 and np.array_equal(gI, gO))
    sti = fit_state(pI, pO, None, unit, 0, 0.2, 'indep', 'IPR_B_CSF|V0')
    hI, hO = selection(sti, Z, pI, pO)
    ok['independent_keys'] = bool(abs(hO.mean() - 0.2) < 0.03 and not np.array_equal(hI, hO))
    kl = lambda R, P: float(np.mean(np.sum(clipn(P) * (np.log(clipn(P)) - np.log(clipn(R))), 1)))
    aI, aO = release_from_twin(st, Z, pI, pO)
    uI, uO = release_from_twin(stc, Z, pI, pO)
    ok['equal_expected_cost'] = bool(abs((kl(aI, pI) / 0.004 + kl(aO, pO) / 0.008) - (kl(uI, pI) / 0.004 + kl(uO, pO) / 0.008))
                                     < 0.2 * (kl(uI, pI) / 0.004 + kl(uO, pO) / 0.008))
    nest = nested(pI, pO, sex, unit, 0, Z, (0.0, 0.1), 'cell', 'IPR_B_CSF|V0')
    ok['nested_zero_is_twin'] = bool(np.allclose(nest[0.0][1], FK.clipn(pO)))
    T = decision_tables(rng.dirichlet(np.ones(5) * 0.7, 5000), 5)
    ok['tables_keep_decisions'] = bool(np.array_equal(T.argmax(1), np.arange(5)))
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(prog='ipr.ib.csf')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    a = ap.parse_args()
    r = selftest()
    print(json.dumps(r))
    sys.exit(0 if r['all_ok'] else 1)

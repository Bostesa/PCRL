"""Designer B, campaign pcrl_input_and_pair_release_v1: reference sketch of CSF, coordinated selective flattening (PROPOSAL.md).

Label-free reference implementation on plain numpy arrays (the coordinator integrates it into the `ipr/` package).

The operation (release side; the task-only twin is fixed):
  1. Twin vectors p_I(x) (n, 2) and p_O(x) (n, 5): the equally trained task-only HGB teachers (passed in, or a stand-in twin is
     fitted here for the self-test).
  2. Flattening tables, fitted on MODEL_FIT with task information only: m_I[d] = mean of the twin's household out-of-fold income
     vectors over the people whose income decision is d, m_O[k] = the same for the occupation vectors and occupation decision k
     (teacher-mean tables, the B1 construction of the finite-routing campaign). If a table row's argmax is not its own decision, it
     is mixed with that decision's one-hot vector by the smallest weight that restores the decision (decisions are always kept).
  3. One keyed uniform per person, u(x) = SHA-256(tag | the six compact fields) / 16^13, SHARED by the two recipients (mode
     'coord'). If u(x) < pi, BOTH recipients receive their decision-cell row: r_I = m_I[argmax p_I(x)], r_O = m_O[argmax p_O(x)];
     otherwise both receive the twin's vectors unchanged. Mode 'indep' (the separately designed control) draws the two recipients'
     uniforms with different tags, so each recipient is flattened for a fraction pi of people, but not the same people.
  4. Modes 'coord' / 'indep' / 'occ_only' / 'inc_only' never read SEX (the `sex` argument is ignored; the strength is selected
     outside, by the MODEL_FIT budget rule, which uses SEX only for the men's / women's occupation guards).
  5. Mode 'cell' (the registered candidate CSF-T, PROPOSAL.md section 1): the same coordinated collapse, with the flattening rate
     allocated over the 10 joint decision cells c = (income decision, occupation decision). On MODEL_FIT, a release-level SEX
     propensity z(p) = P(SEX | the twin's log-ratio pair) is cross-fitted on household folds (the `sex` argument; the SBC path passes
     a permuted SEX column or PSEUDO2 instead); a cell's priority is Var_c(z) / mean_c(per-person flattening cost), the cost being
     the budget-normalized KL kI / 0.004 + kO / 0.008; the rates are water-filled in priority order until the total cost equals
     that of uniform CSF at the same pi. Mode 'cell_indep' (the
     separately designed recipients of D5) allocates each recipient over its own decision cells with a propensity fitted on its own
     vector only, with independent keys.

    fit(Z, y_inc, y_occ, sex, unit, version, strength, twin_inc=None, twin_occ=None, twin_fn=None, mode='coord') -> state
    release(state, Z, twin_inc=None, twin_occ=None) -> (P_inc (n, 2), P_occ (n, 5))
    selftest()  (invented data only)
"""
from __future__ import annotations

import hashlib

import numpy as np

STRENGTHS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)     # registered pi grid (fraction of people flattened in both recipients)
MODES = ('coord', 'indep', 'occ_only', 'inc_only', 'cell', 'cell_indep')
FLOOR = 1e-6
K_INC, K_OCC = 2, 5
COST_NORM = (0.004, 0.008)                               # S2 income / occupation NLL budgets: the 'cell' mode's cost weights


def keyed_uniform(Z, tag):
    """Uniform in [0, 1) per row: SHA-256 of tag | the row's float64 bytes (identical records get identical draws)."""
    Z = np.ascontiguousarray(np.asarray(Z, np.float64))
    return np.array([int(hashlib.sha256(f'{tag}|{r.tobytes().hex()}'.encode()).hexdigest()[:13], 16) / float(16 ** 13)
                     for r in Z])


def clipn(P):
    P = np.clip(np.asarray(P, np.float64), FLOOR, None)
    return P / P.sum(1, keepdims=True)


def decision_tables(P, K):
    """Teacher-mean row per decision cell; a row whose argmax is not its decision is mixed with the one-hot by the smallest weight
    that restores the decision. Empty cells fall back to the one-hot-smoothed global mean."""
    P = clipn(P)
    dec = P.argmax(1)
    glob = P.mean(0)
    T = np.zeros((K, K))
    for k in range(K):
        m = P[dec == k].mean(0) if np.any(dec == k) else 0.5 * glob + 0.5 * np.eye(K)[k]
        if m.argmax() != k:                 # cannot happen for argmax-defined cells (each row's max is k); kept as a guard
            gap = np.max(np.delete(m, k)) - m[k]
            lam = min(1.0, (gap + 1e-9) / (1.0 + gap) + 1e-6)
            m = (1 - lam) * m + lam * np.eye(K)[k]
        T[k] = m / m.sum()
    return T


def household_folds(unit, k=5, seed=0):
    u = np.unique(unit)
    rng = np.random.default_rng(seed)
    f = dict(zip(u, rng.permutation(len(u)) % k))
    return np.array([f[x] for x in unit])


def _standin_twin(Z, y_inc, y_occ, unit, version):
    """Stand-in task-only twin for the self-test (the campaign passes the cached A1_PO_16 HGB teachers instead)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    fold = household_folds(unit, 5, seed=1000 + int(version))
    oofI, oofO = np.zeros((len(Z), K_INC)), np.zeros((len(Z), K_OCC))
    mk = lambda: HistGradientBoostingClassifier(max_leaf_nodes=16, learning_rate=0.05, max_iter=150, min_samples_leaf=50,
                                                random_state=int(version))
    for f in range(5):
        tr, te = fold != f, fold == f
        oofI[te] = mk().fit(Z[tr], y_inc[tr]).predict_proba(Z[te])
        oofO[te] = mk().fit(Z[tr], y_occ[tr]).predict_proba(Z[te])
    mI, mO = mk().fit(Z, y_inc), mk().fit(Z, y_occ)
    return oofI, oofO, (lambda Zn: (mI.predict_proba(Zn), mO.predict_proba(Zn)))


def clog(P):
    L = np.log(clipn(P))
    return L - L.mean(1, keepdims=True)


def release_propensity(F, s, unit, version):
    """Household cross-fitted P(s = 1 | F) on MODEL_FIT, F a function of the twin's released vectors only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    fold = household_folds(unit, 5, seed=2000 + int(version))
    z = np.zeros(len(F))
    for f in range(5):
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                           random_state=int(version)).fit(F[fold != f], s[fold != f])
        z[fold == f] = m.predict_proba(F[fold == f])[:, 1]
    return z


def waterfill(cell, z, cost, budget):
    """Rates per cell: full flattening in decreasing Var_c(z) / mean_c(cost) order until the summed cost reaches `budget`."""
    cells = np.unique(cell)
    prio = {c: float(np.var(z[cell == c]) / max(cost[cell == c].mean(), 1e-12)) for c in cells}
    rate = {int(c): 0.0 for c in cells}
    for c in sorted(prio, key=prio.get, reverse=True):
        cc = float(cost[cell == c].sum())
        rate[int(c)] = min(1.0, budget / cc) if cc > 0 else 1.0
        budget -= rate[int(c)] * cc
        if budget <= 1e-15:
            break
    return rate


def fit(Z, y_inc, y_occ, sex, unit, version, strength, twin_inc=None, twin_occ=None, twin_fn=None, mode='coord'):
    """Fit CSF on MODEL_FIT. twin_inc / twin_occ: the twin's household out-of-fold vectors on these rows (tables are fitted on
    them); twin_fn: Z -> (p_I, p_O), the twin's deployed models (used by release() when no vectors are passed). `sex` is read only
    by the modes 'cell' / 'cell_indep' (the SBC path passes a SEX-independent statistic in its place)."""
    if mode not in MODES:
        raise ValueError(mode)
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError('strength pi must be in [0, 1]')
    Z = np.asarray(Z, np.float64)
    if twin_inc is None or twin_occ is None:
        twin_inc, twin_occ, fn = _standin_twin(Z, np.asarray(y_inc), np.asarray(y_occ), np.asarray(unit), version)
        twin_fn = twin_fn or fn
    st = {'pi': float(strength), 'mode': mode, 'version': int(version), 'tag': f'IPR_B_CSF|V{int(version)}',
          'T_inc': decision_tables(twin_inc, K_INC), 'T_occ': decision_tables(twin_occ, K_OCC), 'twin_fn': twin_fn}
    if mode in ('cell', 'cell_indep'):
        s, unit = np.asarray(sex).astype(int), np.asarray(unit)
        pI, pO = clipn(twin_inc), clipn(twin_occ)
        kI = np.sum(pI * (np.log(pI) - np.log(st['T_inc'][pI.argmax(1)])), 1)
        kO = np.sum(pO * (np.log(pO) - np.log(st['T_occ'][pO.argmax(1)])), 1)
        if mode == 'cell':                                   # one allocation over the joint decision cells, both recipients
            z = release_propensity(np.column_stack([clog(pI), clog(pO)]), s, unit, version)
            cost = kI / COST_NORM[0] + kO / COST_NORM[1]      # budget-normalized flattening KL (S2 NLL components)
            st['rate'] = waterfill(pI.argmax(1) * K_OCC + pO.argmax(1), z, cost, float(strength) * float(np.sum(cost)))
        else:                                                # separately designed: each recipient on its own view and cells
            zI = release_propensity(clog(pI), s, unit, version)
            zO = release_propensity(clog(pO), s, unit, version)
            st['rate_I'] = waterfill(pI.argmax(1), zI, kI, float(strength) * float(np.sum(kI)))
            st['rate_O'] = waterfill(pO.argmax(1), zO, kO, float(strength) * float(np.sum(kO)))
    return st


def selection(state, Z, pI=None, pO=None):
    """(flatten income?, flatten occupation?) per row: one shared keyed uniform ('coord', 'cell'), or two independent ones."""
    pi, mode, tag = state['pi'], state['mode'], state['tag']
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
        r = np.array([state['rate'].get(int(c), 0.0) for c in dI * K_OCC + dO])
        return u < r, u < r
    rI = np.array([state['rate_I'].get(int(c), 0.0) for c in dI])
    rO = np.array([state['rate_O'].get(int(c), 0.0) for c in dO])
    return u < rI, keyed_uniform(Z, tag + '|O') < rO


def release_from_twin(state, Z, pI, pO):
    pI, pO = clipn(pI), clipn(pO)
    fI, fO = selection(state, Z, pI, pO)
    rI, rO = pI.copy(), pO.copy()
    rI[fI] = state['T_inc'][pI[fI].argmax(1)]
    rO[fO] = state['T_occ'][pO[fO].argmax(1)]
    return clipn(rI), clipn(rO)


def release(state, Z, twin_inc=None, twin_occ=None):
    Z = np.asarray(Z, np.float64)
    if twin_inc is None or twin_occ is None:
        twin_inc, twin_occ = state['twin_fn'](Z)
    return release_from_twin(state, Z, twin_inc, twin_occ)


def selftest():
    rng = np.random.default_rng(7)
    n = 2400
    sex = rng.integers(0, 2, n)
    Z = np.column_stack([rng.integers(18, 80, n), np.clip(rng.normal(38 + 4 * sex, 9, n).round(), 1, 99), rng.integers(1, 9, n),
                         rng.integers(1, 25, n), rng.integers(1, 6, n), rng.integers(1, 60, n)]).astype(float)
    unit = rng.integers(0, 1600, n)
    y_inc = (0.05 * Z[:, 1] + 0.15 * Z[:, 3] + rng.normal(0, 1.5, n) > 5.5).astype(int)
    y_occ = np.clip(((Z[:, 3] + 2 * Z[:, 2] + rng.normal(0, 4, n)) / 7).astype(int), 0, 4)
    oI, oO, fn = _standin_twin(Z, y_inc, y_occ, unit, 0)
    st = fit(Z, y_inc, y_occ, sex, unit, 0, 0.25, twin_inc=oI, twin_occ=oO, twin_fn=fn)
    rI, rO = release(st, Z)
    tI, tO = fn(Z)
    fI, fO = selection(st, Z)
    assert np.array_equal(fI, fO), 'coordinated mode must flatten the same people in both recipients'
    assert abs(fI.mean() - 0.25) < 0.04, fI.mean()
    assert np.array_equal(rI.argmax(1), clipn(tI).argmax(1)) and np.array_equal(rO.argmax(1), clipn(tO).argmax(1)), 'decisions kept'
    assert np.allclose(rI[~fI], clipn(tI)[~fI]) and np.allclose(rO[~fO], clipn(tO)[~fO]), 'unselected people get the twin'
    assert len(np.unique(rO[fO].round(12), axis=0)) <= K_OCC and len(np.unique(rI[fI].round(12), axis=0)) <= K_INC
    rI2, rO2 = release(st, Z)
    assert np.array_equal(rI, rI2) and np.array_equal(rO, rO2), 'deterministic in the six fields'
    st_perm = fit(Z, y_inc, y_occ, rng.permutation(sex), unit, 0, 0.25, twin_inc=oI, twin_occ=oO, twin_fn=fn)
    rI3, rO3 = release(st_perm, Z)
    assert np.allclose(rI, rI3) and np.allclose(rO, rO3), 'SEX is never read'
    sti = dict(st, mode='indep')
    gI, gO = selection(sti, Z)
    assert abs(gO.mean() - 0.25) < 0.04 and 0.1 < np.mean(gI & gO) / 0.25 < 0.45, 'independent mode: overlap about pi'
    st0 = dict(st, pi=0.0)
    r0I, r0O = release(st0, Z)
    assert np.allclose(r0I, clipn(tI)) and np.allclose(r0O, clipn(tO)), 'pi = 0 is the twin'
    P = rng.dirichlet(np.ones(K_OCC) * 0.7, 5000)               # a decision-cell mean keeps its decision (each row's max is k)
    T = decision_tables(P, K_OCC)
    assert np.array_equal(T.argmax(1), np.arange(K_OCC))
    # registered candidate CSF-T ('cell'): coordinated, SEX-targeted allocation over joint decision cells at uniform CSF's spend
    sc = fit(Z, y_inc, y_occ, sex, unit, 0, 0.25, twin_inc=oI, twin_occ=oO, twin_fn=fn, mode='cell')
    cI, cO = selection(sc, Z, oI, oO)
    assert np.array_equal(cI, cO), 'CSF-T flattens the same people in both recipients'
    kl = lambda R, P: float(np.sum(clipn(P) * (np.log(clipn(P)) - np.log(clipn(R)))))
    aI, aO = release_from_twin(sc, Z, oI, oO)
    uI, uO = release_from_twin(dict(st, pi=0.25), Z, oI, oO)
    tot_c, tot_u = kl(aI, oI) + kl(aO, oO), kl(uI, oI) + kl(uO, oO)
    assert abs(tot_c - tot_u) < 0.15 * tot_u, (tot_c, tot_u)        # equal expected spend up to keyed-draw noise
    assert np.array_equal(aI.argmax(1), clipn(oI).argmax(1)) and np.array_equal(aO.argmax(1), clipn(oO).argmax(1))
    sbc = fit(Z, y_inc, y_occ, rng.permutation(sex), unit, 0, 0.25, twin_inc=oI, twin_occ=oO, twin_fn=fn, mode='cell')
    assert sbc['rate'] != sc['rate'], 'the SBC path (permuted SEX) changes the allocation'
    si = fit(Z, y_inc, y_occ, sex, unit, 0, 0.25, twin_inc=oI, twin_occ=oO, twin_fn=fn, mode='cell_indep')
    gI, gO = selection(si, Z, oI, oO)
    assert not np.array_equal(gI, gO) and set(si['rate_I']) <= {0, 1} and set(si['rate_O']) <= set(range(K_OCC))
    return 'mech_sketch selftest passed'


if __name__ == '__main__':
    print(selftest())

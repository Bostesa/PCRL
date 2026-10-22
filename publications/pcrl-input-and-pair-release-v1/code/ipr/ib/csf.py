"""IB: coordinated keyed selective flattening (CSF; Designer B's operation, family IB, redirected by the challenger to its uniform,
SEX-blind form). PCRL campaign `pcrl_input_and_pair_release_v1`; EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census
Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is
never a runtime input or output; the candidate reads no SEX (it enters only the MODEL_FIT strength-selection guards).

Operation (card IB_CSF_v1; Designer B's PROPOSAL.md and mech_sketch.py, integrated, with the challenger's redirect, findings 10-14):
  twin      p_I(x), p_O(x): the shared cached task teachers (CTL_TWIN), unchanged
  tables    m_I[d], m_O[k]: means of the twin's household out-of-fold vectors over MODEL_FIT people with income decision d /
            occupation decision k (teacher-mean tables; a cell mean keeps its decision as argmax)
  key       u(x) = SHA-256('IPR_B_CSF|V<v>|' + the six compact fields) / 16^13 (one draw per six-field record, SEX-independent)
  release   CSF (mode 'coord'): if u(x) < pi: r_I = m_I[argmax p_I(x)], r_O = m_O[argmax p_O(x)] (the same people in both
            recipients); else the twin bitwise
  strength  pi* = the largest pi on GRID such that it and every smaller pi meet budget S3 over the twin (nested MODEL_FIT estimate:
            the tables refitted on four outer household folds and applied to the fifth; estimate + 2 household-clustered SE)
Ablations at pi*: CSF-T (mode 'cell': the same coordinated collapse with rates allocated over the 10 joint decision cells in decreasing
order of Var_c(z) / mean_c(kappa), z = P(SEX | the twin's centred log-probability pair) cross-fitted on household folds, kappa =
KL(p_I || m_I[d]) / 0.004 + KL(p_O || m_O[k]) / 0.008, until the cost of uniform flattening at pi; Designer B's original candidate,
the targeting ablation) and IND (mode 'indep': independent per-recipient keys, the separately designed recipients, the coordination
ablation). Matched control (PROTOCOL 4, SEX-blind operation): the SPEND path DQ of the shared twin (ipr.paths), saved once under CTL;
SBC and RO are degenerate by construction and not built. D5: both sequential orders equal the joint design by construction (card).

Commands (real data only through the campaign queue after the receipt and the committed card):
    python3 -m ipr.ib.csf selftest | selftest_stage
    python3 -m ipr.ib.csf fit --version V --card results/pcrl_input_and_pair_release_v1/cards/IB_CSF_v1.md
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


def propensities(pI, pO, s, unit, version, mode, cache=None):
    """The release-level propensities a targeted mode needs (cached: they do not depend on the strength)."""
    cache = {} if cache is None else cache
    s = np.asarray(s).astype(int)
    if mode == 'cell' and 'z' not in cache:
        cache['z'] = release_propensity(np.column_stack([clog(pI), clog(pO)]), s, unit, version, f'IPR_IB_z_S{int(version)}')
    if mode == 'cell_indep':
        if 'zI' not in cache:
            cache['zI'] = release_propensity(clog(pI), s, unit, version, f'IPR_IB_zI_S{int(version)}')
        if 'zO' not in cache:
            cache['zO'] = release_propensity(clog(pO), s, unit, version, f'IPR_IB_zO_S{int(version)}')
    return cache


def fit_state(pI, pO, s, unit, version, pi, mode, tag_key, cache=None):
    """State of one configuration from the twin's out-of-fold vectors on the fitting rows (s: SEX or a SEX-independent target; read
    only by the targeted modes; `cache` holds the strength-independent propensities)."""
    TI, TO = decision_tables(pI, K_INC), decision_tables(pO, K_OCC)
    st = {'pi': float(pi), 'mode': mode, 'tag': tag_key, 'T_inc': TI, 'T_occ': TO}
    if mode in ('cell', 'cell_indep'):
        kI, kO = flatten_cost(pI, pO, TI, TO)
        zc = propensities(pI, pO, s, unit, version, mode, cache)
        if mode == 'cell':
            cost = kI / COST_NORM[0] + kO / COST_NORM[1]
            st['rate'] = waterfill(pI.argmax(1) * K_OCC + pO.argmax(1), zc['z'], cost, float(pi) * float(np.sum(cost)))
        else:
            st['rate_I'] = waterfill(pI.argmax(1), zc['zI'], kI, float(pi) * float(np.sum(kI)))
            st['rate_O'] = waterfill(pO.argmax(1), zc['zO'], kO, float(pi) * float(np.sum(kO)))
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
        cache = {}
        for p in pis:
            st = fit_state(pI[tr], pO[tr], None if s is None else s[tr], unit[tr], version, p, mode, tag_key, cache)
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


# ---------------------------------------------------------------- configurations and the fit stage (card IB_CSF_v1)
BUDGET = {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008,
          'inc_acc': 0.006}                                     # S3 (challenger finding 13)
GRID = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30)
CONFIGS = {
    'IB_CSF': {'kind': 'candidate', 'what': 'coordinated keyed selective flattening (uniform, SEX-blind) at the MODEL_FIT-selected pi*'},
    'IB_CSFT': {'kind': 'ablation', 'what': 'targeting ablation: the same coordinated collapse with rates allocated over the 10 joint decision '
                                        'cells by a release-level SEX propensity (CSF-T, Designer B\'s original candidate) at pi*'},
    'IB_IND': {'kind': 'ablation', 'what': 'coordination ablation (the separately designed recipients): uniform flattening at pi* with '
                                       'independent per-recipient keys'},
}
PURPOSE = ('IB CSF: decision-cell tables and keyed selective flattening, nested strength selection with the SEX guards, the shared '
           'SPEND path, the targeting and coordination ablations on MODEL_FIT (SEX as a guard label and for the CSF-T ablation)')


# ---------------------------------------------------------------- deployment
def _predict_payload(P, Z):
    from ..ctl import twin as TW
    tI, tO = TW.twin_vectors(P, Z)
    if P['op'] == 'csf':
        return release_from_twin(P['state'], Z, tI, tO)
    raise ValueError(f'unknown operation {P["op"]}')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (ipr.deploy contract): the configuration's vectors from admitted features alone; deterministic (the keyed
    selection is a function of the six fields)."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    pi, po = _predict_payload(P, FK.compact(X, names))
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


def stage_fit(version, data, card, save, roles=None, log=print, record_dir=None, save_rich=True, budget=None, grid=GRID):
    from ..ctl import twin as TW
    from .. import paths as PA
    t0 = time.process_time()
    from nrm.fa import prc
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, tsha = TW.load_task_cache(version, row_sha)
    budget = budget or BUDGET
    tI, tO = T['oof']['inc'], T['oof']['occ']
    twin_L = FK.person_losses(tI, tO, yI, yO)
    tag_key = f'IPR_B_CSF|V{int(version)}'
    # 1. nested strength selection of CSF (SEX enters only the guards)
    nest = nested(tI, tO, None, unit, version, Z, grid, 'coord', tag_key)
    L = {p_: FK.person_losses(*nest[p_], yI, yO) for p_ in grid}
    path_sp = {p_: FK.spend(L[p_], twin_L, unit, sex) for p_ in grid}
    pi_star = FK.select_strength(path_sp, grid, budget)
    sel = {'grid': list(grid), 'budget': budget, 'budget_name': 'S3', 'pi_star': pi_star,
           'path': {str(p_): FK.jsonable_spend(s_) for p_, s_ in path_sp.items()},
           'rule': 'per version: the largest grid pi such that it and every smaller grid pi meet estimate + 2 household-clustered SE <= '
                   'S3 on the six components (MODEL_FIT nested: the decision-cell tables refitted on four outer household folds and '
                   'applied to the fifth)'}
    log(json.dumps({'pi_star': pi_star, 'cpu_s': round(time.process_time() - t0, 1)}))
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'task_cache_sha256': tsha}
    exposure = ('MODEL_FIT task labels reach the cached twin (inherited) and the nested strength selection; MODEL_FIT SEX reaches only the '
                'men\'s / women\'s guards of the selection (CSF) and, for the CSF-T ablation, the release-level propensity; no CALIBRATION '
                'or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {'CTL_TWIN': (tI, tO)}

    def put(cfg, payload, oof_pair, extra):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        sp = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], (0, 0), 'MODEL_FIT: income, occupation, sex', exposure,
                        people, hh, time.process_time() - t0, {**extra, 'selection_pi_star': pi_star, 'model_sha256': meta['model_sha256'],
                                                               'oof_spend_over_twin': FK.jsonable_spend(sp)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'oof_spend_over_twin': FK.jsonable_spend(sp), **extra}
        oof_views[cfg] = oof_pair

    out = {'version': int(version), 'selection': sel, 'task_cache_sha256': tsha}
    if pi_star is None:
        log(json.dumps({'stop': 'no grid strength meets the budget: no candidate is released'}))
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'ib') / f'FIT_V{int(version)}.json', out)
        return out
    cand_L = L[pi_star]
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    st_c = fit_state(tI, tO, None, unit, version, pi_star, 'coord', tag_key)
    put('IB_CSF', {'op': 'csf', 'state': st_c}, nest[pi_star], {'op': 'csf', 'mode': 'coord', 'pi': pi_star, 'strength': pi_star})
    # 2. SPEND (DQ) on the shared twin; points saved once under CTL (a SEX-blind operation: no SBC and no RO path)
    sr = PA.spend_and_ro(tI, tO, Z, yI, yO, sex, unit, version, cand_L, twin_L, FAMILY, log=lambda s_: log(s_), ro=False)
    sr_spend = {'SPEND': sr['SPEND']}
    shared_ids, shared_status = PA.save_shared(sr_spend, version, roles, save, {'feature_names': tuple(names),
                                                                              'twin_inc': T['models']['inc'],
                                                                              'twin_occ': T['models']['occ'], 'task_cache_sha256': tsha},
                                               people, hh, time.process_time() - t0, yI, yO, twin_L, unit, sex, FAMILY,
                                               log=lambda s_: log(s_))
    for side, (g, payload, oof_pair) in sr['SPEND']['points'].items():
        oof_views[shared_ids['SPEND'][side]] = oof_pair
    # 3. ablations at pi*: targeting (CSF-T) and coordination (IND)
    nest_t = nested(tI, tO, sex, unit, version, Z, (pi_star,), 'cell', tag_key)
    st_t = fit_state(tI, tO, sex, unit, version, pi_star, 'cell', tag_key)
    put('IB_CSFT', {'op': 'csf', 'state': st_t}, nest_t[pi_star], {'op': 'csf', 'mode': 'cell', 'pi': pi_star, 'rates': st_t['rate']})
    nest_i = nested(tI, tO, None, unit, version, Z, (pi_star,), 'indep', tag_key)
    st_i = fit_state(tI, tO, None, unit, version, pi_star, 'indep', tag_key)
    put('IB_IND', {'op': 'csf', 'state': st_i}, nest_i[pi_star], {'op': 'csf', 'mode': 'indep', 'pi': pi_star})
    rows = sr['SPEND']['bracket']['grid']
    brs = sr['SPEND']['bracket']
    fineness = {'SPEND': {'cand': cand_sp, 'lower': 0.0 if brs['lo'] is None else rows[str(brs['lo'])]['spend'],
                          'upper': rows[str(brs['hi'])]['spend'], 'upper_se': rows[str(brs['hi'])]['se_diff'],
                          'max': max(r['spend'] for r in rows.values())}}
    # 4. descriptive MODEL_FIT-only checks (never a stop)
    t1 = time.process_time()
    check = FK.mf_reader_check(oof_views, sex, unit, f'IPR_mfcheck_IB_S{int(version)}')
    rich_info = None
    if save_rich:
        from .. import richreaders as RR
        rich, rich_cpu = RR.train(oof_views, sex, version)
        rich_info = {'sha256': RR.save(FAMILY, version, rich, {'configs': sorted(oof_views),
                                                               'trained_on': 'MODEL_FIT nested releases, MODEL_FIT SEX'}),
                     'cpu_s': rich_cpu}
    out.update({'saved': saved, 'spend_path_bracket': brs, 'shared_points': shared_ids, 'shared_status': shared_status,
                'csft_rates': st_t['rate'], 'fineness_inputs': fineness, 'cand_spend': cand_sp, 'rich_readers': rich_info,
                'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers (household 5-fold cross-fitted); never a decision '
                                            'input and never a stop; the decisions use the MODEL_SELECT grid',
                                    'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'ib') / f'FIT_V{int(version)}.json', out)
    return out


def selftest_stage(n=2500, seed=5):
    """End-to-end check of stage_fit on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read):
    the cached teachers are replaced by teachers fitted on the invented labels, model files go to a temporary directory, nothing is
    saved to the campaign's runs and no record is written."""
    import tempfile
    from pathlib import Path as _P
    from ..ctl import twin as TW
    from nrm.fa import prc
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    sex = rng.integers(0, 2, n)
    Zf = FK.compact(F['X'][idx], F['feature_names'])
    data = {'X': F['X'][idx], 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': ((Zf[:, 1] + 0.5 * sex + rng.normal(0, 1, n)) > 0.5).astype(np.int64),
            'occupation': np.minimum((Zf[:, 3] > 20).astype(int) + rng.integers(0, 4, n) * (rng.random(n) < 0.5) + sex, 4).astype(np.int64),
            'sex': sex}
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    hp = {**prc.HGB, 'max_iter': 40}
    old_hgb = prc.HGB
    prc.HGB = hp
    try:
        tch = prc.fit_teachers(Z, lab, unit, 0, hp, ('inc', 'occ'))
        sxs = prc.fit_teachers(Z, lab, unit, 0, hp, ('sex',))
    finally:
        prc.HGB = old_hgb
    T = {'T': tch['T'], 'models': tch['models'], 'oof': tch['oof'], 'row_id_sha256': row_sha}
    S = {'T': sxs['T'], 'models': sxs['models'], 'oof': sxs['oof'], 'row_id_sha256': row_sha}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = (C.RUNS, TW.load_task_cache, TW.load_sex_cache)
    tmp = _P(tempfile.mkdtemp(prefix='ipr_ib_selftest_'))
    try:
        C.RUNS = tmp
        TW.load_task_cache = lambda v, rs=None: (T, 'invented')
        TW.load_sex_cache = lambda v, rs=None: (S, 'invented')
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s_: None, record_dir=False, save_rich=False,
                        budget={k: 100.0 for k in BUDGET})
    finally:
        C.RUNS, TW.load_task_cache, TW.load_sex_cache = old
    ok = {'cand_saved': 'IB_CSF' in saved, 'spend_saved': any(c.startswith('CTL_DQ') for c in saved),
          'no_sbc_or_ro': not any(c.startswith('IB_SBC') or c.startswith('CTL_RO') for c in saved),
          'ablations_saved': {'IB_CSFT', 'IB_IND'} <= set(saved),
          'records_valid': all(r['kind'] == (CONFIGS[c]['kind'] if c in CONFIGS else 'control') for c, (_, r) in saved.items())}
    a = saved['IB_CSF'][0]['MODEL_FIT']
    tw = TW.twin_vectors({'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ']}, Z)
    ok['decisions_kept'] = bool(np.array_equal(a['income'].argmax(1), FK.clipn(tw[0]).argmax(1))
                                and np.array_equal(a['occupation'].argmax(1), FK.clipn(tw[1]).argmax(1)))
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['pi_star'] = out['selection']['pi_star']
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ipr.ib.csf')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd in ('selftest', 'selftest_stage'):
        r = selftest() if a.cmd == 'selftest' else selftest_stage()
        print(json.dumps(r, default=str))
        return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'pi_star': out['selection']['pi_star'], 'saved': sorted(out['saved']), 'sbc_target': out.get('sbc_target'),
                      'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""DA revision R1: IPW-STD (card DA_STD_R1; PCRL campaign `pcrl_dual_design_release_v1`; EXPLORATORY DEVELOPMENT; not confirmation).

Trigger (card DA_STD_v1 section 6, R1 (b)): the registered comparison was not separated (DA_REG: D2 PARTIAL_SBC_FRONTIER_ONLY; the
SPEND frontier NOT_SEPARATED_FROM_TASK_SPENDING) and, for at least one recipient, the MODEL_FIT out-of-fold linear R^2 of logit pi on
the lam = 1 release's log-probabilities exceeds half the twin's (`da/R1_TRIGGER_B2.json`): the SEX effect was attributed to proxy
fields by the SEX-aware trees. The reading is ADAPTIVE_R1 (it follows a MODEL_SELECT outcome).

Operation: one SEX-blind HGB per task on Z (A1_PO_16 settings) trained on MODEL_FIT with inverse-propensity weights
omega_i = w / pi_i (women) or (1 - w) / (1 - pi_i) (men), pi = the cached cross-fitted SEX-side propensity (out-of-fold, clipped to
[0.05, 0.95]), w = the registered fit's guard-balanced weight, omega normalized to mean 1; T* by the weighted inner household-CV NLL
(tag OHN_A_cv_S<v>), full refit, out-of-fold predictions on NRM_FA_xf_S<v>. In the reweighted population SEX is independent of x, so the
model targets m_w(x) = sum_s w_s P(y | x, s) without counterfactual evaluation of a SEX input. Release (1 - lam) twin + lam m_IPW;
lam grid (0.5, 0.75, 1.0) and the DA_STD_v1 S3 rule, per version.
Controls: SPEND reused (DA_SPD_LO / DA_SPD_HI); SBC refitted: the same IPW operation with SEX replaced by the DA_STD_v1 SBC target rule
(fixed-seed permutation of SEX; PSEUDO2 with gamma the smallest of 1, 2, 4 reaching the candidate's spend), its propensity a cross-fitted
HGB of the target (SEX-side settings, 200 iterations as in ddr.ctlkit.attribute_profile), two bracket points.

    python3 -m ddr.da.ipw fit --version V --card results/pcrl_dual_design_release_v1/cards/DA_STD_R1.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings

import numpy as np

from .. import common as C
from .. import famkit as FK
from . import std as S0

FAMILY = 'DA'
CONFIGS = {
    'DA_IPW': {'kind': 'candidate', 'what': 'revision R1: IPW-STD (SEX-blind models with inverse-propensity weights toward m_w) at lam*'},
    'DA_IPW_SBC_LO': {'kind': 'control', 'what': 'R1 SBC path, lower bracket point: the IPW operation with a SEX-independent target'},
    'DA_IPW_SBC_HI': {'kind': 'control', 'what': 'R1 SBC path, upper bracket point: the IPW operation with a SEX-independent target'},
}
PURPOSE = 'DA R1: IPW-STD teachers, strength selection and the refitted SBC path on MODEL_FIT (SEX as a training weight / label)'


def _hgb(T, version, cat):
    P = S0._prc()
    return P.make_hgb(P.HGB, T, version, cat)


def _fit(Z, y, K, T, version, sw):
    P = S0._prc()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = _hgb(T, version, P.CAT_Z).fit(Z, y, sample_weight=sw)
    if not np.array_equal(m.classes_, np.arange(K)):
        raise AssertionError('class support differs')
    return m


def weighted_tstar(Z, y, K, unit, version, sw):
    P = S0._prc()
    inner = P.household_folds(unit, P.N_INNER, f'OHN_A_cv_S{version}')
    cur = np.zeros(int(P.HGB['max_iter']))
    for j in range(P.N_INNER):
        tr, va = inner != j, inner == j
        m = _fit(Z[tr], y[tr], K, P.HGB['max_iter'], version, sw[tr])
        for i, r in enumerate(m.staged_decision_function(Z[va])):
            p = P.clipn(P.probs(r))
            cur[i] -= np.sum(sw[va] * np.log(p[np.arange(int(va.sum())), y[va]]))
    return int(np.argmin(cur)) + 1


def ipw_teacher(Z, y, K, unit, version, sw):
    P = S0._prc()
    T = weighted_tstar(Z, y, K, unit, version, sw)
    model = _fit(Z, y, K, T, version, sw)
    xf = P.household_folds(unit, P.N_XF, f'NRM_FA_xf_S{version}')
    oof = np.zeros((len(y), K))
    for f in range(P.N_XF):
        tr, te = xf != f, xf == f
        oof[te] = P._proba(_fit(Z[tr], y[tr], K, T, version, sw[tr]), Z[te])
    return {'T': T, 'model': model, 'oof': oof, 'n_fits': (P.N_INNER + 1, P.N_XF)}


def weights(attr, pi, w):
    pi = np.clip(np.asarray(pi, np.float64), 0.05, 0.95)
    om = np.where(np.asarray(attr) == 1, w / pi, (1 - w) / (1 - pi))
    om = om / om.mean()
    return om, float(om.sum() ** 2 / np.sum(om ** 2) / len(om))


def _predict_payload(Pm, Z):
    prc = S0._prc()
    tI, tO = prc._proba(Pm['twin_inc'], Z), prc._proba(Pm['twin_occ'], Z)
    lam = float(Pm['lambda'])
    rI, rO = prc._proba(Pm['m_inc'], Z), prc._proba(Pm['m_occ'], Z)
    return FK.clipn((1 - lam) * tI + lam * rI), FK.clipn((1 - lam) * tO + lam * rO)


def stage_fit(version, data, card, save, roles=None, log=print, save_rich=True):
    t0 = time.process_time()
    from .. import ctlkit as CK
    prc = S0._prc()
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    row_id = np.asarray(data['row_id'])
    T, S, cache_sha = S0.load_caches(version, row_sha)
    reg = json.loads((C.RESULTS / 'da' / f'FIT_V{int(version)}.json').read_text())
    w = float(reg['selection']['w'])
    budget = S0.BUDGETS[S0.BUDGET]
    tI, tO = T['oof']['inc'], T['oof']['occ']
    twin_L = FK.person_losses(tI, tO, yI, yO)
    pi = np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
    om, ess = weights(sex, pi, w)
    mi, mo = ipw_teacher(Z, yI, 2, unit, version, om), ipw_teacher(Z, yO, 5, unit, version, om)
    rel = {lam: (FK.clipn((1 - lam) * tI + lam * mi['oof']), FK.clipn((1 - lam) * tO + lam * mo['oof'])) for lam in S0.LAMBDA_GRID}
    L = {lam: FK.person_losses(*rel[lam], yI, yO) for lam in S0.LAMBDA_GRID}
    sp = {lam: FK.spend(L[lam], twin_L, unit, sex) for lam in S0.LAMBDA_GRID}
    lam_star = FK.select_strength(sp, S0.LAMBDA_GRID, budget)
    sel = {'w': w, 'ess_share': ess, 'T_inc': mi['T'], 'T_occ': mo['T'], 'lambda_star': lam_star,
           'path': {str(l): FK.jsonable_spend(v) for l, v in sp.items()}}
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'cache_sha256': cache_sha}
    exposure = ('MODEL_FIT income, occupation and SEX of all 20,000 MODEL_FIT people reach the IPW teachers (SEX through the training '
                'weights and the cached cross-fitted propensity), the strength and bracket selection; no CALIBRATION or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {}

    def put(cfg, payload, oof_pair, extra, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        spx = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure,
                        people, hh, time.process_time() - t0,
                        {**extra, 'revision': 1, 'cache_sha256': cache_sha, 'model_sha256': meta['model_sha256'],
                         'oof_spend_over_twin': FK.jsonable_spend(spx)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'oof_spend_over_twin': FK.jsonable_spend(spx), **extra}
        oof_views[cfg] = oof_pair

    out = {'version': int(version), 'selection': sel}
    if lam_star is None:
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        C.write_json(C.RESULTS / 'da' / f'FIT_R1_V{int(version)}.json', out)
        return out
    put('DA_IPW', {'op': 'ipw', 'm_inc': mi['model'], 'm_occ': mo['model'], 'lambda': lam_star}, rel[lam_star],
        {'op': 'ipw', 'lambda': lam_star, 'w': w, 'ess_share': ess}, (mi['n_fits'][0] + mo['n_fits'][0], 10))
    cand_L = L[lam_star]
    tries = []

    def sbc(attr, label):
        prof, q_a = CK.attribute_profile(Z, attr, unit, tI, tO, version, prc.CAT_Z)
        om_a, ess_a = weights(attr, q_a, float(np.mean(attr)))
        ai, ao = ipw_teacher(Z, yI, 2, unit, version, om_a), ipw_teacher(Z, yO, 5, unit, version, om_a)
        relc = {lam: (FK.clipn((1 - lam) * tI + lam * ai['oof']), FK.clipn((1 - lam) * tO + lam * ao['oof'])) for lam in S0.LAMBDA_GRID}
        Lc = {lam: FK.person_losses(*relc[lam], yI, yO) for lam in S0.LAMBDA_GRID}
        br = S0.bracket_points(Lc, cand_L, twin_L, unit, S0.LAMBDA_GRID)
        tries.append({'attribute': label, 'profile': prof, 'ess_share': ess_a, 'bracket': br, 'reaches': br['flag'] is None})
        return {'ai': ai, 'ao': ao, 'rel': relc, 'br': br, 'label': label}
    ch = sbc(S0.permuted_sex(sex, version), 'PERMUTED_SEX')
    if ch['br']['flag'] is not None:
        make, pinfo = CK.pseudo2_select(Z, yI, yO, sex, pi, row_id, version)
        for g in S0.GAMMAS:
            ch = sbc(make(g), f'PSEUDO2_gamma{g:g}')
            if ch['br']['flag'] is None:
                break
    first = True
    for cfg, lam in (('DA_IPW_SBC_LO', ch['br']['lo']), ('DA_IPW_SBC_HI', ch['br']['hi'])):
        if lam is None:
            continue
        put(cfg, {'op': 'ipw', 'm_inc': ch['ai']['model'], 'm_occ': ch['ao']['model'], 'lambda': lam}, ch['rel'][lam],
            {'op': 'ipw_sbc', 'attribute': ch['label'], 'lambda': lam, 'bracket_flag': ch['br']['flag']},
            (12 * len(tries), 10 * len(tries)) if first else (0, 0))
        first = False
    rich_info = None
    if save_rich:                                    # data-rich readers of the revision's releases (as ddr.db.ptc_r1 does)
        import pickle
        from .. import richreaders as RR
        rich, rich_cpu = RR.train(oof_views, sex, version)
        buf = pickle.dumps({'models': rich, 'configs': sorted(oof_views)}, protocol=5)
        with open(C.RUNS / FAMILY / f'RICH_READERS_R1_V{int(version)}.pkl', 'xb') as f:
            f.write(buf)
        (C.RUNS / FAMILY / f'RICH_READERS_R1_V{int(version)}.sha256').write_text(FK.sha_bytes(buf))
        rich_info = {'sha256': FK.sha_bytes(buf), 'cpu_s': rich_cpu}
    check = FK.mf_reader_check({'TWIN': (tI, tO), **oof_views}, sex, unit, f'DDR_mfcheck_DAR1_S{int(version)}')
    out.update({'saved': saved, 'sbc_tries': tries, 'sbc_attribute': ch['label'], 'mf_reader_check': {'auc': check},
                'rich_readers': rich_info, 'cpu_s': time.process_time() - t0})
    C.write_json(C.RESULTS / 'da' / f'FIT_R1_V{int(version)}.json', out)
    return out


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    Pm = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, Pm)
    rI, rO = _predict_payload(Pm, FK.compact(X, names))
    return {'income': rI, 'occupation': rO}


def selftest():
    rng = np.random.default_rng(2)
    n = 1000
    a = rng.integers(0, 2, n)
    pi = np.clip(0.3 + 0.4 * a + rng.normal(0, 0.1, n), 0.01, 0.99)
    om, ess = weights(a, pi, 0.45)
    ok = {'mean_one': abs(om.mean() - 1) < 1e-9, 'ess_in_unit': 0 < ess <= 1,
          'balance': abs(np.average(a, weights=om) - 0.45) < 0.08}
    ok = {k: bool(v) for k, v in ok.items()}
    ok['all_ok'] = all(ok.values())
    return ok


def selftest_stage(n=2500, seed=5):
    """stage_fit end to end on INVENTED labels attached to admitted MODEL_FIT features (label-free): teachers fitted on the invented
    labels stand in for the caches, model files and records go to a temporary directory; nothing touches the campaign's runs."""
    import tempfile
    from pathlib import Path as _P
    prc = S0._prc()
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    Zc = FK.compact(F['X'][idx], F['feature_names'])
    sex = (rng.random(n) < 1.0 / (1.0 + np.exp((Zc[:, 1] - np.median(Zc[:, 1])) / 6.0))).astype(np.int64)
    data = {'X': F['X'][idx], 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': (rng.random(n) < 0.3 + 0.3 * sex).astype(np.int64), 'occupation': np.minimum(rng.integers(0, 5, n) + sex, 4),
            'sex': sex}
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    hp = {**prc.HGB, 'max_iter': 40}
    old_hgb, old_res, old_runs, old_load = prc.HGB, C.RESULTS, C.RUNS, S0.load_caches
    prc.HGB = hp
    tmp = _P(tempfile.mkdtemp(prefix='ddr_ipw_selftest_'))
    saved = {}
    try:
        tch = prc.fit_teachers(Z, lab, unit, 0, hp, ('inc', 'occ'))
        sxs = prc.fit_teachers(Z, lab, unit, 0, hp, ('sex', 'occ_s'))
        Tc = {'T': tch['T'], 'models': tch['models'], 'oof': tch['oof'], 'row_id_sha256': row_sha}
        Sc = {'T': sxs['T'], 'models': sxs['models'], 'oof': sxs['oof'], 'row_id_sha256': row_sha}
        C.RUNS, C.RESULTS = tmp / 'runs', tmp / 'results'
        (C.RESULTS / 'da').mkdir(parents=True)
        C.write_json(C.RESULTS / 'da' / 'FIT_V0.json', {'selection': {'w': 0.45}})
        S0.load_caches = lambda v, rs: (Tc, Sc, {'TASK.pkl': 'invented', 'SEXSIDE.pkl': 'invented'})
        S0.BUDGETS['TEST'] = {k: 100.0 for k in S0.BUDGETS['S3']}
        old_budget = S0.BUDGET
        S0.BUDGET = 'TEST'
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', lambda f, c, v, arr, rec: saved.__setitem__(c, (arr, rec)), roles=roles,
                        log=lambda s: None, save_rich=False)
    finally:
        prc.HGB, C.RESULTS, C.RUNS, S0.load_caches = old_hgb, old_res, old_runs, old_load
        S0.BUDGET = 'S3'
        S0.BUDGETS.pop('TEST', None)
    ok = {'cand_saved': 'DA_IPW' in saved, 'lambda_selected': out['selection']['lambda_star'] in S0.LAMBDA_GRID,
          'sbc_saved': any(c.startswith('DA_IPW_SBC') for c in saved),
          'release_valid': all(np.allclose(a['MODEL_FIT']['occupation'].sum(1), 1) for a, _ in saved.values())}
    ok = {k: bool(v) for k, v in ok.items()}
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ddr.da.ipw')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit'); f.add_argument('--version', type=int, required=True); f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    if a.cmd == 'selftest_stage':
        r = selftest_stage(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'lambda_star': out['selection']['lambda_star'], 'saved': sorted(out['saved']), 'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

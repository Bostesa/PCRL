"""DB revision R1: recipient-split strengths of propensity-tilt cancellation (card DB_PTC_R1; PCRL campaign
`pcrl_dual_design_release_v1`; EXPLORATORY DEVELOPMENT; not confirmation).

Trigger (card DB_PTC_v1 section 6, MODEL_FIT selection, both versions): at alpha* the binding S3 component (the first failing one at
the next grid value) belongs to the income recipient while every occupation component keeps at least 40% budget slack. Read from
results/<campaign>/db/FIT_V<v>.json by `trigger()` (label-free; recorded).

Operation: the DB_PTC_v1 tilt law with separate strengths, log r_I = log p_I - alpha_I (q - qbar) w_I, log r_O = log p_O - alpha_O
(q - qbar_k) W_O[k]; (alpha_I, alpha_O) on the product grid STRENGTHS x STRENGTHS; nested MODEL_FIT spend as in DB_PTC_v1; selection:
the admissible pair maximizing alpha_I + alpha_O (ties: the smaller maximum, then the smaller alpha_I), where a pair is admissible iff it
and every grid pair dominated by it (componentwise <=) meet the DB_PTC_v1 budget (S3 plus each task's summed Brier 0.0045, estimate +
2 household-clustered SE).
Controls: the SPEND path is reused (DB_SPD_LO / DB_SPD_HI; the twin is unchanged; PROTOCOL 8); the SBC path is refitted (the operation
changed): the same target rule as DB_PTC_v1 (permuted SEX, else PSEUDO2 with gamma = 1), the split law along the candidate's ratio
rho = alpha_O* / alpha_I* (points g x m: alpha_I = g m, alpha_O = g m rho, g in STRENGTHS), m by the DB_PTC_v1 scale ladder, two
bracket points.

    python3 -m ddr.db.ptc_r1 trigger
    python3 -m ddr.db.ptc_r1 fit --version V --card results/pcrl_dual_design_release_v1/cards/DB_PTC_R1.md
    python3 -m ddr.db.ptc_r1 ablate --version V --card results/pcrl_dual_design_release_v1/cards/DB_PTC_R1_ABL.md
    python3 -m ddr.db.ptc_r1 ztilt --version V --card results/pcrl_dual_design_release_v1/cards/DB_ZTILT_R1.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np

from .. import common as C
from .. import famkit as FK
from . import ptc as P

FAMILY = 'DB'
CONFIGS = {
    'DB_PTC_R1': {'kind': 'candidate', 'what': 'revision R1: recipient-split strengths (alpha_I, alpha_O) of propensity-tilt cancellation'},
    'DB_SBC_R1_LO': {'kind': 'control', 'what': 'R1 SBC path, lower bracket point: the split tilt law driven by a SEX-independent target'},
    'DB_SBC_R1_HI': {'kind': 'control', 'what': 'R1 SBC path, upper bracket point: the split tilt law driven by a SEX-independent target'},
    'DB_PTC_R1_OCC': {'kind': 'ablation', 'what': 'component ablation of DB_PTC_R1 (card DB_PTC_R1_ABL): its occupation tilt at alpha_O*; '
                                               'income = the twin (per-recipient attribution, as DB_PTC_OCC)'},
    'DB_ZTILT_R1': {'kind': 'control', 'what': 'release-only control of DB_PTC_R1 (card DB_ZTILT_R1): the split tilt law along rho = '
                                             'alpha_O* / alpha_I* with the amplitude P(SEX | the twin\'s release), spend-matched'},
}
ZTILT_PURPOSE = ('DB R1 release-only control (card DB_ZTILT_R1): the ZTILT amplitude teacher, its tilt weights and the nested MODEL_FIT '
                 'spend matching on MODEL_FIT (SEX as a training label)')
ABLATION_PURPOSE = ('DB R1 component ablation (card DB_PTC_R1_ABL): the nested MODEL_FIT occupation release of DB_PTC_R1 at its fitted '
                    'alpha_O*, income = the twin; MODEL_FIT spend record (SEX as a training label of the tilt weights)')
PURPOSE = 'DB R1: recipient-split strengths, nested selection and the refitted SBC path on MODEL_FIT (SEX as a training label)'
SLACK = 0.40


def trigger():
    """The registered MODEL_FIT trigger, from the DB fit records (label-free)."""
    out = {}
    budget = P.BUDGET
    inc_keys = ('inc_nll', 'inc_acc', 'inc_brier')
    for v in C.VERSIONS:
        r = json.loads((C.RESULTS / 'db' / f'FIT_V{v}.json').read_text())
        s = r['selection']
        a = s['alpha_star']
        grid = [float(x) for x in s['path']]
        nxt = next((g for g in grid if g > a), None)
        rows = s['path']
        fails_next = [k for k, lim in budget.items() if nxt is not None and
                      rows[str(nxt)][k]['estimate'] + 2 * rows[str(nxt)][k]['se'] > lim]
        slack = {k: 1 - (rows[str(a)][k]['estimate'] + 2 * rows[str(a)][k]['se']) / lim for k, lim in budget.items()
                 if k not in inc_keys}
        out[f'V{v}'] = {'alpha_star': a, 'next': nxt, 'failing_at_next': fails_next,
                        'binding_is_income_only': bool(fails_next) and all(k in inc_keys for k in fails_next),
                        'occupation_slack_at_alpha_star': slack, 'occupation_slack_ok': all(x >= SLACK for x in slack.values())}
    out['triggered'] = all(x['binding_is_income_only'] and x['occupation_slack_ok'] for x in out.values() if isinstance(x, dict))
    return out


def split_releases(pI, pO, q, s, unit, version, pairs, q_rel=None):
    """{(aI, aO): (rI, rO)} nested on MODEL_FIT (weights refitted on the other outer folds), as ptc.nested_releases."""
    of = FK.household_folds(unit, P.N_OUTER, f'DDR_DB_ncv_S{int(version)}')
    n = len(unit)
    out = {pr: (np.zeros((n, 2)), np.zeros((n, 5))) for pr in pairs}
    qa = q if q_rel is None else q_rel
    for j in range(P.N_OUTER):
        tr, te = of != j, of == j
        st = P.fit_weights(pI[tr], pO[tr], q[tr], s[tr])
        for aI, aO in pairs:
            rI, _ = P.tilt(pI[te], pO[te], qa[te], st, aI, inc=True, occ=False)
            _, rO = P.tilt(pI[te], pO[te], qa[te], st, aO, inc=False, occ=True)
            out[(aI, aO)][0][te], out[(aI, aO)][1][te] = rI, rO
    return out


def select_pair(spends, budget, k_se=2.0):
    ok = {pr: all(sp[k][0] + k_se * sp[k][1] <= lim for k, lim in budget.items()) for pr, sp in spends.items()}
    adm = [pr for pr in ok if ok[pr] and all(ok[q] for q in ok if q[0] <= pr[0] and q[1] <= pr[1])]
    if not adm:
        return None, ok
    return min(adm, key=lambda pr: (-(pr[0] + pr[1]), max(pr), pr[0])), ok


def _predict_payload(Pm, Z):
    prc = P._prc()
    pI, pO = prc._proba(Pm['twin_inc'], Z), prc._proba(Pm['twin_occ'], Z)
    if Pm['op'] == 'ztilt_r1':                   # release-only amplitude: P(SEX | the twin's released log-probabilities)
        q = np.clip(Pm['q_model'].predict_proba(P.zfeat(pI, pO))[:, 1], 0.01, 0.99)
    else:
        q = np.clip(prc._proba(Pm['q_model'], Z)[:, 1], 0.01, 0.99)
    if Pm['op'] == 'ptc_r1_occ':                 # component ablation: the income recipient receives the twin (as DB_PTC_OCC does)
        rI, _ = P.tilt(pI, pO, q, Pm['weights'], 0.0, inc=False, occ=False)
    else:
        rI, _ = P.tilt(pI, pO, q, Pm['weights'], float(Pm['alpha_inc']), inc=True, occ=False)
    _, rO = P.tilt(pI, pO, q, Pm['weights'], float(Pm['alpha_occ']), inc=False, occ=True)
    return rI, rO


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    Pm = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, Pm)
    rI, rO = _predict_payload(Pm, FK.compact(X, names))
    return {'income': rI, 'occupation': rO}


def stage_fit(version, data, card, save, roles=None, log=print, budget_override=None, record_dir=None, save_rich=True):
    t0 = time.process_time()
    prc = P._prc()
    from ..da import std as DA
    from .. import ctlkit as CK
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, S, cache_sha = DA.load_caches(version, row_sha)
    budget = budget_override or P.BUDGET
    pI, pO = T['oof']['inc'], T['oof']['occ']
    q = np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
    twin_L = FK.person_losses(pI, pO, yI, yO)
    pairs = [(a, b) for a in P.STRENGTHS for b in P.STRENGTHS]
    rel = split_releases(pI, pO, q, sex, unit, version, pairs)
    L = {pr: FK.person_losses(*rel[pr], yI, yO) for pr in pairs}
    sp = {pr: FK.spend(L[pr], twin_L, unit, sex) for pr in pairs}
    star, ok = select_pair(sp, budget)
    st_full = P.fit_weights(pI, pO, q, sex)
    sel = {'pair_star': list(star) if star else None, 'admissible_rule': 'it and every dominated grid pair meet the budget',
           'grid_ok': {f'{a}|{b}': v for (a, b), v in ok.items()},
           'spends': {f'{a}|{b}': FK.jsonable_spend(v) for (a, b), v in sp.items()}}
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'cache_sha256': cache_sha}
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the tilt weights and the selection '
                '(SEX or a SEX-independent target as the weights\' label); cached teachers; no CALIBRATION or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {}

    def put(cfg, payload, oof_pair, extra, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        spx = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure,
                        people, hh, time.process_time() - t0,
                        {**extra, 'revision': 1, 'twin_member': 'DA/DA_TWIN', 'cache_sha256': cache_sha,
                         'model_sha256': meta['model_sha256'], 'nested_spend_over_twin': FK.jsonable_spend(spx)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'nested_spend_over_twin': FK.jsonable_spend(spx), **extra}
        oof_views[cfg] = oof_pair

    out = {'version': int(version), 'selection': sel, 'trigger': trigger()}
    if star is None:
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        C.write_json((record_dir or C.RESULTS / 'db') / f'FIT_R1_V{int(version)}.json', out)
        return out
    aI, aO = star
    put('DB_PTC_R1', {'op': 'ptc_r1', 'q_model': S['models']['sex'], 'weights': st_full, 'alpha_inc': aI, 'alpha_occ': aO},
        rel[star], {'op': 'ptc_r1', 'alpha_inc': aI, 'alpha_occ': aO}, (0, 0))
    cand_L = L[star]
    # refitted SBC path: the same target rule, the split law along rho = aO / aI
    rho = aO / aI
    cand_rms = P.rms_tilt(pO, q, st_full)
    tries = []

    def sbc(target, label):
        qm, qa = P.target_teacher(Z, target, unit, version, int(S['T']['sex']))
        stt = P.fit_weights(pI, pO, qa, target)
        if not stt['cov_q_s'] > P.COV_MIN:
            tries.append({'target': label, 'degenerate': True, 'cov_q_s': stt['cov_q_s']})
            return None
        m0 = cand_rms / max(P.rms_tilt(pO, qa, stt), 1e-12)
        res = None
        for mult in P.SBC_SCALE_LADDER:
            m = m0 * mult
            gpairs = [(float(g * m), float(g * m * rho)) for g in P.STRENGTHS]
            relc = split_releases(pI, pO, qa, np.asarray(target), unit, version, gpairs)
            Lc = {g: FK.person_losses(*relc[g], yI, yO) for g in gpairs}
            br = DA.bracket_points(Lc, cand_L, twin_L, unit, gpairs)
            res = {'q_model': qm, 'weights': stt, 'rel': relc, 'br': br, 'label': label, 'scale_m': m}
            if br['flag'] is None:
                break
        tries.append({'target': label, 'degenerate': False, 'cov_q_s': stt['cov_q_s'], 'scale_m': res['scale_m'],
                      'bracket': {**res['br'], 'grid': {str(k): v for k, v in res['br']['grid'].items()}}})
        return res
    ch = sbc(P.permuted(sex, version), 'PERMUTED_SEX')
    if ch is None or ch['br']['flag'] is not None:
        make, _ = CK.pseudo2_select(Z, yI, yO, sex, q, np.asarray(data['row_id']), version, gammas=(1.0,))
        ch = sbc(make(1.0), 'PSEUDO2_gamma1')
    first = True
    if ch is not None:
        for cfg, g in (('DB_SBC_R1_LO', ch['br']['lo']), ('DB_SBC_R1_HI', ch['br']['hi'])):
            if g is None:
                continue
            put(cfg, {'op': 'sbc_r1', 'q_model': ch['q_model'], 'weights': ch['weights'], 'alpha_inc': g[0], 'alpha_occ': g[1]},
                ch['rel'][g], {'op': 'sbc_r1', 'target': ch['label'], 'alpha_inc': g[0], 'alpha_occ': g[1],
                               'bracket_flag': ch['br']['flag']}, (6 * len(tries), 0) if first else (0, 0))
            first = False
    rich_info = None
    if save_rich:
        from .. import richreaders as RR
        rich, rich_cpu = RR.train(oof_views, sex, version)
        import pickle
        d = C.RUNS / FAMILY
        p = d / f'RICH_READERS_R1_V{int(version)}.pkl'
        buf = pickle.dumps({'models': rich, 'configs': sorted(oof_views)}, protocol=5)
        with open(p, 'xb') as f:
            f.write(buf)
        (d / f'RICH_READERS_R1_V{int(version)}.sha256').write_text(FK.sha_bytes(buf))
        rich_info = {'sha256': FK.sha_bytes(buf), 'cpu_s': rich_cpu}
    check = FK.mf_reader_check({'TWIN': (pI, pO), **oof_views}, sex, unit, f'DDR_mfcheck_DBR1_S{int(version)}')
    out.update({'saved': saved, 'sbc_tries': tries, 'sbc_target': (ch or {}).get('label'), 'rich_readers': rich_info,
                'mf_reader_check': {'auc': check, 'note': 'descriptive'}, 'cpu_s': time.process_time() - t0})
    C.write_json((record_dir or C.RESULTS / 'db') / f'FIT_R1_V{int(version)}.json', out)
    return out


def ablation_fit(version, data, card, save, roles=None):
    """Card DB_PTC_R1_ABL: the component ablation DB_PTC_R1_OCC of the strongest DB release. It re-uses DB_PTC_R1's fitted
    (alpha_I*, alpha_O*) (db/FIT_R1_V<v>.json), weights, propensity and twin; nothing is selected here. The income recipient receives
    the twin's vector, the occupation recipient DB_PTC_R1's tilted vector. Its nested MODEL_FIT release gives the spend record."""
    t0 = time.process_time()
    prc = P._prc()
    from ..da import std as DA
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, S, cache_sha = DA.load_caches(version, row_sha)
    fit = json.loads((C.RESULTS / 'db' / f'FIT_R1_V{int(version)}.json').read_text())
    aI, aO = (float(x) for x in fit['selection']['pair_star'])
    pI, pO = T['oof']['inc'], T['oof']['occ']
    q = np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
    twin_L = FK.person_losses(pI, pO, yI, yO)
    rel = split_releases(pI, pO, q, sex, unit, version, [(aI, aO)])[(aI, aO)]
    oof_pair = (P.tilt(pI, pO, q, None, 0.0, inc=False, occ=False)[0], rel[1])
    cand = FK.load_model(FAMILY, 'DB_PTC_R1', version)
    payload = {'feature_names': cand['feature_names'], 'twin_inc': cand['twin_inc'], 'twin_occ': cand['twin_occ'],
               'cache_sha256': cand['cache_sha256'], 'op': 'ptc_r1_occ', 'q_model': cand['q_model'], 'weights': cand['weights'],
               'alpha_inc': 0.0, 'alpha_occ': aO, 'ablation_of': 'DB_PTC_R1', 'ablation_of_alpha_inc': aI}
    meta = FK.write_model(FAMILY, 'DB_PTC_R1_OCC', version, payload)
    arrays = FK.release_all(predict, 'DB_PTC_R1_OCC', version, roles or FK.role_features())
    spx = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the nested tilt weights (SEX as their '
                'label) and the spend record; cached teachers; no CALIBRATION or MODEL_SELECT label; nothing is selected')
    rec = FK.record(card, CONFIGS['DB_PTC_R1_OCC']['kind'], CONFIGS['DB_PTC_R1_OCC']['what'], (0, 0), 'MODEL_FIT: income, occupation, sex',
                    exposure, len(Z), len(np.unique(unit)), time.process_time() - t0,
                    {'op': 'ptc_r1_occ', 'alpha_inc': 0.0, 'alpha_occ': aO, 'ablation_of': 'DB_PTC_R1', 'revision': 1,
                     'twin_member': 'DA/DA_TWIN', 'cache_sha256': cache_sha, 'model_sha256': meta['model_sha256'],
                     'nested_spend_over_twin': FK.jsonable_spend(spx)})
    save(FAMILY, 'DB_PTC_R1_OCC', version, arrays, rec)
    out = {'version': int(version), 'ablation_of': 'DB_PTC_R1', 'pair_star_of_release': [aI, aO],
           'nested_spend_over_twin': FK.jsonable_spend(spx), 'cpu_s': time.process_time() - t0}
    C.write_json(C.RESULTS / 'db' / f'ABLATION_R1_V{int(version)}.json', out)
    return out


def ztilt_fit(version, data, card, save, roles=None):
    """Card DB_ZTILT_R1: the registered ZTILT rule (DB_PTC_v1, PROTOCOL_AMENDMENT_1 finding 2) applied to DB_PTC_R1. Amplitude
    q_z = P(SEX | the twin's released log-probabilities) (ptc.ztilt_teacher, cross-fitted), its own tilt weights (same rule), the split
    law along DB_PTC_R1's rho = alpha_O* / alpha_I*: pairs k (alpha_I* m_z0, alpha_O* m_z0), k in SHUF_MULT, m_z0 the RMS-normalizing
    factor; the first pair whose nested MODEL_FIT mean total-NLL spend reaches DB_PTC_R1's is saved."""
    t0 = time.process_time()
    prc = P._prc()
    from ..da import std as DA
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, S, cache_sha = DA.load_caches(version, row_sha)
    fit = json.loads((C.RESULTS / 'db' / f'FIT_R1_V{int(version)}.json').read_text())
    aI, aO = (float(x) for x in fit['selection']['pair_star'])
    pI, pO = T['oof']['inc'], T['oof']['occ']
    q = np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
    twin_L = FK.person_losses(pI, pO, yI, yO)
    cand_rel = split_releases(pI, pO, q, sex, unit, version, [(aI, aO)])[(aI, aO)]
    cand_sp = float((FK.person_losses(*cand_rel, yI, yO)['tot_nll'] - twin_L['tot_nll']).mean())
    st_full = P.fit_weights(pI, pO, q, sex)
    cand_rms = P.rms_tilt(pO, q, st_full)
    zm, qz = P.ztilt_teacher(pI, pO, sex, unit, version)
    st_z = P.fit_weights(pI, pO, qz, sex)
    mz0 = cand_rms / max(P.rms_tilt(pO, qz, st_z), 1e-12)
    pairs = [(float(aI * mz0 * k), float(aO * mz0 * k)) for k in P.SHUF_MULT]
    rel = split_releases(pI, pO, qz, sex, unit, version, pairs)
    sp = {g: float((FK.person_losses(*rel[g], yI, yO)['tot_nll'] - twin_L['tot_nll']).mean()) for g in pairs}
    g = next((x for x in pairs if sp[x] >= cand_sp), pairs[-1])
    cand = FK.load_model(FAMILY, 'DB_PTC_R1', version)
    payload = {'feature_names': cand['feature_names'], 'twin_inc': cand['twin_inc'], 'twin_occ': cand['twin_occ'],
               'cache_sha256': cand['cache_sha256'], 'op': 'ztilt_r1', 'q_model': zm, 'weights': st_z,
               'alpha_inc': g[0], 'alpha_occ': g[1]}
    meta = FK.write_model(FAMILY, 'DB_ZTILT_R1', version, payload)
    arrays = FK.release_all(predict, 'DB_ZTILT_R1', version, roles or FK.role_features())
    spx = FK.spend(FK.person_losses(*rel[g], yI, yO), twin_L, unit, sex)
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the amplitude teacher (SEX), the tilt '
                'weights and the spend matching; cached teachers; no CALIBRATION or MODEL_SELECT label')
    extra = {'op': 'ztilt_r1', 'alpha_inc': g[0], 'alpha_occ': g[1], 'pairs': [list(x) for x in pairs],
             'spends': {f'{a}|{b}': v for (a, b), v in sp.items()}, 'cand_spend': cand_sp, 'reached_cand_spend': bool(sp[g] >= cand_sp),
             'qz_auc_oof': FK.auc(qz, sex), 'control_of': 'DB_PTC_R1', 'revision': 1, 'twin_member': 'DA/DA_TWIN',
             'cache_sha256': cache_sha, 'model_sha256': meta['model_sha256'], 'nested_spend_over_twin': FK.jsonable_spend(spx)}
    rec = FK.record(card, CONFIGS['DB_ZTILT_R1']['kind'], CONFIGS['DB_ZTILT_R1']['what'], (1, 5), 'MODEL_FIT: income, occupation, sex',
                    exposure, len(Z), len(np.unique(unit)), time.process_time() - t0, extra)
    save(FAMILY, 'DB_ZTILT_R1', version, arrays, rec)
    out = {'version': int(version), 'control_of': 'DB_PTC_R1', 'pair_star_of_release': [aI, aO], 'chosen': list(g),
           'cand_spend': cand_sp, 'spends': extra['spends'], 'reached_cand_spend': extra['reached_cand_spend'],
           'qz_auc_oof': extra['qz_auc_oof'], 'cpu_s': time.process_time() - t0}
    C.write_json(C.RESULTS / 'db' / f'ZTILT_R1_V{int(version)}.json', out)
    return out


def selftest():
    sp = {(a, b): {'x': (0.001 * (a + b), 0.0)} for a in (0.2, 0.5) for b in (0.2, 0.5)}
    star, ok = select_pair(sp, {'x': 0.0008})
    ok1 = star == (0.2, 0.5) or star == (0.5, 0.2) or star is not None
    star2, _ = select_pair(sp, {'x': 0.0003})
    r = {'select_pair_max_sum': star == (0.2, 0.5) or star == (0.5, 0.2), 'select_pair_none': star2 is None, 'ok1': ok1}
    r = {k: bool(v) for k, v in r.items()}
    r['all_ok'] = all(r.values())
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ddr.db.ptc_r1')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('trigger')
    f = sub.add_parser('fit'); f.add_argument('--version', type=int, required=True); f.add_argument('--card', required=True)
    b = sub.add_parser('ablate'); b.add_argument('--version', type=int, required=True); b.add_argument('--card', required=True)
    z = sub.add_parser('ztilt'); z.add_argument('--version', type=int, required=True); z.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    if a.cmd == 'trigger':
        print(json.dumps(trigger(), indent=1)); return 0
    from .. import release as RL
    if a.cmd == 'ztilt':
        data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', ZTILT_PURPOSE, card=a.card)
        out = ztilt_fit(a.version, data, a.card, RL.save)
        print(json.dumps({k: out[k] for k in ('chosen', 'cand_spend', 'reached_cand_spend', 'qz_auc_oof', 'cpu_s')}))
        return 0
    if a.cmd == 'ablate':
        data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', ABLATION_PURPOSE, card=a.card)
        out = ablation_fit(a.version, data, a.card, RL.save)
        print(json.dumps({k: out[k] for k in ('ablation_of', 'pair_star_of_release', 'cpu_s')}))
        return 0
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'pair_star': out['selection']['pair_star'], 'saved': sorted(out['saved']), 'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""IA revision R1: occupation-side SEX-aware field generalization (OSG; card IA_OSG_R1). PCRL campaign
`pcrl_input_and_pair_release_v1`; EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida
person records; not confirmation.

Family IA's first revision, written after its registered reading (IA_REG, NOT_SEPARATED); adaptive as a whole (PROTOCOL 7, tag
ADAPTIVE_R1). Observed failures (card section 1): IA_SFG_v1's shared ladder is held back by income at its first informative step; the
refitted income model leaked more on MODEL_SELECT; the twin's pair recovery is carried by its occupation view.

Operation (card IA_OSG_R1):
  income      the twin's vector (CTL_TWIN's income teacher on the raw record), unchanged
  occupation  P_occ = f_occ,k(h_k(x)): IA_SFG_v1's menu and maps (ipr.ia.sfg); single-step statistics on MODEL_FIT with the task cost
              priced in occupation nats only (dT = dT_occ) and IA_SFG_v1's SEX information dI; build_ladder unchanged; the occupation
              model on h_k(Z) by the twin's construction for the occupation task only (nrm.fa.prc.fit_teachers, tasks ('occ',))
  strength    k* = the largest rung k = 1, 2, ... such that it and every smaller rung meet S3 over the twin (estimate + 2 household-
              clustered SE; the income components are exactly zero); the scan stops at the first failing rung
Matched controls, each restricted to the occupation output (income passes through bitwise), saved under IA:
  SBC     the same ladder rule with IA_SFG_v1's permuted SEX (PSEUDO2 only by the registered fallback rule); two bracket points
  SPEND   SEX-blind dithered quantization of the occupation output (ipr.ctlkit.dq_release's occupation coordinates; DQ_GRID_FINE, tag
          IPR_DQ_S<v>); two bracket points
  RO      the generic release-only tilt (ipr.ctlkit.ro_path; GENERIC_ZTILT) applied to the occupation output; two bracket points
Ablation: IA_OSG_BLIND, the ladder ordered by occupation task spend only, first rung whose MODEL_FIT spend reaches CAND's.
Descriptive MODEL_FIT-only: cross-fitted readers, the data-rich readers (RICH_READERS_R1_V<v>) and the equality of the recomputed
single-step statistics with those recorded by IA_SFG_v1's fit (ia/FIT_V<v>.json).

Commands (real data only through the campaign queue after the receipt and the committed card):
    python3 -m ipr.ia.osg selftest | selftest_stage
    python3 -m ipr.ia.osg fit --version V --card results/pcrl_input_and_pair_release_v1/cards/IA_OSG_R1.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK
from . import sfg as SF

FAMILY = 'IA'
REVISION = 'R1'
BUDGET = SF.BUDGET                      # S3, unchanged
GRID = SF.GRID
CONFIGS = {
    'IA_OSG': {'kind': 'candidate', 'what': "occupation-side SEX-aware field generalization at the MODEL_FIT-selected rung k* (the income "
                                            "recipient receives the twin's vector)"},
    'IA_OSG_SBC_LO': {'kind': 'control', 'what': 'SBC path, lower bracket point: the same occupation-only ladder rule and models with a '
                                                 'SEX-independent target'},
    'IA_OSG_SBC_HI': {'kind': 'control', 'what': 'SBC path, upper bracket point: the same occupation-only ladder rule and models with a '
                                                 'SEX-independent target'},
    'IA_OSG_SPD_LO': {'kind': 'control', 'what': "SPEND path, lower bracket point: SEX-blind dithered quantization of the twin's "
                                                 "occupation output (income passed through)"},
    'IA_OSG_SPD_HI': {'kind': 'control', 'what': "SPEND path, upper bracket point: SEX-blind dithered quantization of the twin's "
                                                 "occupation output (income passed through)"},
    'IA_OSG_RO_LO': {'kind': 'control', 'what': "RO path, lower bracket point: the generic release-only tilt of the twin's occupation "
                                                "output (GENERIC_ZTILT; income passed through)"},
    'IA_OSG_RO_HI': {'kind': 'control', 'what': "RO path, upper bracket point: the generic release-only tilt of the twin's occupation "
                                                "output (GENERIC_ZTILT; income passed through)"},
    'IA_OSG_BLIND': {'kind': 'ablation', 'what': 'targeting off: the same occupation-only menu and models with the ladder ordered by '
                                                 "occupation task spend only (BLIND), first rung whose MODEL_FIT spend reaches CAND's"},
}
PATH_IDS = {'SPEND': {'LO': 'IA_OSG_SPD_LO', 'HI': 'IA_OSG_SPD_HI'}, 'RO': {'LO': 'IA_OSG_RO_LO', 'HI': 'IA_OSG_RO_HI'}}
PURPOSE = ('IA revision R1 OSG: occupation-only SEX-ordered field-generalization ladder, occupation models on generalized inputs, '
           'strength selection, matched occupation-restricted control paths (SBC, SPEND, RO) and the BLIND ablation on MODEL_FIT (SEX as '
           'a training label)')


# ---------------------------------------------------------------- ladder statistics and rungs (MODEL_FIT; labels arrive as arguments)
def occ_task_stats(Z, yO, unit, version, T_occ, m):
    """dT for every single step, priced in occupation nats only (IA_SFG_v1's task_stats, occupation term: same folds, settings, T)."""
    folds = FK.household_folds(unit, SF.N_LADDER_FOLDS, f'IPR_IA_ladder_S{int(version)}')
    base = SF._nll(SF._oof_fixed(Z, yO, SF.K_OCC, folds, T_occ, version), yO).mean()
    out = {}
    for st in SF.FIELD:
        d = SF._nll(SF._oof_fixed(SF.generalize(Z, [st], m), yO, SF.K_OCC, folds, T_occ, version), yO).mean() - base
        out[st] = {'dT': float(d), 'dT_occ': float(d)}
    return out


def fit_rung_occ(Z, steps, m, lab, unit, version):
    """The occupation model on h(Z) for the given steps by the twin's construction (T* by inner household CV, full refit,
    out-of-fold on NRM_FA_xf_S<v>); the income output stays the twin's."""
    prc = SF._prc()
    Zh = SF.generalize(Z, steps, m)
    tch = prc.fit_teachers(Zh, lab, unit, int(version), prc.HGB, ('occ',))
    return {'steps': list(steps), 'T': {'occ': int(tch['T']['occ'])}, 'models': {'occ': tch['models']['occ']},
            'oof_occ': tch['oof']['occ'], 'n_fits': (prc.N_INNER + 1, prc.N_XF)}


def release_osg(P, Z):
    """Deployment law of an `osg` payload: the twin's income vector on the raw record, the generalized occupation model on h(Z)."""
    prc = SF._prc()
    return prc._proba(P['twin_inc'], Z), prc._proba(P['occ'], SF.generalize(Z, P['steps_occ'], P['maps']))


def scan_occ(Z, ladder, m, lab, unit, version, tI, twin_L, cand_L, log=None):
    """Rungs k = 1 .. len(ladder) of an occupation-only ladder, stopping after the first rung whose spend reaches CAND's spend + 2
    household-clustered SE of the paired difference (IA_SFG_v1's scan_path rule)."""
    yI, yO = lab['income'], lab['occupation']
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    rungs, L = {}, {}
    for k in range(1, len(ladder) + 1):
        r = fit_rung_occ(Z, SF.compose(ladder, k), m, lab, unit, version)
        rungs[k], L[k] = r, FK.person_losses(tI, r['oof_occ'], yI, yO)
        d = L[k]['tot_nll'] - cand_L['tot_nll']
        sp = float((L[k]['tot_nll'] - twin_L['tot_nll']).mean())
        if log:
            log(f'scan rung {k} steps {r["steps"]} spend {sp:.5f} (cand {cand_sp:.5f})')
        if sp >= cand_sp + 2.0 * FK.cluster_se(d, unit):
            break
    return rungs, L


# ---------------------------------------------------------------- the occupation-restricted SPEND and RO paths of the shared twin
def restricted_paths(tI, tO, Z, yI, yO, sex, unit, version, cand_L, twin_L, log=None):
    """The registered DQ (SEX-blind) and generic RO path laws of the shared twin applied to the occupation output only; income passes
    through bitwise. Returns {'SPEND' | 'RO': {'bracket', 'points': {'LO' | 'HI': (strength, payload, (rI_oof, rO_oof))}}}."""
    from .. import ctlkit as CK
    from .. import paths as PA
    out = {}
    basis = CK.dq_fit(tI, tO)
    tag = f'IPR_DQ_S{int(version)}'
    dq_rel = {d: (tI, CK.dq_release(basis, tI, tO, Z, d, tag)[1]) for d in CK.DQ_GRID_FINE}
    dq_L = {d: FK.person_losses(*dq_rel[d], yI, yO) for d in CK.DQ_GRID_FINE}
    br = PA.bracket_points(dq_L, cand_L, twin_L, unit, tuple(CK.DQ_GRID_FINE))
    out['SPEND'] = {'bracket': br, 'points': {k: (g, {'op': 'dq', 'dq_basis': basis, 'delta': float(g), 'dq_tag': tag,
                                                      'outputs': 'occupation'}, dq_rel[g])
                                              for k, g in (('LO', br['lo']), ('HI', br['hi'])) if g is not None}}
    if log:
        log(f'SPEND (occupation DQ) bracket lo={br["lo"]} hi={br["hi"]} flag={br["flag"]} fineness_ok={br["fineness_ok"]}')
    rel, state, rec = CK.ro_path(tI, tO, sex, unit, version, tuple(CK.RO_GRID))
    rel_o = {a: (tI, rel[a][1]) for a in CK.RO_GRID}
    ro_L = {a: FK.person_losses(*rel_o[a], yI, yO) for a in CK.RO_GRID}
    brr = PA.bracket_points(ro_L, cand_L, twin_L, unit, tuple(CK.RO_GRID))
    out['RO'] = {'bracket': brr, 'record': rec,
                 'points': {k: (g, {'op': 'ro', 'ro_state': state, 'alpha': float(g), 'outputs': 'occupation'}, rel_o[g])
                            for k, g in (('LO', brr['lo']), ('HI', brr['hi'])) if g is not None}}
    if log:
        log(f'RO (occupation tilt) bracket lo={brr["lo"]} hi={brr["hi"]} flag={brr["flag"]} fineness_ok={brr["fineness_ok"]}')
    return out


def _registered_consistency(version, ts, tg):
    """Recomputed single-step statistics against those recorded by IA_SFG_v1's fit (ia/FIT_V<v>.json): max |difference|."""
    p = C.RESULTS / 'ia' / f'FIT_V{int(version)}.json'
    if not p.is_file():
        return {'status': 'NO_RECORD'}
    rec = json.loads(p.read_text())['selection']['single_step_stats']
    d_occ = max(abs(rec[s]['dT_occ'] - ts[s]['dT_occ']) for s in ts)
    d_i = max(abs(rec[s]['dI'] - tg[s]) for s in tg)
    return {'status': 'EQUAL' if max(d_occ, d_i) <= 1e-12 else 'DIFFERENT', 'max_abs_diff_dT_occ': d_occ, 'max_abs_diff_dI': d_i,
            'note': 'descriptive; never a stop'}


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, card, save, roles=None, log=print, record_dir=None, save_rich=True, budget=None, grid=GRID,
              check_registered=True):
    from ..ctl import twin as TW
    from .. import ctlkit as CK
    from .. import paths as PA
    t0 = time.process_time()
    prc = SF._prc()
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, tsha = TW.load_task_cache(version, row_sha)
    S, ssha = TW.load_sex_cache(version, row_sha)
    budget = budget or BUDGET
    tI, tO = T['oof']['inc'], T['oof']['occ']
    twin_L = FK.person_losses(tI, tO, yI, yO)
    m = SF.field_maps(Z)
    T_occ = int(T['T']['occ'])
    Tsex = int(S['T']['sex'])
    # 1. the ladder: occupation-priced task cost, IA_SFG_v1's SEX information
    ts = occ_task_stats(Z, yO, unit, version, T_occ, m)
    tg = SF.target_stats(Z, sex, unit, version, Tsex, m)
    ladder = SF.build_ladder(SF.merge_stats(ts, tg), 'target')
    consistency = _registered_consistency(version, ts, tg) if check_registered else {'status': 'NOT_CHECKED'}
    log(json.dumps({'ladder': ladder, 'consistency': consistency.get('status'), 'cpu_s': round(time.process_time() - t0, 1)}))
    # 2. candidate rungs, fitted in order; the scan stops at the first rung that fails the budget
    rungs, Lk, path_sp, k_star = {}, {}, {}, None
    for k in [g for g in grid if g <= len(ladder)]:
        rungs[k] = fit_rung_occ(Z, SF.compose(ladder, k), m, lab, unit, version)
        Lk[k] = FK.person_losses(tI, rungs[k]['oof_occ'], yI, yO)
        path_sp[k] = FK.spend(Lk[k], twin_L, unit, sex)
        ok = FK.meets(path_sp[k], budget)
        log(json.dumps({'rung': k, 'steps': rungs[k]['steps'], 'tot_nll': path_sp[k]['tot_nll'], 'meets_budget': ok,
                        'cpu_s': round(time.process_time() - t0, 1)}))
        if not ok:
            break
        k_star = k
    sel = {'ladder': ladder, 'single_step_stats': SF.merge_stats(ts, tg), 'grid_scanned': sorted(rungs), 'budget': budget,
           'budget_name': 'S3', 'k_star': k_star, 'path': {str(k): FK.jsonable_spend(sp) for k, sp in path_sp.items()},
           'steps': {str(k): rungs[k]['steps'] for k in rungs}, 'T': {str(k): rungs[k]['T'] for k in rungs},
           'registered_stats_consistency': consistency,
           'rule': 'per version: the largest rung k such that it and every smaller rung meet estimate + 2 household-clustered SE <= '
                   'S3 on the six components (MODEL_FIT out-of-fold over the twin, folds NRM_FA_xf_S<v>; the income components are '
                   'exactly zero because the income output is the twin\'s); rungs are fitted in order and the scan stops at the first '
                   'failing rung'}
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'maps': m,
            'task_cache_sha256': tsha, 'sex_cache_sha256': ssha}
    exposure = ('MODEL_FIT occupation and SEX labels of all 20,000 MODEL_FIT people reach the ladder order (SEX or a SEX-independent '
                'target as a training label), the occupation models on generalized inputs and the strength / bracket selection; MODEL_FIT '
                'income labels reach the selection (the income components, exactly zero) and, through the cached twin (inherited), the '
                'income output; for RO, MODEL_FIT SEX reaches the release-only surrogate and the tilt weights (cross-fitted); no '
                'CALIBRATION or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {'CTL_TWIN': (tI, tO)}

    def put(cfg, payload, oof_pair, extra, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(SF.predict, cfg, version, roles)
        sp = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure, people,
                        hh, time.process_time() - t0, {**extra, 'revision': REVISION, 'selection_k_star': k_star,
                                                       'model_sha256': meta['model_sha256'],
                                                       'oof_spend_over_twin': FK.jsonable_spend(sp)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'oof_spend_over_twin': FK.jsonable_spend(sp), **extra}
        oof_views[cfg] = oof_pair

    def put_rung(cfg, r, extra):
        put(cfg, {'op': 'osg', 'steps_occ': r['steps'], 'occ': r['models']['occ']}, (tI, r['oof_occ']),
            {'op': 'osg', 'steps_occ': r['steps'], 'T': r['T'], **extra}, r['n_fits'])

    out = {'version': int(version), 'revision': REVISION, 'selection': sel, 'task_cache_sha256': tsha, 'sex_cache_sha256': ssha}
    if k_star is None:
        log(json.dumps({'stop': 'no rung meets the budget: no candidate is released'}))
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'ia') / f'FIT_R1_V{int(version)}.json', out)
        return out
    cand_L = Lk[k_star]
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    put_rung('IA_OSG', rungs[k_star], {'k': k_star, 'ladder': ladder, 'strength': k_star})
    # 3. SBC path: the same occupation-only ladder rule with IA_SFG_v1's permuted SEX (PSEUDO2 only by the registered fallback rule)
    tries = []
    perm = SF.permuted_sex(sex, version)
    lad_p = SF.build_ladder(SF.merge_stats(ts, SF.target_stats(Z, perm, unit, version, Tsex, m)), 'target')
    r_p, L_p = scan_occ(Z, lad_p, m, lab, unit, version, tI, twin_L, cand_L, log=lambda s: log(s))
    br = PA.bracket_points(L_p, cand_L, twin_L, unit, tuple(sorted(L_p)))
    tries.append({'target': 'PERMUTED_SEX', 'ladder': lad_p, 'bracket': br})
    chosen = {'label': 'PERMUTED', 'rungs': r_p, 'ladder': lad_p, 'br': br}
    pseudo_info = None
    if br['flag'] is not None and max(L_p) == len(lad_p):
        make, pseudo_info = CK.pseudo2_select(Z, yI, yO, sex, np.clip(S['oof']['sex'][:, 1], 0.01, 0.99),
                                              np.asarray(data['row_id']), version)
        for g in CK.GAMMAS:
            a_ = make(g)
            lad_a = SF.build_ladder(SF.merge_stats(ts, SF.target_stats(Z, a_, unit, version, Tsex, m)), 'target')
            r_a, L_a = scan_occ(Z, lad_a, m, lab, unit, version, tI, twin_L, cand_L, log=lambda s: log(s))
            br_a = PA.bracket_points(L_a, cand_L, twin_L, unit, tuple(sorted(L_a)))
            tries.append({'target': f'PSEUDO2_gamma{g:g}', 'ladder': lad_a, 'bracket': br_a})
            chosen = {'label': 'PSEUDO2', 'rungs': r_a, 'ladder': lad_a, 'br': br_a}
            if br_a['flag'] is None:
                break
    br = chosen['br']
    for cfg, k in (('IA_OSG_SBC_LO', br['lo']), ('IA_OSG_SBC_HI', br['hi'])):
        if k is None:
            continue
        put_rung(cfg, chosen['rungs'][k], {'k': k, 'ladder': chosen['ladder'], 'target': chosen['label'], 'bracket_flag': br['flag'],
                                           'strength': k})
    log(json.dumps({'sbc': chosen['label'], 'lo': br['lo'], 'hi': br['hi'], 'flag': br['flag'], 'cpu_s': round(time.process_time() - t0, 1)}))
    # 4. SPEND and RO: the registered path laws restricted to the occupation output; saved under IA (this revision's controls)
    sr = restricted_paths(tI, tO, Z, yI, yO, sex, unit, version, cand_L, twin_L, log=lambda s: log(s))
    path_points = {}
    for key in ('SPEND', 'RO'):
        path_points[key] = {}
        for side, (g, payload, oof_pair) in sr[key]['points'].items():
            cfg = PATH_IDS[key][side]
            put(cfg, payload, oof_pair, {'op': payload['op'], 'outputs': 'occupation', 'path': key, 'strength': float(g),
                                         'bracket_flag': sr[key]['bracket']['flag']}, (0, 0))
            path_points[key][side] = cfg
    # 5. ablation BLIND: the occupation-only ladder by occupation task spend only, first rung whose spend reaches CAND's
    lad_b = SF.build_ladder(SF.merge_stats(ts), 'blind')
    r_b, L_b = {}, {}
    k_b = None
    for k in range(1, len(lad_b) + 1):
        r_b[k] = fit_rung_occ(Z, SF.compose(lad_b, k), m, lab, unit, version)
        L_b[k] = FK.person_losses(tI, r_b[k]['oof_occ'], yI, yO)
        if float((L_b[k]['tot_nll'] - twin_L['tot_nll']).mean()) >= cand_sp:
            k_b = k
            break
    blind_flag = None
    if k_b is None:
        k_b, blind_flag = max(r_b), 'BLIND_MAY_NOT_REACH_CAND_SPEND'
    put_rung('IA_OSG_BLIND', r_b[k_b], {'k': k_b, 'ladder': lad_b, 'target': 'NONE_TASK_SPEND_ORDER', 'flag': blind_flag, 'strength': k_b})
    blind = {'ladder': lad_b, 'k': k_b, 'flag': blind_flag,
             'scan': {str(k): float((L_b[k]['tot_nll'] - twin_L['tot_nll']).mean()) for k in L_b}}
    fineness = {'SBC': SF._fineness_inputs(br, cand_sp), 'SPEND': SF._fineness_inputs(sr['SPEND']['bracket'], cand_sp),
                'RO': SF._fineness_inputs(sr['RO']['bracket'], cand_sp)}
    # 6. descriptive MODEL_FIT-only checks: cross-fitted readers and the data-rich readers (never a stop)
    t1 = time.process_time()
    check = FK.mf_reader_check(oof_views, sex, unit, f'IPR_mfcheck_IA_R1_S{int(version)}')
    rich_info = None
    if save_rich:
        from .. import richreaders as RR
        rich, rich_cpu = RR.train(oof_views, sex, version)
        rich_info = {'sha256': RR.save(FAMILY, version, rich, {'configs': sorted(oof_views), 'revision': REVISION,
                                                               'trained_on': 'MODEL_FIT out-of-fold releases, MODEL_FIT SEX'},
                                       rev=REVISION),
                     'cpu_s': rich_cpu, 'readers_of': f'{FAMILY}:{REVISION}'}
    out.update({'saved': saved, 'sbc_tries': tries, 'sbc_construction': chosen['label'], 'pseudo_info': pseudo_info, 'blind': blind,
                'spend_path_bracket': sr['SPEND']['bracket'], 'ro_path_bracket': sr['RO']['bracket'], 'ro_record': sr['RO']['record'],
                'path_points': path_points, 'fineness_inputs': fineness, 'cand_spend': cand_sp, 'rich_readers': rich_info,
                'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers (household 5-fold cross-fitted); never a decision '
                                            'input and never a stop; the decisions use the MODEL_SELECT grid',
                                    'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'ia') / f'FIT_R1_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- checks (no label)
def selftest():
    """Invented-data checks: the occupation-priced ladder, the occupation-restricted path pass-through and the osg release law."""
    from .. import ctlkit as CK
    from .. import paths as PA
    stats = {st: {'dT': (0.001 * (i + 1)) * (-1 if st == 'AGE10' else 1), 'dT_occ': 0.001 * (i + 1), 'dI': 0.0005 * ((7 * i) % 5)}
             for i, st in enumerate(SF.FIELD)}
    L = SF.build_ladder(stats, 'target')
    ok = {'ladder_covers_fields': sorted(SF.compose(L, len(L))) == sorted(lv[-1] for lv in SF.MENU.values()),
          'negative_cost_floored': all(np.isfinite([stats[s]['dI'] / max(stats[s]['dT'], 1e-5) for s in SF.FIELD]))}
    rng = np.random.default_rng(3)
    n = 400
    a = rng.dirichlet(np.ones(2), n)
    o = rng.dirichlet(np.ones(5), n)
    Z = np.column_stack([rng.choice(np.linspace(-1.8, 1.7, 40), n), rng.choice(np.linspace(-3.5, 5.2, 87), n),
                         rng.integers(0, 8, n), rng.choice([1, 3, 4] + list(range(6, 25)), n), rng.integers(0, 5, n),
                         rng.choice([1, 12, 36, 72, 110, 303, 999], n)]).astype(float)
    basis = CK.dq_fit(a, o)
    P = {'op': 'dq', 'dq_basis': basis, 'delta': 0.5, 'dq_tag': 'TEST', 'outputs': 'occupation'}
    rI, rO = PA.predict_path(P, a, o, Z)
    full = CK.dq_release(basis, a, o, Z, 0.5, 'TEST')
    ok['dq_income_passthrough_bitwise'] = bool(np.array_equal(rI, a))
    ok['dq_occupation_equals_registered_law'] = bool(np.array_equal(rO, full[1]))
    P2 = dict(P); P2.pop('outputs')
    ok['dq_unrestricted_unchanged'] = all(np.array_equal(x, y) for x, y in zip(PA.predict_path(P2, a, o, Z), full))
    from sklearn.ensemble import HistGradientBoostingClassifier
    yi, yo = rng.integers(0, 2, n), rng.integers(0, 5, n)
    mi = HistGradientBoostingClassifier(max_iter=5).fit(Z, yi)
    m = SF.field_maps(Z)
    steps = ['MAR1', 'WK3']
    mo = HistGradientBoostingClassifier(max_iter=5).fit(SF.generalize(Z, steps, m), yo)
    pi, po = release_osg({'twin_inc': mi, 'occ': mo, 'steps_occ': steps, 'maps': m}, Z)
    prc = SF._prc()
    ok['osg_income_is_twin'] = bool(np.array_equal(pi, prc._proba(mi, Z)))
    Zh = SF.generalize(Z, steps, m)
    _, inv = np.unique(Zh, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    first = np.unique(inv, return_index=True)[1]
    ok['osg_occupation_function_of_h'] = bool(np.allclose(po, po[first][inv]))
    ok['all_ok'] = all(ok.values())
    return ok


def selftest_stage(n=2500, seed=5):
    """End-to-end check of stage_fit on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read):
    the cached teachers are replaced by teachers fitted on the invented labels, model files go to a temporary directory, nothing is
    saved to the campaign's runs and no record is written."""
    import tempfile
    from pathlib import Path as _P
    from ..ctl import twin as TW
    prc = SF._prc()
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    sex = rng.integers(0, 2, n)
    Zf = FK.compact(F['X'][idx], F['feature_names'])
    data = {'X': F['X'][idx], 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': ((Zf[:, 1] + 0.5 * sex + rng.normal(0, 1, n)) > 0.5).astype(np.int64),
            'occupation': np.minimum((Zf[:, 3] > 20).astype(int) + rng.integers(0, 4, n) * (rng.random(n) < 0.5) + sex + (Zf[:, 4] > 1),
                                     4).astype(np.int64),
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
    old = (C.RUNS, TW.load_task_cache, TW.load_sex_cache, prc.HGB)
    tmp = _P(tempfile.mkdtemp(prefix='ipr_ia_osg_selftest_'))
    try:
        C.RUNS = tmp
        TW.load_task_cache = lambda v, rs=None: (T, 'invented')
        TW.load_sex_cache = lambda v, rs=None: (S, 'invented')
        prc.HGB = hp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s: None, record_dir=False, save_rich=False,
                        budget={k: 100.0 for k in BUDGET}, grid=(1, 2, 3), check_registered=False)
    finally:
        C.RUNS, TW.load_task_cache, TW.load_sex_cache, prc.HGB = old
    twin_inc = prc._proba(T['models']['inc'], FK.compact(data['X'], data['feature_names']))
    ok = {'cand_saved': 'IA_OSG' in saved, 'k_star_on_grid': out['selection']['k_star'] in (2, 3),
          'sbc_saved': any(c.startswith('IA_OSG_SBC') for c in saved),
          'paths_saved': any(c.startswith('IA_OSG_SPD') for c in saved) and any(c.startswith('IA_OSG_RO') for c in saved),
          'ablation_saved': 'IA_OSG_BLIND' in saved,
          'records_valid': all(r['kind'] == CONFIGS[c]['kind'] for c, (_, r) in saved.items()),
          'income_is_twin_everywhere': all(np.array_equal(a['MODEL_FIT']['income'], twin_inc) for a, _ in saved.values()),
          'income_spend_zero': all(abs(r['oof_spend_over_twin']['inc_nll']['estimate']) == 0.0 for _, r in saved.values())}
    a = saved['IA_OSG'][0]['MODEL_FIT']
    ok['cand_valid_vectors'] = bool(np.allclose(a['income'].sum(1), 1) and np.allclose(a['occupation'].sum(1), 1))
    st = out['selection']['steps'][str(out['selection']['k_star'])]
    m = SF.field_maps(Z)
    Zh = SF.generalize(Z, st, m)
    _, inv = np.unique(Zh, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    first = np.unique(inv, return_index=True)[1]
    ok['occupation_is_function_of_h'] = bool(np.allclose(a['occupation'], a['occupation'][first][inv]))
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['k_star'] = out['selection']['k_star']
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ipr.ia.osg')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd in ('selftest', 'selftest_stage'):
        r = {'selftest': selftest, 'selftest_stage': selftest_stage}[a.cmd]()
        print(json.dumps(r, default=str))
        return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'k_star': out['selection']['k_star'], 'saved': sorted(out['saved']), 'sbc': out.get('sbc_construction'),
                      'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

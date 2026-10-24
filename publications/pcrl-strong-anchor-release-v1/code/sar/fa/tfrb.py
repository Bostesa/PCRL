"""FA revision R2: the cross-fit BAG of the occupation-only SUCC-anchored residual (TFRB; card draft roles/investigator_a/CARD_DRAFT_R2.md,
to be committed as cards/FA_TFRB_R2.md). PCRL campaign `pcrl_strong_anchor_release_v1`; EXPLORATORY DEVELOPMENT on previously used
public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation. ADAPTIVE_R2: designed after the FA_TFRO_R1
MODEL_SELECT reading (aggregate facts from the coordinator's adjudication, disclosed as adaptive); no new MODEL_SELECT computation.

Diagnosis (card section 0): v1's guard failure is a SEX-contrast failure (MODEL_SELECT men-minus-women change over SUCC +0.0145 against
the MODEL_FIT out-of-fold contrast -0.0001, se 0.0014; the CALIBRATION level -0.0039 against about -0.0012 on MODEL_SELECT), and the
contrast was measured on the five fold models while a sixth run (the full fit) was deployed. The fold-averaged contrasts were near 0 at
every rho while the deployed runs' MODEL_SELECT contrasts had opposite signs in the two versions (-0.0052 / +0.0145): which SEX group pays
the critic cost is run-specific. lambda = 0 runs are near-unique (label-free check: bag vs full fit KL 4-7e-5 against a movement of
0.022-0.027), so the variability belongs to the critic-trained runs.
Mechanism change (structural): what is DEPLOYED is the bag of the K = 5 household-fold residuals whose guard contrast is measured:
  occupation = softmax(centre(z_S + (1/5) sum_k r_k(phi))), r_k = the R1 occupation-only residual fitted on MODEL_FIT minus fold k with
  R1's fold seed (tfro.train, identical recipe and keys); income = SUCC's vector bitwise. No separate full fit at lambda > 0.
  - the MODEL_FIT out-of-fold contrast (each person scored by the one fold model that did not see them) is then a first-order unbiased
    estimate of the deployed bag's contrast (the bag averages the same five models), and averaging shrinks run-specific asymmetry;
  - its SE is prospectively widened to cover model (run) variance: SE_C = max(household-clustered SE, SD of the five per-fold contrasts
    / sqrt(5)); the CALIBRATION level L and part (a) are measured on the deployed bag (out of sample there).
Rule: R1's (SUCC's cap margins from the committed anchor table; parts (a) and (b); rho in increasing order, first failure stops) with the
two changes above. Controls: CTRL = R1's lambda = 0 path (origin START; the lambda = 0 full fit, bitwise identical to R1's, checked on
the trained weights before any save) bracketed on CALIBRATION total NLL against R2's CAND; points R1 already saved (its LO point and
FA_TFRO_R1_CONT at s = 1) are reused as aliases at no new specification; CONT = FA_TFRO_R1_CONT (reused, gating).
Specifications: CAND 2 + at most 4 new CTRL points = 6 at most.

Commands:
    <repository>/.venv/bin/python -m sar.fa.tfrb selftest
    <repository>/.venv/bin/python -m sar.fa.tfrb preflight --version V
    <repository>/.venv/bin/python -m sar.fa.tfrb fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FA_TFRB_R2.md
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
from .. import famrun as FR
from . import tfr as V1                      # FA_TFR_v1 helpers (unchanged)
from . import tfro as R1                     # FA_TFRO_R1 recipe, keys, margins, guard algebra (unchanged)

FAMILY = 'FA'
REVISION = 'R2'
PREFIX = 'FA_TFRB_R2'
R1_CONT = 'FA_TFRO_R1_CONT'
RHO_GRID = R1.RHO_GRID
S_PATH = V1.S_PATH
K = FR.N_XF
CONFIGS = {
    PREFIX: {'kind': 'candidate', 'role': 'CAND',
             'what': "cross-fit bag of the occupation-only SUCC-anchored residual: SUCC's H3 income bitwise; the occupation output "
                     "corrected from SUCC's JOINT logits by the mean of the five household-fold residuals (R1 recipe and fold seeds) at "
                     'the selected rho*'},
    PREFIX + '_TASK_LO': {'kind': 'control', 'role': 'CTRL',
                          'what': "CTRL lower bracket point on R1's lambda = 0 path (the full lambda = 0 fit, bitwise R1's): the largest "
                                  "CALIBRATION paired total-NLL spend over SUCC that is <= CAND's (income bitwise SUCC)"},
    PREFIX + '_TASK_HI': {'kind': 'control', 'role': 'CTRL',
                          'what': "CTRL upper bracket point on R1's lambda = 0 path: the smallest CALIBRATION paired total-NLL spend over "
                                  "SUCC that is >= CAND's + 2 SE of its paired difference against CAND (income bitwise SUCC)"},
}
PURPOSE = ('FA TFRB R2: the cross-fit bag of the occupation-only residual with the joint SEX critics (MODEL_FIT SEX), the per-fold SEX '
           'contrasts of its guard estimate and R1\'s lambda = 0 control path on MODEL_FIT')
PURPOSE_CAL = ('FA TFRB R2: CALIBRATION occupation (income unchanged by construction) for the task part of the selection, the level of '
               'the guard estimate and the CTRL bracket (paired spends over SUCC; no fit on CALIBRATION; identical exposure for controls)')


# ---------------------------------------------------------------- the bag (deployment algebra; numpy only)
def bag_output(nets, phis, A):
    return np.mean([V1.net_numpy(W, V1.phi_apply(ph, A)) for W, ph in zip(nets, phis)], 0)


def release_payload(P, A, succ):
    """(income, occupation) for a payload: income SUCC's bitwise; 'nets' = the bag, 'net' = one residual (CTRL points); scale 0 = START."""
    inc = np.array(succ['income'], np.float64, copy=True)
    if float(P['scale']) == 0.0:
        return inc, np.array(succ['occupation'], np.float64, copy=True)
    out = bag_output(P['nets'], P['phis'], A) if 'nets' in P else V1.net_numpy(P['net'], V1.phi_apply(P['phi'], A))
    return inc, R1.release_occ(A, out, P['scale'])


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (sar.deploy contract; routed by sar.fa); deterministic."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    if P.get('frozen_sha256') and V1.frozen_sha(version) != P['frozen_sha256']:
        raise PermissionError("SUCC's frozen files differ from those recorded at fit time")
    succ, A = V1.teacher_arrays(P, X, names)
    pi, po = release_payload(P, A, succ)
    return {'income': pi, 'occupation': po}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- the guard estimate with run (model) variance
def guard_estimate_bag(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit, folds):
    """R1's estimate (Delta_men = L + (1 - p) C, Delta_women = L - p C) with SE_C = max(household-clustered SE, SD of the K per-fold
    contrasts / sqrt(K)): each fold's contrast uses the one fold model that did not see those people, so the between-fold spread carries
    the run-to-run variance that the household SE ignores."""
    g = R1.guard_estimate(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit)
    d = np.asarray(mf_c['occ_brier']) - np.asarray(mf_s['occ_brier'])
    men = np.asarray(sex) == 0
    per = []
    for k in np.unique(folds):
        f = folds == k
        per.append(float(d[f & men].mean() - d[f & ~men].mean()))
    se_between = float(np.std(per, ddof=1) / np.sqrt(len(per)))
    se_C = max(g['se_C'], se_between)
    p = g['p_men']
    out = dict(g)
    out.update({'C_per_fold': per, 'se_C_household': g['se_C'], 'se_C_between': se_between, 'se_C': se_C})
    out['men'] = (g['L_cal'] + (1 - p) * g['C_mf'], float(np.sqrt(g['se_L'] ** 2 + ((1 - p) * se_C) ** 2)))
    out['women'] = (g['L_cal'] - p * g['C_mf'], float(np.sqrt(g['se_L'] ** 2 + (p * se_C) ** 2)))
    return out


def evaluate_rho(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit, folds, margins):
    sp = FK.spend(cal_c, cal_s, ucal)
    if abs(sp['inc_nll'][0]) > 0 or abs(sp['inc_brier'][0]) > 0 or abs(sp['inc_acc'][0]) > 0:
        raise AssertionError("the income output must be SUCC's bitwise")
    a = {k: sp[k][0] + 2.0 * sp[k][1] <= margins[k] for k in ('occ_nll', 'occ_brier', 'occ_acc')}
    g = guard_estimate_bag(cal_c, cal_s, ucal, mf_c, mf_s, sex, unit, folds)
    b = {k: g[k][0] + 2.0 * g[k][1] <= margins[k] for k in ('men', 'women')}
    row = {'cal_spend_over_SUCC': FK.jsonable_spend(sp), 'part_a': a, 'part_b': b, 'margins': margins,
           'guard_estimate': {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()}}
    return all(a.values()) and all(b.values()), row


# ---------------------------------------------------------------- R1's saved lambda = 0 points (reuse at no new specification)
def r1_saved_points(version):
    """{s: config id} of the lambda = 0 points FA_TFRO_R1 saved in this version (from its committed fit record), with CONT at s = 1."""
    rec = json.loads((C.RESULTS / 'fa' / f'FIT_R1_V{int(version)}.json').read_text())
    out = {1.0: R1_CONT}
    for side in ('LO', 'HI'):
        x = ((rec.get('ctrl') or {}).get('saved') or {}).get(side) or {}
        if x.get('config'):
            out[float(x['s'])] = x['config']
    return out


def weights_equal(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, cal, card, save, roles=None, log=print, record_dir=None, epochs=None, rho_grid=RHO_GRID, margins=None,
              succ_fn=None, cache=None, frozen=None, r1_points=None, r1_model=None):
    t0 = time.process_time()
    names = tuple(map(str, data['feature_names']))
    X = np.asarray(data['X'], np.float64)
    unit = np.asarray(data['unit'])
    lab = {k: np.asarray(data[k], np.int64) for k in ('income', 'occupation', 'sex')}
    T, tsha = cache if cache is not None else V1.task_cache(version)
    if T.get('row_id_sha256') is not None and 'row_id' in data:
        from nrm.fa import prc
        if T['row_id_sha256'] != prc.array_sha(np.asarray(data['row_id'])):
            raise PermissionError('the cached task teachers were fitted on other rows')
    margins = margins or R1.succ_margins(version)
    frozen = V1.frozen_sha(version) if frozen is None else frozen
    r1_points = r1_points if r1_points is not None else r1_saved_points(version)
    r1_model = r1_model or (lambda cfg: FK.load_model(FAMILY, cfg, version))
    sfn = succ_fn or (lambda v, X_, n_: V1._succ(v, X_, n_))
    succ = sfn(version, X, names)
    A = V1.blocks(FK.compact(X, names), (succ['income'], succ['occupation']), (T['oof']['inc'], T['oof']['occ']))
    succ_pair = (np.asarray(succ['income'], np.float64), np.asarray(succ['occupation'], np.float64))
    A_cal, succ_cal = V1._cal_arrays(cal, version, T, sfn)
    ucal = np.asarray(cal['unit'])
    yIc, yOc = np.asarray(cal['income'], np.int64), np.asarray(cal['occupation'], np.int64)
    L_cal_s = FK.person_losses(*succ_cal, yIc, yOc)
    L_mf_s = FK.person_losses(*succ_pair, lab['income'], lab['occupation'])
    folds = FR.folds(unit, FAMILY, version)                       # the v1 / R1 folds (SAR_xf_FA_S<v>)
    n = len(unit)
    # 1. rho in increasing order: the five fold fits (R1's keys) give the out-of-fold occupation and the deployed bag
    rows, rho_star, cand = {}, None, {}
    for r in rho_grid:
        occ_oof, nets, phis, lams = np.zeros((n, 5)), [], [], []
        for j in range(K):
            tr, te = folds != j, folds == j
            st = R1.train(A[tr], lab['sex'][tr], r, R1._key(version, unit[tr], f'rho{r:g}|f{j}'), epochs=epochs)
            W = st['checkpoints'][st['keep']]
            occ_oof[te] = R1.release_occ(A[te], V1.net_numpy(W, V1.phi_apply(st['phi'], A[te])), 1.0)
            nets.append(W); phis.append(st['phi']); lams.append(st['info']['lambda'])
        P_bag = {'nets': nets, 'phis': phis, 'scale': 1.0}
        cal_c = FK.person_losses(*release_payload(P_bag, A_cal, {'income': succ_cal[0], 'occupation': succ_cal[1]}), yIc, yOc)
        mf_c = FK.person_losses(succ_pair[0], occ_oof, lab['income'], lab['occupation'])
        ok, row = evaluate_rho(cal_c, L_cal_s, ucal, mf_c, L_mf_s, lab['sex'], unit, folds, margins)
        row.update({'lambda_per_fold': lams, 'passes': ok,
                    'mf_oof_spend_over_SUCC_biased': FK.jsonable_spend(FK.spend(mf_c, L_mf_s, unit, lab['sex']))})
        rows[str(r)] = row
        log(json.dumps({'rho': r, 'passes': ok, 'part_a': row['part_a'], 'part_b': row['part_b'],
                        'cpu_s': round(time.process_time() - t0, 1)}))
        if not ok:
            break
        rho_star, cand = r, {'P': P_bag, 'occ_oof': occ_oof, 'cal_L': cal_c, 'lams': lams}
    selection = {'rho_grid': list(rho_grid), 'rho_star': rho_star, 'per_rho': rows, 'margins': margins,
                 'rule': "per version: rho in increasing order; the largest rho such that it and every smaller rho pass (a) the "
                         "deployed bag's CALIBRATION paired occupation spends over SUCC (estimate + 2 SE <= SUCC's margins to the caps) and "
                         "(b) Delta_g = L_CAL(bag) +/- weight x C_MF (out-of-fold, the bag's own fold models), estimate + 2 SE <= SUCC's "
                         "guard margins, SE_C = max(household SE, between-fold SD / sqrt(5)); the scan stops at the first failing rho",
                 'calibration_exposure': "income is SUCC's bitwise; the occupation side of SUCC (JOINT_TREE), the HGB teachers and the five "
                                         'fold residuals are out of sample on CALIBRATION'}
    out = {'version': int(version), 'family': FAMILY, 'revision': REVISION, 'adaptive_tag': 'ADAPTIVE_R2', 'task_cache_sha256': tsha,
           'frozen_sha256': frozen, 'selection': selection,
           'cand': {'config': PREFIX, 'rho_star': rho_star, 'status': 'FITTED' if rho_star is not None else 'NO_FEASIBLE_STRENGTH'},
           'ctrl': None, 'cont': {'config': R1_CONT, 'reused_from': 'FA_TFRO_R1', 'gating': True, 'scale': 1.0, 'rho': 0.0}}
    if rho_star is None:
        log(json.dumps({'stop': 'rho = 0.3 fails: no candidate and no control is released'}))
        out.update({'saved': {}, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_R2_V{int(version)}.json', out)
        return out
    # 2. R1's lambda = 0 path (the full lambda = 0 fit with R1's key): checked against R1's saved model before any save
    full0 = R1.train(A, None, 0.0, R1._key(version, unit, 'rho0|full'), epochs=epochs)
    W0 = full0['checkpoints'][full0['keep']]
    reuse = {}
    for s, cfg in sorted(r1_points.items()):
        reuse[cfg] = weights_equal(W0, r1_model(cfg)['net'])
    if not all(reuse.values()):
        out.update({'saved': {}, 'stop': 'REUSE_NOT_BITWISE', 'reuse_check': reuse, 'cpu_s': time.process_time() - t0})
        log(json.dumps({'stop': "R1's lambda = 0 fit is not reproduced bitwise: nothing is saved", 'reuse_check': reuse}))
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_R2_V{int(version)}.json', out)
        return out
    path_L = {s: FK.person_losses(*R1.release_state(full0, A_cal, succ_cal[0], s, succ_cal[1]), yIc, yOc) for s in S_PATH}
    br = V1.bracket(path_L, cand['cal_L'], L_cal_s, ucal)
    ids = {}
    for side in ('LO', 'HI'):
        s = br[side.lower()]
        if s is None:
            ids[side] = {'config': None, 's': None, 'alias': None}
        elif float(s) == 0.0:
            ids[side] = {'config': None, 's': 0.0, 'alias': 'START'}
        elif float(s) in r1_points:
            ids[side] = {'config': None, 's': float(s), 'alias': r1_points[float(s)]}
        else:
            ids[side] = {'config': f'{PREFIX}_TASK_{side}', 's': float(s), 'alias': None}
        if s is not None:
            ids[side].update({k: br['path'][str(s)][k] for k in ('spend_over_start', 'se_diff_vs_cand')})
    roles = roles or FK.role_features()
    people, hh = len(X), len(np.unique(unit))
    exposure = ('MODEL_FIT SEX of all 20,000 MODEL_FIT people reaches the five fold residuals through the joint critics (each fold model: '
                'MODEL_FIT minus one household fold) and the per-fold SEX contrasts of the guard estimate; no task label reaches the '
                'residuals (distillation toward the admitted HGB occupation teacher, whose out-of-fold values carry MODEL_FIT occupation '
                'labels); CALIBRATION income and occupation labels (never SEX) reach only the task part of the selection, the level of '
                "the guard estimate and the CTRL bracket, identically for the controls; income is SUCC's H3 vector bitwise; MODEL_SELECT "
                'aggregate outcomes of FA_TFRO_R1 informed the design (ADAPTIVE_R2); no MODEL_SELECT label is read by this fit')
    base = {'feature_names': names, 'hgb_inc': T['models']['inc'], 'hgb_occ': T['models']['occ'], 'task_cache_sha256': tsha,
            'succ_config': V1.SUCC_CONFIG, 'version': int(version), 'frozen_sha256': frozen, 'revision': REVISION, 'outputs': 'occupation'}
    saved = {}

    def put(cfg, payload, extra, mf_occ, cal_L, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        spo = FK.spend(FK.person_losses(succ_pair[0], mf_occ, lab['income'], lab['occupation']), L_mf_s, unit, lab['sex'])
        csp = FK.spend(cal_L, L_cal_s, ucal)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits,
                        'MODEL_FIT: income, occupation, sex; CALIBRATION: income, occupation', exposure, people, hh,
                        time.process_time() - t0, {**extra, 'revision': REVISION, 'adaptive_tag': 'ADAPTIVE_R2',
                                                   'role': CONFIGS[cfg]['role'], 'gating': False, 'model_sha256': meta['model_sha256'],
                                                   'selection_rho_star': rho_star, 'cal_spend_over_SUCC': FK.jsonable_spend(csp),
                                                   'mf_oof_spend_over_SUCC_biased': FK.jsonable_spend(spo)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'cal_spend_over_SUCC': FK.jsonable_spend(csp), **extra}

    out['cand']['cal_spend_over_SUCC'] = {**br['cand'], 'all': FK.jsonable_spend(FK.spend(cand['cal_L'], L_cal_s, ucal))}
    put(PREFIX, {**cand['P'], 'train_info': {'lambda_per_fold': cand['lams'], 'rho': rho_star, 'bag': K}},
        {'op': 'tfrb', 'rho': rho_star, 'lambda_per_fold': cand['lams'], 'scale': 1.0, 'strength': rho_star, 'bag_size': K},
        cand['occ_oof'], cand['cal_L'], (0, K, 0, K))
    for side in ('LO', 'HI'):
        if ids[side]['config'] is not None:
            s = ids[side]['s']
            occ_mf = R1.release_state(full0, A, succ_pair[0], s, succ_pair[1])[1]
            put(ids[side]['config'], {'net': W0, 'phi': full0['phi'], 'scale': s, 'train_info': {'rho': 0.0, 'lambda': 0.0}},
                {'op': 'tfro_lambda0', 'rho': 0.0, 'scale': s, 'strength': s, 'bracket_side': side, 'bracket_flags': br['flags']},
                occ_mf, path_L[s], (1, 0, 0, 0))
    out['ctrl'] = {**{k: br[k] for k in ('origin', 'selection_data', 'key', 'path', 'lo', 'hi', 'flags', 'rule')}, 'saved': ids,
                   'path_source': "R1's lambda = 0 full fit (bitwise; checked on the trained weights)", 'reuse_check': reuse,
                   'r1_points': {str(k): v for k, v in r1_points.items()}}
    t1 = time.process_time()
    views = {'SUCC': succ_pair, PREFIX: (succ_pair[0], cand['occ_oof'])}
    check = FK.mf_reader_check(views, lab['sex'], unit, f'SAR_mfcheck_FA_R2_S{int(version)}')
    out.update({'saved': saved, 'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers; never a decision input; SUCC is '
                                                            'in-sample on MODEL_FIT', 'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fa') / f'FIT_R2_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- checks on invented data (no label file, no real teacher)
def selftest(n=1600, epochs=3):
    import tempfile
    from pathlib import Path as _P
    t0 = time.process_time()
    ok = {}
    Zc, yi, yo, sex, unit, succ, hgb = V1._invented(n, 11)
    A = V1.blocks(Zc, succ, hgb)
    # the bag algebra: a one-model bag equals that model; the bag is the mean of the residual outputs
    st = R1.train(A, sex, 1.0, 'r2-a', epochs=epochs)
    st2 = R1.train(A, sex, 1.0, 'r2-b', epochs=epochs)
    W1, W2 = st['checkpoints'][st['keep']], st2['checkpoints'][st2['keep']]
    one = bag_output([W1], [st['phi']], A)
    ok['bag_of_one_is_model'] = bool(np.array_equal(one, V1.net_numpy(W1, V1.phi_apply(st['phi'], A))))
    two = bag_output([W1, W2], [st['phi'], st2['phi']], A)
    ok['bag_is_mean'] = bool(np.allclose(two, 0.5 * (V1.net_numpy(W1, V1.phi_apply(st['phi'], A)) + V1.net_numpy(W2, V1.phi_apply(st2['phi'], A)))))
    inc, occ = release_payload({'nets': [W1, W2], 'phis': [st['phi'], st2['phi']], 'scale': 1.0}, A, {'income': succ[0], 'occupation': succ[1]})
    ok['income_bitwise_succ'] = bool(np.array_equal(inc, succ[0]))
    z = release_payload({'nets': [W1], 'phis': [st['phi']], 'scale': 0.0}, A, {'income': succ[0], 'occupation': succ[1]})
    ok['scale0_is_start'] = bool(np.array_equal(z[1], succ[1]))
    # the guard SE covers between-fold (run) variance: invented per-fold asymmetries of alternating sign
    rng = np.random.default_rng(5)
    nm = 5000
    s_ = rng.integers(0, 2, nm)
    u_ = np.arange(nm) // 2
    f_ = FK.household_folds(u_, 5, 'r2-selftest')
    base = rng.normal(0.6, 0.05, nm)
    shift = np.array([0.006, -0.006, 0.006, -0.006, 0.0])[f_]
    mf_c = {'occ_brier': base + np.where(s_ == 0, shift, -shift)}
    mf_s = {'occ_brier': base}
    cal_s = {'occ_brier': base[:1500]}
    cal_c = {'occ_brier': base[:1500] + 0.0}
    g = guard_estimate_bag(cal_c, cal_s, u_[:1500], mf_c, mf_s, s_, u_, f_)
    ok['between_fold_se_dominates'] = g['se_C_between'] > 3 * g['se_C_household'] and g['se_C'] == g['se_C_between']
    g0 = guard_estimate_bag(cal_c, cal_s, u_[:1500], {'occ_brier': base + 0.001}, mf_s, s_, u_, f_)
    ok['no_run_variance_keeps_household_se'] = g0['se_C'] >= g0['se_C_household'] and abs(g0['C_mf']) < 1e-12
    # end-to-end stage on invented data with fake R1 records / models (temporary model directory, fake save)
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    X = np.column_stack([Zc, np.zeros((n, 2))])
    names = tuple(f'f{i}' for i in range(X.shape[1]))
    data = {'X': X, 'feature_names': names, 'unit': unit, 'income': yi, 'occupation': yo, 'sex': sex}
    Zc2, yi2, yo2, _, unit2, succ2, _ = V1._invented(600, 12)
    X2 = np.column_stack([Zc2, np.zeros((600, 2))])
    cal = {'X': X2, 'feature_names': names, 'unit': unit2, 'income': yi2, 'occupation': yo2}
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from nrm.fa import prc
    mi = HGB(max_iter=10).fit(Zc, yi); mo = HGB(max_iter=10).fit(Zc, yo)
    Tc = {'models': {'inc': mi, 'occ': mo}, 'oof': {'inc': prc._proba(mi, Zc), 'occ': prc._proba(mo, Zc)}}
    table = {X.tobytes(): succ, X2.tobytes(): succ2}
    big = {k: 100.0 for k in ('occ_nll', 'occ_brier', 'occ_acc', 'men', 'women')}
    A_true = V1.blocks(Zc, succ, (Tc['oof']['inc'], Tc['oof']['occ']))
    W0_true = R1.train(A_true, None, 0.0, R1._key(0, unit, 'rho0|full'), epochs=epochs)
    good_model = lambda cfg: {'net': W0_true['checkpoints'][W0_true['keep']]}
    bad_model = lambda cfg: {'net': [w + 1.0 for w in W0_true['checkpoints'][W0_true['keep']]]}
    old = (C.RUNS, FK.compact, V1._succ)
    tmp = _P(tempfile.mkdtemp(prefix='sar_fa_tfrb_selftest_'))
    try:
        C.RUNS = tmp
        FK.compact = lambda X_, n_: np.asarray(X_, np.float64)[:, :6]
        V1._succ = lambda v, X_, n_: dict(zip(('income', 'occupation'), table[np.asarray(X_, np.float64).tobytes()]))
        kw = dict(roles={'MODEL_FIT': {'X': X, 'feature_names': names}}, log=lambda s_: None, record_dir=False, epochs=epochs,
                  rho_grid=(0.3, 1.0), margins=big, cache=(Tc, 'invented'), frozen={})
        res = stage_fit(0, data, cal, 'invented-card', fake_save, r1_points={1.0: R1_CONT}, r1_model=good_model, **kw)
        again = predict(PREFIX, 0, X, names)
        ok['deploy_bitwise'] = all(np.array_equal(again[t], saved[PREFIX][0]['MODEL_FIT'][t]) for t in ('income', 'occupation'))
        ok['saved_income_bitwise_succ'] = all(np.array_equal(a['MODEL_FIT']['income'], succ[0]) for a, _ in saved.values())
        from . import predict as route
        ok['package_routes_r2'] = all(np.array_equal(route(PREFIX, 0, X, names)[t], again[t]) for t in ('income', 'occupation'))
        ok['bag_payload'] = len(FK.load_model(FAMILY, PREFIX, 0)['nets']) == K
        for side in ('LO', 'HI'):
            cfg = res['ctrl']['saved'][side]['config']
            if cfg is not None:
                P = FK.load_model(FAMILY, cfg, 0)
                ok[f'ctrl_{side}_is_r1_lambda0_fit'] = weights_equal(P['net'], W0_true['checkpoints'][W0_true['keep']])
        n_before = len(saved)
        res_bad = stage_fit(0, data, cal, 'invented-card', fake_save, r1_points={1.0: R1_CONT}, r1_model=bad_model, **kw)
        ok['refuses_non_bitwise_reuse'] = res_bad.get('stop') == 'REUSE_NOT_BITWISE' and len(saved) == n_before
        strict = dict(big, men=-1.0)
        res_none = stage_fit(0, data, cal, 'invented-card', fake_save, r1_points={1.0: R1_CONT}, r1_model=good_model,
                             **{**kw, 'margins': strict})
        ok['first_failure_stops'] = res_none['cand']['status'] == 'NO_FEASIBLE_STRENGTH' and list(res_none['selection']['per_rho']) == ['0.3']
        # without R1 aliases the bracket points are saved as new CTRL configurations: they must be R1's lambda = 0 fit, deployed bitwise
        C.RUNS = _P(tempfile.mkdtemp(prefix='sar_fa_tfrb_selftest2_'))
        saved2 = {}
        res3 = stage_fit(0, data, cal, 'invented-card', lambda f_, c_, v_, a_, r_: saved2.__setitem__(c_, (a_, r_)), r1_points={},
                         r1_model=good_model, **kw)
        new_ctrl = [res3['ctrl']['saved'][s]['config'] for s in ('LO', 'HI') if res3['ctrl']['saved'][s]['config']]
        W0t = W0_true['checkpoints'][W0_true['keep']]
        ok['new_ctrl_point_is_r1_lambda0_fit'] = bool(new_ctrl) and all(weights_equal(FK.load_model(FAMILY, c_, 0)['net'], W0t)
                                                                        for c_ in new_ctrl)
        ok['new_ctrl_deploys_bitwise'] = all(np.array_equal(predict(c_, 0, X, names)[t], saved2[c_][0]['MODEL_FIT'][t])
                                             for c_ in new_ctrl for t in ('income', 'occupation'))
        ok['new_ctrl_income_bitwise_succ'] = all(np.array_equal(saved2[c_][0]['MODEL_FIT']['income'], succ[0]) for c_ in new_ctrl)
    finally:
        C.RUNS, FK.compact, V1._succ = old
    ids = res['ctrl']['saved']
    want = {PREFIX} | {ids[s]['config'] for s in ('LO', 'HI') if ids[s]['config']}
    ok['saved_declared_only'] = set(saved) == want and len(saved) <= 3
    ok['cont_reused_not_saved'] = R1_CONT not in saved and res['cont']['reused_from'] == 'FA_TFRO_R1'
    ok['rho_star_largest_feasible'] = res['selection']['rho_star'] == 1.0
    ok['record_fields'] = (res['cand']['status'] == 'FITTED' and {'spend_over_start', 'se'} <= set(res['cand']['cal_spend_over_SUCC'])
                           and res['ctrl']['selection_data'] == 'CALIBRATION' and res['ctrl']['origin'].startswith('START')
                           and all({'C_per_fold', 'se_C_between', 'se_C_household'} <= set(r['guard_estimate'])
                                   for r in res['selection']['per_rho'].values()))
    ok['all_ok'] = all(ok.values())
    ok['cpu_s'] = round(time.process_time() - t0, 1)
    ok['ctrl_points_in_stage_test'] = {s: ids[s] for s in ('LO', 'HI')}
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sar.fa.tfrb')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    p = sub.add_parser('preflight')
    p.add_argument('--version', type=int, required=True)
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest()
        print(json.dumps(r, default=str))
        return 0 if r['all_ok'] else 1
    if a.cmd == 'preflight':
        r = V1.preflight(a.version)
        r['margins'] = R1.succ_margins(a.version)
        r['r1_points'] = {str(k): v for k, v in r1_saved_points(a.version).items()}
        r['r1_models_present'] = {cfg: all(p_.is_file() for p_ in FK.model_paths(FAMILY, cfg, a.version))
                                  for cfg in r1_saved_points(a.version).values()}
        r['ok'] = r['ok'] and all(r['r1_models_present'].values())
        print(json.dumps(r, indent=1, default=str))
        return 0 if r['ok'] else 1
    pf = V1.preflight(a.version)
    if not pf['ok']:
        raise PermissionError(f'frozen inputs not pinned / changed: missing {pf["missing"]}, mismatched {pf["mismatched"]}')
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    cal = C.load('CALIBRATION', ('features', 'income', 'occupation'), 'calibrate', PURPOSE_CAL, card=a.card)
    out = stage_fit(a.version, data, cal, a.card, RL.save)
    print(json.dumps({'rho_star': out['selection']['rho_star'], 'saved': sorted(out['saved']), 'stop': out.get('stop'),
                      'ctrl_flags': (out['ctrl'] or {}).get('flags'), 'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

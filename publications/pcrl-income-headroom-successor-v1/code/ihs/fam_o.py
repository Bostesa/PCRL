"""Family O (investigator O): occupation-side margin repair of the inherited JOINT_TREE occupation output. PCRL income-headroom
successor development campaign: EXPLORATORY DEVELOPMENT on previously used records, not confirmation.

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (development roles of the
protected-tree replication). The service predicts each person's income class (2 classes) and occupation group (5 classes) from
six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) and gives each recipient only its own float64
probability vector and argmax decision. SEX is the dataset's recorded demographic column (sex0 = SEX code 1, men; sex1 = SEX
code 2, women); readers are separate evaluation classifiers that estimate SEX from the released probability vectors.

Release of every configuration (computed from X only; no SEX, no label, no other recipient's hidden quantity at runtime):
- income: H_LJ's own income output, LOCAL_TREE E20 (frozen T_TREE routing + stored stage-B theta), unchanged, so the only
  difference from H_LJ is the occupation output;
- occupation: p' = softmax(z / T + b), with z the centred raw score (= centred log-ratio of the released probability vector) of
  the occupation base output, z = (1 - w) * z_JOINT_TREE_E20 + w * z_<blend_arm>_E20 (w = 0 unless a configuration says
  otherwise), T > 0 one temperature and b a centred per-class bias. For w = 0 and T > 0 the map p -> p' is a bijection of the
  open simplex (distinct vectors stay distinct; p = softmax(T * (clr(log p') - b)) exactly).

Fitting roles (PROTOCOL.md section 3): T (and, for the bias control, b) by overall occupation NLL on CALIBRATION (features +
occupation; no SEX). The SEX-balanced bias of O1_SEXBAL on MODEL_FIT (features + occupation + SEX as a training-time label only)
against the stored U0 MODEL_FIT occupation output: minimax of the two SEX groups' occupation summed-Brier excess vs U0, found as
the SEX weight lambda at which b(lambda) = argmin lambda * mean_men Brier + (1 - lambda) * mean_women Brier equalizes the two
excesses. MODEL_FIT people are in-sample for the teacher routing, the JOINT_TREE corrections and U0; MODEL_FIT values are in-sample
diagnostics, never held-out guard estimates. MODEL_SELECT labels are read only by `python3 -m ihs.devrun score`.

    python3 -m ihs.fam_o selftest                          # invented data only -> roles/investigator_o/INVENTED_DATA_CHECK.json
    python3 -m ihs.fam_o fit --seed 0 --config O1_SEXBAL   # -> .private/.../runs/O/<config>/S<seed>/{RELEASE.npz, RELEASE.json}
    python3 -m ihs.fam_o verify --seed 0 --config O1_SEXBAL  # predict() from features of all six roles == saved release, bitwise
"""
from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

from acs_replication import trees

FAMILY = 'O'
OCC = 'occupation'
K_OCC = 5
INCOME_ARM = 'LOCAL_TREE'        # H_LJ's own income output
BASE_ARM = 'JOINT_TREE'          # the inherited (H_LJ) occupation output
LABEL = 'E20'
OPT = {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11}

# Configuration registry (appended as revisions occur; CARD.md records the reason for each).
CONFIGS = {
    'OC_TCAL': {
        'version': 'O1', 'kind': 'control', 'w': 0.0, 'blend_arm': None, 'matched_control': 'H_LJ (unmodified occupation output)',
        'temperature': 'CALIBRATION_NLL', 'bias': 'NONE',
        'description': 'CONTROL (named by the brief): JOINT_TREE E20 occupation output with one global temperature fitted by overall '
                       'occupation NLL on CALIBRATION (no SEX); income = LOCAL_TREE E20 (H_LJ income)',
    },
    'OC_TBCAL': {
        'version': 'O1', 'kind': 'control', 'w': 0.0, 'blend_arm': None, 'matched_control': 'OC_TCAL',
        'temperature': 'CALIBRATION_NLL_WITH_BIAS', 'bias': 'CALIBRATION_NLL',
        'description': 'CONTROL: JOINT_TREE E20 occupation output with a temperature and a per-class bias fitted jointly by overall '
                       'occupation NLL on CALIBRATION (ordinary bias calibration, no SEX); income = LOCAL_TREE E20 (H_LJ income)',
    },
    'O1_SEXBAL': {
        'version': 'O1', 'kind': 'fit', 'w': 0.0, 'blend_arm': None, 'matched_control': 'OC_TCAL (same temperature, no bias); also OC_TBCAL (bias without SEX)',
        'temperature': 'CALIBRATION_NLL', 'bias': 'MODEL_FIT_SEX_MINIMAX_BRIER',
        'description': 'JOINT_TREE E20 occupation output, temperature as OC_TCAL (CALIBRATION NLL), then a per-class bias fitted on '
                       'MODEL_FIT to equalize the two SEX groups\' occupation summed-Brier excess vs U0 (minimax; SEX a training-time '
                       'label only); income = LOCAL_TREE E20 (H_LJ income)',
    },
    # Attribution control added after the seed-0/seed-1 O1 fits showed lambda* within about 0.001 of each other (near the men's
    # share): the same role, loss and parametrization as O1_SEXBAL with the SEX weighting removed (unweighted Brier on MODEL_FIT).
    'OC_BFIT': {
        'version': 'O1', 'kind': 'control', 'w': 0.0, 'blend_arm': None, 'matched_control': 'O1_SEXBAL (identical except no SEX weighting)',
        'temperature': 'CALIBRATION_NLL', 'bias': 'MODEL_FIT_BRIER_UNWEIGHTED',
        'description': 'CONTROL: JOINT_TREE E20 occupation output, temperature as OC_TCAL (CALIBRATION NLL), then a per-class bias fitted '
                       'by unweighted occupation summed Brier on MODEL_FIT (no SEX: ordinary in-sample bias recalibration); income = '
                       'LOCAL_TREE E20 (H_LJ income)',
    },
}


# ---------------------------------------------------------------- the recalibration map (shared by fit and predict)
def clr(raw):
    raw = np.asarray(raw, dtype=np.float64)
    return raw - raw.mean(1, keepdims=True)


def center(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return b - b.mean()


def base_z(z_base, z_blend, w):
    """Centred occupation score of the base output, optionally blended (w = 0: the JOINT_TREE score itself, bitwise)."""
    if w == 0.0 or z_blend is None:
        return z_base
    return (1.0 - w) * z_base + w * z_blend


def transform(z, T, b):
    """p' = softmax(z / T + b), float64 (trees.probabilities centres and normalizes)."""
    return trees.probabilities(OCC, np.asarray(z, dtype=np.float64) / float(T) + center(b)[None, :])


def release_from_raw(inc_raw, occ_raw, blend_raw, prm):
    """The release from the arms' raw scores and the stored parameters (the one code path used by fit and predict)."""
    z = base_z(clr(occ_raw), None if blend_raw is None else clr(blend_raw), float(prm['w']))
    return {'income': trees.probabilities('income', inc_raw), OCC: transform(z, prm['T'], prm['b'])}


# ---------------------------------------------------------------- metrics, losses and analytic gradients
def onehot(y, k=K_OCC):
    return (np.asarray(y)[:, None] == np.arange(k)[None, :]).astype(np.float64)


def brier_vec(p, y):
    return ((np.asarray(p, dtype=np.float64) - onehot(y, p.shape[1])) ** 2).sum(1)


def nll_vec(p, y):
    """As scm_bench.common.task_vectors (probability clipped at 1e-12)."""
    return -np.log(np.maximum(np.asarray(p, dtype=np.float64)[np.arange(len(y)), y], 1e-12))


def _unpack(theta, fit_T, fit_b, T_fixed, b_fixed):
    i = 0
    if fit_T:
        T = float(np.exp(theta[0])); i = 1
    else:
        T = float(T_fixed)
    b = np.concatenate(([0.0], np.asarray(theta[i:i + K_OCC - 1], dtype=np.float64))) if fit_b else np.asarray(b_fixed, dtype=np.float64)
    return T, b


def loss_grad(theta, z, y, wts, metric, fit_T, fit_b, T_fixed=1.0, b_fixed=None):
    """Weighted mean loss (NLL or summed Brier) of softmax(z / T + b) and its gradient in theta = ([log T], [b_1..b_4]); b_0 = 0."""
    b_fixed = np.zeros(K_OCC) if b_fixed is None else b_fixed
    T, b = _unpack(theta, fit_T, fit_b, T_fixed, b_fixed)
    u = z / T + b[None, :]
    u = u - u.max(1, keepdims=True)
    e = np.exp(u); s = e.sum(1, keepdims=True); p = e / s
    Y = onehot(y)
    W = np.asarray(wts, dtype=np.float64); W = W / W.sum()
    if metric == 'nll':
        lv = np.log(s[:, 0]) - u[np.arange(len(y)), y]
        gu = p - Y
    elif metric == 'brier':
        r = p - Y
        lv = (r ** 2).sum(1)
        gu = 2.0 * p * (r - (p * r).sum(1, keepdims=True))
    else:
        raise ValueError(metric)
    L = float(W @ lv)
    gu = gu * W[:, None]
    g = []
    if fit_T:
        g.append(float(np.sum(gu * (-z / T))))
    if fit_b:
        g.extend(gu.sum(0)[1:].tolist())
    return L, np.asarray(g, dtype=np.float64)


def _solve(z, y, wts, metric, fit_T, fit_b, T_fixed=1.0, b_fixed=None, theta0=None):
    n_par = int(fit_T) + (K_OCC - 1) * int(fit_b)
    theta0 = np.zeros(n_par) if theta0 is None else np.asarray(theta0, dtype=np.float64)
    res = minimize(loss_grad, theta0, args=(z, y, wts, metric, fit_T, fit_b, T_fixed, b_fixed), jac=True, method='L-BFGS-B',
                   bounds=([(-4.0, 4.0)] if fit_T else []) + [(-20.0, 20.0)] * ((K_OCC - 1) * int(fit_b)), options=OPT)
    T, b = _unpack(res.x, fit_T, fit_b, T_fixed, np.zeros(K_OCC) if b_fixed is None else b_fixed)
    gnorm = float(np.max(np.abs(res.jac))) if n_par else 0.0
    return T, center(b), {'converged': bool(res.success), 'message': str(res.message), 'nit': int(res.nit), 'loss': float(res.fun),
                          'max_abs_grad': gnorm, 'theta': res.x.tolist()}


def fit_temperature(z, y, wts=None):
    wts = np.ones(len(y)) if wts is None else wts
    T, b, info = _solve(z, y, wts, 'nll', True, False)
    return T, info


def fit_temperature_bias(z, y, wts=None):
    wts = np.ones(len(y)) if wts is None else wts
    T, b, info = _solve(z, y, wts, 'nll', True, True)
    return T, b, info


def group_excess(p, y, ref_brier, sex):
    br = brier_vec(p, y) - ref_brier
    return float(br[sex == 0].mean()), float(br[sex == 1].mean())


def minimax_bias(z, y, sex, ref_brier, T, iters=60, tol=1e-12):
    """Per-class bias b (T fixed) minimizing max over SEX groups of the occupation summed-Brier excess vs the reference.

    b(lambda) = argmin_b lambda * mean_men Brier + (1 - lambda) * mean_women Brier; lambda* by bisection on the excess gap
    (men minus women), which falls as lambda rises. Returns b(lambda*), and the bisection trace (aggregates only)."""
    men, women = sex == 0, sex == 1
    trace, fits = [], [0]
    warm = {'theta': np.zeros(K_OCC - 1)}

    def at(lam):
        wts = np.where(men, lam / men.sum(), (1.0 - lam) / women.sum())
        _, b, info = _solve(z, y, wts, 'brier', False, True, T_fixed=T, theta0=warm['theta'])
        fits[0] += 1
        warm['theta'] = np.asarray(info['theta'])
        em, ew = group_excess(transform(z, T, b), y, ref_brier, sex)
        trace.append({'lambda': float(lam), 'excess_men': em, 'excess_women': ew, 'gap': em - ew, 'converged': info['converged'],
                      'max_abs_grad': info['max_abs_grad'], 'b': b.tolist()})
        return b, em - ew

    b1, g1 = at(1.0)
    if g1 >= 0.0:
        return b1, {'lambda_star': 1.0, 'status': 'MEN_ONLY_OPTIMUM_STILL_HIGHER', 'trace': trace, 'n_inner_fits': fits[0]}
    warm['theta'] = np.zeros(K_OCC - 1)
    b0, g0 = at(0.0)
    if g0 <= 0.0:
        return b0, {'lambda_star': 0.0, 'status': 'WOMEN_ONLY_OPTIMUM_STILL_HIGHER', 'trace': trace, 'n_inner_fits': fits[0]}
    lo, hi, best = 0.0, 1.0, (b0, g0, 0.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        bm, gm = at(mid)
        if abs(gm) < abs(best[1]):
            best = (bm, gm, mid)
        if gm > 0.0:
            lo = mid
        else:
            hi = mid
        if abs(gm) < tol or hi - lo < 1e-13:
            break
    gaps = [t['gap'] for t in sorted(trace, key=lambda t: t['lambda'])]
    return best[0], {'lambda_star': float(best[2]), 'status': 'EQUALIZED', 'final_gap': float(best[1]), 'trace': trace,
                     'n_inner_fits': fits[0], 'gap_monotone_in_lambda': bool(all(a >= b - 1e-9 for a, b in zip(gaps, gaps[1:])))}


# ---------------------------------------------------------------- real-data fit (registered roles only)
def _cluster_stats(d, unit, mask=None):
    from . import evaluate as EV
    d = np.asarray(d, dtype=np.float64)
    if mask is None:
        return float(d.mean()), EV.cluster_se(d, unit)
    return float(d[mask].mean()), EV.subgroup_se(d, mask, unit)


def _occ_diag(p, y, ref, unit, sex=None):
    """Aggregate occupation diagnostics vs a reference output (no person-level values)."""
    out = {}
    for m, f in (('nll', nll_vec), ('brier', brier_vec)):
        d = f(p, y) - f(ref, y)
        v, se = _cluster_stats(d, unit)
        out[f'{m}_excess'] = {'value': v, 'se': se}
    out['accuracy_loss'] = dict(zip(('value', 'se'), _cluster_stats((ref.argmax(1) == y).astype(float) - (p.argmax(1) == y).astype(float), unit)))
    if sex is not None:
        d = brier_vec(p, y) - brier_vec(ref, y)
        for g, name in ((0, 'men_sex0'), (1, 'women_sex1')):
            v, se = _cluster_stats(d, unit, sex == g)
            out[f'brier_excess_{name}'] = {'value': v, 'se': se}
    return out


def params_path(seed, config_id):
    from . import devrun as D
    return D.release_dir(FAMILY, config_id, seed) / 'PARAMS.json'


def load_params(seed, config_id):
    return json.loads(params_path(seed, config_id).read_text())


def predict(seed, config_id, X, names, prm=None):
    """Recompute the release from the admitted 145-column encoding of the six permitted raw fields only."""
    from . import bases as B
    prm = load_params(seed, config_id) if prm is None else prm
    inc_raw = B.base_raw(seed, prm['income_arm'], X, names, prm['label'], tasks=('income',))['income']
    occ_raw = B.base_raw(seed, prm['occupation_arm'], X, names, prm['label'], tasks=(OCC,))[OCC]
    blend_raw = None
    if float(prm['w']) != 0.0:
        blend_raw = B.base_raw(seed, prm['blend_arm'], X, names, prm['label'], tasks=(OCC,))[OCC]
    return release_from_raw(inc_raw, occ_raw, blend_raw, prm)


def predict_occupation(seed, config_id, X, names, prm=None):
    """The family-O occupation output alone (for combination with another family's income output)."""
    from . import bases as B
    prm = load_params(seed, config_id) if prm is None else prm
    occ_raw = B.base_raw(seed, prm['occupation_arm'], X, names, prm['label'], tasks=(OCC,))[OCC]
    blend_raw = None
    if float(prm['w']) != 0.0:
        blend_raw = B.base_raw(seed, prm['blend_arm'], X, names, prm['label'], tasks=(OCC,))[OCC]
    z = base_z(clr(occ_raw), None if blend_raw is None else clr(blend_raw), float(prm['w']))
    return transform(z, prm['T'], prm['b'])


def fit(seed, config_id):
    from . import common as C
    from . import bases as B
    from . import devrun as D
    from . import evaluate as EV
    cfg = CONFIGS[config_id]
    if not C.registered():
        raise PermissionError('registration not committed')
    t0 = time.process_time()
    roles = C.ROLES6
    inc = B.base_outputs(seed, INCOME_ARM, roles, LABEL)          # recomputed from features; checked vs the stored E20 outputs
    occ = B.base_outputs(seed, BASE_ARM, roles, LABEL)
    blend = B.base_outputs(seed, cfg['blend_arm'], roles, LABEL) if cfg['w'] != 0.0 else None
    z = {r: base_z(clr(occ[r]['raw'][OCC]), None if blend is None else clr(blend[r]['raw'][OCC]), float(cfg['w'])) for r in roles}
    n_fits, info = 0, {}

    # CALIBRATION: features + occupation (no SEX)
    Cal = C.load_role('CALIBRATION', ('features', OCC))
    if not np.array_equal(Cal['row_id'], occ['CALIBRATION']['row_id']):
        raise AssertionError('CALIBRATION row order differs')
    yc, zc = Cal[OCC], z['CALIBRATION']
    if cfg['temperature'] == 'CALIBRATION_NLL':
        T, info['temperature_fit'] = fit_temperature(zc, yc); n_fits += 1
        b = np.zeros(K_OCC)
    elif cfg['temperature'] == 'CALIBRATION_NLL_WITH_BIAS':
        T, b, info['temperature_bias_fit'] = fit_temperature_bias(zc, yc); n_fits += 1
    else:
        raise ValueError(cfg['temperature'])

    # MODEL_FIT: features + occupation + SEX (training-time label for O1's bias; in-sample subgroup diagnostics for every config)
    M = C.load_role('MODEL_FIT', ('features', OCC, 'sex'))
    sa = C.stage_a_outputs(seed)
    if not (np.array_equal(M['row_id'], occ['MODEL_FIT']['row_id']) and np.array_equal(sa['row_id__MODEL_FIT'], M['row_id'])):
        raise AssertionError('MODEL_FIT row order differs')
    u0_fit = B.stored_reference(seed, 'U0', 'MODEL_FIT')[OCC]
    ym, zm, sm = M[OCC], z['MODEL_FIT'], M['sex']
    sex_in_fit = cfg['bias'] == 'MODEL_FIT_SEX_MINIMAX_BRIER'
    if sex_in_fit:
        b, info['minimax'] = minimax_bias(zm, ym, sm, brier_vec(u0_fit, ym), T)
        n_fits += info['minimax']['n_inner_fits']
    elif cfg['bias'] == 'MODEL_FIT_BRIER_UNWEIGHTED':
        _, b, info['bias_fit'] = _solve(zm, ym, np.ones(len(ym)), 'brier', False, True, T_fixed=T)
        n_fits += 1
    elif cfg['bias'] not in ('NONE', 'CALIBRATION_NLL'):
        raise ValueError(cfg['bias'])
    info['MODEL_FIT_men_share'] = float(np.mean(sm == 0))

    prm = {'income_arm': INCOME_ARM, 'occupation_arm': BASE_ARM, 'label': LABEL, 'w': float(cfg['w']), 'blend_arm': cfg['blend_arm'],
           'T': float(T), 'b': [float(v) for v in center(b)], 'map': 'occupation = softmax(clr(raw_base) / T + b); income = LOCAL_TREE E20'}
    outs = {r: release_from_raw(inc[r]['raw']['income'], occ[r]['raw'][OCC],
                                None if blend is None else blend[r]['raw'][OCC], prm) for r in roles}
    fit_cpu = time.process_time() - t0

    # aggregate diagnostics (no person-level values): in-sample MODEL_FIT, CALIBRATION, and identity checks against H_LJ
    base_fit = occ['MODEL_FIT']['prob'][OCC]
    sa_u0_cal = B.stored_reference(seed, 'U0', 'CALIBRATION')[OCC]
    sa_ucal_cal = B.stored_reference(seed, 'Ucal', 'CALIBRATION')[OCC]
    diag = {
        'MODEL_FIT_in_sample_vs_U0': {'H_LJ_occupation': _occ_diag(base_fit, ym, u0_fit, M['unit'], sm),
                                      config_id: _occ_diag(outs['MODEL_FIT'][OCC], ym, u0_fit, M['unit'], sm)},
        'CALIBRATION_vs_U0': {'H_LJ_occupation': _occ_diag(occ['CALIBRATION']['prob'][OCC], yc, sa_u0_cal, Cal['unit']),
                              config_id: _occ_diag(outs['CALIBRATION'][OCC], yc, sa_u0_cal, Cal['unit'])},
        'CALIBRATION_vs_Ucal': {'H_LJ_occupation': _occ_diag(occ['CALIBRATION']['prob'][OCC], yc, sa_ucal_cal, Cal['unit']),
                                config_id: _occ_diag(outs['CALIBRATION'][OCC], yc, sa_ucal_cal, Cal['unit'])},
        'note': 'MODEL_FIT people are in-sample for the teacher routing, the stage-B corrections and U0; CALIBRATION is the fitting '
                'role of the temperature (and of the control bias). Neither is a held-out guard estimate.',
    }
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], occ['MODEL_SELECT']['row_id']):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    hlj = bank['H_LJ']['MODEL_SELECT']
    diag['identity_vs_H_LJ_MODEL_SELECT'] = {
        'income_bitwise_equal': bool(np.array_equal(outs['MODEL_SELECT']['income'], hlj['income'])),
        'income_max_abs_diff': float(np.max(np.abs(outs['MODEL_SELECT']['income'] - hlj['income']))),
        'base_occupation_bitwise_equal': bool(np.array_equal(occ['MODEL_SELECT']['prob'][OCC], hlj[OCC])),
        'occupation_max_abs_change_vs_H_LJ': float(np.max(np.abs(outs['MODEL_SELECT'][OCC] - hlj[OCC]))),
        'occupation_mean_abs_change_vs_H_LJ': float(np.mean(np.abs(outs['MODEL_SELECT'][OCC] - hlj[OCC]))),
        'decision_changes_vs_H_LJ': int(np.sum(outs['MODEL_SELECT'][OCC].argmax(1) != hlj[OCC].argmax(1))),
        'distinct_occupation_vectors': {'H_LJ': int(np.unique(hlj[OCC], axis=0).shape[0]),
                                        config_id: int(np.unique(outs['MODEL_SELECT'][OCC], axis=0).shape[0])},
    }
    diag['predicted_class_shares'] = {r: {'base': occ[r]['prob'][OCC].mean(0).tolist(), config_id: outs[r][OCC].mean(0).tolist()}
                                      for r in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'DEV_REPORT')}
    record = {
        'mechanism': cfg['description'],
        'version': cfg['version'], 'kind': cfg['kind'], 'matched_control': cfg.get('matched_control'),
        'inputs_used': 'admitted 145-column encoding X of AGEP, WKHP, COW, SCHL, MAR, POBP (all roles); no SEX, no label at runtime',
        'roles_read': ('CALIBRATION (features, occupation: temperature' + (' and bias' if cfg['bias'] == 'CALIBRATION_NLL' else '') + ' fit); '
                       'MODEL_FIT (features, occupation' + (' (bias fit, no SEX)' if cfg['bias'] == 'MODEL_FIT_BRIER_UNWEIGHTED' else '') + ', SEX: '
                       + ('training-time label of the SEX-balanced bias and ' if sex_in_fit else '')
                       + 'in-sample subgroup diagnostics; stored U0 MODEL_FIT occupation output); features of MODEL_SELECT, ATTACK_FIT, '
                       'ATTACK_VALIDATION, DEV_REPORT (label-free, release computation); frozen bank H_LJ MODEL_SELECT outputs (identity check, '
                       'no label)'),
        'labels_used_at_training': ('CALIBRATION occupation' + ('; MODEL_FIT occupation and SEX (bias)' if sex_in_fit else '')
                                    + ('; MODEL_FIT occupation (bias; SEX not used)' if cfg['bias'] == 'MODEL_FIT_BRIER_UNWEIGHTED' else '')),
        'n_model_fits': int(n_fits), 'fit_cpu_s': float(fit_cpu), 'params': prm, 'fit_info': info, 'diagnostics': diag,
        'release_contract': 'float64 probability vectors; decisions = argmax; income bitwise H_LJ income when income_bitwise_equal',
    }
    params_path(seed, config_id).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    params_path(seed, config_id).write_text(json.dumps(prm, indent=1, sort_keys=True) + '\n')
    d = D.save_release(FAMILY, config_id, seed, outs, record)
    (d / 'FIT_DIAG.json').write_text(json.dumps({'config_id': config_id, 'seed': seed, 'fit_info': info, 'diagnostics': diag},
                                                indent=1, sort_keys=True, default=float) + '\n')
    v = verify(seed, config_id)
    print(json.dumps({'config': config_id, 'seed': seed, 'T': prm['T'], 'b': prm['b'], 'n_model_fits': n_fits, 'fit_cpu_s': round(fit_cpu, 2),
                      'lambda_star': info.get('minimax', {}).get('lambda_star'), 'minimax_status': info.get('minimax', {}).get('status'),
                      'MODEL_FIT_men_share': info['MODEL_FIT_men_share'],
                      'MODEL_FIT_in_sample': {k: {g: diag['MODEL_FIT_in_sample_vs_U0'][k][g]['value'] for g in ('brier_excess_men_sex0', 'brier_excess_women_sex1', 'nll_excess', 'brier_excess')}
                                              for k in diag['MODEL_FIT_in_sample_vs_U0']},
                      'CALIBRATION_nll_excess_vs_Ucal': {k: diag['CALIBRATION_vs_Ucal'][k]['nll_excess']['value'] for k in diag['CALIBRATION_vs_Ucal']},
                      'identity': diag['identity_vs_H_LJ_MODEL_SELECT'], 'verify': v}, default=float))
    return d


def verify(seed, config_id):
    """predict() from the features of every role reproduces the saved release bitwise."""
    from . import common as C
    from . import bases as B
    from . import devrun as D
    rel, rec = D.load_release(FAMILY, config_id, seed)
    prm = load_params(seed, config_id)
    if prm != rec['params']:
        raise AssertionError('PARAMS.json differs from the release record')
    res = {}
    for r in C.ROLES6:
        X, names, rid, unit = B.load_X(r)
        p = predict(seed, config_id, X, names, prm)
        res[r] = (all(np.array_equal(p[t], rel[r][t]) for t in C.TASKS)
                  and np.array_equal(predict_occupation(seed, config_id, X, names, prm), rel[r][OCC]))
    if not all(res.values()):
        raise AssertionError(f'predict() differs from the saved release: {res}')
    return {'bitwise_equal_all_roles': True, 'roles': sorted(res)}


# ---------------------------------------------------------------- invented-data check (no real data)
def _sample(P, rng):
    """One categorical draw per row of P (inverse CDF; invented labels only)."""
    u = rng.uniform(size=len(P))
    return np.minimum((u[:, None] > np.cumsum(P, axis=1)).sum(1), P.shape[1] - 1).astype(np.int64)


def selftest(out_path=None):
    rng = np.random.default_rng(20261018)
    checks = {}

    # 1. bijection: p is recovered exactly from p' given (T, b); distinct vectors stay distinct
    n = 4000
    z = clr(rng.normal(0, 1.5, (n, K_OCC)))
    z[:200] = z[200:400]                                         # some exact duplicates
    T, b = 1.37, center([0.4, -0.3, 0.1, 0.25, -0.2])
    p = trees.probabilities(OCC, z)
    p2 = transform(z, T, b)
    z_rec = T * (clr(np.log(p2)) - b[None, :])
    p_rec = trees.probabilities(OCC, z_rec)
    checks['bijection'] = {'max_abs_z_recovery_error': float(np.max(np.abs(z_rec - z))),
                           'max_abs_p_recovery_error': float(np.max(np.abs(p_rec - p))),
                           'distinct_before': int(np.unique(p, axis=0).shape[0]), 'distinct_after': int(np.unique(p2, axis=0).shape[0])}
    checks['bijection']['pass'] = bool(checks['bijection']['max_abs_p_recovery_error'] < 1e-10
                                       and checks['bijection']['distinct_before'] == checks['bijection']['distinct_after'])

    # 2. probability contract under extreme scores
    ze = clr(rng.normal(0, 60, (500, K_OCC)))
    pe = transform(ze, 0.3, center([30, -30, 0, 5, -5]))
    checks['contract_extreme'] = {'finite': bool(np.isfinite(pe).all()), 'nonnegative': bool((pe >= 0).all()),
                                  'max_sum_error': float(np.max(np.abs(pe.sum(1) - 1)))}
    checks['contract_extreme']['pass'] = bool(checks['contract_extreme']['finite'] and checks['contract_extreme']['nonnegative']
                                              and checks['contract_extreme']['max_sum_error'] < 1e-12)

    # 3. analytic gradients vs central differences (NLL and Brier; T and b; non-uniform weights)
    m = 600
    zg = clr(rng.normal(0, 1.2, (m, K_OCC))); yg = rng.integers(0, K_OCC, m); wg = rng.uniform(0.2, 2.0, m)
    worst = 0.0
    for metric in ('nll', 'brier'):
        for fit_T, fit_b in ((True, False), (False, True), (True, True)):
            th = rng.normal(0, 0.3, int(fit_T) + 4 * int(fit_b))
            _, g = loss_grad(th, zg, yg, wg, metric, fit_T, fit_b, 1.3, center([0.1, 0, -0.1, 0.2, -0.2]))
            num = np.zeros_like(th)
            for j in range(len(th)):
                e = np.zeros_like(th); e[j] = 1e-6
                num[j] = (loss_grad(th + e, zg, yg, wg, metric, fit_T, fit_b, 1.3, center([0.1, 0, -0.1, 0.2, -0.2]))[0]
                          - loss_grad(th - e, zg, yg, wg, metric, fit_T, fit_b, 1.3, center([0.1, 0, -0.1, 0.2, -0.2]))[0]) / 2e-6
            worst = max(worst, float(np.max(np.abs(g - num)) / max(1e-8, float(np.max(np.abs(num))))))
    checks['gradients'] = {'max_relative_error': worst, 'pass': bool(worst < 1e-5)}

    # loss values agree with the evaluation metrics (scm_bench conventions)
    pv = transform(zg, 1.3, np.zeros(K_OCC))
    Lb, _ = loss_grad(np.array([np.log(1.3)]), zg, yg, np.ones(m), 'brier', True, False)
    Ln, _ = loss_grad(np.array([np.log(1.3)]), zg, yg, np.ones(m), 'nll', True, False)
    checks['loss_matches_metrics'] = {'brier_diff': abs(Lb - float(brier_vec(pv, yg).mean())), 'nll_diff': abs(Ln - float(nll_vec(pv, yg).mean()))}
    checks['loss_matches_metrics']['pass'] = bool(max(checks['loss_matches_metrics'].values()) < 1e-10)

    # 4. temperature and temperature+bias recovery on invented labels drawn from a known calibrated map
    N = 60000
    zt = clr(rng.normal(0, 1.4, (N, K_OCC)))
    T0, b0 = 1.6, center([0.3, -0.2, 0.1, 0.0, -0.2])
    yt = _sample(transform(zt, T0, np.zeros(K_OCC)), rng)
    That, _ = fit_temperature(zt, yt)
    yb = _sample(transform(zt, T0, b0), rng)
    Tb, bb, infb = fit_temperature_bias(zt, yb)
    checks['temperature_recovery'] = {'T_true': T0, 'T_hat': That, 'pass': bool(abs(That - T0) < 0.06)}
    checks['temperature_bias_recovery'] = {'T_true': T0, 'T_hat': Tb, 'b_true': b0.tolist(), 'b_hat': bb.tolist(),
                                           'max_abs_b_error': float(np.max(np.abs(bb - b0))), 'converged': infb['converged']}
    checks['temperature_bias_recovery']['pass'] = bool(abs(Tb - T0) < 0.06 and checks['temperature_bias_recovery']['max_abs_b_error'] < 0.06)

    # 5. minimax SEX-balanced bias on an invented SEX-blind base output that is miscalibrated in opposite directions per group
    Nm = 40000
    sex = (rng.uniform(size=Nm) < 0.47).astype(np.int64)               # 1 = invented "women" (47%), 0 = invented "men"
    zs = clr(rng.normal(0, 1.2, (Nm, K_OCC)))
    shift = np.where(sex[:, None] == 0, 1.0, -1.0) * np.array([0.5, -0.3, 0.0, 0.4, -0.6])[None, :]
    ys = _sample(trees.probabilities(OCC, zs + shift), rng)
    ref = trees.probabilities(OCC, zs + 0.6 * shift)                   # invented "U0" that partly knows the groups
    base_p = trees.probabilities(OCC, zs)
    ref_b = brier_vec(ref, ys)
    em0, ew0 = group_excess(base_p, ys, ref_b, sex)
    # make the invented base output start like the real one: men worse than women (add a women-favouring bias first)
    zs2 = zs + center([-0.25, 0.15, 0.0, -0.2, 0.3])[None, :]
    em1, ew1 = group_excess(trees.probabilities(OCC, zs2), ys, ref_b, sex)
    bmm, mm = minimax_bias(zs2, ys, sex, ref_b, 1.0)
    em2, ew2 = group_excess(transform(zs2, 1.0, bmm), ys, ref_b, sex)
    # brute-force check of the minimax value along random directions around the solution
    worse = 0
    for _ in range(40):
        d = center(rng.normal(0, 0.05, K_OCC))
        a, c = group_excess(transform(zs2, 1.0, bmm + d), ys, ref_b, sex)
        worse += int(max(a, c) >= max(em2, ew2) - 1e-12)
    checks['minimax'] = {'invented_base_excess_men': em1, 'invented_base_excess_women': ew1, 'after_men': em2, 'after_women': ew2,
                         'lambda_star': mm['lambda_star'], 'status': mm['status'], 'n_inner_fits': mm['n_inner_fits'],
                         'gap_monotone_in_lambda': mm.get('gap_monotone_in_lambda'), 'random_perturbations_not_better': f'{worse}/40',
                         'unadjusted_symmetric_base': {'men': em0, 'women': ew0}}
    checks['minimax']['pass'] = bool(mm['status'] == 'EQUALIZED' and abs(em2 - ew2) < 1e-8 and max(em2, ew2) < max(em1, ew1) and worse == 40)

    # 6. parameter round trip through JSON reproduces the release bitwise; release_from_raw path equals transform path
    prm = {'w': 0.0, 'T': float(np.float64(1.2345678901234567)), 'b': [float(v) for v in center([0.123456789, -0.3, 0.2, 0.0, -0.023456789])]}
    prm2 = json.loads(json.dumps(prm))
    raw_inc = rng.normal(0, 2, N // 10); raw_occ = rng.normal(0, 1, (N // 10, K_OCC))
    r1 = release_from_raw(raw_inc, raw_occ, None, prm); r2 = release_from_raw(raw_inc, raw_occ, None, prm2)
    same = all(np.array_equal(r1[t], r2[t]) for t in ('income', OCC))
    direct = np.array_equal(r1[OCC], transform(clr(raw_occ), prm['T'], prm['b']))
    blend_prm = {**prm, 'w': 0.25}
    rb = release_from_raw(raw_inc, raw_occ, raw_occ + 1.0, blend_prm)            # blend of a score with a shifted copy = same clr
    checks['round_trip'] = {'json_round_trip_bitwise': bool(same), 'release_path_equals_transform': bool(direct),
                            'blend_of_equivalent_scores_close': float(np.max(np.abs(rb[OCC] - r1[OCC])))}
    checks['round_trip']['pass'] = bool(same and direct and checks['round_trip']['blend_of_equivalent_scores_close'] < 1e-12)

    res = {'schema': 'IHS1-fam-o-invented-data-check-v1', 'data': 'invented only (numpy default_rng(20261018)); no real data read',
           'checks': checks, 'all_pass': bool(all(c['pass'] for c in checks.values()))}
    if out_path:
        from . import common as C
        C.write_json(out_path, res)
    print(json.dumps({k: v['pass'] for k, v in checks.items()} | {'all_pass': res['all_pass']}))
    return res


# ---------------------------------------------------------------- compact summary of development evaluations (aggregates only)
def _compact(m):
    u = m['utility']
    out = {'guards': {g['group']: {'value': g['value'], 'se': g['se'], 'margin_se_units': g['margin_se_units']} for g in u['guards']},
           'occupation_clauses': {f"{c['reference']}|{c['metric']}": {'value': c['value'], 'se': c.get('se'), 'margin_se_units': c.get('margin_se_units')}
                                  for c in u['clauses'] if c['task'] == 'occupation'},
           'income_clauses_pass': all(c['point_pass'] for c in u['clauses'] if c['task'] == 'income'),
           'point_pass_12': u['point_pass_12'], 'guard_pass': u['guard_pass'], 'distinct': m.get('distinct_outputs')}
    if 'proxy' in m:
        s = m['proxy']['summary']
        out['proxy'] = {v: {'mean': s[v]['mean'], 'min': s[v]['min'], 'max': s[v]['max'], 'cells': s[v]['cells']} for v in s}
    if 'comparisons' in m:
        out['vs'] = {r: {v: {'mean': c[v]['mean'], 'min': c[v]['min'], 'max': c[v]['max']} for v in c} for r, c in m['comparisons'].items()}
    if 'reserve' in m and m['reserve']:
        out['reserve'] = {'status': m['reserve']['status'], 'failed': m['reserve']['failed'], 'occupation_side': m['reserve'].get('occupation_side')}
    return out


def summary(seed, configs):
    from . import common as C
    from . import devrun as D
    ref = D.reference_eval(seed)
    res = {'seed': seed, 'H_LJ_reference': _compact(ref['H_LJ'])}
    for cfg in configs:
        p = D.release_dir(FAMILY, cfg, seed) / 'EVAL_MS.json'
        if p.is_file():
            res[cfg] = _compact(json.loads(p.read_text()))
        rp = D.release_dir(FAMILY, cfg, seed) / 'RELEASE.json'
        if rp.is_file():
            rec = json.loads(rp.read_text())
            res.setdefault(cfg, {})['params'] = rec['params']
            res[cfg]['n_model_fits'] = rec['n_model_fits']
            res[cfg]['fit_cpu_s'] = rec['fit_cpu_s']
    return res


def table(seed, configs):
    """One line per configuration (and H_LJ): guards, occupation NLL/Brier/accuracy vs Ucal and U0, proxies, reserve."""
    res = summary(seed, configs)
    lines = []
    for name in ['H_LJ_reference'] + list(configs):
        d = res.get(name)
        if not d or 'guards' not in d:
            continue
        g, oc, px = d['guards'], d['occupation_clauses'], d.get('proxy', {})
        f = lambda c: f"{oc[c]['value']:+.5f}({oc[c]['margin_se_units']:.2f}SE)"
        lines.append(f"S{seed} {name:16s} men {g['sex0']['value']:+.5f} se {g['sex0']['se']:.5f} ({g['sex0']['margin_se_units']:.2f}SE) | "
                     f"women {g['sex1']['value']:+.5f} se {g['sex1']['se']:.5f} ({g['sex1']['margin_se_units']:.2f}SE) | "
                     f"NLL U0 {f('U0|nll_excess')} Ucal {f('Ucal|nll_excess')} | Brier U0 {f('U0|brier_excess')} Ucal {f('Ucal|brier_excess')} | "
                     f"acc {f('U0|accuracy_loss')} | occ {px.get('occupation', {}).get('mean', float('nan')):.4f} "
                     f"[{px.get('occupation', {}).get('min', float('nan')):.4f},{px.get('occupation', {}).get('max', float('nan')):.4f}] "
                     f"pair {px.get('pair', {}).get('mean', float('nan')):.4f} [{px.get('pair', {}).get('min', float('nan')):.4f},"
                     f"{px.get('pair', {}).get('max', float('nan')):.4f}] | reserve {d.get('reserve', {}).get('status', '-')} {d.get('reserve', {}).get('failed', '')}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('selftest'); s.add_argument('--out', default=None)
    f = sub.add_parser('fit'); f.add_argument('--seed', type=int, required=True); f.add_argument('--config', required=True)
    v = sub.add_parser('verify'); v.add_argument('--seed', type=int, required=True); v.add_argument('--config', required=True)
    sm = sub.add_parser('summary'); sm.add_argument('--seed', type=int, action='append', required=True); sm.add_argument('--config', action='append', default=[])
    sm.add_argument('--out', default=None)
    tb = sub.add_parser('table'); tb.add_argument('--seed', type=int, required=True); tb.add_argument('--config', action='append', default=[])
    a = ap.parse_args(argv)
    if a.cmd == 'table':
        print('\n'.join(table(a.seed, a.config)))
        return 0
    if a.cmd == 'selftest':
        res = selftest(a.out)
        return 0 if res['all_pass'] else 1
    if a.cmd == 'fit':
        fit(a.seed, a.config)
        return 0
    if a.cmd == 'summary':
        res = {f'S{s}': summary(s, a.config) for s in a.seed}
        if a.out:
            from . import common as C
            C.write_json(a.out, {'schema': 'IHS1-fam-o-results-summary-v1', 'what': 'MODEL_SELECT development aggregates (no person-level '
                                 'values) of family O configurations and of H_LJ, from EVAL_MS.json / EVAL_MS_REFERENCES.json', **res})
        print(json.dumps(res if not a.out else {'written': a.out}, indent=1, default=float))
        return 0
    print(json.dumps(verify(a.seed, a.config)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

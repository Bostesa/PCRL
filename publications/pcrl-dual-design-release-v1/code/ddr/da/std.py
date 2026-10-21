"""DA: SEX-standardized task posteriors (STD; Designer A, family DA). PCRL campaign `pcrl_dual_design_release_v1`; EXPLORATORY
DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is
a training input / label on MODEL_FIT only, never a runtime input or output.

Operation (card DA_STD_v1; Designer A's PROPOSAL.md; Z = ohn.fam_a.compact_po(X)):
  twin       tI(x), tO(x): the cached task teachers (HGB, A1_PO_16 settings; MODEL_FIT income / occupation; the new-release-
             mechanisms TASK.pkl, pinned by hash in SOURCE_ADMISSION.json)
  A-aware    qI_a(x) = P(inc | x, A = a), qO_a(x) = P(occ | x, A = a): HGB on [Z, A] (A1_PO_16 settings), evaluated at a = 0, 1.
             For STD, A = SEX: qO_s is the cached SEX-side teacher 'occ_s' (finite-routing SEXSIDE.pkl, pinned); qI_s is NEW (the same
             construction: inner household-CV T*, tag OHN_A_cv_S<v>; full refit; out-of-fold on NRM_FA_xf_S<v>)
  weight     w: the value on 0.30, 0.31, ..., 0.60 minimizing the larger of the men's and women's MODEL_FIT out-of-fold occupation
             Brier excess of the lam = 1 release over the twin (ties to the value nearest the SEX share)
  release    P_inc = (1 - lam) tI + lam [(1 - w) qI_0 + w qI_1];  P_occ = (1 - lam) tO + lam [(1 - w) qO_0 + w qO_1]  (argmax)
  strength   lam* (per version) = the largest lam on (0.5, 0.75, 1.0) such that it and every smaller grid value keep MODEL_FIT
             out-of-fold spend over the twin within budget S3 (estimate + 2 household-clustered SE on six clauses)
Matched controls (PROTOCOL 4; card section 5):
  SBC path   the same operation with A = a fixed-seed permutation of the MODEL_FIT SEX column (A-aware teachers at the SEX-aware
             teachers' iteration counts; w and lam grid by the same rules); if its lam = 1 out-of-fold total-NLL spend cannot reach
             STD's plus 2 SE of the paired difference, the declared fallback A = PSEUDO_gamma: a label-derived binary,
             A_i = 1[u_i < rho_gamma(y_i)], u_i a keyed hash of the row, rho = the MODEL_FIT SEX share of each (income, occupation)
             label cell PERMUTED across the ten cells (fixed seed), sharpened rho_gamma = sigmoid(gamma logit rho + c), c matching the
             SEX share; gamma = the smallest of (1, 2, 4) whose lam = 1 spend reaches that target (else 4, flagged)
  SPEND path SEX-blind KL k-means coarsening of the twin's posteriors with out-of-fold teacher-mean (B1) tables (the finite-routing
             construction), capacities (24, 128), (16, 64), (8, 32)
  Each path saves two bracket points chosen on MODEL_FIT out-of-fold total-NLL spend (PROTOCOL 4).
Ablations: COMP (the same SEX-aware parts mixed with the person's fitted propensity pi(x), cached SEX-side 'sex' teacher, at lam*);
STD_OCC (occupation STD at lam*, income = twin). Descriptive MODEL_FIT-only: UND (the twin's boosting stopped early at matched
spend) and household cross-fitted MODEL_FIT readers; never a decision input and never a stop.

Commands (real data only through the campaign queue after the receipt and the committed card):
    python3 -m ddr.da.std selftest | selftest_stage
    python3 -m ddr.da.std fit --version V --card results/pcrl_dual_design_release_v1/cards/DA_STD_v1.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK

FAMILY = 'DA'
K_INC, K_OCC = 2, 5
LAMBDA_GRID = (0.5, 0.75, 1.0)
W_GRID = tuple(float(x) for x in np.round(np.arange(0.30, 0.6001, 0.01), 2))
BUDGET = 'S3'
BUDGETS = {
    'S2': {'occ_nll': 0.008, 'inc_nll': 0.004, 'occ_brier_men': 0.0030, 'occ_brier_women': 0.0030, 'occ_acc': 0.006, 'inc_acc': 0.004},
    'S3': {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008, 'inc_acc': 0.006},
}
GAMMAS = (1.0, 2.0, 4.0)
CRS_GRID = ((24, 128), (16, 64), (8, 32))
UND_MULT = (0.5, 1.0, 2.0)
CONFIGS = {
    'DA_TWIN': {'kind': 'twin', 'what': 'equally trained task-only twin: the cached HGB task teachers (lam = 0); never reads SEX'},
    'DA_STD': {'kind': 'candidate', 'what': 'SEX-standardized task posteriors at the MODEL_FIT-selected strength lam* (both recipients)'},
    'DA_SBC_LO': {'kind': 'control', 'what': 'SBC path, lower bracket point: the same standardization over a SEX-independent attribute'},
    'DA_SBC_HI': {'kind': 'control', 'what': 'SBC path, upper bracket point: the same standardization over a SEX-independent attribute'},
    'DA_SPD_LO': {'kind': 'control', 'what': 'SPEND path, lower bracket point: SEX-blind KL k-means coarsening of the twin (B1 tables)'},
    'DA_SPD_HI': {'kind': 'control', 'what': 'SPEND path, upper bracket point: SEX-blind KL k-means coarsening of the twin (B1 tables)'},
    'DA_COMP': {'kind': 'ablation', 'what': "the same SEX-aware parts mixed with the person's fitted propensity pi(x) at lam* (weights not fixed)"},
    'DA_STD_OCC': {'kind': 'ablation', 'what': 'occupation STD at lam*; income = the twin (per-recipient attribution)'},
}
PURPOSE = ('DA STD: SEX-aware income teacher, guard-balanced weight, strength selection, matched control paths and ablations on '
           'MODEL_FIT (SEX as a training input and label)')
CODE_MODULES = ('ddr.da.std', 'ddr.famkit', 'nrm.fa.prc', 'frc.fa.route', 'ohn.fam_a')


# ---------------------------------------------------------------- teachers (nrm.fa.prc construction)
def _prc():
    from nrm.fa import prc
    return prc


def teacher_a(Z, y, K, a, unit, version, T=None):
    """Attribute-aware teacher on [Z, A]: T* (first argmin of the pooled inner household-CV NLL, tag OHN_A_cv_S<v>) unless T is
    given, the full refit, and out-of-fold predictions at a = 0 and 1 over the 5 household folds NRM_FA_xf_S<v> (fold models at the
    same T): exactly the nrm.fa.prc.fit_teachers construction of the cached 'occ_s' teacher."""
    P = _prc()
    Zt = np.column_stack([Z, np.asarray(a, dtype=np.float64)])
    cat = np.append(P.CAT_Z, False)
    xf = P.household_folds(unit, P.N_XF, f'NRM_FA_xf_S{version}')
    curve_min, n_inner = None, 0
    if T is None:
        inner = P.household_folds(unit, P.N_INNER, f'OHN_A_cv_S{version}')
        T, cur = P.tstar(P.HGB, Zt, y, K, inner, version, cat)
        curve_min, n_inner = float(cur.min()), P.N_INNER
    model = P.fit_hgb(P.HGB, Zt, y, K, T, version, cat)
    oof = {0: np.zeros((len(Z), K)), 1: np.zeros((len(Z), K))}
    for f in range(P.N_XF):
        tr, te = xf != f, xf == f
        m = P.fit_hgb(P.HGB, Zt[tr], y[tr], K, T, version, cat)
        for s in (0, 1):
            oof[s][te] = P._proba(m, P.sex_aug(Z[te], s))
    return {'T': int(T), 'curve_min': curve_min, 'model': model, 'oof': oof, 'n_fits': (n_inner + 1, P.N_XF)}


def twin_staged_oof(Z, y, K, unit, version, T):
    """Out-of-fold predictions of the twin at EVERY boosting iteration 1..T (fold models refitted at T on NRM_FA_xf_S<v>, staged):
    array (T, n, K). Descriptive UND check only."""
    P = _prc()
    xf = P.household_folds(unit, P.N_XF, f'NRM_FA_xf_S{version}')
    out = np.zeros((T, len(Z), K))
    for f in range(P.N_XF):
        tr, te = xf != f, xf == f
        m = P.fit_hgb(P.HGB, Z[tr], y[tr], K, T, version, P.CAT_Z)
        for i, r in enumerate(m.staged_decision_function(Z[te])):
            out[i, te] = P.clipn(P.probs(r))
    return out


# ---------------------------------------------------------------- release algebra (shared by fit-time OOF and deployment)
def mix(t, q0, q1, w, lam):
    return FK.clipn((1.0 - lam) * t + lam * ((1.0 - w) * q0 + w * q1))


def mix_pi(t, q0, q1, pi, lam):
    pi = np.asarray(pi, dtype=np.float64)[:, None]
    return FK.clipn((1.0 - lam) * t + lam * ((1.0 - pi) * q0 + pi * q1))


def _occ_brier(P, y):
    oh = np.eye(K_OCC)[y]
    return ((P - oh) ** 2).sum(1)


def guard_balanced_w(tO, qO0, qO1, y_occ, sex, grid=W_GRID):
    base = _occ_brier(tO, y_occ)
    men, wom, share = sex == 0, sex == 1, float(np.mean(sex))
    best, table = None, {}
    for w in grid:
        b = _occ_brier(mix(tO, qO0, qO1, w, 1.0), y_occ) - base
        dm, dw = float(b[men].mean()), float(b[wom].mean())
        table[str(w)] = {'men': dm, 'women': dw}
        key = (round(max(dm, dw), 12), abs(w - share))
        if best is None or key < best[0]:
            best = (key, w)
    return float(best[1]), table


def crs_release(book, tabs, tI, tO):
    """Coarsened release from twin posteriors: nearest codeword in cross-entropy (lam = 0 rule), then the B1 table of the cell."""
    k, l = _prc().route({'qI': tI, 'qO': tO}, book)
    return tabs['I'][k].copy(), tabs['O'][l].copy()


def _predict_payload(P, Z):
    prc = _prc()
    tI, tO = prc._proba(P['twin_inc'], Z), prc._proba(P['twin_occ'], Z)
    op = P['op']
    if op == 'twin':
        return tI, tO
    if op == 'crs':
        return crs_release(P['book'], P['tabs'], tI, tO)
    qI0, qI1 = prc._proba(P['a_inc'], prc.sex_aug(Z, 0)), prc._proba(P['a_inc'], prc.sex_aug(Z, 1))
    qO0, qO1 = prc._proba(P['a_occ'], prc.sex_aug(Z, 0)), prc._proba(P['a_occ'], prc.sex_aug(Z, 1))
    lam, w = float(P['lambda']), float(P['w'])
    if op in ('std', 'sbc'):
        return mix(tI, qI0, qI1, w, lam), mix(tO, qO0, qO1, w, lam)
    if op == 'std_occ':
        return tI, mix(tO, qO0, qO1, w, lam)
    if op == 'comp':
        pi = np.clip(prc._proba(P['sex_model'], Z)[:, 1], 0.01, 0.99)
        return mix_pi(tI, qI0, qI1, pi, lam), mix_pi(tO, qO0, qO1, pi, lam)
    raise ValueError(f'unknown operation {op}')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (ddr.deploy contract): the configuration's income (n, 2) and occupation (n, 5) vectors from admitted
    features alone, reading only model_paths(config_id, version). Deterministic: per-request keys and draws other than 0 are refused."""
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


# ---------------------------------------------------------------- caches (hash-pinned in SOURCE_ADMISSION.json)
def load_caches(version, row_sha):
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    tp = C.NRM_FA_CACHE / f'V{int(version)}' / 'TASK.pkl'
    sp = C.FRC_FA_CACHE / f'V{int(version)}' / 'SEXSIDE.pkl'
    want_t = adm['source_private_artefacts'][f'fa_cache/V{int(version)}/TASK.pkl']['sha256']
    want_s = adm['frc_private_artefacts'][f'frc/fa_cache/V{int(version)}/SEXSIDE.pkl']['sha256']
    bt, bs = tp.read_bytes(), sp.read_bytes()
    if FK.sha_bytes(bt) != want_t or FK.sha_bytes(bs) != want_s:
        raise PermissionError('a cached teacher file differs from its admission pin')
    T, S = pickle.loads(bt), pickle.loads(bs)
    if T['row_id_sha256'] != row_sha or S['row_id_sha256'] != row_sha:
        raise PermissionError('a cached teacher was fitted on other rows')
    return T, S, {'TASK.pkl': want_t, 'SEXSIDE.pkl': want_s}


# ---------------------------------------------------------------- SEX-independent attributes of the SBC path
def permuted_sex(sex, version):
    rng = np.random.default_rng(np.random.SeedSequence([int(version), 20261021, 1]))
    return np.asarray(sex)[rng.permutation(len(sex))]


def _u(row_id, tag):
    """Keyed uniform in [0, 1) per row (SHA-256 of tag | row id)."""
    return np.array([int(hashlib.sha256(f'{tag}|{int(r)}'.encode()).hexdigest()[:13], 16) / float(16 ** 13) for r in row_id])


def pseudo_attribute(y_inc, y_occ, sex, row_id, version, gamma):
    """Declared fallback SBC target: A_i = 1[u_i < rho_gamma(y_i)], rho = the MODEL_FIT SEX share of each (income, occupation)
    label cell PERMUTED across the ten cells (fixed seed), sharpened rho_gamma = sigmoid(gamma logit rho + c) with c matching the
    SEX share. A is a label-derived binary of comparable task entanglement whose association direction is unrelated to SEX's."""
    cell = np.asarray(y_inc) * K_OCC + np.asarray(y_occ)
    share = np.array([float(sex[cell == c].mean()) if np.any(cell == c) else float(sex.mean()) for c in range(2 * K_OCC)])
    rng = np.random.default_rng(np.random.SeedSequence([int(version), 20261021, 2]))
    rho = np.clip(share[rng.permutation(2 * K_OCC)], 0.02, 0.98)
    lg = gamma * np.log(rho / (1 - rho))
    target = float(np.mean(sex))
    lo, hi = -20.0, 20.0
    for _ in range(80):                                  # bisection on c: mean over people of sigmoid(lg[cell] + c) = SEX share
        c = 0.5 * (lo + hi)
        m = float(np.mean(1.0 / (1.0 + np.exp(-(lg[cell] + c)))))
        lo, hi = (c, hi) if m < target else (lo, c)
    rg = 1.0 / (1.0 + np.exp(-(lg + 0.5 * (lo + hi))))
    a = (_u(row_id, f'DDR_DA_pseudo_S{int(version)}') < rg[cell]).astype(np.int64)
    return a, {'gamma': gamma, 'rho_permuted': rho.tolist(), 'rho_gamma': rg.tolist(), 'share_A': float(a.mean())}


# ---------------------------------------------------------------- bracket points (PROTOCOL 4)
def bracket_points(path_L, cand_L, twin_L, unit, order):
    """Two saved points of a control path chosen on MODEL_FIT out-of-fold total-NLL spend over the twin: LO = the largest grid point
    whose spend is at most CAND's; HI = the smallest grid point whose spend is at least CAND's + 2 household-clustered SE of the paired
    difference (point minus CAND); if none reaches it, the largest grid point, flagged CONTROL_PATH_MAY_NOT_BRACKET."""
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    rows, lo, hi = {}, None, None
    for g in order:
        d = path_L[g]['tot_nll'] - twin_L['tot_nll']
        diff = path_L[g]['tot_nll'] - cand_L['tot_nll']
        rows[str(g)] = {'spend': float(d.mean()), 'diff_vs_cand': float(diff.mean()), 'se_diff': FK.cluster_se(diff, unit)}
        if rows[str(g)]['spend'] <= cand_sp:
            lo = g
    for g in order:
        r = rows[str(g)]
        if r['spend'] >= cand_sp + 2.0 * r['se_diff']:
            hi = g
            break
    flag = None
    if hi is None:
        hi, flag = order[-1], 'CONTROL_PATH_MAY_NOT_BRACKET'
    if lo == hi:
        lo = None
    return {'cand_spend': cand_sp, 'grid': rows, 'lo': lo, 'hi': hi, 'flag': flag}


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, card, save, roles=None, log=print, budget_override=None, record_dir=None):
    t0 = time.process_time()
    prc = _prc()
    from frc.fa import route as FR
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    row_id = np.asarray(data['row_id'])
    T, S, cache_sha = load_caches(version, row_sha)
    budget = budget_override or BUDGETS[BUDGET]
    tI, tO = T['oof']['inc'], T['oof']['occ']
    twin_L = FK.person_losses(tI, tO, yI, yO)
    # --- CAND parts and selection
    qO0, qO1 = S['oof']['occ_s'][0], S['oof']['occ_s'][1]
    pi_oof = np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
    ts = teacher_a(Z, yI, K_INC, sex, unit, version)
    log(json.dumps({'teacher': 'inc_s', 'T': ts['T'], 'cpu_s': round(time.process_time() - t0, 1)}))
    qI0, qI1 = ts['oof'][0], ts['oof'][1]
    w, w_table = guard_balanced_w(tO, qO0, qO1, yO, sex)
    path = {lam: FK.person_losses(mix(tI, qI0, qI1, w, lam), mix(tO, qO0, qO1, w, lam), yI, yO) for lam in LAMBDA_GRID}
    path_sp = {lam: FK.spend(L, twin_L, unit, sex) for lam, L in path.items()}
    lam_star = FK.select_strength(path_sp, LAMBDA_GRID, budget)
    sel = {'w': w, 'w_table': w_table, 'budget': BUDGET if budget_override is None else 'override', 'budget_limits': budget,
           'lambda_star': lam_star, 'path': {str(l): FK.jsonable_spend(sp) for l, sp in path_sp.items()},
           'rule': 'per version: the largest grid lam such that it and every smaller grid value meet estimate + 2 household-clustered '
                   'SE <= budget on the six clauses (MODEL_FIT out-of-fold, folds NRM_FA_xf_S<v>, T* of the inherited teachers)'}
    log(json.dumps({'lambda_star': lam_star, 'w': w}))
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'cache_sha256': cache_sha}
    exposure_sex = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the attribute-aware teachers '
                    '(SEX or a SEX-derived attribute as a training input), the weight and the strength / bracket selection '
                    '(out-of-fold, household folds); the twin teachers are the cached task teachers; no CALIBRATION or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {'DA_TWIN': (tI, tO)}

    def put(cfg, payload, oof_pair, extra, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        sp = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits,
                        'MODEL_FIT: income, occupation' + ('' if cfg == 'DA_TWIN' else ', sex'),
                        ('MODEL_FIT income and occupation labels reach the cached task teachers (no SEX)' if cfg == 'DA_TWIN'
                         else exposure_sex), people, hh, time.process_time() - t0,
                        {**extra, 'selection_lambda_star': lam_star, 'w_cand': w, 'cache_sha256': cache_sha,
                         'model_sha256': meta['model_sha256'], 'oof_spend_over_twin': FK.jsonable_spend(sp)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'oof_spend_over_twin': FK.jsonable_spend(sp), **extra}
        oof_views[cfg] = oof_pair

    put('DA_TWIN', {'op': 'twin'}, (tI, tO), {'op': 'twin'}, (0, 0))
    out = {'version': int(version), 'selection': sel, 'cache_sha256': cache_sha, 'T_inc_s': ts['T'], 'inc_s_curve_min': ts['curve_min']}
    if lam_star is None:
        log(json.dumps({'stop': 'no grid strength meets the budget: no candidate is released'}))
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'da') / f'FIT_V{int(version)}.json', out)
        return out
    cand_L = path[lam_star]
    std_parts = {'a_inc': ts['model'], 'a_occ': S['models']['occ_s'], 'w': w, 'lambda': lam_star}
    put('DA_STD', {'op': 'std', **std_parts}, (mix(tI, qI0, qI1, w, lam_star), mix(tO, qO0, qO1, w, lam_star)),
        {'op': 'std', 'lambda': lam_star, 'w': w, 'T_inc_s': ts['T'], 'T_occ_s': S['T']['occ_s']}, ts['n_fits'])
    # --- SBC path: permuted SEX first, the declared fallback otherwise
    Tfix = {'inc': ts['T'], 'occ': int(S['T']['occ_s'])}
    tries = []

    def sbc_path(attr, label):
        ai = teacher_a(Z, yI, K_INC, attr, unit, version, T=Tfix['inc'])
        ao = teacher_a(Z, yO, K_OCC, attr, unit, version, T=Tfix['occ'])
        wa, _ = guard_balanced_w(tO, ao['oof'][0], ao['oof'][1], yO, sex)
        L = {lam: FK.person_losses(mix(tI, ai['oof'][0], ai['oof'][1], wa, lam), mix(tO, ao['oof'][0], ao['oof'][1], wa, lam), yI, yO)
             for lam in LAMBDA_GRID}
        br = bracket_points(L, cand_L, twin_L, unit, LAMBDA_GRID)
        reach = br['flag'] is None
        tries.append({'attribute': label, 'w': wa, 'bracket': br, 'reaches': reach,
                      'attribute_share': float(np.mean(attr)), 'attribute_sex_agreement': float(np.mean(np.asarray(attr) == sex))})
        log(json.dumps({'sbc_try': label, 'reaches': reach, 'cpu_s': round(time.process_time() - t0, 1)}))
        return {'ai': ai, 'ao': ao, 'w': wa, 'L': L, 'br': br, 'label': label}
    chosen = sbc_path(permuted_sex(sex, version), 'PERMUTED_SEX')
    pseudo_info = None
    if chosen['br']['flag'] is not None:
        for g in GAMMAS:
            attr, pseudo_info = pseudo_attribute(yI, yO, sex, row_id, version, g)
            chosen = sbc_path(attr, f'PSEUDO_gamma{g:g}')
            if chosen['br']['flag'] is None:
                break
    br = chosen['br']
    n_sbc_fits = (2 * len(tries), 2 * 5 * len(tries))
    first = True
    for cfg, lam in (('DA_SBC_LO', br['lo']), ('DA_SBC_HI', br['hi'])):
        if lam is None:
            continue
        put(cfg, {'op': 'sbc', 'a_inc': chosen['ai']['model'], 'a_occ': chosen['ao']['model'], 'w': chosen['w'], 'lambda': lam},
            (mix(tI, chosen['ai']['oof'][0], chosen['ai']['oof'][1], chosen['w'], lam),
             mix(tO, chosen['ao']['oof'][0], chosen['ao']['oof'][1], chosen['w'], lam)),
            {'op': 'sbc', 'attribute': chosen['label'], 'lambda': lam, 'w': chosen['w'], 'bracket_flag': br['flag']},
            n_sbc_fits if first else (0, 0))
        first = False
    # --- SPEND path: SEX-blind coarsening of the twin (B1 tables)
    E_oof = {'qI': tI, 'qO': tO}
    books, crs_L = {}, {}
    for cap in CRS_GRID:
        book, k0, l0 = prc.twin_book(E_oof, cap[0], cap[1], version, prc.route_params('PRC_TWIN'))
        tabs, tinfo = FR.tables('B1', book, k0, l0, E=E_oof)
        books[cap] = (book, tabs, tinfo, k0, l0)
        crs_L[cap] = FK.person_losses(tabs['I'][k0], tabs['O'][l0], yI, yO)
    brs = bracket_points(crs_L, cand_L, twin_L, unit, CRS_GRID)
    for cfg, cap in (('DA_SPD_LO', brs['lo']), ('DA_SPD_HI', brs['hi'])):
        if cap is None:
            continue
        book, tabs, tinfo, k0, l0 = books[cap]
        put(cfg, {'op': 'crs', 'book': book, 'tabs': tabs, 'capacity': list(cap)}, (tabs['I'][k0], tabs['O'][l0]),
            {'op': 'crs', 'capacity': list(cap), 'bracket_flag': brs['flag'], 'table_info': tinfo}, (0, 0))
    # --- ablations
    put('DA_STD_OCC', {'op': 'std_occ', **std_parts}, (tI, mix(tO, qO0, qO1, w, lam_star)), {'op': 'std_occ', 'lambda': lam_star},
        (0, 0))
    put('DA_COMP', {'op': 'comp', **std_parts, 'sex_model': S['models']['sex']},
        (mix_pi(tI, qI0, qI1, pi_oof, lam_star), mix_pi(tO, qO0, qO1, pi_oof, lam_star)), {'op': 'comp', 'lambda': lam_star}, (0, 0))
    # --- descriptive MODEL_FIT-only checks: UND (early stopping at matched spend) and cross-fitted readers
    t1 = time.process_time()
    stI = twin_staged_oof(Z, yI, K_INC, unit, version, int(T['T']['inc']))
    stO = twin_staged_oof(Z, yO, K_OCC, unit, version, int(T['T']['occ']))
    ar = np.arange(len(yI))
    cI = np.array([float(np.mean(-np.log(stI[i][ar, yI]))) for i in range(stI.shape[0])]) - float(twin_L['inc_nll'].mean())
    cO = np.array([float(np.mean(-np.log(stO[i][ar, yO]))) for i in range(stO.shape[0])]) - float(twin_L['occ_nll'].mean())
    cand_sp = FK.spend(cand_L, twin_L, unit, sex)
    und = {}
    for mult in UND_MULT:
        okI, okO = np.flatnonzero(cI >= mult * cand_sp['inc_nll'][0]), np.flatnonzero(cO >= mult * cand_sp['occ_nll'][0])
        ti, to = (int(okI.max()) + 1 if okI.size else 1), (int(okO.max()) + 1 if okO.size else 1)
        und[f'UND_{mult:g}'] = {'t_inc': ti, 't_occ': to, 'spend': FK.jsonable_spend(FK.spend(
            FK.person_losses(stI[ti - 1], stO[to - 1], yI, yO), twin_L, unit, sex))}
        oof_views[f'UND_{mult:g}'] = (stI[ti - 1], stO[to - 1])
    check = FK.mf_reader_check(oof_views, sex, unit, f'DDR_mfcheck_DA_S{int(version)}')
    out.update({'saved': saved, 'sbc_tries': tries, 'sbc_attribute': chosen['label'], 'pseudo_info': pseudo_info,
                'spend_path_bracket': brs, 'und_descriptive': und,
                'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers (household 5-fold cross-fitted); never a '
                                            'decision input and never a stop; the decisions use the MODEL_SELECT grid',
                                    'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'da') / f'FIT_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- selftests (invented data; no label file)
def selftest():
    rng = np.random.default_rng(11)
    n = 3000
    t = rng.dirichlet(np.ones(5), n)
    q0, q1 = rng.dirichlet(np.ones(5), n), rng.dirichlet(np.ones(5), n)
    ok = {}
    ok['lam0_is_twin'] = np.allclose(mix(t, q0, q1, 0.45, 0.0), FK.clipn(t))
    ok['lam1_is_fixed_average'] = np.allclose(mix(t, q0, q1, 0.45, 1.0), FK.clipn(0.55 * q0 + 0.45 * q1))
    ok['comp_constant_pi_is_std'] = np.allclose(mix_pi(t, q0, q1, np.full(n, 0.45), 0.75), mix(t, q0, q1, 0.45, 0.75))
    y = rng.integers(0, 5, n)
    sex = rng.integers(0, 2, n)
    w, tab = guard_balanced_w(t, q0, q1, y, sex)
    ok['w_on_grid'] = w in W_GRID and len(tab) == len(W_GRID)
    ps = permuted_sex(sex, 0)
    ok['permutation_keeps_margin'] = int(ps.sum()) == int(sex.sum()) and not np.array_equal(ps, sex)
    yi = rng.integers(0, 2, n)
    a, info = pseudo_attribute(yi, y, sex, np.arange(n), 0, 2.0)
    ok['pseudo_share_matches'] = abs(info['share_A'] - float(sex.mean())) < 0.04
    a2, _ = pseudo_attribute(yi, y, sex, np.arange(n), 0, 2.0)
    ok['pseudo_deterministic'] = np.array_equal(a, a2)
    unit = np.arange(n) // 2
    yt = (t.cumsum(1) < rng.random(n)[:, None]).sum(1)          # labels drawn from t: mixing toward uniform spends NLL
    L0 = FK.person_losses(np.full((n, 2), 0.5), t, np.zeros(n, dtype=np.int64), yt)
    Lg = lambda g: FK.person_losses(np.full((n, 2), 0.5), FK.clipn(t * (1 - g) + g * 0.2), np.zeros(n, dtype=np.int64), yt)
    Ls = {g: Lg(g) for g in (0.1, 0.2, 0.35, 0.6)}
    br = bracket_points(Ls, Lg(0.3), L0, unit, (0.1, 0.2, 0.35, 0.6))
    ok['bracket_monotone_law'] = br['lo'] == 0.2 and br['hi'] == 0.35 and br['flag'] is None
    br2 = bracket_points({g: Ls[g] for g in (0.1, 0.2)}, Lg(0.3), L0, unit, (0.1, 0.2))
    ok['bracket_flag_when_unreachable'] = br2['hi'] == 0.2 and br2['flag'] == 'CONTROL_PATH_MAY_NOT_BRACKET'
    ok['all_ok'] = all(ok.values())
    return ok


def selftest_stage(n=2500, seed=5):
    """End-to-end check of stage_fit on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read):
    teachers fitted on the invented labels stand in for the caches, model files go to a temporary directory, nothing is saved to the
    campaign's runs and no record is written. Checks every configuration is written and released through predict()."""
    import tempfile
    from pathlib import Path as _P
    prc = _prc()
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    sex = rng.integers(0, 2, n)
    data = {'X': F['X'][idx], 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': (rng.random(n) < 0.3 + 0.3 * sex).astype(np.int64), 'occupation': np.minimum(rng.integers(0, 5, n) + sex, 4),
            'sex': sex}
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    hp = {**prc.HGB, 'max_iter': 30}
    old_hgb = prc.HGB
    prc.HGB = hp
    try:
        tch = prc.fit_teachers(Z, lab, unit, 0, hp, ('inc', 'occ'))
        sxs = prc.fit_teachers(Z, lab, unit, 0, hp, ('sex', 'occ_s'))
    finally:
        prc.HGB = old_hgb
    T = {'T': tch['T'], 'models': tch['models'], 'oof': tch['oof'], 'row_id_sha256': row_sha}
    S = {'T': sxs['T'], 'models': sxs['models'], 'oof': sxs['oof'], 'row_id_sha256': row_sha}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = (C.RUNS, globals()['load_caches'], prc.HGB)
    tmp = _P(tempfile.mkdtemp(prefix='ddr_da_selftest_'))
    try:
        C.RUNS = tmp
        globals()['load_caches'] = lambda v, rs: (T, S, {'TASK.pkl': 'invented', 'SEXSIDE.pkl': 'invented'})
        prc.HGB = hp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s: None,
                        budget_override={k: 100.0 for k in BUDGETS['S3']}, record_dir=False)
    finally:
        C.RUNS, prc.HGB = old[0], old[2]
        globals()['load_caches'] = old[1]
    ok = {'twin_saved': 'DA_TWIN' in saved}
    tw = saved['DA_TWIN'][0]['MODEL_FIT']
    ok['twin_is_cached_models'] = np.allclose(tw['occupation'], prc._proba(T['models']['occ'], Z))
    lam = out['selection']['lambda_star']
    ok['selection_recorded'] = lam in LAMBDA_GRID
    ok['configs_written'] = {'DA_TWIN', 'DA_STD', 'DA_STD_OCC', 'DA_COMP'} <= set(saved) and any(c.startswith('DA_SBC') for c in saved) \
        and any(c.startswith('DA_SPD') for c in saved)
    ok['records_valid'] = all(r['kind'] == CONFIGS[c]['kind'] for c, (_, r) in saved.items())
    ok['crs_finite'] = all(np.unique(saved[c][0]['MODEL_FIT']['occupation'], axis=0).shape[0] <= 128 for c in saved if c.startswith('DA_SPD'))
    ok['all_ok'] = all(ok.values())
    ok['lambda_star'] = lam
    ok['sbc_attribute'] = out.get('sbc_attribute')
    ok['saved'] = sorted(saved)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ddr.da.std')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    if a.cmd == 'selftest_stage':
        r = selftest_stage(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'lambda_star': out['selection']['lambda_star'], 'w': out['selection']['w'], 'saved': sorted(out['saved']),
                      'sbc_attribute': out.get('sbc_attribute'), 'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

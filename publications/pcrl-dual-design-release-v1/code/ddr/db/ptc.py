"""DB: propensity-tilt cancellation (PTC; Designer B, family DB). PCRL campaign `pcrl_dual_design_release_v1`; EXPLORATORY
DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is
a training label on MODEL_FIT only, never a runtime input or output.

Operation (card DB_PTC_v1; Designer B's PROPOSAL.md):
  p_I(x), p_O(x)  the cached task teachers (TASK.pkl of the new-release-mechanisms campaign, pinned): the twin, unchanged
  q(x)            P(SEX = 1 | x): the cached SEX-side teacher 'sex' (SEXSIDE.pkl of the finite-routing comparison, pinned), clipped
                  to [0.01, 0.99]
  k(x)            argmax p_O(x) (the twin's occupation decision; 5 strata)
  release         log r_I = log p_I - alpha (q - qbar) w_I + const;  log r_O = log p_O - alpha (q - qbar_k) W_O[k] + const  (argmax)
  weights         closed form on MODEL_FIT out-of-fold statistics with the SEX label s, c = centred log-probabilities:
                  w_I = Cov(c(p_I), s) / Cov(q, s);  W_O[k] = Cov(c(p_O), s | k) / Cov(q, s | k), qbar_k = mean(q | k);
                  a stratum with fewer than 200 people or Cov(q, s | k) <= 0 uses the global occupation vector
  strength        alpha* (per version) = the largest alpha on (0.2, 0.35, 0.5, 0.65, 0.8, 1.0) such that it and every smaller grid
                  value keep the NESTED MODEL_FIT spend over the twin (5 outer household folds; weights refitted on the other folds'
                  out-of-fold statistics) within budget S3 plus each task's summed Brier 0.0045 (estimate + 2 household SE)
Matched controls (PROTOCOL 4; card section 5): the twin is DA's (`DA/DA_TWIN`: the same cached teachers, identical wire);
  SBC path   the same operation with the SEX column replaced by a fixed-seed permutation (propensity HGB refitted on it with the
             SEX-side teacher's settings and iteration count; weights by the same rule); fallback when degenerate (Cov(q, s) <= 1e-4)
             or unable to reach PTC's MODEL_FIT spend: a keyed Bernoulli(0.5) label per (SCHL, COW) cell of the compact coding.
             Strength grid = the candidate's grid times a spend-normalizing scale m (RMS tilt of the target's alpha = 1 equal to
             the candidate's), so the path has the same size and spans the candidate's spend
  SPEND path SEX-blind KL k-means coarsening of the twin's posteriors with out-of-fold teacher-mean (B1) tables; rungs (64, 256),
             (32, 128), (24, 96), (16, 64), (12, 48), (8, 32)
  SHUF       the candidate's weights and strata, the person's q replaced by a keyed six-field-hash donor's q from the MODEL_FIT
             out-of-fold bank (deployable; carries no information about this person's SEX); one point, scale chosen on MODEL_FIT so
             that its nested spend first reaches the candidate's
Ablation: PTC_OCC (occupation tilt at alpha*, income = twin). Descriptive MODEL_FIT-only: household cross-fitted readers.

Commands (real data only through the campaign queue after the receipt and the committed card):
    python3 -m ddr.db.ptc selftest | selftest_stage
    python3 -m ddr.db.ptc fit --version V --card results/pcrl_dual_design_release_v1/cards/DB_PTC_v1.md
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

FAMILY = 'DB'
K_INC, K_OCC = 2, 5
FLOOR = 1e-6
MIN_STRATUM = 200
STRENGTHS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
BUDGET = {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008,
          'inc_acc': 0.006, 'occ_brier': 0.0045, 'inc_brier': 0.0045}          # S3 plus each task's summed Brier
DQ_GRID = (0.15, 0.25, 0.4, 0.6, 0.9, 1.4)                                    # SPEND: dithered-quantization steps
COV_MIN = 1e-4                                                                 # degenerate below (global and per stratum)
SBC_SCALE_LADDER = (1.0, 2.0, 4.0, 8.0, 16.0)                                  # x the RMS-normalizing factor of a target
SHUF_MULT = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)                                    # x alpha*: the SHUF scale grid (same size)
N_OUTER = 5
CONFIGS = {
    'DB_PTC': {'kind': 'candidate', 'what': 'propensity-tilt cancellation at the MODEL_FIT-selected strength alpha* (both recipients)'},
    'DB_SBC_LO': {'kind': 'control', 'what': 'SBC path, lower bracket point: the same tilt law driven by a SEX-independent target'},
    'DB_SBC_HI': {'kind': 'control', 'what': 'SBC path, upper bracket point: the same tilt law driven by a SEX-independent target'},
    'DB_SPD_LO': {'kind': 'control', 'what': 'SPEND path, lower bracket point: SEX-blind dithered quantization of the twin (ddr.ctlkit DQ)'},
    'DB_SPD_HI': {'kind': 'control', 'what': 'SPEND path, upper bracket point: SEX-blind dithered quantization of the twin (ddr.ctlkit DQ)'},
    'DB_ZTILT': {'kind': 'control', 'what': 'release-only control: the same tilt law with the amplitude P(SEX | the twin\'s released log-probabilities), spend-matched'},
    'DB_SHUF': {'kind': 'control', 'what': "the candidate's weights and strata with a keyed six-field-hash donor's propensity (spend-matched)"},
    'DB_PTC_OCC': {'kind': 'ablation', 'what': 'occupation tilt at alpha*; income = the twin (per-recipient attribution)'},
}
PURPOSE = ('DB PTC: closed-form tilt weights, nested strength selection, matched control paths and the ablation on MODEL_FIT (SEX as '
           'a training label)')
CODE_MODULES = ('ddr.db.ptc', 'ddr.da.std', 'ddr.famkit', 'ddr.ctlkit', 'nrm.fa.prc', 'ohn.fam_a')


def _prc():
    from nrm.fa import prc
    return prc


# ---------------------------------------------------------------- the tilt law
def clog(p):
    lp = np.log(np.maximum(np.asarray(p, np.float64), FLOOR))
    return lp - lp.mean(1, keepdims=True)


def softmax(l):
    e = np.exp(l - l.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _w(c, q, s):
    sc = s - s.mean()
    cqs = float(np.mean((q - q.mean()) * sc))
    return (c * sc[:, None]).mean(0) / cqs if cqs > COV_MIN else np.zeros(c.shape[1]), float(q.mean()), cqs


def fit_weights(pI, pO, q, s, stratify=True):
    """Closed-form weights on out-of-fold statistics (pI, pO, q) with the binary target s."""
    s = np.asarray(s, np.float64)
    w_inc, qbar, cqs = _w(clog(pI), q, s)
    w_glob, _, _ = _w(clog(pO), q, s)
    dec = np.asarray(pO).argmax(1)
    W, qb, fb = np.tile(w_glob, (K_OCC, 1)), np.full(K_OCC, qbar), []
    if stratify:
        for k in range(K_OCC):
            m = dec == k
            if m.sum() >= MIN_STRATUM:
                wk, qk, ck = _w(clog(pO[m]), q[m], s[m])
                if ck > COV_MIN:
                    W[k], qb[k] = wk, qk
                    continue
            fb.append(k)
    return {'w_inc': w_inc, 'qbar': qbar, 'W_occ': W, 'qbar_occ': qb, 'cov_q_s': cqs, 'fallback_strata': fb,
            'stratum_sizes': np.bincount(dec, minlength=K_OCC).tolist()}


def tilt(pI, pO, q, st, alpha, inc=True, occ=True):
    pI, pO, q = np.asarray(pI, np.float64), np.asarray(pO, np.float64), np.asarray(q, np.float64)
    k = pO.argmax(1)
    rI = softmax(np.log(np.maximum(pI, FLOOR)) - alpha * (q - st['qbar'])[:, None] * st['w_inc'][None, :]) if inc else pI
    rO = softmax(np.log(np.maximum(pO, FLOOR)) - alpha * (q - st['qbar_occ'][k])[:, None] * st['W_occ'][k]) if occ else pO
    return FK.clipn(rI), FK.clipn(rO)


def rms_tilt(pO, q, st):
    """RMS of the alpha = 1 log-shift over people and both recipients (the spend-normalizing scale of a control path)."""
    k = np.asarray(pO).argmax(1)
    dI = (q - st['qbar'])[:, None] * st['w_inc'][None, :]
    dO = (q - st['qbar_occ'][k])[:, None] * st['W_occ'][k]
    return float(np.sqrt(np.mean(np.sum(dI ** 2, 1) + np.sum(dO ** 2, 1))))


def zfeat(pI, pO):
    """The twin's released log-probability features (income log-odds, centred occupation log-probabilities): ZTILT's input."""
    return np.column_stack([np.log(np.maximum(pI[:, 1:], FLOOR) / np.maximum(pI[:, :1], FLOOR)), clog(pO)])


def ztilt_teacher(pI, pO, sex, unit, version):
    """q_z = P(SEX | the twin's released log-probabilities): HGB (SEX-side settings, 300 iterations), cross-fitted on the household
    folds NRM_FA_xf_S<v> for the out-of-fold values, full fit for deployment. A function of the twin's release only."""
    import warnings
    from sklearn.ensemble import HistGradientBoostingClassifier
    P = _prc()
    F = zfeat(pI, pO)
    mk = lambda: HistGradientBoostingClassifier(loss='log_loss', learning_rate=float(P.HGB['lr']), max_iter=300,
                                                max_leaf_nodes=int(P.HGB['leaves']), min_samples_leaf=int(P.HGB['msl']),
                                                l2_regularization=float(P.HGB['l2']), early_stopping=False, random_state=int(version))
    xf = P.household_folds(unit, P.N_XF, f'NRM_FA_xf_S{version}')
    oof = np.zeros(len(sex))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for f in range(P.N_XF):
            tr, te = xf != f, xf == f
            oof[te] = mk().fit(F[tr], sex[tr]).predict_proba(F[te])[:, 1]
        model = mk().fit(F, sex)
    return model, np.clip(oof, 0.01, 0.99)


def donor_index(Z, key, n_bank):
    Z = np.ascontiguousarray(np.asarray(Z, np.float64))
    return np.array([int.from_bytes(hashlib.sha256(key.encode() + r.tobytes()).digest()[:8], 'big') % n_bank for r in Z])


def fallback_target(Z, version):
    """Designer B's original fallback SBC target (a keyed Bernoulli(0.5) label per (SCHL, COW) cell); SUPERSEDED before any fit by
    PSEUDO2 (challenger finding 15: not the same type as SEX). Kept for the record; not used by stage_fit."""
    Z = np.asarray(Z, np.float64)
    return np.array([hashlib.sha256(f'DDR_B_SBC_FALLBACK|{20261021 + int(version)}|{r[3]:.6g}|{r[2]:.6g}'.encode()).digest()[0] & 1
                     for r in Z], dtype=np.int64)


def permuted(sex, version):
    return np.asarray(sex)[np.random.default_rng(np.random.SeedSequence([int(version), 20261021, 3])).permutation(len(sex))]


# ---------------------------------------------------------------- nested selection (weights refitted per outer fold)
def nested_releases(pI, pO, q, s, unit, version, alphas, q_rel=None, inc=True, occ=True):
    """{alpha: (rI, rO)} on MODEL_FIT: weights fitted on the other outer folds' out-of-fold statistics and target s, applied to the
    held-out fold. q_rel (optional) replaces q in the amplitude only (SHUF)."""
    of = FK.household_folds(unit, N_OUTER, f'DDR_DB_ncv_S{int(version)}')
    n = len(unit)
    out = {a: (np.zeros((n, K_INC)), np.zeros((n, K_OCC))) for a in alphas}
    qa = q if q_rel is None else q_rel
    for j in range(N_OUTER):
        tr, te = of != j, of == j
        st = fit_weights(pI[tr], pO[tr], q[tr], s[tr])
        for a in alphas:
            rI, rO = tilt(pI[te], pO[te], qa[te], st, a, inc=inc, occ=occ)
            out[a][0][te], out[a][1][te] = rI, rO
    return out


# ---------------------------------------------------------------- deployment
def _predict_payload(P, Z):
    prc = _prc()
    pI, pO = prc._proba(P['twin_inc'], Z), prc._proba(P['twin_occ'], Z)
    op = P['op']
    if op in ('ptc_r1', 'sbc_r1', 'ptc_r1_occ', 'ztilt_r1'):  # revision R1 configurations (ddr.db.ptc_r1), one family interface
        from .ptc_r1 import _predict_payload as r1
        return r1(P, Z)
    if op == 'dq':
        from .. import ctlkit as CK
        return CK.dq_release(P['dq_basis'], pI, pO, Z, float(P['delta']), P['dq_tag'])
    if op == 'shuf':
        q = P['bank_q'][donor_index(Z, P['donor_key'], len(P['bank_q']))]
    elif op == 'ztilt':
        q = np.clip(P['q_model'].predict_proba(zfeat(pI, pO))[:, 1], 0.01, 0.99)
    else:
        q = np.clip(prc._proba(P['q_model'], Z)[:, 1], 0.01, 0.99)
    return tilt(pI, pO, q, P['weights'], float(P['alpha']), inc=op != 'ptc_occ', occ=True)


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (ddr.deploy contract); deterministic; keys and draws other than 0 are refused."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    rI, rO = _predict_payload(P, FK.compact(X, names))
    return {'income': rI, 'occupation': rO}


def model_paths(config_id, version):
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- propensity teacher on a target (SEX-side settings)
def target_teacher(Z, a, unit, version, T):
    """HGB on Z for a binary target (the SEX-side 'sex' teacher's settings and iteration count T): full fit + out-of-fold P(a = 1 | x)
    over NRM_FA_xf_S<v>."""
    P = _prc()
    xf = P.household_folds(unit, P.N_XF, f'NRM_FA_xf_S{version}')
    a = np.asarray(a, np.int64)
    model = P.fit_hgb(P.HGB, Z, a, 2, T, version, P.CAT_Z)
    oof = np.zeros(len(Z))
    for f in range(P.N_XF):
        tr, te = xf != f, xf == f
        oof[te] = P._proba(P.fit_hgb(P.HGB, Z[tr], a[tr], 2, T, version, P.CAT_Z), Z[te])[:, 1]
    return model, np.clip(oof, 0.01, 0.99)


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, card, save, roles=None, log=print, budget_override=None, record_dir=None, save_rich=True):
    t0 = time.process_time()
    prc = _prc()
    from ..da import std as DA
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, S, cache_sha = DA.load_caches(version, row_sha)
    budget = budget_override or BUDGET
    pI, pO = T['oof']['inc'], T['oof']['occ']
    q = np.clip(S['oof']['sex'][:, 1], 0.01, 0.99)
    twin_L = FK.person_losses(pI, pO, yI, yO)
    # --- CAND: nested strength selection
    rel = nested_releases(pI, pO, q, sex, unit, version, STRENGTHS)
    L = {a: FK.person_losses(*rel[a], yI, yO) for a in STRENGTHS}
    sp = {a: FK.spend(L[a], twin_L, unit, sex) for a in STRENGTHS}
    a_star = FK.select_strength(sp, STRENGTHS, budget)
    st_full = fit_weights(pI, pO, q, sex)
    sel = {'alpha_star': a_star, 'budget': budget, 'path': {str(a): FK.jsonable_spend(v) for a, v in sp.items()},
           'weights_full': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in st_full.items()},
           'rule': 'per version: the largest grid alpha such that it and every smaller grid value meet estimate + 2 household-clustered '
                   'SE <= budget (S3 plus each task summed Brier 0.0045) on the nested MODEL_FIT spend (5 outer household folds '
                   'DDR_DB_ncv_S<v>; weights refitted on the other folds)'}
    log(json.dumps({'alpha_star': a_star}))
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'cache_sha256': cache_sha}
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the tilt weights and the strength / '
                'bracket selection (SEX or a SEX-independent target as the weights\' label); the task teachers and the propensity '
                'are the cached teachers (MODEL_FIT task labels; MODEL_FIT SEX); no CALIBRATION or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {'TWIN': (pI, pO)}

    def put(cfg, payload, oof_pair, extra, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        spx = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure,
                        people, hh, time.process_time() - t0,
                        {**extra, 'selection_alpha_star': a_star, 'twin_member': 'DA/DA_TWIN', 'cache_sha256': cache_sha,
                         'model_sha256': meta['model_sha256'], 'nested_spend_over_twin': FK.jsonable_spend(spx)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'nested_spend_over_twin': FK.jsonable_spend(spx), **extra}
        oof_views[cfg] = oof_pair

    out = {'version': int(version), 'selection': sel, 'cache_sha256': cache_sha}
    if a_star is None:
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'db') / f'FIT_V{int(version)}.json', out)
        return out
    cand_L = L[a_star]
    cand_rms = rms_tilt(pO, q, st_full)
    put('DB_PTC', {'op': 'ptc', 'q_model': S['models']['sex'], 'weights': st_full, 'alpha': a_star}, rel[a_star],
        {'op': 'ptc', 'alpha': a_star}, (0, 0))
    # --- SBC path: permuted SEX, then the PSEUDO2 fallback (ddr.ctlkit; challenger findings 6, 15, 16)
    from .. import ctlkit as CK
    tries, profiles = [], {'SEX': {'propensity_auc': FK.auc(q, sex), 'r2_logit_on_twin_pair': CK.r2_on_pair(q, pI, pO),
                                   'note': 'the cached SEX-side propensity (out-of-fold)'}}
    cand_sp_v = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())

    def sbc(target, label):
        qm, qa = target_teacher(Z, target, unit, version, int(S['T']['sex']))
        stt = fit_weights(pI, pO, qa, target)
        if not stt['cov_q_s'] > COV_MIN:
            tries.append({'target': label, 'degenerate': True, 'cov_q_s': stt['cov_q_s']})
            return None
        m0 = cand_rms / max(rms_tilt(pO, qa, stt), 1e-12)
        res = None
        for mult in SBC_SCALE_LADDER:
            m = m0 * mult
            grid = tuple(float(a * m) for a in STRENGTHS)
            relc = nested_releases(pI, pO, qa, np.asarray(target), unit, version, grid)
            Lc = {g: FK.person_losses(*relc[g], yI, yO) for g in grid}
            br = DA.bracket_points(Lc, cand_L, twin_L, unit, grid)
            res = {'q_model': qm, 'weights': stt, 'rel': relc, 'br': br, 'label': label, 'grid': grid, 'scale_m': m}
            if br['flag'] is None:
                break
        tries.append({'target': label, 'degenerate': False, 'cov_q_s': stt['cov_q_s'], 'scale_m': res['scale_m'],
                      'scale_m0': m0, 'bracket': res['br'], 'target_sex_corr': float(np.corrcoef(target, sex)[0, 1])})
        return res
    ch = sbc(permuted(sex, version), 'PERMUTED_SEX')
    pseudo_info = None
    if ch is None or ch['br']['flag'] is not None:
        make, pseudo_info = CK.pseudo2_select(Z, yI, yO, sex, q, np.asarray(data['row_id']), version, gammas=(1.0,))
        attr = make(1.0)
        profiles['PSEUDO2_gamma1'], _ = CK.attribute_profile(Z, attr, unit, pI, pO, version, prc.CAT_Z)
        ch = sbc(attr, 'PSEUDO2_gamma1')
    sbc_saved = []
    if ch is not None:
        for cfg, g in (('DB_SBC_LO', ch['br']['lo']), ('DB_SBC_HI', ch['br']['hi'])):
            if g is None:
                continue
            put(cfg, {'op': 'sbc', 'q_model': ch['q_model'], 'weights': ch['weights'], 'alpha': g}, ch['rel'][g],
                {'op': 'sbc', 'target': ch['label'], 'alpha': g, 'scale_m': ch['scale_m'], 'bracket_flag': ch['br']['flag']},
                (6 * len(tries), 0) if not sbc_saved else (0, 0))
            sbc_saved.append(cfg)
    # --- SPEND path: SEX-blind dithered quantization of the twin (ddr.ctlkit DQ; continuous, information removing)
    dq_basis = CK.dq_fit(pI, pO)
    dq_tag = f'DDR_DQ_S{int(version)}'
    dq_rel = {d: CK.dq_release(dq_basis, pI, pO, Z, d, dq_tag) for d in DQ_GRID}
    dq_L = {d: FK.person_losses(*dq_rel[d], yI, yO) for d in DQ_GRID}
    brs = DA.bracket_points(dq_L, cand_L, twin_L, unit, DQ_GRID)
    for cfg, d in (('DB_SPD_LO', brs['lo']), ('DB_SPD_HI', brs['hi'])):
        if d is None:
            continue
        put(cfg, {'op': 'dq', 'dq_basis': dq_basis, 'delta': d, 'dq_tag': dq_tag}, dq_rel[d],
            {'op': 'dq', 'delta': d, 'bracket_flag': brs['flag']}, (0, 0))
    # --- ZTILT (required; challenger finding 2): amplitude = P(SEX | the twin's release); first scale reaching CAND's nested spend
    zm, qz = ztilt_teacher(pI, pO, sex, unit, version)
    st_z = fit_weights(pI, pO, qz, sex)
    mz0 = cand_rms / max(rms_tilt(pO, qz, st_z), 1e-12)
    grid_z = tuple(float(a_star * mz0 * m) for m in SHUF_MULT)
    rel_z = nested_releases(pI, pO, qz, sex, unit, version, grid_z)
    Lz = {g: FK.person_losses(*rel_z[g], yI, yO) for g in grid_z}
    g_z = next((g for g in grid_z if float((Lz[g]['tot_nll'] - twin_L['tot_nll']).mean()) >= cand_sp_v), grid_z[-1])
    put('DB_ZTILT', {'op': 'ztilt', 'q_model': zm, 'weights': st_z, 'alpha': g_z}, rel_z[g_z],
        {'op': 'ztilt', 'alpha': g_z, 'grid': list(grid_z), 'reached_cand_spend': bool(float((Lz[g_z]['tot_nll'] - twin_L['tot_nll']).mean()) >= cand_sp_v),
         'spends': {str(g): float((Lz[g]['tot_nll'] - twin_L['tot_nll']).mean()) for g in grid_z},
         'qz_auc_oof': FK.auc(qz, sex)}, (1, 5))
    # --- SHUF: the candidate's weights, a keyed donor's propensity; one point at the first scale whose nested spend reaches CAND's
    key = f'DDR_DB_SHUF|V{int(version)}'
    q_don = q[donor_index(Z, key, len(q))]
    grid_s = tuple(float(a_star * m) for m in SHUF_MULT)
    rel_s = nested_releases(pI, pO, q, sex, unit, version, grid_s, q_rel=q_don)
    Ls = {g: FK.person_losses(*rel_s[g], yI, yO) for g in grid_s}
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    g_s = next((g for g in grid_s if float((Ls[g]['tot_nll'] - twin_L['tot_nll']).mean()) >= cand_sp), grid_s[-1])
    put('DB_SHUF', {'op': 'shuf', 'weights': st_full, 'alpha': g_s, 'bank_q': q.copy(), 'donor_key': key}, rel_s[g_s],
        {'op': 'shuf', 'alpha': g_s, 'grid': list(grid_s),
         'spends': {str(g): float((Ls[g]['tot_nll'] - twin_L['tot_nll']).mean()) for g in grid_s}}, (0, 0))
    # --- ablation: occupation-only tilt
    rel_o = nested_releases(pI, pO, q, sex, unit, version, (a_star,), inc=False)
    put('DB_PTC_OCC', {'op': 'ptc_occ', 'q_model': S['models']['sex'], 'weights': st_full, 'alpha': a_star}, rel_o[a_star],
        {'op': 'ptc_occ', 'alpha': a_star}, (0, 0))
    t1 = time.process_time()
    check = FK.mf_reader_check(oof_views, sex, unit, f'DDR_mfcheck_DB_S{int(version)}')
    rich_info = None
    if save_rich:
        from .. import richreaders as RR
        views = {k: v for k, v in oof_views.items() if k in saved or k == 'TWIN'}
        rich, rich_cpu = RR.train(views, sex, version)
        rich_info = {'sha256': RR.save(FAMILY, version, rich, {'configs': sorted(views), 'trained_on':
                                                                 'MODEL_FIT nested releases (TWIN: out-of-fold), MODEL_FIT SEX'}),
                     'cpu_s': rich_cpu}
    out.update({'saved': saved, 'sbc_tries': tries, 'sbc_target': (ch or {}).get('label'), 'spend_path_bracket': brs,
                'cand_rms_tilt': cand_rms, 'pseudo_info': pseudo_info, 'attribute_profiles': profiles, 'rich_readers': rich_info,
                'mf_reader_check': {'note': 'descriptive MODEL_FIT nested-release readers (household 5-fold cross-fitted); never a '
                                            'decision input and never a stop', 'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'db') / f'FIT_V{int(version)}.json', out)
    return out


# ---------------------------------------------------------------- selftests (invented data; no label file)
def selftest():
    rng = np.random.default_rng(5)
    n = 4000
    s = rng.integers(0, 2, n)
    q = np.clip(0.5 + 0.3 * (s - 0.5) + rng.normal(0, 0.15, n), 0.01, 0.99)
    pI = FK.clipn(np.column_stack([1 - (0.3 + 0.2 * s), 0.3 + 0.2 * s]) + rng.uniform(0, 0.05, (n, 2)))
    pO = FK.clipn(rng.dirichlet(np.ones(5), n) + 0.3 * np.eye(5)[np.minimum(rng.integers(0, 4, n) + s, 4)])
    st = fit_weights(pI, pO, q, s)
    rI, rO = tilt(pI, pO, q, st, 1.0)
    sc = s - s.mean()
    ok = {'income_cov_cancelled': float(np.abs((clog(rI) * sc[:, None]).mean(0)).max()) < 1e-3}
    k = pO.argmax(1)
    ok['occ_cov_cancelled_in_strata'] = max(float(np.abs((clog(rO[k == j]) * (s[k == j] - s[k == j].mean())[:, None]).mean(0)).max())
                                            for j in range(5) if (k == j).sum() >= MIN_STRATUM) < 1e-3
    ok['alpha0_identity'] = np.allclose(tilt(pI, pO, q, st, 0.0)[1], FK.clipn(pO))
    ok['decisions_shift_bounded'] = float(np.mean(rO.argmax(1) != pO.argmax(1))) < 0.5
    unit = np.arange(n) // 2
    rel = nested_releases(pI, pO, q, s, unit, 0, (0.5, 1.0))
    ok['nested_shapes'] = rel[1.0][1].shape == (n, 5) and np.allclose(rel[1.0][1].sum(1), 1)
    ok['fallback_binary'] = set(np.unique(fallback_target(rng.integers(0, 9, (50, 6)).astype(float), 0))) <= {0, 1}
    ok['all_ok'] = all(ok.values())
    return ok


def selftest_stage(n=2500, seed=9, max_iter=30):
    """stage_fit end to end on INVENTED labels attached to admitted MODEL_FIT features (label-free); teachers fitted on the invented
    labels stand in for the caches; model files go to a temporary directory; nothing is saved to the campaign's runs."""
    import tempfile
    from pathlib import Path as _P
    from ..da import std as DA
    prc = _prc()
    F = C.features('MODEL_FIT')
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(F['row_id']), n, replace=False))
    Zc = FK.compact(F['X'][idx], F['feature_names'])
    hours = Zc[:, 1]
    sex = (rng.random(n) < 1.0 / (1.0 + np.exp((hours - np.median(hours)) / 6.0))).astype(np.int64)   # invented: x-predictable
    data = {'X': F['X'][idx], 'feature_names': F['feature_names'], 'row_id': F['row_id'][idx], 'unit': F['unit'][idx],
            'income': (rng.random(n) < 0.3 + 0.3 * sex).astype(np.int64), 'occupation': np.minimum(rng.integers(0, 5, n) + sex, 4),
            'sex': sex}
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    hp = {**prc.HGB, 'max_iter': int(max_iter)}
    old_hgb = prc.HGB
    prc.HGB = hp
    try:
        tch = prc.fit_teachers(Z, lab, unit, 0, hp, ('inc', 'occ'))
        sxs = prc.fit_teachers(Z, lab, unit, 0, hp, ('sex', 'occ_s'))
    finally:
        prc.HGB = old_hgb
    Tc = {'T': tch['T'], 'models': tch['models'], 'oof': tch['oof'], 'row_id_sha256': row_sha}
    Sc = {'T': sxs['T'], 'models': sxs['models'], 'oof': sxs['oof'], 'row_id_sha256': row_sha}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = (C.RUNS, DA.load_caches, prc.HGB)
    tmp = _P(tempfile.mkdtemp(prefix='ddr_db_selftest_'))
    try:
        C.RUNS = tmp
        DA.load_caches = lambda v, rs: (Tc, Sc, {'TASK.pkl': 'invented', 'SEXSIDE.pkl': 'invented'})
        prc.HGB = hp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s: None,
                        budget_override={k: 100.0 for k in BUDGET}, record_dir=False, save_rich=False)
    finally:
        C.RUNS, DA.load_caches, prc.HGB = old
    ok = {'cand_saved': 'DB_PTC' in saved, 'alpha_selected': out['selection']['alpha_star'] in STRENGTHS}
    ok['controls_saved'] = any(c.startswith('DB_SPD') for c in saved) and 'DB_SHUF' in saved and 'DB_PTC_OCC' in saved \
        and 'DB_ZTILT' in saved
    ok['records_valid'] = all(r['kind'] == CONFIGS[c]['kind'] for c, (_, r) in saved.items())
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['sbc_target'] = out.get('sbc_target')
    ok['sbc_tries'] = [{k: v for k, v in t.items() if k in ('target', 'degenerate', 'cov_q_s')} for t in out.get('sbc_tries', [])]
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ddr.db.ptc')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    st2 = sub.add_parser('selftest_stage'); st2.add_argument('--n', type=int, default=2500); st2.add_argument('--max-iter', type=int, default=30)
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    if a.cmd == 'selftest_stage':
        r = selftest_stage(n=a.n, max_iter=a.max_iter); print(json.dumps(r, default=str)); return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'alpha_star': out['selection']['alpha_star'], 'saved': sorted(out['saved']), 'sbc_target': out.get('sbc_target'),
                      'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

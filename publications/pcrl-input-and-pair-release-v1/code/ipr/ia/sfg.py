"""IA: shared SEX-aware field generalization (SFG; Designer A, family IA). PCRL campaign `pcrl_input_and_pair_release_v1`;
EXPLORATORY DEVELOPMENT on previously used public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not
confirmation.

The service maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women) is a
training label on MODEL_FIT only (it orders a fixed menu of field generalizations), never a runtime input or output.

Operation (card IA_SFG_v1; Designer A's PROPOSAL.md and mech_sketch.py, integrated, with the challenger's required changes:
budget S3, rung k = 1 in the grid, the intermediate levels promoted into the menu, the permuted-SEX ladder as the only gating SBC,
BLIND as the targeting-off ablation, the generic release-only tilt as the declared RO fallback, no PROJ / ATTR):
  menu      17 fixed, label-free generalization steps, a chain of two or three coarser levels per field (MENU below; label-free maps of the compact
            coding Z = ohn.fam_a.compact_po(X): COW and MAR are block indices = admitted code - 1, SCHL the SCHL code, POBP the
            numeric place-of-birth code, AGEP / WKHP standardized with their admitted levels)
  ladder    single-step statistics on MODEL_FIT (5 household folds, tag IPR_IA_ladder_S<v>, HGB A1_PO_16 settings at the cached
            twin's iteration counts): dT_s = out-of-fold income + occupation NLL increase over the identity, dI_s = out-of-fold
            log-loss increase of an HGB model of the target (SEX for the candidate) on h_s(Z) over one on Z; greedy order by
            incremental dI / dT, each field moving only to a coarser level (build_ladder)
  models    both task models on h_k(Z) by the twin's construction (nrm.fa.prc.fit_teachers: inner household-CV T*, tag
            OHN_A_cv_S<v>; full refit; out-of-fold on NRM_FA_xf_S<v>, the folds of the cached twin's out-of-fold predictions)
  release   P_inc = f_inc(h_k(x)), P_occ = f_occ(h_k(x)); argmax decisions; deterministic; no SEX and no propensity at runtime;
            k = 0 is the twin
  strength  k* = the largest rung k = 1, 2, ... such that it and every smaller rung keep MODEL_FIT out-of-fold spend over the twin
            within budget S3 (estimate + 2 household-clustered SE on every component); rungs are fitted in order and the scan stops
            at the first failing rung (no larger rung can be selected)
Matched controls (PROTOCOL 4; card IA_SFG_v1 fixes them before the first fit):
  SBC path  the same menu, ladder rule and models with the target SEX replaced by a fixed-seed permutation of the MODEL_FIT SEX column
            (fallback PSEUDO2 only by the protocol rule: if the permuted path cannot reach CAND's spend + 2 SE); rungs k = 1 .. L;
            two bracket points
  SPEND     DQ of the shared twin (ipr.paths, 1.1-ratio step grid); two bracket points, saved once under CTL (card CTL_PATHS_v1)
  RO        GENERIC_ZTILT, the card's declared fallback (the twin's release identifies the record for about 93% of MODEL_FIT people,
            so the SUBSTITUTED ladder is SFG's up to the surrogate learner; it is computed here on MODEL_FIT, descriptively, and
            compared with SFG's ladder, but not released); two bracket points, saved once under CTL
Ablation: BLIND, the same menu and models with the ladder ordered by task spend only (the targeting component removed), at the
first rung whose MODEL_FIT spend reaches CAND's. Descriptive MODEL_FIT-only: cross-fitted readers and the data-rich readers.

Commands (real data only through the campaign queue after the receipt and the committed card):
    python3 -m ipr.ia.sfg selftest | selftest_maps
    python3 -m ipr.ia.sfg fit --version V --card results/pcrl_input_and_pair_release_v1/cards/IA_SFG_v1.md
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

FAMILY = 'IA'
K_INC, K_OCC = 2, 5
COL = {'AGEP': 0, 'WKHP': 1, 'COW': 2, 'SCHL': 3, 'MAR': 4, 'POBP': 5}
MENU = {'AGEP': ('AGE5', 'AGE10'), 'WKHP': ('WK6', 'WK5', 'WK3'), 'COW': ('COW5', 'COW4', 'COW3'), 'SCHL': ('SCH5', 'SCH4', 'SCH3'),
        'MAR': ('MAR3', 'MAR2', 'MAR1'), 'POBP': ('POB4', 'POB2', 'POB1')}
FIELD = {s: f for f, lv in MENU.items() for s in lv}
GRID = tuple(range(1, 18))            # rungs k = 1 .. 17 (the scan stops at the first rung that fails the budget)
N_LADDER_FOLDS = 5
CODE_MODULES = ('ipr.ia.sfg', 'ipr.famkit', 'ipr.paths', 'ipr.ctlkit', 'ipr.ctl.twin', 'nrm.fa.prc', 'ohn.fam_a')


def _prc():
    from nrm.fa import prc
    return prc


# ---------------------------------------------------------------- the menu (label-free, deterministic from Z)
def field_maps(Z):
    """Age levels (40 admitted levels = ages 25-64) and the hours scale of the standardized WKHP (modal value = 40 h, minimum = 1 h)."""
    age_lv = np.unique(Z[:, 0])
    wk_lv, wk_ct = np.unique(Z[:, 1], return_counts=True)
    return {'age_levels': age_lv, 'z40': float(wk_lv[np.argmax(wk_ct)]), 'zmin': float(wk_lv.min())}


def _hours(Z, m):
    return np.round(1 + (Z[:, 1] - m['zmin']) * 39.0 / (m['z40'] - m['zmin']))


def apply_step(Zh, Z, st, m):
    c = Z[:, COL[FIELD[st]]]
    if st in ('AGE5', 'AGE10'):
        idx = np.clip(np.searchsorted(m['age_levels'], c), 0, len(m['age_levels']) - 1)
        Zh[:, 0] = idx // (5 if st == 'AGE5' else 10)
    elif st == 'WK6':
        Zh[:, 1] = np.digitize(_hours(Z, m), [20, 35, 40, 41, 50])
    elif st == 'WK5':                          # <35 | 35-39 | 40 | 41-49 | 50+ (WK6 with 1-19 and 20-34 merged)
        Zh[:, 1] = np.digitize(_hours(Z, m), [35, 40, 41, 50])
    elif st == 'WK3':
        Zh[:, 1] = np.digitize(_hours(Z, m), [35, 45])
    elif st == 'SCH5':
        Zh[:, 3] = np.digitize(c, [16, 18, 21, 22])
    elif st == 'SCH4':                         # <=17 | 18-20 | 21 | >=22 (SCH5 with <=15 and 16-17 merged)
        Zh[:, 3] = np.digitize(c, [18, 21, 22])
    elif st == 'SCH3':
        Zh[:, 3] = np.digitize(c, [18, 21])
    elif st == 'COW5':                         # for-profit | nonprofit | government (local, state, federal) | self-employed | unpaid
        Zh[:, 2] = np.array([0, 1, 2, 2, 2, 3, 3, 4, 0])[np.clip(c.astype(int), 0, 8)]
    elif st == 'COW4':                         # private (for-profit and nonprofit) | government | self-employed | unpaid
        Zh[:, 2] = np.array([0, 0, 1, 1, 1, 2, 2, 3, 0])[np.clip(c.astype(int), 0, 8)]
    elif st == 'COW3':                         # private (incl. unpaid) | government | self-employed
        Zh[:, 2] = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0])[np.clip(c.astype(int), 0, 8)]
    elif st == 'MAR3':                         # married | widowed, divorced, separated | never married
        Zh[:, 4] = np.array([0, 1, 1, 1, 2, 0])[np.clip(c.astype(int), 0, 5)]
    elif st == 'MAR2':                         # married | not married (previously and never married merged)
        Zh[:, 4] = np.array([0, 1, 1, 1, 1, 0])[np.clip(c.astype(int), 0, 5)]
    elif st == 'MAR1':
        Zh[:, 4] = 0
    elif st == 'POB4':                         # Florida | other US state | US territories | foreign or other
        Zh[:, 5] = np.where(c == 12, 0, np.where(c < 60, 1, np.where(c < 100, 2, 3)))
    elif st == 'POB2':                         # Florida | elsewhere
        Zh[:, 5] = np.where(c == 12, 0, 1)
    elif st == 'POB1':
        Zh[:, 5] = 0
    else:
        raise KeyError(st)
    return Zh


def generalize(Z, steps, m):
    Zh = np.array(Z, dtype=np.float64, copy=True)
    for st in steps:
        apply_step(Zh, Z, st, m)
    return Zh


def compose(ladder, k):
    """The first k ladder steps; a field keeps its last (coarsest) step."""
    cur = {}
    for st in ladder[:k]:
        cur[FIELD[st]] = st
    return [cur[f] for f in MENU if f in cur]


def build_ladder(stats, order):
    """Greedy: each field moves only to a coarser level; incremental ratio against the field's current level.
    order 'target': largest incremental dI / dT first; 'blind': smallest incremental dT first (no target)."""
    cur = {f: None for f in MENU}
    ladder = []
    while True:
        cand = []
        for f, lv in MENU.items():
            i0 = -1 if cur[f] is None else lv.index(cur[f])
            T0 = 0.0 if cur[f] is None else stats[cur[f]]['dT']
            I0 = 0.0 if cur[f] is None else stats[cur[f]].get('dI', 0.0)
            for st in lv[i0 + 1:]:
                dT = max(stats[st]['dT'] - T0, 1e-5)
                dI = stats[st].get('dI', 0.0) - I0
                key = -dT if order == 'blind' else dI / dT
                cand.append((key, -dT, st))
        if not cand:
            return ladder
        st = max(cand)[2]
        cur[FIELD[st]] = st
        ladder.append(st)


# ---------------------------------------------------------------- ladder statistics (MODEL_FIT; labels arrive as arguments)
def _oof_fixed(Zh, y, K, folds, T, version):
    prc = _prc()
    P = np.zeros((len(y), K))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        P[te] = prc._proba(prc.fit_hgb(prc.HGB, Zh[tr], y[tr], K, T, version, prc.CAT_Z), Zh[te])
    return P


def _nll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))


def task_stats(Z, yI, yO, unit, version, T, m):
    """dT for every single step (the task part of the ladder; shared by every ordering)."""
    folds = FK.household_folds(unit, N_LADDER_FOLDS, f'IPR_IA_ladder_S{int(version)}')
    base = {'inc': _nll(_oof_fixed(Z, yI, K_INC, folds, T['inc'], version), yI).mean(),
            'occ': _nll(_oof_fixed(Z, yO, K_OCC, folds, T['occ'], version), yO).mean()}
    out = {}
    for st in FIELD:
        Zh = generalize(Z, [st], m)
        dI_ = _nll(_oof_fixed(Zh, yI, K_INC, folds, T['inc'], version), yI).mean() - base['inc']
        dO_ = _nll(_oof_fixed(Zh, yO, K_OCC, folds, T['occ'], version), yO).mean() - base['occ']
        out[st] = {'dT': float(dI_ + dO_), 'dT_inc': float(dI_), 'dT_occ': float(dO_)}
    return out


def target_stats(Z, target, unit, version, T_sex, m):
    """dI for every single step: the out-of-fold log-loss increase of an HGB model of the binary target."""
    folds = FK.household_folds(unit, N_LADDER_FOLDS, f'IPR_IA_ladder_S{int(version)}')
    t = np.asarray(target, dtype=np.int64)
    base = _nll(_oof_fixed(Z, t, 2, folds, T_sex, version), t).mean()
    return {st: float(_nll(_oof_fixed(generalize(Z, [st], m), t, 2, folds, T_sex, version), t).mean() - base) for st in FIELD}


def _oof_soft(Zh, z, folds, T, version):
    """Out-of-fold P(target = 1 | Zh) for a soft target z in [0, 1]: HGB on the rows stacked twice with labels 1 / 0 and weights
    z / 1 - z (the A1_PO_16 settings at T iterations)."""
    prc = _prc()
    P = np.zeros(len(z))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        X2 = np.vstack([Zh[tr], Zh[tr]])
        y2 = np.concatenate([np.ones(tr.sum(), np.int64), np.zeros(tr.sum(), np.int64)])
        w2 = np.concatenate([z[tr], 1.0 - z[tr]])
        m = prc.make_hgb(prc.HGB, T, version, prc.CAT_Z)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(X2, y2, sample_weight=w2)
        P[te] = np.clip(m.predict_proba(Zh[te])[:, 1], 1e-6, 1 - 1e-6)
    return P


def target_stats_soft(Z, z, unit, version, T_sex, m):
    """dI for every single step with a soft target z (the soft log-loss increase of the out-of-fold model on h_s(Z) over Z)."""
    folds = FK.household_folds(unit, N_LADDER_FOLDS, f'IPR_IA_ladder_S{int(version)}')
    z = np.clip(np.asarray(z, np.float64), 1e-6, 1 - 1e-6)
    ll = lambda P: float(np.mean(-(z * np.log(P) + (1 - z) * np.log(1 - P))))
    base = ll(_oof_soft(Z, z, folds, T_sex, version))
    return {st: ll(_oof_soft(generalize(Z, [st], m), z, folds, T_sex, version)) - base for st in FIELD}


def merge_stats(task, target=None):
    return {st: {**task[st], 'dI': (target or {}).get(st, 0.0)} for st in task}


# ---------------------------------------------------------------- models on h_k(Z) (the twin's construction)
def fit_rung(Z, steps, m, lab, unit, version):
    """Task models on h(Z) for the given steps: T* (inner household CV), full refit, out-of-fold on NRM_FA_xf_S<v>."""
    prc = _prc()
    Zh = generalize(Z, steps, m)
    tch = prc.fit_teachers(Zh, lab, unit, int(version), prc.HGB, ('inc', 'occ'))
    return {'steps': list(steps), 'T': {k: int(v) for k, v in tch['T'].items()}, 'models': tch['models'],
            'oof': (tch['oof']['inc'], tch['oof']['occ']), 'n_fits': (2 * (prc.N_INNER + 1), 2 * prc.N_XF)}


def release_steps(P, Z):
    prc = _prc()
    Zh = generalize(Z, P['steps'], P['maps'])
    return prc._proba(P['inc'], Zh), prc._proba(P['occ'], Zh)


def selftest_maps():
    """Label-free check of the menu on the admitted MODEL_FIT features: every step maps into a small set of levels and only coarsens."""
    F = C.features('MODEL_FIT')
    Z = FK.compact(F['X'], F['feature_names'])
    m = field_maps(Z)
    ok = {'hours_range': bool(_hours(Z, m).min() == 1 and 95 <= _hours(Z, m).max() <= 99),
          'age_levels_40': len(m['age_levels']) == 40}
    for st in FIELD:
        col = COL[FIELD[st]]
        h = generalize(Z, [st], m)[:, col]
        n0, n1 = len(np.unique(Z[:, col])), len(np.unique(h))
        ok[f'{st}_coarsens'] = bool(n1 < n0)
        _, inv = np.unique(Z[:, col], return_inverse=True)
        ok[f'{st}_function_of_field'] = bool(all(len(np.unique(h[np.asarray(inv).ravel() == i])) == 1 for i in range(n0)))
    ok['all_ok'] = all(ok.values())
    return ok


def selftest():
    """Invented-data checks of the ladder and composition logic (no file, no label)."""
    stats = {st: {'dT': 0.001 * (i + 1), 'dI': 0.0005 * ((7 * i) % 5)} for i, st in enumerate(FIELD)}
    L = build_ladder(stats, 'target')
    B = build_ladder(stats, 'blind')
    ok = {'ladder_covers_fields': sorted(compose(L, len(L))) == sorted(lv[-1] for lv in MENU.values()),
          'blind_cheapest_first': stats[B[0]]['dT'] == min(stats[s]['dT'] for s in FIELD if s in ('AGE5', 'WK6', 'COW5', 'SCH5', 'MAR3', 'POB4')
                                                              or s in ('AGE10', 'WK3', 'COW3', 'SCH3', 'MAR1', 'POB1')),
          'one_step_per_field': all(len({FIELD[x] for x in compose(L, k)}) == len(compose(L, k)) for k in range(len(L) + 1)),
          'only_coarser': all(MENU[FIELD[s]].index(s) >= 0 for s in L)}
    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.choice(np.linspace(-1.8, 1.7, 40), 500), rng.choice(np.linspace(-3.5, 5.2, 87), 500),
                         rng.integers(0, 8, 500), rng.choice([1, 3, 4] + list(range(6, 25)), 500), rng.integers(0, 5, 500),
                         rng.choice([1, 12, 36, 72, 110, 303, 999], 500)]).astype(float)
    m = field_maps(Z)
    Zh = generalize(Z, compose(L, 4), m)
    ok['generalize_shape'] = Zh.shape == Z.shape
    ok['all_ok'] = all(ok.values())
    return ok


# ---------------------------------------------------------------- configurations, budget and control switches (fixed by the card)
BUDGET = {'occ_nll': 0.012, 'inc_nll': 0.006, 'occ_brier_men': 0.0045, 'occ_brier_women': 0.0045, 'occ_acc': 0.008, 'inc_acc': 0.006}  # S3
CTL_PATHS_CARD = 'results/pcrl_input_and_pair_release_v1/cards/CTL_PATHS_v1.md'
CONFIGS = {
    'IA_SFG': {'kind': 'candidate', 'what': 'shared SEX-aware field generalization at the MODEL_FIT-selected rung k* (both recipients)'},
    'IA_SBC_LO': {'kind': 'control', 'what': 'SBC path, lower bracket point: the same menu, ladder rule and models with a SEX-independent target'},
    'IA_SBC_HI': {'kind': 'control', 'what': 'SBC path, upper bracket point: the same menu, ladder rule and models with a SEX-independent target'},
    'IA_BLIND': {'kind': 'ablation', 'what': 'targeting off: the same menu and models with the ladder ordered by task spend only (BLIND), '
                                         'first rung whose MODEL_FIT spend reaches CAND\'s'},
}
PURPOSE = ('IA SFG: SEX-ordered field-generalization ladder, task models on generalized inputs, strength selection, matched control paths '
           '(SBC, SPEND, RO, BLIND) and the ablation on MODEL_FIT (SEX as a training label)')


def permuted_sex(sex, version):
    rng = np.random.default_rng(np.random.SeedSequence([int(version), 20261022, 1]))
    return np.asarray(sex)[rng.permutation(len(sex))]


# ---------------------------------------------------------------- deployment
def _predict_payload(P, Z):
    from ..ctl import twin as TW
    op = P['op']
    if op == 'sfg':
        return release_steps(P, Z)
    if op == 'osg':                                  # IA revision R1 (card IA_OSG_R1): occupation-side generalization
        from .osg import release_osg
        return release_osg(P, Z)
    tI, tO = TW.twin_vectors(P, Z)
    if op in ('dq', 'ro'):
        from .. import paths as PA
        return PA.predict_path(P, tI, tO, Z)
    raise ValueError(f'unknown operation {op}')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (ipr.deploy contract): the configuration's vectors from admitted features alone; deterministic."""
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


# ---------------------------------------------------------------- the fit stage (one unit per version)
def scan_path(Z, ladder, m, lab, unit, version, twin_L, cand_L, start=1, log=None):
    """Rungs k = start .. len(ladder) of a ladder by the twin's construction, stopping after the first rung whose spend reaches CAND's
    spend + 2 household-clustered SE of the paired difference (the bracket rule needs nothing beyond it)."""
    yI, yO = lab['income'], lab['occupation']
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    rungs, L = {}, {}
    for k in range(start, len(ladder) + 1):
        r = fit_rung(Z, compose(ladder, k), m, lab, unit, version)
        rungs[k], L[k] = r, FK.person_losses(*r['oof'], yI, yO)
        d = L[k]['tot_nll'] - cand_L['tot_nll']
        sp = float((L[k]['tot_nll'] - twin_L['tot_nll']).mean())
        if log:
            log(f'scan rung {k} steps {r["steps"]} spend {sp:.5f} (cand {cand_sp:.5f})')
        if sp >= cand_sp + 2.0 * FK.cluster_se(d, unit):
            break
    return rungs, L


def _fineness_inputs(br, cand_sp):
    """grid_fineness inputs of one path and version from a bracket record (ipr.paths.bracket_points)."""
    rows = br['grid']
    lo = 0.0 if br['lo'] is None else rows[str(br['lo'])]['spend']
    hi = rows[str(br['hi'])]
    return {'cand': cand_sp, 'lower': lo, 'upper': hi['spend'], 'upper_se': hi['se_diff'],
            'max': max(r['spend'] for r in rows.values())}


def stage_fit(version, data, card, save, roles=None, log=print, record_dir=None, save_rich=True, budget=None, grid=GRID):
    from ..ctl import twin as TW
    from .. import paths as PA
    from .. import ctlkit as CK
    t0 = time.process_time()
    prc = _prc()
    Z, lab, unit, names, row_sha = prc._prep(data, sex=True)
    yI, yO, sex = lab['income'], lab['occupation'], lab['sex']
    T, tsha = TW.load_task_cache(version, row_sha)
    S, ssha = TW.load_sex_cache(version, row_sha)
    budget = budget or BUDGET
    tI, tO = T['oof']['inc'], T['oof']['occ']
    twin_L = FK.person_losses(tI, tO, yI, yO)
    m = field_maps(Z)
    Tfix = {'inc': int(T['T']['inc']), 'occ': int(T['T']['occ'])}
    Tsex = int(S['T']['sex'])
    # 1. the ladder (the task part is shared by every ordering)
    ts = task_stats(Z, yI, yO, unit, version, Tfix, m)
    tg = target_stats(Z, sex, unit, version, Tsex, m)
    ladder = build_ladder(merge_stats(ts, tg), 'target')
    log(json.dumps({'ladder': ladder, 'cpu_s': round(time.process_time() - t0, 1)}))
    # 2. candidate rungs, fitted in order; the scan stops at the first rung that fails the budget
    rungs, Lk, path_sp, k_star = {}, {}, {}, None
    for k in [g for g in grid if g <= len(ladder)]:
        rungs[k] = fit_rung(Z, compose(ladder, k), m, lab, unit, version)
        Lk[k] = FK.person_losses(*rungs[k]['oof'], yI, yO)
        path_sp[k] = FK.spend(Lk[k], twin_L, unit, sex)
        ok = FK.meets(path_sp[k], budget)
        log(json.dumps({'rung': k, 'steps': rungs[k]['steps'], 'tot_nll': path_sp[k]['tot_nll'], 'meets_budget': ok,
                        'cpu_s': round(time.process_time() - t0, 1)}))
        if not ok:
            break
        k_star = k
    sel = {'ladder': ladder, 'single_step_stats': merge_stats(ts, tg), 'grid_scanned': sorted(rungs), 'budget': budget,
           'budget_name': 'S3', 'k_star': k_star, 'path': {str(k): FK.jsonable_spend(sp) for k, sp in path_sp.items()},
           'steps': {str(k): rungs[k]['steps'] for k in rungs}, 'T': {str(k): rungs[k]['T'] for k in rungs},
           'rule': 'per version: the largest rung k such that it and every smaller rung meet estimate + 2 household-clustered SE <= '
                   'S3 on the six components (MODEL_FIT out-of-fold over the twin, folds NRM_FA_xf_S<v>, the twin construction); '
                   'rungs are fitted in order and the scan stops at the first failing rung'}
    roles = roles or FK.role_features()
    base = {'feature_names': tuple(names), 'twin_inc': T['models']['inc'], 'twin_occ': T['models']['occ'], 'maps': m,
            'task_cache_sha256': tsha, 'sex_cache_sha256': ssha}
    exposure = ('MODEL_FIT income, occupation and SEX labels of all 20,000 MODEL_FIT people reach the ladder order (SEX or a SEX-'
                'independent target as a training label), the task models on generalized inputs (income, occupation) and the strength / '
                'bracket selection (out-of-fold, household folds); no CALIBRATION or MODEL_SELECT label')
    people, hh = len(Z), len(np.unique(unit))
    saved, oof_views = {}, {'CTL_TWIN': (tI, tO)}

    def put(cfg, payload, oof_pair, extra, fits):
        meta = FK.write_model(FAMILY, cfg, version, {**base, **payload})
        arrays = FK.release_all(predict, cfg, version, roles)
        sp = FK.spend(FK.person_losses(*oof_pair, yI, yO), twin_L, unit, sex)
        rec = FK.record(card, CONFIGS[cfg]['kind'], CONFIGS[cfg]['what'], fits, 'MODEL_FIT: income, occupation, sex', exposure, people,
                        hh, time.process_time() - t0, {**extra, 'selection_k_star': k_star, 'model_sha256': meta['model_sha256'],
                                                       'oof_spend_over_twin': FK.jsonable_spend(sp)})
        save(FAMILY, cfg, version, arrays, rec)
        saved[cfg] = {'oof_spend_over_twin': FK.jsonable_spend(sp), **extra}
        oof_views[cfg] = oof_pair

    def put_rung(cfg, r, extra):
        put(cfg, {'op': 'sfg', 'steps': r['steps'], 'inc': r['models']['inc'], 'occ': r['models']['occ']}, r['oof'],
            {'op': 'sfg', 'steps': r['steps'], 'T': r['T'], **extra}, r['n_fits'])

    out = {'version': int(version), 'selection': sel, 'task_cache_sha256': tsha, 'sex_cache_sha256': ssha}
    if k_star is None:
        log(json.dumps({'stop': 'no rung meets the budget: no candidate is released'}))
        out.update({'saved': saved, 'stop': 'NO_FEASIBLE_STRENGTH', 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'ia') / f'FIT_V{int(version)}.json', out)
        return out
    cand_L = Lk[k_star]
    cand_sp = float((cand_L['tot_nll'] - twin_L['tot_nll']).mean())
    put_rung('IA_SFG', rungs[k_star], {'k': k_star, 'ladder': ladder, 'strength': k_star})
    # 3. SBC path: the same ladder rule with a permuted SEX column (fallback PSEUDO2 only by the protocol rule)
    tries = []
    perm = permuted_sex(sex, version)
    lad_p = build_ladder(merge_stats(ts, target_stats(Z, perm, unit, version, Tsex, m)), 'target')
    r_p, L_p = scan_path(Z, lad_p, m, lab, unit, version, twin_L, cand_L, log=lambda s: log(s))
    br = PA.bracket_points(L_p, cand_L, twin_L, unit, tuple(sorted(L_p)))
    tries.append({'target': 'PERMUTED_SEX', 'ladder': lad_p, 'bracket': br})
    chosen = {'label': 'PERMUTED', 'rungs': r_p, 'ladder': lad_p, 'br': br}
    pseudo_info = None
    if br['flag'] is not None and max(L_p) == len(lad_p):
        make, pseudo_info = CK.pseudo2_select(Z, yI, yO, sex, np.clip(S['oof']['sex'][:, 1], 0.01, 0.99),
                                              np.asarray(data['row_id']), version)
        for g in CK.GAMMAS:
            a_ = make(g)
            lad_a = build_ladder(merge_stats(ts, target_stats(Z, a_, unit, version, Tsex, m)), 'target')
            r_a, L_a = scan_path(Z, lad_a, m, lab, unit, version, twin_L, cand_L, log=lambda s: log(s))
            br_a = PA.bracket_points(L_a, cand_L, twin_L, unit, tuple(sorted(L_a)))
            tries.append({'target': f'PSEUDO2_gamma{g:g}', 'ladder': lad_a, 'bracket': br_a})
            chosen = {'label': 'PSEUDO2', 'rungs': r_a, 'ladder': lad_a, 'br': br_a}
            if br_a['flag'] is None:
                break
    br = chosen['br']
    for cfg, k in (('IA_SBC_LO', br['lo']), ('IA_SBC_HI', br['hi'])):
        if k is None:
            continue
        put_rung(cfg, chosen['rungs'][k], {'k': k, 'ladder': chosen['ladder'], 'target': chosen['label'], 'bracket_flag': br['flag'],
                                           'strength': k})
    log(json.dumps({'sbc': chosen['label'], 'lo': br['lo'], 'hi': br['hi'], 'flag': br['flag'], 'cpu_s': round(time.process_time() - t0, 1)}))
    # 4. ablation BLIND: ladder by task spend only, first rung whose spend reaches CAND's
    lad_b = build_ladder(merge_stats(ts), 'blind')
    r_b, L_b = {}, {}
    k_b = None
    for k in range(1, len(lad_b) + 1):
        r_b[k] = fit_rung(Z, compose(lad_b, k), m, lab, unit, version)
        L_b[k] = FK.person_losses(*r_b[k]['oof'], yI, yO)
        if float((L_b[k]['tot_nll'] - twin_L['tot_nll']).mean()) >= cand_sp:
            k_b = k
            break
    blind_flag = None
    if k_b is None:
        k_b, blind_flag = max(r_b), 'BLIND_MAY_NOT_REACH_CAND_SPEND'
    put_rung('IA_BLIND', r_b[k_b], {'k': k_b, 'ladder': lad_b, 'target': 'NONE_TASK_SPEND_ORDER', 'flag': blind_flag, 'strength': k_b})
    blind = {'ladder': lad_b, 'k': k_b, 'flag': blind_flag,
             'scan': {str(k): float((L_b[k]['tot_nll'] - twin_L['tot_nll']).mean()) for k in L_b}}
    # 5. SPEND (DQ) and RO (GENERIC_ZTILT, declared fallback) on the shared twin; points saved once under CTL
    sr = PA.spend_and_ro(tI, tO, Z, yI, yO, sex, unit, version, cand_L, twin_L, FAMILY, log=lambda s: log(s))
    shared_ids, shared_status = PA.save_shared(sr, version, roles, save, {'feature_names': tuple(names), 'twin_inc': T['models']['inc'],
                                                                         'twin_occ': T['models']['occ'], 'task_cache_sha256': tsha},
                                               people, hh, time.process_time() - t0, yI, yO, twin_L, unit, sex, FAMILY,
                                               log=lambda s: log(s))
    for key in ('SPEND', 'RO'):
        for side, (g, payload, oof_pair) in sr[key]['points'].items():
            oof_views[shared_ids[key][side]] = oof_pair
    # 6. descriptive: the SUBSTITUTED ladder (the release-only surrogate as a soft target), computed and compared, not released
    _, qz = CK.ro_teacher(tI, tO, sex, unit, version)
    lad_s = build_ladder(merge_stats(ts, target_stats_soft(Z, qz, unit, version, Tsex, m)), 'target')
    k_cmp = min(len(lad_s), len(ladder))
    substituted = {'ladder': lad_s, 'equals_sfg_ladder': lad_s == ladder,
                   'first_rungs_equal': [compose(lad_s, k) == compose(ladder, k) for k in range(1, k_cmp + 1)],
                   'rung_k_star_steps_equal': compose(lad_s, k_star) == compose(ladder, k_star),
                   'note': 'descriptive (not released; not a configuration): the SFG ladder rule with the MODEL_FIT SEX column replaced '
                           'by the cross-fitted release-only surrogate P(SEX | the twin\'s released log-probabilities) as a soft target'}
    fineness = {'SBC': _fineness_inputs(br, cand_sp), 'SPEND': _fineness_inputs(sr['SPEND']['bracket'], cand_sp),
                'RO': _fineness_inputs(sr['RO']['bracket'], cand_sp)}
    # 7. descriptive MODEL_FIT-only checks: cross-fitted readers and the data-rich readers (never a stop)
    t1 = time.process_time()
    check = FK.mf_reader_check(oof_views, sex, unit, f'IPR_mfcheck_IA_S{int(version)}')
    rich_info = None
    if save_rich:
        from .. import richreaders as RR
        rich, rich_cpu = RR.train(oof_views, sex, version)
        rich_info = {'sha256': RR.save(FAMILY, version, rich, {'configs': sorted(oof_views),
                                                               'trained_on': 'MODEL_FIT out-of-fold releases, MODEL_FIT SEX'}),
                     'cpu_s': rich_cpu}
    out.update({'saved': saved, 'sbc_tries': tries, 'sbc_construction': chosen['label'], 'pseudo_info': pseudo_info, 'blind': blind,
                'spend_path_bracket': sr['SPEND']['bracket'], 'ro_path_bracket': sr['RO']['bracket'], 'ro_record': sr['RO']['record'],
                'shared_points': shared_ids, 'shared_status': shared_status, 'substituted_ladder': substituted,
                'fineness_inputs': fineness, 'cand_spend': cand_sp, 'rich_readers': rich_info,
                'mf_reader_check': {'note': 'descriptive MODEL_FIT out-of-fold readers (household 5-fold cross-fitted); never a decision '
                                            'input and never a stop; the decisions use the MODEL_SELECT grid',
                                    'auc': check, 'cpu_s': time.process_time() - t1},
                'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'ia') / f'FIT_V{int(version)}.json', out)
    return out


def selftest_stage(n=2500, seed=5):
    """End-to-end check of stage_fit on INVENTED labels attached to admitted MODEL_FIT features (label-free; no label file is read):
    the cached teachers are replaced by teachers fitted on the invented labels, model files go to a temporary directory, nothing is
    saved to the campaign's runs and no record is written."""
    import tempfile
    from pathlib import Path as _P
    from ..ctl import twin as TW
    prc = _prc()
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
    old = (C.RUNS, TW.load_task_cache, TW.load_sex_cache, prc.HGB)
    tmp = _P(tempfile.mkdtemp(prefix='ipr_ia_selftest_'))
    try:
        C.RUNS = tmp
        TW.load_task_cache = lambda v, rs=None: (T, 'invented')
        TW.load_sex_cache = lambda v, rs=None: (S, 'invented')
        prc.HGB = hp
        roles = {'MODEL_FIT': {'X': data['X'], 'feature_names': data['feature_names']}}
        out = stage_fit(0, data, 'invented-card', fake_save, roles=roles, log=lambda s: None, record_dir=False, save_rich=False,
                        budget={k: 100.0 for k in BUDGET}, grid=(1, 2, 3))
    finally:
        C.RUNS, TW.load_task_cache, TW.load_sex_cache, prc.HGB = old
    ok = {'cand_saved': 'IA_SFG' in saved, 'k_star_on_grid': out['selection']['k_star'] in (2, 3),
          'controls_saved': any(c.startswith('IA_SBC') for c in saved) and any(c.startswith('CTL_DQ') for c in saved)
          and any(c.startswith('CTL_RO') for c in saved),
          'ablation_saved': 'IA_BLIND' in saved,
          'records_valid': all(r['kind'] == (CONFIGS[c]['kind'] if c in CONFIGS else 'control') for c, (_, r) in saved.items())}
    a = saved['IA_SFG'][0]['MODEL_FIT']
    ok['cand_valid_vectors'] = bool(np.allclose(a['income'].sum(1), 1) and np.allclose(a['occupation'].sum(1), 1))
    st = out['selection']['steps'][str(out['selection']['k_star'])]
    m = field_maps(Z)
    Zh = generalize(Z, st, m)
    _, inv = np.unique(Zh, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    first = np.unique(inv, return_index=True)[1]
    ok['release_is_function_of_h'] = bool(np.allclose(a['occupation'], a['occupation'][first][inv]))
    ok['all_ok'] = all(ok.values())
    ok['saved'] = sorted(saved)
    ok['k_star'] = out['selection']['k_star']
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ipr.ia.sfg')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('selftest_maps')
    sub.add_parser('selftest_stage')
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    if a.cmd in ('selftest', 'selftest_maps', 'selftest_stage'):
        r = {'selftest': selftest, 'selftest_maps': selftest_maps, 'selftest_stage': selftest_stage}[a.cmd]()
        print(json.dumps(r, default=str))
        return 0 if r['all_ok'] else 1
    from .. import release as RL
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = stage_fit(a.version, data, a.card, RL.save)
    print(json.dumps({'k_star': out['selection']['k_star'], 'saved': sorted(out['saved']), 'sbc_target': out.get('sbc_target'),
                      'cpu_s': round(out['cpu_s'], 1)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""FS_CPI: the campaign's replacement family. A conditional pair-increment residual on SUCC's income, with the occupation frozen
(card FS_CPI_v1, draft by investigator B). PCRL campaign `pcrl_strong_anchor_release_v1`; EXPLORATORY DEVELOPMENT on previously used
public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation. ADAPTIVE as a whole: designed after the
FA / FB MODEL_SELECT readings.

START = SUCC_S<v> (H3_STRONG_LJDIR_PB income + JOINT_TREE E20 occupation; ihs.fam_h.predict, hash-pinned frozen files). On the SAR1
grid its linked pair (0.6166) sits 0.040 above its occupation view (0.5768): the H3 income vector, never protected jointly with JOINT
occupation, carries SEX information CONDITIONAL on the released occupation vector.

Operation (sequential, occupation first):
  occupation  SUCC's JOINT E20 occupation vector, frozen bitwise (its clauses and guards, the binding men's guard included, are SUCC's)
  income      p = softmax(-(d_S + g)/2, (d_S + g)/2): d_S = SUCC's H3 income log-odds; g = g_theta(phi(x)) a 64-64 ReLU residual on
              phi = [standardized compact fields, SUCC's logits (d_S, z_J), the admitted HGB teachers' logits (d_H, z_H)] (sar.fa.tfr
              feature map, unchanged); zero-initialized output layer (theta = 0 is SUCC)
  training    MODEL_FIT, all 20,000 people, SEX only inside P; no task label is read by the fit:
              KL(sigma(d_S) || sigma(d_S + g)) (anchor to START's own income; zero gradient at g = 0, so lambda = 0 IS START)
              + lambda * P_FS,  P_FS = [Phi(D_pair / sqrt 2) - 1/2] + relu([Phi(D_inc / sqrt 2) - 1/2] - [Phi(D_inc,S / sqrt 2) - 1/2])
              D_v^2 = FB_PFD_v1's closed-form whitened quadratic best response on the privacy batch (sar.fb.pfd.fisher_d2 / poly2);
              D_pair on (trainable income log-odds, the frozen occupation log-ratios as released); the income-view term is a hinge
              at START's in-batch value (it only acts if the income view would become MORE separable than START's)
              AdamW lr 1e-3, weight decay 0.01, clip 5; 400 steps (anchor batch 256, privacy batch 2,048)
  strength    lambda on LAMBDA_GRID (nats of anchor KL per unit of AUC proxy; no gradient calibration: at g = 0 the anchor has no
              gradient); lambda* = the largest grid value such that it and every smaller one keep, on CALIBRATION, estimate + 2
              household-clustered SE <= BUDGET_CAL on the three income components of the paired spend over START_CF (H3 with its maps
              household-cross-fitted on CALIBRATION, H3's own budget estimator); the occupation components are exactly 0
Controls (origin START; selection data CALIBRATION; spends over START_CF):
  CTRL  FS_CPI_DQ_HI: SEX-blind subtractive-dithered quantization of SUCC's income log-odds (sar.ctlkit, occupation passed through
        bitwise), the single point with the smallest spend >= max(CAND's + 2 SE, CTRL_HEDGE x CAND's) (hedge for MODEL_SELECT
        transfer, prospective)
  ALT   FS_CPI_BETA_LO / _HI: SUCC's own protection operation continued: H3's LJDIR transplant a (s_B + D_L + beta D_J) + b with beta
        beyond SUCC's (0.25 / 0.20) on H3's own grid and the 2-parameter Platt map refitted on CALIBRATION for each beta (exactly
        ihs.fam_h's recipe); LO <= CAND's spend, HI >= CAND's + 2 SE (sar.fb.pfd.bracket)
  CONT  lambda = 0 is START bitwise (no specification)
Configurations: at most 8 new (CAND 2 + CTRL 2 + ALT 4).

Commands (real data only through the campaign queue after the committed card):
    <repository>/.venv/bin/python -m sar.fs.cpi selftest | selftest_stage
    <repository>/.venv/bin/python -m sar.fs.cpi fit --version V --card results/pcrl_strong_anchor_release_v1/cards/FS_CPI_v1.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from .. import common as C
from .. import famkit as FK

FAMILY = 'FS'                      # release family (card prefix FS_); the reading's family name is READING
READING = 'FS_CPI'
SUCC_CONFIG = 'H3_STRONG_LJDIR_PB'
LAMBDA_GRID = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
STEPS = 400
CTRL_HEDGE = 1.5
# CALIBRATION income components, estimate + 2 SE <= limit: about 0.5 of SUCC's smallest recorded MODEL_SELECT margin to the cap per
# component over both versions and both references (income NLL 0.0136, Brier 0.0073, accuracy 0.0083; anchors/ANCHOR_TABLE.json),
# an adaptive, disclosed choice
BUDGET_CAL = {'inc_nll': 0.0065, 'inc_brier': 0.0035, 'inc_acc': 0.0040}
RECIPE = {'schema': 'SAR1-FS-CPI-recipe-v1', 'hidden': (64, 64), 'lr': 1e-3, 'weight_decay': 0.01, 'clip': 5.0, 'anchor_batch': 256,
          'privacy_batch': 2048, 'steps': STEPS, 'lambda_grid': list(LAMBDA_GRID), 'ridge': 1e-3,
          'anchor': 'KL(sigma(d_S) || sigma(d_S + g)) on MODEL_FIT (no label)', 'ctrl_hedge': CTRL_HEDGE,
          'privacy': 'Phi-AUC of the whitened quadratic pair discriminant + hinge on the income view at START (sar.fb.pfd functions)'}
CONFIGS = {
    'FS_CPI': {'kind': 'candidate', 'path': 'CAND',
               'what': "SUCC's occupation frozen; income = SUCC's H3 log-odds + a zero-initialized residual trained against the closed-form "
                       "whitened pair discriminant (and an income-view hinge at START), anchored to START's own income by KL"},
    'FS_CPI_DQ_HI': {'kind': 'control', 'path': 'CTRL',
                     'what': "CTRL: SEX-blind subtractive-dithered quantization of SUCC's income (occupation passed through bitwise) at the "
                             "smallest CALIBRATION spend over START >= max(CAND's + 2 SE, 1.5 x CAND's)"},
    'FS_CPI_BETA_LO': {'kind': 'control', 'path': 'ALT',
                       'what': "ALT lower point: SUCC's own LJDIR transplant continued (beta beyond SUCC's, Platt refitted on CALIBRATION) at "
                               "the largest CALIBRATION spend over START <= CAND's"},
    'FS_CPI_BETA_HI': {'kind': 'control', 'path': 'ALT',
                       'what': "ALT upper point: SUCC's own LJDIR transplant continued at the smallest CALIBRATION spend over START >= "
                               "CAND's + 2 SE"},
}
PURPOSE = ('FS_CPI: train the income residual on SUCC with the closed-form pair objective (MODEL_FIT SEX as a training label of the '
           'protection objective only; no task label)')
PURPOSE_CAL = ('FS_CPI: CALIBRATION income for the CAND strength rule, the CTRL / ALT brackets (paired spends over SUCC with its maps '
               'cross-fitted on CALIBRATION) and the ALT Platt refits (H3 recipe); no SEX')


# ---------------------------------------------------------------- frozen pieces (label-free)
def _tf():
    from ..fa import tfr as TF
    return TF


def succ_arrays(version, X, names):
    """SUCC_S<v> probability vectors from admitted features (ihs.fam_h.predict via sar.fa.tfr._succ)."""
    s = _tf()._succ(int(version), X, names)
    return np.asarray(s['income'], np.float64), np.asarray(s['occupation'], np.float64)


def h3_state(version):
    from ihs import fam_h
    return json.loads(fam_h.state_path(int(version), SUCC_CONFIG).read_text())


def h3_components(version, X, names):
    from ihs import fam_h
    hgb, s1 = fam_h.load_step1(int(version))
    return fam_h.components(int(version), X, tuple(map(str, names)), hgb, s1['cv']['selected'])


def h3_logodds(comp, cfg, beta, wb, wp):
    from ihs import fam_h
    return fam_h.release_logodds(fam_h.base_design(comp, cfg['base']), fam_h.offset(comp, cfg, beta), np.asarray(wb), np.asarray(wp))


def pins(version):
    TF = _tf()
    out = dict(TF.frozen_sha(version))
    out['fa_cache/TASK.pkl'] = TF.task_cache(version)[1]
    return out


def inc_probs(d):
    from acs_replication import trees
    return trees.probabilities('income', np.asarray(d, np.float64).reshape(-1))


# ---------------------------------------------------------------- the residual (torch float64 for training, numpy for deployment)
def _net(d, seed):
    import torch
    torch.manual_seed(int(seed) % 2 ** 31)
    h1, h2 = RECIPE['hidden']
    m = torch.nn.Sequential(torch.nn.Linear(d, h1, dtype=torch.float64), torch.nn.ReLU(), torch.nn.Linear(h1, h2, dtype=torch.float64),
                            torch.nn.ReLU(), torch.nn.Linear(h2, 1, dtype=torch.float64))
    with torch.no_grad():
        m[4].weight.zero_(); m[4].bias.zero_()                 # theta = 0 is START
    return m


def net_state(m):
    return [p.detach().cpu().numpy().copy() for p in m.parameters()]


def net_numpy(W, F):
    h = np.maximum(np.asarray(F, np.float64) @ W[0].T + W[1], 0.0)
    h = np.maximum(h @ W[2].T + W[3], 0.0)
    return (h @ W[4].T + W[5]).reshape(-1)


def privacy_terms_inc(d, lo_const, sex, d_start):
    """P_FS terms: the pair (trainable income log-odds d, constant occupation log-ratios) and the income-view hinge at START."""
    import torch
    from ..fb import pfd as PD
    li = d.reshape(-1, 1)
    d2_pair = PD.fisher_d2(PD.poly2(torch.cat([li, lo_const], 1)), sex)
    d2_inc = PD.fisher_d2(PD.poly2(li), sex)
    with torch.no_grad():
        d2_inc0 = PD.fisher_d2(PD.poly2(d_start.reshape(-1, 1)), sex)
    pair = PD.auc_proxy(d2_pair)
    hinge = torch.relu(PD.auc_proxy(d2_inc) - PD.auc_proxy(d2_inc0))
    return pair + hinge, {'pair': float(pair.detach()), 'hinge': float(hinge.detach()), 'd2_pair': float(d2_pair.detach()),
                          'd2_inc': float(d2_inc.detach()), 'd2_inc_start': float(d2_inc0)}


def train(F, d_S, z_occ, sex, version, lam, steps=STEPS, log_every=40):
    """One residual on MODEL_FIT (all rows given): KL anchor to START + lam * P_FS. lam = 0 leaves the residual's output exactly 0."""
    import torch
    from acs_replication import leafcorr as lc
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    Ft = torch.from_numpy(np.ascontiguousarray(F, np.float64))
    dS = torch.from_numpy(np.ascontiguousarray(d_S, np.float64))
    z = np.asarray(z_occ, np.float64)
    lo = torch.from_numpy(np.ascontiguousarray(z[:, 1:] - z[:, :1]))
    sx = None if sex is None else torch.from_numpy(np.asarray(sex, np.int64))
    n = len(d_S)
    m = _net(F.shape[1], lc.seed_for('SAR1fs-net', int(version)))
    opt = torch.optim.AdamW(m.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    task_rng = np.random.default_rng(lc.seed_for('SAR1fs-anchor', int(version)) % 2 ** 32)
    priv_rng = random.Random(lc.seed_for('SAR1fs-privacy', int(version)))
    pS = torch.sigmoid(dS)
    logs = []
    for step in range(steps):
        rows = torch.from_numpy(task_rng.choice(n, RECIPE['anchor_batch'], replace=False))
        g = m(Ft[rows]).reshape(-1)
        q = torch.sigmoid(dS[rows] + g).clamp(1e-12, 1 - 1e-12)
        p = pS[rows].clamp(1e-12, 1 - 1e-12)
        loss = (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).mean()
        info = {}
        if lam > 0:
            pr = torch.from_numpy(np.asarray(priv_rng.sample(range(n), min(RECIPE['privacy_batch'], n)), np.int64))
            gp = m(Ft[pr]).reshape(-1)
            P, info = privacy_terms_inc(dS[pr] + gp, lo[pr], sx[pr], dS[pr])
            loss = loss + float(lam) * P
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), RECIPE['clip'])
        opt.step()
        if step % log_every == 0:
            logs.append({'step': step, 'loss': float(loss.detach()), **info})
    return {'W': net_state(m), 'lambda': float(lam), 'steps': steps, 'logs': logs}


# ---------------------------------------------------------------- deployment (numpy only; python3)
def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function of the FS_CPI configurations (sar.deploy contract via sar.fs); deterministic. The occupation output is
    SUCC's bitwise; SUCC's frozen files must hash to the values recorded at fit time."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    P = FK.load_model(FAMILY, config_id, version)
    X, names = FK.check_X(X, feature_names, P)
    TF = _tf()
    if P.get('frozen_sha256') and TF.frozen_sha(version) != P['frozen_sha256']:
        raise PermissionError("SUCC's frozen files differ from those recorded at fit time")
    pI, pO = succ_arrays(version, X, names)
    op = P['op']
    if op == 'res':
        from nrm.fa import prc
        Zc = FK.compact(X, names)
        A = TF.blocks(Zc, (pI, pO), (prc._proba(P['hgb_inc'], Zc), prc._proba(P['hgb_occ'], Zc)))
        g = net_numpy(P['net'], TF.phi_apply(P['phi'], A))
        rI = inc_probs(A[:, 0] + g)
    elif op == 'dq_inc':
        from .. import ctlkit as CK
        rI = CK.dq_release(P['dq_basis'], pI, pO, FK.compact(X, names), float(P['delta']), P['dq_tag'])[0]
    elif op == 'alt':
        rI = inc_probs(h3_logodds(h3_components(version, X, names), P['h3_config'], float(P['beta']), P['stack_w'], P['platt_w']))
    else:
        raise ValueError(f'unknown FS_CPI payload op {op!r}')
    return {'income': rI, 'occupation': pO}


def model_paths(config_id, version):
    """The configuration's own model files; predict() also reads SUCC's frozen files (sar.fa.tfr.frozen_inputs, hash-checked)."""
    return FK.model_paths(FAMILY, config_id, version)


# ---------------------------------------------------------------- bracket rules (CALIBRATION spends over START_CF)
def ctrl_point(path_L, cand_L, start_L, unit, grid, hedge=CTRL_HEDGE):
    """The single CTRL point: the grid point with the smallest spend over START >= max(CAND's + 2 SE, hedge x CAND's), SE = the
    household-clustered SE of that point's paired difference against CAND; none reaching it: the largest-spend point, flag
    CONTROL_PATH_MAY_NOT_BRACKET; CAND's spend <= 0: CTRL_BRACKET_IS_ORIGIN (nothing saved)."""
    from ..fb import pfd as PD
    b = PD.bracket(path_L, cand_L, start_L, unit, grid)
    c = b['cand_spend_over_start']['value']
    out = {**b, 'lo': None, 'hi': None, 'flags': [f for f in b['flags'] if f == 'CTRL_BRACKET_IS_ORIGIN'], 'hedge': hedge,
           'rule': 'smallest spend >= max(CAND + 2 SE, hedge x CAND)'}
    if c <= 0:
        return out
    rows = {g: b['grid'][str(g)] for g in grid}
    ok = [g for g in grid if rows[g]['spend_over_start'] >= max(c + 2.0 * rows[g]['se_diff_vs_cand'], hedge * c)]
    if ok:
        out['hi'] = min(ok, key=lambda g: rows[g]['spend_over_start'])
    else:
        out['hi'] = max(grid, key=lambda g: rows[g]['spend_over_start'])
        out['flags'].append('CONTROL_PATH_MAY_NOT_BRACKET')
    return out


# ---------------------------------------------------------------- the fit stage (one unit per version)
def stage_fit(version, data, cal, card, save, roles=None, log=print, record_dir=None, lambda_grid=LAMBDA_GRID, steps=STEPS,
              budget=None, cache=None, reader_check=True):
    """data: MODEL_FIT (features, sex); cal: CALIBRATION (features, income). No other label is read."""
    from ihs import common as IC
    from ihs import fam_h
    from nrm.fa import prc
    from .. import ctlkit as CK
    from ..fb import pfd as PD
    TF = _tf()
    t0 = time.process_time()
    v = int(version)
    budget = budget or BUDGET_CAL
    names = tuple(map(str, data['feature_names']))
    X = np.asarray(data['X'], np.float64)
    unit, sex = np.asarray(data['unit']), np.asarray(data['sex'], np.int64)
    T, tsha = cache if cache is not None else TF.task_cache(v)
    if T.get('row_id_sha256') is not None and 'row_id' in data and T['row_id_sha256'] != prc.array_sha(np.asarray(data['row_id'])):
        raise PermissionError('the cached task teachers were fitted on other rows')
    pin = {**TF.frozen_sha(v), 'fa_cache/TASK.pkl': tsha}
    st = h3_state(v)
    cfg, beta0 = st['config'], float(st['beta'])
    # 1. MODEL_FIT arrays (SUCC in-sample there; the HGB teachers' household out-of-fold values) and the feature map
    sI, sO = succ_arrays(v, X, names)
    Zc = FK.compact(X, names)
    A = TF.blocks(Zc, (sI, sO), (T['oof']['inc'], T['oof']['occ']))
    phi = TF.phi_fit(A)
    F = TF.phi_apply(phi, A)
    # 2. one residual per lambda on all MODEL_FIT people (CALIBRATION is never seen by the residual)
    fits = {}
    for lam in lambda_grid:
        fits[lam] = train(F, A[:, 0], A[:, 1:6], sex, v, lam, steps)
        log(json.dumps({'lambda': lam, 'last_log': fits[lam]['logs'][-1] if fits[lam]['logs'] else None,
                        'cpu_s': round(time.process_time() - t0, 1)}))
    # 3. CALIBRATION: START_CF = H3 with its maps household-cross-fitted on CALIBRATION (H3's own budget estimator)
    cn = tuple(map(str, cal['feature_names']))
    Xc = np.asarray(cal['X'], np.float64)
    yc, uc = np.asarray(cal['income'], np.int64), np.asarray(cal['unit'])
    cc = h3_components(v, Xc, cn)
    Zb = fam_h.base_design(cc, cfg['base'])
    folds = IC.household_folds(uc, fam_h.N_FOLDS, f'IHS_H_budget_S{v}')
    d_cf = fam_h.crossfit(Zb, fam_h.offset(cc, cfg, beta0), yc, folds)
    cI, cO = succ_arrays(v, Xc, cn)
    Zcc = FK.compact(Xc, cn)
    p_cf = inc_probs(d_cf)
    hgb_c = (prc._proba(T['models']['inc'], Zcc), prc._proba(T['models']['occ'], Zcc))
    A_cf = TF.blocks(Zcc, (p_cf, cO), hgb_c)
    F_cf = TF.phi_apply(phi, A_cf)
    yO_dummy = np.zeros(len(yc), np.int64)                      # occupation is identical in every release: its losses cancel
    Lc = lambda pi: FK.person_losses(pi, cO, yc, yO_dummy)
    L_start = Lc(p_cf)
    cand_L = {lam: Lc(inc_probs(d_cf + net_numpy(fits[lam]['W'], F_cf))) for lam in lambda_grid}
    sp = {lam: FK.spend(cand_L[lam], L_start, uc) for lam in lambda_grid}
    lam_star = FK.select_strength(sp, tuple(lambda_grid), budget)
    rec_check = st.get('calibration_cost_vs_base', {})
    try:                                                         # START_CF is H3's own cross-fitted CALIBRATION release
        h3_nll = float(rec_check['base_reference']['nll']) + float(rec_check['nll'])
    except (KeyError, TypeError):
        h3_nll = None
    sel = {'rule': 'lambda* = the largest LAMBDA_GRID value such that it and every smaller value keep estimate + 2 household-clustered '
                   'SE <= BUDGET_CAL on the CALIBRATION paired income spend over START_CF (H3 with household-cross-fitted CALIBRATION '
                   'maps, folds IHS_H_budget_S<v>); occupation spends are exactly 0',
           'budget': budget, 'lambda_star': lam_star, 'per_lambda': {str(l): FK.jsonable_spend(sp[l]) for l in lambda_grid},
           'selection_data': 'CALIBRATION',
           'exposure_note': "the residual and the HGB teachers are out of sample on CALIBRATION; START_CF's maps are cross-fitted there "
                            "(H3's stack, beta and Platt map were fitted on all CALIBRATION: beta was chosen by H3's own CALIBRATION rule)",
           'start_cf_mean_nll': float(L_start['inc_nll'].mean()), 'h3_recorded_calibration_cost_vs_base': rec_check,
           'start_cf_mean_nll_minus_h3_record': None if h3_nll is None else float(L_start['inc_nll'].mean()) - h3_nll}
    cand = {'config': 'FS_CPI', 'strength': lam_star, 'status': 'NO_FEASIBLE_STRENGTH' if lam_star is None else 'FITTED',
            'spend_over_start': None}
    out = {'version': v, 'reading': READING, 'label_prefix': 'ADAPTIVE_', 'recipe': RECIPE, 'pins_sha256': pin, 'h3_beta': beta0,
           'selection': sel, 'cand': cand, 'cont': {'declared': False, 'reason': 'lambda = 0 leaves the residual output exactly 0: '
                                                                                  'the zero-protection continuation is START bitwise'},
           'paths': {}, 'train_logs': {str(l): fits[l]['logs'] for l in lambda_grid}}
    if lam_star is None:
        out.update({'stop': 'NO_FEASIBLE_STRENGTH', 'saved': {}, 'cpu_s': time.process_time() - t0})
        if record_dir is not False:
            C.write_json((record_dir or C.RESULTS / 'fs') / f'FIT_CPI_V{v}.json', out)
        return out
    L_cand = cand_L[lam_star]
    dc = L_cand['tot_nll'] - L_start['tot_nll']
    cand['spend_over_start'] = {'value': float(dc.mean()), 'se': FK.cluster_se(dc, uc), 'data': 'CALIBRATION', 'start': 'START_CF'}
    # 4. CTRL: DQ of SUCC's income (basis on SUCC's MODEL_FIT outputs, label-free); ALT: H3's beta continued (cross-fitted maps)
    basis = CK.dq_fit(sI, sO)
    tag = f'SAR_DQ_FS_S{v}'
    dq_grid = tuple(CK.DQ_GRID_FINE)
    dq_L = {g: Lc(CK.dq_release(basis, p_cf, cO, Zcc, g, tag)[0]) for g in dq_grid}
    alt_grid = tuple(b for b in fam_h.BETA_GRID if b > beta0 + 1e-9)
    alt_d = {b: fam_h.crossfit(Zb, fam_h.offset(cc, cfg, b), yc, folds) for b in alt_grid}
    alt_L = {b: Lc(inc_probs(alt_d[b])) for b in alt_grid}
    br = {'CTRL': ctrl_point(dq_L, L_cand, L_start, uc, dq_grid), 'ALT': PD.bracket(alt_L, L_cand, L_start, uc, alt_grid)}
    ids = {('CTRL', 'hi'): 'FS_CPI_DQ_HI', ('ALT', 'lo'): 'FS_CPI_BETA_LO', ('ALT', 'hi'): 'FS_CPI_BETA_HI'}
    for key, b in br.items():
        out['paths'][key] = {**b, 'origin': 'START = SUCC (on CALIBRATION: START_CF)', 'selection_data': 'CALIBRATION',
                             'lo_config': ids.get((key, 'lo')) if b['lo'] is not None else None,
                             'hi_config': ids.get((key, 'hi')) if b['hi'] is not None else None,
                             'operation': ("SEX-blind subtractive-dithered quantization of SUCC's income (sar.ctlkit DQ_GRID_FINE, tag "
                                           f"{tag}; occupation passed through bitwise)" if key == 'CTRL' else
                                           f"H3's LJDIR transplant with beta in {alt_grid[0] if alt_grid else '-'}..{alt_grid[-1] if alt_grid else '-'} "
                                           "(ihs.fam_h BETA_GRID), stack and Platt maps refitted on CALIBRATION per beta (H3 recipe)")}
        log(json.dumps({key: {'lo': b['lo'], 'hi': b['hi'], 'flags': b['flags']}}))
    # 5. deployed configurations
    checks = {}
    roles = roles or FK.role_features()
    payload_base = {'feature_names': names, 'frozen_sha256': TF.frozen_sha(v), 'pins_sha256': pin, 'reading': READING}
    plan = [('FS_CPI', {'op': 'res', 'net': fits[lam_star]['W'], 'phi': phi, 'hgb_inc': T['models']['inc'],
                        'hgb_occ': T['models']['occ'], 'lambda': lam_star}, lam_star, L_cand)]
    if br['CTRL']['hi'] is not None:
        g = br['CTRL']['hi']
        plan.append(('FS_CPI_DQ_HI', {'op': 'dq_inc', 'dq_basis': basis, 'delta': float(g), 'dq_tag': tag}, float(g), dq_L[g]))
    for side in ('lo', 'hi'):
        b = br['ALT'][side]
        if b is not None:
            wb, wp = fam_h.fit_maps(Zb, fam_h.offset(cc, cfg, b), yc)           # H3's own all-CALIBRATION maps at this beta
            plan.append((ids[('ALT', side)], {'op': 'alt', 'beta': float(b), 'stack_w': wb.tolist(), 'platt_w': wp.tolist(),
                                              'h3_config': cfg}, float(b), alt_L[b]))
    wb0, wp0 = fam_h.fit_maps(Zb, fam_h.offset(cc, cfg, beta0), yc)
    d0 = h3_logodds(cc, cfg, beta0, wb0, wp0)
    checks['alt_recipe_at_succ_beta_reproduces_succ_income_on_CAL'] = float(np.max(np.abs(inc_probs(d0) - cI)))
    people, hh = len(X), len(np.unique(unit))
    exposure = ('MODEL_FIT SEX of all 20,000 MODEL_FIT people reaches the residual through the pair objective (no task label reaches it); '
                'CALIBRATION income (all 2,982) reaches the strength rule, the CTRL / ALT brackets and the ALT Platt / stack refits; '
                "through SUCC (inherited) MODEL_FIT and CALIBRATION labels reach every release; no MODEL_SELECT label")
    saved, views = {}, {'SUCC': (sI, sO)}
    for cfg_id, pl, strength, Lx in plan:
        meta = FK.write_model(FAMILY, cfg_id, v, {**payload_base, **pl})
        arrays = FK.release_all(predict, cfg_id, v, roles)
        d = Lx['tot_nll'] - L_start['tot_nll']
        key = CONFIGS[cfg_id]['path']
        extra = {'reading': READING, 'label_prefix': 'ADAPTIVE_', 'path': key, 'strength': strength, 'lambda_star': lam_star,
                 'model_sha256': meta['model_sha256'], 'pins_sha256': pin, 'checks': checks, 'selection_data': 'CALIBRATION',
                 'cal_spend_over_start': {'value': float(d.mean()), 'se': FK.cluster_se(d, uc)},
                 'cal_spend_components': FK.jsonable_spend(FK.spend(Lx, L_start, uc)),
                 'bracket_flags': br[key]['flags'] if key in br else None}
        nfits = (1, 0, 0, 0) if pl['op'] == 'res' else ((0, 0, 1, 0) if pl['op'] == 'alt' else (0, 0, 0, 0))
        rec = FK.record(card, CONFIGS[cfg_id]['kind'], CONFIGS[cfg_id]['what'], nfits,
                        'MODEL_FIT: sex; CALIBRATION: income', exposure, people, hh, time.process_time() - t0, extra)
        save(FAMILY, cfg_id, v, arrays, rec)
        saved[cfg_id] = extra
        if 'MODEL_FIT' in arrays:
            views[cfg_id] = (arrays['MODEL_FIT']['income'], arrays['MODEL_FIT']['occupation'])
            checks.setdefault('occupation_is_succ_bitwise', True)
            checks['occupation_is_succ_bitwise'] &= bool(np.array_equal(arrays['MODEL_FIT']['occupation'], sO))
    if reader_check:
        t1 = time.process_time()
        out['mf_reader_check'] = {'note': 'descriptive MODEL_FIT readers (SUCC and the residual are in-sample there); never a decision input',
                                  'auc': FK.mf_reader_check(views, sex, unit, f'SAR_mfcheck_FS_CPI_S{v}'), 'cpu_s': time.process_time() - t1}
    out.update({'saved': saved, 'n_saved_new': len(saved), 'checks': checks, 'cpu_s': time.process_time() - t0})
    if record_dir is not False:
        C.write_json((record_dir or C.RESULTS / 'fs') / f'FIT_CPI_V{v}.json', out)
    return out


# ---------------------------------------------------------------- checks (invented data / admitted features only; no label file)
def selftest():
    import tempfile
    from pathlib import Path
    import torch
    from .. import ctlkit as CK
    torch.set_num_threads(1)
    ok, info = {}, {}
    rng = np.random.default_rng(4)
    n = 4000
    # invented world: occupation log-ratios carry SEX; income carries extra SEX information GIVEN occupation (the increment)
    sex = rng.integers(0, 2, n)
    s = sex - 0.5
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.8 * s
    z = z - z.mean(1, keepdims=True)
    d_S = 0.9 * (z[:, 1] - z[:, 0]) + 0.6 * s + rng.normal(0, 1, n)
    F = np.column_stack([rng.normal(size=(n, 6)), d_S, z[:, 1] - z[:, 0], s * 0 + rng.normal(size=n)])
    F = np.column_stack([F, (sex - 0.5) + rng.normal(0, 0.8, n)])          # an input that predicts SEX (as the real fields do)
    r0 = train(F, d_S, z, sex, 0, 0.0, steps=60)
    ok['lambda0_output_exactly_zero'] = bool(np.all(net_numpy(r0['W'], F) == 0.0))
    r1 = train(F, d_S, z, sex, 0, 1.0, steps=300)
    g = net_numpy(r1['W'], F)
    ok['lambda_moves_income'] = bool(np.abs(g).max() > 0)
    tt = lambda a: torch.from_numpy(np.asarray(a, np.float64))
    lo = tt(z[:, 1:] - z[:, :1]); sx = torch.from_numpy(sex)
    _, i0 = privacy_terms_inc(tt(d_S), lo, sx, tt(d_S))
    _, i1 = privacy_terms_inc(tt(d_S + g), lo, sx, tt(d_S))
    info['invented'] = {'start': i0, 'lambda1': i1}
    from ..fb import pfd as PD
    d2_occ = float(PD.fisher_d2(PD.poly2(lo), sx))
    info['invented']['d2_occupation_view'] = d2_occ
    ok['pair_increment_reduced'] = (i1['d2_pair'] - d2_occ) < 0.7 * (i0['d2_pair'] - d2_occ)
    ok['income_view_not_above_start'] = i1['d2_inc'] <= i0['d2_inc'] * 1.05 + 1e-4
    ok['hinge_zero_at_start'] = i0['hinge'] == 0.0
    # numpy deployment of the residual = torch forward
    m = _net(F.shape[1], 1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.from_numpy(rng.normal(0, 0.1, size=tuple(p.shape))))
        gt = m(tt(F)).reshape(-1).numpy()
    ok['numpy_forward_matches_torch'] = float(np.abs(net_numpy(net_state(m), F) - gt).max()) < 1e-12
    # the CTRL point rule (hedged) on invented spends
    ub = np.arange(n) // 2
    base = {'tot_nll': np.ones(n)}
    mk = lambda v: {'tot_nll': np.ones(n) + v}
    grid = (1, 2, 3, 4, 5, 6)
    path = {k: mk(0.001 * k) for k in grid}
    b = ctrl_point(path, mk(0.0019), base, ub, grid)
    ok['ctrl_hedge_rule'] = b['hi'] == 3 and b['flags'] == []                # max(0.0019 + 0, 1.5 x 0.0019 = 0.00285) -> point 3
    b2 = ctrl_point({k: mk(0.0001 * k) for k in grid}, mk(0.002), base, ub, grid)
    ok['ctrl_may_not_bracket'] = b2['hi'] == 6 and 'CONTROL_PATH_MAY_NOT_BRACKET' in b2['flags']
    b3 = ctrl_point(path, mk(-0.001), base, ub, grid)
    ok['ctrl_origin'] = b3['hi'] is None and b3['flags'] == ['CTRL_BRACKET_IS_ORIGIN']
    # deployment paths on admitted features (label-free): occupation is SUCC's bitwise; 'alt' at SUCC's beta and maps is SUCC's income;
    # 'dq_inc' is the shared DQ law on SUCC's income; 'res' with a zero residual is SUCC
    Fe = C.features('MODEL_FIT')
    Xr, nm = Fe['X'][:300], Fe['feature_names']
    sI, sO = succ_arrays(0, np.asarray(Xr, np.float64), nm)
    st = h3_state(0)
    TF = _tf()
    T, _ = TF.task_cache(0)
    Zc = FK.compact(np.asarray(Xr, np.float64), nm)
    from nrm.fa import prc
    A = TF.blocks(Zc, (sI, sO), (prc._proba(T['models']['inc'], Zc), prc._proba(T['models']['occ'], Zc)))
    phi = TF.phi_fit(A)
    W0 = net_state(_net(TF.phi_apply(phi, A).shape[1], 2))
    basis = CK.dq_fit(sI, sO)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='sar_fs_cpi_selftest_'))
    try:
        C.RUNS = tmp
        base_pl = {'feature_names': tuple(nm), 'frozen_sha256': TF.frozen_sha(0)}
        FK.write_model(FAMILY, 'FS_CPI', 0, {**base_pl, 'op': 'res', 'net': W0, 'phi': phi, 'hgb_inc': T['models']['inc'],
                                              'hgb_occ': T['models']['occ']})
        FK.write_model(FAMILY, 'FS_CPI_DQ_HI', 0, {**base_pl, 'op': 'dq_inc', 'dq_basis': basis, 'delta': 0.5, 'dq_tag': 'TEST'})
        FK.write_model(FAMILY, 'FS_CPI_BETA_LO', 0, {**base_pl, 'op': 'alt', 'beta': float(st['beta']), 'stack_w': st['stack_w'],
                                                      'platt_w': st['platt_w'], 'h3_config': st['config']})
        from . import predict as pkg_predict
        res = pkg_predict('FS_CPI', 0, Xr, nm)
        dq = pkg_predict('FS_CPI_DQ_HI', 0, Xr, nm)
        alt = pkg_predict('FS_CPI_BETA_LO', 0, Xr, nm)
    finally:
        C.RUNS = old
    ok['occupation_is_succ_bitwise'] = all(np.array_equal(r['occupation'], sO) for r in (res, dq, alt))
    ok['res_zero_residual_is_succ_income'] = float(np.abs(res['income'] - sI).max()) < 1e-12
    ok['alt_at_succ_beta_is_succ_income'] = float(np.abs(alt['income'] - sI).max()) < 1e-12
    want = CK.dq_release(basis, sI, sO, Zc, 0.5, 'TEST')[0]
    ok['dq_inc_is_shared_law_bitwise'] = bool(np.array_equal(dq['income'], want))
    ok['all_ok'] = all(ok.values())
    return {**ok, 'info': info}


def selftest_stage(n_cal=1500, seed=6, lambda_grid=(0.05, 0.2, 0.8), steps=120):
    """stage_fit end to end with INVENTED labels on admitted features: MODEL_FIT SEX invented from SUCC's own outputs (so that the
    income carries SEX information given occupation), CALIBRATION income drawn from SUCC's own probabilities on a CALIBRATION subset;
    SUCC's frozen files, the HGB cache and H3's state are the admitted ones (label-free); models go to a temporary directory and nothing
    is saved to the campaign's runs."""
    import tempfile
    from pathlib import Path
    rng = np.random.default_rng(seed)
    Fm = C.features('MODEL_FIT')
    names = Fm['feature_names']
    sI, sO = succ_arrays(0, np.asarray(Fm['X'], np.float64), names)
    TF = _tf()
    dS, zO = TF.logodds(sI), TF.clogits(sO)
    eta = 1.2 * (dS - dS.mean()) / dS.std() - 0.8 * (zO[:, 3] - zO[:, 1])
    sex = (rng.random(len(dS)) < 1 / (1 + np.exp(-eta))).astype(np.int64)
    data = {'X': Fm['X'], 'feature_names': names, 'row_id': Fm['row_id'], 'unit': Fm['unit'], 'sex': sex}
    Fc = C.features('CALIBRATION')
    idx = np.sort(rng.choice(len(Fc['row_id']), n_cal, replace=False))
    cI, _ = succ_arrays(0, np.asarray(Fc['X'][idx], np.float64), names)
    cal = {'X': Fc['X'][idx], 'feature_names': names, 'unit': Fc['unit'][idx],
           'income': (rng.random(n_cal) < cI[:, 1]).astype(np.int64)}
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        saved[cfg] = (arrays, rec)
    old = C.RUNS
    tmp = Path(tempfile.mkdtemp(prefix='sar_fs_cpi_stage_'))
    try:
        C.RUNS = tmp
        roles = {'CALIBRATION': {'X': cal['X'], 'feature_names': names}}
        out = stage_fit(0, data, cal, 'invented-card', fake_save, roles=roles, log=lambda s_: None, record_dir=False,
                        lambda_grid=lambda_grid, steps=steps, budget={k: 100.0 for k in BUDGET_CAL}, reader_check=False)
        re = {c: predict(c, 0, cal['X'], names) for c in saved}
    finally:
        C.RUNS = old
    _, cO = succ_arrays(0, np.asarray(cal['X'], np.float64), names)
    c_sp = out['cand']['spend_over_start']
    ok = {'cand_saved_at_last_lambda': 'FS_CPI' in saved and out['cand']['strength'] == lambda_grid[-1],
          'at_most_four_new': len(saved) <= 4,
          'occupation_is_succ_bitwise_everywhere': all(np.array_equal(a['CALIBRATION']['occupation'], cO) for a, _ in saved.values()),
          'release_is_predict_bitwise': all(np.array_equal(saved[c][0]['CALIBRATION'][t], re[c][t]) for c in saved
                                            for t in ('income', 'occupation')),
          'fit_record_fields': all(set(out['paths'][k]) >= {'origin', 'selection_data', 'grid', 'lo', 'hi', 'flags', 'hi_config'}
                                   for k in ('CTRL', 'ALT')) and out['cont']['declared'] is False,
          'records_valid': all(r['kind'] == CONFIGS[c]['kind'] and r['path'] == CONFIGS[c]['path'] for c, (_, r) in saved.items()),
          'bracket_branch_exercised': c_sp is not None and c_sp['value'] > 0 and any(c != 'FS_CPI' for c in saved)}
    ok['all_ok'] = all(ok.values())
    # with INVENTED CALIBRATION labels on a subset the refitted H3 maps differ from H3's; the reproduction is a real-data fit check
    ok['alt_recipe_max_abs_diff_with_invented_labels'] = out['checks']['alt_recipe_at_succ_beta_reproduces_succ_income_on_CAL']
    ok['saved'] = sorted(saved)
    ok['cand_spend_over_start'] = c_sp
    ok['brackets'] = {k: {'lo': out['paths'][k]['lo'], 'hi': out['paths'][k]['hi'], 'flags': out['paths'][k]['flags']} for k in ('CTRL', 'ALT')}
    ok['per_lambda_inc_nll'] = {k: v['inc_nll'] for k, v in out['selection']['per_lambda'].items()}
    ok['cpu_s'] = round(out['cpu_s'], 1)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sar.fs.cpi')
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
    data = C.load('MODEL_FIT', ('features', 'sex'), 'fit', PURPOSE, card=a.card)
    cal = C.load('CALIBRATION', ('features', 'income'), 'calibrate', PURPOSE_CAL, card=a.card)
    out = stage_fit(a.version, data, cal, a.card, RL.save)
    print(json.dumps({'lambda_star': out['selection']['lambda_star'], 'status': out['cand']['status'],
                      'cand_spend_over_start': out['cand']['spend_over_start'],
                      'brackets': {k: {'lo': v['lo'], 'hi': v['hi'], 'flags': v['flags']} for k, v in out['paths'].items()},
                      'saved': sorted(out['saved']), 'checks': out.get('checks'), 'cpu_s': round(out['cpu_s'], 1)}, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())

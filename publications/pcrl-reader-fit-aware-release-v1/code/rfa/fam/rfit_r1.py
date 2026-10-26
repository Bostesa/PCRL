"""RFA1 revision R1 reader family: 'softbin', a bounded, differentiable TABLE-LIKE reader added to v1's SIMPLE and RFF families.

PCRL campaign `pcrl_reader_fit_aware_release_v1`; mechanism investigator, 2026-10-25 (revision R1, designed label-free after the
coordinator's descriptive v1 MODEL_SELECT summary: the readers that win on RF_FULL are categorical-interaction / tree / MLP readers on
probabilities and decisions; the v1 inner families are linear and smooth). This module imports `rfa.fam.rfit` and reuses it
UNCHANGED: the Newton solve, the implicit-function fit path, the guards, the binormal surrogate, the ALTCRIT critic update and the
monitor are rfit's; only the per-family feature map (and a per-family ridge) is added. With families ['simple', 'rff'] the R1
objective reproduces rfit.EpisodeObjective exactly (selftest check `r1_reproduces_v1_families`).

softbin features of view v (built from the same detached-moment, soft-bounded standardized coordinates s as v1, and from the centred
logits for the decisions):
  (a) soft one-hot bins of every coordinate of s: B = 8 Gaussian kernels at the standard-normal quantile centres
      c_b = Phi^-1((b - 1/2) / B), common width h = 0.3, normalized per coordinate (memberships sum to 1): m_b(s) = softmax_b(-(s - c_b)^2 / (2 h^2))
  (b) soft decisions d_r = softmax(z_r / (tau * scale_r)), tau = 0.25, scale_r = the DETACHED A-row RMS SD of recipient r's logit
      differences (so a global temperature move of the release cannot blur the soft decision across steps; the wire's argmax decision
      is the tau -> 0 limit); income view: d_inc (2); occupation view: d_occ (5)
  (c) pair view: the 10 products d_inc (x) d_occ (a soft CATI-like decision interaction) instead of the two marginals (they are its sums)
  plus an intercept. Widths: income 8 + 2 + 1 = 11; occupation 32 + 5 + 1 = 38; pair 40 + 10 + 1 = 51.
  Ridge 1e-3 (mean-loss form; intercept unpenalized): memberships are [0, 1] features with variance ~ 1/B, so a ridge 10x smaller than
  v1's 1e-2 on unit-variance features gives a comparable per-feature shrinkage (a bin holding 1/8 of the rows: data curvature ~0.03
  vs 1e-3, ~3% shrinkage; v1 linear feature: ~0.25 vs 1e-2, ~4%). All values fixed a priori (no data, no outcome, no grid).

Commands:
    <repository>/.venv/bin/python -m rfa.fam.rfit_r1 selftest [--out PATH]
    <repository>/.venv/bin/python -m rfa.fam.rfit_r1 bench [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

from . import rfit as RF
from .rfit import (MODES, VIEWS, ReaderRefused, binormal_auc, exact_auc, fit_reader, monitor_fisher2,  # noqa: F401 (re-exported)
                   moments_of, standardize, view_inputs)

REVISION = 'R1'
FAMILIES_R1 = ('simple', 'rff', 'softbin')
RECIPE_RF_R1 = copy.deepcopy(RF.RECIPE_RF)
RECIPE_RF_R1.update({
    'schema': 'RFA1-reader-fit-recipe-R1',
    'revision': REVISION,
    'base_recipe': RF.RECIPE_RF['schema'] + ' (every v1 value unchanged except families, family_features and ridge_by_family)',
    'families': list(FAMILIES_R1),
    'family_features': {**RF.RECIPE_RF['family_features'],
                        'softbin': '[soft one-hot bins (B per coordinate of s), soft decision(s), 1]; pair: bins of 5 coordinates + '
                                   'the 10 products d_inc (x) d_occ + 1'},
    'softbin_bins': 8,
    'softbin_centres': 'standard-normal quantiles Phi^-1((b - 1/2) / B), b = 1..B',
    'softbin_width': 0.3,
    'softbin_membership': 'softmax over the B bins of -(s - c_b)^2 / (2 width^2), per coordinate (memberships sum to 1)',
    'softbin_tau': 0.25,
    'softbin_decision': 'd_r = softmax(z_r / (tau * scale_r)); scale_r = DETACHED A-row RMS SD of recipient r logit differences '
                        '(income: SD of z1 - z0; occupation: sqrt(mean_k Var(z_k - z0)))',
    'ridge_by_family': {'simple': RF.RECIPE_RF['ridge'], 'rff': RF.RECIPE_RF['ridge'], 'softbin': 1e-3},
    'softbin_ridge_rationale': 'memberships have variance ~ 1/B; ridge 1e-3 gives about the per-feature shrinkage of ridge 1e-2 on the '
                               'unit-variance v1 features (fixed a priori)',
})
RECIPE_RF = RECIPE_RF_R1                    # drop-in name for rfa/fam/fit.py


# ---------------------------------------------------------------- the softbin feature map
_CENTRES = {}


def bin_centres(B):
    import torch
    if B not in _CENTRES:
        nd = statistics.NormalDist()
        _CENTRES[B] = torch.tensor([nd.inv_cdf((b - 0.5) / B) for b in range(1, B + 1)], dtype=torch.float64)
    return _CENTRES[B]


def soft_bins(s, rc=RECIPE_RF_R1):
    """(n, d) standardized coordinates -> (n, d * B) per-coordinate soft one-hot memberships (each coordinate's B sum to 1)."""
    import torch
    c = bin_centres(int(rc['softbin_bins']))
    h = float(rc['softbin_width'])
    m = torch.softmax(-(s[:, :, None] - c[None, None, :]) ** 2 / (2.0 * h * h), dim=2)
    return m.reshape(s.shape[0], -1)


def decision_scales(uA, moments=None, rc=RECIPE_RF_R1):
    """Detached A-row scales of the two recipients' logit differences (from the frozen pair moments when given)."""
    import torch
    if moments is not None and 'pair' in moments:
        sd = moments['pair'][1].detach()
    elif moments is not None and 'income' in moments and 'occupation' in moments:
        sd = torch.cat([moments['income'][1].detach(), moments['occupation'][1].detach()])
    else:
        sd = uA['pair'].detach().std(0, unbiased=False)
    inc, occ = sd[0], torch.sqrt((sd[1:] ** 2).mean())
    if not (bool(torch.isfinite(sd).all()) and float(inc) > rc['min_sd'] and float(occ) > rc['min_sd']):
        raise ReaderRefused('degenerate decision scale')
    return {'income': inc, 'occupation': occ}


def soft_decision(z, scale, rc=RECIPE_RF_R1):
    import torch
    return torch.softmax(z / (float(rc['softbin_tau']) * scale), dim=1)


def softbin_basis(view, s, z_inc, z_occ, dsc, rc=RECIPE_RF_R1):
    import torch
    n = s.shape[0]
    parts = [soft_bins(s, rc)]
    if view == 'income':
        parts.append(soft_decision(z_inc, dsc['income'], rc))
    elif view == 'occupation':
        parts.append(soft_decision(z_occ, dsc['occupation'], rc))
    elif view == 'pair':
        di, do = soft_decision(z_inc, dsc['income'], rc), soft_decision(z_occ, dsc['occupation'], rc)
        parts.append((di[:, :, None] * do[:, None, :]).reshape(n, -1))
    else:
        raise ValueError(f'unknown view {view!r}')
    parts.append(torch.ones(n, 1, dtype=s.dtype))
    return torch.cat(parts, 1)


def family_features(view, family, s, z_inc, z_occ, dsc, rc=RECIPE_RF_R1):
    if family == 'softbin':
        return softbin_basis(view, s, z_inc, z_occ, dsc, rc)
    return RF.basis(s, view, family, rc)                 # v1 families: rfit's map, unchanged


def ridge_vector_family(p, family, rc=RECIPE_RF_R1):
    import torch
    k = torch.full((p,), float(rc.get('ridge_by_family', {}).get(family, rc['ridge'])), dtype=torch.float64)
    k[-1] = 0.0
    return k


# ---------------------------------------------------------------- the R1 episode objective (rfit's loop with the family hook)
class EpisodeObjective(RF.EpisodeObjective):
    """rfit.EpisodeObjective with the R1 family map: same modes (FULL / STOPFIT / ALTCRIT), same objective / objective_parts
    signatures and info keys; S = sum_v mean_f R_vf over the R1 families."""

    def __init__(self, mode, seed=0, recipe=RECIPE_RF_R1):
        super().__init__(mode, seed, recipe)

    def objective_parts(self, zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments=None, diagnostics=True):
        import torch
        rc, mode = self.rc, self.mode
        zA, zE = (zA_inc, zA_occ), (zE_inc, zE_occ)
        for z in (*zA, *zE):
            if z.dtype != torch.float64:
                raise TypeError('logits must be float64')
            if not bool(torch.isfinite(z.detach()).all()):
                raise ReaderRefused('non-finite release logits')
        if mode == 'FULL' and not (zA_inc.requires_grad or zA_occ.requires_grad):
            raise ValueError('FULL needs the A-row logits with their autograd graph (the fit path flows through them)')
        yA, yE, mE = sA.to(torch.float64), sE.to(torch.float64), sE.bool()
        uA, uE = view_inputs(*zA), view_inputs(*zE)
        dsc = decision_scales(uA, moments, rc) if 'softbin' in self.families else None
        S_dir = torch.zeros((), dtype=torch.float64)
        S_fit = torch.zeros((), dtype=torch.float64)
        info = {'mode': mode, 'refused': False, 'terms': {}, 'revision': REVISION}
        nf = len(self.families)
        for v in self.views:
            mom = moments[v] if moments is not None else moments_of(uA[v], rc)
            sa, se = standardize(uA[v], mom, rc), standardize(uE[v], mom, rc)
            for f in self.families:
                key = (v, f)
                phiA = family_features(v, f, sa, zA_inc, zA_occ, dsc, rc)
                phiE = family_features(v, f, se, zE_inc, zE_occ, dsc, rc)
                kap = ridge_vector_family(phiA.shape[1], f, rc)
                if mode == 'ALTCRIT':
                    rec = self._critic_update(key, phiA.detach(), yA, kap)
                    w, H = self.critic[key].detach(), None
                else:
                    w0 = self.w_prev.get(key) if rc['newton_warm_start'] else None
                    w, H, rec = fit_reader(phiA, yA, kap, rc, w0)
                    self.w_prev[key] = w
                wl = w.detach().clone().requires_grad_(mode == 'FULL')
                tE = phiE @ wl
                auc_b, dm = binormal_auc(tE, mE)
                R = (auc_b - 0.5) ** 2
                if not bool(torch.isfinite(R)):
                    raise ReaderRefused('non-finite surrogate')
                S_dir = S_dir + R / nf
                rec['R'] = float(R.detach())
                if mode == 'FULL':
                    gw = torch.autograd.grad(R, wl, retain_graph=True)[0].detach()
                    vv = torch.linalg.solve(H, gw)
                    gA = phiA.T @ (torch.sigmoid(phiA @ w) - yA) / len(yA) + kap * w
                    fs = -(vv * gA).sum()
                    if not bool(torch.isfinite(fs)):
                        raise ReaderRefused('non-finite fit-path term')
                    S_fit = S_fit + fs / nf
                    rec.update({'norm_grad_w_R': float(gw.norm()), 'norm_v': float(vv.norm()), 'fit_scalar_value': float(fs.detach())})
                if diagnostics:
                    with torch.no_grad():
                        rec.update({'auc_binormal_E': float(auc_b), 'orientation_sign_E': int(np.sign(float(dm))),
                                    'auc_exact_E': exact_auc(tE.detach().numpy(), sE.numpy()),
                                    'auc_exact_A': exact_auc((phiA @ w).detach().numpy(), sA.numpy())})
                info['terms'][f'{v}/{f}'] = rec
        if mode != 'FULL':
            S_fit = 0.0 * S_dir
        info['S_dir'] = info['P'] = float(S_dir.detach())
        info['S_fit'] = float(S_fit.detach())
        if dsc is not None:
            info['decision_scales'] = {k: float(x) for k, x in dsc.items()}
        if mode == 'ALTCRIT':
            lags = [t['critic_rel_lag'] for t in info['terms'].values()]
            info['critic_rel_lag_median'] = float(np.median(lags))
            info['critic_max_abs_grad_A_max'] = max(t['critic_max_abs_grad_A'] for t in info['terms'].values())
        if mode == 'FULL' and diagnostics:
            eo = [z for z in zE if z.requires_grad]
            ao = [z for z in zA if z.requires_grad]
            gd = torch.autograd.grad(S_dir, eo, retain_graph=True, allow_unused=True) if eo else []
            gf = torch.autograd.grad(S_fit, ao, retain_graph=True, allow_unused=True) if ao else []
            nd = math.sqrt(sum(float((x ** 2).sum()) for x in gd if x is not None))
            nfp = math.sqrt(sum(float((x ** 2).sum()) for x in gf if x is not None))
            info['output_grad_norms'] = {'direct_wrt_E_outputs': nd, 'fit_wrt_A_outputs': nfp,
                                         'fit_share': nfp / (nd + nfp) if nd + nfp > 0 else 0.0}
        self.calls += 1
        return S_dir, S_fit, info


EpisodeObjectiveR1 = EpisodeObjective


# ---------------------------------------------------------------- invented-data checks (float64, deterministic)
def _r1(**kw):
    rc = copy.deepcopy(RECIPE_RF_R1)
    rc.update(kw)
    return rc


def _grads(cls, theta0, XA, sA, XE, sE, rc, moments=None, freeze_income=False):
    import torch
    out = {}
    for mode in ('FULL', 'STOPFIT'):
        th = theta0.clone().requires_grad_(True)
        (ziA, zoA), (ziE, zoE) = RF._tiny_logits(th, XA), RF._tiny_logits(th, XE)
        if freeze_income:
            ziA, ziE = RF._tiny_logits(theta0, XA)[0].detach(), RF._tiny_logits(theta0, XE)[0].detach()
        S, info = cls(mode, 0, rc).objective(ziA, zoA, sA, ziE, zoE, sE, moments=moments)
        out[mode] = (torch.autograd.grad(S, th)[0].detach(), info, float(S.detach()))
    return out


def _P(theta, theta0, XA, sA, XE, sE, rc, moments, freeze_income=False):
    (ziA, zoA), (ziE, zoE) = RF._tiny_logits(theta, XA), RF._tiny_logits(theta, XE)
    if freeze_income:
        ziA, ziE = RF._tiny_logits(theta0, XA)[0].detach(), RF._tiny_logits(theta0, XE)[0].detach()
    return EpisodeObjective('STOPFIT', 0, rc).objective(ziA, zoA, sA, ziE, zoE, sE, moments=moments, diagnostics=False)[1]['P']


def _fd(theta0, idx, XA, sA, XE, sE, rc, moments, eps=1e-5, freeze_income=False):
    import torch
    fd = torch.zeros(len(idx), dtype=torch.float64)
    for k, j in enumerate(idx):
        e = torch.zeros_like(theta0); e[j] = eps
        fd[k] = (_P(theta0 + e, theta0, XA, sA, XE, sE, rc, moments, freeze_income)
                 - _P(theta0 - e, theta0, XA, sA, XE, sE, rc, moments, freeze_income)) / (2 * eps)
    return fd


def selftest():
    import torch
    from acs_replication import neural
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    t0 = time.process_time()
    res, num = {}, {}
    XA, sA = RF._invented(256, 11); XE, sE = RF._invented(256, 12)
    th0 = RF._tiny(3)
    zA0 = RF._tiny_logits(th0, XA)
    mom = {v: moments_of(view_inputs(*zA0)[v]) for v in VIEWS}
    # 1. regression: with the v1 families the R1 class reproduces rfit.EpisodeObjective (values and gradients)
    rv1 = RF._recipe(newton_warm_start=False, views=list(VIEWS))
    G1 = RF._grads(th0, XA, sA, XE, sE, rv1, mom)
    G2 = _grads(EpisodeObjective, th0, XA, sA, XE, sE, _r1(newton_warm_start=False, views=list(VIEWS), families=['simple', 'rff']), mom)
    d_reg = max(float((G1[m][0] - G2[m][0]).abs().max()) for m in ('FULL', 'STOPFIT'))
    res['r1_reproduces_v1_families'] = d_reg == 0.0 and G1['FULL'][1]['P'] == G2['FULL'][1]['P']
    num['r1_reproduces_v1_families'] = {'max_abs_grad_diff': d_reg, 'P_v1': G1['FULL'][1]['P'], 'P_r1': G2['FULL'][1]['P']}
    # 2. complete-derivative central finite differences, softbin alone, all three views, all 49 release parameters
    rsb = _r1(newton_warm_start=False, views=list(VIEWS), families=['softbin'])
    G = _grads(EpisodeObjective, th0, XA, sA, XE, sE, rsb, mom)
    gf, gs = G['FULL'][0], G['STOPFIT'][0]
    fd = _fd(th0, range(len(th0)), XA, sA, XE, sE, rsb, mom)
    rf_, rs_ = float((fd - gf).norm() / fd.norm()), float((fd - gs).norm() / fd.norm())
    res['fd_softbin_complete_derivative'] = rf_ < 1e-6 and rs_ > 1e-3
    num['fd_softbin_complete_derivative'] = {
        'parameters': len(th0), 'n_A': 256, 'n_E': 256, 'eps': 1e-5, 'rel_err_FULL_vs_fd': rf_,
        'max_abs_err_FULL_vs_fd': float((fd - gf).abs().max()), 'rel_err_STOPFIT_vs_fd': rs_, **RF._cmp(gf, gs),
        'max_newton_iterations': max(t['newton_iterations'] for t in G['FULL'][1]['terms'].values()),
        'max_cond_H': max(t['cond_H'] for t in G['FULL'][1]['terms'].values()),
        'max_abs_fit_scalar_value': max(abs(t['fit_scalar_value']) for t in G['FULL'][1]['terms'].values())}
    # 3. the registered R1 recipe in the loop's pattern: three families, views (occupation, pair), income logits constant
    rreg = _r1(newton_warm_start=False)
    Gr = _grads(EpisodeObjective, th0, XA, sA, XE, sE, rreg, mom, freeze_income=True)
    o = 14
    fdo = _fd(th0, range(o, len(th0)), XA, sA, XE, sE, rreg, mom, freeze_income=True)
    ro = float((fdo - Gr['FULL'][0][o:]).norm() / fdo.norm())
    res['fd_registered_r1_income_constant'] = (ro < 1e-6 and float(Gr['FULL'][0][:o].abs().max()) == 0.0
                                              and sorted(Gr['FULL'][1]['terms']) == sorted(f'{v}/{f}' for v in ('occupation', 'pair')
                                                                                          for f in FAMILIES_R1))
    num['fd_registered_r1_income_constant'] = {'parameters': len(th0) - o, 'rel_err_FULL_vs_fd': ro,
                                              'terms': sorted(Gr['FULL'][1]['terms']), **RF._cmp(Gr['FULL'][0][o:], Gr['STOPFIT'][0][o:])}
    # 4. optimization sign along -grad(FULL), softbin alone
    steps = []
    for eta in (1e-3, 1e-2):
        pn = _P(th0 - eta * gf, th0, XA, sA, XE, sE, rsb, mom)
        steps.append({'eta': eta, 'P0': G['FULL'][1]['P'], 'P_after': pn, 'decrease': G['FULL'][1]['P'] - pn,
                      'first_order_prediction': eta * float(gf @ gf)})
    res['optimization_sign_softbin'] = all(s['decrease'] > 0 for s in steps) and \
        abs(steps[0]['decrease'] / steps[0]['first_order_prediction'] - 1) < 0.05
    num['optimization_sign_softbin'] = steps
    # 5. FULL vs STOPFIT for softbin per view at campaign-like sizes (invented: mean shift + variance signal)
    XA4, sA4 = RF._invented(4096, 41, signal=0.3, nonlinear=0.2); XE4, sE4 = RF._invented(2400, 42, signal=0.3, nonlinear=0.2)
    per = {}
    for v in VIEWS:
        Gx = _grads(EpisodeObjective, th0, XA4, sA4, XE4, sE4, _r1(newton_warm_start=False, views=[v], families=['softbin']))
        per[f'{v}/softbin'] = {**RF._cmp(Gx['FULL'][0], Gx['STOPFIT'][0]),
                               'auc_exact_E': Gx['FULL'][1]['terms'][f'{v}/softbin']['auc_exact_E'],
                               'cond_H': Gx['FULL'][1]['terms'][f'{v}/softbin']['cond_H']}
    Gall = _grads(EpisodeObjective, th0, XA4, sA4, XE4, sE4, _r1(newton_warm_start=False), freeze_income=False)
    num['fit_path_softbin_4096_2400_invented'] = {'per_view': per, 'registered_all_terms': RF._cmp(Gall['FULL'][0], Gall['STOPFIT'][0]),
                                                  'registered_output_grad_norms': Gall['FULL'][1]['output_grad_norms']}
    res['softbin_fit_path_nonzero'] = all(per[k]['norm_fit_over_direct'] > 1e-3 for k in per)
    # 6. structure: per-coordinate memberships, decisions and the pair products each sum to 1, all in [0, 1]
    with torch.no_grad():
        uA0 = view_inputs(*zA0)
        dsc = decision_scales(uA0, mom)
        ph = softbin_basis('pair', standardize(uA0['pair'], mom['pair']), zA0[0], zA0[1], dsc)
        B = RECIPE_RF_R1['softbin_bins']
        sums = [float((ph[:, j * B:(j + 1) * B].sum(1) - 1).abs().max()) for j in range(5)]
        sums.append(float((ph[:, 5 * B:5 * B + 10].sum(1) - 1).abs().max()))
    res['softbin_structure'] = max(sums) < 1e-12 and float(ph.min()) >= 0 and float(ph.max()) <= 1 and ph.shape[1] == 51
    num['softbin_structure'] = {'pair_width': int(ph.shape[1]), 'max_abs_sum_minus_1': max(sums),
                                'centres': [float(x) for x in bin_centres(B)], 'decision_scales': {k: float(x) for k, x in dsc.items()}}
    # 7. ALTCRIT: newton inner == STOPFIT bitwise; adam lags; objective_parts consistency
    torch.manual_seed(9)
    nets = [neural.network(2, 145), neural.network(5, 145)]
    rng = np.random.default_rng(7)
    Xc = torch.from_numpy(rng.normal(size=(1024, 145)).astype(np.float32)); sc = torch.from_numpy(rng.integers(0, 2, 1024))
    Xc[:, :10] += 0.4 * (2 * sc[:, None] - 1).float()
    Ac, Ec, Wc = torch.arange(512), torch.arange(512, 1024), torch.arange(256, 768)
    with torch.no_grad():
        zinc = RF.centred_logits(nets[0], Xc)
    occp = list(nets[1].parameters())

    def g_of(obj, warm=None):
        if warm is not None:
            with torch.no_grad():
                zw = RF.centred_logits(nets[1], Xc[warm])
            obj.objective(zinc[warm], zw, sc[warm], zinc[warm], zw, sc[warm], diagnostics=False)
        S_, inf_ = obj.objective(zinc[Ac], RF.centred_logits(nets[1], Xc[Ac]), sc[Ac], zinc[Ec], RF.centred_logits(nets[1], Xc[Ec]),
                                 sc[Ec])
        return torch.cat([x.reshape(-1).double() for x in torch.autograd.grad(S_, occp)]), inf_

    a_s, _ = g_of(EpisodeObjective('STOPFIT', 0, rreg))
    a_n, inf_n = g_of(EpisodeObjective('ALTCRIT', 0, _r1(newton_warm_start=False, altcrit_inner='newton')), warm=Wc)
    a_l, inf_l = g_of(EpisodeObjective('ALTCRIT', 0, rreg), warm=Wc)
    res['altcrit_newton_alias_of_stopfit_r1'] = float((a_s - a_n).abs().max()) == 0.0 and inf_n['critic_rel_lag_median'] == 0.0
    res['altcrit_adam_lagging_r1'] = float((a_s - a_l).norm() / a_s.norm()) > 1e-3 and inf_l['critic_rel_lag_median'] > 0.0
    num['altcrit_r1'] = {'max_abs_diff_newton_vs_stopfit': float((a_s - a_n).abs().max()),
                         'rel_diff_adam_vs_stopfit': float((a_s - a_l).norm() / a_s.norm()),
                         'cos_adam_vs_stopfit': float(a_s @ a_l / (a_s.norm() * a_l.norm())),
                         'rel_lag_median_adam': inf_l['critic_rel_lag_median']}
    th = th0.clone().requires_grad_(True)
    (ziA, zoA), (ziE, zoE) = RF._tiny_logits(th, XA), RF._tiny_logits(th, XE)
    Sd, Sf, inf_p = EpisodeObjective('FULL', 0, rsb).objective_parts(ziA, zoA, sA, ziE, zoE, sE, moments=mom)
    g_d = torch.autograd.grad(Sd, th, retain_graph=True)[0]
    g_f = torch.autograd.grad(Sf, th)[0]
    sc_ = float(gf.abs().max())
    res['objective_parts_consistent_r1'] = (float((g_d - gs).abs().max()) <= 1e-14 * sc_ and
                                            float((g_f - (gf - gs)).abs().max()) <= 1e-13 * sc_ and abs(inf_p['S_fit']) < 1e-12)
    num['objective_parts_r1'] = {'max_abs_grad_S_dir_minus_STOPFIT': float((g_d - gs).abs().max()),
                                 'max_abs_grad_S_fit_minus_FULL_minus_STOPFIT': float((g_f - (gf - gs)).abs().max()),
                                 'S_fit_value': inf_p['S_fit']}
    # 8. guards, interface and monitor compatibility
    try:
        EpisodeObjective('STOPFIT', 0, rreg).objective(zA0[0].detach(), zA0[1].detach(), torch.zeros(256, dtype=torch.int64),
                                                        zA0[0].detach(), zA0[1].detach(), sA)
        res['one_class_A_refused_r1'] = False
    except FloatingPointError:
        res['one_class_A_refused_r1'] = True
    mon = monitor_fisher2(zA0[0].detach(), zA0[1].detach(), sA, *(z.detach() for z in RF._tiny_logits(th0, XE)), sE)
    res['interface_r1'] = (RF._plain(G['FULL'][1]) and RF._plain(inf_l) and RF._plain(mon) and RECIPE_RF['n_A'] == 4096
                           and RECIPE_RF['n_E'] == 2400 and RECIPE_RF['views'] == ['occupation', 'pair']
                           and RECIPE_RF['families'] == list(FAMILIES_R1) and issubclass(EpisodeObjective, RF.EpisodeObjective))
    res['all_pass'] = all(v for k, v in res.items() if isinstance(v, bool))
    res['numbers'] = num
    res['cpu_s'] = time.process_time() - t0
    return res


def bench(n_a=4096, n_e=2400, reps=4):
    """CPU seconds per step in the training loop's pattern (occupation network trained, START income logits constant, diagnostics
    on, 256-row task batch, Adam, clip 5): R1 (three families) and v1 (two families) on identical invented rows. Random weights."""
    import torch
    import torch.nn.functional as F
    from acs_replication import neural
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = {'schema': 'RFA1-rfit-r1-bench-v1', 'threads': torch.get_num_threads(), 'n_A': n_a, 'n_E': n_e, 'rows': {}}
    rng = np.random.default_rng(0)
    N = 16000
    X = torch.from_numpy(rng.normal(size=(N, 145)).astype(np.float32))
    sex = torch.from_numpy(rng.integers(0, 2, N))
    X[:, :10] += 0.3 * (2 * sex[:, None] - 1).float()
    y = torch.from_numpy(rng.integers(0, 5, N))
    hh = RF.household_index(np.repeat(np.arange(N // 2), 2))
    torch.manual_seed(0)
    with torch.no_grad():
        zi = RF.centred_logits(neural.network(2, 145), X)
    for tag, cls, rc in (('R1', EpisodeObjective, RECIPE_RF_R1), ('v1', RF.EpisodeObjective, RF.RECIPE_RF)):
        for name in ('FULL', 'STOPFIT', 'ALTCRIT', 'TASK'):
            if tag == 'v1' and name == 'TASK':
                continue
            torch.manual_seed(1)
            occ = neural.network(5, 145)
            params = list(occ.parameters())
            opt = torch.optim.Adam(params, lr=1e-4)
            obj = None if name == 'TASK' else cls(name, 0, rc)
            erng = np.random.default_rng(5)
            times = []
            for r in range(reps + 1):
                A, E = (torch.from_numpy(x) for x in RF.episode_rows(hh, erng, n_a, n_e))
                tr = torch.from_numpy(erng.choice(N, 256, replace=False))
                t0 = time.process_time()
                if obj is not None:
                    S, info = obj.objective(zi[A], RF.centred_logits(occ, X[A]), sex[A], zi[E], RF.centred_logits(occ, X[E]), sex[E])
                    gp = torch.autograd.grad(S, params)
                else:
                    gp = [torch.zeros_like(p) for p in params]
                gt = torch.autograd.grad(F.cross_entropy(RF.centred_logits(occ, X[tr]), y[tr]), params)
                opt.zero_grad(set_to_none=True)
                for p, a, c in zip(params, gt, gp):
                    p.grad = a + c
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                if r > 0:
                    times.append(time.process_time() - t0)
            out['rows'][f'{tag}_{name}'] = {'cpu_s_per_step_mean': float(np.mean(times)), 'cpu_s_per_step_max': float(np.max(times)),
                                            'steps_timed': reps}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rfa.fam.rfit_r1')
    ap.add_argument('cmd', choices=('selftest', 'bench'))
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    t0 = time.process_time()
    rep = selftest() if a.cmd == 'selftest' else bench()
    rep['process_cpu_s'] = time.process_time() - t0
    txt = json.dumps(rep, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write(txt + '\n')
    print(txt)
    if a.cmd == 'selftest' and not rep['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()

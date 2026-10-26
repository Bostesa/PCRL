#!/usr/bin/env python3
"""RFA1 verifier, wave 2 addition: derivative check of the revision R1 objective rfa/fam/rfit_r1.py (softbin family) on INVENTED data.

The verifier's own numpy implementation of the R1 episode objective (v1 views / standardization / SIMPLE / RFF from
v1_graph_check, plus the softbin map written from the docstring and recipe text: per-coordinate softmax memberships over 8 Gaussian
kernels at Phi^-1((b - 1/2) / 8) (scipy.stats.norm.ppf, not statistics.NormalDist), width 0.3; soft decisions softmax(z / (0.25 x
scale)) with scale_inc = SD of z1 - z0 and scale_occ = sqrt(mean_k Var(z_k - z0)) of the A rows (population SDs); pair products
d_inc (x) d_occ; intercept; ridge 1e-3 for softbin) defines J(theta) with w* re-solved at every theta and the A-row moments AND the
decision scales frozen at theta0 (the module detaches both). Central FD of J vs the autograd gradient of the module's FULL scalar; the
STOPFIT direct path vs FD at fixed w; the optimization sign; R1 with families ['simple', 'rff'] vs v1 (rfit) bitwise.

Run: <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_r1_check.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vguard  # noqa: E402

vguard.install()
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402
from scipy.stats import norm  # noqa: E402

import v1_graph_check as G  # noqa: E402  (the verifier's wave-1 implementation: views, moments, features, Newton, binormal, FD)

OUT = HERE.parent / 'out' / 'V2_R1_CHECK.json'
B, H_W, TAU, RIDGE_SB = 8, 0.3, 0.25, 1e-3
CENTRES = norm.ppf((np.arange(1, B + 1) - 0.5) / B)


def sb_bins(s):
    a = -(s[:, :, None] - CENTRES[None, None, :]) ** 2 / (2.0 * H_W * H_W)
    return softmax(a, axis=2).reshape(s.shape[0], -1)


def sb_features(view, s, zi, zo, dsc):
    di = softmax(zi / (TAU * dsc['income']), axis=1)
    do = softmax(zo / (TAU * dsc['occupation']), axis=1)
    parts = [sb_bins(s)]
    if view == 'income':
        parts.append(di)
    elif view == 'occupation':
        parts.append(do)
    else:
        parts.append((di[:, :, None] * do[:, None, :]).reshape(len(s), -1))
    parts.append(np.ones((len(s), 1)))
    return np.hstack(parts)


def scales_of(uA):
    sd = uA['pair'].std(0)
    return {'income': float(sd[0]), 'occupation': float(math.sqrt(float(np.mean(sd[1:] ** 2))))}


def J_r1(zA, zE, yA, yE, mom, dsc, views, fams, w_fixed=None, w_warm=None):
    uA, uE = G.v_views(*zA), G.v_views(*zE)
    m = yE.astype(bool)
    tot, ws, terms = 0.0, {}, {}
    for v in views:
        sa, se = G.v_std(uA[v], mom[v]), G.v_std(uE[v], mom[v])
        for f in fams:
            if f == 'softbin':
                pa, pe = sb_features(v, sa, zA[0], zA[1], dsc), sb_features(v, se, zE[0], zE[1], dsc)
                kap = G.v_kap(pa.shape[1], RIDGE_SB)
            else:
                pa, pe = G.v_feats(sa, v, f), G.v_feats(se, v, f)
                kap = G.v_kap(pa.shape[1], 1e-2)
            if w_fixed is not None:
                w = w_fixed[(v, f)]
            else:
                w = G.v_newton(pa, yA, kap, None if w_warm is None else w_warm[(v, f)])[0]
            ws[(v, f)] = w
            R = (G.v_binormal(pe @ w, m) - 0.5) ** 2
            terms[f'{v}/{f}'] = R
            tot += R / len(fams)
    return tot, ws, terms


def module_grad(R1, mode, theta, ziA, ziE, XA, XE, sA, sE, recipe, cls=None, obj=None, inc_theta=False):
    import torch
    th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    zoA, zoE = G.occ_t(th, torch.from_numpy(XA)), G.occ_t(th, torch.from_numpy(XE))
    obj = obj if obj is not None else (cls or R1.EpisodeObjective)(mode, 0, recipe)
    S, info = obj.objective(torch.from_numpy(ziA), zoA, torch.from_numpy(sA), torch.from_numpy(ziE), zoE, torch.from_numpy(sE))
    return torch.autograd.grad(S, th)[0].detach().numpy().copy(), info, float(S.detach()), obj


def main():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0 = time.process_time()
    from rfa.fam import rfit as RF
    from rfa.fam import rfit_r1 as R1
    res = {'schema': 'RFA1-verifier-r1-check-v1', 'rfit_r1_sha256': hashlib.sha256((ROOT / 'rfa/fam/rfit_r1.py').read_bytes()).hexdigest(),
           'rfit_sha256': hashlib.sha256((ROOT / 'rfa/fam/rfit.py').read_bytes()).hexdigest(), 'checks': {}, 'numbers': {}}
    chk, num = res['checks'], res['numbers']
    rc = copy.deepcopy(R1.RECIPE_RF_R1)
    views, fams = tuple(rc['views']), tuple(rc['families'])
    num['recipe'] = {'views': list(views), 'families': list(fams), 'softbin_bins': rc['softbin_bins'], 'softbin_width': rc['softbin_width'],
                     'softbin_tau': rc['softbin_tau'], 'ridge_by_family': rc['ridge_by_family']}
    chk['recipe_constants_as_documented'] = (rc['softbin_bins'] == B and rc['softbin_width'] == H_W and rc['softbin_tau'] == TAU and
                                             rc['ridge_by_family'] == {'simple': 1e-2, 'rff': 1e-2, 'softbin': RIDGE_SB} and
                                             fams == ('simple', 'rff', 'softbin'))
    num['centres_max_abs_diff_vs_module'] = float(np.max(np.abs(R1.bin_centres(B).numpy() - CENTRES)))
    chk['bin_centres_match_scipy_normal_quantiles (abs < 1e-14)'] = num['centres_max_abs_diff_vs_module'] < 1e-14
    # invented household-separated episode, tiny MLP occupation release, constant income logits (as in the training loop)
    X, s, unit = G.invented(420, 101)
    A, E = G.household_split(unit, 400, 400, 5)
    XA, XE, sA, sE = X[A], X[E], s[A], s[E]
    ziA, ziE = G.inc_const(XA), G.inc_const(XE)
    theta0 = np.random.default_rng(3).normal(size=G.N_THETA) * 0.5
    zA0, zE0 = (ziA, G.occ_np(theta0, XA)), (ziE, G.occ_np(theta0, XE))
    uA0 = G.v_views(*zA0)
    mom0 = {v: G.v_moments(uA0[v]) for v in G.VIEWS}
    dsc0 = scales_of(uA0)
    yA, yE = sA.astype(float), sE.astype(float)
    J0, ws0, terms0 = J_r1(zA0, zE0, yA, yE, mom0, dsc0, views, fams)
    # feature map agreement on the same inputs (module helpers vs the verifier's)
    fdiff = 0.0
    # run 1 passed torch.tensor(python float) = FLOAT32 scales here (harness bug; output kept as V2_R1_CHECK_run1_*); float64 now
    tsc = {k: torch.tensor(x, dtype=torch.float64) for k, x in dsc0.items()}
    for v in views:
        sa = G.v_std(uA0[v], mom0[v])
        ph_m = R1.softbin_basis(v, torch.from_numpy(sa), torch.from_numpy(zA0[0]), torch.from_numpy(zA0[1]), tsc).numpy()
        fdiff = max(fdiff, float(np.max(np.abs(ph_m - sb_features(v, sa, zA0[0], zA0[1], dsc0)))))
    mods = R1.decision_scales({k: torch.from_numpy(x) for k, x in uA0.items()})
    num['softbin_feature_max_abs_diff_vs_module'] = fdiff
    num['decision_scales'] = {'verifier': dsc0, 'module': {k: float(x) for k, x in mods.items()}}
    chk['softbin_features_match_verifier (abs < 1e-13)'] = fdiff < 1e-13
    chk['decision_scales_match_verifier (abs < 1e-13)'] = all(abs(float(mods[k]) - dsc0[k]) < 1e-13 for k in dsc0)
    # module value and gradients
    gF, infF, SF, _ = module_grad(R1, 'FULL', theta0, ziA, ziE, XA, XE, sA, sE, rc)
    gS, infS, SS, _ = module_grad(R1, 'STOPFIT', theta0, ziA, ziE, XA, XE, sA, sE, rc)
    num['forward'] = {'module_P': infS['P'], 'verifier_J0': J0, 'abs_diff': abs(infS['P'] - J0),
                      'max_abs_term_diff': max(abs(infS['terms'][k]['R'] - terms0[k]) for k in terms0), 'terms': terms0,
                      'newton_iterations': {k: t['newton_iterations'] for k, t in infF['terms'].items()},
                      'cond_H': {k: t['cond_H'] for k, t in infF['terms'].items()}}
    chk['forward_value_matches_verifier (abs < 1e-12)'] = abs(infS['P'] - J0) < 1e-12

    def J_resolve(th):
        return J_r1((ziA, G.occ_np(th, XA)), (ziE, G.occ_np(th, XE)), yA, yE, mom0, dsc0, views, fams, w_warm=ws0)[0]

    def J_fixed(th):
        return J_r1((ziA, G.occ_np(th, XA)), (ziE, G.occ_np(th, XE)), yA, yE, mom0, dsc0, views, fams, w_fixed=ws0)[0]
    fd_r = {h: G.fd(J_resolve, theta0, h) for h in (1e-4, 1e-5)}
    fd_f = G.fd(J_fixed, theta0, 1e-5)
    num['fd'] = {'FULL_vs_FD_resolved': {str(h): G.rel(gF, x) for h, x in fd_r.items()}, 'STOPFIT_vs_FD_fixed_w': G.rel(gS, fd_f),
                 'STOPFIT_vs_FD_resolved': G.rel(gS, fd_r[1e-5]), 'cos_FULL_STOPFIT': G.cos(gF, gS),
                 'norm_fit_path_over_direct': float(np.linalg.norm(gF - gS) / np.linalg.norm(gS)),
                 'max_abs_err_FULL_1e-5': float(np.max(np.abs(gF - fd_r[1e-5])))}
    chk['R1_FULL_matches_FD_of_resolved_J (rel < 1e-6)'] = min(G.rel(gF, x) for x in fd_r.values()) < 1e-6
    chk['R1_STOPFIT_matches_FD_at_fixed_w (rel < 1e-6)'] = G.rel(gS, fd_f) < 1e-6
    chk['R1_STOPFIT_is_not_the_bilevel_gradient (rel > 1e-2)'] = G.rel(gS, fd_r[1e-5]) > 1e-2
    # softbin terms alone (the new family isolated)
    rc_sb = copy.deepcopy(rc); rc_sb['families'] = ['softbin']
    gFb = module_grad(R1, 'FULL', theta0, ziA, ziE, XA, XE, sA, sE, rc_sb)[0]
    J0b, wsb, _ = J_r1(zA0, zE0, yA, yE, mom0, dsc0, views, ('softbin',))
    fdb = G.fd(lambda th: J_r1((ziA, G.occ_np(th, XA)), (ziE, G.occ_np(th, XE)), yA, yE, mom0, dsc0, views, ('softbin',), w_warm=wsb)[0],
               theta0, 1e-5)
    num['softbin_only'] = {'FULL_vs_FD_resolved_rel': G.rel(gFb, fdb), 'J0': J0b}
    chk['softbin_only_FULL_matches_FD (rel < 1e-6)'] = G.rel(gFb, fdb) < 1e-6
    # sign
    u = gF / np.linalg.norm(gF)
    steps = []
    for d in (1e-4, 1e-3):
        Jn = J_resolve(theta0 - d * u)
        steps.append({'step': d, 'decrease': J0 - Jn, 'ratio_actual_over_predicted': (J0 - Jn) / (d * float(np.linalg.norm(gF)))})
    num['sign'] = steps
    chk['R1_sign_step_along_minus_grad_lowers_J'] = all(x['decrease'] > 0 for x in steps) and abs(steps[0]['ratio_actual_over_predicted'] - 1) < 0.01
    # R1 with families ['simple', 'rff'] reproduces v1 bitwise (FULL, STOPFIT, ALTCRIT over two consecutive episodes)
    rc_v1fam = copy.deepcopy(rc); rc_v1fam['families'] = ['simple', 'rff']
    rc_v1 = copy.deepcopy(RF.RECIPE_RF)
    A2, E2 = G.household_split(unit, 400, 400, 6)
    eps = [(A, E, theta0), (A2, E2, theta0 - 1e-2 * u)]
    same = {}
    for mode in ('FULL', 'STOPFIT', 'ALTCRIT'):
        o1, o2 = R1.EpisodeObjective(mode, 0, rc_v1fam), RF.EpisodeObjective(mode, 0, rc_v1)
        ok, worst = True, 0.0
        for a_, e_, th in eps:
            g1, i1, s1, _ = module_grad(R1, mode, th, G.inc_const(X[a_]), G.inc_const(X[e_]), X[a_], X[e_], s[a_], s[e_], rc_v1fam, obj=o1)
            g2, i2, s2, _ = module_grad(R1, mode, th, G.inc_const(X[a_]), G.inc_const(X[e_]), X[a_], X[e_], s[a_], s[e_], rc_v1, obj=o2)
            ok &= bool(np.array_equal(g1, g2) and s1 == s2)
            worst = max(worst, float(np.max(np.abs(g1 - g2))), abs(s1 - s2))
        same[mode] = {'bitwise': ok, 'max_abs_diff': worst}
    num['r1_v1_families_vs_v1'] = same
    chk['R1_with_v1_families_reproduces_v1_bitwise (FULL, STOPFIT, ALTCRIT, 2 calls)'] = all(x['bitwise'] for x in same.values())
    res['rfit_r1_sha256_at_end'] = hashlib.sha256((ROOT / 'rfa/fam/rfit_r1.py').read_bytes()).hexdigest()
    res['unchanged_during_check'] = res['rfit_r1_sha256_at_end'] == res['rfit_r1_sha256']
    res['guard_refused_opens'] = list(vguard.REFUSED)
    res['all_pass'] = all(bool(x) for x in chk.values())
    res['cpu_s'] = time.process_time() - t0
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'checks': chk, 'fd': num['fd'], 'softbin_only': num['softbin_only'],
                      'sign': num['sign'], 'r1_v1': same, 'forward': {k: num['forward'][k] for k in ('abs_diff', 'max_abs_term_diff', 'cond_H')},
                      'cpu_s': res['cpu_s']}, indent=1, default=float))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

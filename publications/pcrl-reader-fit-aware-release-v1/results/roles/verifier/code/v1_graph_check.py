#!/usr/bin/env python3
"""RFA1 independent verifier, wave 1, check 1: computational graph and derivative of rfa/fam/rfit.py (INVENTED data only).

The verifier's own float64 NUMPY implementation of the episode objective (view inputs, A-row standardization, soft bound, SIMPLE / RFF
features, ridge-logistic inner fit by the verifier's own damped Newton solver polished to the floating-point floor, binormal AUC,
symmetric surrogate) defines the bilevel objective
    J(theta) = sum_v mean_f (AUC_b(phi_f(E; theta) w*_vf(theta; A)) - 1/2)^2,   w* RE-SOLVED at every perturbed theta,
with the A-row standardization moments frozen at theta0 (the module detaches them: constants of the step). Central finite
differences of J (numpy forward of the release model; no torch, no module code) are compared with the autograd gradient of the
module's scalar S. The module is only CALLED (EpisodeObjective(...).objective, fit_reader for a solver comparison, rff_params for a
bitwise comparison with the verifier's regeneration); none of its gradient code is used to produce a reference value.

Run:  <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v1_graph_check.py
Writes roles/verifier/out/V1_GRAPH_CHECK.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vguard  # noqa: E402  (sets one numerical thread before numpy / torch)

vguard.install()
sys.path.insert(1, str(ROOT))

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit, ndtr  # noqa: E402

T0 = time.process_time()
OUT = HERE.parent / 'out' / 'V1_GRAPH_CHECK.json'
RFIT = ROOT / 'rfa' / 'fam' / 'rfit.py'

# ---------------------------------------------------------------- the verifier's own implementation (numpy, float64)
SOFT = 6.0
VIEW_DIM = {'income': 1, 'occupation': 4, 'pair': 5}
VIEWS = ('income', 'occupation', 'pair')
FAMS = ('simple', 'rff')


def v_seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:15], 16)


def v_rff(view, width=64):
    """Regenerated from the recipe TEXT: torch.Generator().manual_seed(seed_for('RFA1-rff', view) mod 2^62); Omega ~ N(0, 1/d) (d x 64),
    b ~ U(0, 2 pi). (The recipe defines the frequencies through torch's generator, so torch is used here only to draw them.)"""
    import torch
    g = torch.Generator().manual_seed(v_seed_for('RFA1-rff', view) % 2 ** 62)
    d = VIEW_DIM[view]
    om = torch.randn(d, width, generator=g, dtype=torch.float64) / math.sqrt(d)
    b = torch.rand(width, generator=g, dtype=torch.float64) * (2 * math.pi)
    return om.numpy().copy(), b.numpy().copy()


RFF = {v: v_rff(v) for v in VIEWS}


def v_views(zi, zo):
    li = (zi[:, 1] - zi[:, 0])[:, None]
    lo = zo[:, 1:] - zo[:, :1]
    return {'income': li, 'occupation': lo, 'pair': np.hstack([li, lo])}


def v_moments(u):
    return u.mean(0), u.std(0)          # population SD (ddof 0)


def v_std(u, mom):
    return SOFT * np.tanh(((u - mom[0]) / mom[1]) / SOFT)


def v_feats(s, view, fam):
    one = np.ones((s.shape[0], 1))
    if fam == 'simple':
        return np.hstack([s, one])
    om, b = RFF[view]
    return np.hstack([s, math.sqrt(2.0) * np.cos(s @ om + b), one])


def v_kap(p, ridge):
    k = np.full(p, float(ridge))
    k[-1] = 0.0
    return k


def v_inner(phi, y, w, kap):
    t = phi @ w
    return float(np.mean(np.logaddexp(0.0, t) - y * t) + 0.5 * np.sum(kap * w * w))


def v_grad_inner(phi, y, w, kap):
    return phi.T @ (expit(phi @ w) - y) / len(y) + kap * w


def v_newton(phi, y, kap, w0=None, max_iter=200):
    """Damped Newton (Armijo backtracking) to max|grad| < 1e-9, then pure Newton polishing until the gradient stops decreasing
    (the floating-point floor). Returns w*, final max|grad|, iterations."""
    n, p = phi.shape
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=np.float64)
    L = v_inner(phi, y, w, kap)
    best, gm, it, worse = np.inf, np.inf, 0, 0
    for it in range(1, max_iter + 1):
        q = expit(phi @ w)
        g = phi.T @ (q - y) / n + kap * w
        gm = float(np.max(np.abs(g)))
        if gm == 0.0:
            break
        if gm < 1e-9:
            if gm >= best:
                worse += 1
                if worse >= 2:
                    break
            best = min(best, gm)
        H = (phi.T * (q * (1.0 - q))) @ phi / n + np.diag(kap)
        st = np.linalg.solve(H, g)
        a = 1.0
        if gm >= 1e-9:
            dec = float(g @ st)
            while v_inner(phi, y, w - a * st, kap) > L - 1e-4 * a * dec and a > 1e-14:
                a *= 0.5
        w = w - a * st
        L = v_inner(phi, y, w, kap)
    gfin = float(np.max(np.abs(v_grad_inner(phi, y, w, kap))))
    return w, gfin, it


def v_binormal(t, m):
    a, b = t[m], t[~m]
    dm = a.mean() - b.mean()
    v = a.var() + b.var()
    return float(ndtr(dm / math.sqrt(v)))


def v_J(zA, zE, yA, yE, mom, ridge=1e-2, views=VIEWS, fams=FAMS, w_fixed=None, w_warm=None):
    """J and its parts. mom: {view: (mu, sd)} frozen moments, or None = recompute from the A rows at this theta.
    w_fixed: {(view, fam): w} (stopped fit); otherwise re-solve (w_warm only speeds the unique solve)."""
    uA, uE = v_views(*zA), v_views(*zE)
    m = yE.astype(bool)
    tot, ws, terms, gmax = 0.0, {}, {}, 0.0
    for v in views:
        mo = mom[v] if mom is not None else v_moments(uA[v])
        sa, se = v_std(uA[v], mo), v_std(uE[v], mo)
        for f in fams:
            pa, pe = v_feats(sa, v, f), v_feats(se, v, f)
            kap = v_kap(pa.shape[1], ridge)
            if w_fixed is not None:
                w = w_fixed[(v, f)]
            else:
                w, g, _ = v_newton(pa, yA, kap, None if w_warm is None else w_warm[(v, f)])
                gmax = max(gmax, g)
            ws[(v, f)] = w
            R = (v_binormal(pe @ w, m) - 0.5) ** 2
            terms[f'{v}/{f}'] = R
            tot += R / len(fams)
    return tot, ws, terms, gmax


# ---------------------------------------------------------------- invented data and a tiny release model
def invented(n_house, seed, d=6, signal=0.5, nonlinear=0.3, max_size=4):
    """Invented households (1..4 people, a shared household effect), SEX ~ Bernoulli(1/2), a SEX mean shift plus a SEX-dependent
    spread of three coordinates."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, max_size + 1, n_house)
    unit = np.repeat(np.arange(n_house), sizes)
    n = len(unit)
    s = rng.integers(0, 2, n)
    sg = (2 * s - 1)[:, None]
    hh = rng.normal(size=(n_house, d))[unit] * 0.5
    X = rng.normal(size=(n, d)) + hh + signal * sg * np.linspace(1.0, 0.2, d)[None, :]
    X[:, :3] *= np.exp(nonlinear * sg)
    return X, s.astype(np.int64), unit


def household_split(unit, n_a, n_e, seed):
    """Whole households in a random order: A until >= n_a rows, then E until >= n_e rows."""
    rng = np.random.default_rng(seed)
    hh = np.unique(unit)
    A, E, na, ne = [], [], 0, 0
    for h in rng.permutation(hh):
        rows = np.flatnonzero(unit == h)
        if na < n_a:
            A.append(rows); na += len(rows)
        elif ne < n_e:
            E.append(rows); ne += len(rows)
        else:
            break
    A, E = np.sort(np.concatenate(A)), np.sort(np.concatenate(E))
    assert not np.intersect1d(unit[A], unit[E]).size
    return A, E


D_IN, HID = 6, 8
N_THETA = D_IN * HID + HID + HID * 5 + 5


def unpack(theta):
    o = 0
    W1 = theta[o:o + D_IN * HID].reshape(D_IN, HID); o += D_IN * HID
    b1 = theta[o:o + HID]; o += HID
    W2 = theta[o:o + HID * 5].reshape(HID, 5); o += HID * 5
    b2 = theta[o:o + 5]
    return W1, b1, W2, b2


def occ_np(theta, X):
    W1, b1, W2, b2 = unpack(theta)
    z = np.tanh(X @ W1 + b1) @ W2 + b2
    return z - z.mean(1, keepdims=True)


def occ_t(th, X):
    import torch
    W1, b1, W2, b2 = unpack(th)
    z = torch.tanh(X @ W1 + b1) @ W2 + b2
    return z - z.mean(1, keepdim=True)


def inc_const(X, seed=77):
    rng = np.random.default_rng(seed)
    Wi, bi = rng.normal(size=(X.shape[1], 2)) * 0.6, rng.normal(size=2) * 0.2
    z = X @ Wi + bi
    return z - z.mean(1, keepdims=True)


# ---------------------------------------------------------------- helpers
def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def fd(fun, theta, h):
    g = np.zeros_like(theta)
    for j in range(len(theta)):
        e = np.zeros_like(theta); e[j] = h
        g[j] = (fun(theta + e) - fun(theta - e)) / (2 * h)
    return g


def module_grad(RF, mode, theta, ziA, ziE, XA, XE, sA, sE, recipe, obj=None):
    import torch
    th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    XA_t, XE_t = torch.from_numpy(XA), torch.from_numpy(XE)
    zA = (torch.from_numpy(ziA), occ_t(th, XA_t))
    zE = (torch.from_numpy(ziE), occ_t(th, XE_t))
    obj = obj if obj is not None else RF.EpisodeObjective(mode, 0, recipe)
    S, info = obj.objective(zA[0], zA[1], torch.from_numpy(sA), zE[0], zE[1], torch.from_numpy(sE))
    g = torch.autograd.grad(S, th)[0].detach().numpy().copy()
    return g, info, float(S.detach()), obj


def my_ift_grad(theta, ziA, ziE, XA, XE, sA, sE, mom, ws, ridge=1e-2, views=VIEWS, fams=FAMS, fit_path=True):
    """Verifier's own analytic hypergradient (torch autograd of the VERIFIER's forward): one Newton step from the detached
    solution, w1(theta) = w* - H^-1 grad_w L_A(w*, theta) (H, w* detached), has dw1/dtheta = -H^-1 d(grad_w L_A)/dtheta (the implicit
    function theorem) and value w*; R(w1(theta), E(theta)) therefore has the complete derivative. fit_path=False: R(w*, E(theta))."""
    import torch
    th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    zA = (torch.from_numpy(ziA), occ_t(th, torch.from_numpy(XA)))
    zE = (torch.from_numpy(ziE), occ_t(th, torch.from_numpy(XE)))

    def views_t(zi, zo):
        li = (zi[:, 1] - zi[:, 0])[:, None]
        lo = zo[:, 1:] - zo[:, :1]
        return {'income': li, 'occupation': lo, 'pair': torch.cat([li, lo], 1)}
    uA, uE = views_t(*zA), views_t(*zE)
    yA = torch.from_numpy(sA.astype(np.float64)); m = torch.from_numpy(sE.astype(bool))
    tot = torch.zeros((), dtype=torch.float64)
    for v in views:
        mu, sd = (torch.from_numpy(np.asarray(x)) for x in mom[v])
        sa = SOFT * torch.tanh(((uA[v] - mu) / sd) / SOFT)
        se = SOFT * torch.tanh(((uE[v] - mu) / sd) / SOFT)
        for f in fams:
            def ft(s):
                one = torch.ones(s.shape[0], 1, dtype=torch.float64)
                if f == 'simple':
                    return torch.cat([s, one], 1)
                om, b = (torch.from_numpy(x) for x in RFF[v])
                return torch.cat([s, math.sqrt(2.0) * torch.cos(s @ om + b), one], 1)
            pa, pe = ft(sa), ft(se)
            kap = torch.from_numpy(v_kap(pa.shape[1], ridge))
            w = torch.from_numpy(ws[(v, f)])
            if fit_path:
                pad = pa.detach()
                q = torch.sigmoid(pad @ w)
                H = (pad.T * (q * (1 - q))) @ pad / len(yA) + torch.diag(kap)
                g = pa.T @ (torch.sigmoid(pa @ w) - yA) / len(yA) + kap * w
                w = w - torch.linalg.solve(H, g)
            t = pe @ w
            a, b_ = t[m], t[~m]
            dm = a.mean() - b_.mean()
            var = ((a - a.mean()) ** 2).mean() + ((b_ - b_.mean()) ** 2).mean()
            auc = 0.5 * (1 + torch.erf(dm / torch.sqrt(var) / math.sqrt(2.0)))
            tot = tot + (auc - 0.5) ** 2 / len(fams)
    return torch.autograd.grad(tot, th)[0].numpy().copy()


def main():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from rfa.fam import rfit as RF
    res = {'schema': 'RFA1-verifier-graph-check-v1', 'checked_file': 'rfa/fam/rfit.py',
           'rfit_sha256_at_start': hashlib.sha256(RFIT.read_bytes()).hexdigest(), 'checks': {}, 'numbers': {},
           'guard': vguard.selfcheck()}
    chk, num = res['checks'], res['numbers']
    rc = copy.deepcopy(RF.RECIPE_RF)                     # the REGISTERED recipe (warm start on, ridge 1e-2, adam critic)
    ridge = rc['ridge']
    # ---- 0. registered constants equal the verifier's reading of the docstring / recipe text
    num['recipe_constants'] = {k: rc[k] for k in ('soft_bound', 'ridge', 'rff_width', 'newton_tol', 'newton_target', 'cond_max',
                                                  'min_class_rows', 'altcrit_inner', 'altcrit_rounds', 'altcrit_lr', 'n_A', 'n_E',
                                                  'newton_warm_start')}
    REG = tuple(rc['views'])
    num['registered_views'] = list(REG)
    chk['recipe_constants_match_docstring'] = (rc['soft_bound'] == SOFT and rc['ridge'] == 1e-2 and rc['rff_width'] == 64 and
                                               rc['families'] == ['simple', 'rff'] and set(REG) <= set(VIEWS))
    # ---- 0b. feature / basis helpers vs the verifier's regeneration (bitwise RFF; max abs diff of features)
    rff_bitwise = all(np.array_equal(RF.rff_params(v, 64)[0].numpy(), RFF[v][0]) and
                      np.array_equal(RF.rff_params(v, 64)[1].numpy(), RFF[v][1]) for v in VIEWS)
    chk['rff_frequencies_bitwise_equal_regeneration'] = rff_bitwise
    num['rff_frequency_moments'] = {v: {'omega_var_times_d': float(RFF[v][0].var() * VIEW_DIM[v]),
                                        'b_min': float(RFF[v][1].min()), 'b_max': float(RFF[v][1].max())} for v in VIEWS}

    # ---- invented episode for the tiny release (household-separated A / E)
    X, s, unit = invented(420, 101)
    A, E = household_split(unit, 400, 400, 5)
    XA, XE, sA, sE = X[A], X[E], s[A], s[E]
    ziA, ziE = inc_const(XA), inc_const(XE)
    theta0 = np.random.default_rng(3).normal(size=N_THETA) * 0.5
    zA0, zE0 = (ziA, occ_np(theta0, XA)), (ziE, occ_np(theta0, XE))
    uA0 = v_views(*zA0)
    mom0 = {v: v_moments(uA0[v]) for v in VIEWS}
    yA, yE = sA.astype(np.float64), sE.astype(np.float64)
    J0, ws0, terms0, g0 = v_J(zA0, zE0, yA, yE, mom0, ridge, views=REG)
    num['episode'] = {'n_A': int(len(A)), 'n_E': int(len(E)), 'households_A': int(len(np.unique(unit[A]))),
                      'households_E': int(len(np.unique(unit[E]))), 'household_overlap': 0, 'theta_dim': N_THETA,
                      'income_logits': 'constant (not a function of theta)', 'J0': J0, 'terms0': terms0,
                      'verifier_newton_max_abs_grad': g0}
    # helper comparisons: module features on the same logits
    tA = {v: torch.from_numpy(uA0[v]) for v in VIEWS}
    feat_diff = 0.0
    for v in VIEWS:
        mo_t = RF.moments_of(tA[v])
        feat_diff = max(feat_diff, float(np.max(np.abs(mo_t[0].numpy() - mom0[v][0]))), float(np.max(np.abs(mo_t[1].numpy() - mom0[v][1]))))
        sa_t = RF.standardize(tA[v], mo_t)
        for f in FAMS:
            ph = RF.basis(sa_t, v, f).numpy()
            feat_diff = max(feat_diff, float(np.max(np.abs(ph - v_feats(v_std(uA0[v], mom0[v]), v, f)))))
    vi = RF.view_inputs(torch.from_numpy(zA0[0]), torch.from_numpy(zA0[1]))
    feat_diff = max(feat_diff, max(float(np.max(np.abs(vi[v].numpy() - uA0[v]))) for v in VIEWS))
    num['helpers_max_abs_diff_views_moments_standardize_basis'] = feat_diff
    chk['helpers_match_verifier_features'] = feat_diff < 1e-13
    # module Newton vs verifier Newton (solution agreement)
    wdiff, newt = 0.0, {}
    for v in REG:
        sa = torch.from_numpy(v_std(uA0[v], mom0[v]))
        for f in FAMS:
            ph = RF.basis(sa, v, f)
            kap = RF.ridge_vector(ph.shape[1])
            w_m, H_m, facts = RF.fit_reader(ph, torch.from_numpy(yA), kap)
            wdiff = max(wdiff, float(np.max(np.abs(w_m.numpy() - ws0[(v, f)]))))
            newt[f'{v}/{f}'] = {'module_newton_iterations': facts['newton_iterations'], 'module_max_abs_grad': facts['newton_max_abs_grad'],
                                'cond_H': facts['cond_H'], 'max_abs_w_diff_vs_verifier': float(np.max(np.abs(w_m.numpy() - ws0[(v, f)])))}
    num['inner_solve_comparison'] = newt
    chk['module_inner_solution_matches_verifier_newton'] = wdiff < 1e-8

    # ---- 1. forward value: module S (FULL and STOPFIT) vs verifier J(theta0)
    gF, infF, SF, _ = module_grad(RF, 'FULL', theta0, ziA, ziE, XA, XE, sA, sE, rc)
    gS, infS, SS, _ = module_grad(RF, 'STOPFIT', theta0, ziA, ziE, XA, XE, sA, sE, rc)
    term_diff = max(abs(infS['terms'][k]['R'] - terms0[k]) for k in terms0)
    num['forward'] = {'module_P_STOPFIT': infS['P'], 'module_S_FULL': SF, 'module_S_STOPFIT': SS, 'verifier_J0': J0,
                      'abs_diff_P_vs_J0': abs(infS['P'] - J0), 'max_abs_term_R_diff': term_diff,
                      'abs_S_FULL_minus_S_STOPFIT': abs(SF - SS),
                      'max_abs_fit_scalar_value': max(abs(t['fit_scalar_value']) for t in infF['terms'].values())}
    chk['forward_value_matches_verifier'] = abs(infS['P'] - J0) < 1e-12 and term_diff < 1e-12
    chk['full_scalar_value_equals_surrogate_to_solve_tolerance'] = abs(SF - SS) < 1e-10

    # ---- 2. FULL gradient vs central FD of the re-solved bilevel objective (moments frozen at theta0)
    def J_resolve(th):
        return v_J((ziA, occ_np(th, XA)), (ziE, occ_np(th, XE)), yA, yE, mom0, ridge, views=REG, w_warm=ws0)[0]

    def J_fixed(th):
        return v_J((ziA, occ_np(th, XA)), (ziE, occ_np(th, XE)), yA, yE, mom0, ridge, views=REG, w_fixed=ws0)[0]

    def J_moments(th):
        return v_J((ziA, occ_np(th, XA)), (ziE, occ_np(th, XE)), yA, yE, None, ridge, views=REG, w_warm=ws0)[0]
    t1 = time.process_time()
    fd_r = {h: fd(J_resolve, theta0, h) for h in (1e-4, 1e-5)}
    fd_f = {h: fd(J_fixed, theta0, h) for h in (1e-4, 1e-5)}
    rich_r = (4 * fd_r[1e-5] - fd_r[1e-4]) / 3            # not a true Richardson pair (ratio 10); reported only as a stability view
    g_ift = my_ift_grad(theta0, ziA, ziE, XA, XE, sA, sE, mom0, ws0, ridge, views=REG, fit_path=True)
    g_dir = my_ift_grad(theta0, ziA, ziE, XA, XE, sA, sE, mom0, ws0, ridge, views=REG, fit_path=False)
    num['fd_tiny_release'] = {
        'parameters': N_THETA, 'steps': [1e-4, 1e-5], 'cpu_s': time.process_time() - t1,
        'FULL_vs_FD_resolved': {str(h): {'rel_err_l2': rel(gF, fd_r[h]), 'max_abs_err': float(np.max(np.abs(gF - fd_r[h])))} for h in fd_r},
        'STOPFIT_vs_FD_resolved': {str(h): rel(gS, fd_r[h]) for h in fd_r},
        'STOPFIT_vs_FD_fixed_w': {str(h): {'rel_err_l2': rel(gS, fd_f[h]), 'max_abs_err': float(np.max(np.abs(gS - fd_f[h])))} for h in fd_f},
        'FULL_vs_FD_fixed_w': {str(h): rel(gF, fd_f[h]) for h in fd_f},
        'FD_step_consistency_resolved': rel(fd_r[1e-5], fd_r[1e-4]), 'FD_step_consistency_fixed': rel(fd_f[1e-5], fd_f[1e-4]),
        'verifier_IFT_vs_module_FULL_rel': rel(gF, g_ift), 'verifier_direct_vs_module_STOPFIT_rel': rel(gS, g_dir),
        'verifier_IFT_vs_FD_resolved_1e-5': rel(g_ift, fd_r[1e-5]), 'norm_FULL': float(np.linalg.norm(gF)),
        'norm_STOPFIT': float(np.linalg.norm(gS)), 'cos_FULL_STOPFIT': cos(gF, gS),
        'norm_fit_path_over_direct': float(np.linalg.norm(gF - gS) / np.linalg.norm(gS)), 'stability_view_rel': rel(gF, rich_r)}
    best_full = min(rel(gF, fd_r[h]) for h in fd_r)
    best_stop = min(rel(gS, fd_f[h]) for h in fd_f)
    chk['FULL_gradient_matches_FD_of_resolved_bilevel_J (rel < 1e-6)'] = best_full < 1e-6
    chk['STOPFIT_gradient_matches_FD_of_fixed_w_objective (rel < 1e-6)'] = best_stop < 1e-6
    chk['STOPFIT_is_not_the_bilevel_gradient (rel > 1e-2)'] = min(rel(gS, fd_r[h]) for h in fd_r) > 1e-2
    chk['verifier_IFT_equals_module_FULL (rel < 1e-10)'] = rel(gF, g_ift) < 1e-10
    chk['verifier_direct_equals_module_STOPFIT (rel < 1e-12)'] = rel(gS, g_dir) < 1e-12

    # ---- 2b. the same check with all three views (covers the income-view code path; constant income => zero income gradient)
    rc_all = copy.deepcopy(rc); rc_all['views'] = list(VIEWS)
    gF_all = module_grad(RF, 'FULL', theta0, ziA, ziE, XA, XE, sA, sE, rc_all)[0]
    J0_all, ws_all, _, _ = v_J(zA0, zE0, yA, yE, mom0, ridge, views=VIEWS)

    def J_all(th):
        return v_J((ziA, occ_np(th, XA)), (ziE, occ_np(th, XE)), yA, yE, mom0, ridge, views=VIEWS, w_warm=ws_all)[0]
    fd_all = fd(J_all, theta0, 1e-5)
    num['fd_tiny_release_all_three_views'] = {'FULL_vs_FD_resolved_rel': rel(gF_all, fd_all), 'J0': J0_all,
                                              'rel_diff_vs_registered_two_view_gradient': rel(gF_all, gF)}
    chk['FULL_matches_FD_with_all_three_views (rel < 1e-6)'] = rel(gF_all, fd_all) < 1e-6

    # ---- 3. the A-row moment path is detached (by design): its size, informational
    t1 = time.process_time()
    fd_m = fd(J_moments, theta0, 1e-5)
    num['moment_path_omitted_by_design'] = {'rel_FULL_vs_FD_with_moments_recomputed': rel(gF, fd_m),
                                            'moment_path_norm_over_fit_path_norm': float(np.linalg.norm(fd_m - gF) / np.linalg.norm(gF - gS)),
                                            'moment_path_norm_over_direct_norm': float(np.linalg.norm(fd_m - gF) / np.linalg.norm(gS)),
                                            'cos_FULL_vs_FD_with_moments_recomputed': cos(gF, fd_m),
                                            'cpu_s': time.process_time() - t1,
                                            'note': 'the module detaches mu_A / sd_A (docstring: constants of the step); FULL is the '
                                                    'gradient of J with frozen moments, not of the fully re-standardized objective'}

    # ---- 4. optimization sign along -grad FULL (true bilevel J re-solved; frozen moments) and with moments recomputed
    u = gF / np.linalg.norm(gF)
    steps = []
    for delta in (1e-4, 1e-3, 1e-2):
        Jn = J_resolve(theta0 - delta * u)
        Jm = J_moments(theta0 - delta * u)
        steps.append({'step_length': delta, 'J0': J0, 'J_after': Jn, 'decrease': J0 - Jn, 'first_order_prediction': delta * float(np.linalg.norm(gF)),
                      'ratio_actual_over_predicted': (J0 - Jn) / (delta * float(np.linalg.norm(gF))),
                      'decrease_with_moments_recomputed': J_moments(theta0) - Jm})
    num['optimization_sign'] = steps
    chk['optimization_sign_step_along_minus_grad_lowers_J'] = all(st['decrease'] > 0 for st in steps) and \
        abs(steps[0]['ratio_actual_over_predicted'] - 1) < 0.01

    # ---- 5. FULL vs STOPFIT where they should differ: small n_A, RFF, pair view, misspecified (variance) signal
    X2, s2, unit2 = invented(40, 202, signal=0.15, nonlinear=0.35)
    X3, s3, unit3 = invented(900, 203, signal=0.15, nonlinear=0.35)
    A2 = np.arange(len(s2)); E3 = np.arange(len(s3))           # different invented households (disjoint by construction)
    rc5 = copy.deepcopy(rc); rc5['families'] = ['rff']; rc5['views'] = ['pair']
    zi2, zi3 = inc_const(X2), inc_const(X3)
    gF5 = module_grad(RF, 'FULL', theta0, zi2, zi3, X2, X3, s2, s3, rc5)[0]
    gS5 = module_grad(RF, 'STOPFIT', theta0, zi2, zi3, X2, X3, s2, s3, rc5)[0]
    mom5 = {'pair': v_moments(v_views(zi2, occ_np(theta0, X2))['pair'])}
    _, ws5, _, _ = v_J((zi2, occ_np(theta0, X2)), (zi3, occ_np(theta0, X3)), s2.astype(float), s3.astype(float), mom5, ridge,
                       views=('pair',), fams=('rff',))

    def J5(th):
        return v_J((zi2, occ_np(th, X2)), (zi3, occ_np(th, X3)), s2.astype(float), s3.astype(float), mom5, ridge, views=('pair',),
                   fams=('rff',), w_warm=ws5)[0]
    fd5 = fd(J5, theta0, 1e-5)
    num['should_differ_case'] = {'n_A': int(len(A2)), 'n_E': int(len(E3)), 'family': 'rff', 'view': 'pair', 'cos_FULL_STOPFIT': cos(gF5, gS5),
                                 'norm_FULL_over_STOPFIT': float(np.linalg.norm(gF5) / np.linalg.norm(gS5)),
                                 'norm_fit_path_over_direct': float(np.linalg.norm(gF5 - gS5) / np.linalg.norm(gS5)),
                                 'FULL_vs_FD_resolved_rel': rel(gF5, fd5), 'STOPFIT_vs_FD_resolved_rel': rel(gS5, fd5)}
    chk['FULL_and_STOPFIT_differ_materially_in_small_nA_rff_case (fit/direct > 0.3)'] = \
        num['should_differ_case']['norm_fit_path_over_direct'] > 0.3
    chk['FULL_matches_FD_in_small_nA_rff_case (rel < 1e-6)'] = rel(gF5, fd5) < 1e-6

    # ---- 6. ALTCRIT: first call = STOPFIT bitwise; second call (10 Adam rounds) direct path vs FD with critic frozen; Adam replicated
    A_e1, E_e1 = A, E
    A_e2, E_e2 = household_split(unit, 400, 400, 6)
    alt = RF.EpisodeObjective('ALTCRIT', 0, rc)
    gA1, _, _, alt = module_grad(RF, 'ALTCRIT', theta0, ziA, ziE, XA, XE, sA, sE, rc, obj=alt)
    critic1 = {k: alt.critic[k].detach().numpy().copy() for k in alt.critic}
    chk['ALTCRIT_first_call_equals_STOPFIT_bitwise'] = bool(np.array_equal(gA1, gS))
    XA2, XE2, sA2, sE2 = X[A_e2], X[E_e2], s[A_e2], s[E_e2]
    ziA2, ziE2 = inc_const(XA2), inc_const(XE2)
    gA2, infA2, _, alt = module_grad(RF, 'ALTCRIT', theta0, ziA2, ziE2, XA2, XE2, sA2, sE2, rc, obj=alt)
    critic2 = {k: alt.critic[k].detach().numpy().copy() for k in alt.critic}
    uA2 = v_views(ziA2, occ_np(theta0, XA2))
    mom2 = {v: v_moments(uA2[v]) for v in VIEWS}
    # verifier's own Adam (torch's update order), 10 rounds from critic1 on episode 2's A features
    adam_diff, lag = 0.0, {}
    for v in REG:
        sa2 = v_std(uA2[v], mom2[v])
        for f in FAMS:
            ph = v_feats(sa2, v, f); kap = v_kap(ph.shape[1], ridge)
            w = critic1[(v, f)].copy(); m1 = np.zeros_like(w); m2 = np.zeros_like(w)
            for t in range(1, rc['altcrit_rounds'] + 1):
                g = v_grad_inner(ph, sA2.astype(float), w, kap)
                m1 = 0.9 * m1 + 0.1 * g
                m2 = 0.999 * m2 + 0.001 * g * g
                denom = np.sqrt(m2) / math.sqrt(1 - 0.999 ** t) + 1e-8
                w = w - (rc['altcrit_lr'] / (1 - 0.9 ** t)) * m1 / denom
            adam_diff = max(adam_diff, float(np.max(np.abs(w - critic2[(v, f)]))))
            wstar2 = v_newton(ph, sA2.astype(float), kap)[0]
            lag[f'{v}/{f}'] = {'critic_max_abs_grad_A2': float(np.max(np.abs(v_grad_inner(ph, sA2.astype(float), critic2[(v, f)], kap)))),
                               'max_abs_critic_minus_exact_fit_A2': float(np.max(np.abs(critic2[(v, f)] - wstar2))),
                               'verifier_rel_lag': float(np.linalg.norm(critic2[(v, f)] - wstar2) / np.linalg.norm(wstar2))}

    def J_alt(th):
        return v_J((ziA2, occ_np(th, XA2)), (ziE2, occ_np(th, XE2)), sA2.astype(float), sE2.astype(float), mom2, ridge, views=REG,
                   w_fixed=critic2)[0]
    fd_alt = fd(J_alt, theta0, 1e-5)
    gS_e2 = module_grad(RF, 'STOPFIT', theta0, ziA2, ziE2, XA2, XE2, sA2, sE2, rc)[0]
    num['altcrit'] = {'first_call_max_abs_diff_vs_STOPFIT': float(np.max(np.abs(gA1 - gS))),
                      'second_call_vs_FD_critic_frozen_rel': rel(gA2, fd_alt),
                      'second_call_max_abs_err': float(np.max(np.abs(gA2 - fd_alt))),
                      'verifier_adam_replication_max_abs_diff': adam_diff, 'lag_after_10_rounds': lag,
                      'second_call_cos_vs_STOPFIT_same_episode': cos(gA2, gS_e2),
                      'second_call_rel_diff_vs_STOPFIT_same_episode': rel(gA2, gS_e2),
                      'module_reported_critic_max_abs_grad_A': {k: t.get('critic_max_abs_grad_A') for k, t in infA2['terms'].items()},
                      'module_reported_critic_rel_lag': {k: t.get('critic_rel_lag') for k, t in infA2['terms'].items()}}
    chk['ALTCRIT_direct_path_matches_FD_with_critic_frozen (rel < 1e-6)'] = rel(gA2, fd_alt) < 1e-6
    lag_diff = max(abs(lag[k]['verifier_rel_lag'] - infA2['terms'][k]['critic_rel_lag']) for k in lag)
    num['altcrit']['critic_rel_lag_module_vs_verifier_max_abs_diff'] = lag_diff
    chk['ALTCRIT_lag_diagnostic_matches_verifier (abs < 1e-9)'] = lag_diff < 1e-9
    chk['ALTCRIT_critic_update_is_10_Adam_rounds_from_previous_critic (max abs < 1e-12)'] = adam_diff < 1e-12
    chk['ALTCRIT_adam_critic_lags_the_exact_fit (not an alias)'] = rel(gA2, gS_e2) > 1e-3

    # ---- 7. alias claim: ALTCRIT with altcrit_inner = 'newton' == STOPFIT (bitwise without warm start; tolerance with it)
    episodes = [(A, E), (A_e2, E_e2), household_split(unit, 400, 400, 7)]
    th_seq = [theta0, theta0 - 1e-2 * u, theta0 - 2e-2 * u]
    def seq(mode, recipe):
        obj = RF.EpisodeObjective(mode, 0, recipe)
        out = []
        for (a_, e_), th in zip(episodes, th_seq):
            out.append(module_grad(RF, mode, th, inc_const(X[a_]), inc_const(X[e_]), X[a_], X[e_], s[a_], s[e_], recipe, obj=obj)[0])
        return out
    rc_nw = copy.deepcopy(rc); rc_nw['newton_warm_start'] = False
    rc_alt_newton = copy.deepcopy(rc); rc_alt_newton['altcrit_inner'] = 'newton'
    rc_alt_newton_nw = copy.deepcopy(rc_nw); rc_alt_newton_nw['altcrit_inner'] = 'newton'
    s_nw = seq('STOPFIT', rc_nw); a_nw = seq('ALTCRIT', rc_alt_newton_nw)
    s_w = seq('STOPFIT', rc); a_w = seq('ALTCRIT', rc_alt_newton)
    num['alias'] = {'no_warm_start_max_abs_diff_per_call': [float(np.max(np.abs(x - y))) for x, y in zip(s_nw, a_nw)],
                    'registered_warm_start_rel_diff_per_call': [rel(y, x) for x, y in zip(s_w, a_w)],
                    'registered_warm_start_max_abs_diff_per_call': [float(np.max(np.abs(x - y))) for x, y in zip(s_w, a_w)]}
    chk['alias_ALTCRIT_newton_equals_STOPFIT_bitwise_without_warm_start'] = all(np.array_equal(x, y) for x, y in zip(s_nw, a_nw))
    chk['alias_ALTCRIT_newton_equals_STOPFIT_to_solve_tolerance_with_registered_warm_start (rel < 1e-8)'] = \
        all(rel(y, x) < 1e-8 for x, y in zip(s_w, a_w))
    num['alias']['bitwise_with_registered_warm_start'] = all(np.array_equal(x, y) for x, y in zip(s_w, a_w))

    # ---- 8. U0-size float64 networks (d = 145, 64-64-16-K): directional derivatives vs FD of the verifier's numpy J
    from acs_replication import neural
    torch.manual_seed(11)
    net_o = neural.network(5, 145).double()
    net_i = neural.network(2, 145).double()
    rngb = np.random.default_rng(12)
    nb_house = 700
    sizes = rngb.integers(1, 4, nb_house); ub = np.repeat(np.arange(nb_house), sizes)
    sb = rngb.integers(0, 2, len(ub)).astype(np.int64)
    Xb = rngb.normal(size=(len(ub), 145)); Xb[:, :10] += 0.4 * (2 * sb[:, None] - 1)
    Ab, Eb = household_split(ub, 512, 512, 13)
    with torch.no_grad():
        zi_b = net_i(torch.from_numpy(Xb)).numpy(); zi_b = zi_b - zi_b.mean(1, keepdims=True)
    lay = [(net_o[i].weight.detach().numpy().copy(), net_o[i].bias.detach().numpy().copy()) for i in (0, 2, 4, 5)]

    def fwd_np(layers, Xr):
        h = Xr
        for j, (w_, b_) in enumerate(layers):
            h = h @ w_.T + b_
            if j < 2:
                h = np.maximum(h, 0.0)
        return h - h.mean(1, keepdims=True)
    zoA, zoE = fwd_np(lay, Xb[Ab]), fwd_np(lay, Xb[Eb])
    momb = {v: v_moments(v_views(zi_b[Ab], zoA)[v]) for v in VIEWS}
    yAb, yEb = sb[Ab].astype(float), sb[Eb].astype(float)
    Jb0, wsb, _, _ = v_J((zi_b[Ab], zoA), (zi_b[Eb], zoE), yAb, yEb, momb, ridge, views=REG)
    params = list(net_o.parameters())
    gb = {}
    for mode in ('FULL', 'STOPFIT'):
        zA_t = net_o(torch.from_numpy(Xb[Ab])); zA_t = zA_t - zA_t.mean(1, keepdim=True)
        zE_t = net_o(torch.from_numpy(Xb[Eb])); zE_t = zE_t - zE_t.mean(1, keepdim=True)
        S_b, _ = RF.EpisodeObjective(mode, 0, rc).objective(torch.from_numpy(zi_b[Ab]), zA_t, torch.from_numpy(sb[Ab]),
                                                             torch.from_numpy(zi_b[Eb]), zE_t, torch.from_numpy(sb[Eb]))
        gb[mode] = [x.detach().numpy().copy() for x in torch.autograd.grad(S_b, params)]
    pidx = {id(p): i for i, p in enumerate(params)}
    lay_idx = [(pidx[id(net_o[i].weight)], pidx[id(net_o[i].bias)]) for i in (0, 2, 4, 5)]
    dirs = []
    for k in range(3):
        rg = np.random.default_rng(300 + k)
        dv = [rg.normal(size=p.shape) for p in params]
        nr = math.sqrt(sum(float((x ** 2).sum()) for x in dv)); dv = [x / nr for x in dv]

        def lay_at(hh):
            return [(lay[j][0] + hh * dv[lay_idx[j][0]], lay[j][1] + hh * dv[lay_idx[j][1]]) for j in range(4)]

        def Jd(hh, fixed=False):
            la = lay_at(hh)
            return v_J((zi_b[Ab], fwd_np(la, Xb[Ab])), (zi_b[Eb], fwd_np(la, Xb[Eb])), yAb, yEb, momb, ridge, views=REG,
                       w_fixed=wsb if fixed else None, w_warm=None if fixed else wsb)[0]
        h = 1e-5
        fdv = (Jd(h) - Jd(-h)) / (2 * h)
        fdv_fixed = (Jd(h, True) - Jd(-h, True)) / (2 * h)
        an_full = sum(float((a * b).sum()) for a, b in zip(gb['FULL'], dv))
        an_stop = sum(float((a * b).sum()) for a, b in zip(gb['STOPFIT'], dv))
        dirs.append({'FULL_analytic': an_full, 'FD_resolved': fdv, 'FULL_rel_err': abs(an_full - fdv) / abs(fdv),
                     'STOPFIT_analytic': an_stop, 'FD_fixed_w': fdv_fixed, 'STOPFIT_rel_err': abs(an_stop - fdv_fixed) / abs(fdv_fixed)})
    gfl = np.concatenate([x.ravel() for x in gb['FULL']]); gst = np.concatenate([x.ravel() for x in gb['STOPFIT']])
    num['u0_size_float64'] = {'n_A': int(len(Ab)), 'n_E': int(len(Eb)), 'parameters': int(sum(p.numel() for p in params)), 'J0': Jb0,
                              'directions': dirs, 'cos_FULL_STOPFIT': cos(gfl, gst),
                              'norm_fit_path_over_direct': float(np.linalg.norm(gfl - gst) / np.linalg.norm(gst))}
    chk['u0_size_FULL_directional_derivatives_match_FD (rel < 1e-5)'] = all(d['FULL_rel_err'] < 1e-5 for d in dirs)
    chk['u0_size_STOPFIT_directional_derivatives_match_FD (rel < 1e-5)'] = all(d['STOPFIT_rel_err'] < 1e-5 for d in dirs)

    # ---- 9. income view, SIMPLE family, theta-dependent income: fit path exactly zero (scale / shift invariance of AUC_b)
    th_i = np.random.default_rng(21).normal(size=D_IN * 2 + 2) * 0.5

    def inc_t(th, Xt):
        z = Xt @ th[:D_IN * 2].reshape(D_IN, 2) + th[D_IN * 2:]
        return z - z.mean(1, keepdim=True)
    rc9 = copy.deepcopy(rc); rc9['families'] = ['simple']; rc9['views'] = ['income']
    g9 = {}
    for mode in ('FULL', 'STOPFIT'):
        th = torch.tensor(th_i, requires_grad=True)
        S9, _ = RF.EpisodeObjective(mode, 0, rc9).objective(inc_t(th, torch.from_numpy(XA)), torch.from_numpy(zA0[1]), torch.from_numpy(sA),
                                                            inc_t(th, torch.from_numpy(XE)), torch.from_numpy(zE0[1]), torch.from_numpy(sE))
        g9[mode] = torch.autograd.grad(S9, th)[0].numpy()
    num['income_simple_exact_zero'] = {'norm_fit_path_over_direct': float(np.linalg.norm(g9['FULL'] - g9['STOPFIT']) / np.linalg.norm(g9['STOPFIT']))}
    chk['income_simple_fit_path_zero (rel < 1e-8)'] = num['income_simple_exact_zero']['norm_fit_path_over_direct'] < 1e-8

    # ---- 10. no A path in STOPFIT / ALTCRIT; an A path in FULL (gradients w.r.t. the A-row occupation logits)
    gA_paths = {}
    for mode in ('FULL', 'STOPFIT', 'ALTCRIT'):
        zAo = torch.tensor(zA0[1], requires_grad=True); zEo = torch.tensor(zE0[1], requires_grad=True)
        S10, _ = RF.EpisodeObjective(mode, 0, rc).objective(torch.from_numpy(ziA), zAo, torch.from_numpy(sA), torch.from_numpy(ziE), zEo,
                                                            torch.from_numpy(sE))
        ga, ge = torch.autograd.grad(S10, [zAo, zEo], allow_unused=True)
        gA_paths[mode] = {'norm_grad_A_outputs': 0.0 if ga is None else float(ga.norm()), 'norm_grad_E_outputs': 0.0 if ge is None else float(ge.norm())}
    num['output_paths'] = gA_paths
    chk['A_outputs_reach_only_FULL'] = (gA_paths['FULL']['norm_grad_A_outputs'] > 0 and gA_paths['STOPFIT']['norm_grad_A_outputs'] == 0.0 and
                                        gA_paths['ALTCRIT']['norm_grad_A_outputs'] == 0.0)

    # ---- 11. campaign-size invented episode (registered recipe, n_A = n_E = 2048, U0-size float64 nets): how different FULL is
    rngc = np.random.default_rng(31)
    nh = 3400
    sz = rngc.integers(1, 4, nh); uc = np.repeat(np.arange(nh), sz)
    sc = rngc.integers(0, 2, len(uc)).astype(np.int64)
    Xc = rngc.normal(size=(len(uc), 145)); Xc[:, :10] += 0.3 * (2 * sc[:, None] - 1); Xc[:, 10:13] *= np.exp(0.2 * (2 * sc[:, None] - 1))
    Ac, Ec = household_split(uc, rc['n_A'], rc['n_E'], 32)
    with torch.no_grad():
        zic = net_i(torch.from_numpy(Xc)); zic = (zic - zic.mean(1, keepdim=True))
    gc = {}
    for mode in ('FULL', 'STOPFIT'):
        zA_t = net_o(torch.from_numpy(Xc[Ac])); zA_t = zA_t - zA_t.mean(1, keepdim=True)
        zE_t = net_o(torch.from_numpy(Xc[Ec])); zE_t = zE_t - zE_t.mean(1, keepdim=True)
        S_c, inf_c = RF.EpisodeObjective(mode, 0, rc).objective(zic[Ac], zA_t, torch.from_numpy(sc[Ac]), zic[Ec], zE_t, torch.from_numpy(sc[Ec]))
        gc[mode] = np.concatenate([x.detach().numpy().ravel() for x in torch.autograd.grad(S_c, params)])
    num['campaign_size_invented'] = {'n_A': int(len(Ac)), 'n_E': int(len(Ec)), 'cos_FULL_STOPFIT': cos(gc['FULL'], gc['STOPFIT']),
                                     'norm_fit_path_over_direct': float(np.linalg.norm(gc['FULL'] - gc['STOPFIT']) / np.linalg.norm(gc['STOPFIT']))}

    res['rfit_sha256_at_end'] = hashlib.sha256(RFIT.read_bytes()).hexdigest()
    res['rfit_unchanged_during_check'] = res['rfit_sha256_at_end'] == res['rfit_sha256_at_start']
    res['guard_refused_opens'] = list(vguard.REFUSED)
    res['all_pass'] = all(bool(v) for v in chk.values()) and all(res['guard'].values())
    res['cpu_s'] = time.process_time() - T0
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'checks': chk, 'cpu_s': res['cpu_s']}, indent=1, default=float))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

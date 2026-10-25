"""RFA1 reader-fit-aware release objective: held-out recovery of a FITTED auxiliary-data reader, differentiated through the fit.

PCRL campaign `pcrl_reader_fit_aware_release_v1` (code package `rfa/`); mechanism investigator draft, 2026-10-25. Label-free code:
this module never opens data; its checks use invented numbers only.

Release wire (unchanged contract): recipient r releases softmax(centred float64 logits z_r) and argmax decision. The readers here see
the logit differences, a smooth sufficient parametrization of the wire (income u = z1 - z0; occupation u_k = z_k - z0, k = 1..4;
pair = both, 5 coordinates).

One episode (inside MODEL_FIT, per release step): household-separated rows A (auxiliary training) and E (auxiliary evaluation),
supplied by the training loop. For every view v and reader family f:
    s      = c tanh(((u - mu_A) / sd_A) / c), c = 6          mu_A, sd_A: DETACHED moments of the A rows (constants of the step)
    phi_f  = SIMPLE [s, 1]  |  RFF [s, sqrt2 cos(s Omega_v + b_v) (64 frozen random frequencies, Omega_v ~ N(0, I / d_v)), 1]
    w*     = argmin_w L_A(w, theta) = mean_A BCE(phi w, SEX) + kappa/2 ||w_(-intercept)||^2      (Newton, float64, detached)
    t_E    = phi_f(E) w*                                     held-out attack score, oriented by A (higher = SEX 1)
    AUC_b  = Phi(dmu / sqrt(var_1 + var_0)) of t_E           binormal smooth AUC: exactly invariant to scale and shift of t (and w)
    R_vf   = (AUC_b - 1/2)^2                                 symmetric: an inverted held-out score is not rewarded
S = sum_v mean_f R_vf (to DECREASE). Derivatives with respect to the release parameters theta:
    direct path  dR/dtheta at fixed w* (through the E outputs only)
    fit path     -(d/dtheta)[ v . grad_w L_A(w*, theta) ],  v = H^-1 grad_w R,  H = Hessian of L_A at w*   (implicit function
                 theorem; v and w* detached; flows through the A outputs only)
    FULL = direct + fit;  STOPFIT = direct only (identical everything else);  ALTCRIT = direct path through a persistent lagging
    reader (same family, features, loss, ridge) updated by `altcrit_rounds` full-batch Adam steps on each episode's A rows before the
    release gradient. ALTCRIT with altcrit_inner = 'newton' re-solves exactly: w = w*(A_t, theta_t), an EXACT ALIAS of STOPFIT.
Guards: non-finite logits / features / surrogate, degenerate A moments, < min_class_rows of a SEX class, Newton not converged to
newton_tol, Hessian condition number > cond_max, degenerate held-out score variance -> ReaderRefused (a FloatingPointError): the
training loop records it and stops; refusal is never a reward.

Commands:
    <repository>/.venv/bin/python -m rfa.fam.rfit selftest [--out PATH]
    <repository>/.venv/bin/python -m rfa.fam.rfit bench [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

VIEWS = ('income', 'occupation', 'pair')
VIEW_DIM = {'income': 1, 'occupation': 4, 'pair': 5}
MODES = ('FULL', 'STOPFIT', 'ALTCRIT')
RECIPE_RF = {
    'schema': 'RFA1-reader-fit-recipe-v1',
    'n_A': 4096, 'n_E': 2400,
    'episode': {'n_A': 4096, 'n_E': 2400, 'episodes_per_step': 1, 'separation': 'household (no unit in both A and E)',
                'pool': 'the training rows of the unit (fold training rows / all MODEL_FIT rows for the deployed fit)',
                'sizes_rationale': 'n_A ~ the declared recipient auxiliary FIT sample (~4,000, MODEL_SELECT FIT parts 0-4), so the fit '
                                   'path describes a reader with that much auxiliary data; n_E ~ its SCORE part (~2,400)'},
    'views': ['occupation', 'pair'],
    'views_rationale': 'the loop trains the occupation network only; income logits are START constants, so the income-view term has '
                       'no gradient and is skipped; the pair view keeps the constant income coordinate',
    'view_inputs': 'income: z1 - z0 (1); occupation: z_k - z0, k = 1..4 (4); pair: both (5); z = centred float64 logits',
    'standardization': 'per view: (u - mean_A) / sd_A with the DETACHED A-row mean and population SD; then c tanh(s / c)',
    'moments_convention': 'mu_A and sd_A are detached constants of each call, so the FULL fit path runs through w* only (no '
                          'derivative through the standardization); STOPFIT and ALTCRIT share the convention (as FB_PFD poly2)',
    'soft_bound': 6.0,
    'families': ['simple', 'rff'],
    'family_features': {'simple': '[s, 1]',
                        'rff': '[s, sqrt(2) cos(s Omega_v + b_v), 1]; Omega_v ~ N(0, 1 / d_v) (d_v x 64), b_v ~ U(0, 2 pi)'},
    'rff_width': 64,
    'rff_seed': 'torch.Generator().manual_seed(seed_for("RFA1-rff", view) mod 2^62); frozen, identical in every version / fold',
    'ridge': 1e-2,
    'inner_loss': 'mean binary cross-entropy of SEX + ridge / 2 * ||w||^2 over all non-intercept weights (intercept unpenalized)',
    'newton': 'from w = 0 (or the previous call\'s w* when newton_warm_start), Armijo backtracking (c = 1e-4, halving) while max '
              '|grad| >= 1e-5, full steps below; stop at max |grad| <= newton_target or when it stalls below newton_tol',
    'newton_target': 1e-13, 'newton_tol': 1e-10, 'newton_max_iter': 60, 'newton_warm_start': True,
    'cond_max': 1e8, 'min_class_rows': 16, 'min_sd': 1e-8,
    'surrogate': 'S = sum over views of the MEAN over families of (AUC_b - 1/2)^2; AUC_b = Phi(dmu / sqrt(var_1 + var_0)) of the '
                 'held-out reader score t_E = phi(E) w*; population variances, NOT detached (no temperature; exactly scale- and '
                 'shift-invariant in t)',
    'altcrit_inner': 'adam', 'altcrit_rounds': 10, 'altcrit_lr': 3e-3,
    'altcrit_init': 'exact fit on the first call\'s A rows (a converged warm critic at START)',
    'altcrit_order': 'per call: critic update on this episode\'s A rows at the current release, then the release gradient through E',
    'diagnostics': 'per term R, AUC_b (E), orientation sign of dmu (E), exact Mann-Whitney AUC (E and A), Newton iterations, final '
                   'max |grad|, cond(H); FULL: ||grad_w R||, ||v||, output-space norms of the direct (E outputs) and fit (A outputs) '
                   'contributions',
}
SQRT2 = math.sqrt(2.0)


class ReaderRefused(FloatingPointError):
    """A guard failed (non-finite value, failed inner solve, degenerate episode): the loop records it and stops."""


def seed_for(*items):
    return int(hashlib.sha256('|'.join(map(str, items)).encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------- frozen reader feature maps
_RFF = {}


def rff_params(view, width):
    import torch
    key = (view, int(width))
    if key not in _RFF:
        g = torch.Generator().manual_seed(seed_for('RFA1-rff', view) % 2 ** 62)
        d = VIEW_DIM[view]
        om = torch.randn(d, int(width), generator=g, dtype=torch.float64) / math.sqrt(d)
        b = torch.rand(int(width), generator=g, dtype=torch.float64) * (2 * math.pi)
        _RFF[key] = (om, b)
    return _RFF[key]


def view_inputs(z_inc, z_occ):
    import torch
    li = (z_inc[:, 1] - z_inc[:, 0])[:, None]
    lo = z_occ[:, 1:] - z_occ[:, :1]
    return {'income': li, 'occupation': lo, 'pair': torch.cat([li, lo], 1)}


def moments_of(u_a, rc=RECIPE_RF):
    import torch
    u = u_a.detach()
    mu, sd = u.mean(0), u.std(0, unbiased=False)
    if not (bool(torch.isfinite(mu).all()) and bool(torch.isfinite(sd).all())) or float(sd.min()) < rc['min_sd']:
        raise ReaderRefused('degenerate or non-finite A-row moments')
    return mu, sd


def standardize(u, mom, rc=RECIPE_RF):
    import torch
    c = rc['soft_bound']
    return c * torch.tanh(((u - mom[0]) / mom[1]) / c)


def basis(s, view, family, rc=RECIPE_RF):
    import torch
    one = torch.ones(s.shape[0], 1, dtype=s.dtype)
    if family == 'simple':
        return torch.cat([s, one], 1)
    if family == 'rff':
        om, b = rff_params(view, rc['rff_width'])
        return torch.cat([s, SQRT2 * torch.cos(s @ om + b), one], 1)
    raise ValueError(f'unregistered reader family {family!r}')


def ridge_vector(p, rc=RECIPE_RF):
    import torch
    k = torch.full((p,), rc['ridge'], dtype=torch.float64)
    k[-1] = 0.0
    return k


def inner_loss(phi, y, w, kap):
    import torch.nn.functional as F
    t = phi @ w
    return (F.softplus(t) - y * t).mean() + 0.5 * (kap * w * w).sum()


def fit_reader(phi, y, kap, rc=RECIPE_RF, w0=None):
    """Exact ridge-logistic fit (Newton, float64, detached). Returns w*, the Hessian at w*, and solver facts; refuses on failure."""
    import torch
    phi, y = phi.detach(), y.detach()
    if not bool(torch.isfinite(phi).all()):
        raise ReaderRefused('non-finite reader features')
    n1, n = int(y.sum()), len(y)
    if min(n1, n - n1) < rc['min_class_rows']:
        raise ReaderRefused('too few rows of a SEX class in A')
    w = torch.zeros(phi.shape[1], dtype=torch.float64) if w0 is None else w0.detach().clone()
    L = float(inner_loss(phi, y, w, kap))
    prev, it = float('inf'), 0
    for it in range(1, rc['newton_max_iter'] + 1):
        q = torch.sigmoid(phi @ w)
        g = phi.T @ (q - y) / n + kap * w
        gmax = float(g.abs().max())
        if gmax <= rc['newton_target'] or (gmax <= rc['newton_tol'] and gmax > 0.25 * prev):
            break
        prev = gmax
        H = (phi.T * (q * (1 - q))) @ phi / n + torch.diag(kap)
        step = torch.linalg.solve(H, g)
        if gmax < 1e-5:                         # local quadratic convergence: full Newton steps
            w = w - step
            continue
        a, dec = 1.0, float(g @ step)
        while True:
            w_new = w - a * step
            L_new = float(inner_loss(phi, y, w_new, kap))
            if L_new <= L - 1e-4 * a * dec or a < 1e-10:
                break
            a *= 0.5
        w, L = w_new, L_new
    q = torch.sigmoid(phi @ w)
    g = phi.T @ (q - y) / n + kap * w
    gmax = float(g.abs().max())
    H = (phi.T * (q * (1 - q))) @ phi / n + torch.diag(kap)
    ev = torch.linalg.eigvalsh(H)
    cond = float(ev[-1] / ev[0]) if float(ev[0]) > 0 else float('inf')
    if not (math.isfinite(gmax) and gmax <= rc['newton_tol']):
        raise ReaderRefused(f'Newton not converged (max |grad| {gmax:.3e} after {it} iterations)')
    if not (math.isfinite(cond) and cond <= rc['cond_max']):
        raise ReaderRefused(f'reader Hessian condition number {cond:.3e} above the registered maximum')
    return w, H, {'newton_iterations': it, 'newton_max_abs_grad': gmax, 'cond_H': cond}


def binormal_auc(t, m):
    """Binormal smooth AUC of score t between m (SEX 1) and ~m; exactly invariant to positive scale and shift of t."""
    import torch
    a, b = t[m], t[~m]
    dm = a.mean() - b.mean()
    v = ((a - a.mean()) ** 2).mean() + ((b - b.mean()) ** 2).mean()
    if not (bool(torch.isfinite(v.detach())) and float(v.detach()) > 1e-24 * (1.0 + float(dm.detach()) ** 2)):
        raise ReaderRefused('degenerate held-out score variance')
    return 0.5 * (1.0 + torch.erf(dm / torch.sqrt(v) / SQRT2)), dm


def exact_auc(t, y):
    """Mann-Whitney AUC (average ranks for ties) of a detached score: orientation and exploit monitor (not differentiated)."""
    t = np.asarray(t, dtype=np.float64); y = np.asarray(y).astype(bool)
    _, inv, cnt = np.unique(t, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- the episode objective (the training loop's interface)
class EpisodeObjective:
    """mode 'FULL' | 'STOPFIT' | 'ALTCRIT'. objective(zA_inc, zA_occ, sA, zE_inc, zE_occ, sE) -> (S, info): S is a scalar to DECREASE
    whose autograd gradient w.r.t. the network parameters is the update direction of the mode; its value is the surrogate (the FULL
    fit-path scalar adds -v . grad_w L_A(w*), which is 0 up to the solve tolerance). info holds plain floats. `seed` is recorded only:
    every random element (the RFF frequencies) is frozen in the recipe, identical in every version and fold."""

    def __init__(self, mode, seed=0, recipe=RECIPE_RF):
        if mode not in MODES:
            raise ValueError(f'unregistered mode {mode!r}')
        if recipe['altcrit_inner'] not in ('adam', 'newton'):
            raise ValueError('altcrit_inner must be adam or newton')
        self.mode, self.seed, self.rc = mode, int(seed), recipe
        self.families, self.views = tuple(recipe['families']), tuple(recipe['views'])
        self.w_prev = {}                         # FULL / STOPFIT: Newton warm starts (the solution is unique; exactness unchanged)
        self.critic, self.critic_opt, self.calls = {}, {}, 0

    def _critic_update(self, key, phiA, yA, kap):
        import torch
        rc = self.rc
        if key not in self.critic or rc['altcrit_inner'] == 'newton':
            self.critic[key] = fit_reader(phiA, yA, kap, rc, None)[0].clone().requires_grad_(True)
            self.critic_opt[key] = torch.optim.Adam([self.critic[key]], lr=rc['altcrit_lr'])
            rounds = 0
        else:
            for _ in range(rc['altcrit_rounds']):
                self.critic_opt[key].zero_grad(set_to_none=True)
                loss = inner_loss(phiA, yA, self.critic[key], kap)
                if not bool(torch.isfinite(loss)):
                    raise ReaderRefused('non-finite alternating-critic loss')
                loss.backward()
                self.critic_opt[key].step()
            rounds = rc['altcrit_rounds']
        with torch.no_grad():
            wc = self.critic[key].detach()
            g = phiA.T @ (torch.sigmoid(phiA @ wc) - yA) / len(yA) + kap * wc
            # lag diagnostic only (never in the gradient): the exact solve on the same A rows at the same release
            wx = fit_reader(phiA, yA, kap, rc, self.w_prev.get(key) if rc['newton_warm_start'] else None)[0]
            self.w_prev[key] = wx
        return {'critic_rounds': rounds, 'critic_max_abs_grad_A': float(g.abs().max()),
                'critic_rel_lag': float((wc - wx).norm() / wx.norm())}

    def objective(self, zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments=None, diagnostics=True):
        """(S, info), S = S_dir + S_fit of objective_parts (S_fit = 0 for STOPFIT and ALTCRIT)."""
        S_dir, S_fit, info = self.objective_parts(zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments, diagnostics)
        S = S_dir + S_fit
        info['S'] = float(S.detach())
        return S, info

    def objective_parts(self, zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments=None, diagnostics=True):
        """(S_dir, S_fit, info). S_dir = sum_v mean_f R(phi_E(theta); w const) is exactly STOPFIT's objective at this call's w*
        (FULL / STOPFIT) or at the critic's w (ALTCRIT); S_fit = sum_v mean_f [-v . grad_w L_A(w*, theta)] (FULL) carries the fit
        path through the A outputs and has value ~0; for STOPFIT / ALTCRIT S_fit = 0 * S_dir (graph-connected, zero gradient).
        moments: optional {view: (mean, sd)} constants (the finite-difference checks freeze them); default: the detached A-row
        moments of this call."""
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
        S_dir = torch.zeros((), dtype=torch.float64)
        S_fit = torch.zeros((), dtype=torch.float64)
        info = {'mode': mode, 'refused': False, 'terms': {}}
        nf = len(self.families)
        for v in self.views:
            mom = moments[v] if moments is not None else moments_of(uA[v], rc)
            sa, se = standardize(uA[v], mom, rc), standardize(uE[v], mom, rc)
            for f in self.families:
                key = (v, f)
                phiA, phiE = basis(sa, v, f, rc), basis(se, v, f, rc)
                kap = ridge_vector(phiA.shape[1], rc)
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


def _poly2(s):
    """Degree-2 polynomial features [s, s_i s_j (i <= j)] (FB_PFD's poly2 layout, without its re-standardization)."""
    import torch
    d = s.shape[1]
    iu = torch.triu_indices(d, d)
    return torch.cat([s, (s[:, :, None] * s[:, None, :])[:, iu[0], iu[1]]], 1)


def monitor_fisher2(zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, views=VIEWS, ridge=1e-3):
    """HELD-OUT degree-2 Fisher monitor (diagnostic only, never in any gradient). Per view: the pooled within-SEX covariance
    discriminant w = (S + ridge tr(S)/p I)^-1 Delta (FB_PFD's reader and ridge rule) on degree-2 polynomial features of the
    logit differences standardized by the A-row mean / SD, fitted on A and scored on E: the exact Mann-Whitney AUC on E (oriented
    by A: > 1/2 means the A-fitted direction recovers SEX on E). Also the per-coordinate population SD of the logit differences on A
    (to compare with START's: a > 10x shrink flags a collapsed output scale). Returns plain floats."""
    import torch
    out = {}
    with torch.no_grad():
        uA = view_inputs(zA_inc.detach().double(), zA_occ.detach().double())
        uE = view_inputs(zE_inc.detach().double(), zE_occ.detach().double())
        mA = sA.bool()
        for v in views:
            mu, sd = uA[v].mean(0), uA[v].std(0, unbiased=False)
            if not bool(torch.isfinite(sd).all()) or float(sd.min()) < 1e-12:
                out[v] = float('nan')
                continue
            pa, pe = _poly2((uA[v] - mu) / sd), _poly2((uE[v] - mu) / sd)
            a, b = pa[~mA], pa[mA]
            delta = b.mean(0) - a.mean(0)
            ca, cb = a - a.mean(0), b - b.mean(0)
            S = (ca.T @ ca + cb.T @ cb) / max(len(pa) - 2, 1)
            p = S.shape[0]
            S = S + ridge * torch.trace(S) / p * torch.eye(p, dtype=S.dtype)
            w = torch.linalg.solve(S, delta)
            out[v] = exact_auc((pe @ w).numpy(), sE.numpy())
        out['sd_A'] = {'income': [float(x) for x in uA['income'].std(0, unbiased=False)],
                       'occupation': [float(x) for x in uA['occupation'].std(0, unbiased=False)]}
    return out


# ---------------------------------------------------------------- helpers: episodes, logits, a reference release step
def household_index(unit):
    unit = np.asarray(unit)
    order = np.argsort(unit, kind='mergesort')
    u_sorted = unit[order]
    cuts = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1], True])
    return [order[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


def episode_rows(hh, rng, n_a, n_e):
    """Household-separated episode: households in a random order fill A (n_a rows), then E (n_e rows); a household cut at the A
    boundary loses its remaining members (they enter neither part). rng: numpy Generator."""
    A, E, na, ne = [], [], 0, 0
    for h in rng.permutation(len(hh)):
        rows = hh[h]
        if na < n_a:
            take = rows[:n_a - na]; A.append(take); na += len(take)
        elif ne < n_e:
            take = rows[:n_e - ne]; E.append(take); ne += len(take)
        else:
            break
    if na < n_a or ne < n_e:
        raise ValueError('pool too small for one episode')
    return np.concatenate(A), np.concatenate(E)


def centred_logits(model, Xrows):
    z = model(Xrows.to(next(model.parameters()).dtype)).double()
    return z - z.mean(1, keepdim=True)


# ---------------------------------------------------------------- invented-data checks
def _tiny(seed, d=6):
    """A tiny linear release (theta: income d x 2 + 2, occupation d x 5 + 5 = 49 numbers for d = 6), float64."""
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn(7 * d + 7, generator=g, dtype=torch.float64) * 0.5


def _tiny_logits(theta, X, d=6):
    Wi, bi = theta[:2 * d].reshape(d, 2), theta[2 * d:2 * d + 2]
    o = 2 * d + 2
    Wo, bo = theta[o:o + 5 * d].reshape(d, 5), theta[o + 5 * d:o + 5 * d + 5]
    zi, zo = X @ Wi + bi, X @ Wo + bo
    return zi - zi.mean(1, keepdim=True), zo - zo.mean(1, keepdim=True)


def _invented(n, seed, d=6, signal=0.5, nonlinear=0.0):
    """Invented rows: SEX ~ Bernoulli(1/2); a SEX-dependent mean shift (signal) plus an optional SEX-dependent spread of the first
    three coordinates (nonlinear: a variance signal that a linear reader cannot use)."""
    import torch
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    sgn = (2 * s - 1)[:, None]
    X = rng.normal(size=(n, d)) + signal * sgn * np.linspace(1.0, 0.2, d)[None, :]
    if nonlinear:
        X[:, :3] *= np.exp(nonlinear * sgn)
    return torch.from_numpy(X), torch.from_numpy(s.astype(np.int64))


def _recipe(**kw):
    rc = copy.deepcopy(RECIPE_RF)
    rc.update(kw)
    return rc


def _grads(theta0, XA, sA, XE, sE, rc, moments=None):
    import torch
    out = {}
    for mode in ('FULL', 'STOPFIT'):
        th = theta0.clone().requires_grad_(True)
        zA, zE = _tiny_logits(th, XA), _tiny_logits(th, XE)
        S, info = EpisodeObjective(mode, 0, rc).objective(*zA, sA, *zE, sE, moments=moments)
        out[mode] = (torch.autograd.grad(S, th)[0].detach(), info)
    return out


def _P(theta, XA, sA, XE, sE, rc, moments):
    zA, zE = _tiny_logits(theta, XA), _tiny_logits(theta, XE)
    return EpisodeObjective('STOPFIT', 0, rc).objective(*zA, sA, *zE, sE, moments=moments, diagnostics=False)[1]['P']


def _cmp(gf, gs):
    fitp = gf - gs
    return {'cos_full_vs_stopfit': float(gf @ gs / (gf.norm() * gs.norm() + 1e-300)),
            'norm_fit_over_direct': float(fitp.norm() / (gs.norm() + 1e-300)),
            'cos_fit_vs_direct': float(fitp @ gs / (fitp.norm() * gs.norm() + 1e-300)),
            'norm_full': float(gf.norm()), 'norm_direct': float(gs.norm())}


def _plain(x):
    if isinstance(x, dict):
        return all(isinstance(k, str) and _plain(v) for k, v in x.items())
    if isinstance(x, list):
        return all(_plain(v) for v in x)
    return isinstance(x, (bool, int, float, str)) or x is None


def selftest():
    """{check_name: bool, ..., 'all_pass': bool, 'numbers': {...}, 'cpu_s': float}; invented data, deterministic, float64."""
    import torch
    from acs_replication import neural
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    t0 = time.process_time()
    res, num = {}, {}
    rc = _recipe(newton_warm_start=False, views=list(VIEWS))          # all three views: the checks cover the income view too
    # 1. complete-derivative central finite differences over all 49 release parameters (A moments frozen at theta0, as in a step)
    XA, sA = _invented(256, 11); XE, sE = _invented(256, 12)
    th0 = _tiny(3)
    zA0 = _tiny_logits(th0, XA)
    mom = {v: moments_of(view_inputs(*zA0)[v]) for v in VIEWS}
    G = _grads(th0, XA, sA, XE, sE, rc, mom)
    gf, gs = G['FULL'][0], G['STOPFIT'][0]
    eps = 1e-5
    fd = torch.zeros_like(th0)
    for j in range(len(th0)):
        e = torch.zeros_like(th0); e[j] = eps
        fd[j] = (_P(th0 + e, XA, sA, XE, sE, rc, mom) - _P(th0 - e, XA, sA, XE, sE, rc, mom)) / (2 * eps)
    r_full, r_stop = float((fd - gf).norm() / fd.norm()), float((fd - gs).norm() / fd.norm())
    res['fd_complete_derivative_tiny_release'] = r_full < 1e-6 and r_stop > 1e-2
    num['fd_complete_derivative_tiny_release'] = {
        'n_A': 256, 'n_E': 256, 'parameters': len(th0), 'eps': eps, 'rel_err_FULL_vs_fd': r_full,
        'max_abs_err_FULL_vs_fd': float((fd - gf).abs().max()), 'rel_err_STOPFIT_vs_fd': r_stop, **_cmp(gf, gs),
        'max_abs_fit_scalar_value': max(abs(t['fit_scalar_value']) for t in G['FULL'][1]['terms'].values()),
        'max_newton_iterations': max(t['newton_iterations'] for t in G['FULL'][1]['terms'].values()),
        'P': G['FULL'][1]['P'], 'output_grad_norms': G['FULL'][1]['output_grad_norms']}
    # 1b. a float64 copy of a U0-size network pair (d = 145, d-64-64-16-K), 3 random directions in all parameters
    torch.manual_seed(5)
    nets = [neural.network(2, 145).double(), neural.network(5, 145).double()]
    rng = np.random.default_rng(7)
    Xb = torch.from_numpy(rng.normal(size=(512, 145))); sb = torch.from_numpy(rng.integers(0, 2, 512).astype(np.int64))
    Xb[:, :10] += 0.4 * (2 * sb[:, None] - 1).double()
    A, E = torch.arange(256), torch.arange(256, 512)
    flat = [p for m in nets for p in m.parameters()]
    with torch.no_grad():
        momb = {v: moments_of(view_inputs(*(centred_logits(m, Xb[A]) for m in nets))[v]) for v in VIEWS}

    def pnet():
        with torch.no_grad():
            za = [centred_logits(m, Xb[A]) for m in nets]; ze = [centred_logits(m, Xb[E]) for m in nets]
        return EpisodeObjective('STOPFIT', 0, rc).objective(za[0], za[1], sb[A], ze[0], ze[1], sb[E], moments=momb,
                                                            diagnostics=False)[1]['P']

    za = [centred_logits(m, Xb[A]) for m in nets]; ze = [centred_logits(m, Xb[E]) for m in nets]
    S, _ = EpisodeObjective('FULL', 0, rc).objective(za[0], za[1], sb[A], ze[0], ze[1], sb[E], moments=momb, diagnostics=False)
    gnet = torch.autograd.grad(S, flat)
    dirs = []
    for k in range(3):
        g = torch.Generator().manual_seed(100 + k)
        dv = [torch.randn(p.shape, generator=g, dtype=torch.float64) for p in flat]
        nrm = math.sqrt(sum(float((x ** 2).sum()) for x in dv)); dv = [x / nrm for x in dv]
        an = sum(float((a * b).sum()) for a, b in zip(gnet, dv))
        h = 1e-5
        with torch.no_grad():
            for p, x in zip(flat, dv): p.add_(h * x)
        pp = pnet()
        with torch.no_grad():
            for p, x in zip(flat, dv): p.add_(-2 * h * x)
        pm = pnet()
        with torch.no_grad():
            for p, x in zip(flat, dv): p.add_(h * x)
        fdv = (pp - pm) / (2 * h)
        dirs.append({'analytic': an, 'fd': fdv, 'rel_err': abs(an - fdv) / max(abs(fdv), 1e-300)})
    res['fd_complete_derivative_u0_size_float64'] = all(d['rel_err'] < 1e-5 for d in dirs)
    num['fd_complete_derivative_u0_size_float64'] = {'n_A': 256, 'n_E': 256, 'directions': dirs}
    # 2. optimization sign: a small step along -grad(FULL) lowers the surrogate; first-order prediction
    steps = []
    for eta in (1e-3, 1e-2):
        pn = _P(th0 - eta * gf, XA, sA, XE, sE, rc, mom)
        steps.append({'eta': eta, 'P0': G['FULL'][1]['P'], 'P_after': pn, 'decrease': G['FULL'][1]['P'] - pn,
                      'first_order_prediction': eta * float(gf @ gf)})
    res['optimization_sign'] = all(s['decrease'] > 0 for s in steps) and \
        abs(steps[0]['decrease'] / steps[0]['first_order_prediction'] - 1) < 0.05
    num['optimization_sign'] = steps
    # 3. envelope theorem: E = A and outer = -L_A (regularized) => fit path vanishes; out of sample it does not
    def env(XE_, sE_):
        out = {}
        for mode in ('FULL', 'STOPFIT'):
            th = th0.clone().requires_grad_(True)
            zA, zE = _tiny_logits(th, XA), _tiny_logits(th, XE_)
            uA, uE = view_inputs(*zA), view_inputs(*zE)
            tot = torch.zeros((), dtype=torch.float64)
            for v in VIEWS:
                sa, se = standardize(uA[v], mom[v]), standardize(uE[v], mom[v])
                for f in RECIPE_RF['families']:
                    pa, pe = basis(sa, v, f), basis(se, v, f)
                    kap = ridge_vector(pa.shape[1])
                    w, H, _ = fit_reader(pa, sA.double(), kap, rc)
                    wl = w.clone().requires_grad_(True)
                    R = -inner_loss(pe, sE_.double(), wl, kap)
                    tot = tot + R
                    if mode == 'FULL':
                        vv = torch.linalg.solve(H, torch.autograd.grad(R, wl, retain_graph=True)[0])
                        tot = tot - (vv * (pa.T @ (torch.sigmoid(pa @ w) - sA.double()) / len(sA) + kap * w)).sum()
            out[mode] = torch.autograd.grad(tot, th)[0]
        return _cmp(out['FULL'], out['STOPFIT'])
    e_in, e_out = env(XA, sA), env(XE, sE)
    res['envelope_identity'] = e_in['norm_fit_over_direct'] < 1e-8 and e_out['norm_fit_over_direct'] > 1e-3
    num['envelope_identity'] = {'in_sample_E_equals_A_outer_minus_L_A': e_in, 'held_out_E_not_A_outer_minus_L_E': e_out}
    # 3b. FB_PFD's in-batch Fisher criterion: its autograd gradient == the stopped-fit gradient (w = S^-1 Delta detached)
    from sar.fb import pfd as PD
    th = th0.clone().requires_grad_(True)
    zi, zo = _tiny_logits(th, XA)
    d2 = PD.fisher_d2(PD.poly2(PD.view_features(zi, zo)['pair']), sA)
    g_aut = torch.autograd.grad(PD.auc_proxy(d2), th)[0]
    th2 = th0.clone().requires_grad_(True)
    zi2, zo2 = _tiny_logits(th2, XA)
    phi2 = PD.poly2(PD.view_features(zi2, zo2)['pair'])
    m = sA.bool(); a, b = phi2[~m], phi2[m]
    delta = b.mean(0) - a.mean(0)
    ca, cb = a - a.mean(0), b - b.mean(0)
    Sw = (ca.T @ ca + cb.T @ cb) / max(len(phi2) - 2, 1)
    pp = Sw.shape[0]
    Sw = Sw + PD.RECIPE['ridge'] * torch.trace(Sw).detach() / pp * torch.eye(pp, dtype=Sw.dtype)
    wst = torch.linalg.solve(Sw, delta).detach()
    J = (wst @ delta) ** 2 / (wst @ Sw @ wst)                   # Rayleigh quotient at the detached in-batch best response
    g_stp = torch.autograd.grad(PD.auc_proxy(J), th2)[0]
    rel = float((g_aut - g_stp).norm() / g_aut.norm())
    res['fb_pfd_gradient_is_stopped_fit'] = rel < 1e-10
    num['fb_pfd_gradient_is_stopped_fit'] = {'source': 'sar.fb.pfd.poly2 / fisher_d2 / auc_proxy (imported, pair view)',
                                             'rel_diff_autograd_vs_stopped': rel, 'J_minus_D2': float(J.detach() - d2.detach())}
    # 4. when the fit path matters: exact zero, near-coincidence, material difference, campaign size
    cases = {}
    Gi = _grads(th0, XA, sA, XE, sE, _recipe(newton_warm_start=False, families=['simple'], views=['income']))
    cases['income_view_simple_exact_zero'] = {**_cmp(Gi['FULL'][0], Gi['STOPFIT'][0]),
                                              'why': '1 feature + intercept: AUC_b is invariant to positive scale and shift of w'}
    XA2, sA2 = _invented(32768, 21); XE2, sE2 = _invented(32768, 22)
    Gc = _grads(th0, XA2, sA2, XE2, sE2, _recipe(newton_warm_start=False, families=['simple'], views=['occupation'], ridge=1e-4))
    cases['large_n_gaussian_simple_occupation_small_ridge_near_coincide'] = {
        'n_A': 32768, 'n_E': 32768, 'ridge': 1e-4, **_cmp(Gc['FULL'][0], Gc['STOPFIT'][0]),
        'why': 'well-specified equal-covariance Gaussian logits, negligible ridge bias, large n: the logistic direction is the '
               'AUC_b-optimal direction, so grad_w R = O(n^-1/2) at w*'}
    Gr = _grads(th0, XA2, sA2, XE2, sE2, _recipe(newton_warm_start=False, families=['simple'], views=['occupation']))
    cases['large_n_gaussian_simple_occupation_registered_ridge'] = {
        'n_A': 32768, 'n_E': 32768, 'ridge': RECIPE_RF['ridge'], **_cmp(Gr['FULL'][0], Gr['STOPFIT'][0]),
        'why': 'the same data with the registered ridge: the ridge-biased direction is not AUC_b-optimal, so the fit path does '
               'not vanish as n grows'}
    Gp = _grads(th0, XA2, sA2, XE2, sE2, _recipe(newton_warm_start=False, families=['simple'], views=['pair'], ridge=1e-4))
    cases['large_n_gaussian_simple_pair_small_ridge'] = {
        'n_A': 32768, 'n_E': 32768, 'ridge': 1e-4, **_cmp(Gp['FULL'][0], Gp['STOPFIT'][0]),
        'why': 'nearly collinear pair coordinates: H^-1 amplifies the O(n^-1/2) grad_w R along weak directions'}
    XA3, sA3 = _invented(96, 31, signal=0.15, nonlinear=0.3); XE3, sE3 = _invented(2048, 32, signal=0.15, nonlinear=0.3)
    Gd = _grads(th0, XA3, sA3, XE3, sE3, _recipe(newton_warm_start=False, families=['rff'], views=['pair']))
    cases['small_n_A_rff_misspecified_differ'] = {'n_A': 96, 'n_E': 2048, **_cmp(Gd['FULL'][0], Gd['STOPFIT'][0])}
    XA4, sA4 = _invented(2048, 41, signal=0.3, nonlinear=0.2); XE4, sE4 = _invented(2048, 42, signal=0.3, nonlinear=0.2)
    Gm = _grads(th0, XA4, sA4, XE4, sE4, rc)
    per = {}
    for v in VIEWS:
        for fam in RECIPE_RF['families']:
            Gx = _grads(th0, XA4, sA4, XE4, sE4, _recipe(newton_warm_start=False, families=[fam], views=[v]))
            per[f'{v}/{fam}'] = _cmp(Gx['FULL'][0], Gx['STOPFIT'][0])
    cases['campaign_size_invented'] = {'n_A': 2048, 'n_E': 2048, 'all_terms': _cmp(Gm['FULL'][0], Gm['STOPFIT'][0]), 'per_term': per,
                                       'output_grad_norms': Gm['FULL'][1]['output_grad_norms']}
    res['fit_path_exact_zero_income_simple'] = cases['income_view_simple_exact_zero']['norm_fit_over_direct'] < 1e-8
    res['fit_path_differs_materially_example'] = (cases['small_n_A_rff_misspecified_differ']['norm_fit_over_direct'] > 0.3 and
                                                  cases['small_n_A_rff_misspecified_differ']['norm_fit_over_direct'] >
                                                  10 * cases['large_n_gaussian_simple_occupation_small_ridge_near_coincide']
                                                  ['norm_fit_over_direct'])
    num['when_the_fit_path_matters'] = cases
    # 5. ALTCRIT: inner 'newton' == STOPFIT bitwise; inner 'adam' (lagging) is not
    torch.manual_seed(9)
    nets2 = [neural.network(2, 145), neural.network(5, 145)]
    Xc = torch.from_numpy(rng.normal(size=(1024, 145)).astype(np.float32)); sc = torch.from_numpy(rng.integers(0, 2, 1024))
    Xc[:, :10] += 0.4 * (2 * sc[:, None] - 1).float()
    Ac, Ec, Wc = torch.arange(512), torch.arange(512, 1024), torch.arange(256, 768)
    flat2 = [p for m in nets2 for p in m.parameters()]

    def g_of(obj, warm=None):
        if warm is not None:                                            # one earlier episode (warm-up of the critic)
            with torch.no_grad():
                zw = [centred_logits(m, Xc[warm]) for m in nets2]
            obj.objective(zw[0], zw[1], sc[warm], zw[0], zw[1], sc[warm], diagnostics=False)
        za_ = [centred_logits(m, Xc[Ac]) for m in nets2]; ze_ = [centred_logits(m, Xc[Ec]) for m in nets2]
        S_, inf_ = obj.objective(za_[0], za_[1], sc[Ac], ze_[0], ze_[1], sc[Ec])
        return torch.cat([x.reshape(-1).double() for x in torch.autograd.grad(S_, flat2)]), inf_

    a_s, _ = g_of(EpisodeObjective('STOPFIT', 0, rc))
    a_n, inf_n = g_of(EpisodeObjective('ALTCRIT', 0, _recipe(newton_warm_start=False, altcrit_inner='newton', views=list(VIEWS))),
                      warm=Wc)
    a_l, inf_l = g_of(EpisodeObjective('ALTCRIT', 0, rc), warm=Wc)
    res['altcrit_newton_alias_of_stopfit'] = float((a_s - a_n).abs().max()) == 0.0
    res['altcrit_adam_lagging_not_alias'] = float((a_s - a_l).norm() / a_s.norm()) > 1e-3
    num['altcrit'] = {'max_abs_diff_newton_altcrit_vs_stopfit': float((a_s - a_n).abs().max()),
                      'rel_diff_adam_altcrit_vs_stopfit': float((a_s - a_l).norm() / a_s.norm()),
                      'cos_adam_altcrit_vs_stopfit': float(a_s @ a_l / (a_s.norm() * a_l.norm())),
                      'adam_critic_max_abs_grad_A_after_rounds': max(t.get('critic_max_abs_grad_A', 0.0) for t in inf_l['terms'].values()),
                      'note': 'alias iff the critic is re-solved exactly on the current A rows at the current release before the '
                              'release gradient (altcrit_inner = newton); a converged first-order inner loop is an alias up to its '
                              'tolerance; 10 Adam rounds at lr 3e-3 from the previous episode lag'}
    res['altcrit_lag_diagnostics'] = (inf_n['critic_rel_lag_median'] == 0.0 and inf_l['critic_rel_lag_median'] > 0.0
                                      and _plain(inf_l))
    num['altcrit'].update({'rel_lag_median_newton': inf_n['critic_rel_lag_median'], 'rel_lag_median_adam': inf_l['critic_rel_lag_median'],
                           'max_abs_grad_A_adam': inf_l['critic_max_abs_grad_A_max']})
    # 5a. objective_parts: S_dir is STOPFIT's objective at the same w*; S_fit carries exactly FULL - STOPFIT
    th = th0.clone().requires_grad_(True)
    zA_, zE_ = _tiny_logits(th, XA), _tiny_logits(th, XE)
    Sd, Sf, inf_p = EpisodeObjective('FULL', 0, rc).objective_parts(*zA_, sA, *zE_, sE, moments=mom)
    g_d = torch.autograd.grad(Sd, th, retain_graph=True)[0]
    g_f = torch.autograd.grad(Sf, th)[0]
    scale = float(gf.abs().max())
    d_dir, d_fit = float((g_d - gs).abs().max()), float((g_f - (gf - gs)).abs().max())
    res['objective_parts_consistent'] = d_dir <= 1e-14 * scale and d_fit <= 1e-13 * scale and abs(inf_p['S_fit']) < 1e-12
    num['objective_parts'] = {'max_abs_grad_S_dir_minus_STOPFIT': d_dir, 'max_abs_grad_S_fit_minus_FULL_minus_STOPFIT': d_fit,
                              'S_fit_value': inf_p['S_fit'], 'S_dir_value': inf_p['S_dir']}
    # 5c. held-out degree-2 Fisher monitor (diagnostic only)
    mon = monitor_fisher2(*(z.detach() for z in zA0), sA, *(z.detach() for z in _tiny_logits(th0, XE)), sE)
    res['monitor_fisher2_runs_oriented'] = _plain(mon) and all(0.5 < mon[v] <= 1.0 for v in VIEWS) and len(mon['sd_A']['occupation']) == 4
    num['monitor_fisher2'] = mon
    # 5b. the training loop's pattern under the REGISTERED recipe: views (occupation, pair), income logits constants (no graph);
    #     FD over the 35 occupation parameters against FULL; the income-view term is absent
    rc_reg = _recipe(newton_warm_start=False)
    zi_A, zi_E = _tiny_logits(th0, XA)[0].detach(), _tiny_logits(th0, XE)[0].detach()
    o = 14

    def p_occ(th):
        return EpisodeObjective('STOPFIT', 0, rc_reg).objective(zi_A, _tiny_logits(th, XA)[1], sA, zi_E, _tiny_logits(th, XE)[1], sE,
                                                                moments=mom, diagnostics=False)[1]['P']

    th = th0.clone().requires_grad_(True)
    S, inf_o = EpisodeObjective('FULL', 0, rc_reg).objective(zi_A, _tiny_logits(th, XA)[1], sA, zi_E, _tiny_logits(th, XE)[1], sE,
                                                             moments=mom)
    g_all = torch.autograd.grad(S, th)[0]
    fd_o = torch.zeros(len(th0) - o, dtype=torch.float64)
    for j in range(o, len(th0)):
        e = torch.zeros_like(th0); e[j] = eps
        fd_o[j - o] = (p_occ(th0 + e) - p_occ(th0 - e)) / (2 * eps)
    r_o = float((fd_o - g_all[o:]).norm() / fd_o.norm())
    res['fd_registered_views_income_constant'] = (r_o < 1e-6 and float(g_all[:o].abs().max()) == 0.0
                                                  and sorted(inf_o['terms']) == ['occupation/rff', 'occupation/simple', 'pair/rff',
                                                                                 'pair/simple'])
    num['fd_registered_views_income_constant'] = {'parameters': len(th0) - o, 'rel_err_FULL_vs_fd': r_o,
                                                  'max_abs_grad_income_params': float(g_all[:o].abs().max()),
                                                  'terms': sorted(inf_o['terms']), 'output_grad_norms': inf_o['output_grad_norms']}
    # 6. guards and interface
    g6 = {}
    zi0, zo0 = (z.detach() for z in zA0)
    for name, call in (('one_class_A_refused', lambda: EpisodeObjective('STOPFIT', 0, rc).objective(
                            zi0, zo0, torch.zeros(256, dtype=torch.int64), zi0, zo0, sA)),
                       ('nan_logits_refused', lambda: EpisodeObjective('STOPFIT', 0, rc).objective(
                            zi0.clone().index_fill_(0, torch.tensor([0]), float('nan')), zo0, sA, zi0, zo0, sA)),
                       ('constant_view_refused', lambda: EpisodeObjective('STOPFIT', 0, rc).objective(
                            torch.zeros_like(zi0), zo0, sA, zi0, zo0, sA))):
        try:
            call(); g6[name] = False
        except FloatingPointError:
            g6[name] = True
    tt = torch.from_numpy(np.random.default_rng(1).normal(size=400)); yy = torch.from_numpy(np.r_[np.ones(200), np.zeros(200)]).bool()
    tt = tt + 0.8 * yy.double()
    g6['inverted_score_not_rewarded'] = abs(float((binormal_auc(tt, yy)[0] - 0.5) ** 2 - (binormal_auc(-tt, yy)[0] - 0.5) ** 2)) < 1e-15
    g6['scale_shift_invariant_surrogate'] = abs(float(binormal_auc(3.0 * tt + 7.0, yy)[0] - binormal_auc(tt, yy)[0])) < 1e-14
    g6['info_plain_floats'] = all(_plain(G[k][1]) for k in G) and _plain(inf_l)
    res.update(g6)
    res['all_pass'] = all(v for k, v in res.items() if isinstance(v, bool))
    res['numbers'] = num
    res['cpu_s'] = time.process_time() - t0
    return res


def bench(sizes=((2048, 2048), (4096, 2400), (4096, 4096)), reps=4):
    """CPU seconds per release step on one thread in the training loop's pattern (rfa/fam/fit.py): the occupation network
    (d = 145, d-64-64-16-5, float32) trained, START's income logits constant; registered views and readers; the A and E rows
    forwarded with their graph in every mode; info diagnostics on (the loop's default call); 256-row task batch, Adam, clip 5.
    Random weights and invented X only."""
    import torch
    import torch.nn.functional as F
    from acs_replication import neural
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = {'schema': 'RFA1-rfit-bench-v2', 'threads': torch.get_num_threads(), 'torch': torch.__version__,
           'views': RECIPE_RF['views'], 'families': RECIPE_RF['families'], 'rows': {}}
    rng = np.random.default_rng(0)
    N = 16000
    X = torch.from_numpy(rng.normal(size=(N, 145)).astype(np.float32))
    sex = torch.from_numpy(rng.integers(0, 2, N))
    X[:, :10] += 0.3 * (2 * sex[:, None] - 1).float()
    y = torch.from_numpy(rng.integers(0, 5, N))
    hh = household_index(np.repeat(np.arange(N // 2), 2))
    torch.manual_seed(0)
    inc = neural.network(2, 145)
    with torch.no_grad():
        zi = centred_logits(inc, X)
    for na, ne in sizes:
        row = {}
        for name in ('FULL', 'STOPFIT', 'ALTCRIT', 'TASK'):
            torch.manual_seed(1)
            occ = neural.network(5, 145)
            params = list(occ.parameters())
            opt = torch.optim.Adam(params, lr=1e-4)
            obj = None if name == 'TASK' else EpisodeObjective(name, 0)
            erng = np.random.default_rng(5)
            times, its = [], []
            for r in range(reps + 1):
                A, E = (torch.from_numpy(x) for x in episode_rows(hh, erng, na, ne))
                tr = torch.from_numpy(erng.choice(N, 256, replace=False))
                t0 = time.process_time()
                if obj is not None:
                    S, info = obj.objective(zi[A], centred_logits(occ, X[A]), sex[A], zi[E], centred_logits(occ, X[E]), sex[E])
                    gp = torch.autograd.grad(S, params)
                    its.append(max(t.get('newton_iterations', 0) for t in info['terms'].values()))
                else:
                    gp = [torch.zeros_like(p) for p in params]
                gt = torch.autograd.grad(F.cross_entropy(centred_logits(occ, X[tr]), y[tr]), params)
                opt.zero_grad(set_to_none=True)
                for p, a, c in zip(params, gt, gp):
                    p.grad = a + c
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                if r > 0:
                    times.append(time.process_time() - t0)
            row[name] = {'cpu_s_per_step_mean': float(np.mean(times)), 'cpu_s_per_step_max': float(np.max(times)),
                         'steps_timed': reps, 'max_newton_iterations': its}
        out['rows'][f'n_A={na},n_E={ne}'] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rfa.fam.rfit')
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

"""Challenger check (INVENTED DATA ONLY; no real data, no label, no model file is read).

Questions:
  E1  Envelope: the gradient of FB_PFD's in-batch Fisher D^2 (autograd through the solve) equals the gradient
      with the discriminant weight w* = (S + eps I)^-1 Delta held fixed (a stopped-fit gradient).
  E2  The same envelope fails for a held-out (fit on A, evaluate on E) objective; the implicit-function (IFT)
      fit-path term is checked against a central finite difference of the re-solved objective.
  E3  Size of the fit path for the proposed operation on a tiny synthetic law: ridge logistic regression on fixed
      linear + random-ReLU features of the pair logit differences, fitted on A (n_A rows), objective
      J = (smoothAUC_E - 1/2)^2 on E; direct path (reader detached) vs fit path (IFT through the ridge solve).
Run from the worktree root:  <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/challenger/fitpath_check.py
"""
import json
import math
import sys
import time

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
T0 = time.process_time()
D_IN = 6


# ------------------------------------------------------------------ invented law and tiny networks
def draw(n, gen, law):
    s = (torch.rand(n, generator=gen) < 0.5).double()
    x = torch.randn(n, D_IN, generator=gen)
    mu = torch.tensor([0.35, -0.25, 0.20, 0.0, 0.0, 0.10])
    x = x + s[:, None] * mu
    if law == 'nonlin':                      # SEX also changes the spread of two inputs (nonlinear signal)
        x[:, 3] = x[:, 3] * (1 + 0.35 * s)
        x[:, 4] = x[:, 4] * (1 - 0.25 * s)
    return x, s


def nets(seed, linear=False):
    torch.manual_seed(seed)
    if linear:
        return [torch.nn.Linear(D_IN, 2), torch.nn.Linear(D_IN, 5)]
    return [torch.nn.Sequential(torch.nn.Linear(D_IN, 16), torch.nn.ReLU(), torch.nn.Linear(16, K)) for K in (2, 5)]


def pair_u(models, x):
    zi, zo = models[0](x), models[1](x)
    return torch.cat([zi[:, 1:] - zi[:, :1], zo[:, 1:] - zo[:, :1]], 1)          # 5 logit differences


class FeatureMap:
    """Fixed (detached) reference standardization + fixed random ReLU features; p = 5 + m."""
    def __init__(self, u_ref, m, seed):
        g = torch.Generator().manual_seed(seed)
        self.mean, self.sd = u_ref.mean(0).detach(), u_ref.std(0).detach()
        self.R = torch.randn(m, u_ref.shape[1], generator=g) / math.sqrt(u_ref.shape[1])
        self.b = torch.rand(m, generator=g) * 2 - 1

    def __call__(self, u):
        t = (u - self.mean) / self.sd
        return torch.cat([t, torch.relu(t @ self.R.T + self.b)], 1) if len(self.b) else t


# ------------------------------------------------------------------ ridge logistic regression (intercept unpenalized)
def design(phi):
    return torch.cat([phi, torch.ones(len(phi), 1)], 1)


def grad_L(beta, Phi1, s, alpha):
    r = torch.sigmoid(Phi1 @ beta) - s
    g = Phi1.T @ r / len(s)
    pen = alpha * beta.clone(); pen[-1] = 0.0
    return g + pen


def hess_L(beta, Phi1, s, alpha):
    p = torch.sigmoid(Phi1 @ beta)
    H = (Phi1 * (p * (1 - p))[:, None]).T @ Phi1 / len(s)
    d = torch.full((Phi1.shape[1],), alpha); d[-1] = 0.0
    return H + torch.diag(d)


def solve(Phi1, s, alpha, tol=1e-13, it=60):
    beta = torch.zeros(Phi1.shape[1])
    for _ in range(it):
        g = grad_L(beta, Phi1, s, alpha)
        if g.abs().max() < tol:
            break
        beta = beta - torch.linalg.solve(hess_L(beta, Phi1, s, alpha), g)
    return beta


def smooth_auc(score, s, tau=0.2):
    t = (score - score.mean()) / score.std()
    a, b = t[s > 0.5], t[s < 0.5]
    return torch.sigmoid((a[:, None] - b[None, :]) / tau).mean()


def exact_auc(score, s):
    a, b = score[s > 0.5], score[s < 0.5]
    return float(((a[:, None] > b[None, :]).double().mean() + 0.5 * (a[:, None] == b[None, :]).double().mean()))


def flat(gs, params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(gs, params)])


def episode(models, fmap, xA, sA, xE, sE, alpha, tau=0.2):
    params = [p for m in models for p in m.parameters()]
    PhiA = design(fmap(pair_u(models, xA)))
    beta = solve(PhiA.detach(), sA, alpha)
    bvar = beta.clone().requires_grad_(True)
    PhiE = design(fmap(pair_u(models, xE)))
    auc = smooth_auc(PhiE @ bvar, sE, tau)
    J = (auc - 0.5) ** 2
    gs = torch.autograd.grad(J, params + [bvar], allow_unused=True)
    g_dir, dJ_db = flat(gs[:-1], params), gs[-1]
    v = torch.linalg.solve(hess_L(beta, PhiA.detach(), sA, alpha), dJ_db)
    G = (v.detach() * grad_L(beta.detach(), PhiA, sA, alpha)).sum()            # v' grad_beta L_A(beta*; theta)
    g_fit = -flat(torch.autograd.grad(G, params, allow_unused=True), params)
    return g_dir, g_fit, float(auc), exact_auc((PhiE @ beta).detach(), sE), beta


def objective_resolved(models, fmap, xA, sA, xE, sE, alpha, tau=0.2):
    with torch.no_grad():
        PhiA = design(fmap(pair_u(models, xA)))
        beta = solve(PhiA, sA, alpha)
        PhiE = design(fmap(pair_u(models, xE)))
        return float((smooth_auc(PhiE @ beta, sE, tau) - 0.5) ** 2)


def perturb(models, d, h):
    with torch.no_grad():
        k = 0
        for m in models:
            for p in m.parameters():
                n = p.numel(); p.add_(h * d[k:k + n].view_as(p)); k += n


# ------------------------------------------------------------------ E1: envelope for FB_PFD's statistic
def pfd_fisher_d2(phi, sex, ridge=1e-3):          # verbatim law of sar.fb.pfd.fisher_d2 (re-implemented, not imported)
    m = sex.bool()
    a, b = phi[~m], phi[m]
    delta = b.mean(0) - a.mean(0)
    ca, cb = a - a.mean(0), b - b.mean(0)
    S = (ca.T @ ca + cb.T @ cb) / max(len(phi) - 2, 1)
    p = S.shape[0]
    S = S + ridge * torch.trace(S).detach() / p * torch.eye(p)
    return delta @ torch.linalg.solve(S, delta), delta, S


def poly2(u):
    s = (u - u.mean(0, keepdim=True).detach()) / (u.std(0, keepdim=True).detach() + 1e-9)
    d = s.shape[1]
    iu = torch.triu_indices(d, d)
    return torch.cat([s, (s[:, :, None] * s[:, None, :])[:, iu[0], iu[1]]], 1)


def check_envelope():
    gen = torch.Generator().manual_seed(11)
    models = nets(3)
    params = [p for m in models for p in m.parameters()]
    x, s = draw(2048, gen, 'nonlin')
    d2, _, _ = pfd_fisher_d2(poly2(pair_u(models, x)), s)
    g_full = flat(torch.autograd.grad(d2, params), params)
    d2b, delta, S = pfd_fisher_d2(poly2(pair_u(models, x)), s)
    w = torch.linalg.solve(S, delta).detach()                       # the in-batch best response, then STOPPED
    Q = 2 * w @ delta - w @ S @ w                                  # max_w Q(w) = D^2 ; Q(w*) = D^2
    g_stop = flat(torch.autograd.grad(Q, params), params)
    # same check for an in-sample objective that is NOT the inner objective: smooth AUC of the in-sample ridge-logistic reader
    fmap = FeatureMap(pair_u(models, x).detach(), 16, 5)
    gd, gf, *_ = episode(models, fmap, x, s, x, s, 1e-2)
    return {'pfd_D2': float(d2), 'pfd_rel_diff_full_vs_stopped': float((g_full - g_stop).norm() / g_full.norm()),
            'insample_logistic_auc_fitpath_share': float(gf.norm() / gd.norm())}


def check_ift():
    gen = torch.Generator().manual_seed(12)
    models = nets(4)
    xA, sA = draw(2048, gen, 'nonlin'); xE, sE = draw(2048, gen, 'nonlin')
    fmap = FeatureMap(pair_u(models, draw(4096, gen, 'nonlin')[0]).detach(), 64, 6)
    out = {}
    for alpha in (1e-2, 1.0):
        gd, gf, *_ = episode(models, fmap, xA, sA, xE, sE, alpha)
        d = torch.randn(gd.numel(), generator=torch.Generator().manual_seed(99)); d = d / d.norm()
        h = 1e-5
        perturb(models, d, h); fp = objective_resolved(models, fmap, xA, sA, xE, sE, alpha)
        perturb(models, d, -2 * h); fm = objective_resolved(models, fmap, xA, sA, xE, sE, alpha)
        perturb(models, d, h)
        fd = (fp - fm) / (2 * h)
        out[str(alpha)] = {'finite_difference': fd, 'direct_plus_fit': float((gd + gf) @ d), 'direct_only': float(gd @ d),
                           'rel_err_total': abs(fd - float((gd + gf) @ d)) / abs(fd)}
    return out


# ------------------------------------------------------------------ E3: size of the fit path
def measure(cfg, n_ep):
    gen = torch.Generator().manual_seed(cfg['seed'])
    models = nets(cfg['net_seed'], linear=cfg.get('linear', False))
    fmap = FeatureMap(pair_u(models, draw(8192, gen, cfg['law'])[0]).detach(), cfg['m'], cfg['seed'] + 1)
    GD, GF, aucs, eaucs, norms = [], [], [], [], []
    for _ in range(n_ep):
        xA, sA = draw(cfg['nA'], gen, cfg['law']); xE, sE = draw(cfg['nE'], gen, cfg['law'])
        gd, gf, a, ea, beta = episode(models, fmap, xA, sA, xE, sE, cfg['alpha'])
        GD.append(gd); GF.append(gf); aucs.append(a); eaucs.append(ea); norms.append(float(beta[:-1].norm()))
    GD, GF = torch.stack(GD), torch.stack(GF)
    GT = GD + GF
    G = 8                                 # unbiased cross-group estimates of mean-vector norms; jackknife over groups

    def stats(keep):
        md = [GD[g::G].mean(0) for g in keep]; mf = [GF[g::G].mean(0) for g in keep]
        pairs = [(i, j) for i in range(len(keep)) for j in range(len(keep)) if i != j]
        dd = sum(float(md[i] @ md[j]) for i, j in pairs) / len(pairs)
        ff = sum(float(mf[i] @ mf[j]) for i, j in pairs) / len(pairs)
        fd = sum(float(mf[i] @ md[j]) for i, j in pairs) / len(pairs)
        ratio = math.sqrt(max(ff, 0.0) / dd) if dd > 0 else float('nan')
        cos = fd / math.sqrt(dd * ff) if dd > 0 and ff > 0 else float('nan')
        orth = math.sqrt(max(ff - fd * fd / dd, 0.0) / dd) if dd > 0 else float('nan')
        tt = dd + ff + 2 * fd                                              # ||E g_total||^2
        cos_td = (dd + fd) / math.sqrt(dd * tt) if dd > 0 and tt > 0 else float('nan')
        return {'ratio': ratio, 'cos': cos, 'orth': orth, 'cos_total_direct': cos_td, 'ff': ff, 'dd': dd}

    full = stats(range(G))
    jk = [stats([g for g in range(G) if g != k]) for k in range(G)]
    def se(key):
        vals = [r[key] for r in jk if not math.isnan(r[key])]
        if len(vals) < 2:
            return float('nan')
        m = sum(vals) / len(vals)
        return math.sqrt((len(vals) - 1) / len(vals) * sum((v - m) ** 2 for v in vals))
    noise_tot = float(((GT - GT.mean(0)) ** 2).sum(1).mean().sqrt())   # per-step noise of the full update
    noise_dir = float(((GD - GD.mean(0)) ** 2).sum(1).mean().sqrt())   # per-step noise of the direct (stopped-fit) update
    per = (GF.norm(dim=1) / GD.norm(dim=1))
    cos_ep = torch.nn.functional.cosine_similarity(GF, GD, dim=1)
    mean_fit = math.sqrt(max(full['ff'], 0.0))
    return {**{k: v for k, v in cfg.items() if k != 'seed'}, 'p': 5 + cfg['m'], 'episodes': n_ep,
            'smooth_auc_E_mean': sum(aucs) / n_ep, 'exact_auc_E_mean': sum(eaucs) / n_ep, 'reader_norm_mean': sum(norms) / n_ep,
            'per_step_fit_over_direct_median': float(per.median()), 'per_step_fit_over_direct_p90': float(per.quantile(0.9)),
            'per_step_cos_fit_direct_mean': float(cos_ep.mean()),
            'mean_fit_over_mean_direct': full['ratio'], 'mean_fit_over_mean_direct_se': se('ratio'),
            'cos_of_means_fit_direct': full['cos'], 'cos_of_means_fit_direct_se': se('cos'),
            'orthogonal_mean_fit_over_mean_direct': full['orth'], 'orthogonal_se': se('orth'),
            'cos_of_means_total_vs_direct': full['cos_total_direct'], 'cos_total_vs_direct_se': se('cos_total_direct'),
            'per_step_noise_direct': noise_dir, 'per_step_noise_total': noise_tot,
            'mean_direct_norm': math.sqrt(max(full['dd'], 0.0)), 'mean_fit_norm': mean_fit,
            'mean_fit_over_step_noise_total': mean_fit / noise_tot,
            'steps_for_2sigma_fit_drift': (2 * noise_tot / mean_fit) ** 2 if mean_fit > 0 else float('inf')}


def main():
    out = {'invented_data_only': True}
    out['E1_envelope'] = check_envelope()
    out['E2_ift_vs_finite_difference'] = check_ift()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    grid = []
    for m in (0, 16, 64):
        for alpha in (1e-4, 1e-2, 1.0):
            grid.append({'law': 'nonlin', 'nA': 2048, 'nE': 2048, 'm': m, 'alpha': alpha, 'seed': 100 + m, 'net_seed': 7})
    for nA in (512, 8192):
        grid.append({'law': 'nonlin', 'nA': nA, 'nE': 2048, 'm': 64, 'alpha': 1e-2, 'seed': 300 + nA, 'net_seed': 7})
    for alpha in (1e-4, 1.0):                       # well-specified: linear nets, Gaussian shift, linear features
        grid.append({'law': 'shift', 'nA': 2048, 'nE': 2048, 'm': 0, 'alpha': alpha, 'seed': 500, 'net_seed': 7, 'linear': True})
    grid.append({'law': 'shift', 'nA': 8192, 'nE': 2048, 'm': 0, 'alpha': 1e-4, 'seed': 501, 'net_seed': 7, 'linear': True})
    res = []
    for cfg in grid:
        t = time.process_time()
        r = measure(cfg, N); r['cpu_s'] = round(time.process_time() - t, 2); res.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    out['E3_fit_path_size'] = res
    out['episodes_per_config'] = N
    out['cpu_seconds_total'] = round(time.process_time() - T0, 2)
    with open('results/pcrl_reader_fit_aware_release_v1/roles/challenger/fitpath_check.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ('E1_envelope', 'E2_ift_vs_finite_difference', 'cpu_seconds_total')}, indent=1))


if __name__ == '__main__':
    main()

"""Joint confirmation power for the frozen L_TREE release (runnable on public aggregate inputs only).

Model. For a future study with G independent evaluation groups (duplicate groups; NOT bootstrap draws and NOT
model seeds -- both seeds are evaluated on the same people), the vector of clause statistics is approximately
    T_hat ~ N(mu, A Sigma A' / G),
where Sigma is the covariance of per-group influence functions estimated on INNER (already used; validated
against the original 2048-draw bootstrap) and A maps base statistics (24 utility clause statistics and the
coalition/recipient AUCs of the candidate and every comparator, both seeds) to clause statistics.
Prevalence shifts re-weight the class-split second moments (AUC influence terms scale with 1/pi_class).

Decision rules (prospective; they do not alter any historical result):
  SIM  -- the original style: every clause's two-sided Bonferroni bound over a family of K endpoints must lie on
          the passing side (critical value z_{1 - 0.05/(2K)}; large-G normal limit of the Student-t value).
  IUT  -- one fixed global claim: every clause's one-sided level-alpha test rejects its own 'fails' null
          (critical value z_{1-alpha}, alpha = 0.025 one-sided).
For one draw z ~ N(0, I), clause j (direction d_j = +1 for 'statistic <= bound', -1 for 'statistic > bound',
slack_j = -d_j (mu_j - bound_j) > 0) passes iff sqrt(G) >= (d_j (L z)_j + c sigma_j) / slack_j. Hence the smallest
G at which ALL clauses pass is G*(z) = max_j [max(0, d_j (L z)_j + c sigma_j) / slack_j]^2 and the joint power is
P(G* <= G). A clause with slack_j <= 0 (true effect at or beyond its bound) cannot reach high power at any G.
Known-variance approximation; a nonparametric resampling check with estimated SEs is reported separately.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

UTILITY_CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
CODES = ('H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT')
ORIGINAL_REQUIRED = ('RAW_PROJ', 'N_EQ', 'T_TREE', 'H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT')
STRONGEST_FEASIBLE_CODE = 'H_SEQ21_CODE'   # lowest INNER coalition AUC among codes passing all 24 INNER point caps


def load(path):
    f = json.loads(Path(path).read_text())
    names = f['names']; idx = {n: i for i, n in enumerate(names)}
    return f, names, idx, np.array(f['points']), np.array(f['cov_phi']), np.array(f['second_moment_pos']), np.array(f['second_moment_neg'])


def claim(idx, kind, required=ORIGINAL_REQUIRED, superiority_vs=STRONGEST_FEASIBLE_CODE):
    """Clauses as (name, a-vector, direction, bound, group)."""
    M = len(idx); rows = []
    for nm, i in idx.items():
        if nm.startswith('U|'):
            _, s, t, ref, metric = nm.split('|'); a = np.zeros(M); a[i] = 1
            rows.append((f'utility|s{s}|{t}|{ref}|{metric}', a, +1, UTILITY_CAPS[metric], 'utility'))

    def auc(svc, s, v): return idx[f'AUC|{svc}|{s}|{v}']

    def benefit(c, margin, tag):
        a = np.zeros(M)
        for s in (0, 1):
            a[auc(c, s, 'pair')] += 0.5; a[auc('L_TREE', s, 'pair')] -= 0.5
        rows.append((f'{tag}|benefit_vs_{c}', a, -1, margin, tag))

    def excess(c):
        for v in ('income', 'occupation'):
            a = np.zeros(M)
            for s in (0, 1):
                a[auc('L_TREE', s, v)] += 0.5; a[auc(c, s, v)] -= 0.5
            rows.append((f'excess|{v}_vs_{c}', a, +1, 0.010, 'excess'))
    if kind == 'A':
        for c in required:
            benefit(c, 0.020, 'benefit'); excess(c)
    elif kind == 'B':
        benefit('RAW_PROJ', 0.020, 'benefit'); excess('RAW_PROJ')
        benefit(superiority_vs, 0.0, 'superiority'); excess(superiority_vs)
    else:
        raise ValueError(kind)
    return rows


def sigma_for(cov, Mpos, Mneg, pi1_obs, names, minority_share=None, var_mult=1.0, class_means=None):
    """(Sigma, delta): covariance of per-group influence values and the shift of base-statistic means under a SEX
    prevalence (label) shift with unchanged class-conditional distributions. AUC influence terms scale with
    1/pi_class and have zero within-class mean; utility statistics' means move to pi' m_pos + (1-pi') m_neg
    (review fix 2026-10-15: means, not only variances, are re-weighted)."""
    if minority_share is None:
        return cov * var_mult, np.zeros(len(names))
    pi1_new = 1 - minority_share if pi1_obs >= 0.5 else minority_share
    is_auc = np.array([not n.startswith('U|') for n in names])
    dpos = np.where(is_auc, pi1_obs / pi1_new, 1.0); dneg = np.where(is_auc, (1 - pi1_obs) / (1 - pi1_new), 1.0)
    S = pi1_new * (dpos[:, None] * Mpos * dpos[None, :]) + (1 - pi1_new) * (dneg[:, None] * Mneg * dneg[None, :])
    delta = np.zeros(len(names))
    if class_means is not None:
        mp, mn = np.asarray(class_means[0]), np.asarray(class_means[1])
        delta = np.where(is_auc, 0.0, pi1_new * mp + (1 - pi1_new) * mn)
    S = S - np.outer(delta, delta)
    return S * var_mult, delta


def gstar(mu, Sig, dirs, bounds, crit, draws=20000, seed=20261015, grid=None):
    """Draws of the smallest G at which all clauses pass (inf if some clause has non-positive slack), plus the
    joint power computed directly at each G in `grid` (valid also when some slack <= 0)."""
    sd = np.sqrt(np.diag(Sig)); slack = -dirs * (mu - bounds)
    # PSD square root (the clause covariance is exactly singular: U0 and Ucal share decisions, so their accuracy-loss clauses coincide).
    w, V = np.linalg.eigh((Sig + Sig.T) / 2); L = V * np.sqrt(np.clip(w, 0, None))[None, :]
    z = np.random.default_rng(seed).standard_normal((draws, len(mu))) @ L.T
    need = dirs[None, :] * z + crit * sd[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(slack[None, :] > 0, (np.maximum(need, 0) / slack[None, :]) ** 2, np.inf)
    direct = {}
    for G in (grid or []):
        direct[G] = float(np.mean(np.all(need <= slack[None, :] * math.sqrt(G), axis=1)))
    gstar.direct = direct
    return g.max(1), slack, sd


def marginal_power(mu, sd, dirs, bounds, crit, G):
    slack = -dirs * (mu - bounds)
    return stats.norm.cdf(slack * math.sqrt(G) / sd - crit)


def summarize(gs, grid, targets=(0.8, 0.9), direct=None):
    B = len(gs); out = {'power_at': {}, 'N_for': {}}
    for G in grid:
        p = float(direct[G]) if direct and G in direct else float(np.mean(gs <= G))
        out['power_at'][str(G)] = {'power': p, 'mc_se': math.sqrt(max(p * (1 - p), 1e-12) / B)}
    finite = np.isfinite(gs)
    for t in targets:
        if finite.mean() < t:
            out['N_for'][str(t)] = {'N': None, 'status': 'NOT_ACHIEVABLE (some clause has non-positive true slack: power stays at or below the clause level alpha at its boundary and tends to 0 beyond it)'}
            continue
        q = np.quantile(gs, t); k = int(round(t * B)); h = int(math.ceil(1.96 * math.sqrt(B * t * (1 - t))))
        srt = np.sort(gs)
        out['N_for'][str(t)] = {'N': float(q), 'mc_95_interval': [float(srt[max(0, k - h)]), float(srt[min(B - 1, k + h)])], 'status': 'ACHIEVABLE'}
    return out


def run(inputs, outdir, draws=20000, supplement=None):
    f, names, idx, theta, cov, Mpos, Mneg = load(inputs)
    cm = None
    if supplement:
        sup = json.loads(Path(supplement).read_text())
        if sup['names'] != names:
            raise ValueError('supplement names differ')
        cm = (sup['class_mean_phi_pos'], sup['class_mean_phi_neg'])
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    grid = [2234, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    benefits = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]
    brier_shifts = [-0.002, -0.001, 0.0, 0.0005]
    var_mults = [0.8, 1.0, 1.25, 1.5]
    minority = [None, 0.25, 0.45]
    designs = {'IUT_alpha0.025': None, 'SIM_claimK': 'claim', 'SIM_K78': 78}
    pi1 = f['sex_prevalence_pi1']
    rows, marg_rows, bind_rows = [], [], []
    claims = {'A_original_6': claim(idx, 'A'), 'A_plus_RAW_J_NEQ': claim(idx, 'A', ORIGINAL_REQUIRED + ('RAW_J_NEQ',)), 'B_narrow': claim(idx, 'B')}
    for cname, cl in claims.items():
        A = np.array([c[1] for c in cl]); dirs = np.array([c[2] for c in cl], dtype=float); bounds = np.array([c[3] for c in cl])
        names_c = [c[0] for c in cl]; K = len(cl)
        mu_obs = A @ theta
        for b in benefits + ['observed']:
            for shift in brier_shifts:
                for vm in var_mults:
                    for ms in minority:
                        if (vm != 1.0 and ms is not None):
                            continue   # vary one nuisance at a time
                        if (vm != 1.0 or ms is not None) and (shift != 0.0 or b not in (0.020, 0.025, 'observed')):
                            continue
                        mu = mu_obs.copy()
                        for j, nm in enumerate(names_c):
                            if b != 'observed' and ((cname.startswith('A') and nm.startswith('benefit|benefit_vs_') and nm.split('_vs_')[1] in CODES + ('RAW_J_NEQ',) and nm.split('_vs_')[1] in CODES)
                                                    or (cname == 'B_narrow' and nm.startswith('superiority|'))):
                                mu[j] = b
                            if nm.endswith('|occupation|U0|brier_excess') or nm.endswith('|occupation|Ucal|brier_excess'):
                                mu[j] = mu_obs[j] + shift
                        Sb, delta = sigma_for(cov, Mpos, Mneg, pi1, names, ms, vm, cm)
                        Sig = A @ Sb @ A.T
                        mu = mu + A @ delta
                        for dname, dk in designs.items():
                            crit = stats.norm.ppf(1 - 0.025) if dk is None else stats.norm.ppf(1 - 0.05 / (2 * (K if dk == 'claim' else dk)))
                            gs, slack, sd = gstar(mu, Sig, dirs, bounds, crit, draws, grid=grid)
                            sm = summarize(gs, grid, direct=gstar.direct)
                            row = {'claim': cname, 'clauses': K, 'design': dname, 'critical': round(float(crit), 4), 'true_benefit_vs_binding': b,
                                   'occ_brier_shift': shift, 'variance_multiplier': vm, 'minority_share': 'observed' if ms is None else ms,
                                   'N80': sm['N_for']['0.8'].get('N'), 'N80_status': sm['N_for']['0.8']['status'], 'N80_mc95': sm['N_for']['0.8'].get('mc_95_interval'),
                                   'N90': sm['N_for']['0.9'].get('N'), 'N90_status': sm['N_for']['0.9']['status'],
                                   **{f'power_G{G}': round(sm['power_at'][str(G)]['power'], 4) for G in grid}}
                            rows.append(row)
                            if b in ('observed', 0.020) and shift == 0.0 and vm == 1.0 and ms is None:
                                for G in (20000, 100000, 500000):
                                    mp = marginal_power(mu, sd, dirs, bounds, crit, G)
                                    for j, nm in enumerate(names_c):
                                        marg_rows.append({'claim': cname, 'design': dname, 'true_benefit_vs_binding': b, 'G': G, 'clause': nm, 'true_value': float(mu[j]),
                                                          'bound': float(bounds[j]), 'slack': float(slack[j]), 'sd_per_group': float(sd[j]), 'marginal_power': float(mp[j])})
                                single = [((max(0.0, crit * sd[j]) / slack[j]) ** 2 if slack[j] > 0 else float('inf')) for j in range(K)]
                                order = np.argsort(single)[::-1][:6]
                                for j in order:
                                    bind_rows.append({'claim': cname, 'design': dname, 'true_benefit_vs_binding': b, 'clause': names_c[j], 'true_value': float(mu[j]),
                                                      'slack': float(slack[j]), 'sd_per_group': float(sd[j]), 'N_for_50pct_marginal': single[j]})
    # Privacy-only sizing (shows why the study must NOT be sized from privacy alone) and an illustrative
    # beyond-cap utility scenario (added for illustration; not part of the locked grid).
    extra = []
    for cname, cl in claims.items():
        A = np.array([c[1] for c in cl]); dirs = np.array([c[2] for c in cl], dtype=float); bounds = np.array([c[3] for c in cl])
        names_c = [c[0] for c in cl]; mu_obs = A @ theta; Sig_full = A @ cov @ A.T
        keep = [j for j, nm in enumerate(names_c) if not nm.startswith('utility')]
        for b in benefits + ['observed']:
            mu = mu_obs.copy()
            for j, nm in enumerate(names_c):
                if b != 'observed' and ((cname.startswith('A') and nm.startswith('benefit|benefit_vs_') and nm.split('_vs_')[1] in CODES) or (cname == 'B_narrow' and nm.startswith('superiority|'))):
                    mu[j] = b
            for dname, crit in (('IUT_alpha0.025', stats.norm.ppf(0.975)), ('SIM_claimK', stats.norm.ppf(1 - 0.05 / (2 * len(cl))))):
                gs, _, _ = gstar(mu[keep], Sig_full[np.ix_(keep, keep)], dirs[keep], bounds[keep], crit, draws, grid=grid)
                sm = summarize(gs, grid, direct=gstar.direct)
                extra.append({'analysis': 'PRIVACY_CLAUSES_ONLY', 'claim': cname, 'design': dname, 'true_benefit_vs_binding': b,
                              'N80': sm['N_for']['0.8'].get('N'), 'N80_status': sm['N_for']['0.8']['status'], 'N90': sm['N_for']['0.9'].get('N')})
        mu = mu_obs.copy()
        for j, nm in enumerate(names_c):
            if nm.endswith('|occupation|U0|brier_excess') or nm.endswith('|occupation|Ucal|brier_excess'):
                mu[j] = mu_obs[j] + 0.001
            if cname.startswith('A') and nm.startswith('benefit|benefit_vs_') and nm.split('_vs_')[1] in CODES:
                mu[j] = 0.025
        gs, _, _ = gstar(mu, Sig_full, dirs, bounds, stats.norm.ppf(0.975), draws, grid=grid)
        sm = summarize(gs, grid, direct=gstar.direct)
        extra.append({'analysis': 'ILLUSTRATIVE_BEYOND_CAP (occupation Brier +0.001, seed-1 vs U0 true excess 0.00522 > 0.005)', 'claim': cname, 'design': 'IUT_alpha0.025',
                      'true_benefit_vs_binding': 0.025 if cname.startswith('A') else 'observed', 'N80': sm['N_for']['0.8'].get('N'), 'N80_status': sm['N_for']['0.8']['status'],
                      'power_G1000000': sm['power_at']['1000000']['power']})

    def write(path, rs):
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with Path(path).open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rs]
    write(out / 'JOINT_POWER_TABLES.csv', rows); write(out / 'PRIVACY_ONLY_AND_BEYOND_CAP.csv', extra); write(out / 'MARGINAL_POWER.csv', marg_rows); write(out / 'BINDING_CLAUSES.csv', bind_rows)
    meta = {'inputs_sha256': __import__('hashlib').sha256(Path(inputs).read_bytes()).hexdigest(), 'draws': draws, 'seed': 20261015, 'grid_G': grid,
            'benefits': benefits, 'brier_shifts': brier_shifts, 'variance_multipliers': var_mults, 'minority_shares': minority,
            'observed_minority_share': (1 - pi1) if pi1 >= 0.5 else pi1, 'prevalence_mean_shift': cm is not None, 'designs': list(designs), 'claims': {k: [c[0] for c in v] for k, v in claims.items()},
            'model': __doc__}
    (out / 'POWER_META.json').write_text(json.dumps(meta, indent=2))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--inputs', required=True); p.add_argument('--out', required=True); p.add_argument('--draws', type=int, default=20000); p.add_argument('--supplement')
    a = p.parse_args(); run(a.inputs, a.out, a.draws, a.supplement)


def assurance(inputs, out_json, claim_kind='B', n_mu=2000, n_z=2000, G_values=(316601, 1000000, 5000000), seed=20261017):
    """Flat-prior assurance (planning sensitivity requested by the statistics review): draw the true clause means
    from N(mu_hat, Sigma/G_INNER) -- the flat-prior posterior around the development estimates -- and report
    P(every clause truly passes) and the average joint IUT power at each G. Selection of the candidate makes even
    these optimistic."""
    f, names, idx, theta, cov, Mpos, Mneg = load(inputs)
    cl = claim(idx, claim_kind)
    A = np.array([c[1] for c in cl]); dirs = np.array([c[2] for c in cl], dtype=float); bounds = np.array([c[3] for c in cl])
    S = A @ cov @ A.T; mu_hat = A @ theta; G0 = f['G']
    w, V = np.linalg.eigh((S + S.T) / 2); L = V * np.sqrt(np.clip(w, 0, None))[None, :]
    rng = np.random.default_rng(seed)
    mus = mu_hat[None, :] + (rng.standard_normal((n_mu, len(mu_hat))) @ L.T) / math.sqrt(G0)
    zs = rng.standard_normal((n_z, len(mu_hat))) @ L.T
    sd = np.sqrt(np.diag(S)); crit = stats.norm.ppf(0.975)
    true_pass = np.all(-dirs[None, :] * (mus - bounds[None, :]) > 0, axis=1)
    res = {'claim': claim_kind, 'P_all_clauses_truly_pass': float(true_pass.mean()), 'n_mu': n_mu, 'n_z': n_z, 'G_inner': G0, 'average_power': {}}
    for G in G_values:
        pw = []
        for m in mus:
            slack = -dirs * (m - bounds)
            need = dirs[None, :] * zs + crit * sd[None, :]
            pw.append(np.mean(np.all(need <= slack[None, :] * math.sqrt(G), axis=1)))
        res['average_power'][str(G)] = float(np.mean(pw))
    res['note'] = assurance.__doc__
    Path(out_json).write_text(json.dumps(res, indent=2))
    return res

"""D2 data-free check 1: linear-Gaussian model of the coalition (no real data).

Purpose. Check, with exact covariance algebra (and a Monte Carlo cross-check), the three claims PROPOSAL.md relies on:
  (A) amplification law: pair linear information W_pair = W_occ + (c1 - b'c2)^2 / Var(f1 | f2)   (Gaussian identity);
  (B) driver vs projection: the calibrated task cost of removing ALL linear S-information from a release F with a
      counter-shift driven by the sex propensity is ||c||^2 / Var(pi); a deterministic post-processing (projection) of F
      costs ||c||^2 / Var(E[pi | F]);
  (C) redundancy vs minimality: with the income release frozen, an occupation counter-shift that minimises the PAIR
      information (ALIGN) beats one that minimises occupation-only information (MIN) at equal occupation cost; the
      model is calibrated to published Gaussian-equivalent d'^2 values (EVIDENCE_PACK / SERVICE_TABLE, fitted readers).

Conventions. Coordinates F = (f1, g, h): f1 income probability; g occupation-probability coordinate along the
income-predictive direction (Corr(f1, g)^2 = R2); h S-relevant occupation coordinate uncorrelated with f1 and g.
pi = P(S = 1 | X); c = Cov(F, pi) = Cov(F, S). Linear information of a view = c_v' Sigma_v^{-1} c_v = Var(E_lin[pi | view]).
Gaussian-equivalent d'^2 = W / (p q)^2 with p = 0.47; AUC = Phi(sqrt(d'^2) / sqrt(2)). The driver u is the part of pi
orthogonal to F (variance V = Var(pi) - W_pair). A counter-shift adds k * u; calibrated releases are E[target | view].
Costs are summed squared error in probability units (summed Brier excess at the population level).
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
from statistics import NormalDist

import numpy as np

PQ = 0.47 * 0.53
ND = NormalDist()


def d2_from_auc(a):
    return 2.0 * ND.inv_cdf(a) ** 2


def auc_from_d2(d2):
    return ND.cdf(math.sqrt(max(d2, 0.0)) / math.sqrt(2.0))


def W_to_d2(W):
    return W / PQ ** 2


def build(R2, alpha, d2X, d2inc, d2occ, s1=0.071, sg=0.03, sh=0.02):
    """Covariances of (f1, g, h) and c = Cov(F, pi) matching the given Gaussian-equivalent d'^2 values."""
    cov1g = math.sqrt(R2 * s1 * sg)
    Sig = np.array([[s1, cov1g, 0.0], [cov1g, sg, 0.0], [0.0, 0.0, sh]])
    c1 = -math.sqrt(d2inc) * PQ * math.sqrt(s1)
    cg = alpha * c1 * sg / cov1g                      # occupation 'explains' alpha of c1 through g
    rest = d2occ * PQ ** 2 - cg ** 2 / sg
    if rest < 0:
        raise ValueError('alpha too large for the occupation d2')
    ch = math.sqrt(rest * sh)
    c = np.array([c1, cg, ch])
    W_pair = float(c @ np.linalg.solve(Sig, c))
    varpi = d2X * PQ ** 2
    V = varpi - W_pair
    if V <= 0:
        raise ValueError('d2X must exceed the pair information')
    return {'Sig': Sig, 'c': c, 'V': V, 'varpi': varpi, 'W_pair': W_pair}


def view_info(M, k, idx):
    """Linear information of view idx after the counter-shift k*u (u orthogonal to F, variance V)."""
    Sig, c, V = M['Sig'], M['c'], M['V']
    cn = c + V * k
    Sn = Sig + V * np.outer(k, k)
    idx = list(idx)
    return float(cn[idx] @ np.linalg.solve(Sn[np.ix_(idx, idx)], cn[idx]))


def calibrated_cost(M, k, target, view):
    """E||T - E[T | view]||^2 where T = original coordinates `target` and view = shifted coordinates `view`."""
    Sig, V = M['Sig'], M['V']
    Sn = Sig + V * np.outer(k, k)                     # covariance of shifted coordinates
    C = Sig[np.ix_(target, view)]                     # Cov(original target, shifted view) (u is orthogonal to F)
    S = Sn[np.ix_(view, view)]
    return float(np.trace(Sig[np.ix_(target, target)]) - np.trace(C @ np.linalg.solve(S, C.T)))


def check_A():
    """Amplification identity on random positive-definite instances (exact algebra)."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        A = rng.normal(size=(5, 5)); Sig = A @ A.T + 0.1 * np.eye(5); c = rng.normal(size=5)
        W = c @ np.linalg.solve(Sig, c)
        S11, S12, S22 = Sig[0, 0], Sig[0, 1:], Sig[1:, 1:]
        c1, c2 = c[0], c[1:]
        W2 = c2 @ np.linalg.solve(S22, c2)
        b = np.linalg.solve(S22, S12)                 # regression of f1 on the occupation block
        part = (c1 - b @ c2) ** 2 / (S11 - S12 @ b)
        worst = max(worst, abs(W - (W2 + part)) / W)
    return {'max_relative_error': worst, 'pass': worst < 1e-10}


def check_B(n_mc=400_000):
    """Full linear removal: driver cost ||c||^2/Var(pi) vs projection cost ||c||^2/Var(E[pi|F]); MC cross-check."""
    rng = np.random.default_rng(2)
    out = []
    for trial in range(3):
        A = rng.normal(size=(3, 3)) * 0.1; Sig = A @ A.T + 0.01 * np.eye(3)
        c = rng.normal(size=3) * 0.01
        W = float(c @ np.linalg.solve(Sig, c)); V = 0.5 * W * (trial + 1)
        k = -c / V                                     # residual-driver shift that zeroes Cov(F', pi)
        M = {'Sig': Sig, 'c': c, 'V': V}
        cost_closed = calibrated_cost(M, k, [0, 1, 2], [0, 1, 2])
        cost_formula = float(c @ c) / (W + V)
        W_after = view_info(M, k, [0, 1, 2])
        # Monte Carlo: F ~ N(0, Sig), u ~ N(0, V) independent; pi = c'Sig^{-1}F + u (linear model)
        L = np.linalg.cholesky(Sig)
        F = rng.normal(size=(n_mc, 3)) @ L.T
        u = rng.normal(size=n_mc) * math.sqrt(V)
        pi = F @ np.linalg.solve(Sig, c) + u
        Fp = F + np.outer(u, k)
        B, *_ = np.linalg.lstsq(Fp, F, rcond=None)
        mc_cost = float(np.mean(np.sum((F - Fp @ B) ** 2, 1)))
        b, *_ = np.linalg.lstsq(Fp, pi, rcond=None)
        mc_W_after = float(np.var(Fp @ b))
        proj_cost = float(c @ c) / W                   # V -> 0 limit: project out the discriminant direction
        out.append({'W': W, 'V': V, 'cost_closed': cost_closed, 'cost_formula_c2_over_varpi': cost_formula,
                    'mc_cost': mc_cost, 'W_after_closed': W_after, 'W_after_mc': mc_W_after,
                    'projection_cost_c2_over_W': proj_cost, 'driver_gain_factor': proj_cost / cost_formula})
    ok = all(abs(r['cost_closed'] - r['cost_formula_c2_over_varpi']) < 1e-12 + 1e-9 * r['cost_formula_c2_over_varpi']
             and abs(r['mc_cost'] / r['cost_closed'] - 1) < 0.02 and r['W_after_closed'] < 1e-12 for r in out)
    return {'trials': out, 'pass': ok}


def _batch_info(c, Sig, V, K, idx):
    """Vectorised linear information of view idx for shifts K (N x 3)."""
    idx = list(idx)
    cn = c[None, :] + V * K                                   # (N, 3)
    Sn = Sig[None, :, :] + V * K[:, :, None] * K[:, None, :]  # (N, 3, 3)
    cv = cn[:, idx]
    Sv = Sn[:, idx][:, :, idx]
    x = np.linalg.solve(Sv, cv[:, :, None])[:, :, 0]
    return np.sum(cv * x, 1)


def _batch_cost(Sig, V, K, target, view):
    Sn = Sig[None, :, :] + V * K[:, :, None] * K[:, None, :]
    C = np.broadcast_to(Sig[np.ix_(target, view)], (K.shape[0], len(target), len(view)))
    S = Sn[:, view][:, :, view]
    X = np.linalg.solve(S, np.transpose(C, (0, 2, 1)))        # (N, view, target)
    return np.trace(Sig[np.ix_(target, target)]) - np.einsum('nij,nji->n', C, X)


def frontier(M, budgets, grid=np.linspace(-12, 12, 481), income_aware=True):
    """MIN vs ALIGN occupation counter-shifts (income frozen) on a dense grid of shifts (k_g, k_h)."""
    Sig, c, V = M['Sig'], M['c'], M['V']
    kg, kh = np.meshgrid(grid, grid, indexing='ij')
    K = np.stack([np.zeros(kg.size), kg.ravel(), kh.ravel()], 1)
    view = [0, 1, 2] if income_aware else [1, 2]
    cost = _batch_cost(Sig, V, K, [1, 2], view)
    Wp = _batch_info(c, Sig, V, K, [0, 1, 2])
    Wo = _batch_info(c, Sig, V, K, [1, 2])
    rows = []
    for b in budgets:
        ok = cost <= b + 1e-15
        r = {'budget': float(b)}
        for arm, obj in (('MIN', Wo), ('ALIGN', Wp)):
            j = int(np.argmin(np.where(ok, obj, np.inf)))
            r[arm] = {'pair_auc': auc_from_d2(W_to_d2(Wp[j])), 'occ_auc': auc_from_d2(W_to_d2(Wo[j])),
                      'cost': float(cost[j]), 'k': [float(K[j, 1]), float(K[j, 2])]}
        rows.append(r)
    return rows


def scenario(R2, alpha, d2X, d2inc, d2occ, label, local_occ_auc=0.582, target_pair_auc=0.606):
    M = build(R2, alpha, d2X, d2inc, d2occ)
    base_pair = auc_from_d2(W_to_d2(M['W_pair']))
    budgets = list(np.linspace(0.0, 0.03, 61))
    fr = frontier(M, budgets)
    # unit of cost: the MIN cost that brings occupation-only AUC down to LOCAL_TREE's level (0.582)
    unit = next((r['budget'] for r in fr if r['MIN']['occ_auc'] <= local_occ_auc), None)
    at_unit = next((r for r in fr if r['budget'] == unit), None)
    reach = next((r['budget'] for r in fr if r['ALIGN']['pair_auc'] <= target_pair_auc), None)
    full_driver = float(M['c'][1:] @ M['c'][1:])      # reference scale (occupation part of ||c||^2)
    return {'label': label, 'R2': R2, 'alpha': alpha, 'd2X': d2X, 'auc_X': auc_from_d2(d2X), 'd2inc': d2inc,
            'd2occ': d2occ, 'task_pair_auc': base_pair, 'V_over_varpi': M['V'] / M['varpi'],
            'cost_unit_MIN_to_occ_0582': unit,
            'MIN_at_unit': at_unit['MIN'] if at_unit else None, 'ALIGN_at_unit': at_unit['ALIGN'] if at_unit else None,
            'ALIGN_cost_to_reach_0606': reach,
            'ALIGN_cost_to_reach_0606_in_units': (reach / unit) if (reach is not None and unit) else None,
            'frontier_every_5th': fr[::5], 'occ_c2': full_driver}


def main():
    res = {'A_amplification_identity': check_A(), 'B_driver_vs_projection': check_B()}
    # Published Gaussian-equivalent anchors (selected readers, DEV_REPORT, seed 0; EVIDENCE_PACK / SERVICE_TABLE):
    anchors = {k: d2_from_auc(v) for k, v in {
        'T_TREE_income': 0.569, 'T_TREE_occ': 0.650, 'T_TREE_pair': 0.664,
        'LOCAL_income': 0.558, 'LOCAL_occ': 0.582, 'LOCAL_pair': 0.657,
        'JOINT_income': 0.535, 'JOINT_occ': 0.560, 'JOINT_pair': 0.603,
        'H_LJ_pair': 0.641, 'H_JL_pair': 0.625, 'NEURAL_income': 0.546, 'NEURAL_occ': 0.622, 'NEURAL_pair': 0.640,
        'TASK_REFIT_income': 0.572, 'TASK_REFIT_occ': 0.650, 'TASK_REFIT_pair': 0.664, 'U0_income': 0.560,
        'U0_occ': 0.641, 'U0_pair': 0.654, 'target_material': 0.606, 'target_plain': 0.628}.items()}
    a = anchors
    amp = {
        'LOCAL': (a['LOCAL_pair'] - a['LOCAL_occ']) / a['LOCAL_income'],
        'JOINT': (a['JOINT_pair'] - a['JOINT_occ']) / a['JOINT_income'],
        'H_LJ (LOCAL inc, JOINT occ)': (a['H_LJ_pair'] - a['JOINT_occ']) / a['LOCAL_income'],
        'H_JL (JOINT inc, LOCAL occ)': (a['H_JL_pair'] - a['LOCAL_occ']) / a['JOINT_income'],
        'NEURAL_RAW': (a['NEURAL_pair'] - a['NEURAL_occ']) / a['NEURAL_income'],
        'TASK_REFIT': (a['TASK_REFIT_pair'] - a['TASK_REFIT_occ']) / a['TASK_REFIT_income'],
        'T_TREE': (a['T_TREE_pair'] - a['T_TREE_occ']) / a['T_TREE_income'],
        'U0': (a['U0_pair'] - a['U0_occ']) / a['U0_income']}
    res['published_d2_anchors'] = anchors
    res['published_amplification_(pair-occ)/inc'] = amp
    # Calibration check: does the model's MIN arm reproduce LOCAL_TREE's pair from LOCAL's income and occupation levels?
    scen = []
    for R2 in (0.70, 0.82, 0.90):
        for alpha in (0.4, 0.6):
            for d2X in (0.40, 0.49, 0.64):
                try:
                    scen.append(scenario(R2, alpha, d2X, a['T_TREE_income'], a['T_TREE_occ'],
                                         f'T_TREE base R2={R2} alpha={alpha} d2X={d2X}'))
                except ValueError as e:
                    scen.append({'label': f'R2={R2} alpha={alpha} d2X={d2X}', 'error': str(e)})
    res['scenarios_income_frozen_at_T_TREE'] = scen
    # Calibration replay: income at LOCAL's level, occupation MIN to LOCAL's occupation level -> predicted pair
    rep = []
    for R2 in (0.70, 0.82, 0.90):
        for d2X in (0.40, 0.49, 0.64):
            try:
                M = build(R2, 0.6, d2X, a['LOCAL_income'], a['T_TREE_occ'])
                fr = frontier(M, list(np.linspace(0.0, 0.03, 121)))
                hit = next((r for r in fr if r['MIN']['occ_auc'] <= 0.582), None)
                rep.append({'R2': R2, 'd2X': d2X, 'MIN_pair_auc_at_occ_0582': hit['MIN']['pair_auc'] if hit else None,
                            'ALIGN_pair_auc_same_cost': hit['ALIGN']['pair_auc'] if hit else None,
                            'cost': hit['budget'] if hit else None, 'published_LOCAL_pair': 0.657})
            except ValueError as e:
                rep.append({'R2': R2, 'd2X': d2X, 'error': str(e)})
    res['calibration_replay_LOCAL'] = rep
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

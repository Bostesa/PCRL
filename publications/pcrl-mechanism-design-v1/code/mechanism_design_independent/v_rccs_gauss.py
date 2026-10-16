"""Verifier checks for designer 2 (RCCS) - linear-information algebra, the Gaussian model and published-AUC readings.

Synthetic / published aggregates only. Independent implementation from the proposal's description (not its code).
1. Exact algebra on random instances: amplification identity W_pair = W_occ + (c1 - b'c2)^2 / Var(f1|f2);
   counter-shift costs: calibrated ||c||^2/Var(pi), RAW (unrecalibrated) ||c||^2/V, projection ||c||^2/W.
2. The designer's Gaussian model (R2, alpha, d2X grid; income frozen at T_TREE) under four cost/view conventions:
   A  designer convention: cost = income-aware linear recalibration E||T - E_lin[T | f1, g', h']||^2, but the
      occupation-only information is that of the UNcalibrated shifted coordinates (g', h');
   B  consistent: same recalibrated release, own-view information of that release;
   C  occupation-only recalibration (release E_lin[T | g', h']);
   D  RAW: the literal RCCS map (no recalibration step): cost = V ||k||^2.
   Reported: unit (MIN cost to bring occupation-only AUC to 0.582), ALIGN(=RCCS target) cost to reach pair 0.606 in
   units, and pair AUC at one unit, per convention; plus D priced in convention-A units (LOCAL-anchored).
3. Published AUCs (EVIDENCE_PACK table A): Gaussian-equivalent d'^2, the amplification ratio
   (d'2_pair - d'2_occ)/d'2_inc for BOTH seeds, its sampling spread (Hanley-McNeil SEs at DEV_REPORT size), the
   implied R^2 under c2 = 0 and under misalignment, the 'share of pair signal that is occupation-given-income'
   for every service, and reader-class sensitivity (linear-only readers) from published reader matrices.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_RCCS_GAUSS_V1.json'
ND = NormalDist()
P = 0.47
PQ = P * (1 - P)


def d2(auc):
    return 2.0 * ND.inv_cdf(auc) ** 2


def auc_of_d2(x):
    return ND.cdf(math.sqrt(max(x, 0.0) / 2.0))


def W_of_auc(a):
    return d2(a) * PQ ** 2


# ---------------------------------------------------------------- 1. exact algebra
def algebra_checks(seed=3):
    rng = np.random.default_rng(seed)
    worst_amp = 0.0
    for _ in range(500):
        d = rng.integers(2, 7)
        A = rng.normal(size=(d, d))
        S = A @ A.T + 0.05 * np.eye(d)
        c = rng.normal(size=d)
        W = c @ np.linalg.solve(S, c)
        S22 = S[1:, 1:]
        S21 = S[1:, 0]
        b = np.linalg.solve(S22, S21)              # regression of f1 on f2
        part = (c[0] - b @ c[1:]) ** 2 / (S[0, 0] - S21 @ b)
        W2 = c[1:] @ np.linalg.solve(S22, c[1:])
        worst_amp = max(worst_amp, abs(W - W2 - part) / W)
    # costs
    worst_cal = worst_raw = worst_proj = 0.0
    for _ in range(200):
        d = rng.integers(1, 6)
        A = rng.normal(size=(d, d))
        S = A @ A.T + 0.05 * np.eye(d)
        c = rng.normal(size=d)
        W = c @ np.linalg.solve(S, c)
        V = rng.uniform(0.05, 3.0) * W
        k = -c / V                                     # zero Cov(F', pi)
        Sp = S + V * np.outer(k, k)
        cal = np.trace(S) - np.trace(S @ np.linalg.solve(Sp, S))
        worst_cal = max(worst_cal, abs(cal - c @ c / (V + W)) / (c @ c / (V + W)))
        raw = V * k @ k
        worst_raw = max(worst_raw, abs(raw - c @ c / V) / (c @ c / V))
        # best linear post-processing that zeroes linear information: project out the whitened discriminant
        L = np.linalg.cholesky(S)
        a = np.linalg.solve(L, c)                      # Cov(Z, pi) for Z = L^{-1}F (white)
        Pm = np.eye(d) - np.outer(a, a) / (a @ a)
        resid = L @ (np.eye(d) - Pm)                   # F - E[F | Pm Z] = L (I - Pm) Z
        proj = np.trace(resid @ resid.T)
        worst_proj = max(worst_proj, abs(proj - c @ c / W) / (c @ c / W))
    return {'amplification_identity_max_rel_err': worst_amp, 'calibrated_cost_c2_over_varpi_max_rel_err': worst_cal,
            'raw_cost_c2_over_V_max_rel_err': worst_raw, 'projection_cost_c2_over_W_max_rel_err': worst_proj,
            'note': 'raw/calibrated = (V+W)/V; driver beats projection on RAW cost only if V > W'}


# ---------------------------------------------------------------- 2. Gaussian model
def build(R2, alpha, d2X, d2inc, d2occ, s1=0.071, sg=0.03, sh=0.02):
    cov1g = math.sqrt(R2 * s1 * sg)
    S = np.array([[s1, cov1g, 0.0], [cov1g, sg, 0.0], [0.0, 0.0, sh]])
    c1 = -math.sqrt(d2inc) * PQ * math.sqrt(s1)
    cg = alpha * c1 * sg / cov1g
    rest = d2occ * PQ ** 2 - cg ** 2 / sg
    if rest < 0:
        return None
    c = np.array([c1, cg, math.sqrt(rest * sh)])
    Wp = c @ np.linalg.solve(S, c)
    V = d2X * PQ ** 2 - Wp
    if V <= 0:
        return None
    return S, c, V


def grid_quantities(S, c, V, grid):
    kg, kh = np.meshgrid(grid, grid, indexing='ij')
    K = np.stack([np.zeros(kg.size), kg.ravel(), kh.ravel()], 1)          # (N, 3); income never shifted
    cn = c[None, :] + V * K
    Sn = S[None] + V * K[:, :, None] * K[:, None, :]

    def info(cv, Sv):
        x = np.linalg.solve(Sv, cv[..., None])[..., 0]
        return np.sum(cv * x, -1)
    Wp = info(cn, Sn)
    Wo_uncal = info(cn[:, 1:], Sn[:, 1:, 1:])
    T = [1, 2]
    # A/B: income-aware recalibration  R2 = C S'^{-1} v, v = (f1, g', h'), C = Cov((g,h), v) = S[T, :]
    C = np.broadcast_to(S[T, :], (K.shape[0], 2, 3))
    X = np.linalg.solve(Sn, np.transpose(C, (0, 2, 1)))                 # S'^{-1} C^T  (N, 3, 2)
    CSC = np.einsum('nij,njk->nik', C, X)                                # C S'^{-1} C^T (N, 2, 2) = Cov(R2)
    cost_cal_inc = np.trace(S[np.ix_(T, T)]) - np.trace(CSC, axis1=1, axis2=2)
    cR = np.einsum('nji,nj->ni', X, cn)                                  # Cov(R2, pi) = C S'^{-1} c'
    Wo_cal_inc = info(cR, CSC)
    # C: occupation-only recalibration
    C2 = np.broadcast_to(S[np.ix_(T, T)], (K.shape[0], 2, 2))
    X2 = np.linalg.solve(Sn[:, 1:, 1:], np.transpose(C2, (0, 2, 1)))
    cost_cal_occ = np.trace(S[np.ix_(T, T)]) - np.trace(np.einsum('nij,njk->nik', C2, X2), axis1=1, axis2=2)
    # D: raw
    cost_raw = V * (K[:, 1] ** 2 + K[:, 2] ** 2)
    return {'Wp': Wp, 'Wo_uncal': Wo_uncal, 'Wo_cal_inc': Wo_cal_inc, 'cost_cal_inc': cost_cal_inc,
            'cost_cal_occ': cost_cal_occ, 'cost_raw': cost_raw}


def convention_summary(Wp, Wo, cost, occ_target=0.582, pair_target=0.606, unit=None):
    Wo_t = W_of_auc(occ_target)
    Wp_t = W_of_auc(pair_target)
    ok_min = Wo <= Wo_t
    own_unit = float(np.min(cost[ok_min])) if ok_min.any() else None
    u = own_unit if unit is None else unit
    ok_al = Wp <= Wp_t
    reach = float(np.min(cost[ok_al])) if ok_al.any() else None
    res = {'unit_MIN_to_occ_0582': own_unit, 'unit_used': u, 'ALIGN_cost_to_pair_0606': reach,
           'ALIGN_reach_in_units': (reach / u) if (reach is not None and u) else None}
    if u:
        feas = cost <= u * (1 + 1e-12)
        res['ALIGN_pair_auc_at_1_unit'] = auc_of_d2(float(np.min(Wp[feas])) / PQ ** 2)
        j = np.argmin(np.where(feas, Wo, np.inf))
        res['MIN_pair_auc_at_1_unit'] = auc_of_d2(float(Wp[j]) / PQ ** 2)
        res['MIN_occ_auc_at_1_unit'] = auc_of_d2(float(Wo[j]) / PQ ** 2)
    return res


def designer_budget_grid(Wp, Wo, cost):
    """Reproduce the proposal's discretised reading (budgets 0..0.03 step 0.0005; first budget that reaches)."""
    budgets = np.linspace(0.0, 0.03, 61)
    unit = reach = None
    rows = {}
    for b in budgets:
        ok = cost <= b + 1e-15
        if not ok.any():
            continue
        jm = np.argmin(np.where(ok, Wo, np.inf))
        ja = np.argmin(np.where(ok, Wp, np.inf))
        occ = auc_of_d2(Wo[jm] / PQ ** 2)
        pair_al = auc_of_d2(Wp[ja] / PQ ** 2)
        rows[float(b)] = (occ, auc_of_d2(Wp[jm] / PQ ** 2), pair_al)
        if unit is None and occ <= 0.582:
            unit = float(b)
        if reach is None and pair_al <= 0.606:
            reach = float(b)
    out = {'unit': unit, 'reach': reach, 'reach_in_units': (reach / unit) if (unit and reach is not None) else None}
    if unit is not None:
        out['MIN_pair_at_unit'] = rows[unit][1]
        out['ALIGN_pair_at_unit'] = rows[unit][2]
    return out


def gaussian_model():
    grid = np.linspace(-12, 12, 481)
    a_inc, a_occ = d2(0.569), d2(0.650)
    scen = []
    for R2 in (0.70, 0.82, 0.90):
        for alpha in (0.4, 0.6):
            for d2X in (0.40, 0.49, 0.64):
                m = build(R2, alpha, d2X, a_inc, a_occ)
                if m is None:
                    scen.append({'R2': R2, 'alpha': alpha, 'd2X': d2X, 'valid': False})
                    continue
                S, c, V = m
                q = grid_quantities(S, c, V, grid)
                A = convention_summary(q['Wp'], q['Wo_uncal'], q['cost_cal_inc'])
                rec = {'R2': R2, 'alpha': alpha, 'd2X': d2X, 'valid': True, 'V_over_varpi': V / (d2X * PQ ** 2),
                       'task_pair_auc': auc_of_d2(float(c @ np.linalg.solve(S, c)) / PQ ** 2),
                       'A_designer_convention': A,
                       'A_designer_budget_grid': designer_budget_grid(q['Wp'], q['Wo_uncal'], q['cost_cal_inc']),
                       'B_consistent_income_aware': convention_summary(q['Wp'], q['Wo_cal_inc'], q['cost_cal_inc']),
                       'C_occupation_only_recalibration': convention_summary(q['Wp'], q['Wo_uncal'], q['cost_cal_occ']),
                       'D_raw_literal_map': convention_summary(q['Wp'], q['Wo_uncal'], q['cost_raw']),
                       'D_raw_in_A_units': convention_summary(q['Wp'], q['Wo_uncal'], q['cost_raw'], unit=A['unit_MIN_to_occ_0582'])}
                scen.append(rec)
    # calibration replay (income at LOCAL's 0.558, occupation from T_TREE's 0.650; MIN to 0.582 -> pair)
    rep = []
    for R2 in (0.70, 0.82, 0.90):
        for d2X in (0.40, 0.49, 0.64):
            m = build(R2, 0.6, d2X, d2(0.558), a_occ)
            if m is None:
                rep.append({'R2': R2, 'd2X': d2X, 'valid': False})
                continue
            S, c, V = m
            q = grid_quantities(S, c, V, np.linspace(-12, 12, 481))
            out = {'R2': R2, 'd2X': d2X}
            for lab, Wo, cost in (('A', q['Wo_uncal'], q['cost_cal_inc']), ('B', q['Wo_cal_inc'], q['cost_cal_inc']),
                                  ('D', q['Wo_uncal'], q['cost_raw'])):
                s = convention_summary(q['Wp'], Wo, cost)
                out[lab + '_MIN_pair_at_occ_0582'] = s.get('MIN_pair_auc_at_1_unit')
            rep.append(out)
    return {'scenarios': scen, 'calibration_replay_LOCAL_published_pair_0657': rep}


# ---------------------------------------------------------------- 3. published AUC readings
SERV = {  # (income, occupation, pair) seed0 / seed1, EVIDENCE_PACK table A
    'T_TREE': ((0.569, 0.650, 0.664), (0.568, 0.649, 0.664)),
    'TASK_REFIT': ((0.572, 0.650, 0.664), (0.567, 0.645, 0.666)),
    'U0': ((0.560, 0.641, 0.654), (0.567, 0.638, 0.649)),
    'LOCAL_TREE': ((0.558, 0.582, 0.657), (0.549, 0.584, 0.649)),
    'JOINT_TREE': ((0.535, 0.560, 0.603), (0.537, 0.583, 0.606)),
    'NEURAL_RAW': ((0.546, 0.622, 0.640), (0.551, 0.623, 0.638)),
    'H_LJ': ((0.558, 0.560, 0.641), (0.549, 0.583, 0.643)),
    'H_JL': ((0.535, 0.582, 0.625), (0.537, 0.584, 0.625)),
}
LINEAR_ONLY = {  # published reader matrices (DEV_REPORT for LOCAL; ATTACK_VALIDATION for NEURAL_RAW seed 0)
    'LOCAL_TREE s0 OUT_LR (DEV)': (0.5583, 0.5471, 0.6427), 'LOCAL_TREE s0 LOG_LR (DEV)': (0.5582, 0.5618, 0.6460),
    'LOCAL_TREE s1 OUT_LR (DEV)': (0.5573, 0.5614, 0.6418), 'LOCAL_TREE s1 LOG_LR (DEV)': (0.5565, 0.5727, 0.6421),
    'NEURAL_RAW s0 OUT_LR (AV)': (0.539056, 0.597073, 0.609817), 'NEURAL_RAW s0 LOG_LR (AV)': (0.539082, 0.616082, 0.618266),
    'NEURAL_RAW s0 LOG_MLP (AV)': (0.539845, 0.623589, 0.628773),
}


def hm_se(auc, n1, n0):
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))


def published(seed=5, B=20000):
    rng = np.random.default_rng(seed)
    n = 5943
    n1 = round(P * n)
    n0 = n - n1
    out = {}
    for name, seeds in SERV.items():
        for si, (ai, ao, ap) in enumerate(seeds):
            di, do, dp = d2(ai), d2(ao), d2(ap)
            amp = (dp - do) / di
            sims = []
            for _ in range(B):
                xi = min(max(ai + rng.normal() * hm_se(ai, n1, n0), 0.5005), 0.99)
                xo = min(max(ao + rng.normal() * hm_se(ao, n1, n0), 0.5005), 0.99)
                xp = min(max(ap + rng.normal() * hm_se(ap, n1, n0), 0.5005), 0.99)
                sims.append((d2(xp) - d2(xo)) / d2(xi))
            lo, med, hi = np.percentile(sims, [2.5, 50, 97.5])
            rec = {'d2_inc': di, 'd2_occ': do, 'd2_pair': dp, 'amplification': amp,
                   'amplification_95pct_indep_errors': [float(lo), float(med), float(hi)],
                   'share_pair_signal_not_in_income_alone(1-d2inc/d2pair)': 1 - di / dp,
                   'share_pair_signal_occupation_alone(d2occ/d2pair)': do / dp}
            if amp > 1:
                rec['implied_R2_if_c2_zero'] = 1 - 1 / amp
                rec['implied_R2_if_bc2_over_c1=+0.3'] = 1 - 0.49 / amp
                rec['implied_R2_if_bc2_over_c1=-0.3'] = 1 - 1.69 / amp
            out[f'{name}|s{si}'] = rec
    lin = {}
    for name, (ai, ao, ap) in LINEAR_ONLY.items():
        lin[name] = {'amplification': (d2(ap) - d2(ao)) / d2(ai), 'share_not_in_income_alone': 1 - d2(ai) / d2(ap)}
    return {'services': out, 'reader_class_sensitivity': lin, 'n1_n0_used': [n1, n0],
            'note': 'independent normal errors per AUC (ignores positive pair/occupation correlation); AUCs rounded to 3 dp'}


def main():
    res = {'algebra': algebra_checks()}
    print(json.dumps(res['algebra']), flush=True)
    res['gaussian_model'] = gaussian_model()
    for s in res['gaussian_model']['scenarios']:
        if s.get('valid'):
            print(s['R2'], s['alpha'], s['d2X'], 'V/varpi=%.3f' % s['V_over_varpi'],
                  'A grid:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s['A_designer_budget_grid'].items()},
                  '| B reach units:', s['B_consistent_income_aware']['ALIGN_reach_in_units'],
                  '| D reach units (own):', s['D_raw_literal_map']['ALIGN_reach_in_units'],
                  '| D reach in A units:', s['D_raw_in_A_units']['ALIGN_reach_in_units'],
                  '| D pair@1 A-unit:', s['D_raw_in_A_units'].get('ALIGN_pair_auc_at_1_unit'), flush=True)
        else:
            print('invalid', s)
    print('replay', json.dumps(res['gaussian_model']['calibration_replay_LOCAL_published_pair_0657']), flush=True)
    res['published'] = published()
    for k, v in res['published']['services'].items():
        print(k, 'amp=%.2f' % v['amplification'], '95%%[%.2f, %.2f]' % (v['amplification_95pct_indep_errors'][0], v['amplification_95pct_indep_errors'][2]),
              'notIncome=%.3f' % v['share_pair_signal_not_in_income_alone(1-d2inc/d2pair)'], 'occAlone=%.3f' % v['share_pair_signal_occupation_alone(d2occ/d2pair)'])
    print(json.dumps(res['published']['reader_class_sensitivity']))
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()

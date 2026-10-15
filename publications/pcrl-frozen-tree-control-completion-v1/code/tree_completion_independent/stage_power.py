"""Stage POWER: own per-group influence functions on INNER and own joint-power simulation.

(a) Base statistics (24 L_TREE utility clauses + winner-reader SEX AUCs of the candidate and every comparator,
    both seeds) and their per-row influence functions: mean statistic d: IF_i = d_i - mean(d); AUC: Hajek
    projection. Rows are summed within duplicate groups and scaled by G/n, so stat - truth ~= mean_g phi_g.
(b) Normal model T ~ N(mu, A Sigma A'/G); per draw the smallest G at which all clauses pass; N80/N90 are the
    0.8/0.9 quantiles. IUT: c = z_0.975; SIM: c = z_{1-0.05/(2K)}, K = number of clauses in the claim.
    Prevalence (label) shift: AUC influence terms rescale by pi_c/pi_c', utility means move to the
    class-mixture pi1' m_pos + pi0' m_neg, covariance = class-reweighted second moments about the new means.
"""
from __future__ import annotations

import argparse
import csv
import io
import math

import numpy as np
from scipy import stats

from . import core

NEW = ('T_REFIT_E20', 'RAW_J_NEQ', 'FARE_TC')
SERVICES = ('FARE_TC', 'H_DPC_JOINT', 'H_JOINT_CODE', 'H_SEQ21_CODE', 'L_TREE', 'N_EQ', 'RAW_J_NEQ', 'RAW_PROJ', 'T_REFIT_E20', 'T_TREE')
CODES = ('H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT')
REQUIRED_A = ('RAW_PROJ', 'N_EQ', 'T_TREE', 'H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT')
VIEWS = ('income', 'occupation', 'pair')
BINDING = ['U|1|occupation|U0|brier_excess', 'AUC|L_TREE|0|pair', 'AUC|L_TREE|1|pair', 'AUC|RAW_PROJ|0|pair',
           'AUC|RAW_PROJ|1|pair', 'AUC|H_SEQ21_CODE|0|pair', 'AUC|H_SEQ21_CODE|1|pair']


def build_clauses(names, kind):
    """(clause name, coefficient vector over base statistics, direction, bound, group)."""
    idx = {n: i for i, n in enumerate(names)}; M = len(names); out = []
    for n in names:
        if n.startswith('U|'):
            _, s, t, ref, m = n.split('|'); a = np.zeros(M); a[idx[n]] = 1.0
            out.append((f'utility|s{s}|{t}|{ref}|{m}', a, +1, core.CAPS[m], 'utility'))

    def benefit(c, margin, tag):
        a = np.zeros(M)
        for s in ('0', '1'):
            a[idx[f'AUC|{c}|{s}|pair']] += 0.5; a[idx[f'AUC|L_TREE|{s}|pair']] -= 0.5
        out.append((f'{tag}|vs_{c}', a, -1, margin, tag))

    def excess(c):
        for v in ('income', 'occupation'):
            a = np.zeros(M)
            for s in ('0', '1'):
                a[idx[f'AUC|L_TREE|{s}|{v}']] += 0.5; a[idx[f'AUC|{c}|{s}|{v}']] -= 0.5
            out.append((f'excess|{v}_vs_{c}', a, +1, 0.010, 'excess'))
    if kind in ('A', 'A+'):
        for c in REQUIRED_A + (('RAW_J_NEQ',) if kind == 'A+' else ()):
            benefit(c, 0.020, 'benefit'); excess(c)
    elif kind == 'B':
        benefit('RAW_PROJ', 0.020, 'benefit'); excess('RAW_PROJ')
        benefit('H_SEQ21_CODE', 0.0, 'superiority'); excess('H_SEQ21_CODE')
    return out


def clause_arrays(clauses, mu_base, Sigma, overrides=None, shift=None):
    Amat = np.array([c[1] for c in clauses])
    mu = Amat @ mu_base
    names = [c[0] for c in clauses]
    for j, nm in enumerate(names):
        if overrides and nm in overrides:
            mu[j] = overrides[nm]
        if shift and nm in shift:
            mu[j] += shift[nm]
    return names, mu, np.array([c[3] for c in clauses]), np.array([c[2] for c in clauses], float), Amat @ Sigma @ Amat.T


def run(spec):
    from tree_completion.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec, 'verification', allow_inner_truth=True)
    R = core.Receipt('POWER', active, cfg)
    cfg['_code_sha256'] = core.code_hashes(verify, cfg)
    J = lambda it: core.load_json(verify, it)
    fin = J(cfg['feasibility_inputs']); sup = J(cfg['feasibility_supplement']); meta = J(cfg['power_meta'])
    R.check('feasibility_inputs_copies_identical', core.file_sha(verify, cfg['feasibility_inputs']) == core.file_sha(verify, cfg['feasibility_inputs_private'])
            == meta['inputs_sha256'])
    names = list(fin['names'])
    R.check('feasibility_names_84_and_supplement_aligned', len(names) == 84 and sup['names'] == names)

    # ---------------------------------------------------------------- INNER data
    guard()
    truth = {k: core.load_npz(verify, cfg['inner_truth'][k], ['row_id', 'unit', k]) for k in ('sex', 'income', 'occupation')}
    row_id, unit = truth['sex']['row_id'], truth['sex']['unit']
    R.check('truth_rows_aligned', all(np.array_equal(truth[k]['row_id'], row_id) and np.array_equal(truth[k]['unit'], unit) for k in truth))
    y = truth['sex']['sex'].astype(np.int64); lab = {'income': truth['income']['income'].astype(np.int64), 'occupation': truth['occupation']['occupation'].astype(np.int64)}
    gidx, G = core.group_index(unit); n = len(y)
    gsex = np.zeros(G, dtype=np.int64); gsex[gidx] = y
    R.check('groups_2234_rows_2235_sex_pure', n == 2235 and G == 2234 and all(len(np.unique(y[gidx == g])) == 1 for g in np.flatnonzero(np.bincount(gidx) > 1))
            and fin['G'] == G and fin['n_rows'] == n)
    refs = {s: core.load_npz(verify, cfg['inner_references'][s], ['row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2']) for s in ('0', '1')}
    rec = {b: {s: J(cfg['banks'][b][s]['receipt']) for s in ('0', '1')} for b in ('old', 'new')}
    sco = {b: {s: core.load_npz(verify, cfg['banks'][b][s]['scores']) for s in ('0', '1')} for b in ('old', 'new')}
    R.check('aligned_rows_refs_scores', all(np.array_equal(refs[s]['row_id'], row_id) for s in refs) and
            all(np.array_equal(sco[b][s]['row_id'], row_id) and np.array_equal(sco[b][s]['unit'], unit) for b in sco for s in sco[b]))
    cand = {s: core.load_npz(verify, cfg['candidate_release'][s], ['row_id', 'p1', 'd1', 'p2', 'd2']) for s in ('0', '1')}

    # ---------------------------------------------------------------- base statistics and influence functions
    scale = G / n
    point = {}; IF = {}
    for s in ('0', '1'):
        for ti, task in enumerate(core.TASKS):
            p = cand[s]['p1' if task == 'income' else 'p2']
            for ref in ('U0', 'Ucal'):
                pt, rows = core.clause_point_and_rows(lab[task], p, refs[s][f'{ref}_{ti + 1}'], core.KCLS[task])
                for m in core.METRICS:
                    nm = f'U|{s}|{task}|{ref}|{m}'
                    point[nm] = pt[m]; IF[nm] = rows[m] - rows[m].mean()
    for svc in SERVICES:
        b = 'new' if svc in NEW else 'old'
        for s in ('0', '1'):
            for v in VIEWS:
                w = rec[b][s]['services'][svc]['winner'][v]['reader']
                inf, a = core.auc_influence(y, sco[b][s][f'{svc}__{v}__{w}'])
                nm = f'AUC|{svc}|{s}|{v}'
                point[nm] = core.auc(y, sco[b][s][f'{svc}__{v}__{w}']); IF[nm] = inf
                if abs(a - point[nm]) > 1e-12:
                    R.check(f'influence_auc_mean_equals_auc_{nm}', False)
    R.check('base_statistics_cover_feasibility_names', sorted(point) == sorted(names))
    mu = np.array([point[nm] for nm in names])
    pdiff = np.abs(mu - np.array(fin['points']))
    R.check('base_points_exact_vs_FEASIBILITY_INPUTS', pdiff.max() <= 1e-12, max_abs=float(pdiff.max()))
    phi = np.zeros((len(names), G))
    for i, nm in enumerate(names):
        phi[i] = scale * np.bincount(gidx, weights=IF[nm], minlength=G)
    Sig0 = phi @ phi.T / G; Sig1 = phi @ phi.T / (G - 1)
    pub = np.array(fin['cov_phi'])
    rel0 = np.abs(np.diag(Sig0) / np.diag(pub) - 1); rel1 = np.abs(np.diag(Sig1) / np.diag(pub) - 1)
    conv = 'ddof0' if rel0.max() <= rel1.max() else 'ddof1'
    Sig = Sig0 if conv == 'ddof0' else Sig1
    reld = np.abs(np.diag(Sig) / np.diag(pub) - 1)
    ii = {nm: i for i, nm in enumerate(names)}
    bind = {nm: {'mine': float(Sig[ii[nm], ii[nm]]), 'published': float(pub[ii[nm], ii[nm]]), 'rel_diff': float(reld[ii[nm]])} for nm in BINDING}
    R.check('binding_statistic_variances_within_2pct', all(v['rel_diff'] < 0.02 for v in bind.values()), detail=bind)
    R.check('all_84_variances_within_2pct', reld.max() < 0.02, max_rel=float(reld.max()), convention=conv)
    offd = np.abs(Sig - pub).max()
    R.check('full_covariance_close', offd <= 0.02 * np.abs(pub).max(), max_abs=float(offd))
    # private per-group phi (published as aggregate only): direct comparison
    ph = core.load_npz(verify, cfg['phi_private'], ['phi', 'group_sex', 'names'])
    R.check('PHI_PER_GROUP_names_and_group_sex_aligned', [str(x) for x in ph['names']] == names and np.array_equal(ph['group_sex'], gsex))
    R.check('per_group_phi_equal_to_private_phi', np.abs(ph['phi'] - phi).max() <= 1e-9, max_abs=float(np.abs(ph['phi'] - phi).max()))
    # class-conditional means (prevalence supplement)
    mpos = phi[:, gsex == 1].mean(1); mneg = phi[:, gsex == 0].mean(1)
    cm = max(np.abs(mpos - np.array(sup['class_mean_phi_pos'])).max(), np.abs(mneg - np.array(sup['class_mean_phi_neg'])).max())
    R.check('class_conditional_phi_means_equal_supplement', cm <= 1e-9 and sup['groups_pos'] == int((gsex == 1).sum()) and sup['groups_neg'] == int((gsex == 0).sum()), max_abs=float(cm))
    pi1_groups = float((gsex == 1).mean())
    R.check('sex_prevalence_equals_inputs', abs(pi1_groups - fin['sex_prevalence_pi1']) < 1e-15, pi1=pi1_groups)

    # influence SE vs bootstrap SE for the benefit vs RAW_PROJ (validation analogue, own bootstrap)
    clB = build_clauses(names, 'B')
    a_b = [c for c in clB if c[0] == 'benefit|vs_RAW_PROJ'][0][1]
    se_if = math.sqrt(a_b @ Sig @ a_b / G)
    rng = np.random.default_rng(int(cfg['bootstrap_seed']))
    Mb = core.multiplicities(rng, G, 2000)
    bvals = np.zeros(2000)
    for s in ('0', '1'):
        for svc, sign in (('RAW_PROJ', 0.5), ('L_TREE', -0.5)):
            b = 'new' if svc in NEW else 'old'
            w = rec[b][s]['services'][svc]['winner']['pair']['reader']
            f = core.WeightedAUC(y, sco[b][s][f'{svc}__pair__{w}'])
            for a0 in range(0, 2000, 500):
                bvals[a0:a0 + 500] += sign * f(Mb[a0:a0 + 500][:, gidx])
    se_bs = float(np.std(bvals, ddof=1))
    R.check('influence_se_validates_against_own_bootstrap', abs(se_if / se_bs - 1) < 0.05 and abs(se_if / fin['validation']['benefit_L_TREE_vs_RAW_PROJ_se_influence'] - 1) < 0.02,
            se_influence=se_if, se_bootstrap_own=se_bs, published=fin['validation'])
    R.numbers['binding_variances'] = bind
    R.numbers['covariance_convention_matching'] = conv

    # ---------------------------------------------------------------- joint power (own simulation)
    guard()
    draws = int(cfg['draws']); seed = int(cfg['sim_seed'])
    zi = stats.norm.ppf(0.975)
    jp = list(csv.DictReader(io.StringIO(verify(cfg['joint_power']['path'], cfg['joint_power']['sha256']).read_text())))
    po = list(csv.DictReader(io.StringIO(verify(cfg['privacy_only']['path'], cfg['privacy_only']['sha256']).read_text())))

    def pub_row(claim, design, benefit, shift, minority='observed'):
        for r in jp:
            if (r['claim'] == claim and r['design'] == design and r['true_benefit_vs_binding'] == benefit and float(r['occ_brier_shift']) == shift
                    and float(r['variance_multiplier']) == 1.0 and r['minority_share'] == minority):
                return r
        return None

    occ_brier = [c for c in names if c.startswith('U|') and '|occupation|' in c and c.endswith('brier_excess')]

    def scenario(kind, benefit=None, shift=0.0, shift_mode='occupation_brier', minority=None, privacy_only=False, mu_base=None, Sigma=None):
        mu_b = mu if mu_base is None else mu_base
        S = Sig if Sigma is None else Sigma
        if minority is not None:
            pi1n = 1 - minority if pi1_groups >= 0.5 else minority
            pi1r = float(y.mean()); pi0r = 1 - pi1r
            isu = np.array([nm.startswith('U|') for nm in names])
            delta = np.where(isu, pi1n * mpos + (1 - pi1n) * mneg, 0.0)
            ph2 = phi.copy()
            pos = gsex == 1
            ph2[np.ix_(~isu, pos)] *= pi1r / pi1n
            ph2[np.ix_(~isu, ~pos)] *= pi0r / (1 - pi1n)
            ph2[isu] -= delta[isu][:, None]
            S = pi1n * (ph2[:, pos] @ ph2[:, pos].T) / pos.sum() + (1 - pi1n) * (ph2[:, ~pos] @ ph2[:, ~pos].T) / (~pos).sum()
            mu_b = mu_b + delta
        cl = build_clauses(names, kind)
        if privacy_only:
            cl = [c for c in cl if c[4] != 'utility']
        over = {f'benefit|vs_{c}': benefit for c in CODES} if (benefit is not None and kind in ('A', 'A+')) else None
        sh = None
        if shift:
            if shift_mode == 'occupation_brier':
                sh = {f"utility|s{nm.split('|')[1]}|occupation|{nm.split('|')[3]}|brier_excess": shift for nm in occ_brier}
            elif shift_mode == 's1_U0_only':
                sh = {'utility|s1|occupation|U0|brier_excess': shift}
            else:
                sh = {c[0]: shift for c in cl if c[0].endswith('brier_excess')}
        cn, m, bnd, d, C = clause_arrays(cl, mu_b, S, over, sh)
        return cn, m, bnd, d, C

    def n80n90(kind, design, **kw):
        cn, m, bnd, d, C = scenario(kind, **kw)
        c = zi if design == 'IUT' else core.z_critical_two_sided(len(cn))
        rng2 = np.random.Generator(np.random.SFC64(seed))
        g = core.gstar_draws(m, bnd, d, C, c, rng2, draws)
        if g is None:
            return {'status': 'NOT_ACHIEVABLE', 'K': len(cn), 'c': c}
        return {'status': 'ACHIEVABLE', 'K': len(cn), 'c': c, 'N80': float(np.quantile(g, 0.8)), 'N90': float(np.quantile(g, 0.9)),
                '_scen': (m, bnd, d, C)}

    targets = [
        ('B_observed_IUT', 'B', 'IUT', {}, ('B_narrow', 'IUT_alpha0.025', 'observed', 0.0)),
        ('B_observed_SIM_K30', 'B', 'SIM', {}, ('B_narrow', 'SIM_claimK', 'observed', 0.0)),
        ('A_benefit0.025_IUT', 'A', 'IUT', {'benefit': 0.025}, ('A_original_6', 'IUT_alpha0.025', '0.025', 0.0)),
        ('A_benefit0.025_SIM_K42', 'A', 'SIM', {'benefit': 0.025}, ('A_original_6', 'SIM_claimK', '0.025', 0.0)),
        ('B_brier_shift_-0.001_IUT', 'B', 'IUT', {'shift': -0.001}, ('B_narrow', 'IUT_alpha0.025', 'observed', -0.001)),
        ('B_brier_shift_+0.0005_IUT', 'B', 'IUT', {'shift': 0.0005}, ('B_narrow', 'IUT_alpha0.025', 'observed', 0.0005)),
        ('B_minority0.25_IUT', 'B', 'IUT', {'minority': 0.25}, ('B_narrow', 'IUT_alpha0.025', 'observed', 0.0, '0.25')),
    ]
    results = {}; worst = 0.0
    for name, kind, design, kw, key in targets:
        r = n80n90(kind, design, **kw)
        pr = pub_row(key[0], key[1], key[2], key[3], key[4] if len(key) > 4 else 'observed')
        out = {k: v for k, v in r.items() if not k.startswith('_')}
        if pr is not None and pr['N80']:
            out['published_N80'] = float(pr['N80']); out['published_N90'] = float(pr['N90'])
            out['rel_diff_N80'] = r['N80'] / out['published_N80'] - 1 if 'N80' in r else None
            out['rel_diff_N90'] = r['N90'] / out['published_N90'] - 1 if 'N90' in r else None
            ok = r['status'] == 'ACHIEVABLE' and abs(out['rel_diff_N80']) < 0.03 and abs(out['rel_diff_N90']) < 0.03
            worst = max(worst, abs(out['rel_diff_N80'] or 0), abs(out['rel_diff_N90'] or 0)) if r['status'] == 'ACHIEVABLE' else worst
        else:
            ok = False; out['published'] = 'row missing'
        R.check(f'power_{name}_within_3pct', ok, **out)
        if '_scen' in r:
            m, bnd, d, C = r['_scen']
            pw = core.direct_power(m, bnd, d, C, r['c'], r['N80'], np.random.Generator(np.random.SFC64(seed + 1)), 100000)
            out['direct_power_at_own_N80'] = pw
        results[name] = out
    R.numbers['power_targets'] = results
    R.numbers['power_worst_rel_diff'] = worst
    R.check('direct_simulation_power_at_own_N80_is_0.8', all(abs(v['direct_power_at_own_N80'] - 0.8) < 0.01 for v in results.values() if 'direct_power_at_own_N80' in v))

    # NOT_ACHIEVABLE checks
    na = {}
    for b in (0.010, 0.015, 0.020):
        for design in ('IUT', 'SIM'):
            na[f'A_benefit{b}_{design}'] = n80n90('A', design, benefit=b)['status']
    na['A_beyond_cap_+0.001_IUT_benefit0.025'] = n80n90('A', 'IUT', benefit=0.025, shift=0.001)['status']
    na['A+RAW_J_beyond_cap_+0.001_IUT_benefit0.025'] = n80n90('A+', 'IUT', benefit=0.025, shift=0.001)['status']
    na['B_beyond_cap_+0.001_IUT_observed'] = n80n90('B', 'IUT', shift=0.001)['status']
    na['B_minority0.45_IUT'] = n80n90('B', 'IUT', minority=0.45)['status']
    na['A_observed_IUT'] = n80n90('A', 'IUT')['status']
    R.check('not_achievable_cases', all(v == 'NOT_ACHIEVABLE' for v in na.values()), cases=na)
    s1u0 = point['U|1|occupation|U0|brier_excess']
    R.numbers['beyond_cap_true_excess'] = s1u0 + 0.001
    pubna = [r for r in jp if r['claim'] == 'A_original_6' and r['true_benefit_vs_binding'] in ('0.01', '0.015', '0.02') and r['design'] != 'SIM_K78']
    R.check('published_A_benefit_le_0.020_NOT_ACHIEVABLE', pubna and all(r['N80_status'].startswith('NOT_ACHIEVABLE') for r in pubna))
    pubbc = [r for r in po if r['analysis'].startswith('ILLUSTRATIVE_BEYOND_CAP')]
    R.check('published_beyond_cap_NOT_ACHIEVABLE', pubbc and all(r['N80_status'].startswith('NOT_ACHIEVABLE') for r in pubbc))
    pm45 = pub_row('B_narrow', 'IUT_alpha0.025', 'observed', 0.0, '0.45')
    R.check('published_minority0.45_NOT_ACHIEVABLE', pm45 is not None and pm45['N80_status'].startswith('NOT_ACHIEVABLE'))

    # privacy-only (B observed IUT ~ 4,079)
    r = n80n90('B', 'IUT', privacy_only=True)
    prow = [x for x in po if x['analysis'] == 'PRIVACY_CLAUSES_ONLY' and x['claim'] == 'B_narrow' and x['design'] == 'IUT_alpha0.025' and x['true_benefit_vs_binding'] == 'observed'][0]
    rd = r['N80'] / float(prow['N80']) - 1
    R.check('privacy_only_B_observed_IUT_N80', abs(rd) < 0.03, mine=r['N80'], published=float(prow['N80']), rel_diff=rd, N90_mine=r['N90'], N90_published=float(prow['N90']))

    # interpretation probes for the Brier shift (transparency)
    probes = {}
    for mode in ('occupation_brier', 's1_U0_only', 'all_brier'):
        rr = n80n90('B', 'IUT', shift=-0.001, shift_mode=mode)
        probes[mode] = rr.get('N80')
    R.numbers['brier_shift_-0.001_interpretation_N80'] = probes

    # secondary run: published points and covariance through the own simulation
    sec = {}
    for name, kind, design, kw, key in targets[:6]:
        rr = n80n90(kind, design, mu_base=np.array(fin['points']), Sigma=pub, **kw)
        sec[name] = {'N80': rr.get('N80'), 'N90': rr.get('N90'), 'vs_primary_N80': (rr['N80'] / results[name]['N80'] - 1) if rr.get('N80') else None}
    R.numbers['secondary_published_inputs_own_simulation'] = sec
    R.numbers['simulation'] = {'rng': 'numpy SFC64', 'seed': seed, 'draws': draws, 'IUT_c': float(zi),
                               'SIM_c_K30': core.z_critical_two_sided(30), 'SIM_c_K42': core.z_critical_two_sided(42)}
    return R.write()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)

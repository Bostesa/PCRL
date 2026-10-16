"""Independent check 8: DEV_REPORT utility, recovery, subgroup endpoints, IUT / simultaneous bounds and claim decisions.

Own code only (no acs_replication / task_headroom / task_conditioned_representation import):
  * AUC point: Mann-Whitney via average ranks (scipy.stats.rankdata).
  * Weighted (bootstrap) AUC: sparse indicator aggregation of per-draw positive / negative weight at each distinct score.
  * Household bootstrap, two variants, both B = 2048:
      REPLAY  - draws re-implemented from the registered seed 2026101603 (default_rng.integers over sorted households);
                must reproduce the lead's SEs and every decision exactly.
      OWN     - independent seed 20261016 (Generator(PCG64).choice per draw); SEs must agree within MC error (~5%);
                decision differences are reported as Monte Carlo sensitivity.
  * Endpoints, bounds (t with G-1 df, one-sided alpha/3 with alpha = 0.025; K = 40 Bonferroni two-sided 95%) and the three claims
    re-derived from SCIENCE_LOCK.inference.claim_rules.
DEV_REPORT truth is RE-READ for verification (already opened once by the lead). Aggregates only are written.
"""
import json
from pathlib import Path

import numpy as np
from scipy import sparse, stats
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
OUT = RES / 'independent'
TASKS = ('income', 'occupation')
KCLS = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
B = 2048
REG_SEED = 2026101603
OWN_SEED = 20261016
ALPHA = 0.025
K40 = 40
FILES = ('STAGE_A_{s}/STAGE_A_OUTPUTS.npz', 'STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz', 'NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz',
         'FARE_{s}/FARE_OUTPUTS.npz', 'FAGTB_{s}/FAGTB_OUTPUTS.npz')


def npz(path, field):
    with np.load(path, allow_pickle=False) as z:
        return z[field].copy(), z['row_id'].copy(), z['unit'].copy()


def auc_rank(y, s):
    r = rankdata(np.asarray(s, dtype=np.float64), method='average')
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_weighted(y, s, W):
    """W: (D, n) nonnegative weights. Returns (D,) weighted Mann-Whitney AUCs (ties count 1/2)."""
    s = np.asarray(s, dtype=np.float64)
    u, inv = np.unique(s, return_inverse=True)
    n = len(s); U = len(u)
    M = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, U))
    pos = (y == 1).astype(np.float64); neg = 1.0 - pos
    MT = M.T.tocsr()                                   # (U, n) indicator of distinct score values
    P = np.asarray(MT @ (W * pos[None, :]).T).T        # (D, U) positive weight at each distinct score
    N = np.asarray(MT @ (W * neg[None, :]).T).T        # (D, U) negative weight at each distinct score
    below = np.cumsum(N, axis=1) - N
    num = (P * (below + 0.5 * N)).sum(axis=1)
    return num / (P.sum(axis=1) * N.sum(axis=1))


def boot_replay(unit):
    hh, inv = np.unique(unit, return_inverse=True)
    G = len(hh)
    rng = np.random.default_rng(REG_SEED)
    picks = rng.integers(0, G, size=(B, G))
    C = np.empty((B, G))
    for b in range(B):
        C[b] = np.bincount(picks[b], minlength=G)
    return C[:, inv], G


def boot_own(unit):
    hh, inv = np.unique(unit, return_inverse=True)
    G = len(hh)
    rng = np.random.Generator(np.random.PCG64(OWN_SEED))
    C = np.zeros((B, G))
    for b in range(B):
        idx = rng.choice(G, size=G, replace=True)
        np.add.at(C[b], idx, 1.0)
    return C[:, inv], G


def task_vecs(y, p):
    p = np.asarray(p, dtype=np.float64); n = len(y)
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1.0, atol=1e-9, rtol=0)
    oh = np.zeros_like(p); oh[np.arange(n), y] = 1.0
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(n), y], 1e-12)),
            'brier': ((p - oh) ** 2).sum(axis=1)}


def endpoint(point, draws, kind, margin, c1, cs):
    se = float(np.std(draws, ddof=1))
    lo1, hi1 = point - c1 * se, point + c1 * se
    los, his = point - cs * se, point + cs * se
    if kind == 'benefit':
        iut = lo1 > margin
        sim = 'ESTABLISHED' if los > margin else ('ESTABLISHED_SHORTFALL' if his < margin else 'UNRESOLVED')
        pp = point > margin
    else:
        iut = hi1 <= margin
        sim = 'PASS' if his <= margin else ('ESTABLISHED_VIOLATION' if los > margin else 'UNRESOLVED')
        pp = point <= margin
    return {'point': float(point), 'se': se, 'iut_lower': float(lo1), 'iut_upper': float(hi1), 'sim95_lower': float(los), 'sim95_upper': float(his),
            'iut_pass': bool(iut), 'sim_status': sim, 'point_pass': bool(pp), 'kind': kind, 'margin': margin,
            'pct_lower': float(np.quantile(draws, ALPHA / 3)), 'pct_upper': float(np.quantile(draws, 1 - ALPHA / 3)),
            'kurt': float(stats.kurtosis(draws, fisher=False))}


def mc_z(se_a, se_b, kurt):
    """z-score of the difference of two independent B-draw bootstrap SEs (delta method: rel sd of one SE = sqrt((kurt-1)/(4B)))."""
    rel = np.sqrt(max(kurt - 1.0, 1e-12) / (4.0 * B))
    return float((se_a - se_b) / (se_b * np.sqrt(2.0) * rel))


def load_outputs(seed):
    out = {}
    for pat in FILES:
        with np.load(PRIV / 'runs' / pat.format(s=seed)) as z:
            for k in z.files:
                parts = k.split('__')
                if len(parts) == 3 and parts[1] == 'DEV_REPORT':
                    out.setdefault(parts[0], {})[parts[2]] = z[k].copy()
                if k == 'row_id__DEV_REPORT':
                    out.setdefault('__row_id__', {})[pat] = z[k].copy()
    return out


def main():
    rep = json.loads((RES / 'DEV_REPORT_RESULTS.json').read_text())
    sel = json.loads((RES / 'SELECTION_LOCK.json').read_text())
    lockj = json.loads((RES / 'SCIENCE_LOCK.json').read_text())
    T = PRIV / 'exports' / 'DEV_REPORT_TRUTH'
    y = {}
    y['income'], rid, unit = npz(T / 'income.npz', 'income')
    y['occupation'], rid2, unit2 = npz(T / 'occupation.npz', 'occupation')
    sex, rid3, unit3 = npz(T / 'sex.npz', 'sex')
    pw, rid4, unit4 = npz(T / 'weights.npz', 'pwgtp')
    with np.load(PRIV / 'exports' / 'DEV_REPORT' / 'features.npz', allow_pickle=False) as z:
        frid, funit = z['row_id'].copy(), z['unit'].copy()
    align = all(np.array_equal(a, frid) for a in (rid, rid2, rid3, rid4)) and all(np.array_equal(a, funit) for a in (unit, unit2, unit3, unit4))
    pw = pw.astype(np.float64)
    WR, G = boot_replay(unit)
    WO, G2 = boot_own(unit)
    assert G == G2
    c1 = float(stats.t.ppf(1 - ALPHA / 3, G - 1)); cs = float(stats.t.ppf(1 - 0.05 / (2 * K40), G - 1))
    hh_sex = {g: int(len(np.unique(unit[sex == g]))) for g in (0, 1)}
    estimable = all(v >= lockj['benchmark']['support_thresholds']['subgroup_households_min_per_sex_DEV_REPORT'] for v in hh_sex.values())
    roster = sel['attack_roster']
    pts, drR, drO = {}, {}, {}
    util = {}
    sg = {}
    recov_cmp = []
    util_cmp = []
    wauc_cmp = []
    row_align = True
    for seed in (0, 1):
        outs = load_outputs(seed)
        row_align &= all(np.array_equal(v, frid) for v in outs['__row_id__'].values())
        rr = json.loads((PRIV / 'runs' / f'READERS_{seed}' / 'READERS_SEED.json').read_text())
        with np.load(PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz') as z:
            scores = {k: z[k].copy() for k in z.files}
        row_align &= np.array_equal(scores['row_id'], frid)
        vt_U0 = {t: task_vecs(y[t], outs['U0'][t]) for t in TASKS}
        vt_Ucal = {t: task_vecs(y[t], outs['Ucal'][t]) for t in TASKS}
        for svc in roster:
            lead_svc = rep['per_seed'][str(seed)]['services'][svc]
            for view in ('income', 'occupation', 'pair'):
                win = rr['services'][svc]['winner'][view]['reader']
                s = scores[f'{svc}__{view}__{win}']
                a = auc_rank(sex, s)
                a_w1 = float(auc_weighted(sex, s, np.ones((1, len(s))))[0])
                pts[(svc, seed, view)] = a
                drR[(svc, seed, view)] = auc_weighted(sex, s, WR)
                drO[(svc, seed, view)] = auc_weighted(sex, s, WO)
                lr = lead_svc['recovery'][view]
                recov_cmp.append({'service': svc, 'seed': seed, 'view': view, 'reader': win, 'reader_matches_lead': win == lr['reader'],
                                  'auc_independent': a, 'auc_lead': lr['auc'], 'abs_diff': abs(a - lr['auc']), 'rank_vs_weighted_formula_diff': abs(a - a_w1),
                                  'se_replay': float(np.std(drR[(svc, seed, view)], ddof=1)), 'se_own_seed': float(np.std(drO[(svc, seed, view)], ddof=1)), 'se_lead': lr['auc_se'],
                                  'mc_z_own_vs_lead': mc_z(float(np.std(drO[(svc, seed, view)], ddof=1)), lr['auc_se'], float(stats.kurtosis(drR[(svc, seed, view)], fisher=False)))})
                wa = float(auc_weighted(sex, s, pw[None, :])[0])
                wauc_cmp.append(abs(wa - lr['pwgtp_weighted_auc_point']))
            for t in TASKS:
                v = task_vecs(y[t], outs[svc][t])
                for ref, rv in (('U0', vt_U0[t]), ('Ucal', vt_Ucal[t])):
                    diffs = {'accuracy_loss': rv['accuracy'] - v['accuracy'], 'nll_excess': v['nll'] - rv['nll'], 'brier_excess': v['brier'] - rv['brier']}
                    for m, d in diffs.items():
                        eR = endpoint(d.mean(), (WR @ d) / WR.sum(1), 'no_excess', CAPS[m], c1, cs)
                        eO = endpoint(d.mean(), (WO @ d) / WO.sum(1), 'no_excess', CAPS[m], c1, cs)
                        util[(svc, seed, t, ref, m)] = (eR, eO)
                        le = lead_svc['utility'][f'{t}|{ref}|{m}']
                        util_cmp.append({'key': f'{svc}|s{seed}|{t}|{ref}|{m}', 'point_diff': abs(eR['point'] - le['point']), 'se_replay_diff': abs(eR['se'] - le['se']),
                                         'se_own_ratio': eO['se'] / le['se'] if le['se'] > 0 else None, 'mc_z_own_vs_lead': mc_z(eO['se'], le['se'], eR['kurt']) if le['se'] > 0 else 0.0, 'pw_point_diff': abs(float(np.sum(pw * d) / pw.sum()) - le['pwgtp_weighted_point'])})
            vo = task_vecs(y['occupation'], outs[svc]['occupation']); ro = vt_U0['occupation']
            d = vo['brier'] - ro['brier']
            for g in (0, 1):
                m = sex == g
                eR = endpoint(d[m].mean(), (WR[:, m] @ d[m]) / WR[:, m].sum(1), 'no_excess', 0.005, c1, cs)
                eO = endpoint(d[m].mean(), (WO[:, m] @ d[m]) / WO[:, m].sum(1), 'no_excess', 0.005, c1, cs)
                sg[(svc, seed, g)] = (eR, eO)
    # ------------------------------------------------------------------ the 40 primary endpoints
    def build(which):
        k = 0 if which == 'REPLAY' else 1
        dr = drR if which == 'REPLAY' else drO
        E = {}
        for seed in (0, 1):
            E[f'RECIPE|coalition_benefit_vs_TASK_REFIT|s{seed}'] = endpoint(pts[('TASK_REFIT', seed, 'pair')] - pts[('LOCAL_TREE', seed, 'pair')],
                                                                         dr[('TASK_REFIT', seed, 'pair')] - dr[('LOCAL_TREE', seed, 'pair')], 'benefit', 0.020, c1, cs)
            for view in ('income', 'occupation'):
                E[f'RECIPE|{view}_recipient_worsening_vs_TASK_REFIT|s{seed}'] = endpoint(pts[('LOCAL_TREE', seed, view)] - pts[('TASK_REFIT', seed, view)],
                                                                                      dr[('LOCAL_TREE', seed, view)] - dr[('TASK_REFIT', seed, view)], 'no_excess', 0.010, c1, cs)
            for t in TASKS:
                for ref in ('U0', 'Ucal'):
                    for m in CAPS:
                        E[f'RECIPE|utility|{t}|{ref}|{m}|s{seed}'] = util[('LOCAL_TREE', seed, t, ref, m)][k]
            for fam in ('NEURAL_RAW', 'FAGTB', 'FARE'):
                svc = sel['frozen'][fam]['configuration']
                E[f'COMPETITIVE|coalition_benefit_vs_{fam}|s{seed}'] = endpoint(pts[(svc, seed, 'pair')] - pts[('LOCAL_TREE', seed, 'pair')],
                                                                              dr[(svc, seed, 'pair')] - dr[('LOCAL_TREE', seed, 'pair')], 'benefit', 0.0, c1, cs)
                e = E[f'COMPETITIVE|coalition_benefit_vs_{fam}|s{seed}']
                e['material_iut_pass'] = bool(e['iut_lower'] > 0.020)
            for g in (0, 1):
                E[f'SUBGROUP|occupation_brier_excess_vs_U0|sex{g}|s{seed}'] = sg[('LOCAL_TREE', seed, g)][k]
        assert len(E) == K40
        return E

    def claims(E, lt_select_feasible):
        ben = [E[f'RECIPE|coalition_benefit_vs_TASK_REFIT|s{s}'] for s in (0, 1)]
        wor = [E[f'RECIPE|{v}_recipient_worsening_vs_TASK_REFIT|s{s}'] for s in (0, 1) for v in ('income', 'occupation')]
        utl = [E[k] for k in E if k.startswith('RECIPE|utility|')]
        assert len(utl) == 24
        six = all(e['iut_pass'] for e in ben) and all(e['iut_pass'] for e in wor)
        upf = all(e['point_pass'] for e in utl)
        recipe = 'RECIPE_EFFECT_ESTABLISHED' if six and upf else 'RECIPE_EFFECT_NOT_ESTABLISHED'
        nested = 'RECIPE_EFFECT_AND_UTILITY_BOUNDS_ESTABLISHED' if six and all(e['iut_pass'] for e in utl) else 'RECIPE_EFFECT_AND_UTILITY_BOUNDS_NOT_ESTABLISHED'
        fams = ('NEURAL_RAW', 'FAGTB', 'FARE')
        complete = all(sel['frozen'][f].get('configuration') for f in fams) and bool(sel['frozen']['FAGTB'].get('two_task_valid'))
        same = lt_select_feasible and upf
        eligible = [f for f in fams if sel['frozen'][f]['status'] == 'ELIGIBLE']
        beats = {f: all(E[f'COMPETITIVE|coalition_benefit_vs_{f}|s{s}']['iut_pass'] for s in (0, 1)) for f in fams}
        comp = ('INCOMPLETE' if not complete else 'LOCAL_TREE_FAILS_SAME_UTILITY_REQUIREMENTS' if not same else
                'NO_ELIGIBLE_BASELINE' if not eligible else 'SUPERIOR_TO_ALL_ELIGIBLE' if all(beats[f] for f in eligible) else 'NOT_SUPERIOR_TO_ALL_ELIGIBLE')
        sgs = [E[k] for k in E if k.startswith('SUBGROUP|')]
        sub = ('NOT_ESTIMABLE' if not estimable else 'SUBGROUP_QUALIFIED' if all(e['iut_pass'] for e in sgs) else
               'ESTABLISHED_SUBGROUP_HARM' if any(e['sim95_lower'] > 0.005 for e in sgs) else 'SUBGROUP_UNRESOLVED')
        return {'RECIPE_EFFECT': recipe, 'RECIPE_nested_utility_bounds': nested, 'utility_point_feasible_24': upf,
                'utility_iut_bound_pass_count_of_24': int(sum(e['iut_pass'] for e in utl)),
                'utility_sim_counts': {s: int(sum(e['sim_status'] == s for e in utl)) for s in ('PASS', 'UNRESOLVED', 'ESTABLISHED_VIOLATION')},
                'COMPETITIVE_COMPARISON': comp, 'eligible': eligible, 'complete': bool(complete), 'beats': beats,
                'SUBGROUP_SUITABILITY': sub, 'subgroup_iut': {k: E[k]['iut_pass'] for k in E if k.startswith('SUBGROUP|')}}

    c7 = OUT / 'check07_nominations.json'
    if c7.is_file():
        lt_feas = json.loads(c7.read_text())['evidence']['fixed_services']['LOCAL_TREE']['point_feasible_24']; lt_src = 'independent check07'
    else:
        lt_feas = bool(sel['fixed_services']['LOCAL_TREE']['point_feasible_24']); lt_src = 'SELECTION_LOCK (check07 output absent)'
    ER, EO = build('REPLAY'), build('OWN')
    CR, CO = claims(ER, lt_feas), claims(EO, lt_feas)
    LE = rep['endpoints']; LC = rep['claims']
    ep_cmp = {}
    worst = {'point': 0.0, 'replay_se_rel': 0.0, 'replay_bound': 0.0, 'own_se_ratio_min': 9.0, 'own_se_ratio_max': 0.0, 'replay_pct': 0.0}
    dec_mismatch_replay, dec_diff_own, sim_diff_own = [], [], []
    for k, le in LE.items():
        r, o = ER[k], EO[k]
        worst['point'] = max(worst['point'], abs(r['point'] - le['point']))
        worst['replay_se_rel'] = max(worst['replay_se_rel'], abs(r['se'] - le['se']) / le['se'])
        worst['replay_bound'] = max(worst['replay_bound'], *(abs(r[f] - le[f]) for f in ('iut_lower', 'iut_upper', 'sim95_lower', 'sim95_upper')))
        worst['replay_pct'] = max(worst['replay_pct'], abs(r['pct_lower'] - le['sensitivity_percentile_lower']), abs(r['pct_upper'] - le['sensitivity_percentile_upper']))
        ratio = o['se'] / le['se']
        worst.setdefault('own_mc_z_absmax', 0.0); worst['own_mc_z_absmax'] = max(worst['own_mc_z_absmax'], abs(mc_z(o['se'], le['se'], r['kurt'])))
        worst['own_se_ratio_min'] = min(worst['own_se_ratio_min'], ratio); worst['own_se_ratio_max'] = max(worst['own_se_ratio_max'], ratio)
        if r['iut_pass'] != le['iut_pass'] or r['sim_status'] != le['sim_status'] or r['point_pass'] != le['point_pass']:
            dec_mismatch_replay.append(k)
        if o['iut_pass'] != le['iut_pass']:
            dec_diff_own.append(k)
        if o['sim_status'] != le['sim_status']:
            sim_diff_own.append(k)
        if 'material_iut_pass' in r and r['material_iut_pass'] != le['material_margin_0.020_iut_pass']:
            dec_mismatch_replay.append(k + '|material')
        ep_cmp[k] = {'point_indep': r['point'], 'point_lead': le['point'], 'se_replay': r['se'], 'se_own_seed': o['se'], 'se_lead': le['se'],
                     'iut_lower_own': o['iut_lower'], 'iut_upper_own': o['iut_upper'], 'iut_pass_replay': r['iut_pass'], 'iut_pass_own': o['iut_pass'], 'iut_pass_lead': le['iut_pass'],
                     'sim_status_replay': r['sim_status'], 'sim_status_own': o['sim_status'], 'sim_status_lead': le['sim_status'], 'point_pass': r['point_pass']}
    lead_claims = {'RECIPE_EFFECT': LC['RECIPE_EFFECT']['decision'], 'RECIPE_nested_utility_bounds': LC['RECIPE_EFFECT']['nested_decision_utility_bounds'],
                   'utility_point_feasible_24': LC['RECIPE_EFFECT']['utility_point_feasible_24'],
                   'utility_iut_bound_pass_count_of_24': LC['RECIPE_EFFECT']['utility_iut_bound_pass_count_of_24'],
                   'COMPETITIVE_COMPARISON': LC['COMPETITIVE_COMPARISON']['decision'], 'eligible': LC['COMPETITIVE_COMPARISON']['eligible_required_baselines'],
                   'SUBGROUP_SUITABILITY': LC['SUBGROUP_SUITABILITY']['decision']}
    claims_replay_agree = all(CR[k] == v for k, v in lead_claims.items())
    claims_own_agree = all(CO[k] == v for k, v in lead_claims.items() if k not in ('utility_iut_bound_pass_count_of_24',))
    recov_point_max = max(x['abs_diff'] for x in recov_cmp)
    recov_reader_ok = all(x['reader_matches_lead'] for x in recov_cmp)
    recov_se_replay_rel = max(abs(x['se_replay'] - x['se_lead']) / x['se_lead'] for x in recov_cmp)
    recov_se_own = [x['se_own_seed'] / x['se_lead'] for x in recov_cmp]
    util_point_max = max(x['point_diff'] for x in util_cmp)
    util_se_replay_max = max(x['se_replay_diff'] for x in util_cmp)
    util_own_ratios = [x['se_own_ratio'] for x in util_cmp if x['se_own_ratio'] is not None]
    crit_ok = abs(c1 - rep['critical']['t_one_sided']) < 1e-12 and abs(cs - rep['critical']['t_simultaneous_K40']) < 1e-12
    counts_ok = G == rep['households'] and len(sex) == rep['persons'] and hh_sex == {int(k): v for k, v in rep['subgroup_households_per_sex'].items()}
    # PRIMARY_ENDPOINTS.csv consistency with DEV_REPORT_RESULTS / independent values
    import csv
    csv_rows = list(csv.DictReader(open(RES / 'PRIMARY_ENDPOINTS.csv')))
    csv_ok = len(csv_rows) == 40 and all(abs(float(rw['point']) - ER[rw['endpoint']]['point']) <= 1e-9 and (rw['iut_pass'] == str(ER[rw['endpoint']]['iut_pass']))
                                        and rw['sim_status'] == ER[rw['endpoint']]['sim_status'] for rw in csv_rows)
    z_all = [abs(x['mc_z_own_vs_lead']) for x in recov_cmp] + [abs(x['mc_z_own_vs_lead']) for x in util_cmp] + [worst['own_mc_z_absmax']]
    mc_ok = max(z_all) <= 4.0
    sens_dis_replay = sorted(k for k, r in ER.items() if ((r['pct_lower'] > r['margin']) if r['kind'] == 'benefit' else (r['pct_upper'] <= r['margin'])) != r['iut_pass'])
    # ---- extra: subgroup endpoints for every roster service, descriptive LOCAL_TREE - TASK_REFIT utility, full reader-slate matrix
    sg_cmp = []
    for (svc, seed, g), (eR, eO) in sg.items():
        le = rep['per_seed'][str(seed)]['services'][svc]['subgroup'][f'sex{g}']
        sg_cmp.append({'key': f'{svc}|s{seed}|sex{g}', 'point_diff': abs(eR['point'] - le['point']), 'se_replay_diff': abs(eR['se'] - le['se']),
                       'decision_match': eR['iut_pass'] == le['iut_pass'] and eR['sim_status'] == le['sim_status']})
    sg_ok = max(x['point_diff'] for x in sg_cmp) <= 1e-9 and max(x['se_replay_diff'] for x in sg_cmp) <= 1e-12 and all(x['decision_match'] for x in sg_cmp)
    desc = rep['descriptive']['local_minus_task_refit_utility']
    desc_diff = max(abs(util[('LOCAL_TREE', s, t, 'U0', m)][0]['point'] - util[('TASK_REFIT', s, t, 'U0', m)][0]['point'] - desc[f's{s}|{t}|{m}'])
                    for s in (0, 1) for t in TASKS for m in CAPS)
    slate_diff, slate_n = 0.0, 0
    for seed in (0, 1):
        with np.load(PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz') as z:
            for svc in ('LOCAL_TREE', 'TASK_REFIT'):
                for view in ('income', 'occupation', 'pair'):
                    lm = rep['descriptive']['reader_slate_auc_matrix'][f'{svc}|s{seed}|{view}']
                    for rd, v in lm.items():
                        slate_diff = max(slate_diff, abs(auc_rank(sex, z[f'{svc}__{view}__{rd}']) - v)); slate_n += 1
    extra_ok = sg_ok and desc_diff <= 1e-12 and slate_diff <= 1e-12
    ok = (align and row_align and crit_ok and counts_ok and recov_point_max <= 1e-9 and recov_reader_ok and recov_se_replay_rel <= 1e-9 and util_point_max <= 1e-9
          and util_se_replay_max <= 1e-9 and worst['point'] <= 1e-9 and worst['replay_se_rel'] <= 1e-9 and not dec_mismatch_replay and claims_replay_agree
          and mc_ok and extra_ok and csv_ok and sens_dis_replay == sorted(rep['sensitivity_disagreements']))
    res = {'id': 8, 'name': 'UTILITY, RECOVERY, SUBGROUP, INFERENCE (DEV_REPORT)', 'status': 'PASS' if ok else 'FAIL',
           'DEV_REPORT_truth_reread': True, 'DEV_REPORT_truth_reread_note': 'verification re-read of the already-opened DEV_REPORT role (exports/DEV_REPORT_TRUTH); no new opening event written',
           'evidence': {'households_G': G, 'persons': int(len(sex)), 'households_per_sex': hh_sex, 'row_alignment_truth_features': bool(align), 'row_alignment_outputs_scores': bool(row_align),
                        'critical_values': {'t_one_sided_alpha_over_3': c1, 't_sim_K40': cs, 'match_lead': crit_ok}, 'counts_match_lead': counts_ok,
                        'recovery_winner_aucs_checked': len(recov_cmp), 'recovery_point_max_abs_diff': recov_point_max, 'recovery_readers_match': recov_reader_ok,
                        'recovery_se_replay_max_rel_diff': recov_se_replay_rel, 'recovery_se_own_seed_ratio_range': [min(recov_se_own), max(recov_se_own)],
                        'pwgtp_weighted_auc_max_abs_diff': max(wauc_cmp),
                        'utility_endpoints_checked_all_services': len(util_cmp), 'utility_point_max_abs_diff': util_point_max, 'utility_se_replay_max_abs_diff': util_se_replay_max,
                        'utility_se_own_seed_ratio_range': [min(util_own_ratios), max(util_own_ratios)],
                        'utility_pwgtp_point_max_abs_diff': max(x['pw_point_diff'] for x in util_cmp),
                        'primary_endpoints': 40, 'primary_point_max_abs_diff': worst['point'], 'primary_se_replay_max_rel_diff': worst['replay_se_rel'],
                        'primary_bounds_replay_max_abs_diff': worst['replay_bound'], 'primary_percentile_replay_max_abs_diff': worst['replay_pct'],
                        'primary_se_own_seed_ratio_range': [worst['own_se_ratio_min'], worst['own_se_ratio_max']],
                        'own_seed_se_mc_z_absmax_primary': worst['own_mc_z_absmax'], 'own_seed_se_mc_z_absmax_all': max(z_all), 'own_seed_se_comparisons': len(z_all) - 1 + 40,
                        'own_seed_se_mc_criterion': '|z| <= 4 where z = (SE_own - SE_lead) / (SE_lead * sqrt(2) * sqrt((kurtosis-1)/(4B)))',
                        'decision_mismatches_replay': dec_mismatch_replay, 'iut_pass_differences_own_seed_MC': dec_diff_own, 'sim_status_differences_own_seed_MC': sim_diff_own,
                        'claims_lead': lead_claims, 'claims_independent_replay': {k: CR[k] for k in lead_claims}, 'claims_independent_own_seed': {k: CO[k] for k in lead_claims},
                        'claims_replay_agree': claims_replay_agree, 'claims_own_seed_agree': claims_own_agree,
                        'utility_sim_counts_replay': CR['utility_sim_counts'], 'utility_sim_counts_own': CO['utility_sim_counts'],
                        'LOCAL_TREE_model_select_feasibility_source': lt_src, 'PRIMARY_ENDPOINTS_csv_consistent': csv_ok,
                        'subgroup_endpoints_all_services_checked': len(sg_cmp), 'subgroup_all_services_match': sg_ok,
                        'subgroup_all_services_point_max_abs_diff': max(x['point_diff'] for x in sg_cmp),
                        'descriptive_local_minus_task_refit_max_abs_diff': desc_diff, 'reader_slate_matrix_aucs_checked': slate_n, 'reader_slate_matrix_max_abs_diff': slate_diff,
                        'sensitivity_disagreements_replay': sens_dis_replay, 'sensitivity_disagreements_lead': rep['sensitivity_disagreements']},
           'endpoints': ep_cmp, 'recovery': recov_cmp,
           'scope': 'point estimates recomputed from stored DEV_REPORT reader scores and stored service probabilities (replay of the scoring), with own AUC/metrics; '
                    'bootstrap both replayed from the registered seed (own implementation of the draw rule) and redrawn with an independent seed',
           'notes': 'PASS requires exact (<=1e-9) points, exact replay SEs/decisions/claims, and own-seed SEs within Monte Carlo error (|z| <= 4, kurtosis-adjusted; '
                    'the nominal ~5% band is reported as the ratio range); own-seed decision differences are MC sensitivity, not FAIL.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check08_dev_report_inference.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print(json.dumps(res['evidence'], indent=1, default=float), res['status'])


if __name__ == '__main__':
    main()

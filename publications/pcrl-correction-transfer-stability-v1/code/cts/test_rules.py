"""Invented-data tests of cts.rules (statistical reviewer). No real data, no file other than RULES.json.

Run: python3 -m cts.test_rules   (prints a JSON summary; exit status 1 on any failure)
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
import traceback

import numpy as np

from . import rules as R

TESTS = []


def test(f):
    TESTS.append(f)
    return f


def _invented(seed, n=1500, scale_delta=1.5):
    """Invented people: households of 1-3 people, SEX 0/1, 5 occupation classes, witness logits z0 and a correction delta."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, size=n)
    unit = np.repeat(np.arange(n), sizes)[:n]
    sex = rng.integers(0, 2, size=n)
    z0 = R.logp(R.probs(rng.normal(0, 1.2, size=(n, 5))))
    y = np.array([rng.choice(5, p=p) for p in R.probs(z0)])
    delta = rng.normal(0, scale_delta, size=(n, 5)) + 0.8 * (sex[:, None] == 0) * np.array([1.0, -1.0, 0.5, 0.0, -0.5])
    return z0, delta, y, sex, unit


def _slack_for_interior(z0, delta, y, sex, f=0.8):
    """Reserve slacks chosen so that the f-level budget is half of the s = 1 cost of every positive statistic."""
    base = R.task_vectors(y, R.probs(z0))
    st1 = R.cost_stats(R.probs(z0 + delta), base, y, sex)
    slack = {}
    for k, cs in R.SLACK_OF.items():
        v = 0.5 * st1[k] / f if st1[k] > 0 else 1.0
        for c in cs:
            slack[c] = v
    return slack


def _baseline(rho=0.004, margin=0.01, se_used=0.0015):
    return {c: {'slack': rho, 'margin': margin, 'se_used': se_used} for c in R.OCC_CONDITIONS}


def _flat(costs_by_stat):
    """{stat: list of 10 values} -> ladder table {s: {stat: value}} (s = 0 included, zero cost)."""
    out = {0.0: {k: 0.0 for k in R.STATS}}
    for j, s in enumerate(R.LADDER[1:]):
        out[s] = {k: float(costs_by_stat[k][j]) for k in R.STATS}
    return out


# ------------------------------------------------------------------ rule file and metric identity
@test
def t_rules_file_matches_constants():
    r = R.load_rules()
    assert r['schema'] == R.SCHEMA and r['rules']['SIMPLE']['gamma'] == 0.5
    assert abs(R.Z - R._norm_ppf(0.95)) < 1e-15
    # gross construction: net split variance + half the reader-seed variance (source's published income components)
    gross = (0.02123682861190588 ** 2 + 0.0018486366498115536 ** 2 / 2) / 0.008962013990683278 ** 2
    assert abs(R.PHI_PRIOR['income'] - gross) < 1e-12 and R.PHI_PRIOR_INCOME == R.PHI_PRIOR['income']
    assert r['screens']['aggregate_paired_rule']['phi_prior'] == R.PHI_PRIOR
    assert R.LADDER == tuple(r['ladder']['strengths'])
    return {'rules_sha256': r['_sha256']}


@test
def t_metrics_identical_to_sources():
    from scm_bench import common as SB
    from ihs import evaluate as E
    from acs_replication import trees
    z0, delta, y, sex, unit = _invented(1)
    p = R.probs(z0 + 0.7 * delta)
    assert np.array_equal(p, trees.probabilities('occupation', z0 + 0.7 * delta))
    a, b = R.task_vectors(y, p), SB.task_vectors(y, p, 5)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    d = a['brier'] - R.task_vectors(y, R.probs(z0))['brier']
    assert R.cluster_se(d, unit) == E.cluster_se(d, unit)
    assert R.subgroup_se(d, sex == 0, unit) == E.subgroup_se(d, sex == 0, unit)
    # the 1e-12 NLL floor is the source's
    q = np.full((2, 5), 0.25); q[:, 0] = 0.0
    assert abs(R.task_vectors(np.array([0, 1]), q)['nll'][0] + math.log(1e-12)) < 1e-12
    return {'ok': True}


@test
def t_cost_stats_identical_to_ohn():
    from ohn import protect as P
    z0, delta, y, sex, unit = _invented(2)
    base = R.task_vectors(y, R.probs(z0))
    p = R.probs(z0 + 0.4 * delta)
    assert R.cost_stats(p, base, y, sex) == P.cost_stats(p, base, y, sex)
    assert R.cost_stats_se(p, base, y, sex, unit) == P.cost_stats_se(p, base, y, sex, unit)
    slack = {c: 0.001 * (i + 1) for i, c in enumerate(R.OCC_CONDITIONS)}
    assert R.budget_at(slack, 0.8) == P.budget_at(slack, 0.8)
    assert R.SLACK_OF == P.SLACK_OF and R.STATS == P.STATS
    return {'ok': True}


# ------------------------------------------------------------------ OLD = ohn.protect.project
@test
def t_old_equals_ohn_project():
    from ohn import protect as P
    out = {}
    for seed in (3, 4, 5):
        z0, delta, y, sex, unit = _invented(seed)
        slack = _slack_for_interior(z0, delta, y, sex)
        for level, f in R.OLD_LEVELS.items():
            mine = R.old_rule(z0, delta, y, sex, slack, level)
            s_src, st_src, curve_src, _ = P.project(z0, delta, y, sex, P.budget_at(slack, f))
            assert mine['selected'] == s_src, (mine['selected'], s_src)          # bitwise equality of the chosen scale
            assert mine['oof_stats'] == st_src
            s2, _, curve2, _ = R.project(z0, delta, y, sex, R.budget_at(slack, f))
            assert curve2 == curve_src and s2 == s_src
            out[f'{seed}_{level}'] = s_src
    assert all(0.0 < v < 1.0 for v in out.values()), out                       # the bisection was exercised
    # edge cases: every grid point feasible (s = 1) and the first nonzero grid point infeasible
    z0, delta, y, sex, unit = _invented(6)
    big = {c: 10.0 for c in R.OCC_CONDITIONS}
    assert R.old_rule(z0, delta, y, sex, big)['selected'] == P.project(z0, delta, y, sex, P.budget_at(big, 0.8))[0] == 1.0
    tiny = {c: 1e-9 for c in R.OCC_CONDITIONS}
    a = R.old_rule(z0, delta, y, sex, tiny)['selected']
    assert a == P.project(z0, delta, y, sex, P.budget_at(tiny, 0.8))[0] and a < 0.01
    neg = {c: -0.001 for c in R.OCC_CONDITIONS}
    rr = R.old_rule(z0, delta, y, sex, neg)
    assert rr['selected'] == 0.0 and 'OLD_NONPOSITIVE_SLACK' in rr['flags']
    return {'scales': out}


# ------------------------------------------------------------------ reserve slack conventions
@test
def t_reserve_slack_floor_handling():
    a = R.reserve_slack(0.010, 0.0, 0.0018)
    assert a['se_source'] == 'floor' and a['own_se_zero'] and abs(a['slack'] - (0.010 - 2 * 0.0018)) < 1e-15
    b = R.reserve_slack(0.010, 1e-9, 0.0018)
    assert b['se_source'] == 'floor' and not b['own_se_zero'] and b['se_used'] == 0.0018
    c = R.reserve_slack(0.010, 0.0025, 0.0018)
    assert c['se_source'] == 'own' and abs(c['slack'] - 0.005) < 1e-15
    for bad in (0.0, -1e-3, float('nan'), None):
        try:
            R.reserve_slack(0.01, 0.001, bad)
            raise AssertionError('floor accepted')
        except ValueError:
            pass
    try:
        R.reserve_slack(0.01, -0.001, 0.001)
        raise AssertionError('negative SE accepted')
    except ValueError:
        pass
    d = R.reserve_slack(float('nan'), 0.001, 0.001)
    assert d['slack'] == -math.inf and not d['met']
    e = R.reserve_slack(0.01, float('inf'), 0.001)
    assert e['slack'] == -math.inf
    return {'ok': True}


@test
def t_witness_slacks_rescale_and_caps():
    conds = {c: {'value': -0.004 if c.startswith('guard') else 0.0, 'se': 0.0,
                 'cap': R.GUARD_CAP if c.startswith('guard') else R.CAPS[c.split('|')[2]]} for c in R.OCC_CONDITIONS}
    w = R.witness_slacks(conds, 0, 1326)
    fl = R.load_rules()['reserve']['floor_se']['0']['guard|sex0'] * math.sqrt(5306 / 1326)
    assert abs(w['guard|sex0']['se_used'] - fl) < 1e-15 and w['guard|sex0']['se_source'] == 'floor'
    assert abs(w['guard|sex0']['slack'] - (0.009 - 2 * fl)) < 1e-15 and w['guard|sex0']['own_se_zero']
    conds['guard|sex1']['cap'] = 0.006
    try:
        R.witness_slacks(conds, 0, 5306)
        raise AssertionError('changed cap accepted')
    except ValueError:
        pass
    return {'ok': True}


# ------------------------------------------------------------------ SIMPLE
@test
def t_simple_fixed_sequence_stops_at_first_failure():
    rho = 0.004
    men = [0.0004, 0.0012, 0.0025, 0.0015, 0.0010, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]   # 0.3 fails, 0.4+ would pass
    gc = _flat({'men_brier': men, 'women_brier': [0.0] * 10, 'brier': [0.0] * 10, 'nll': [0.0] * 10, 'acc_loss': [0.0] * 10})
    r = R.simple_rule(gc, {c: rho for c in R.OCC_CONDITIONS})
    assert r['selected'] == 0.2 and r['first_failure'] == 0.3 and r['accepted_after_first_failure_not_used'][0] == 0.4
    assert r['per_strength'][0.3]['failing'] == ['guard|sex0']
    return {'selected': r['selected']}


@test
def t_simple_identity_when_first_fails_and_all_pass():
    rho = 0.004
    gc = _flat({k: [0.0025] + [0.0] * 9 for k in R.STATS})
    r = R.simple_rule(gc, {c: rho for c in R.OCC_CONDITIONS})
    assert r['selected'] == 0.0 and r['identity'] and r['first_failure'] == 0.1
    gc2 = _flat({k: [0.0019] * 10 for k in R.STATS})
    assert R.simple_rule(gc2, {c: rho for c in R.OCC_CONDITIONS})['selected'] == 1.0
    # string keys and a non-finite cost
    gc3 = {str(s): v for s, v in gc2.items()}
    gc3['0.5'] = dict(gc3['0.5'], nll=float('nan'))
    assert R.simple_rule(gc3, {c: rho for c in R.OCC_CONDITIONS})['selected'] == 0.4
    return {'ok': True}


@test
def t_simple_nonpositive_slack_is_never_loosened():
    sl = {c: 0.004 for c in R.OCC_CONDITIONS}
    sl['guard|sex1'] = -0.001
    gc = _flat({'men_brier': [0.0] * 10, 'women_brier': [-0.0007] * 10, 'brier': [0.0] * 10, 'nll': [0.0] * 10,
                'acc_loss': [0.0] * 10})
    r = R.simple_rule(gc, sl)
    assert r['thresholds']['guard|sex1'] == -0.001 and r['selected'] == 0.0 and 'NONPOSITIVE_SLACK' in r['flags']
    gc[0.1]['women_brier'] = -0.0011
    assert R.simple_rule(gc, sl)['selected'] == 0.1          # a correction that improves the condition enough
    return {'ok': True}


# ------------------------------------------------------------------ UNCERTAINTY
N_G = {'all': 2800, 'men': 1450, 'women': 1350}
N_MS = {'all': 8001, 'men': 4150, 'women': 3851}


def _se_table(v):
    return _flat({k: [v] * 10 for k in R.STATS})


@test
def t_uncertainty_fixed_sequence_nonmonotone_and_identity():
    base = _baseline()
    men = [0.0003, 0.0008, 0.0035, 0.0010, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008]   # 0.3: 0.0035 + z tau > rho
    gc = _flat({'men_brier': men, 'women_brier': [0.0] * 10, 'brier': [0.0] * 10, 'nll': [0.0] * 10, 'acc_loss': [0.0] * 10})
    r = R.uncertainty_rule(gc, _se_table(0.0004), base, N_G, N_MS)
    assert r['selected'] == 0.2 and r['first_failure'] == 0.3 and 0.4 in r['accepted_after_first_failure_not_used']
    gc[0.1]['men_brier'] = 0.0038                                         # 0.0038 + z tau > 0.004
    r2 = R.uncertainty_rule(gc, _se_table(0.0004), base, N_G, N_MS)
    assert r2['selected'] == 0.0 and r2['identity'] and r2['first_failure'] == 0.1
    return {'selected': r['selected'], 'tau_inflation_men': r['inflation']['men_brier']}


@test
def t_uncertainty_formula_and_predictive_inflation():
    base = _baseline(rho=0.004)
    gc = _flat({k: [0.001] * 10 for k in R.STATS})
    r = R.uncertainty_rule(gc, _se_table(0.0005), base, N_G, N_MS)
    rec = r['per_strength'][0.5]['conditions']['guard|sex1']
    infl = math.sqrt(1 + 1350 / 3851)
    assert abs(rec['tau'] - 0.0005 * infl) < 1e-15 and abs(rec['bound_U1'] - (0.001 + R.Z * 0.0005 * infl)) < 1e-15
    assert abs(rec['bound_U2'] - (0.001 + R.Z * math.sqrt(0.0005 ** 2 + 0.0015 ** 2))) < 1e-15
    assert abs(r['inflation']['brier'] - math.sqrt(1 + 2800 / 8001)) < 1e-15
    return {'bound_U1': rec['bound_U1']}


@test
def t_uncertainty_intersection_union():
    """Every one of the 8 conditions must pass; one failing condition blocks the strength; z does not grow with the number
    of conditions (no Bonferroni)."""
    base = _baseline(rho=0.004)
    base['occupation|Ucal|nll_excess'] = {'slack': 0.0011, 'margin': 0.01, 'se_used': 0.0015}   # 0.001 + z tau = 0.00119
    gc = _flat({k: [0.001] * 10 for k in R.STATS})
    r = R.uncertainty_rule(gc, _se_table(0.0001), base, N_G, N_MS)
    assert r['selected'] == 0.0 and r['per_strength'][0.1]['failing'] == ['occupation|Ucal|nll_excess']
    assert r['z'] == R.Z                                          # one-sided 0.05 for each component, 8 conditions
    base['occupation|Ucal|nll_excess'] = {'slack': 0.004, 'margin': 0.01, 'se_used': 0.0015}
    assert R.uncertainty_rule(gc, _se_table(0.0001), base, N_G, N_MS)['selected'] == 1.0
    return {'ok': True}


@test
def t_uncertainty_U1_implies_U2():
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(4000):
        se_b = rng.uniform(1e-4, 4e-3); m = rng.uniform(0, 0.02); rho = m - 2 * se_b
        d = rng.normal(0, 0.003); se = rng.uniform(0, 0.002); infl = math.sqrt(1 + rng.uniform(0, 1))
        if d + R.Z * se * infl <= rho:
            checked += 1
            assert d + R.Z * math.sqrt(se ** 2 + se_b ** 2) <= m + 1e-15
    assert checked > 100
    return {'cases_with_U1': checked}


@test
def t_uncertainty_tiny_se_floor():
    """acc_loss with no changed decision: estimate 0, SE 0 -> the one-decision floor 1/n_GUARD is used; it binds when the
    accuracy slack is tighter than z * floor * inflation."""
    base = _baseline(rho=0.004)
    gc = _flat({k: [0.0] * 10 for k in R.STATS})
    se = _se_table(0.0)
    r = R.uncertainty_rule(gc, se, base, N_G, N_MS)
    rec = r['per_strength'][0.1]['conditions']['occupation|U0|accuracy_loss']
    assert rec['se_floor_used'] and abs(rec['se_used_increment'] - 1 / 2800) < 1e-18 and r['selected'] == 1.0
    tight = R.Z * (1 / 2800) * math.sqrt(1 + 2800 / 8001) * 0.9
    for c in ('occupation|U0|accuracy_loss', 'occupation|Ucal|accuracy_loss'):
        base[c] = {'slack': tight, 'margin': 0.01, 'se_used': 0.0035}
    assert R.uncertainty_rule(gc, se, base, N_G, N_MS)['selected'] == 0.0
    # Brier with SE 0 and zero estimate: no floor (an unchanged release is accepted)
    assert not r['per_strength'][0.1]['conditions']['guard|sex0']['se_floor_used']
    try:
        R.uncertainty_rule(gc, _se_table(-1e-4), _baseline(), N_G, N_MS)
        raise AssertionError('negative SE accepted')
    except ValueError:
        pass
    return {'ok': True}


@test
def t_fixed_sequence_fwer_simulation():
    """Non-monotone true costs with one infeasible strength (0.4) between feasible ones: under normal noise with the stated
    SE, P(selected strength infeasible) <= alpha (here the infeasible point sits exactly on the boundary)."""
    rng = np.random.default_rng(11)
    rho, se = 0.004, 0.0006
    true_men = np.array([0.0005, 0.0010, 0.0020, 0.0040000001, 0.0015, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010])
    base = _baseline(rho=rho, margin=0.02, se_used=0.0015)
    zero = [0.0] * 10
    n_sim, bad, ident = 3000, 0, 0
    se_tab = _flat({'men_brier': [se] * 10, 'women_brier': zero, 'brier': zero, 'nll': zero, 'acc_loss': zero})
    for _ in range(n_sim):
        est = true_men + rng.normal(0, se, size=10)
        r = R.uncertainty_rule(_flat({'men_brier': est, 'women_brier': zero, 'brier': zero, 'nll': zero, 'acc_loss': zero}),
                               se_tab, base, N_G, N_MS)
        j = round(r['selected'] * 10)
        bad += bool(j >= 1 and true_men[j - 1] > rho)
        ident += r['identity']
    rate = bad / n_sim
    assert rate <= R.ALPHA + 3 * math.sqrt(R.ALPHA * (1 - R.ALPHA) / n_sim), rate
    return {'infeasible_selected_rate': rate, 'identity_rate': ident / n_sim}


# ------------------------------------------------------------------ diagnostics and question A
@test
def t_transfer_errors_and_outcomes():
    slack = {c: 0.004 for c in R.OCC_CONDITIONS}
    pred = _flat({k: [0.001] * 10 for k in R.STATS})
    real = _flat({k: [0.0025] * 10 for k in R.STATS})
    e = R.transfer_errors(pred, real, slack)
    assert abs(e[0.5]['men_brier']['signed'] - 0.0015) < 1e-15 and abs(e[0.5]['men_brier']['rel_slack'] - 0.375) < 1e-12
    base = _baseline(rho=0.002, margin=0.0024)
    o = R.rule_outcome(0.5, real[0.5], base)
    assert o['reserve_crossing'] and o['point_cap_crossing'] and abs(o['max_slack_use'] - 1.25) < 1e-12
    o2 = R.rule_outcome(0.0, {k: 0.0 for k in R.STATS}, base)
    assert o2['identity'] and not o2['reserve_crossing']
    real2 = _flat({k: [0.0005, 0.001, 0.0015, 0.0019, 0.0025, 0.003, 0.003, 0.003, 0.003, 0.003] for k in R.STATS})
    ac = R.accepted_crossings({'selected': 0.6}, real2, _baseline(rho=0.002, margin=0.0027))
    assert ac['accepted'] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] and ac['n_reserve_crossings'] == 2 and ac['n_point_cap_crossings'] == 1
    z0, delta, y, sex, unit = _invented(8)
    m0 = R.correction_magnitude(z0, delta, 0.0)
    m1 = R.correction_magnitude(z0, delta, 0.5)
    assert m0 == {'mean_abs_logit_change': 0.0, 'flip_rate': 0.0} and m1['mean_abs_logit_change'] > 0 and m1['flip_rate'] > 0
    return {'ok': True}


@test
def t_prediction_comparison_bootstrap():
    """GUARD unbiased, OOF biased by +40% of the slack on men's Brier (cost under-predicted): D > 0, bootstrap 5% quantile > 0."""
    rng = np.random.default_rng(12)

    def make(n, bias_men=0.0):
        unit = np.repeat(np.arange(n), 2)[:n]
        sex = rng.integers(0, 2, size=n)
        s = np.arange(1, 11) / 10
        db = (0.003 * s)[None, :] * (1 + 0.8 * (sex[:, None] == 0)) + rng.normal(0, 0.02, size=(n, 10)) * s[None, :]
        db = db - bias_men * (sex[:, None] == 0) * s[None, :]
        dn = 1.5 * db + rng.normal(0, 0.01, size=(n, 10)) * s[None, :]
        da = (rng.random((n, 10)) < 0.01 * s[None, :]).astype(float) - (rng.random((n, 10)) < 0.008 * s[None, :])
        return {'db': db, 'dn': dn, 'da': da, 'unit': unit, 'sex': sex}
    slack = {c: 0.004 for c in R.OCC_CONDITIONS}
    sets = {'OOF': make(6000, bias_men=0.0016 * 2), 'GUARD': make(2000), 'CHECK': make(2000)}
    r = R.prediction_comparison(sets, slack, B=400)
    assert r['D'] > 0 and r['q05'] > 0 and r['label'] == 'PREDICTION_IMPROVED', r
    assert r['mean_signed_error']['OOF']['men_brier'] > 0
    sets2 = {'OOF': make(6000), 'GUARD': make(2000), 'CHECK': make(2000)}
    r2 = R.prediction_comparison(sets2, slack, B=400)
    assert r2['label'] == 'NOT_IMPROVED'                     # unbiased OOF with 3x the people predicts at least as well
    return {'biased_oof': {k: round(r[k], 4) for k in ('D', 'q05')}, 'unbiased_oof_D': round(r2['D'], 4)}


# ------------------------------------------------------------------ screens
SRC = {'sd_split': 0.02123682861190588, 'sd_reader': 0.0018486366498115536, 'se_cell': 0.008962013990683278}


@test
def t_per_cell_gate():
    cells = {f'S{s}R{r}k{k}': 0.0 for s in (0, 1) for r in (1, 2) for k in (0, 1)}
    assert R.per_cell_gate(cells)['pass']
    cells['S0R1k0'] = 0.0100
    assert R.per_cell_gate(cells)['pass']                    # inclusive
    cells['S0R1k1'] = 0.0101
    g = R.per_cell_gate(cells)
    assert not g['pass'] and g['cells_above'] == ['S0R1k1']
    cells['S1R2k1'] = float('nan')
    assert R.per_cell_gate(cells)['label'] == 'INCOMPLETE'
    return {'ok': True}


@test
def t_aggregate_formula_and_alias():
    cells = {'S0R1k0': 0.004, 'S0R1k1': 0.006, 'S0R2k0': -0.010, 'S0R2k1': -0.008,
             'S1R1k0': 0.002, 'S1R1k1': 0.002, 'S1R2k0': -0.012, 'S1R2k1': -0.012}
    ses = {c: 0.004 for c in cells}
    a = R.aggregate_paired_screen(cells, ses, R.PHI_PRIOR_INCOME)
    y1, y2 = 0.0035, -0.0105
    dbar = (y1 + y2) / 2
    v_samp = 2 * 0.004 ** 2 / 4
    phi_emp = ((y1 - dbar) ** 2 + (y2 - dbar) ** 2) / 0.004 ** 2
    U = dbar + R.Z * math.sqrt(max(1, R.PHI_PRIOR_INCOME, phi_emp) * v_samp)
    assert abs(a['dbar'] - dbar) < 1e-15 and abs(a['phi_emp'] - phi_emp) < 1e-9 and abs(a['U'] - U) < 1e-12
    assert a['pass'] == (U <= 0.010)
    zero = {c: 0.0 for c in cells}
    al = R.aggregate_paired_screen(zero, ses, R.PHI_PRIOR_INCOME, alias=True)
    assert al['pass'] and al['label'] == 'EXACT_ALIAS' and R.per_cell_gate(zero)['pass']
    try:
        R.aggregate_paired_screen(cells, ses, R.PHI_PRIOR_INCOME, alias=True)
        raise AssertionError('alias with nonzero cells accepted')
    except ValueError:
        pass
    try:
        R.aggregate_paired_screen(cells, ses, 'COMPUTE_FROM_FROZEN_SOURCE_GRID_BEFORE_NEW_OUTCOMES')
        raise AssertionError('missing phi_prior accepted')
    except ValueError:
        pass
    inc = dict(cells); inc.pop('S1R2k1')
    assert R.aggregate_paired_screen(inc, {c: 0.004 for c in inc}, 5.0)['label'] == 'INCOMPLETE'
    return {'U': a['U'], 'pass': a['pass']}


@test
def t_screens_zero_and_nonzero_shift():
    """Law L1 with small noise (near-identical releases read by separately fitted readers): zero shift passes both screens
    most of the time; a +0.020 shift under the source's income-view noise almost never passes the aggregate screen."""
    small = R.oc_rates(0.0, 2000, 21, sd_split=0.003, sd_reader=0.002, se_cell=0.003, rho_seed=0.76)
    assert small['per_cell_gate_pass'] > 0.95 and small['aggregate_pass'] > 0.5, small
    big = R.oc_rates(0.020, 2000, 22, rho_seed=0.76, **SRC)
    assert big['aggregate_pass'] < 0.01, big
    neg = R.oc_rates(-0.030, 2000, 23, rho_seed=0.76, **SRC)
    assert neg['aggregate_pass'] > 0.5
    return {'zero_small_noise': small, 'plus_0020_source_noise': big, 'minus_0030_source_noise': neg}


@test
def t_oc_reproduces_source_gate_false_failure():
    """Under the source's own assumptions (independent seeds, income-view components), the per-cell gate fails a true zero
    difference with probability about 0.806 (published 0.8058 for one comparator, 8 cells)."""
    r = R.oc_rates(0.0, 20000, 31, rho_seed=0.0, **SRC)
    fail = 1 - r['per_cell_gate_pass']
    assert abs(fail - 0.806) < 0.015, fail
    return {'gate_fail_at_zero': fail, 'aggregate_pass_at_zero': r['aggregate_pass']}


@test
def t_aggregate_type1_at_allowance_boundary():
    """True difference exactly +0.010 (the allowance) under source income-view noise with shared splits (rho 0.76): the
    aggregate screen admits at a rate <= alpha (simulated, this law only)."""
    r = R.oc_rates(0.010, 20000, 41, rho_seed=0.76, **SRC)
    se_mc = math.sqrt(R.ALPHA * (1 - R.ALPHA) / 20000)
    assert r['aggregate_pass'] <= R.ALPHA + 3 * se_mc, r
    return {'aggregate_pass_at_boundary': r['aggregate_pass'], 'per_cell_gate_pass_at_boundary': r['per_cell_gate_pass']}


# ------------------------------------------------------------------ selection, comparisons, claims
def _grid4(vals, se=0.004):
    cells = {f'S0R{r}k{k}': v for (r, k), v in zip([(1, 0), (1, 1), (2, 0), (2, 1)], vals)}
    return {'cells': cells, 'cell_se': {c: se for c in cells}}


def _cand(rid, arm, rule, routing, deployment, entry, vals0, vals1, ratios=(3.0,) * 8, crossing=False, sel=0.5):
    D = {}
    for P, v in (('D0', vals0), ('D1', vals1)):
        D[P] = {'selected': sel, 'reserve_crossing_check': crossing, 'reserve_crossing_ms': False, 'point_pass_ms': True,
                'cont_grid': _grid4(v), 'ms_reserve_ratios': list(ratios)}
    return {'recipe': rid, 'arm': arm, 'rule': rule, 'routing': routing, 'deployment': deployment, 'entry': entry, 'D': D}


@test
def t_select_recipe_tie_breaks():
    good = [0.030, 0.031, 0.029, 0.030]
    cands = [
        _cand('A', 'PAIR', 'OLD-B80', 'REFIT', 'full', 1, good, good, ratios=(2.5,) * 8),
        _cand('B', 'LOC', 'SIMPLE', 'REFIT', 'full', 2, good, [0.029, 0.030, 0.028, 0.029], ratios=(2.5,) * 8),
        _cand('C', 'PAIR', 'SIMPLE', 'REFIT', 'full', 3, [0.08, 0.08, 0.08, 0.08], [0.0, 0.002, -0.002, 0.0]),  # one low AUC
        _cand('D', 'PAIR', 'UNCERTAINTY', 'REFIT', 'full', 4, good, good, crossing=True),
        _cand('E', 'CONT', 'SIMPLE', 'REFIT', 'full', 5, good, good),
        _cand('F', 'PAIR', 'SIMPLE', 'REFIT', 'full', 6, good, good, sel=0.0),
    ]
    r = R.select_recipe(cands)
    lab = {x['recipe']: x['label'] for x in r['records']}
    assert lab['D'] == 'GATE_FAILED' and lab['E'] == 'NOT_A_RECIPE_ARM' and lab['F'] == 'GATE_FAILED'
    assert lab['C'] == 'ELIGIBLE_NOT_CHOSEN'                 # repeatable benefit beats one exceptional partition
    assert set(r['near_tie_set']) == {'A', 'B'} and r['selected'] == 'B'   # equal ratios -> rule order SIMPLE before OLD
    cands[0] = _cand('A', 'PAIR', 'OLD-B80', 'REFIT', 'full', 1, good, good, ratios=(2.6,) * 8)
    assert R.select_recipe(cands)['selected'] == 'A'          # larger realized reserve ratio first
    return {'selected': r['selected']}


@test
def t_compare_rules_shrinkage_is_not_improvement():
    a = {'rule': 'UNCERTAINTY', 'selected': 0.0, 'reserve_crossing': False, 'pair_auc': 0.64}
    b = {'rule': 'OLD-B80', 'selected': 0.8, 'reserve_crossing': True, 'pair_auc': 0.58}
    c = R.compare_rules(a, b)
    assert c['category'] == 'A_IDENTITY_NO_IMPROVEMENT' and not c['a_improves']
    a2 = dict(a, selected=0.5, pair_auc=0.60)
    c2 = R.compare_rules(a2, b)
    assert c2['category'] == 'A_SAFER_AVOIDS_CROSSING' and c2['a_improves'] and abs(c2['protection_change_b_minus_a'] + 0.02) < 1e-12
    b2 = dict(b, reserve_crossing=False)
    assert R.compare_rules(a2, b2)['category'] == 'A_WEAKER_NO_GAIN'
    assert R.compare_rules(dict(a2, selected=0.9), b2)['category'] == 'A_STRONGER_WITHOUT_CROSSING'
    return {'ok': True}


@test
def t_claim_ladder_order_and_fixed_labels():
    ev = {'arm': 'PAIR', 'prediction_labels': {P: 'PREDICTION_IMPROVED' for P in ('D0', 'D1', 'R0', 'R1')},
          'utility': {P: {'nonzero': True, 'ms_reserve_kept': True, 'check_no_crossing': True, 'point_pass': True} for P in ('R0', 'R1')},
          'matched_controls': {'CONT': 'STABLE_POSITIVE', 'LOC': 'POSITIVE_NOT_STABLE', 'COMP': 'STABLE_POSITIVE', 'W': 'STABLE_POSITIVE'},
          'frontier': {'R0': True, 'R1': True},
          'benchmark': {'point_and_gate': True, 'superiority_all_required': True, 'target_label': 'DEV_GRID_TARGET_POINT_ONLY'}}
    c = R.claim_ladder(ev)
    assert c['IMPROVED_COST_PREDICTION'] == 'CLAIMABLE' and c['UTILITY_RETAINED_AT_NONZERO_CORRECTION'] == 'CLAIMABLE'
    assert c['INCREMENTAL_PROTECTION_BEYOND_MATCHED_CONTROLS'] == 'NOT_MET'
    assert c['STABLE_FRONTIER_IMPROVEMENT'] == 'MET_BUT_NOT_CLAIMABLE_LOWER_RUNG_UNMET'
    assert c['highest_claimable_rung'] == 2 and c['JOINT_DESIGN_ADVANTAGE'] == 'NOT_CLAIMABLE' and c['NOVELTY'] == 'NOT_CLAIMED'
    assert c['MATERIAL_TARGET_0020'] == 'DEV_GRID_TARGET_POINT_ONLY'
    ev2 = dict(ev, arm='LOC')                                    # LOC control not required for a LOC recipe
    assert R.claim_ladder(ev2)['highest_claimable_rung'] == 5
    ev['prediction_labels']['R1'] = 'NOT_IMPROVED'               # D improvement not replicated on a repair panel
    assert R.claim_ladder(ev)['highest_claimable_rung'] == 0
    return {'ok': True}


@test
def t_reserve_qualified_frontier():
    pts = [{'id': 'SUCC', 'kind': 'reference', 'min_ratio': {0: 0.321, 1: 0.624}, 'pair_auc': {0: 0.60, 1: 0.60}},
           {'id': 'NEURAL_RAW', 'kind': 'reference', 'min_ratio': {0: 1.947, 1: 1.558}, 'pair_auc': {0: 0.62, 1: 0.62}},
           {'id': 'X', 'kind': 'candidate', 'min_ratio': {0: 2.2, 1: 2.7}, 'pair_auc': {0: 0.61, 1: 0.62}},
           {'id': 'Y', 'kind': 'candidate', 'min_ratio': {0: 2.9, 1: 2.8}, 'pair_auc': {0: 0.63, 1: 0.63}},
           {'id': 'Z', 'kind': 'control', 'min_ratio': {0: 2.1, 1: 2.1}, 'pair_auc': {0: 0.64, 1: 0.64}}]
    r = R.reserve_qualified_frontier(pts)
    assert r['references_failing_reserve'] == ['SUCC', 'NEURAL_RAW'] and set(r['frontier']) == {'X', 'Y'}
    return {'frontier': r['frontier']}


def main():
    results, failed = {}, []
    for t in TESTS:
        try:
            results[t.__name__] = {'status': 'PASS', 'detail': t()}
        except Exception as e:                                   # report every failure, then exit nonzero
            results[t.__name__] = {'status': 'FAIL', 'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]}
            failed.append(t.__name__)
    raw = R.RULES_PATH.read_bytes()
    summary = {'unit': 'CTS1_RULES_TEST', 'n_tests': len(TESTS), 'n_pass': len(TESTS) - len(failed), 'failed': failed,
               'rules_sha256': hashlib.sha256(raw).hexdigest(), 'results': results}
    print(json.dumps(summary, indent=1, default=float))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

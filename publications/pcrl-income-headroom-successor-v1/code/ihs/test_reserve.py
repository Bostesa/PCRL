"""Invented-data checks of ihs.reserve (no real data, no label, no private file). Run from the worktree root:

    OMP_NUM_THREADS=1 python3 -m ihs.test_reserve

Every number below is invented. The only real file touched is a STRUCTURAL check of the registered RESERVE_RULE.json (keys,
counts, signs, unchanged caps); no statistic is computed from it.
"""
from __future__ import annotations
import copy
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ihs import reserve as R
else:
    from . import reserve as R

CAP = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}
NAMES = R.INCOME + R.OCCUPATION + R.GUARDS


def invented_rule(k=2.0):
    floors = {s: {n: 0.001 for n in NAMES} for s in ('0', '1')}
    inh_vals = {s: {n: (0.004 if n == 'guard|sex0' else 0.001) for n in R.OCCUPATION + R.GUARDS} for s in ('0', '1')}
    caps = {n: (0.005 if n.startswith('guard|') else CAP[n.split('|')[2]]) for n in R.OCCUPATION + R.GUARDS}
    inh_marg = {s: {n: caps[n] - inh_vals[s][n] for n in inh_vals[s]} for s in ('0', '1')}
    return {'schema': R.SCHEMA, 'multiplier_k': k, 'floor': {'households': 5306, 'se': floors},
            'occupation': {'inherited_values': inh_vals, 'inherited_margins': inh_marg, 'identity_tolerance': 1e-12,
                           'inherited_wire_sha256': None}}


def invented_m(seed, rule, income=None, occupation=None, households=5306, se=0.0012):
    """A dict laid out like ihs.evaluate.measure: income clauses at value 0.001 / SE `se` unless overridden; occupation clauses
    and guards equal to the invented inherited values unless overridden. Overrides: {name: (value, se)}."""
    s = str(seed)
    income, occupation = income or {}, occupation or {}
    cl, gu = [], []
    for n in R.INCOME + R.OCCUPATION:
        t, ref, met = n.split('|')
        if t == 'income':
            v, e = income.get(n, (0.001, se))
        else:
            v, e = occupation.get(n, (rule['occupation']['inherited_values'][s][n], 0.0015))
        cap = CAP[met]
        cl.append({'seed': seed, 'task': t, 'reference': ref, 'metric': met, 'value': v, 'cap': cap, 'normalized': v / cap,
                   'point_pass': v <= cap, 'se': e, 'margin': cap - v, 'margin_se_units': (cap - v) / e if e > 0 else math.inf})
    for n in R.GUARDS:
        v, e = occupation.get(n, (rule['occupation']['inherited_values'][s][n], 0.0017))
        gu.append({'seed': seed, 'group': n.split('|')[1], 'value': v, 'cap': 0.005, 'se': e, 'margin': 0.005 - v,
                   'margin_se_units': (0.005 - v) / e, 'point_pass': v <= 0.005})
    return {'utility': {'clauses': cl, 'guards': gu, 'households': households, 'point_pass_12': True, 'guard_pass': True},
            'wire_sha256': {'income': 'inv-inc', 'occupation': 'inv-occ'}}


def expect_raise(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def run():
    rule = invented_rule()
    res = {}

    # 1. comfortable income margins + inherited occupation -> RESERVE_MET; inherited report flags the thin men's guard
    sc = R.screen(invented_m(0, rule), 0, rule=rule)
    res['met_with_inherited_occupation'] = (sc['status'] == 'RESERVE_MET' and sc['failed'] == [] and sc['occupation_side'] == 'INHERITED'
                                            and sc['inherited_constraint_report']['status'] == 'INHERITED_CONSTRAINTS_BIND'
                                            and sc['inherited_constraint_report']['below_reserve'] == ['guard|sex0'])

    # 2. boundary is inclusive: margin == k * se_used meets; a hair less does not, and only that condition is named
    n = 'income|Ucal|brier_excess'
    exact = R.screen(invented_m(0, rule, income={n: (0.005 - 2 * 0.0015, 0.0015)}), 0, rule=rule)
    below = R.screen(invented_m(0, rule, income={n: (0.005 - 2 * 0.0015 + 1e-9, 0.0015)}), 0, rule=rule)
    res['boundary_inclusive'] = exact['status'] == 'RESERVE_MET' and below['status'] == 'RESERVE_NOT_MET' and below['failed'] == [n]

    # 3. floor binds: tiny own SE (barely differs from a reference) cannot buy reserve below k * floor
    fl = R.screen(invented_m(1, rule, income={n: (0.0035, 0.0002)}), 1, rule=rule)
    rec = [r for r in fl['income'] if r['condition'] == n][0]
    res['floor_binds'] = (fl['status'] == 'RESERVE_NOT_MET' and rec['se_source'] == 'floor' and rec['met_with_own_se']
                          and not rec['met'] and fl['income_met_with_own_se_only'] == [n] and abs(rec['ratio'] - 1.5) < 1e-12)

    # 4. own SE above the floor is used as is
    big = R.screen(invented_m(1, rule, income={n: (0.001, 0.0025)}), 1, rule=rule)
    rec = [r for r in big['income'] if r['condition'] == n][0]
    res['own_se_used_when_larger'] = rec['se_source'] == 'own' and abs(rec['required_margin'] - 0.005) < 1e-15 and big['status'] == 'RESERVE_NOT_MET'

    # 5. changed occupation: a guard worse than inherited and below the reserve fails; better than inherited passes as non-worsening
    worse = R.screen(invented_m(0, rule, occupation={'guard|sex0': (0.0045, 0.0017)}), 0, rule=rule)
    better = R.screen(invented_m(0, rule, occupation={'guard|sex0': (0.003, 0.0017)}), 0, rule=rule)
    allres = R.screen(invented_m(0, rule, occupation={'guard|sex0': (0.0001, 0.0017)}), 0, rule=rule)
    res['changed_occupation'] = (worse['status'] == 'RESERVE_NOT_MET' and worse['failed'] == ['guard|sex0']
                                 and worse['occupation_side'] == 'CHANGED_WORSENED'
                                 and better['status'] == 'RESERVE_MET' and better['occupation_side'] == 'CHANGED_NON_WORSENING'
                                 and allres['status'] == 'RESERVE_MET' and allres['occupation_side'] == 'CHANGED_RESERVE_MET')

    # 6. a changed occupation clause above its cap fails even though income is fine
    over = R.screen(invented_m(1, rule, occupation={'occupation|U0|nll_excess': (0.0101, 0.003)}), 1, rule=rule)
    res['changed_occupation_cap_exceeded'] = over['status'] == 'RESERVE_NOT_MET' and over['failed'] == ['occupation|U0|nll_excess'] and not over['point_pass_all']

    # 7. non-finite values are failures, never silent passes
    nf = R.screen(invented_m(0, rule, income={'income|U0|nll_excess': (math.inf, 0.001), n: (0.001, math.nan)}), 0, rule=rule)
    res['nonfinite_fails'] = nf['status'] == 'RESERVE_NOT_MET' and sorted(nf['failed']) == sorted(['income|U0|nll_excess:NONFINITE', n + ':NONFINITE'])

    # 8. malformed input and changed caps are refused
    m = invented_m(0, rule)
    m_cap = copy.deepcopy(m); m_cap['utility']['clauses'][2]['cap'] = 0.006
    m_miss = copy.deepcopy(m); m_miss['utility']['clauses'].pop(0)
    m_dup = copy.deepcopy(m); m_dup['utility']['clauses'].append(copy.deepcopy(m_dup['utility']['clauses'][0]))
    m_g = copy.deepcopy(m); m_g['utility']['guards'][0]['cap'] = 0.006
    res['refuses_malformed'] = all([expect_raise(lambda: R.screen(m_cap, 0, rule=rule)), expect_raise(lambda: R.screen(m_miss, 0, rule=rule)),
                                    expect_raise(lambda: R.screen(m_dup, 0, rule=rule)), expect_raise(lambda: R.screen(m, 1, rule=rule)),
                                    expect_raise(lambda: R.screen(m_g, 0, rule=rule)), expect_raise(lambda: R.screen(m, 2, rule=rule))])

    # 9. household rescaling of the floor: a quarter of the households doubles the floor SE
    rq = copy.deepcopy(rule); rq['floor']['households'] = 5304
    q = R.screen(invented_m(0, rq, income={n: (0.0015, 0.0001)}, households=1326), 0, rule=rq)
    rec = [r for r in q['income'] if r['condition'] == n][0]
    res['floor_rescaled'] = abs(q['floor_scale'] - 2.0) < 1e-12 and abs(rec['floor_se'] - 0.002) < 1e-15 and q['status'] == 'RESERVE_NOT_MET'

    # 10. hash identity (when the rule records the inherited wire) agrees with value identity, and disagreement is an error
    rh = copy.deepcopy(rule); rh['occupation']['inherited_wire_sha256'] = {'0': 'inv-occ', '1': 'other'}
    h = R.screen(invented_m(0, rh), 0, rule=rh)
    res['hash_identity'] = (h['occupation_side'] == 'INHERITED' and h['occupation_identity_method'] == 'wire_sha256'
                            and expect_raise(lambda: R.screen(invented_m(1, rh), 1, rule=rh), AssertionError))

    # 11. two-seed labels
    s0, s1 = R.screen(invented_m(0, rule), 0, rule=rule), R.screen(invented_m(1, rule), 1, rule=rule)
    s1bad = R.screen(invented_m(1, rule, income={n: (0.004, 0.0012)}), 1, rule=rule)
    s0bad = R.screen(invented_m(0, rule, income={n: (0.004, 0.0012)}), 0, rule=rule)
    t_both, t_one = R.two_seed_status([s0, s1]), R.two_seed_status({0: s0, 1: s1bad})
    res['two_seed_labels'] = (t_both['status'] == 'RESERVE_MET' and t_both['meets_reserve']
                              and t_one['status'] == 'RESERVE_ONE_SEED_ONLY' and not t_one['meets_reserve'] and t_one['failed'] == ['s1|' + n]
                              and R.two_seed_status([s0bad, s1bad])['status'] == 'RESERVE_NOT_MET'
                              and R.two_seed_status([s0])['status'] == 'RESERVE_PENDING_SECOND_SEED'
                              and R.two_seed_status([s1bad])['status'] == 'RESERVE_NOT_MET'
                              and abs(t_one['min_income_ratio'] - (0.001 / 0.0012)) < 1e-12
                              and expect_raise(lambda: R.two_seed_status([s0, s0])) and expect_raise(lambda: R.two_seed_status([])))

    # 12. the returned dict carries what ihs.ledger.summarize reads
    res['ledger_fields'] = isinstance(s0['status'], str) and isinstance(s0['failed'], list)

    # 13. STRUCTURAL check of the registered rule file (no statistic computed from it)
    reg = R.load_rule()
    ok = reg['schema'] == R.SCHEMA and reg['multiplier_k'] == 2.0 and reg['floor']['households'] == 5306
    ok &= reg['original_caps_unchanged'] == {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005, 'within_sex_occupation_guard': 0.005}
    for s in ('0', '1'):
        ok &= set(reg['floor']['se'][s]) == set(NAMES) and all(v > 0 for v in reg['floor']['se'][s].values())
        ok &= set(reg['occupation']['inherited_values'][s]) == set(R.OCCUPATION + R.GUARDS)
        ok &= set(reg['occupation']['inherited_margins'][s]) == set(R.OCCUPATION + R.GUARDS)
    res['registered_rule_structure'] = bool(ok) and len(reg['_sha256']) == 64

    out = {'unit': 'IHS_RESERVE_TEST', 'all_pass': all(res.values()), 'checks': res, 'rule_sha256': reg['_sha256']}
    print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    sys.exit(0 if run()['all_pass'] else 1)

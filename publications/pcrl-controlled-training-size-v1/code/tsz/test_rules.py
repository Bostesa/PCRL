"""Invented-data tests of tsz.rules (no real data, no label). Run: python3 -m tsz.test_rules

Covers: the registered rule file and its refusal of altered constants; the reserve (ohn.rules.reserve reused unchanged, inclusive
k = 2 boundary); trigger pass / fail edges (T0-T3, income excluded, boundary); strength selection edges (identity, non-monotone
estimates, inclusive allowance boundary, version-specific allowance, non-finite, malformed tables); contrast summaries and the
size / control / reproduction / encoding labels; margin patterns and interpretation rules; reader stability labels (source
equivalence when every reader seed is effective, duplicated cells, target labels) and the effective-seed check; claim labels and
the closure outcome."""
from __future__ import annotations
import copy
import json
import math
import sys
import tempfile
from pathlib import Path

from ohn import rules as OR
from tsz import rules as R

CAPS = R.CAPS
OCC = {('U0', 'accuracy_loss'): -0.0098, ('U0', 'nll_excess'): -0.0128, ('U0', 'brier_excess'): -0.0055,
       ('Ucal', 'accuracy_loss'): -0.0098, ('Ucal', 'nll_excess'): -0.0128, ('Ucal', 'brier_excess'): -0.0055}
INC = {('U0', 'accuracy_loss'): 0.0017, ('U0', 'nll_excess'): -0.0050, ('U0', 'brier_excess'): -0.0030,
       ('Ucal', 'accuracy_loss'): 0.0017, ('Ucal', 'nll_excess'): -0.0050, ('Ucal', 'brier_excess'): -0.0030}
GUARDS = {'sex0': -0.0023, 'sex1': -0.0087}


def util(v, occ=None, inc=None, guards=None, se=None, households=5306):
    """Invented evaluator utility dict of one version: 12 clauses + 2 guards; se = {condition name: own SE} (default 0.001,
    below every floor, so the floor is used)."""
    occ = {**OCC, **(occ or {})}
    inc = {**INC, **(inc or {})}
    guards = {**GUARDS, **(guards or {})}
    se = se or {}
    clauses = []
    for task, vals in (('income', inc), ('occupation', occ)):
        for (ref, m), x in vals.items():
            clauses.append({'task': task, 'reference': ref, 'metric': m, 'value': x, 'cap': CAPS[m],
                            'se': se.get(f'{task}|{ref}|{m}', 0.001), 'seed': v})
    gl = [{'group': g, 'value': x, 'cap': R.GUARD_CAP, 'se': se.get(f'guard|{g}', 0.001), 'seed': v} for g, x in guards.items()]
    return {'households': households, 'clauses': clauses, 'guards': gl}


def evals(over=None):
    return {v: util(v, **(over or {}).get(v, {})) for v in R.VERSIONS}


DEPLOY_OK = {0: {'bitwise': True, 'refusals_all_pass': True}, 1: {'bitwise': True, 'refusals_all_pass': True}}
H3_OK = {0: True, 1: True}


def exact_margin_value(cap, target_margin):
    """A float value with cap - value == target_margin exactly (searched with nextafter)."""
    x = cap - target_margin
    for _ in range(64):
        d = cap - x
        if d == target_margin:
            return x
        x = math.nextafter(x, math.inf if d > target_margin else -math.inf)
    raise AssertionError('no exact boundary value found')


# ---------------------------------------------------------------- rule file
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'TSZ1-rules-v1' and len(r['_sha256']) == 64
    assert OR.load_rules()['_sha256'] == R.OHN_RULE_SHA256
    raw = json.loads(Path(R.RULES_PATH).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='tsz_rules_test_'))
    for path_, val in ((('selection', 'allowance_lambda'), 2.0), (('reserve', 'multiplier_k'), 1.5),
                       (('correction', 'strengths'), [0, 0.1, 1.0]), (('original_limits', 'within_sex_guard_cap'), 0.006),
                       (('correction', 'primary_arm'), 'PAIR'), (('correction', 'guard_none_targets_constant'), {'men': 1, 'women': 1})):
        bad = copy.deepcopy(raw)
        bad[path_[0]][path_[1]] = val
        p = tmp / f'bad_{path_[1]}.json'
        p.write_text(json.dumps(bad))
        try:
            R.load_rules(p)
        except ValueError:
            continue
        raise AssertionError(f'altered {path_} was accepted')
    al = R.allowances()
    for v in R.VERSIONS:
        fl = OR.load_rules()['reserve']['floor']['se'][str(v)]
        for j, cs in R.SLACK_OF.items():
            assert al[v][j] == R.ALLOWANCE_LAMBDA * min(fl[c] for c in cs)


# ---------------------------------------------------------------- reserve and trigger
def t_reserve_matches_source_and_inclusive_boundary():
    o = R.occupation_reserve(evals())
    assert o['status'] == 'OCC_RESERVE_MET_BOTH'
    src = OR.reserve(evals())
    for v in R.VERSIONS:
        for c in R.OCC_SIDE:
            assert o['per_version'][v]['conditions'][c]['ratio'] == src['per_seed'][v]['conditions'][c]['ratio']
        assert o['per_version'][v]['binding'] == 'guard|sex0'
    # boundary: own SE 2^-8 (above the floor) and margin exactly 2 x 2^-8 -> met; one ulp less margin -> not met
    se = 0.00390625
    x = exact_margin_value(0.01, 2 * se)
    e = evals({1: {'occ': {('U0', 'accuracy_loss'): x}, 'se': {'occupation|U0|accuracy_loss': se}}})
    rec = R.occupation_reserve(e)['per_version'][1]['conditions']['occupation|U0|accuracy_loss']
    assert rec['margin'] == 2 * se and rec['met'] and rec['se_source'] == 'own'
    x2 = math.nextafter(x, math.inf)
    while not (0.01 - x2) < 2 * se:                                   # the smallest representable step below the boundary
        x2 = math.nextafter(x2, math.inf)
    e2 = evals({1: {'occ': {('U0', 'accuracy_loss'): x2}, 'se': {'occupation|U0|accuracy_loss': se}}})
    o2 = R.occupation_reserve(e2)
    assert o2['per_version'][1]['status'] == 'OCC_RESERVE_NOT_MET' and o2['status'] == 'OCC_RESERVE_ONE_VERSION'
    assert o2['per_version'][1]['point_pass']                       # the point cap still passes: reserve and point are separate
    rho = o['per_version'][0]['conditions']['guard|sex0']['rho']
    assert abs(rho - ((0.005 - -0.0023) - 2 * OR.load_rules()['reserve']['floor']['se']['0']['guard|sex0'])) < 1e-15


def t_trigger_pass_and_fail_edges():
    t = R.trigger(evals(), DEPLOY_OK, H3_OK)
    assert t['fires'] and t['decision'] == 'TRIGGER_FIRES' and t['failing_parts'] == []
    # T1 + T2: a guard over the point cap on version 1
    t = R.trigger(evals({1: {'guards': {'sex0': 0.0051}}}), DEPLOY_OK, H3_OK)
    assert not t['fires'] and 'T1' in t['failing_parts'] and 'T2' in t['failing_parts']
    assert t['parts']['T1']['detail'][1] == ['guard|sex0']
    # T2 only: point passes, reserve fails (men's guard margin 0.0030 < 2 x floor 0.0017)
    t = R.trigger(evals({0: {'guards': {'sex0': 0.0020}}}), DEPLOY_OK, H3_OK)
    assert not t['fires'] and t['failing_parts'] == ['T2'] and t['parts']['T2']['detail'][0]['failed'] == ['guard|sex0']
    # T2 boundary: margin exactly 2 x own SE (inclusive) still fires
    se = 0.001953125
    x = exact_margin_value(0.005, 2 * se)
    t = R.trigger(evals({1: {'guards': {'sex0': x}, 'se': {'guard|sex0': se}}}), DEPLOY_OK, H3_OK)
    assert t['fires'], t['failing_parts']
    # T3: deployment not bitwise on version 0, or a refusal check failing, or the record missing
    for dep in ({0: {'bitwise': False, 'refusals_all_pass': True}, 1: DEPLOY_OK[1]},
                {0: DEPLOY_OK[0], 1: {'bitwise': True, 'refusals_all_pass': False}}, {0: DEPLOY_OK[0]}, {}):
        t = R.trigger(evals(), dep, H3_OK)
        assert not t['fires'] and t['failing_parts'] == ['T3']
    # T0: a version missing; H3 income not verified; a non-finite value
    t = R.trigger({0: util(0)}, DEPLOY_OK, H3_OK)
    assert not t['fires'] and 'T0' in t['failing_parts']
    t = R.trigger(evals(), DEPLOY_OK, {0: True, 1: False})
    assert not t['fires'] and 'T0' in t['failing_parts']
    t = R.trigger(evals({0: {'occ': {('Ucal', 'nll_excess'): math.nan}}}), DEPLOY_OK, H3_OK)
    assert not t['fires'] and 'T0' in t['failing_parts']
    # income plays no role: income accuracy below the reserve (ratio < 2) and even over the point cap does not block the trigger
    t = R.trigger(evals({1: {'inc': {('U0', 'accuracy_loss'): 0.0105, ('Ucal', 'accuracy_loss'): 0.0105}}}), DEPLOY_OK, H3_OK)
    assert t['fires'] and t['income_reported_separately'][1]['income_reserve_met'] is False


# ---------------------------------------------------------------- strength selection
def table(costs, se=0.0):
    """costs: {s: {stat: estimate}} or a function s -> estimate for every statistic."""
    out = {}
    for s in R.STRENGTHS:
        if s == 0.0:
            continue
        row = costs(s) if callable(costs) else costs[s]
        out[s] = {j: {'estimate': (row[j] if isinstance(row, dict) else row), 'se': se} for j in R.STATS}
    return out


def t_select_all_pass_identity_and_boundary():
    r = R.select_strength(table(lambda s: 1e-5 * s), 0)
    assert r['selected'] == 1.0 and not r['identity'] and r['first_failing_strength'] is None
    r = R.select_strength(table(lambda s: 1.0), 0)
    assert r['selected'] == 0.0 and r['identity'] and 'IDENTITY_SELECTED' in r['flags'] and r['first_failing_strength'] == 0.01
    al = R.allowances()[0]['men_brier']
    exact = {s: {j: (al if j == 'men_brier' else 0.0) for j in R.STATS} for s in R.STRENGTHS if s > 0}
    r = R.select_strength(table(exact), 0)
    assert r['selected'] == 1.0                                       # upper == allowance passes (inclusive)
    over = {s: {j: (math.nextafter(al, math.inf) if (j == 'men_brier' and s >= 0.3) else 0.0) for j in R.STATS}
            for s in R.STRENGTHS if s > 0}
    r = R.select_strength(table(over), 0)
    assert r['selected'] == 0.2 and r['binding_at_first_failure'] == ['men_brier']
    # z = 2 enters: estimate 0.5 x allowance with SE 0.24 x allowance passes (upper 0.98 x); SE 0.26 x fails (upper 1.02 x)
    t = {s: {j: {'estimate': 0.5 * R.allowances()[0][j], 'se': 0.24 * R.allowances()[0][j]} for j in R.STATS}
         for s in R.STRENGTHS if s > 0}
    assert R.select_strength(t, 0)['selected'] == 1.0
    t2 = {s: {j: {'estimate': 0.5 * R.allowances()[0][j], 'se': 0.26 * R.allowances()[0][j]} for j in R.STATS}
          for s in R.STRENGTHS if s > 0}
    assert R.select_strength(t2, 0)['selected'] == 0.0


def t_select_non_monotone_and_version_allowance():
    # pass at 0.01, 0.03; fail at 0.1; pass again at 0.2..1.0 -> 0.03 selected, PASS_AFTER_FAIL recorded
    def f(s):
        return 1.0 if s == 0.1 else -1e-4
    r = R.select_strength(table(f), 1)
    assert r['selected'] == 0.03 and 'PASS_AFTER_FAIL' in r['flags'] and r['pass_after_fail_strengths'] == [0.2, 0.3, 0.5, 0.7, 1.0]
    # negative estimates allowed (a correction that lowers a cost)
    assert R.select_strength(table(lambda s: -0.01), 1)['selected'] == 1.0
    # version-specific allowance: men's 0.00175 passes version 0 (0.00181) and fails version 1 (0.00170)
    c = {s: {j: (0.00175 if j == 'men_brier' else 0.0) for j in R.STATS} for s in R.STRENGTHS if s > 0}
    assert R.select_strength(table(c), 0)['selected'] == 1.0
    assert R.select_strength(table(c), 1)['selected'] == 0.0
    # non-finite estimate or SE fails at that strength
    n = table(lambda s: 0.0)
    n[0.3]['nll'] = {'estimate': math.nan, 'se': 0.0}
    assert R.select_strength(n, 0)['selected'] == 0.2
    n = table(lambda s: 0.0)
    n[0.5]['acc_loss'] = {'estimate': 0.0, 'se': math.inf}
    assert R.select_strength(n, 0)['selected'] == 0.3
    # string keys accepted; an identity row must be zero; missing / unknown strengths and missing statistics are errors
    t = {str(s): row for s, row in table(lambda s: 0.0).items()}
    t['0'] = {j: {'estimate': 0.0, 'se': 0.0} for j in R.STATS}
    assert R.select_strength(t, 0)['selected'] == 1.0
    for bad in ({**table(lambda s: 0.0), 0.0: {j: {'estimate': 1e-3, 'se': 0.0} for j in R.STATS}},
                {s: row for s, row in table(lambda s: 0.0).items() if s != 0.7},
                {**table(lambda s: 0.0), 0.25: {j: {'estimate': 0.0, 'se': 0.0} for j in R.STATS}},
                {s: {j: row[j] for j in R.STATS if j != 'brier'} for s, row in table(lambda s: 0.0).items()}):
        try:
            R.select_strength(bad, 0)
        except ValueError:
            continue
        raise AssertionError('malformed cost table accepted')
    try:
        R.select_strength(table(lambda s: 0.0), 2)
    except ValueError:
        pass
    else:
        raise AssertionError('unknown version accepted')


# ---------------------------------------------------------------- contrasts and labels
def t_summarize_orderings():
    s = R.summarize_orderings({'O0': 0.004, 'O1': 0.005, 'O2': 0.006, 'O3': 0.005}, {o: 0.001 for o in R.ORDERINGS})
    sd = math.sqrt(sum((x - 0.005) ** 2 for x in (0.004, 0.005, 0.006, 0.005)) / 3)
    assert abs(s['se_comb'] - math.sqrt(0.001 ** 2 + sd ** 2 / 4)) < 1e-15 and s['label'] == 'RESOLVED_POSITIVE'
    s = R.summarize_orderings({'O0': 0.004, 'O1': -0.0001, 'O2': 0.006, 'O3': 0.005}, {o: 0.0001 for o in R.ORDERINGS})
    assert s['label'] == 'UNRESOLVED' and s['sign_reversal'] and s['sign_reversals'] == 1   # mean - 2 se > 0 but a reversal
    s = R.summarize_orderings({o: -0.003 for o in R.ORDERINGS}, {o: 0.001 for o in R.ORDERINGS})
    assert s['label'] == 'RESOLVED_NEGATIVE'
    s = R.summarize_orderings({o: 0.0015 for o in R.ORDERINGS}, {o: 0.001 for o in R.ORDERINGS})
    assert s['label'] == 'UNRESOLVED' and not s['sign_reversal']
    s = R.summarize_orderings({'S100': 0.003}, {'S100': 0.001})
    assert s['label'] == 'RESOLVED_POSITIVE' and s['se_comb'] == 0.001 and s['sd_ord'] == 0.0
    assert R.summarize_orderings({'O0': 0.1}, {'O1': 0.1})['label'] == 'INCOMPLETE'
    assert R.summarize_orderings({'O0': math.nan}, {'O0': 0.1})['label'] == 'INCOMPLETE'
    assert R.across_versions({0: 'RESOLVED_POSITIVE', 1: 'RESOLVED_POSITIVE'}) == 'POSITIVE_BOTH_VERSIONS'
    assert R.across_versions({0: 'RESOLVED_POSITIVE', 1: 'UNRESOLVED'}) == 'VERSION_DEPENDENT'
    assert R.across_versions({0: 'RESOLVED_POSITIVE', 1: 'RESOLVED_NEGATIVE'}) == 'VERSION_DEPENDENT'
    assert R.across_versions({0: 'UNRESOLVED', 1: 'UNRESOLVED'}) == 'NOT_RESOLVED'
    assert R.across_versions({0: 'RESOLVED_POSITIVE'}) == 'INCOMPLETE'


def _block(change_by_version, se=0.0008):
    return {v: {'values': {o: x for o, x in zip(R.ORDERINGS, change_by_version[v])}, 'se': {o: se for o in R.ORDERINGS}}
            for v in change_by_version}


def t_size_summary_labels():
    gain = {0: [-0.0040, -0.0035, -0.0045, -0.0038], 1: [-0.0030, -0.0034, -0.0029, -0.0036]}   # loss falls with size
    ch = {c: {m: _block(gain) for m in R.METRICS} for c in R.CONTRASTS}
    s = R.size_summary(ch)
    assert s['size_effect'] == 'SIZE_GAIN_RESOLVED' and s['contrasts']['50->100']['men_brier']['per_version'][0]['mean'] > 0
    ch['50->100']['men_brier'] = _block({0: gain[0], 1: [-0.0030, 0.0005, -0.0029, -0.0036]})
    s = R.size_summary(ch)
    assert s['size_effect'] == 'SIZE_EFFECT_UNRESOLVED' and s['primary_endpoint']['across'] == 'VERSION_DEPENDENT'
    ch['50->100']['men_brier'] = _block({0: [0.004] * 4, 1: [0.0041] * 4})
    assert R.size_summary(ch)['size_effect'] == 'SIZE_LOSS_RESOLVED'
    del ch['50->100']['men_brier']
    assert R.size_summary(ch)['size_effect'] == 'INCOMPLETE'
    ch['50->100']['men_brier'] = {0: _block(gain)[0]}
    assert R.size_summary(ch)['size_effect'] == 'INCOMPLETE'


def _status(w100=(True, True), s50=True, s70=True):
    def grid(x):
        return {o: {v: (x(o, v) if callable(x) else x) for v in R.VERSIONS} for o in R.ORDERINGS}
    return {'W100': dict(zip(R.VERSIONS, w100)), 'S50': grid(s50), 'S70': grid(s70)}


def t_margin_pattern_and_interpretation():
    assert R.margin_pattern(_status(s50=False, s70=False))['pattern'] == 'FULL_ONLY'
    m = R.margin_pattern(_status(s50=lambda o, v: o != 'O2', s70=True))
    assert m['pattern'] == 'FULL_ONLY' and m['n50_met_of_8'] == 6 and 'FULL_ONLY_70_RETAINS' in m['flags']
    assert R.margin_pattern(_status())['pattern'] == 'ALL_SIZES'
    assert R.margin_pattern(_status(w100=(True, False)))['pattern'] == 'FULL_NOT_REPRODUCED'
    m = R.margin_pattern(_status(s50=True, s70=lambda o, v: o != 'O1'))
    assert m['pattern'] == 'FULL_ONLY' and 'NON_NESTED_PATTERN_50_RETAINS_70_LOSES' in m['flags']
    bad = _status(); bad['S70']['O3'][1] = None
    assert R.margin_pattern(bad)['pattern'] == 'INCOMPLETE'
    bad = _status(); del bad['S50']['O0']
    assert R.margin_pattern(bad)['pattern'] == 'INCOMPLETE'
    E = R.size_explanation
    assert E('FULL_ONLY', 'SIZE_GAIN_RESOLVED') == 'SUPPORTED_ON_THIS_POOL'
    assert E('FULL_ONLY', 'SIZE_EFFECT_UNRESOLVED') == 'UNRESOLVED'
    assert E('ALL_SIZES', 'SIZE_GAIN_RESOLVED') == 'NOT_SUPPORTED'
    assert E('FULL_ONLY', 'SIZE_LOSS_RESOLVED') == 'NOT_SUPPORTED'
    assert E('FULL_NOT_REPRODUCED', 'SIZE_GAIN_RESOLVED') == 'NOT_APPLICABLE_FULL_NOT_REPRODUCED'
    assert E('INCOMPLETE', 'SIZE_GAIN_RESOLVED') == 'INCOMPLETE'
    i = R.interpretation('FULL_ONLY', 'SIZE_EFFECT_UNRESOLVED', {'men_brier': 'CONTROL_EXPLAINS_GAIN'}, {100: True, 50: False})
    rules = [x['rule'] for x in i['readings']]
    assert rules == ['IR1', 'IR3', 'IR4'] and i['size_explanation'] == 'UNRESOLVED'
    i = R.interpretation('FULL_NOT_REPRODUCED', 'SIZE_GAIN_RESOLVED')
    assert [x['rule'] for x in i['readings']] == ['IR2']
    i = R.interpretation('ALL_SIZES', 'SIZE_GAIN_RESOLVED')
    assert [x['rule'] for x in i['readings']] == ['ALL_SIZES'] and i['size_explanation'] == 'NOT_SUPPORTED'
    i = R.interpretation('FULL_ONLY', 'SIZE_GAIN_RESOLVED')
    assert [x['rule'] for x in i['readings']] == ['IR1'] and i['size_explanation'] == 'SUPPORTED_ON_THIS_POOL'


def t_control_labels():
    better = {0: [-0.002] * 4, 1: [-0.0021] * 4}                      # loss(A1) - loss(control) < 0: recipe better
    lv = R.control_level({m: _block(better, se=0.0005) for m in R.METRICS})
    assert all(lv[m]['label'] == 'RECIPE_BETTER_RESOLVED' for m in R.METRICS)
    lv = R.control_level({m: _block({0: [0.002] * 4, 1: [0.0021] * 4}, se=0.0005) for m in R.METRICS})
    assert lv['nll']['label'] == 'RECIPE_WORSE_RESOLVED'
    one = {v: {'values': {'S100': -0.002}, 'se': {'S100': 0.0005}} for v in R.VERSIONS}
    assert R.control_level({m: one for m in R.METRICS})['brier']['label'] == 'RECIPE_BETTER_RESOLVED'
    d = R.control_did({m: _block({0: [0.0001, -0.0002, 0.0001, 0.0], 1: [0.0, 0.0001, -0.0001, 0.0]}) for m in R.METRICS})
    assert d['men_brier']['across'] == 'NOT_RESOLVED'
    CE = R.control_explains
    assert CE('POSITIVE_BOTH_VERSIONS', 'POSITIVE_BOTH_VERSIONS', 'NOT_RESOLVED') == 'CONTROL_EXPLAINS_GAIN'
    assert CE('POSITIVE_BOTH_VERSIONS', 'POSITIVE_BOTH_VERSIONS', 'POSITIVE_BOTH_VERSIONS') == 'RECIPE_SPECIFIC_SIZE_GAIN'
    assert CE('POSITIVE_BOTH_VERSIONS', 'NOT_RESOLVED', 'NOT_RESOLVED') == 'RECIPE_GAIN_CONTROL_UNRESOLVED'
    assert CE('VERSION_DEPENDENT', 'POSITIVE_BOTH_VERSIONS', 'NOT_RESOLVED') == 'NO_RECIPE_SIZE_GAIN'
    assert CE('POSITIVE_BOTH_VERSIONS', 'INCOMPLETE', 'NOT_RESOLVED') == 'INCOMPLETE'
    rs = {('O0', 0): True, ('O1', 0): False}
    assert R.control_also_meets_reserve(rs, {('O0', 0): True, ('O1', 0): False})
    assert not R.control_also_meets_reserve(rs, {('O0', 0): False, ('O1', 0): True})
    assert not R.control_also_meets_reserve({('O0', 0): False}, {('O0', 0): True})


def t_reproduction_and_encoding():
    import numpy as np
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(5), size=50)
    rec = R.reproduction_record(p, p.copy(), 412, 412)
    assert rec['sha_equal'] and rec['max_abs_diff'] == 0.0
    assert R.reproduction({0: rec, 1: [rec, rec]})['label'] == 'REPRODUCED_BITWISE'
    q = p.copy(); q[0, 0] = math.nextafter(q[0, 0], 1.0)
    rec_t = R.reproduction_record(q, p, 412, 412)
    assert not rec_t['sha_equal'] and rec_t['max_abs_diff'] <= 1e-12 and rec_t['decisions_equal']
    assert R.reproduction({0: rec, 1: rec_t})['label'] == 'REPRODUCED_WITHIN_TOLERANCE'
    assert R.reproduction({0: rec, 1: R.reproduction_record(p, p, 412, 413)})['label'] == 'NOT_REPRODUCED'
    q2 = p.copy(); q2[3] = q2[3][::-1]
    assert R.reproduction({0: R.reproduction_record(q2, p, 1, 1), 1: rec})['label'] == 'NOT_REPRODUCED'
    assert R.reproduction({0: rec})['label'] == 'INCOMPLETE'
    assert R.margin_reproduced({'status': 'OCC_RESERVE_MET_BOTH'}) == 'MARGIN_REPRODUCED'
    assert R.margin_reproduced({'status': 'OCC_RESERVE_ONE_VERSION'}) == 'MARGIN_NOT_REPRODUCED'
    E = {m: {'values': {o: 0.002 for o in R.ORDERINGS}, 'se': {o: 0.0004 for o in R.ORDERINGS}} for m in R.METRICS}
    E['nll'] = {'values': {'O0': 0.001, 'O1': -0.0005, 'O2': 0.0007, 'O3': 0.0002}, 'se': {o: 0.0004 for o in R.ORDERINGS}}
    del E['acc_loss']
    s = R.encoding_summary(E)
    assert s['brier']['encoding_label'] == 'ENCODING_EFFECT_RESOLVED' and s['nll']['encoding_label'] == 'ENCODING_EFFECT_UNRESOLVED'
    assert s['acc_loss']['encoding_label'] == 'INCOMPLETE'
    assert R.encoding_share(0.001, 0.004) == 0.25 and R.encoding_share(-0.001, 0.004) is None


# ---------------------------------------------------------------- readers
def cells_from(fn):
    return {c: fn(c) for c in R.CELLS}


def t_stability_labels_and_source_equivalence():
    vals = dict(zip(R.CELLS, [0.030, 0.034, 0.028, 0.031, 0.027, 0.033, 0.029, 0.032]))
    ses = {c: 0.004 for c in R.CELLS}
    s = R.stability_label(vals, ses)
    assert s['label'] == 'STABLE_POSITIVE' and s['n_eff'] == 8 and s['reader_seed_qualifier'] == 'READER_SEEDS_EFFECTIVE_ALL_PAIRS'
    src = OR.benefit_summary({'cells': vals, 'cell_se': ses}, 2.0)
    assert src['label'] == 'STABLE_POSITIVE' and abs(src['sd_total'] - s['sd_total']) < 1e-15   # identical when n_eff = 8
    neg = dict(vals); neg['S1R2k1'] = -0.001
    assert R.stability_label(neg, ses)['label'] == 'SIGN_CHANGES'
    wide = dict(zip(R.CELLS, [0.001, 0.060, 0.002, 0.050, 0.001, 0.070, 0.003, 0.040]))
    assert R.stability_label(wide, ses)['label'] == 'POSITIVE_NOT_STABLE'
    low = {c: -0.002 for c in R.CELLS}; low['S0R1k0'] = 0.003
    assert R.stability_label(low, ses)['label'] == 'NOT_POSITIVE'
    miss = dict(vals); del miss['S0R2k1']
    assert R.stability_label(miss, ses)['label'] == 'INCOMPLETE'
    assert R.stability_label(vals, {**ses, 'S0R1k0': -1.0})['label'] == 'INCOMPLETE'


def t_stability_duplicated_cells():
    # every winner seed-deterministic: k0 == k1 in all four (v, R) pairs -> n_eff = 4, df 2, SS over the 4 distinct cells
    rep = {(0, 1): 0.020, (0, 2): 0.030, (1, 1): 0.022, (1, 2): 0.034}
    vals = {f'S{v}R{r}k{k}': rep[(v, r)] for v in R.VERSIONS for r in (1, 2) for k in (0, 1)}
    ses = {c: 0.003 for c in R.CELLS}
    s = R.stability_label(vals, ses)
    d0, d1 = 0.025, 0.028
    ss = (0.020 - d0) ** 2 + (0.030 - d0) ** 2 + (0.022 - d1) ** 2 + (0.034 - d1) ** 2
    assert s['n_eff'] == 4 and s['spread_df'] == 2 and s['reader_seed_qualifier'] == 'READER_SEED_REPLICATION_ABSENT'
    assert abs(s['sd_reader_split'] ** 2 - ss / 2) < 1e-15
    src = OR.benefit_summary({'cells': vals, 'cell_se': ses}, 2.0)
    assert src['sd_reader_split'] < s['sd_reader_split']               # the source formula would shrink v_rep by 2/3 here
    assert abs(src['sd_reader_split'] ** 2 - (2 * ss) / 6) < 1e-15
    assert all(v is False for v in s['k_differs'].values())
    # partly effective: one pair differs -> n_eff = 5, qualifier PARTLY
    vals2 = dict(vals); vals2['S1R2k1'] = 0.036
    s2 = R.stability_label(vals2, ses)
    assert s2['n_eff'] == 5 and s2['reader_seed_qualifier'] == 'READER_SEEDS_PARTLY_EFFECTIVE' and s2['k_differs']['S1R2']
    assert R.seed_distinct_pairs(vals2) == 1 and R.seed_distinct_pairs({'S0R1k0': 1.0}) is None


def t_target_and_recipient_gate():
    vals = dict(zip(R.CELLS, [0.040, 0.042, 0.039, 0.041, 0.038, 0.043, 0.040, 0.041]))
    ses = {c: 0.003 for c in R.CELLS}
    assert R.target_label(vals, ses)['label'] == 'DEV_GRID_TARGET_STABLE'
    mixed = dict(zip(R.CELLS, [0.040, 0.010, 0.039, 0.041, 0.018, 0.043, 0.040, 0.041]))
    assert R.target_label(mixed, ses)['label'] == 'DEV_GRID_TARGET_POINT_ONLY'
    assert R.target_label({c: 0.015 for c in R.CELLS}, ses)['label'] == 'DEV_GRID_TARGET_NOT_MET'
    g = R.recipient_gate({**{c: 0.0 for c in R.CELLS}, 'S0R1k0': 0.010})
    assert g['pass']                                                   # <= 0.010 inclusive
    g = R.recipient_gate({**{c: 0.0 for c in R.CELLS}, 'S1R2k1': 0.0101})
    assert not g['pass'] and g['cells_above'] == ['S1R2k1']


def t_reader_seed_check():
    seeds = {'S0k0': 2026201800, 'S0k1': 2026201801, 'S1k0': 2026201810, 'S1k1': 2026201811}
    checks = [{'recipe': r, 'view': 'pair', 'wire': 'W100', 'v': 0, 'R': 'R1', 'equal_k0_k1': False} for r in R.SEED_DEPENDENT]
    checks += [{'recipe': r, 'view': 'pair', 'wire': 'W100', 'v': 0, 'R': 'R1', 'equal_k0_k1': True} for r in R.SEED_INVARIANT]
    c = R.reader_seed_check(seeds, checks)
    assert c['label'] == 'READER_SEEDS_EFFECTIVE' and not c['flags']
    bad = copy.deepcopy(checks); bad[0]['equal_k0_k1'] = True
    assert R.reader_seed_check(seeds, bad)['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    assert R.reader_seed_check(seeds, [x for x in checks if x['recipe'] != 'MLP64'])['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    old = dict(seeds); old['S1k1'] = R.MODEL_SEED_BASE + 300 + 10 * 1 + 1
    assert R.reader_seed_check(old, checks)['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    dup = dict(seeds); dup['S1k1'] = dup['S1k0']
    assert R.reader_seed_check(dup, checks)['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    nd = copy.deepcopy(checks); nd[-1]['equal_k0_k1'] = False
    c = R.reader_seed_check(seeds, nd)
    assert c['label'] == 'READER_SEEDS_EFFECTIVE' and c['flags'] == ['NONDETERMINISTIC_RECIPE']


# ---------------------------------------------------------------- equal reserve and claims
def t_equal_reserve_table():
    pts = [{'id': 'LOC', 'kind': 'candidate', 'min_ratio': {0: 2.4, 1: 2.1}, 'pair_auc': {0: 0.60, 1: 0.61}},
           {'id': 'W', 'kind': 'control', 'min_ratio': {0: 4.8, 1: 4.3}, 'pair_auc': {0: 0.63, 1: 0.63}},
           {'id': 'CONT', 'kind': 'control', 'min_ratio': {0: 4.0, 1: 3.0}, 'pair_auc': {0: 0.64, 1: 0.64}},
           {'id': 'H_LJ', 'kind': 'reference', 'min_ratio': {0: 0.3, 1: 0.6}, 'pair_auc': {0: 0.58, 1: 0.58}}]
    t = R.equal_reserve_table(pts)
    assert t['frontier'] == ['LOC', 'W'] and t['references_failing_reserve'] == ['H_LJ']
    assert [p['qualified'] for p in t['points']] == [True, True, True, False]


def arm(selected=(0.3, 0.2), point=(True, True), reserve=(True, True), cont='STABLE_POSITIVE', w='STABLE_POSITIVE', loc=None,
        comp=(False, False, False), target='DEV_GRID_TARGET_NOT_MET'):
    a = {'selected': dict(zip(R.VERSIONS, selected)),
         'utility': {v: {'point_pass': point[v], 'occ_reserve_met': reserve[v]} for v in R.VERSIONS},
         'vs_CONT': cont, 'vs_W': w, 'competitive': dict(zip(('point_all', 'recipient_gate_all', 'stable_vs_all_required'), comp)),
         'target': target}
    if loc is not None:
        a['vs_LOC'] = loc
    return a


def t_claim_labels_and_closure():
    c = R.claim_labels({'trigger_fired': False, 'size_explanation': 'UNRESOLVED'})
    assert c['outcome'] == 'NOT_RUN_TRIGGER_NOT_FIRED' and c['C1']['label'] == 'NOT_ESTABLISHED' and c['C7'] == 'NOT_CLAIMABLE'
    ev = {'trigger_fired': True, 'size_explanation': 'SUPPORTED_ON_THIS_POOL',
          'arms': {'LOC': arm(), 'PAIR': arm(loc='SIGN_CHANGES')}}
    c = R.claim_labels(ev)
    assert c['outcome'] == 'PRIMARY_ARM_QUALIFIES' and c['C1']['label'] == 'MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL'
    assert c['arms']['LOC']['C2'] == 'UTILITY_PRESERVED_BOTH_VERSIONS' and c['arms']['LOC']['C3']['label'] == 'RECOVERY_BELOW_CONTINUATION'
    assert c['arms']['PAIR']['C4']['label'] == 'NOT_MET' and not c['arms']['PAIR']['qualifies']
    assert c['C8'] == 'NOT_CLAIMED' and c['arms']['LOC']['closure_uses'] == 'C2'
    # LOC selects identity on version 1 (no 8-cell stability possible) and PAIR qualifies -> secondary arm only
    ev['arms'] = {'LOC': arm(selected=(0.2, 0.0), cont='SIGN_CHANGES', w='SIGN_CHANGES'), 'PAIR': arm(loc='STABLE_POSITIVE')}
    c = R.claim_labels(ev)
    assert c['arms']['LOC']['C2'] == 'UTILITY_PRESERVED_ONE_VERSION' and c['outcome'] == 'SECONDARY_ARM_ONLY'
    # stable recovery reduction but the reserve fails on a nonzero selection: C2 fails, C2p (point only) passes, closure
    ev['arms'] = {'LOC': arm(reserve=(True, False)), 'PAIR': arm(reserve=(False, True), loc='STABLE_POSITIVE')}
    c = R.claim_labels(ev)
    assert c['arms']['LOC']['C2'] == 'UTILITY_NOT_PRESERVED' and c['arms']['LOC']['C2p'] == 'UTILITY_POINT_PRESERVED_BOTH_VERSIONS'
    assert c['arms']['LOC']['C3']['label'] == 'MET_WITHOUT_UTILITY' and c['outcome'] == 'CORRECTION_RECIPE_CLOSED'
    assert c['arms']['LOC']['selected_release_status'][1] == {'selected': 0.2, 'point_pass_12_plus_2': True, 'occ_reserve_met': False}
    # point failure on a nonzero selection: both C2 and C2p fail
    ev['arms'] = {'LOC': arm(point=(True, False)), 'PAIR': arm(selected=(0.0, 0.0), cont='NOT_POSITIVE', w='NOT_POSITIVE',
                                                                  loc='NOT_POSITIVE')}
    c = R.claim_labels(ev)
    assert c['arms']['LOC']['C2p'] == 'UTILITY_POINT_NOT_PRESERVED' and c['arms']['PAIR']['C2'] == 'NO_NONZERO_SELECTION'
    assert c['outcome'] == 'CORRECTION_RECIPE_CLOSED'
    # competitiveness and the 0.020 target are separate from the closure outcome
    ev['arms'] = {'LOC': arm(comp=(True, True, True), target='DEV_GRID_TARGET_POINT_ONLY'), 'PAIR': arm(loc='NOT_POSITIVE')}
    c = R.claim_labels(ev)
    assert c['arms']['LOC']['C5']['label'] == 'COMPETITIVE_WITH_ALL_REQUIRED'
    assert c['arms']['LOC']['C6']['label'] == 'DEV_GRID_TARGET_POINT_ONLY' and c['arms']['LOC']['C6']['claimable']


TESTS = [v for k, v in sorted(globals().items()) if k.startswith('t_') and callable(v)]


def main():
    failed = []
    for t in TESTS:
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception as ex:                                            # report every failure, then exit non-zero
            failed.append(t.__name__)
            print(f'FAIL {t.__name__}: {type(ex).__name__}: {ex}')
    print(f'{len(TESTS) - len(failed)} of {len(TESTS)} PASS; RULES.json sha256 {R.load_rules()["_sha256"]}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

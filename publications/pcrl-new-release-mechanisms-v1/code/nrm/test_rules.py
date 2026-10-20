"""Invented-data tests of nrm.rules and known-law checks of the aggregate screen (no real data, no label).
Run: python3 -m nrm.test_rules        (writes results/pcrl_new_release_mechanisms_v1/roles/statistical_reviewer/SCREEN_CHECKS.json)

Covers: the rule file and its refusal of altered constants; point eligibility (inclusive limits, worst draw); the reserve applied
identically (MC SE in quadrature); randomized cells; the aggregate estimator (hand-computed components); EXACT ALIASES (a
candidate identical to its comparator never passes; identical views pass non-worsening; aliases counted once); near aliases
(SE floor); one noisy cell (no veto; divergence flagged); recipient labels per version; the 0.020 screen; reader-stability
summaries; grid validity (A-SC1, prospective); the frontier bracketing rule; the reserve frontier; claims G1-G7; selection
(tiers, near ties, deterministic tie-breaks, control may win, aliases, output mismatch). Known-law simulation: correlated cells
(version, split, interaction, reader-seed and shared-household components), true means 0 / 0.005 / 0.010 / 0.020; empirical
pass rates of the new and the old rules (not a universal error guarantee)."""
from __future__ import annotations
import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ohn import rules as OR
from tsz import rules as TR
from nrm import rules as R

OUT = R.RULES_PATH.parent / 'SCREEN_CHECKS.json'
CAPS = R.CAPS
OCC = {('U0', 'accuracy_loss'): -0.0098, ('U0', 'nll_excess'): -0.0128, ('U0', 'brier_excess'): -0.0055,
       ('Ucal', 'accuracy_loss'): -0.0098, ('Ucal', 'nll_excess'): -0.0128, ('Ucal', 'brier_excess'): -0.0055}
INC = {('U0', 'accuracy_loss'): 0.0017, ('U0', 'nll_excess'): -0.0050, ('U0', 'brier_excess'): -0.0030,
       ('Ucal', 'accuracy_loss'): 0.0017, ('Ucal', 'nll_excess'): -0.0050, ('Ucal', 'brier_excess'): -0.0030}
GUARDS = {'sex0': -0.0023, 'sex1': -0.0087}


def util(v, occ=None, inc=None, guards=None, se=0.001, se_mc=None, draws=None):
    """Invented evaluator utility dict of one version (12 clauses + 2 guards)."""
    occ, inc, guards = {**OCC, **(occ or {})}, {**INC, **(inc or {})}, {**GUARDS, **(guards or {})}
    clauses = []
    for task, vals in (('income', inc), ('occupation', occ)):
        for (ref, m), x in vals.items():
            c = {'task': task, 'reference': ref, 'metric': m, 'value': x, 'cap': CAPS[m], 'se': se, 'seed': v}
            if se_mc is not None:
                c['se_mc'] = se_mc
            if draws is not None:
                c['draw_values'] = [x + d for d in draws]
            clauses.append(c)
    gl = []
    for g, x in guards.items():
        c = {'group': g, 'value': x, 'cap': R.GUARD_CAP, 'se': se, 'seed': v}
        if se_mc is not None:
            c['se_mc'] = se_mc
        if draws is not None:
            c['draw_values'] = [x + d for d in draws]
        gl.append(c)
    return {'households': 5306, 'clauses': clauses, 'guards': gl}


def evals(**kw):
    return {v: util(v, **kw) for v in R.VERSIONS}


def entry(vals, se=0.004, mc=None):
    """Grid entry from {cell: value} (registered cells by default)."""
    e = {'cells': dict(vals), 'cell_se': {c: (se[c] if isinstance(se, dict) else se) for c in vals}}
    if mc is not None:
        e['cell_se_mc'] = {c: mc for c in vals}
    return e


def reg(fn):
    return {c: fn(c) for c in R.REGISTERED}


def grid_entry(pair, inc=0.0, occ=0.0, se=0.004):
    return {'pair': entry(reg(pair) if callable(pair) else reg(lambda c: pair), se),
            'income': entry(reg(inc) if callable(inc) else reg(lambda c: inc), se),
            'occupation': entry(reg(occ) if callable(occ) else reg(lambda c: occ), se)}


ALIAS = {'pair': entry(reg(lambda c: 0.0), 0.0), 'income': entry(reg(lambda c: 0.0), 0.0), 'occupation': entry(reg(lambda c: 0.0), 0.0)}


# ---------------------------------------------------------------- rule file
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'NRM1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(Path(R.RULES_PATH).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='nrm_rules_test_'))
    for path_, val in ((('aggregate_screen', 'z'), 1.5), (('aggregate_screen', 'se_floor_auc'), 0.0),
                       (('original_limits', 'recipient_allowance_auc'), 0.02), (('original_limits', 'material_target_auc'), 0.01),
                       (('selection', 'near_tie_width'), 0.01), (('reserve', 'multiplier_k'), 1.0),
                       (('original_limits', 'target_references'), ['H_LJ'])):
        bad = copy.deepcopy(raw)
        bad[path_[0]][path_[1]] = val
        p = tmp / f'bad_{path_[1]}.json'
        p.write_text(json.dumps(bad))
        try:
            R.load_rules(p)
        except ValueError:
            continue
        raise AssertionError(f'altered {path_} was accepted')
    pub = r['references']['published_labels']
    assert set(pub) == set(R.BENCHMARK) and all(pub[x]['point'] == 'POINT_ELIGIBLE' for x in R.BENCHMARK)
    assert [x for x in R.BENCHMARK if pub[x]['reserve'] == 'RESERVE_MET'] == ['W', 'TR100', 'TT100']


# ---------------------------------------------------------------- G1 core, reserve, randomized
def t_point_eligibility_and_worst_draw():
    assert R.point_eligibility(evals())['label'] == 'POINT_ELIGIBLE'
    e = evals()
    e[1] = util(1, guards={'sex0': 0.005})                          # inclusive cap
    assert R.point_eligibility(e)['label'] == 'POINT_ELIGIBLE'
    e[1] = util(1, guards={'sex0': math.nextafter(0.005, 1)})
    p = R.point_eligibility(e)
    assert p['label'] == 'POINT_INELIGIBLE' and p['failed'] == ['v1|guard|sex0']
    assert R.point_eligibility({0: util(0)})['label'] == 'INCOMPLETE'
    p = R.point_eligibility(evals(draws=[-0.001, 0.0, 0.0151]))      # one draw breaks every guard's / clause's cap
    assert p['label'] == 'POINT_ELIGIBLE' and p['worst_draw'][0]['every_draw_passes'] is False


def t_reserve_symmetric_with_mc():
    r0 = R.reserve_status(evals())
    assert r0['qualified'] and r0['status'] == OR.MET
    # occupation accuracy margin 0.0198 vs floor ~0.0035: ratio ~5.6; an MC SE of 0.012 pushes the own SE above the floor
    r1 = R.reserve_status(evals(se_mc=0.012))
    assert not r1['qualified'] and r1['without_mc']['status'] == OR.MET
    src = OR.reserve(evals())
    assert r0['ratio_vector'] == src['ratio_vector']                 # identical rule for every method


def t_mc_cell():
    c = R.mc_cell([0.010, 0.014, 0.012], [0.003, 0.004, 0.005])
    assert abs(c['value'] - 0.012) < 1e-15 and abs(c['se'] - math.sqrt((9 + 16 + 25) / 3) * 1e-3) < 1e-15
    assert abs(c['se_mc'] - 0.002 / math.sqrt(3)) < 1e-15
    assert R.mc_cell([0.01], [0.003])['flags'] == ['MC_SE_NOT_ESTIMABLE']
    try:
        R.mc_cell([0.01, math.nan], [0.003, 0.003])
    except ValueError:
        return
    raise AssertionError('non-finite draw accepted')


# ---------------------------------------------------------------- aggregate estimator and exact aliases
def t_aggregate_components_by_hand():
    vals = dict(zip(R.REGISTERED, [0.010, 0.014, 0.006, 0.008, 0.012, 0.016, 0.004, 0.010]))
    ses = dict(zip(R.REGISTERED, [0.003, 0.004, 0.003, 0.005, 0.004, 0.004, 0.002, 0.003]))
    a = R.aggregate(entry(vals, ses))
    mb = [0.012, 0.007, 0.014, 0.007]
    x = sum(mb) / 4
    assert abs(a['mean'] - x) < 1e-15
    assert abs(a['se_household'] - math.sqrt(sum(s * s for s in ses.values()) / 8)) < 1e-15
    assert abs(a['se_reader_split'] - math.sqrt(sum((m - x) ** 2 for m in mb) / 4)) < 1e-15
    assert abs(a['per_version'][0]['mean'] - 0.0095) < 1e-15
    assert abs(a['per_version'][1]['se_reader_split'] - abs(0.014 - 0.007) / 2) < 1e-15
    assert a['n_eff_distinct_cells'] == 8 and a['reader_seed_qualifier'] == 'READER_SEEDS_EFFECTIVE_ALL_BLOCKS'
    # seed-deterministic duplicates: counted once in n_eff, mean unchanged
    dup = dict(vals); dup['S0R1k1'] = dup['S0R1k0']; dup['S1R2k1'] = dup['S1R2k0']
    a2 = R.aggregate(entry(dup, ses))
    assert a2['n_eff_distinct_cells'] == 6 and a2['reader_seed_qualifier'] == 'READER_SEEDS_PARTLY_EFFECTIVE'
    # fraction cells and unknown names are ignored; a missing registered cell is incomplete
    extra = dict(vals); extra['S0R1k0@0.5'] = 9.0
    assert abs(R.aggregate(entry(extra, 0.003))['mean'] - x) < 1e-15
    miss = dict(vals); del miss['S1R1k0']
    assert R.benefit_screen(entry(miss, 0.003))['label'] == 'INCOMPLETE'


def t_exact_alias_never_passes():
    b = R.benefit_screen(ALIAS['pair'])
    assert b['label'] == 'ALIAS_ZERO' and not b['resolved']
    assert R.target_screen(ALIAS['pair'])['label'] == 'TARGET_AGG_NOT_MET'
    rc = R.recipient_screen(ALIAS['income'])
    assert rc['label'] == 'ALIAS_ZERO' and rc['gate_pass']             # an identical view does not worsen
    cmpr = R.comparison(ALIAS)
    assert not cmpr['benefit_positive'] and cmpr['recipient_gate_pass']
    # a candidate identical to its twin, alternative and every reference: no claim beyond G1, never selectable
    rec = {'evals': evals(), 'grid': {n: ALIAS for n in R.BENCHMARK + ('TWIN', 'ALT')}, 'matched': {'twin': 'TWIN', 'alternative': 'ALT'},
           'grid_valid': True}
    cl = R.claims(rec)
    assert cl['G1']['label'] == 'TASK_FEASIBLE' and cl['G2']['label'] == 'NOT_MET' and cl['G4']['label'] == 'NOT_SUPERIOR'
    assert cl['tier'] is None and cl['G5']['label'] == 'NOT_MET'
    # an alias with a tiny positive cell SE but zero differences still fails (mean 0 is never > 0)
    z = entry(reg(lambda c: 0.0), 0.004)
    assert R.benefit_screen(z)['label'] == 'AGG_NOT_POSITIVE'


def t_near_alias_floor():
    tiny = entry(reg(lambda c: 0.0004), 1e-5)                        # consistent but negligible difference, tiny SEs
    b = R.benefit_screen(tiny)
    assert b['aggregate']['floor_binds'] and b['label'] == 'AGG_POSITIVE_POINT_ONLY'
    b2 = R.benefit_screen(entry(reg(lambda c: 0.0025), 1e-5))         # above 2 x floor: resolves
    assert b2['label'] == 'AGG_POSITIVE'


def t_one_noisy_cell_no_veto():
    vals = reg(lambda c: 0.012)
    vals['S1R2k1'] = -0.002                                           # one noisy cell
    b = R.benefit_screen(entry(vals, 0.003))
    assert b['old_every_cell_rule']['label'] == 'SIGN_CHANGES' and b['label'] == 'AGG_POSITIVE'
    assert b['flags'] == ['EVERY_CELL_RULE_DIVERGES'] and b['aggregate']['sign_reversals']['cells_opposite_to_mean'] == 1
    r = reg(lambda c: 0.001)
    r['S0R2k0'] = 0.013                                               # one recipient cell above the allowance
    rc = R.recipient_screen(entry(r, 0.003))
    assert rc['old_every_cell_rule']['pass'] is False and rc['gate_pass'] and rc['cells_above_limit'] == ['S0R2k0']
    assert 'EVERY_CELL_RULE_DIVERGES' in rc['flags']


def t_recipient_labels_per_version():
    assert R.recipient_screen(entry(reg(lambda c: 0.001), 0.002))['label'] == 'RECIPIENT_AGG_RESOLVED'
    assert R.recipient_screen(entry(reg(lambda c: 0.008), 0.003))['label'] == 'RECIPIENT_AGG_EDGE'
    assert R.recipient_screen(entry(reg(lambda c: 0.010), 0.0))['label'] == 'RECIPIENT_AGG_EDGE'     # point inclusive
    v1 = reg(lambda c: 0.002 if c.startswith('S0') else 0.012)       # version 1 above the allowance: fails per version
    rc = R.recipient_screen(entry(v1, 0.003))
    assert rc['label'] == 'RECIPIENT_AGG_POINT_FAILED' and not rc['gate_pass'] and rc['aggregate']['mean'] < 0.010
    assert R.recipient_screen(entry(reg(lambda c: 0.030), 0.003))['label'] == 'RECIPIENT_AGG_RESOLVED_WORSE'


def t_target_screen():
    assert R.target_screen(entry(reg(lambda c: 0.040), 0.003))['label'] == 'TARGET_AGG_RESOLVED'
    t = R.target_screen(entry(reg(lambda c: 0.023), 0.004))
    assert t['label'] == 'TARGET_AGG_POINT_ONLY'
    v = reg(lambda c: 0.045 if c.startswith('S0') else 0.018)         # mean 0.0315 but version 1 below 0.020
    assert R.target_screen(entry(v, 0.001))['label'] == 'TARGET_AGG_POINT_ONLY'
    assert R.target_screen(entry(reg(lambda c: 0.015), 0.003))['label'] == 'TARGET_AGG_NOT_MET'


def t_robustness_cells():
    vals = reg(lambda c: 0.012)
    vals.update({f'S{v}{r}k{k}': 0.011 for v in R.VERSIONS for r in R.ROB_REPS for k in R.REG_K})
    e = entry(vals, 0.003)
    rb = R.robustness({'pair': e})
    assert rb['label'] == 'ROBUSTNESS_CONFIRMED' and rb['robustness_cells']['aggregate']['n_cells'] == 8
    assert rb['all_cells']['aggregate']['n_blocks'] == 8 and R.benefit_screen(e)['aggregate']['n_cells'] == 8   # decision: registered
    vals.update({f'S{v}R3k{k}': -0.01 for v in R.VERSIONS for k in R.REG_K})
    assert R.robustness({'pair': entry(vals, 0.003)})['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    assert R.robustness({'pair': entry(reg(lambda c: 0.012), 0.003)})['label'] == 'NOT_RUN'


# ---------------------------------------------------------------- reader stability, grid validity, aliases
def t_reader_stability_and_aliases():
    cells = {}
    for i, c in enumerate(R.REGISTERED):
        cells[c] = {'A': {v: {'winner': ('MLP64' if i % 2 else 'OUT_LR'), 'near_tie': i == 3, 'score_auc': 0.60 + 0.001 * i}
                          for v in R.VIEWS},
                    'B': {v: {'winner': 'OUT_LR', 'near_tie': False, 'score_auc': 0.62} for v in R.VIEWS}}
    s = R.reader_stability(cells, 'A', 'B')
    p = s['pair']
    assert p['winner_changes']['A']['n_distinct'] == 2 and p['near_ties']['A'] == 1 and p['deterministic_cells'] == 4
    assert p['reader_seed_qualifier'] == 'READER_SEEDS_EFFECTIVE_ALL_BLOCKS' and p['sign_changes'] == 0
    w = {'A': {0: {'income': 'h', 'occupation': 'o1'}, 1: {'income': 'h', 'occupation': 'o2'}},
         'B': {0: {'income': 'h', 'occupation': 'x1'}, 1: {'income': 'h', 'occupation': 'x2'}},
         'C': {0: {'income': 'h', 'occupation': 'x1'}, 1: {'income': 'h', 'occupation': 'x2'}}}
    s2 = R.reader_stability(cells, 'A', 'B', wires=w)
    assert s2['income']['label'] == 'EXACT_ALIAS_COUNTED_ONCE' and s2['pair']['label'] == 'SUMMARY'
    g = R.alias_groups(w)
    assert g['income'] == [['A', 'B', 'C']] and g['occupation'] == [['B', 'C']] and g['pair'] == [['B', 'C']]
    assert R.distinct_comparators(['B', 'C', 'A'], g, 'occupation') == ['B=C', 'A']


def t_grid_validity():
    seeds = {'S0k0': R.MODEL_SEED_BASE + 600, 'S0k1': R.MODEL_SEED_BASE + 601, 'S1k0': R.MODEL_SEED_BASE + 610,
             'S1k1': R.MODEL_SEED_BASE + 611}
    checks = [{'recipe': r, 'view': vw, 'wire': w, 'v': 0, 'R': 'R1', 'equal_k0_k1': False}
              for r in R.SEED_DEPENDENT for vw in R.VIEWS for w in ('CAND', 'TWIN', 'REF')]
    assert R.grid_validity(seeds, checks, ['CAND', 'TWIN'], {})['label'] == 'READER_SEEDS_EFFECTIVE'
    ch = copy.deepcopy(checks)
    for c in ch:
        if c['recipe'] in ('ET5', 'ET25') and c['wire'] == 'REF' and c['view'] == 'income':
            c['equal_k0_k1'] = True
    assert R.grid_validity(seeds, ch, ['CAND', 'TWIN'], {})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'   # count unrecorded
    ok = R.grid_validity(seeds, ch, ['CAND', 'TWIN'], {'REF|V0': {'income': 6}})
    assert ok['label'] == 'READER_SEEDS_EFFECTIVE' and ok['explained_equalities'] == 2
    for c in ch:
        if c['wire'] == 'CAND' and c['recipe'] == 'MLP64':
            c['equal_k0_k1'] = True
    assert R.grid_validity(seeds, ch, ['CAND', 'TWIN'], {'REF|V0': {'income': 6}})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    old = dict(seeds); old['S1k1'] = R.MODEL_SEED_BASE + 500 + 11
    assert R.grid_validity(old, checks, ['CAND'], {})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    lone = [c for c in checks if c['wire'] != 'REF'] + [{'recipe': 'ET5', 'view': 'income', 'wire': 'REF', 'v': 0, 'R': 'R1',
                                                         'equal_k0_k1': True}]
    assert R.grid_validity(seeds, lone, ['CAND', 'TWIN'], {'REF|V0': {'income': 6}})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    assert R.grid_validity({'S0k0': seeds['S0k0']}, checks, ['CAND'], {})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'


# ---------------------------------------------------------------- frontier and reserve frontier
def costs(total, se=0.0005, extra=None):
    c = {'total|nll': {'value': total, 'se': se}}
    for m in R.SECONDARY:
        c[m] = {'value': 0.0, 'se': 0.0005}
    c.update(extra or {})
    return {v: copy.deepcopy(c) for v in R.VERSIONS}


def t_frontier_bracketing():
    path = [{'id': 'C1'}, {'id': 'C2'}, {'id': 'C3'}]
    pt = {'id': 'P', 'costs_vs': {'C1': costs(0.004), 'C2': costs(-0.002), 'C3': costs(-0.008)},
          'grid_vs': {'C1': grid_entry(0.020), 'C2': grid_entry(0.012), 'C3': grid_entry(0.003)}}
    f = R.frontier_comparison(pt, path)
    assert f['label'] == 'DOMINATES_CONTROL_FRONTIER' and f['bracketing_control'] == 'C2' and 'interpolation' in f
    assert abs(f['interpolation']['lambda'] - 0.004 / 0.006) < 1e-12
    pt2 = copy.deepcopy(pt); pt2['grid_vs']['C2'] = grid_entry(0.002)
    f2 = R.frontier_comparison(pt2, path)
    assert f2['label'] == 'NOT_SEPARATED_FROM_TASK_SPENDING' and f2['reason'] == 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'
    pt3 = copy.deepcopy(pt); pt3['costs_vs'] = {k: costs(0.003) for k in ('C1', 'C2', 'C3')}
    f3 = R.frontier_comparison(pt3, path)
    assert f3['label'] == 'NOT_SEPARATED_FROM_TASK_SPENDING' and f3['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    pt4 = copy.deepcopy(pt); pt4['costs_vs']['C2'] = costs(-0.002, extra={'occupation|men_brier': {'value': 0.004, 'se': 0.001}})
    f4 = R.frontier_comparison(pt4, path)
    pt6 = copy.deepcopy(pt)
    for v in R.VERSIONS:
        del pt6['costs_vs']['C2'][v]['income|brier']
    assert R.frontier_comparison(pt6, path)['label'] == 'INCOMPLETE'
    pt7 = copy.deepcopy(pt); pt7['costs_vs']['C2'][0]['occupation|men_brier'] = {'value': 0.05, 'se': math.nan}
    assert R.frontier_comparison(pt7, path)['label'] == 'INCOMPLETE'
    assert f4['label'] == 'SEPARATED_ON_PRIMARY_COST_ONLY' and f4['secondary_more_spending'] == ['occupation|men_brier']
    pt5 = copy.deepcopy(pt)
    for k in pt5['costs_vs']:
        for v in R.VERSIONS:
            t = pt5['costs_vs'][k][v].pop('total|nll')
            pt5['costs_vs'][k][v]['income|nll'] = {'value': t['value'] / 2, 'se': 0.0003}
            pt5['costs_vs'][k][v]['occupation|nll'] = {'value': t['value'] / 2, 'se': 0.0003}
    f5 = R.frontier_comparison(pt5, path)
    assert f5['label'] == 'DOMINATES_CONTROL_FRONTIER' and f5['flags'] == ['COST_SE_BOUND']
    assert R.frontier_comparison(pt, [])['reason'] == 'CONTROL_PATH_MISSING'
    assert R.frontier_comparison({'id': 'P', 'costs_vs': {}}, path)['label'] == 'INCOMPLETE'


def t_reserve_frontier():
    pts = [{'id': 'CAND', 'kind': 'candidate', 'reserve_qualified': True, 'pair_auc': 0.600, 'min_ratio': 2.4},
           {'id': 'W', 'kind': 'reference', 'reserve_qualified': True, 'pair_auc': 0.648, 'min_ratio': 2.86},
           {'id': 'CTL', 'kind': 'control', 'reserve_qualified': True, 'pair_auc': 0.640, 'min_ratio': 2.1},
           {'id': 'H_LJ', 'kind': 'reference', 'reserve_qualified': False, 'pair_auc': 0.618, 'min_ratio': 0.32}]
    paired = {('CAND', 'W'): grid_entry(0.040), ('CAND', 'CTL'): grid_entry(0.030)}
    t = R.reserve_frontier(pts, paired)
    rec = {p['id']: p for p in t['points']}
    assert rec['CAND']['label'] == 'RESERVE_FRONTIER_IMPROVEMENT' and rec['CAND']['on_reserve_pareto_frontier']
    assert rec['W']['on_reserve_pareto_frontier'] and not rec['CTL']['on_reserve_pareto_frontier']
    assert t['references_failing_reserve'] == ['H_LJ'] and 'not superiority' in t['note']
    paired[('CAND', 'CTL')] = grid_entry(0.001)
    assert {p['id']: p for p in R.reserve_frontier(pts, paired)['points']}['CAND']['label'] == 'NO_RESERVE_FRONTIER_IMPROVEMENT'


# ---------------------------------------------------------------- claims and selection
def record(benefit=0.030, matched=0.015, frontier='DOMINATES_CONTROL_FRONTIER', ev=None, inc=0.0, occ=0.0, joint=None, **kw):
    grid = {n: grid_entry(benefit, inc, occ) for n in R.BENCHMARK}
    grid['TWIN'] = grid_entry(matched)
    grid['ALT'] = grid_entry(matched)
    r = {'evals': ev or evals(), 'grid': grid, 'matched': {'twin': 'TWIN', 'alternative': 'ALT'},
         'frontier': {'label': frontier}, 'joint': joint, 'grid_valid': True}
    r.update(kw)
    return r


def t_claims_g1_to_g7():
    c = R.claims(record())
    assert [c[g]['label'] for g in ('G1', 'G2', 'G3', 'G4', 'G5')] == [
        'TASK_FEASIBLE', 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS', 'STABLE_FRONTIER_IMPROVEMENT',
        'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'MATERIAL_0020_RESOLVED']
    assert c['tier'] == 'T5' and c['useful_component'] is None and c['G7']['label'] == 'NOT_CLAIMED'
    assert c['G6']['label'] == 'NOT_APPLICABLE_NOT_JOINT'
    c = R.claims(record(benefit=0.012))                               # positive but below the material target
    assert c['G4']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' and c['G5']['label'] == 'POSITIVE_BELOW_MATERIAL_TARGET'
    assert c['tier'] == 'T4'
    c = R.claims(record(benefit=-0.004))                              # useful component, not competitive
    assert c['G2']['label'] == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' and c['G4']['label'] == 'NOT_SUPERIOR'
    assert c['useful_component'] == 'USEFUL_COMPONENT_NOT_COMPETITIVE' and c['tier'] == 'T3'
    c = R.claims(record(ev=evals(occ={('U0', 'nll_excess'): 0.0101})))  # protection without feasibility is never claimable
    assert c['G1']['label'] == 'TASK_INFEASIBLE' and c['G2']['label'] == 'MET_WITHOUT_TASK_FEASIBILITY' and c['tier'] is None
    assert c['G5']['label'] == 'MET_WITHOUT_TASK_FEASIBILITY'
    c = R.claims(record(occ=0.012))                                   # a recipient view above the allowance blocks G4
    assert c['G4']['label'] == 'NOT_SUPERIOR' and c['G2']['label'] == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS'
    c = R.claims(record(frontier='NOT_SEPARATED_FROM_TASK_SPENDING'))
    assert c['G3']['label'] == 'NOT_SEPARATED_FROM_TASK_SPENDING' and c['tier'] == 'T2'
    r = record(); r['matched'] = {'twin': 'TWIN'}
    assert R.claims(r)['G2']['label'] == 'NO_MATCHED_CONTROLS'
    r = record(); del r['grid']['TT100']
    assert R.claims(r)['G4']['label'] == 'INCOMPLETE'
    # joint design: one output changed; missing controls; supported
    c = R.claims(record(joint={'outputs_changed': ['occupation'], 'jointly_designed': False}))
    assert c['G6']['label'] == 'NOT_APPLICABLE_ONE_OUTPUT_CHANGED' and 'occupation' in c['G6']['statement']
    c = R.claims(record(joint={'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': 'TWIN'}))
    assert c['G6']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS'
    j = {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': 'SEP', 'income_first': 'IF',
         'occupation_first': 'OF', 'costs_vs': {n: costs(-0.001) for n in ('SEP', 'IF', 'OF')}}
    r = record(joint=j)
    r['grid'].update({n: grid_entry(0.010) for n in ('SEP', 'IF', 'OF')})
    assert R.claims(r)['G6']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED'
    r['joint']['costs_vs']['IF'] = costs(0.004)                      # the joint design spends resolved-more than one order
    assert R.claims(r)['G6']['label'] == 'NOT_SUPPORTED'
    assert R.claims(record(novelty='NOVEL_PER_SEPARATE_REVIEW'))['G7']['label'] == 'NOVEL_PER_SEPARATE_REVIEW'
    assert R.claims(record(novelty='BEST_EVER'))['G7']['label'] == 'NOT_CLAIMED'
    # review findings 4, 6, 7
    r = record(); del r['grid_valid']
    assert R.claims(r)['G4']['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN' and R.claims(r)['tier'] is None
    assert R.claims(record(grid_valid=np.False_))['G4']['label'] == 'NOT_COMPUTED_GRID_INVALID'
    c = R.claims(record(joint={'outputs_changed': ['income', 'occupation'], 'jointly_designed': False}))
    assert c['G6']['label'] == 'NOT_APPLICABLE_NOT_JOINT'
    c = R.claims(record(joint={'outputs_changed': ['income', 'income'], 'jointly_designed': True}))
    assert c['G6']['label'] == 'NOT_APPLICABLE_ONE_OUTPUT_CHANGED' and c['G6']['changed_outputs'] == ['income']
    c = R.claims(record(benefit=0.012, ev=evals(occ={('U0', 'nll_excess'): 0.0101})))
    assert c['G5']['label'] == 'MET_WITHOUT_TASK_FEASIBILITY' and c['G5']['level'] == 'POSITIVE_BELOW_MATERIAL_TARGET'
    c = R.claims(record(benefit=0.040, occ=0.030))
    assert c['G5']['label'] == 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED' and c['G4']['label'] == 'NOT_SUPERIOR'
    c = R.claims(record(grid_valid=False))                           # invalid reader grid: no claim label beyond G1
    assert c['G1']['label'] == 'TASK_FEASIBLE' and c['G4']['label'] == 'NOT_COMPUTED_GRID_INVALID' and c['tier'] is None


def wires(tag):
    return {v: {'income': f'i{tag}{v}', 'occupation': f'o{tag}{v}'} for v in R.VERSIONS}


REF_WIRES = {r: wires('REF' + r) for r in R.BENCHMARK}


def ledger_entry(cid, entry_no, kind='candidate', comps=3, **kw):
    r = record(**kw)
    r.update({'config_id': cid, 'family': cid[:2], 'kind': kind, 'entry': entry_no, 'n_fitted_components': comps,
              'wire_sha256_eval': wires(cid), 'wire_sha256_grid': wires(cid)})
    return r


def t_selection_rules():
    led = [ledger_entry('FA_1', 1, benefit=0.012), ledger_entry('FB_1', 2, benefit=0.014),
           ledger_entry('CT_1', 3, kind='control', benefit=0.030)]
    s = R.select_candidate(led, REF_WIRES)
    assert s['selected'] == 'CT_1' and s['tier'] == 'T5'                  # a control may win; tier first
    ch = next(r for r in s['records'] if r['config_id'] == 'CT_1')
    assert 'CONTROL_DESIGNATED_CANDIDATE' in ch['flags']
    led = [ledger_entry('FA_1', 1, benefit=0.012), ledger_entry('FB_1', 2, benefit=0.014)]
    s = R.select_candidate(led, REF_WIRES)                                       # same tier, near tie (0.002 < 0.005): fewer components? equal
    assert s['near_tie_set'] == ['FA_1', 'FB_1'] and s['selected'] == 'FA_1'   # equal reserve and components -> earlier entry
    led = [ledger_entry('FA_1', 1, benefit=0.012, comps=5), ledger_entry('FB_1', 2, benefit=0.014, comps=2)]
    assert R.select_candidate(led, REF_WIRES)['selected'] == 'FB_1'                 # fewer fitted components
    led = [ledger_entry('FA_1', 1, benefit=0.012), ledger_entry('FB_1', 2, benefit=0.0175)]
    assert R.select_candidate(led, REF_WIRES)['selected'] == 'FB_1'                 # outside the near-tie width: score decides
    led = [ledger_entry('FA_1', 1, benefit=0.012, ev=evals(se=0.004)), ledger_entry('FB_1', 2, benefit=0.013)]
    sel = R.select_candidate(led, REF_WIRES)
    assert sel['selected'] == 'FB_1'                                     # reserve-qualified first (FA_1 fails the reserve)
    dup = ledger_entry('FB_2', 4, benefit=0.040)
    dup['wire_sha256_eval'] = dup['wire_sha256_grid'] = wires('FB_1')
    s = R.select_candidate([ledger_entry('FB_1', 2, benefit=0.012), dup], REF_WIRES)
    assert s['selected'] == 'FB_1' and next(r for r in s['records'] if r['config_id'] == 'FB_2')['status'] == 'ALIAS_OF_FB_1'
    mm = ledger_entry('FC_1', 5, benefit=0.040); mm['wire_sha256_grid'] = wires('FC_1x')
    s = R.select_candidate([mm], REF_WIRES)
    assert s['outcome'] == 'NO_DEVELOPMENT_CANDIDATE' and s['records'][0]['status'] == 'NOT_SELECTABLE_OUTPUT_MISMATCH'
    al = ledger_entry('FD_1', 6, benefit=0.040)
    s = R.select_candidate([al], {**REF_WIRES, 'H_LJ': wires('FD_1')})
    assert s['records'][0]['status'] == 'ALIAS_OF_H_LJ' and s['selected'] is None
    s = R.select_candidate([ledger_entry('FE_1', 7, matched=0.0)], REF_WIRES)      # no incremental reduction: below the minimum tier
    assert s['outcome'] == 'NO_DEVELOPMENT_CANDIDATE' and s['records'][0]['status'] == 'BELOW_MINIMUM_TIER'
    led = [ledger_entry('FA_1', 1, benefit=0.012, comps=5), ledger_entry('FB_1', 2, benefit=0.013, comps=0)]
    assert R.select_candidate(led, REF_WIRES)['selected'] == 'FB_1'                    # 0 components is fewest, not infinity
    led = [ledger_entry('FA_1', None, benefit=0.012, comps={0: 3, 1: 3}), ledger_entry('FB_1', 2, benefit=0.013, comps={0: 2, 1: 3})]
    assert R.select_candidate(led, REF_WIRES)['selected'] == 'FB_1'                    # summed over versions; a None entry sorts last
    try:
        R.select_candidate(led, None)
        raise AssertionError('missing reference wires accepted')
    except ValueError:
        pass
    az = ledger_entry('FG_1', 9, benefit=0.030); az['grid']['H_LJ'] = ALIAS
    s = R.select_candidate([az], REF_WIRES)
    assert s['records'][0]['status'] == 'ALIAS_OF_H_LJ' and s['selected'] is None
    led = [ledger_entry('FA_1', 1, benefit=0.012, ev=evals(se=0.0015)), ledger_entry('FB_1', 2, benefit=0.013)]
    sel = R.select_candidate(led, REF_WIRES)                                          # both qualified: leximin decides
    assert all(r['reserve_qualified'] for r in sel['records']) and sel['selected'] == 'FB_1'
    inc = ledger_entry('FF_1', 8, frontier='DOMINATES_CONTROL_FRONTIER')
    inc['grid'] = {k: g for k, g in inc['grid'].items() if k != 'W'}  # T3 claims, but the reference grid is incomplete
    s = R.select_candidate([inc], REF_WIRES)
    assert s['records'][0]['status'] == 'NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID' and s['selected'] is None


# ---------------------------------------------------------------- known-law simulation of the aggregate screen
LAWS = {
    'sampling_dominated': dict(s=0.004, sV=0.001, sR=0.001, sG=0.001, sK=0.002, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.0),
    'reader_split_dominated': dict(s=0.003, sV=0.002, sR=0.003, sG=0.002, sK=0.004, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.0),
    'version_dominated': dict(s=0.003, sV=0.005, sR=0.001, sG=0.001, sK=0.002, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.0),
    'deterministic_winners': dict(s=0.004, sV=0.001, sR=0.001, sG=0.001, sK=0.002, rb=0.8, rv=0.5, rr=0.15, pdet=0.5, contam=0.0),
    'one_contaminated_cell': dict(s=0.004, sV=0.001, sR=0.001, sG=0.001, sK=0.002, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.2),
    'large_noise': dict(s=0.008, sV=0.003, sR=0.003, sG=0.002, sK=0.005, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.0),
    'version_extreme': dict(s=0.002, sV=0.008, sR=0.001, sG=0.001, sK=0.001, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.0),
    'split_extreme': dict(s=0.002, sV=0.001, sR=0.008, sG=0.001, sK=0.001, rb=0.8, rv=0.5, rr=0.15, pdet=0.0, contam=0.0),
}
IDX = [(v, r, k) for v in R.VERSIONS for r in (0, 1) for k in (0, 1)]           # registered cell order (R.REGISTERED)


def _corr(rb, rv, rr):
    C = np.empty((8, 8))
    for i, (v, r, k) in enumerate(IDX):
        for j, (w, q, m) in enumerate(IDX):
            C[i, j] = 1.0 if i == j else (rb if (v, r) == (w, q) else rv if r == q else rr)
    return C


def simulate(law, mu, n, rng):
    """n replicates x 8 registered cells: x = mu + version + split (shared by both versions) + interaction + reader seed +
    shared-household sampling error (correlation rb same block, rv same split other version, rr other split); reported SEs carry
    the B = 400 bootstrap noise; seed-deterministic blocks (prob pdet) duplicate k0 into k1; contamination shifts one cell by
    +/- 0.015 with probability contam."""
    L = np.linalg.cholesky(_corr(law['rb'], law['rv'], law['rr']))
    a = rng.normal(0, law['sV'], (n, 2)); b = rng.normal(0, law['sR'], (n, 2)); g = rng.normal(0, law['sG'], (n, 2, 2))
    d = rng.normal(0, law['sK'], (n, 8)); e = rng.normal(size=(n, 8)) @ L.T * law['s']
    x = np.empty((n, 8)); se = law['s'] * (1 + rng.normal(0, 1 / math.sqrt(2 * 399), (n, 8)))
    for i, (v, r, k) in enumerate(IDX):
        x[:, i] = mu + a[:, v] + b[:, r] + g[:, v, r] + d[:, i] + e[:, i]
    if law['pdet'] > 0:
        dup = rng.random((n, 4)) < law['pdet']
        for blk in range(4):
            x[dup[:, blk], 2 * blk + 1] = x[dup[:, blk], 2 * blk]; se[dup[:, blk], 2 * blk + 1] = se[dup[:, blk], 2 * blk]
    if law['contam'] > 0:
        hit = rng.random(n) < law['contam']
        cell = rng.integers(0, 8, n); sign = rng.choice([-1.0, 1.0], n)
        x[hit, cell[hit]] += 0.015 * sign[hit]
    return x, np.abs(se)


def screen_vec(x, se, threshold=0.0, kind='benefit', z=R.Z, floor=R.SE_FLOOR, limit=R.RECIPIENT_LIMIT):
    """Vectorized copy of nrm.rules (registered cells) and of the old every-cell rules, for the simulation."""
    mb = x.reshape(-1, 4, 2).mean(2)                                   # block means (S0R1, S0R2, S1R1, S1R2)
    w2 = se.reshape(-1, 4, 2) ** 2
    if kind == 'benefit':
        m = mb.mean(1)
        hh = np.sqrt(w2.mean((1, 2)))
        rs = np.sqrt(((mb - m[:, None]) ** 2).sum(1) / 4)
        sa = np.maximum(np.sqrt(hh ** 2 + rs ** 2), floor)
        vm = mb.reshape(-1, 2, 2).mean(2)
        new = (m - z * sa > threshold) & (vm > threshold).all(1)
        # old rule (tsz.rules.stability_label): every cell > threshold and d_v - z sd_total > threshold, distinct-cell SS
        d = vm
        xb = x.reshape(-1, 2, 2, 2)
        eq = xb[..., 0] == xb[..., 1]
        dev = (xb - d[:, :, None, None]) ** 2
        ss = dev[..., 0].sum((1, 2)) + np.where(eq, 0.0, dev[..., 1]).sum((1, 2))
        neff = 8 - eq.sum((1, 2))
        sd = np.sqrt(ss / (neff - 2) + (se ** 2).mean(1))
        old = (x > threshold).all(1) & ((d - z * sd[:, None]) > threshold).all(1)
        return new, old, m
    vm = mb.reshape(-1, 2, 2).mean(2)
    hh = np.sqrt(w2.reshape(-1, 2, 4).mean(2))
    rs = np.sqrt(((mb.reshape(-1, 2, 2) - vm[:, :, None]) ** 2).sum(2) / 2)
    sv = np.maximum(np.sqrt(hh ** 2 + rs ** 2), floor)
    gate = (vm <= limit).all(1)
    resolved = (vm + z * sv <= limit).all(1)
    old = (x <= limit).all(1)
    return gate, resolved, old


def _rate(a):
    return round(float(np.mean(a)), 4)


def t_known_law_screen_checks():
    rng = np.random.default_rng(20261020)
    n = 20000
    t0 = time.process_time()
    out = {'schema': 'NRM1-screen-checks-v1', 'replicates_per_setting': n, 'rng_seed': 20261020, 'laws': {},
           'model': 'x_vrk = mu + a_v + b_r + g_vr + d_vrk + e_vrk; a: version (procedure repeat), b: split shared by both versions '
                    '(same split tag), g: version-specific readers on a split, d: reader seed, e: SCORE-household sampling error with '
                    'correlation rb (same block), rv (same split, other version), rr (other split); reported cell SE = s x (1 + '
                    'N(0, 1/sqrt(798))) (B = 400 bootstrap noise); deterministic blocks duplicate k0 into k1; contamination shifts one '
                    'cell by +/- 0.015. Pass rates are empirical under these invented laws only: not a universal error guarantee.',
           'z': R.Z, 'se_floor': R.SE_FLOOR}
    consistency = 0
    for name, law in LAWS.items():
        rec = {'law': law, 'benefit_screen': {}, 'target_screen': {}, 'recipient_screen': {}}
        true_sd_mean = None
        for mu in (0.0, 0.005, 0.010, 0.020):
            x, se = simulate(law, mu, n, rng)
            new, old, m = screen_vec(x, se)
            if true_sd_mean is None:
                true_sd_mean = float(np.std(m))
            rec['benefit_screen'][f'mu={mu}'] = {'new_AGG_POSITIVE': _rate(new), 'old_STABLE_POSITIVE': _rate(old),
                                                 'role': 'false-pass rate (null)' if mu == 0 else 'pass rate (alternative)'}
            for i in range(150):                                       # the scalar implementation agrees with the vectorized one
                e = entry(dict(zip(R.REGISTERED, x[i].tolist())), dict(zip(R.REGISTERED, se[i].tolist())))
                b = R.benefit_screen(e)
                assert b['resolved'] == bool(new[i]) and bool(b['old_every_cell_rule']['stable']) == bool(old[i]), (name, mu, i)
                consistency += 1
        rec['empirical_sd_of_aggregate_mean'] = round(true_sd_mean, 6)
        for mu in (0.020, 0.025, 0.030, 0.040):
            x, se = simulate(law, mu, n, rng)
            new, old, _ = screen_vec(x, se, threshold=R.TARGET)
            rec['target_screen'][f'mu={mu}'] = {'new_TARGET_AGG_RESOLVED': _rate(new), 'old_DEV_GRID_TARGET_STABLE': _rate(old),
                                                'role': 'false-pass rate (null at 0.020)' if mu == 0.020 else 'pass rate'}
        for mu in (0.0, 0.005, 0.010, 0.015, 0.020):
            x, se = simulate(law, mu, n, rng)
            gate, resolved, old = screen_vec(x, se, kind='recipient')
            rec['recipient_screen'][f'mu={mu}'] = {'new_gate_point_per_version': _rate(gate), 'new_RECIPIENT_AGG_RESOLVED': _rate(resolved),
                                                   'old_every_cell_gate': _rate(old)}
            for i in range(100):
                e = entry(dict(zip(R.REGISTERED, x[i].tolist())), dict(zip(R.REGISTERED, se[i].tolist())))
                rc = R.recipient_screen(e)
                assert rc['gate_pass'] == bool(gate[i]) and (rc['label'] == 'RECIPIENT_AGG_RESOLVED') == bool(resolved[i])
                assert rc['old_every_cell_rule']['pass'] == bool(old[i])
                consistency += 1
        out['laws'][name] = rec
    # exact aliases: identical candidate and comparator in every replicate -> never passes
    alias_pass = sum(R.benefit_screen(ALIAS['pair'])['resolved'] for _ in range(1))
    out['exact_alias'] = {'benefit_pass_rate': float(alias_pass), 'target_pass_rate': 0.0 if R.target_screen(ALIAS['pair'])['label']
                          != 'TARGET_AGG_RESOLVED' else 1.0, 'recipient_gate_pass': R.recipient_screen(ALIAS['income'])['gate_pass'],
                          'note': 'deterministic: an exact alias has every cell value and SE exactly 0 (shared readers through the '
                                  'wire-hash cache); the screen labels it ALIAS_ZERO and it can never be positive'}
    out['scalar_vector_consistency_checks'] = consistency
    out['cpu_s'] = round(time.process_time() - t0, 1)
    # sanity assertions (behaviour, not a guarantee): monotone power; alias never passes
    for name, rec in out['laws'].items():
        rates = [rec['benefit_screen'][f'mu={mu}']['new_AGG_POSITIVE'] for mu in (0.0, 0.005, 0.010, 0.020)]
        assert rates == sorted(rates), (name, rates)
    assert out['exact_alias']['benefit_pass_rate'] == 0.0 and out['exact_alias']['target_pass_rate'] == 0.0
    OUT.write_text(json.dumps(out, indent=1) + '\n')


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

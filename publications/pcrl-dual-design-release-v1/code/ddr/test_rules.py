"""Invented-data tests of ddr.rules (no real data, no label). Run: python3 -m ddr.test_rules (prints JSON; exit 0 iff all pass).

Covers: the rule file and its refusal of altered constants; nrm.rules left unchanged (never monkeypatched); reader-seed validity
of the DDR1 grid (+800 registered, 300-700 forbidden, bootstrap seeds, the prospective extra-trees explanation with its support
bound, seed pairs judged separately); the genuinely effective refit; the bracket rule (equal to nrm.rules.frontier_comparison);
D1 labels; D2 conjuncts, partial labels, the recipient gate at the bracket, domination by a cheaper control, missing brackets,
aliases, controls at the identity, gates and missing inputs; randomized releases (MC SE through NR unchanged); D3 / D4 with the
reader-budget credit rule; D5 / D6; the outcome class (every class and flag); registered vs adaptive labelling and the campaign
summary; the strongest feasible nonzero release; D2R robustness. Helpers for invented utility records and grid entries come from
nrm.test_rules (unchanged)."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

from nrm import rules as NR
from nrm import test_rules as NT                 # invented-data helpers (util, evals, entry, reg, grid_entry, costs, ALIAS)
from ddr import rules as R

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- invented records
SBC = ['SBC_1', 'SBC_2']
SPEND = ['SPD_1', 'SPD_2']
POINTS = ('TWIN',) + tuple(SBC) + tuple(SPEND)


def d2_rec(pair=0.02, twin_pair=0.03, refs=0.03, g1=True, grid_valid=True, nolook=0.01, status='FITTED', revision=0,
           family='DA', early=True):
    """A complete invented family reading: CAND spends more than TWIN (+0.003 nats), SBC_1 / SPD_1 are cheaper than CAND and
    SBC_2 / SPD_2 more expensive (the brackets)."""
    ev = NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.011})
    grid = {i: NT.grid_entry(pair) for i in POINTS}
    grid['TWIN'] = NT.grid_entry(twin_pair)
    grid.update({r: NT.grid_entry(refs) for r in NR.BENCHMARK})
    costs = {'TWIN': NT.costs(0.003), 'SBC_1': NT.costs(0.001), 'SBC_2': NT.costs(-0.001), 'SPD_1': NT.costs(0.0015),
             'SPD_2': NT.costs(-0.0005)}
    return {'family': family, 'card': f'{family}_x', 'revision': revision, 'card_committed_before_any_model_select_outcome': early,
            'grid_valid': grid_valid, 'twin': {'config_id': 'TWIN', 'evals': NT.evals()},
            'cand': {'status': status, 'config_id': 'CAND', 'evals': ev}, 'strengths': [{'config_id': 'CAND', 'strength': 2.0,
                                                                                          'evals': ev}],
            'grid': grid, 'paths': {'SBC': list(SBC), 'SPEND': list(SPEND)}, 'alias_of': {}, 'costs_vs': costs,
            'nolookup': {i: {'cells': NT.reg(lambda c: nolook)} for i in POINTS + NR.BENCHMARK}}


def seeds_all(offset=R.SEED_OFFSET, ks=(0, 1, 2, 3)):
    return {f'S{v}k{k}': NR.MODEL_SEED_BASE + offset + 10 * v + k for v in NR.VERSIONS for k in ks}


def boots(reps=('R1', 'R2')):
    return {f'S{v}{r}': R.registered_boot_seed(v, r) for v in NR.VERSIONS for r in reps}


def checks(wires=('CAND', 'TWIN', 'REF'), reps=('R1', 'R2'), k_pair=None):
    out = []
    for r in NR.SEED_DEPENDENT:
        for vw in NR.VIEWS:
            for w in wires:
                for v in NR.VERSIONS:
                    for rp in reps:
                        c = {'recipe': r, 'view': vw, 'wire': w, 'v': v, 'R': rp, 'equal_k0_k1': False}
                        if k_pair is not None:
                            c['k_pair'] = list(k_pair)
                        out.append(c)
    return out


def with_eq(ch, keys):
    """Mark the checks whose (wire, view, v, R, recipe[, k_pair]) is in keys as seed-pair equal."""
    ch = copy.deepcopy(ch)
    for c in ch:
        k5 = (c['wire'], c['view'], c['v'], c['R'], c['recipe'])
        if k5 in keys or k5 + (tuple(c.get('k_pair', (0, 1))),) in keys:
            c['equal_k0_k1'] = True
    return ch


DI = {f'{w}|V{v}': {'income': 16, 'occupation': 64, 'pair': 1000} for w in ('CAND', 'TWIN', 'REF') for v in NR.VERSIONS}


def reg_seeds():
    return {k: x for k, x in seeds_all().items() if k[-1] in '01'}


# ---------------------------------------------------------------- rule file and the inherited module
@test
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'DDR1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(R.RULES_PATH.read_text())
    for path, val in ((('reader_grid', 'seed_offset'), 700), (('reader_grid', 'forbidden_offsets'), [300, 400, 500, 600]),
                      (('reader_grid', 'bootstrap_seed_base'), 2026102100), (('er_explanation', 'min_leaf', 'ET25'), 5),
                      (('er_explanation', 'pool_people'), 4000), (('z',), 1.96), (('schema',), 'FRC1-rules-v1'),
                      (('decisions', 'D2', 'met_label'), 'COMPONENT_BEYOND_MATCHED_CONTROLS'),
                      (('outcome_class', 'labels'), list(R.OUTCOMES[:-1])), (('registration_status', 'kinds'), ['ADAPTIVE']),
                      (('decisions', 'D4', 'target_references'), ['H_LJ'])):
        bad = copy.deepcopy(raw)
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'RULES.json'
            p.write_text(json.dumps(bad))
            try:
                R.load_rules(p)
            except ValueError:
                continue
            raise AssertionError(f'altered {path} accepted')


@test
def t_nrm_rules_unchanged_not_monkeypatched():
    assert hashlib.sha256(Path(NR.__file__).read_bytes()).hexdigest() == R.NRM_RULES_PY_SHA256
    assert NR.load_rules()['_sha256'] == R.NRM1_RULES_SHA256 and NR.Z == R.Z
    before = NR.forbidden_seeds()
    R.grid_validity(reg_seeds(), checks(), ['CAND'], DI, boots())
    R.refit_effectiveness({}, seeds_all(), checks(), ['CAND'], DI, boots(R.ALL_REPS))
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.forbidden_seeds() == before
    assert R.forbidden_seeds() > NR.forbidden_seeds() and len(R.forbidden_seeds()) == 100
    assert R.registered_seed(1, 3) == NR.MODEL_SEED_BASE + 813 and R.registered_boot_seed(1, 'R4') == 2026102303
    assert R.support_bound('ET5') == 1600 and R.support_bound('ET25') == 320 and R.support_bound('MLP64') is None


# ---------------------------------------------------------------- (9) grid validity
@test
def t_grid_validity_seeds_and_bootstrap():
    g = R.grid_validity(reg_seeds(), checks(), ['CAND', 'TWIN'], DI, boots())
    assert g['label'] == 'READER_SEEDS_EFFECTIVE' and g['strict_nrm1']['label'] == 'READER_SEEDS_EFFECTIVE', g['errors']
    for off in R.FORBIDDEN_OFFSETS:
        old = {k: x for k, x in seeds_all(off).items() if k[-1] in '01'}
        g = R.grid_validity(old, checks(), ['CAND'], DI, boots())
        assert g['label'] == 'READER_SEEDS_NOT_EFFECTIVE' and any('reuse earlier' in e for e in g['errors']), off
    seven = {k: x for k, x in seeds_all(700).items() if k[-1] in '01'}
    assert NR.grid_validity(seven, checks(), ['CAND'], DI)['label'] == 'READER_SEEDS_EFFECTIVE'   # nrm alone accepts +700
    shifted = dict(reg_seeds())
    shifted['S1k1'] = NR.MODEL_SEED_BASE + 800 + 19                                            # unregistered, not forbidden
    g = R.grid_validity(shifted, checks(), ['CAND'], DI, boots())
    assert not g['effective'] and any('differ from the registered' in e for e in g['errors'])
    assert not R.grid_validity({'S0k0': reg_seeds()['S0k0']}, checks(), ['CAND'], DI, boots())['effective']
    dup = dict(reg_seeds())
    dup['S1k1'] = dup['S1k0']
    assert not R.grid_validity(dup, checks(), ['CAND'], DI, boots())['effective']
    weird = dict(reg_seeds())
    weird['X'] = NR.MODEL_SEED_BASE + 899
    assert not R.grid_validity(weird, checks(), ['CAND'], DI, boots())['effective']
    g = R.grid_validity(reg_seeds(), checks(), ['CAND'], DI)                                  # bootstrap seeds not recorded
    assert not g['effective'] and any('bootstrap seeds not recorded' in e for e in g['errors'])
    b = boots()
    b['S1R2'] += 1
    assert any('S1R2' in e for e in R.grid_validity(reg_seeds(), checks(), ['CAND'], DI, b)['errors'])
    b = boots()
    b['S0R5'] = 1
    assert not R.grid_validity(reg_seeds(), checks(), ['CAND'], DI, b)['effective']
    b = boots()
    del b['S0R1']
    assert not R.grid_validity(reg_seeds(), checks(), ['CAND'], DI, b)['effective']
    g = R.grid_validity(reg_seeds(), [c for c in checks() if c['recipe'] != 'ET5'], ['CAND'], DI, boots())
    assert any('not checked' in e for e in g['errors'])
    assert not R.grid_validity(reg_seeds(), [], ['CAND'], DI, boots())['effective']             # nothing checked
    bad = checks() + [{'recipe': 'ET5', 'view': 'pair', 'wire': 'CAND', 'v': 0, 'R': 'R1', 'equal_k0_k1': False, 'k_pair': [0]}]
    assert any('malformed' in e for e in R.grid_validity(reg_seeds(), bad, ['CAND'], DI, boots())['errors'])
    nd = checks() + [{'recipe': 'LOOKUP', 'view': 'pair', 'wire': 'CAND', 'v': 0, 'R': 'R1', 'equal_k0_k1': False}]
    g = R.grid_validity(reg_seeds(), nd, ['CAND'], DI, boots())
    assert g['effective'] and g['flags'] == ['NONDETERMINISTIC_RECIPE']


@test
def t_grid_validity_extra_trees_explanation():
    ok = with_eq(checks(), {('CAND', 'income', 0, 'R1', 'ET5'), ('TWIN', 'occupation', 1, 'R2', 'ET25')})
    g = R.grid_validity(reg_seeds(), ok, ['CAND', 'TWIN'], DI, boots())
    assert g['label'] == 'READER_SEEDS_EFFECTIVE', g['errors']
    assert g['explained_minimum_set_equalities']['count'] == 2 and g['strict_nrm1']['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    ne = g['explained_minimum_set_equalities']['n_eff_implied']['CAND|income|ET5|k0k1']
    assert ne['blocks_checked'] == 4 and ne['blocks_seed_pair_equal'] == 1 and ne['n_eff_seed_cells_if_this_recipe_wins'] == 7
    # an MLP equality on the minimum set is never explained (no seed-invariance mechanism), even with (a) + (b) + (c)
    mlp = with_eq(checks(), {('CAND', 'income', 0, 'R1', 'MLP64')})
    g = R.grid_validity(reg_seeds(), mlp, ['CAND', 'TWIN'], DI, boots())
    assert not g['effective'] and any('on the minimum set' in e for e in g['errors'])
    # ... while off the minimum set NRM1 (iii) applies unchanged and explains it
    assert R.grid_validity(reg_seeds(), mlp, ['TWIN'], DI, boots())['effective']
    # support bound: ET25 on a view with 321 > 8001 // 25 distinct inputs is not explained; 320 is
    for n, want in ((321, False), (320, True), (0, False)):
        di = copy.deepcopy(DI)
        di['TWIN|V1']['occupation'] = n
        eq = with_eq(checks(), {('TWIN', 'occupation', 1, 'R2', 'ET25')})
        assert R.grid_validity(reg_seeds(), eq, ['CAND', 'TWIN'], di, boots())['effective'] is want, n
    di = copy.deepcopy(DI)
    del di['CAND|V0']['income']                                                                    # (c) not recorded
    eq = with_eq(checks(), {('CAND', 'income', 0, 'R1', 'ET5')})
    assert not R.grid_validity(reg_seeds(), eq, ['CAND'], di, boots())['effective']
    di['CAND|V0']['income'] = True                                                                 # a bool is not a count
    assert not R.grid_validity(reg_seeds(), eq, ['CAND'], di, boots())['effective']
    allrec = with_eq(checks(), {('CAND', 'income', 0, 'R1', r) for r in NR.SEED_DEPENDENT})    # (a) fails
    assert not R.grid_validity(reg_seeds(), allrec, ['CAND'], DI, boots())['effective']
    allview = with_eq(checks(), {('CAND', w, 0, 'R1', 'ET5') for w in NR.VIEWS})                # (b) fails
    assert not R.grid_validity(reg_seeds(), allview, ['CAND'], DI, boots())['effective']
    other = with_eq(checks(), {('REF', 'income', 0, 'R1', 'ET5')})                               # off the minimum set
    assert R.grid_validity(reg_seeds(), other, ['CAND'], DI, boots())['effective']
    assert not R.grid_validity(reg_seeds(), other, ['CAND'], {}, boots())['effective']


@test
def t_seed_pairs_judged_separately():
    ch = checks(reps=('R3', 'R4'), k_pair=(0, 1)) + checks(reps=R.ALL_REPS, k_pair=(2, 3))
    g = R.grid_validity(seeds_all(), ch, ['CAND'], DI, boots(R.ALL_REPS), required_k=R.ALL_K, reps=R.ALL_REPS)
    assert g['effective'] and set(g['per_seed_pair']) == {'k0k1', 'k2k3'}, g['errors']
    # in pair k2 / k3 every seed-dependent recipe is equal in one block: (a) fails there, although pair k0 / k1 differs
    keys = {('REF', 'income', 0, 'R3', r, (2, 3)) for r in NR.SEED_DEPENDENT}
    g = R.grid_validity(seeds_all(), with_eq(ch, keys), ['CAND'], DI, boots(R.ALL_REPS), required_k=R.ALL_K, reps=R.ALL_REPS)
    assert not g['effective'] and g['per_seed_pair']['k2k3']['unexplained_other'] == len(NR.SEED_DEPENDENT)
    pooled = [{k: v for k, v in c.items() if k != 'k_pair'} for c in with_eq(ch, keys)]            # NRM1 pooling would pass it
    assert NR.grid_validity(seeds_all(), pooled, [], DI)['label'] == 'READER_SEEDS_EFFECTIVE'


# ---------------------------------------------------------------- (8) refit effectiveness
def splits(overlap=0.3):
    reps = R.ALL_REPS
    return {r: {'assignment_sha256': f'hash{r}', 'score_overlap': {q: overlap for q in reps if q != r}} for r in reps}


def refit_checks(eq=()):
    ch = checks(reps=('R3', 'R4'), k_pair=(0, 1)) + checks(reps=R.ALL_REPS, k_pair=(2, 3))
    return with_eq(ch, set(eq))


@test
def t_refit_effectiveness():
    args = (seeds_all(), refit_checks(), ['CAND', 'TWIN'], DI, boots(R.ALL_REPS))
    ok = R.refit_effectiveness(splits(), *args)
    assert ok['label'] == 'REFIT_GENUINELY_EFFECTIVE' and ok['counted_once'] == [], ok['reasons']
    sp = splits()
    sp['R4']['assignment_sha256'] = sp['R2']['assignment_sha256']
    bad = R.refit_effectiveness(sp, *args)
    assert bad['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and bad['counted_once'] == ['robustness repetitions']
    assert R.refit_effectiveness(splits(1.0), *args)['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'
    sp = splits()
    del sp['R3']['score_overlap']['R1']
    del sp['R1']['score_overlap']['R3']
    assert R.refit_effectiveness(sp, *args)['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'
    # the prospective extra-trees explanation applies to the refit (FRC1 kept the strict rule there)
    et = R.refit_effectiveness(splits(), seeds_all(), refit_checks({('CAND', 'income', 0, 'R3', 'ET5', (2, 3))}),
                               ['CAND', 'TWIN'], DI, boots(R.ALL_REPS))
    assert et['label'] == 'REFIT_GENUINELY_EFFECTIVE' and et['grid_validity']['explained_minimum_set_equalities']['count'] == 1
    mlp = R.refit_effectiveness(splits(), seeds_all(), refit_checks({('CAND', 'income', 0, 'R3', 'MLP64', (2, 3))}),
                                ['CAND', 'TWIN'], DI, boots(R.ALL_REPS))
    assert mlp['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and mlp['counted_once'] == ['robustness reader seeds']
    s = seeds_all()
    del s['S1k3']
    assert R.refit_effectiveness(splits(), s, refit_checks(), ['CAND'], DI, boots(R.ALL_REPS))['label'] == \
        'REFIT_NOT_GENUINELY_EFFECTIVE'
    assert R.refit_effectiveness(splits(), seeds_all(), refit_checks(), ['CAND'], DI, boots())['label'] == \
        'REFIT_NOT_GENUINELY_EFFECTIVE'                                                         # R3 / R4 bootstrap seeds missing


# ---------------------------------------------------------------- (3) brackets and D1
@test
def t_brackets_equal_nrm_rule():
    r = d2_rec()
    b = R.brackets(r['costs_vs'], r['paths'])
    assert b['SBC']['bracket'] == 'SBC_2' and b['SPEND']['bracket'] == 'SPD_2' and b['SBC']['path'] == ['TWIN', 'SBC_1', 'SBC_2']
    for key, ids in (('SBC', SBC), ('SPEND', SPEND)):
        fr = NR.frontier_comparison({'id': 'CAND', 'costs_vs': r['costs_vs'], 'grid_vs': r['grid']},
                                    [{'id': i} for i in R.control_path(ids)])
        assert fr['bracketing_control'] == b[key]['bracket']
    cheap = copy.deepcopy(r['costs_vs'])
    cheap['SBC_2'] = NT.costs(0.0004)
    assert R.brackets(cheap, r['paths'])['SBC']['status'] == 'NO_BRACKET'
    tie = copy.deepcopy(r['costs_vs'])
    tie['SBC_1'] = NT.costs(-0.001)                                                             # tie: path order kept
    assert R.brackets(tie, r['paths'])['SBC']['bracket'] == 'SBC_1'
    miss = copy.deepcopy(r['costs_vs'])
    del miss['TWIN']
    assert R.brackets(miss, r['paths'])['SBC']['status'] == 'INCOMPLETE'
    assert R.brackets(r['costs_vs'], {'SBC': None})['SBC']['status'] == 'PATH_NOT_GIVEN'
    assert R.control_path(['SBC_1', 'SBC_2'], {'SBC_1': 'TWIN'}) == ['TWIN', 'SBC_2'] and R.control_path(None) is None
    assert R.control_path([], {}) == ['TWIN'] and R.control_path(['A', 'B'], {'B': 'A'}) == ['TWIN', 'A']


@test
def t_d1_labels():
    assert R.d1(d2_rec())['label'] == 'TASK_FEASIBLE'
    r = d2_rec()
    r['twin']['evals'] = NT.evals(inc={('U0', 'accuracy_loss'): 0.0101})
    d = R.d1(r)
    assert d['label'] == 'CAND_FEASIBLE_TWIN_INFEASIBLE' and d['twin_failed'] == ['v0|income|U0|accuracy_loss',
                                                                                   'v1|income|U0|accuracy_loss']
    assert R.d1(d2_rec(g1=False))['label'] == 'CAND_INFEASIBLE'
    r = d2_rec(g1=False)
    r['twin']['evals'] = {0: NT.util(0)}                                     # TWIN incomplete: CAND's infeasibility still known
    assert R.d1(r)['label'] == 'CAND_INFEASIBLE'
    r = d2_rec()
    r['twin']['evals'] = {0: NT.util(0)}
    assert R.d1(r)['label'] == 'INCOMPLETE'
    assert R.d1(d2_rec(status='NO_FEASIBLE_STRENGTH'))['label'] == 'NO_FEASIBLE_STRENGTH'
    assert R.d1(d2_rec(status=None))['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['cand']['evals'] = {1: NT.util(1)}
    assert R.d1(r)['label'] == 'INCOMPLETE'
    e = NT.evals()
    e[1] = NT.util(1, guards={'sex0': 0.005})                                # inclusive cap
    r = d2_rec()
    r['cand']['evals'] = e
    assert R.d1(r)['label'] == 'TASK_FEASIBLE'
    r['strengths'].append({'config_id': 'ZERO', 'strength': 0.0, 'evals': NT.evals()})
    assert R.d1(r)['ignored_zero_strength_entries'] == ['ZERO']


# ---------------------------------------------------------------- (3) D2
@test
def t_d2_met_and_structure():
    d = R.d2(d2_rec())
    assert d['label'] == R.D2_MET and d['met'] and d['passed'] == list(R.D2_COMPARISONS), d
    assert d['comparisons']['SBC_FRONTIER']['point'] == 'SBC_2' and d['comparisons']['SPEND_FRONTIER']['point'] == 'SPD_2'
    assert d['comparisons']['SBC_FRONTIER']['path'] == ['TWIN', 'SBC_1', 'SBC_2'] and set(d['targeting']) == {'TWIN', 'SBC_2',
                                                                                                                 'SPD_2'}
    assert d['flags'] == [] and d['comparisons']['SBC_FRONTIER']['recipient_gate_vs_bracket'] is True


@test
def t_d2_recipient_gate_at_bracket():
    r = d2_rec()
    r['grid']['SBC_2']['income'] = NT.grid_entry(0.0, inc=0.02)['income']                       # income view worse by 0.02
    fr = NR.frontier_comparison({'id': 'CAND', 'costs_vs': r['costs_vs'], 'grid_vs': r['grid']},
                                [{'id': i} for i in R.control_path(SBC)])
    assert fr['label'] == 'DOMINATES_CONTROL_FRONTIER'                                          # the inherited rule alone passes
    d = R.d2(r)
    c = d['comparisons']['SBC_FRONTIER']
    assert not c['pass'] and c['reason'] == 'RECIPIENT_GATE_FAILED_AT_BRACKETING_CONTROL'
    assert 'RECIPIENT_GATE_FAILED_VS_SBC_2' in d['flags'] and d['label'] == 'PARTIAL_TWIN+SPEND_FRONTIER_ONLY'
    r = d2_rec()
    r['grid']['SPD_2']['occupation'] = NT.grid_entry(0.0, occ=0.0105)['occupation']             # just above the allowance
    assert R.d2(r)['label'] == 'PARTIAL_TWIN+SBC_FRONTIER_ONLY'
    r = d2_rec()
    r['grid']['SPD_2']['occupation'] = NT.grid_entry(0.0, occ=0.010)['occupation']              # inclusive allowance passes
    assert R.d2(r)['met']
    r = d2_rec()
    del r['grid']['SPD_2']['income']
    assert R.d2(r)['label'] == 'INCOMPLETE'


@test
def t_d2_cheaper_control_domination():
    r = d2_rec()
    r['grid']['SBC_1'] = NT.grid_entry(-0.02)                       # SBC_1 is cheaper AND resolved less recoverable than CAND
    d = R.d2(r)
    c = d['comparisons']['SBC_FRONTIER']
    assert c['label'] == 'DOMINATES_CONTROL_FRONTIER' and not c['pass'] and c['dominated_by_cheaper'] == ['SBC_1']
    assert c['reason'] == 'DOMINATED_BY_CHEAPER_CONTROL' and 'DOMINATED_BY_CHEAPER_SBC_1' in d['flags']
    assert d['label'] == 'PARTIAL_TWIN+SPEND_FRONTIER_ONLY'
    r['grid']['SBC_1'] = NT.grid_entry(-0.005)                      # point-only lower recovery: not a resolved domination
    assert R.d2(r)['met']
    r['grid']['SBC_1'] = NT.grid_entry(lambda c: -0.02 if c.startswith('S0') else 0.001)          # a version disagrees
    assert R.d2(r)['met']
    r['grid']['SBC_1'] = copy.deepcopy(NT.ALIAS)                    # an exact alias of CAND is not 'less recoverable'
    assert R.d2(r)['met']
    del r['grid']['SBC_1']
    assert R.d2(r)['label'] == 'INCOMPLETE'


@test
def t_d2_brackets_and_secondary_spending():
    r = d2_rec()
    r['costs_vs']['SBC_2'] = NT.costs(0.0004)                       # every SBC point cheaper than CAND: no bracket
    d = R.d2(r)
    c = d['comparisons']['SBC_FRONTIER']
    assert c['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT' and c['point'] is None and 'NO_BRACKET_SBC' in d['flags']
    assert d['label'] == 'PARTIAL_TWIN+SPEND_FRONTIER_ONLY' and set(d['targeting']) == {'TWIN', 'SPD_2'}
    r = d2_rec()
    r['costs_vs']['SBC_2'] = NT.costs(-0.001, extra={'occupation|acc_loss': {'value': 0.004, 'se': 0.001}})
    d = R.d2(r)
    assert d['comparisons']['SBC_FRONTIER']['label'] == 'SEPARATED_ON_PRIMARY_COST_ONLY' and not d['met']
    r = d2_rec()
    r['grid']['SBC_2'] = NT.grid_entry(0.005)                       # point-only benefit at the bracket
    d = R.d2(r)
    assert d['comparisons']['SBC_FRONTIER']['reason'] == 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'
    r = d2_rec()
    del r['costs_vs']['SBC_2'][0]['occupation|men_brier']
    assert R.d2(r)['label'] == 'INCOMPLETE'                         # a missing secondary metric is never skipped
    r = d2_rec()
    del r['costs_vs']['TWIN']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['costs_vs']['TWIN'] = NT.costs(-0.0001)                       # CAND cheaper than TWIN: TWIN is the bracket of both paths
    d = R.d2(r)
    assert d['comparisons']['SBC_FRONTIER']['point'] == 'TWIN' and d['comparisons']['SPEND_FRONTIER']['point'] == 'TWIN'
    assert d['met'] and set(d['targeting']) == {'TWIN'}


@test
def t_d2_targeting_partials_and_not_separated():
    r = d2_rec()
    r['nolookup']['SPD_2'] = {'cells': NT.reg(lambda c: -0.01 if c.startswith('S1') else 0.01)}
    d = R.d2(r)
    assert d['label'] == 'READER_TARGETED' and 'READER_TARGETED_SPEND_FRONTIER' in d['flags'] and not d['met']
    r = d2_rec()
    r['nolookup']['TWIN'] = {'cells': NT.reg(lambda c: 0.0)}        # point benefit must be > 0, not >= 0
    assert R.d2(r)['label'] == 'READER_TARGETED'
    r = d2_rec()
    del r['nolookup']['SBC_2']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['nolookup']['SBC_1'] = {'cells': NT.reg(lambda c: -0.05)}     # a cheaper (non-bracketing) point is not a targeting point
    assert R.d2(r)['met']
    assert R.d2(d2_rec(pair=-0.01, twin_pair=-0.01))['label'] == 'NOT_SEPARATED'
    assert R.d2(d2_rec(pair=0.0))['label'] == 'PARTIAL_TWIN_ONLY'
    r = d2_rec(twin_pair=0.0)
    assert R.d2(r)['label'] == 'PARTIAL_SBC_FRONTIER+SPEND_FRONTIER_ONLY'
    r = d2_rec()
    r['grid']['TWIN']['income'] = NT.grid_entry(0.0, inc=0.02)['income']                       # recipient gate vs TWIN
    assert 'TWIN' not in R.d2(r)['passed']
    r = d2_rec()
    r['enhanced'] = {'SBC_2': {'cells': NT.reg(lambda c: -0.01)}}   # enhanced winner: reported only
    d = R.d2(r)
    assert d['met'] and 'READER_SENSITIVE_ENHANCED_SBC_2' in d['flags']


@test
def t_d2_aliases_identity_and_gates():
    r = d2_rec()
    r['grid']['TWIN'] = copy.deepcopy(NT.ALIAS)
    d = R.d2(r)
    assert 'CAND_ALIAS_OF_TWIN' in d['flags'] and 'TWIN' not in d['passed']
    r = d2_rec()
    r['paths']['SPEND'] = []
    d = R.d2(r)
    assert 'SPEND_AT_IDENTITY' in d['flags'] and d['comparisons']['SPEND_FRONTIER']['path'] == ['TWIN']
    assert d['comparisons']['SPEND_FRONTIER']['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    r = d2_rec()
    r['alias_of'] = {'SBC_2': 'SBC_1'}                              # alias counted once: the bracket disappears
    d = R.d2(r)
    assert d['comparisons']['SBC_FRONTIER']['path'] == ['TWIN', 'SBC_1'] and 'SBC_FRONTIER' not in d['passed']
    r = d2_rec()
    r['alias_of'] = {'SPD_1': 'TWIN', 'SPD_2': 'TWIN'}              # every SPEND point an alias of TWIN
    assert 'SPEND_AT_IDENTITY' in R.d2(r)['flags']
    assert R.d2(d2_rec(status='NO_FEASIBLE_STRENGTH'))['label'] == 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH'
    r = d2_rec()
    r['cand']['every_nonzero_strength_alias_of_twin'] = True
    d = R.d2(r)
    assert d['label'] == 'NOT_ASSESSED_EXACT_ALIAS' and d['family_stops']
    assert R.d2(d2_rec(g1=False))['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    assert R.d2(d2_rec(g1=False, grid_valid=False))['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'     # feasibility first
    assert R.d2(d2_rec(grid_valid=None))['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    assert R.d2(d2_rec(grid_valid=False))['label'] == 'NOT_COMPUTED_GRID_INVALID'
    assert R.d2(d2_rec(grid_valid='true'))['label'] == 'NOT_COMPUTED_GRID_INVALID'               # only the boolean True
    assert R.d2(d2_rec(status='SAVED'))['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['paths']['SBC'] = None
    r['controls_not_run'] = {'SBC': True}
    assert R.d2(r)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    r['controls_not_run'] = {}
    assert R.d2(r)['label'] == 'INCOMPLETE'                         # a missing path is never 'not run' unless recorded
    r = d2_rec()
    del r['grid']['TWIN']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['cand']['evals'] = {0: NT.util(0)}
    assert R.d2(r)['label'] == 'INCOMPLETE'


@test
def t_randomized_mc_through_nr():
    c = NR.mc_cell([0.018, 0.022, 0.020], [0.004, 0.004, 0.004])
    assert abs(c['value'] - 0.02) < 1e-15 and abs(c['se_mc'] - 0.002 / math.sqrt(3)) < 1e-15
    r = d2_rec()
    r['grid']['SBC_2']['pair'] = NT.entry(NT.reg(lambda x: c['value']), c['se'], mc=c['se_mc'])
    assert R.d2(r)['met']                                           # small MC SE: still resolved
    r['grid']['SBC_2']['pair'] = NT.entry(NT.reg(lambda x: 0.02), 0.004, mc=0.012)
    d = R.d2(r)                                                     # large MC SE enters se_agg: no longer resolved
    assert d['comparisons']['SBC_FRONTIER']['reason'] == 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL' and not d['met']
    one = NR.mc_cell([0.02], [0.004])
    r = d2_rec()
    r['mc_flags'] = one['flags']
    out = R.decisions(r)
    assert 'MC_SE_NOT_ESTIMABLE' in out['outcome_class']['flags']
    r = d2_rec()                                                    # MC SE on a paired cost moves the secondary check
    r['costs_vs']['SBC_2'] = NT.costs(-0.001, extra={'occupation|acc_loss': {'value': 0.003, 'se': 0.001, 'se_mc': 0.002}})
    assert R.d2(r)['met']                                           # 0.003 - 2 sqrt(0.001^2 + 0.002^2) < 0: not resolved


# ---------------------------------------------------------------- (2-7) decisions and the outcome class
@test
def t_outcome_classes():
    out = R.decisions(d2_rec())
    assert out['outcome_class']['label'] == 'CANDIDATE' and out['D3']['met'] and out['D4']['label'] == 'MATERIAL_0020_RESOLVED'
    assert out['D5']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS' and out['D6']['label'] == 'NOT_CLAIMED' and out['registered']
    out = R.decisions(d2_rec(refs=0.001))
    assert out['outcome_class']['label'] == 'COMPONENT' and out['D3']['label'] == 'NOT_SUPERIOR'
    out = R.decisions(d2_rec(refs=0.012))
    assert out['D4']['label'] == 'POSITIVE_BELOW_MATERIAL_TARGET' and not out['D4']['met'] and out['D3']['met']
    r = d2_rec()
    r['grid']['SBC_2'] = NT.grid_entry(0.0)
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'TASK_ONLY_FEASIBLE_NO_PROTECTION' and 'D3_WITHOUT_D2' in out['outcome_class']['flags']
    r = d2_rec(g1=False)
    r['strengths'].append({'config_id': 'LOW', 'strength': 1.0, 'evals': NT.evals(occ={('U0', 'nll_excess'): 0.012})})
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'TASK_INFEASIBLE' and out['D2']['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    assert out['D3']['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE' and out['D4']['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    r['strengths'][-1]['evals'] = NT.evals()
    assert R.decisions(r)['outcome_class']['label'] == 'SELECTED_STRENGTH_INFEASIBLE'
    r['strengths'][-1]['evals'] = {0: NT.util(0)}
    assert R.decisions(r)['outcome_class']['label'] == 'INCOMPLETE'
    r['strengths'] = None
    assert R.decisions(r)['outcome_class']['label'] == 'INCOMPLETE'
    r = d2_rec(g1=False, grid_valid=None)
    r['strengths'] = []                                             # CAND the only fitted strength: infeasible whatever the grid
    assert R.decisions(r)['outcome_class']['label'] == 'TASK_INFEASIBLE'
    out = R.decisions(d2_rec(status='NO_FEASIBLE_STRENGTH'))
    assert out['outcome_class']['label'] == 'TASK_INFEASIBLE' and 'NO_STRENGTH_MEETS_MODEL_FIT_BUDGET' in \
        out['outcome_class']['flags'] and out['D3']['label'] == 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH'
    r = d2_rec()
    r['paths']['SPEND'] = None
    r['controls_not_run'] = {'SPEND': True}
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'PROTECTION_NOT_ASSESSED' and out['D3']['met'] and \
        'D3_WITHOUT_D2' in out['outcome_class']['flags']
    assert R.decisions(d2_rec(grid_valid=None))['outcome_class']['label'] == 'INCOMPLETE'
    assert R.decisions(d2_rec(grid_valid=False))['D3']['label'] == 'NOT_COMPUTED_GRID_INVALID'
    r = d2_rec()
    r['twin']['evals'] = NT.evals(occ={('Ucal', 'brier_excess'): 0.0051})
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'CANDIDATE' and 'TWIN_INFEASIBLE' in out['outcome_class']['flags']
    assert out['D1']['label'] == 'CAND_FEASIBLE_TWIN_INFEASIBLE'
    r = d2_rec()
    r['cand']['every_nonzero_strength_alias_of_twin'] = True
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'TASK_ONLY_FEASIBLE_NO_PROTECTION' and 'FAMILY_STOPS_EXACT_ALIAS' in \
        out['outcome_class']['flags'] and out['D3']['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    tight = d2_rec()
    tight['cand']['evals'] = NT.evals(occ={('U0', 'nll_excess'): 0.0095}, se=0.002)
    assert 'POINT_FEASIBLE_NOT_RESERVE_QUALIFIED' in R.decisions(tight)['outcome_class']['flags']
    r = d2_rec()
    del r['grid']['SPD_1']                                          # D2 incomplete: the class is incomplete, D3 still read
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'INCOMPLETE' and out['D3']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES'


@test
def t_d3_d4_reader_budget_and_incomplete_classes():
    r = d2_rec()
    r['reader_budget'] = {'W': 'READER_BUDGET_ASYMMETRIC', 'H_LJ': 'READER_BUDGET_COMPARABLE'}
    out = R.decisions(r)
    assert out['D3']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' and out['D3']['reader_budget_credit'] == {'W': 'CREDITED'}
    r['nolookup']['W'] = {'cells': NT.reg(lambda c: -0.01 if c.startswith('S1') else 0.01)}
    out = R.decisions(r)
    assert out['D3']['label'] == 'NOT_SUPERIOR' and out['D3']['not_credited'] == ['W'] and 'W' in out['D3']['failing']
    assert out['outcome_class']['label'] == 'COMPONENT' and out['D4']['label'] == 'MATERIAL_0020_RESOLVED'
    r = d2_rec()
    r['reader_budget'] = {'TT100': 'READER_BUDGET_ASYMMETRIC'}
    del r['nolookup']['TT100']                                      # missing -> INCOMPLETE, never 'not beaten'
    out = R.decisions(r)
    assert out['D3']['label'] == 'INCOMPLETE' and out['outcome_class']['label'] == 'INCOMPLETE'   # D2 met, D3 undecided
    r['grid']['SBC_2'] = NT.grid_entry(0.0)                         # D2 not met: the class no longer needs D3
    assert R.decisions(r)['outcome_class']['label'] == 'TASK_ONLY_FEASIBLE_NO_PROTECTION'
    r = d2_rec()
    r['reader_budget'] = {'TT100': 'READER_BUDGET_ASYMMETRIC'}
    del r['nolookup']['TT100']
    r['grid']['SUCC'] = NT.grid_entry(0.0)                          # failing on its own result: NOT_SUPERIOR stays
    out = R.decisions(r)
    assert out['D3']['label'] == 'NOT_SUPERIOR' and out['D3']['incomplete_references'] == ['TT100']
    r = d2_rec()
    r['reader_budget'] = {'NEURAL_RAW': 'READER_BUDGET_ASYMMETRIC', 'H_LJ': 'READER_BUDGET_ASYMMETRIC'}
    r['nolookup']['NEURAL_RAW'] = {'cells': NT.reg(lambda c: -0.002)}
    out = R.decisions(r)
    assert out['D4']['label'] == 'NOT_MET' and out['D4']['reader_budget_credit']['NEURAL_RAW'] == 'NOT_CREDITED'
    r = d2_rec()
    r['reader_budget'] = {'H_LJ': 'READER_BUDGET_ASYMMETRIC'}
    del r['nolookup']['H_LJ']
    assert R.decisions(r)['D4']['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['grid']['TT100']                                          # a reference comparison missing
    assert R.decisions(r)['D3']['label'] == 'INCOMPLETE'


@test
def t_d5_d6():
    r = d2_rec()
    r['joint'] = {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': 'SEP'}
    assert R.decisions(r)['D5']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS'
    for n in ('SEP', 'IF', 'OF'):
        r['grid'][n] = NT.grid_entry(0.02)
    r['joint'].update({'income_first': 'IF', 'occupation_first': 'OF', 'costs_vs': {n: NT.costs(-0.001) for n in ('SEP', 'IF', 'OF')}})
    assert R.decisions(r)['D5']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED'
    r['joint']['costs_vs']['IF'] = NT.costs(0.004)                  # resolved more spending than a sequential order
    assert R.decisions(r)['D5']['label'] == 'NOT_SUPPORTED'
    r = d2_rec()
    r['novelty'] = 'NOVEL_PER_SEPARATE_REVIEW'
    assert R.decisions(r)['D6']['label'] == 'NOVEL_PER_SEPARATE_REVIEW'
    r['novelty'] = 'NOVEL_BECAUSE_IT_WORKS'
    assert R.decisions(r)['D6']['label'] == 'NOT_CLAIMED'


# ---------------------------------------------------------------- (7) registered vs adaptive readings, campaign summary
@test
def t_registered_and_adaptive_labelling():
    out = R.decisions(d2_rec())
    assert out['registered'] and out['adaptive_tag'] is None and out['outcome_class']['label'] == 'CANDIDATE'
    assert out['outcome_class']['label_unprefixed'] == 'CANDIDATE' and out['D2']['label_unprefixed'] == R.D2_MET
    out = R.decisions(d2_rec(revision=1, early=False))
    assert not out['registered'] and out['adaptive_tag'] == 'ADAPTIVE_R1'
    assert out['outcome_class']['label'] == 'ADAPTIVE_R1__CANDIDATE' and out['D2']['label'] == f'ADAPTIVE_R1__{R.D2_MET}'
    assert out['D1']['label'] == 'ADAPTIVE_R1__TASK_FEASIBLE' and out['outcome_class']['label_unprefixed'] == 'CANDIDATE'
    assert out['D2']['met'] is True                                 # the raw booleans are kept for the summary
    cases = {(1, 'DA', True): 'REVISED_R1', (2, 'DB', False): 'ADAPTIVE_R2', (0, 'FS_X', False): 'ADAPTIVE_FS',
             (0, 'FS_X', True): 'REVISED_FS', (1, 'FS_X', False): 'ADAPTIVE_FS_R1', (0, 'DA', False): 'ADAPTIVE_R0_LATE_CARD',
             (0, 'DB', True): None}
    for (n, fam, early), want in cases.items():
        assert R.adaptive_tag({'revision': n, 'family': fam, 'card_committed_before_any_model_select_outcome': early}) == want
    assert R.adaptive_tag({'revision': 0, 'family': 'DA'}) == 'ADAPTIVE_R0_LATE_CARD'           # not recorded: not registered
    assert R.adaptive_tag({'revision': 0, 'family': 'DA', 'card_committed_before_any_model_select_outcome': 'yes'}) == \
        'ADAPTIVE_R0_LATE_CARD'


@test
def t_campaign_summary():
    da = R.decisions(d2_rec(family='DA'))
    db_rec = d2_rec(family='DB')
    db_rec['grid']['SBC_2'] = NT.grid_entry(0.0)
    db = R.decisions(db_rec)
    da1 = R.decisions(d2_rec(family='DA', revision=1, early=False))
    inf = R.decisions(d2_rec(family='FS_Y', g1=False, early=False))
    s = R.campaign_summary([da, db, da1, inf])
    assert set(s['registered']) == {'DA', 'DB'} and s['d2_looks'] == 3 and len(s['d2_met']) == 2 and s['flags'] == []
    assert s['statement'].startswith('D2 was met in 1 of 2 registered family readings and in 1 of 1 non-registered readings '
                                     '(3 D2 looks in total)')
    assert s['registered']['DB']['outcome_class'] == 'TASK_ONLY_FEASIBLE_NO_PROTECTION'
    assert [x['tag'] for x in s['non_registered']] == ['ADAPTIVE_R1', 'ADAPTIVE_FS']
    s = R.campaign_summary([db, da1])
    assert s['flags'] == ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS']
    try:
        R.campaign_summary([da, R.decisions(d2_rec(family='DA'))])
    except ValueError:
        return
    raise AssertionError('two registered readings of one family accepted')


# ---------------------------------------------------------------- (8) strongest nonzero release and D2R
def cfg(cid, strength, x, g1=True, alias=None, revision=0, twin='TWIN'):
    return {'config_id': cid, 'strength': strength, 'revision': revision, 'twin_id': twin,
            'evals': NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.02}),
            'pair_vs_twin': NT.grid_entry(x)['pair'] if x is not None else copy.deepcopy(NT.ALIAS['pair']), 'alias_of': alias}


@test
def t_strongest_nonzero():
    s = R.strongest_nonzero([cfg('TW', 0.0, 0.05), cfg('P1', 0.5, 0.004), cfg('P2', 2.0, 0.012), cfg('P3', 8.0, 0.03, g1=False)])
    assert s['selected'] == 'P2' and s['flags'] == []
    s = R.strongest_nonzero([cfg('P3', 4.0, 0.012), cfg('P2', 1.0, 0.012), cfg('P1', 0.25, 0.004)])
    assert s['selected'] == 'P2' and s['tied'] == ['P2', 'P3']
    s = R.strongest_nonzero([cfg('R1P', 1.0, 0.012, revision=1), cfg('R0P', 1.0, 0.012)])           # tie: lower revision
    assert s['selected'] == 'R0P' and s['revision'] == 0
    s = R.strongest_nonzero([cfg('P1', 0.25, None), cfg('P2', 1.0, 0.02, alias='P1')])
    assert s['label'] == 'NO_FEASIBLE_NONZERO_RELEASE' and s['records'][0]['status'] == 'ALIAS_OF_TWIN'
    assert R.strongest_nonzero([])['label'] == 'NO_FEASIBLE_NONZERO_RELEASE'
    assert R.strongest_nonzero([cfg('P1', 0.25, -0.002)])['selected'] == 'P1'                     # negative still the strongest
    s = R.strongest_nonzero([cfg('PV', {0: 1.0, 1: 2.0}, 0.01), cfg('PZ', {0: 1.0, 1: 0.0}, 0.02)])  # zero in one version
    assert s['selected'] == 'PV' and s['records'][1]['status'] == 'NOT_NONZERO'
    s = R.strongest_nonzero([cfg('A', 1.0, 0.01), cfg('B', 1.0, 0.012, revision=1, twin='TWIN_R1')])
    assert s['selected'] == 'B' and s['flags'] == ['MIXED_TWINS']
    bad = cfg('Q', 1.0, 0.01)
    bad['pair_vs_twin'] = {'cells': {}, 'cell_se': {}}
    assert R.strongest_nonzero([bad])['records'][0]['status'] == 'INCOMPLETE_GRID'


def rob_entry(x, reg_cells=True, rob=None):
    """Merged entry: registered cells (optional) + the 24 robustness cells, constant values, SE 0.004."""
    rob_cells = [f'S{v}{r}k{k}' for v in NR.VERSIONS for r in ('R3', 'R4') for k in range(4)] + \
                [f'S{v}{r}k{k}' for v in NR.VERSIONS for r in ('R1', 'R2') for k in (2, 3)]
    vals = {c: (rob if rob is not None else x) for c in rob_cells}
    if reg_cells:
        vals.update({c: x for c in NR.REGISTERED})
    e = NT.entry(vals, 0.004)
    return {'pair': e, 'income': NT.entry({c: 0.0 for c in vals}, 0.004), 'occupation': NT.entry({c: 0.0 for c in vals}, 0.004)}


@test
def t_d2r():
    strongest = R.strongest_nonzero([cfg('CAND', 2.0, 0.02)])
    refit = {'label': 'REFIT_GENUINELY_EFFECTIVE', 'counted_once': []}
    rob = {c: rob_entry(0.02) for c in ('TWIN', 'SBC_2', 'SPD_2', 'NEURAL_RAW', 'H_LJ')}
    base = {'grid_valid': True, 'strongest': strongest, 'refit': refit, 'rob': rob, 'brackets': {'SBC': 'SBC_2', 'SPEND': 'SPD_2'}}
    d = R.d2r(base)
    assert d['label'] == 'ROBUSTNESS_CONFIRMED' and d['decisive'] == ['TWIN', 'SBC_2', 'SPD_2'] and \
        set(d['per_comparator']) == {'TWIN', 'SBC_2', 'SPD_2', 'NEURAL_RAW', 'H_LJ'}
    assert d['per_comparator']['TWIN']['n_eff_robustness_cells'] == 8                            # constant cells once per block (8 blocks)
    r = copy.deepcopy(base)
    r['brackets'] = R.brackets(d2_rec()['costs_vs'], d2_rec()['paths'])                         # a brackets() record is accepted
    assert R.d2r(r)['decisive'] == ['TWIN', 'SBC_2', 'SPD_2']
    r = copy.deepcopy(base)
    r['refit'] = {'label': 'REFIT_NOT_GENUINELY_EFFECTIVE', 'counted_once': ['robustness reader seeds']}
    d = R.d2r(r)
    assert d['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and 'REFIT_NOT_GENUINELY_EFFECTIVE' in d['per_comparator']['TWIN']['reasons']
    r = copy.deepcopy(base)
    r['rob']['SBC_2'] = rob_entry(0.02, reg_cells=False)
    d = R.d2r(r)
    assert d['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d['per_comparator']['SBC_2']['reasons'] == ['REGISTERED_CELLS_ABSENT']
    r = copy.deepcopy(base)
    r['rob']['SPD_2'] = rob_entry(0.02, rob=0.001)                  # registered positive, robustness cells not
    d = R.d2r(r)
    assert d['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d['per_comparator']['SPD_2']['nrm_label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    r = copy.deepcopy(base)
    r['rob']['NEURAL_RAW'] = rob_entry(-0.02)                       # references are reported, not decisive
    assert R.d2r(r)['label'] == 'ROBUSTNESS_CONFIRMED'
    r = copy.deepcopy(base)
    del r['rob']['TWIN']
    assert R.d2r(r)['label'] == 'NOT_RUN'
    r = copy.deepcopy(base)
    r['strongest'] = R.strongest_nonzero([])
    assert R.d2r(r)['label'] == 'NOT_RUN'
    r = copy.deepcopy(base)
    r['grid_valid'] = False
    assert R.d2r(r)['label'] == 'NOT_COMPUTED_GRID_INVALID'
    r['grid_valid'] = None
    assert R.d2r(r)['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'


def main():
    t0 = time.process_time()
    res = []
    for t in TESTS:
        try:
            t()
            res.append({'test': t.__name__, 'result': 'PASS'})
        except Exception as ex:                                            # report every failure, then exit non-zero
            res.append({'test': t.__name__, 'result': 'FAIL', 'error': f'{type(ex).__name__}: {ex}'})
    npass = sum(r['result'] == 'PASS' for r in res)
    try:
        sha = R.load_rules()['_sha256']
    except (ValueError, KeyError, OSError) as ex:                         # a refused rule file fails the run, reported
        sha = f'REFUSED: {ex}'
    out = {'schema': R.SCHEMA, 'rules_sha256': sha, 'tests': res, 'passed': npass, 'total': len(res),
           'all_pass': npass == len(res) and not sha.startswith('REFUSED'), 'cpu_s': round(time.process_time() - t0, 3),
           'note': 'invented data only; no label read, no real-data computation'}
    print(json.dumps(out, indent=1))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())

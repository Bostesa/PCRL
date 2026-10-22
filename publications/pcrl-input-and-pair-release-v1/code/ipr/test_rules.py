"""Invented-data tests of ipr.rules (no real data, no label). Run: python3 -m ipr.test_rules (prints JSON; exit 0 iff all pass).

Covers: the rule file and its refusal of altered constants; nrm.rules left unchanged (never monkeypatched); reader-seed validity
of the IPR1 grid (+900 registered, 300-800 forbidden, bootstrap seeds, the prospective extra-trees explanation with its support
bound, seed pairs judged separately); the genuinely effective refit; the bracket rule on all three control paths (equal to
nrm.rules.frontier_comparison); D1 labels; D2 conjuncts with the SBC, SPEND and RO paths, partial labels named by what passed,
the recipient gate at the bracket, domination by a cheaper control, missing brackets, aliases, controls at the identity, gates and
missing inputs; the RO declaration (SUBSTITUTED / GENERIC_ZTILT, the release-only operation's own met label, misdeclarations);
a SEX-blind operation (no SBC path, D2 cannot be met); the MODEL_FIT control-grid fineness check; randomized releases (MC SE
through NR unchanged); D3 / D4 with the reader-budget credit rule; D5 = NR G6 (pass-through, INCOMPLETE for missing records,
gated like D3 / D4) and D6; the outcome class (every class and flag, the by-version flags); families IA / IB / FS_<name> only;
registered vs adaptive labelling and the campaign summary; component-vector strengths; the strongest feasible nonzero release;
D2R robustness over the three paths. Helpers for invented utility records and grid entries come from nrm.test_rules (unchanged)."""
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
from ipr import rules as R

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- invented records
SBC = ['SBC_1', 'SBC_2']
SPEND = ['SPD_1', 'SPD_2']
RO = ['RO_1', 'RO_2']
POINTS = ('TWIN',) + tuple(SBC) + tuple(SPEND) + tuple(RO)
ALL3 = 'PARTIAL_TWIN+SBC_FRONTIER+SPEND_FRONTIER_ONLY'          # every comparison but RO_FRONTIER passed


def d2_rec(pair=0.02, twin_pair=0.03, refs=0.03, g1=True, grid_valid=True, nolook=0.01, status='FITTED', revision=0,
           family='IA', early=True):
    """A complete invented family reading: CAND spends more than TWIN (+0.003 nats); SBC_1 / SPD_1 / RO_1 are cheaper than CAND
    and SBC_2 / SPD_2 / RO_2 more expensive (the brackets); the RO path is SUBSTITUTED and the operation is not release-only."""
    ev = NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.011})
    grid = {i: NT.grid_entry(pair) for i in POINTS}
    grid['TWIN'] = NT.grid_entry(twin_pair)
    grid.update({r: NT.grid_entry(refs) for r in NR.BENCHMARK})
    costs = {'TWIN': NT.costs(0.003), 'SBC_1': NT.costs(0.001), 'SBC_2': NT.costs(-0.001), 'SPD_1': NT.costs(0.0015),
             'SPD_2': NT.costs(-0.0005), 'RO_1': NT.costs(0.002), 'RO_2': NT.costs(-0.0008)}
    return {'family': family, 'card': f'{family}_x', 'revision': revision, 'card_committed_before_any_model_select_outcome': early,
            'grid_valid': grid_valid, 'twin': {'config_id': 'TWIN', 'evals': NT.evals()},
            'cand': {'status': status, 'config_id': 'CAND', 'evals': ev, 'strength': 2.0},
            'strengths': [{'config_id': 'CAND', 'strength': 2.0, 'evals': ev}],
            'grid': grid, 'paths': {'SBC': list(SBC), 'SPEND': list(SPEND), 'RO': list(RO)}, 'alias_of': {}, 'costs_vs': costs,
            'ro': {'kind': 'SUBSTITUTED', 'operation_release_only': False},
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
    assert r['schema'] == 'IPR1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(R.RULES_PATH.read_text())
    for path, val in ((('reader_grid', 'seed_offset'), 800), (('reader_grid', 'forbidden_offsets'), [300, 400, 500, 600, 700]),
                      (('reader_grid', 'bootstrap_seed_base'), 2026102200), (('reader_grid', 'split_tag'), 'DDR1_devgrid_R{r}'),
                      (('er_explanation', 'min_leaf', 'ET25'), 5), (('er_explanation', 'pool_people'), 4000), (('z',), 1.96),
                      (('schema',), 'DDR1-rules-v1'),
                      (('decisions', 'D2', 'met_labels'), ['IMPROVEMENT_BEYOND_MATCHED_CONTROLS']),
                      (('decisions', 'D2', 'comparisons'), ['TWIN', 'SBC_FRONTIER', 'SPEND_FRONTIER']),
                      (('decisions', 'D2', 'paths'), [['SBC_FRONTIER', 'SBC'], ['SPEND_FRONTIER', 'SPEND']]),
                      (('decisions', 'D2', 'ro_kinds'), ['SUBSTITUTED']),
                      (('decisions', 'D2', 'ro_scope', 'GENERIC_ZTILT'), 'beyond release-only information'),
                      (('decisions', 'D2', 'grid_fineness', 'fraction'), 0.5),
                      (('registration_status', 'registered_families'), ['IA', 'IB', 'FS_X']),
                      (('registration_status', 'family_pattern'), '.*'),
                      (('decisions', 'D2R', 'labels'), list(R.D2R_LABELS[:-1])),
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
    R.decisions(d2_rec())
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.forbidden_seeds() == before
    assert R.forbidden_seeds() > NR.forbidden_seeds() and len(R.forbidden_seeds()) == 120
    assert R.registered_seed(1, 3) == NR.MODEL_SEED_BASE + 913 and R.registered_boot_seed(1, 'R4') == 2026102403
    assert R.registered_seed(0, 0) == 2026102105 and R.registered_boot_seed(0, 'R1') == 2026102300
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
    eight = {k: x for k, x in seeds_all(800).items() if k[-1] in '01'}
    assert NR.grid_validity(eight, checks(), ['CAND'], DI)['label'] == 'READER_SEEDS_EFFECTIVE'   # nrm alone accepts DDR1's +800
    shifted = dict(reg_seeds())
    shifted['S1k1'] = NR.MODEL_SEED_BASE + 900 + 19                                            # unregistered, not forbidden
    g = R.grid_validity(shifted, checks(), ['CAND'], DI, boots())
    assert not g['effective'] and any('differ from the registered' in e for e in g['errors'])
    assert not any('reuse earlier' in e for e in g['errors'])
    assert not R.grid_validity({'S0k0': reg_seeds()['S0k0']}, checks(), ['CAND'], DI, boots())['effective']
    dup = dict(reg_seeds())
    dup['S1k1'] = dup['S1k0']
    assert not R.grid_validity(dup, checks(), ['CAND'], DI, boots())['effective']
    weird = dict(reg_seeds())
    weird['X'] = NR.MODEL_SEED_BASE + 999
    assert not R.grid_validity(weird, checks(), ['CAND'], DI, boots())['effective']
    g = R.grid_validity(reg_seeds(), checks(), ['CAND'], DI)                                  # bootstrap seeds not recorded
    assert not g['effective'] and any('bootstrap seeds not recorded' in e for e in g['errors'])
    b = boots()
    b['S1R2'] += 1
    assert any('S1R2' in e for e in R.grid_validity(reg_seeds(), checks(), ['CAND'], DI, b)['errors'])
    b = {f'S{v}{r}': 2026102200 + 100 * v + R.REP_INDEX[r] for v in NR.VERSIONS for r in ('R1', 'R2')}   # DDR1's bootstrap base
    assert not R.grid_validity(reg_seeds(), checks(), ['CAND'], DI, b)['effective']
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
    mlp = with_eq(checks(), {('CAND', 'income', 0, 'R1', 'MLP64')})
    g = R.grid_validity(reg_seeds(), mlp, ['CAND', 'TWIN'], DI, boots())
    assert not g['effective'] and any('on the minimum set' in e for e in g['errors'])
    assert R.grid_validity(reg_seeds(), mlp, ['TWIN'], DI, boots())['effective']
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


# ---------------------------------------------------------------- (3) brackets, fineness and D1
@test
def t_brackets_equal_nrm_rule():
    r = d2_rec()
    b = R.brackets(r['costs_vs'], r['paths'])
    assert b['SBC']['bracket'] == 'SBC_2' and b['SPEND']['bracket'] == 'SPD_2' and b['RO']['bracket'] == 'RO_2'
    assert b['RO']['path'] == ['TWIN', 'RO_1', 'RO_2']
    for key, ids in (('SBC', SBC), ('SPEND', SPEND), ('RO', RO)):
        fr = NR.frontier_comparison({'id': 'CAND', 'costs_vs': r['costs_vs'], 'grid_vs': r['grid']},
                                    [{'id': i} for i in R.control_path(ids)])
        assert fr['bracketing_control'] == b[key]['bracket']
    cheap = copy.deepcopy(r['costs_vs'])
    cheap['RO_2'] = NT.costs(0.0004)
    assert R.brackets(cheap, r['paths'])['RO']['status'] == 'NO_BRACKET'
    tie = copy.deepcopy(r['costs_vs'])
    tie['SBC_1'] = NT.costs(-0.001)                                                             # tie: path order kept
    assert R.brackets(tie, r['paths'])['SBC']['bracket'] == 'SBC_1'
    miss = copy.deepcopy(r['costs_vs'])
    del miss['TWIN']
    assert all(R.brackets(miss, r['paths'])[k]['status'] == 'INCOMPLETE' for k in ('SBC', 'SPEND', 'RO'))
    assert R.brackets(r['costs_vs'], {'SBC': None})['RO']['status'] == 'PATH_NOT_GIVEN'
    assert R.control_path(['SBC_1', 'SBC_2'], {'SBC_1': 'TWIN'}) == ['TWIN', 'SBC_2'] and R.control_path(None) is None
    assert R.control_path([], {}) == ['TWIN'] and R.control_path(['A', 'B'], {'B': 'A'}) == ['TWIN', 'A']


def sp(cand=0.010, lower=0.008, upper=0.0135, se=0.001, mx=0.03):
    return {v: {'cand': cand, 'lower': lower, 'upper': upper, 'upper_se': se, 'max': mx} for v in NR.VERSIONS}


@test
def t_grid_fineness():
    g = R.grid_fineness({'SBC': sp(), 'SPEND': sp(), 'RO': sp()})
    assert all(g['per_path'][k] == {0: 'FINE', 1: 'FINE'} for k in ('SBC', 'SPEND', 'RO')) and g['flags'] == [], g
    # the dual-design release-only grid: 0.90x below, 2.08x above
    g = R.grid_fineness({'SBC': sp(), 'SPEND': sp(), 'RO': sp(lower=0.009, upper=0.0208)})
    assert g['per_path']['RO'][0] == 'COARSE_ABOVE' and g['flags'] == ['CONTROL_GRID_COARSE_RO']
    assert R.grid_fineness({'SBC': sp(lower=0.007), 'SPEND': sp(), 'RO': sp()})['per_path']['SBC'][1] == 'COARSE_BELOW'
    assert R.grid_fineness({'SBC': sp(lower=0.0074), 'SPEND': sp(), 'RO': sp()})['per_path']['SBC'][1] == 'COARSE_BELOW'
    assert R.grid_fineness({'SBC': sp(lower=0.0076), 'SPEND': sp(), 'RO': sp()})['per_path']['SBC'][1] == 'FINE'
    assert R.grid_fineness({'SBC': sp(lower=0.0, upper=0.03), 'SPEND': sp(), 'RO': sp()})['per_path']['SBC'][0] == 'COARSE_BOTH'
    assert R.grid_fineness({'SBC': sp(upper=0.0146), 'SPEND': sp(), 'RO': sp()})['per_path']['SBC'][0] == 'COARSE_ABOVE'
    g = R.grid_fineness({'SBC': sp(upper=0.011, mx=0.011), 'SPEND': sp(), 'RO': sp()})            # path never reaches s + 2 SE
    assert g['per_path']['SBC'][0] == 'CONTROL_PATH_MAY_NOT_BRACKET' and 'CONTROL_PATH_MAY_NOT_BRACKET_SBC' in g['flags']
    g = R.grid_fineness({'SBC': sp(upper=0.0115), 'SPEND': sp(lower=0.0101), 'RO': sp()})         # saved-point rule broken
    assert g['per_path']['SBC'][0] == g['per_path']['SPEND'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'
    g = R.grid_fineness({'SBC': sp(cand=-0.001), 'SPEND': sp(cand=0.0), 'RO': sp()})              # CAND not above TWIN
    assert g['per_path']['SBC'][0] == g['per_path']['SPEND'][1] == 'NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN' and g['flags'] == []
    g = R.grid_fineness({'SBC': None, 'SPEND': {0: sp()[0]}, 'RO': sp(se=-1.0)})
    assert g['per_path']['SBC'] == 'NOT_GIVEN' and g['per_path']['SPEND'][1] == 'INCOMPLETE' and g['per_path']['RO'][0] == 'INCOMPLETE'
    assert 'CONTROL_GRID_FINENESS_INCOMPLETE_SPEND' in g['flags']
    r = d2_rec()                                                    # reported beside D2, never gating
    r['fineness'] = R.grid_fineness({'SBC': sp(), 'SPEND': sp(), 'RO': sp(upper=0.0208)})
    d = R.d2(r)
    assert d['met'] and 'CONTROL_GRID_COARSE_RO' in d['flags']


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
    assert d['comparisons']['RO_FRONTIER']['point'] == 'RO_2' and d['comparisons']['RO_FRONTIER']['path'] == ['TWIN', 'RO_1', 'RO_2']
    assert set(d['targeting']) == {'TWIN', 'SBC_2', 'SPD_2', 'RO_2'}
    assert d['flags'] == ['RO_SUBSTITUTED'] and d['comparisons']['RO_FRONTIER']['recipient_gate_vs_bracket'] is True
    assert d['ro'] == {'kind': 'SUBSTITUTED', 'operation_release_only': False, 'scope': R.RO_SCOPE['SUBSTITUTED']}


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
    assert 'RECIPIENT_GATE_FAILED_VS_SBC_2' in d['flags'] and d['label'] == 'PARTIAL_TWIN+SPEND_FRONTIER+RO_FRONTIER_ONLY'
    r = d2_rec()
    r['grid']['SPD_2']['occupation'] = NT.grid_entry(0.0, occ=0.0105)['occupation']             # just above the allowance
    assert R.d2(r)['label'] == 'PARTIAL_TWIN+SBC_FRONTIER+RO_FRONTIER_ONLY'
    r = d2_rec()
    r['grid']['SPD_2']['occupation'] = NT.grid_entry(0.0, occ=0.010)['occupation']              # inclusive allowance passes
    assert R.d2(r)['met']
    r = d2_rec()
    r['grid']['RO_2']['income'] = NT.grid_entry(0.0, inc=0.02)['income']                        # ... and at the RO bracket
    d = R.d2(r)
    assert 'RECIPIENT_GATE_FAILED_VS_RO_2' in d['flags'] and d['label'] == ALL3
    r = d2_rec()
    del r['grid']['SPD_2']['income']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['grid']['RO_2']['occupation']
    assert R.d2(r)['label'] == 'INCOMPLETE'


@test
def t_d2_cheaper_control_domination():
    r = d2_rec()
    r['grid']['SBC_1'] = NT.grid_entry(-0.02)                       # SBC_1 is cheaper AND resolved less recoverable than CAND
    d = R.d2(r)
    c = d['comparisons']['SBC_FRONTIER']
    assert c['label'] == 'DOMINATES_CONTROL_FRONTIER' and not c['pass'] and c['dominated_by_cheaper'] == ['SBC_1']
    assert c['reason'] == 'DOMINATED_BY_CHEAPER_CONTROL' and 'DOMINATED_BY_CHEAPER_SBC_1' in d['flags']
    assert d['label'] == 'PARTIAL_TWIN+SPEND_FRONTIER+RO_FRONTIER_ONLY'
    r['grid']['SBC_1'] = NT.grid_entry(-0.005)                      # point-only lower recovery: not a resolved domination
    assert R.d2(r)['met']
    r['grid']['SBC_1'] = NT.grid_entry(lambda c: -0.02 if c.startswith('S0') else 0.001)          # a version disagrees
    assert R.d2(r)['met']
    r['grid']['SBC_1'] = copy.deepcopy(NT.ALIAS)                    # an exact alias of CAND is not 'less recoverable'
    assert R.d2(r)['met']
    del r['grid']['SBC_1']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['grid']['RO_1'] = NT.grid_entry(-0.02)                        # the cheaper release-only point leaks less than CAND
    d = R.d2(r)
    assert 'DOMINATED_BY_CHEAPER_RO_1' in d['flags'] and d['label'] == ALL3


@test
def t_d2_brackets_and_secondary_spending():
    r = d2_rec()
    r['costs_vs']['SBC_2'] = NT.costs(0.0004)                       # every SBC point cheaper than CAND: no bracket
    d = R.d2(r)
    c = d['comparisons']['SBC_FRONTIER']
    assert c['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT' and c['point'] is None and 'NO_BRACKET_SBC' in d['flags']
    assert d['label'] == 'PARTIAL_TWIN+SPEND_FRONTIER+RO_FRONTIER_ONLY' and set(d['targeting']) == {'TWIN', 'SPD_2', 'RO_2'}
    r = d2_rec()
    r['costs_vs']['RO_2'] = NT.costs(0.0004)                        # every RO point cheaper than CAND
    d = R.d2(r)
    assert 'NO_BRACKET_RO' in d['flags'] and d['label'] == ALL3
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
    del r['costs_vs']['RO_2'][1]['income|nll']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['costs_vs']['TWIN']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['costs_vs']['TWIN'] = NT.costs(-0.0001)                       # CAND cheaper than TWIN: TWIN is the bracket of every path
    d = R.d2(r)
    assert all(d['comparisons'][n]['point'] == 'TWIN' for n in ('SBC_FRONTIER', 'SPEND_FRONTIER', 'RO_FRONTIER'))
    assert d['met'] and set(d['targeting']) == {'TWIN'}


@test
def t_d2_targeting_partials_and_not_separated():
    r = d2_rec()
    r['nolookup']['SPD_2'] = {'cells': NT.reg(lambda c: -0.01 if c.startswith('S1') else 0.01)}
    d = R.d2(r)
    assert d['label'] == 'READER_TARGETED' and 'READER_TARGETED_SPEND_FRONTIER' in d['flags'] and not d['met']
    r = d2_rec()
    r['nolookup']['RO_2'] = {'cells': NT.reg(lambda c: -0.01)}
    d = R.d2(r)
    assert d['label'] == 'READER_TARGETED' and 'READER_TARGETED_RO_FRONTIER' in d['flags']
    r = d2_rec()
    r['nolookup']['TWIN'] = {'cells': NT.reg(lambda c: 0.0)}        # point benefit must be > 0, not >= 0
    assert R.d2(r)['label'] == 'READER_TARGETED'
    r = d2_rec()
    del r['nolookup']['SBC_2']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['nolookup']['RO_2']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['nolookup']['SBC_1'] = {'cells': NT.reg(lambda c: -0.05)}     # a cheaper (non-bracketing) point is not a targeting point
    assert R.d2(r)['met']
    assert R.d2(d2_rec(pair=-0.01, twin_pair=-0.01))['label'] == 'NOT_SEPARATED'
    assert R.d2(d2_rec(pair=0.0))['label'] == 'PARTIAL_TWIN_ONLY'
    r = d2_rec(twin_pair=0.0)
    assert R.d2(r)['label'] == 'PARTIAL_SBC_FRONTIER+SPEND_FRONTIER+RO_FRONTIER_ONLY'
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
    for key, name in (('SPEND', 'SPEND_FRONTIER'), ('RO', 'RO_FRONTIER')):
        r = d2_rec()
        r['paths'][key] = []
        d = R.d2(r)
        assert f'{key}_AT_IDENTITY' in d['flags'] and d['comparisons'][name]['path'] == ['TWIN'] and name not in d['passed']
        assert d['comparisons'][name]['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    r = d2_rec()
    r['alias_of'] = {'SBC_2': 'SBC_1'}                              # alias counted once: the bracket disappears
    d = R.d2(r)
    assert d['comparisons']['SBC_FRONTIER']['path'] == ['TWIN', 'SBC_1'] and 'SBC_FRONTIER' not in d['passed']
    r = d2_rec()
    r['alias_of'] = {'RO_1': 'TWIN', 'RO_2': 'TWIN'}                # every RO point an alias of TWIN
    assert 'RO_AT_IDENTITY' in R.d2(r)['flags']
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
    for key in ('SBC', 'SPEND', 'RO'):
        r = d2_rec()
        r['paths'][key] = None
        r['controls_not_run'] = {key: True}
        assert R.d2(r)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN' and R.d2(r)['not_run'] == [key]
        r['controls_not_run'] = {}
        assert R.d2(r)['label'] == 'INCOMPLETE'                     # a missing path is never 'not run' unless recorded
    r = d2_rec()
    del r['grid']['TWIN']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['cand']['evals'] = {0: NT.util(0)}
    assert R.d2(r)['label'] == 'INCOMPLETE'


@test
def t_d2_release_only_path_declaration():
    r = d2_rec()
    r['ro'] = {'kind': 'GENERIC_ZTILT', 'operation_release_only': False}   # generic tilt for an operation that is not release-only
    d = R.d2(r)
    assert d['label'] == R.D2_MET and 'RO_GENERIC_ZTILT' in d['flags'] and d['ro']['scope'] == R.RO_SCOPE['GENERIC_ZTILT']
    r['ro'] = {'kind': 'GENERIC_ZTILT', 'operation_release_only': True}    # the operation is itself release-only
    out = R.decisions(r)
    assert out['D2']['label'] == R.D2_MET_RELEASE_ONLY and out['D2']['met'] and out['outcome_class']['label'] == 'CANDIDATE'
    assert 'RELEASE_ONLY_OPERATION' in out['outcome_class']['flags']
    assert out['outcome_class']['ro_scope'] == R.RO_SCOPE['RELEASE_ONLY_OPERATION']
    s = R.campaign_summary([out])
    assert s['d2_looks'] == 1 and len(s['d2_met']) == 1 and s['registered']['IA']['D2'] == R.D2_MET_RELEASE_ONLY
    r2 = copy.deepcopy(r)
    r2['grid']['RO_2'] = NT.grid_entry(0.005)                       # a release-only map not better than the generic tilt
    assert R.d2(r2)['label'] == ALL3 and 'RELEASE_ONLY_OPERATION' in R.d2(r2)['flags']
    r['ro'] = {'kind': 'SUBSTITUTED', 'operation_release_only': True}      # misdeclared: the substitution is the operation
    assert R.d2(r)['label'] == 'INCOMPLETE' and 'generic ZTILT' in R.d2(r)['error']
    for bad in (None, {}, {'kind': 'ZTILT', 'operation_release_only': False}, {'kind': 'SUBSTITUTED',
                                                                                 'operation_release_only': 'no'}):
        r['ro'] = bad
        assert R.d2(r)['label'] == 'INCOMPLETE', bad
    r = d2_rec()                                                    # the RO declaration is not needed when RO is not run
    r['paths']['RO'] = None
    r['controls_not_run'] = {'RO': True}
    r['ro'] = None
    out = R.decisions(r)
    assert out['D2']['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN' and out['outcome_class']['label'] == 'PROTECTION_NOT_ASSESSED'
    r = d2_rec()                                                    # RO fails at its bracket: named by what passed
    r['grid']['RO_2'] = NT.grid_entry(0.005)
    out = R.decisions(r)
    assert out['D2']['label'] == ALL3 and out['outcome_class']['label'] == 'TASK_ONLY_FEASIBLE_NO_PROTECTION'
    assert out['D2']['comparisons']['RO_FRONTIER']['reason'] == 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'
    assert 'D3_WITHOUT_D2' in out['outcome_class']['flags']


def sex_blind_rec(ro=False):
    """A SEX-blind, release-only operation (reads no SEX; a function of the twin's release and a SEX-independent key): no SBC
    path; with ro=True a generic ZTILT path is given (reported beside), else no RO path."""
    r = d2_rec(family='IB')
    r['operation_sex_blind'] = True
    r['paths']['SBC'] = None
    if ro:
        r['ro'] = {'kind': 'GENERIC_ZTILT', 'operation_release_only': True}
    else:
        r['paths']['RO'] = None
        r['ro'] = None
    return r


@test
def t_d2_sex_blind_operation():
    r = sex_blind_rec()
    out = R.decisions(r)
    d = out['D2']
    assert d['label'] == R.D2_MET_SEX_BLIND and d['met'] and d['gating'] == ['TWIN', 'SPEND_FRONTIER'], d
    assert d['passed'] == ['TWIN', 'SPEND_FRONTIER'] and set(d['targeting']) == {'TWIN', 'SPD_2'}
    assert {'SEX_BLIND_OPERATION', 'SBC_DEGENERATE_BY_CONSTRUCTION', 'RO_DEGENERATE_BY_CONSTRUCTION'} <= set(d['flags'])
    assert d['comparisons']['SBC_FRONTIER']['label'] == d['comparisons']['RO_FRONTIER']['label'] == 'DEGENERATE_BY_CONSTRUCTION'
    assert d['scope'] == R.RO_SCOPE['SEX_BLIND_OPERATION'] and out['outcome_class']['label'] == 'CANDIDATE'
    assert out['outcome_class']['ro_scope'] == R.RO_SCOPE['SEX_BLIND_OPERATION']
    x = sex_blind_rec()
    x['grid']['SPD_2'] = NT.grid_entry(0.005)                       # not resolved below the SPEND bracket
    assert R.d2(x)['label'] == 'PARTIAL_TWIN_ONLY'
    x = sex_blind_rec()
    x['nolookup']['SPD_2'] = {'cells': NT.reg(lambda c: -0.01)}
    assert R.d2(x)['label'] == 'READER_TARGETED'
    assert R.d2(sex_blind_rec() | {'grid': {**d2_rec()['grid'], 'TWIN': NT.grid_entry(0.0)}})['label'] == \
        'PARTIAL_SPEND_FRONTIER_ONLY'
    x = sex_blind_rec()
    x['paths']['SPEND'] = None
    x['controls_not_run'] = {'SPEND': True}
    assert R.d2(x)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    x['controls_not_run'] = {'SBC': True, 'RO': True}               # SBC / RO of a SEX-blind operation are never 'not run'
    assert R.d2(x)['label'] == 'INCOMPLETE'                         # (SPEND missing and not recorded as not run)
    x = sex_blind_rec()
    x['controls_not_run'] = {'SBC': True, 'RO': True}
    assert R.d2(x)['label'] == R.D2_MET_SEX_BLIND
    # a generic ZTILT path built for it is computed and reported beside, never gating
    x = sex_blind_rec(ro=True)
    x['grid']['RO_2'] = NT.grid_entry(0.005)                        # it would fail if it gated
    d = R.d2(x)
    assert d['label'] == R.D2_MET_SEX_BLIND and d['comparisons']['RO_FRONTIER']['gating'] is False
    assert d['comparisons']['RO_FRONTIER']['passed_if_it_were_gating'] is False and 'RO_GENERIC_ZTILT_REPORTED_BESIDE' in d['flags']
    assert 'RO_2' not in d['targeting']
    x = sex_blind_rec(ro=True)
    del x['costs_vs']['RO_2']                                       # an incomplete beside path never makes D2 incomplete
    d = R.d2(x)
    assert d['label'] == R.D2_MET_SEX_BLIND and d['comparisons']['RO_FRONTIER']['label'] == 'INCOMPLETE'
    x = sex_blind_rec(ro=True)
    x['ro'] = {'kind': 'SUBSTITUTED', 'operation_release_only': False}   # a substituted RO path would be the operation itself
    assert R.d2(x)['label'] == 'INCOMPLETE'
    x = sex_blind_rec()
    x['paths']['SBC'] = list(SBC)                                   # an SBC path for a SEX-blind operation is refused
    assert R.d2(x)['label'] == 'INCOMPLETE'
    x = sex_blind_rec()
    x['operation_sex_blind'] = 'yes'
    assert R.d2(x)['label'] == 'INCOMPLETE'
    s = R.campaign_summary([out])
    assert s['d2_looks'] == 1 and len(s['d2_met']) == 1
    r = d2_rec()
    r['operation_sex_blind'] = False
    assert R.d2(r)['label'] == R.D2_MET


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
    assert out['D5']['label'] == 'NOT_APPLICABLE_NOT_JOINT' and out['D6']['label'] == 'NOT_CLAIMED' and out['registered']
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
    assert all(out[k]['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE' for k in ('D3', 'D4', 'D5'))
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
    assert out['D5']['label'] == 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH'
    r = d2_rec()
    r['paths']['SPEND'] = None
    r['controls_not_run'] = {'SPEND': True}
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'PROTECTION_NOT_ASSESSED' and out['D3']['met'] and \
        'D3_WITHOUT_D2' in out['outcome_class']['flags']
    assert R.decisions(d2_rec(grid_valid=None))['outcome_class']['label'] == 'INCOMPLETE'
    out = R.decisions(d2_rec(grid_valid=False))
    assert out['D3']['label'] == 'NOT_COMPUTED_GRID_INVALID' and out['D5']['label'] == 'NOT_COMPUTED_GRID_INVALID'
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


def joint_rec():
    r = d2_rec(family='IB')
    r['joint'] = {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': 'SEP',
                  'income_first': 'IF', 'occupation_first': 'OF', 'costs_vs': {n: NT.costs(-0.001) for n in ('SEP', 'IF', 'OF')}}
    for n in ('SEP', 'IF', 'OF'):
        r['grid'][n] = NT.grid_entry(0.02)
    return r


@test
def t_d5_nr_g6():
    r = joint_rec()
    out = R.decisions(r)
    assert out['D5']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED' and out['outcome_class']['label'] == 'CANDIDATE'
    cl = NR.claims({'evals': r['cand']['evals'], 'grid': r['grid'], 'grid_valid': True, 'matched': {}, 'joint': r['joint']})
    assert cl['G6']['label'] == out['D5']['label_unprefixed'] == 'JOINT_CONTRIBUTION_SUPPORTED'   # NR G6 unchanged
    cases = []
    x = joint_rec()
    del x['joint']['occupation_first']                              # a sequential order not built (not named)
    cases.append((x, 'NOT_ASSESSED_MISSING_CONTROLS'))
    x = joint_rec()
    del x['grid']['IF']                                             # named but its grid entry missing: INCOMPLETE (NR: missing)
    cases.append((x, 'INCOMPLETE'))
    x = joint_rec()
    del x['grid']['SEP']['occupation']                              # a recipient view missing
    cases.append((x, 'INCOMPLETE'))
    x = joint_rec()
    del x['joint']['costs_vs']['OF']                                # a paired cost missing: INCOMPLETE (NR: NOT_SUPPORTED)
    cases.append((x, 'INCOMPLETE'))
    x = joint_rec()
    x['joint']['costs_vs']['IF'] = NT.costs(0.004)                  # CAND resolved more spending than a sequential order
    cases.append((x, 'NOT_SUPPORTED'))
    x = joint_rec()
    x['grid']['OF'] = NT.grid_entry(0.005)                          # not resolved below the occupation-first order
    cases.append((x, 'NOT_SUPPORTED'))
    x = joint_rec()
    x['grid']['SEP']['income'] = NT.grid_entry(0.0, inc=0.02)['income']   # recipient gate vs the separate design
    cases.append((x, 'NOT_SUPPORTED'))
    x = joint_rec()
    x['joint']['outputs_changed'] = ['occupation']
    cases.append((x, 'NOT_APPLICABLE_ONE_OUTPUT_CHANGED'))
    x = joint_rec()
    x['joint']['jointly_designed'] = False
    cases.append((x, 'NOT_APPLICABLE_NOT_JOINT'))
    x = joint_rec()
    x['joint'] = None
    cases.append((x, 'NOT_APPLICABLE_NOT_JOINT'))
    x = joint_rec()
    x['cand']['evals'] = NT.evals(occ={('U0', 'nll_excess'): 0.011})
    cases.append((x, 'NOT_ASSESSED_TASK_INFEASIBLE'))               # gated like D3 / D4
    x = joint_rec()
    x['grid_valid'] = None
    cases.append((x, 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'))
    x = joint_rec()
    x['cand']['every_nonzero_strength_alias_of_twin'] = True
    cases.append((x, 'NOT_ASSESSED_EXACT_ALIAS'))
    for rec, want in cases:
        assert R.decisions(rec)['D5']['label'] == want, (want, R.decisions(rec)['D5'])
    x = joint_rec()
    x['joint']['costs_vs']['IF'] = NT.costs(0.004)
    assert R.decisions(x)['outcome_class']['label'] == 'CANDIDATE'  # D5 never changes the class
    x = joint_rec()
    x['revision'], x['card_committed_before_any_model_select_outcome'] = 1, False
    assert R.decisions(x)['D5']['label'] == 'ADAPTIVE_R1__JOINT_CONTRIBUTION_SUPPORTED'
    r = d2_rec()
    r['novelty'] = 'NOVEL_PER_SEPARATE_REVIEW'
    assert R.decisions(r)['D6']['label'] == 'NOVEL_PER_SEPARATE_REVIEW'
    r['novelty'] = 'NOVEL_BECAUSE_IT_WORKS'
    assert R.decisions(r)['D6']['label'] == 'NOT_CLAIMED'


@test
def t_d5_collapsed_sequential_orders():
    # a separately designed comparator (independent per-recipient keys) is built; both sequential orders equal the separate
    # design by construction (card): the reduced test reads NR G6 with the separate comparator standing in for them
    r = joint_rec()
    for k in ('income_first', 'occupation_first'):
        del r['joint'][k]
    del r['grid']['IF'], r['grid']['OF']
    assert R.decisions(r)['D5']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS'          # undeclared: missing controls
    r['joint']['collapsed'] = {'income_first': 'SEPARATE', 'occupation_first': 'SEPARATE'}
    out = R.decisions(r)
    assert out['D5']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED' and out['D5']['reduced_test']
    assert {'D5_REDUCED_INCOME_FIRST_EQUALS_SEPARATE', 'D5_REDUCED_OCCUPATION_FIRST_EQUALS_SEPARATE'} <= \
        set(out['outcome_class']['flags']) and 'reduced test' in out['D5']['statement']
    x = copy.deepcopy(r)
    x['grid']['SEP'] = NT.grid_entry(0.005)                         # not resolved below the separate design
    assert R.decisions(x)['D5']['label'] == 'NOT_SUPPORTED'
    x = copy.deepcopy(r)
    x['joint']['income_first'] = 'SEP'                              # naming the separate comparator itself is consistent
    assert R.decisions(x)['D5']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED'
    x['joint']['income_first'] = 'OTHER'                            # another comparator for a collapsed order: inconsistent
    assert R.decisions(x)['D5']['label'] == 'INCOMPLETE'
    x = copy.deepcopy(r)                                            # one order collapses, the other is built
    x['joint']['collapsed'] = {'income_first': 'SEPARATE'}
    x['joint']['occupation_first'] = 'OF'
    x['grid']['OF'] = NT.grid_entry(0.02)
    x['joint']['costs_vs']['OF'] = NT.costs(-0.001)
    assert R.decisions(x)['D5']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED'
    x['grid']['OF'] = NT.grid_entry(0.0)
    assert R.decisions(x)['D5']['label'] == 'NOT_SUPPORTED'
    x = copy.deepcopy(r)                                            # an order equal to the joint design itself
    x['joint']['collapsed'] = {'income_first': 'SEPARATE', 'occupation_first': 'JOINT'}
    out = R.decisions(x)
    assert out['D5']['label'] == 'NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL' and out['outcome_class']['label'] == 'CANDIDATE'
    for bad in ({}, {'income_first': 'BOTH'}, {'separate': 'SEPARATE'}, ['income_first']):
        x = copy.deepcopy(r)
        x['joint']['collapsed'] = bad
        assert R.decisions(x)['D5']['label'] == 'INCOMPLETE', bad
    x = copy.deepcopy(r)                                            # the reduced test still needs the separate comparator
    del x['joint']['separate']
    assert R.decisions(x)['D5']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS'
    x = copy.deepcopy(r)                                            # not declared jointly designed: NR's label first
    x['joint']['jointly_designed'] = False
    assert R.decisions(x)['D5']['label'] == 'NOT_APPLICABLE_NOT_JOINT'
    x = sex_blind_rec()                                             # the coordinator's case: SEX-blind, release-only, keyed
    x['joint'] = copy.deepcopy(r['joint'])
    x['grid']['SEP'] = NT.grid_entry(0.02)
    out = R.decisions(x)
    assert out['D2']['label'] == R.D2_MET_SEX_BLIND and out['D5']['label'] == 'JOINT_CONTRIBUTION_SUPPORTED'


@test
def t_flags_by_version_and_strength_components():
    r = d2_rec()
    r['cand']['strength'] = {0: 1.0, 1: 0.75}
    assert 'CAND_STRENGTH_DIFFERS_BY_VERSION' in R.decisions(r)['outcome_class']['flags']
    for same in (2.0, {0: [0.35, 0.65], '1': [0.35, 0.65]}, [0.35, 0.65]):
        r['cand']['strength'] = same
        assert 'CAND_STRENGTH_DIFFERS_BY_VERSION' not in R.decisions(r)['outcome_class']['flags'], same
    r['cand']['strength'] = {0: [0.35, 0.65], 1: [0.35, 0.5]}
    assert 'CAND_STRENGTH_DIFFERS_BY_VERSION' in R.decisions(r)['outcome_class']['flags']
    r = d2_rec()
    r['sbc_construction'] = {0: 'PERMUTED', 1: 'PSEUDO2'}           # reported; the reading is unchanged
    d = R.d2(r)
    assert d['met'] and 'SBC_CONSTRUCTION_DIFFERS_BY_VERSION' in d['flags']
    r['sbc_construction'] = {0: 'PERMUTED', 1: 'PERMUTED'}
    assert 'SBC_CONSTRUCTION_DIFFERS_BY_VERSION' not in R.d2(r)['flags']
    # component-vector strengths are read (DDR1 read [0.35, 0.65] as 'not nonzero': its deviation 8)
    assert R._nonzero_strength([0.35, 0.65]) and R._nonzero_strength({0: [0.35, 0.65], 1: [0.0, 0.65]})
    assert R._nonzero_strength([0.0, 0.5]) and not R._nonzero_strength([0.0, 0.0]) and not R._nonzero_strength([-0.1, 0.5])
    assert not R._nonzero_strength([]) and not R._nonzero_strength({0: 1.0}) and not R._nonzero_strength('0.5')
    assert R._strength_value([0.35, 0.65]) == 2.0 and R._strength_value({0: 1.0, 1: 0.75}) == 1.75
    assert R._strength_value([-0.1, 0.5]) is None
    r = d2_rec()
    r['strengths'] = [{'config_id': 'CAND', 'strength': [0.35, 0.65], 'evals': NT.evals()}]
    d = R.d1(r)
    assert d['fitted_strengths'] == {'CAND': 'POINT_ELIGIBLE'} and d['ignored_zero_strength_entries'] == []


@test
def t_families_refused_and_registered():
    for fam in ('DA', 'CTL', 'FS_', 'fs_x', None, 'IA2', 'FS_A-B'):
        for fn in (R.decisions, R.adaptive_tag):
            try:
                fn(d2_rec(family=fam))
            except ValueError:
                continue
            raise AssertionError(f'family {fam!r} accepted by {fn.__name__}')
    assert R.decisions(d2_rec(family='IB'))['registered']
    out = R.decisions(d2_rec(family='FS_RANK1'))
    assert not out['registered'] and out['adaptive_tag'] == 'REVISED_FS'


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
    cases = {(1, 'IA', True): 'REVISED_R1', (2, 'IB', False): 'ADAPTIVE_R2', (0, 'FS_X', False): 'ADAPTIVE_FS',
             (0, 'FS_X', True): 'REVISED_FS', (1, 'FS_X', False): 'ADAPTIVE_FS_R1', (0, 'IA', False): 'ADAPTIVE_R0_LATE_CARD',
             (0, 'IB', True): None}
    for (n, fam, early), want in cases.items():
        assert R.adaptive_tag({'revision': n, 'family': fam, 'card_committed_before_any_model_select_outcome': early}) == want
    assert R.adaptive_tag({'revision': 0, 'family': 'IA'}) == 'ADAPTIVE_R0_LATE_CARD'           # not recorded: not registered
    assert R.adaptive_tag({'revision': 0, 'family': 'IA', 'card_committed_before_any_model_select_outcome': 'yes'}) == \
        'ADAPTIVE_R0_LATE_CARD'


@test
def t_campaign_summary():
    ia = R.decisions(d2_rec(family='IA'))
    ib_rec = d2_rec(family='IB')
    ib_rec['grid']['RO_2'] = NT.grid_entry(0.0)                     # IB not separated from its release-only path
    ib = R.decisions(ib_rec)
    ia1 = R.decisions(d2_rec(family='IA', revision=1, early=False))
    inf = R.decisions(d2_rec(family='FS_Y', g1=False, early=False))
    s = R.campaign_summary([ia, ib, ia1, inf])
    assert set(s['registered']) == {'IA', 'IB'} and s['d2_looks'] == 3 and len(s['d2_met']) == 2 and s['flags'] == []
    assert s['statement'].startswith('D2 was met in 1 of 2 registered family readings and in 1 of 1 non-registered readings '
                                     '(3 D2 looks in total)')
    assert 'intersection-union conjunction of four' in s['statement']
    assert s['registered']['IB']['outcome_class'] == 'TASK_ONLY_FEASIBLE_NO_PROTECTION' and s['registered']['IB']['D2'] == ALL3
    assert [x['tag'] for x in s['non_registered']] == ['ADAPTIVE_R1', 'ADAPTIVE_FS']
    s = R.campaign_summary([ib, ia1])
    assert s['flags'] == ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS']
    try:
        R.campaign_summary([ia, R.decisions(d2_rec(family='IA'))])
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
    assert s['selected'] == 'P2' and s['config_id'] == 'P2' and s['flags'] == []
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
    s = R.strongest_nonzero([cfg('SPLIT', [0.35, 0.65], 0.024), cfg('ONE', 0.35, 0.016)])           # a component vector is read
    assert s['selected'] == 'SPLIT' and s['records'][0]['status'] == 'ELIGIBLE'
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
    rob = {c: rob_entry(0.02) for c in ('TWIN', 'SBC_2', 'SPD_2', 'RO_2', 'NEURAL_RAW', 'H_LJ')}
    base = {'grid_valid': True, 'strongest': strongest, 'refit': refit, 'rob': rob,
            'brackets': {'SBC': 'SBC_2', 'SPEND': 'SPD_2', 'RO': 'RO_2'}}
    d = R.d2r(base)
    assert d['label'] == 'ROBUSTNESS_CONFIRMED' and d['decisive'] == ['TWIN', 'SBC_2', 'SPD_2', 'RO_2'] and \
        set(d['per_comparator']) == {'TWIN', 'SBC_2', 'SPD_2', 'RO_2', 'NEURAL_RAW', 'H_LJ'} and d['release'] == 'CAND'
    assert d['per_comparator']['TWIN']['n_eff_robustness_cells'] == 8                            # constant cells once per block (8 blocks)
    r = copy.deepcopy(base)
    r['brackets'] = R.brackets(d2_rec()['costs_vs'], d2_rec()['paths'])                         # a brackets() record is accepted
    assert R.d2r(r)['decisive'] == ['TWIN', 'SBC_2', 'SPD_2', 'RO_2']
    r = copy.deepcopy(base)
    r['rob']['RO_2'] = rob_entry(0.02, rob=0.001)                   # not robust against the release-only bracket
    d = R.d2r(r)
    assert d['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d['per_comparator']['RO_2']['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    r = copy.deepcopy(base)
    del r['brackets']['RO']                                         # an unrecorded bracket never shrinks the decisive set
    assert R.d2r(r)['label'] == 'INCOMPLETE'
    r['brackets']['RO'] = {'bracket': None, 'status': 'INCOMPLETE'}
    assert R.d2r(r)['label'] == 'INCOMPLETE'
    r['brackets']['RO'] = None                                      # explicitly no bracket (NO_BRACKET)
    assert R.d2r(r)['label'] == 'ROBUSTNESS_CONFIRMED' and R.d2r(r)['decisive'] == ['TWIN', 'SBC_2', 'SPD_2']
    r = copy.deepcopy(base)
    r['strongest'] = {'label': 'SELECTED', 'config_id': 'CAND'}     # a stub with config_id only (DDR1 deviation 12 (c))
    assert R.d2r(r)['release'] == 'CAND'
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

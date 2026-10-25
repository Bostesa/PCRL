"""Invented-data tests of drf.rules (no real data, no label). Run: python3 -m drf.test_rules (prints JSON; exit 0 iff all pass).

Covers: the rule file and its refusal of altered constants (an absolute AUC margin included); nrm.rules left unchanged (never
monkeypatched); reader-seed validity of the DRF1 grid (+1100 registered, 300-1000 forbidden, bootstrap seeds, the extra-trees
support explanation, the few-valued explanation and its finite-pair extension above 320, seed pairs judged separately); the
genuinely effective refit; the CTRL bracket rule with origin TWIN (equal to nrm.rules.frontier_comparison); MODEL_FIT nested
fineness of the saved CTRL points; D1 (G1 gating; confidence conjunction and reserve beside); D2 (the CTRL frontier, OBJ_MI with the
single-point spend states, partials named by what passed, no bracket, cheaper-point domination, recipient gates, reader targeting,
gates and missing inputs); FC and TWIN reported apart (TWIN never sufficient); D3 / D4; the optimistic diagnostic (comparator-
based, shift-invariant, never gating); the failure stages 1-5; the outcome classes and the three separate outcomes; registration
(family DR, revisions 0-2 with prospective notes, late cards and MODEL_SELECT triggers adaptive); the campaign summary; D2R with
sign changes and robustness recipient failures carried into the headline. Helpers for invented utility records and grid entries
come from nrm.test_rules (unchanged)."""
from __future__ import annotations
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

from nrm import rules as NR
from nrm import test_rules as NT                 # invented-data helpers (util, evals, entry, reg, grid_entry)
from drf import rules as R

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- invented records
def costs(total, extra=None, se=0.0005):
    c = {'total|nll': {'value': total, 'se': se}}
    for m in NR.SECONDARY:
        c[m] = {'value': 0.0, 'se': 0.0005}
    c.update(extra or {})
    return {v: copy.deepcopy(c) for v in NR.VERSIONS}


CTRL = ['C1', 'C2']


def rec(ctrl_pair=0.02, obj_pair=0.02, start_pair=0.02, twin_pair=0.04, refs=0.03, nolook=0.01, obj_total=-0.0005,
        cand_ev=None, grid_valid=True, revision=0, early=True, note=None, opt=None):
    """A complete invented DR reading: DR spends +0.003 nats over TWIN, C1 is cheaper than DR, C2 the bracket (spends more);
    OBJ_MI spends obj_total relative to DR (cost(DR) - cost(OBJ_MI); negative = OBJ_MI spends more); CTRL bracket data
    MODEL_FIT_NESTED. Every benefit is per cell with SE 0.004 (se_agg 0.004 when constant)."""
    grid = {r: NT.grid_entry(refs) for r in NR.BENCHMARK}
    grid['SUCC'] = NT.grid_entry(start_pair)
    grid['TWIN'] = NT.grid_entry(twin_pair)
    grid.update({i: NT.grid_entry(ctrl_pair) for i in CTRL})
    grid['OBJ_MI'] = NT.grid_entry(obj_pair)
    cv = {'TWIN': costs(0.003), 'C1': costs(0.001), 'C2': costs(-0.001), 'OBJ_MI': costs(obj_total), 'SUCC': costs(0.002)}
    ev = cand_ev if cand_ev is not None else NT.evals()
    r = {'family': 'DR', 'card': 'DR_card', 'revision': revision, 'card_committed_before_any_model_select_outcome': early,
         'grid_valid': grid_valid, 'cand': {'status': 'FITTED', 'config_id': 'DR', 'evals': ev, 'alias_of_twin': False},
         'start': {'id': 'SUCC', 'evals': NT.evals()}, 'twin': {'id': 'TWIN', 'evals': NT.evals()},
         'obj_mi': {'config_id': 'OBJ_MI', 'evals': NT.evals()},
         'paths': {'CTRL': list(CTRL)}, 'selection_data': {'CTRL': 'MODEL_FIT_NESTED'}, 'alias_of': {}, 'grid': grid,
         'costs_vs': cv, 'nolookup': {i: {'cells': NT.reg(lambda c: nolook)} for i in list(NR.BENCHMARK) + CTRL +
                                      ['OBJ_MI', 'TWIN']},
         'optimistic': opt if opt is not None else opt_rec()}
    if note is not None:
        r['revision_note'] = note
    return r


def path_pts(law='TRUE_SEX_EMPIRICAL', shift=0.0):
    pts = [('TWIN', 0.0, 0.70), ('P1', 0.004, 0.69), ('P2', 0.008, 0.68), ('P3', 0.020, 0.66)]
    return [{'id': i, 'law': law, 'spend': s, 'auc': {'pair': a + shift, 'income': 0.60 + shift, 'occupation': 0.65 + shift}}
            for i, s, a in pts]


def opt_rec(gain_s=0.030, gain_l=0.010, nested_gain=0.025, within=True, spend=0.006, law='TRUE_SEX_EMPIRICAL', shift=0.0,
            inc_excess=0.0):
    """OPT_S / OPT_L at MODEL_FIT spend `spend` (bracket P2: spend 0.008, pair AUC 0.68) with the given pair gains."""
    def x(g, mode_law=law):
        return {'law': mode_law, 'spend': spend, 'within_budget': within,
                'auc': {'pair': 0.68 + shift - g, 'income': 0.60 + shift + inc_excess, 'occupation': 0.65 + shift}}
    out = {'CTRL': {v: {'in_sample': path_pts(law, shift), 'nested': path_pts(law, shift)} for v in NR.VERSIONS}}
    out['OPT_S'] = {v: {'in_sample': x(gain_s), 'nested': x(nested_gain)} for v in NR.VERSIONS}
    out['OPT_L'] = {v: {'in_sample': x(gain_l), 'nested': x(gain_l / 2)} for v in NR.VERSIONS}
    return out


def seeds_all(offset=R.SEED_OFFSET, ks=(0, 1, 2, 3)):
    return {f'S{v}k{k}': NR.MODEL_SEED_BASE + offset + 10 * v + k for v in NR.VERSIONS for k in ks}


def reg_seeds(offset=R.SEED_OFFSET):
    return seeds_all(offset, ks=(0, 1))


def boots(reps=('R1', 'R2')):
    return {f'S{v}{r}': R.registered_boot_seed(v, r) for v in NR.VERSIONS for r in reps}


def checks(wires=('DR', 'SUCC', 'REF'), reps=('R1', 'R2'), k_pair=None):
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


def with_eq(ch, pred):
    ch = copy.deepcopy(ch)
    for c in ch:
        if pred(c):
            c['equal_k0_k1'] = True
    return ch


def di(income=6, occupation=5000, pair=7000, wires=('DR', 'SUCC', 'REF'), over=None):
    out = {f'{w}|V{v}': {'income': income, 'occupation': occupation, 'pair': pair} for w in wires for v in NR.VERSIONS}
    for k, x in (over or {}).items():
        out[k].update(x)
    return out


MIN = ['DR', 'SUCC']


# ---------------------------------------------------------------- rule file and the inherited module
@test
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'DRF1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(Path(R.RULES_PATH).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='drf_rules_test_'))
    alter = [(('reader_grid', 'seed_offset'), 1000), (('reader_grid', 'forbidden_offsets'), [300, 400, 500, 600, 700, 800, 900]),
             (('reader_grid', 'bootstrap_seed_base'), 2026102400), (('reader_grid', 'split_tag'), 'SAR1_devgrid_R{r}'),
             (('er_explanation', 'few_valued_max'), 1600), (('er_explanation', 'pair_finite_max'), 4096),
             (('decisions', 'D2', 'met_label'), 'COMPONENT_GAIN_BEYOND_START_AND_CONTROLS'),
             (('decisions', 'D2', 'comparisons'), ['CTRL_FRONTIER']),
             (('decisions', 'D2', 'selection_data', 'allowed'), ['MODEL_FIT_NESTED', 'MODEL_SELECT']),
             (('comparators', 'START'), 'H_LJ'), (('registration_status', 'family'), 'FA'),
             (('registration_status', 'max_revisions'), 3), (('optimistic', 'useful_margin_auc'), 0.58),
             (('recipient_limit_auc',), 0.02), (('z',), 1.5),
             (('failure_stage', 'labels'), ['STAGE_1', 'STAGE_2']),
             (('outcome_class', 'labels'), ['CANDIDATE', 'COMPONENT'])]
    for i, (path_, val) in enumerate(alter):
        bad = copy.deepcopy(raw)
        node = bad
        for k in path_[:-1]:
            node = node[k]
        node[path_[-1]] = val
        p = tmp / f'bad_{i}.json'
        p.write_text(json.dumps(bad))
        try:
            R.load_rules(p)
        except ValueError:
            continue
        raise AssertionError(f'altered {path_} was accepted')
    bad = copy.deepcopy(raw); bad['schema'] = 'SAR1-rules-v1'
    p = tmp / 'bad_schema.json'; p.write_text(json.dumps(bad))
    try:
        R.load_rules(p)
        raise AssertionError('wrong schema accepted')
    except ValueError:
        pass


@test
def t_nrm_rules_unchanged_not_monkeypatched():
    assert hashlib.sha256(Path(NR.__file__).read_bytes()).hexdigest() == R.NRM_RULES_PY_SHA256
    assert NR.load_rules()['_sha256'] == R.NRM1_RULES_SHA256
    assert R.NR is NR
    for name in ('point_eligibility', 'reserve_status', 'aggregate', 'benefit_screen', 'target_screen', 'recipient_screen',
                 'comparison', 'frontier_comparison', 'robustness', 'claims', 'grid_validity', 'forbidden_seeds'):
        assert getattr(NR, name).__module__ == 'nrm.rules', name
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.TARGET == 0.020 and NR.RECIPIENT_LIMIT == 0.010
    assert NR.CAPS == {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005} and NR.GUARD_CAP == 0.005
    assert NR.point_eligibility(NT.evals())['n_conditions'] == 28 and R.USEFUL_MARGIN == NR.TARGET


# ---------------------------------------------------------------- (9) grid validity
@test
def t_grid_validity_seeds_and_bootstrap():
    assert R.registered_seed(0, 0) == 2026102305 and R.registered_seed(1, 3) == 2026102318
    assert R.registered_boot_seed(0, 'R1') == 2026102500 and R.registered_boot_seed(1, 'R4') == 2026102603
    fs = R.forbidden_seeds()
    assert len(fs) == 160 and NR.MODEL_SEED_BASE + 1000 + 19 in fs and not set(seeds_all().values()) & fs
    ch = checks()
    ok = R.grid_validity(reg_seeds(), ch, MIN, di(occupation=5000, income=5000), boot_seeds=boots())
    assert ok['effective'] and ok['label'] == 'READER_SEEDS_EFFECTIVE', ok['errors']
    for off in (1000, 900):                                         # SAR1's and IPR1's realized grids are forbidden
        old = R.grid_validity(reg_seeds(off), ch, MIN, di(income=5000), boot_seeds=boots())
        assert not old['effective'] and any('reuse earlier grid seeds' in e for e in old['errors'])
    shifted = reg_seeds(); shifted['S1k1'] = NR.MODEL_SEED_BASE + 1200 + 11                    # unregistered, not forbidden
    x = R.grid_validity(shifted, ch, MIN, di(income=5000), boot_seeds=boots())
    assert not x['effective'] and any('differ from the registered' in e for e in x['errors'])
    nb = R.grid_validity(reg_seeds(), ch, MIN, di(income=5000), boot_seeds=None)
    assert not nb['effective'] and any('bootstrap seeds not recorded' in e for e in nb['errors'])
    wb = boots(); wb['S1R2'] = 2026102501
    assert not R.grid_validity(reg_seeds(), ch, MIN, di(income=5000), boot_seeds=wb)['effective']
    dup = reg_seeds(); dup['S1k1'] = dup['S1k0']
    assert not R.grid_validity(dup, ch, MIN, di(income=5000), boot_seeds=boots())['effective']
    missing = [c for c in ch if c['recipe'] != 'MLP64']
    assert not R.grid_validity(reg_seeds(), missing, MIN, di(income=5000), boot_seeds=boots())['effective']


@test
def t_grid_validity_explanations():
    ch = checks()
    # (1) few-valued DR income view (6 inputs): every seed-dependent recipe equal there, each differs elsewhere -> FEW_VALUED
    fv = with_eq(ch, lambda c: c['wire'] == 'DR' and c['view'] == 'income')
    g = R.grid_validity(reg_seeds(), fv, MIN, di(income=6), boot_seeds=boots())
    assert g['effective'] and 'FEW_VALUED_SEED_EQUALITIES_COUNTED_ONCE' in g['flags'], g['errors']
    assert g['strict_nrm1']['label'] == 'READER_SEEDS_NOT_EFFECTIVE' and g['sar1_reading']['label'] == 'READER_SEEDS_EFFECTIVE'
    k = 'DR|income|MLP64|k0k1'
    assert g['explained_equalities']['n_eff_implied'][k]['n_eff_seed_cells_if_this_recipe_wins'] == 4
    # (2) the same equalities on a 1000-valued income view: not few-valued (and not a pair view) -> invalid
    assert not R.grid_validity(reg_seeds(), fv, MIN, di(income=1000), boot_seeds=boots())['effective']
    # (3) extra-trees support on the minimum set
    et = with_eq(ch, lambda c: c['wire'] == 'DR' and c['view'] == 'income' and c['recipe'] == 'ET5')
    g3 = R.grid_validity(reg_seeds(), et, MIN, di(income=1000), boot_seeds=boots())
    assert g3['effective'] and {i['explanation'] for i in g3['explained_equalities']['items']} == {'ER_EXTRA_TREES_SUPPORT'}
    assert not R.grid_validity(reg_seeds(), et, MIN, di(income=2000), boot_seeds=boots())['effective']       # > 1600
    # (4) a non-extra-trees recipe equal on a many-valued minimum-set view: invalid; off the minimum set NRM1 (iii) explains it
    ml = with_eq(ch, lambda c: c['wire'] == 'DR' and c['view'] == 'income' and c['recipe'] == 'MLP64')
    assert not R.grid_validity(reg_seeds(), ml, MIN, di(income=1000), boot_seeds=boots())['effective']
    assert R.grid_validity(reg_seeds(), ml, ['SUCC'], di(income=1000), boot_seeds=boots())['effective']
    # (5) the recipe never differs anywhere in that block and seed pair: the seed is not shown to reach it -> invalid
    dead = with_eq(ch, lambda c: c['recipe'] == 'MLP64' and c['v'] == 0 and c['R'] == 'R1')
    assert not R.grid_validity(reg_seeds(), dead, MIN, di(income=6, occupation=6, pair=6), boot_seeds=boots())['effective']
    # (6) a distinct-input count not recorded: never few-valued
    assert not R.grid_validity(reg_seeds(), fv, MIN, {}, boot_seeds=boots())['effective']


@test
def t_finite_pair_extension_above_320():
    ch = checks()
    pe = with_eq(ch, lambda c: c['wire'] == 'DR' and c['view'] == 'pair')      # every recipe equal on DR's pair view
    fin = {f'DR|V{v}': {'income': 14, 'occupation': 60, 'pair': 700} for v in NR.VERSIONS}
    g = R.grid_validity(reg_seeds(), pe, MIN, di(over=fin), boot_seeds=boots())
    assert g['effective'] and 'FEW_VALUED_PAIR_ABOVE_320' in g['flags'], g['errors']
    assert {i['explanation'] for i in g['explained_equalities']['items']} == {'FEW_VALUED_PAIR_ABOVE_320'}
    assert g['sar1_reading']['label'] == 'READER_SEEDS_NOT_EFFECTIVE'                  # SAR1's 320 bound reported beside
    # more distinct pair inputs than the product of the wire's own single-view counts: not explained
    big = {f'DR|V{v}': {'income': 10, 'occupation': 60, 'pair': 700} for v in NR.VERSIONS}
    assert not R.grid_validity(reg_seeds(), pe, MIN, di(over=big), boot_seeds=boots())['effective']
    # a single view not few-valued (continuous occupation): not a finite release -> not explained
    cont = {f'DR|V{v}': {'income': 14, 'occupation': 5000, 'pair': 700} for v in NR.VERSIONS}
    assert not R.grid_validity(reg_seeds(), pe, MIN, di(over=cont), boot_seeds=boots())['effective']
    # MLP64 equal on DR's pair view; the seed shown on DR's own single views but never on another wire's pair view: not explained
    one = with_eq(ch, lambda c: c['wire'] == 'DR' and c['view'] == 'pair' and c['recipe'] == 'MLP64')
    g1 = R.grid_validity(reg_seeds(), one, MIN, di(over=fin), boot_seeds=boots())
    assert g1['effective'] and {i['explanation'] for i in g1['explained_equalities']['items']} == {'FEW_VALUED_PAIR_ABOVE_320'}
    no_pair_evidence = [c for c in one if not (c['recipe'] == 'MLP64' and c['view'] == 'pair' and c['wire'] != 'DR')]
    g2 = R.grid_validity(reg_seeds(), no_pair_evidence, MIN, di(over=fin), boot_seeds=boots())
    assert not g2['effective'] and any('minimum set' in e for e in g2['errors'])
    # above the family's structural maximum 1024: not explained
    huge = {f'DR|V{v}': {'income': 300, 'occupation': 300, 'pair': 2000} for v in NR.VERSIONS}
    assert not R.grid_validity(reg_seeds(), pe, MIN, di(over=huge), boot_seeds=boots())['effective']
    # at or below 320 the unchanged SAR1 few-valued explanation applies
    small = {f'DR|V{v}': {'income': 14, 'occupation': 60, 'pair': 300} for v in NR.VERSIONS}
    g3 = R.grid_validity(reg_seeds(), pe, MIN, di(over=small), boot_seeds=boots())
    assert g3['effective'] and g3['sar1_reading']['label'] == 'READER_SEEDS_EFFECTIVE'


@test
def t_seed_pairs_judged_separately():
    ch = checks(k_pair=(0, 1)) + checks(k_pair=(2, 3))
    eq = with_eq(ch, lambda c: c.get('k_pair') == [2, 3] and c['wire'] == 'DR' and c['view'] == 'income'
                 and c['recipe'] == 'MLP64')
    g = R.grid_validity(seeds_all(), eq, MIN, di(income=1000), boot_seeds=boots(), required_k=R.ALL_K)
    assert not g['effective'] and any(e.startswith('k2k3') for e in g['errors'])
    assert R.grid_validity(seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(), required_k=R.ALL_K)['effective']


def splits(reps=('R1', 'R2', 'R3', 'R4'), overlap=0.3):
    return {r: {'assignment_sha256': hashlib.sha256(r.encode()).hexdigest(),
                'score_overlap': {q: overlap for q in reps if q != r}} for r in reps}


@test
def t_refit_effectiveness():
    reps4 = ('R1', 'R2', 'R3', 'R4')
    ch = checks(reps=reps4, k_pair=(0, 1)) + checks(reps=reps4, k_pair=(2, 3))
    ok = R.refit_effectiveness(splits(), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4))
    assert ok['label'] == 'REFIT_GENUINELY_EFFECTIVE' and not ok['flags'], ok['reasons']
    reps3 = ('R1', 'R2', 'R3')
    ch3 = checks(reps=reps3, k_pair=(0, 1)) + checks(reps=reps3, k_pair=(2, 3))
    one = R.refit_effectiveness(splits(reps3), seeds_all(), ch3, MIN, di(income=1000), boot_seeds=boots(reps3))
    assert one['label'] == 'REFIT_GENUINELY_EFFECTIVE' and one['flags'] == ['ONE_ROBUSTNESS_SPLIT_ONLY'], one['reasons']
    same = splits(); same['R3']['assignment_sha256'] = same['R1']['assignment_sha256']
    assert R.refit_effectiveness(same, seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4))['label'] == \
        'REFIT_NOT_GENUINELY_EFFECTIVE'
    none = R.refit_effectiveness(splits(('R1', 'R2')), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots())
    assert none['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and 'NO_ROBUSTNESS_SPLIT' in none['reasons']


# ---------------------------------------------------------------- (4) brackets and fineness
@test
def t_brackets_origin_twin_equal_nrm_rule():
    r = rec()
    b = R.brackets(r['costs_vs'], r['paths'])['CTRL']
    pt = {'id': 'DR', 'costs_vs': r['costs_vs'], 'grid_vs': r['grid']}
    f = NR.frontier_comparison(pt, [{'id': i} for i in ['TWIN'] + CTRL])
    assert b['status'] == 'BRACKETED' and b['bracket'] == f['bracketing_control'] == 'C2' and b['path'] == ['TWIN', 'C1', 'C2']
    r2 = rec(); r2['costs_vs']['C2'] = costs(0.0005)
    assert R.brackets(r2['costs_vs'], r2['paths'])['CTRL']['status'] == 'NO_BRACKET'
    assert R.brackets(r['costs_vs'], {'CTRL': None})['CTRL']['status'] == 'PATH_NOT_GIVEN'
    assert R.control_path(['C1', 'C2', 'C1'], {'C1': 'TWIN'}) == ['TWIN', 'C2']


def sp(cand=0.010, lower=0.008, upper=0.0135, se=0.001, mx=0.03, grid=None, data='MODEL_FIT_NESTED'):
    x = {'cand': cand, 'lower': lower, 'upper': upper, 'upper_se': se, 'max': mx}
    if grid is not None:
        x['grid'] = grid
    out = {v: dict(x) for v in NR.VERSIONS}
    if data is not None:
        out['selection_data'] = data
    return {'CTRL': out}


@test
def t_grid_fineness_model_fit_nested_only():
    assert R.grid_fineness(sp())['per_path']['CTRL'] == {0: 'FINE', 1: 'FINE'}
    assert R.grid_fineness(sp(lower=0.004))['per_path']['CTRL'][0] == 'COARSE_BELOW'
    assert R.grid_fineness(sp(mx=0.011))['per_path']['CTRL'][0] == 'CONTROL_PATH_MAY_NOT_BRACKET'
    assert R.grid_fineness(sp(upper=0.011))['per_path']['CTRL'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'
    assert R.grid_fineness(sp(cand=-0.001))['per_path']['CTRL'][0] == 'NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN'
    for bad in ('MODEL_SELECT', 'CALIBRATION', 'MODEL_FIT_OOF'):
        g = R.grid_fineness(sp(data=bad))
        assert g['per_path']['CTRL'][0] == 'SELECTION_DATA_NOT_ALLOWED' and 'CONTROL_SELECTION_DATA_NOT_ALLOWED_CTRL' in g['flags']
    assert 'CONTROL_SELECTION_DATA_NOT_RECORDED_CTRL' in R.grid_fineness(sp(data=None))['flags']
    grid = [{'spend': 0.009, 'se': 0.001}, {'spend': 0.0125, 'se': 0.001}, {'spend': 0.0135, 'se': 0.001}]
    assert R.grid_fineness(sp(grid=grid))['per_path']['CTRL'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'   # 0.009 / 0.0125 exist
    assert R.grid_fineness(sp(lower=0.009, upper=0.0125, grid=grid))['per_path']['CTRL'][0] == 'FINE'


# ---------------------------------------------------------------- (2) D1
@test
def t_d1_point_gating_confidence_reserve_beside():
    d = R.d1(rec())
    assert d['label'] == 'TASK_FEASIBLE' and d['met'] and d['confidence']['cand']['label'] == 'CONFIDENCE_ALL_PASS'
    assert set(d['confidence']) == {'cand', 'START', 'TWIN', 'OBJ_MI'} and set(d['reserve']) == {'cand', 'START', 'TWIN', 'OBJ_MI'}
    near = NT.evals(guards={'sex0': 0.0045})                    # point pass, + 2 se crosses the cap: unresolved, still feasible
    d2 = R.d1(rec(cand_ev=near))
    assert d2['label'] == 'TASK_FEASIBLE' and d2['confidence']['cand']['label'] == 'CONFIDENCE_UNRESOLVED'
    bad = NT.evals(inc={('U0', 'nll_excess'): 0.0101})
    d3 = R.d1(rec(cand_ev=bad))
    assert d3['label'] == 'CAND_INFEASIBLE' and d3['cand_failed']
    r = rec(); r['obj_mi']['evals'] = bad
    assert 'OBJ_MI_POINT_INELIGIBLE' in R.d1(r)['flags'] and R.d2(r)['met']        # an ineligible control stays a control
    r2 = rec(); r2['cand']['status'] = None
    assert R.d1(r2)['label'] == 'INCOMPLETE'


# ---------------------------------------------------------------- (4) D2
@test
def t_d2_met_and_structure():
    d = R.d2(rec())
    assert d['label'] == R.D2_MET and d['met'] and d['passed'] == ['CTRL_FRONTIER', 'OBJ'], d
    assert d['comparisons']['CTRL_FRONTIER']['point'] == 'C2' and d['comparisons']['OBJ']['spend']['state'] == \
        'OBJ_MI_SPENDS_AT_LEAST_AS_MUCH'
    assert 'SUCC' not in d['comparisons'] and 'TWIN' not in d['comparisons']       # FC and TWIN are separate records


@test
def t_protocol_ids():
    """The protocol's names (DR_AUC, DR_MI, TWB_C64, DR_CTRL_LO / HI) work through comparator_ids; a missing key is INCOMPLETE."""
    r = rec()
    ren = {'TWIN': 'TWB_C64', 'OBJ_MI': 'DR_MI', 'C1': 'DR_CTRL_LO', 'C2': 'DR_CTRL_HI'}
    for k in ('grid', 'costs_vs', 'nolookup'):
        r[k] = {ren.get(i, i): x for i, x in r[k].items()}
    r['cand']['config_id'] = 'DR_AUC'
    r['twin']['id'] = 'TWB_C64'
    r['obj_mi']['config_id'] = 'DR_MI'
    r['paths']['CTRL'] = ['DR_CTRL_LO', 'DR_CTRL_HI']
    d = R.decisions(r)
    assert d['D2']['met'] and d['D2']['comparisons']['CTRL_FRONTIER']['path'] == ['TWB_C64', 'DR_CTRL_LO', 'DR_CTRL_HI']
    assert d['D2']['comparisons']['OBJ']['point'] == 'DR_MI' and d['TWIN']['comparator'] == 'TWB_C64'
    assert R.comparator_ids(r) == ('TWB_C64', 'DR_MI') and R.comparator_ids({}) == ('TWIN', 'OBJ_MI')
    r2 = copy.deepcopy(r); r2['obj_mi']['config_id'] = 'OBJ_MI'          # the key does not exist under the default name
    assert R.d2(r2)['label'] == 'INCOMPLETE'


@test
def t_d2_obj_spend_states():
    un = R.d2(rec(obj_total=0.0008))                                # DR spends 0.0008 more, SE 0.0005: not resolved
    assert un['met'] and 'OBJ_MI_SPENDS_LESS_UNRESOLVED' in un['flags']
    assert un['comparisons']['OBJ']['spend']['state'] == 'OBJ_MI_SPENDS_LESS_UNRESOLVED' and un['comparisons']['OBJ']['statement']
    rs = R.d2(rec(obj_total=0.002))                                 # resolved: the objective is not credited for spending more
    o = rs['comparisons']['OBJ']
    assert rs['label'] == 'PARTIAL_CTRL_FRONTIER_ONLY' and o['reason'] == 'NOT_SEPARATED_FROM_SPEND'
    assert o['passed_if_spend_matched'] is True and 'not credited for a spend change' in rs['statement']
    assert o['inherited_point_rule_beside']['label'] == 'NOT_SEPARATED_FROM_TASK_SPENDING'
    r = rec(); r['costs_vs']['OBJ_MI'] = costs(-0.001, {'occupation|men_brier': {'value': 0.002, 'se': 0.0004}})
    assert R.d2(r)['comparisons']['OBJ']['reason'] == 'SECONDARY_METRIC_RESOLVED_HIGHER'
    nb = R.d2(rec(obj_pair=0.004))                                  # benefit not resolved vs OBJ_MI
    assert nb['label'] == 'PARTIAL_CTRL_FRONTIER_ONLY' and 'SEX-aware routing' in nb['statement']
    al = rec(); al['grid']['OBJ_MI'] = copy.deepcopy(NT.ALIAS)
    da = R.d2(al)
    assert 'CAND_ALIAS_OF_OBJ_MI' in da['flags'] and da['comparisons']['OBJ']['reason'] == 'DR_EQUALS_OBJ_MI'
    rg = rec(); rg['grid']['OBJ_MI'] = NT.grid_entry(0.02, inc=0.02)  # income view of DR 0.020 more recoverable
    assert R.d2(rg)['comparisons']['OBJ']['reason'] == 'RECIPIENT_GATE_FAILED'


@test
def t_d2_ctrl_partials_no_bracket_domination():
    p = R.d2(rec(ctrl_pair=0.004))
    assert p['label'] == 'PARTIAL_OBJ_ONLY' and 'task spending' in p['statement']
    assert R.d2(rec(ctrl_pair=0.004, obj_pair=0.004))['label'] == 'NOT_SEPARATED'
    r = rec(); r['costs_vs']['C2'] = costs(0.0005)                   # every CTRL point spends less than DR: no bracket
    d = R.d2(r)
    assert d['label'] == 'PARTIAL_OBJ_ONLY' and 'NO_BRACKET_CTRL' in d['flags']
    assert d['comparisons']['CTRL_FRONTIER']['reason'] == 'NO_BRACKET'
    r2 = rec(); r2['grid']['C1'] = NT.grid_entry(-0.03)              # the cheaper C1 is resolved less recoverable than DR
    d2 = R.d2(r2)
    assert 'DOMINATED_BY_CHEAPER_C1' in d2['flags'] and not d2['comparisons']['CTRL_FRONTIER']['pass']
    r3 = rec(); r3['grid']['C2'] = NT.grid_entry(0.02, occ=0.02)    # recipient gate fails at Q*
    assert 'RECIPIENT_GATE_FAILED_VS_C2' in R.d2(r3)['flags']
    r4 = rec(); r4['costs_vs']['TWIN'] = costs(-0.0001)              # DR spends no more than TWIN: the origin is the bracket
    assert 'CTRL_BRACKET_IS_ORIGIN' in R.d2(r4)['flags']
    r5 = rec(); r5['paths']['CTRL'] = []
    assert 'CTRL_AT_ORIGIN' in R.d2(r5)['flags']


@test
def t_d2_targeting():
    t = R.d2(rec(nolook=-0.002))
    assert t['label'] == 'READER_TARGETED' and 'READER_TARGETED_OBJ' in t['flags']
    r = rec(); r['nolookup']['OBJ_MI'] = {'cells': NT.reg(lambda c: -0.001 if c.startswith('S1') else 0.01)}
    assert R.d2(r)['label'] == 'READER_TARGETED'
    r2 = rec(); del r2['nolookup']['C2']
    assert R.d2(r2)['label'] == 'INCOMPLETE'


@test
def t_d2_gates_and_inputs():
    a = rec(); a['cand']['alias_of_twin'] = True
    assert R.d2(a)['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    bad = NT.evals(occ={('Ucal', 'brier_excess'): 0.0051})
    assert R.d2(rec(cand_ev=bad))['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    assert R.d2(rec(grid_valid=None))['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    assert R.d2(rec(grid_valid=False))['label'] == 'NOT_COMPUTED_GRID_INVALID'
    nr = rec(); nr['paths']['CTRL'] = None; nr['controls_not_run'] = {'CTRL': True}
    assert R.d2(nr)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    no = rec(); del no['grid']['OBJ_MI']; no['controls_not_run'] = {'OBJ': True}
    assert R.d2(no)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    mp = rec(); mp['paths']['CTRL'] = None
    assert R.d2(mp)['label'] == 'INCOMPLETE'                          # missing path, not recorded as not run
    mo = rec(); del mo['grid']['OBJ_MI']
    assert R.d2(mo)['label'] == 'INCOMPLETE'
    for sd in (None, 'MODEL_SELECT', 'CALIBRATION'):
        s = rec(); s['selection_data'] = {'CTRL': sd}
        assert R.d2(s)['label'] == 'INCOMPLETE'
    mc = rec(); del mc['costs_vs']['OBJ_MI']
    assert R.d2(mc)['label'] == 'INCOMPLETE'
    st = rec(); st['reader_stage'] = 'STOPPED_USEFULNESS_FAILURE'      # stopping is allowed only after a usefulness failure
    assert R.d2(st)['label'] == 'INCOMPLETE'
    ok = rec(cand_ev=bad); ok['reader_stage'] = 'STOPPED_USEFULNESS_FAILURE'
    assert R.d2(ok)['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'


@test
def t_fc_and_twin_reported_apart():
    d = R.decisions(rec(start_pair=-0.01))                          # DR more recoverable than SUCC: FC fails, D3 fails
    assert d['FC']['label'] == 'FAMILY_CHANGE_NOT_RESOLVED' and d['D2']['met'] and not d['D3']['met']
    assert d['outcome_class']['label'] == 'COMPONENT_GAIN_NOT_COMPETITIVE' and 'SUCC' in d['D3']['failing']
    t = R.decisions(rec(ctrl_pair=0.0, obj_pair=0.0, twin_pair=0.05, refs=-0.01, start_pair=-0.01))
    assert t['TWIN']['label'] == 'BELOW_TWIN_RESOLVED' and not t['D2']['met']
    assert t['outcome_class']['label'] == 'FEASIBLE_NO_RELIABLE_GAIN'
    assert 'TWIN_ONLY_ATTRIBUTION_INSUFFICIENT' in t['outcome_class']['flags']
    f = R.decisions(rec())
    assert f['FC']['label'] == 'FAMILY_CHANGE_BELOW_START_RESOLVED' and f['outcomes']['family_change_vs_start']['met']


# ---------------------------------------------------------------- (5) D3, D4
@test
def t_d3_d4_lists_and_credit():
    d = R.decisions(rec())
    assert d['D3']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' and d['D4']['label'] == 'MATERIAL_0020_RESOLVED'
    r = rec(); r['grid']['H_LJ'] = NT.grid_entry(0.004)
    d2 = R.decisions(r)
    assert d2['D3']['label'] == 'NOT_SUPERIOR' and 'H_LJ' in d2['D3']['failing'] and d2['D4']['label'] != 'MATERIAL_0020_RESOLVED'
    r3 = rec(); r3['reader_budget'] = {'NEURAL_RAW': 'READER_BUDGET_ASYMMETRIC'}
    r3['nolookup']['NEURAL_RAW'] = {'cells': NT.reg(lambda c: -0.001)}
    d3 = R.decisions(r3)
    assert d3['D3']['label'] == 'NOT_SUPERIOR' and 'NEURAL_RAW' in d3['D3']['not_credited'] and d3['D4']['label'] == 'NOT_MET'
    r4 = rec(refs=0.025)                                            # 0.025 - 2 x 0.004 = 0.017 < 0.020: point only
    assert R.decisions(r4)['D4']['label'] == 'MATERIAL_0020_POINT_ONLY'


# ---------------------------------------------------------------- (5) optimistic diagnostic and the failure stage
@test
def t_useful_optimistic_comparator_based():
    u = R.useful_optimistic(opt_rec())
    assert u['label'] == 'USEFUL_OPTIMISTIC_POINT' and u['useful_in_sample'] == ['OPT_S'] and u['held_out_survives'] == ['OPT_S']
    assert u['per_construction']['OPT_S']['in_sample']['by_version'][0]['bracket'] == 'P2'
    n = R.useful_optimistic(opt_rec(gain_s=0.015, gain_l=0.010))
    assert n['label'] == 'NO_USEFUL_OPTIMISTIC_POINT'
    for sh in (-0.15, 0.2):                                         # no absolute threshold: a common shift changes nothing
        assert R.useful_optimistic(opt_rec(shift=sh))['label'] == u['label']
        assert R.useful_optimistic(opt_rec(gain_s=0.015, shift=sh))['label'] == n['label']
    assert R.useful_optimistic(opt_rec(within=False))['label'] == 'NO_USEFUL_OPTIMISTIC_POINT'
    assert R.useful_optimistic(opt_rec(inc_excess=0.011))['label'] == 'NO_USEFUL_OPTIMISTIC_POINT'
    ub = R.useful_optimistic(opt_rec(gain_s=0.015, spend=0.03))      # beyond every CTRL path spend: unbracketed
    assert ub['label'] == 'UNDETERMINED'
    lm = opt_rec(gain_s=0.015); lm['CTRL'][0]['in_sample'][1]['law'] = 'ESTIMATED_PI'
    assert R.useful_optimistic(lm)['label'] == 'UNDETERMINED'          # comparator on another law: incomplete
    ho = R.useful_optimistic(opt_rec(nested_gain=0.005))
    assert ho['label'] == 'USEFUL_OPTIMISTIC_POINT' and ho['held_out_survives'] == []
    assert R.useful_optimistic(None)['label'] == 'UNDETERMINED'


@test
def t_failure_stages():
    useful, none = opt_rec(), opt_rec(gain_s=0.015)
    s5 = R.decisions(rec())
    assert s5['failure_stage']['label'] == 'STAGE_5_COMPETITIVE_CONDITIONS_PASS'
    s4 = R.decisions(rec(refs=-0.01))
    assert s4['failure_stage']['label'] == 'STAGE_4_BEATS_CONTROLS_MISSES_STRONG_REFERENCES'
    assert 'COMPONENT_RESULT_NOT_ROBUSTNESS_SUPPORTED' in s4['failure_stage']['flags']
    rr = rec(refs=-0.01); rr['d2r'] = {'label': 'ROBUSTNESS_CONFIRMED', 'sign_changes': [], 'recipient_failures': []}
    assert R.decisions(rr)['failure_stage']['scoped_component_result'] is True
    rr['d2r']['recipient_failures'] = ['C2']
    assert R.decisions(rr)['failure_stage']['scoped_component_result'] is False
    s3 = R.decisions(rec(ctrl_pair=0.0, refs=-0.01, opt=useful))
    assert s3['failure_stage']['label'] == 'STAGE_3_KEEPS_UTILITY_TIES_CONTROLS'
    s1 = R.decisions(rec(ctrl_pair=0.0, refs=-0.01, opt=none))
    assert s1['failure_stage']['label'] == 'STAGE_1_NO_USEFUL_OPTIMISTIC_POINT'
    assert s1['failure_stage']['legal_stage'] == 'L3_KEEPS_UTILITY_TIES_CONTROLS'
    bad = NT.evals(occ={('U0', 'nll_excess'): 0.011})
    s2 = R.decisions(rec(cand_ev=bad, opt=useful))
    assert s2['failure_stage']['label'] == 'STAGE_2_OPTIMISTIC_SUCCEEDS_LEGAL_MAPPING_FAILS'
    assert R.decisions(rec(cand_ev=bad, opt=none))['failure_stage']['label'] == 'STAGE_1_NO_USEFUL_OPTIMISTIC_POINT'
    al = rec(opt=useful); al['cand']['alias_of_twin'] = True
    assert R.decisions(al)['failure_stage']['label'] == 'STAGE_2_OPTIMISTIC_SUCCEEDS_LEGAL_MAPPING_FAILS'
    ud = R.decisions(rec(cand_ev=bad, opt=opt_rec(gain_s=0.015, spend=0.03)))
    assert ud['failure_stage']['label'] == 'UNDETERMINED_OPTIMISTIC_DIAGNOSTIC'
    assert R.decisions(rec(grid_valid=None))['failure_stage']['label'] == 'INCOMPLETE'
    pg = R.decisions(rec(refs=-0.01, opt=none))                     # never gating: the legal mapping still progresses
    assert pg['failure_stage']['label'].startswith('STAGE_4') and \
        'NO_USEFUL_OPTIMISTIC_POINT_BUT_LEGAL_MAPPING_PROGRESSED' in pg['failure_stage']['flags']
    assert pg['D2']['met'] and pg['outcome_class']['label'] == 'COMPONENT_GAIN_NOT_COMPETITIVE'


# ---------------------------------------------------------------- (6) outcome classes
def _oc(r):
    return R.decisions(r)['outcome_class']['label']


@test
def t_outcome_classes_and_three_outcomes():
    assert _oc(rec()) == 'METHOD_CANDIDATE'
    assert _oc(rec(refs=-0.01)) == 'COMPONENT_GAIN_NOT_COMPETITIVE'
    d = R.decisions(rec(ctrl_pair=0.0))
    assert d['outcome_class']['label'] == 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED' and 'D3_WITHOUT_D2' in d['outcome_class']['flags']
    assert 'D4_WITHOUT_D2' in d['outcome_class']['flags']
    assert _oc(rec(ctrl_pair=0.0, refs=-0.01)) == 'FEASIBLE_NO_RELIABLE_GAIN'
    a = rec(); a['cand']['alias_of_twin'] = True
    assert _oc(a) == 'NO_OUTPUT_CHANGE'
    bad = NT.evals(inc={('Ucal', 'accuracy_loss'): 0.0101})
    ti = rec(cand_ev=bad); ti['reader_stage'] = 'STOPPED_USEFULNESS_FAILURE'
    dt = R.decisions(ti)
    assert dt['outcome_class']['label'] == 'TASK_INFEASIBLE' and dt['D3']['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    assert 'READER_STAGE_STOPPED_RELEASE_PRESERVED' in dt['outcome_class']['flags']
    nr = rec(); nr['paths']['CTRL'] = None; nr['controls_not_run'] = {'CTRL': True}
    assert _oc(nr) == 'PROTECTION_NOT_ASSESSED'
    assert _oc(rec(grid_valid=False)) == 'INCOMPLETE'
    near = NT.evals(guards={'sex0': 0.0045})
    f = R.decisions(rec(cand_ev=near))['outcome_class']['flags']
    assert 'POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED' in f                                # beside, never gating
    o = R.decisions(rec(refs=-0.01))['outcomes']
    assert o['1_component_gain']['met'] and not o['2_strong_references']['met'] and not o['3_material_target']['met']
    assert '(1) component gain' in o['statement'] and 'failure stage' in o['statement']


# ---------------------------------------------------------------- (7) registration and the campaign summary
@test
def t_registration_and_revisions():
    assert R.adaptive_tag(rec()) is None
    assert R.adaptive_tag(rec(early=False)) == 'ADAPTIVE_R0_LATE_CARD'
    note = {'observed_failure': 'tiny exact instance: local search 0.004 above the exhaustive optimum',
            'changed_operation': 'two-cell swap moves added', 'trigger': 'SYNTHETIC'}
    assert R.adaptive_tag(rec(revision=1, note=note)) is None                    # pre-outcome, non-MODEL_SELECT trigger
    ms = dict(note, trigger='MODEL_SELECT')
    assert R.adaptive_tag(rec(revision=1, note=ms)) == 'ADAPTIVE_R1'
    assert R.adaptive_tag(rec(revision=2, note=note, early=False)) == 'ADAPTIVE_R2'
    assert R.adaptive_tag(rec(revision=1)) == 'ADAPTIVE_R1_NO_NOTE'
    assert R.adaptive_tag(rec(revision=1, note=dict(note, changed_operation=' '))) == 'ADAPTIVE_R1_NO_NOTE'
    for bad in ({'revision': 3}, {'family': 'FA'}, {'family': 'OBJ_MI'}):
        r = rec(); r.update(bad)
        try:
            R.decisions(r)
            raise AssertionError(f'{bad} accepted')
        except ValueError:
            pass
    d = R.decisions(rec(revision=1, note=ms))
    assert not d['registered'] and d['D2']['label'].startswith('ADAPTIVE_R1__') and d['D2']['label_unprefixed'] == R.D2_MET
    assert d['failure_stage']['label'].startswith('ADAPTIVE_R1__') and d['outcome_class']['label'].startswith('ADAPTIVE_R1__')
    nn = R.decisions(rec(revision=1))
    assert 'REVISION_NOTE_MISSING' in nn['outcome_class']['flags']


@test
def t_campaign_summary():
    note = {'observed_failure': 'x', 'changed_operation': 'y', 'trigger': 'MODEL_FIT'}
    r0 = R.decisions(rec(refs=-0.01))
    r1 = R.decisions(rec(revision=1, note=note, ctrl_pair=0.0))
    r2 = R.decisions(rec(revision=2, note=dict(note, trigger='MODEL_SELECT')))
    s = R.campaign_summary([r0, r1, r2])
    assert s['registered']['revision'] == 1 and [x['revision'] for x in s['registered_superseded']] == [0]
    assert 'MULTIPLE_REGISTERED_READINGS' in s['flags'] and s['d2_looks'] == 3
    assert 'D2_MET_ONLY_IN_NON_REGISTERED_LOOKS' in s['flags']                    # the headline (revision 1) did not meet D2
    assert 'D2 was met in 0 of 1 registered headline readings and in 2 of 2 other readings' in s['statement']
    assert 'no multiplicity correction' in s['statement']
    try:
        R.campaign_summary([r0, R.decisions(rec(refs=-0.01))])
        raise AssertionError('two registered readings with the same revision accepted')
    except ValueError:
        pass


# ---------------------------------------------------------------- (8) D2R
def rob_entry(reg_x, rob_x, reps=('R3', 'R4'), inc_rob=0.0):
    cells, ses, inc = {}, {}, {}
    for v in NR.VERSIONS:
        for r in ('R1', 'R2') + reps:
            for k in (0, 1, 2, 3):
                c = f'S{v}{r}k{k}'
                cells[c] = reg_x if c in NR.REGISTERED else rob_x
                inc[c] = 0.0 if c in NR.REGISTERED else inc_rob
                ses[c] = 0.004
    z = {'cells': {c: 0.0 for c in cells}, 'cell_se': dict(ses)}
    return {'pair': {'cells': cells, 'cell_se': ses}, 'income': {'cells': inc, 'cell_se': dict(ses)},
            'occupation': copy.deepcopy(z)}


@test
def t_d2r():
    refit = {'label': 'REFIT_GENUINELY_EFFECTIVE', 'counted_once': []}
    base = {'grid_valid': True, 'refit': refit, 'comparators': {'CTRL': 'C2', 'OBJ': 'OBJ_MI'},
            'rob': {'C2': rob_entry(0.02, 0.02), 'OBJ_MI': rob_entry(0.02, 0.02), 'SUCC': rob_entry(0.03, 0.03),
                    'NEURAL_RAW': rob_entry(0.03, 0.03), 'H_LJ': rob_entry(0.03, 0.03)}}
    d = R.d2r(base)
    assert d['label'] == 'ROBUSTNESS_CONFIRMED' and d['decisive'] == ['C2', 'OBJ_MI'] and not d['sign_changes'], d
    assert 'SUCC' in d['beside'] and d['headline_caveat'] is None
    sc = copy.deepcopy(base); sc['rob']['OBJ_MI'] = rob_entry(0.02, -0.01)
    d2 = R.d2r(sc)
    assert d2['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d2['sign_changes'] == ['OBJ_MI'] and 'headline' in d2['headline_caveat']
    rf = copy.deepcopy(base); rf['rob']['C2'] = rob_entry(0.02, 0.02, inc_rob=0.02)   # income view fails on robustness cells
    d3 = R.d2r(rf)
    assert d3['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d3['recipient_failures'] == ['C2'] and 'single-view' in d3['headline_caveat']
    bs = copy.deepcopy(base); bs['rob']['H_LJ'] = rob_entry(0.03, 0.03, inc_rob=0.02)   # beside: carried, label unchanged
    d4 = R.d2r(bs)
    assert d4['label'] == 'ROBUSTNESS_CONFIRMED' and d4['recipient_failures'] == ['H_LJ'] and d4['headline_caveat']
    nf = copy.deepcopy(base); nf['refit'] = {'label': 'REFIT_NOT_GENUINELY_EFFECTIVE'}
    assert R.d2r(nf)['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    nr = copy.deepcopy(base); del nr['rob']['OBJ_MI']
    assert R.d2r(nr)['label'] == 'NOT_RUN'
    inc = copy.deepcopy(base); del inc['comparators']['OBJ']
    assert R.d2r(inc)['label'] == 'INCOMPLETE'
    br = copy.deepcopy(base); br['comparators']['CTRL'] = R.brackets(rec()['costs_vs'], rec()['paths'])
    assert R.d2r(br)['decisive'] == ['C2', 'OBJ_MI']
    assert R.d2r({**base, 'grid_valid': None})['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    dd = rec(); dd['d2r'] = d3
    dec = R.decisions(dd)
    assert 'D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE' in dec['outcome_class']['flags'] and \
        'single-view' in dec['outcomes']['statement']


# ---------------------------------------------------------------- runner
def main():
    t0 = time.process_time()
    res = []
    for fn in TESTS:
        try:
            fn()
            res.append({'test': fn.__name__, 'status': 'PASS'})
        except Exception as ex:                                            # noqa: BLE001 - report every failure
            res.append({'test': fn.__name__, 'status': 'FAIL', 'error': f'{type(ex).__name__}: {ex}'[:600]})
    n_pass = sum(r['status'] == 'PASS' for r in res)
    out = {'module': 'drf.test_rules', 'rules_sha256': R.load_rules()['_sha256'],
           'drf_rules_py_sha256': hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(), 'passed': n_pass,
           'total': len(res), 'cpu_seconds': round(time.process_time() - t0, 3), 'results': res}
    print(json.dumps(out, indent=1))
    return 0 if n_pass == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())

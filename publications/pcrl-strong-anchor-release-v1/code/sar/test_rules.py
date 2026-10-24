"""Invented-data tests of sar.rules (no real data, no label). Run: python3 -m sar.test_rules (prints JSON; exit 0 iff all pass).

Covers: the rule file and its refusal of altered constants; nrm.rules left unchanged (never monkeypatched); reader-seed validity of
the SAR1 grid (+1000 registered, 300-900 forbidden, bootstrap seeds, the extra-trees support explanation, the few-valued
explanation counted once, seed pairs judged separately); the genuinely effective refit (one or two robustness splits); the CTRL
bracket rule (equal to nrm.rules.frontier_comparison); MODEL_FIT fineness of the saved CTRL points; D1 (point eligibility gating,
the confidence conjunction and the reserve reported separately, MC SE in quadrature); UM (binding condition from START's record
and the frozen anchor table, pooled resolved improvement with both versions, recovery profile, whole-release point margin, every
label); D2 (START, CONT gating iff declared, the CTRL frontier with origin, no bracket, bracket at the origin, recipient gate at the
bracket, cheaper-point domination, reader targeting, partials named by what passed, gates and missing inputs, aliases); D3 passed
/ failed lists and the reader-budget credit; D4 levels; D5 not claimed; the outcome classes and the three separate outcomes;
families FA / FB / FS_<name>; registered vs adaptive labelling; the campaign summary; strongest_nonzero; D2R with sign changes.
Helpers for invented utility records and grid entries come from nrm.test_rules (unchanged)."""
from __future__ import annotations
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

from nrm import rules as NR
from nrm import test_rules as NT                 # invented-data helpers (util, evals, entry, reg, grid_entry, costs)
from sar import rules as R

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- invented records
def patch_se(ev, name, se, se_mc=None):
    """Set the SE (and optionally se_mc) of one condition in both versions of an evals dict."""
    ev = copy.deepcopy(ev)
    for v in NR.VERSIONS:
        u = ev[v]
        for c in u['clauses']:
            if f"{c['task']}|{c['reference']}|{c['metric']}" == name:
                c['se'] = se
                if se_mc is not None:
                    c['se_mc'] = se_mc
        for g in u['guards']:
            if f"guard|{g['group']}" == name:
                g['se'] = se
                if se_mc is not None:
                    g['se_mc'] = se_mc
    return ev


def succ_evals(sex0=0.0044, sex1=-0.0049):
    """SUCC-like: the men's guard binds (+0.0044, se 0.0018 -> reserve ratio about 0.33), everything else has wide margin."""
    return patch_se(NT.evals(guards={'sex0': sex0, 'sex1': sex1}), 'guard|sex0', 0.0018)


def neural_raw_evals():
    """NEURAL_RAW-like: v0 binds on the men's guard, v1 on income Brier vs U0 (as the frozen anchor table)."""
    ev = {0: NT.util(0, guards={'sex0': 0.0015, 'sex1': -0.001}),
          1: NT.util(1, guards={'sex0': 0.0015, 'sex1': -0.001}, inc={('U0', 'brier_excess'): 0.003})}
    return patch_se(ev, 'guard|sex0', 0.0005)


def costs(total, extra=None, se=0.0005):
    return NT.costs(total, se=se, extra=extra)


def men(x, se=0.0004):
    return {'occupation|men_brier': {'value': x, 'se': se}}


CTRL = ['C1', 'C2']
ALT = ['A1', 'A2']


def rec(start_pair=0.03, ctrl_pair=0.02, refs=0.03, men_diff=-0.0014, cand_ev=None, grid_valid=True, nolook=0.01,
        family='FA', revision=0, early=True, cont=None, alt=None):
    """A complete invented family reading. FA: START = SUCC; CAND spends +0.003 nats over START; C1 cheaper than CAND, C2 more
    expensive (the bracket); CAND improves the men's guard by men_diff (paired, se 0.0004). cont = None (not declared) or the
    pair benefit vs a declared CONT; alt = None (not declared) or the pair benefit vs both points of a declared ALT path (A1
    cheaper than CAND, A2 the bracket). Bracket selection data MODEL_FIT_OOF."""
    sid = R.STARTS.get(family, 'SUCC')
    sev = succ_evals() if sid == 'SUCC' else neural_raw_evals()
    cev = cand_ev if cand_ev is not None else (succ_evals(sex0=0.0030) if sid == 'SUCC' else neural_raw_evals())
    grid = {r: NT.grid_entry(refs) for r in NR.BENCHMARK}
    grid[sid] = NT.grid_entry(start_pair)
    grid.update({i: NT.grid_entry(ctrl_pair) for i in CTRL})
    cv = {sid: costs(0.003, men(men_diff)), 'C1': costs(0.001), 'C2': costs(-0.001)}
    r = {'family': family, 'card': f'{family}_x', 'revision': revision, 'card_committed_before_any_model_select_outcome': early,
         'grid_valid': grid_valid, 'start': {'id': sid, 'evals': sev, 'binding': dict(R.START_BINDING[sid])},
         'cand': {'status': 'FITTED', 'config_id': 'CAND', 'evals': cev, 'strength': 2.0},
         'strengths': [{'config_id': 'CAND', 'strength': 2.0, 'evals': cev}], 'cont_declared': cont is not None,
         'paths': {'CTRL': list(CTRL)}, 'alias_of': {}, 'grid': grid, 'costs_vs': cv,
         'selection_data': {'CTRL': 'MODEL_FIT_OOF'},
         'nolookup': {i: {'cells': NT.reg(lambda c: nolook)} for i in list(NR.BENCHMARK) + CTRL + ALT + ['CONT']}}
    if cont is not None:
        r['cont'] = {'config_id': 'CONT', 'evals': NT.evals()}
        grid['CONT'] = NT.grid_entry(cont)
    if alt is not None:
        r['alt_declared'] = True
        r['paths']['ALT'] = list(ALT)
        r['selection_data']['ALT'] = 'MODEL_FIT_OOF'
        grid.update({i: NT.grid_entry(alt) for i in ALT})
        cv.update({'A1': costs(0.0015), 'A2': costs(-0.0008)})
    return r


def seeds_all(offset=R.SEED_OFFSET, ks=(0, 1, 2, 3)):
    return {f'S{v}k{k}': NR.MODEL_SEED_BASE + offset + 10 * v + k for v in NR.VERSIONS for k in ks}


def reg_seeds(offset=R.SEED_OFFSET):
    return seeds_all(offset, ks=(0, 1))


def boots(reps=('R1', 'R2')):
    return {f'S{v}{r}': R.registered_boot_seed(v, r) for v in NR.VERSIONS for r in reps}


def checks(wires=('CAND', 'START', 'REF'), reps=('R1', 'R2'), k_pair=None):
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


def di(income=6, occupation=5000, pair=7000, wires=('CAND', 'START', 'REF')):
    return {f'{w}|V{v}': {'income': income, 'occupation': occupation, 'pair': pair} for w in wires for v in NR.VERSIONS}


MIN = ['CAND', 'START']


# ---------------------------------------------------------------- rule file and the inherited module
@test
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'SAR1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(Path(R.RULES_PATH).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='sar_rules_test_'))
    alter = [(('reader_grid', 'seed_offset'), 900), (('reader_grid', 'forbidden_offsets'), [300, 400, 500, 600, 700, 800]),
             (('reader_grid', 'bootstrap_seed_base'), 2026102300), (('er_explanation', 'few_valued_max'), 1600),
             (('decisions', 'UM', 'start_binding'), {'SUCC': {'0': 'guard|sex1', '1': 'guard|sex0'},
                                                     'NEURAL_RAW': {'0': 'guard|sex0', '1': 'income|U0|brier_excess'}}),
             (('decisions', 'D2', 'met_label'), 'IMPROVEMENT_BEYOND_MATCHED_CONTROLS'),
             (('decisions', 'D2', 'comparisons'), ['START', 'CONT', 'CTRL_FRONTIER']),
             (('decisions', 'D2', 'frontier_paths'), [['CTRL_FRONTIER', 'CTRL']]),
             (('decisions', 'D2', 'selection_data', 'allowed'), ['MODEL_FIT_OOF', 'CALIBRATION', 'MODEL_SELECT']),
             (('registration_status', 'family_pattern'), '(IA|IB|FS_[A-Za-z0-9]+)'),
             (('registration_status', 'starts'), {'FA': 'H_LJ', 'FB': 'NEURAL_RAW'}),
             (('recipient_limit_auc',), 0.02), (('z',), 1.5),
             (('decisions', 'D1', 'confidence_labels'), ['CONFIDENCE_ALL_PASS', 'CONFIDENCE_VIOLATION', 'INCOMPLETE']),
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
    bad = copy.deepcopy(raw); bad['schema'] = 'IPR1-rules-v1'
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
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.SECONDARY == tuple(sorted(set(R.condition_metric(n) for n in
                                                                                     NR.OR.CONDITIONS), key=NR.SECONDARY.index))


# ---------------------------------------------------------------- (9) grid validity
@test
def t_grid_validity_seeds_and_bootstrap():
    assert R.registered_seed(0, 0) == 2026102205 and R.registered_seed(1, 3) == 2026102218
    assert R.registered_boot_seed(0, 'R1') == 2026102400 and R.registered_boot_seed(1, 'R4') == 2026102503
    fs = R.forbidden_seeds()
    assert len(fs) == 140 and NR.MODEL_SEED_BASE + 900 + 19 in fs and not set(seeds_all().values()) & fs
    ch = checks()
    ok = R.grid_validity(reg_seeds(), ch, MIN, di(occupation=5000, income=5000), boot_seeds=boots())
    assert ok['effective'] and ok['label'] == 'READER_SEEDS_EFFECTIVE', ok['errors']
    old = R.grid_validity(reg_seeds(900), ch, MIN, di(income=5000), boot_seeds=boots())
    assert not old['effective'] and any('reuse earlier grid seeds' in e for e in old['errors'])
    shifted = reg_seeds(); shifted['S1k1'] = NR.MODEL_SEED_BASE + 1100 + 11                    # unregistered, not forbidden
    x = R.grid_validity(shifted, ch, MIN, di(income=5000), boot_seeds=boots())
    assert not x['effective'] and any('differ from the registered' in e for e in x['errors'])
    nb = R.grid_validity(reg_seeds(), ch, MIN, di(income=5000), boot_seeds=None)
    assert not nb['effective'] and any('bootstrap seeds not recorded' in e for e in nb['errors'])
    wb = boots(); wb['S1R2'] = 2026102301
    assert not R.grid_validity(reg_seeds(), ch, MIN, di(income=5000), boot_seeds=wb)['effective']
    dup = reg_seeds(); dup['S1k1'] = dup['S1k0']
    assert not R.grid_validity(dup, ch, MIN, di(income=5000), boot_seeds=boots())['effective']
    missing = [c for c in ch if c['recipe'] != 'MLP64']
    assert not R.grid_validity(reg_seeds(), missing, MIN, di(income=5000), boot_seeds=boots())['effective']


@test
def t_grid_validity_explanations():
    ch = checks()
    # (1) few-valued CAND income view (6 inputs): EVERY seed-dependent recipe equal there (NRM1 (a) fails), each differs on the
    # occupation view of the same wire -> explained as FEW_VALUED, counted once, grid valid; strict NRM1 reading invalid
    fv = with_eq(ch, lambda c: c['wire'] == 'CAND' and c['view'] == 'income')
    g = R.grid_validity(reg_seeds(), fv, MIN, di(income=6), boot_seeds=boots())
    assert g['effective'] and 'FEW_VALUED_SEED_EQUALITIES_COUNTED_ONCE' in g['flags'], g['errors']
    assert g['strict_nrm1']['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    assert all(i['explanation'] == 'FEW_VALUED' for i in g['explained_equalities']['items'])
    k = 'CAND|income|MLP64|k0k1'
    assert g['explained_equalities']['n_eff_implied'][k]['n_eff_seed_cells_if_this_recipe_wins'] == 4
    # (2) the same equalities on a view with 1000 distinct inputs (> 320): not few-valued; MLP equalities invalidate
    assert not R.grid_validity(reg_seeds(), fv, MIN, di(income=1000), boot_seeds=boots())['effective']
    # (3) extra-trees support on the minimum set: ET5 equal on CAND income (1000 <= 1600) with (a) + (b) -> explained (ER)
    et = with_eq(ch, lambda c: c['wire'] == 'CAND' and c['view'] == 'income' and c['recipe'] == 'ET5')
    g3 = R.grid_validity(reg_seeds(), et, MIN, di(income=1000), boot_seeds=boots())
    assert g3['effective'] and {i['explanation'] for i in g3['explained_equalities']['items']} == {'ER_EXTRA_TREES_SUPPORT'}
    assert not R.grid_validity(reg_seeds(), et, MIN, di(income=2000), boot_seeds=boots())['effective']       # > 1600
    # (4) a non-extra-trees recipe equal on a continuous minimum-set view: invalid
    ml = with_eq(ch, lambda c: c['wire'] == 'CAND' and c['view'] == 'income' and c['recipe'] == 'MLP64')
    assert not R.grid_validity(reg_seeds(), ml, MIN, di(income=1000), boot_seeds=boots())['effective']
    # (5) off the minimum set NRM1 (iii) explains it
    assert R.grid_validity(reg_seeds(), ml, ['START'], di(income=1000), boot_seeds=boots())['effective']
    # (6) the recipe never differs anywhere in that version / repetition / seed pair: the seed is not shown effective -> invalid
    dead = with_eq(ch, lambda c: c['recipe'] == 'MLP64' and c['v'] == 0 and c['R'] == 'R1')
    assert not R.grid_validity(reg_seeds(), dead, MIN, di(income=6, occupation=6, pair=6), boot_seeds=boots())['effective']
    # (7) a distinct-input count not recorded: never few-valued
    assert not R.grid_validity(reg_seeds(), fv, MIN, {}, boot_seeds=boots())['effective']


@test
def t_seed_pairs_judged_separately():
    ch = checks(k_pair=(0, 1)) + checks(k_pair=(2, 3))
    eq = with_eq(ch, lambda c: c.get('k_pair') == [2, 3] and c['wire'] == 'CAND' and c['view'] == 'income'
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
    ident = splits(overlap=1.0)
    assert R.refit_effectiveness(ident, seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4))['label'] == \
        'REFIT_NOT_GENUINELY_EFFECTIVE'
    none = R.refit_effectiveness(splits(('R1', 'R2')), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots())
    assert none['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and 'NO_ROBUSTNESS_SPLIT' in none['reasons']
    bad = with_eq(ch, lambda c: c.get('k_pair') == [2, 3] and c['wire'] == 'CAND' and c['view'] == 'pair'
                  and c['recipe'] == 'MLP6432')
    assert R.refit_effectiveness(splits(), seeds_all(), bad, MIN, di(income=1000), boot_seeds=boots(reps4))['label'] == \
        'REFIT_NOT_GENUINELY_EFFECTIVE'


# ---------------------------------------------------------------- (4) brackets and fineness
@test
def t_brackets_equal_nrm_rule():
    r = rec()
    b = R.brackets(r['costs_vs'], r['paths'], origin='SUCC')['CTRL']
    pt = {'id': 'CAND', 'costs_vs': r['costs_vs'], 'grid_vs': r['grid']}
    f = NR.frontier_comparison(pt, [{'id': i} for i in ['SUCC'] + CTRL])
    assert b['status'] == 'BRACKETED' and b['bracket'] == f['bracketing_control'] == 'C2' and b['path'] == ['SUCC', 'C1', 'C2']
    r2 = rec(); r2['costs_vs']['C2'] = costs(0.0005)
    assert R.brackets(r2['costs_vs'], r2['paths'], origin='SUCC')['CTRL']['status'] == 'NO_BRACKET'
    r3 = rec(); r3['costs_vs']['SUCC'] = costs(-0.0005)
    assert R.brackets(r3['costs_vs'], r3['paths'], origin='SUCC')['CTRL']['bracket'] == 'SUCC'
    assert R.brackets(r['costs_vs'], {'CTRL': None})['CTRL']['status'] == 'PATH_NOT_GIVEN'
    del r['costs_vs']['C1']
    assert R.brackets(r['costs_vs'], r['paths'], origin='SUCC')['CTRL']['status'] == 'INCOMPLETE'
    assert R.control_path(['C1', 'C2', 'C1'], {'C1': 'SUCC'}, 'SUCC') == ['SUCC', 'C2']


def sp(cand=0.010, lower=0.008, upper=0.0135, se=0.001, mx=0.03, grid=None, data='MODEL_FIT_OOF', key='CTRL'):
    x = {'cand': cand, 'lower': lower, 'upper': upper, 'upper_se': se, 'max': mx}
    if grid is not None:
        x['grid'] = grid
    out = {v: dict(x) for v in NR.VERSIONS}
    if data is not None:
        out['selection_data'] = data
    return {key: out}


@test
def t_grid_fineness():
    assert R.grid_fineness(sp())['per_path']['CTRL'] == {0: 'FINE', 1: 'FINE'}
    assert R.grid_fineness(sp(upper=0.020))['per_path']['CTRL'][0] == 'COARSE_ABOVE'
    assert R.grid_fineness(sp(lower=0.002))['per_path']['CTRL'][0] == 'COARSE_BELOW'
    assert R.grid_fineness(sp(mx=0.011))['flags'] == ['CONTROL_PATH_MAY_NOT_BRACKET_CTRL']
    assert R.grid_fineness(sp(lower=0.011))['per_path']['CTRL'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'
    assert R.grid_fineness(sp(cand=-0.001))['per_path']['CTRL'][0] == 'NOT_APPLICABLE_CAND_NOT_ABOVE_START'
    g = [{'spend': 0.004, 'se': 0.001}, {'spend': 0.009, 'se': 0.001}, {'spend': 0.0135, 'se': 0.001}, {'spend': 0.03, 'se': 0.001}]
    assert R.grid_fineness(sp(lower=0.009, grid=g))['per_path']['CTRL'][0] == 'FINE'
    assert R.grid_fineness(sp(lower=0.004, grid=g))['per_path']['CTRL'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'   # not the largest
    g2 = g + [{'spend': 0.0125, 'se': 0.001}]                                                                 # 0.0125 >= 0.012
    assert R.grid_fineness(sp(lower=0.009, grid=g2))['per_path']['CTRL'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'
    assert R.grid_fineness({'CTRL': {0: {'cand': 0.01}, 'selection_data': 'MODEL_FIT_OOF'}})['flags'] == \
        ['CONTROL_GRID_FINENESS_INCOMPLETE_CTRL']
    assert R.grid_fineness({'CTRL': None})['per_path']['CTRL'] == 'NOT_GIVEN'
    # declared selection data: MODEL_FIT out-of-fold, or CALIBRATION only when START is in-sample on MODEL_FIT; never MODEL_SELECT
    ms = R.grid_fineness(sp(data='MODEL_SELECT'))
    assert ms['per_path']['CTRL'][0] == 'SELECTION_DATA_NOT_ALLOWED' and ms['flags'] == ['CONTROL_SELECTION_DATA_NOT_ALLOWED_CTRL']
    assert R.grid_fineness(sp(data='CALIBRATION'))['per_path']['CTRL'][1] == 'SELECTION_DATA_NOT_ALLOWED'
    cal = R.grid_fineness(sp(data='CALIBRATION'), start_in_sample_on_model_fit=True)
    assert cal['per_path']['CTRL'] == {0: 'FINE', 1: 'FINE'} and cal['selection_data'] == {'CTRL': 'CALIBRATION'}
    nr = R.grid_fineness(sp(data=None))
    assert nr['per_path']['CTRL'][0] == 'INCOMPLETE' and nr['flags'] == ['CONTROL_SELECTION_DATA_NOT_RECORDED_CTRL']
    both = R.grid_fineness({**sp(), **sp(upper=0.020, key='ALT')})
    assert both['per_path']['ALT'][0] == 'COARSE_ABOVE' and both['flags'] == ['CONTROL_GRID_COARSE_ALT']
    assert R.selection_data_error('MODEL_FIT_OOF', None) is None and R.selection_data_error('MODEL_SELECT', True) == 'NOT_ALLOWED'
    assert R.selection_data_error('CALIBRATION', True) is None and R.selection_data_error(None, True) == 'NOT_RECORDED'


# ---------------------------------------------------------------- (2) D1
@test
def t_d1_point_confidence_reserve_separate():
    r = rec()
    d = R.d1(r)
    assert d['label'] == 'TASK_FEASIBLE' and d['met'] and d['start'] == 'POINT_ELIGIBLE'
    cs = R.confidence_conjunction(r['start']['evals'])                     # SUCC-like: the men's guard is unresolved
    assert cs['label'] == 'CONFIDENCE_UNRESOLVED' and cs['unresolved'] == ['v0|guard|sex0', 'v1|guard|sex0'] and not cs['violations']
    assert d['confidence']['start']['label'] == 'CONFIDENCE_UNRESOLVED' and not d['reserve']['start']['qualified']
    assert R.confidence_conjunction(NT.evals())['label'] == 'CONFIDENCE_ALL_PASS'
    bad = NT.evals(occ={('U0', 'nll_excess'): 0.02})                     # resolved beyond the cap
    cb = R.confidence_conjunction(bad)
    assert cb['label'] == 'CONFIDENCE_VIOLATION' and 'v0|occupation|U0|nll_excess' in cb['violations']
    edge = NT.evals(occ={('U0', 'nll_excess'): 0.0105})                  # point failure within 2 SE: unresolved, never a pass
    ce = R.confidence_conjunction(edge)
    assert ce['label'] == 'CONFIDENCE_UNRESOLVED' and 'v1|occupation|U0|nll_excess' in ce['unresolved']
    r2 = rec(cand_ev=edge)
    d2 = R.d1(r2)
    assert d2['label'] == 'CAND_INFEASIBLE' and d2['confidence']['cand']['label'] == 'CONFIDENCE_UNRESOLVED'
    # MC SE in quadrature: 0.0040 + 2 * 0.0004 = 0.0048 passes; with se_mc 0.0004 the bound is 0.00513 (unresolved)
    ev = patch_se(NT.evals(guards={'sex0': 0.0040, 'sex1': -0.0049}), 'guard|sex0', 0.0004)
    assert R.confidence_conjunction(ev)['label'] == 'CONFIDENCE_ALL_PASS'
    assert R.confidence_conjunction(patch_se(ev, 'guard|sex0', 0.0004, se_mc=0.0004))['label'] == 'CONFIDENCE_UNRESOLVED'
    nose = copy.deepcopy(ev); nose[0]['guards'][0]['se'] = None
    assert R.confidence_conjunction(nose)['label'] == 'INCOMPLETE'
    r3 = rec(); r3['cand'] = {'status': 'NO_FEASIBLE_STRENGTH'}
    assert R.d1(r3)['label'] == 'NO_FEASIBLE_STRENGTH'
    r4 = rec(); r4['cand']['status'] = None
    assert R.d1(r4)['label'] == 'INCOMPLETE'
    r5 = rec(); r5['start']['evals'] = bad
    assert 'START_POINT_INELIGIBLE' in R.d1(r5)['flags']


# ---------------------------------------------------------------- (3) UM
@test
def t_um_margin_recovered_and_binding():
    r = rec()
    u = R.um(r)
    assert u['label'] == 'MARGIN_RECOVERED' and u['met'], u
    assert u['binding'] == {0: 'guard|sex0', 1: 'guard|sex0'} and u['binding_metric'][0] == 'occupation|men_brier'
    assert abs(u['binding_difference']['pooled'] + 0.0014) < 1e-12 and u['binding_difference']['resolved_improvement']
    assert u['binding_confidence']['start'] == {0: 'UNRESOLVED', 1: 'UNRESOLVED'}
    assert u['whole_release']['point_margin']['not_below_start'] and 'preparatory' in u['statement']
    b, _ = R.binding_condition(neural_raw_evals())
    assert b == {0: 'guard|sex0', 1: 'income|U0|brier_excess'} == R.START_BINDING['NEURAL_RAW']
    rb = rec(family='FB')
    rb['costs_vs']['NEURAL_RAW'] = costs(0.003, {**men(-0.001), 'income|brier': {'value': -0.001, 'se': 0.0004}})
    ub = R.um(rb)
    assert ub['label'] == 'MARGIN_RECOVERED' and 'BINDING_CONDITION_DIFFERS_BY_VERSION' in ub['flags']
    assert ub['binding_metric'] == {0: 'occupation|men_brier', 1: 'income|brier'}
    for name in NR.OR.CONDITIONS:
        assert R.condition_metric(name) in NR.SECONDARY


@test
def t_um_labels():
    assert R.um(rec(men_diff=-0.0005))['label'] == 'MARGIN_NOT_RESOLVED'                 # upper -0.0005 + 0.0008 > 0
    r = rec(); r['costs_vs']['SUCC'] = costs(0.003, {'occupation|men_brier': {'value': -0.003, 'se': 0.0004}})
    for v in (1,):
        r['costs_vs']['SUCC'][v]['occupation|men_brier'] = {'value': 0.0001, 'se': 0.0004}
    assert R.um(r)['label'] == 'MARGIN_NOT_RESOLVED'                                     # pooled resolved, one version > 0
    assert R.um(rec(start_pair=-0.03))['label'] == 'MARGIN_GAIN_PROFILE_NOT_RETAINED'      # pair 0.03 more recoverable
    r2 = rec(); r2['grid']['SUCC'] = NT.grid_entry(-0.008, se=0.001)                      # within 0.010, resolved worse
    u2 = R.um(r2)
    assert u2['label'] == 'MARGIN_GAIN_PROFILE_NOT_RETAINED' and u2['recovery_profile']['pair']['resolved_worsening']
    rp = rec(); rp['grid']['SUCC'] = NT.grid_entry(-0.012, se=0.01)                       # 0.012 > 0.010, not resolved
    up = R.um(rp)
    assert up['label'] == 'MARGIN_GAIN_PROFILE_NOT_RETAINED' and not up['recovery_profile']['pair']['resolved_worsening']
    r3 = rec(); r3['grid']['SUCC'] = NT.grid_entry(0.0, inc=0.02)                          # income view worse by 0.02
    assert R.um(r3)['label'] == 'MARGIN_GAIN_PROFILE_NOT_RETAINED'
    sh = rec(cand_ev=succ_evals(sex0=0.0030, sex1=0.0046))                               # women's guard now tighter
    us = R.um(sh)
    assert us['label'] == 'MARGIN_GAIN_SHIFTED_ELSEWHERE' and 'WHOLE_RELEASE_POINT_MARGIN_BELOW_START' in us['flags']
    assert R.um(rec(grid_valid=False))['label'] == 'NOT_COMPUTED_GRID_INVALID'
    assert R.um(rec(grid_valid=None, men_diff=0.0))['label'] == 'MARGIN_NOT_RESOLVED'     # needs no grid
    bad = rec(); bad['start']['binding'] = {0: 'guard|sex1', 1: 'guard|sex0'}
    assert R.um(bad)['label'] == 'INCOMPLETE'
    moved = rec(); moved['start']['evals'] = NT.evals(); moved['start']['binding'] = R.binding_condition(NT.evals())[0]
    assert R.um(moved)['label'] == 'INCOMPLETE'                                           # disagrees with the frozen anchor table
    nc = rec(); del nc['costs_vs']['SUCC'][0]['occupation|women_brier']
    assert R.um(nc)['label'] == 'INCOMPLETE'
    inf = rec(cand_ev=NT.evals(occ={('U0', 'nll_excess'): 0.011}))
    assert R.um(inf)['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    al = rec(); al['cand']['every_nonzero_strength_alias_of_start'] = True
    assert R.um(al)['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    worse = rec(); worse['costs_vs']['SUCC'] = costs(0.003, {**men(-0.0014), 'income|brier': {'value': 0.002, 'se': 0.0004}})
    assert 'CONDITION_RESOLVED_WORSE_income|brier' in R.um(worse)['flags']


# ---------------------------------------------------------------- (4) D2
@test
def t_d2_met_and_structure():
    d = R.d2(rec())
    assert d['label'] == R.D2_MET and d['met'] and d['gating'] == ['START', 'CTRL_FRONTIER'], d
    c = d['comparisons']['CTRL_FRONTIER']
    assert c['point'] == 'C2' and c['path'] == ['SUCC', 'C1', 'C2'] and c['pass']
    assert abs(c['residual_spend_mismatch']['at_bracket']['value'] - (-0.001)) < 1e-12
    assert d['start'] == 'SUCC' and d['scope'] == R.D2_SCOPE


@test
def t_d2_partials_no_bracket_and_origin():
    d = R.d2(rec(ctrl_pair=0.0))
    assert d['label'] == 'PARTIAL_START_ONLY' and 'task spending' in d['statement']
    assert R.d2(rec(start_pair=0.0, ctrl_pair=0.0))['label'] == 'NOT_SEPARATED'
    assert R.d2(rec(start_pair=0.0))['label'] == 'PARTIAL_CTRL_FRONTIER_ONLY'
    nb = rec(); nb['costs_vs']['C2'] = costs(0.0005)
    d2 = R.d2(nb)
    c = d2['comparisons']['CTRL_FRONTIER']
    assert d2['label'] == 'PARTIAL_START_ONLY' and c['label'] == 'NOT_SEPARATED' and c['reason'] == 'NO_BRACKET'
    assert 'NO_BRACKET_CTRL' in d2['flags'] and 'never evidence' in c['statement']
    og = rec(); og['costs_vs']['SUCC'] = costs(-0.0005, men(-0.0014)); og['costs_vs']['C2'] = costs(-0.002)
    d3 = R.d2(og)
    assert d3['comparisons']['CTRL_FRONTIER']['point'] == 'SUCC' and 'CTRL_BRACKET_IS_ORIGIN' in d3['flags']
    assert d3['label'] == R.D2_MET
    ali = rec(); ali['alias_of'] = {'C1': 'START'}
    assert R.d2(ali)['comparisons']['CTRL_FRONTIER']['path'] == ['SUCC', 'C2']
    idn = rec(); idn['paths']['CTRL'] = []
    assert 'CTRL_AT_ORIGIN' in R.d2(idn)['flags']


@test
def t_d2_gate_domination_targeting():
    g = rec(); g['grid']['C2'] = NT.grid_entry(0.02, inc=0.02)
    d = R.d2(g)
    assert d['label'] == 'PARTIAL_START_ONLY' and 'RECIPIENT_GATE_FAILED_VS_C2' in d['flags']
    dom = rec(); dom['grid']['C1'] = NT.grid_entry(-0.02)
    d2 = R.d2(dom)
    assert d2['label'] == 'PARTIAL_START_ONLY' and 'DOMINATED_BY_CHEAPER_C1' in d2['flags']
    sec = rec(); sec['costs_vs']['C2'] = costs(-0.001, {'occupation|men_brier': {'value': 0.004, 'se': 0.001}})
    assert R.d2(sec)['comparisons']['CTRL_FRONTIER']['frontier_label'] == 'SEPARATED_ON_PRIMARY_COST_ONLY'
    tg = rec(); tg['nolookup']['C2'] = {'cells': NT.reg(lambda c: -0.01)}
    d3 = R.d2(tg)
    assert d3['label'] == 'READER_TARGETED' and not d3['met'] and 'READER_TARGETED_CTRL_FRONTIER' in d3['flags']
    miss = rec(); del miss['nolookup']['C2']
    assert R.d2(miss)['label'] == 'INCOMPLETE'
    enh = rec(); enh['enhanced'] = {'C2': {'cells': NT.reg(lambda c: -0.001)}}
    d4 = R.d2(enh)
    assert d4['label'] == R.D2_MET and 'READER_SENSITIVE_ENHANCED_C2' in d4['flags']           # reported only


@test
def t_d2_cont():
    d = R.d2(rec(cont=0.03))
    assert d['label'] == R.D2_MET and d['gating'] == ['START', 'CONT', 'CTRL_FRONTIER']
    d2 = R.d2(rec(cont=0.005))                                                                 # not resolved vs CONT
    assert d2['label'] == 'PARTIAL_START+CTRL_FRONTIER_ONLY' and 'retraining alone' in d2['statement']
    beside = rec(); beside['grid']['CONT'] = NT.grid_entry(0.0)                                # given, not declared
    d3 = R.d2(beside)
    assert d3['label'] == R.D2_MET and 'CONT_NOT_SEPARATED_REPORTED_BESIDE' in d3['flags']
    assert d3['comparisons']['CONT']['gating'] is False
    nr = rec(cont=0.03); del nr['grid']['CONT']; nr['controls_not_run'] = {'CONT': True}
    assert R.d2(nr)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    mi = rec(cont=0.03); del mi['grid']['CONT']
    assert R.d2(mi)['label'] == 'INCOMPLETE'
    o = rec(cont=0.03); o['ctrl_origin'] = 'CONT'; o['costs_vs']['CONT'] = costs(0.002)
    d4 = R.d2(o)
    assert d4['comparisons']['CTRL_FRONTIER']['path'] == ['CONT', 'C1', 'C2'] and d4['label'] == R.D2_MET
    bad = rec(); bad['ctrl_origin'] = 'CONT'                                                  # CONT origin, CONT not declared
    bad['grid']['CONT'] = NT.grid_entry(0.03); bad['costs_vs']['CONT'] = costs(0.002)
    db = R.d2(bad)
    assert db['label'] == 'INCOMPLETE' and 'ctrl_origin' in db['error']
    nd = rec(); del nd['cont_declared']
    assert R.d2(nd)['label'] == 'INCOMPLETE'


@test
def t_d2_alt_path():
    d = R.d2(rec(alt=0.02))
    assert d['label'] == R.D2_MET and d['gating'] == ['START', 'CTRL_FRONTIER', 'ALT_FRONTIER'] and d['scope'] == R.D2_SCOPE_ALT
    a = d['comparisons']['ALT_FRONTIER']
    assert a['point'] == 'A2' and a['path'] == ['SUCC', 'A1', 'A2'] and a['pass'] and a['gating']
    assert d['selection_data'] == {'CTRL': 'MODEL_FIT_OOF', 'ALT': 'MODEL_FIT_OOF'}
    d2 = R.d2(rec(alt=0.0))                                                       # the start's own operation reproduces the gain
    assert d2['label'] == 'PARTIAL_START+CTRL_FRONTIER_ONLY' and 'own protection operation' in d2['statement']
    assert R.d2(rec(alt=0.02, cont=0.03, ctrl_pair=0.0))['label'] == 'PARTIAL_START+CONT+ALT_FRONTIER_ONLY'
    assert R.d2(rec(alt=0.0, ctrl_pair=0.0, start_pair=0.0))['label'] == 'NOT_SEPARATED'
    nr = rec(alt=0.02); nr['paths']['ALT'] = None; nr['controls_not_run'] = {'ALT': True}
    assert R.d2(nr)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN' and R.d2(nr)['not_run'] == ['ALT']
    mi = rec(alt=0.02); mi['paths']['ALT'] = None
    assert R.d2(mi)['label'] == 'INCOMPLETE'
    bd = rec(alt=0.02); bd['alt_declared'] = 'yes'
    assert R.d2(bd)['label'] == 'INCOMPLETE'
    nb = rec(alt=0.02); nb['costs_vs']['A2'] = costs(0.0005)                      # CAND spends more than every ALT point
    d3 = R.d2(nb)
    assert d3['label'] == 'PARTIAL_START+CTRL_FRONTIER_ONLY' and 'NO_BRACKET_ALT' in d3['flags']
    assert d3['comparisons']['ALT_FRONTIER']['reason'] == 'NO_BRACKET' and 'ALT point' in d3['comparisons']['ALT_FRONTIER']['statement']
    tg = rec(alt=0.02); tg['nolookup']['A2'] = {'cells': NT.reg(lambda c: -0.01)}   # targeting fails at the ALT bracket only
    d4 = R.d2(tg)
    assert d4['label'] == 'READER_TARGETED' and 'READER_TARGETED_ALT_FRONTIER' in d4['flags']
    dm = rec(alt=0.02); dm['grid']['A1'] = NT.grid_entry(-0.02)                   # a cheaper ALT point less recoverable than CAND
    assert 'DOMINATED_BY_CHEAPER_A1' in R.d2(dm)['flags'] and R.d2(dm)['label'] == 'PARTIAL_START+CTRL_FRONTIER_ONLY'
    o = rec(alt=0.02, cont=0.03); o['ctrl_origin'] = 'CONT'; o['costs_vs']['CONT'] = costs(0.002)
    d5 = R.d2(o)                                                                  # ALT's origin stays START
    assert d5['comparisons']['ALT_FRONTIER']['path'][0] == 'SUCC' and d5['comparisons']['CTRL_FRONTIER']['path'][0] == 'CONT'
    # not declared: nothing changes; a given ALT path is reported beside and never gates
    bs = rec(); bs['paths']['ALT'] = list(ALT); bs['costs_vs'].update({'A1': costs(0.0015), 'A2': costs(-0.0008)})
    bs['grid'].update({i: NT.grid_entry(0.0) for i in ALT})
    d6 = R.d2(bs)
    assert d6['label'] == R.D2_MET and d6['gating'] == ['START', 'CTRL_FRONTIER'] and d6['scope'] == R.D2_SCOPE
    assert {'ALT_REPORTED_BESIDE_NOT_DECLARED', 'ALT_NOT_SEPARATED_REPORTED_BESIDE'} <= set(d6['flags'])
    assert d6['comparisons']['ALT_FRONTIER']['gating'] is False and d6['comparisons']['ALT_FRONTIER']['passed_if_it_were_gating'] is False
    plain = R.d2(rec())
    assert 'ALT_FRONTIER' not in plain['comparisons'] and plain['label'] == R.D2_MET
    b = R.brackets(rec(alt=0.02)['costs_vs'], rec(alt=0.02)['paths'], origin='SUCC')
    assert b['ALT']['bracket'] == 'A2' and b['CTRL']['bracket'] == 'C2' and R.brackets({}, {'CTRL': None})['ALT']['status'] == \
        'PATH_NOT_GIVEN'
    dd = R.decisions(rec(family='FB', alt=0.02))
    assert dd['outcome_class']['label'] == 'METHOD_CANDIDATE' and dd['D2']['gating'][-1] == 'ALT_FRONTIER'


@test
def t_d2_selection_data():
    nr = rec(); nr['selection_data'] = {}
    assert R.d2(nr)['label'] == 'INCOMPLETE' and 'not recorded' in R.d2(nr)['error']
    ms = rec(); ms['selection_data'] = {'CTRL': 'MODEL_SELECT'}
    d = R.d2(ms)
    assert d['label'] == 'INCOMPLETE' and 'CONTROL_SELECTION_DATA_NOT_ALLOWED_CTRL' in d['flags'] and 'never MODEL_SELECT' in d['error']
    ca = rec(); ca['selection_data'] = {'CTRL': 'CALIBRATION'}
    assert R.d2(ca)['label'] == 'INCOMPLETE'                                       # START not recorded in-sample on MODEL_FIT
    ca['start_in_sample_on_model_fit'] = True
    d2 = R.d2(ca)
    assert d2['label'] == R.D2_MET and 'SELECTION_DATA_CALIBRATION_CTRL' in d2['flags']
    al = rec(alt=0.02); del al['selection_data']['ALT']
    assert R.d2(al)['label'] == 'INCOMPLETE'
    al2 = rec(alt=0.02); al2['selection_data']['ALT'] = 'MODEL_SELECT'
    assert 'CONTROL_SELECTION_DATA_NOT_ALLOWED_ALT' in R.d2(al2)['flags']
    nd = rec(); nd['selection_data'] = {'CTRL': 'MODEL_FIT_OOF', 'ALT': 'MODEL_SELECT'}   # ALT not declared: not read
    assert R.d2(nd)['label'] == R.D2_MET
    assert R.decisions(ms)['outcome_class']['label'] == 'INCOMPLETE'


@test
def t_d2_gates_and_inputs():
    assert R.d2(rec(grid_valid=None))['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    assert R.d2(rec(grid_valid=False))['label'] == 'NOT_COMPUTED_GRID_INVALID'
    inf = rec(cand_ev=NT.evals(occ={('U0', 'nll_excess'): 0.011}))
    assert R.d2(inf)['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    al = rec(); al['cand']['every_nonzero_strength_alias_of_start'] = True
    assert R.d2(al)['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    nf = rec(); nf['cand'] = {'status': 'NO_FEASIBLE_STRENGTH'}
    assert R.d2(nf)['label'] == 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH'
    ws = rec(); ws['start']['id'] = 'NEURAL_RAW'; ws['costs_vs']['NEURAL_RAW'] = costs(0.003)   # complete inputs, wrong START
    dw = R.d2(ws)
    assert dw['label'] == 'INCOMPLETE' and 'frozen anchor' in dw['error']
    nr = rec(); nr['paths']['CTRL'] = None; nr['controls_not_run'] = {'CTRL': True}
    assert R.d2(nr)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    np_ = rec(); np_['paths']['CTRL'] = None
    assert R.d2(np_)['label'] == 'INCOMPLETE'
    nc = rec(); del nc['costs_vs']['C1']
    assert R.d2(nc)['label'] == 'INCOMPLETE'
    ng = rec(); del ng['grid']['SUCC']
    assert R.d2(ng)['label'] == 'INCOMPLETE'
    al2 = rec(); al2['grid']['SUCC'] = copy.deepcopy(NT.ALIAS)
    assert 'CAND_ALIAS_OF_START' in R.d2(al2)['flags']
    fn = rec(); fn['fineness'] = R.grid_fineness(sp(upper=0.020))
    assert 'CONTROL_GRID_COARSE_CTRL' in R.d2(fn)['flags'] and R.d2(fn)['label'] == R.D2_MET


@test
def t_randomized_mc_through_nr():
    e = NT.entry(NT.reg(lambda c: 0.02), se=0.004, mc=0.003)
    a = NR.aggregate(e)
    assert abs(a['se_mc'] - 0.003) < 1e-12 and abs(a['se_agg'] - 0.005) < 1e-12
    r = rec(); r['grid']['C2'] = {'pair': NT.entry(NT.reg(lambda c: 0.009), se=0.004, mc=0.003),
                                  'income': NT.entry(NT.reg(lambda c: 0.0)), 'occupation': NT.entry(NT.reg(lambda c: 0.0))}
    assert R.d2(r)['label'] == 'PARTIAL_START_ONLY'                         # 0.009 - 2 * 0.005 < 0: the MC SE counts
    r['grid']['C2']['pair'] = NT.entry(NT.reg(lambda c: 0.009), se=0.004)
    assert R.d2(r)['label'] == R.D2_MET                                     # 0.009 - 2 * 0.004 > 0 without it
    x = NR.mc_cell([0.01, 0.012], [0.004, 0.004])
    assert x['draws'] == 2 and x['se_mc'] is not None and NR.mc_cell([0.01], [0.004])['flags'] == ['MC_SE_NOT_ESTIMABLE']


# ---------------------------------------------------------------- (5) D3, D4, D5, D6
@test
def t_d3_d4_lists_and_credit():
    d = R.decisions(rec())
    assert d['D3']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' and d['D3']['passed'] == list(NR.BENCHMARK)
    assert d['D4']['label'] == 'MATERIAL_0020_RESOLVED' and d['D4']['met']
    r = rec(refs=0.0)
    d2 = R.decisions(r)
    assert d2['D3']['label'] == 'NOT_SUPERIOR' and d2['D3']['passed'] == ['SUCC'] and 'H_LJ' in d2['D3']['failing']
    assert d2['D4']['label'] == 'NOT_MET'
    r3 = rec(refs=0.025)
    assert R.decisions(r3)['D4']['label'] == 'MATERIAL_0020_POINT_ONLY'
    rb = rec(); rb['reader_budget'] = {'TWJ': 'READER_BUDGET_ASYMMETRIC'}; rb['nolookup']['TWJ'] = {'cells': NT.reg(lambda c: -0.01)}
    d4 = R.decisions(rb)
    assert d4['D3']['label'] == 'NOT_SUPERIOR' and 'TWJ' in d4['D3']['failing'] and 'TWJ' not in d4['D3']['passed']
    rc = rec(); rc['reader_budget'] = {'H_LJ': 'READER_BUDGET_ASYMMETRIC'}; del rc['nolookup']['H_LJ']
    d5 = R.decisions(rc)
    assert d5['D3']['label'] == 'INCOMPLETE' and d5['D4']['label'] == 'INCOMPLETE'
    assert d['D5']['label'] == 'NOT_APPLICABLE_NOT_JOINT' and d['D6']['label'] == 'NOT_CLAIMED'
    rn = rec(); rn['novelty'] = 'PRIOR_ART_FOUND'
    assert R.decisions(rn)['D6']['label'] == 'PRIOR_ART_FOUND'


# ---------------------------------------------------------------- (6) outcome classes and the three outcomes
def _oc(r):
    return R.decisions(r)['outcome_class']['label']


@test
def t_outcome_classes():
    d = R.decisions(rec())
    assert d['outcome_class']['label'] == 'METHOD_CANDIDATE' and d['registered']
    o = d['outcomes']
    assert o['1_component_gain']['met'] and o['2_strong_references']['met'] and o['3_material_target']['met']
    assert '(1) component gain' in o['statement'] and '(3) material target' in o['statement']
    assert 'POINT_FEASIBLE_NOT_RESERVE_QUALIFIED' in d['outcome_class']['flags']
    assert _oc(rec(refs=0.0)) == 'COMPONENT_GAIN_NOT_COMPETITIVE'
    ra = R.decisions(rec(ctrl_pair=0.0))
    assert ra['outcome_class']['label'] == 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED' and 'D3_WITHOUT_D2' in ra['outcome_class']['flags']
    assert not ra['outcomes']['1_component_gain']['met'] and ra['outcomes']['2_strong_references']['met']
    assert _oc(rec(start_pair=0.0, ctrl_pair=0.0, refs=0.0)) == 'USEFULNESS_MARGIN_ONLY'
    assert _oc(rec(start_pair=0.0, ctrl_pair=0.0, refs=0.0, men_diff=0.0)) == 'FEASIBLE_NO_RELIABLE_GAIN'
    al = rec(); al['cand']['every_nonzero_strength_alias_of_start'] = True
    da = R.decisions(al)
    assert da['outcome_class']['label'] == 'NO_OUTPUT_CHANGE' and 'FAMILY_STOPS_EXACT_ALIAS' in da['outcome_class']['flags']
    assert da['D3']['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    nr = rec(); nr['paths']['CTRL'] = None; nr['controls_not_run'] = {'CTRL': True}
    assert _oc(nr) == 'PROTECTION_NOT_ASSESSED'
    assert _oc(rec(grid_valid=None)) == 'INCOMPLETE'
    bad = NT.evals(occ={('U0', 'nll_excess'): 0.011})
    inf = rec(cand_ev=bad)
    assert _oc(inf) == 'TASK_INFEASIBLE'
    inf['strengths'] = [{'config_id': 'CAND', 'strength': 2.0, 'evals': bad}, {'config_id': 'S1', 'strength': 1.0,
                                                                               'evals': NT.evals()}]
    assert _oc(inf) == 'SELECTED_STRENGTH_INFEASIBLE'
    inf['strengths'] = None
    assert _oc(inf) == 'INCOMPLETE'
    nf = rec(); nf['cand'] = {'status': 'NO_FEASIBLE_STRENGTH'}
    assert _oc(nf) == 'TASK_INFEASIBLE'
    rr = rec(); rr['d2r'] = {'label': 'ROBUSTNESS_NOT_CONFIRMED', 'sign_changes': ['C2'], 'headline_caveat': 'sign change vs C2'}
    dr = R.decisions(rr)
    assert dr['outcome_class']['label'] == 'METHOD_CANDIDATE' and 'D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE' in \
        dr['outcome_class']['flags'] and 'sign change vs C2' in dr['outcomes']['statement']
    ps = rec(cand_ev=succ_evals(sex0=0.0030))
    assert 'POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED' in R.decisions(ps)['outcome_class']['flags']


# ---------------------------------------------------------------- (7) families, registration, campaign summary
@test
def t_families_and_registration():
    for fam in ('IA', 'CTL', 'FC', 'fs_x', None):
        try:
            R.decisions(rec(family='FA') | {'family': fam})
            raise AssertionError(f'family {fam!r} accepted')
        except ValueError:
            pass
    fs = rec(); fs['family'] = 'FS_SWAP2'
    d = R.decisions(fs)
    assert d['adaptive_tag'] == 'REVISED_FS' and d['D2']['label'] == 'REVISED_FS__' + R.D2_MET and not d['registered']
    fs['start']['id'] = 'U0'
    assert R.decisions(fs)['D2']['label_unprefixed'] == 'INCOMPLETE'
    assert R.adaptive_tag(rec(revision=1)) == 'REVISED_R1'
    assert R.adaptive_tag(rec(early=False)) == 'ADAPTIVE_R0_LATE_CARD'
    assert R.adaptive_tag(rec(family='FB')) is None
    late = R.decisions(rec(revision=2, early=False))
    assert late['outcome_class']['label'] == 'ADAPTIVE_R2__METHOD_CANDIDATE' and late['UM']['label'].startswith('ADAPTIVE_R2__')
    db = R.decisions(rec(family='FB', start_pair=0.0, ctrl_pair=0.0, refs=0.0))
    assert db['start'] == 'NEURAL_RAW' and db['registered']


@test
def t_campaign_summary():
    a = R.decisions(rec())
    b = R.decisions(rec(family='FB', start_pair=0.0, ctrl_pair=0.0, refs=0.0))
    c = R.decisions(rec(revision=1))
    s = R.campaign_summary([a, b, c])
    assert set(s['registered']) == {'FA', 'FB'} and s['d2_looks'] == 3 and len(s['d2_met']) == 2
    assert 'D2 was met in 1 of 2 registered family readings and in 1 of 1 non-registered readings (3 D2 looks in total)' in \
        s['statement'] and 'no multiplicity correction' in s['statement'] and len(s['d4_met']) == 2
    try:
        R.campaign_summary([a, a])
        raise AssertionError('two registered readings accepted')
    except ValueError:
        pass
    s2 = R.campaign_summary([b, c])
    assert s2['flags'] == ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS']


# ---------------------------------------------------------------- (8) strongest_nonzero, D2R
def cfg(cid, strength, x, g1=True, alias=None, revision=0, start='SUCC'):
    return {'config_id': cid, 'strength': strength, 'revision': revision, 'start_id': start, 'alias_of': alias,
            'evals': NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.011}), 'pair_vs_start': NT.grid_entry(x)['pair']}


@test
def t_strongest_nonzero():
    s = R.strongest_nonzero([cfg('a', 1.0, 0.01), cfg('b', 2.0, 0.02), cfg('c', 3.0, 0.03, g1=False), cfg('d', 0.0, 0.05),
                             cfg('e', 4.0, 0.04, alias='b')])
    assert s['selected'] == 'b' and s['config_id'] == 'b'
    t = R.strongest_nonzero([cfg('x', 2.0, 0.02), cfg('y', [0.5, 0.5], 0.02)])
    assert t['selected'] == 'y' and t['tied'] == ['y', 'x']
    assert R.strongest_nonzero([cfg('z', 0.0, 0.01)])['label'] == 'NO_FEASIBLE_NONZERO_RELEASE'
    al = cfg('w', 1.0, 0.0); al['pair_vs_start'] = copy.deepcopy(NT.ALIAS['pair'])
    assert R.strongest_nonzero([al])['records'][0]['status'] == 'ALIAS_OF_START'


def rob_entry(reg_x, rob_x, reps=('R3', 'R4')):
    cells, ses = {}, {}
    for v in NR.VERSIONS:
        for r in ('R1', 'R2') + reps:
            for k in (0, 1, 2, 3):
                c = f'S{v}{r}k{k}'
                cells[c] = reg_x if c in NR.REGISTERED else rob_x
                ses[c] = 0.004
    e = {'cells': cells, 'cell_se': ses}
    z = {'cells': {c: 0.0 for c in cells}, 'cell_se': dict(ses)}
    return {'pair': e, 'income': z, 'occupation': copy.deepcopy(z)}


@test
def t_d2r():
    refit = {'label': 'REFIT_GENUINELY_EFFECTIVE', 'counted_once': []}
    base = {'grid_valid': True, 'refit': refit, 'comparators': {'START': 'SUCC', 'CONT': None, 'CTRL': 'C2'},
            'rob': {'SUCC': rob_entry(0.03, 0.03), 'C2': rob_entry(0.02, 0.02), 'NEURAL_RAW': rob_entry(0.03, 0.03),
                    'H_LJ': rob_entry(0.03, 0.03)}}
    d = R.d2r(base)
    assert d['label'] == 'ROBUSTNESS_CONFIRMED' and d['decisive'] == ['SUCC', 'C2'] and not d['sign_changes']
    sc = copy.deepcopy(base); sc['rob']['C2'] = rob_entry(0.02, -0.01)
    d2 = R.d2r(sc)
    assert d2['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d2['sign_changes'] == ['C2'] and 'headline' in d2['headline_caveat']
    sb = copy.deepcopy(base); sb['rob']['H_LJ'] = rob_entry(0.03, -0.01)
    d3 = R.d2r(sb)
    assert d3['label'] == 'ROBUSTNESS_CONFIRMED' and d3['sign_changes'] == ['H_LJ']        # beside: still carried
    nf = copy.deepcopy(base); nf['refit'] = {'label': 'REFIT_NOT_GENUINELY_EFFECTIVE'}
    assert R.d2r(nf)['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    nr = copy.deepcopy(base); del nr['rob']['C2']
    assert R.d2r(nr)['label'] == 'NOT_RUN'
    inc = copy.deepcopy(base); del inc['comparators']['CTRL']
    assert R.d2r(inc)['label'] == 'INCOMPLETE'
    br = copy.deepcopy(base); br['comparators']['CTRL'] = {'bracket': 'C2', 'status': 'BRACKETED'}
    assert R.d2r(br)['decisive'] == ['SUCC', 'C2']
    nb = copy.deepcopy(base); nb['comparators']['CTRL'] = None
    assert R.d2r(nb)['decisive'] == ['SUCC']
    one = copy.deepcopy(base); one['rob'] = {k: rob_entry(0.03, 0.03, reps=('R3',)) for k in base['rob']}
    assert R.d2r(one)['label'] == 'ROBUSTNESS_CONFIRMED'
    al = copy.deepcopy(base); al['alt_declared'] = True
    assert R.d2r(al)['label'] == 'INCOMPLETE'                                      # a declared ALT needs its bracket recorded
    al['comparators']['ALT'] = {'bracket': 'A2', 'status': 'BRACKETED'}
    assert R.d2r(al)['decisive'] == ['SUCC', 'C2', 'A2'] and R.d2r(al)['label'] == 'NOT_RUN'     # A2's entry missing
    al['rob']['A2'] = rob_entry(0.02, -0.01)
    da = R.d2r(al)
    assert da['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and da['sign_changes'] == ['A2']
    nd = copy.deepcopy(base); nd['comparators']['ALT'] = 'A2'                      # not declared: ALT is not decisive
    assert R.d2r(nd)['decisive'] == ['SUCC', 'C2']
    assert R.d2r({**base, 'grid_valid': None})['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'


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
    out = {'module': 'sar.test_rules', 'rules_sha256': R.load_rules()['_sha256'],
           'sar_rules_py_sha256': hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(), 'passed': n_pass,
           'total': len(res), 'cpu_seconds': round(time.process_time() - t0, 3), 'results': res}
    print(json.dumps(out, indent=1))
    return 0 if n_pass == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())

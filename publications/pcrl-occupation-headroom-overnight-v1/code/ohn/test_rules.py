"""Invented-data tests of ohn.rules (no real data, no label, no array). Run through the queue:
python3 -m ohn.jobqueue run --unit OHN_RULES_TEST_V1 --stage selftest --rss-gib 1 -- python3 -m ohn.test_rules"""
from __future__ import annotations
import copy
import json
import math
import sys
import traceback

from . import rules as R

RULES = R.load_rules()
DEFAULT = {'accuracy_loss': (0.0, 0.001), 'nll_excess': (-0.005, 0.001), 'brier_excess': (-0.002, 0.0005)}
GUARD_DEFAULT = (-0.001, 0.0005)


def util(seed, over=None, G=5306):
    """Invented utility dict of one seed: comfortable margins unless overridden {name: (value, se)}."""
    over = over or {}
    clauses = []
    for t in ('income', 'occupation'):
        for r in R.REFS:
            for m in R.METRICS:
                v, se = over.get(f'{t}|{r}|{m}', DEFAULT[m])
                clauses.append({'seed': seed, 'task': t, 'reference': r, 'metric': m, 'value': v, 'cap': R.CAPS[m], 'se': se,
                                'margin': R.CAPS[m] - v, 'point_pass': v <= R.CAPS[m]})
    guards = []
    for g in ('sex0', 'sex1'):
        v, se = over.get(f'guard|{g}', GUARD_DEFAULT)
        guards.append({'seed': seed, 'group': g, 'value': v, 'cap': 0.005, 'se': se, 'margin': 0.005 - v, 'point_pass': v <= 0.005})
    return {'clauses': clauses, 'guards': guards, 'households': G}


def evals(over0=None, over1=None):
    return {0: util(0, over0), 1: util(1, over1)}


def view(vals, ses=0.002):
    cells = {c: (vals[i] if isinstance(vals, (list, tuple)) else vals) for i, c in enumerate(R.CELLS)}
    return {'cells': cells, 'cell_se': {c: ses for c in R.CELLS}}


def grid_entry(pair=0.02, inc=-0.005, occ=-0.002, se=0.002):
    return {'pair': view(pair, se), 'income': view(inc, se), 'occupation': view(occ, se)}


def config(cid, pair=0.02, over0=None, over1=None, entry=1, kind='privacy', fits=2, mc=True, grids=None, **kw):
    c = {'config_id': cid, 'family': 'A', 'kind': kind, 'entry': entry, 'evals': evals(over0, over1),
         'grid': grids or {x: grid_entry(pair) for x in R.REQUIRED}, 'matched_component': {'config_id': cid + '_TASK', 'grid': grid_entry(0.01)} if mc else None,
         'runnable_package': True, 'six_raw_fields': True, 'alias_of': None, 'n_model_fits': fits, 'n_calibrator_fits': 0}
    c.update(kw)
    return c


def expect_raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------------------------------------- tests
def t_rules_file():
    assert RULES['schema'] == 'OHN1-rules-v1' and len(RULES['_sha256']) == 64
    assert RULES['reserve']['multiplier_k'] == 2.0 and RULES['stability']['z'] == 2.0
    assert RULES['nomination']['near_tie_width'] == 0.005
    assert abs(RULES['final_stage']['c1_clause'] - 2.3949990680014097) < 1e-12
    assert set(RULES['reserve']['floor']['se']['0']) == set(R.CONDITIONS)


def t_reserve_met_and_floor():
    res = R.reserve(evals())
    assert res['status'] == 'RESERVE_MET' and res['meets_reserve'] and not res['failed']
    c = res['per_seed'][0]['conditions']['income|U0|accuracy_loss']
    assert c['se_source'] == 'floor' and abs(c['se_used'] - RULES['reserve']['floor']['se']['0']['income|U0|accuracy_loss']) < 1e-15
    assert len(res['ratio_vector']) == 28 and res['ratio_vector'] == sorted(res['ratio_vector'])


def t_reserve_zero_se_uses_floor():
    res = R.reserve(evals({'income|U0|accuracy_loss': (0.0, 0.0)}))
    c = res['per_seed'][0]['conditions']['income|U0|accuracy_loss']
    assert c['own_se_zero'] and c['se_source'] == 'floor' and c['se_used'] > 0 and math.isfinite(c['ratio'])
    assert 's0|income|U0|accuracy_loss' in res['own_se_zero']
    # a value close to the cap with zero own SE is NOT met (the floor bites)
    res2 = R.reserve(evals({'income|U0|accuracy_loss': (0.009, 0.0)}))
    assert res2['status'] == 'RESERVE_ONE_SEED_ONLY' and 's0|income|U0|accuracy_loss' in res2['failed']


def t_reserve_own_se_larger_than_floor():
    res = R.reserve(evals({'occupation|U0|nll_excess': (0.0, 0.006)}))
    c = res['per_seed'][0]['conditions']['occupation|U0|nll_excess']
    assert c['se_source'] == 'own' and c['se_used'] == 0.006 and not c['met']          # 0.010 < 2 x 0.006
    assert res['status'] == 'RESERVE_ONE_SEED_ONLY'


def t_reserve_guard_and_status():
    res = R.reserve(evals({'guard|sex0': (0.003, 0.001)}, {'guard|sex0': (0.003, 0.001)}))   # margin 0.002 < 2 x 0.0018
    assert res['status'] == 'RESERVE_NOT_MET' and res['binding_condition'].endswith('guard|sex0')
    assert res['point_pass_all']                                                            # point pass, reserve not met
    one = R.reserve({0: util(0)})
    assert one['status'] == 'RESERVE_PENDING_SECOND_SEED' and not one['meets_reserve']


def t_reserve_refusals():
    bad_cap = util(0); bad_cap['clauses'][0]['cap'] = 0.02
    assert expect_raises(lambda: R.reserve({0: bad_cap, 1: util(1)}))
    assert expect_raises(lambda: R.reserve(evals({'income|U0|nll_excess': (0.0, -0.001)})))
    floor = copy.deepcopy(RULES['reserve']['floor']['se']); floor['0']['guard|sex1'] = 0.0
    assert expect_raises(lambda: R.reserve(evals(), floor=floor))
    missing = util(0); missing['guards'] = missing['guards'][:1]
    assert expect_raises(lambda: R.reserve({0: missing, 1: util(1)}))
    assert expect_raises(lambda: R.reserve({2: util(0)}))


def t_reserve_nonfinite_fails():
    res = R.reserve(evals({'occupation|Ucal|brier_excess': (float('nan'), 0.001)}))
    c = res['per_seed'][0]['conditions']['occupation|Ucal|brier_excess']
    assert c['nonfinite'] and not c['met'] and not c['point_pass'] and res['status'] == 'RESERVE_ONE_SEED_ONLY'
    res2 = R.reserve(evals({'occupation|Ucal|brier_excess': (0.0, None)}))
    assert not res2['per_seed'][0]['conditions']['occupation|Ucal|brier_excess']['met']


def t_reserve_household_rescaling():
    u0, u1 = util(0, G=1326), util(1, G=1326)
    res = R.reserve({0: u0, 1: u1})
    c = res['per_seed'][0]['conditions']['guard|sex0']
    assert abs(res['per_seed'][0]['floor_scale'] - math.sqrt(5306 / 1326)) < 1e-12
    assert abs(c['floor_se'] - RULES['reserve']['floor']['se']['0']['guard|sex0'] * math.sqrt(5306 / 1326)) < 1e-15


def t_income_accuracy_only_failure():
    res = R.reserve(evals(None, {'income|U0|accuracy_loss': (0.005, 0.002), 'income|Ucal|accuracy_loss': (0.005, 0.002)}))
    assert res['status'] == 'RESERVE_ONE_SEED_ONLY' and res['income_accuracy_only_failure']
    res2 = R.reserve(evals(None, {'income|U0|accuracy_loss': (0.005, 0.002), 'guard|sex0': (0.004, 0.001)}))
    assert not res2['income_accuracy_only_failure']


def t_witness_guard_cannot_be_compensated():
    strong_nll_bad_guard = evals({'occupation|U0|nll_excess': (-0.030, 0.001), 'occupation|Ucal|nll_excess': (-0.030, 0.001),
                                  'guard|sex0': (0.003, 0.001)})
    w = R.witness(strong_nll_bad_guard)
    assert w['label'] == 'OCCUPATION_RESERVE_NOT_MET' and 's0|guard|sex0' in w['occupation_reserve_failed']
    assert len(w['full_constraints']) == 28                                                 # full constraints reported
    sel = R.select_witnesses({'STRONG_NLL': {'evals': strong_nll_bad_guard, 'task_only': True, 'entry': 1},
                              'PLAIN': {'evals': evals(), 'task_only': True, 'entry': 2}})
    assert sel['selected'] == ['PLAIN'] and sel['records']['STRONG_NLL']['selection'] == 'NOT_A_WITNESS'


def t_witness_point_ineligible_and_one_seed():
    w = R.witness(evals({'guard|sex1': (0.0051, 0.0005)}))
    assert w['label'] == 'OCCUPATION_POINT_INELIGIBLE'
    assert R.witness({0: util(0)})['label'] == 'NOT_CONSIDERED_ONE_SEED_ONLY'
    w2 = R.witness(evals({'income|Ucal|brier_excess': (0.006, 0.001)}))
    assert w2['label'] == 'RELEASE_POINT_INELIGIBLE'


def t_witness_ranking_and_cap_of_two():
    good = evals({'guard|sex0': (-0.004, 0.001)}, {'guard|sex0': (-0.004, 0.001)})
    cands = {
        'A_PLAIN': {'evals': evals(), 'task_only': True, 'entry': 5, 'n_model_fits': 2},
        'B_BETTER_GUARD': {'evals': good, 'task_only': True, 'entry': 9, 'n_model_fits': 2},
        'C_PLAIN_SIMPLER': {'evals': evals(), 'task_only': True, 'entry': 7, 'n_model_fits': 1},
        'D_SEX_TRAINED': {'evals': good, 'task_only': False, 'entry': 1},
    }
    sel = R.select_witnesses(cands)
    # the binding occupation-side minimum is the occupation accuracy clause (ratio ~2.82) for all three, equal across them;
    # leximin then reaches the guard ratios, where B is better; A and C tie fully -> simpler C wins
    assert sel['selected'] == ['B_BETTER_GUARD', 'C_PLAIN_SIMPLER'], sel['ranking']
    assert sel['records']['A_PLAIN']['selection'] == 'PROMISING_NOT_SELECTED'
    assert sel['records']['D_SEX_TRAINED']['label'] == 'NOT_TASK_ONLY'


def t_stability_labels():
    z = RULES['stability']['z']
    st = R.stability(grid_entry(pair=0.02, se=0.003))
    assert st['benefit']['label'] == 'STABLE_POSITIVE' and st['benefit_stable']
    vals = [0.02, 0.02, 0.02, -0.001, 0.02, 0.02, 0.02, 0.02]
    assert R.stability({'pair': view(vals), 'income': view(0.0), 'occupation': view(0.0)})['benefit']['label'] == 'SIGN_CHANGES'
    noisy = [0.002, 0.03, 0.004, 0.028, 0.003, 0.03, 0.002, 0.03]       # all positive, large spread
    b = R.stability({'pair': view(noisy, 0.004), 'income': view(0.0), 'occupation': view(0.0)})['benefit']
    assert b['label'] == 'POSITIVE_NOT_STABLE' and b['cells_above'] == 8
    assert all(b['lower_per_seed'][s] <= 0 for s in (0, 1))
    small_vs_se = R.stability({'pair': view(0.005, 0.004), 'income': view(0.0), 'occupation': view(0.0)})['benefit']
    assert small_vs_se['label'] == 'POSITIVE_NOT_STABLE'                   # 0.005 < 2 x 0.004: sampling noise not resolved
    assert R.stability({'pair': view(-0.01), 'income': view(0.0), 'occupation': view(0.0)})['benefit']['label'] == 'NOT_POSITIVE'
    inc = {'pair': {'cells': {c: 0.02 for c in R.CELLS[:7]}, 'cell_se': {c: 0.002 for c in R.CELLS[:7]}}, 'income': view(0.0), 'occupation': view(0.0)}
    assert R.stability(inc)['benefit']['label'] == 'INCOMPLETE'
    noses = {'pair': {'cells': {c: 0.02 for c in R.CELLS}}, 'income': view(0.0), 'occupation': view(0.0)}
    assert R.stability(noses)['benefit']['label'] == 'INCOMPLETE'
    assert z == 2.0


def t_stability_spread_formula():
    vals = [0.010, 0.014, 0.012, 0.016, 0.020, 0.018, 0.022, 0.024]
    b = R.benefit_summary(view(vals, 0.003), 2.0)
    d0, d1 = sum(vals[:4]) / 4, sum(vals[4:]) / 4
    v_rep = (sum((x - d0) ** 2 for x in vals[:4]) + sum((x - d1) ** 2 for x in vals[4:])) / 6
    sd = math.sqrt(v_rep + 0.003 ** 2)
    assert abs(b['sd_total'] - sd) < 1e-15 and abs(b['per_seed_mean'][0] - d0) < 1e-15
    assert b['stable'] == (d0 - 2 * sd > 0 and d1 - 2 * sd > 0)


def t_recipient_point_and_edge():
    st = R.stability(grid_entry(inc=[0.0, 0.0, 0.0, 0.011, 0.0, 0.0, 0.0, 0.0]))
    assert not st['recipient_point_pass'] and st['recipient']['income']['label'] == 'RECIPIENT_POINT_FAILED'
    st2 = R.stability(grid_entry(occ=0.008, se=0.002))                    # every cell <= 0.010 but 0.008 + 2 x 0.002 > 0.010
    assert st2['recipient_point_pass'] and st2['recipient']['occupation']['label'] == 'RECIPIENT_EDGE'
    st3 = R.stability(grid_entry(occ=0.010))                              # inclusive limit
    assert st3['recipient_point_pass']


def t_material_target():
    assert R.stability(grid_entry(pair=0.03, se=0.002))['target']['label'] == 'DEV_GRID_TARGET_STABLE'
    assert R.stability(grid_entry(pair=0.021, se=0.002))['target']['label'] == 'DEV_GRID_TARGET_POINT_ONLY'
    assert R.stability(grid_entry(pair=0.015, se=0.002))['target']['label'] == 'DEV_GRID_TARGET_NOT_MET'


def t_income_branch_trigger():
    sel_yes = {'promising_exists': True}
    sel_no = {'promising_exists': False}
    met = R.reserve(evals())
    t = R.income_branch_trigger(sel_yes, met)
    assert t['leg_B_published_dev'] and not t['leg_A_model_select'] and t['fires']        # registered H3 anchor: leg B true
    assert abs(t['leg_B_detail']['s1|income|U0|accuracy_loss']['ratio'] - 0.82962) < 1e-4
    assert not t['leg_B_detail']['s1|income|U0|accuracy_loss']['meets_r2'] and t['leg_B_detail']['s0|income|U0|accuracy_loss']['meets_r2']
    assert not R.income_branch_trigger(sel_no, met)['fires']
    clean_dev = {'clauses': {'s1|income|U0|accuracy_loss': {'cap': 0.01, 'point': 0.0, 'se_anchor': 0.003, 'se_h_lj': 0.0035}}}
    assert not R.income_branch_trigger(sel_yes, met, clean_dev)['fires']                   # neither leg
    acc_fail = R.reserve(evals(None, {'income|U0|accuracy_loss': (0.005, 0.002), 'income|Ucal|accuracy_loss': (0.005, 0.002)}))
    t2 = R.income_branch_trigger(sel_yes, acc_fail, clean_dev)
    assert t2['leg_A_model_select'] and t2['fires']
    other_fail = R.reserve(evals(None, {'guard|sex0': (0.004, 0.001)}))
    assert not R.income_branch_trigger(sel_yes, other_fail, clean_dev)['fires']


def t_nominate_choice_and_tie_breaks():
    best = config('BEST_SCORE', pair=0.030, over0={'guard|sex0': (0.002, 0.001)}, entry=1)   # higher score, thinner guard
    near = config('NEAR_TIE_MORE_RESERVE', pair=0.027, entry=2)                              # within 0.005, more reserve
    far = config('FAR', pair=0.020, entry=3)                                                 # outside window
    out = R.nominate([best, near, far])
    lab = {r['config_id']: r['label'] for r in out['configurations']}
    # BEST_SCORE's seed-0 men's guard margin 0.003 < 2 x 0.0018 fails the reserve, so it cannot qualify at all
    assert lab['BEST_SCORE'] == 'RESERVE_NOT_MET' and 'EDGE_STABLE_BENEFIT_WITHOUT_RESERVE' in out['configurations'][0]['flags']
    assert out['nominated'] == 'NEAR_TIE_MORE_RESERVE' and lab['FAR'] == 'QUALIFYING_NOT_CHOSEN'
    a = config('A', pair=0.030, over0={'guard|sex0': (0.0015, 0.001)}, entry=1)             # guard ratio ~1.93? -> check reserve
    b = config('B', pair=0.027, entry=2)
    out2 = R.nominate([a, b])
    lab2 = {r['config_id']: r['label'] for r in out2['configurations']}
    assert lab2['A'] in ('QUALIFYING_NOT_CHOSEN', 'RESERVE_NOT_MET', 'NOMINATED_FINALIST')
    # two qualifying configurations in a near tie: the larger whole-release minimum ratio wins over the higher score
    hi = config('HI_SCORE_LOWER_RESERVE', pair=0.030, over0={'occupation|U0|nll_excess': (0.0035, 0.001)}, entry=1)
    lo = config('LO_SCORE_HIGHER_RESERVE', pair=0.026, entry=2)
    out3 = R.nominate([hi, lo])
    assert out3['nominated'] == 'LO_SCORE_HIGHER_RESERVE', [(r['config_id'], r['reserve']['min_ratio']) for r in out3['configurations']]
    # exact reserve tie -> simpler implementation
    s1 = config('COMPLEX', pair=0.028, entry=1, fits=6)
    s2 = config('SIMPLE', pair=0.026, entry=2, fits=2)
    assert R.nominate([s1, s2])['nominated'] == 'SIMPLE'
    # full tie -> earliest entry
    e1 = config('LATE', pair=0.026, entry=9)
    e2 = config('EARLY', pair=0.026, entry=4)
    assert R.nominate([e1, e2])['nominated'] == 'EARLY'


def t_nominate_gates():
    grids = {x: grid_entry(0.02) for x in R.REQUIRED}
    grids['RETIRED_B3CTL'] = grid_entry([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, -0.002])
    unstable = config('UNSTABLE_VS_RETIRED', grids=grids)
    rec_bad = {x: grid_entry(0.02) for x in R.REQUIRED}
    rec_bad['H_LJ'] = grid_entry(0.02, inc=[0.0] * 7 + [0.0105])
    recip = config('RECIPIENT_FAIL', grids=rec_bad)
    point = config('POINT_FAIL', over1={'occupation|Ucal|brier_excess': (0.0051, 0.001)})
    nomc = config('NO_MC', mc=False)
    alias = config('ALIAS', alias_of='SUCC')
    one = config('ONE_SEED'); one['evals'] = {0: util(0)}
    nopkg = config('NO_PACKAGE', runnable_package=False)
    miss = config('MISSING_GRID'); miss['grid'] = {x: grid_entry(0.02) for x in R.REQUIRED[:5]}
    out = R.nominate([unstable, recip, point, nomc, alias, one, nopkg, miss])
    lab = {r['config_id']: r['label'] for r in out['configurations']}
    assert lab == {'UNSTABLE_VS_RETIRED': 'BENEFIT_NOT_STABLE', 'RECIPIENT_FAIL': 'RECIPIENT_POINT_FAILED',
                   'POINT_FAIL': 'UTILITY_POINT_INELIGIBLE', 'NO_MC': 'NO_MATCHED_COMPONENT_COMPARISON', 'ALIAS': 'ALIAS_OF_SUCC',
                   'ONE_SEED': 'NOT_CONSIDERED_ONE_SEED_ONLY', 'NO_PACKAGE': 'NOT_ADMISSIBLE', 'MISSING_GRID': 'NOT_ADMISSIBLE'}, lab
    assert out['outcome'] == 'NO_QUALIFYING_CONFIGURATION' and out['dev_report'] == 'STAYS_CLOSED' and out['nominated'] is None
    u = [r for r in out['configurations'] if r['config_id'] == 'UNSTABLE_VS_RETIRED'][0]
    assert u['outcomes']['reader_stability'] == 'BENEFIT_NOT_STABLE' and u['benefit_not_stable'] == {'RETIRED_B3CTL': 'SIGN_CHANGES'}
    assert u['outcomes']['point'] == 'POINT_ELIGIBLE' and u['outcomes']['reserve'] == 'RESERVE_MET'   # partial passes recorded separately


def t_nominate_control_and_outcomes():
    ctl = config('CONTROL_X', pair=0.03, kind='control', entry=1)
    oth = config('OTHER', pair=0.02, entry=2)
    out = R.nominate([ctl, oth])
    c = [r for r in out['configurations'] if r['config_id'] == 'CONTROL_X'][0]
    assert out['nominated'] == 'CONTROL_X' and 'CONTROL_DESIGNATED_FINALIST' in c['flags']
    assert set(c['outcomes']) == set(RULES['nomination']['separate_outcomes'])
    assert c['outcomes']['material_target'] == 'DEV_GRID_TARGET_STABLE'
    assert c['outcomes']['conditional_confidence'] == 'UTILITY_CONFIDENCE_RESOLVED'
    unres = config('UNRES', over0={'guard|sex0': (-0.001, 0.0025)})            # 0.0068 > 0.005 at z 3.12; reserve uses 0.0025
    r = R.nominate([unres])['configurations'][0]
    assert r['outcomes']['conditional_confidence'].startswith('UTILITY_CONFIDENCE_UNRESOLVED (1')


def t_score_is_worst_required_comparator():
    grids = {x: grid_entry(0.03) for x in R.REQUIRED}
    grids['SUCC'] = grid_entry(0.012, se=0.002)
    r = R.nominate([config('X', grids=grids)])['configurations'][0]
    assert abs(r['score_min_required_mean_benefit'] - 0.012) < 1e-12


def t_learning_curve():
    assert R.learning_curve_flag({0.25: 0.03, 0.5: 0.02, 1.0: 0.02})['flag'] == 'BENEFIT_SHRINKS_WITH_READER_DATA'
    assert R.learning_curve_flag({0.25: 0.02, 0.5: 0.02, 1.0: 0.018})['flag'] == 'NO_SHRINKAGE_FLAG'
    assert R.learning_curve_flag({0.25: 0.02, 1.0: 0.018})['flag'] == 'LEARNING_CURVE_INCOMPLETE'


def t_crossfit_flag():
    assert R.crossfit_gap_flag(0.80, 0.79, 0.003)['flag'] == 'CROSSFIT_DEPLOYED_GAP'
    assert R.crossfit_gap_flag(0.80, 0.799, 0.003)['flag'] == 'NO_CROSSFIT_GAP_FLAG'
    assert R.crossfit_gap_flag(0.80, None, 0.003)['flag'] == 'CROSSFIT_CHECK_INCOMPLETE'


def t_leximin():
    assert R.leximin_cmp([2.0, 3.0], [2.0, 2.5]) == -1 and R.leximin_cmp([1.9, 9.0], [2.0, 2.0]) == 1
    assert R.leximin_cmp([2.0, 3.0], [2.0 + 1e-12, 3.0]) == 0
    assert R.leximin_cmp([-math.inf, 3.0], [0.1, 0.2]) == 1


TESTS = [v for k, v in sorted(globals().items()) if k.startswith('t_') and callable(v)]


def main():
    results = {}
    for t in TESTS:
        try:
            t()
            results[t.__name__] = 'PASS'
        except Exception:                                   # noqa: BLE001 - every failure is reported, none hidden
            results[t.__name__] = 'FAIL: ' + traceback.format_exc(limit=3).strip().splitlines()[-1]
    n_pass = sum(v == 'PASS' for v in results.values())
    print(json.dumps({'rules_sha256': RULES['_sha256'], 'passed': n_pass, 'of': len(results), 'results': results}, indent=1))
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())

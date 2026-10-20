"""Invented-data tests of frc.rules (no real data, no label). Run: python3 -m frc.test_rules (prints JSON; exit 0 iff all pass).

Covers every new function: the rule file and its refusal of altered constants; Q-PE labels (supported, point only, a version
disagreeing in sign, exact alias, missing inputs, share removed); the construction choice (ties, task-sum fallback); QUAL (Q1, Q2,
inclusive ties, every increment); the original capacity gate (reported, never gating) and GATE reproduction; the capacity
cascade 64 -> 128 -> 256 -> repair -> stop and its untriggered-result flags; D1-D6 and the outcome class (partials, aliases,
controls at the identity, missing grid validity); the strongest feasible nonzero release (ties to the smaller lambda); the
genuinely effective refit; reader-seed validity with offsets 300 / 400 / 500 / 600 forbidden and +700 registered; nrm.rules
left unchanged. Helpers for invented utility records and grid entries are imported from nrm.test_rules (unchanged)."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

from nrm import rules as NR
from nrm import test_rules as NT                 # invented-data helpers (util, evals, entry, reg, grid_entry, costs, ALIAS)
from frc import rules as R

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def nested(occ=(-0.004, 0.001), tot=(-0.005, 0.0012), inc=(-0.001, 0.0004), spend_a=0.0107, spend_b1=0.0067, versions=(0, 1),
           fold_offsets=(-0.0005, -0.00025, 0.0, 0.00025, 0.0005)):
    """Invented T1 record: B1 - A paired differences and spends over CONT, per version (dicts, tuples or None per version)."""
    out = {}
    for v in versions:
        pick = (lambda x: x[v] if isinstance(x, dict) else x)
        b = {'occ_nll': pick(occ), 'inc_nll': pick(inc), 'occ_brier': (-0.001, 0.0005), 'inc_brier': (-0.0002, 0.0001),
             'occ_acc': (0.0, 0.001), 'inc_acc': (0.0, 0.0005)}
        if pick(tot) is not None:
            b['tot_nll'] = pick(tot)
        offs = pick(fold_offsets) if fold_offsets is not None else None
        folds = None if offs is None else [{'paired_fold_means': {k: (x[0] + o if x is not None else None) for k, x in b.items()}}
                                           for o in offs]
        out[v] = {'B1-A': b, 'B0-B1': {'occ_nll': (0.0003, 0.0001)}, 'folds': folds,
                  'spend_over_cont': {'A': {'occ_nll': (pick(spend_a), 0.0013)}, 'B1': {'occ_nll': (pick(spend_b1), 0.0012)}}}
    return out


SMALL = (-0.0005, -0.00025, 0.0, 0.00025, 0.0005)          # fold SE 0.000177 < household SE: the household SE binds
WIDE = (-0.008, 0.008, 0.0, -0.004, 0.004)                   # fold SE 0.002828: binds over a household SE of 0.001


def ms(occ=-0.002, inc=-0.0005):
    """Invented T3 paired costs B1 vs A (nrm.evaluate.paired_costs format), per version."""
    def one(v):
        o = occ[v] if isinstance(occ, dict) else occ
        return {'occupation|nll': {'value': o, 'se': 0.001}, 'income|nll': {'value': inc, 'se': 0.0004},
                'total|nll': {'value': o + inc, 'se': 0.0011}, 'occupation|men_brier': {'value': -0.0004, 'se': 0.0003},
                'occupation|women_brier': {'value': -0.0003, 'se': 0.0003}}
    return {v: one(v) for v in NR.VERSIONS}


def cap_rec(qualifies=True, choice_b1=True, reproduced=True, qpe_supported=True):
    """Invented capacity record whose chosen construction does / does not qualify."""
    n = nested() if choice_b1 else nested(occ=(0.001, 0.001), tot=(0.001, 0.001))
    ev_ok = NT.evals()
    ev_bad = NT.evals(occ={('U0', 'nll_excess'): 0.0065})               # point feasible, fails Q2 (0.0065 > 0.006)
    return {'reproduced': reproduced, 'nested': n, 'ms': ms() if qpe_supported else ms(occ={0: -0.001, 1: 0.0002}),
            'evals': {'A': ev_ok if qualifies else ev_bad, 'B1': ev_ok if qualifies else ev_bad}}


# ---------------------------------------------------------------- rule file and the inherited module
@test
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'FRC1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(R.RULES_PATH.read_text())
    for path, val in ((('qual', 'q2_increments', 'guard'), 0.002), (('reader_seeds', 'forbidden_offsets'), [300, 400, 500]),
                      (('original_capacity_gate', 'limits_estimate_plus_2se', 'occ_nll'), 0.005), (('z',), 1.96)):
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
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.load_rules()['_sha256'] == R.NRM1_RULES_SHA256
    R.grid_validity({}, [], [], {})
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.forbidden_seeds() == {
        NR.MODEL_SEED_BASE + o + 10 * v + k for o in (300, 500) for v in NR.VERSIONS for k in range(10)}
    assert R.forbidden_seeds() > NR.forbidden_seeds()                   # strict superset: 400 and 600 added locally
    assert len(R.forbidden_seeds()) == 80 and R.registered_seed(1, 3) == NR.MODEL_SEED_BASE + 713


# ---------------------------------------------------------------- (a) Q-PE
@test
def t_qpe_labels():
    assert R.qpe_label(nested(), ms())['label'] == 'PE_ERROR_SUPPORTED'
    assert R.qpe_label(nested(occ=(-0.001, 0.001)), ms())['label'] == 'PE_ERROR_POINT_ONLY'          # not resolved
    assert R.qpe_label(nested(), ms(occ={0: -0.001, 1: 0.0001}))['label'] == 'PE_ERROR_POINT_ONLY'   # MODEL_SELECT disagrees
    assert R.qpe_label(nested(), ms(occ={0: -0.001, 1: 0.0}))['label'] == 'PE_ERROR_POINT_ONLY'      # MS tie at 0 is not < 0
    sign = nested(occ={0: (-0.006, 0.001), 1: (0.0005, 0.001)})                                      # a version disagrees in sign
    assert R.qpe_label(sign, ms())['label'] == 'PE_ERROR_NOT_SUPPORTED'
    assert R.qpe_label(nested(occ=(0.0, 0.0)), ms())['label'] == 'PE_ERROR_NOT_SUPPORTED'            # exact alias / tie at 0
    assert R.qpe_label(nested(versions=(0,)), ms())['label'] == 'INCOMPLETE'                         # a version missing
    assert R.qpe_label(nested(occ={0: (-0.004, float('nan')), 1: (-0.004, 0.001)}), ms())['label'] == 'INCOMPLETE'
    assert R.qpe_label(nested(), None)['label'] == 'INCOMPLETE'                                      # resolved, MS needed
    assert R.qpe_label(nested(occ=(-0.001, 0.001)), None)['label'] == 'PE_ERROR_POINT_ONLY'          # MS not needed
    assert R.qpe_label(sign, None)['label'] == 'PE_ERROR_NOT_SUPPORTED'
    rec = R.qpe_label(nested(), ms())
    assert rec['reported_beside']['tot_nll']['nested_reading'] == 'RESOLVED_NEGATIVE_BOTH'
    assert rec['reported_beside']['occ_acc']['nested_reading'] == 'NOT_NEGATIVE_SOME_VERSION'
    assert rec['reported_beside']['occupation|men_brier']['model_select_point'][1] == -0.0004
    assert abs(rec['b0_minus_b1']['occ_nll'][0][0] - 0.0003) < 1e-15
    assert rec['occ_nll'][0]['binding'] == 'household' and rec['occ_nll'][0]['se_used'] == 0.001


@test
def t_qpe_fold_se():
    w = R.qpe_label(nested(fold_offsets=WIDE), ms())                # fold SE binds: -0.004 + 2 x 0.002828 > 0 -> not resolved
    assert w['label'] == 'PE_ERROR_POINT_ONLY' and w['occ_nll'][1]['binding'] == 'fold'
    assert abs(w['occ_nll'][0]['se_fold'] - 0.0028284271247461905) < 1e-12 and w['occ_nll'][0]['se_household'] == 0.001
    assert R.qpe_label(nested(fold_offsets=None), ms())['label'] == 'INCOMPLETE'            # fold means absent
    four = nested()
    four[1]['folds'] = four[1]['folds'][:4]
    assert R.qpe_label(four, ms())['label'] == 'INCOMPLETE'                                  # not exactly 5 folds
    nanf = nested()
    nanf[0]['folds'][2]['paired_fold_means']['occ_nll'] = float('nan')
    assert R.qpe_label(nanf, ms())['label'] == 'INCOMPLETE'
    alt = nested(fold_offsets=None)                                                          # the alternative record form
    for v in (0, 1):
        alt[v]['fold_means'] = {'occ_nll': [-0.004 + o for o in SMALL]}
    assert R.qpe_label(alt, ms())['label'] == 'PE_ERROR_SUPPORTED'
    assert 'FOLD_MEANS_MISSING_V0' in R.qpe_label(alt, ms())['reported_beside']['tot_nll']['flags']
    # a sign-disagreeing version is NOT_SUPPORTED whatever the SE
    assert R.qpe_label(nested(occ={0: (-0.006, 0.001), 1: (0.0005, 0.001)}, fold_offsets=WIDE), ms())['label'] == \
        'PE_ERROR_NOT_SUPPORTED'


@test
def t_share_removed():
    s = R.share_removed(nested(spend_a=0.010, spend_b1=0.006))
    assert abs(s['per_version'][0]['share'] - 0.4) < 1e-12 and abs(s['pooled'] - 0.4) < 1e-12
    s = R.share_removed(nested(spend_a={0: 0.010, 1: 0.014}, spend_b1={0: 0.006, 1: 0.007}))
    assert abs(s['pooled'] - (0.0055 / 0.012)) < 1e-12
    s = R.share_removed(nested(spend_a={0: -0.001, 1: 0.01}, spend_b1=0.0))
    assert s['per_version'][0]['share'] is None and s['per_version'][0]['reason'] == 'NOT_DEFINED_NONPOSITIVE_A_SPEND'
    assert R.share_removed({0: {}, 1: {}})['pooled'] is None


# ---------------------------------------------------------------- (b) construction choice
@test
def t_construction_choice():
    assert R.construction_choice(nested())['choice'] == 'B1'
    assert R.construction_choice(nested(occ=(-0.0001, 0.003)))['choice'] == 'B1'          # point rule, not resolved
    assert R.construction_choice(nested(tot={0: (-0.001, 0.001), 1: (0.0002, 0.001)}))['choice'] == 'A'
    assert R.construction_choice(nested(occ={0: (-0.001, 0.001), 1: (0.0, 0.0)}))['choice'] == 'A'     # tie at 0 -> A
    fb = R.construction_choice(nested(tot=None))
    assert fb['choice'] == 'B1' and fb['flags'] == ['TOTAL_FROM_TASK_SUM']
    assert R.construction_choice(nested(tot=None, inc=(0.005, 0.001)))['choice'] == 'A'   # occ < 0 but inc + occ > 0
    assert R.construction_choice(nested(versions=(1,)))['label'] == 'INCOMPLETE'


# ---------------------------------------------------------------- (c) QUAL
@test
def t_qual():
    q = R.qual(NT.evals())
    assert q['label'] == 'QUALIFIED' and q['reasons'] == []
    q = R.qual(NT.evals(occ={('U0', 'nll_excess'): 0.0061}))
    assert q['label'] == 'NOT_QUALIFIED' and q['per_version'][0]['q2_failed'] == ['occupation|U0|nll_excess']
    assert q['per_version'][0]['q1'] is True and 'Q2_FAILED_V0' in q['reasons']
    tie = 0.010 - 0.008                                                                 # inclusive: value == cap - d passes
    assert R.qual(NT.evals(occ={('Ucal', 'nll_excess'): tie}))['label'] == 'QUALIFIED'
    for cond, inc in (('income|U0|accuracy_loss', 0.004), ('income|Ucal|nll_excess', 0.004), ('income|U0|brier_excess', 0.002),
                      ('occupation|U0|accuracy_loss', 0.006), ('occupation|Ucal|brier_excess', 0.003), ('guard|sex0', 0.003),
                      ('guard|sex1', 0.003), ('occupation|U0|nll_excess', 0.008)):           # the S2 budget
        assert R.q2_increment(cond) == inc, cond
    over = NT.evals(guards={'sex1': 0.0021})                                           # 0.0021 > 0.005 - 0.0030
    assert R.qual(over)['per_version'][1]['q2_failed'] == ['guard|sex1']
    # disclosure arithmetic: a v1 men's guard of 0.00255 (A at (16, 64), published) fails the S2 form, passes the old S1 form
    pub = {0: NT.util(0), 1: NT.util(1, guards={'sex0': 0.00255})}
    q = R.qual(pub)
    assert q['label'] == 'NOT_QUALIFIED' and q['per_version'][1]['q2_failed'] == ['guard|sex0'] and q['per_version'][1]['q1']
    assert 0.00255 + 0.0015 <= 0.005 < 0.00255 + R.Q2_INCREMENTS['guard']
    q1 = R.qual(NT.evals(inc={('U0', 'accuracy_loss'): 0.0101}))                       # G1 fails at the point
    assert q1['label'] == 'NOT_QUALIFIED' and 'Q1_FAILED_V0' in q1['reasons'] and 'Q2_FAILED_V0' in q1['reasons']
    one = {0: NT.util(0)}
    assert R.qual(one)['label'] == 'INCOMPLETE'
    recap = NT.evals()
    recap[0]['clauses'][0]['cap'] = 0.02                                               # a re-capped clause is refused
    assert R.qual(recap)['label'] == 'INCOMPLETE'


# ---------------------------------------------------------------- (d) original capacity gate (reported) and reproduction
GATE_V0 = {'gate': 'FAIL', 'nested_means': {'CONT': {'occ_nll': 1.2133, 'inc_nll': 0.5175}, 'PRC_TWIN': {'occ_nll': 1.2240,
                                                                                                       'inc_nll': 0.5190}},
           'spend_twin_over_cont': {'inc_acc': [0.0012, 0.00088], 'inc_nll': [0.00153, 0.000365], 'occ_acc': [0.0026, 0.0019],
                                    'occ_nll': [0.01068, 0.00129]}}


@test
def t_capacity_gate_reported():
    sp = {k: tuple(x) for k, x in GATE_V0['spend_twin_over_cont'].items()}
    g = R.capacity_gate({0: sp, 1: sp})
    assert g['label'] == 'ORIGINAL_CAPACITY_GATE_FAIL' and g['gating'] is False
    assert g['per_version'][0]['occ_nll']['pass'] is False and g['per_version'][0]['inc_nll']['pass'] is True
    ok = {'occ_nll': (0.002, 0.001), 'inc_nll': (0.001, 0.0005), 'occ_acc': (0.001, 0.001), 'inc_acc': (0.001, 0.001)}
    assert R.capacity_gate({0: ok, 1: ok})['label'] == 'ORIGINAL_CAPACITY_GATE_PASS'          # 0.002 + 0.002 = 0.004 inclusive
    assert R.capacity_gate({0: ok, 1: sp})['label'] == 'ORIGINAL_CAPACITY_GATE_FAIL'
    assert R.capacity_gate({0: ok})['label'] == 'INCOMPLETE'
    # never gating: a failing original gate (both constructions, both versions) does not change the capacity decision
    rec = cap_rec()
    for v in NR.VERSIONS:
        for c in ('A', 'B1'):
            rec['nested'][v]['spend_over_cont'][c] = dict(sp)
    assert R.capacity_gate({v: rec['nested'][v]['spend_over_cont']['B1'] for v in NR.VERSIONS})['label'] == \
        'ORIGINAL_CAPACITY_GATE_FAIL'
    d = R.capacity_decision({'16x64': rec})
    assert d['state'] == 'PROTECT' and d['capacity'] == '16x64' and R.d1(d)['label'] == 'REPAIRED_TASK_ONLY_RELEASE'


@test
def t_gate_reproduction():
    rc = {'nested_means': copy.deepcopy(GATE_V0['nested_means']),
          'spend_over_cont': {k: tuple(x) for k, x in GATE_V0['spend_twin_over_cont'].items()}}
    rc['nested_means']['CONT']['occ_nll'] += 5e-13
    assert R.gate_reproduction(rc, GATE_V0, 'PRC_TWIN')['label'] == 'REPRODUCED'
    rc['nested_means']['CONT']['occ_nll'] += 1e-9
    assert R.gate_reproduction(rc, GATE_V0, 'PRC_TWIN')['label'] == 'NOT_REPRODUCED'
    del rc['nested_means']['PRC_TWIN']
    assert R.gate_reproduction(rc, GATE_V0, 'PRC_TWIN')['label'] == 'INCOMPLETE'
    rc = {'nested_means': copy.deepcopy(GATE_V0['nested_means']),
          'spend_over_cont': {k: tuple(x) for k, x in GATE_V0['spend_twin_over_cont'].items()}}
    assert R.gate_reproduction(rc, {**GATE_V0, 'gate': 'PASS'}, 'PRC_TWIN')['label'] == 'NOT_REPRODUCED'   # gate label differs


# ---------------------------------------------------------------- (c) capacity order, triggers, stop
@test
def t_capacity_cascade():
    no = cap_rec(qualifies=False)
    yes = cap_rec(qualifies=True)
    assert R.capacity_decision({})['state'] == 'INCOMPLETE'
    assert R.capacity_decision({'16x64': no})['awaiting'] == '24x128'
    d = R.capacity_decision({'16x64': no, '24x128': no})
    assert d['state'] == 'TRIGGER_T5' and d['run'] == '32x256'
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no})
    assert d['state'] == 'TRIGGER_T6'
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair=no)          # repair without a committed card
    assert d['state'] == 'STOP_FAMILY_FA' and 'REPAIR_WITHOUT_COMMITTED_CARD_IGNORED' in d['flags']
    rep_no = {**no, 'card_committed_before_fit': True}
    rep_yes = {**yes, 'card_committed_before_fit': True}
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair=rep_no)       # 64 -> 128 -> 256 -> repair -> stop
    assert d['state'] == 'STOP_FAMILY_FA' and [s['capacity'] for s in d['steps']] == ['16x64', '24x128', '32x256', 'REPAIR']
    assert 'protected routing remains untested' in d['statement']
    assert R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair='DECLINED')['state'] == 'STOP_FAMILY_FA'
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair=rep_yes)
    assert d['state'] == 'PROTECT' and d['capacity'] == 'REPAIR'
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': yes})
    assert d['state'] == 'PROTECT' and d['capacity'] == '32x256' and d['construction'] == 'B1'
    d = R.capacity_decision({'16x64': no, '24x128': yes, '32x256': yes}, repair=rep_yes)   # untriggered results ignored
    assert d['capacity'] == '24x128' and set(d['flags']) == {'T5_RESULT_WITHOUT_TRIGGER_IGNORED', 'REPAIR_WITHOUT_TRIGGER_IGNORED'}
    d = R.capacity_decision({'16x64': cap_rec(choice_b1=False)})
    assert d['capacity'] == '16x64' and d['construction'] == 'A'
    assert R.capacity_decision({'16x64': cap_rec(reproduced=False)})['state'] == 'STOPPED_FOR_DIAGNOSIS'
    # only the CHOSEN construction is qualified: B1 chosen and failing, A passing -> not qualified at that capacity
    mixed = cap_rec()
    mixed['evals']['B1'] = no['evals']['B1']
    d = R.capacity_decision({'16x64': mixed})
    assert d['awaiting'] == '24x128' and d['steps'][0]['choice'] == 'B1' and d['steps'][0]['other_construction'] == 'A'
    assert d['steps'][0]['other_would_qualify'] is True                       # recorded, never used
    d = R.capacity_decision({'16x64': cap_rec()})
    assert d['steps'][0]['other_would_qualify'] is True and d['construction'] == 'B1'
    one = cap_rec()
    del one['evals']['A']
    assert R.capacity_decision({'16x64': one})['steps'][0]['other_would_qualify'] is None
    rep_ms = {**yes, 'card_committed_before_model_select_read': True}
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair=rep_ms)
    assert d['state'] == 'PROTECT' and d['capacity'] == 'REPAIR'
    noev = cap_rec()
    del noev['evals']['B1']
    assert R.capacity_decision({'16x64': noev})['state'] == 'INCOMPLETE'


# ---------------------------------------------------------------- (e) decisions
POINTS = ('TWIN', 'HQ_S2', 'HQ_BR', 'SHUF_S2', 'SHUF_BR', 'ORD_S1', 'ORD_S2', 'ORD_BR')


def d2_rec(pair=0.02, ord_pair=0.03, g1=True, grid_valid=True, refs=0.03, status='FITTED', twin_alias=False, nolook=0.01,
           nest=0.01, f4d=0.01):
    ev = NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.011})
    grid = {i: NT.grid_entry(pair) for i in POINTS}
    grid.update({o: NT.grid_entry(ord_pair) for o in ('ORD_S1', 'ORD_S2', 'ORD_BR')})
    if twin_alias:
        grid['TWIN'] = copy.deepcopy(NT.ALIAS)
    grid.update({r: NT.grid_entry(refs) for r in NR.BENCHMARK})
    costs = {'TWIN': NT.costs(0.003), 'HQ_S2': NT.costs(0.001), 'HQ_BR': NT.costs(-0.001), 'SHUF_S2': NT.costs(0.001),
             'SHUF_BR': NT.costs(-0.001), 'ORD_S1': NT.costs(0.001), 'ORD_S2': NT.costs(-0.002), 'ORD_BR': NT.costs(-0.0005)}
    return {'capacity_decision': R.capacity_decision({'16x64': cap_rec()}), 'grid_valid': grid_valid,
            'prc_s2': {'status': status, 'config_id': 'PRC_S2', 'evals': ev}, 'grid': grid,
            'paths': {'HQ': ['HQ_S2', 'HQ_BR'], 'SHUF': ['SHUF_S2', 'SHUF_BR'], 'ORD': ['TWIN', 'ORD_S1', 'ORD_S2', None, 'ORD_BR']},
            'alias_of': {}, 'costs_vs': costs, 'f4': {'version': {v: {'diff': f4d, 'se': 0.003} for v in NR.VERSIONS}},
            'nolookup': {i: {'cells': NT.reg(lambda c: nolook)} for i in POINTS + NR.BENCHMARK},
            'nested_lookup': {i: {v: nest for v in NR.VERSIONS} for i in POINTS + NR.BENCHMARK}}


@test
def t_d1_labels():
    assert R.d1(R.capacity_decision({'16x64': cap_rec()}))['label'] == 'REPAIRED_TASK_ONLY_RELEASE'
    assert R.d1(R.capacity_decision({'16x64': cap_rec(qpe_supported=False)}))['label'] == 'QUALIFIED_B1_REPAIR_NOT_RESOLVED'
    assert R.d1(R.capacity_decision({'16x64': cap_rec(choice_b1=False)}))['label'] == 'QUALIFIED_WITHOUT_PROBABILITY_REPAIR'
    no = cap_rec(qualifies=False)
    assert R.d1(R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, 'DECLINED'))['label'] == \
        'NO_QUALIFYING_TASK_ONLY_RELEASE'
    assert R.d1(R.capacity_decision({'16x64': no, '24x128': no}))['label'] == 'INCOMPLETE'           # T5 pending
    rep = {**cap_rec(), 'card_committed_before_model_select_read': True}                             # supported B1 repair
    d = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair=rep)
    assert R.d1(d)['label'] == 'POST_HOC_REPAIR_QUALIFIED'                                           # never REPAIRED_...


@test
def t_d2_conjuncts_partials_and_aliases():
    d = R.d2(d2_rec())
    assert d['label'] == 'COMPONENT_BEYOND_MATCHED_CONTROLS' and d['met'] and d['passed'] == list(R.D2_COMPARISONS)
    assert d['comparisons']['ORD_FRONTIER']['point'] == 'ORD_BR' and d['comparisons']['HQ_FRONTIER']['point'] == 'HQ_BR'
    assert d['comparisons']['ORD_FRONTIER']['path'] == ['TWIN', 'ORD_S1', 'ORD_S2', 'ORD_BR']        # ORD_S3 absent
    assert set(d['targeting']) == {'TWIN', 'SHUF_BR', 'HQ_BR', 'ORD_BR'}
    d = R.d2(d2_rec(f4d=0.005))                                                                      # 0.005 - 2 x 0.003 < 0
    assert d['label'] == 'NOT_MET_F4' and not d['met'] and d['passed'] == list(R.D2_COMPARISONS)
    r = d2_rec()
    r['f4']['version'][1] = {'diff': 0.005, 'se': 0.003}                                             # F4 fails in one version
    assert R.d2(r)['label'] == 'NOT_MET_F4'
    r = d2_rec()
    r['nolookup']['HQ_BR'] = {'cells': NT.reg(lambda c: -0.01 if c.startswith('S1') else 0.01)}    # v1 non-table mean < 0
    d = R.d2(r)
    assert d['label'] == 'READER_TARGETED' and 'READER_TARGETED_HQ_FRONTIER' in d['flags'] and not d['met']
    r = d2_rec()
    r['nested_lookup']['ORD_BR'] = {0: 0.01, 1: -0.001}                                              # nested LOOKUP fails
    assert R.d2(r)['label'] == 'READER_TARGETED'
    r = d2_rec()
    r['paths']['HQ'] = [None, None]                                                                  # empty HQ path
    d = R.d2(r)
    assert 'HQ_AT_IDENTITY' in d['flags'] and d['comparisons']['HQ_FRONTIER']['path'] == ['TWIN']
    assert d['comparisons']['HQ_FRONTIER']['reason'] == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    assert d['label'] == 'PARTIAL_TWIN+SHUF_FRONTIER+ORD_FRONTIER_ONLY'
    r = d2_rec()
    r['alias_of'] = {'HQ_BR': 'HQ_S2'}                                                               # alias counted once
    d = R.d2(r)
    assert d['comparisons']['HQ_FRONTIER']['path'] == ['HQ_S2'] and 'HQ_FRONTIER' not in d['passed']
    r = d2_rec()
    r['grid']['SHUF_BR'] = NT.grid_entry(0.0005)                                                     # no resolved benefit
    d = R.d2(r)
    assert d['label'] == 'PARTIAL_TWIN+HQ_FRONTIER+ORD_FRONTIER_ONLY'
    assert d['comparisons']['SHUF_FRONTIER']['reason'] == 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'
    assert R.d2(d2_rec(pair=-0.01, ord_pair=-0.01))['label'] == 'NO_REDUCTION_BEYOND_ANY_CONTROL'
    assert R.d2(d2_rec(g1=False))['label'] == 'MET_WITHOUT_TASK_FEASIBILITY'
    d = R.d2(d2_rec(twin_alias=True))
    assert 'PRC_S2_ALIAS_OF_TWIN' in d['flags'] and 'TWIN' not in d['passed']
    r = d2_rec()
    r['prc_s2']['every_nonzero_selection_alias_of_twin'] = True
    d = R.d2(r)
    assert d['label'] == 'NOT_ASSESSED_EXACT_ALIAS' and d['family_stops']
    r = d2_rec()
    r['grid']['TWIN']['income'] = NT.grid_entry(0.0, inc=0.02)['income']                            # recipient gate fails
    assert 'TWIN' not in R.d2(r)['passed']
    r = d2_rec()
    del r['f4']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['nolookup']['SHUF_BR']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['nested_lookup']['TWIN']
    d = R.d2(r)
    assert d['met'] and 'TWIN_NESTED_LOOKUP_FROM_F4' in d['flags']
    r = d2_rec()
    del r['costs_vs']['HQ_S2']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    r = d2_rec()
    del r['paths']['ORD']
    assert R.d2(r)['label'] == 'INCOMPLETE'
    assert R.d2(d2_rec(grid_valid=None))['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    assert R.d2(d2_rec(grid_valid=False))['label'] == 'NOT_COMPUTED_GRID_INVALID'
    assert R.d2(d2_rec(status='NOT_FEASIBLE'))['label'] == 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH'


@test
def t_decisions_and_outcome_class():
    out = R.decisions(d2_rec())
    assert out['outcome_class']['label'] == 'CANDIDATE' and out['D3']['met'] and out['D4']['label'] == 'MATERIAL_0020_RESOLVED'
    assert out['D5']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS' and out['D6']['label'] == 'NOT_CLAIMED'
    out = R.decisions(d2_rec(refs=0.001))
    assert out['outcome_class']['label'] == 'COMPONENT' and out['D3']['label'] == 'NOT_SUPERIOR'
    r = d2_rec(refs=0.03)
    r['grid']['SHUF_BR'] = NT.grid_entry(0.0)
    out = R.decisions(r)
    assert out['outcome_class']['label'] == 'ENGINEERING_REPAIR' and 'D3_WITHOUT_D2' in out['outcome_class']['flags']
    r['capacity_decision'] = R.capacity_decision({'16x64': cap_rec(choice_b1=False)})
    assert R.decisions(r)['outcome_class']['label'] == 'FINITE_NEGATIVE_RESULT'
    no = cap_rec(qualifies=False)
    rep = {**cap_rec(), 'card_committed_before_model_select_read': True}
    r['capacity_decision'] = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, repair=rep)
    out = R.decisions(r)                                                              # a post hoc repair never gives the class
    assert out['D1']['label'] == 'POST_HOC_REPAIR_QUALIFIED' and out['outcome_class']['label'] == 'FINITE_NEGATIVE_RESULT'
    r['capacity_decision'] = R.capacity_decision({'16x64': no, '24x128': no, '32x256': no}, 'DECLINED')
    out = R.decisions(r)
    assert out['D2']['label'] == 'NOT_ASSESSED_PROTECTED_ROUTING_UNTESTED' and out['outcome_class']['label'] == \
        'FINITE_NEGATIVE_RESULT' and 'PROTECTED_ROUTING_UNTESTED' in out['outcome_class']['flags']
    assert R.decisions(d2_rec(grid_valid=None))['outcome_class']['label'] == 'INCOMPLETE'
    r = d2_rec()
    r['capacity_decision'] = R.capacity_decision({'16x64': no, '24x128': no})
    assert R.decisions(r)['outcome_class']['label'] == 'INCOMPLETE'
    out = R.decisions(d2_rec(status='NOT_FEASIBLE'))
    assert out['D3']['label'] == 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH' and out['outcome_class']['label'] == 'ENGINEERING_REPAIR'
    r = d2_rec()
    r['prc_s2']['every_nonzero_selection_alias_of_twin'] = True
    out = R.decisions(r)
    assert out['D3']['label'] == 'NOT_ASSESSED_EXACT_ALIAS' and 'FAMILY_STOPS_EXACT_ALIAS' in out['outcome_class']['flags']
    tight = d2_rec()
    tight['prc_s2']['evals'] = NT.evals(occ={('U0', 'nll_excess'): 0.0095}, se=0.002)    # point feasible, reserve not met
    assert 'POINT_FEASIBLE_NOT_RESERVE_QUALIFIED' in R.decisions(tight)['outcome_class']['flags']
    j = d2_rec()
    j['joint'] = {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': 'ABL_LOC'}
    assert R.decisions(j)['D5']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS'               # ABL_LOC alone is not enough


@test
def t_d3_d4_reader_budget_asymmetry():
    r = d2_rec()
    r['reader_budget'] = {'W': 'READER_BUDGET_ASYMMETRIC', 'H_LJ': 'READER_BUDGET_SYMMETRIC'}
    r['nested_lookup']['W'] = {0: -0.001, 1: -0.001}                    # irrelevant: no nested LOOKUP condition for references
    out = R.decisions(r)
    assert out['D3']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' and out['D3']['reader_budget_credit'] == {'W': 'CREDITED'}
    r['nolookup']['W'] = {'cells': NT.reg(lambda c: -0.01 if c.startswith('S1') else 0.01)}    # non-table v1 mean < 0
    out = R.decisions(r)
    assert out['D3']['label'] == 'NOT_SUPERIOR' and out['D3']['not_credited'] == ['W'] and 'W' in out['D3']['failing']
    assert out['outcome_class']['label'] == 'COMPONENT' and out['D4']['label'] == 'MATERIAL_0020_RESOLVED'   # W not a target
    r = d2_rec()
    r['reader_budget'] = {'TT100': 'READER_BUDGET_ASYMMETRIC'}
    del r['nolookup']['TT100']                                          # missing -> INCOMPLETE, never 'not beaten'
    out = R.decisions(r)
    assert out['D3']['label'] == 'INCOMPLETE' and out['D3']['incomplete_references'] == ['TT100']
    assert out['outcome_class']['label'] == 'INCOMPLETE'
    r['grid']['SUCC'] = NT.grid_entry(0.0)                              # another reference fails on its own result
    out = R.decisions(r)
    assert out['D3']['label'] == 'NOT_SUPERIOR' and out['D3']['incomplete_references'] == ['TT100'] and \
        'TT100' not in out['D3']['failing']
    # D4: the same rule for NEURAL_RAW and H_LJ
    r = d2_rec()
    r['reader_budget'] = {'NEURAL_RAW': 'READER_BUDGET_ASYMMETRIC', 'H_LJ': 'READER_BUDGET_ASYMMETRIC'}
    assert R.decisions(r)['D4']['label'] == 'MATERIAL_0020_RESOLVED'
    r['nolookup']['NEURAL_RAW'] = {'cells': NT.reg(lambda c: -0.002)}
    out = R.decisions(r)
    assert out['D4']['label'] == 'NOT_MET' and out['D4']['reader_budget_credit']['NEURAL_RAW'] == 'NOT_CREDITED'
    assert out['D3']['label'] == 'NOT_SUPERIOR'
    r = d2_rec()
    r['reader_budget'] = {'H_LJ': 'READER_BUDGET_ASYMMETRIC'}
    del r['nolookup']['H_LJ']
    out = R.decisions(r)
    assert out['D4']['label'] == 'INCOMPLETE' and out['D3']['label'] == 'INCOMPLETE'
    weak = d2_rec(refs=0.001)                                            # level NOT_MET on its own: stays NOT_MET
    weak['reader_budget'] = {'H_LJ': 'READER_BUDGET_ASYMMETRIC'}
    del weak['nolookup']['H_LJ']
    assert R.decisions(weak)['D4']['label'] == 'NOT_MET'
    nof = d2_rec(g1=False)                                               # without G1 a credited level is never claimable
    assert R.decisions(nof)['D4']['label'] == 'MET_WITHOUT_TASK_FEASIBILITY'


# ---------------------------------------------------------------- (f) strongest nonzero release, effective refit
@test
def t_strongest_nonzero():
    def cfg(cid, lam, x, g1=True, alias=None):
        return {'config_id': cid, 'lam': lam, 'evals': NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.02}),
                'pair_vs_twin': NT.grid_entry(x)['pair'] if x is not None else copy.deepcopy(NT.ALIAS['pair']),
                'alias_of': alias}
    s = R.strongest_nonzero([cfg('TW', 0.0, 0.05), cfg('P1', 0.5, 0.004), cfg('P2', 2.0, 0.012), cfg('P3', 8.0, 0.03, g1=False)])
    assert s['selected'] == 'P2'
    s = R.strongest_nonzero([cfg('P3', 4.0, 0.012), cfg('P2', 1.0, 0.012), cfg('P1', 0.25, 0.004)])   # tie -> smaller lambda
    assert s['selected'] == 'P2' and s['tied'] == ['P2', 'P3']
    s = R.strongest_nonzero([cfg('P1', 0.25, None), cfg('P2', 1.0, 0.02, alias='P1')])
    assert s['label'] == 'NO_FEASIBLE_NONZERO_RELEASE' and s['records'][0]['status'] == 'ALIAS_OF_TWIN'
    assert R.strongest_nonzero([])['label'] == 'NO_FEASIBLE_NONZERO_RELEASE'
    s = R.strongest_nonzero([cfg('P1', 0.25, -0.002)])                                 # negative benefit still the strongest
    assert s['selected'] == 'P1'


def seeds8(offset=R.SEED_OFFSET):
    return {f'S{v}k{k}': NR.MODEL_SEED_BASE + offset + 10 * v + k for v in NR.VERSIONS for k in range(4)}


def checks(wires=('CAND', 'TWIN', 'REF'), reps=('R1', 'R2', 'R3', 'R4')):
    return [{'recipe': r, 'view': vw, 'wire': w, 'v': v, 'R': rp, 'equal_k0_k1': False}
            for r in NR.SEED_DEPENDENT for vw in NR.VIEWS for w in wires for v in NR.VERSIONS for rp in reps]


def splits(overlap=0.3):
    reps = ('R1', 'R2', 'R3', 'R4')
    return {r: {'assignment_sha256': f'hash{r}', 'score_overlap': {q: overlap for q in reps if q != r}} for r in reps}


@test
def t_refit_effectiveness():
    ok = R.refit_effectiveness(splits(), seeds8(), checks(), ['CAND', 'TWIN'], {})
    assert ok['label'] == 'REFIT_GENUINELY_EFFECTIVE' and ok['counted_once'] == []
    sp = splits()
    sp['R4']['assignment_sha256'] = sp['R2']['assignment_sha256']                       # R4 repeats R2's split
    bad = R.refit_effectiveness(sp, seeds8(), checks(), ['CAND', 'TWIN'], {})
    assert bad['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and bad['counted_once'] == ['robustness repetitions']
    assert R.refit_effectiveness(splits(1.0), seeds8(), checks(), ['CAND'], {})['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'
    sp = splits()
    del sp['R3']['score_overlap']['R1']
    del sp['R1']['score_overlap']['R3']
    assert R.refit_effectiveness(sp, seeds8(), checks(), ['CAND'], {})['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'
    ch = checks()
    for c in ch:
        if c['wire'] == 'CAND' and c['recipe'] == 'ET25' and c['R'] == 'R3':
            c['equal_k0_k1'] = True                                                  # seed changed nothing on the minimum set
    r = R.refit_effectiveness(splits(), seeds8(), ch, ['CAND', 'TWIN'], {})
    assert r['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and r['counted_once'] == ['robustness reader seeds']
    s = seeds8()
    del s['S1k3']
    assert R.refit_effectiveness(splits(), s, checks(), ['CAND'], {})['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'


# ---------------------------------------------------------------- (g) reader-seed validity of the FRC1 grid
@test
def t_grid_validity_frc_offsets():
    reg = {k: x for k, x in seeds8().items() if k[-1] in '01'}
    assert R.grid_validity(reg, checks(reps=('R1', 'R2')), ['CAND', 'TWIN'], {})['label'] == 'READER_SEEDS_EFFECTIVE'
    for off in (300, 400, 500, 600):
        old = {k: x for k, x in seeds8(off).items() if k[-1] in '01'}
        g = R.grid_validity(old, checks(), ['CAND'], {})
        assert g['label'] == 'READER_SEEDS_NOT_EFFECTIVE' and any('reuse earlier' in e for e in g['errors']), off
    six = {k: x for k, x in seeds8(600).items() if k[-1] in '01'}
    assert NR.grid_validity(six, checks(), ['CAND'], {})['label'] == 'READER_SEEDS_EFFECTIVE'      # nrm alone accepts +600
    shifted = dict(reg)
    shifted['S1k1'] = NR.MODEL_SEED_BASE + 700 + 19                                    # unregistered, not forbidden
    g = R.grid_validity(shifted, checks(), ['CAND'], {})
    assert g['label'] == 'READER_SEEDS_NOT_EFFECTIVE' and any('differ from the registered' in e for e in g['errors'])
    assert R.grid_validity({'S0k0': reg['S0k0']}, checks(), ['CAND'], {})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    dup = dict(reg)
    dup['S1k1'] = dup['S1k0']
    assert R.grid_validity(dup, checks(), ['CAND'], {})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    weird = dict(reg)
    weird['X'] = NR.MODEL_SEED_BASE + 799
    assert R.grid_validity(weird, checks(), ['CAND'], {})['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    ch = checks()
    for c in ch:
        if c['wire'] == 'CAND' and c['recipe'] == 'MLP64':
            c['equal_k0_k1'] = True
    g = R.grid_validity(reg, ch, ['CAND', 'TWIN'], {})                                 # nrm's recipe-equality part unchanged
    assert g['label'] == 'READER_SEEDS_NOT_EFFECTIVE' and any('minimum set' in e for e in g['errors'])
    g = R.grid_validity(reg, [c for c in checks() if c['recipe'] != 'ET5'], ['CAND'], {})
    assert any('not checked' in e for e in g['errors'])


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

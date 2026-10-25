"""Invented-data tests of rfa.rules (no real data, no label). Run: python3 -m rfa.test_rules (prints JSON; exit 0 iff all pass).

Covers: the rule file and its refusal of altered constants; nrm.rules left unchanged (never monkeypatched); the usefulness definitions
at the cited lines of the governing code (source text only, nothing imported that reads data); signs; reader-seed validity of the
RFA1 grid (+1200 registered, 300-1100 forbidden, bootstrap base 2026102520, B = 400 recorded, the explanations, seed pairs judged
separately); the refit; the brackets (equal to nrm.rules.frontier_comparison) and fineness on MODEL_FIT out-of-fold data only; D1
(G1 gating, the confidence conjunction and the reserve reported separately, CAND's selection data); D2 (START, CONT, STOPFIT,
ALTCRIT as path / point / exact alias counted once, RAWJ; no bracket; bracket at the origin; recipient gate at the bracket;
cheaper-point domination; the single-point spend rule; reader targeting; partials named by what passed; gates and missing inputs;
bounds); the MC SE through NR; D3 (no reserve gate removes a reference) and D4 (0.020 resolved, strict); the learning curve; the
outcome classes and the five outcomes; registration; the campaign summary; D2R with sign changes and robustness recipient failures.
Helpers for invented utility records and grid entries come from nrm.test_rules (unchanged)."""
from __future__ import annotations
import copy
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from nrm import rules as NR
from nrm import test_rules as NT                 # invented-data helpers (util, evals, entry, reg, grid_entry, costs)
from rfa import rules as R

ROOT = Path(R.__file__).resolve().parent.parent
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- invented records
def patch_se(ev, name, se, se_mc=None):
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


def neural_raw_evals():
    ev = {0: NT.util(0, guards={'sex0': 0.0015, 'sex1': -0.001}),
          1: NT.util(1, guards={'sex0': 0.0015, 'sex1': -0.001}, inc={('U0', 'brier_excess'): 0.003})}
    return patch_se(ev, 'guard|sex0', 0.0005)


def costs(total, extra=None, se=0.0005):
    return NT.costs(total, se=se, extra=extra)


STOP = ['S1', 'S2']
ALT = ['A1', 'A2']
RAWJ = ['J1', 'J2']
IDS = ['NEURAL_RAW', 'CONT'] + STOP + ALT + RAWJ + ['AP']


def rec(start_pair=0.03, cont_pair=0.03, stop_pair=0.02, alt_pair=0.02, rawj_pair=0.02, refs=0.03, cand_ev=None,
        grid_valid=True, nolook=0.01, family='RFA', revision=0, early=True, altcrit='PATH'):
    """A complete invented reading. START = NEURAL_RAW; CAND spends +0.003 nats over START; on each path the first saved point is
    cheaper than CAND and the second spends more (the bracket); altcrit = 'PATH' | 'POINT' (one point AP spending more than CAND)
    | 'ALIAS_OF_STOPFIT'."""
    cev = cand_ev if cand_ev is not None else NT.evals()
    grid = {r: NT.grid_entry(refs) for r in NR.BENCHMARK}
    grid['NEURAL_RAW'] = NT.grid_entry(start_pair)
    grid['CONT'] = NT.grid_entry(cont_pair)
    grid.update({i: NT.grid_entry(stop_pair) for i in STOP})
    grid.update({i: NT.grid_entry(rawj_pair) for i in RAWJ})
    cv = {'NEURAL_RAW': costs(0.003), 'CONT': costs(0.004), 'S1': costs(0.001), 'S2': costs(-0.001), 'J1': costs(0.002),
          'J2': costs(-0.0005)}
    paths = {'STOPFIT': list(STOP), 'RAWJ': list(RAWJ)}
    sel = {'STOPFIT': 'MODEL_FIT_OOF', 'RAWJ': 'MODEL_FIT_OOF'}
    if altcrit == 'PATH':
        grid.update({i: NT.grid_entry(alt_pair) for i in ALT})
        cv.update({'A1': costs(0.0015), 'A2': costs(-0.0008)})
        paths['ALTCRIT'] = list(ALT)
        sel['ALTCRIT'] = 'MODEL_FIT_OOF'
    elif altcrit == 'POINT':
        grid['AP'] = NT.grid_entry(alt_pair)
        cv['AP'] = costs(-0.0005)
        paths['ALTCRIT'] = ['AP']
        sel['ALTCRIT'] = 'MODEL_FIT_OOF'
    r = {'family': family, 'card': f'{family}_x', 'revision': revision, 'card_committed_before_any_model_select_outcome': early,
         'grid_valid': grid_valid, 'start': {'id': 'NEURAL_RAW', 'evals': neural_raw_evals()},
         'cand': {'status': 'FITTED', 'config_id': 'CAND', 'evals': cev, 'operating_point': {0: {'epoch': 6}, 1: {'epoch': 6}},
                  'selection_data': 'MODEL_FIT_OOF'},
         'points': [{'config_id': 'CAND', 'evals': cev}],
         'cont': {'config_id': 'CONT', 'evals': NT.evals()}, 'paths': paths, 'control_kind': {'ALTCRIT': altcrit},
         'selection_data': sel, 'alias_of': {}, 'grid': grid, 'costs_vs': cv,
         'nolookup': {i: {'cells': NT.reg(lambda c: nolook)} for i in list(NR.BENCHMARK) + IDS}}
    if altcrit == 'ALIAS_OF_STOPFIT':
        r['altcrit_alias_wire_check'] = True
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


def gv(seeds, ch, mins, dist, boot=None, B=400, **kw):
    return R.grid_validity(seeds, ch, mins, dist, boot_seeds=boots() if boot is None else boot, boot_B=B, **kw)


# ---------------------------------------------------------------- rule file, inherited module, governing code
@test
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'RFA1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(Path(R.RULES_PATH).read_text())
    tmp = Path(tempfile.mkdtemp(prefix='rfa_rules_test_'))
    alter = [(('reader_grid', 'seed_offset'), 1100), (('reader_grid', 'forbidden_offsets'), [300, 400, 500, 600, 700, 800, 900, 1000]),
             (('reader_grid', 'bootstrap_seed_base'), 2026102500), (('reader_grid', 'bootstrap_B'), 200),
             (('er_explanation', 'few_valued_max'), 1600), (('usefulness', 'caps'), {'accuracy_loss': 0.02, 'nll_excess': 0.01,
                                                                                    'brier_excess': 0.005}),
             (('usefulness', 'within_sex_guard_cap'), 0.01), (('usefulness', 'n_conditions'), 24),
             (('decisions', 'D2', 'met_label'), 'COMPONENT_GAIN_BEYOND_START_AND_CONTROLS'),
             (('decisions', 'D2', 'comparisons'), ['START', 'CONT', 'STOPFIT', 'RAWJ']),
             (('decisions', 'D2', 'control_kinds'), {'STOPFIT': ['PATH', 'POINT'], 'ALTCRIT': ['PATH', 'POINT', 'ALIAS_OF_STOPFIT'],
                                                     'RAWJ': ['PATH']}),
             (('decisions', 'D2', 'bracket_selection_data'), ['MODEL_FIT_OOF', 'MODEL_SELECT']),
             (('decisions', 'D1', 'cand_selection_data'), ['MODEL_FIT_OOF', 'MODEL_SELECT']),
             (('decisions', 'D4', 'threshold_auc'), 0.015), (('registration_status', 'family_pattern'), '(FA|FS_[A-Za-z0-9]+)'),
             (('registration_status', 'starts'), {'RFA': 'H_LJ'}), (('recipient_limit_auc',), 0.02), (('z',), 1.5),
             (('se_floor_auc',), 0.0), (('decisions', 'LC', 'labels'), ['BENEFIT_SHRINKS_POINT', 'NO_SHRINKAGE_AT_POINT']),
             (('outcome_class', 'labels'), ['METHOD_CANDIDATE', 'COMPONENT']),
             (('registration_status', 'family_aliases'), {'FAM': 'FS_FAM'}),
             (('decisions', 'D2', 'altcrit_near_alias_lag'), 0.05),
             (('decisions', 'F0', 'labels'), ['MECHANISM_INERT'])]
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
                 'comparison', 'frontier_comparison', 'robustness', 'claims', 'grid_validity', 'forbidden_seeds', '_primary',
                 '_pooled'):
        assert getattr(NR, name).__module__ == 'nrm.rules', name
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and NR.SE_FLOOR == 0.001 and NR.RECIPIENT_LIMIT == 0.010 and NR.TARGET == 0.020


def _line(rel, n):
    return (ROOT / rel).read_text().splitlines()[n - 1]


@test
def t_usefulness_definitions_at_the_cited_code():
    """The 28 conditions and their caps where RULES.md section 2 says they live (source text only)."""
    assert _line('scm_bench/selection.py', 38).startswith('def clauses(labels, probs, refs, seed):')
    assert "for ref in ('U0', 'Ucal'):" in (ROOT / 'scm_bench/selection.py').read_text()
    assert _line('scm_bench/common.py', 45) == "CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}"
    assert _line('scm_bench/common.py', 213).startswith('def task_vectors(y, p, k):')
    assert '((p - target) ** 2).sum(1)' in _line('scm_bench/common.py', 220)                 # summed Brier over classes
    assert _line('ihs/evaluate.py', 75).startswith('def utility(labels, outs, refs, sex, unit, seed):')
    assert "d = own['occupation']['brier'] - u0['brier']" in _line('ihs/evaluate.py', 89)    # guards vs U0, within SEX
    assert _line('ihs/evaluate.py', 57).startswith('def cluster_se(d, unit):')
    assert re.match(r'GUARD = 0\.005\s+# within-SEX occupation summed-Brier excess vs U0', _line('ihs/common.py', 63))
    assert _line('ohn/evaluate.py', 120).startswith('def complete_eval(')
    assert _line('sar/evaluate.py', 160).startswith('def utility_of(v, rel, S, ctx):')
    assert _line('tsz/evaluate.py', 38).startswith('def refs_of(v, S):')
    assert _line('ohn/rules.py', 103).startswith('def conditions(e, seed):')
    assert _line('ohn/rules.py', 30) == "CAPS = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}"
    assert _line('ohn/rules.py', 31) == 'GUARD_CAP = 0.005'
    assert _line('nrm/rules.py', 157).startswith('def point_eligibility(evals_by_version):')
    assert _line('nrm/rules.py', 48) == 'RECIPIENT_LIMIT = 0.010' and _line('nrm/rules.py', 49) == 'TARGET = 0.020'
    assert _line('nrm/rules.py', 356).startswith('def recipient_screen(') and _line('nrm/rules.py', 387).startswith('def comparison(')
    assert _line('nrm/rules.py', 344).startswith('def target_screen(')
    # structure: 12 clauses + 2 guards per version, 28 in total, inclusive caps
    assert len(NR.OR.CONDITIONS) == 14 and sum(n.startswith('guard|') for n in NR.OR.CONDITIONS) == 2
    assert tuple(R.load_rules()['usefulness']['conditions_per_version']) == NR.OR.CONDITIONS
    p = NR.point_eligibility(NT.evals())
    assert p['label'] == 'POINT_ELIGIBLE' and p['n_conditions'] == 28
    at_cap = NT.evals(occ={('Ucal', 'brier_excess'): 0.005}, inc={('U0', 'accuracy_loss'): 0.010}, guards={'sex1': 0.005})
    assert NR.point_eligibility(at_cap)['label'] == 'POINT_ELIGIBLE'                         # inclusive
    over = NT.evals(guards={'sex1': 0.0050001})
    assert NR.point_eligibility(over)['failed'] == ['v0|guard|sex1', 'v1|guard|sex1']


@test
def t_signs():
    # pair benefit: comparator AUC 0.62, candidate 0.60 -> b = +0.02 favours the candidate
    e = NT.grid_entry(0.62 - 0.60, se=0.002)
    assert NR.comparison(e)['benefit']['label'] == 'AGG_POSITIVE'
    # recipient: candidate single view 0.015 more recoverable than the comparator's -> gate fails
    assert not NR.comparison(NT.grid_entry(0.02, inc=0.015, se=0.002))['recipient_gate_pass']
    assert NR.comparison(NT.grid_entry(0.02, inc=0.010, se=0.002))['recipient_gate_pass']        # inclusive 0.010
    # paired cost: cost(CAND) - cost(point) <= 0 -> the point spends at least as much (a bracket)
    b = R.brackets({'NEURAL_RAW': costs(0.003), 'S1': costs(0.001), 'S2': costs(-0.001)}, {'STOPFIT': STOP})
    assert b['STOPFIT']['bracket'] == 'S2'
    # learning curve: b(FULL) 0.01 < b(HALF) 0.03 -> d = -0.02 (shrinks)
    x = R._lc_one(NT.grid_entry(0.01)['pair'], NT.grid_entry(0.03)['pair'])
    assert x['change_mean'] < 0 and x['limitation']


# ---------------------------------------------------------------- (9) grid validity, (8) refit
@test
def t_grid_validity_seeds_and_bootstrap():
    assert R.registered_seed(0, 0) == 2026102405 and R.registered_seed(1, 3) == 2026102418
    assert R.registered_boot_seed(0, 'R1') == 2026102520 and R.registered_boot_seed(1, 'R4') == 2026102623
    sar_drf_boot = {2026102400 + 100 * v + i for v in (0, 1) for i in range(4)} | {2026102500 + 100 * v + i for v in (0, 1)
                                                                                  for i in range(4)}
    assert not {R.registered_boot_seed(v, r) for v in (0, 1) for r in R.REP_INDEX} & sar_drf_boot
    fs = R.forbidden_seeds()
    assert len(fs) == 180 and NR.MODEL_SEED_BASE + 1100 + 19 in fs and not set(seeds_all().values()) & fs
    ch = checks()
    ok = gv(reg_seeds(), ch, MIN, di(income=5000))
    assert ok['effective'] and ok['label'] == 'READER_SEEDS_EFFECTIVE', ok['errors']
    old = gv(reg_seeds(1100), ch, MIN, di(income=5000))
    assert not old['effective'] and any('reuse earlier grid seeds' in e for e in old['errors'])
    shifted = reg_seeds(); shifted['S1k1'] = NR.MODEL_SEED_BASE + 1300 + 11
    assert any('differ from the registered' in e for e in gv(shifted, ch, MIN, di(income=5000))['errors'])
    assert not R.grid_validity(reg_seeds(), ch, MIN, di(income=5000), boot_seeds=None, boot_B=400)['effective']
    nb = gv(reg_seeds(), ch, MIN, di(income=5000), B=None)
    assert not nb['effective'] and any('replicate count not recorded' in e for e in nb['errors'])
    assert not gv(reg_seeds(), ch, MIN, di(income=5000), B=200)['effective']
    wb = boots(); wb['S1R2'] = 2026102601
    assert not gv(reg_seeds(), ch, MIN, di(income=5000), boot=wb)['effective']
    dup = reg_seeds(); dup['S1k1'] = dup['S1k0']
    assert not gv(dup, ch, MIN, di(income=5000))['effective']
    assert not gv(reg_seeds(), [c for c in ch if c['recipe'] != 'MLP64'], MIN, di(income=5000))['effective']


@test
def t_grid_validity_explanations():
    ch = checks()
    fv = with_eq(ch, lambda c: c['wire'] == 'CAND' and c['view'] == 'income')
    g = gv(reg_seeds(), fv, MIN, di(income=6))
    assert g['effective'] and 'FEW_VALUED_SEED_EQUALITIES_COUNTED_ONCE' in g['flags'], g['errors']
    assert g['strict_nrm1']['label'] == 'READER_SEEDS_NOT_EFFECTIVE'
    assert g['explained_equalities']['n_eff_implied']['CAND|income|MLP64|k0k1']['n_eff_seed_cells_if_this_recipe_wins'] == 4
    assert not gv(reg_seeds(), fv, MIN, di(income=1000))['effective']
    et = with_eq(ch, lambda c: c['wire'] == 'CAND' and c['view'] == 'income' and c['recipe'] == 'ET5')
    g3 = gv(reg_seeds(), et, MIN, di(income=1000))
    assert g3['effective'] and {i['explanation'] for i in g3['explained_equalities']['items']} == {'ER_EXTRA_TREES_SUPPORT'}
    assert not gv(reg_seeds(), et, MIN, di(income=2000))['effective']
    ml = with_eq(ch, lambda c: c['wire'] == 'CAND' and c['view'] == 'income' and c['recipe'] == 'MLP64')
    assert not gv(reg_seeds(), ml, MIN, di(income=1000))['effective']
    assert gv(reg_seeds(), ml, ['START'], di(income=1000))['effective']
    dead = with_eq(ch, lambda c: c['recipe'] == 'MLP64' and c['v'] == 0 and c['R'] == 'R1')
    assert not gv(reg_seeds(), dead, MIN, di(income=6, occupation=6, pair=6))['effective']
    assert not gv(reg_seeds(), fv, MIN, {})['effective']
    pairs = checks(k_pair=(0, 1)) + checks(k_pair=(2, 3))
    eq = with_eq(pairs, lambda c: c.get('k_pair') == [2, 3] and c['wire'] == 'CAND' and c['view'] == 'income'
                 and c['recipe'] == 'MLP64')
    g4 = gv(seeds_all(), eq, MIN, di(income=1000), required_k=R.ALL_K)
    assert not g4['effective'] and any(e.startswith('k2k3') for e in g4['errors'])


def splits(reps=('R1', 'R2', 'R3', 'R4'), overlap=0.3):
    return {r: {'assignment_sha256': hashlib.sha256(r.encode()).hexdigest(),
                'score_overlap': {q: overlap for q in reps if q != r}} for r in reps}


@test
def t_refit_effectiveness():
    reps4 = ('R1', 'R2', 'R3', 'R4')
    ch = checks(reps=reps4, k_pair=(0, 1)) + checks(reps=reps4, k_pair=(2, 3))
    ok = R.refit_effectiveness(splits(), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4), boot_B=400)
    assert ok['label'] == 'REFIT_GENUINELY_EFFECTIVE' and not ok['flags'], ok['reasons']
    reps3 = ('R1', 'R2', 'R3')
    ch3 = checks(reps=reps3, k_pair=(0, 1)) + checks(reps=reps3, k_pair=(2, 3))
    one = R.refit_effectiveness(splits(reps3), seeds_all(), ch3, MIN, di(income=1000), boot_seeds=boots(reps3), boot_B=400)
    assert one['label'] == 'REFIT_GENUINELY_EFFECTIVE' and one['flags'] == ['ONE_ROBUSTNESS_SPLIT_ONLY'], one['reasons']
    same = splits(); same['R3']['assignment_sha256'] = same['R1']['assignment_sha256']
    assert R.refit_effectiveness(same, seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4), boot_B=400)['label'] == \
        'REFIT_NOT_GENUINELY_EFFECTIVE'
    assert R.refit_effectiveness(splits(overlap=1.0), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4),
                                 boot_B=400)['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'
    none = R.refit_effectiveness(splits(('R1', 'R2')), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(), boot_B=400)
    assert none['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE' and 'NO_ROBUSTNESS_SPLIT' in none['reasons']
    nb = R.refit_effectiveness(splits(), seeds_all(), ch, MIN, di(income=1000), boot_seeds=boots(reps4))
    assert nb['label'] == 'REFIT_NOT_GENUINELY_EFFECTIVE'                                   # B not recorded


# ---------------------------------------------------------------- (4) brackets and fineness
@test
def t_brackets_equal_nrm_rule():
    r = rec()
    b = R.brackets(r['costs_vs'], r['paths'], r['control_kind'])
    pt = {'id': 'CAND', 'costs_vs': r['costs_vs'], 'grid_vs': r['grid']}
    for key, ids, want in (('STOPFIT', STOP, 'S2'), ('ALTCRIT', ALT, 'A2'), ('RAWJ', RAWJ, 'J2')):
        f = NR.frontier_comparison(pt, [{'id': i} for i in ['NEURAL_RAW'] + ids])
        assert b[key]['status'] == 'BRACKETED' and b[key]['bracket'] == f['bracketing_control'] == want
    r2 = rec(); r2['costs_vs']['S2'] = costs(0.0005)
    assert R.brackets(r2['costs_vs'], r2['paths'])['STOPFIT']['status'] == 'NO_BRACKET'
    r3 = rec(); r3['costs_vs']['NEURAL_RAW'] = costs(-0.0005)
    assert R.brackets(r3['costs_vs'], r3['paths'])['STOPFIT']['bracket'] == 'NEURAL_RAW'
    assert R.brackets(r['costs_vs'], {'STOPFIT': None})['STOPFIT']['status'] == 'PATH_NOT_GIVEN'
    p = rec(altcrit='POINT')
    assert R.brackets(p['costs_vs'], p['paths'], p['control_kind'])['ALTCRIT'] == {'bracket': 'AP', 'status': 'POINT',
                                                                                     'path': ['AP']}
    assert R.brackets({}, {}, {'ALTCRIT': 'ALIAS_OF_STOPFIT'})['ALTCRIT']['status'] == 'ALIAS_OF_STOPFIT'
    assert R.control_path(['S1', 'S2', 'S1'], {'S1': 'NEURAL_RAW'}) == ['NEURAL_RAW', 'S2']


def sp(cand=0.010, lower=0.008, upper=0.0135, se=0.001, mx=0.03, grid=None, data='MODEL_FIT_OOF', key='STOPFIT'):
    x = {'cand': cand, 'lower': lower, 'upper': upper, 'upper_se': se, 'max': mx}
    if grid is not None:
        x['grid'] = grid
    out = {v: dict(x) for v in NR.VERSIONS}
    if data is not None:
        out['selection_data'] = data
    return {key: out}


@test
def t_grid_fineness():
    assert R.grid_fineness(sp())['per_path']['STOPFIT'] == {0: 'FINE', 1: 'FINE'}
    assert R.grid_fineness(sp(upper=0.020))['per_path']['STOPFIT'][0] == 'COARSE_ABOVE'
    assert R.grid_fineness(sp(lower=0.002))['per_path']['STOPFIT'][0] == 'COARSE_BELOW'
    assert R.grid_fineness(sp(mx=0.011))['flags'] == ['CONTROL_PATH_MAY_NOT_BRACKET_STOPFIT']
    assert R.grid_fineness(sp(lower=0.011))['per_path']['STOPFIT'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'
    assert R.grid_fineness(sp(cand=-0.001))['per_path']['STOPFIT'][0] == 'NOT_APPLICABLE_CAND_NOT_ABOVE_START'
    g = [{'spend': 0.004, 'se': 0.001}, {'spend': 0.009, 'se': 0.001}, {'spend': 0.0135, 'se': 0.001}, {'spend': 0.03, 'se': 0.001}]
    assert R.grid_fineness(sp(lower=0.009, grid=g))['per_path']['STOPFIT'][0] == 'FINE'
    assert R.grid_fineness(sp(lower=0.004, grid=g))['per_path']['STOPFIT'][0] == 'SAVED_POINT_RULE_NOT_FOLLOWED'
    for bad in ('MODEL_SELECT', 'CALIBRATION', 'MODEL_FIT_EPISODES'):
        x = R.grid_fineness(sp(data=bad))
        assert x['per_path']['STOPFIT'][0] == 'SELECTION_DATA_NOT_ALLOWED' and x['flags'] == ['CONTROL_SELECTION_DATA_NOT_ALLOWED_STOPFIT']
    nr = R.grid_fineness(sp(data=None))
    assert nr['per_path']['STOPFIT'][0] == 'INCOMPLETE' and nr['flags'] == ['CONTROL_SELECTION_DATA_NOT_RECORDED_STOPFIT']
    both = R.grid_fineness({**sp(), **sp(upper=0.020, key='RAWJ')})
    assert both['per_path']['RAWJ'][0] == 'COARSE_ABOVE' and both['flags'] == ['CONTROL_GRID_COARSE_RAWJ']
    assert R.selection_data_error('MODEL_FIT_OOF') is None and R.selection_data_error('MODEL_SELECT') == 'NOT_ALLOWED'
    assert R.selection_data_error('MODEL_FIT_EPISODES', 'POINT') is None and R.selection_data_error(None) == 'NOT_RECORDED'


# ---------------------------------------------------------------- (2) D1
@test
def t_d1_point_confidence_reserve_separate():
    d = R.d1(rec())
    assert d['label'] == 'TASK_FEASIBLE' and d['met'] and d['start'] == 'POINT_ELIGIBLE'
    assert d['confidence']['cand']['label'] == 'CONFIDENCE_ALL_PASS'
    assert R.confidence_conjunction(neural_raw_evals())['label'] == 'CONFIDENCE_ALL_PASS'      # 0.003 + 2 * 0.001 <= 0.005
    cs = R.confidence_conjunction(NT.evals(inc={('U0', 'brier_excess'): 0.0035}))           # 0.0055 > 0.005: unresolved
    assert cs['label'] == 'CONFIDENCE_UNRESOLVED' and 'v1|income|U0|brier_excess' in cs['unresolved'] and not cs['violations']
    bad = NT.evals(occ={('U0', 'nll_excess'): 0.02})
    assert R.confidence_conjunction(bad)['label'] == 'CONFIDENCE_VIOLATION'
    edge = NT.evals(occ={('U0', 'nll_excess'): 0.0105})                  # point failure within 2 SE: unresolved, never a pass
    d2 = R.d1(rec(cand_ev=edge))
    assert d2['label'] == 'CAND_INFEASIBLE' and d2['confidence']['cand']['label'] == 'CONFIDENCE_UNRESOLVED'
    near = NT.evals(occ={('U0', 'nll_excess'): 0.0095})                  # point pass, not confidence-resolved
    d3 = R.d1(rec(cand_ev=near))
    assert d3['label'] == 'TASK_FEASIBLE' and 'POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED' in d3['flags']
    ev = patch_se(NT.evals(guards={'sex0': 0.0040, 'sex1': -0.0049}), 'guard|sex0', 0.0004)
    assert R.confidence_conjunction(ev)['label'] == 'CONFIDENCE_ALL_PASS'
    assert R.confidence_conjunction(patch_se(ev, 'guard|sex0', 0.0004, se_mc=0.0004))['label'] == 'CONFIDENCE_UNRESOLVED'
    r3 = rec(); r3['cand'] = {'status': 'NO_FEASIBLE_POINT'}
    assert R.d1(r3)['label'] == 'NO_FEASIBLE_POINT'
    for sd in (None, 'MODEL_SELECT', 'CALIBRATION'):
        r4 = rec(); r4['cand']['selection_data'] = sd
        assert R.d1(r4)['label'] == 'INCOMPLETE' and R.d2(r4)['label'] == 'INCOMPLETE'
    r5 = rec(); r5['cand']['selection_uses_sex_recovery_diagnostic'] = True
    assert 'CAND_SELECTED_ON_MODEL_FIT_SEX_RECOVERY_DIAGNOSTIC' in R.d1(r5)['flags']
    r6 = rec(); r6['start']['evals'] = bad
    assert 'START_POINT_INELIGIBLE' in R.d1(r6)['flags']


# ---------------------------------------------------------------- (4) D2
@test
def t_d2_met_and_structure():
    d = R.d2(rec())
    assert d['label'] == R.D2_MET and d['met'] and d['gating'] == ['START', 'CONT', 'STOPFIT', 'ALTCRIT', 'RAWJ'], d
    s = d['comparisons']['STOPFIT']
    assert s['point'] == 'S2' and s['path'] == ['NEURAL_RAW', 'S1', 'S2'] and s['pass']
    assert abs(s['residual_spend_mismatch']['at_bracket']['value'] - (-0.001)) < 1e-12
    assert d['comparisons']['RAWJ']['point'] == 'J2' and d['comparisons']['ALTCRIT']['point'] == 'A2'
    b = d['bounds']['per_comparison']['STOPFIT']
    assert abs(b['mean'] - 0.02) < 1e-12 and abs(b['upper'] - (0.02 + 2 * b['se_agg'])) < 1e-12
    assert d['start'] == 'NEURAL_RAW' and d['scope'] == R.D2_SCOPE and 'not information removal' in d['scope']


@test
def t_d2_partials_named_by_what_passed():
    d = R.d2(rec(stop_pair=0.0))
    assert d['label'] == 'PARTIAL_START+CONT+ALTCRIT+RAWJ_ONLY' and 'fit-aware derivative is not isolated' in d['statement']
    assert R.d2(rec(start_pair=0.0, cont_pair=0.0, stop_pair=0.0, alt_pair=0.0, rawj_pair=0.0))['label'] == 'NOT_SEPARATED'
    d2 = R.d2(rec(rawj_pair=0.0, alt_pair=0.0))
    assert d2['label'] == 'PARTIAL_START+CONT+STOPFIT_ONLY' and 'online-critic recipe' in d2['statement'] and \
        'alternating persistent critic' in d2['statement']
    assert 'continued task training alone' in R.d2(rec(cont_pair=0.0))['statement']
    assert R.d2(rec(start_pair=0.0))['label'] == 'PARTIAL_CONT+STOPFIT+ALTCRIT+RAWJ_ONLY'


@test
def t_d2_no_bracket_origin_alias():
    nb = rec(); nb['costs_vs']['S2'] = costs(0.0005)
    d = R.d2(nb)
    c = d['comparisons']['STOPFIT']
    assert d['label'] == 'PARTIAL_START+CONT+ALTCRIT+RAWJ_ONLY' and c['label'] == 'NOT_SEPARATED' and c['reason'] == 'NO_BRACKET'
    assert 'NO_BRACKET_STOPFIT' in d['flags'] and 'never evidence' in c['statement']
    assert d['bounds']['per_comparison']['STOPFIT']['mean'] is None
    og = rec(); og['costs_vs']['NEURAL_RAW'] = costs(-0.0005)
    for k, ids in (('S2', STOP), ('A2', ALT), ('J2', RAWJ)):
        og['costs_vs'][k] = costs(-0.002)
    d3 = R.d2(og)
    assert d3['comparisons']['STOPFIT']['point'] == 'NEURAL_RAW' and 'STOPFIT_BRACKET_IS_ORIGIN' in d3['flags']
    assert d3['label'] == R.D2_MET
    ali = rec(); ali['alias_of'] = {'S1': 'START'}
    assert R.d2(ali)['comparisons']['STOPFIT']['path'] == ['NEURAL_RAW', 'S2']
    idn = rec(); idn['paths']['STOPFIT'] = []
    assert 'STOPFIT_AT_ORIGIN' in R.d2(idn)['flags']
    same = rec(); same['grid']['S2'] = copy.deepcopy(NT.ALIAS)                   # CAND identical to the STOPFIT bracket point
    ds = R.d2(same)
    assert 'CAND_ALIAS_OF_STOPFIT_S2' in ds['flags'] and not ds['comparisons']['STOPFIT']['pass']


@test
def t_d2_gate_domination_targeting():
    g = rec(); g['grid']['S2'] = NT.grid_entry(0.02, inc=0.02)
    d = R.d2(g)
    assert not d['comparisons']['STOPFIT']['pass'] and 'RECIPIENT_GATE_FAILED_VS_S2' in d['flags']
    dom = rec(); dom['grid']['J1'] = NT.grid_entry(-0.02)
    d2 = R.d2(dom)
    assert not d2['comparisons']['RAWJ']['pass'] and 'DOMINATED_BY_CHEAPER_J1' in d2['flags']
    sec = rec(); sec['costs_vs']['S2'] = costs(-0.001, {'occupation|men_brier': {'value': 0.004, 'se': 0.001}})
    assert R.d2(sec)['comparisons']['STOPFIT']['frontier_label'] == 'SEPARATED_ON_PRIMARY_COST_ONLY'
    tg = rec(); tg['nolookup']['S2'] = {'cells': NT.reg(lambda c: -0.01)}
    d3 = R.d2(tg)
    assert d3['label'] == 'READER_TARGETED' and not d3['met'] and 'READER_TARGETED_STOPFIT' in d3['flags']
    miss = rec(); del miss['nolookup']['J2']
    assert R.d2(miss)['label'] == 'INCOMPLETE'
    enh = rec(); enh['enhanced'] = {'S2': {'cells': NT.reg(lambda c: -0.001)}}
    d4 = R.d2(enh)
    assert d4['label'] == R.D2_MET and 'READER_SENSITIVE_ENHANCED_S2' in d4['flags']


@test
def t_d2_altcrit_point_and_alias():
    p = R.d2(rec(altcrit='POINT'))
    a = p['comparisons']['ALTCRIT']
    assert p['label'] == R.D2_MET and a['kind'] == 'POINT' and a['spend_state'] == 'SPENDS_AT_LEAST_AS_MUCH' and a['pass']
    un = rec(altcrit='POINT'); un['costs_vs']['AP'] = costs(0.0005)                     # 0.0005 - 2 * 0.0005 <= 0
    du = R.d2(un)
    assert du['label'] == R.D2_MET and 'ALTCRIT_SPENDS_LESS_UNRESOLVED' in du['flags']
    rs = rec(altcrit='POINT'); rs['costs_vs']['AP'] = costs(0.002)                      # resolved cheaper: not credited
    dr = R.d2(rs)
    x = dr['comparisons']['ALTCRIT']
    assert dr['label'] == 'PARTIAL_START+CONT+STOPFIT+RAWJ_ONLY' and x['reason'] == 'NOT_SEPARATED_FROM_SPEND' and \
        x['passed_if_spend_matched'] is True
    sc = rec(altcrit='POINT'); sc['costs_vs']['AP'] = costs(-0.0005, {'income|nll': {'value': 0.003, 'se': 0.001}})
    assert R.d2(sc)['comparisons']['ALTCRIT']['label'] == 'SEPARATED_ON_PRIMARY_COST_ONLY'
    two = rec(altcrit='POINT'); two['paths']['ALTCRIT'] = ['AP', 'A2']
    assert R.d2(two)['label'] == 'INCOMPLETE'
    ps = rec(altcrit='POINT'); ps['selection_data']['ALTCRIT'] = 'MODEL_FIT_EPISODES'   # a point uses CAND's own sources
    assert R.d2(ps)['label'] == R.D2_MET
    al = R.d2(rec(altcrit='ALIAS_OF_STOPFIT'))
    assert al['label'] == R.D2_MET and al['gating'] == ['START', 'CONT', 'STOPFIT', 'RAWJ']
    assert 'ALTCRIT_EXACT_ALIAS_OF_STOPFIT_COUNTED_ONCE' in al['flags'] and al['comparisons']['ALTCRIT']['gating'] is False
    nw = rec(altcrit='ALIAS_OF_STOPFIT'); del nw['altcrit_alias_wire_check']
    assert R.d2(nw)['label'] == 'INCOMPLETE' and 'wire check' in R.d2(nw)['error']
    assert R.d2(rec(altcrit='ALIAS_OF_STOPFIT', stop_pair=0.0))['label'] == 'PARTIAL_START+CONT+RAWJ_ONLY'
    bk = rec(); bk['control_kind'] = {'STOPFIT': 'POINT'}
    assert R.d2(bk)['label'] == 'INCOMPLETE'                                             # STOPFIT is always a path


@test
def t_d2_controls_not_run_gates_and_inputs():
    for key in ('STOPFIT', 'ALTCRIT', 'RAWJ'):
        nr = rec(); nr['paths'][key] = None; nr['controls_not_run'] = {key: True}
        d = R.d2(nr)
        assert d['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN' and d['not_run'] == [key]
        mi = rec(); mi['paths'][key] = None
        assert R.d2(mi)['label'] == 'INCOMPLETE'
    nc = rec(); del nc['grid']['CONT']; nc['controls_not_run'] = {'CONT': True}
    assert R.d2(nc)['label'] == 'NOT_ASSESSED_CONTROLS_NOT_RUN'
    nc2 = rec(); del nc2['grid']['CONT']
    assert R.d2(nc2)['label'] == 'INCOMPLETE'
    assert R.d2(rec(grid_valid=None))['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    assert R.d2(rec(grid_valid=False))['label'] == 'NOT_COMPUTED_GRID_INVALID'
    assert R.d2(rec(cand_ev=NT.evals(occ={('U0', 'nll_excess'): 0.011})))['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'
    al = rec(); al['cand']['alias_of_start'] = True
    assert R.d2(al)['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    nf = rec(); nf['cand'] = {'status': 'NO_FEASIBLE_POINT'}
    assert R.d2(nf)['label'] == 'NOT_ASSESSED_NO_FEASIBLE_POINT'
    ws = rec(); ws['start']['id'] = 'H_LJ'
    assert R.d2(ws)['label'] == 'INCOMPLETE' and 'NEURAL_RAW' in R.d2(ws)['error']
    for key in ('STOPFIT', 'RAWJ'):
        sd = rec(); sd['selection_data'][key] = 'MODEL_SELECT'
        x = R.d2(sd)
        assert x['label'] == 'INCOMPLETE' and f'CONTROL_SELECTION_DATA_NOT_ALLOWED_{key}' in x['flags']
        ns = rec(); del ns['selection_data'][key]
        assert R.d2(ns)['label'] == 'INCOMPLETE'
    cv = rec(); del cv['costs_vs']['S1']
    assert R.d2(cv)['label'] == 'INCOMPLETE'
    ng = rec(); del ng['grid']['NEURAL_RAW']
    assert R.d2(ng)['label'] == 'INCOMPLETE'
    a2 = rec(); a2['grid']['NEURAL_RAW'] = copy.deepcopy(NT.ALIAS)
    assert 'CAND_ALIAS_OF_START' in R.d2(a2)['flags']
    fn = rec(); fn['fineness'] = R.grid_fineness(sp(upper=0.020))
    assert 'CONTROL_GRID_COARSE_STOPFIT' in R.d2(fn)['flags'] and R.d2(fn)['label'] == R.D2_MET


@test
def t_randomized_mc_through_nr():
    r = rec(); r['grid']['S2'] = {'pair': NT.entry(NT.reg(lambda c: 0.009), se=0.004, mc=0.003),
                                  'income': NT.entry(NT.reg(lambda c: 0.0)), 'occupation': NT.entry(NT.reg(lambda c: 0.0))}
    assert not R.d2(r)['comparisons']['STOPFIT']['pass']                         # 0.009 - 2 * 0.005 < 0: the MC SE counts
    r['grid']['S2']['pair'] = NT.entry(NT.reg(lambda c: 0.009), se=0.004)
    assert R.d2(r)['comparisons']['STOPFIT']['pass']                             # 0.009 - 2 * 0.004 > 0 without it


# ---------------------------------------------------------------- (6) D3, D4
@test
def t_d3_d4_lists_credit_and_no_reserve_removal():
    d = R.decisions(rec())
    assert d['D3']['label'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' and d['D3']['passed'] == list(NR.BENCHMARK)
    assert d['D3']['references'] == list(NR.BENCHMARK)                           # all nine, whatever their reserve status
    assert d['D4']['label'] == 'MATERIAL_0020_RESOLVED' and d['D4']['met']
    d2 = R.decisions(rec(refs=0.0))
    assert d2['D3']['label'] == 'NOT_SUPERIOR' and d2['D3']['passed'] == ['NEURAL_RAW'] and 'H_LJ' in d2['D3']['failing']
    assert d2['D4']['label'] == 'NOT_MET'
    assert R.decisions(rec(refs=0.025))['D4']['label'] == 'MATERIAL_0020_POINT_ONLY'
    # 0.020 is strict: mean 0.028 with se_agg 0.004 (lower bound exactly 0.020) is not resolved
    ex = rec(); ex['grid']['H_LJ'] = NT.grid_entry(0.028, se=0.004)
    assert R.decisions(ex)['D4']['label'] == 'MATERIAL_0020_POINT_ONLY'
    rb = rec(); rb['reader_budget'] = {'TWJ': 'READER_BUDGET_ASYMMETRIC'}; rb['nolookup']['TWJ'] = {'cells': NT.reg(lambda c: -0.01)}
    d4 = R.decisions(rb)
    assert d4['D3']['label'] == 'NOT_SUPERIOR' and 'TWJ' in d4['D3']['failing'] and 'TWJ' not in d4['D3']['passed']
    rc = rec(); rc['reader_budget'] = {'H_LJ': 'READER_BUDGET_ASYMMETRIC'}; del rc['nolookup']['H_LJ']
    d5 = R.decisions(rc)
    assert d5['D3']['label'] == 'INCOMPLETE' and d5['D4']['label'] == 'INCOMPLETE'
    assert d['D5']['label'] == 'NOT_APPLICABLE_NOT_JOINT' and d['D6']['label'] == 'NOT_CLAIMED'
    rn = rec(); rn['novelty'] = 'PRIOR_ART_FOUND'
    assert R.decisions(rn)['D6']['label'] == 'PRIOR_ART_FOUND'


# ---------------------------------------------------------------- (5) learning curve
def lc(r, half=0.03, diff_se=None, only=None, inner=2000):
    ids = only or ['NEURAL_RAW', 'CONT', 'S2', 'A2', 'J2', 'H_LJ']
    r['learning_curve'] = {'grid_valid': True, 'budgets': {'FULL': 4000, 'HALF': 2000}, 'inner_episode_fit_size': inner,
                           'half': {i: NT.grid_entry(half) for i in ids}}
    if diff_se is not None:
        r['learning_curve']['diff_cell_se'] = {i: {c: diff_se for c in NR.REGISTERED} for i in ids}
    return r


@test
def t_learning_curve():
    d = R.decisions(lc(rec(), half=0.03))                                       # FULL 0.03 (0.02 vs controls) vs HALF 0.03
    x = d['LC']['per_comparator']['NEURAL_RAW']
    assert x['label'] == 'NO_SHRINKAGE_AT_POINT' and abs(x['change_mean']) < 1e-12 and x['se_source'] == 'UNPAIRED_BOUND'
    assert d['LC']['per_comparator']['S2']['label'] == 'BENEFIT_SHRINKS_POINT' and d['LC']['per_comparator']['S2']['limitation']
    rs = R.decisions(lc(rec(), half=0.03, diff_se=0.002))                         # paired SE: -0.01 + 2 * 0.002 < 0
    assert rs['LC']['per_comparator']['S2']['label'] == 'BENEFIT_SHRINKS_RESOLVED' and 'LC_SE_UNPAIRED_BOUND' not in rs['LC']['flags']
    assert d['LC']['label'] == 'LEARNABILITY_LIMITATION' and 'S2' in d['LC']['limitations']
    assert 'LEARNABILITY_LIMITATION_CARRIED_INTO_HEADLINE' in d['outcome_class']['flags']
    assert 'never as information removal' in d['outcomes']['statement']
    assert d['outcome_class']['label'] == 'COMPETITIVE_CANDIDATE'                # descriptive: never changes the class
    ok = R.decisions(lc(rec(), half=0.01))                                       # every half benefit smaller: no shrinkage
    assert ok['LC']['label'] == 'NO_SHRINKAGE_SEEN' and not ok['LC']['headline_caveat']
    assert 'LEARNABILITY_LIMITATION_CARRIED_INTO_HEADLINE' not in ok['outcome_class']['flags']
    pt = R._lc_one(NT.grid_entry(0.02)['pair'], NT.grid_entry(0.024)['pair'])     # -0.004, RSS se 0.0057: point only
    assert pt['label'] == 'BENEFIT_SHRINKS_POINT' and pt['limitation']
    pd = R._lc_one(NT.grid_entry(0.02)['pair'], NT.grid_entry(0.024)['pair'], {c: 0.001 for c in NR.REGISTERED})
    assert pd['label'] == 'BENEFIT_SHRINKS_RESOLVED' and pd['se_source'] == 'PAIRED_BOOTSTRAP'
    neg = R._lc_one(NT.grid_entry(-0.01)['pair'], NT.grid_entry(0.0)['pair'])     # no benefit at FULL: no limitation
    assert neg['label'] == 'BENEFIT_SHRINKS_POINT' and not neg['limitation']
    mh = R.decisions(lc(rec(), half=0.01, only=['NEURAL_RAW', 'CONT', 'S2', 'A2', 'J2']))       # H_LJ not run
    assert mh['LC']['label'] == 'INCOMPLETE' and mh['LC']['per_comparator']['H_LJ']['label'] == 'NOT_RUN'
    assert R.decisions(rec())['LC']['label'] == 'NOT_RUN'
    ig = lc(rec(), half=0.01); ig['learning_curve']['grid_valid'] = None
    assert R.decisions(ig)['LC']['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    big = R.decisions(lc(rec(), half=0.01, inner=4000))
    assert 'INNER_EPISODE_BUDGET_AT_LEAST_EVALUATION_FIT' in big['LC']['flags']
    ni = lc(rec(), half=0.01); del ni['learning_curve']['inner_episode_fit_size']
    assert 'LC_INNER_EPISODE_SIZE_NOT_RECORDED' in R.decisions(ni)['LC']['flags']
    inf = lc(rec(cand_ev=NT.evals(occ={('U0', 'nll_excess'): 0.011})))
    assert R.decisions(inf)['LC']['label'] == 'NOT_ASSESSED_TASK_INFEASIBLE'


# ---------------------------------------------------------------- (7) outcome classes and the five outcomes
def _oc(r):
    return R.decisions(r)['outcome_class']['label']


@test
def t_outcome_classes_and_five_outcomes():
    d = R.decisions(rec())
    assert d['outcome_class']['label'] == 'COMPETITIVE_CANDIDATE' and d['registered']
    o = d['outcomes']
    assert set(o) == {'1_usefulness', '2_gain_beyond_mechanism_controls', '3_strong_references', '4_material_target',
                      '5_joint_design_and_novelty', 'statement'}
    assert o['1_usefulness']['met'] and o['2_gain_beyond_mechanism_controls']['met'] and o['4_material_target']['met']
    assert o['5_joint_design_and_novelty']['joint_design'] == 'NOT_APPLICABLE_NOT_JOINT'
    for part in ('(1) usefulness', '(2) gain beyond the mechanism-specific controls', '(3) strong references',
                 '(4) material target', '(5) joint design'):
        assert part in o['statement'], part
    assert _oc(rec(refs=0.0)) == 'MECHANISM_COMPONENT_NOT_COMPETITIVE'
    ra = R.decisions(rec(stop_pair=0.0))
    assert ra['outcome_class']['label'] == 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED' and 'D3_WITHOUT_D2' in ra['outcome_class']['flags']
    assert ra['outcome_class']['partial'] == 'PARTIAL_START+CONT+ALTCRIT+RAWJ_ONLY'
    bn = R.decisions(rec(stop_pair=0.0, alt_pair=0.0, refs=0.0))
    assert bn['outcome_class']['label'] == 'BOUNDED_NEGATIVE_EXPERIMENT' and bn['outcome_class']['partial'] == \
        'PARTIAL_START+CONT+RAWJ_ONLY' and 'STOPFIT' in bn['outcome_class']['bounds']['per_comparison']
    assert 'not a population bound' in bn['outcome_class']['bounds']['note']
    tg = rec(refs=0.0); tg['nolookup']['S2'] = {'cells': NT.reg(lambda c: -0.01)}
    dt = R.decisions(tg)
    assert dt['outcome_class']['label'] == 'BOUNDED_NEGATIVE_EXPERIMENT' and dt['outcome_class']['partial'] == 'READER_TARGETED'
    al = rec(); al['cand']['alias_of_start'] = True
    da = R.decisions(al)
    assert da['outcome_class']['label'] == 'NO_OUTPUT_CHANGE' and da['D3']['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    nr = rec(); nr['paths']['STOPFIT'] = None; nr['controls_not_run'] = {'STOPFIT': True}
    assert _oc(nr) == 'MECHANISM_NOT_ASSESSED'
    assert _oc(rec(grid_valid=None)) == 'INCOMPLETE'
    bad = NT.evals(occ={('U0', 'nll_excess'): 0.011})
    inf = rec(cand_ev=bad)
    assert _oc(inf) == 'TASK_INFEASIBLE'
    inf['points'] = [{'config_id': 'CAND', 'evals': bad}, {'config_id': 'P2', 'evals': NT.evals()}]
    assert _oc(inf) == 'SELECTED_POINT_INFEASIBLE'
    inf['points'] = None
    assert _oc(inf) == 'INCOMPLETE'
    nf = rec(); nf['cand'] = {'status': 'NO_FEASIBLE_POINT'}
    assert _oc(nf) == 'TASK_INFEASIBLE'
    rr = rec(); rr['d2r'] = {'label': 'ROBUSTNESS_NOT_CONFIRMED', 'sign_changes': ['S2'], 'recipient_failures': ['H_LJ'],
                             'headline_caveat': 'sign change vs S2; recipient failure vs H_LJ'}
    dr = R.decisions(rr)
    assert dr['outcome_class']['label'] == 'COMPETITIVE_CANDIDATE' and {'D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE',
                                                                        'D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE'} <= \
        set(dr['outcome_class']['flags']) and 'recipient failure vs H_LJ' in dr['outcomes']['statement']
    nq = R.decisions(rec(cand_ev=NT.evals(occ={('U0', 'nll_excess'): 0.0095})))
    assert 'POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED' in nq['outcome_class']['flags']
    ov = rec(); ov['cand']['operating_point'] = {0: {'epoch': 6}, 1: {'epoch': 9}}
    assert 'CAND_OPERATING_POINT_DIFFERS_BY_VERSION' in R.decisions(ov)['outcome_class']['flags']


# ---------------------------------------------------------------- registration and the campaign summary
@test
def t_families_and_registration():
    for fam in ('FA', 'FB', 'CTL', 'rfa', None):
        try:
            R.decisions(rec() | {'family': fam})
            raise AssertionError(f'family {fam!r} accepted')
        except ValueError:
            pass
    assert R.adaptive_tag(rec()) is None
    assert R.adaptive_tag(rec(revision=1)) == 'REVISED_R1' and R.adaptive_tag(rec(revision=1, early=False)) == 'ADAPTIVE_R1'
    assert R.adaptive_tag(rec(early=False)) == 'ADAPTIVE_R0_LATE_CARD'
    r1 = R.decisions(rec(revision=1, early=False))
    assert r1['outcome_class']['label'] == 'ADAPTIVE_R1__COMPETITIVE_CANDIDATE' and r1['D2']['label'].startswith('ADAPTIVE_R1__')
    assert r1['LC']['label'].startswith('ADAPTIVE_R1__') and not r1['registered']
    assert 'REVISION_BEYOND_PLANNED' in R.decisions(rec(revision=2))['outcome_class']['flags']
    fs = rec(); fs['family'] = 'FS_X2'; fs['start']['id'] = 'NEURAL_RAW'
    assert R.decisions(fs)['adaptive_tag'] == 'REVISED_FS'


@test
def t_campaign_summary():
    a = R.decisions(rec())
    b = R.decisions(lc(rec(revision=1), half=0.03))
    c = R.decisions(rec(revision=1, early=False, stop_pair=0.0))
    s = R.campaign_summary([a, b, c])
    assert set(s['registered']) == {'RFA'} and s['d2_looks'] == 3 and len(s['d2_met']) == 2
    assert 'D2 was met in 1 of 1 registered readings and in 1 of 2 non-registered readings (3 D2 looks in total)' in \
        s['statement'] and 'no multiplicity correction' in s['statement'] and len(s['learnability_limitations']) == 1
    try:
        R.campaign_summary([a, a])
        raise AssertionError('two registered readings accepted')
    except ValueError:
        pass
    assert R.campaign_summary([b, c])['flags'] == ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS']


# ---------------------------------------------------------------- (8) D2R
def rob_entry(reg_x, rob_x, reps=('R3', 'R4'), rob_inc=0.0):
    cells, ses, inc = {}, {}, {}
    for v in NR.VERSIONS:
        for r in ('R1', 'R2') + reps:
            for k in (0, 1, 2, 3):
                c = f'S{v}{r}k{k}'
                cells[c] = reg_x if c in NR.REGISTERED else rob_x
                inc[c] = 0.0 if c in NR.REGISTERED else rob_inc
                ses[c] = 0.004
    z = {'cells': {c: 0.0 for c in cells}, 'cell_se': dict(ses)}
    return {'pair': {'cells': cells, 'cell_se': ses}, 'income': {'cells': inc, 'cell_se': dict(ses)}, 'occupation': z}


@test
def t_d2r():
    refit = {'label': 'REFIT_GENUINELY_EFFECTIVE', 'counted_once': []}
    comps = {'START': 'NEURAL_RAW', 'CONT': 'CONT', 'STOPFIT': 'S2', 'ALTCRIT': 'A2', 'RAWJ': 'J2'}
    base = {'grid_valid': True, 'refit': refit, 'comparators': dict(comps),
            'rob': {k: rob_entry(0.03, 0.03) for k in ('NEURAL_RAW', 'CONT', 'S2', 'A2', 'J2', 'H_LJ', 'SUCC')}}
    d = R.d2r(base)
    assert d['label'] == 'ROBUSTNESS_CONFIRMED' and d['decisive'] == ['NEURAL_RAW', 'CONT', 'S2', 'A2', 'J2'], d
    assert d['beside'] == ['H_LJ', 'SUCC'] and len(set(d['beside'])) == len(d['beside'])
    sc = copy.deepcopy(base); sc['rob']['S2'] = rob_entry(0.02, -0.01)
    d2 = R.d2r(sc)
    assert d2['label'] == 'ROBUSTNESS_NOT_CONFIRMED' and d2['sign_changes'] == ['S2'] and 'headline' in d2['headline_caveat']
    rf = copy.deepcopy(base); rf['rob']['H_LJ'] = rob_entry(0.03, 0.03, rob_inc=0.02)          # beside: still carried
    d3 = R.d2r(rf)
    assert d3['label'] == 'ROBUSTNESS_CONFIRMED' and d3['recipient_failures'] == ['H_LJ'] and 'recipient' in d3['headline_caveat']
    rd = copy.deepcopy(base); rd['rob']['J2'] = rob_entry(0.03, 0.03, rob_inc=0.02)            # decisive: not confirmed
    assert R.d2r(rd)['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    nf = copy.deepcopy(base); nf['refit'] = {'label': 'REFIT_NOT_GENUINELY_EFFECTIVE'}
    assert R.d2r(nf)['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    nr = copy.deepcopy(base); del nr['rob']['A2']
    assert R.d2r(nr)['label'] == 'NOT_RUN'
    inc = copy.deepcopy(base); del inc['comparators']['RAWJ']
    assert R.d2r(inc)['label'] == 'INCOMPLETE'
    al = copy.deepcopy(base); al['comparators']['ALTCRIT'] = 'ALIAS_OF_STOPFIT'
    assert R.d2r(al)['decisive'] == ['NEURAL_RAW', 'CONT', 'S2', 'J2']
    bad = copy.deepcopy(base); bad['comparators']['RAWJ'] = 'ALIAS_OF_STOPFIT'
    assert R.d2r(bad)['label'] == 'INCOMPLETE'
    br = copy.deepcopy(base); br['comparators']['STOPFIT'] = {'bracket': 'S2', 'status': 'BRACKETED'}
    br['comparators']['ALTCRIT'] = {'bracket': None, 'status': 'NO_BRACKET'}
    assert R.d2r(br)['decisive'] == ['NEURAL_RAW', 'CONT', 'S2', 'J2']
    one = copy.deepcopy(base); one['rob'] = {k: rob_entry(0.03, 0.03, reps=('R3',)) for k in base['rob']}
    assert R.d2r(one)['label'] == 'ROBUSTNESS_CONFIRMED'
    assert R.d2r({**base, 'grid_valid': None})['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'


# ---------------------------------------------------------------- card RF_RFIT_v1
def card_rec(stop=0.02, altc=0.02, rawj=0.02, lag=0.05, **kw):
    """The card's record shape: CAND RF_FULL, CONT RF_TASK, STOPFIT [RF_STOP_LO, RF_STOP_HI], ALTCRIT [RF_ALTC_HI] and RAWJ
    [RF_RAWJ_HI] (LO recorded, not saved), F0 active in both versions, family FAM."""
    r = rec(**kw)
    g, cv, nl = r['grid'], r['costs_vs'], r['nolookup']
    for old in STOP + ALT + RAWJ + ['CONT']:
        g.pop(old, None); cv.pop(old, None)
    g.update({'RF_TASK': NT.grid_entry(0.03), 'RF_STOP_LO': NT.grid_entry(stop), 'RF_STOP_HI': NT.grid_entry(stop),
              'RF_ALTC_HI': NT.grid_entry(altc), 'RF_RAWJ_HI': NT.grid_entry(rawj)})
    cv.update({'RF_TASK': costs(0.004), 'RF_STOP_LO': costs(0.001), 'RF_STOP_HI': costs(-0.001), 'RF_ALTC_HI': costs(-0.0008),
               'RF_RAWJ_HI': costs(-0.0005)})
    nl.update({i: {'cells': NT.reg(lambda c: 0.01)} for i in ('RF_TASK', 'RF_STOP_LO', 'RF_STOP_HI', 'RF_ALTC_HI', 'RF_RAWJ_HI')})
    r.update({'family': 'FAM', 'card': 'RF_RFIT_v1', 'cont': {'config_id': 'RF_TASK', 'evals': NT.evals()},
              'paths': {'STOPFIT': ['RF_STOP_LO', 'RF_STOP_HI'], 'ALTCRIT': ['RF_ALTC_HI'], 'RAWJ': ['RF_RAWJ_HI']},
              'control_kind': {'ALTCRIT': 'PATH'}, 'lo_not_saved': ['ALTCRIT', 'RAWJ'],
              'f0': {0: 'MECHANISM_ACTIVE', 1: 'MECHANISM_ACTIVE'}, 'altcrit_median_relative_lag': lag})
    r['cand']['config_id'] = 'RF_FULL'
    r['cand']['selection_uses_sex_recovery_diagnostic'] = False
    return r


@test
def t_card_rf_rfit_v1():
    d = R.decisions(card_rec())
    assert d['family'] == 'RFA' and d['registered'] and d['outcome_class']['label'] == 'COMPETITIVE_CANDIDATE', d['D2'].get('error')
    c = d['D2']['comparisons']
    assert c['ALTCRIT']['path'] == ['NEURAL_RAW', 'RF_ALTC_HI'] and c['ALTCRIT']['point'] == 'RF_ALTC_HI' and not c['ALTCRIT']['lo_saved']
    assert c['STOPFIT']['point'] == 'RF_STOP_HI' and c['STOPFIT']['lo_saved'] and c['RAWJ']['point'] == 'RF_RAWJ_HI'
    assert {'LO_NOT_SAVED_SPEC_BUDGET_ALTCRIT', 'LO_NOT_SAVED_SPEC_BUDGET_RAWJ', 'F0_SEX_BASED_FIT_GATE_DISCLOSED'} <= \
        set(d['outcome_class']['flags']) and c['ALTCRIT']['independent']
    f = d['falsifiers']['per_falsifier']
    assert f['F0'] == 'NOT_TRIGGERED' and f['F2'] == 'NOT_TRIGGERED' and f['F4'] == 'NOT_RUN' and f['F3'] == 'NOT_RECORDED'
    # STOPFIT tie: F2 triggered, PARTIAL named, the fit path not isolated
    t = R.decisions(card_rec(stop=0.0))
    assert t['falsifiers']['per_falsifier']['F2'] == 'TRIGGERED' and 'FALSIFIER_F2_TRIGGERED' in t['outcome_class']['flags']
    assert t['D2']['label'] == 'PARTIAL_START+CONT+ALTCRIT+RAWJ_ONLY'
    # near alias by the 1% lag rule: still gating, never independent; a failing near-alias ALTCRIT still blocks D2
    na = R.decisions(card_rec(lag=0.004))
    assert na['D2']['label'] == R.D2_MET and na['D2']['gating'] == ['START', 'CONT', 'STOPFIT', 'ALTCRIT', 'RAWJ']
    assert 'ALTCRIT_NEAR_ALIAS_OF_STOPFIT_NOT_INDEPENDENT' in na['D2']['flags'] and not na['D2']['comparisons']['ALTCRIT']['independent']
    assert na['D2']['scope'] == R.D2_SCOPE_NEAR_ALIAS and 'not counted as an independent control' in \
        na['outcomes']['2_gain_beyond_mechanism_controls']['scope']
    assert R.d2(card_rec(lag=0.004, altc=0.0))['label'] == 'PARTIAL_START+CONT+STOPFIT+RAWJ_ONLY'
    assert 'ALTCRIT_NEAR_ALIAS_OF_STOPFIT_NOT_INDEPENDENT' not in R.d2(card_rec(lag=0.01))['flags']        # strict < 1%
    nl = card_rec(); del nl['altcrit_median_relative_lag']
    assert 'ALTCRIT_LAG_NOT_RECORDED' in R.d2(nl)['flags']
    bd = card_rec(lag=0.05); bd['altcrit_near_alias'] = True
    assert R.d2(bd)['label'] == 'INCOMPLETE'                                         # declaration disagrees with the lag rule
    # LO not saved: at most one saved point
    two = card_rec(); two['lo_not_saved'] = ['STOPFIT']
    assert R.d2(two)['label'] == 'INCOMPLETE'
    # a missing HI (the bracket point does not exist): the path is START alone; CAND spends more -> NO_BRACKET
    nh = card_rec(); nh['paths']['RAWJ'] = []
    dn = R.d2(nh)
    assert dn['comparisons']['RAWJ']['reason'] == 'NO_BRACKET' and dn['label'] == 'PARTIAL_START+CONT+STOPFIT+ALTCRIT_ONLY'
    # F0: inert in both / one version; inconsistent records
    for inert in ({0: 'MECHANISM_INERT', 1: 'MECHANISM_INERT'}, {0: 'MECHANISM_ACTIVE', 1: 'MECHANISM_INERT'}):
        m = card_rec(); m['f0'] = inert; m['cand'] = {'status': 'MECHANISM_INERT'}
        dm = R.decisions(m)
        assert dm['outcome_class']['label'] == 'MECHANISM_INERT' and dm['D1']['label'] == 'NOT_FITTED_MECHANISM_INERT'
        assert dm['D2']['label'] == 'NOT_ASSESSED_MECHANISM_INERT' and dm['D3']['label'] == 'NOT_ASSESSED_MECHANISM_INERT'
        assert dm['LC']['label'] == 'NOT_ASSESSED_MECHANISM_INERT' and dm['falsifiers']['per_falsifier']['F0'] == 'TRIGGERED'
    assert 'MECHANISM_INERT_IN_ONE_VERSION_ONLY' in dm['outcome_class']['flags']
    bad = card_rec(); bad['f0'] = {0: 'MECHANISM_ACTIVE', 1: 'MECHANISM_INERT'}
    assert R.decisions(bad)['outcome_class']['label'] == 'INCOMPLETE'                 # FITTED with an inert version
    nr = card_rec(); nr['cand'] = {'status': 'MECHANISM_INERT'}
    assert R.decisions(nr)['outcome_class']['label'] == 'INCOMPLETE'                  # inert without the inert record
    nf = card_rec(); del nf['f0']
    assert 'F0_NOT_RECORDED' in R.decisions(nf)['outcome_class']['flags']
    # the family alias is one family: two registered readings under RFA and FAM are refused
    try:
        R.campaign_summary([R.decisions(card_rec()), R.decisions(rec())])
        raise AssertionError('two registered readings (FAM and RFA) accepted')
    except ValueError:
        pass
    assert R.adaptive_tag(card_rec(revision=1)) == 'REVISED_R1'
    fr = card_rec(); fr['falsifier_records'] = {'F3': 'TRIGGERED', 'F6': 'maybe'}
    ff = R.decisions(fr)['falsifiers']['per_falsifier']
    assert ff['F3'] == 'TRIGGERED' and ff['F6'] == 'NOT_RECORDED'
    lcr = R.decisions(lc(card_rec(), half=0.01, only=['NEURAL_RAW', 'RF_TASK', 'RF_STOP_HI', 'RF_ALTC_HI', 'RF_RAWJ_HI', 'H_LJ'],
                         diff_se=0.001))
    assert lcr['LC']['label'] == 'NO_SHRINKAGE_SEEN' and lcr['falsifiers']['per_falsifier']['F4'] == 'NOT_TRIGGERED'


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
    out = {'module': 'rfa.test_rules', 'rules_sha256': R.load_rules()['_sha256'],
           'rfa_rules_py_sha256': hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(), 'passed': n_pass,
           'total': len(res), 'cpu_seconds': round(time.process_time() - t0, 3), 'results': res}
    print(json.dumps(out, indent=1))
    return 0 if n_pass == len(res) else 1


if __name__ == '__main__':
    sys.exit(main())

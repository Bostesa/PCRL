"""Independent arithmetic fixtures; stdlib only apart from the test runner."""
import ast
import copy
import itertools
import json
import math
from pathlib import Path

from scripts import verify_acs_source_guard_comparisons as replay


RULES = json.loads((Path(__file__).parents[1]/'results/redesign_20260909_acs_source_guard_v1/comparison_rules.json').read_text())


def point(loss=.30, gain=.02):
    return {'utility': {t: loss for t in replay.TASKS}, 'gain': {t: gain for t in replay.AUDITS},
            'ids': {t: 'fixed' for t in replay.AUDITS}}


def compare(j=None, local=None, parent=None, panel=None, delta=.001):
    return replay.independent_pair(j or point(gain=.01), local or point(), parent or {t: .30 for t in replay.SOURCE},
        panel or replay.TASKS, delta, 'SEX', 'G_L025')


def test_independent_source_floors_use_both_methods_every_task_and_current_pool():
    j = point(gain=.01);j['utility']['income_binary'] = .32;j['utility']['civilian_at_work'] = .20
    assert compare(j)['both_source_feasible'] is False
    local = point();local['utility']['public_coverage'] = .32
    result = compare(local=local)
    assert 'G_L025/source/public_coverage' in result['exclusion_reasons']['close']
    assert compare(point(loss=.305, gain=.01))['qualifies_close'] is False  # Utility mismatch, source passes.
    assert replay.source_floor(point(loss=.305)['utility'], {t: .30 for t in replay.SOURCE})['pass']
    assert not replay.source_floor(point(loss=.305)['utility'], {t: .28 for t in replay.SOURCE})['pass']


def test_close_directional_missing_and_signed_gain_are_distinct():
    result = compare(point(loss=.20, gain=-.01))
    assert result['qualifies_directional'] is True and result['qualifies_close'] is False
    assert result['guarded_J_gain'] == -.01
    j = point(gain=.01);j['utility']['income_binary'] = .4;j['utility']['same_residence'] = None
    result = compare(j)
    assert result['qualifies_close'] is None and 'G_J/source/income_binary' in result['exclusion_reasons']['close']
    j = point(gain=.01);j['utility']['commute_over20'] = None
    assert compare(j, panel=replay.SOURCE)['qualifies_close'] is True
    assert compare(j)['qualifies_close'] is None


def test_three_seed_aggregation_preserves_failed_seeds_and_eligibility():
    rows = {}
    for seed, gain in enumerate([-.08, -.18, .22]):
        rows[str(seed)] = compare(point(gain=gain))
    result = replay.independent_aggregate(rows)
    assert result['close_source_and_utility_eligible_seed_count'] == 3
    assert result['close_qualifying_seed_count'] == 2
    assert math.isclose(result['gain_difference_mean'], -.1/3)
    assert result['gain_difference_per_seed']['2'] > 0
    rows['1']['gain_difference'] = None
    assert replay.independent_aggregate(rows)['gain_difference_mean'] is None


def test_ordinary_pareto_uses_full_vector_roundoff_and_unknown_alternatives():
    points = {'a': {'u': .3005, 'g': .01, 'source': .4}, 'b': {'u': .30, 'g': .02, 'source': .3}}
    result = replay.ordinary_pareto(points, ('u', 'g', 'source'))
    assert {row['status'] for row in result.values()} == {'nondominated'}
    points['missing'] = {'u': .1, 'g': None, 'source': .1}
    result = replay.ordinary_pareto(points, ('u', 'g', 'source'))
    assert result['missing']['status'] == 'unassessable_point'
    assert result['a']['status'] == 'not_dominated_among_assessable_points'
    tied = replay.ordinary_pareto({'a': {'u': .3}, 'b': {'u': .3+5e-13}}, ['u'])
    assert all(row['status'] == 'nondominated' for row in tied.values())


def synthetic_replay():
    r = replay.Replay.__new__(replay.Replay);r.rules = copy.deepcopy(RULES)
    r.entries = [dict(seed=seed, interface=interface, arm=arm, condition=interface+'_'+arm,
        reused=arm not in replay.NEW) for seed, interface, arm in itertools.product((0, 1, 2), ('F', 'P'), replay.ARMS)]
    r.alias = {(e['seed'], e['interface'], e['arm']): e['condition'] for e in r.entries}
    r.names = {e['condition']: e['interface'] for e in r.entries if e['seed'] == 0}
    r.parent = {(seed, split, weight, task): .30 for seed, split, weight, task in itertools.product((0, 1, 2), RULES['evaluation_splits'], RULES['weights'], replay.TASKS)}
    r.points = {(e['seed'], e['condition'], split, weight, scope, budget): point(gain=.01 if e['arm']=='G_J' else .02)
        for e, split, weight, scope, budget in itertools.product(r.entries, RULES['evaluation_splits'], RULES['weights'], RULES['audit_scopes'], RULES['audit_budgets'])}
    r.metric = lambda *args: {'log_loss': .30, 'prior_loss': .5, 'present': True}
    return r


def test_complete_fixed_identity_counts_and_validation_weight_separation():
    r = synthetic_replay();ids = list(r.identities())
    assert len(ids) == len(set(ids)) == 3072
    assert len({(seed, identity) for seed, identity in itertools.product((0, 1, 2), ids)}) == 9216
    selected = next(i for i in ids if i[0]=='F' and i[2]=='G_L025' and i[6]=='validation' and i[7]=='person_weighted' and i[8]=='expanded_catchup' and i[9]==360)
    changed_key = (1, 'F_G_J', 'validation', 'person_weighted', 'expanded_catchup', 360)
    r.points[changed_key]['utility']['public_coverage'] = .32
    assert r.pair(1, selected)['both_source_feasible'] is False
    development = tuple('development_evaluation' if k==6 else v for k,v in enumerate(selected))
    assert r.pair(1, development)['both_source_feasible'] is True
    assert r.pair(0, selected)['both_source_feasible'] is True
    assert len(list(r.endpoints())) == 66
    contrasts = list(r.contrast_expected())
    assert len(contrasts) == 19536
    assert len({identity for _, identity, _ in contrasts}) == 6512


def test_legacy_criteria_preserve_signed_and_undefined_headrooms():
    residence = replay.residence_margin(.3, .35, .4, .5)
    assert residence['pass'] is True and math.isclose(residence['retained_fraction'], .5)
    undefined = replay.residence_margin(.4, .3, .3, .5)
    assert undefined['ratio_defined'] is False and undefined['pass'] is None
    race = replay.halving_margin(.4, .5, .6, False)
    assert race['numeric_halving_inequality'] is True and race['pass'] is None
    assert race['retained_fraction'] is None and race['undefined_reason'] == 'coverage_limited'
    assert replay.legacy_joint((False, None)) is False
    assert replay.conjunction((False, None)) is None


def test_replay_does_not_import_training_or_reporter_arithmetic():
    for module in (replay, replay.plumbing):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(name.startswith(('numpy', 'torch', 'scipy', 'experiments')) or 'summarize_' in name
                       or 'report_acs' in name or name.endswith('acs_coalition_strength_comparisons') for name in imports)

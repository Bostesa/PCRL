"""Focused read-only joins, split identities and fixed guarded comparisons."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from scripts import report_acs_source_guard as report

RULES=json.loads((Path(__file__).parents[1]/'results/redesign_20260909_acs_source_guard_v1/comparison_rules.json').read_text())


def manifest():
    entries=[]
    for seed, interface, arm in itertools.product(RULES['seeds'],RULES['interfaces'],report.ARMS):
        entries.append(dict(seed=seed,interface=interface,arm=arm,condition=interface+'_'+arm,
            reused=arm not in report.NEW_ARMS,forward_alias=f'seed_{seed}/T_shared' if arm=='T' else None))
    return {'systems':entries}


def synthetic_points(entries):
    points={}
    for entry, split, weight, scope, budget in itertools.product(entries,RULES['evaluation_splits'],RULES['weights'],RULES['audit_scopes'],RULES['audit_budgets']):
        utility={t:.3 for t in report.UTILITY_TASKS};parent={t:.3 for t in report.SOURCE_TASKS}
        gains={v+'/'+t:.01 if entry['arm']=='G_J' else .02 for v,ts in report.AUDIT_ROLES.items() for t in ts}
        points[entry['seed'],entry['condition'],split,weight,scope,budget]={**entry,'evaluation_split':split,'weighting':weight,
            'scope':scope,'audit_budget':budget,'utility':utility,'parent':parent,'gains':gains,
            'source_feasibility':report.source_check({'utility':utility},parent),
            'candidate_ids':{k:'fixed' for k in gains},'coverage':{k:{'complete':not k.endswith('RAC1P')} for k in gains}}
    return points


def test_matrix48_alias_keeps_distinct_source_only_interfaces():
    entries,aliases=report.registry(manifest(),RULES)
    assert len(entries)==len(aliases)==48
    assert sum(not r['reused'] for r in entries)==24
    for seed in RULES['seeds']:
        a,b=aliases[seed,'F','T'],aliases[seed,'P','T']
        assert a is not b and a['condition']!=b['condition'] and a['forward_alias']==b['forward_alias']
    broken=manifest();broken['systems'][0]['reused']=True
    with pytest.raises(ValueError):report.registry(broken,RULES)


def test_exact_comparison_cube_and_no_split_or_local_stitching():
    entries,aliases=report.registry(manifest(),RULES)
    cube=report.pair_cube(synthetic_points(entries),aliases,RULES)
    assert len(cube)==9216
    assert len({tuple(r[k] for k in ('seed',*report.PAIR_FIELDS)) for r in cube})==9216
    aggregate=report.fixed_seed_summary(cube,report.PAIR_FIELDS)
    assert len(aggregate)==3072
    assert all(r['close_qualifying_seed_count']==3 and r['present_seeds']==[0,1,2] for r in aggregate)
    assert all(r['attribute_full_schema_assessment']=='unassessable' for r in cube if r['attribute']=='RAC1P')
    assert {r['evaluation_split'] for r in aggregate}=={'validation','development_evaluation'}


def test_validation_source_pass_does_not_fill_failed_development_or_weighted_parent():
    entries,aliases=report.registry(manifest(),RULES);points=synthetic_points(entries)
    name=aliases[1,'F','G_J']['condition']
    for scope,budget in itertools.product(RULES['audit_scopes'],RULES['audit_budgets']):
        points[1,name,'development_evaluation','unweighted',scope,budget]['utility']['public_coverage']=.32
    cube=report.pair_cube(points,aliases,RULES)
    use=[r for r in cube if r['seed']==1 and r['interface']=='F' and r['attribute']=='SEX']
    assert all(r['qualifies_close'] is False for r in use if r['evaluation_split']=='development_evaluation' and r['weighting']=='unweighted')
    assert all(r['qualifies_close'] is True for r in use if r['evaluation_split']=='validation' or r['weighting']=='person_weighted')
    assert report.parent_scores(RULES,0,'validation','person_weighted')['income_binary']==RULES['original_parent_metric_identity']['0']['tasks']['income_binary']['scores']['validation_person_weighted']
    assert report.parent_scores(RULES,0,'development_evaluation','unweighted')['income_binary']!=report.parent_scores(RULES,0,'validation','unweighted')['income_binary']


def test_unavailable_guarded_point_not_replaced_by_unguarded_or_source_only():
    entries,aliases=report.registry(manifest(),RULES);points=synthetic_points(entries)
    for key,p in points.items():
        if p['seed']==2 and p['arm']=='G_L20':
            p['utility']['public_coverage']=None
    cube=report.pair_cube(points,aliases,RULES)
    assert all(r['qualifies_close'] is None for r in cube if r['seed']==2 and r['local_arm']=='G_L20')
    assert all(r['qualifies_close'] is True for r in cube if r['local_arm']=='G_L025')
    summary=report.fixed_seed_summary(cube,report.PAIR_FIELDS)
    assert all(r['gain_difference_mean']==pytest.approx(-.01) for r in summary)  # Preserve observed gain despite source failure.
    assert all(r['close_qualifying_seed_count']==2 and r['close_all_seeds_qualify'] is None for r in summary if r['local_arm']=='G_L20')


def test_native_source_scores_cannot_substitute_for_selected_probe():
    entry=manifest()['systems'][0]
    rows=[]
    for task in report.SOURCE_TASKS:
        rows.append(dict(seed=entry['seed'],condition=entry['condition'],role='native',view=report.UTILITY_VIEW[task],target=task,
            audit_budget=None,candidate_id='native',selected_scopes=['native'],scores={'test':{'log_loss':.001}}))
    index=report.historical.build_index(rows)
    points=report.points_from_index(index,[entry],RULES)
    assert all(p['source_feasibility']['pass'] is None and all(v is None for v in p['utility'].values()) for p in points.values())


def test_single_budget_scope_weight_change_stays_in_its_exact_identity():
    entries,aliases=report.registry(manifest(),RULES);points=synthetic_points(entries)
    key=(0,'F_G_J','development_evaluation','person_weighted','expanded_catchup',360)
    points[key]['gains']['AB/SEX']=.03
    cube=report.pair_cube(points,aliases,RULES)
    changed=[r for r in cube if r['gain_difference']>0]
    assert len(changed)==2*4*4
    assert all((r['seed'],r['interface'],r['evaluation_split'],r['weighting'],r['scope'],r['audit_budget'],r['attribute'])==(0,'F','development_evaluation','person_weighted','expanded_catchup',360,'SEX') for r in changed)


def test_all_contrasts_are_fixed_and_T_alias_not_an_extra_independent_fit():
    _,aliases=report.registry(manifest(),RULES)
    endpoints=report.comparison_endpoints(aliases,RULES)
    assert len(endpoints)==66
    assert len({tuple(sorted(r.items())) for r in endpoints})==66
    assert sum(r['comparison']=='guarded_minus_unguarded' for r in endpoints)==18
    assert sum(r['comparison']=='F_minus_P' and r['left_arm']=='T' for r in endpoints)==3


def test_SEX_improvement_is_not_full_vector_dominance():
    entries,aliases=report.registry(manifest(),RULES);points=synthetic_points(entries)
    key=(0,'F_G_J','development_evaluation','unweighted','expanded_catchup',360)
    points[key]['gains']['B/RAC1P']=.03
    rows=report.vector_tradeoffs(report.pair_cube(points,aliases,RULES))
    selected=[r for r in rows if r['seed']==0 and r['interface']=='F' and r['evaluation_split']=='development_evaluation' and r['weighting']=='unweighted' and r['scope']=='expanded_catchup' and r['audit_budget']==360]
    assert len(selected)==2
    assert all(r['guarded_J_dominates_local_numeric'] is False and r['components_worse_for_guarded_J']==['gain/B/RAC1P'] for r in selected)

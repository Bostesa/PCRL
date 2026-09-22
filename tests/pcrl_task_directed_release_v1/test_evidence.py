"""Synthetic aggregate-only evidence contract; no ACS or fitted artifacts."""
import copy
import importlib
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import config, reporting, run


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.evidence')


def scores(value):
    return {'unweighted': value, 'weighted': value + .02, 'balanced': value + .01}


def fixture():
    roles = ['attack:' + r for r in config.PRIMARY + config.SECONDARY] + ['utility:' + r for r in config.UTILITY]
    records = {}; receipts = []; audits = []; ledger = []
    for i, name in enumerate(('H','J','continuous_task','T0_code')):
        records[name] = {}
        for anchor in (0,1,2):
            native = {}
            for role in roles:
                ce = scores(.5 + anchor * .03 - .01*i)
                nclass = 9 if role.endswith('/RAC1P') else 2
                per_class = [{'class': c, 'support': 2 if c < 2 else 0,
                              'weight_sum': 2. if c < 2 else 0., 'ess': 2. if c < 2 else 0.,
                              'ce': ce if c < 2 else {k: None for k in ce}} for c in range(nclass)]
                entry = {'ce': ce, 'accuracy': {'unweighted': .7, 'weighted': .72},
                         'support': 4, 'weight_sum': 4., 'ess': 4., 'per_class': per_class,
                         'selection': 'chosen', 'independent_selection': 'own',
                         'independent_ce': scores(ce['unweighted'] + .03),
                         'selection_source': 'frozen validation registry; no evaluation selection'}
                if role == 'utility:A/same_residence' and name != 'T0_code':
                    entry['fixed_decoder_ce'] = scores(ce['unweighted'] + .04)
                native[role] = entry
            records[name][str(anchor)] = {'test': native}
            receipts.append({'configuration': name, 'anchor': anchor,
                             'summary_sha256': chr(97+i)*64, 'receipt_sha256': 'f'*64})
            audits.append({'configuration': name, 'anchor': anchor, 'registry_sha256': chr(97+i)*64,
                           'receipt_sha256': 'e'*64, 'role_audits': len(roles),
                           'new_role_fits': len(roles) if name == 'H' else 9,
                           'reused_role_audits': 0 if name == 'H' else len(roles)-9})
            ledger.append({'configuration': name, 'anchor': anchor, 'kind': 'control'})
    grid = {'phase': 'evaluation', 'pool': 'test', 'selection_frozen': True,
            'records': records, 'receipts': receipts, 'audit_receipts': audits,
            'map_receipts': [], 'selection_sha256': '1'*64}
    choices = {n: {a: {r: {'selection': 'chosen', 'independent_selection': 'own'} for r in roles}
                   for a in ('0','1','2')} for n in records}
    selection = {'selection_frozen': True, 'evaluation_configurations': list(records),
                 'predictor_choices': choices, 'families': {}, 'routes': {},
                 'contrasts_sha256':'2'*64,'contrast_family_size':1}
    return grid, selection, ledger


def test_complete_equal_anchor_means_decomposition_recovery_secondary_and_support():
    m = module(); grid, selected, ledger = fixture()
    result = m.build_evidence(grid, selection=selected, ledger=ledger)
    q = result['configurations']['continuous_task']
    assert q['task']['selected']['PWGTP'] == pytest.approx(.53)
    assert q['task']['independent']['unweighted'] == pytest.approx(.54)
    assert q['task']['fixed']['unweighted'] == pytest.approx(.55)
    sensitive = q['primary_sensitive']['A/SEX']['PWGTP']
    assert sensitive['recovery_over_H'] == pytest.approx(.02)
    assert sensitive['recovery_increment_over_J'] == pytest.approx(.01)
    assert sensitive['J_minus_H_loss'] == pytest.approx(-.01)
    assert len(q['secondary_roles']) == 7 and len(q['original_source_utility']) == 4
    support = q['roles']['attack:A/RAC1P']['support_by_anchor']['0']
    assert len(support['per_class']) == 9 and support['per_class'][8]['support'] == 0
    assert support['per_class'][8]['ce']['weighted'] is None
    assert result['configurations']['T0_code']['task']['fixed'] is None
    assert len(result['baseline_families']) == 8
    assert 'optnet16_C1' in result['historical_baselines']


def test_partial_anchors_are_listed_not_averaged_and_counts_do_not_invent_fits():
    m = module(); grid, selected, ledger = fixture()
    grid['records']['continuous_task'].pop('2')
    grid['receipts'] = [r for r in grid['receipts'] if (r['configuration'],r['anchor']) != ('continuous_task',2)]
    evidence = m.build_evidence(grid, selection=selected, ledger=ledger)
    assert evidence['configurations']['continuous_task']['status'] == 'incomplete'
    assert 'task' not in evidence['configurations']['continuous_task']
    counts = evidence['execution']
    assert counts['nominal_release_anchor_units'] == 12
    assert counts['accepted_evaluation_units'] == 11 and counts['incomplete_evaluation_units'] == 1
    assert counts['accepted_audit_units'] == 12
    assert counts['new_role_fit_units'] + counts['reused_role_audit_units'] == 12*16
    assert counts['unique_registry_artifacts'] == 4
    assert counts['mathematical_duplicate_release_count'] is None


@pytest.mark.parametrize('fault', ['wrong_selection','duplicate_receipt','missing_receipt','nonfinite',
                                  'injected_person_ids','class_schema','mixed_pool'])
def test_unaccepted_misaligned_or_nonpublic_native_summaries_fail(fault):
    m = module(); grid, selected, ledger = fixture()
    entry = grid['records']['H']['0']['test']['utility:A/same_residence']
    if fault == 'wrong_selection': entry['selection'] = 'evaluation_selected'
    if fault == 'duplicate_receipt': grid['receipts'].append(copy.deepcopy(grid['receipts'][0]))
    if fault == 'missing_receipt': grid['receipts'].pop()
    if fault == 'nonfinite': entry['ce']['weighted'] = float('nan')
    if fault == 'injected_person_ids': entry['ids'] = ['private-person']
    if fault == 'class_schema': entry['per_class'][1]['class'] = 0
    if fault == 'mixed_pool': grid['pool'] = 'attacker_validation'
    with pytest.raises(ValueError): m.build_evidence(grid, selection=selected, ledger=ledger)


def test_selected_claims_are_copied_separately_without_promoting_eligibility():
    m = module(); grid, selected, ledger = fixture()
    selected['routes'] = {'utility_first': {'nominee': 'continuous_task', 'screen_passed': True,
                                           'competitive_validation_eligible': True}}
    claims = {'family_size': 1, 'claim_results': {'utility_first': {'competitive': {'passed': False}}},
              'attribution_claim_results': {'utility_first': {'risk': {'passed': True}}},
              'provenance': {'selection_sha256': '1'*64, 'contrasts_sha256': '2'*64}}
    bounds = {'family_size': 1, 'bounds': {'endpoint': {'estimate': -.1, 'lower': -.2, 'upper': .1}},
              'provenance': claims['provenance']}
    result = m.build_evidence(grid, selection=selected, ledger=ledger, claims=claims, bounds=bounds)
    assert result['selected_inference']['main']['utility_first']['competitive']['passed'] is False
    assert result['selected_inference']['attribution']['utility_first']['risk']['passed'] is True
    claims['provenance']['selection_sha256'] = '0'*64
    with pytest.raises(ValueError, match='provenance'):m.build_evidence(grid, selection=selected, ledger=ledger, claims=claims,bounds=bounds)


def test_machine_tables_are_written_without_person_data_or_claim_invention(tmp_path):
    m = module(); grid, selected, ledger = fixture()
    evidence = m.build_evidence(grid, selection=selected, ledger=ledger)
    m.write_evidence(evidence, grid, out_dir=tmp_path)
    for name in ('EVIDENCE.json','EVALUATION_GRID.json','UTILITY_DECOMPOSITION.md','BASELINES.md','DEVELOPMENT_RESULTS.md'):
        assert (tmp_path/name).is_file()
    assert json.loads((tmp_path/'EVALUATION_GRID.json').read_text())['records'] == grid['records']
    text = (tmp_path/'DEVELOPMENT_RESULTS.md').read_text()
    assert 'descriptive' in text.lower() and 'AB/RAC1P' in text and 'PWGTP' in text
    assert 'fixed decoder unavailable' in (tmp_path/'UTILITY_DECOMPOSITION.md').read_text()
    with pytest.raises(FileExistsError):m.write_evidence(evidence, grid, out_dir=tmp_path)


def test_collector_verifies_frozen_receipts_and_summary_before_aggregate_read(tmp_path,monkeypatch):
    m = module(); grid, selected, ledger = fixture()
    selected.update(configuration_digest=config.digest(config.configuration()),config_hash=config.digest(config.configuration()),
                    source_hashes=run.source_fingerprint(),inference_source_hashes=reporting.inference_source_fingerprint(),frozen_audits={})
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    for audit in grid['audit_receipts']:
        a,n = audit['anchor'],audit['configuration']; base=tmp_path/'private/run'/f'anchor_{a}'/'audits'/n
        record={**run._provenance('audit'),**audit}
        write(base/'COMPLETE.json',record)
        selected['frozen_audits'].setdefault(n,{})[str(a)]={'registry_sha256':audit['registry_sha256'],
                                                       'receipt_sha256':run.sha(base/'COMPLETE.json')}
    write(tmp_path/'SELECTION.json',selected); selection_sha=run.sha(tmp_path/'SELECTION.json')
    for item in grid['receipts']:
        a,n=item['anchor'],item['configuration'];base=tmp_path/'private/run'/f'anchor_{a}'/'evaluation'/n
        write(base/'summary.json',grid['records'][n][str(a)])
        marker={**run._provenance('evaluation'),'anchor':a,'configuration':n,'selection_sha256':selection_sha,
                'registry_sha256':selected['frozen_audits'][n][str(a)]['registry_sha256'],
                'artifact_hashes':{'summary.json':run.sha(base/'summary.json')}}
        write(base/'COMPLETE.json',marker)
    result=m.collect_evaluation_grid(out_root=tmp_path,ledger=ledger)
    assert result['records']==grid['records'] and result['phase']=='evaluation'
    bad=tmp_path/'private/run/anchor_0/evaluation/H/summary.json';bad.write_text('{}')
    with pytest.raises(ValueError,match='summary'):m.collect_evaluation_grid(out_root=tmp_path,ledger=ledger)


@pytest.mark.parametrize('location', ['grid','receipt','audit_receipt','map_receipt'])
def test_public_grid_cannot_carry_extra_private_arrays_or_paths(location):
    m=module();grid,selected,ledger=fixture()
    if location=='grid':grid['person_ids']=['private']
    elif location=='receipt':grid['receipts'][0]['private_path']='/private/persons.npz'
    elif location=='audit_receipt':grid['audit_receipts'][0]['person_ids']=['private']
    else:grid['map_receipts']=[{'configuration':'H','anchor':0,'solution_sha256':'a'*64,
                               'receipt_sha256':'b'*64,'private_rows':[1,2]}]
    with pytest.raises(ValueError,match='public|unexpected|Unexpected'):
        m.build_evidence(grid,selection=selected,ledger=ledger)


def test_native_grid_is_accepted_by_figure_builder_without_metric_translation():
    from experiments.pcrl_task_directed_release_v1 import figures
    grid,selected,ledger=fixture();m=module()
    evidence=m.build_evidence(grid,selection=selected,ledger=ledger)
    points=figures.build_point_table(grid)
    assert len(points['tradeoff_points'])==len(evidence['configurations'])*8
    point=next(p for p in points['tradeoff_points'] if p['configuration']=='continuous_task'
               and p['role']=='AB/RAC1P' and p['weighting']=='PWGTP')
    assert point['sensitive_recovery']==pytest.approx(evidence['configurations']['continuous_task']['primary_sensitive']['AB/RAC1P']['PWGTP']['recovery_over_H'])


@pytest.mark.parametrize('mutation',['contrast_hash','family_size','config_hash','bounds_private_field'])
def test_inference_must_match_frozen_contrast_family_and_public_schema(mutation):
    m=module();grid,selected,ledger=fixture()
    provenance={'selection_sha256':'1'*64,'contrasts_sha256':'2'*64,'config_hash':'3'*64}
    selected['config_hash']='3'*64
    claims={'family_size':1,'claim_results':{},'attribution_claim_results':{},'provenance':provenance}
    bounds={'family_size':1,'bounds':{'e':{'estimate':0.,'lower':0.,'upper':0.}},'provenance':provenance}
    if mutation=='contrast_hash':selected['contrasts_sha256']='9'*64
    if mutation=='family_size':selected['contrast_family_size']=2
    if mutation=='config_hash':selected['config_hash']='8'*64
    if mutation=='bounds_private_field':bounds['bounds']['e']['ids']=['private-person']
    with pytest.raises(ValueError):m.build_evidence(grid,selection=selected,ledger=ledger,claims=claims,bounds=bounds)


def test_generate_checks_contrast_bytes_endpoint_ids_and_recomputes_only_claim_logic(tmp_path,monkeypatch):
    m=module();grid,selected,ledger=fixture()
    contrast={'family_size':1,'endpoints':[{'id':'e','checks':[{'name':'strict','bound':'upper','operator':'<','threshold':0.}]}],
              'claim_formulas':{'utility_first':{'competitive':{'endpoint':'e','check':'strict'}}},
              'attribution_claim_formulas':{}}
    (tmp_path/'CONTRASTS.json').write_text(json.dumps(contrast))
    selected['contrasts_sha256']=run.sha(tmp_path/'CONTRASTS.json')
    (tmp_path/'SELECTION.json').write_text(json.dumps(selected));grid['selection_sha256']=run.sha(tmp_path/'SELECTION.json')
    provenance={'selection_sha256':grid['selection_sha256'],'contrasts_sha256':selected['contrasts_sha256']}
    bounds={'family_size':1,'bounds':{'e':{'estimate':-.1,'lower':-.2,'upper':.01}},'provenance':provenance}
    claims={**reporting.evaluate_claims(contrast,bounds),'provenance':provenance}
    (tmp_path/'PAIRED_BOUNDS.json').write_text(json.dumps(bounds))
    (tmp_path/'CLAIM_RESULTS.json').write_text(json.dumps(claims))
    monkeypatch.setattr(m,'collect_evaluation_grid',lambda **kwargs:copy.deepcopy(grid))
    written=m.generate(out_root=tmp_path,out_dir=tmp_path/'public',ledger=ledger)
    assert set(written)=={'EVIDENCE.json','EVALUATION_GRID.json','UTILITY_DECOMPOSITION.md','BASELINES.md','DEVELOPMENT_RESULTS.md'}
    claims['claim_results']['utility_first']['competitive']['passed']=True
    (tmp_path/'CLAIM_RESULTS.json').write_text(json.dumps(claims))
    with pytest.raises(ValueError,match='formulas'):m.generate(out_root=tmp_path,out_dir=tmp_path/'tampered',ledger=ledger)
    (tmp_path/'CLAIM_RESULTS.json').write_text(json.dumps({**reporting.evaluate_claims(contrast,bounds),'provenance':provenance}))
    bounds['bounds']['wrong']=bounds['bounds'].pop('e')
    (tmp_path/'PAIRED_BOUNDS.json').write_text(json.dumps(bounds))
    with pytest.raises(ValueError,match='endpoint'):m.generate(out_root=tmp_path,out_dir=tmp_path/'wrong',ledger=ledger)


def test_amendment5_diagnostics_and_missing_vs_ineligible_families_stay_distinct():
    m=module();grid,selected,ledger=fixture()
    selected['routes']={'utility_first':{'nominee':'continuous_task','screen_passed':True,
        'competitive_validation_eligible':True,'empty_required_families':['supervised_SPLINCE'],
        'incomplete_required_families':[],'scientifically_ineligible_families':['supervised_SPLINCE'],
        'eligible_comparator_families':['continuous_task'],'strict_all_families_validation_eligible':False,
        'family_nominees':{'supervised_SPLINCE':{'configuration':None,'required_frontier_complete':True,
            'missing_required_configurations':[],'scientifically_ineligible':True,'eligible':[]}}}}
    claims={'family_size':1,'claim_results':{'utility_first':{'competitive':{'passed':True}}},
        'attribution_claim_results':{},'diagnostic_claim_results':{'utility_first':{'competitive_all_families':{'passed':False}}},
        'provenance':{'selection_sha256':'1'*64,'contrasts_sha256':'2'*64}}
    bounds={'family_size':1,'bounds':{'e':{'estimate':0.,'lower':0.,'upper':0.}},'provenance':claims['provenance']}
    result=m.build_evidence(grid,selection=selected,ledger=ledger,claims=claims,bounds=bounds)
    assert result['selected_inference']['main']['utility_first']['competitive']['passed']
    assert not result['selected_inference']['diagnostic']['utility_first']['competitive_all_families']['passed']
    assert result['selected_routes']['utility_first']['incomplete_required_families']==[]
    assert result['family_eligibility']['utility_first']['supervised_SPLINCE']['scientifically_ineligible']
    assert 'does not veto' in m.render_tables(result)['BASELINES.md']


def test_native_selection_blocked_family_metadata_is_public_and_preserved():
    from experiments.pcrl_task_directed_release_v1 import selection
    from test_selection import base, A
    chosen = selection.select_validation(base(), candidate_ids=[A])
    contrasts = selection.build_contrast_family(chosen)
    checks = {(endpoint['id'], check['name']): False
        for endpoint in contrasts['endpoints'] for check in endpoint['checks']}
    observed = set()
    for group in ('claim_formulas', 'attribution_claim_formulas', 'diagnostic_claim_formulas'):
        for route in contrasts[group].values():
            for raw in route.values():
                evaluated = reporting.evaluate_formula(raw, checks)
                before = copy.deepcopy(evaluated)
                module()._public_formula(evaluated)
                assert evaluated == before
                if evaluated.get('kind') == 'blocked':
                    observed.update(set(evaluated) & {'incomplete_required_families',
                        'eligible_comparator_families', 'empty_required_families'})
    assert observed == {'incomplete_required_families', 'eligible_comparator_families', 'empty_required_families'}


@pytest.mark.parametrize('field', ['incomplete_required_families', 'eligible_comparator_families', 'empty_required_families'])
@pytest.mark.parametrize('bad_value', ['family', {'family': True}, [1], [['family']], [None]])
def test_blocked_family_metadata_requires_public_string_lists(field, bad_value):
    with pytest.raises(ValueError):
        module()._public_formula({'kind': 'blocked', 'passed': False, field: bad_value})


def test_family_metadata_is_blocked_only_and_private_fields_still_rejected():
    m = module()
    for node in ({'endpoint': 'e', 'check': 'cap', 'passed': False},
                 {'kind': 'any', 'clauses': [{'passed': False}], 'passed': False}):
        with pytest.raises(ValueError):
            m._public_formula({**node, 'empty_required_families': ['public_family']})
    with pytest.raises(ValueError):
        m._public_formula({'kind': 'blocked', 'passed': False,
            'incomplete_required_families': ['supervised_LEACE'], 'ids': ['private-person']})

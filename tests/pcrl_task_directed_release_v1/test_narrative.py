"""Final prose is generated from explicit synthetic public aggregates only."""
import copy
import importlib
import json

import pytest


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.narrative')


def evidence(passed=False, inference=True):
    score={'unweighted':.5,'PWGTP':.4}
    task={k:copy.deepcopy(score) for k in ('fixed','independent','selected')}
    task.update(loss_delta_H={'unweighted':-.03,'PWGTP':-.02},
                loss_delta_J={'unweighted':-.01,'PWGTP':-.005})
    formula={'kind':'all','clauses':[{'endpoint':'e','check':'cap','passed':passed}],'passed':passed}
    return {'schema':1,'phase':'evaluation','pool':'test','selection_frozen':True,
        'selection_sha256':'1'*64,'grid_digest':'2'*64,
        'selected_routes':{r:{'nominee':'Trisk_C_0.002_a17','screen_passed':True,
            'descriptive_only':False,'competitive_validation_eligible':True,
            'incomplete_required_families':[],'eligible_comparator_families':['continuous_task']}
            for r in ('utility_first','protection_first')},
        'configurations':{'Trisk_C_0.002_a17':{'status':'complete','task':task,
            'primary_sensitive':{'A/SEX':{w:{'selected_ce':.6,'recovery_over_H':.01,
                'recovery_increment_over_J':-.001} for w in score}}}},
        'execution':{'nominal_map_units':81,'accepted_map_units':81,'new_role_fit_units':123},
        'selected_inference':({'status':'supplied_adjusted_results','family_size':1,
            'provenance':{'selection_sha256':'1'*64},'bounds':{'e':{'estimate':0.,'lower':-.01,'upper':.01}},
            'main':{r:{'competitive':copy.deepcopy(formula),'historical_J':copy.deepcopy(formula)}
                for r in ('utility_first','protection_first')},'attribution':{},'diagnostic':{}}
            if inference else {'status':'not_supplied','main':{},'attribution':{},'diagnostic':{}})}


def render(e=None,**kwargs):
    provenance=kwargs.pop('provenance',{} if e is None else {'EVIDENCE.json':'a'*64})
    return module().render_reports(evidence=e,provenance=provenance,**kwargs)


def test_pending_template_retains_scope_without_inventing_method_success():
    output=render()
    assert set(output['documents'])=={'RESEARCH_DECISION.md','PAPER_ADDENDUM.md','VALIDATION.md'}
    assert output['manifest']['competitive_pass_established'] is False
    all_text='\n'.join(output['documents'].values())
    for text in ('pending','supervised residence','globally unseen','actual venue reviews','HISTORICAL_SCORE_INDEX.json'):
        assert text in all_text
    assert 'The registered adjusted competitive criterion passed' not in all_text


def test_favorable_point_metrics_cannot_establish_method_pass_before_inference():
    output=render(evidence(True,False))
    assert not output['manifest']['competitive_pass_established']
    assert output['manifest']['readiness']=='inference_pending'
    assert '-0.01' in output['documents']['RESEARCH_DECISION.md']


def test_passed_adjusted_routes_are_reported_with_scope_and_exact_source_pointers():
    output=render(evidence(True))
    assert output['manifest']['competitive_pass_established']
    assert output['manifest']['competitive_pass_routes']==['utility_first','protection_first']
    assert '/selected_inference/main/utility_first/competitive' in output['documents']['RESEARCH_DECISION.md']
    assert 'conditional development' in output['documents']['PAPER_ADDENDUM.md']
    assert 'verification pending' in output['documents']['VALIDATION.md']


def test_failed_criterion_is_not_rendered_as_population_or_information_impossibility():
    output=render(evidence(False))
    assert not output['manifest']['competitive_pass_established']
    assert 'No adjusted competitive pass is established' in output['documents']['RESEARCH_DECISION.md']
    assert 'not an impossibility result' in output['documents']['RESEARCH_DECISION.md']


@pytest.mark.parametrize('fault',['missing_hash','wrong_selection','incomplete_family','false_screen','bad_formula','missing_endpoint'])
def test_claim_or_provenance_inconsistencies_fail_closed(fault):
    e=evidence(True);pins={'EVIDENCE.json':'a'*64}
    if fault=='missing_hash':pins={}
    elif fault=='wrong_selection':e['selected_inference']['provenance']['selection_sha256']='3'*64
    elif fault=='incomplete_family':e['selected_routes']['utility_first']['incomplete_required_families']=['supervised_LEACE']
    elif fault=='false_screen':e['selected_routes']['utility_first']['screen_passed']=False
    elif fault=='bad_formula':e['selected_inference']['main']['utility_first']['competitive']['passed']=False
    else:e['selected_inference']['bounds']={}
    with pytest.raises(ValueError):render(e,provenance=pins)


def test_private_fields_are_not_copied_into_public_prose_or_manifest():
    e=evidence();e['private_labels']=[7,8,9];e['configurations']['Trisk_C_0.002_a17']['person_ids']=['SECRET_PERSON']
    output=render(e)
    assert 'SECRET_PERSON' not in json.dumps(output)
    assert 'private_labels' not in json.dumps(output)


def test_exact_equivalence_counts_stay_distinct_from_nominal_and_artifact_counts():
    eq={'counts':{'nominal_map_units':81,'accepted_verified_map_units':81,
         'unique_literal_kernel_groups':60,'unique_reduced_kernel_groups':50},'independent_evidence_claimed':False}
    output=render(evidence(),equivalence=eq,provenance={'EVIDENCE.json':'a'*64,'EXACT_EQUIVALENCE.json':'b'*64})
    text=output['documents']['VALIDATION.md']
    assert 'unique_reduced_kernel_groups' in text and '50' in text
    assert 'independent observations' in text


def test_prediction_bets_remain_separate_and_unassessed_stays_unassessed():
    p={'scientific_gate':False,'inference_performed':False,'predictions':{
        f'P{i}':{'probability':.5,'status':'unassessed','value':None,'coverage':{'complete':False}}
        for i in range(1,8)}}
    output=render(prediction_assessment=p,provenance={'PREDICTION_ASSESSMENT.json':'a'*64})
    assert 'unassessed' in output['documents']['RESEARCH_DECISION.md']
    assert not output['manifest']['competitive_pass_established']


def test_writer_refuses_to_overwrite_existing_report(tmp_path):
    report=render();module().write_reports(report,out_dir=tmp_path)
    before=(tmp_path/'RESEARCH_DECISION.md').read_bytes()
    with pytest.raises(FileExistsError):module().write_reports(report,out_dir=tmp_path)
    assert (tmp_path/'RESEARCH_DECISION.md').read_bytes()==before


def test_native_evidence_schema_interoperability_without_file_discovery():
    from tests.pcrl_task_directed_release_v1.test_evidence import fixture
    from experiments.pcrl_task_directed_release_v1.evidence import build_evidence
    grid,selected,ledger=fixture()
    e=build_evidence(grid,selection=selected,ledger=ledger)
    output=render(e)
    manifest=output['manifest']
    assert manifest['complete_configurations']==4
    assert manifest['numerical_versions']['registered'] is False
    assert manifest['execution_counts']['nominal_release_anchor_units']==12
    assert next(r for r in manifest['task_points'] if r['configuration']=='T0_code')['fixed'] is None
    assert len(manifest['primary_sensitive_points'])==4*4*2


def test_missing_h_or_j_is_unavailable_instead_of_inventing_recovery():
    e=evidence(False,False)
    row=e['configurations']['Trisk_C_0.002_a17']
    row['task']['loss_delta_H']=None
    row['primary_sensitive']['A/SEX']['unweighted']['recovery_over_H']=None
    output=render(e)
    assert output['manifest']['primary_sensitive_points'][0]['recovery_over_H'] is None
    assert 'unavailable' in output['documents']['RESEARCH_DECISION.md']


def test_native_blocked_families_are_reported_as_blocked_not_an_evaluated_failure():
    e=evidence()
    e['selected_inference']['main']['utility_first']['competitive']={
        'kind':'blocked','passed':False,'reason':'validation screen did not pass; descriptive nominee only',
        'incomplete_required_families':[],'eligible_comparator_families':[]}
    output=render(e)
    claim=next(c for c in output['manifest']['claims'] if c['route']=='utility_first' and c['claim']=='competitive')
    assert claim['status']=='blocked'
    assert not output['manifest']['competitive_pass_established']


def versions():
    m=module();records=[]
    for anchor,name in ((0,'Ttask_C_0.0005_a17'),(0,'Ttask_C_0.01_a17'),(0,'Ttask_L_0.002_a17'),(2,'Trisk_C_0.01_a17')):
        row={'configuration':name,'anchor':anchor,'attempts':1,'retry_completed':True,'retry_accepted':True,
             'installation_decision':'installed_accepted_retry','reaudit_completed':True}
        row.update({k:'a'*64 for k in m.VERSION_HASHES})
        row.update({prefix+k:v for prefix in ('original_','current_')
                    for k,v in (('audit_role_units',16),('new_role_fits',9),('reused_role_audits',7))})
        records.append(row)
    return {'schema':1,'registered':True,'registration_sha256':'b'*64,'registered_slots':4,
            'new_nominal_configurations':0,'coverage_complete':True,'records':records,
            'scope':'Receipt-only provenance/byte checks; no score, model or prediction replay.'}


def test_embedded_four_versions_preserve_receipts_and_do_not_add_configurations():
    e=evidence();e['numerical_versions']=versions()
    output=render(e);v=output['manifest']['numerical_versions']
    assert len(v['records'])==4 and v['new_nominal_configurations']==0
    assert v['records'][0]['original_registry_sha256']=='a'*64
    assert v['records'][0]['current_registry_sha256']=='a'*64
    text=output['documents']['VALIDATION.md']
    assert text.count('installed_accepted_retry')==4


@pytest.mark.parametrize('fault',['missing_reaudit','bad_roles','contradictory_decision','unmatched_standalone'])
def test_invalid_numerical_versions_fail_closed(fault):
    e=evidence();e['numerical_versions']=versions();kwargs={}
    if fault=='missing_reaudit':e['numerical_versions']['records'][0]['reaudit_completed']=False
    elif fault=='bad_roles':e['numerical_versions']['records'][0]['current_audit_role_units']=15
    elif fault=='contradictory_decision':e['numerical_versions']['records'][0]['installation_decision']='retained_original'
    else:
        kwargs={'numerical_versions':versions(),'provenance':{'EVIDENCE.json':'a'*64,'NUMERICAL_VERSION_INDEX.json':'b'*64}}
        kwargs['numerical_versions']['records'][0]['current_solution_sha256']='b'*64
    with pytest.raises(ValueError):render(e,**kwargs)


def test_rejected_retry_is_distinct_from_a_completed_reaudit():
    e=evidence();v=e['numerical_versions']=versions()
    v['records'][0].update(retry_accepted=False,installation_decision='retained_original',reaudit_completed=False)
    output=render(e)
    assert output['manifest']['numerical_versions']['records'][0]['reaudit_completed'] is False
    assert 'retained_original' in output['documents']['VALIDATION.md']


def verified():
    return {'passed':True,'all_passed':True,'source_selection_closure_unchanged':True,
            'planned_units':6,'passed_units':6,'failed_or_incomplete_units':0,
            'global_selection':{'selection_sha256':'1'*64}}


def test_complete_pinned_verification_is_reported_only_for_its_coverage():
    out=render(evidence(),verification=verified(),provenance={'EVIDENCE.json':'a'*64,'PARALLEL_VERIFICATION.json':'b'*64})
    assert out['manifest']['verification']['passed'] is True
    text=out['documents']['VALIDATION.md']
    assert 'not every unselected fitted artifact' in text
    assert 'restore evidence must be reported separately' in text


@pytest.mark.parametrize('fault',['missing_units','changed_closure','wrong_selection','missing_selection'])
def test_incomplete_verification_cannot_claim_a_pass(fault):
    v=verified()
    if fault=='missing_units':v['passed_units']=5
    elif fault=='changed_closure':v['source_selection_closure_unchanged']=False
    elif fault=='wrong_selection':v['global_selection']['selection_sha256']='2'*64
    else:v['global_selection']={}
    with pytest.raises(ValueError):render(evidence(),verification=v,provenance={'EVIDENCE.json':'a'*64,'PARALLEL_VERIFICATION.json':'b'*64})


def test_native_empty_prediction_assessment_is_preserved_without_assessing_outcomes():
    from experiments.pcrl_task_directed_release_v1.predictions import assess_predictions
    assessment=assess_predictions({})
    output=render(prediction_assessment=assessment,provenance={'PREDICTION_ASSESSMENT.json':'a'*64})
    assert len(output['manifest']['prediction_assessments'])==7
    assert all(r['status']=='unassessed' for r in output['manifest']['prediction_assessments'])


def test_incompatible_independent_holdout_scope_cannot_be_silently_rewritten():
    e=evidence();e['cross_anchor_scope']={'globally_unseen_people':True,'globally_untouched_labels':False,
        'retraining_uncertainty_included':False,'bounds_conditional_on_fitted_models':True}
    with pytest.raises(ValueError):render(e)


def test_omitted_adversarial_baseline_and_historical_kernel_slate_are_explicit():
    text='\n'.join(render()['documents'].values())
    assert 'optional neural-adversarial baseline was not executed' in text
    assert 'historical expanded/kernel-expanded audit scores' in text

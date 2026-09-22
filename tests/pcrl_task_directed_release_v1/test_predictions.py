"""Prospective bet assessment using synthetic native aggregate schemas only."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pcrl_task_directed_release_v1 import config


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.predictions')


def score(x):return {'unweighted':x,'weighted':x+.02,'balanced':x+.01}


def fixture():
    values={'H':(.70,.20),'J':(.68,None),'continuous_task':(.90,.65),
            'T0_U_unconstrained_a17':(.80,.6505)}
    for policy in ('L','C'):
        for budget in config.POSITIVE_BUDGETS:values[config.mechanism_id('T0',policy,budget)]=(.66,None)
        values[config.mechanism_id('T0',policy,.002)]=(.64,None)
        values[config.mechanism_id('Ttask',policy,.002)]=(.635,None)
        values[config.mechanism_id('Trisk',policy,.002)]=(.63 if policy=='L' else .6305,None)
    records={}
    for name,(task,fixed) in values.items():
        records[name]={}
        for anchor in (0,1,2):
            entry={'validation':score(task+.001*anchor),'independent_validation':score(task+.03+.001*anchor),
                   'fixed_decoder':None if fixed is None else score(fixed+.001*anchor)}
            native={'utility:A/same_residence':entry}
            sensitive=.6 if name=='H' else (.5895 if name=='Trisk_L_0.002_a17' else .592 if name=='Trisk_C_0.002_a17' else .59)
            native.update({'attack:'+r:{'validation':score(sensitive+.001*anchor)} for r in config.PRIMARY})
            records[name][str(anchor)]=native
    return records


def selected(*,complete=True):
    records=[{'configuration':r['configuration'],'anchor':r['anchor'],'complete':True,'scope':'primary'}
             for r in config.release_ledger()['records']]
    if not complete:records[-1]['complete']=False
    return {'selection_frozen':True,'completion_at_freeze':{'records':records,'missing':[] if complete else [records[-1]]},
            'routes':{r:{'nominee':'Trisk_C_0.002_a17','screen_passed':True,'incomplete_required_families':[],
                         'eligible_comparator_families':['continuous_task'],'competitive_validation_eligible':True}
                      for r in ('utility_first','protection_first')},'registered_controls':{}}


def claims(passed=False):
    formula={'kind':'all','clauses':[{'endpoint':'e','check':'cap','passed':passed}],'passed':passed}
    return {'claim_results':{r:{'competitive':copy.deepcopy(formula)} for r in ('utility_first','protection_first')},
            'checks':{'e':{'cap':passed}},'family_size':1,
            'provenance':{'selection_sha256':'1'*64,'contrasts_sha256':'2'*64}}


def certificates(nonconstant=False,all_units=True):
    from experiments.pcrl_task_directed_release_v1.zero_certificate import certify
    if nonconstant:
        law=np.array([[[9,1,1]],[[1,9,9]]]);mass=[10,10,10]
    else:
        law=np.eye(3,dtype=int)[:,None,:];mass=[1,1,1]
    proof,_=certify({'role':law},mass,[0,1,2])
    specs=[s for s in config.configuration()['maps'] if s['budget']==0]
    if not all_units:specs=specs[:1]
    rows=[]
    for spec in specs:
        row={**proof,'anchor':spec['anchor'],'configuration':spec['configuration'],'map_receipt_sha256':'a'*64}
        if nonconstant:row['private_witness_sha256']='b'*64
        rows.append(row)
    return {'certificates':rows,'unavailable':[],'P7_supported':not nonconstant}


def test_registered_probabilities_and_P1_to_P5_use_correct_decoder_routes():
    result=module().assess_predictions(fixture())
    assert {k:v['probability'] for k,v in result['predictions'].items()}=={'P1':.85,'P2':.60,'P3':.60,'P4':.40,'P5':.65,'P6':.25,'P7':.70}
    assert all(result['predictions'][p]['status']=='supported' for p in ('P1','P2','P3','P4','P5'))
    assert result['predictions']['P1']['evidence'][0]['gain']==pytest.approx(.05)
    assert result['predictions']['P2']['evidence'][0]['loss']==pytest.approx(.0005)
    assert result['predictions']['P5']['evidence'][0]['task_penalty']==pytest.approx(.0005)
    assert result['predictions']['P5']['evidence'][0]['coalition_max_recovery']<result['predictions']['P5']['evidence'][0]['local_max_recovery']
    assert not result['scientific_gate'] and not result['inference_performed']


def test_prospective_rules_artifact_matches_assessor_contract():
    path=Path(__file__).resolve().parents[2]/'results/pcrl_task_directed_release_v1/PREDICTION_ASSESSMENT_RULES.json'
    assert json.loads(path.read_text())==module().assessment_rules()


def test_every_mean_requires_all_three_anchors_and_never_uses_partial_mean():
    records=fixture();records['continuous_task'].pop('2')
    result=module().assess_predictions(records)
    assert result['predictions']['P1']['status']==result['predictions']['P2']['status']=='unassessed'
    assert result['predictions']['P1']['coverage']['missing']
    assert result['predictions']['P3']['status']=='supported'


def test_existential_true_retains_partial_coverage_but_false_needs_complete_grid():
    records=fixture();missing='T0_C_0.01_a17';records.pop(missing)
    result=module().assess_predictions(records)['predictions']['P3']
    assert result['status']=='supported' and not result['coverage']['complete']
    for name,anchors in records.items():
        if name.startswith('T0_') and '_U_' not in name:
            for native in anchors.values():native['utility:A/same_residence']['validation']=score(.8)
    result=module().assess_predictions(records)['predictions']['P3']
    assert result['status']=='unassessed'
    records[missing]=copy.deepcopy(records['T0_L_0.01_a17'])
    assert module().assess_predictions(records)['predictions']['P3']['status']=='refuted'


def test_P4_caps_both_weightings_against_each_comparator_and_uses_same_policy():
    records=fixture()
    for policy in ('L','C'):
        for entry in records[f'Trisk_{policy}_0.002_a17'].values():
            scores=entry['attack:AB/RAC1P']['validation'];scores['weighted']=.50
            scores['balanced']=.5*(scores['weighted']+scores['unweighted'])
    assert module().assess_predictions(records)['predictions']['P4']['status']=='refuted'
    records=fixture()
    for entry in records['Trisk_L_0.002_a17'].values():entry['utility:A/same_residence']['validation']=score(.8)
    records.pop('Ttask_C_0.002_a17')
    assert module().assess_predictions(records)['predictions']['P4']['status']=='unassessed'


def test_P5_uses_max_of_balanced_AB_recoveries_not_max_over_weightings_or_A():
    records=fixture()
    for entry in records['Trisk_C_0.002_a17'].values():
        entry['attack:A/SEX']['validation']=score(.1)
        entry['attack:AB/SEX']['validation']={'unweighted':.50,'weighted':.71,'balanced':.605}
        entry['attack:AB/RAC1P']['validation']=score(.593)
    result=module().assess_predictions(records)['predictions']['P5']
    assert result['status']=='supported'
    assert result['evidence'][0]['coalition_max_recovery']==pytest.approx(.008)


def test_strict_P4_task_and_P5_recovery_ties_do_not_pass():
    records=fixture()
    for policy in ('L','C'):
        for anchor,entry in records[f'Trisk_{policy}_0.002_a17'].items():
            entry['utility:A/same_residence']['validation']=copy.deepcopy(
                records[f'Ttask_{policy}_0.002_a17'][anchor]['utility:A/same_residence']['validation'])
    assert module().assess_predictions(records)['predictions']['P4']['status']=='refuted'
    records=fixture()
    for anchor,entry in records['Trisk_C_0.002_a17'].items():
        for role in ('AB/SEX','AB/RAC1P'):
            entry['attack:'+role]=copy.deepcopy(records['Trisk_L_0.002_a17'][anchor]['attack:'+role])
    assert module().assess_predictions(records)['predictions']['P5']['status']=='refuted'


def test_P6_true_adjusted_witness_can_be_supported_with_partial_programme():
    result=module().assess_predictions(fixture(),claim_results=claims(True),selection=selected(complete=False))['predictions']['P6']
    assert result['status']=='supported' and not result['coverage']['complete']


def test_P6_failed_adjusted_tests_require_complete_programme_to_refute():
    m=module()
    assert m.assess_predictions(fixture(),claim_results=claims(),selection=selected())['predictions']['P6']['status']=='refuted'
    assert m.assess_predictions(fixture(),claim_results=claims(),selection=selected(complete=False))['predictions']['P6']['status']=='unassessed'
    assert m.assess_predictions(fixture(),claim_results=claims())['predictions']['P6']['status']=='unassessed'


def test_P6_complete_validation_screen_failures_or_empty_union_are_refutations():
    m=module();s=selected()
    for r in s['routes'].values():r['screen_passed']=False;r['competitive_validation_eligible']=False
    assert m.assess_predictions(fixture(),selection=s)['predictions']['P6']['status']=='refuted'
    s=selected();c=claims()
    for route,r in s['routes'].items():
        r['competitive_validation_eligible']=False;r['eligible_comparator_families']=[]
        c['claim_results'][route]['competitive']={'kind':'blocked','passed':False,
            'reason':'mandatory comparison attempts incomplete or total feasible comparator set empty',
            'incomplete_required_families':[],'eligible_comparator_families':[]}
    assert m.assess_predictions(fixture(),claim_results=c,selection=s)['predictions']['P6']['status']=='refuted'
    c['claim_results']['utility_first']['competitive']['reason']='unknown future block'
    assert m.assess_predictions(fixture(),claim_results=c,selection=s)['predictions']['P6']['status']=='unassessed'


def test_P6_missing_required_family_and_inconsistent_claim_booleans_are_not_success():
    m=module();s=selected();s['routes']['utility_first']['incomplete_required_families']=['supervised_SPLINCE']
    assert m.assess_predictions(fixture(),claim_results=claims(),selection=s)['predictions']['P6']['status']=='unassessed'
    c=claims(True);c['claim_results']['utility_first']['competitive']['passed']=False
    with pytest.raises(ValueError,match='formula'):m.assess_predictions(fixture(),claim_results=c,selection=selected())


def test_P6_missing_required_supplement_cannot_refute_bet():
    s=selected()
    s['registered_controls']={f'{method}_{scope}':{'required':True}
        for method in ('leace_supervised','splince_supervised') for scope in ('mechanism40','union88')}
    result=module().assess_predictions(fixture(),claim_results=claims(),selection=s)['predictions']['P6']
    assert result['status']=='unassessed' and len(result['coverage']['missing'])==12
    s['completion_at_freeze']['records'] += [{'configuration':name,'anchor':anchor,'complete':True}
        for name in s['registered_controls'] for anchor in (0,1,2)]
    assert module().assess_predictions(fixture(),claim_results=claims(),selection=s)['predictions']['P6']['status']=='refuted'


@pytest.mark.parametrize('field,value',[('config_hash','a'*64),('source_hashes',{'finite.py':'b'*64}),
                                      ('inference_source_hashes',{'selection.py':'c'*64})])
def test_P6_claim_source_and_configuration_pins_must_match_selection(field,value):
    s=selected();s[field]=value;c=claims()
    with pytest.raises(ValueError,match='provenance'):
        module().assess_predictions(fixture(),claim_results=c,selection=s)
    c['provenance'][field]=value
    assert module().assess_predictions(fixture(),claim_results=c,selection=s)['predictions']['P6']['status']=='refuted'


def test_P7_requires_exact_observed_support_certificates_and_ignores_global_flag():
    m=module()
    assert m.assess_predictions({},zero_certificates=certificates(True,False))['predictions']['P7']['status']=='supported'
    assert m.assess_predictions({},zero_certificates=certificates(False,False))['predictions']['P7']['status']=='unassessed'
    assert m.assess_predictions({},zero_certificates=certificates(False,True))['predictions']['P7']['status']=='refuted'
    c=certificates(True,False);c['certificates'][0]['rational_equations_verified']=False
    with pytest.raises(ValueError,match='certificate'):m.assess_predictions({},zero_certificates=c)


@pytest.mark.parametrize('fault',['nan','negative','balanced','duplicate_anchor','evaluation'])
def test_malformed_or_wrong_phase_inputs_fail_closed(fault):
    records=fixture()
    if fault=='evaluation':value={'phase':'evaluation','records':records}
    else:
        value=records
        if fault=='duplicate_anchor':records['H'][0]=records['H']['0']
        elif fault=='nan':records['H']['0']['utility:A/same_residence']['validation']['weighted']=float('nan')
        elif fault=='negative':records['H']['0']['utility:A/same_residence']['validation']['unweighted']=-.1
        else:records['H']['0']['utility:A/same_residence']['validation']['balanced']=99.
    with pytest.raises(ValueError):module().assess_predictions(value)


def test_provenance_is_preserved_and_selection_hash_mismatch_is_rejected():
    m=module();provenance={'VALIDATION_GRID.json':'a'*64,'SELECTION.json':'1'*64,'CLAIM_RESULTS.json':'b'*64}
    result=m.assess_predictions({'evaluation_opened':False,'records':fixture()},claim_results=claims(),selection=selected(),provenance=provenance)
    assert result['provenance']==provenance and result['rules_digest']==config.digest(m.assessment_rules())
    provenance['SELECTION.json']='3'*64
    with pytest.raises(ValueError,match='provenance'):m.assess_predictions(fixture(),claim_results=claims(),selection=selected(),provenance=provenance)

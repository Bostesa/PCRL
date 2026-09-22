"""Positive-control diagnostics on synthetic JSON aggregates only."""
import copy
import importlib
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import config,run


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.attack_calibration')


def native(base=.5,improvement=.01):
    offsets={'logistic':.08,'hist_gb_20':.05,'hist_gb_5':.04,
             'sampled_hist_gb_20':.05,'sampled_hist_gb_5':.04,'mlp_120':.02,'mlp_360':.02-improvement}
    scores={c:{'unweighted':base+x,'weighted':base+x+.02,'balanced':base+x+.01} for c,x in offsets.items()}
    selected=min(scores,key=lambda c:(scores[c]['balanced'],c))
    return {'selection':selected,'independent_selection':selected,'validation_scores':scores,
            'fit_rows':100,'validation_rows':20,'view':'A','target':'SEX','kind':'attack',
            'candidates':{c:{'origin':'independent','route':{'kind':'model','source_view':'A','wire':'release'},
                              'model_path':'/private/not-for-publication'} for c in scores},
            'validation_loss_path':'/private/ids-and-losses.npz'}


def trajectory(epoch):
    return {'candidate_id':f'mlp_{epoch}','seed':7,'schedule_seed':700007,'initial_state_hash':'a'*64,
            'trajectory_epochs':360,'parameters':{'epochs':epoch},'selected_epoch':min(epoch,100),
            'trajectory_continuation':'actual last weights and Adam/RNG state; no selected-state rewind'}


def payload(improvement=.01):
    m=module();rows=[]
    for index,name in enumerate(m.CONTROLS):
        for anchor in (0,1,2):
            for role in m.ROLES:
                n=native(1.-index*.1+anchor*.01,improvement)
                n['view'],n['target']=role.split(':',1)[1].split('/')
                for c in n['candidates'].values():c['route'].update(source_view=n['view'],wire='H' if name=='H' else 'release')
                rows.append(m.role_record(name,anchor,role,n,{120:trajectory(120),360:trajectory(360)},
                    receipt_sha256='b'*64,selection_sha256='c'*64,metadata_hashes={'120':'d'*64,'360':'e'*64}))
    return {'phase':'validation','evaluation_opened':False,'records':rows,'incomplete_units':[],
            'config_hash':config.digest(config.configuration()),'resource_schedule_sha256':'f'*64}


def test_reports_all_candidates_frozen_choices_trajectory_and_positive_stress_gain():
    m=module();p=payload();result=m.build_calibration(p)
    assert result['status']=='complete' and result['accepted_role_units']==48
    j=result['three_anchor_means']['J']['attack:A/SEX']
    assert j['selected_recovery_over_H']['balanced']==pytest.approx(.1)
    assert j['continued360_gain_over120']['balanced']==pytest.approx(.01)
    assert j['catchup_gain_over_standard']['balanced']==pytest.approx(.01)
    assert j['best_tree_gain_over_logistic']['balanced']==pytest.approx(.04)
    assert result['diagnostic_status']=='stress_gain_demonstrated_on_informative_controls'
    assert len(result['records'][0]['candidate_validation'])==7
    serialized=json.dumps(result)
    assert '/private/' not in serialized and 'validation_loss_path' not in serialized
    assert result['records'][0]['trajectory']['same_initialization_and_schedule'] is True


def test_no_gain_is_honest_and_partial_anchors_are_not_averaged():
    m=module();p=payload(improvement=0);result=m.build_calibration(p)
    assert result['diagnostic_status']=='no_stress_gain_demonstrated_on_informative_controls'
    assert result['nonlinear_upgrade_status']=='nonlinear_gain_demonstrated_on_informative_controls'
    p['records']=[r for r in p['records'] if not (r['configuration']=='J' and r['anchor']==2)]
    p['incomplete_units']=[{'configuration':'J','anchor':2}]
    result=m.build_calibration(p)
    assert result['status']=='incomplete' and 'J' not in result['three_anchor_means']
    assert result['diagnostic_status']=='incomplete_calibration'


@pytest.mark.parametrize('fault',['continued_seed','missing_candidate','wrong_winner','nonfinite','extra_control'])
def test_invalid_calibration_sources_are_rejected(fault):
    m=module();n=native();meta={120:trajectory(120),360:trajectory(360)};name='J'
    if fault=='continued_seed':meta[360]['seed']=99
    if fault=='missing_candidate':n['candidates'].pop('mlp_360');n['validation_scores'].pop('mlp_360')
    if fault=='wrong_winner':n['selection']='logistic'
    if fault=='nonfinite':n['validation_scores']['logistic']['balanced']=float('nan')
    if fault=='extra_control':name='posthoc_Q'
    with pytest.raises(ValueError):m.role_record(name,0,'attack:A/SEX',n,meta,
        receipt_sha256='b'*64,selection_sha256='c'*64,metadata_hashes={'120':'d'*64,'360':'e'*64})


def test_collect_reads_only_receipt_hashed_json_and_never_unaccepted_roles(tmp_path):
    m=module();cfg=config.digest(config.configuration())
    (tmp_path/'RESOURCE_SCHEDULE.json').write_text(json.dumps({'frozen_before_comparative_outcomes':True,'config_hash':cfg}))
    base=tmp_path/'private/run/anchor_0/audits/J';base.mkdir(parents=True)
    files={}
    for role in m.ROLES:
        n=native();n['view'],n['target']=role.split(':',1)[1].split('/')
        for c in n['candidates'].values():c['route']['source_view']=n['view']
        safe=role.replace(':','__').replace('/','__')
        values={f'{safe}/selection.json':n,**{f'{safe}/models/mlp_{e}/metadata.json':trajectory(e) for e in (120,360)}}
        for rel,value in values.items():
            path=base/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));files[rel]=run.sha(path)
    receipt={**run._provenance('audit'),'anchor':0,'configuration':'J','artifact_hashes':files}
    (base/'COMPLETE.json').write_text(json.dumps(receipt))
    got=m.collect_calibration(out_root=tmp_path)
    assert len(got['records'])==4 and len(got['incomplete_units'])==11
    assert '/private/' not in json.dumps(got)
    path=base/'attack__A__SEX/selection.json';path.write_text('{}')
    with pytest.raises(ValueError,match='hash'):m.collect_calibration(out_root=tmp_path)


def test_output_is_immutable_and_marks_validation_diagnostic(tmp_path):
    m=module();result=m.build_calibration(payload())
    m.write_calibration(result,out_dir=tmp_path)
    assert (tmp_path/'ATTACK_CALIBRATION.json').is_file()
    text=(tmp_path/'ATTACK_CALIBRATION.md').read_text()
    assert 'validation' in text and 'not a' in text
    with pytest.raises(FileExistsError):m.write_calibration(result,out_dir=tmp_path)


@pytest.mark.parametrize('fault',['private_array','changed_score','duplicate','private_candidate_path'])
def test_public_builder_rejects_extra_rows_and_inconsistent_derived_values(fault):
    m=module();p=payload()
    if fault=='private_array':p['records'][0]['ids']=['private']
    elif fault=='changed_score':p['records'][0]['comparison_ce']['mlp120']['balanced']+=.1
    elif fault=='duplicate':p['records'].append(copy.deepcopy(p['records'][0]))
    else:p['records'][0]['candidate_validation']['logistic']['model_path']='/private/model'
    with pytest.raises(ValueError):m.build_calibration(p)

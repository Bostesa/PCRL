"""Exact channel equivalence on synthetic stored maps only."""
import copy
import importlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.equivalence')


def record(name, family, q, *, anchor=0, identity='a'*64, offsets=(0.,1.)):
    return {'configuration':name,'anchor':anchor,'input':family,'Q':np.asarray(q,float),
            'offsets':np.asarray(offsets,float),'coarse_states':2,'encoder_sha256':identity,
            'receipt_sha256':'b'*64,'Q_artifact_sha256':'c'*64,'solution_sha256':'d'*64,
            'solver_status':'optimal','solver':'CLARABEL','coarse_configuration':None}


def test_exact_child_copies_cross_families_reduce_but_literal_groups_remain_separate():
    m=module();q=np.array([[.2,.8],[.7,.3]])
    records=[record('coarse','T0',q),record('task','Ttask',np.repeat(q,2,axis=0)),
             record('risk','Trisk',np.repeat(q,2,axis=0)),record('coarse_copy','T0',q)]
    report=m.group_maps(records)
    assert report['counts']['accepted_verified_map_units']==4
    assert report['counts']['unique_literal_kernel_groups']==3
    assert report['counts']['unique_reduced_kernel_groups']==1
    assert report['reduced_groups'][0]['canonical_domain']=='T0'
    assert len(report['reduced_groups'][0]['members'])==4
    assert report['independent_evidence_claimed'] is False
    assert '"Q":' not in json.dumps(report) and '"offsets":' not in json.dumps(report)


def test_nonconstant_refined_coincidences_and_one_ulp_differences_never_merge():
    m=module();q=np.array([[.2,.8],[.21,.79],[.7,.3],[.71,.29]])
    near=q.copy();near[0,0]=np.nextafter(near[0,0],1.)
    records=[record('task','Ttask',q),record('risk','Trisk',q),record('near','Ttask',near)]
    report=m.group_maps(records)
    assert report['counts']['unique_literal_kernel_groups']==3
    assert report['counts']['unique_reduced_kernel_groups']==3
    assert {g['canonical_domain'] for g in report['reduced_groups']}=={'Ttask','Trisk'}


def test_constant_reduction_checks_every_row_including_absent_states():
    m=module();q=np.tile([.4,.6],(4,1));different=q.copy();different[-1]=[1.,0.]
    rows=[record('coarse','T0',q[:2]),record('task','Ttask',q),record('risk','Trisk',q),
          record('absent_diff','Trisk',different)]
    report=m.group_maps(rows)
    assert report['counts']['unique_reduced_kernel_groups']==2
    constant=next(g for g in report['reduced_groups'] if g['canonical_domain']=='constant')
    assert len(constant['members'])==3


def test_anchor_code_construction_and_action_order_are_required_identity():
    m=module();q=[[.2,.8],[.7,.3]]
    records=[record('base','T0',q),record('anchor','T0',q,anchor=1),
             record('encoder','T0',q,identity='e'*64),record('actions','T0',q,offsets=(1.,0.)),
             record('near_action','T0',q,offsets=(0.,np.nextafter(1.,2.)))]
    report=m.group_maps(records)
    assert report['counts']['unique_reduced_kernel_groups']==5
    assert report['counts']['unique_literal_kernel_groups']==5


def test_signed_zero_is_numeric_equality_and_integer_conversion_never_rounds():
    m=module();q=np.array([[1.,0.],[0.,1.]]);other=q.copy();other[0,1]=-0.
    records=[record('base','T0',q),record('copy','T0',other)]
    records[1]['offsets'][0]=-0.
    assert m.group_maps(records)['counts']['unique_literal_kernel_groups']==1
    records[1]['offsets']=np.array([0,2**53+1],dtype=np.int64)
    with pytest.raises(ValueError,match='round'):m.group_maps(records)


def test_witness_reuse_is_checked_separately_from_solver_declaration():
    m=module();q=np.array([[.2,.8],[.7,.3]])
    coarse=record('coarse','T0',q);fine=record('task','Ttask',np.repeat(q,2,axis=0))
    fine.update(coarse_configuration='coarse',coarse_solution_sha256=coarse['solution_sha256'],
                solver_status='feasible_witness',solver='embedded')
    report=m.group_maps([coarse,fine]);status=next(r for r in report['maps'] if r['configuration']=='task')['witness']
    assert status['solver_returned_embedded_witness'] and status['exact_copy_of_declared_coarse']
    fine['Q'][0]=[.3,.7]
    report=m.group_maps([coarse,fine]);status=next(r for r in report['maps'] if r['configuration']=='task')['witness']
    assert not status['exact_copy_of_declared_coarse'] and status['declaration_disagrees_with_exact_copy']


@pytest.mark.parametrize('fault',['shape','nonfinite','negative','sum','identity','duplicate'])
def test_malformed_map_records_fail(fault):
    m=module();r=record('map','T0',[[.2,.8],[.7,.3]]);rows=[r]
    if fault=='shape':r['Q']=r['Q'][:1]
    elif fault=='nonfinite':r['Q'][0,0]=np.nan
    elif fault=='negative':r['Q'][0]=[-.1,1.1]
    elif fault=='sum':r['Q'][0]=[.2,.7]
    elif fault=='identity':r['encoder_sha256']='not-a-hash'
    else:rows.append(copy.deepcopy(r))
    with pytest.raises(ValueError):m.group_maps(rows)


def artifacts(tmp_path):
    from experiments.pcrl_task_directed_release_v1 import config,run
    from experiments.pcrl_task_directed_release_v1.encoding import Codebook
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    schedule={'frozen_before_comparative_outcomes':True,'config_hash':config.digest(config.configuration())}
    write(tmp_path/'RESOURCE_SCHEDULE.json',schedule)
    base=tmp_path/'private/run/anchor_0';encoder_path=base/'encoder/encoder.joblib';encoder_path.parent.mkdir(parents=True)
    code=Codebook(np.array([0.]),np.zeros(2),np.zeros(2),np.zeros((2,2,11)),{})
    encoder=SimpleNamespace(code=code,dictionaries={17:{'offsets':np.array([0.,1.]),'zero_action':0}})
    joblib.dump(encoder,encoder_path)
    prep={**run._provenance('prepare'),'anchor':0,'cache_sha256':'e'*64,
          'artifact_hashes':{'encoder/encoder.joblib':run.sha(encoder_path)}}
    write(base/'PREPARED.json',prep)
    specs=[]
    for family in ('T0','Ttask','Trisk'):
        name=config.mechanism_id(family,'L',0.)
        spec=next(s for s in config.configuration()['maps'] if s['configuration']==name and s['anchor']==0)
        specs.append(spec);path=base/'maps'/name;path.mkdir(parents=True)
        q=np.array([[.2,.8],[.7,.3]])
        if family!='T0':q=np.repeat(q,2,axis=0)
        np.savez(path/'Q.npz',Q=q)
        meta={'feasible':True,'input':family,'configuration':name,'actions':17,'status':'optimal','solver':'CLARABEL'}
        write(path/'metadata.json',meta)
        marker={**run._provenance('map'),'configuration':name,'anchor':0,'sha256':'f'*64,
                'spec_hash':config.digest(spec),'prepared_cache_sha256':prep['cache_sha256'],
                'coarse_configuration':None,'coarse_solution_sha256':None,
                'artifact_hashes':{p.name:run.sha(p) for p in (path/'Q.npz',path/'metadata.json')}}
        write(path/'ACCEPTED.json',marker)
    return specs


def test_collector_reads_only_verified_encoder_q_and_json_and_handles_pending(tmp_path,monkeypatch):
    m=module();specs=artifacts(tmp_path)
    original_load=joblib.load;loaded=[]
    def load(path,*args,**kwargs):
        loaded.append(str(path));assert str(path).endswith('/encoder/encoder.joblib')
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(joblib,'load',load)
    report=m.collect_equivalence(out_root=tmp_path,specs=specs)
    assert report['counts']['accepted_verified_map_units']==3
    assert report['counts']['unique_reduced_kernel_groups']==1
    assert len(loaded)==1
    path=tmp_path/'private/run/anchor_0/maps'/specs[2]['configuration']/'ACCEPTED.json';path.unlink()
    report=m.collect_equivalence(out_root=tmp_path,specs=specs)
    assert report['counts']['unfinished_map_units']==1 and report['counts']['accepted_verified_map_units']==2
    from experiments.pcrl_task_directed_release_v1 import config
    absent=next(s for s in config.configuration()['maps'] if s['anchor']==1 and s['input']=='T0')
    report=m.collect_equivalence(out_root=tmp_path,specs=specs+[absent])
    assert report['counts']['pending_missing_map_units']==1


@pytest.mark.parametrize('tamper',['Q','encoder','metadata','receipt'])
def test_collector_reports_invalid_artifacts_without_merging_them(tmp_path,tamper):
    m=module();specs=artifacts(tmp_path);base=tmp_path/'private/run/anchor_0'
    if tamper=='encoder':path=base/'encoder/encoder.joblib'
    else:path=base/'maps'/specs[0]['configuration']/({'Q':'Q.npz','metadata':'metadata.json','receipt':'ACCEPTED.json'}[tamper])
    path.write_bytes(b'changed')
    report=m.collect_equivalence(out_root=tmp_path,specs=specs)
    assert report['counts']['invalid_map_units']==(3 if tamper=='encoder' else 1)
    assert report['counts']['accepted_verified_map_units']==(0 if tamper=='encoder' else 2)


def test_collector_requires_frozen_schedule_and_refuses_output_overwrite(tmp_path):
    m=module();specs=artifacts(tmp_path);out=tmp_path/'EQUIVALENCE.json'
    m.collect_equivalence(out_root=tmp_path,specs=specs,out_path=out)
    assert out.is_file()
    with pytest.raises(FileExistsError):m.collect_equivalence(out_root=tmp_path,specs=specs,out_path=out)
    (tmp_path/'RESOURCE_SCHEDULE.json').write_text('{}')
    with pytest.raises(ValueError,match='schedule'):m.collect_equivalence(out_root=tmp_path,specs=specs)


@pytest.mark.parametrize('branch',['A','C'])
def test_branch_provenance_pins_registration_sources_schedule_and_spec(tmp_path,branch):
    m=module();artifacts(tmp_path)
    from experiments.pcrl_task_directed_release_v1 import config,run,branches,robustness
    if branch=='A':
        maps,controls=branches.action33_specs(),branches.action33_controls()
        regfile,planfile,key,sourcekey='EXTRA_CONFIGS.json','EXTRA_RESOURCE_SCHEDULE.json','extra_registration_sha256','branch_source_sha256'
        source=branches.__file__
    else:
        maps,controls=robustness.robustness_specs(),[]
        regfile,planfile,key,sourcekey='ROBUSTNESS_CONFIGS.json','ROBUSTNESS_RESOURCE_SCHEDULE.json','robustness_registration_sha256','robustness_source_sha256'
        source=robustness.__file__
    registration={'schema':1,'branch':branch,'registered':True,'primary_config_hash':config.digest(config.configuration()),
                  'scientific_source_hashes':run.source_fingerprint(),sourcekey:run.sha(source),
                  'maps':maps,'controls':controls,'trigger':{'triggered':True}}
    registration['registration_payload_hash']=config.digest(registration)
    (tmp_path/regfile).write_text(json.dumps(registration))
    spec={**maps[0],key:run.sha(tmp_path/regfile),sourcekey:run.sha(source)}
    plan={'frozen_before_affected_outcomes':True,key:spec[key],'unit_ids':[spec['id']],
          'primary_resource_schedule_sha256':run.sha(tmp_path/'RESOURCE_SCHEDULE.json')}
    plan['schedule_payload_hash']=config.digest(plan);(tmp_path/planfile).write_text(json.dumps(plan))
    receipt={'extra_resource_schedule_sha256':run.sha(tmp_path/planfile)}
    m._extra_provenance(tmp_path,spec,receipt)
    bad=copy.deepcopy(spec);bad['budget']=123.
    with pytest.raises(ValueError,match='spec|registration'):m._extra_provenance(tmp_path,bad,receipt)
    plan['unit_ids']=[];plan['schedule_payload_hash']=config.digest({k:v for k,v in plan.items() if k!='schedule_payload_hash'})
    (tmp_path/planfile).write_text(json.dumps(plan));receipt['extra_resource_schedule_sha256']=run.sha(tmp_path/planfile)
    with pytest.raises(ValueError,match='schedule'):m._extra_provenance(tmp_path,spec,receipt)

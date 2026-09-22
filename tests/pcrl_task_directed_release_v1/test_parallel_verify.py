"""Synthetic selected replay scheduling, including actual spawned workers."""
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import time

import pytest


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.parallel_verify')


def selection_and_contrasts():
    names=['H','J','Q','control']
    selected={'evaluation_configurations':names,'selected_contrast_configurations':['Q','control','J']}
    contrasts={'family_size':2,'endpoints':[
        {'id':'e1','terms':[{'configuration':'Q'},{'configuration':'J'}]},
        {'id':'e2','terms':[{'configuration':'Q'},{'configuration':'control'}]}]}
    return selected,contrasts


def synthetic_worker(job):
    if job.get('sleep'):time.sleep(job['sleep'])
    if job.get('raise'):raise RuntimeError('synthetic failed unit')
    if job.get('missing'):return None
    return {'id':job['id'],'configuration':job['configuration'],'anchor':job['anchor'],
            'passed':True,'worker_pid':os.getpid(),'seconds':.001,'peak_rss_bytes':1234}


def test_plan_contains_every_endpoint_control_H_J_and_three_anchors():
    m=module();selected,contrasts=selection_and_contrasts();plan=m.selected_plan(selected,contrasts)
    assert len(plan)==12 and len({j['id'] for j in plan})==12
    assert {j['configuration'] for j in plan}=={'H','J','Q','control'}
    assert all({j['anchor'] for j in plan if j['configuration']==n}=={0,1,2} for n in selected['evaluation_configurations'])
    selected['evaluation_configurations'].remove('control')
    with pytest.raises(ValueError):m.selected_plan(selected,contrasts)


def test_resource_limit_obeys_physical_cores_and_available_RAM():
    m=module();gib=1024**3
    assert m.worker_limit(8,physical_cores=16,available_bytes=32*gib,total_bytes=64*gib)['workers']==8
    assert m.worker_limit(8,physical_cores=4,available_bytes=64*gib,total_bytes=64*gib)['workers']==4
    assert m.worker_limit(8,physical_cores=16,available_bytes=11*gib,total_bytes=64*gib)['workers']==2
    with pytest.raises(ValueError):m.worker_limit(8,physical_cores=16,available_bytes=7*gib,total_bytes=64*gib)
    with pytest.raises(ValueError):m.worker_limit(9,physical_cores=16,available_bytes=64*gib,total_bytes=64*gib)


def test_spawn_serial_and_parallel_execute_identical_complete_unit_sets():
    m=module();selected,contrasts=selection_and_contrasts();plan=m.selected_plan(selected,contrasts)
    deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=45)
    serial=m._execute(plan,workers=1,deadline=deadline,worker=synthetic_worker)
    parallel=m._execute(plan,workers=2,deadline=deadline,worker=synthetic_worker)
    assert set(serial)==set(parallel)=={j['id'] for j in plan}
    assert all(r['passed'] for r in serial.values()) and all(r['passed'] for r in parallel.values())
    assert all(r['worker_pid']!=os.getpid() for r in parallel.values())


def test_failed_and_missing_worker_returns_never_count_as_passing():
    m=module();plan=[{'id':'a','configuration':'H','anchor':0,'raise':True},
                    {'id':'b','configuration':'H','anchor':1,'missing':True},
                    {'id':'c','configuration':'H','anchor':2}]
    result=m._execute(plan,workers=2,deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=45),worker=synthetic_worker)
    assert not result['a']['passed'] and not result['b']['passed'] and result['c']['passed']


def test_deadline_terminates_own_spawned_work_and_marks_every_unit_incomplete():
    m=module();plan=[{'id':str(i),'configuration':'H','anchor':i,'sleep':10} for i in range(3)]
    started=time.monotonic()
    result=m._execute(plan,workers=1,deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=.2),worker=synthetic_worker)
    assert time.monotonic()-started<5
    assert set(result)=={'0','1','2'} and not any(r['passed'] for r in result.values())
    assert {r['status'] for r in result.values()}=={'deadline_incomplete'}


def helper_result():
    m=module();roles={r:{} for r in m.ROLES}
    return {'passed':True,'anchor':0,'configuration':'H','selection_sha256':'a'*64,
            'original_fallback_allowed':False,'original_selection_pin_checked':True,
            'preparation':{'passed':True},'mathematical_replay':{'passed':True},
            'unit':{'passed':True,'validation':{'passed':True,'roles':roles},'ancestor_parity':{'passed':True,'roles':roles},
                    'evaluation':{'passed':True,'pools':{'test':{r:{'wire_check':{'passed':True,'repetitions':512}} for r in m.ROLES}}}}}


@pytest.mark.parametrize('fault',['missing_role','weak_wire','failed_parity','fallback','wrong_selection'])
def test_worker_summary_requires_all16_roles_fullwire_and_frozen_pins(fault):
    m=module();result=helper_result()
    if fault=='missing_role':result['unit']['evaluation']['pools']['test'].pop(m.ROLES[0])
    elif fault=='weak_wire':result['unit']['evaluation']['pools']['test'][m.ROLES[0]]['wire_check']['repetitions']=32
    elif fault=='failed_parity':result['unit']['ancestor_parity']['passed']=False
    elif fault=='fallback':result['original_fallback_allowed']=True
    else:result['selection_sha256']='b'*64
    with pytest.raises(ValueError):m.coverage_summary(result,configuration='H',anchor=0,selection_sha256='a'*64)


def test_writers_must_be_closed_before_global_selection_or_any_replay(tmp_path,monkeypatch):
    m=module();called=[]
    monkeypatch.setattr(m,'verify_frozen_selection',lambda **kwargs:called.append(True))
    with pytest.raises(ValueError,match='writers'):
        m.verify_parallel(artifact_root=tmp_path,original_root=tmp_path,out_dir=tmp_path/'private/new',writers_closed=False)
    assert called==[]


@pytest.mark.parametrize('fault',[None,'closure','missing','tampered_report'])
def test_parent_reconstructs_selection_once_rechecks_closure_and_every_unit_report(tmp_path,monkeypatch,fault):
    m=module();root=tmp_path/'artifacts';study=root/'results'/m.STUDY;study.mkdir(parents=True)
    selected,contrasts=selection_and_contrasts();selection_file=study/'SELECTION.json';selection_file.write_text(json.dumps(selected))
    pin=m._sha(selection_file);calls=[]
    def frozen(**kwargs):
        calls.append(kwargs)
        return {'selection':selected,'contrasts':contrasts,'report':{'passed':True,'selection_sha256':pin}}
    monkeypatch.setattr(m,'verify_frozen_selection',frozen)
    monkeypatch.setattr(m,'worker_limit',lambda *args,**kwargs:{'workers':2})
    monkeypatch.setattr(m,'_deadline',lambda value:dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=30))
    monkeypatch.setattr(m,'source_snapshot',lambda r:{'frozen_documents':{'SELECTION.json':m._sha(selection_file)}})
    def execute(jobs,**kwargs):
        result={}
        for job in jobs:
            if fault=='missing' and job==jobs[-1]:continue
            helper=helper_result();helper.update(configuration=job['configuration'],anchor=job['anchor'],selection_sha256=pin)
            path=Path(job['out_dir'])/'restore_verification.json';m._write(path,helper)
            coverage=m.coverage_summary(helper,configuration=job['configuration'],anchor=job['anchor'],selection_sha256=pin)
            row={k:job[k] for k in ('id','configuration','anchor')}
            row.update(passed=True,status='passed',**coverage,worker_pid=os.getpid(),seconds=.1,peak_rss_bytes=100,
                       unit_report_sha256=m._sha(path),unit_report_relative_path=str(path.relative_to(Path(job['run_dir']))))
            result[job['id']]=row
            if fault=='tampered_report' and job==jobs[0]:path.write_text('{}')
        if fault=='closure':selection_file.write_text('{}')
        return result
    monkeypatch.setattr(m,'_execute',execute)
    result=m.verify_parallel(artifact_root=root,original_root='/nonexistent/original',
        out_dir=study/'private/parallel_test',writers_closed=True,public_report=study/'PUBLIC_VERIFY.json')
    assert len(calls)==1 and result['planned_units']==12
    assert result['passed'] is (fault is None)
    assert len(result['units'])==12
    if fault is None:
        assert result['validation_role_checks']==result['evaluation_role_checks']==192
        assert result['wire_draw_repetitions_total']==192*512
        assert result['source_selection_closure_unchanged']
    assert '/nonexistent/original' not in json.dumps(result)

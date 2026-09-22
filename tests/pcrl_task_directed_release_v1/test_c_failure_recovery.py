"""Synthetic recovery orchestration only; no live study artifacts or AWS."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from experiments.pcrl_task_directed_release_v1 import finite, mechanisms, run

from experiments.pcrl_task_directed_release_v1 import c_failure_recovery as c


class Code:
    def parents(self,family):return np.array([0,0,1,1])


@pytest.fixture
def synthetic(tmp_path,monkeypatch,request):
    policy=getattr(request,'param','L');name=f'Trisk_{policy}_0.002_a17_fineC';anchor=2
    out=tmp_path/'results'/run.STUDY
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUT',out)
    spec={'id':name+'/anchor_2','anchor':anchor,'configuration':name,'input':'Trisk','policy':policy,
        'budget':.002,'max_actions':17,'branch':'C','coarse_witness':'none',
        'robustness_registration_sha256':'reg','robustness_source_sha256':'robust'}
    def lookup(a,n,*args):
        if a!=anchor or n!=name:raise ValueError('Unexpected C slot')
        return copy.deepcopy(spec)
    monkeypatch.setattr(c,'spec_for',lookup)
    monkeypatch.setattr(run,'map_spec',lookup)
    # Native run.map_spec argument order is name,anchor.
    monkeypatch.setattr(run,'map_spec',lambda n,a:lookup(a,n))
    monkeypatch.setattr(run,'_extra_module',lambda s:SimpleNamespace(require_scheduled=lambda s:'sched'))
    monkeypatch.setattr(run,'_provenance',lambda stage:{'schema':2,'stage':stage,'config_hash':'cfg',
        'source_hashes':{'synthetic':'source'},'runner_sha256':'runner'})
    monkeypatch.setattr(c,'context',lambda out,root,controller_source=None:{
        'config_hash':'cfg','source_hashes':{'numerical_recovery':run.sha(c.recovery.__file__)},
        'controller_source_sha256':run.sha(c.__file__ if controller_source is None else controller_source),
        'resource_schedule_sha256':'resource','robustness_registration_sha256':'reg',
        'robustness_resource_schedule_sha256':'sched'})
    p=c.paths(out,anchor,name);p['map'].mkdir(parents=True)
    cost=np.array([[.2,.1,.3]]*4);mass=np.array([4,4,4,4])
    roles={f'{v}/{s}/{w}{suffix}':np.full((2 if s=='SEX' else 9,4 if suffix else 2,4),
        1/((2 if s=='SEX' else 9)*(4 if suffix else 2)*4))
        for v in ('A','AB') for s in ('SEX','RAC1P') for w in ('U','W') for suffix in ('','/fineC')}
    table={'cost_U':cost.copy(),'cost_W':cost.copy(),'cost':cost,'state_mass':mass,
        'roles':roles,'support':{},'actions':17,'aggregation':{'maximum_error':0.},'population':{'mechanism_rows':16}}
    prepared={'ctx':{'pools':{pool:{} for pool in run.FIT_POOLS}},
        'encoder':SimpleNamespace(code=Code(),dictionaries={17:{'zero_action':0}})}
    joblib.dump(prepared,p['base']/'prepared.joblib')
    c.new_json(p['base']/'PREPARED.json',{**run._provenance('prepare'),'anchor':anchor,
        'cache_sha256':run.sha(p['base']/'prepared.joblib'),
        'artifact_hashes':{'prepared.joblib':run.sha(p['base']/'prepared.joblib')}})
    fine=p['base']/'branches/fineC';fine.mkdir(parents=True)
    joblib.dump({'tables':{'Trisk':table}},fine/'tables.joblib')
    c.new_json(fine/'TABLES.json',{'anchor':anchor,'branch':'C',
        'prepared_cache_sha256':run.sha(p['base']/'prepared.joblib'),
        'robustness_registration_sha256':'reg','robustness_source_sha256':'robust',
        'robustness_resource_schedule_sha256':'sched','artifact_hashes':{'tables.joblib':run.sha(fine/'tables.joblib')}})
    selected={k:v for k,v in roles.items() if policy=='C' or k.startswith('A/')}
    _,_,_,support=finite._support(4,selected,mass,np.array([0,0,1,1]),cost)
    original={'Q':None,'objective':None,'status':'failed','solver':None,'feasible':False,'optimal':False,
        'residuals':{'feasible':False,'reason':'no_accepted_solver_candidate'},'rank':0,'role_ranks':{},
        'support':support,'attempts':[{'solver':'CLARABEL','status':'solver_error','accepted':False},
                                    {'solver':'SCS','status':'optimal','accepted':False,'raw_simplex':1.988526323604134e-7}],
        'witness':None,'tolerances':dict(finite.ACCEPTANCE_TOLERANCES),'budget':.002,'information_units':'nats',
        'input':'Trisk','policy':policy,'configuration':name,'actions':17,'constrained_roles':sorted(selected),
        'embedding':None,'aggregation':table['aggregation'],'population':table['population'],
        'constant_action':1,'cost':cost}
    joblib.dump(original,p['map']/'solution.joblib')
    c.new_json(p['map']/'metadata.json',{k:v for k,v in original.items() if k not in ('Q','cost')})
    arrays={k:table[k] for k in ('cost_U','cost_W','cost','state_mass')}
    arrays.update({'joint/'+k:v for k,v in roles.items()})
    np.savez_compressed(p['map']/'tables.npz',**arrays)
    return SimpleNamespace(out=out,root=tmp_path,anchor=anchor,name=name,p=p,table=table,
                           selected=selected,original=original,spec=spec)


def registered(s):
    record=c.register(s.anchor,s.name)
    return record,run.sha(s.p['registration'])


@pytest.mark.parametrize('synthetic',['L','C'],indirect=True)
def test_once_only_exact_input_recovery_installs_native_map_and_preserves_original(synthetic,monkeypatch):
    s=synthetic;record,pin=registered(s);before=c.tree(s.p['map']);calls=[]
    solver=c.recovery.solve_normalized_retry
    def spy(cost,roles,budget,**kwargs):
        calls.append(True)
        np.testing.assert_array_equal(cost,s.table['cost'])
        assert set(roles)==set(s.selected) and len(roles)==(8 if s.spec['policy']=='L' else 16)
        assert all(np.array_equal(roles[k],s.selected[k]) for k in roles)
        assert budget==.002 and kwargs['embedded_q'] is None and kwargs['zero_action']==0
        np.testing.assert_array_equal(kwargs['parent'],[0,0,1,1])
        np.testing.assert_array_equal(kwargs['state_mass'],s.table['state_mass'])
        return solver(cost,roles,budget,**kwargs)
    monkeypatch.setattr(c.recovery,'solve_normalized_retry',spy)
    result=c.execute(s.anchor,s.name,registration_sha256=pin)
    assert result['accepted'] and calls==[True] and c.tree(s.p['map'])==before
    with pytest.raises(FileExistsError):c.execute(s.anchor,s.name,registration_sha256=pin)
    assert calls==[True]
    installed=c.install(s.anchor,s.name,registration_sha256=pin)
    assert installed['decision']=='installed_accepted_retry' and installed['audit_allowed']
    assert c.tree(s.p['backup']/'map')==before
    assert c.verify_installed(s.anchor,s.name)['verified']
    receipt=run._map_receipt(s.anchor,s.name)
    assert 'c_failure_retry' in receipt and 'numerical_retry' not in receipt
    assert run._load_map(s.anchor,s.name)['feasible']
    assert not s.p['audit'].exists()
    assert c.verify_all()['installed_retries']==1


def test_failed_retry_remains_unresolved_and_cannot_be_solved_again(synthetic,monkeypatch):
    s=synthetic;_,pin=registered(s);before=c.tree(s.p['map'])
    result={'Q':None,'accepted':False,'feasible':False,'optimal':False,'status':'failed',
        'solver_status':'solver_error','attempts':[{'solver':'CLARABEL'}],
        'tolerances':dict(finite.ACCEPTANCE_TOLERANCES),'solver_settings':dict(finite.SOLVER_SETTINGS['CLARABEL'])}
    monkeypatch.setattr(c.recovery,'solve_normalized_retry',lambda *a,**k:copy.deepcopy(result))
    assert not c.execute(s.anchor,s.name,registration_sha256=pin)['accepted']
    receipt=c.install(s.anchor,s.name,registration_sha256=pin)
    assert receipt['decision']=='unresolved_rejected_retry' and receipt['audit_allowed'] is False
    assert c.tree(s.p['map'])==before and not (s.p['map']/'ACCEPTED.json').exists()
    with pytest.raises(FileExistsError):c.execute(s.anchor,s.name,registration_sha256=pin)


def test_exception_consumes_claim_without_changing_original(synthetic,monkeypatch):
    s=synthetic;_,pin=registered(s);before=c.tree(s.p['map'])
    def crash(*args,**kwargs):raise RuntimeError('synthetic process failure')
    monkeypatch.setattr(c.recovery,'solve_normalized_retry',crash)
    with pytest.raises(RuntimeError,match='synthetic'):c.execute(s.anchor,s.name,registration_sha256=pin)
    assert json.loads((s.p['stage']/'FAILED.json').read_text())['attempt_consumed']
    assert c.tree(s.p['map'])==before
    with pytest.raises(FileExistsError):c.execute(s.anchor,s.name,registration_sha256=pin)


@pytest.mark.parametrize('change',['accepted','feasible','witness','missing_attempt','audit'])
def test_only_complete_failed_unaccepted_no_witness_slots_are_eligible(synthetic,change):
    s=synthetic
    if change=='accepted':(s.p['map']/'ACCEPTED.json').write_text('{}')
    elif change=='audit':s.p['audit'].mkdir(parents=True)
    else:
        value=copy.deepcopy(s.original)
        if change=='feasible':value['feasible']=True
        if change=='witness':value['witness']={'feasible':True}
        if change=='missing_attempt':value['attempts']=value['attempts'][:1]
        joblib.dump(value,s.p['map']/'solution.joblib')
    with pytest.raises(ValueError):c.register(s.anchor,s.name)
    assert not s.p['registration'].exists()


def test_registration_pin_and_prepared_dependency_checked_before_solver(synthetic,monkeypatch):
    s=synthetic;_,pin=registered(s)
    monkeypatch.setattr(c.recovery,'solve_normalized_retry',lambda *a,**k:pytest.fail('unexpected solve'))
    with pytest.raises(ValueError,match='SHA'):c.execute(s.anchor,s.name,registration_sha256='0'*64)
    (s.p['base']/'prepared.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):c.execute(s.anchor,s.name,registration_sha256=pin)
    assert not s.p['stage'].exists()


def test_exact_fine_law_and_saved_tables_required(synthetic):
    s=synthetic;record,pin=registered(s)
    p=s.p['base']/'branches/fineC/tables.joblib'
    payload=joblib.load(p);payload['tables']['Trisk']['roles'].pop('A/SEX/U/fineC');joblib.dump(payload,p)
    with pytest.raises(ValueError,match='tables differ'):c.inputs(record,s.p)


def test_independent_feasibility_refuses_falsely_accepted_channel(synthetic,monkeypatch):
    s=synthetic;_,pin=registered(s)
    bad={'Q':np.full((4,3),.9),'accepted':True,'feasible':True,'optimal':True,'status':'optimal',
        'solver_status':'optimal','objective':0.,'attempts':[{'solver':'CLARABEL'}],
        'tolerances':dict(finite.ACCEPTANCE_TOLERANCES),'solver_settings':dict(finite.SOLVER_SETTINGS['CLARABEL'])}
    monkeypatch.setattr(c.recovery,'solve_normalized_retry',lambda *a,**k:bad)
    with pytest.raises(ValueError):c.execute(s.anchor,s.name,registration_sha256=pin)
    assert not (s.p['map']/'ACCEPTED.json').exists()


def test_target_and_selection_locks_exclusive_but_unrelated_A_unit_remains_available(synthetic):
    s=synthetic
    with c.slot_locks(s.anchor,s.name):
        for key in ('programme/selection_freeze',f'map/{s.anchor}/{s.name}',f'audit/{s.anchor}/{s.name}'):
            with pytest.raises(RuntimeError,match='locked'):
                with run.unit_lock(key):pass
        with run.unit_lock('audit/0/Trisk_U_a33'):pass


def test_frozen_selection_blocks_all_recovery(synthetic):
    s=synthetic;c.new_json(s.out/'SELECTION.json',{'selection_frozen':True})
    with pytest.raises(RuntimeError,match='frozen'):c.register(s.anchor,s.name)


def test_restored_verifier_uses_only_explicit_relocated_artifacts(synthetic,tmp_path):
    s=synthetic;_,pin=registered(s);c.execute(s.anchor,s.name,registration_sha256=pin)
    c.install(s.anchor,s.name,registration_sha256=pin)
    relocated=tmp_path/'relocated';restored_out=relocated/'results'/run.STUDY
    shutil.copytree(s.out,restored_out)
    s.out.rename(s.out.with_name('original_inaccessible'))
    assert c.verify_installed(s.anchor,s.name,study_out=restored_out,source_root=relocated)['verified']
    assert c.verify_all(study_out=restored_out,source_root=relocated)['installed_retries']==1
    (restored_out/'private/c_recovery/originals/anchor_2'/s.name/'map/metadata.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        c.verify_installed(s.anchor,s.name,study_out=restored_out,source_root=relocated)


def test_installation_failure_rolls_back_original_bytes(synthetic,monkeypatch):
    s=synthetic;_,pin=registered(s);c.execute(s.anchor,s.name,registration_sha256=pin)
    before=c.tree(s.p['map'])
    monkeypatch.setattr(c,'verify_installed',lambda *a,**k:(_ for _ in ()).throw(ValueError('synthetic verification failure')))
    with pytest.raises(ValueError,match='synthetic'):c.install(s.anchor,s.name,registration_sha256=pin)
    assert c.tree(s.p['map'])==before
    assert json.loads((s.p['stage']/'INSTALLATION.json').read_text())['decision']=='rolled_back_requires_manual_review'


def test_recovery_cannot_read_through_symlinked_map_ancestor(synthetic,tmp_path):
    s=synthetic;maps=s.p['map'].parent;renamed=maps.with_name('moved_maps')
    maps.rename(renamed);maps.symlink_to(renamed,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):c.register(s.anchor,s.name)


def test_all_registered_recoveries_must_be_resolved_before_closure(synthetic):
    s=synthetic;registered(s)
    with pytest.raises(FileNotFoundError):c.verify_all()


@pytest.mark.parametrize('field,value',[('anchor',0),('configuration','other'),('accepted_sha256','0'*64),
    ('registration_sha256','0'*64),('retry_receipt_sha256','0'*64),('controller_source_sha256','0'*64),('audit_allowed',False)])
def test_all_verifier_binds_completed_installation_to_accepted_map(synthetic,field,value):
    s=synthetic;_,pin=registered(s);c.execute(s.anchor,s.name,registration_sha256=pin)
    c.install(s.anchor,s.name,registration_sha256=pin)
    path=s.p['stage']/'INSTALLATION.json';record=json.loads(path.read_text());record[field]=value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='installation receipt'):c.verify_all()

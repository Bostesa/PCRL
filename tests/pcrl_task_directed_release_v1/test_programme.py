"""Prospective orchestration integrity with synthetic receipts and stubs."""
import copy
import hashlib
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import programme as p,run,config,branches,robustness,reporting,selection


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def root(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUT',tmp_path);monkeypatch.setattr(run,'OUT',tmp_path)
    write(tmp_path/'RESOURCE_SCHEDULE.json',{'frozen_before_comparative_outcomes':True,'config_hash':config.digest(config.configuration())})
    monkeypatch.setattr(p,'active_scientific_writers',lambda:[])


def test_inventory_requires_all180_primary_and_verified_receipts(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch)
    assert len(p.completion_inventory())==180
    assert not any(r['complete'] for r in p.completion_inventory())
    path=tmp_path/'private/run/anchor_0/audits/H/COMPLETE.json';write(path,{})
    with pytest.raises(ValueError):p.completion_inventory()


def test_inventory_delegates_full_dependency_validation_and_rejects_bad_registry(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);base=tmp_path/'private/run/anchor_0/audits/H'
    write(base/'summary.json',{});(base/'registry.joblib').write_bytes(b'synthetic registry, never deserialized')
    receipt={**run._provenance('audit'),'anchor':0,'configuration':'H','role_audits':16,
             'registry_sha256':run.sha(base/'registry.joblib'),
             'artifact_hashes':{n:run.sha(base/n) for n in ('summary.json','registry.joblib')}}
    write(base/'COMPLETE.json',receipt);calls=[]
    def verified(anchor,name):
        calls.append((anchor,name));return run._read_marker(base/'COMPLETE.json','audit',anchor=anchor,name=name)
    monkeypatch.setattr(run,'_audit_receipt',verified)
    rows=p.completion_inventory();assert calls==[(0,'H')]
    assert sum(r['complete'] for r in rows)==1
    (base/'registry.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash'):p.completion_inventory()


def test_freeze_fails_before_comparative_reads_for_missing_units_or_evaluation_attempt(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch)
    monkeypatch.setattr(reporting,'collect_validation',lambda:pytest.fail('validation read before integrity gate'))
    with pytest.raises(RuntimeError,match='incomplete'):p.freeze()
    key='evaluation/0/H';write(tmp_path/'private/locks'/(hashlib.sha256(key.encode()).hexdigest()+'.lock'),{'key':key})
    with pytest.raises(RuntimeError,match='evaluation'):p.freeze(closeout_reason='fixed resource deadline')


def test_whitespace_closeout_is_not_a_genuine_reason(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch)
    monkeypatch.setattr(reporting,'collect_validation',lambda:pytest.fail('blank closeout bypassed gate'))
    with pytest.raises(ValueError,match='closeout'):p.freeze(closeout_reason='  ')


def test_active_writers_block_freeze_before_validation(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);monkeypatch.setattr(p,'active_scientific_writers',lambda:[123])
    monkeypatch.setattr(reporting,'collect_validation',lambda:pytest.fail('active writer ignored'))
    with pytest.raises(RuntimeError,match='writers'):p.freeze()


def test_numerical_retry_writers_block_selection_but_registration_is_read_only(monkeypatch):
    from types import SimpleNamespace
    module='experiments.pcrl_task_directed_release_v1.numerical_recovery_run'
    processes=[SimpleNamespace(pid=100+i,info={'cmdline':['python','-m',module,command]})
               for i,command in enumerate(('execute','install','register'))]
    monkeypatch.setattr(p.psutil,'process_iter',lambda fields:processes)
    monkeypatch.setattr(p.os,'getpid',lambda:999)
    assert p.active_scientific_writers()==[100,101]
    monkeypatch.setattr(p.os,'getpid',lambda:100)
    assert p.active_scientific_writers()==[101]


def test_extra_schedule_is_resolved_against_verified_registry(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);spec=branches.action33_specs()[0]
    write(tmp_path/'EXTRA_RESOURCE_SCHEDULE.json',{'unit_ids':[spec['id']]})
    monkeypatch.setattr(branches,'_registration',lambda:{'maps':branches.action33_specs(),'controls':branches.action33_controls()})
    monkeypatch.setattr(branches,'lookup_release',lambda name,anchor:{**spec})
    checked=[];monkeypatch.setattr(branches,'require_scheduled',lambda value:checked.append(value) or 'a'*64)
    rows=p.completion_inventory();assert len(rows)==181 and checked==[spec]
    write(tmp_path/'EXTRA_RESOURCE_SCHEDULE.json',{'unit_ids':['unknown/anchor_0']})
    with pytest.raises(ValueError,match='registered'):p.completion_inventory()
    write(tmp_path/'EXTRA_RESOURCE_SCHEDULE.json',{'unit_ids':[spec['id'],spec['id']]})
    with pytest.raises(ValueError,match='Duplicate'):p.completion_inventory()


def test_common_extension_specs_must_match_all_three_anchors(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);write(tmp_path/'EXTRA_CONFIGS.json',{})
    maps=branches.action33_specs();controls=branches.action33_controls()
    monkeypatch.setattr(branches,'_registration',lambda:{'maps':maps,'controls':controls})
    def lookup(name,anchor):return next(s for s in maps+controls if s['configuration']==name and s['anchor']==anchor)
    monkeypatch.setattr(branches,'lookup_release',lookup)
    result=p.registration_inputs()
    assert len(result['registered_extensions'])==21 and len(result['registered_controls'])==20
    assert len(result['candidate_ids'])==36
    original=lookup
    monkeypatch.setattr(branches,'lookup_release',lambda name,anchor:{**original(name,anchor),'unexpected':anchor})
    with pytest.raises(ValueError,match='across anchors'):p.registration_inputs()


def test_common_extension_cannot_silently_drop_an_anchor(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);write(tmp_path/'EXTRA_CONFIGS.json',{})
    maps=branches.action33_specs()[:2]
    monkeypatch.setattr(branches,'_registration',lambda:{'maps':maps,'controls':[]})
    monkeypatch.setattr(branches,'lookup_release',lambda name,anchor:next(s for s in maps if s['anchor']==anchor))
    with pytest.raises(ValueError,match='three anchors'):p.registration_inputs()


def test_explicit_resource_closeout_preserves_missing_inventory_without_inventing_completion(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch)
    def collect():write(tmp_path/'VALIDATION_GRID.json',{'synthetic':True});return {'synthetic':True}
    monkeypatch.setattr(reporting,'collect_validation',collect)
    monkeypatch.setattr(selection,'select_validation',lambda value,**kwargs:{'synthetic_selected':True})
    monkeypatch.setattr(reporting,'bind_frozen_provenance',lambda selected:selected)
    def freeze(selected,**kwargs):
        write(tmp_path/'SELECTION.json',selected);write(tmp_path/'CONTRASTS.json',{});return selected
    monkeypatch.setattr(selection,'freeze_selection',freeze)
    result=p.freeze(closeout_reason=' fixed resource deadline ')
    completion=result['completion_at_freeze']
    assert len(completion['missing'])==180 and not any(r['complete'] for r in completion['records'])
    assert completion['closeout_reason']=='fixed resource deadline'


@pytest.mark.parametrize('outer',[False,True])
def test_branch_A_uses_fixed_whole_comparison_order_and_valid_dependencies(tmp_path,monkeypatch,outer):
    root(tmp_path,monkeypatch);maps=branches.action33_specs();controls=branches.action33_controls()
    monkeypatch.setattr(branches,'register_action33',lambda:{'registered':True,'maps':maps,'controls':controls})
    monkeypatch.setattr(branches,'freeze_action33_schedule',lambda ids,**kwargs:{'ids':ids,**kwargs})
    result=p.branch_a_schedule(resource_decision={'frozen':'synthetic'},include_outer_budgets=outer)
    ids=result['ids'];lookup={s['id']:s for s in maps+controls}
    assert len(ids)==(123 if outer else 87) and len(set(ids))==len(ids)
    assert all(lookup[i].get('policy')=='U' for i in ids[:9])
    assert all(lookup[i].get('budget')==.002 for i in ids[9:27])
    assert all(lookup[i].get('kind')=='control' for i in ids[27:87])
    if outer:
        assert all(lookup[i]['budget']==.0005 for i in ids[87:105])
        assert all(lookup[i]['budget']==.01 for i in ids[105:])
    for position,ident in enumerate(ids):
        s=lookup[ident]
        if s.get('kind')=='control':required=s['required_map']
        elif s['input']!='T0':required=config.mechanism_id('T0',s['policy'],s['budget'],33)
        else:required=None
        if required:assert required+f"/anchor_{s['anchor']}" in ids[:position]


def supplement_stub(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    import experiments.pcrl_task_directed_release_v1 as package
    specs=[]
    for scope in ('mechanism40','union88'):
        for method,family in (('leace','supervised_LEACE'),('splince','supervised_SPLINCE')):
            name=f'{method}_supervised_{scope}'
            for anchor in (0,1,2):
                specs.append({'id':f'{name}/anchor_{anchor}','configuration':name,'anchor':anchor,
                    'branch':'baseline_supplement','kind':'control','required':True,'label_matched':True,
                    'scope':scope,'method':method,'baseline_family':family,'max_actions':17,
                    'required_map':None,'canonical_release':name})
    checked=[]
    def lookup(name,anchor):
        return {**next(s for s in specs if s['configuration']==name and s['anchor']==anchor),
                'baseline_registration_sha256':'a'*64,'baseline_source_sha256':'b'*64}
    supplement=SimpleNamespace(registration=lambda:{'controls':specs},specs=lambda:specs,
                               lookup_release=lookup,require_scheduled=lambda spec:checked.append(spec) or 'c'*64)
    monkeypatch.setitem(sys.modules,package.__name__+'.baseline_supplement',supplement)
    monkeypatch.setattr(package,'baseline_supplement',supplement,raising=False)
    write(tmp_path/'BASELINE_SUPPLEMENTS.json',{})
    return specs,checked


def test_baseline_supplement_is_all12_required_units_and_four_common_family_controls(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);specs,checked=supplement_stub(tmp_path,monkeypatch)
    write(tmp_path/'BASELINE_SUPPLEMENT_SCHEDULE.json',{'unit_ids':[s['id'] for s in specs]})
    rows=p.completion_inventory();extra=[r for r in rows if r['scope']=='baseline_supplement']
    assert len(rows)==192 and len(extra)==12 and all(r['required'] for r in extra)
    assert len(checked)==12 and not any(r['complete'] for r in extra)
    common=p.registration_inputs()['registered_controls']
    assert len(common)==4 and all(s['required'] and s['baseline_registration_sha256']=='a'*64 for s in common.values())
    assert {s['baseline_family'] for s in common.values()}=={'supervised_LEACE','supervised_SPLINCE'}
    monkeypatch.setattr(reporting,'collect_validation',lambda:pytest.fail('missing mandatory supplement allowed selection'))
    with pytest.raises(RuntimeError,match='incomplete'):p.freeze()


def test_registered_but_unscheduled_supplement_cannot_disappear_from_closeout(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);specs,checked=supplement_stub(tmp_path,monkeypatch)
    rows=p.completion_inventory()
    assert len(rows)==192 and len([r for r in rows if r['scope']=='baseline_supplement'])==12
    assert not checked
    def collect():write(tmp_path/'VALIDATION_GRID.json',{});return {}
    monkeypatch.setattr(reporting,'collect_validation',collect)
    monkeypatch.setattr(selection,'select_validation',lambda value,**kwargs:{'arguments':kwargs})
    monkeypatch.setattr(reporting,'bind_frozen_provenance',lambda selected:selected)
    def freeze(selected,**kwargs):
        write(tmp_path/'SELECTION.json',selected);write(tmp_path/'CONTRASTS.json',{});return selected
    monkeypatch.setattr(selection,'freeze_selection',freeze)
    result=p.freeze(closeout_reason='documented access failure after registration')
    assert len([r for r in result['completion_at_freeze']['missing'] if r['scope']=='baseline_supplement'])==12
    assert len(result['arguments']['registered_controls'])==4


def test_baseline_schedule_cannot_choose_subset_or_fit_before_schedule(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);specs,_=supplement_stub(tmp_path,monkeypatch)
    path=tmp_path/'BASELINE_SUPPLEMENT_SCHEDULE.json'
    write(path,{'unit_ids':[s['id'] for s in specs[:-1]]})
    with pytest.raises(ValueError,match='twelve|12|all'):p.completion_inventory()
    path.unlink()
    s=specs[0];write(tmp_path/'private/run'/f"anchor_{s['anchor']}"/'audits'/s['configuration']/'COMPLETE.json',{})
    with pytest.raises(ValueError,match='schedule'):p.completion_inventory()


def test_amendment_activation_prevents_deleted_registry_from_bypassing_required_controls(tmp_path,monkeypatch):
    root(tmp_path,monkeypatch);supplement_stub(tmp_path,monkeypatch)
    from experiments.pcrl_task_directed_release_v1 import baseline_supplement
    (tmp_path/'AMENDMENT_6_BASELINE_FAIRNESS.md').write_text('Prospective synthetic amendment')
    path=tmp_path/'BASELINE_SUPPLEMENTS.json';path.unlink()
    monkeypatch.setattr(baseline_supplement,'registration',lambda:json.loads(path.read_text()))
    assert p.baseline_supplement_active()
    with pytest.raises(FileNotFoundError):p.completion_inventory()
    with pytest.raises(FileNotFoundError):p.registration_inputs()


def test_real_baseline_registration_and_schedule_integrate_without_fitting(tmp_path,monkeypatch):
    import importlib
    from experiments.pcrl_task_directed_release_v1 import scheduler
    supplement=importlib.import_module('experiments.pcrl_task_directed_release_v1.baseline_supplement')
    root(tmp_path,monkeypatch);monkeypatch.setattr(scheduler,'OUT',tmp_path)
    (tmp_path/supplement.AMENDMENT).write_text('Synthetic prospective fairness amendment; no data')
    registry=supplement.register()
    assert len(registry['controls'])==12
    schedule=supplement.freeze_schedule(resource_decision={'registry_commit':'a'*40,'scope':'synthetic metadata-only test'})
    assert schedule['unit_ids']==[s['id'] for s in supplement.specs()]
    arguments=p.registration_inputs();assert len(arguments['registered_controls'])==4
    rows=p.completion_inventory();assert len(rows)==192
    assert not any(r['complete'] for r in rows)
    block=scheduler.extension_phases('baseline')
    assert len(block[0][1])==12
    assert not (tmp_path/'private/run').exists()

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

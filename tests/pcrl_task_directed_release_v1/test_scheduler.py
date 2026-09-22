from experiments.pcrl_task_directed_release_v1.scheduler import phases
from experiments.pcrl_task_directed_release_v1.config import release_ledger


def test_complete_schedule_and_required_first_block():
    blocks=phases();all_audits=[j for _,js in blocks for j in js if j['command']=='audit']
    actual={(j['name'],j['anchor']) for j in all_audits}
    expected={(j['configuration'],j['anchor']) for j in release_ledger()['records']}
    assert actual==expected
    assert len(all_audits)==len(actual)==180
    first=dict(blocks)['mandatory_T0']
    assert len(first)==18
    assert all(j['name'].startswith('T0_') for j in first)
    assert [n for n,_ in blocks].index('mandatory_T0')<[n for n,_ in blocks].index('paired_middle')
    assert not any(j['command']=='evaluate' for _,js in blocks for j in js)


def test_dependency_scheduler_gates_refinements_until_complete_t0(tmp_path,monkeypatch):
 from experiments.pcrl_task_directed_release_v1 import scheduler as s
 monkeypatch.setattr(s,'OUT',tmp_path)
 base=tmp_path/'private/run/anchor_0';base.mkdir(parents=True)
 assert s.dependencies_ready(s.job('prepare',0))
 assert not s.dependencies_ready(s.job('audit',0,'H'))
 (base/'PREPARED.json').write_text('{}')
 assert s.dependencies_ready(s.job('audit',0,'H'))
 (base/'audits/H').mkdir(parents=True);(base/'audits/H/COMPLETE.json').write_text('{}')
 j={**s.job('audit',0,'T0_L_0.002_a17'),'phase':'mandatory_T0'}
 assert s.dependencies_ready(j)
 refined={**s.job('audit',0,'Trisk_L_0.002_a17'),'phase':'paired_middle'}
 assert not s.dependencies_ready(refined)
 for item in dict(s.phases())['mandatory_T0']:
  marker=s.marker_for(item);marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text('{}')
 assert not s.dependencies_ready(refined)
 q=base/'maps/T0_L_0.002_a17/ACCEPTED.json';q.parent.mkdir(parents=True);q.write_text('{}')
 assert s.dependencies_ready(refined)


def test_evaluation_schedule_covers_frozen_extensions_and_rejects_missing_audit(tmp_path,monkeypatch):
 import json
 import pytest
 from experiments.pcrl_task_directed_release_v1 import scheduler as s
 monkeypatch.setattr(s,'OUT',tmp_path)
 names=['H','Trisk_C_0.002_a33','Trisk_L_0.01_a17_fineC']
 (tmp_path/'SELECTION.json').write_text(json.dumps({'selection_frozen':True,'evaluation_configurations':names}))
 for name in names:
  for anchor in (0,1,2):
   marker=tmp_path/'private/run'/f'anchor_{anchor}'/'audits'/name/'COMPLETE.json'
   marker.parent.mkdir(parents=True);marker.write_text('{}')
 block=s.evaluation_phases()
 assert len(block[0][1])==9
 assert {j['name'] for j in block[0][1]}==set(names)
 marker.unlink()
 with pytest.raises(ValueError,match='lacks its accepted audit'):s.evaluation_phases()


def test_baseline_supplement_is_separate_fixed_twelve_unit_schedule(tmp_path,monkeypatch):
 import importlib.util
 import json
 from pathlib import Path
 from experiments.pcrl_task_directed_release_v1 import scheduler as s
 spec=importlib.util.spec_from_file_location('programme_test_helpers',Path(__file__).with_name('test_programme.py'))
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 monkeypatch.setattr(s,'OUT',tmp_path)
 before=s.phases();specs,checked=helper.supplement_stub(tmp_path,monkeypatch)
 order=[r['id'] for r in specs]
 (tmp_path/'BASELINE_SUPPLEMENT_SCHEDULE.json').write_text(json.dumps({'unit_ids':order}))
 block=s.extension_phases('baseline')
 assert len(block)==1 and block[0][0]=='extension_baseline'
 assert [j['name']+f"/anchor_{j['anchor']}" for j in block[0][1]]==order
 assert len(block[0][1])==len(checked)==12 and all(j['command']=='audit' for j in block[0][1])
 assert s.phases()==before
 assert len([j for _,jobs in before for j in jobs if j['command']=='audit'])==180
 (tmp_path/'BASELINE_SUPPLEMENT_SCHEDULE.json').write_text(json.dumps({'unit_ids':order[:-1]}))
 import pytest
 with pytest.raises(ValueError,match='12|twelve|all'):s.extension_phases('baseline')
 (tmp_path/'BASELINE_SUPPLEMENT_SCHEDULE.json').write_text(json.dumps({'unit_ids':list(reversed(order))}))
 with pytest.raises(ValueError,match='fixed order'):s.extension_phases('baseline')


def test_shared_preparation_waits_for_owner_but_not_unrelated_work(tmp_path,monkeypatch):
 from experiments.pcrl_task_directed_release_v1 import scheduler as s
 monkeypatch.setattr(s,'OUT',tmp_path)
 first={**s.job('audit',0,'leace_supervised_mechanism40'),
        'preparation':{'key':'baseline/0/mechanism40','marker':'private/run/anchor_0/baseline_supplement/mechanism40/FITTED.json'}}
 second={**s.job('audit',0,'splince_supervised_mechanism40'),'preparation':first['preparation']}
 other={**s.job('audit',1,'splince_supervised_mechanism40'),
        'preparation':{'key':'baseline/1/mechanism40','marker':'private/run/anchor_1/baseline_supplement/mechanism40/FITTED.json'}}
 assert s.preparation_ready(first,[])
 assert not s.preparation_ready(second,[first])
 assert s.preparation_ready(other,[first])
 assert s.preparation_ready(s.job('audit',0,'J'),[first])
 marker=tmp_path/first['preparation']['marker'];marker.parent.mkdir(parents=True);marker.write_text('{}')
 assert s.preparation_ready(second,[first])


def test_preparation_receipt_keys_match_actual_extension_cache_layouts():
 from experiments.pcrl_task_directed_release_v1 import scheduler as s
 a=s.preparation_dependency('A',{'anchor':2})
 c=s.preparation_dependency('C',{'anchor':2})
 b=s.preparation_dependency('baseline',{'anchor':2,'scope':'union88'})
 assert a=={'key':'A/2','marker':'private/run/anchor_2/branches/action33/TABLES.json'}
 assert c=={'key':'C/2','marker':'private/run/anchor_2/branches/fineC/TABLES.json'}
 assert b=={'key':'baseline/2/union88','marker':'private/run/anchor_2/baseline_supplement/union88/FITTED.json'}

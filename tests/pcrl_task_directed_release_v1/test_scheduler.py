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

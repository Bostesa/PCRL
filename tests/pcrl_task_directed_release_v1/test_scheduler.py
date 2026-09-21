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

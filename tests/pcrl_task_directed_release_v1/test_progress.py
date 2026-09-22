"""Conservative evaluation-exposure inventory; synthetic filesystem only."""
import hashlib
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import progress


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_selection_permit_alone_is_not_an_evaluation_attempt(tmp_path):
    write(tmp_path/'SELECTION.json',{'selection_frozen':True})
    assert not progress.evaluation_exposure(out_root=tmp_path)['evaluation_attempted_or_opened']


@pytest.mark.parametrize('kind',['empty_lock','lock','partial_output','log','scheduler_active','scheduler_failed','quarantine','previous_ledger'])
def test_started_or_aborted_evaluation_is_never_reported_sealed(tmp_path,kind):
    key='evaluation/0/H'
    lock=tmp_path/'private/locks'/(hashlib.sha256(key.encode()).hexdigest()+'.lock')
    if kind=='empty_lock':lock.parent.mkdir(parents=True);lock.touch()
    elif kind=='lock':write(lock,{'key':key,'pid':1,'utc':'synthetic'})
    elif kind=='partial_output':(tmp_path/'private/run/anchor_0/evaluation/H/test').mkdir(parents=True)
    elif kind=='log':p=tmp_path/'private/logs/evaluate__H__anchor_0.log';p.parent.mkdir(parents=True);p.touch()
    elif kind=='scheduler_active':write(tmp_path/'private/SCHEDULER.json',{'active_jobs':[{'command':'evaluate','anchor':0,'name':'H'}]})
    elif kind=='scheduler_failed':write(tmp_path/'private/SCHEDULER.json',{'jobs':[{'command':'evaluate','anchor':0,'name':'H','status':'incident'}]})
    elif kind=='quarantine':(tmp_path/'private/quarantine/evaluation-0-H-synthetic').mkdir(parents=True)
    else:write(tmp_path/'RUN_LEDGER.json',{'evaluation_opened':True})
    result=progress.evaluation_exposure(out_root=tmp_path)
    assert result['evaluation_attempted_or_opened'] and result['evaluation_opened']
    assert result['evidence'] and 'conservative' in result['interpretation']


def test_pending_evaluation_schedule_and_training_locks_do_not_imply_opened(tmp_path):
    write(tmp_path/'private/locks/training.lock',{'key':'audit/0/H'})
    write(tmp_path/'private/SCHEDULER.json',{'jobs':[{'command':'evaluate','status':'dependency_blocked'}]})
    assert not progress.evaluation_exposure(out_root=tmp_path)['evaluation_opened']


def test_snapshot_propagates_partial_evaluation_evidence_without_reading_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(progress,'OUT',tmp_path)
    p=tmp_path/'private/run/anchor_0/evaluation/H/test/private_predictions.npz';p.parent.mkdir(parents=True);p.write_bytes(b'not opened')
    result=progress.snapshot()
    assert result['completed_evaluations']==0 and result['evaluation_opened']
    assert result['evaluation_attempted_or_opened'] and result['comparative_performance_read'] is False
    assert 'attempted or opened' in (tmp_path/'RUN_STATUS.md').read_text()


def test_comparative_access_is_not_reset_after_validation_opens(tmp_path,monkeypatch):
    monkeypatch.setattr(progress,'OUT',tmp_path)
    write(tmp_path/'VALIDATION_OPENED.json',{'scope':'registered validation comparisons'})
    first=progress.snapshot()
    assert first['comparative_performance_read'] and not first['evaluation_opened']
    (tmp_path/'VALIDATION_OPENED.json').unlink()
    second=progress.snapshot()
    assert second['comparative_performance_read'] and not second['evaluation_opened']
    assert 'RUN_LEDGER.json:prior_comparative_access' in second['comparative_access_evidence']


def test_baseline_metadata_registration_alone_is_not_comparative_access(tmp_path):
    write(tmp_path/'BASELINE_SUPPLEMENTS.json',{'metadata_only':True})
    assert not progress.comparative_access(out_root=tmp_path)['comparative_performance_read']
    write(tmp_path/'BRANCH_A_TRIGGER.json',{'registered':False})
    assert progress.comparative_access(out_root=tmp_path)['comparative_performance_read']

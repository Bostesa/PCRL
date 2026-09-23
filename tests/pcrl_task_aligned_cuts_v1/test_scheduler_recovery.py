"""The queue must accept a valid receipt when its state path is relative."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import pytest

from experiments.pcrl_task_aligned_cuts_v1 import scheduler


def test_relative_state_path_accepts_receipt_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protocol = tmp_path / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL.md'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('synthetic frozen protocol\n')
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    script = (
        "import hashlib,json,os,pathlib; "
        "p=pathlib.Path('private/unit');p.mkdir(parents=True,exist_ok=True); "
        "(p/'artifact.txt').write_text('ok'); "
        "h=hashlib.sha256((p/'artifact.txt').read_bytes()).hexdigest(); "
        "(p/'COMPLETE.json').write_text(json.dumps({"
        "'unit_id':os.environ['PCRL_UNIT_ID'],"
        "'queue_sha256':os.environ['PCRL_QUEUE_SHA256'],"
        "'artifacts':{'artifact.txt':h}}))"
    )
    queue = {'schema':'pcrl-task-aligned-queue-v1',
             'frozen_before_comparative_outcomes':True,
             'protocol_sha256':protocol_sha,
             'units':[{'id':'fixture','tier':'A',
                       'argv':[sys.executable,'-c',script],
                       'output_dir':'private/unit'}]}
    queue_path = tmp_path / 'RUN_QUEUE.json'
    queue_path.write_text(json.dumps(queue))
    queue_sha = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    (protocol.parent/'PROTOCOL_LOCK.json').write_text(json.dumps({
        'protocol_sha256':protocol_sha,'run_queue_sha256':queue_sha}))
    state_path = Path('results/pcrl_task_aligned_cuts_v1/private/QUEUE_STATUS.json')
    kwargs = dict(state_path=state_path,workers=1,
                  deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=1),
                  only_ids=('fixture',),minimum_free_memory_bytes=0,
                  minimum_free_disk_bytes=0)
    first = scheduler.run_queue(tmp_path,queue_path,**kwargs)
    assert first['units']['fixture']['status'] == 'complete'
    assert first['attempts'][0]['log'].startswith('results/pcrl_task_aligned_cuts_v1/private/logs/')
    second = scheduler.run_queue(tmp_path,queue_path,**kwargs)
    assert second['units']['fixture']['status'] == 'complete'
    assert len(second['attempts']) == 1


def test_sidecar_accepts_only_hash_verified_distinct_completed_control(tmp_path):
    root = tmp_path.resolve()
    lock = root / 'results/pcrl_task_aligned_cuts_v1/PROTOCOL_LOCK.json'
    lock.parent.mkdir(parents=True)
    lock.write_text('{}')
    lock_sha = scheduler._sha(lock)
    output = root / 'results/pcrl_task_aligned_cuts_v1/private/sidecar_audits/a0_simple_D17'
    output.mkdir(parents=True)
    (output / 'artifact.txt').write_text('frozen fitted route\n')
    sidecar = {
        'schema':'pcrl-control-audit-sidecar-v1', 'outer_pool_opened':False,
        'protocol_lock_sha256':lock_sha,
        'input_index_sha256':'1'*64,
        'shared_H_source_complete_sha256':'2'*64,
        'simple_map_audit_units':{'D17':{
            'status':'registered_unrun', 'release_id':'a0_simple_D17',
            'canonical_release_id':'a0_simple_D17',
            'output_relative_dir':str(output.relative_to(root)),
            'source_file_sha256':'3'*64,'source_array_sha256':'4'*64}},
        'fitted_control_audits':{},
    }
    sidecar_path = root / 'results/pcrl_task_aligned_cuts_v1/CONTROL_SIDECAR.json'
    sidecar_path.write_text(json.dumps(sidecar))
    sidecar_sha = scheduler._sha(sidecar_path)
    receipt = {
        'schema':'pcrl-control-audit-complete-v1', 'unit_id':'a0_simple_D17',
        'release_id':'a0_simple_D17', 'sidecar_sha256':sidecar_sha,
        'source_file_sha256':'3'*64, 'source_array_sha256':'4'*64,
        'protocol_lock_sha256':lock_sha, 'input_index_sha256':'1'*64,
        'H_source_complete_sha256':'2'*64,
        'artifacts':{'artifact.txt':scheduler._sha(output/'artifact.txt')},
    }
    receipt_path = output/'SIDECAR_COMPLETE.json'
    original_receipt_bytes = json.dumps(receipt)
    receipt_path.write_text(original_receipt_bytes)
    kwargs = dict(state_path=Path('results/pcrl_task_aligned_cuts_v1/private/SIDECAR_STATUS.json'),
                  deadline=dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=1),
                  only_ids=('D17',),minimum_free_memory_bytes=0,minimum_free_disk_bytes=0)
    first = scheduler.run_control_sidecar(root, sidecar_path, sidecar_sha, **kwargs)
    assert first['units']['a0_simple_D17']['status'] == 'complete'
    assert first['attempts'] == []
    second = scheduler.run_control_sidecar(root, sidecar_path, sidecar_sha, **kwargs)
    assert second['units']['a0_simple_D17']['status'] == 'complete'
    (output/'unlisted.txt').write_text('not in receipt')
    with pytest.raises(ValueError, match='extra or missing'):
        scheduler.run_control_sidecar(root, sidecar_path, sidecar_sha, **kwargs)
    (output/'unlisted.txt').unlink()
    receipt_path.write_text(json.dumps(receipt, indent=2))
    with pytest.raises(RuntimeError, match='previously accepted control receipt changed'):
        scheduler.run_control_sidecar(root, sidecar_path, sidecar_sha, **kwargs)
    receipt_path.write_text(original_receipt_bytes)
    (output/'artifact.txt').write_text('changed\n')
    with pytest.raises(ValueError, match='artifact mismatch'):
        scheduler.run_control_sidecar(root, sidecar_path, sidecar_sha, **kwargs)

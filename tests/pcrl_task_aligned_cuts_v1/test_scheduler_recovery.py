"""The queue must accept a valid receipt when its state path is relative."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

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

import datetime as dt
import json
import pytest
from experiments.pcrl_task_directed_release_v1 import archive


def proof(path,**changes):
    record={'schema':1,'bucket':'task-bucket','operation':'GetPublicAccessBlock',
            'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'response':{'PublicAccessBlockConfiguration':{key:True for key in
                ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')}}}
    record.update(changes);path.write_text(json.dumps(record));return path,archive.sha(path)


def test_attested_metadata_still_requires_actual_empty_prefix(tmp_path,monkeypatch):
    attestation=proof(tmp_path/'bucket.json');calls=[]
    def api(*args):
        calls.append(args);assert args[1]=='list-objects-v2';return {'KeyCount':0}
    monkeypatch.setattr(archive,'aws_json',api)
    result=archive.validate_destination('task-bucket','pcrl-study/new',bucket_attestation=attestation)
    assert result['prefix_empty_before_upload'] and len(calls)==1
    monkeypatch.setattr(archive,'aws_json',lambda *a:{'KeyCount':1})
    with pytest.raises(ValueError,match='must be empty'):
        archive.validate_destination('task-bucket','pcrl-study/new',bucket_attestation=attestation)


@pytest.mark.parametrize('change',[
    {'bucket':'wrong-bucket'}, {'operation':'unverified'},
    {'checked_utc':(dt.datetime.now(dt.timezone.utc)-dt.timedelta(hours=1)).isoformat()},
    {'response':{'PublicAccessBlockConfiguration':{}}},
])
def test_unscoped_stale_or_public_metadata_rejected(tmp_path,monkeypatch,change):
    attestation=proof(tmp_path/'bucket.json',**change)
    monkeypatch.setattr(archive,'aws_json',lambda *a:pytest.fail('Rejected attestation reached AWS'))
    with pytest.raises(ValueError):archive.validate_destination('task-bucket','new',bucket_attestation=attestation)


def test_altered_attestation_rejected(tmp_path):
    path,expected=proof(tmp_path/'bucket.json');path.write_text('{}')
    with pytest.raises(ValueError,match='hash differs'):
        archive.bucket_block_from_attestation(path,expected,'task-bucket')

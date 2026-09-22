"""Synthetic successor archive patch: all C recovery chains are restorable."""
import hashlib
import json

import pytest

from experiments.pcrl_task_directed_release_v1 import archive


def write(root,name,payload):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(payload if isinstance(payload,bytes) else json.dumps(payload).encode())
    return {'path':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture_records(root):
    stem=f'results/{archive.STUDY}';base=stem+'/private/run/anchor_2'
    name='Trisk_L_0.002_a17_fineC';active=base+'/maps/'+name
    records=[];dependencies={}
    for path in ('PREPARED.json','prepared.joblib','branches/fineC/TABLES.json','branches/fineC/tables.joblib'):
        record=write(root,base+'/'+path,b'synthetic pinned dependency');records.append(record)
        dependencies[path]=record['sha256']
    registration={'anchor':2,'configuration':name,'dependency_files':dependencies}
    record=write(root,active+'/recovery_evidence/REGISTRATION.json',registration);records.append(record)
    records.append(write(root,active+'/ACCEPTED.json',{'anchor':2,'configuration':name,
        'c_failure_retry':{'registration_sha256':record['sha256']}}))
    records.append(write(root,active+'/Q.npz',b'synthetic accepted Q'))
    records.append(write(root,stem+'/private/c_recovery/originals/anchor_2/'+name+'/map/solution.joblib',b'failed original'))
    records.append(write(root,stem+'/private/c_recovery/attempts/anchor_2/'+name+'/CLAIM.json',{'attempt':1}))
    records.append(write(root,base+'/audits/unrelated/models/large_weights.bin',b'unrelated nonrepresentative audit'))
    return stem,base,active,records


def test_all_installed_c_maps_and_dependencies_roundtrip_outside_anchor0(tmp_path):
    root=tmp_path/'root';stem,base,active,records=fixture_records(root)
    prefixes=archive.c_recovery_restore_prefixes(root,records)
    assert stem+'/private/c_recovery' in prefixes and active in prefixes
    assert base+'/prepared.joblib' in prefixes and base+'/branches/fineC/tables.joblib' in prefixes
    packed=tmp_path/'part.tar.zst';archive.pack(root,records,packed)
    restored=tmp_path/'restored'
    result=archive.verify_and_restore(packed,records,restore_root=restored,restore_prefixes=prefixes)
    assert result['verified_files']==len(records) and result['restored_files']==len(records)-1
    assert (restored/(active+'/Q.npz')).read_bytes()==b'synthetic accepted Q'
    assert not (restored/(base+'/audits')).exists()


def test_c_dependency_or_registration_pin_mismatch_blocks_restore_plan(tmp_path):
    root=tmp_path/'root';stem,base,active,records=fixture_records(root)
    with pytest.raises(ValueError,match='missing or changed'):
        archive.c_recovery_restore_prefixes(root,[r for r in records if r['path']!=base+'/prepared.joblib'])
    bad=[dict(r) for r in records]
    next(r for r in bad if r['path']==active+'/recovery_evidence/REGISTRATION.json')['sha256']='0'*64
    with pytest.raises(ValueError,match='pin differs'):archive.c_recovery_restore_prefixes(root,bad)


def test_c_staged_accepted_result_requires_complete_archive_coverage(tmp_path):
    root=tmp_path/'root';stem=f'results/{archive.STUDY}'
    stage=stem+'/private/c_recovery/attempts/anchor_2/Trisk_L_0.002_a17_fineC'
    content=write(root,stage+'/map/solution.joblib',b'accepted retry')
    marker=write(root,stage+'/COMPLETE.json',{'artifact_hashes':{'map/solution.joblib':content['sha256']}})
    with pytest.raises(ValueError,match='missing, excluded, or changed'):
        archive._verify_receipt_coverage(root,[marker])
    archive._verify_receipt_coverage(root,[marker,content])

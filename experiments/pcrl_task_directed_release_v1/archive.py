"""Encrypted task-private archival with complete stream/file verification."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import tarfile
from .config import ROOT,OUT,STUDY
from .run import atomic,sha,now

EXCLUDED_PARTS={'__pycache__','.pytest_cache','venv','staging','archives'}


def safe_relative(name):
    if not isinstance(name,str) or '\\' in name or '\x00' in name:
        raise ValueError('Unsafe archive path')
    p=PurePosixPath(name)
    if (p.is_absolute() or not p.parts or '..' in p.parts or p.as_posix()!=name
            or '2016' in name or 'ss16' in name):
        raise ValueError('Unsafe or sealed archive path')
    return p


def _record_map(records):
    result={}
    for record in records:
        name=record['path'];safe_relative(name)
        if name in result:raise ValueError('Duplicate manifest paths')
        if (not isinstance(record['bytes'],int) or isinstance(record['bytes'],bool) or record['bytes']<0
                or len(record['sha256'])!=64 or any(c not in '0123456789abcdef' for c in record['sha256'])):
            raise ValueError('Invalid artifact size/hash in manifest')
        result[name]=record
    if not result:raise ValueError('Refusing an empty artifact manifest')
    return result


def _contained_file(root,relative):
    """Reject every symbolic-link component before resolving or reading a file."""
    path=root
    for part in safe_relative(relative).parts:
        path=path/part
        if path.is_symlink():raise ValueError('Archive refuses symbolic link paths')
    if not path.resolve().is_relative_to(root):raise ValueError('Archive path escape')
    return path


def _verify_receipt_coverage(root,records):
    """Require all active accepted artifacts, including deployable input maps."""
    indexed=_record_map(records)
    run_root=root/'results'/STUDY/'private/run'
    receipts=sorted(p for name in ('PREPARED.json','ACCEPTED.json','COMPLETE.json')
                    for p in run_root.rglob(name) if 'quarantine' not in p.parts
                    and not set(p.relative_to(root).parts)&EXCLUDED_PARTS)
    manifests=list((root/'results'/STUDY/'private/deployment_inputs').glob('*/MANIFEST.json'))
    for marker in receipts+manifests:
        record=json.loads(marker.read_text())
        hashes=record.get('artifact_hashes') if marker in receipts else record.get('files')
        if not isinstance(hashes,dict) or not hashes:
            raise ValueError('Accepted artifact manifest missing')
        for relative,spec in hashes.items():
            expected=spec.get('sha256') if isinstance(spec,dict) else spec
            path=_contained_file(marker.parent.resolve(),relative)
            name=path.relative_to(root).as_posix()
            if name not in indexed or indexed[name]['sha256']!=expected:
                raise ValueError(f'Accepted artifact missing, excluded, or changed: {name}')


def inventory(root=ROOT):
    root=Path(root).resolve()
    candidates=[]
    for directory in (root/'experiments'/STUDY,root/'tests'/STUDY,root/'results'/STUDY):
        candidates.extend(directory.rglob('*'))
    candidates.append(root/'experiments/acs_transfer_data.py')
    records=[]
    for path in sorted(set(candidates)):
        relative=path.relative_to(root)
        if set(relative.parts)&EXCLUDED_PARTS:continue
        if path.is_symlink():raise ValueError('Archive refuses symbolic links')
        if not path.is_file():continue
        safe_relative(relative.as_posix())
        records.append({'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    if not records:raise ValueError('Refusing an empty study archive')
    _verify_receipt_coverage(root,records)
    return records


def chunk_records(records,target_bytes=1_000_000_000):
    groups=[];part=[];size=0
    for record in records:
        if part and size+record['bytes']>target_bytes:
            groups.append(part);part=[];size=0
        part.append(record);size+=record['bytes']
    if part:groups.append(part)
    return groups


def pack(root,records,path):
    root=Path(root).resolve();path=Path(path);_record_map(records)
    if path.exists():raise FileExistsError('Archive output already exists')
    with path.open('xb') as output:
        compressor=subprocess.Popen(['zstd','-3','-T1','-c'],stdin=subprocess.PIPE,stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin,mode='w|',dereference=True) as tf:
                for record in records:
                    file=_contained_file(root,record['path'])
                    if not file.is_file() or file.stat().st_size!=record['bytes'] or sha(file)!=record['sha256']:
                        raise ValueError('Artifact changed during archive sealing')
                    tf.add(file,arcname=record['path'],recursive=False)
            compressor.stdin.close()
            if compressor.wait()!=0:raise RuntimeError('Archive compression failed')
        except BaseException:
            compressor.kill();compressor.wait();raise
    return {'sha256':sha(path),'bytes':path.stat().st_size,'uncompressed_bytes':sum(r['bytes'] for r in records),'files':len(records)}


def verify_and_restore(path,records,*,restore_root=None,restore_prefixes=()):
    """Independently hash every uncompressed member; restore only declared copies."""
    expected=_record_map(records);seen=set();restored=[]
    for prefix in restore_prefixes:safe_relative(prefix.rstrip('/'))
    if restore_root is not None and Path(restore_root).is_symlink():
        raise ValueError('Restore refuses symbolic root')
    base=Path(restore_root).resolve() if restore_root is not None else None
    if base is not None:base.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['zstd','-d','-c',str(path)],stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as tf:
            for member in tf:
                safe_relative(member.name)
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unexpected archive member/type/duplicate')
                spec=expected[member.name]
                if member.size!=spec['bytes']:raise ValueError('Archive member size mismatch')
                stream=tf.extractfile(member);h=hashlib.sha256();output=None;temp=None;target=None
                try:
                    restore=base is not None and any(member.name==p or member.name.startswith(p.rstrip('/')+'/') for p in restore_prefixes)
                    if restore:
                        target=_contained_file(base,member.name)
                        if target.exists():raise ValueError('Restore refuses escapes and overwrites')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        temp=target.with_name(target.name+'.restore-tmp');output=temp.open('xb')
                    for block in iter(lambda:stream.read(8*1024*1024),b''):
                        h.update(block)
                        if output is not None:output.write(block)
                    if h.hexdigest()!=spec['sha256']:raise ValueError('Restored artifact hash mismatch')
                    if output is not None:
                        output.flush();os.fsync(output.fileno());output.close();output=None
                        # Atomic create-if-absent: another process cannot be overwritten.
                        os.link(temp,target);temp.unlink();restored.append(member.name)
                    seen.add(member.name)
                finally:
                    if output is not None:output.close()
                    if temp is not None:temp.unlink(missing_ok=True)
        proc.stdout.close()
        if proc.wait()!=0:raise RuntimeError('Archive decompression failed')
    except BaseException:
        proc.kill();proc.wait();raise
    if seen!=set(expected):raise ValueError('Archive omitted manifest members')
    return {'verified_files':len(seen),'verified_uncompressed_bytes':sum(expected[n]['bytes'] for n in seen),
            'restored_files':len(restored),'restored_paths':restored}


def aws_json(*args):
    p=subprocess.run(['aws',*args,'--output','json'],capture_output=True,text=True,check=True)
    return json.loads(p.stdout) if p.stdout.strip() else {}


def bucket_block_from_attestation(path,expected_sha256,bucket):
    """Consume a fresh scoped read-only AWS response from the authenticated host.

    The execution role need not receive account/bucket administration rights.
    This is an explicit recorded provenance transfer, not an inferred policy.
    """
    path=Path(path)
    if sha(path)!=expected_sha256:raise ValueError('Bucket metadata attestation hash differs')
    record=json.loads(path.read_text())
    if record.get('schema')!=1 or record.get('bucket')!=bucket or record.get('operation')!='GetPublicAccessBlock':
        raise ValueError('Bucket metadata attestation has the wrong scope')
    checked=dt.datetime.fromisoformat(record['checked_utc'].replace('Z','+00:00'))
    if checked.tzinfo is None:raise ValueError('Bucket attestation timestamp must include timezone')
    age=(dt.datetime.now(dt.timezone.utc)-checked).total_seconds()
    if not -60<=age<=1800:raise ValueError('Bucket metadata attestation is not fresh')
    return record['response'].get('PublicAccessBlockConfiguration',{}),record['checked_utc']


def validate_destination(bucket,prefix,*,bucket_attestation=None):
    """Read-only preflight; privacy is verified, never inferred from encryption."""
    if not bucket or not prefix or prefix.startswith('/') or '..' in PurePosixPath(prefix).parts:
        raise ValueError('An explicit task-private archive destination is required')
    provenance={'bucket_metadata_verification':'direct AWS API'}
    if bucket_attestation is None:
        block=aws_json('s3api','get-public-access-block','--bucket',bucket).get('PublicAccessBlockConfiguration',{})
    else:
        path,expected=bucket_attestation
        block,checked=bucket_block_from_attestation(path,expected,bucket)
        provenance={'bucket_metadata_verification':'fresh authenticated-host AWS response',
                    'bucket_attestation_sha256':expected,'bucket_checked_utc':checked}
    required=('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')
    if any(block.get(k) is not True for k in required):
        raise ValueError('Archive requires all four bucket public-access blocks for private storage')
    contents=aws_json('s3api','list-objects-v2','--bucket',bucket,'--prefix',prefix.rstrip('/')+'/',
                      '--max-keys','1')
    if contents.get('KeyCount',0) or contents.get('Contents'):
        raise ValueError('Archive destination prefix must be empty')
    return {'bucket_public_access_block':{k:block[k] for k in required},'prefix_empty_before_upload':True,**provenance}


def encrypted_put(bucket,key,path):
    """Conditionally create one encrypted object, then verify that exact version."""
    path=Path(path);size=path.stat().st_size
    if size>5_000_000_000:raise ValueError('Archive part exceeds single-PutObject size limit')
    result=aws_json('s3api','put-object','--bucket',bucket,'--key',key,'--body',str(path),
                    '--server-side-encryption','AES256','--if-none-match','*')
    args=['s3api','head-object','--bucket',bucket,'--key',key]
    if result.get('VersionId'):args+=['--version-id',result['VersionId']]
    header=aws_json(*args)
    if (header.get('ServerSideEncryption')!='AES256' or header.get('ContentLength')!=size
            or result.get('VersionId')!=header.get('VersionId')):
        raise ValueError('Archive object encryption/size/version differs')
    return header


def publish(bucket,prefix,directory,restore_root,source_commit,*,bucket_attestation=None):
    """Call only after scientific writes are closed; never deletes source artifacts."""
    selection=json.loads((OUT/'SELECTION.json').read_text())
    if not selection.get('selection_frozen'):raise RuntimeError('Archive closeout requires frozen scientific selection')
    destination=(validate_destination(bucket,prefix) if bucket_attestation is None else
                 validate_destination(bucket,prefix,bucket_attestation=bucket_attestation))
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    records=inventory();manifest={'created_utc':now(),'study':STUDY,'source_commit':source_commit,
          'source_root':str(ROOT),'files':records,'selection_sha256':sha(OUT/'SELECTION.json')}
    atomic(directory/'MANIFEST.private.json',manifest)
    prefixes=[f'experiments/{STUDY}',f'tests/{STUDY}','experiments/acs_transfer_data.py',f'results/{STUDY}/private/inputs',
              f'results/{STUDY}/private/deployment_inputs',f'results/{STUDY}/private/run/anchor_0']
    # The representative restore needs the global selection permit, prospective
    # contrasts, manifests, protocol, and reports in addition to anchor0 weights.
    prefixes+=sorted(r['path'] for r in records if PurePosixPath(r['path']).parent==PurePosixPath('results')/STUDY)
    expected_restore=sum(any(r['path']==p or r['path'].startswith(p+'/') for p in prefixes) for r in records)
    index={'created_utc':now(),'study':STUDY,'source_commit':source_commit,'private_access':True,
           'destination_preflight':destination,'restore_prefixes':prefixes,'expected_restored_files':expected_restore,
           'encryption':'AES256','manifest_sha256':sha(directory/'MANIFEST.private.json'),
           'files':len(records),'uncompressed_bytes':sum(r['bytes'] for r in records),'chunks':[],
           'restore_scope':'all indexed global study documents, tests and executable sources; all owned inputs and deployment input maps; complete indexed anchor0 artifacts; other anchors verified in stream'}
    for i,group in enumerate(chunk_records(records)):
        name=f'part-{i:04d}.tar.zst';path=directory/name;info=pack(ROOT,group,path)
        key=prefix.rstrip('/')+'/'+name
        header=encrypted_put(bucket,key,path)
        readback=directory/(name+'.readback')
        args=['s3api','get-object','--bucket',bucket,'--key',key]
        if header.get('VersionId'):args+=['--version-id',header['VersionId']]
        aws_json(*args,str(readback))
        if sha(readback)!=info['sha256']:raise ValueError('Archive compressed stream readback hash mismatch')
        verification=verify_and_restore(readback,group,restore_root=restore_root,restore_prefixes=prefixes)
        index['chunks'].append({'key':key,'version_id':header.get('VersionId'),**info,
                               'stream_readback_verified':True,'per_file_verified':verification['verified_files'],
                               'restored_files':verification['restored_files']})
        atomic(directory/'INDEX.json',index)
    # Private per-file inventory also receives encrypted read-back verification.
    manifest_key=prefix.rstrip('/')+'/MANIFEST.private.json'
    manifest_header=encrypted_put(bucket,manifest_key,directory/'MANIFEST.private.json')
    args=['s3api','get-object','--bucket',bucket,'--key',manifest_key]
    if manifest_header.get('VersionId'):args+=['--version-id',manifest_header['VersionId']]
    aws_json(*args,str(directory/'MANIFEST.readback.json'))
    if sha(directory/'MANIFEST.readback.json')!=index['manifest_sha256']:raise ValueError('Private inventory readback mismatch')
    if sum(c['restored_files'] for c in index['chunks'])!=expected_restore:
        raise ValueError('Restore did not cover every declared scope member')
    index.update(completed_utc=now(),compressed_bytes=sum(c['bytes'] for c in index['chunks']),
                 all_streams_and_files_verified=True,restore_completed=True,
                 archive_bucket=bucket,archive_prefix=prefix,manifest_key=manifest_key,
                 manifest_version_id=manifest_header.get('VersionId'))
    atomic(OUT/'ARCHIVE_INDEX.json',index)
    atomic(directory/'INDEX.json',index)
    return index


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bucket',required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--directory',required=True);p.add_argument('--restore-root',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--bucket-attestation');p.add_argument('--bucket-attestation-sha256')
    a=p.parse_args()
    if bool(a.bucket_attestation)!=bool(a.bucket_attestation_sha256):p.error('Attestation path and hash must be supplied together')
    proof=(a.bucket_attestation,a.bucket_attestation_sha256) if a.bucket_attestation else None
    print(json.dumps(publish(a.bucket,a.prefix,a.directory,a.restore_root,a.source_commit,bucket_attestation=proof),indent=2))

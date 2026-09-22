"""Encrypted task-private archival with complete stream/file verification."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import tarfile
from .config import ROOT,OUT,STUDY
from .run import atomic,sha,now

EXCLUDED_PARTS={'__pycache__','.pytest_cache','venv','staging','archives'}
MAX_LOCK_METADATA_BYTES=4096


def _task_lock_digest(name):
    """Recognize only this study's canonical hashed unit-lock namespace."""
    match=re.fullmatch(rf'results/{re.escape(STUDY)}/private/locks/([0-9a-f]{{64}})\.lock',name)
    return match.group(1) if match else None


def safe_relative(name):
    if not isinstance(name,str) or '\\' in name or '\x00' in name:
        raise ValueError('Unsafe archive path')
    p=PurePosixPath(name)
    if (p.is_absolute() or not p.parts or '..' in p.parts or p.as_posix()!=name
            or (('2016' in name or 'ss16' in name) and _task_lock_digest(name) is None)):
        raise ValueError('Unsafe or sealed archive path')
    return p


def _validate_lock_metadata(name,payload):
    """Bind a bounded metadata record to its filename; never assert PID liveness.

    Hash names can contain a sealed-year substring by chance. Such a name is
    admissible only in the exact namespace above and only with the same strict
    JSON record that unit_lock writes. All hashed unit locks are checked, even
    when their digest does not contain the otherwise forbidden substring.
    """
    digest=_task_lock_digest(name)
    if digest is None:return
    if not 0<len(payload)<=MAX_LOCK_METADATA_BYTES:
        raise ValueError('Invalid lock metadata size')
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate lock metadata key')
            result[key]=value
        return result
    try:
        record=json.loads(payload.decode('utf-8'),object_pairs_hook=unique_object)
    except (ValueError,UnicodeError) as error:
        raise ValueError('Invalid lock metadata JSON') from error
    if not isinstance(record,dict) or set(record)!={'pid','key','utc'}:
        raise ValueError('Invalid lock metadata schema')
    pid=record['pid'];key=record['key'];utc=record['utc']
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0:
        raise ValueError('Invalid lock metadata PID')
    # These are the unit-key namespaces emitted by the registered study and
    # its read-only closeout driver. Do not admit arbitrary path-like payloads.
    key_pattern=(r'(?:prepare/[012]|(?:map|audit|evaluation)/[012]/[A-Za-z0-9_.-]+'
                 r'|benchmark|programme/(?:selection_freeze|final_evidence)'
                 r'|branch/[AC]/(?:registration|resource_schedule|tables/[012])'
                 r'|baseline_supplement/(?:registration|schedule|[012]/(?:mechanism40|union88)))')
    if (not isinstance(key,str) or re.fullmatch(key_pattern,key) is None
            or '2016' in key or 'ss16' in key or any(part in ('.','..') for part in key.split('/'))
            or hashlib.sha256(key.encode('utf-8')).hexdigest()!=digest):
        raise ValueError('Invalid lock metadata key or filename digest')
    if not isinstance(utc,str) or re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:\+00:00|Z)',utc) is None:
        raise ValueError('Invalid lock metadata UTC timestamp')
    try:dt.datetime.fromisoformat(utc.replace('Z','+00:00'))
    except ValueError as error:raise ValueError('Invalid lock metadata UTC timestamp') from error


def _read_lock_metadata(name,path):
    if _task_lock_digest(name) is None:return None
    with path.open('rb') as handle:payload=handle.read(MAX_LOCK_METADATA_BYTES+1)
    _validate_lock_metadata(name,payload)
    return payload


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
    """Require active, staged and preserved receipts in their actual directories.

    Preserved audit manifests are relative to the preserved audit directory;
    their original model-path strings are not used or silently redirected to
    replacement models. Installed retry pins additionally require the complete
    registered original/staged version chain, even if an entire receipt vanished.
    """
    indexed=_record_map(records)
    out=root/'results'/STUDY;run_root=out/'private/run'
    receipt_roots=(run_root,out/'private/numerical_recovery',out/'private/numerical_originals',
                   out/'private/c_recovery/attempts')
    receipts=sorted({p for base in receipt_roots for name in ('PREPARED.json','ACCEPTED.json','COMPLETE.json')
                    for p in base.rglob(name) if 'quarantine' not in p.parts
                    and not set(p.relative_to(root).parts)&EXCLUDED_PARTS})
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
        if marker.is_relative_to(run_root) and marker.name=='ACCEPTED.json' and record.get('numerical_retry') is not None:
            from .numerical_recovery_run import verify_installed_retry
            verify_installed_retry(record['anchor'],record['configuration'],record,
                                   study_out=out,source_root=root)


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
        payload=_read_lock_metadata(relative.as_posix(),path)
        records.append({'path':relative.as_posix(),
                        'bytes':len(payload) if payload is not None else path.stat().st_size,
                        'sha256':hashlib.sha256(payload).hexdigest() if payload is not None else sha(path)})
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
                    if file.is_file() and _task_lock_digest(record['path']) is not None:
                        payload=_read_lock_metadata(record['path'],file)
                        if len(payload)!=record['bytes'] or hashlib.sha256(payload).hexdigest()!=record['sha256']:
                            raise ValueError('Lock metadata changed during archive sealing')
                        info=tf.gettarinfo(file,arcname=record['path']);info.size=len(payload)
                        # Serialize the validated bytes, not a second mutable file read.
                        tf.addfile(info,io.BytesIO(payload))
                        continue
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
                lock_payload=bytearray() if _task_lock_digest(member.name) is not None else None
                if lock_payload is not None and not 0<member.size<=MAX_LOCK_METADATA_BYTES:
                    raise ValueError('Invalid lock metadata size in archive')
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
                        if lock_payload is not None:lock_payload.extend(block)
                        if output is not None:output.write(block)
                    if h.hexdigest()!=spec['sha256']:raise ValueError('Restored artifact hash mismatch')
                    if lock_payload is not None:_validate_lock_metadata(member.name,bytes(lock_payload))
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


def c_recovery_restore_prefixes(root,records):
    """Restore every installed C retry and its declared replay dependencies.

    This uses hashed receipt metadata only. Nonrepresentative anchors contribute
    just their installed C maps and prepared/fineC inputs, never extra audit fits.
    """
    root=Path(root).resolve();indexed=_record_map(records)
    study=PurePosixPath('results')/STUDY;run_root=study/'private/run'
    prefixes={str(study/'private/c_recovery')}
    for name in indexed:
        path=PurePosixPath(name)
        if (len(path.parts)!=8 or path.parts[:4]!=run_root.parts
                or re.fullmatch(r'anchor_[012]',path.parts[4]) is None
                or path.parts[5]!='maps' or path.name!='ACCEPTED.json'):continue
        marker=json.loads(_contained_file(root,name).read_text())
        pin=marker.get('c_failure_retry')
        if pin is None:continue
        anchor=int(path.parts[4][-1]);configuration=path.parts[6]
        registration_name=str(path.parent/'recovery_evidence/REGISTRATION.json')
        if (marker.get('anchor')!=anchor or marker.get('configuration')!=configuration
                or registration_name not in indexed
                or indexed[registration_name]['sha256']!=pin.get('registration_sha256')):
            raise ValueError('C restore registration/slot pin differs')
        registration=json.loads(_contained_file(root,registration_name).read_text())
        if registration.get('anchor')!=anchor or registration.get('configuration')!=configuration:
            raise ValueError('C restore registration identity differs')
        dependencies=registration.get('dependency_files')
        required={'PREPARED.json','prepared.joblib','branches/fineC/TABLES.json','branches/fineC/tables.joblib'}
        if not isinstance(dependencies,dict) or set(dependencies)!=required:
            raise ValueError('C restore dependency schema differs')
        for relative,digest in dependencies.items():
            safe_relative(relative)
            dependency=str(run_root/f'anchor_{anchor}'/relative)
            if dependency not in indexed or indexed[dependency]['sha256']!=digest:
                raise ValueError('C restore dependency missing or changed')
            prefixes.add(dependency)
        prefixes.add(str(path.parent))
    return sorted(prefixes)


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
              f'results/{STUDY}/private/deployment_inputs',f'results/{STUDY}/private/run/anchor_0',
              f'results/{STUDY}/private/numerical_recovery',f'results/{STUDY}/private/numerical_originals']
    # The representative restore needs the global selection permit, prospective
    # contrasts, manifests, protocol, and reports in addition to anchor0 weights.
    prefixes+=sorted(r['path'] for r in records if PurePosixPath(r['path']).parent==PurePosixPath('results')/STUDY)
    prefixes+=c_recovery_restore_prefixes(ROOT,records)
    expected_restore=sum(any(r['path']==p or r['path'].startswith(p+'/') for p in prefixes) for r in records)
    index={'created_utc':now(),'study':STUDY,'source_commit':source_commit,'private_access':True,
           'destination_preflight':destination,'restore_prefixes':prefixes,'expected_restored_files':expected_restore,
           'encryption':'AES256','manifest_sha256':sha(directory/'MANIFEST.private.json'),
           'files':len(records),'uncompressed_bytes':sum(r['bytes'] for r in records),'chunks':[],
           'restore_scope':'all indexed global study documents, tests and executable sources; all owned inputs and deployment input maps; complete indexed anchor0 artifacts; numerical retry and preserved original versions; every installed C retry plus its registered prepared/fineC replay dependencies; remaining other-anchor files verified in stream'}
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

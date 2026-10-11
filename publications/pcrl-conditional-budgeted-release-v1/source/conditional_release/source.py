"""Admitted role-only sources. Never scan or checksum an all-role archive."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROLES = {'OSF_DEFENSE_FIT': (15434,15428), 'ATTACK_FIT_NEW':(4064,4061), 'INNER_SELECTION':(2235,2234)}
WIRE_FIELDS = {'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'}
REFERENCE_FIELDS = {'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'}
FEATURE_FIELDS = {'row_id','unit','X','feature_names'}
FOLD_SALT = 'IVR1|FIT-crossfold|20261010|'

def admitted_path(path):
    p = Path(path)
    lowered = p.as_posix().lower()
    if any(x in lowered for x in ('head_validation','osf_development_assessment','all_role','old_assessment')):
        raise PermissionError('barred source path')
    if p.name == 'release.npz' and any(x in lowered for x in ('teacher','rel__s')):
        raise PermissionError('historical all-role teacher archive is barred, including hashing')
    if p.is_symlink(): raise PermissionError('symlink source refused')
    return p

def sha_file(path):
    with admitted_path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def pinned(item):
    p=admitted_path(item['path'])
    if sha_file(p)!=item['sha256']: raise PermissionError('admitted source hash differs')
    return p

def read_exact(item, fields):
    path=pinned(item)
    with ZipFile(path) as archive:
        names=[i.filename for i in archive.infolist()]
        if len(names)!=len(fields) or set(names)!={x+'.npy' for x in fields}:
            raise PermissionError('archive fields differ before deserialization')
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name].copy() for name in fields}

def load_manifest(path, expected_sha256=None):
    if expected_sha256 and sha_file(path)!=expected_sha256: raise PermissionError('source manifest differs')
    value=json.loads(admitted_path(path).read_text())
    if value.get('schema')!='CBR1-role-safe-private-source-v1': raise PermissionError('source manifest schema differs')
    return value

def group_folds(unit):
    return np.array([int(hashlib.sha256((FOLD_SALT+str(x)).encode()).hexdigest(),16)%3 for x in unit],dtype=np.int64)

def load_allowed_role(manifest,seed,role,fields=(), *, reporting_seal=None):
    """Load only requested allowed fields, check exact pairing and role counts."""
    if role not in ROLES or seed not in (0,1): raise PermissionError('unadmitted role/seed')
    fields=set(fields)
    allowed={'features','references','income','occupation','sex'}
    if not fields<=allowed: raise PermissionError('unregistered field')
    if role=='ATTACK_FIT_NEW' and fields & {'income','occupation'}: raise PermissionError('attack task labels unadmitted')
    if role=='INNER_SELECTION' and 'sex' in fields:
        if reporting_seal is None: raise PermissionError('INNER SEX embargo')
        raise PermissionError('INNER SEX belongs to final report owner, absent from source manifest')
    record=manifest['seeds'][str(seed)]
    wire=read_exact(record['roles'][role]['release'],WIRE_FIELDS)
    rows,groups=ROLES[role]
    if len(wire['row_id'])!=rows or len(np.unique(wire['unit']))!=groups or len(np.unique(wire['row_id']))!=rows:
        raise PermissionError('source role identity counts differ')
    output={'wire':wire,'fold':group_folds(wire['unit']) if role=='OSF_DEFENSE_FIT' else None}
    for name in fields:
        item=manifest['roles'][role][f'references_s{seed}' if name=='references' else name]
        keys=FEATURE_FIELDS if name=='features' else REFERENCE_FIELDS if name=='references' else {'row_id','unit',name}
        values=read_exact(item,keys)
        if any(not np.array_equal(values[k],wire[k]) for k in ('row_id','unit')):
            raise PermissionError('role pairing differs')
        output[name]=values
    return output

def load_bank(manifest,seed):
    item=manifest['seeds'][str(seed)]['public_bank']
    return json.loads(pinned(item).read_text())

def load_historical_policy(manifest,seed,arm):
    from incumbent_view.r2_package import load_policy
    item=manifest['seeds'][str(seed)]['policies'][arm]
    pinned(item['package']);receipt=pinned(item['receipt'])
    return load_policy(receipt.parent,expected_receipt_sha256=item['receipt']['sha256'])

def donor_law(manifest,seed,c1,c2):
    """Frozen V-J gate with eta=1; reconstruct direct weights, never divide laws."""
    from incumbent_view.r2_policy import Policy
    policy=load_historical_policy(manifest,seed,'V_J')
    return Policy(policy.space,'V_J',policy.weights,1.,1.).law(c1,c2)

def pin(path):
    path=admitted_path(path).absolute()
    return {'path':str(path),'sha256':sha_file(path),'size_bytes':path.stat().st_size}

def write_once(path,value,private=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700 if private else 0o755)
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644),'w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return sha_file(path)

def inventory(predecessor,out,public):
    pred=Path(predecessor)
    source_path=pred/'results/pcrl_incumbent_view_risk_v1/r1/.private/P0_SOURCE_PATHS.json'
    original=json.loads(source_path.read_text())
    result={'schema':'CBR1-role-safe-private-source-v1','source_execution_sha':'320c07c81beba3a82ad4d1f9fe0adc490f65c043',
            'source_publication_sha':'3603b7464fc844681acb766d405502da6f165274','roles':{},'seeds':{}}
    entries={}
    def admit(alias,item):
        actual=pin(item['path'])
        if actual['sha256']!=item['sha256']: raise PermissionError('historical pin mismatch: '+alias)
        entries[alias]={k:v for k,v in actual.items() if k!='path'}
        return actual
    for role,fields in original['roles'].items():
        result['roles'][role]={}
        for field,item in fields.items():
            if role=='INNER_SELECTION' and field=='sex': continue
            result['roles'][role][field]=admit('roles/'+role+'/'+field,item)
    for seed in (0,1):
        old=original['seeds'][str(seed)]
        record={'pins':{},'public_bank':admit(f's{seed}/public_bank',old['public_bank']),'roles':{},'policies':{}}
        for name,item in old['pins'].items():record['pins'][name]=admit(f's{seed}/frozen/'+name,item)
        for role,item in old['roles'].items():
            record['roles'][role]={'release':admit(f's{seed}/{role}/wire',item),
              'receipt':admit(f's{seed}/{role}/receipt',{'path':item['receipt_path'],'sha256':item['receipt_sha256']})}
        for arm in ('V_J','V_S21'):
            base=pred/f'results/pcrl_incumbent_view_risk_v1/r2/.private/units/R2_{arm}_S{seed}_FIT'
            receipt_path=base/'policy/POLICY_RECEIPT.json';receipt=json.loads(receipt_path.read_text())
            record['policies'][arm]={'receipt':admit(f's{seed}/{arm}/receipt',pin(receipt_path)),
              'package':admit(f's{seed}/{arm}/package',{'path':str(base/'policy/policy.npz'),'sha256':receipt['policy_package_sha256']})}
            if arm=='V_J':
                index_path=base/'PRIVATE_ARCHIVE_INDEX.json';index=json.loads(index_path.read_text())
                record['fixed_readers']={'index':admit(f's{seed}/reader_index',pin(index_path)),
                  'arrays':admit(f's{seed}/reader_arrays',{'path':str(base/'READERS.npz'),'sha256':index['reader_file_sha256']})}
        sample=pred/f'results/pcrl_incumbent_view_risk_v1/r1/.private/R1_P0_S{seed}_ADMISSION/OSF_DEFENSE_FIT/sample_input.npz'
        record['sample_input']=admit(f's{seed}/raw32',pin(sample))
        result['seeds'][str(seed)]=record
    manifest_sha=write_once(out,result)
    report={'schema':'CBR1-source-hash-inventory-v1','status':'HASH_ADMITTED_NUMERICAL_PARITY_PENDING','source_execution_sha':result['source_execution_sha'],
            'source_publication_sha':result['source_publication_sha'],'private_manifest_sha256':manifest_sha,'entries':entries,
            'barred_archive_opened':False,'person_arrays_deserialized':False,'inner_sex_file_hashed':False,'role_counts':ROLES,
            'predecessor_full_source_manifest_read':'metadata only; INNER SEX path/pin excluded from all admitted inputs'}
    write_once(public,report,False)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--predecessor',required=True);parser.add_argument('--out',required=True);parser.add_argument('--public',required=True)
    args=parser.parse_args();print(json.dumps(inventory(args.predecessor,args.out,args.public)))

"""Role-only source admission with label embargo before deserialization or hashing."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from zipfile import ZipFile

ROLES={'OSF_DEFENSE_FIT':(15434,15428),'ATTACK_FIT_NEW':(4064,4061),'INNER_SELECTION':(2235,2234)}
FOLD_SALT='IVR1|FIT-crossfold|20261010|'
FEATURE_FIELDS={'row_id','unit','X','feature_names'}
REFERENCE_FIELDS={'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'}


def admitted_path(path):
    p=Path(path); low=p.as_posix().lower()
    if any(x in low for x in ('head_validation','old_assessment','osf_development_assessment','all_role')):
        raise PermissionError('barred source path, including hashing')
    if p.name=='release.npz' and ('teacher' in low or 'rel__s' in low):
        raise PermissionError('all-role source release barred, including hashing')
    if p.name=='sex.npz' and 'inner_selection' in low:
        raise PermissionError('INNER SEX embargo: final endpoint owner requires separate sealed admission')
    if p.is_symlink():raise PermissionError('symlink refused')
    return p


def sha_file(path):
    with admitted_path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pinned(item):
    p=admitted_path(item['path'])
    if sha_file(p)!=item['sha256']:raise PermissionError('source hash differs: '+p.name)
    return p


def read_exact(item,fields):
    import numpy as np
    p=pinned(item)
    with ZipFile(p) as z:
        names=[i.filename for i in z.infolist()]
        if len(names)!=len(fields) or set(names)!={x+'.npy' for x in fields}:raise PermissionError('unexpected source fields')
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in fields}


def load_manifest(path,expected_sha256=None):
    if expected_sha256 and sha_file(path)!=expected_sha256:raise PermissionError('manifest hash differs')
    m=json.loads(admitted_path(path).read_text())
    if m.get('schema') not in ('CBR1-role-safe-private-source-v1','TCR1-role-safe-private-source-v1'):raise PermissionError('source schema differs')
    if 'sex' in m['roles']['INNER_SELECTION']:raise PermissionError('INNER SEX must be absent from prefit manifest')
    return m


def group_folds(unit):
    import numpy as np
    return np.array([int(hashlib.sha256((FOLD_SALT+str(x)).encode()).hexdigest(),16)%3 for x in unit],dtype=np.int64)


def load_role(manifest,seed,role,fields=('features',),*,nomination_seal=None):
    import numpy as np
    if role not in ROLES or seed not in (0,1):raise PermissionError('unadmitted role or seed')
    fields=set(fields)
    if not fields <= {'features','references','income','occupation','sex'}:raise PermissionError('unknown field')
    if role=='INNER_SELECTION':
        if 'sex' in fields:raise PermissionError('INNER SEX embargo')
        if fields & {'income','occupation'}:
            if nomination_seal is None:raise PermissionError('INNER task labels require frozen family nominees')
            seal=json.loads(Path(nomination_seal).read_text()) if isinstance(nomination_seal,(str,Path)) else nomination_seal
            if seal.get('status')!='FROZEN' or not seal.get('nominees'):raise PermissionError('invalid nomination seal')
    if role=='ATTACK_FIT_NEW' and fields & {'income','occupation'}:raise PermissionError('attack task labels unadmitted')
    # Features provide permitted identities without opening unrelated historical wires.
    f=read_exact(manifest['roles'][role]['features'],FEATURE_FIELDS)
    n,g=ROLES[role]
    if len(f['row_id'])!=n or len(np.unique(f['row_id']))!=n or len(np.unique(f['unit']))!=g:raise PermissionError('role counts differ')
    if f['X'].shape!=(n,83) or f['X'].dtype!=np.float32 or not np.isfinite(f['X']).all():raise PermissionError('raw83 source differs')
    names=tuple(map(str,f['feature_names']))
    if len(names)!=83 or len(set(names))!=83 or any(x.lower().split('=')[0] in {'sex','race','income','occupation','fnlwgt','row_id','unit'} for x in names):raise PermissionError('forbidden schema')
    out=dict(f);out['fold']=group_folds(f['unit']) if role=='OSF_DEFENSE_FIT' else None
    for name in fields-{'features'}:
        item=manifest['roles'][role][f'references_s{seed}' if name=='references' else name]
        z=read_exact(item,REFERENCE_FIELDS if name=='references' else {'row_id','unit',name})
        if any(not np.array_equal(z[k],f[k]) for k in ('row_id','unit')):raise PermissionError('person pairing differs')
        if name=='references':out[name]=z
        else:
            a=z[name];k={'income':2,'occupation':6,'sex':2}[name]
            if a.shape!=(n,) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=k):raise PermissionError('label convention differs')
            out[name]=a
    return out


def write_json(path,value,private=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644),'w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return sha_file(p)


def make_manifest(prior_path,out,public):
    old=load_manifest(prior_path)
    m={'schema':'TCR1-role-safe-private-source-v1','roles':old['roles'],'seeds':{},
       'upstream_source_execution_sha':old['source_execution_sha'],'upstream_source_publication_sha':old['source_publication_sha']}
    for s in ('0','1'):
        r=old['seeds'][s]
        m['seeds'][s]={'pins':{k:r['pins'][k] for k in ('model','head0','head1','record','complete')},'public_bank':r['public_bank']}
        cal=Path('/Users/nathansamson/PCRL_eval_cache_private/hcal_v1/run/units')/f'calU__s{s}'
        m['seeds'][s]['ucal']={k:{'path':str(cal/v),'sha256':sha_file(cal/v),'size_bytes':(cal/v).stat().st_size} for k,v in [('complete','COMPLETE.json'),('record','record.json')]}
    entries={}
    def walk(v,prefix=''):
        for k,x in v.items():
            if isinstance(x,dict) and 'path' in x and 'sha256' in x:
                p=pinned(x);entries[prefix+k]={'sha256':x['sha256'],'size_bytes':p.stat().st_size}
            elif isinstance(x,dict):walk(x,prefix+k+'/')
    walk(m)
    h=write_json(out,m,True)
    write_json(public,{'schema':'TCR1-source-hash-inventory-v1','manifest_sha256':h,'entries':entries,'INNER_SEX_opened_or_hashed':False,'barred_arrays_opened':False})
    return h

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prior-manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--public',required=True)
    a=p.parse_args();print(json.dumps({'source_manifest_sha256':make_manifest(a.prior_manifest,a.out,a.public)}))

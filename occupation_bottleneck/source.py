"""Queued role-only source access with a separate canonical INNER reporting seal."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile
from . import admission

ROLES = admission.ROLES
FOLD_SALT = admission.FOLD_SALT
FEATURE_FIELDS = admission.FEATURE_FIELDS
REFERENCE_FIELDS = admission.REFERENCE_FIELDS
LABELS = admission.LABELS
ROOT = Path(__file__).resolve().parents[1]
REPORTING_LOCK = ROOT / 'results/pcrl_occupation_bottleneck_v1/REPORTING_LOCK.json'


def _queued():
    from .scheduler import assert_active_from_env
    assert_active_from_env()


def _reporting_seal(seal):
    """Check exact root lock and its committed bytes before touching INNER labels."""
    if not isinstance(seal, dict) or set(('path', 'sha256')) - seal.keys():
        raise PermissionError('explicit root reporting seal required')
    p = admission.admitted_path(seal['path'])
    if p != REPORTING_LOCK or admission.sha_metadata(p) != seal['sha256']:
        raise PermissionError('canonical reporting lock binding differs')
    env = os.environ.get('PCRL_OB_REPORTING_LOCK_SHA256')
    if env and env != seal['sha256']:
        raise PermissionError('queue reporting lock hash differs')
    lock = admission.read_metadata(p)
    if lock.get('status') != 'ALL_READERS_FROZEN' or lock.get('owner') != 'ROOT':
        raise PermissionError('root reporting lock is not frozen')
    if not lock.get('candidates_frozen') or not lock.get('readers_frozen'):
        raise PermissionError('reporting candidates and readers must be frozen')
    relative = p.relative_to(ROOT).as_posix()
    committed = subprocess.run(['git', 'show', 'HEAD:' + relative], cwd=ROOT,
                               capture_output=True, check=False)
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != seal['sha256']:
        raise PermissionError('reporting lock exact bytes must be committed at HEAD')
    return lock


def admitted_path(path, reporting_seal=None):
    p = Path(os.path.abspath(os.path.expanduser(str(path))))
    if admission.inner_truth_path(p):
        _reporting_seal(reporting_seal)
        # Reuse all ordinary guards without dropping any ancestry or forbidden-role checks.
        low = p.as_posix().lower()
        if any(x in low for x in ('head_validation','old_assessment','osf_development_assessment','all_role','all-role')):
            raise PermissionError('barred source')
        if any(q.is_symlink() for q in (p,*p.parents)):
            raise PermissionError('symlink path refused')
        return p
    return admission.admitted_path(p)


def sha_file(path, reporting_seal=None):
    _queued()
    p = admitted_path(path, reporting_seal)
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned(item, reporting_seal=None):
    p = admitted_path(item['path'], reporting_seal)
    if sha_file(p, reporting_seal) != item['sha256']:
        raise PermissionError('source hash differs: ' + p.name)
    return p


def load_manifest(path, expected_sha256=None):
    return admission.load_manifest(path, expected_sha256)


def read_exact(item, fields, reporting_seal=None):
    import numpy as np
    p = pinned(item, reporting_seal)
    with ZipFile(p) as archive:
        names = [x.filename for x in archive.infolist()]
        if len(names) != len(fields) or set(names) != {x+'.npy' for x in fields}:
            raise PermissionError('unexpected source fields')
    with np.load(p, allow_pickle=False) as archive:
        return {k:archive[k].copy() for k in fields}


def group_folds(unit):
    return admission.group_folds(unit)


def load_role(manifest, seed, role, fields=('features',), reporting_seal=None):
    _queued()
    import numpy as np
    fields = set(fields)
    if seed not in (0,1) or role not in ROLES or not fields <= {'features','references',*LABELS}:
        raise PermissionError('unadmitted role, seed or field')
    label_items = None
    if role == 'INNER_SELECTION' and fields & LABELS:
        _reporting_seal(reporting_seal)
        # Label pins enter only as a separately frozen reporting supplement.
        lock = _reporting_seal(reporting_seal)
        label_items = reporting_seal.get('roles',{}).get(role,{})
        if not (fields & LABELS) <= label_items.keys():
            raise PermissionError('sealed INNER label supplement missing')
        for name in fields & LABELS:
            if lock.get('role_label_sha256',{}).get(role,{}).get(name) != label_items[name]['sha256']:
                raise PermissionError('INNER supplement differs from committed public label hash')
    if role == 'ATTACK_FIT_NEW' and fields & {'income','occupation'}:
        if not (fields & {'income','occupation'}) <= manifest['roles'][role].keys():
            raise PermissionError('ATT task exports absent; use separate FIT diagnostic folds')
    f = read_exact(manifest['roles'][role]['features'], FEATURE_FIELDS)
    n,groups = ROLES[role]
    if len(f['row_id'])!=n or len(np.unique(f['row_id']))!=n or len(np.unique(f['unit']))!=groups:
        raise PermissionError('actual role counts differ')
    if f['X'].shape!=(n,83) or f['X'].dtype!=np.float32 or not np.isfinite(f['X']).all():
        raise PermissionError('raw83 source differs')
    names=tuple(map(str,f['feature_names']))
    forbidden={'sex','race','income','occupation','fnlwgt','row_id','unit','id','hidden_flag'}
    if len(names)!=83 or len(set(names))!=83 or any(x.lower().split('=')[0] in forbidden for x in names):
        raise PermissionError('forbidden input schema')
    out=dict(f);out['fold']=group_folds(f['unit']) if role=='OSF_DEFENSE_FIT' else None
    for name in fields-{'features'}:
        key=f'references_s{seed}' if name=='references' else name
        item=(label_items if role=='INNER_SELECTION' and name in LABELS else manifest['roles'][role])[key]
        if role=='INNER_SELECTION' and name in LABELS:
            expected_path=Path(manifest['roles'][role]['features']['path']).parent/(name+'.npz')
            if Path(item['path']) != expected_path:
                lock = _reporting_seal(reporting_seal)
                admitted = lock.get('inherited_label_custody', {}).get(name, {})
                inherited = ROOT.parent / 'pcrl-task-conditioned-representations-v1' / admitted.get('relative_path','')
                if name != 'sex' or admitted.get('sha256') != item['sha256'] or Path(item['path']) != inherited:
                    raise PermissionError('INNER supplement must use inherited exact role export or pinned SEX custody')
        z=read_exact(item,REFERENCE_FIELDS if name=='references' else {'row_id','unit',name},reporting_seal)
        if any(not np.array_equal(z[k],f[k]) for k in ('row_id','unit')):
            raise PermissionError('person pairing differs')
        if name=='references':out[name]=z
        else:
            a=z[name];k={'income':2,'occupation':6,'sex':2}[name]
            if a.shape!=(n,) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=k):
                raise PermissionError('label convention differs')
            out[name]=a
    return out


def write_json(path,value,private=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644),'w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return admission.sha_metadata(p)

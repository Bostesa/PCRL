"""Checkpointed scientific units. Benchmark output deliberately excludes performance."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback
import uuid
import joblib
import numpy as np
import psutil
from scipy.special import logit
from .config import OUT,ROOT,STUDY,configuration,mechanism_id,digest
from .data import load_anchor,RuntimeInputs,array_hash
from .encoding import fit_encoder
from .baselines import fit_supervised_erasers
from .evaluation import fit_release_audits,evaluate_frozen_audits

FIT_POOLS=('representation_fit','downstream_fit','downstream_validation','attacker_fit','attacker_validation')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,default=clean,indent=2,allow_nan=False)+'\n'
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('x') as handle:
            handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_joblib(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        joblib.dump(value,temp,compress=3)
        with temp.open('rb') as handle:os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def source_fingerprint(stage=None):
    # Scientific dependencies only: orchestration, report and prose edits do
    # not invalidate completed fits. The runner SHA is retained separately.
    dependencies={
        'prepare':('config','data','encoding','audits','baselines','finite','mechanisms'),
        'map':('config','finite','mechanisms'),
        'audit':('config','audits','evaluation','mechanisms'),
        'evaluation':('config','data','encoding','audits','evaluation','mechanisms'),
    }
    names=dependencies[stage] if stage is not None else tuple(sorted(set().union(*dependencies.values())))
    return {n:sha(ROOT/'experiments'/STUDY/(n+'.py')) for n in names}


def _provenance(stage):
    return {'schema':2,'stage':stage,'config_hash':digest(configuration()),
            'source_hashes':source_fingerprint(stage),
            'runner_sha256':sha(ROOT/'experiments'/STUDY/'run.py')}


def _validate_provenance(record,stage):
    for key,value in _provenance(stage).items():
        if key=='runner_sha256':continue
        if record.get(key)!=value:raise ValueError(f'Accepted {stage} {key} changed or missing')


@contextmanager
def unit_lock(key):
    directory=OUT/'private/locks';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(hashlib.sha256(key.encode()).hexdigest()+'.lock')
    with path.open('a+') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError(f'Unit is locked by another worker: {key}') from error
        try:
            handle.seek(0);handle.truncate();handle.write(json.dumps({'pid':os.getpid(),'key':key,'utc':now()}));handle.flush()
            yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def _quarantine(paths,label):
    paths=[Path(p) for p in paths if Path(p).exists()]
    if not paths:return
    target=OUT/'private/quarantine'/f'{label}-{time.time_ns()}-{uuid.uuid4().hex[:8]}'
    target.mkdir(parents=True,exist_ok=False)
    for path in paths:os.replace(path,target/path.name)
    atomic(target/'QUARANTINE.json',{'created_utc':now(),'reason':'unaccepted incomplete attempt',
                                   'original_paths':[str(p) for p in paths]})


def _artifact_hashes(base,paths):
    base=Path(base);files=[]
    for path in paths:
        path=Path(path)
        files.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
    return {str(p.relative_to(base)):sha(p) for p in files}


def _verify_artifacts(base,record):
    hashes=record.get('artifact_hashes')
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Accepted artifact manifest missing')
    base=Path(base).resolve()
    for relative,expected in hashes.items():
        path=(base/relative).resolve()
        if not path.is_relative_to(base) or sha(path)!=expected:
            raise ValueError(f'Accepted artifact hash mismatch: {relative}')


def _read_marker(path,stage,*,anchor=None,name=None):
    record=json.loads(Path(path).read_text());_validate_provenance(record,stage)
    if anchor is not None and record.get('anchor')!=anchor:raise ValueError('Accepted anchor changed')
    if name is not None and record.get('configuration')!=name:raise ValueError('Accepted configuration changed')
    _verify_artifacts(Path(path).parent,record)
    return record


def anchor_dir(anchor):return OUT/'private/run'/f'anchor_{anchor}'


def peak_rss_bytes():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if platform.system()=='Darwin' else n*1024)


def _prepared_receipt(anchor):
    base=anchor_dir(anchor)
    record=_read_marker(base/'PREPARED.json','prepare',anchor=anchor)
    if sha(base/'prepared.joblib')!=record['cache_sha256']:raise ValueError('Prepared cache hash mismatch')
    return record


def prepare(anchor,*,allow_fit=True):
    if (anchor_dir(anchor)/'PREPARED.json').exists():
        return _prepare(anchor,allow_fit=False)
    with unit_lock(f'prepare/{anchor}'):
        return _prepare(anchor,allow_fit=allow_fit)


def _prepare(anchor,*,allow_fit):
    from .mechanisms import estimate_tables
    base=anchor_dir(anchor);base.mkdir(parents=True,exist_ok=True)
    cache=base/'prepared.joblib';marker=base/'PREPARED.json'
    if marker.exists():
        _prepared_receipt(anchor)
        return joblib.load(cache)
    if not allow_fit:raise FileNotFoundError(f'Frozen preparation missing for anchor {anchor}')
    _quarantine([base/'encoder',base/'erasers',cache,*base.glob('prepared.joblib.*.tmp')],f'prepare-{anchor}')
    tick=time.perf_counter()
    ctx=load_anchor(anchor,pools=FIT_POOLS)
    encoder,roles=fit_encoder(ctx,base/'encoder')
    encoded={p:encoder.encode(RuntimeInputs(d['x'],d['ha'])) for p,d in ctx['pools'].items()}
    tables=estimate_tables(ctx,encoder,encoded,roles)
    rf=ctx['pools']['representation_fit'];ix=roles['teacher_fit']
    x=np.column_stack((rf['x'][ix],logit(encoded['representation_fit']['p'][ix])))
    erasers=fit_supervised_erasers(x,{k:rf['labels'][k][ix] for k in ('SEX','RAC1P')},
                  {k:rf['labels'][k][ix] for k in ('same_residence','income_binary','civilian_at_work')},
                  out_dir=base/'erasers')
    result={'ctx':ctx,'encoder':encoder,'encoded':encoded,'tables':tables,'roles':roles,'erasers':erasers}
    atomic_joblib(cache,result)
    split={p:{'people':len(d['ids']),'households':len(np.unique(d['households'])),
               'raw_row_hash':array_hash(d['raw_rows']),
               'person_order_hash':digest(d['ids'].tolist())} for p,d in ctx['pools'].items()}
    split['RF_subroles']={k:{'people':len(ix),'households':len(np.unique(rf['households'][ix])),
                             'raw_row_hash':array_hash(rf['raw_rows'][ix])} for k,ix in roles.items()}
    atomic(marker,{**_provenance('prepare'),'created_utc':now(),'anchor':anchor,'cache_sha256':sha(cache),
                   'artifact_hashes':_artifact_hashes(base,[cache,base/'encoder',base/'erasers']),
                   'seconds':time.perf_counter()-tick,'split':split,'portability':ctx['portability']})
    return result


def map_spec(name,anchor):
    primary=next((m for m in configuration()['maps'] if m['configuration']==name and m['anchor']==anchor),None)
    if primary is not None:return primary
    from .branches import lookup_spec
    extra=lookup_spec(name,anchor)
    if extra is not None:return extra
    from .robustness import lookup_spec as robustness_spec
    return robustness_spec(name,anchor)


def _extra_release(name,anchor):
    if name in {method+'_'+scope for method in ('leace_supervised','splince_supervised')
                for scope in ('mechanism40','union88')}:
        from .baseline_supplement import lookup_release
        return lookup_release(name,anchor)
    if name.endswith('_fineC'):
        from .robustness import lookup_spec
        return lookup_spec(name,anchor)
    from .branches import lookup_release
    return lookup_release(name,anchor)


def _extra_module(spec):
    if spec['branch']=='baseline_supplement':
        from . import baseline_supplement
        return baseline_supplement
    if spec['branch']=='C':
        from . import robustness
        return robustness
    from . import branches
    return branches


def ensure_map(anchor,name,prepared=None):
    if prepared is not None and 'test' in prepared['ctx']['pools']:
        raise ValueError('Cannot fit or ensure a map from an evaluation/test context')
    if (anchor_dir(anchor)/'maps'/name/'ACCEPTED.json').exists():
        return _load_map(anchor,name)
    with unit_lock(f'map/{anchor}/{name}'):
        return _ensure_map(anchor,name,prepared)


def _map_receipt(anchor,name):
    base=anchor_dir(anchor)/'maps'/name
    spec=map_spec(name,anchor)
    if spec is None:raise ValueError(f'Unregistered primary map {name}')
    record=_read_marker(base/'ACCEPTED.json','map',anchor=anchor,name=name)
    if sha(base/'solution.joblib')!=record.get('sha256'):raise ValueError('Accepted channel solution hash changed')
    if record.get('spec_hash')!=digest(spec):raise ValueError('Accepted channel specification changed')
    if spec.get('branch') in ('A','C'):
        if record.get('extra_resource_schedule_sha256')!=_extra_module(spec).require_scheduled(spec):
            raise ValueError('Accepted branch resource schedule changed')
    if record.get('prepared_cache_sha256')!=_prepared_receipt(anchor)['cache_sha256']:
        raise ValueError('Accepted channel preparation changed')
    if record.get('coarse_configuration') is not None:
        coarse=_map_receipt(anchor,record['coarse_configuration'])
        if record.get('coarse_solution_sha256')!=coarse['sha256']:
            raise ValueError('Accepted coarse witness changed')
    return record


def _load_map(anchor,name):
    _map_receipt(anchor,name)
    result=joblib.load(anchor_dir(anchor)/'maps'/name/'solution.joblib')
    if not result.get('feasible') or result.get('Q') is None:
        raise ValueError('Accepted map does not contain a feasible channel')
    return result


def _ensure_map(anchor,name,prepared):
    from .mechanisms import fit_map
    base=anchor_dir(anchor)/'maps'/name;path=base/'solution.joblib'
    if (base/'ACCEPTED.json').exists():
        return _load_map(anchor,name)
    spec=map_spec(name,anchor)
    if spec is None:raise ValueError(f'Unregistered primary map {name}')
    p=prepared if prepared is not None else prepare(anchor)
    preparation=_prepared_receipt(anchor)
    coarse=None;coarse_name=None
    if spec['input']!='T0' and spec.get('branch')!='C':
        coarse_name=mechanism_id('T0',spec['policy'],spec['budget'],spec.get('max_actions',17))
        coarse=ensure_map(anchor,coarse_name,p)
    extra_schedule_sha=None
    if spec.get('branch') in ('A','C'):
        branch_module=_extra_module(spec)
        p=branch_module.prepared_for_spec(anchor,p,spec)
        extra_schedule_sha=branch_module.require_scheduled(spec)
    _quarantine([base],f'map-{anchor}-{name}')
    tick=time.perf_counter()
    result=fit_map(p['ctx'],p['encoder'],p['encoded'],p['tables'],spec,base,coarse_solution=coarse)
    atomic_joblib(path,result)
    if not result.get('feasible') or result.get('Q') is None:
        raise ValueError('Solver did not return an independently feasible channel')
    atomic(base/'ACCEPTED.json',{**_provenance('map'),'created_utc':now(),'configuration':name,
             'anchor':anchor,'sha256':sha(path),'spec_hash':digest(spec),
             'prepared_cache_sha256':preparation['cache_sha256'],
             'extra_resource_schedule_sha256':extra_schedule_sha,
             'coarse_configuration':coarse_name,
             'coarse_solution_sha256':None if coarse_name is None else _map_receipt(anchor,coarse_name)['sha256'],
             'seconds':time.perf_counter()-tick,'artifact_hashes':_artifact_hashes(base,[base])})
    return result


def _release_map_name(name,anchor):
    if map_spec(name,anchor) is not None:return name
    if name=='constant_best':return mechanism_id('T0','U',None)
    if any(name.startswith(c+'_withhold_') or name.startswith(c+'_rr_') for c in ('T0','Ttask','Trisk')):
        if name.endswith('_a33'):
            from .branches import lookup_release
            branch=lookup_release(name,anchor)
            if branch is None:raise ValueError('Unregistered expanded-action control')
            return branch['required_map']
        return mechanism_id(name.split('_')[0],'U',None)
    if name.endswith('_a33'):
        from .branches import lookup_release
        branch=lookup_release(name,anchor)
        if branch is not None:return branch.get('required_map')
    return None


def release_for(name,p,anchor,*,allow_fit=True):
    from .mechanisms import build_release
    if allow_fit and 'test' in p['ctx']['pools']:
        raise ValueError('Evaluation/test releases require allow_fit=False')
    branch=_extra_release(name,anchor)
    if branch is not None:_extra_module(branch).require_scheduled(branch)
    if branch is not None and branch['branch']=='baseline_supplement':
        p=_extra_module(branch).prepared_for_spec(anchor,p,branch,allow_fit=allow_fit)
    required=_release_map_name(name,anchor)
    mechanism=None
    if required is not None:
        mechanism=ensure_map(anchor,required,p) if allow_fit else _load_map(anchor,required)
    spec=branch if branch is not None else map_spec(name,anchor)
    actions=spec.get('max_actions',17) if spec is not None else 17
    canonical=branch.get('canonical_release',name) if branch is not None else name
    return build_release(p['ctx'],p['encoder'],p['encoded'],canonical,mechanism=mechanism,erasers=p['erasers'],actions=actions)


def audit_unit(anchor,name,prepared=None):
    if prepared is not None and 'test' in prepared['ctx']['pools']:
        raise ValueError('Cannot fit audits on an evaluation/test context')
    if (anchor_dir(anchor)/'audits'/name/'COMPLETE.json').exists():
        return _load_audit(anchor,name)
    with unit_lock(f'audit/{anchor}/{name}'):
        return _audit_unit(anchor,name,prepared)


def _audit_receipt(anchor,name):
    base=anchor_dir(anchor)/'audits'/name
    record=_read_marker(base/'COMPLETE.json','audit',anchor=anchor,name=name)
    if sha(base/'registry.joblib')!=record.get('registry_sha256'):raise ValueError('Accepted audit registry hash changed')
    if record.get('prepared_cache_sha256')!=_prepared_receipt(anchor)['cache_sha256']:
        raise ValueError('Accepted audit preparation changed')
    required=_release_map_name(name,anchor)
    if required is not None and record.get('map_solution_sha256')!=_map_receipt(anchor,required)['sha256']:
        raise ValueError('Accepted audit channel changed')
    if name!='H' and record.get('H_registry_sha256')!=_audit_receipt(anchor,'H')['registry_sha256']:
        raise ValueError('Accepted H ancestor changed')
    branch=_extra_release(name,anchor)
    if branch is not None:
        if (record.get('branch_release_hash')!=digest(branch)
                or record.get('extra_resource_schedule_sha256')!=_extra_module(branch).require_scheduled(branch)):
            raise ValueError('Accepted extra release registration/schedule changed')
        if (branch['branch']=='baseline_supplement' and record.get('supplement_fit_sha256')!=
                _extra_module(branch).artifact_receipt(anchor,branch)['receipt_sha256']):
            raise ValueError('Accepted supplement fit dependency changed')
    return record


def _load_audit(anchor,name):
    _audit_receipt(anchor,name)
    base=anchor_dir(anchor)/'audits'/name;path=base/'registry.joblib'
    return {'registry':joblib.load(path),'summary':json.loads((base/'summary.json').read_text()),'registry_path':str(path)}


def _audit_unit(anchor,name,prepared):
    base=anchor_dir(anchor)/'audits'/name;marker=base/'COMPLETE.json'
    if marker.exists():
        return _load_audit(anchor,name)
    p=prepared if prepared is not None else prepare(anchor)
    preparation=_prepared_receipt(anchor)
    h=None if name=='H' else audit_unit(anchor,'H',p)
    release=release_for(name,p,anchor)
    _quarantine([base],f'audit-{anchor}-{name}')
    tick=time.perf_counter()
    result=fit_release_audits(name,p['ctx'],release,base,h_baseline=h,b_baseline=h,
                             global_offsets={pool:e.get('global_offsets',e['actions'][17]) for pool,e in p['encoded'].items()})
    required=_release_map_name(name,anchor)
    branch=_extra_release(name,anchor)
    atomic(marker,{**_provenance('audit'),'created_utc':now(),'anchor':anchor,'configuration':name,
                   'registry_sha256':sha(base/'registry.joblib'),
                   'prepared_cache_sha256':preparation['cache_sha256'],
                   'map_solution_sha256':None if required is None else _map_receipt(anchor,required)['sha256'],
                   'H_registry_sha256':None if name=='H' else _audit_receipt(anchor,'H')['registry_sha256'],
                   'branch_release_hash':None if branch is None else digest(branch),
                   'extra_resource_schedule_sha256':None if branch is None else _extra_module(branch).require_scheduled(branch),
                   'supplement_fit_sha256':None if branch is None or branch['branch']!='baseline_supplement' else
                       _extra_module(branch).artifact_receipt(anchor,branch)['receipt_sha256'],
                   'artifact_hashes':_artifact_hashes(base,[base]),
                   'seconds':time.perf_counter()-tick,'peak_rss_bytes':peak_rss_bytes(),
                   'role_audits':len(result['registry']['roles']),
                   'new_role_fits':sum(not s['reused_B'] for s in result['summary'].values()),
                   'reused_role_audits':sum(s['reused_B'] for s in result['summary'].values())})
    return result


def benchmark():
    with unit_lock('benchmark'):
        if (OUT/'RESOURCE_SCHEDULE.json').exists():raise RuntimeError('Benchmark must precede resource schedule')
        name=mechanism_id('T0','C',.002);tick=time.perf_counter()
        rec={'created_utc':now(),'anchor':0,'configuration':name,'status':'running',
             'threads':1,'scientific_workers':1,'source_hashes':source_fingerprint(),
             'runner_sha256':sha(ROOT/'experiments'/STUDY/'run.py'),
             'config_hash':digest(configuration()),'performance_values_inspected_for_scheduling':False}
        log=OUT/'private/benchmark'/f'{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
        base=anchor_dir(0)
        stages=[('preparation',base/'PREPARED.json'),('H_all_roles',base/'audits/H/COMPLETE.json'),
                ('map',base/'maps'/name/'ACCEPTED.json'),('release_all_roles',base/'audits'/name/'COMPLETE.json')]
        rec['reused_stages']=[stage for stage,path in stages if path.exists()]
        # Warnings or fit diagnostics remain private; the scheduler receives no CE.
        with log.open('x') as handle,redirect_stdout(handle),redirect_stderr(handle):
            try:
                start=time.perf_counter();p=prepare(0);rec['observed_preparation_seconds']=time.perf_counter()-start
                start=time.perf_counter();audit_unit(0,'H',p);rec['observed_H_all_roles_seconds']=time.perf_counter()-start
                start=time.perf_counter();ensure_map(0,name,p);rec['observed_map_seconds']=time.perf_counter()-start
                start=time.perf_counter();audit_unit(0,name,p);rec['observed_release_all_roles_seconds']=time.perf_counter()-start
                # Completed-stage timings remain valid on a restart; cache loading
                # time is never substituted for the cost of fitting a fresh unit.
                for stage,path in stages:rec[stage+'_seconds']=json.loads(path.read_text())['seconds']
                receipt=_audit_receipt(0,name)
                rec.update(status='complete',integrity='accepted complete unit',
                    primary_privacy_maps_fitted_and_audited=1,completed_role_audits=receipt['role_audits'],
                    new_release_role_fits=receipt['new_role_fits'],same_B_role_audits_reused=receipt['reused_role_audits'])
            except Exception as error:
                traceback.print_exc()
                rec.update(status='failed',integrity='complete unit not accepted',error_type=type(error).__name__)
        rec.update(total_observed_seconds=time.perf_counter()-tick,peak_rss_bytes=peak_rss_bytes(),
                   physical_memory=psutil.virtual_memory()._asdict(),cpu_count=psutil.cpu_count())
        atomic(OUT/'BENCHMARK.json',rec)
        return rec


def evaluate_unit(anchor,name):
    with unit_lock(f'evaluation/{anchor}/{name}'):
        return _evaluate_unit(anchor,name)


def _evaluate_unit(anchor,name):
    permit=OUT/'SELECTION.json'
    selection=json.loads(permit.read_text())
    if not selection.get('selection_frozen'):raise PermissionError('Selection is not frozen')
    if 'config_hash' in selection and selection['config_hash']!=digest(configuration()):
        raise ValueError('Frozen selection configuration changed')
    if 'source_hashes' in selection and selection['source_hashes']!=source_fingerprint():
        raise ValueError('Frozen selection scientific sources changed')
    base=anchor_dir(anchor);out=base/'evaluation'/name
    # Every fitting dependency is required and verified before test arrays open.
    audit=_audit_receipt(anchor,name)
    pin=selection.get('frozen_audits',{}).get(name,{}).get(str(anchor))
    if not pin or name not in selection.get('evaluation_configurations',[]):
        raise ValueError('No selection-time audit pin for this evaluation')
    if (pin.get('registry_sha256')!=audit['registry_sha256']
            or pin.get('receipt_sha256')!=sha(base/'audits'/name/'COMPLETE.json')):
        raise ValueError('Accepted audit changed after selection freeze')
    required=_release_map_name(name,anchor)
    if required is not None:_map_receipt(anchor,required)
    if (out/'COMPLETE.json').exists():
        record=_read_marker(out/'COMPLETE.json','evaluation',anchor=anchor,name=name)
        if record['selection_sha256']!=sha(permit) or record['registry_sha256']!=audit['registry_sha256']:
            raise ValueError('Frozen evaluation inputs changed')
        return json.loads((out/'summary.json').read_text())
    p=prepare(anchor,allow_fit=False)
    _quarantine([out],f'evaluation-{anchor}-{name}')
    ctx=load_anchor(anchor,pools=('test',),evaluation_permit=permit)
    encoded={pool:p['encoder'].encode(RuntimeInputs(d['x'],d['ha'])) for pool,d in ctx['pools'].items()}
    eval_p={**p,'ctx':ctx,'encoded':encoded}
    release=release_for(name,eval_p,anchor,allow_fit=False)
    reg=base/'audits'/name/'registry.joblib'
    result=evaluate_frozen_audits(reg,ctx,release,out)
    atomic(out/'COMPLETE.json',{**_provenance('evaluation'),'created_utc':now(),'anchor':anchor,
                               'configuration':name,'selection_sha256':sha(permit),
                               'registry_sha256':sha(reg),'artifact_hashes':_artifact_hashes(out,[out])})
    return result['summary']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','benchmark','audit','map','evaluate'))
    parser.add_argument('--anchor',type=int,default=0);parser.add_argument('--name')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    started=time.perf_counter()
    try:
        if args.command=='prepare':prepare(args.anchor)
        elif args.command=='benchmark':
            result=benchmark();print(json.dumps(result,default=clean,indent=2),flush=True)
            if result['status']!='complete':raise SystemExit(1)
        elif args.command=='audit':audit_unit(args.anchor,args.name)
        elif args.command=='map':ensure_map(args.anchor,args.name)
        else:evaluate_unit(args.anchor,args.name)
        if args.command!='benchmark':print(json.dumps({'status':'complete','command':args.command,
                  'anchor':args.anchor,'name':args.name,'seconds':time.perf_counter()-started}),flush=True)
    except Exception as error:
        path=OUT/'private/incidents'/f'{time.time_ns()}-{args.command}-{args.anchor}.json'
        try:
            from .incidents import preserve_arrays
            arrays=preserve_arrays(error,path.with_suffix('.arrays'))
        except Exception as snapshot_error:
            arrays={'snapshot_error_type':type(snapshot_error).__name__,'message':str(snapshot_error)}
        atomic(path,{'created_utc':now(),'command':vars(args),'type':type(error).__name__,
                     'message':str(error),'traceback':traceback.format_exc(),'private_array_snapshot':arrays})
        raise


if __name__=='__main__':main()

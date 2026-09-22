"""Once-only equivalent entropy retry for a failed, scheduled Branch C map.

Private operational wrapper. No scientific source, registration, evaluation,
or audit is changed. Registration precedes the only solver call. An accepted
retry is installed regardless of scores; rejection leaves the failed original
and blocks its audit. Original-formulation reruns are NOT authorized here.
The ordinary map manifest covers the self-contained recovery evidence as well
as the new map. A separate explicit-root verifier checks its semantic chain.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import joblib
import numpy as np
from experiments.pcrl_task_directed_release_v1 import finite, mechanisms, numerical_recovery as recovery, robustness, run

SOURCES=('config','data','encoding','audits','baselines','finite','mechanisms','evaluation','run','robustness','numerical_recovery')
IDENTITY=('input','policy','budget','configuration','actions','constrained_roles','embedding','aggregation','population','constant_action','cost')


def new_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('x') as handle:
        os.chmod(path,0o600)
        handle.write(json.dumps(value,default=run.clean,sort_keys=True,indent=2,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def owned(path, root):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    try:relative=path.relative_to(root)
    except ValueError as error:raise ValueError('Path outside explicit owned root') from error
    if '..' in relative.parts:raise ValueError('Path traversal refused')
    current=root
    for part in relative.parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlinked recovery path refused')
    return path


def layout(study_out=None,source_root=None):
    if (study_out is None)!=(source_root is None):raise ValueError('Both explicit relocation roots are required')
    root=Path(run.ROOT if source_root is None else source_root).resolve()
    out=owned(run.OUT if study_out is None else study_out,root)
    owned(out/'private',root);owned(out/'private/run',root);owned(out/'private/locks',root)
    return out,root


def tree(base):
    base=Path(base)
    if base.is_symlink() or not base.is_dir():raise ValueError('Artifact tree missing or symlinked')
    result={}
    for path in sorted(base.rglob('*')):
        owned(path,base)
        if path.is_file():result[path.relative_to(base).as_posix()]=run.sha(path)
    if not result:raise ValueError('Empty artifact tree')
    return result


def verify_files(base, files, *, exact=False):
    if not isinstance(files,dict) or not files:raise ValueError('Missing file hash manifest')
    for name,digest in files.items():
        if run.sha(owned(Path(base)/name,base))!=digest:raise ValueError('Pinned artifact changed: '+name)
    if exact and tree(base)!=files:raise ValueError('Pinned artifact file set changed')


def context(out,root,*,controller_source=None):
    code=root/'experiments'/run.STUDY
    return {'config_hash':run.digest(run.configuration()),
        'source_hashes':{name:run.sha(owned(code/(name+'.py'),root)) for name in SOURCES},
        'controller_source_sha256':run.sha(Path(__file__) if controller_source is None else controller_source),
        'resource_schedule_sha256':run.sha(owned(out/'RESOURCE_SCHEDULE.json',root)),
        'robustness_registration_sha256':run.sha(owned(out/'ROBUSTNESS_CONFIGS.json',root)),
        'robustness_resource_schedule_sha256':run.sha(owned(out/'ROBUSTNESS_RESOURCE_SCHEDULE.json',root))}


def spec_for(anchor,name,out,root):
    if type(anchor) is not int or anchor not in (0,1,2):raise ValueError('Invalid anchor')
    record=json.loads(owned(out/'ROBUSTNESS_CONFIGS.json',root).read_text())
    if record.get('maps')!=robustness.robustness_specs() or record.get('branch')!='C' or not record.get('registered'):
        raise ValueError('Registered C map set changed')
    if record.get('registration_payload_hash')!=run.digest({k:v for k,v in record.items() if k!='registration_payload_hash'}):
        raise ValueError('C registration hash mismatch')
    raw=next((s for s in record['maps'] if s['anchor']==anchor and s['configuration']==name),None)
    if raw is None or raw.get('coarse_witness')!='none' or raw.get('max_actions')!=17:
        raise ValueError('Only registered C17 no-witness maps may be retried')
    schedule=json.loads(owned(out/'ROBUSTNESS_RESOURCE_SCHEDULE.json',root).read_text())
    if (schedule.get('schedule_payload_hash')!=run.digest({k:v for k,v in schedule.items() if k!='schedule_payload_hash'})
            or raw['id'] not in schedule.get('unit_ids',[]) or not schedule.get('frozen_before_affected_outcomes')
            or schedule.get('robustness_registration_sha256')!=run.sha(out/'ROBUSTNESS_CONFIGS.json')
            or schedule.get('primary_resource_schedule_sha256')!=run.sha(out/'RESOURCE_SCHEDULE.json')):
        raise ValueError('C slot is not in the frozen schedule')
    expected_sources={n:run.sha(root/'experiments'/run.STUDY/(n+'.py'))
                      for n in ('config','data','encoding','audits','baselines','finite','mechanisms','evaluation')}
    if (record.get('scientific_source_hashes')!=expected_sources
            or record.get('primary_config_hash')!=run.digest(run.configuration())
            or record.get('robustness_source_sha256')!=run.sha(root/'experiments'/run.STUDY/'robustness.py')):
        raise ValueError('Frozen C scientific dependencies changed')
    return {**raw,'robustness_registration_sha256':run.sha(out/'ROBUSTNESS_CONFIGS.json'),
            'robustness_source_sha256':record['robustness_source_sha256']}


def paths(out,anchor,name):
    base=out/'private/run'/f'anchor_{anchor}'
    home=out/'private/c_recovery'
    result={'base':base,'map':base/'maps'/name,'audit':base/'audits'/name,
        'registration':home/'registrations'/f'anchor_{anchor}'/(name+'.json'),
        'stage':home/'attempts'/f'anchor_{anchor}'/name,
        'backup':home/'originals'/f'anchor_{anchor}'/name}
    for path in result.values():owned(path,out)
    return result


def unfrozen(out):
    selection=owned(out/'SELECTION.json',out)
    if selection.exists() and json.loads(selection.read_text()).get('selection_frozen'):
        raise RuntimeError('C recovery cannot modify frozen selection')
    if any((out/'private/run'/f'anchor_{a}'/'evaluation').exists() for a in (0,1,2)):
        raise RuntimeError('C recovery is forbidden after evaluation access')


@contextmanager
def slot_locks(anchor,name):
    out,root=layout()
    with ExitStack() as stack:
        for key in ('programme/selection_freeze',f'audit/{anchor}/{name}',f'map/{anchor}/{name}'):
            owned(out/'private/locks'/(hashlib.sha256(key.encode()).hexdigest()+'.lock'),root)
            stack.enter_context(run.unit_lock(key))
        unfrozen(out)
        yield


def failure(base,spec):
    if (base/'ACCEPTED.json').exists():raise ValueError('Accepted maps must never be retried')
    if set(tree(base))!={'tables.npz','metadata.json','solution.joblib'}:
        raise ValueError('Failed map must have its exact complete original artifact set')
    original=joblib.load(base/'solution.joblib')
    if (original.get('status')!='failed' or original.get('Q') is not None
            or original.get('feasible') is not False or original.get('optimal') is not False
            or original.get('embedding') is not None or original.get('witness') is not None):
        raise ValueError('Only failed no-witness maps may be retried')
    attempts=original.get('attempts',[])
    if len(attempts)!=2 or [a.get('solver') for a in attempts]!=['CLARABEL','SCS'] or any(a.get('accepted') is not False for a in attempts):
        raise ValueError('Original bounded CLARABEL/SCS attempts must both have failed acceptance')
    for key,value in (('input','Trisk'),('policy',spec['policy']),('budget',spec['budget']),('configuration',spec['configuration']),('actions',17)):
        if original.get(key)!=value:raise ValueError('Original failed map identity differs')
    metadata=json.loads((base/'metadata.json').read_text())
    if metadata!=mechanisms._jsonable({k:v for k,v in original.items() if k not in ('Q','cost')}):
        raise ValueError('Original failed metadata and solution disagree')
    return original


def dependencies(p,spec):
    base=p['base'];prepared=json.loads((base/'PREPARED.json').read_text())
    if prepared.get('cache_sha256')!=run.sha(base/'prepared.joblib'):
        raise ValueError('Frozen prepared cache differs')
    fine=base/'branches/fineC';record=json.loads((fine/'TABLES.json').read_text())
    expected={'anchor':spec['anchor'],'branch':'C','prepared_cache_sha256':prepared['cache_sha256'],
        'robustness_registration_sha256':spec['robustness_registration_sha256'],
        'robustness_source_sha256':spec['robustness_source_sha256']}
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen C table dependency differs')
    verify_files(fine,record['artifact_hashes'])
    return {name:run.sha(base/name) for name in ('PREPARED.json','prepared.joblib','branches/fineC/TABLES.json','branches/fineC/tables.joblib')}


def register(anchor,name):
    out,root=layout();spec=spec_for(anchor,name,out,root);p=paths(out,anchor,name)
    with slot_locks(anchor,name):
        if p['registration'].exists() or p['stage'].exists():raise FileExistsError('C retry registration/attempt already exists')
        if p['audit'].exists():raise ValueError('Failed map must not already have audit artifacts')
        before=context(out,root);original=failure(p['map'],spec)
        deps=dependencies(p,spec)
        fine=json.loads((p['base']/'branches/fineC/TABLES.json').read_text())
        if fine.get('robustness_resource_schedule_sha256')!=before['robustness_resource_schedule_sha256']:
            raise ValueError('C tables use a different schedule')
        result={'schema':1,'purpose':'failed_C_map_equivalent_retry','created_utc':run.now(),
            'anchor':anchor,'configuration':name,'spec':spec,'context':before,
            'original_files':tree(p['map']),'dependency_files':deps,
            'original_attempts':original['attempts'],'attempt_limit':1,
            'tolerances':dict(finite.ACCEPTANCE_TOLERANCES),'solver_settings':dict(finite.SOLVER_SETTINGS['CLARABEL']),
            'original_location':str(p['map']),'original_map_relative':p['map'].relative_to(out).as_posix(),
            'replacement_rule':'install every accepted retry; rejected retry remains unresolved; no scores consulted',
            'evaluation_read':False,'audit_fits':0,'new_nominal_configurations':0}
        if context(out,root)!=before:raise ValueError('Sources changed during registration')
        new_json(p['registration'],result)
        return result


def registration(anchor,name,expected_sha,*,out,root):
    p=paths(out,anchor,name);file=owned(p['registration'],root)
    if run.sha(file)!=expected_sha:raise ValueError('C retry registration SHA differs')
    record=json.loads(file.read_text())
    if (record.get('schema')!=1 or record.get('purpose')!='failed_C_map_equivalent_retry'
            or record.get('anchor')!=anchor or record.get('configuration')!=name or record.get('attempt_limit')!=1
            or record.get('spec')!=spec_for(anchor,name,out,root) or record.get('context')!=context(out,root)
            or record.get('tolerances')!=dict(finite.ACCEPTANCE_TOLERANCES)
            or record.get('solver_settings')!=dict(finite.SOLVER_SETTINGS['CLARABEL'])):
        raise ValueError('Registered recovery identity/sources/settings changed')
    verify_files(p['base'],record['dependency_files'])
    return record,p


def inputs(record,p):
    verify_files(p['map'],record['original_files'],exact=True)
    original=failure(p['map'],record['spec'])
    prepared=joblib.load(p['base']/'prepared.joblib')
    if set(prepared['ctx']['pools'])!=set(run.FIT_POOLS):raise ValueError('Recovery requires fitting-only prepared pools')
    table=joblib.load(p['base']/'branches/fineC/tables.joblib')['tables']['Trisk']
    expected={k:table[k] for k in ('cost_U','cost_W','cost','state_mass')}
    expected.update({'joint/'+k:v for k,v in table['roles'].items()})
    expected.update({'support/'+k+'/'+f:v for k,fields in table['support'].items() for f,v in fields.items()})
    with np.load(p['map']/'tables.npz',allow_pickle=False) as arrays:
        if set(arrays.files)!=set(expected) or any(not np.array_equal(arrays[k],v) for k,v in expected.items()):
            raise ValueError('Original and prepared fineC empirical tables differ')
    all_roles={f'{v}/{s}/{w}{suffix}' for v in ('A','AB') for s in ('SEX','RAC1P') for w in ('U','W') for suffix in ('','/fineC')}
    selected={k:v for k,v in table['roles'].items() if record['spec']['policy']=='C' or k.startswith('A/')}
    required={k for k in all_roles if record['spec']['policy']=='C' or k.startswith('A/')}
    if set(table['roles'])!=all_roles or set(selected)!=required or sorted(selected)!=original['constrained_roles']:
        raise ValueError('Exact original/fineC U/PWGTP role set differs')
    encoder=prepared['encoder'];parents=np.asarray(encoder.code.parents('Trisk'))
    if (table['actions']!=17 or not np.array_equal(original['cost'],table['cost'])
            or not np.array_equal(parents,original['support']['parent'])
            or not np.array_equal(table['state_mass'],original['support']['state_mass'])
            or original['aggregation']!=table['aggregation'] or original['population']!=table['population']):
        raise ValueError('Original objective/support/population contract differs')
    return original,table,selected,parents,encoder.dictionaries[17]['zero_action']


def independent(result,table,selected,parents,zero_action,budget):
    if (len(result.get('attempts',[]))!=1 or result['attempts'][0].get('solver')!='CLARABEL'
            or result.get('tolerances')!=dict(finite.ACCEPTANCE_TOLERANCES)
            or result.get('solver_settings')!=dict(finite.SOLVER_SETTINGS['CLARABEL'])):
        raise ValueError('Retry changed solver count/settings/tolerances')
    if not result['accepted']:
        if result.get('Q') is not None or result.get('feasible') or result.get('optimal'):
            raise ValueError('Rejected retry cannot expose a channel')
        return
    if (not result.get('feasible') or result.get('status') not in ('optimal','optimal_inaccurate')
            or result.get('optimal')!=(result.get('status')=='optimal')
            or result.get('solver_status')!=result.get('status')):
        raise ValueError('Invalid accepted retry status')
    q=mechanisms._probabilities(result['Q'],len(parents))
    _,ties,absent,_=finite._support(len(parents),selected,table['state_mass'],parents,table['cost'])
    equations,rhs=finite._support_equations(len(parents),q.shape[1],ties,absent,zero_action)
    matrices={k:finite.privacy_matrix(v) for k,v in selected.items()}
    stacked=np.vstack(list(matrices.values()));scales=np.max(np.abs(stacked),axis=1,initial=0)
    normalized=stacked[scales>0]/scales[scales>0,None]
    residuals=finite._evaluate(q,selected,budget,matrices,normalized,equations,rhs,table['cost'])
    if not residuals['feasible'] or abs(residuals['objective']-result['objective'])>1e-12:
        raise ValueError('Retry failed independent feasibility/objective check')
    result['independent_cmi']={k:finite.cmi(v,q) for k,v in table['roles'].items()}
    result['independent_objective']=residuals['objective']


def execute(anchor,name,*,registration_sha256):
    out,root=layout()
    with slot_locks(anchor,name):
        record,p=registration(anchor,name,registration_sha256,out=out,root=root)
        if p['audit'].exists():raise ValueError('Audit must not precede recovery')
        verify_files(p['map'],record['original_files'],exact=True)
        p['stage'].parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        p['stage'].mkdir(mode=0o700,exist_ok=False)  # Consumed permanently, even on crash.
        new_json(p['stage']/'CLAIM.json',{'anchor':anchor,'configuration':name,'pid':os.getpid(),
            'claimed_utc':run.now(),'registration_sha256':registration_sha256,'attempt':1})
        tick=time.perf_counter()
        try:
            original,table,selected,parents,zero_action=inputs(record,p)
            result=recovery.solve_normalized_retry(table['cost'],selected,record['spec']['budget'],
                parent=parents,state_mass=table['state_mass'],zero_action=zero_action,embedded_q=None)
            independent(result,table,selected,parents,zero_action,record['spec']['budget'])
            result.update({k:copy.deepcopy(original[k]) for k in IDENTITY})
            target=p['stage']/'map';target.mkdir(mode=0o700)
            shutil.copy2(p['map']/'tables.npz',target/'tables.npz')
            if result['Q'] is not None:np.savez_compressed(target/'Q.npz',Q=result['Q'])
            run.atomic_joblib(target/'solution.joblib',result)
            new_json(target/'metadata.json',{k:v for k,v in result.items() if k not in ('Q','cost')})
            registration(anchor,name,registration_sha256,out=out,root=root)
            verify_files(p['map'],record['original_files'],exact=True)
            complete={'schema':1,'anchor':anchor,'configuration':name,'registration_sha256':registration_sha256,
                'created_utc':run.now(),'seconds':time.perf_counter()-tick,'attempts':1,
                **{k:result[k] for k in ('accepted','feasible','optimal','status','solver_status')},
                'artifact_hashes':{'map/'+k:v for k,v in tree(target).items()},
                'originals_unchanged':True,'evaluation_read':False,'audit_fits':0}
            new_json(p['stage']/'COMPLETE.json',complete)
            return complete
        except BaseException as error:
            new_json(p['stage']/'FAILED.json',{'created_utc':run.now(),'error_type':type(error).__name__,
                'message':str(error),'attempt_consumed':True,'originals_replaced':False})
            raise


def completion(p,registration_sha256):
    complete=json.loads((p['stage']/'COMPLETE.json').read_text())
    claim=json.loads((p['stage']/'CLAIM.json').read_text())
    if (complete.get('registration_sha256')!=registration_sha256 or complete.get('attempts')!=1
            or claim.get('registration_sha256')!=registration_sha256 or claim.get('attempt')!=1
            or claim.get('anchor')!=complete.get('anchor') or claim.get('configuration')!=complete.get('configuration')):
        raise ValueError('Retry completion/claim differs')
    verify_files(p['stage'],complete['artifact_hashes'])
    if tree(p['stage']/'map')!={k.removeprefix('map/'):v for k,v in complete['artifact_hashes'].items()}:
        raise ValueError('Staged map file set differs')
    metadata=json.loads((p['stage']/'map/metadata.json').read_text())
    if any(metadata.get(k)!=complete.get(k) for k in ('configuration','accepted','feasible','optimal','status','solver_status')):
        raise ValueError('Staged solver metadata disagrees with completion')
    result=joblib.load(p['stage']/'map/solution.joblib')
    if metadata!=mechanisms._jsonable({k:v for k,v in result.items() if k not in ('Q','cost')}):
        raise ValueError('Staged metadata and solution disagree')
    return complete


def install(anchor,name,*,registration_sha256):
    out,root=layout()
    with slot_locks(anchor,name):
        record,p=registration(anchor,name,registration_sha256,out=out,root=root)
        complete=completion(p,registration_sha256)
        if (complete.get('anchor')!=anchor or complete.get('configuration')!=name
                or run.sha(p['stage']/'map/tables.npz')!=record['original_files']['tables.npz']):
            raise ValueError('Staged retry identity or empirical tables differ')
        if (p['stage']/'INSTALLATION.json').exists():raise FileExistsError('Installation already claimed; inspect its receipt')
        original,table,selected,parents,zero_action=inputs(record,p)
        if p['audit'].exists():raise ValueError('Recovery installation refuses existing audit artifacts')
        result=joblib.load(p['stage']/'map/solution.joblib')
        independent(result,table,selected,parents,zero_action,record['spec']['budget'])
        if not complete['accepted']:
            receipt={'anchor':anchor,'configuration':name,'decision':'unresolved_rejected_retry',
                'retry_receipt_sha256':run.sha(p['stage']/'COMPLETE.json'),'audit_allowed':False,'created_utc':run.now()}
            new_json(p['stage']/'INSTALLATION.json',receipt);return receipt
        if not result['accepted']:raise ValueError('Completion and solution acceptance differ')
        candidate=p['stage']/'installation_candidate'
        if p['backup'].exists() or candidate.exists():raise RuntimeError('Partial installation requires manual review')
        shutil.copytree(p['stage']/'map',candidate)
        shutil.copytree(p['map'],candidate/'original_failure')
        evidence=candidate/'recovery_evidence';evidence.mkdir()
        for source,target in ((p['registration'],'REGISTRATION.json'),(p['stage']/'CLAIM.json','CLAIM.json'),
                (p['stage']/'COMPLETE.json','STAGED_RESULT.json'),(Path(__file__),'controller.py'),
                (Path(recovery.__file__),'normalized_solver.py')):
            shutil.copy2(source,evidence/target)
        pin={'schema':1,'registration_sha256':registration_sha256,
            'retry_receipt_sha256':run.sha(p['stage']/'COMPLETE.json'),
            'original_solution_sha256':record['original_files']['solution.joblib'],
            'controller_source_sha256':record['context']['controller_source_sha256'],
            'solver_source_sha256':record['context']['source_hashes']['numerical_recovery']}
        new_json(candidate/'RECOVERY_PROVENANCE.json',pin)
        accepted={**run._provenance('map'),'created_utc':run.now(),'anchor':anchor,'configuration':name,
            'sha256':run.sha(candidate/'solution.joblib'),'spec_hash':run.digest(record['spec']),
            'prepared_cache_sha256':record['dependency_files']['prepared.joblib'],
            'extra_resource_schedule_sha256':record['context']['robustness_resource_schedule_sha256'],
            'coarse_configuration':None,'coarse_solution_sha256':None,'seconds':complete['seconds'],
            'artifact_hashes':tree(candidate),'c_failure_retry':pin}
        new_json(candidate/'ACCEPTED.json',accepted)
        p['backup'].mkdir(parents=True,exist_ok=False,mode=0o700)
        journal={'anchor':anchor,'configuration':name,'decision':'installation_started','created_utc':run.now(),**pin}
        new_json(p['stage']/'INSTALLATION.json',journal)
        moved=installed=False
        try:
            os.rename(p['map'],p['backup']/'map');moved=True
            os.rename(candidate,p['map']);installed=True
            verify_installed(anchor,name,study_out=out,source_root=root)
            receipt={**journal,'decision':'installed_accepted_retry','installed_utc':run.now(),
                'accepted_sha256':run.sha(p['map']/'ACCEPTED.json'),'audit_allowed':True,
                'preserved_original_files':record['original_files'],'original_location':record['original_location']}
            run.atomic(p['stage']/'INSTALLATION.json',receipt);return receipt
        except BaseException as error:
            if installed:os.rename(p['map'],candidate)
            if moved:os.rename(p['backup']/'map',p['map'])
            run.atomic(p['stage']/'INSTALLATION.json',{**journal,'decision':'rolled_back_requires_manual_review',
                'error_type':type(error).__name__,'message':str(error)})
            raise


def verify_installed(anchor,name,*,study_out=None,source_root=None):
    """No fitting. Explicit relocation roots never consult original artifact paths."""
    out,root=layout(study_out,source_root);p=paths(out,anchor,name)
    active=owned(p['map'],root);receipt=json.loads((active/'ACCEPTED.json').read_text())
    evidence=active/'recovery_evidence'
    record=json.loads((evidence/'REGISTRATION.json').read_text())
    pin=json.loads((active/'RECOVERY_PROVENANCE.json').read_text())
    if (receipt.get('anchor')!=anchor or receipt.get('configuration')!=name
            or record.get('anchor')!=anchor or record.get('configuration')!=name
            or record.get('attempt_limit')!=1 or pin!=receipt.get('c_failure_retry')
            or receipt.get('numerical_retry') is not None):raise ValueError('Installed C recovery identity differs')
    if (pin.get('registration_sha256')!=run.sha(evidence/'REGISTRATION.json')
            or pin.get('retry_receipt_sha256')!=run.sha(evidence/'STAGED_RESULT.json')
            or pin.get('controller_source_sha256')!=run.sha(evidence/'controller.py')
            or pin.get('solver_source_sha256')!=run.sha(evidence/'normalized_solver.py')
            or pin.get('original_solution_sha256')!=record['original_files']['solution.joblib']):
        raise ValueError('Installed C recovery evidence hash differs')
    if (run.sha(p['registration'])!=pin['registration_sha256']
            or run.sha(p['stage']/'COMPLETE.json')!=pin['retry_receipt_sha256']
            or run.sha(p['stage']/'CLAIM.json')!=run.sha(evidence/'CLAIM.json')):
        raise ValueError('Installed evidence differs from original registration/claim/completion')
    if record['context']!=context(out,root,controller_source=evidence/'controller.py'):
        raise ValueError('Installed recovery source/configuration closure differs')
    spec=spec_for(anchor,name,out,root)
    if (record['spec']!=spec or receipt.get('spec_hash')!=run.digest(spec)
            or receipt.get('prepared_cache_sha256')!=record['dependency_files']['prepared.joblib']
            or receipt.get('extra_resource_schedule_sha256')!=record['context']['robustness_resource_schedule_sha256']
            or receipt.get('coarse_configuration') is not None or receipt.get('coarse_solution_sha256') is not None):
        raise ValueError('Installed C native dependency receipt differs')
    if (record.get('tolerances')!=dict(finite.ACCEPTANCE_TOLERANCES)
            or record.get('solver_settings')!=dict(finite.SOLVER_SETTINGS['CLARABEL'])):
        raise ValueError('Installed recovery changed registered tolerances/settings')
    verify_files(p['base'],record['dependency_files'])
    verify_files(active/'original_failure',record['original_files'],exact=True)
    verify_files(p['backup']/'map',record['original_files'],exact=True)
    verify_files(active,receipt['artifact_hashes'])
    if {k:v for k,v in tree(active).items() if k!='ACCEPTED.json'}!=receipt['artifact_hashes']:
        raise ValueError('Installed recovery manifest is incomplete')
    staged=json.loads((evidence/'STAGED_RESULT.json').read_text())
    claim=json.loads((evidence/'CLAIM.json').read_text())
    if (not staged.get('accepted') or not staged.get('feasible') or staged.get('attempts')!=1
            or staged.get('anchor')!=anchor or staged.get('configuration')!=name
            or staged.get('registration_sha256')!=pin['registration_sha256']
            or claim.get('attempt')!=1 or claim.get('registration_sha256')!=pin['registration_sha256']
            or claim.get('anchor')!=anchor or claim.get('configuration')!=name):
        raise ValueError('Installed channel lacks an accepted once-only result')
    expected={k.removeprefix('map/'):v for k,v in staged['artifact_hashes'].items()}
    if expected.get('tables.npz')!=record['original_files']['tables.npz']:
        raise ValueError('Installed empirical table bytes differ from original failure')
    verify_files(active,expected)
    if receipt.get('sha256')!=expected.get('solution.joblib'):raise ValueError('Installed solution differs from retry')
    replay_paths={**p,'map':active/'original_failure'}
    original,table,selected,parents,zero_action=inputs(record,replay_paths)
    result=joblib.load(active/'solution.joblib')
    independent(result,table,selected,parents,zero_action,spec['budget'])
    if not result['accepted']:raise ValueError('Installed solution is not accepted')
    with np.load(active/'Q.npz',allow_pickle=False) as arrays:
        if arrays.files!=['Q'] or not np.array_equal(arrays['Q'],result['Q']):
            raise ValueError('Installed wire Q differs from solution')
    return {'verified':True,'anchor':anchor,'configuration':name,'registration_sha256':pin['registration_sha256'],
        'solution_sha256':receipt['sha256'],'status':result['status'],'optimal':result['optimal'],'evaluation_read':False}


def verify_all(*,study_out=None,source_root=None):
    """Verify every registered recovery, irrespective of inferential nomination."""
    out,root=layout(study_out,source_root);home=owned(out/'private/c_recovery',root)
    registered={}
    for file in sorted((home/'registrations').glob('anchor_*/*.json')):
        record=json.loads(owned(file,root).read_text())
        key=(record['anchor'],record['configuration'])
        if key in registered:raise ValueError('Duplicate C retry registration')
        p=paths(out,*key)
        if file!=p['registration']:raise ValueError('C retry registration path/identity differs')
        installed=json.loads((p['stage']/'INSTALLATION.json').read_text())
        if installed.get('decision')!='installed_accepted_retry':
            raise RuntimeError('Registered C recovery is unresolved; no complete freeze/replay pass')
        accepted_path=owned(p['map']/'ACCEPTED.json',root)
        accepted=json.loads(accepted_path.read_text());pin=accepted.get('c_failure_retry')
        if (installed.get('anchor')!=key[0] or installed.get('configuration')!=key[1]
                or installed.get('audit_allowed') is not True or not isinstance(pin,dict)
                or any(installed.get(k)!=v for k,v in pin.items())
                or installed.get('accepted_sha256')!=run.sha(accepted_path)):
            raise ValueError('Completed C installation receipt differs from accepted map/pins')
        registered[key]=record
    active=set()
    for file in sorted((out/'private/run').glob('anchor_*/maps/*/ACCEPTED.json')):
        record=json.loads(owned(file,root).read_text())
        if record.get('c_failure_retry') is not None:active.add((record['anchor'],record['configuration']))
    if active!=set(registered):raise ValueError('Installed C retry set differs from registered complete recoveries')
    results=[verify_installed(a,n,study_out=out,source_root=root) for a,n in sorted(registered)]
    return {'verified':True,'installed_retries':len(results),'results':results,
            'scope':'all registered installed C recoveries; no nomination filter','evaluation_read':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','execute','install','verify','verify-all'))
    parser.add_argument('--anchor',type=int);parser.add_argument('--name')
    parser.add_argument('--registration-sha256')
    parser.add_argument('--study-out');parser.add_argument('--source-root')
    args=parser.parse_args()
    if args.command!='verify-all' and (args.anchor is None or not args.name):parser.error('Anchor and name required')
    if args.command not in ('verify','verify-all') and (args.study_out or args.source_root):parser.error('Relocation is verification-only')
    if args.command in ('execute','install') and not args.registration_sha256:parser.error('Explicit registration SHA required')
    if args.command=='register':result=register(args.anchor,args.name)
    elif args.command=='verify':result=verify_installed(args.anchor,args.name,study_out=args.study_out,source_root=args.source_root)
    elif args.command=='verify-all':result=verify_all(study_out=args.study_out,source_root=args.source_root)
    else:result=globals()[args.command](args.anchor,args.name,registration_sha256=args.registration_sha256)
    print(json.dumps({k:result[k] for k in ('anchor','configuration','accepted','status','optimal','decision','verified','installed_retries') if k in result}))

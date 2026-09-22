"""Bounded parallel replay of every frozen endpoint configuration and H/J.

Only the public verify_frozen_selection and verify_restored_unit helpers execute
scientific checks. No fits, repairs, fallback paths, role subsets or smaller
wire checks are available. Invoke after scientific writers are closed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import contextmanager,ExitStack
import datetime as dt
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import resource
import signal
import time

import psutil
from .config import STUDY,PRIMARY,SECONDARY,UTILITY,configuration
from .verify import verify_frozen_selection,verify_restored_unit

ROLES=tuple('attack:'+r for r in PRIMARY+SECONDARY)+tuple('utility:'+r for r in UTILITY)
WIRE_REPETITIONS=512
GIB=1024**3


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def _now():return dt.datetime.now(dt.timezone.utc)


def selected_plan(selected,contrasts):
    endpoints=contrasts['endpoints'];ids=[e['id'] for e in endpoints]
    if len(ids)!=len(set(ids)) or len(ids)!=contrasts['family_size']:raise ValueError('Invalid frozen endpoint family')
    names={'H','J'}
    for endpoint in endpoints:
        if not endpoint.get('terms'):raise ValueError('Endpoint has no explicit comparison terms')
        names.update(term['configuration'] for term in endpoint['terms'])
    if any(not isinstance(n,str) or not re.fullmatch('[A-Za-z0-9_.-]+',n) for n in names):raise ValueError('Unsafe configuration name')
    if not names<=set(selected['evaluation_configurations']):raise ValueError('Required endpoint/H/J configuration was not frozen for evaluation')
    endpoint_names={t['configuration'] for e in endpoints for t in e['terms']}
    if endpoint_names!=set(selected['selected_contrast_configurations']):raise ValueError('Frozen selected configuration list differs from endpoint terms')
    return [{'id':f'{name}/anchor_{anchor}','configuration':name,'anchor':anchor}
            for name in sorted(names) for anchor in (0,1,2)]


def worker_limit(requested=8,*,physical_cores=None,available_bytes=None,total_bytes=None):
    if isinstance(requested,bool) or not isinstance(requested,int) or not 1<=requested<=8:
        raise ValueError('Verification maximum must be between one and eight workers')
    physical=physical_cores if physical_cores is not None else psutil.cpu_count(logical=False) or 1
    memory=psutil.virtual_memory() if available_bytes is None or total_bytes is None else None
    available=memory.available if available_bytes is None else available_bytes
    total=memory.total if total_bytes is None else total_bytes
    ram_workers=int((min(available,total)-4*GIB)//(3.5*GIB))
    workers=min(requested,physical,ram_workers)
    if workers<1:raise ValueError('Insufficient RAM after the four-GiB reserve for one verification worker')
    return {'workers':workers,'requested_maximum':requested,'physical_cores':physical,
            'available_memory_bytes':available,'total_memory_bytes':total,
            'per_worker_memory_budget_bytes':int(3.5*GIB),'memory_reserve_bytes':4*GIB,
            'process_start_method':'spawn','threads_per_worker':1}


def coverage_summary(result,*,configuration,anchor,selection_sha256):
    if (result.get('passed') is not True or result.get('configuration')!=configuration or result.get('anchor')!=anchor
            or result.get('selection_sha256')!=selection_sha256 or result.get('original_fallback_allowed') is not False
            or result.get('original_selection_pin_checked') is not True):raise ValueError('Unit failed its frozen identity/pin/no-fallback contract')
    unit=result['unit'];validation=unit['validation'];parity=unit['ancestor_parity'];evaluation=unit['evaluation']
    if any(r.get('passed') is not True for r in (unit,validation,parity,evaluation,result['preparation'],result['mathematical_replay'])):
        raise ValueError('A required scientific replay stage did not pass')
    expected=set(ROLES)
    if set(validation['roles'])!=expected or set(parity['roles'])!=expected or set(evaluation['pools'])!={'test'}:
        raise ValueError('Replay did not cover all sixteen registered roles')
    tests=evaluation['pools']['test']
    if set(tests)!=expected:raise ValueError('Evaluation did not cover all sixteen registered roles')
    for role,record in tests.items():
        wire=record.get('wire_check',{})
        if wire.get('passed') is not True or wire.get('repetitions')!=WIRE_REPETITIONS:
            raise ValueError('Every role requires its full 512-draw independent wire check')
    return {'validation_roles':len(expected),'ancestor_roles':len(expected),'evaluation_roles':len(expected),
            'wire_checked_roles':len(expected),'wire_repetitions_per_role':WIRE_REPETITIONS,
            'mathematical_replay_passed':True,'preparation_replay_passed':True,'original_fallback_allowed':False}


def _initialize_worker():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    global _WORKER_THREAD_LIMITS
    _WORKER_THREAD_LIMITS=threadpool_limits(limits=1)


def _verify_worker(job):
    tick=time.monotonic();directory=Path(job['out_dir'])
    report={'id':job['id'],'configuration':job['configuration'],'anchor':job['anchor'],
            'passed':False,'worker_pid':os.getpid()}
    try:
        result=verify_restored_unit(job['configuration'],artifact_root=job['artifact_root'],
            original_root=job['original_root'],out_dir=directory,anchor=job['anchor'],include_evaluation=True,
            expected_selection_sha256=job['selection_sha256'],wire_repetitions=WIRE_REPETITIONS)
        coverage=coverage_summary(result,configuration=job['configuration'],anchor=job['anchor'],selection_sha256=job['selection_sha256'])
        path=directory/'restore_verification.json'
        if not path.is_file() or json.loads(path.read_text())!=result:raise ValueError('Persisted unit report does not match helper return')
        report.update(passed=True,status='passed',**coverage,unit_report_sha256=_sha(path),
                      unit_report_relative_path=str(path.relative_to(Path(job['run_dir']))))
    except Exception as error:
        report.update(status='failed',failure_type=type(error).__name__)
        _write(directory/'parallel_incident.json',{'type':type(error).__name__,'message':str(error)})
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(seconds=time.monotonic()-tick,peak_rss_bytes=int(peak if platform.system()=='Darwin' else peak*1024))
    _write(directory/'worker.json',report)
    return report


def _stop_pool(pool):
    # Python3.13 lacks terminate_workers(). These Process objects belong only
    # to this executor; never discover or kill unrelated operating-system PIDs.
    processes=list((getattr(pool,'_processes',None) or {}).values())
    pool.shutdown(wait=False,cancel_futures=True)
    for process in processes:
        if process.is_alive():process.terminate()
    end=time.monotonic()+1.
    for process in processes:process.join(timeout=max(0.,end-time.monotonic()))
    for process in processes:
        if process.is_alive():process.kill()
    for process in processes:process.join(timeout=.1)


def _execute(plan,*,workers,deadline,worker=_verify_worker,on_result=None):
    """Bounded spawn queue. Test injection is internal; production uses the helper worker."""
    pending=iter(plan);active={};results={};expired=False
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_initialize_worker)
    try:
        def submit():
            job=next(pending,None)
            if job is not None:active[pool.submit(worker,job)]=job
        for _ in range(workers):submit()
        while active:
            remaining=(deadline-_now()).total_seconds()
            if remaining<=0:expired=True;break
            done,_=wait(active,timeout=min(.5,remaining),return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try:
                    result=future.result()
                    if (not isinstance(result,dict) or result.get('id')!=job['id']
                            or result.get('configuration')!=job['configuration'] or result.get('anchor')!=job['anchor']
                            or not isinstance(result.get('passed'),bool)):raise ValueError('Missing or misidentified worker result')
                except Exception as error:
                    result={k:job[k] for k in ('id','configuration','anchor')}
                    result.update(passed=False,status='failed',failure_type=type(error).__name__)
                results[job['id']]=result
                if on_result:on_result(result)
                if _now()<deadline:submit()
        if expired:_stop_pool(pool)
        else:pool.shutdown(wait=True,cancel_futures=False)
    except BaseException:
        _stop_pool(pool);raise
    for job in plan:
        if job['id'] not in results:
            results[job['id']]={k:job[k] for k in ('id','configuration','anchor')}
            results[job['id']].update(passed=False,status='deadline_incomplete')
    return {key:results[key] for key in sorted(results)}


def _owned(root,path):
    path=Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Required owned source/provenance file missing or escaped')
    return path


def _confirm_unit_reports(directory,rows,selection_sha256):
    for row in rows.values():
        if row.get('passed') is not True:continue
        try:
            relative=f"units/anchor_{row['anchor']}/{row['configuration']}/restore_verification.json"
            if row.get('unit_report_relative_path')!=relative:raise ValueError('Worker report path changed')
            path=_owned(directory,directory/relative)
            if _sha(path)!=row.get('unit_report_sha256'):raise ValueError('Unit report hash changed before parent readback')
            coverage=coverage_summary(json.loads(path.read_text()),configuration=row['configuration'],anchor=row['anchor'],
                                      selection_sha256=selection_sha256)
            if any(row.get(k)!=v for k,v in coverage.items()):raise ValueError('Worker coverage differs from persisted report')
            row['parent_report_readback_verified']=True
        except Exception as error:
            row.update(passed=False,status='report_readback_failed',failure_type=type(error).__name__,parent_report_readback_verified=False)
            _write(directory/'parent_incidents'/f"{row['configuration']}__anchor_{row['anchor']}.json",{'type':type(error).__name__,'message':str(error)})


def source_snapshot(root):
    """Byte closure of all running/owned study modules and frozen registrations."""
    root=Path(root).resolve();source=Path(__file__).parent;owned=root/'experiments'/STUDY
    names={p.name for p in source.glob('*.py')}
    if names!={p.name for p in owned.glob('*.py')}:raise ValueError('Running and owned study module sets differ')
    sources={}
    for name in sorted(names):
        current=_sha(source/name);saved=_sha(_owned(root,owned/name))
        if current!=saved:raise ValueError('Running and owned study source bytes differ')
        sources[name]=current
    out=root/'results'/STUDY
    required=('SELECTION.json','CONTRASTS.json','VALIDATION_GRID.json','RESOURCE_SCHEDULE.json')
    optional=('EXTRA_CONFIGS.json','EXTRA_RESOURCE_SCHEDULE.json','ROBUSTNESS_CONFIGS.json','ROBUSTNESS_RESOURCE_SCHEDULE.json',
              'BASELINE_SUPPLEMENTS.json','BASELINE_SUPPLEMENT_SCHEDULE.json')
    documents={name:_sha(_owned(root,out/name)) for name in required}
    for name in optional:
        if (out/name).exists():documents[name]=_sha(_owned(root,out/name))
    return {'study_sources':sources,'frozen_documents':documents}


@contextmanager
def _closed_writers(out):
    with ExitStack() as stack:
        for path in [out/'private/scheduler.lock',*sorted((out/'private/locks').glob('*.lock'))]:
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.open('a'))
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ValueError('Scientific writers are active') from error
        state=out/'private/SCHEDULER.json'
        if state.exists() and json.loads(state.read_text()).get('status')=='running':raise ValueError('Scientific scheduler still reports running')
        yield


def _deadline(value):
    absolute=dt.datetime.fromisoformat(configuration()['absolute_deadline_utc'].replace('Z','+00:00'))
    cutoff=absolute.replace(hour=8,minute=19,second=50,microsecond=0)
    wanted=cutoff if value is None else dt.datetime.fromisoformat(value.replace('Z','+00:00')) if isinstance(value,str) else value
    if wanted.tzinfo is None or wanted>cutoff or wanted<=_now():raise ValueError('Verification needs a future UTC deadline before the08:20 closeout limit')
    return wanted.astimezone(dt.timezone.utc)


@contextmanager
def _alarm(deadline):
    old_handler=signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL)[0]:raise ValueError('An existing process deadline prevents safe verification alarm ownership')
    def stop(*_):raise TimeoutError('Verification deadline reached')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,max(.001,(deadline-_now()).total_seconds()))
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)


def verify_parallel(*,artifact_root,original_root,out_dir,writers_closed=False,max_workers=8,
                    deadline_utc=None,public_report=None):
    if writers_closed is not True:raise ValueError('Caller must explicitly confirm scientific writers closed')
    root=Path(artifact_root).resolve();original=Path(original_root).resolve();out=root/'results'/STUDY
    directory=Path(out_dir).resolve()
    if not root.is_dir() or not directory.is_relative_to(out/'private') or directory.exists():
        raise ValueError('Verification requires explicit artifact root and fresh task-private output directory')
    public_target=None if public_report is None else Path(public_report).resolve()
    if public_target is not None and (not public_target.is_relative_to(out) or public_target.exists()):
        raise ValueError('Public report needs a new owned study path')
    deadline=_deadline(deadline_utc);resources=worker_limit(max_workers);directory.mkdir(parents=True)
    report={'schema':1,'passed':False,'all_passed':False,'started_utc':_now().isoformat(),'resources':resources,
            'deadline_utc':deadline.isoformat(),'scientific_writers_closed':True,'original_fallback_allowed':False,
            'evaluation_requested':True,'role_count_per_unit':len(ROLES),'wire_repetitions_per_role':WIRE_REPETITIONS,
            'units':{},'planned_units':0,'scope':'all frozen endpoint configurations plus H/J, every anchor, all16roles and wire512; global selection rebuilt once'}
    tick=time.monotonic();plan=[];stage='global_selection';before=None
    try:
        with _closed_writers(out),_alarm(deadline):
            before=source_snapshot(root);report['source_selection_closure_before']=before
            selection_tick=time.monotonic()
            rebuilt=verify_frozen_selection(artifact_root=root,original_root=original)
            report['global_selection_seconds']=time.monotonic()-selection_tick
            if rebuilt['report'].get('passed') is not True:raise ValueError('Global selection reconstruction did not pass')
            report['global_selection']=rebuilt['report'];_write(directory/'global_selection.json',rebuilt['report'])
            report['global_selection_report_sha256']=_sha(directory/'global_selection.json')
            plan=selected_plan(rebuilt['selection'],rebuilt['contrasts'])
            report['planned_units']=len(plan);report['configurations']=sorted({j['configuration'] for j in plan})
            pin=rebuilt['report']['selection_sha256']
            if pin!=before['frozen_documents']['SELECTION.json']:raise ValueError('Selection changed during global reconstruction')
            jobs=[{**j,'artifact_root':str(root),'original_root':str(original),'run_dir':str(directory),
                   'out_dir':str(directory/'units'/f"anchor_{j['anchor']}"/j['configuration']),'selection_sha256':pin} for j in plan]
            _write(directory/'PLAN.json',{'plan':plan,'selection_sha256':pin,'resources':resources})
            stage='parallel_units'
            def completed(row):
                report['units'][row['id']]=row
                _write(directory/'completed'/f"{row['configuration']}__anchor_{row['anchor']}.json",row)
            parallel_tick=time.monotonic()
            report['units']=_execute(jobs,workers=resources['workers'],deadline=deadline,on_result=completed)
            report['parallel_units_wall_seconds']=time.monotonic()-parallel_tick
            _confirm_unit_reports(directory,report['units'],pin)
            stage='final_closure';after=source_snapshot(root);report['source_selection_closure_after']=after
            if after!=before:raise ValueError('Source/selection closure changed during parallel replay')
            report['source_selection_closure_unchanged']=True
            report['passed']=report['all_passed']=bool(len(report['units'])==len(plan) and all(r['passed'] for r in report['units'].values()))
    except Exception as error:
        report.update(failed_stage=stage,failure_type=type(error).__name__)
        _write(directory/'incident.json',{'stage':stage,'type':type(error).__name__,'message':str(error)})
    for job in plan:
        if job['id'] not in report['units']:
            report['units'][job['id']]={**job,'passed':False,'status':'deadline_incomplete' if _now()>=deadline else 'incomplete'}
    report['units']={key:report['units'][key] for key in sorted(report['units'])}
    passed=[r for r in report['units'].values() if r.get('passed') is True]
    report.update(completed_utc=_now().isoformat(),seconds=time.monotonic()-tick,
                  passed_units=len(passed),failed_or_incomplete_units=len(report['units'])-len(passed),
                  validation_role_checks=sum(r.get('validation_roles',0) for r in passed),
                  ancestor_role_checks=sum(r.get('ancestor_roles',0) for r in passed),
                  evaluation_role_checks=sum(r.get('evaluation_roles',0) for r in passed),
                  wire_role_checks=sum(r.get('wire_checked_roles',0) for r in passed),
                  wire_draw_repetitions_total=sum(r.get('wire_checked_roles',0)*r.get('wire_repetitions_per_role',0) for r in passed),
                  observed_worker_pids=sorted({r['worker_pid'] for r in report['units'].values() if 'worker_pid' in r}),
                  maximum_worker_peak_rss_bytes=max((r.get('peak_rss_bytes',0) for r in report['units'].values()),default=0),
                  summed_worker_seconds=math.fsum(r.get('seconds',0.) for r in report['units'].values()))
    _write(directory/'PARALLEL_VERIFICATION.json',report)
    if public_target is not None:_write(public_target,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact-root',required=True,type=Path)
    parser.add_argument('--original-root',required=True,type=Path);parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--writers-closed',action='store_true');parser.add_argument('--max-workers',type=int,default=8)
    parser.add_argument('--deadline-utc');parser.add_argument('--public-report',type=Path)
    args=parser.parse_args(argv)
    result=verify_parallel(artifact_root=args.artifact_root,original_root=args.original_root,out_dir=args.out_dir,
        writers_closed=args.writers_closed,max_workers=args.max_workers,deadline_utc=args.deadline_utc,public_report=args.public_report)
    print(json.dumps({'passed':result['passed'],'planned_units':result['planned_units'],'passed_units':result['passed_units'],
                      'failed_or_incomplete_units':result['failed_or_incomplete_units'],'seconds':result['seconds']},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

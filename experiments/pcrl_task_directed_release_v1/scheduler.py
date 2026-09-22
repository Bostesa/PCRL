"""Detached, checkpointed task-owned science scheduler with a frozen resource plan."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import psutil
from .config import OUT,ROOT,configuration,mechanism_id,release_ledger,digest
from .run import atomic,now,sha


def job(command,anchor,name=None):
    return {'id':f'{command}/{name or "encoder"}/anchor_{anchor}',
            'command':command,'anchor':anchor,'name':name}


def phases():
    cfg=configuration();anchors=cfg['seeds'];codes=('Ttask','Trisk')
    order=[]
    order.append(('prepare',[job('prepare',a) for a in anchors]))
    order.append(('H',[job('audit',a,'H') for a in anchors]))
    order.append(('mandatory_T0',[job('audit',a,mechanism_id('T0',p,b))
                  for b in (.002,.0005,.01) for p in ('L','C') for a in anchors]))
    order.append(('unconstrained_T0',[job('audit',a,mechanism_id('T0','U',None)) for a in anchors]))
    order.append(('unconstrained_refinements',[job('audit',a,mechanism_id(c,'U',None)) for c in codes for a in anchors]))
    order.append(('matched_operating_controls',[job('audit',a,name) for name in ('J','continuous_task') for a in anchors]))
    order.append(('paired_middle',[job('audit',a,mechanism_id(c,p,.002))
                  for p in ('L','C') for a in anchors for c in codes]))
    order.append(('paired_other',[job('audit',a,mechanism_id(c,p,b))
                  for b in (.0005,.01) for p in ('L','C') for a in anchors for c in codes]))
    order.append(('zero_T0',[job('audit',a,mechanism_id('T0',p,0.)) for p in ('L','C') for a in anchors]))
    order.append(('zero_refinements',[job('audit',a,mechanism_id(c,p,0.))
                  for p in ('L','C') for a in anchors for c in codes]))
    priority=['leace_supervised','splince_supervised',
              'leace_A0','splince_A0','optnet16_L1','optnet16_L2','optnet16_C1',
              'constant_best','independent_token','T0_code','Ttask_code','Trisk_code']
    controls=[r['configuration'] for r in release_ledger()['records'] if r['kind']=='control' and r['anchor']==0 and r['configuration'] not in ('H','J','continuous_task')]
    priority+=sorted(set(controls)-set(priority))
    order.append(('matched_and_historical_controls',[job('audit',a,name) for name in priority for a in anchors]))
    return order


def marker_for(j):
    base=OUT/'private/run'/f"anchor_{j['anchor']}"
    if j['command']=='prepare':return base/'PREPARED.json'
    if j['command']=='evaluate':return base/'evaluation'/j['name']/'COMPLETE.json'
    return base/'audits'/j['name']/'COMPLETE.json'


def extension_phases(branch):
    """Use only an explicit prospectively frozen extension schedule."""
    if branch=='A':
        from . import branches
        registry=branches._registration()
        schedule=json.loads((OUT/'EXTRA_RESOURCE_SCHEDULE.json').read_text())
        specs={r['id']:r for r in registry['maps']+registry['controls']}
        lookup,require=branches.lookup_release,branches.require_scheduled
    elif branch=='C':
        from . import robustness
        registry=robustness._registration()
        schedule=json.loads((OUT/'ROBUSTNESS_RESOURCE_SCHEDULE.json').read_text())
        specs={r['id']:r for r in registry['maps']}
        lookup,require=robustness.lookup_spec,robustness.require_scheduled
    elif branch=='baseline':
        from . import baseline_supplement
        registry=baseline_supplement.registration()
        schedule=json.loads((OUT/'BASELINE_SUPPLEMENT_SCHEDULE.json').read_text())
        controls=registry['controls'];expected=baseline_supplement.specs()
        if len(controls)!=12 or controls!=expected or schedule.get('unit_ids')!=[r['id'] for r in expected]:
            raise ValueError('Baseline supplement schedule requires all twelve registered controls in fixed order')
        specs={r['id']:r for r in controls}
        lookup,require=baseline_supplement.lookup_release,baseline_supplement.require_scheduled
    else:raise ValueError('Unknown registered extension')
    jobs=[]
    for ident in schedule['unit_ids']:
        spec=specs[ident]
        require(lookup(spec['configuration'],spec['anchor']))
        jobs.append(job('audit',spec['anchor'],spec['configuration']))
    return [(f'extension_{branch}',jobs)]


def evaluation_phases():
    selection=json.loads((OUT/'SELECTION.json').read_text())
    if not selection.get('selection_frozen'):raise RuntimeError('No frozen evaluation permit')
    jobs=[job('evaluate',anchor,name) for name in selection['evaluation_configurations'] for anchor in (0,1,2)]
    for j in jobs:
        if not (OUT/'private/run'/f"anchor_{j['anchor']}"/'audits'/j['name']/'COMPLETE.json').exists():
            raise ValueError('Frozen evaluation configuration lacks its accepted audit')
    return [('evaluation',jobs)]



def dependencies_ready(j):
    """Read accepted markers only; scheduling never reads performance values."""
    base=OUT/'private/run'/f"anchor_{j['anchor']}"
    if marker_for(j).exists():return True
    if j['command']=='prepare':return True
    if not (base/'PREPARED.json').exists():return False
    if j['command']=='evaluate':return (base/'audits'/j['name']/'COMPLETE.json').exists()
    if j['name']=='H':return True
    if not (base/'audits/H/COMPLETE.json').exists():return False
    mandatory=dict(phases())['mandatory_T0']
    if j.get('phase')!='mandatory_T0' and any(not marker_for(x).exists() for x in mandatory):return False
    from .run import map_spec,_release_map_name
    spec=map_spec(j['name'],j['anchor'])
    if spec is not None and spec['input']!='T0' and not spec.get('conditioning_family'):
        coarse=mechanism_id('T0',spec['policy'],spec['budget'],spec.get('max_actions',17))
        if not (base/'maps'/coarse/'ACCEPTED.json').exists():return False
    dependency=_release_map_name(j['name'],j['anchor'])
    if dependency is not None and dependency!=j['name']:
        if not (base/'maps'/dependency/'ACCEPTED.json').exists():return False
    return True

def execute(j,deadline):
    logdir=OUT/'private/logs';logdir.mkdir(parents=True,exist_ok=True)
    log=logdir/(j['id'].replace('/','__')+'.log')
    args=[sys.executable,'-m','experiments.pcrl_task_directed_release_v1.run',j['command'],'--anchor',str(j['anchor'])]
    if j['name']:args+=['--name',j['name']]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    start=time.perf_counter()
    with log.open('a') as f:
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            if dt.datetime.now(dt.timezone.utc)>=deadline:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                return {**j,'status':'resource_incomplete','returncode':proc.returncode,'seconds':time.perf_counter()-start,'log':str(log.relative_to(ROOT))}
            time.sleep(1)
    marker=marker_for(j)
    return {**j,'status':'complete' if proc.returncode==0 and marker.exists() else 'incident',
            'returncode':proc.returncode,'seconds':time.perf_counter()-start,
            'log':str(log.relative_to(ROOT)),'marker_sha256':sha(marker) if marker.exists() else None}


def run(selected_phases=None,evaluation=False,extension=None):
    plan_path=OUT/'RESOURCE_SCHEDULE.json'
    plan=json.loads(plan_path.read_text())
    if not plan.get('frozen_before_comparative_outcomes'):raise RuntimeError('Resource plan is not frozen')
    if plan['config_hash']!=digest(configuration()):raise RuntimeError('Configuration differs from resource schedule')
    workers=int(plan['scientific_workers'])
    physical=psutil.cpu_count(logical=False) or psutil.cpu_count()
    if workers<1 or workers>physical:raise ValueError('Frozen concurrency exceeds physical CPU cores')
    if workers*configuration()['physical_memory_worker_ceiling_gib']*1024**3 > psutil.virtual_memory().total-4*1024**3:
        raise ValueError('Frozen concurrency exceeds physical memory reserve')
    deadline=dt.datetime.fromisoformat(plan['science_deadline_utc'].replace('Z','+00:00'))
    path=OUT/'private/SCHEDULER.json';lock=OUT/'private/scheduler.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if evaluation and extension:raise ValueError('Choose evaluation or fitting extension')
        if evaluation:work=evaluation_phases()
        elif extension:work=extension_phases(extension)
        else:work=phases()
        if selected_phases:work=[p for p in work if p[0] in selected_phases]
        state={'started_utc':now(),'pid':os.getpid(),'resource_schedule_sha256':sha(plan_path),
               'workers':workers,'phases':[p[0] for p in work],'jobs':[],'status':'running'}
        atomic(path,state)
        pending=[{**j,'phase':phase} for phase,jobs in work for j in jobs];active={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending or active:
                current=dt.datetime.now(dt.timezone.utc)
                if current>=deadline:
                    state['jobs'] += [{**j,'status':'resource_incomplete'} for j in pending];pending=[]
                memory_wait=False
                while pending and len(active)<workers and current<deadline:
                    mem=psutil.virtual_memory().available;free=shutil.disk_usage(ROOT).free
                    if free<plan.get('minimum_free_disk_bytes',8*1024**3):
                        state['status']='disk_safety_stop';atomic(path,state)
                        raise RuntimeError('Task volume below registered free disk reserve')
                    if mem<plan.get('minimum_free_memory_bytes',4*1024**3):memory_wait=True;break
                    ready=next((i for i,j in enumerate(pending) if dependencies_ready(j)),None)
                    if ready is None:break
                    j=pending.pop(ready);active[pool.submit(execute,j,deadline)]=j
                if not active:
                    if pending and memory_wait:
                        state['resource_wait_utc']=now();atomic(path,state);time.sleep(5);continue
                    if pending:
                        state['jobs'] += [{**j,'status':'dependency_blocked'} for j in pending];pending=[]
                    break
                done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    info=active.pop(future)
                    try:r=future.result()
                    except Exception as e:r={**info,'status':'scheduler_error','message':str(e)}
                    state['jobs'].append({**r,'phase':info['phase'],'completed_utc':now()})
                state['active_jobs']=list(active.values());state['remaining_jobs']=len(pending);state['updated_utc']=now();atomic(path,state)
        # Failed units are retained for bounded diagnosis, never retried for favorable scores.
        state['status']='finished_with_incidents' if any(j['status']!='complete' for j in state['jobs']) else 'complete'
        state['finished_utc']=now();state['active_jobs']=[];atomic(path,state)
        print(json.dumps({'status':state['status'],'complete':sum(j['status']=='complete' for j in state['jobs']),
                          'incidents_or_incomplete':sum(j['status']!='complete' for j in state['jobs'])}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phases',nargs='*');ap.add_argument('--evaluate',action='store_true')
    ap.add_argument('--extension',choices=('A','C','baseline'))
    a=ap.parse_args();run(a.phases,a.evaluate,a.extension)

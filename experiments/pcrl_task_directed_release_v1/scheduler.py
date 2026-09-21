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
    order.append(('paired_middle',[job('audit',a,mechanism_id(c,p,.002))
                  for p in ('L','C') for a in anchors for c in codes]))
    order.append(('paired_other',[job('audit',a,mechanism_id(c,p,b))
                  for b in (.0005,.01) for p in ('L','C') for a in anchors for c in codes]))
    order.append(('zero_T0',[job('audit',a,mechanism_id('T0',p,0.)) for p in ('L','C') for a in anchors]))
    order.append(('zero_refinements',[job('audit',a,mechanism_id(c,p,0.))
                  for p in ('L','C') for a in anchors for c in codes]))
    priority=['J','continuous_task','leace_supervised','splince_supervised',
              'leace_A0','splince_A0','optnet16_L1','optnet16_L2','optnet16_C1',
              'constant_best','independent_token','T0_code','Ttask_code','Trisk_code']
    controls=[r['configuration'] for r in release_ledger()['records'] if r['kind']=='control' and r['anchor']==0 and r['configuration']!='H']
    priority+=sorted(set(controls)-set(priority))
    order.append(('matched_and_historical_controls',[job('audit',a,name) for name in priority for a in anchors]))
    return order


def marker_for(j):
    base=OUT/'private/run'/f"anchor_{j['anchor']}"
    if j['command']=='prepare':return base/'PREPARED.json'
    if j['command']=='evaluate':return base/'evaluation'/j['name']/'COMPLETE.json'
    return base/'audits'/j['name']/'COMPLETE.json'


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


def run(selected_phases=None,evaluation=False):
    plan_path=OUT/'RESOURCE_SCHEDULE.json'
    plan=json.loads(plan_path.read_text())
    if not plan.get('frozen_before_comparative_outcomes'):raise RuntimeError('Resource plan is not frozen')
    if plan['config_hash']!=digest(configuration()):raise RuntimeError('Configuration differs from resource schedule')
    workers=int(plan['scientific_workers'])
    if workers<1 or workers>8:raise ValueError('Invalid frozen concurrency')
    deadline=dt.datetime.fromisoformat(plan['science_deadline_utc'].replace('Z','+00:00'))
    path=OUT/'private/SCHEDULER.json';lock=OUT/'private/scheduler.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if evaluation:
            selection=json.loads((OUT/'SELECTION.json').read_text())
            if not selection.get('selection_frozen'):raise RuntimeError('No frozen evaluation permit')
            registered=release_ledger()['records']
            work=[('evaluation',[job('evaluate',r['anchor'],r['configuration']) for r in registered
                 if (OUT/'private/run'/f"anchor_{r['anchor']}"/'audits'/r['configuration']/'COMPLETE.json').exists()])]
        else:work=phases()
        if selected_phases:work=[p for p in work if p[0] in selected_phases]
        state={'started_utc':now(),'pid':os.getpid(),'resource_schedule_sha256':sha(plan_path),
               'workers':workers,'phases':[p[0] for p in work],'jobs':[],'status':'running'}
        atomic(path,state)
        for phase,jobs in work:
            state['current_phase']=phase;atomic(path,state)
            pending=list(jobs);active={}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while pending or active:
                    current=dt.datetime.now(dt.timezone.utc)
                    if current>=deadline:
                        state['jobs'] += [{**j,'status':'resource_incomplete','phase':phase} for j in pending];pending=[]
                    while pending and len(active)<workers and current<deadline:
                        mem=psutil.virtual_memory().available;free=shutil.disk_usage(ROOT).free
                        if free<plan.get('minimum_free_disk_bytes',8*1024**3):
                            state['status']='disk_safety_stop';atomic(path,state)
                            raise RuntimeError('Task volume below registered free disk reserve')
                        if mem<plan.get('minimum_free_memory_bytes',4*1024**3):break
                        j=pending.pop(0);active[pool.submit(execute,j,deadline)]={**j,'phase':phase}
                    if not active:
                        if pending:
                            state['resource_wait_utc']=now();atomic(path,state);time.sleep(5)
                            continue
                        break
                    done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        info=active.pop(future)
                        try:r=future.result()
                        except Exception as e:r={**info,'status':'scheduler_error','message':str(e)}
                        state['jobs'].append({**r,'phase':phase,'completed_utc':now()})
                    state['active_jobs']=list(active.values());state['remaining_in_phase']=len(pending);state['updated_utc']=now();atomic(path,state)
            # Failed units are retained for bounded diagnosis; independent phases continue.
        state['status']='finished_with_incidents' if any(j['status']!='complete' for j in state['jobs']) else 'complete'
        state['finished_utc']=now();state['active_jobs']=[];atomic(path,state)
        print(json.dumps({'status':state['status'],'complete':sum(j['status']=='complete' for j in state['jobs']),
                          'incidents_or_incomplete':sum(j['status']!='complete' for j in state['jobs'])}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phases',nargs='*');ap.add_argument('--evaluate',action='store_true')
    a=ap.parse_args();run(a.phases,a.evaluate)

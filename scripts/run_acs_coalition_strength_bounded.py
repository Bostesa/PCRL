"""Conservative whole-process scientific accounting for the fixed strength study."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_coalition_strength_v1');ap.add_argument('--phase',choices=['train','evaluate'],required=True);ap.add_argument('--max-units',type=int,default=36);a=ap.parse_args();out=a.out.resolve()
    cfg=json.loads((out/'config.json').read_text());directory=out/'processes';directory.mkdir(exist_ok=True)
    old=[json.loads(p.read_text()) for p in sorted(directory.glob('process_*.json'))]
    if any(p['status']=='running' for p in old):raise RuntimeError('Resolve the recorded running process before recovery')
    utc=datetime.datetime.now(datetime.timezone.utc);science=sum(p['elapsed_seconds'] for p in old);elapsed=(utc-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
    remaining=min(cfg['maximum_scientific_seconds']-science,cfg['maximum_total_work_seconds']-cfg['reserved_reporting_seconds']-elapsed)
    if remaining<=0:raise RuntimeError('Scientific or reserved reporting deadline reached')
    cmd=[sys.executable,'-u','-m','experiments.run_acs_coalition_strength','--out',str(out),'--phase',a.phase,'--max-units',str(a.max_units)]
    path=directory/f'process_{len(old):03d}.json';log=path.with_suffix('.log');env=os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    record={'started_utc':utc.isoformat(),'status':'running','phase':a.phase,'command':cmd,'timeout_seconds':remaining,'previous_scientific_seconds':science,'accounting':'entire process including load,fit,freeze,selection,score and operational failure','one_numerical_thread':True,'stdout_local_path':str(log)}
    with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    tick=time.perf_counter()
    try:
        with log.open('x') as f:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        record.update(status='complete' if p.returncode==0 else 'failed',returncode=p.returncode)
    except subprocess.TimeoutExpired:record.update(status='deadline_terminated',returncode=124,reason='Scientific/total deadline; preserve completed and partial artifacts')
    finally:
        record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-tick);path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if record['returncode']:raise SystemExit(record['returncode'])
if __name__=='__main__':main()

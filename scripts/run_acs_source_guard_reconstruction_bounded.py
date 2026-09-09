"""Charge the exact T diagnostic reconstruction conservatively to study ceilings."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_source_guard_v1');a=ap.parse_args();out=a.out.resolve();cfg=json.loads((out/'config.json').read_text());directory=out/'processes'
    records=[json.loads(p.read_text()) for p in sorted(directory.glob('process_*.json'))]
    assert all(r['status']!='running' for r in records),'Only one scientific process at a time'
    now=datetime.datetime.now(datetime.timezone.utc);spent=sum(r['elapsed_seconds'] for r in records);work=(now-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
    remaining=min(180.,cfg['maximum_scientific_seconds']-spent,cfg['maximum_total_work_seconds']-cfg['reserved_reporting_seconds']-work)
    if remaining<=0:raise RuntimeError('No remaining bounded reconstruction time')
    path=directory/f'process_{len(records):03d}.json';log=path.with_suffix('.log');cmd=[sys.executable,'-u','scripts/reconstruct_acs_source_guard_T.py','--out',str(out)]
    env=os.environ.copy();env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    record={'started_utc':now.isoformat(),'status':'running','phase':'T_diagnostic_reconstruction','command':cmd,'timeout_seconds':remaining,'previous_scientific_seconds':spent,'accounting':'Full disposable existing-trajectory reconstruction process; conservatively charged to scientific ceiling; no new result model or observer/auditor fit','one_numerical_thread':True,'stdout_local_path':str(log)}
    with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    tick=time.perf_counter()
    try:
        with log.open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        record.update(status='complete' if r.returncode==0 else 'failed',returncode=r.returncode)
    except subprocess.TimeoutExpired:record.update(status='deadline_terminated',returncode=124,reason='Bounded diagnostic reconstruction stopped; original fits remain immutable')
    finally:
        record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-tick);path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2));raise SystemExit(record['returncode'])
if __name__=='__main__':main()

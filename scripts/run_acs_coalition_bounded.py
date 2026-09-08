"""Local finite subprocess wrapper with conservative scientific/total deadlines."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260908_acs_coalition_v1');ap.add_argument('--max-seeds',type=int,default=1);a=ap.parse_args();out=a.out.resolve()
    cfg=json.loads((out/'config.json').read_text());processes=out/'processes';processes.mkdir(exist_ok=True)
    old=[json.loads(p.read_text()) for p in sorted(processes.glob('process_*.json'))]
    if any(p.get('status')=='running' for p in old):raise RuntimeError('Unresolved running process record; inspect actual process before recovery')
    now=datetime.datetime.now(datetime.timezone.utc);science=sum(p['elapsed_seconds'] for p in old)
    elapsed=(now-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds()
    remaining=min(cfg['maximum_scientific_seconds']-science,cfg['maximum_total_work_seconds']-cfg['reserved_reporting_seconds']-elapsed)
    if remaining<=0:raise RuntimeError('No scientific time remains after the reporting reserve')
    command=[sys.executable,'-u','-m','experiments.run_acs_coalition','--out',str(out),'--run','--max-seeds',str(a.max_seeds)]
    index=len(old);path=processes/f'process_{index:03d}.json';log=processes/f'process_{index:03d}.log'
    record={'started_utc':now.isoformat(),'status':'running','command':command,'timeout_seconds':remaining,'previous_scientific_seconds':science,'scientific_accounting':'entire subprocess wall time, including loading, validation and scoring','stdout_local_path':str(log)}
    with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    started=time.perf_counter()
    try:
        with log.open('x') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        record.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode)
    except subprocess.TimeoutExpired:
        record.update(status='deadline_terminated',returncode=124,reason='Scientific or total-work deadline; preserve partial seed and report')
    finally:
        record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if record['returncode']:raise SystemExit(record['returncode'])
if __name__=='__main__':main()

"""Bounded sequential post-fit verification, conservatively charged to science."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');p.add_argument('--step',choices=['tests','replay','summary','enrich','comparisons'],required=True);a=p.parse_args();out=a.out.resolve();cfg=json.loads((out/'config.json').read_text())
    assert json.loads((out/'EXECUTED_MATRIX.json').read_text())['complete']
    processes=out/'processes';previous=[json.loads(p.read_text()) for p in processes.glob('process_*.json')];assert not any(r['status']=='running' for r in previous)
    utc=datetime.datetime.now(datetime.timezone.utc);science=sum(r['elapsed_seconds'] for r in previous);elapsed=(utc-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds();remaining=min(5400-science,14400-1800-elapsed);assert remaining>0
    if a.step=='tests':cmd=[sys.executable,'-m','pytest','-q','tests/test_acs_fixed_predictions.py','tests/test_acs_coalition_training.py::test_equal_purpose_source_loss_masks_and_prior_entropy','tests/test_acs_coalition_training.py::test_exact_targeted_extra_local_sum_algebra_and_role_seeds']
    else:
        module={'replay':'replay_acs_fixed_predictions','summary':'summarize_acs_fixed_predictions','enrich':'enrich_acs_fixed_predictions','comparisons':'verify_acs_fixed_predictions_comparisons'}[a.step];cmd=[sys.executable,'-u','-m','scripts.'+module,'--out',str(out)]
    i=len(previous);path=processes/f'process_{i:03d}.json';log=path.with_suffix('.log');record={'phase':'verification_'+a.step,'started_utc':utc.isoformat(),'status':'running','command':cmd,'timeout_seconds':remaining,'previous_scientific_seconds':science,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('scripts/*fixed_predictions*.py')},'accounting':'whole post-fit process conservatively included in scientific ceiling','stdout_local_path':str(log)};path.write_text(json.dumps(record,indent=2)+'\n');tick=time.perf_counter();env=os.environ.copy();env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    try:
        with log.open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
        record.update(returncode=r.returncode,status='complete' if r.returncode==0 else 'verification_failed')
    except subprocess.TimeoutExpired:record.update(returncode=124,status='deadline_terminated')
    finally:record.update(elapsed_seconds=time.perf_counter()-tick,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2));raise SystemExit(record['returncode'])
if __name__=='__main__':main()

"""Repair numerical layout in saved projected predictions only; zero model fits.

All maps, fitted objects, validation choices and old historical evidence remain
unchanged. Original generated evidence is preserved under local revisions/.
"""
from __future__ import annotations
import argparse,json,time,datetime,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.run_acs_residual_spectral import ROOT,OUT,HIST_ROOT,HIST,ARMS,parent,wires,read,arrays
from experiments.acs_spectral_audits import load_base
from experiments.acs_transfer_data import sha_file,write_json,array_hash
from experiments.acs_transfer_heads import metrics

def repair(out,root,seeds):
    started=time.perf_counter(); result=[]
    for seed in seeds:
        dest=out/f'seed_{seed}'; labels=arrays(dest/'evaluation_labels.npz'); historical=parent(root)/f'seed_{seed}'
        for condition in HIST+ARMS:
            unit=dest/condition
            if not (unit/'complete.json').exists():continue
            if (unit/'projection_layout_repair.json').exists():
                receipt=read(unit/'projection_layout_repair.json')
                assert sha_file(unit/'predictions.npz')==receipt['after']['predictions.npz']
                assert sha_file(unit/'metrics.json')==receipt['after']['metrics.json']
                result.append(receipt);continue
            tick=time.perf_counter(); completion=read(unit/'complete.json');rows=read(unit/'metrics.json')
            assert sha_file(unit/'predictions.npz')==completion['prediction_sha256']
            assert sha_file(unit/'metrics.json')==completion['metric_sha256']
            meta=read(unit/'audits/audit_selection.json'); predictions=arrays(unit/'predictions.npz')
            selected_hash=sha_file(unit/'selection_before_test.json');audit_hash=sha_file(unit/'audits/audit_selection.json')
            oldkeys=set()
            if condition in HIST:
                for row in read(historical/condition/'metrics.json')['raw_metrics']:
                    oldkeys.add((row['role'],row['view'],row['target'],row['audit_budget'],row['candidate_id']))
                release=historical/'training'/condition/'releases.npz'
            else:release=dest/'releases'/condition/'releases.npz'
            w,d=wires(release); models={}; cache={};changed=[];maxerr=0.;scoreerr=0.;checked=0
            for row in rows['raw_metrics']:
                v,t,b,cid=(row[k] for k in ('view','target','audit_budget','candidate_id'))
                if row['role']!='audit' or ('audit',v,t,b,cid) in oldkeys:continue
                record=meta['candidates'][str(b)][v+'/'+t][cid]
                cols=record['projection_columns']
                if cols is None:continue
                path=record['base_candidate_directory']
                if path not in models:models[path]=load_base(path)
                base=models[path]
                for split,pool in [('validation','attacker_validation'),('test','test')]:
                    y=labels[pool+'/'+t];valid=y>=0;view=(w if record['space']=='wire' else d)[v][pool]
                    x=np.ascontiguousarray(view[valid][:,cols]);key=(path,pool,array_hash(x))
                    if key not in cache:cache[key]=base.predict_proba(x)
                    p=cache[key];pk=f'audit/{v}/{t}/{b}/{cid}/{split}';old=predictions[pk];checked+=1
                    error=float(np.max(np.abs(old-p)));maxerr=max(maxerr,error)
                    if not np.array_equal(old,p):
                        predictions[pk]=p;changed.append({'key':pk,'max_abs_probability_difference':error})
                    # Scoring is deterministic and always based on fixed validation choices.
                    for weight in (False,True):
                        sk=split+('_person_weighted' if weight else '')
                        actual=metrics(y[valid],p,9 if t=='RAC1P' else 2,labels['weights/'+pool][valid] if weight else None)
                        scoreerr=max(scoreerr,abs(actual['log_loss']-row['scores'][sk]['log_loss']))
                        row['scores'][sk]=actual
            before={p:sha_file(unit/p) for p in ('predictions.npz','metrics.json','complete.json')}
            revision=unit/'revisions'/'pre_projection_layout'
            revision.mkdir(parents=True,exist_ok=False)
            for p in before:shutil.copy2(unit/p,revision/p)
            np.savez_compressed(unit/'predictions.repaired.npz',**predictions)
            write_json(unit/'metrics.repaired.json',rows)
            (unit/'predictions.repaired.npz').replace(unit/'predictions.npz')
            (unit/'metrics.repaired.json').replace(unit/'metrics.json')
            completion.update(prediction_sha256=sha_file(unit/'predictions.npz'),metric_sha256=sha_file(unit/'metrics.json'),projection_layout_repaired=True)
            write_json(unit/'complete.json',completion)
            assert sha_file(unit/'selection_before_test.json')==selected_hash and sha_file(unit/'audits/audit_selection.json')==audit_hash
            receipt={'seed':seed,'condition':condition,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'runtime_seconds':time.perf_counter()-tick,'new_fits':0,'reselections':0,'projection_arrays_checked':checked,
                'changed_arrays':len(changed),'max_abs_probability_difference':maxerr,'max_abs_logloss_difference':scoreerr,
                'before':before,'after':{p:sha_file(unit/p) for p in before},'unchanged_selection_sha256':selected_hash,
                'unchanged_audit_selection_sha256':audit_hash,'source_sha256':sha_file(Path(__file__)),
                'reason':'Use authoritative contiguous projected coordinates before saved standardizer/model, matching AuditCandidate route.'}
            write_json(unit/'projection_layout_repair.json',receipt);result.append(receipt)
            print('REPAIRED',seed,condition,len(changed),maxerr,round(time.perf_counter()-tick,2),flush=True)
    write_json(out/'PROJECTION_LAYOUT_REPAIR.json',{'complete_units_repaired':len(result),'expected_units':42,'complete':len(result)==42,
        'new_fits':0,'reselections':0,'max_abs_probability_difference':max((r['max_abs_probability_difference'] for r in result),default=0.),
        'max_abs_logloss_difference':max((r['max_abs_logloss_difference'] for r in result),default=0.),'units':result,
        'unique_repair_unit_seconds':sum(r['runtime_seconds'] for r in result),'invocation_elapsed_seconds':time.perf_counter()-started})

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--historical-root',type=Path,default=HIST_ROOT);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):repair(a.out,a.historical_root,a.seeds)
if __name__=='__main__':main()

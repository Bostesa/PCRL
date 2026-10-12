"""Four predeclared independent pair-reader refits, with INNER SEX sealed."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args()
    from task_conditioned_representation.scheduler import assert_active_from_env
    guard=assert_active_from_env(stage='verification')
    spec=json.loads(Path(a.spec).read_text())
    import numpy as np
    import joblib
    from .readers import READERS,features,split,refit,predict
    from .science import auc
    if len(spec['banks'])!=4 or {(r['family'],r['seed']) for r in spec['banks']}!={(f,s) for f in ('U_NEW','JOINT_CONDITIONAL') for s in (0,1)}:
        raise ValueError('unregistered independent refit scope')
    results=[]
    for entry in spec['banks']:
        assert_active_from_env(stage='verification')
        root=Path(entry['directory'])
        receipt_path=root/'FROZEN_COMPLETE.json'
        if digest(receipt_path)!=entry['receipt_sha256']:raise ValueError('reader freeze changed')
        receipt=json.loads(receipt_path.read_text())
        if receipt['status']!='FROZEN_COMPLETE' or receipt['source_seed']!=entry['seed'] or receipt['INNER_SEX_opened']:
            raise ValueError('reader freeze scope mismatch')
        for name,key in [('ATT_RELEASE.npz','att_release_sha256'),('INNER_RELEASE.npz','inner_release_sha256')]:
            if digest(root/name)!=receipt[key]:raise ValueError('release freeze changed')
        with np.load(root/'ATT_RELEASE.npz',allow_pickle=False) as a:att={k:a[k] for k in a.files}
        with np.load(root/'INNER_RELEASE.npz',allow_pickle=False) as a:inner={k:a[k] for k in a.files}
        if 'sex' in inner:raise ValueError('INNER SEX must be sealed during refits')
        train,val=split(att['unit'])
        if not np.array_equal(train,att['train']) or not np.array_equal(val,att['validation']):raise ValueError('ATT split differs')
        # Frozen tie order, source validation only. No reporting labels enter.
        best=max(READERS,key=lambda n:receipt['validation']['pair'][n]['fixed_orientation_auc'])
        x=features(att['r1'],att['r2'],'pair',best)
        started=time.process_time()
        model=refit(x[train],att['sex'][train],best,entry['seed'])
        vs=predict(model,x[val]);sign=1 if auc(att['sex'][val],vs)>=.5 else -1
        score=predict(model,features(inner['r1'],inner['r2'],'pair',best))
        if sign==-1:score=1-score
        dest=Path(spec['private_output'])/f"{entry['family']}__s{entry['seed']}"
        dest.mkdir(parents=True,exist_ok=False)
        path=dest/'MODEL.joblib';joblib.dump(model,path,compress=3);os.chmod(path,0o600)
        pred=dest/'INNER_SCORE.npz';np.savez_compressed(pred,score=score,row_id=inner['row_id'],unit=inner['unit']);os.chmod(pred,0o600)
        results.append({'family':entry['family'],'seed':entry['seed'],'reader':best,'orientation':sign,
          'ATT_validation_auc':auc(att['sex'][val],vs if sign==1 else 1-vs),'model_sha256':digest(path),
          'score_sha256':digest(pred),'cpu_s':time.process_time()-started,'columns':x.shape[1],'fit_rows':len(train),
          'validation_rows':len(val),'INNER_SEX_opened':False})
    out={'schema':'TCR1-R6-independent-reader-refits-v1','status':'FROZEN_COMPLETE','guard':guard,'fits':results,
         'numerical_fit_count':4,'claim_scope':'secondary reader-refit sensitivity only; no competitive promotion'}
    p=Path(spec['output']);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()

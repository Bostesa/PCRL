from pathlib import Path
import json,gzip,csv,hashlib,time
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from scripts.replay_acs_residual_spectral import literal_spectral,literal_basis,literal_labels,arrays,sha,SPECTRAL
root=Path('/Users/nathansamson/PCRL');out=Path.cwd()/'results/redesign_20260910_acs_residual_spectral_v1'
started=time.perf_counter();sources={};expected={};counts=0;maxerror=0.;diagchecks=0
raw=pd.read_csv(root/'data/folktables/2018/1-Year/psam_p06.csv',usecols=['SEX','RAC1P','PUBCOV','PINCP','ESR','MIG','JWMNP'])
with threadpool_limits(limits=1):
 for seed in range(3):
  d=out/f'seed_{seed}';h=root/'results/redesign_20260909_acs_fixed_predictions_v1'/f'seed_{seed}'
  model=joblib.load(d/'maps.joblib');train=json.loads((d/'matrix_diagnostics.json').read_text());held=json.loads((d/'heldout_moments.json').read_text())
  sources.update({str(d/n):sha(d/n) for n in ('maps.joblib','matrix_diagnostics.json','heldout_moments.json')})
  pca,anchor,rows=arrays(h/'pca.npz'),arrays(h/'anchors.npz'),arrays(h/'split_rows.npz')
  folds=np.asarray(train['fold_assignments'])
  for pool in ('representation_fit','source_validation','test'):
   labels=literal_labels(raw.iloc[rows[pool]]);V=literal_spectral(model,pca[pool],anchor[pool+'/A'])
   bases={'local':literal_basis(model.qA,anchor[pool+'/A'][:,[1,3]]),'coalition':literal_basis(model.qAB,np.column_stack((anchor[pool+'/A'][:,[1,3]],anchor[pool+'/B'][:,1])))}
   for role,basis in bases.items():
    attrnames=('SEX','RAC1P','public_coverage') if role=='local' else ('SEX','RAC1P');den=len(attrnames)
    perarm={arm:[] for arm in SPECTRAL}
    for target in attrnames:
     classes=9 if target=='RAC1P' else 2;y=labels[target];valid=y>=0
     def predict(m,x):
      if m.estimator is None:return np.tile(m.constant,(len(x),1))
      z=np.zeros((len(x),classes));z[:,m.estimator.classes_.astype(int)]=m.estimator.predict_proba(x);return z
     models=model.nuisance_models[role][target]
     if pool=='representation_fit':
      p=np.zeros((len(y),classes))
      for fold,m in enumerate(models):p[folds==fold]=predict(m,basis[folds==fold])
     else:p=sum(predict(m,basis) for m in models)/3
     details=train['penalties'][role]['attributes'][target] if pool=='representation_fit' else held[pool][role][target]
     trace=train['penalties'][role]['attributes'][target]['raw_trace'];norms={arm:0. for arm in SPECTRAL}
     for k in range(classes):
      residual=(y[valid]==k).astype(float)-p[valid,k]
      G=V[valid].T@(basis[valid]*residual[:,None])/valid.sum()
      stored=np.asarray(details['classes'][k]['moment_matrix'])
      error=float(np.max(abs(G-stored)));maxerror=max(maxerror,error);assert error<2e-9,(seed,pool,role,target,k,error)
      for arm,W in model.maps.items():
       value=float(np.sum((W.T@G)**2));saved=details['classes'][k]['output_squared_norms'][arm]
       error=abs(value-saved);maxerror=max(maxerror,error);assert error<2e-9
       norms[arm]+=value;diagchecks+=1
     for arm,value in norms.items():
      normalized=value/trace if trace>1e-12 else 0. if pool=='representation_fit' else None
      e={'raw_projected_moment':value,'training_attribute_trace':trace,'normalized_projected_moment':normalized,'fixed_role_denominator':den,'zero_training_trace':trace<=1e-12,'valid_rows':int(valid.sum()),'unsupported_classes':np.flatnonzero(np.bincount(y[valid],minlength=classes)==0).tolist()}
      expected[seed,arm,pool,role,target]=e;perarm[arm].append(e)
    for arm,items in perarm.items():
     expected[seed,arm,pool,role,'__mean__']={'raw_projected_moment':sum(r['raw_projected_moment'] for r in items)/den,'training_attribute_trace':None,'normalized_projected_moment':sum(r['normalized_projected_moment'] for r in items)/den if all(r['normalized_projected_moment'] is not None for r in items) else None,'fixed_role_denominator':den,'zero_training_trace':any(r['zero_training_trace'] for r in items),'valid_rows':None,'unsupported_classes':{t:r['unsupported_classes'] for t,r in zip(attrnames,items)}}
 path=out/'SURROGATE_DIAGNOSTICS.csv.gz';sources[str(path)]=sha(path)
 with gzip.open(path,'rt') as f:
  for row in csv.DictReader(f):
   key=(int(row['seed']),row['condition'],row['pool'],row['role'],row['attribute']);e=expected.pop(key)
   for name,v in e.items():
    value=row[name]
    if isinstance(v,(dict,list)):assert json.loads(value)==v
    elif isinstance(v,bool):assert value==str(v)
    elif v is None:assert value==''
    else:
     error=abs(float(value)-v);maxerror=max(maxerror,error);assert error<2e-9,(key,name,error)
   counts+=1
 assert not expected and counts==504
for p,digest in sources.items():assert sha(p)==digest
result={'passed':True,'rows':counts,'independent_class_arm_moment_checks':diagchecks,'max_absolute_error':maxerror,'runtime_seconds':time.perf_counter()-started,'scientific_fits':0,'optimizer_updates':0,'training_prediction_rule':'saved three grouped OOF models, each applied only to its own holdout fold','heldout_prediction_rule':'equal ensemble of same three frozen models','normalization':'fixed original training trace for each attribute; fixed local3/coalition2 denominators','source_sha256':sources,'script_sha256':sha(__file__)}
(out/'INDEPENDENT_SURROGATES.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

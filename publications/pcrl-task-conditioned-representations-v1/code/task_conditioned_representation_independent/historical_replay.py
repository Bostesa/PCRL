"""Independent restoration of the additional fixed historical raw83 service."""
from __future__ import annotations
import json
import numpy as np
from .science import encoder,own_release

def verify_restoration(entries,manifest_pin,checked):
    import joblib,torch
    if len(entries)!=2 or {(e['family'],e['seed']) for e in entries}!={('H_RAW_CSTAR',s) for s in (0,1)}:
        raise ValueError('complete additional historical reference required')
    manifest=json.loads(checked(manifest_pin).read_text());reports=[]
    from .checkpoint_io import read
    for entry in entries:
        pins=manifest['seeds'][str(entry['seed'])]['pins']
        complete=json.loads(checked(pins['complete']).read_text())
        for file,key in [('model.pt','model'),('head_0.joblib','head0'),('head_1.joblib','head1')]:
            if complete['files'][file]!=pins[key]['sha256']:raise ValueError('historical COMPLETE source binding differs')
        original=torch.load(checked(pins['model']),map_location='cpu',weights_only=True)
        payload=read(checked({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']}))
        if payload['family']!='H_RAW_CSTAR' or payload['seed']!=entry['seed'] or payload['updates_per_recipient']!=[0,0]:
            raise ValueError('historical reference has new updates or wrong identity')
        if any(not torch.equal(t,original[k]) for k,t in payload['model'].items() if k.startswith('enc.')):
            raise ValueError('historical source encoder tensors differ')
        heads=[joblib.load(checked(pins[f'head{r}'])) for r in (0,1)]
        role_reports=[]
        for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):
            with np.load(checked(manifest['roles'][role]['features']),allow_pickle=False) as z:x=z['X'];names=z['feature_names']
            if list(map(str,names))!=payload['feature_names']:raise ValueError('historical raw83 order differs')
            release=own_release(payload['model'],x);checks=[]
            for r,head in enumerate(heads):
                z=encoder(original,r,x);scaler,clf=head.steps[0][1],head.steps[-1][1]
                w=np.asarray(clf.coef_,float)/np.asarray(scaler.scale_,float)
                b=np.asarray(clf.intercept_,float)-w@np.asarray(scaler.mean_,float)
                if w.shape[0]==1:
                    divisor=1 if getattr(clf,'multi_class','auto')=='multinomial' else 2
                    w=np.concatenate((-w/divisor,w/divisor));b=np.concatenate((-b/divisor,b/divisor))
                else:w-=w.mean(0);b-=b.mean()
                expected=z@w.T+b;expected-=expected.mean(1,keepdims=True)
                p=head.predict_proba(z)
                zgap=float(np.max(abs(z-release[f'r{r+1}'])));lgap=float(np.max(abs(expected-release[f'c{r+1}'])));pgap=float(np.max(abs(p-release[f'p{r+1}'])))
                if zgap!=0 or max(lgap,pgap)>2e-12 or not np.array_equal(p.argmax(1),release[f'hard{r+1}']):raise ValueError('independent historical source restoration differs')
                checks.append({'representation_error':zgap,'centered_logits_error':lgap,'probability_error':pgap,'binary_convention':getattr(clf,'multi_class','auto')})
            role_reports.append({'role':role,'rows':len(x),'tasks':checks})
        reports.append({'family':'H_RAW_CSTAR','seed':entry['seed'],'roles':role_reports,'new_updates':0,'primary_pool_changed':False})
    return reports

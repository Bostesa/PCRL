"""Independent metrics, functional forward, projection, readers and resampling.

Every numerical operation runs inside the root-owned queue. Imports from the
campaign are limited to its queue authenticator. Source paths and data bindings
are explicit; no all-role reader and no implicit INNER label capability exists.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
FIT_SALT='IVR1|FIT-crossfold|20261010|'
ATT_SALT='pcrl-method-portfolio-v1|attack-tune|20261008|'
BASE=('LR_C1','MLP_64x32','HGB_15x200','DA_MLP_64x32','REP_LR_C1','REP_MLP_64x32','REP_HGB_15x200','OUTPUT_LR_C1','OUTPUT_HGB_15x200')


class SavedReference:
    """Neutral custody of serialized legacy reference attributes; no methods."""
    pass


def torch_load(path,weights_only=False):
    """Avoid implicit runner imports caused by legacy pickled dataclasses."""
    import torch,pickle,types
    if weights_only:return torch.load(path,map_location='cpu',weights_only=True)
    class IndependentUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module.startswith(('task_conditioned_representation','occupation_bottleneck.')):
                if module=='task_conditioned_representation.kernels' and name=='DiagonalReference':return SavedReference
                raise PermissionError('unregistered scientific pickle class '+module+'.'+name)
            return super().find_class(module,name)
    neutral=types.ModuleType('occupation_independent_pickle');neutral.Unpickler=IndependentUnpickler;neutral.load=pickle.load;neutral.loads=pickle.loads;neutral.dump=pickle.dump;neutral.dumps=pickle.dumps
    return torch.load(path,map_location='cpu',weights_only=False,pickle_module=neutral)


class Inputs:
    def __init__(self,path):
        from occupation_bottleneck.scheduler import assert_active_from_env
        self.guard=assert_active_from_env
        active=self.guard(stage='verification')
        state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
        self.pins={str(Path(x['path']).absolute()):x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
        self.cfg=json.loads(self.path(path).read_text())
        if self.cfg['unit_id']!=active['unit_id']:raise PermissionError('verifier unit mismatch')
        self.checked={}

    def seal(self):
        item=self.cfg.get('reporting_lock')
        if not item:raise PermissionError('INNER labels remain closed')
        p=Path(item['path']).absolute()
        canonical=ROOT/'results/pcrl_occupation_bottleneck_v1/REPORTING_LOCK.json'
        if p!=canonical:raise PermissionError('noncanonical reporting lock')
        self.path(item)
        lock=json.loads(p.read_text())
        if lock.get('owner')!='ROOT' or lock.get('status')!='ALL_READERS_FROZEN' or not lock.get('candidates_frozen') or not lock.get('readers_frozen'):raise PermissionError('reporting not globally frozen')
        committed=subprocess.run(['git','show','HEAD:'+p.relative_to(ROOT).as_posix()],cwd=ROOT,capture_output=True)
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest()!=item['sha256']:raise PermissionError('lock bytes not committed')

    def path(self,item):
        self.guard(stage='verification')
        p=Path(item['path'] if isinstance(item,dict) else item).absolute()
        low=p.as_posix().lower()
        if any(x in low for x in ('head_validation','old_assessment','osf_development_assessment','all_role','all-role')):raise PermissionError('barred source')
        if p.name=='release.npz' and ('teacher' in low or 'rel__s' in low):raise PermissionError('barred all-role release')
        if any(q.is_symlink() for q in (p,*p.parents)):raise PermissionError('symlink refused')
        if ('inner_selection' in low or 'inner_truth' in low) and p.stem.lower() in ('sex','income','occupation'):self.seal()
        if p.stem.lower() in ('inner_utility_inputs','inner_truth','inner_labels'):self.seal()
        sha=self.pins.get(str(p))
        if not sha or isinstance(item,dict) and item['sha256']!=sha:raise PermissionError('unregistered independent source')
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=sha:raise PermissionError('source differs')
        return p

    def json(self,item):return json.loads(self.path(item).read_text())

    def arrays(self,item,fields=None):
        import numpy as np
        p=self.path(item)
        with ZipFile(p) as z:
            names=[v.filename for v in z.infolist()]
            if len(names)!=len(set(names)) or any(not x.endswith('.npy') for x in names):raise ValueError('unsafe NPZ schema')
            if fields is not None and set(names)!={x+'.npy' for x in fields}:raise ValueError('exact NPZ fields differ')
            if 'inner' in p.as_posix().lower() and {'sex.npy','income.npy','occupation.npy'}&set(names):self.seal()
        with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def vectors(y,p,order=None):
    import numpy as np
    y=np.asarray(y);p=np.asarray(p,dtype=np.float64)
    order=np.arange(p.shape[1]) if order is None else np.asarray(order)
    if p.ndim!=2 or y.shape!=(len(p),) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0):raise ValueError('probability convention')
    if len(set(order))!=p.shape[1] or not np.isin(y,order).all():raise ValueError('class order')
    truth=(y[:,None]==order).astype(float)
    return {'accuracy':(order[p.argmax(1)]==y).astype(float),'nll':-(truth*np.log(np.clip(p,1e-12,None))).sum(1),'brier':((p-truth)**2).sum(1)}


def metrics(y,p,prior=None):
    import numpy as np
    v=vectors(y,p);d=p.argmax(1);classes=range(p.shape[1])
    recall=[float((d[y==k]==k).mean()) if (y==k).any() else 0. for k in classes]
    out={k:float(x.mean()) for k,x in v.items()}
    out.update(balanced_accuracy=float(np.mean(recall)),per_class_recall=recall,class_order=list(classes),class_counts=[int((y==k).sum()) for k in classes],confidence=float(p.max(1).mean()),confidence_by_true_class=[float(p[y==k].max(1).mean()) if (y==k).any() else None for k in classes],true_class_probability_by_class=[float(p[y==k,k].mean()) if (y==k).any() else None for k in classes],per_class_recall_available=[bool((y==k).any()) for k in classes])
    if prior is not None:
        cv=vectors(y,np.broadcast_to(prior,p.shape));base={k:float(x.mean()) for k,x in cv.items()}
        out['constant_prediction']=base;out['gain_over_constant']={'accuracy':out['accuracy']-base['accuracy'],'nll':base['nll']-out['nll'],'brier':base['brier']-out['brier']}
    return out


def weighted_auc(y,s,weights=None):
    """Independent tie-block concordance with half credit for ties."""
    import numpy as np
    y=np.asarray(y);s=np.asarray(s,dtype=float)
    w=np.ones((1,len(y))) if weights is None else np.atleast_2d(weights).astype(float)
    if y.shape!=s.shape or not np.isin(y,[0,1]).all() or np.unique(y).size!=2 or not np.isfinite(s).all() or not np.isfinite(w).all() or w.shape[1]!=len(y) or (w<0).any():raise ValueError('AUC inputs')
    _,rank=np.unique(s,return_inverse=True);n=rank.max()+1
    result=[]
    for row in w:
        positive=np.bincount(rank,weights=row*y,minlength=n);negative=np.bincount(rank,weights=row*(1-y),minlength=n)
        denominator=positive.sum()*negative.sum()
        if denominator<=0:raise ValueError('missing AUC class')
        below=np.r_[0,np.cumsum(negative)[:-1]]
        result.append(float(np.dot(positive,below+.5*negative)/denominator))
    return np.array(result)


def folds(unit):
    import numpy as np
    return np.array([int.from_bytes(hashlib.sha256((FIT_SALT+str(v)).encode()).digest(),'big')%3 for v in unit])


def forward(payload,x):
    import torch
    import torch.nn.functional as f
    import numpy as np
    torch.set_num_threads(1)
    if payload['schema']!='TCR1-trainable-own-head-checkpoint-v1' or payload['class_orders']!=[[0,1],list(range(6))]:raise ValueError('checkpoint schema/order')
    state=payload['model'];result=[]
    with torch.no_grad():
        for i,k in enumerate((2,6)):
            z=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32)
            for layer in (0,2,4):
                z=f.linear(z,state[f'enc.{i}.{layer}.weight'],state[f'enc.{i}.{layer}.bias'])
                if layer!=4:z=f.relu(z)
            l=f.linear(z.double(),state[f'head.{i}.weight'],state[f'head.{i}.bias']);l=l-l.mean(1,keepdim=True);p=torch.softmax(l,1)
            result.append(np.column_stack((z.numpy(),l.numpy(),p.numpy(),p.argmax(1).numpy())))
    return result


def geometry(weight,bias,z,rule):
    import numpy as np
    w=np.asarray(weight,dtype=float);b=np.asarray(bias,dtype=float);a=w-w.mean(0);c=b-b.mean()
    _,s,v=np.linalg.svd(a,full_matrices=False)
    cutoff=max(rule['relative_rank_cutoff'],rule['epsilon_multiplier']*max(a.shape)*np.finfo(float).eps)*s[0]
    if ((s>0)&(s>=.5*cutoff)&(s<=2*cutoff)).any():raise ValueError('ambiguous SVD rank')
    basis=v[s>cutoff].copy()
    for row in basis:
        if row[np.argmax(abs(row))]<0:row*=-1
    projector=basis.T@basis;zp=(z@basis.T)@basis
    def head(value):
        l=value@w.T+b;l-=l.mean(1,keepdims=True);e=np.exp(l-l.max(1,keepdims=True));p=e/e.sum(1,keepdims=True)
        return l,p,p.argmax(1)
    l,p,d=head(z);ll,pp,dd=head(zp);gap=np.sort(l,axis=1)[:,-1]-np.sort(l,axis=1)[:,-2]
    return {'rank':len(basis),'B':basis,'P':projector,'projected_z':zp,'AP_max_absolute_residual':float(abs(a@projector-a).max()),'max_absolute_logit_difference':float(abs(l-ll).max()),'max_absolute_probability_difference':float(abs(p-pp).max()),'decision_mismatch_rows':int((d!=dd).sum()),'near_tie_rows':int((gap<=rule['near_tie_absolute_logit_gap']).sum()),'near_tie_mismatch_rows':int(((gap<=rule['near_tie_absolute_logit_gap'])&(d!=dd)).sum())}


def reader_table(r1,r2,name,view='pair'):
    import numpy as np
    kind='defense' if name.startswith('DA_') else 'representation' if name.startswith('REP_') else 'output' if name.startswith('OUTPUT_') else 'logits_output' if name.startswith('LOGITS_OUTPUT_') else 'decision' if name.startswith('DECISION_') else 'confidence' if name.startswith('CONFIDENCE_') else 'full'
    tables=[]
    for wire,k in ((r1,2),(r2,6)):
        p=wire[:,16+k:16+2*k]
        tables.append({'full':wire,'representation':wire[:,:16],'output':wire[:,16+k:],'logits_output':wire[:,16:],'decision':wire[:,-1:],'confidence':p.max(1,keepdims=True),'defense':np.column_stack((wire,np.log(np.maximum(p,np.finfo(float).eps))))}[kind])
    if view!='pair':return tables[0 if view=='r1' else 1]
    pair=np.column_stack(tables)
    if kind=='defense':pair=np.column_stack((pair,(r1[:,:16,None]*r2[:,None,:16]).reshape(len(r1),-1),(r1[:,16:18,None]*r2[:,None,16:22]).reshape(len(r1),-1)))
    return pair


def refit(inputs):
    import numpy as np, joblib, warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    cfg=inputs.cfg;out=Path(cfg['private_output']);out.mkdir(parents=True,exist_ok=False,mode=0o700);records=[]
    for target in cfg['targets']:
        inputs.guard(stage='verification');a=inputs.arrays(target['att_release']);b=inputs.arrays(target['inner_release'],{'row_id','unit','r1','r2'})
        unique=np.unique(a['unit']);ranked=sorted((hashlib.sha256((ATT_SALT+str(int(g))).encode()).hexdigest(),int(g)) for g in unique)
        val=np.isin(a['unit'],[g for _,g in ranked[:1015]]);train=~val
        if not np.array_equal(np.flatnonzero(train),a['train']) or not np.array_equal(np.flatnonzero(val),a['validation']):raise ValueError('ATT split replay')
        name=target['reader'];view=target.get('view','pair');x=reader_table(a['r1'],a['r2'],name,view);xi=reader_table(b['r1'],b['r2'],name,view)
        mean=x[train].mean(0);sd=x[train].std(0);scale=np.where(sd>0,sd,1.)
        if 'MLP' in name:model=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=1e-4,max_iter=300,early_stopping=False,random_state=target['model_seed'])
        elif 'HGB' in name:model=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=200,learning_rate=.05,early_stopping=False,random_state=target['model_seed'])
        else:model=LogisticRegression(C=1,max_iter=500,solver='lbfgs')
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');model.fit((x[train]-mean)/scale,a['sex'][train])
        raw=model.predict_proba((x[val]-mean)/scale)[:,1];positive=float(weighted_auc(a['sex'][val],raw)[0]);sign=1 if positive>=.5 else -1
        inner=model.predict_proba((xi-mean)/scale)[:,1];inner=inner if sign==1 else 1-inner
        key=target['target_id'];joblib.dump({'model':model,'mean':mean,'scale':scale},out/(key+'.joblib'));np.savez_compressed(out/(key+'.npz'),row_id=b['row_id'],unit=b['unit'],score=inner)
        def sha(path):
            with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
        records.append({'target_id':key,'arm_id':target['arm_id'],'seed':target['seed'],'reader':name,'orientation':sign,'ATT_positive_auc':positive,'ATT_oriented_auc':max(positive,1-positive),'orientation_matches_target':sign==target['orientation'],'model_sha256':sha(out/(key+'.joblib')),'score_sha256':sha(out/(key+'.npz')),'ATT_release_sha256':target['att_release']['sha256'],'INNER_label_free_release_sha256':target['inner_release']['sha256'],'INNER_labels_opened':False,'warnings':[str(w.message) for w in ws]})
    return {'schema':'OB1-independent-refits-v1','records':records,'numerical_fit_count':len(records),'scientific_runner_imports':False}


def development(inputs):
    import numpy as np
    cfg=inputs.cfg;m=inputs.json(cfg['manifest']);pins=m['roles']['OSF_DEFENSE_FIT']
    f=inputs.arrays(pins['features'],{'row_id','unit','X','feature_names'});labels={}
    for k in ('income','occupation','sex'):
        a=inputs.arrays(pins[k],{'row_id','unit',k})
        if any(not np.array_equal(a[q],f[q]) for q in ('row_id','unit')):raise ValueError('diagnostic labels unpaired')
        labels[k]=a[k]
    fold=folds(f['unit']);train=fold!=2;test=~train
    if set(f['unit'][train])&set(f['unit'][test]):raise ValueError('diagnostic group leakage')
    result={}
    for view,k,label in (('r1',2,labels['income']),('r2',6,labels['occupation']),('pair',12,labels['income']*6+labels['occupation'])):
        counts=np.array([[int(((label[train]==v)&(labels['sex'][train]==s)).sum()) for s in (0,1)] for v in range(k)])
        table=(counts[:,1]+1)/(counts.sum(1)+2);score=table[label[test]]
        result[view]={'auc':float(weighted_auc(labels['sex'][test],score)[0]),'training_counts':counts.tolist(),'table':table.tolist()}
    banks=[]
    for bank in cfg.get('banks',[]):
        release=inputs.arrays(bank['release'],{'row_id','unit','r1','r2'});scores=inputs.arrays(bank['scores'])
        if any(not np.array_equal(release[key],f[key][test]) for key in ('row_id','unit')):raise ValueError('C3 independent bank rows')
        recovery=envelopes(scores,labels['sex'][test]);strata={}
        for view,k,label in (('r1',2,labels['income'][test]),('r2',6,labels['occupation'][test]),('pair',12,labels['income'][test]*6+labels['occupation'][test])):
            records=[];y=labels['sex'][test];group=f['unit'][test]
            for value in range(k):
                mask=label==value;support=[len(np.unique(group[mask&(y==s)])) for s in (0,1)]
                item={'class_or_pair_id':value,'rows':int(mask.sum()),'SEX_group_counts':support}
                if min(support)<32:item.update(status='MISSING_SPARSE_STRATUM',auc=None)
                else:
                    values={n:float(weighted_auc(y[mask],s[mask])[0]) for n,s in scores.items() if n.startswith(view+'__') and n.split('__')[-1] in BASE}
                    item.update(status='ESTIMATED_DESCRIPTIVE' if values else 'MISSING_FROZEN_READER_SCORES',auc=max(values.values()) if values else None)
                records.append(item)
            strata[view]=records
        banks.append({'arm_id':bank['arm_id'],'seed':bank['seed'],'recovery':recovery,'within_true_task_strata':strata})
    return {'schema':'OB1-independent-C3-v1','training_rows':int(train.sum()),'scoring_rows':int(test.sum()),'training_groups':len(np.unique(f['unit'][train])),'scoring_groups':len(np.unique(f['unit'][test])),'task_association':result,'reader_banks':banks,'training_role':'FIT folds0/1','scoring_role':'FIT fold2','source_training_saw_all_FIT':True,'INNER_labels_opened':False,'universal_privacy_floor':False}


def close(actual,expected,label,tolerance=2e-12):
    import numpy as np
    if isinstance(actual,list) and any(x is None for x in actual):
        if len(actual)!=len(expected):raise AssertionError(label+' sparse length')
        for i,(a,b) in enumerate(zip(actual,expected)):
            if a is None or b is None:
                if a is not b:raise AssertionError(label+' sparse availability')
            else:close(a,b,label+str(i),tolerance)
        return
    if not np.allclose(actual,expected,rtol=0,atol=tolerance):raise AssertionError(label+' independent discrepancy')


def envelopes(scores,y,indices=None):
    import numpy as np
    ix=np.arange(len(y)) if indices is None else indices;out={}
    for view in ('r1','r2','pair'):
        values={name.split('__',1)[1]:float(weighted_auc(y[ix],value[ix])[0]) for name,value in scores.items() if name.startswith(view+'__') and not name.split('__',1)[1].startswith(('SHUFFLED','PLANTED'))}
        contracts={}
        for contract in ('primary','representation','output','logits_output','decision','confidence'):
            def eligible(name):
                n=name.split('__')[-1]
                return n in BASE if contract=='primary' else n.startswith({'representation':'REP_','output':'OUTPUT_','logits_output':'LOGITS_OUTPUT_','decision':'DECISION_','confidence':'CONFIDENCE_'}[contract])
            chosen={n:v for n,v in values.items() if eligible(n)}
            contracts[contract]=None if not chosen else {'auc':max(chosen.values()),'reader_count':len(chosen)}
        out[view]={'reader_auc':values,'contracts':contracts}
    for view,other in (('r1','r2'),('r2','r1')):
        for name in out[view]['reader_auc']:
            alias='pair__IGNORE_'+other+'__'+name
            if alias not in scores or not np.array_equal(scores[view+'__'+name],scores[alias]):raise AssertionError('local-ignore alias absent or unequal')
    return out


def audit(inputs):
    import numpy as np,torch
    cfg=inputs.cfg;result={'schema':'OB1-independent-scientific-replay-v1','scientific_runner_imports':False,'trajectory':[],'geometry':[],'banks':[],'gradients':[]}
    priors={}
    if cfg.get('source_admission'):
        result['source_admission']=source_audit(inputs,cfg['source_admission'])
        m=inputs.json(cfg['source_admission']['manifest'])
        for task,k in (('income',2),('occupation',6)):
            y=inputs.arrays(m['roles']['OSF_DEFENSE_FIT'][task],{'row_id','unit',task})[task]
            priors[task]=np.array([(y==value).mean() for value in range(k)])
    for item in cfg.get('trajectory',[]):
        a=inputs.arrays(item['inputs']);report=inputs.json(item['summary']);records={}
        for i,task in enumerate(('income','occupation'),1):
            p=a[f"s{report['seed']}__{report['arm_id']}__p{i}"];actual=metrics(a[task],p,priors.get(task))
            expected=report['metrics'][task]['metrics']
            for key in ('accuracy','nll','brier','balanced_accuracy','per_class_recall','class_order','class_counts','confidence','confidence_by_true_class','true_class_probability_by_class','per_class_recall_available'):close(actual[key],expected[key],task+' '+key)
            if task in priors:
                for collection in ('constant_prediction','gain_over_constant'):
                    for key,value in actual[collection].items():close(value,expected[collection][key],task+' '+collection+' '+key)
            deltas={}
            for ref in ('U0','Ucal'):
                # A distinct reference namespace prevents candidate overwrite.
                key=item.get('reference_keys',{}).get(f'{ref}_{i}',f"s{report['seed']}__REF_{ref}__p{i}")
                reference=metrics(a[task],a[key]);delta={'accuracy_loss':reference['accuracy']-actual['accuracy'],'nll_excess':actual['nll']-reference['nll'],'brier_excess':actual['brier']-reference['brier']}
                for endpoint,value in delta.items():close(value,report['metrics'][task]['references'][ref]['difference'][endpoint],task+' '+ref+' '+endpoint)
                deltas[ref]=delta
            records[task]={'metrics':actual,'references':deltas}
        result['trajectory'].append({'arm_id':report['arm_id'],'seed':report['seed'],'metrics':records,'status':'PASS'})
    for item in cfg.get('geometry',[]):
        payload=torch_load(inputs.path(item['checkpoint']));f=inputs.arrays(item['features'],{'row_id','unit','X','feature_names'});wire=forward(payload,f['X']);receipts=[]
        for i in range(2):
            g=geometry(payload['model'][f'head.{i}.weight'].numpy(),payload['model'][f'head.{i}.bias'].numpy(),wire[i][:,:16],item['rule'])
            if item.get('public_geometry'):
                saved=inputs.json(item['public_geometry'])
                saved=saved[i] if isinstance(saved,list) else saved.get('recipients',saved)[i]
                close(g['B'],saved['B'],'public SVD basis',1e-10);close(g['P'],saved['P'],'public projector',1e-10)
            for key in ('B','P','projected_z'):g.pop(key)
            if g['decision_mismatch_rows'] or g['max_absolute_probability_difference']>item['rule']['probability_atol'] or g['AP_max_absolute_residual']>item['rule']['algebra_atol']:raise AssertionError('projection decision/probability parity')
            receipts.append(g)
        result['geometry'].append({'arm_id':item['arm_id'],'seed':item['seed'],'role':item['role'],'checks':receipts,'status':'PASS'})
    for item in cfg.get('banks',[]):
        a=inputs.arrays(item['release']);scores=inputs.arrays(item['scores']);freeze=inputs.json(item['freeze'])
        if 'sex' in a:y=a['sex']
        else:
            truth=inputs.arrays(item['sex'],{'row_id','unit','sex'})
            if any(not np.array_equal(truth[k],a[k]) for k in ('row_id','unit')):raise ValueError('recovery truth unpaired')
            y=truth['sex']
        ix=a.get('validation') if item.get('ATT_validation',False) else None
        actual=envelopes(scores,y,ix)
        if item.get('ATT_validation',False):
            for view in actual:
                for name,value in actual[view]['reader_auc'].items():close(value,freeze['ATT_validation'][view]['reader_auc'][name],view+' '+name)
        result['banks'].append({'arm_id':item['arm_id'],'seed':item['seed'],'recovery':actual,'exact_ignore_other_aliases':'PASS'})
    for item in cfg.get('gradients',[]):
        tensors=torch_load(inputs.path(item['tensors']));summary=inputs.json(item['summary']);receipts=[]
        for batch,blocks in tensors['batches'].items():
            for block,vectors_ in blocks.items():
                vectors_={n:v.double().numpy().reshape(-1) for n,v in vectors_.items()};expected=summary['records'][int(batch)]['blocks'][block];norm={n:float(np.linalg.norm(v)) for n,v in vectors_.items()}
                for name,value in norm.items():close(value,expected['norms'][name],block+' '+name,tolerance=1e-10)
                for key,pair in expected['pairs'].items():
                    left,right=key.split('__');dot=float(vectors_[left]@vectors_[right]);close(dot,pair['dot'],key,tolerance=1e-10)
                    if norm[left]*norm[right]>0:close(dot/(norm[left]*norm[right]),pair['cosine'],key+' cosine',tolerance=1e-10)
                receipts.append({'batch':batch,'block':block,'tensor_norm_and_conflict_replay':'PASS'})
        result['gradients'].append({'checkpoint_sha256':summary['checkpoint_sha256'],'seed':summary['seed'],'checks':receipts,'replayed_regenerated_tensors':True,'independently_regenerated_gradients':False})
    if cfg.get('gradient_regeneration'):
        result['independently_regenerated_gradients']=[regenerate(inputs,item) for item in cfg['gradient_regeneration']]
    if cfg.get('paths'):
        result['path_diagnostics']=[path_diagnostics(inputs,path) for path in cfg['paths']]
    return result


def path_diagnostics(inputs,path):
    import numpy as np
    from scipy.stats import rankdata
    points=[]
    for point in sorted(path['points'],key=lambda p:p['epoch']):
        selection=inputs.json(point['selection']);bank=inputs.json(point['bank_freeze']);utility=inputs.json(point['utility_summary']);task=utility['metrics']['occupation']['metrics']
        points.append({'epoch':point['epoch'],'saved_sensitive_objective':selection['sensitive'],'occupation_nll':task['nll'],'occupation_brier':task['brier'],'occupation_accuracy':task['accuracy'],'occupation_confidence':task['confidence'],'r1_recovery':bank['ATT_validation']['r1']['contracts']['primary']['auc'],'r2_recovery':bank['ATT_validation']['r2']['contracts']['primary']['auc'],'pair_recovery':bank['ATT_validation']['pair']['contracts']['primary']['auc']})
    transitions=[]
    for left,right in zip(points,points[1:]):
        delta={key:right[key]-left[key] for key in left if key!='epoch'};delta.update(from_epoch=left['epoch'],to_epoch=right['epoch']);transitions.append(delta)
    ranks={}
    for endpoint in ('r2_recovery','pair_recovery'):
        for objective in ('saved_sensitive_objective','occupation_nll'):
            x=np.array([p[objective] for p in points]);y=np.array([p[endpoint] for p in points]);valid=len(points)>=3 and np.std(x)>0 and np.std(y)>0
            ranks[objective+'__'+endpoint]=float(np.corrcoef(rankdata(x,method='average'),rankdata(y,method='average'))[0,1]) if valid else None
    def first(predicate):return next(({'from_epoch':p['from_epoch'],'to_epoch':p['to_epoch']} for p in transitions if predicate(p)),None)
    return {'family':path['family'],'strength':path['strength'],'seed':path['seed'],'points':points,'transitions':transitions,'within_path_spearman':ranks,'first_occupation_confidence_or_loss_cost':first(lambda p:p['occupation_nll']>0 or p['occupation_brier']>0 or p['occupation_confidence']<0),'first_occupation_recovery_decline':first(lambda p:p['r2_recovery']<0),'first_pair_recovery_decline':first(lambda p:p['pair_recovery']<0),'interpretation':'fixed coarse saved checkpoints; no causal or continuous timing inference'}


def regenerate(inputs,item):
    """Independent actual-checkpoint and saved-critic FIT differentiation."""
    import numpy as np,torch
    import torch.nn.functional as F
    torch.set_num_threads(1);m=inputs.json(item['manifest']);pins=m['roles']['OSF_DEFENSE_FIT'];f=inputs.arrays(pins['features'],{'row_id','unit','X','feature_names'});labels={}
    for key in ('income','occupation','sex'):
        a=inputs.arrays(pins[key],{'row_id','unit',key})
        if any(not np.array_equal(a[k],f[k]) for k in ('row_id','unit')):raise ValueError('regeneration FIT labels')
        labels[key]=a[key]
    indices=item['batch_indices']
    if len(set(indices))!=len(indices) or not 0<len(indices)<=256 or not np.isin(folds(f['unit'])[indices],[0,1]).all():raise ValueError('fixed regeneration batch')
    payload=torch_load(inputs.path(item['checkpoint']));state={k:v.detach().clone().requires_grad_(True) for k,v in payload['model'].items()};x=torch.from_numpy(f['X'][indices]);ys={k:torch.from_numpy(v[indices]).long() for k,v in labels.items()};wires=[];probabilities=[];logits=[]
    for i in range(2):
        z=x
        for layer in (0,2,4):
            z=F.linear(z,state[f'enc.{i}.{layer}.weight'],state[f'enc.{i}.{layer}.bias'])
            if layer!=4:z=F.relu(z)
        l=F.linear(z.double(),state[f'head.{i}.weight'],state[f'head.{i}.bias']);l=l-l.mean(1,keepdim=True);p=torch.softmax(l,1);wires.append(torch.cat((z.double(),l,p,p.argmax(1)[:,None].double()),1));probabilities.append(p);logits.append(l)
    loss={'task_ce':F.cross_entropy(logits[0],ys['income'])+F.cross_entropy(logits[1],ys['occupation']),'occupation_ce':F.cross_entropy(logits[1],ys['occupation']),'occupation_nll':-probabilities[1][torch.arange(len(indices)),ys['occupation']].clamp_min(1e-12).log().mean(),'occupation_brier':(probabilities[1]-F.one_hot(ys['occupation'],6).double()).square().sum(1).mean()}
    objective=payload.get('objective_state')
    if objective and 'critics' in objective:
        terms=[]
        for wire,reference,bank in zip((wires[0],wires[1],torch.cat(wires,1)),objective['references'],objective['critics']):
            normalized=(wire-reference.mean)/reference.scale;ces=[]
            for critic in bank:
                value=normalized
                layers=sorted([k for k in critic if k.endswith('weight')],key=lambda k:-1 if k=='weight' else int(k.split('.')[0]))
                for number,key in enumerate(layers):
                    value=F.linear(value,critic[key],critic[key.replace('weight','bias')])
                    if number!=len(layers)-1:value=F.relu(value)
                ces.append(F.cross_entropy(value,ys['sex']))
            ces.append(wire.sum()*0+objective['entropy']);terms.append(1-torch.stack(ces).min()/objective['entropy'])
        loss['sensitive_penalty']=sum(terms)
    original=torch_load(inputs.path(item['tensors'])) if item.get('tensors') else None;records=[]
    blocks={'all_actual_parameters':list(state),'occupation_encoder_and_own_head':[k for k in state if k.startswith(('enc.1.','head.1.'))]}
    for block,names in blocks.items():
        for name,value in loss.items():
            gradient=torch.autograd.grad(value,[state[k] for k in names],retain_graph=True,allow_unused=True);vector=np.concatenate([np.zeros(state[k].numel()) if g is None else g.detach().double().numpy().reshape(-1) for k,g in zip(names,gradient)])
            difference=None
            if original:
                expected=original['batches'][str(item.get('batch_number',0))][block][name].double().numpy();difference=float(abs(vector-expected).max());close(vector,expected,block+' independently regenerated '+name,tolerance=1e-10)
            records.append({'block':block,'loss':name,'norm':float(np.linalg.norm(vector)),'max_absolute_vector_difference':difference})
    return {'checkpoint_sha256':item['checkpoint']['sha256'],'seed':payload['seed'],'batch_rows':len(indices),'batch_index_sha256':hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),'checks':records,'status':'PASS','independently_regenerated_gradients':True,'new_critic_fits':0,'model_updates':0,'INNER_labels_opened':False}


def source_audit(inputs,item):
    import numpy as np,torch,joblib
    m=inputs.json(item['manifest']);roster=inputs.json(item['roster']);roles={};models={};checks=[]
    for seed in (0,1):
        pins=m['seeds'][str(seed)]['pins'];state=torch_load(inputs.path(pins['model']),weights_only=True);own={k:v for k,v in state.items() if k.startswith('enc.')}
        heads=[]
        for i,name in enumerate(('head0','head1')):
            pipeline=joblib.load(inputs.path(pins[name]));scaler,lr=list(pipeline.named_steps.values());heads.append(pipeline)
            scale=scaler.scale_ if scaler.with_std else np.ones(16);mean=scaler.mean_ if scaler.with_mean else np.zeros(16)
            w=lr.coef_.astype(float)/scale;b=lr.intercept_.astype(float)-w@mean
            if len(lr.classes_)==2:
                factor=1 if getattr(lr,'multi_class','auto')=='multinomial' else .5;w=np.vstack((-factor*w,factor*w));b=np.r_[-factor*b,factor*b]
            else:w=w-w.mean(0);b=b-b.mean()
            own[f'head.{i}.weight']=torch.from_numpy(w);own[f'head.{i}.bias']=torch.from_numpy(b)
        payload={'schema':'TCR1-trainable-own-head-checkpoint-v1','class_orders':[[0,1],list(range(6))],'model':own};models[seed]=payload
        cal=inputs.json(m['seeds'][str(seed)]['ucal']['record'])
        for role,rolepins in m['roles'].items():
            f=inputs.arrays(rolepins['features'],{'row_id','unit','X','feature_names'});ref=inputs.arrays(rolepins[f'references_s{seed}'],{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
            counts={'OSF_DEFENSE_FIT':(15434,15428),'ATTACK_FIT_NEW':(4064,4061),'INNER_SELECTION':(2235,2234)}
            if (len(f['row_id']),len(np.unique(f['unit'])))!=counts[role] or len(np.unique(f['row_id']))!=len(f['row_id']):raise AssertionError('source role counts')
            if f['X'].shape!=(counts[role][0],83) or f['X'].dtype!=np.float32 or not np.isfinite(f['X']).all():raise AssertionError('source raw83 convention')
            names=tuple(map(str,f['feature_names']));forbidden={'sex','race','income','occupation','fnlwgt','row_id','unit','id','hidden_flag'}
            if len(names)!=83 or len(set(names))!=83 or any(v.lower().split('=')[0] in forbidden for v in names):raise AssertionError('source forbidden feature schema')
            if any(not np.array_equal(f[k],ref[k]) for k in ('row_id','unit')):raise ValueError('source reference rows unpaired')
            if role in roles and any(not np.array_equal(f[k],roles[role][k]) for k in ('row_id','unit')):raise ValueError('two source seed people differ')
            roles[role]=f;wire=forward(payload,f['X']);tasks=[]
            for i,k in enumerate((2,6)):
                p=wire[i][:,16+k:16+2*k];reference=heads[i].predict_proba(wire[i][:,:16]);close(p,reference,'source folded parity');close(reference,ref[f'U0_{i+1}'],'source stored parity')
                if not np.array_equal(p.argmax(1),ref[f'd_{i+1}']):raise AssertionError('source decisions')
                alpha=cal['tables'][f'H-GLOBAL-TEMP|r{i+1}']['alpha'];logp=np.log(np.maximum(reference,1e-12));v=alpha*logp;v=np.exp(v-v.max(1,keepdims=True));q=v/v.sum(1,keepdims=True);close(q,ref[f'Ucal_{i+1}'],'calibrated reference')
                tasks.append({'recipient':i+1,'max_folded_probability_difference':float(abs(p-reference).max()),'decision_parity':'PASS'})
            checks.append({'seed':seed,'role':role,'rows':len(f['X']),'groups':len(np.unique(f['unit'])),'tasks':tasks})
    for a in roles:
        for b in roles:
            if a<b and (np.intersect1d(roles[a]['row_id'],roles[b]['row_id']).size or np.intersect1d(roles[a]['unit'],roles[b]['unit']).size):raise AssertionError('role duplicate groups overlap')
    aliases=[];missing=[];canonical=set()
    for entry in roster['entries']:
        if not entry['exists']:missing.append(entry['unit_id']);continue
        canonical.add(entry['canonical_model_alias'])
        if entry['canonical_model_alias'].startswith('U0_S'):
            actual=torch_load(inputs.path({'path':entry['checkpoint'],'sha256':entry['checkpoint_sha256']}))
            if not all(torch.equal(v,models[entry['seed']]['model'][k]) for k,v in actual['model'].items()):raise AssertionError('epoch-zero source tensor alias differs')
            aliases.append(entry['unit_id'])
    return {'checks':checks,'exact_source_tensor_aliases':aliases,'alias_count':len(aliases),'nominal_packages':len(roster['entries']),'unique_canonical_packages':len(canonical),'missing':missing,'status':'PASS','INNER_label_bytes_read_or_hashed':False}


def assessment(inputs):
    """All endpoints share duplicate-group draws across candidates and seeds."""
    import numpy as np
    from scipy.stats import t
    cfg=inputs.cfg;inputs.seal();registry=inputs.json(cfg['endpoint_registry']);units=inputs.arrays(cfg['unit_input']);group=units['unit'];unique,inverse=np.unique(group,return_inverse=True)
    if group.ndim!=1 or group.dtype.kind not in 'iu' or not len(group):raise ValueError('assessment unit vector')
    if 'row_id' not in units or units['row_id'].shape!=group.shape or len(np.unique(units['row_id']))!=len(group):raise ValueError('assessment unique row identity')
    expected_rows=registry.get('expected_rows',2235);expected_groups=registry.get('expected_groups',2234)
    if (len(group),len(unique))!=(expected_rows,expected_groups):raise ValueError('assessment admitted row/group counts')
    endpoints=registry['endpoints'];K=len(endpoints)
    if K!=registry['K'] or len({x['endpoint_id'] for x in endpoints})!=K:raise ValueError('actual endpoint family count')
    B=registry['replicates'];seed=registry['bootstrap_seed']
    if not isinstance(B,int) or B<2 or not isinstance(seed,int) or not 0<registry.get('alpha',.05)<1:raise ValueError('bootstrap registry')
    if registry.get('family') in ('privacy','utility'):
        privacy=registry['family']=='privacy';configs=registry['configuration_count'];per=3 if privacy else 24
        if configs!=36 or K!=configs*per or B!=2048 or seed!=2026101301:raise ValueError('frozen campaign assessment dimensions')
        for endpoint in endpoints:
            if endpoint['kind']!=('auc_envelope_difference' if privacy else 'mean_difference'):raise ValueError('endpoint kind differs from registered family')
            if len(endpoint['paired_seeds'])!=(2 if privacy else 1):raise ValueError('privacy seed average versus seed-specific utility clause')
    rng=np.random.default_rng(seed);draws=rng.integers(len(unique),size=(B,len(unique)))
    count=np.zeros((B,len(unique)),dtype=np.int32)
    for b,draw in enumerate(draws):count[b]=np.bincount(draw,minlength=len(unique))
    weights=count[:,inverse];den=weights.sum(1);tables={};values={};out=[]
    for key,item in cfg['arrays'].items():
        table=inputs.arrays(item)
        if any(name not in table or not np.array_equal(table[name],units[name]) for name in ('row_id','unit')):raise ValueError('assessment table person/order alignment '+key)
        tables[key]=table
    for endpoint in endpoints:
        if endpoint['direction'] not in ('at_most','at_least') or not np.isfinite(endpoint['threshold']) or not endpoint['paired_seeds']:raise ValueError('frozen endpoint direction/threshold/seeds')
        parts=[];point=[]
        for part in endpoint['paired_seeds']:
            if part.get('sign',1) not in (-1,1):raise ValueError('endpoint sign convention')
            if endpoint['kind']=='mean_difference':
                left=tables[part['left_table']][part['left_key']];right=tables[part['right_table']][part['right_key']]
                if left.shape!=group.shape or right.shape!=group.shape or not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('assessment finite aligned metric vectors')
                delta=(left-right)*part.get('sign',1);point.append(float(delta.mean()));parts.append(weights@delta/den)
            elif endpoint['kind']=='auc_envelope_difference':
                y=tables[part['truth_table']][part['truth_key']]
                def envelope(which):
                    tab=tables[part[which+'_table']];keys=part[which+'_keys']
                    if not keys or len(set(keys))!=len(keys):raise ValueError('nonempty unique frozen reader envelope')
                    p=max(float(weighted_auc(y,tab[k])[0]) for k in keys);boots=np.zeros(B)
                    for k in keys:
                        for start in range(0,B,32):boots[start:start+32]=np.maximum(boots[start:start+32],weighted_auc(y,tab[k],weights[start:start+32]))
                    return p,boots
                lp,lb=envelope('left');rp,rb=envelope('right');point.append((lp-rp)*part.get('sign',1));parts.append((lb-rb)*part.get('sign',1))
            else:raise ValueError('unknown frozen endpoint')
        p=float(np.mean(point));boot=np.mean(parts,axis=0);se=float(np.std(boot,ddof=1));critical=float(t.ppf(1-registry.get('alpha',.05)/(2*K),len(unique)-1));lo=p-critical*se;hi=p+critical*se;cap=endpoint['threshold']
        if endpoint['direction']=='at_most':status='PASS' if hi<=cap else 'VIOLATION' if lo>cap else 'UNRESOLVED'
        else:status='PASS' if lo>=cap else 'SHORTFALL' if hi<cap else 'UNRESOLVED'
        out.append({'endpoint_id':endpoint['endpoint_id'],'point':p,'bootstrap_se':se,'lower':lo,'upper':hi,'status':status,'threshold':cap,'K':K});values[endpoint['endpoint_id']]=boot
    if cfg.get('private_output'):
        p=Path(cfg['private_output']);p.mkdir(parents=True,exist_ok=False,mode=0o700);np.savez_compressed(p/'BOOTSTRAP.npz',**values);np.savez_compressed(p/'WEIGHTS.npz',weights=weights)
    return {'schema':'OB1-independent-paired-inference-v1','K':K,'replicates':B,'bootstrap_seed':seed,'rows':len(group),'groups':len(unique),'endpoints':out,'coverage':'approximate fixed-model fixed-reader exploratory coverage; two seeds averaged inside each shared group draw'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=Inputs(args.spec)
    result={'refit':refit,'development':development,'audit':audit,'assessment':assessment}[inputs.cfg['operation']](inputs)
    path=Path(inputs.cfg['public_output']);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'INDEPENDENT_REPLAY_COMPLETE','operation':inputs.cfg['operation']}))

if __name__=='__main__':main()

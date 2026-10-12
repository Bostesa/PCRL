"""Fresh reader banks for the literal continuous representation service."""
from __future__ import annotations
import hashlib,json,os,time,warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from .inference import auc

READERS=('LR_C1','MLP_64x32','HGB_15x200','DA_MLP_64x32','REP_LR_C1','REP_MLP_64x32','REP_HGB_15x200','OUTPUT_LR_C1','OUTPUT_HGB_15x200')
OUTPUT_READERS=('OUTPUT_LR_C1','OUTPUT_HGB_15x200')
REP_READERS=('REP_LR_C1','REP_MLP_64x32','REP_HGB_15x200')
MODEL_SEED_BASE=2026101205
ATT_NULL_SEED=2026101206
INNER_NULL_SEED=2026101207
SPLIT_SALT='pcrl-method-portfolio-v1|attack-tune|20261008|'


def sha_file(path):
    p=Path(path)
    if p.is_symlink():raise PermissionError('symlink refused')
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value,private=True):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600 if private else 0o644),'w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return sha_file(p)


def save_npz(path,**arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    return sha_file(p)


class FiniteScaler:
    """Preserve every finite coordinate, including small positive variance."""
    def fit(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or not np.isfinite(x).all() or len(x)<2:raise ValueError('scaler needs finite training matrix')
        self.mean_=x.mean(0);self.raw_scale_=x.std(0)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.raw_scale_).all():raise ValueError('empirical scaler moments must be finite')
        self.scale_=np.where(self.raw_scale_>0,self.raw_scale_,1.)
        return self
    def transform(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.mean_) or not np.isfinite(x).all():raise ValueError('scaler input differs')
        return (x-self.mean_)/self.scale_


def attack_split(unit,require_adult=True):
    g=np.asarray(unit);unique=np.unique(g)
    if g.ndim!=1 or g.dtype.kind not in 'iu' or (require_adult and (len(g)!=4064 or len(unique)!=4061)):raise ValueError('ATT group/row counts differ')
    ranked=sorted((hashlib.sha256((SPLIT_SALT+str(int(i))).encode()).hexdigest(),int(i)) for i in unique)
    count=1015 if require_adult else max(1,len(unique)//4)
    val=np.isin(g,[v for _,v in ranked[:count]])
    if val.all() or not val.any():raise ValueError('empty ATT split')
    return np.flatnonzero(~val),np.flatnonzero(val)


def feature_tables(r1,r2):
    a=np.asarray(r1,dtype=np.float64);b=np.asarray(r2,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape!=(len(a),21) or b.shape!=(len(a),29) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('actual full-wire dimensions/values differ')
    result={}
    for key,x,k in (('r1',a,2),('r2',b,6)):
        p=x[:,16+k:16+2*k];z=x[:,:16];l=x[:,16:16+k]
        result[key]={'full':x,'representation':z,'output':x[:,16+k:],
                     'defense':np.column_stack((x,np.log(np.maximum(p,np.finfo(float).eps))))}
    product=(a[:,:16,None]*b[:,None,:16]).reshape(len(a),-1)
    logits=(a[:,16:18,None]*b[:,None,16:22]).reshape(len(a),-1)
    result['pair']={kind:np.column_stack((result['r1'][kind],result['r2'][kind])) for kind in ('full','representation','output')}
    result['pair']['defense']=np.column_stack((result['r1']['defense'],result['r2']['defense'],product,logits))
    return result


def orientation(y,s):return 1 if auc(y,s)>=.5 else -1

def orient(s,sign):
    if sign not in (1,-1):raise ValueError('orientation must be frozen signed one')
    return np.asarray(s) if sign==1 else 1-np.asarray(s)


def group_shuffle(y,unit,seed):
    if seed not in (ATT_NULL_SEED,INNER_NULL_SEED):raise ValueError('unregistered null seed')
    y=np.asarray(y);g=np.asarray(unit);rng=np.random.default_rng(seed);buckets={};out=np.empty_like(y)
    for v in np.unique(g):
        index=np.flatnonzero(g==v)
        if len(np.unique(y[index]))!=1:raise ValueError('duplicate group label differs')
        buckets.setdefault(len(index),[]).append(index)
    for size in sorted(buckets):
        arr=buckets[size]
        for target,source in zip(arr,rng.permutation(len(arr))):out[target]=y[arr[int(source)]]
    return out


def add_ignore_other(scores):
    out={v:dict(a) for v,a in scores.items()}
    for view,other in (('r1','r2'),('r2','r1')):
        for name,s in scores[view].items():out['pair'][f'IGNORE_{other}__{name}']=s
    return out


def reader_kind(name):return 'defense' if name.startswith('DA_') else 'representation' if name.startswith('REP_') else 'output' if name.startswith('OUTPUT_') else 'full'


def fit_reader(x,y,name,seed):
    scaler=FiniteScaler().fit(x);xx=scaler.transform(x)
    if 'MLP' in name:model=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',solver='adam',alpha=1e-4,max_iter=300,early_stopping=False,random_state=seed)
    elif 'HGB' in name:model=HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=200,learning_rate=.05,early_stopping=False,random_state=seed)
    else:model=LogisticRegression(C=1,max_iter=500,solver='lbfgs')
    model.fit(xx,y)
    return {'model':model,'scaler':scaler}


def predict_reader(record,x):
    model=record['model']
    if not np.array_equal(model.classes_,[0,1]):raise ValueError('reader class order differs')
    return model.predict_proba(record['scaler'].transform(x))[:,1].astype(np.float64)


def fit_bank(att,inner,*,seed,private_output,queue_guard,bindings=None):
    queue_guard();root=Path(private_output)
    if '.private' not in root.resolve().parts or root.exists():raise PermissionError('fresh private reader directory required')
    root.mkdir(parents=True,mode=0o700);y=np.asarray(att['sex']);train,val=attack_split(att['unit'])
    if np.unique(y[train]).size!=2 or np.unique(y[val]).size!=2:raise ValueError('ATT split lacks both SEX classes')
    tables=feature_tables(att['r1'],att['r2']);inner_tables=feature_tables(inner['r1'],inner['r2'])
    save_npz(root/'ATT_RELEASE.npz',**att,train=train,validation=val);save_npz(root/'INNER_RELEASE.npz',**inner)
    scores={v:{} for v in tables};val_receipt={};model_hashes={};account=[];controls={}
    def fit_save(x,labels,name,view,derived_kind):
        queue_guard();t=time.perf_counter();cpu=time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');record=fit_reader(x,labels,name,MODEL_SEED_BASE+seed)
        path=root/f'MODEL_{view}__{name}.joblib';joblib.dump(record,path,compress=3);path.chmod(0o600);model_hashes[path.name]=sha_file(path)
        account.append({'view':view,'reader':name,'input':derived_kind,'train_rows':len(x),'columns':x.shape[1],'cpu_s':time.process_time()-cpu,'wall_s':time.perf_counter()-t,'scaler_positive_variance_floor':None,'zero_variance_scale':1.,'zero_variance_coordinates':int(np.sum(record['scaler'].raw_scale_==0)),'warnings':[str(w.message) for w in caught]})
        return record
    for view,tab in tables.items():
        val_receipt[view]={}
        for name in READERS:
            kind=reader_kind(name);record=fit_save(tab[kind][train],y[train],name,view,kind)
            score=predict_reader(record,tab[kind][val]);sign=orientation(y[val],score)
            val_receipt[view][name]={'positive_auc':auc(y[val],score),'orientation':sign,'fixed_orientation_auc':auc(y[val],orient(score,sign))}
            scores[view][name]=orient(predict_reader(record,inner_tables[view][kind]),sign)
        shuffled=group_shuffle(y[train],att['unit'][train],ATT_NULL_SEED)
        record=fit_save(tab['full'][train],shuffled,'SHUFFLED',view,'full')
        controls[f'{view}__SHUFFLED']=predict_reader(record,inner_tables[view]['full'])
        planted=np.column_stack((tab['full'][train],y[train]))
        record=fit_save(planted,y[train],'PLANTED',view,'full+engineering_true_SEX_bit')
        for bit in (0,1):controls[f'{view}__PLANTED_{bit}']=predict_reader(record,np.column_stack((inner_tables[view]['full'],np.full(len(inner['unit']),bit))))
    scores=add_ignore_other(scores);flat={f'{view}__{name}':s for view,records in scores.items() for name,s in records.items()};flat.update(controls)
    score_sha=save_npz(root/'INNER_SCORES.npz',**flat)
    receipt={'schema':'TCR1-fresh-continuous-reader-freeze-v1','status':'FROZEN_COMPLETE','source_seed':seed,'bindings':bindings,
             'score_sha256':score_sha,'att_release_sha256':sha_file(root/'ATT_RELEASE.npz'),'inner_release_sha256':sha_file(root/'INNER_RELEASE.npz'),
             'model_files':model_hashes,'readers':list(READERS),'model_seed':MODEL_SEED_BASE+seed,'fit_rows':len(train),'validation_rows':len(val),
             'fit_groups':len(np.unique(att['unit'][train])),'validation_groups':len(np.unique(att['unit'][val])),'numerical_fit_count':len(account),
             'actual_wire_dimensions':{'r1':21,'r2':29,'pair':50},'score_only_dimensions':{'r1':3,'r2':7,'pair':10},
             'wire_order':'z16,centered_own_head_logitsK,own_head_probabilitiesK,numeric_own_head_decision;pair_income_then_occupation',
             'defense_aware_transform':'all_full_wire + log_own_probability; pair adds z_outer_products and centered_logit_outer_products; no extra provider inputs',
             'primary_envelope':'max all nine fixed oriented readers plus every pair local-ignore alias','secondary_envelope':'output LR/HGB and pair output local-ignore aliases',
             'validation':val_receipt,'accounting':account,'INNER_SEX_opened':False,'sklearn_version':__import__('sklearn').__version__,'numpy_version':np.__version__}
    write_json(root/'FROZEN_COMPLETE.json',receipt)
    return receipt


def score_bank(scores,sex,unit):
    y=np.asarray(sex);shuffled=group_shuffle(y,unit,INNER_NULL_SEED);report={}
    for view in ('r1','r2','pair'):
        prefix=view+'__';science={k[len(prefix):]:v for k,v in scores.items() if k.startswith(prefix) and not k[len(prefix):].startswith(('PLANTED','SHUFFLED'))}
        all_auc={name:auc(y,s) for name,s in science.items()}
        output={name:a for name,a in all_auc.items() if name.split('__')[-1] in OUTPUT_READERS}
        rep={name:a for name,a in all_auc.items() if name.split('__')[-1] in REP_READERS}
        control={'null_auc':auc(y,np.full(len(y),.5)),'shuffled_independent_auc':max(auc(shuffled,scores[prefix+'SHUFFLED']),1-auc(shuffled,scores[prefix+'SHUFFLED'])),
                 'planted_auc':auc(y,np.where(y==1,scores[prefix+'PLANTED_1'],scores[prefix+'PLANTED_0']))}
        gates={'null_exact':control['null_auc']==.5,'shuffle_at_most_0.60':control['shuffled_independent_auc']<=.60,'planted_at_least_0.99':control['planted_auc']>=.99}
        report[view]={'reader_auc':all_auc,'envelope_auc':max(all_auc.values()),'strongest_reader':max(all_auc,key=all_auc.get),'output_only_auc':max(output.values()),'representation_only_auc':max(rep.values()),'controls':control,'control_gates':gates,'controls_pass':all(gates.values())}
    if report['pair']['envelope_auc']+1e-12<max(report[v]['envelope_auc'] for v in ('r1','r2')):raise AssertionError('pair ignore envelope monotonicity failed')
    return {'views':report,'all_controls_pass':all(v['controls_pass'] for v in report.values()),'ignore_other_monotonicity':True}


def oracle_diagnostics(scores,sex,unit,task_labels,oracle_scores):
    """True-class conditional AUCs are interpretation-only, never primary inputs."""
    y=np.asarray(sex);g=np.asarray(unit);income=np.asarray(task_labels['income']);occupation=np.asarray(task_labels['occupation']);out={}
    for view,classes,label in (('r1',2,income),('r2',6,occupation),('pair',12,income*6+occupation)):
        prefix=view+'__';science={k[len(prefix):]:s for k,s in scores.items() if k.startswith(prefix) and not k[len(prefix):].startswith(('PLANTED','SHUFFLED'))}
        strata=[]
        for value in range(classes):
            mask=label==value;support=[len(np.unique(g[mask&(y==sex_class)])) for sex_class in (0,1)]
            if not all(support):strata.append({'class_or_pair_id':value,'rows':int(mask.sum()),'SEX_group_counts':support,'status':'ABSENT_SEX_CLASS','auc':None});continue
            reader_auc={name:auc(y[mask],s[mask]) for name,s in science.items()}
            strata.append({'class_or_pair_id':value,'rows':int(mask.sum()),'SEX_group_counts':support,'status':'ESTIMATED_DESCRIPTIVE','auc':max(reader_auc.values()),'reader_auc':reader_auc})
        out[view]={'FIT_trained_true_label_oracle_auc':auc(y,oracle_scores[view]),'conditional_fresh_reader_auc':strata}
    return {'interpretation_only':True,'true_task_labels_absent_primary_inputs':True,'condition':'local true class for recipients, income*6+occupation for pair','views':out}

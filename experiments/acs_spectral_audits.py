"""Fixed spectral-study audits; historical fitters and their objects stay immutable."""
from __future__ import annotations
import copy, json, time
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import pdist
from experiments import acs_fixed_predictions_audits as old
from experiments.acs_transfer_heads import InputStandardizer, metrics, _hash_array, load_candidate
from experiments.acs_transfer_data import sha_file, write_json
from experiments.acs_preservation_audits import fit_extended_auditors, save_extended_audits

ALPHAS=(.0001,.01,1.)

def expected_losses(base, augmented, p):
    if not 0<=p<=1 or np.shape(base)!=np.shape(augmented): raise ValueError('Aligned losses and probability required')
    return (1-p)*np.asarray(base)+p*np.asarray(augmented)

def route_ancestor(view,width_a,columns):
    route=list(range(4)) if view=='A' else [0,1,2,3,width_a,width_a+1] if view=='AB' else [0,1]
    return tuple(route[j] for j in columns) if columns is not None else tuple(route)

class KernelCandidate:
    family='kernel_ridge'
    def __init__(self, preprocessing, omega, phase, center, prior, coef, metadata):
        self.preprocessing=preprocessing; self.omega=omega; self.phase=phase
        self.center=center; self.prior=prior; self.coef=coef; self.metadata=metadata
        self.n_classes=len(prior)
    def predict_proba(self,x):
        f=np.sqrt(2/len(self.phase))*np.cos(self.preprocessing.transform(x)@self.omega+self.phase)
        p=np.clip(self.prior+(f-self.center)@self.coef,1e-12,1.)
        return p/p.sum(1,keepdims=True)

def fit_kernel(xf,yf,xv,yv,k,seed,directory,*,features=256):
    directory=Path(directory)
    if (directory/'complete.json').exists():
        manifest=json.loads((directory/'complete.json').read_text())
        for name,digest in manifest['files'].items(): assert sha_file(directory/name)==digest
        result={str(a):joblib.load(directory/str(a)/'model.joblib') for a in ALPHAS}
        for candidate in result.values():
            meta=candidate.metadata
            assert meta['fit_hashes']=={'x':_hash_array(np.asarray(xf,np.float64)),'y':_hash_array(yf)}
            assert meta['validation_hashes']=={'x':_hash_array(np.asarray(xv,np.float64)),'y':_hash_array(yv)}
            assert meta['seed']==seed and meta['n_classes']==k and meta['parameters']['features']==features
        return result
    tick=time.perf_counter(); xf=np.asarray(xf,np.float64); xv=np.asarray(xv,np.float64)
    prep=InputStandardizer.fit(xf,1e-12); x=prep.transform(xf)
    rng=np.random.default_rng(seed); ix=rng.choice(len(x),min(512,len(x)),replace=False)
    distances=pdist(x[ix]); positive=distances[distances>1e-12]
    # Constant input has no nonlinear distance scale: fixed sigma1 is explicit.
    sigma=float(np.median(positive)) if len(positive) else 1.
    omega=rng.normal(size=(x.shape[1],features))/sigma; phase=rng.uniform(0,2*np.pi,features)
    f=np.sqrt(2/features)*np.cos(x@omega+phase); center=f.mean(0); f-=center
    target=np.eye(k)[yf]; prior=target.mean(0); residual=target-prior
    gram=f.T@f/len(x); cross=f.T@residual/len(x)
    eig,vec=np.linalg.eigh(gram); eig=np.maximum(eig,0); result={}
    for alpha in ALPHAS:
        coef=vec@((vec.T@cross)/(eig[:,None]+alpha))
        error=float(np.mean(np.sum((f@coef-residual)**2,axis=1)))
        meta={'family':'kernel_ridge','input_dim':x.shape[1],'n_classes':k,'seed':seed,
              'fit_rows':len(yf),'validation_rows':len(yv),'fit_support':np.bincount(yf,minlength=k).tolist(),
              'fit_hashes':{'x':_hash_array(xf),'y':_hash_array(yf)},
              'validation_hashes':{'x':_hash_array(xv),'y':_hash_array(yv)},
              'parameters':{'features':features,'alpha':alpha,'bandwidth':sigma,'bandwidth_subset':len(ix),'constant_input':not len(positive)},
              'squared_loss':error,'penalized_objective':error+alpha*float(np.sum(coef**2)),
              'normal_equation_max_error':float(np.max(np.abs((gram+alpha*np.eye(features))@coef-cross))),
              'objective':'mean row-summed onehot squared loss + alpha ||B||F²; centered features, unpenalized prior intercept',
              'probability_conversion':'clip regression coordinates to [1e-12,1], normalize row',
              'selection':'unweighted attacker validation log loss then candidate ID','fit_weighted':False}
        candidate=KernelCandidate(prep,omega,phase,center,prior,coef,meta)
        meta['validation_scores']=metrics(yv,candidate.predict_proba(xv),k)
        path=directory/str(alpha); path.mkdir(parents=True,exist_ok=True)
        write_json(path/'metadata.json',meta); joblib.dump(candidate,path/'model.joblib'); result[str(alpha)]=candidate
    write_json(directory/'complete.json',{'runtime_seconds':time.perf_counter()-tick,'feature_fits':1,'ridge_solves':3,
        'files':{str(p.relative_to(directory)):sha_file(p) for p in directory.glob('*/*') if p.is_file()}})
    return result

def load_base(path):
    p=Path(path); meta=json.loads((p/'metadata.json').read_text())
    return joblib.load(p/'model.joblib') if meta['family']=='kernel_ridge' else load_candidate(p)

def load_audits(directory):
    record=json.loads((Path(directory)/'audit_selection.json').read_text()); cache={}; cs={}
    for b,roles in record['candidates'].items():
        cs[int(b)]={}
        for role,candidates in roles.items():
            cs[int(b)][role]={}
            for cid,meta in candidates.items():
                path=meta['base_candidate_directory']
                if path not in cache:
                    assert sha_file(Path(path)/'metadata.json')==meta['base_metadata_sha256']
                    cache[path]=load_base(path)
                cs[int(b)][role][cid]=old.AuditCandidate(cache[path],meta['space'],tuple(meta['projection_columns']) if meta['projection_columns'] is not None else None,meta)
    return {'candidates':cs,'selection':{int(k):v for k,v in record['selections'].items()},'metadata':record}

def selections(candidates):
    # Retain original scopes, add kernel to separate common scopes.
    original={cid:c for cid,c in candidates.items() if 'kernel__' not in cid}
    chosen,pools=old.select_pools(original)
    common,commonp=old.select_pools(candidates)
    for name in old.SCOPES: chosen['kernel_'+name]=common[name]; pools['kernel_'+name]=commonp[name]
    return chosen,pools

def build_audits(wire,derived,labels,indices,seed,directory,*,historical=None,ancestor=None):
    """Only attacker fitting and validation accepted; no spectral saved observer."""
    assert set(labels)==set(old.FIT_POOLS)
    directory=Path(directory)
    if (directory/'audit_selection.json').exists():
        restored=load_audits(directory)
        for b,roles in restored['candidates'].items():
            for role,current in roles.items():
                view,target=role.split('/'); ix=indices[target]; valid=labels['attacker_validation'][target]>=0
                for c in current.values():
                    space=wire if c.space=='wire' else derived
                    xf=space[view]['attacker_fit'][ix]; xv=space[view]['attacker_validation'][valid]
                    if c.columns is not None: xf=xf[:,c.columns]; xv=xv[:,c.columns]
                    meta=c.metadata
                    if meta.get('fit_hashes'):
                        assert meta['fit_hashes']=={'x':_hash_array(xf),'y':_hash_array(labels['attacker_fit'][target][ix])}
                    assert meta['validation_hashes']=={'x':_hash_array(xv),'y':_hash_array(labels['attacker_validation'][target][valid])}
        return restored
    directory.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    spaces={'wire':wire}
    if derived is not None: spaces['derived']=derived
    for space,views in spaces.items():
        for pool in old.FIT_POOLS:
            assert all(views[v][pool].dtype==np.float64 for v in old.VIEWS)
            assert np.array_equal(views['AB'][pool],np.column_stack((views['A'][pool],views['B'][pool])))
    cs={b:{v+'/'+t:{} for v in old.VIEWS for t in old.AUDIT_ROLES[v]} for b in (120,360)}
    if historical:
        for b,roles in historical['candidates'].items():
            for role,candidates in roles.items(): cs[b][role].update(candidates)
    counts={'new_five_candidate_roles':0,'fresh_mlp_trajectories':0,'static_fits':0,'kernel_feature_fits':0,'kernel_ridge_solves':0,'own_catchup_trajectories':0}
    for view in old.VIEWS:
        for target in old.AUDIT_ROLES[view]:
            if view=='B' and ancestor is not None: continue
            role=view+'/'+target; ix=indices[target]; valid=labels['attacker_validation'][target]>=0
            yf=labels['attacker_fit'][target][ix]; yv=labels['attacker_validation'][target][valid]; k=old.CLASSES[target]
            for space,views in spaces.items():
                if view=='B' and space=='derived':continue
                xf=views[view]['attacker_fit'][ix]; xv=views[view]['attacker_validation'][valid]
                if not historical:
                    assert space=='wire'
                    path=directory/'fitted'/space/view/target/'fresh'
                    if not (path/'unit_complete.json').exists():
                        if path.exists():
                            # Preserve incomplete serialized role; no completed role is overwritten.
                            quarantine=path.with_name('incomplete_'+str(time.time_ns()))
                            path.rename(quarantine)
                            write_json(quarantine/'recovery.json',{'reason':'incomplete role serialization','counted_partial_attempt':True,'preserved':True})
                        static=old.fit_static_auditors(xf,yf,xv,yv,k,old.role_seed(seed,view,target))
                        r=fit_extended_auditors(xf,yf,xv,yv,k,old.role_seed(seed,view,target),static_candidates=static,epochs=360,nested_epochs=120)
                        save_extended_audits(r,path)
                        write_json(path/'unit_complete.json',{'complete':True,'mlp_trajectories':2,'static_fits':3,'files':{str(p.relative_to(path)):sha_file(p) for p in path.rglob('*') if p.is_file()}})
                    for name,digest in json.loads((path/'unit_complete.json').read_text())['files'].items():assert sha_file(path/name)==digest
                    for tag,b in [('nested120',120),('nested360',360)]:
                        for cid in old.FRESH:
                            p=path/tag/cid; base=load_candidate(p)
                            assert base.metadata['fit_hashes']=={'x':_hash_array(xf),'y':_hash_array(yf)}
                            assert base.metadata['validation_hashes']=={'x':_hash_array(xv),'y':_hash_array(yv)}
                            assert base.metadata['seed']==old.role_seed(seed,view,target)+(10000 if cid=='mlp_1' else 0)
                            if cid.startswith('mlp'):assert base.metadata['parameters']['epochs']==b
                            cs[b][role][space+'__'+cid]=old._wrapped(base,space+'__'+cid,space,view,target,b,p)
                    counts['new_five_candidate_roles']+=1; counts['fresh_mlp_trajectories']+=2; counts['static_fits']+=3
                path=directory/'fitted'/space/view/target/'kernel'
                rngseed=20263910+100*seed+10*old.VIEWS.index(view)+old.TARGETS.index(target)
                kernels=fit_kernel(xf,yf,xv,yv,k,rngseed,path)
                for a,base in kernels.items():
                    cid=space+'__kernel__'+a
                    for b in cs: cs[b][role][cid]=old._wrapped(base,cid,space,view,target,b,path/a)
                counts['kernel_feature_fits']+=1;counts['kernel_ridge_solves']+=3
    if ancestor is not None:
        for b,roles in cs.items():
            for role,current in roles.items():
                view,target=role.split('/')
                if view=='B':
                    # Canonical B candidate identities shared unchanged in every system.
                    current.clear();current.update(ancestor['candidates'][b][role]);continue
                for cid,c in ancestor['candidates'][b][role].items():
                    cols=route_ancestor(view,wire['A']['attacker_fit'].shape[1],c.columns)
                    ident='anchor__'+cid; meta=copy.deepcopy(c.metadata)
                    meta.update(candidate_id=ident,view=view,projection_columns=list(cols),source_condition='H',anchor_ancestor=True,inherited_singleton=False)
                    current[ident]=old.AuditCandidate(c.base,'wire',cols,meta)
    widths={s:(v['A']['attacker_fit'].shape[1],v['B']['attacker_fit'].shape[1]) for s,v in spaces.items()}
    select={};pools={};record={}; parity=[]
    for b,roles in cs.items():
        old.inherit_singletons(roles,widths);select[b]={};pools[b]={};record[b]={}
        for role,current in roles.items():
            view,target=role.split('/');valid=labels['attacker_validation'][target]>=0
            bundle={s:v[view]['attacker_validation'][valid] for s,v in spaces.items()}
            for cid,c in current.items():
                # Verify every inherited route; fresh validation was scored by fitter.
                if c.columns is not None:
                    actual=metrics(labels['attacker_validation'][target][valid],c.predict_proba(bundle),old.CLASSES[target])
                    assert actual==c.metadata['validation_scores'],(role,cid)
                    parity.append([b,role,cid])
            select[b][role],pools[b][role]=selections(current)
            record[b][role]={cid:c.metadata for cid,c in current.items()}
    meta={'seed':seed,'selections':select,'selection_pools':pools,'candidates':record,'counts':counts,
          'runtime_seconds':time.perf_counter()-start,'development_received':False,'projection_checks':parity,
          'spectral_saved_observer':'not applicable; no trained observer','historical_catchup_retained':True}
    write_json(directory/'audit_selection.json',meta)
    return {'candidates':cs,'selection':select,'metadata':meta}

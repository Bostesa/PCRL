"""Independent historical complete-code-wire finite-law replay.

Shared bank/schema definitions only. No conditional_release scientific module
is imported. This service remains separate from the new continuous full wire.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scipy import sparse
from .readers import load_saved_reader,split

LEGACY_READERS=('LR_C1','MLP_64x32','HGB_15x200','DA_MLP_64x32','CELL_A1','CODE_BIT_LR_C1','PROB_LR_C1','PROB_HGB_15x200')

def tables(bank,prior):
    prob=[np.asarray(bank['recipients'][str(r)]['probability'],dtype=np.float64) for r in (1,2)]
    result={};bits=[]
    for r,p in enumerate(prob):
        n,k=p.shape;width=(n-1).bit_length();codes=np.arange(n)
        binary=((codes[:,None]>>np.arange(max(1,width)))&1).astype(float);bits.append(binary)
        full=np.column_stack((np.eye(n),binary,p,np.eye(k)[p.argmax(1)],np.full(n,r+1),np.full(n,width)))
        marginal=prior.sum(axis=1 if r==0 else 0)
        result[f'r{r+1}']={'x':full,'bits':binary,'probability':p,
            'defense':np.column_stack((full,np.log(np.maximum(p,1e-12)),np.log(np.maximum(marginal,1e-12))))}
    left=np.repeat(np.arange(len(prob[0])),len(prob[1]));right=np.tile(np.arange(len(prob[1])),len(prob[0]))
    result['pair']={name:np.column_stack((result['r1'][name][left],result['r2'][name][right])) for name in ('x','bits','probability')}
    product=np.einsum('ni,nj->nij',bits[0][left],bits[1][right]).reshape(len(left),-1)
    result['pair']['defense']=np.column_stack((result['r1']['defense'][left],result['r2']['defense'][right],product,np.log(np.maximum(prior.ravel(),1e-12))))
    return result,left,right

def law_check(law,bank,sex_record):
    if set(law)!={'row_id','unit','code_r1','code_r2','mass'}:raise ValueError('legacy law fields differ')
    if not all(np.array_equal(law[k],sex_record[k]) for k in ('row_id','unit')):raise ValueError('legacy people differ')
    a,b,mass=law['code_r1'],law['code_r2'],law['mass']
    sizes=[len(bank['recipients'][str(r)]['probability']) for r in (1,2)]
    if a.ndim!=2 or b.shape!=a.shape or mass.shape!=a.shape or not np.isfinite(mass).all() or (mass<0).any() or not np.allclose(mass.sum(1),1,atol=1e-12,rtol=0):raise ValueError('legacy finite law invalid')
    if (a<0).any() or (a>=sizes[0]).any() or (b<0).any() or (b>=sizes[1]).any():raise ValueError('legacy public code IDs out of range')
    return {'r1':a,'r2':b,'pair':a*sizes[1]+b},mass

def aggregate_codes(index,mass,sex,group_index,group_count,size):
    rows=np.repeat(group_index,index.shape[1]);cols=index.ravel();weight=mass.ravel()
    label=np.repeat(sex,index.shape[1]);matrices=[]
    for s in (1,0):
        keep=(weight>0)&(label==s)
        matrix=sparse.coo_matrix((weight[keep],(rows[keep],cols[keep])),shape=(group_count,size)).tocsr()
        matrix.sum_duplicates();matrices.append(matrix)
    return matrices

def table_auc(score,positive,negative):
    score=np.asarray(score);order=np.argsort(score,kind='stable');s=score[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    p=np.add.reduceat(np.asarray(positive)[...,order],starts,axis=-1)
    n=np.add.reduceat(np.asarray(negative)[...,order],starts,axis=-1)
    denominator=p.sum(-1)*n.sum(-1)
    if (denominator<=0).any():raise ValueError('legacy weighted AUC lacks a SEX class')
    return np.sum(p*(np.cumsum(n,axis=-1)-.5*n),axis=-1)/denominator

def shuffle(sex,groups):
    salt='TCR1|legacy-INNER-null|2026101207'
    seed=int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16],'big')
    rng=np.random.Generator(np.random.PCG64DXSM(seed));out=np.empty_like(sex);buckets={}
    for g in np.unique(groups):
        indices=np.flatnonzero(groups==g);buckets.setdefault(len(indices),[]).append(indices)
    for size in sorted(buckets):
        rows=buckets[size]
        for dst,src in zip(rows,rng.permutation(len(rows))):out[dst]=sex[rows[src]]
    return out

def replay(entries,sex_record,checked):
    if len(entries)!=4 or {(e['family'],e['seed']) for e in entries}!={(f,s) for f in ('P0','S12') for s in (0,1)}:raise ValueError('complete four legacy bank scope required')
    groups=sex_record['unit'];sex=sex_record['sex'];unique,first,inverse=np.unique(groups,return_index=True,return_inverse=True);g=len(unique)
    draws=np.random.default_rng(2026101208).integers(0,g,size=(2048,g));counts=np.zeros((2048,g),dtype=np.int32)
    for i,row in enumerate(draws):counts[i]=np.bincount(row,minlength=g)
    def npz(pin):
        with np.load(checked(pin),allow_pickle=False) as z:return {k:z[k] for k in z.files}
    points={};boots={};reports=[]
    for entry in entries:
        root=Path(entry['directory']);freeze=json.loads(checked({'path':str(root/'FROZEN_COMPLETE.json'),'sha256':entry['receipt_sha256']}).read_text())
        if freeze['status']!='FROZEN_COMPLETE' or freeze['source_seed']!=entry['seed']:raise ValueError('legacy reader freeze differs')
        bank=json.loads(checked(entry['bank']).read_text());law=npz(entry['inner_law']);indexes,mass=law_check(law,bank,sex_record)
        frozen=npz({'path':str(root/'FROZEN_SCORE_TABLES.npz'),'sha256':freeze['score_table_sha256']})
        prior=npz({'path':str(root/'PUBLIC_UNIFORM_PRIOR.npz'),'sha256':freeze['public_uniform_prior_sha256']})['pair']
        feature,left,right=tables(bank,prior)
        training=npz({'path':str(root/'TRAINING_INPUTS.npz'),'sha256':freeze['input_sha256']})
        tt={'row_id':training['row_id'],'unit':training['unit'],'sex':training['sex']}
        train_law={k:training[k] for k in ('row_id','unit','code_r1','code_r2','mass')}
        train_index,train_mass=law_check(train_law,bank,tt);fit,tune=split(training['unit'])
        if not np.array_equal(fit,training['fit']) or not np.array_equal(tune,training['tune']):raise ValueError('legacy source ATT split differs')
        error=0.;views={}
        for view in ('r1','r2','pair'):
            size=len(feature[view]['x']);cell=np.zeros((size,2))
            np.add.at(cell,(train_index[view][fit].ravel(),np.repeat(training['sex'][fit],train_index[view].shape[1])),train_mass[fit].ravel())
            tune_cell=np.zeros((size,2));np.add.at(tune_cell,(train_index[view][tune].ravel(),np.repeat(training['sex'][tune],train_index[view].shape[1])),train_mass[tune].ravel())
            for name in LEGACY_READERS:
                file=f'MODEL_{view}__{name}.joblib';record=load_saved_reader(checked({'path':str(root/file),'sha256':freeze['model_files'][file]}),validate=False)
                if name=='CELL_A1':
                    baseline=(cell[:,1].sum()+1)/(cell.sum()+2);pred=np.full(size,baseline);seen=cell.sum(1)>0;pred[seen]=(cell[seen,1]+1)/(cell[seen].sum(1)+2)
                    if not np.allclose(record['counts'],cell,atol=1e-10,rtol=0):raise ValueError('legacy CELL weighted counts differ')
                else:
                    kind='defense' if name.startswith('DA_') else 'bits' if name=='CODE_BIT_LR_C1' else 'probability' if name.startswith('PROB_') else 'x'
                    z=feature[view][kind];scaler=record['scaler'];model=record['model']
                    if not np.array_equal(model.classes_,[0,1]):raise ValueError('legacy reader classes differ')
                    if scaler is not None:
                        wires,labels=np.nonzero(cell>0);weights=cell[wires,labels]
                        mean=np.average(z[wires],axis=0,weights=weights)
                        sd=np.sqrt(np.average((z[wires]-mean)**2,axis=0,weights=weights))
                        if not np.array_equal(scaler.mean_,mean) or not np.allclose(scaler.scale_,np.where(sd>0,sd,1.),atol=0,rtol=1e-12):raise ValueError('legacy weighted positive-SD scaler law differs')
                    pred=model.predict_proba(z if scaler is None else (z-scaler.mean_)/scaler.scale_)[:,1]
                orientation=1 if float(table_auc(pred,tune_cell[:,1],tune_cell[:,0]))>=.5 else -1
                if orientation==-1:pred=1-pred
                difference=float(np.max(abs(pred-frozen[f'{view}__{name}'])))
                if difference>1e-10:raise ValueError('legacy saved reader/table replay differs')
                error=max(error,difference)
            bank_scores={name[len(view)+2:]:v for name,v in frozen.items() if name.startswith(view+'__') and name[len(view)+2:] not in ('SHUFFLED','PLANTED')}
            matrices=aggregate_codes(indexes[view],mass,sex,inverse,g,size)
            pos=np.asarray(matrices[0].sum(axis=0)).ravel();neg=np.asarray(matrices[1].sum(axis=0)).ravel()
            reader_auc={name:float(table_auc(s,pos,neg)) for name,s in bank_scores.items()};point=max(reader_auc.values());rep=np.empty(2048)
            for start in range(0,2048,64):
                stop=min(start+64,2048);p=np.asarray(counts[start:stop]@matrices[0]);n=np.asarray(counts[start:stop]@matrices[1])
                rep[start:stop]=np.max(np.stack([table_auc(s,p,n) for s in bank_scores.values()]),axis=0)
            # Exact finite-law null, independently shuffled and planted probes.
            sh=shuffle(sex,groups);sm=aggregate_codes(indexes[view],mass,sh,inverse,g,size)
            sp=np.asarray(sm[0].sum(axis=0)).ravel();sn=np.asarray(sm[1].sum(axis=0)).ravel();sauc=float(table_auc(frozen[f'{view}__SHUFFLED'],sp,sn))
            planted=frozen[f'{view}__PLANTED'];combined_score=np.r_[planted[:,0],planted[:,1]]
            planted_auc=float(table_auc(combined_score,np.r_[np.zeros(size),pos],np.r_[neg,np.zeros(size)]))
            control={'null_auc':float(table_auc(np.zeros(size),pos,neg)),'shuffled_auc':max(sauc,1-sauc),'planted_auc':planted_auc}
            control['pass']=control['null_auc']==.5 and control['shuffled_auc']<=.6 and control['planted_auc']>=.99
            views[view]={'reader_auc':reader_auc,'envelope_auc':point,'controls':control};points[(entry['family'],entry['seed'],view)]=point;boots[(entry['family'],entry['seed'],view)]=rep
        for local,code,other in [('r1',left,'R2'),('r2',right,'R1')]:
            for name in LEGACY_READERS:
                if not np.array_equal(frozen[f'pair__IGNORE_{other}__{name}'],frozen[f'{local}__{name}'][code]):raise ValueError('legacy ignore-other alias differs')
        if views['pair']['envelope_auc']+1e-12<max(views[v]['envelope_auc'] for v in ('r1','r2')):raise ValueError('legacy coalition envelope not monotone')
        reports.append({'family':entry['family'],'seed':entry['seed'],'reader_prediction_max_abs_error':error,'readers_replayed':24,
            'views':views,'contract':'literal legacy complete code/probability/decision/metadata wire; separate from new full features'})
    return points,boots,reports

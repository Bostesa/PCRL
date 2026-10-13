"""Standalone raw83 runtime: actual encoder, own head, optional fixed head projection.

No source loaders, training models, teacher scores or labels are imported. The
projection transmits Pz16 only; rank coordinates remain internal.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

SCHEMA='OB1-raw83-own-head-deployment-v1'
STATE_KEYS={f'enc.{i}.{layer}.{kind}' for i in range(2) for layer in (0,2,4) for kind in ('weight','bias')}|{f'head.{i}.{kind}' for i in range(2) for kind in ('weight','bias')}
CONTRACT={'recipient_full_dimensions':[21,29],'pair_full_dimensions':50,'recipient_score_dimensions':[3,7],'pair_score_dimensions':10,'deterministic':True,'head_reads_released_representation':True}


def sha_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def schema_hash(names):return hashlib.sha256(json.dumps(names,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def validate_names(names):
    if not isinstance(names,(tuple,list)) or len(names)!=83 or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=83:raise ValueError('exact ordered83 named columns required')
    forbidden={'sex','race','income','occupation','fnlwgt','row_id','unit','id','person_id','sampling_id','group_id','hidden_flag','y','label','target','teacher','teacher_score'}
    if any(n.lower().split('=')[0] in forbidden for n in names):raise ValueError('forbidden deployment feature')
    return list(names)


class RuntimeModel(torch.nn.Module):
    def __init__(self,names,projection=None):
        super().__init__();self.names=tuple(validate_names(names));self.projection=projection
        self.enc=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(83,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,16)) for _ in range(2)])
        self.head=torch.nn.ModuleList([torch.nn.Linear(16,k,dtype=torch.float64) for k in (2,6)])

    def release(self,raw,names,**extra):
        if extra:raise ValueError('unregistered inputs or output channels refused')
        if tuple(validate_names(names))!=self.names:raise ValueError('wrong or reordered input columns')
        x=np.asarray(raw)
        if x.ndim!=2 or x.shape[1]!=83 or x.dtype.kind!='f' or not np.isfinite(x).all():raise ValueError('finite floating raw83 required')
        with np.errstate(over='ignore'):x=np.ascontiguousarray(x,dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('raw83 overflows encoder precision')
        views=[]
        with torch.inference_mode():
            for i,(encoder,head) in enumerate(zip(self.enc,self.head)):
                z=encoder(torch.from_numpy(x))
                if self.projection is None:
                    released=z.to(torch.float64);logits=head(released);logits-=logits.mean(1,keepdim=True);p=torch.softmax(logits,1);d=p.argmax(1)
                    r= released.numpy();l=logits.numpy();prob=p.numpy();decision=d.numpy()
                else:
                    # Preserve geometry.py's frozen NumPy arithmetic exactly.
                    basis=self.projection[i]['B'];r=(z.numpy().astype(np.float64)@basis.T)@basis
                    l=r@head.weight.detach().numpy().T+head.bias.detach().numpy();l-=l.mean(1,keepdims=True)
                    e=np.exp(l-l.max(1,keepdims=True));prob=e/e.sum(1,keepdims=True);decision=prob.argmax(1)
                full=np.column_stack((r,l,prob,decision))
                if not np.isfinite(full).all():raise ValueError('nonfinite release refused')
                views.append({'representation':r,'centered_logits':l,'probabilities':prob,'decision':decision,'full':full,'output_only':np.column_stack((prob,decision))})
        return views


def load_package(directory,expected_manifest_sha256):
    root=Path(directory);mfile=root/'DEPLOYMENT.json';weights=root/'model_state.pt'
    if not expected_manifest_sha256 or any(p.is_symlink() for p in (root,*root.parents,mfile,weights)) or sha_file(mfile)!=expected_manifest_sha256:raise ValueError('manifest pin differs')
    m=json.loads(mfile.read_text());fields={'schema','feature_names','feature_schema_sha256','class_orders','model_sha256','source_checkpoint_sha256','provenance','contract','parameter_count','projection'}
    if set(m)!=fields or m['schema']!=SCHEMA or m['class_orders']!=[[0,1],list(range(6))] or m['parameter_count']!=21288 or m['contract']!=CONTRACT:raise ValueError('deployment schema or contract differs')
    names=validate_names(m['feature_names'])
    if schema_hash(names)!=m['feature_schema_sha256'] or sha_file(weights)!=m['model_sha256']:raise ValueError('model or schema pin differs')
    state=torch.load(weights,map_location='cpu',weights_only=True)
    if set(state)!=STATE_KEYS or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() or v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) for k,v in state.items()):raise ValueError('model state differs')
    projection=None
    if m['projection'] is not None:
        meta=m['projection'];file=root/'projection.npz'
        if set(meta)!={'sha256','rule','ranks','representation_dimension','release','head_binding_sha256'} or meta['head_binding_sha256']!=m['model_sha256'] or meta['release']!='factorized_Pz16' or meta['representation_dimension']!=16 or file.is_symlink() or sha_file(file)!=meta['sha256']:raise ValueError('projection binding differs')
        with np.load(file,allow_pickle=False) as z:
            if set(z.files)!={'B0','P0','B1','P1'}:raise ValueError('projection fields differ')
            projection=[]
            for i in range(2):
                b=z[f'B{i}'].copy();p=z[f'P{i}'];w=state[f'head.{i}.weight'].numpy();a=w-w.mean(0,keepdims=True)
                if not isinstance(meta['ranks'][i],int) or not 0<=meta['ranks'][i]<=len(w)-1 or b.dtype!=np.float64 or b.shape!=(meta['ranks'][i],16) or p.dtype!=np.float64 or p.shape!=(16,16) or not np.isfinite(b).all() or not np.array_equal(b.T@b,p) or not np.allclose(b@b.T,np.eye(len(b)),atol=1e-10,rtol=1e-10) or not np.allclose(a@p,a,atol=1e-10,rtol=1e-10):raise ValueError('invalid own-head projection')
                projection.append({'B':b})
    model=RuntimeModel(names,projection);model.load_state_dict(state,strict=True);model.eval();return model,m


def export_checkpoint(checkpoint,checkpoint_sha256,directory,variant='full'):
    if sha_file(checkpoint)!=checkpoint_sha256:raise ValueError('checkpoint hash differs')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('schema')!='TCR1-trainable-own-head-checkpoint-v1' or payload.get('class_orders')!=[[0,1],list(range(6))]:raise ValueError('checkpoint schema differs')
    names=validate_names(payload['feature_names']);state=payload['model'];model=RuntimeModel(names)
    if set(state)!=STATE_KEYS or any(v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('encoder/head state differs')
    model.load_state_dict(state,strict=True)
    root=Path(directory);root.mkdir(parents=True,exist_ok=False,mode=0o700);weights=root/'model_state.pt';torch.save(state,weights);weights.chmod(0o600);model_sha=sha_file(weights);projection=None
    if variant=='head_row_space':
        from .geometry import build_geometry,RULE
        g=[build_geometry(state[f'head.{i}.weight'].numpy(),state[f'head.{i}.bias'].numpy(),payload['class_orders'][i]) for i in range(2)]
        file=root/'projection.npz';np.savez(file,**{f'{key}{i}':v[key] for i,v in enumerate(g) for key in ('B','P')});file.chmod(0o600)
        projection={'sha256':sha_file(file),'rule':RULE,'ranks':[v['rank'] for v in g],'representation_dimension':16,'release':'factorized_Pz16','head_binding_sha256':model_sha}
    elif variant!='full':raise ValueError('unregistered variant')
    m={'schema':SCHEMA,'feature_names':names,'feature_schema_sha256':schema_hash(names),'class_orders':payload['class_orders'],'model_sha256':model_sha,'source_checkpoint_sha256':checkpoint_sha256,'provenance':{k:payload[k] for k in ('family','seed','strength','coefficient','epoch') if k in payload},'parameter_count':sum(v.numel() for v in state.values()),'contract':CONTRACT,'projection':projection}
    with os.fdopen(os.open(root/'DEPLOYMENT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(m,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return sha_file(root/'DEPLOYMENT.json')


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--package',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--input',required=True);p.add_argument('--input-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    model,m=load_package(a.package,a.manifest_sha256)
    if sha_file(a.input)!=a.input_sha256:raise ValueError('input pin differs')
    with np.load(a.input,allow_pickle=False) as z:
        if set(z.files)!={'X','feature_names'}:raise ValueError('feature-only raw83 input required')
        v=model.release(z['X'],z['feature_names'].tolist())
    with os.fdopen(os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:np.savez(f,r1=v[0]['full'],r2=v[1]['full']);f.flush();os.fsync(f.fileno())
    print(json.dumps({'schema':SCHEMA,'rows':len(v[0]['full']),'output_sha256':sha_file(a.output),'dimensions':[21,29,50]}))

if __name__=='__main__':main()

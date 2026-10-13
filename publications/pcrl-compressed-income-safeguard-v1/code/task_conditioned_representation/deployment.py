"""Standalone deterministic raw83 deployment of saved NEW encoders and own heads.

This runtime imports no source admission, historical service or training code.
Exported packages contain only ordered schema, model tensors and provenance hashes.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'TCR1-raw83-own-head-deployment-v1'
STATE_KEYS = {f'enc.{i}.{layer}.{kind}' for i in range(2) for layer in (0,2,4) for kind in ('weight','bias')} | {f'head.{i}.{kind}' for i in range(2) for kind in ('weight','bias')}


def sha_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def schema_hash(names):
    return hashlib.sha256(json.dumps(names,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def validate_names(names):
    if not isinstance(names,(list,tuple)) or len(names)!=83 or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=83:raise ValueError('exact ordered 83 named columns required')
    forbidden={'sex','race','income','occupation','fnlwgt','row_id','unit','person_id','sampling_id','group_id'}
    if any(n.lower().split('=')[0] in forbidden for n in names):raise ValueError('forbidden deployment feature')
    return list(names)


class RuntimeModel(torch.nn.Module):
    def __init__(self,names):
        super().__init__();self.names=tuple(validate_names(names))
        self.enc=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(83,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,16)) for _ in range(2)])
        self.head=torch.nn.ModuleList([torch.nn.Linear(16,k,dtype=torch.float64) for k in (2,6)])

    def release(self,raw,names,**extra):
        if extra:raise ValueError('unregistered inputs or output channels refused')
        if tuple(validate_names(names))!=self.names:raise ValueError('wrong or reordered input columns')
        a=np.asarray(raw)
        if a.ndim!=2 or a.shape[1]!=83 or a.dtype.kind!='f' or not np.isfinite(a).all():raise ValueError('finite floating raw83 matrix required')
        with np.errstate(over='ignore'):a=np.ascontiguousarray(a,dtype=np.float32)
        if not np.isfinite(a).all():raise ValueError('raw83 overflows encoder precision')
        x=torch.from_numpy(a);views=[]
        with torch.inference_mode():
            for enc,head in zip(self.enc,self.head):
                z=enc(x);logits=head(z.to(torch.float64));logits=logits-logits.mean(1,keepdim=True)
                p=torch.softmax(logits,1);d=p.argmax(1)
                full=torch.cat((z.to(torch.float64),logits,p,d[:,None].to(torch.float64)),1)
                output=torch.cat((p,d[:,None].to(torch.float64)),1)
                if not torch.isfinite(full).all():raise ValueError('nonfinite release refused')
                views.append({'representation':z.numpy(),'centered_logits':logits.numpy(),'probabilities':p.numpy(),'decision':d.numpy(),'full':full.numpy(),'output_only':output.numpy()})
        return views


def load_package(directory,expected_manifest_sha256):
    root=Path(directory);manifest=root/'DEPLOYMENT.json';weights=root/'model_state.pt'
    if not expected_manifest_sha256 or manifest.is_symlink() or weights.is_symlink() or sha_file(manifest)!=expected_manifest_sha256:raise ValueError('deployment manifest binding differs')
    m=json.loads(manifest.read_text())
    fields={'schema','feature_names','feature_schema_sha256','class_orders','model_sha256','source_checkpoint_sha256','provenance','contract','parameter_count'}
    if set(m)!=fields or m['schema']!=SCHEMA or m['class_orders']!=[[0,1],list(range(6))] or m['parameter_count']!=21288:raise ValueError('model/schema/class orders differ')
    names=validate_names(m['feature_names'])
    if schema_hash(names)!=m['feature_schema_sha256'] or sha_file(weights)!=m['model_sha256']:raise ValueError('schema/model hash differs')
    if m['contract']!={'recipient_full_dimensions':[21,29],'pair_full_dimensions':50,'recipient_score_dimensions':[3,7],'pair_score_dimensions':10,'deterministic':True,'head_reads_released_representation':True}:raise ValueError('hidden outputs or release contract differs')
    state=torch.load(weights,map_location='cpu',weights_only=True)
    if set(state)!=STATE_KEYS or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() or v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) for k,v in state.items()):raise ValueError('invalid deployment state')
    model=RuntimeModel(names);model.load_state_dict(state,strict=True);model.eval()
    return model,m


def export_checkpoint(checkpoint,checkpoint_sha256,directory):
    if sha_file(checkpoint)!=checkpoint_sha256:raise ValueError('selected checkpoint hash differs')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('schema')!='TCR1-trainable-own-head-checkpoint-v1' or payload.get('class_orders')!=[[0,1],list(range(6))]:raise ValueError('selected checkpoint schema differs')
    names=validate_names(payload['feature_names']);state=payload['model'];model=RuntimeModel(names)
    if set(state)!=STATE_KEYS or any(v.dtype!=(torch.float32 if k.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('selected encoder/head state differs')
    model.load_state_dict(state,strict=True)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    weights=directory/'model_state.pt';torch.save(state,weights);os.chmod(weights,0o600)
    provenance={k:payload[k] for k in ('family','seed','strength','coefficient','epoch','updates_per_recipient') if k in payload}
    m={'schema':SCHEMA,'feature_names':names,'feature_schema_sha256':schema_hash(names),'class_orders':payload['class_orders'],'model_sha256':sha_file(weights),'source_checkpoint_sha256':checkpoint_sha256,'provenance':provenance,'parameter_count':sum(v.numel() for v in state.values()),'contract':{'recipient_full_dimensions':[21,29],'pair_full_dimensions':50,'recipient_score_dimensions':[3,7],'pair_score_dimensions':10,'deterministic':True,'head_reads_released_representation':True}}
    manifest=directory/'DEPLOYMENT.json'
    with os.fdopen(os.open(manifest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(m,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return sha_file(manifest)


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--package',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--input',required=True);p.add_argument('--input-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--contract',choices=('full','output_only'),default='full');a=p.parse_args(argv)
    torch.set_num_threads(1)
    model,m=load_package(a.package,a.manifest_sha256)
    if sha_file(a.input)!=a.input_sha256:raise ValueError('input hash differs')
    with np.load(a.input,allow_pickle=False) as z:
        if set(z.files)!={'X','feature_names'}:raise ValueError('inference archive must contain only X and feature_names')
        views=model.release(z['X'],z['feature_names'].tolist())
    with os.fdopen(os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:np.savez(f,recipient1=views[0][a.contract],recipient2=views[1][a.contract]);f.flush();os.fsync(f.fileno())
    print(json.dumps({'schema':SCHEMA,'rows':len(views[0]['full']),'contract':a.contract,'output_sha256':sha_file(a.output),'model_sha256':m['model_sha256']},sort_keys=True))


if __name__=='__main__':main()

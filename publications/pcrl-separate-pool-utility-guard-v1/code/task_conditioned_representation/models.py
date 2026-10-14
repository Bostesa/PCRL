"""Actual recipient wire and private trainable checkpoint restoration."""
from pathlib import Path
import hashlib
import torch
from .useful_service import UsefulModel


def own_wire(z, head, class_order, **extra):
    if extra:raise ValueError('output bypass or unregistered channel refused')
    if z.ndim!=2 or z.shape[1]!=head.in_features:raise ValueError('actual representation shape differs')
    logits=head(z.to(head.weight.dtype));logits=logits-logits.mean(1,keepdim=True)
    probability=torch.softmax(logits,1)
    order=torch.tensor(class_order,device=z.device,dtype=torch.long)
    if len(order)!=logits.shape[1]:raise ValueError('class order differs')
    decision=order[probability.argmax(1)]
    return {'representation':z,'centered_logits':logits,'probabilities':probability,
            'decision':decision,'full':torch.cat((z.to(logits.dtype),logits,probability,decision[:,None].to(logits.dtype)),1),
            'output_only':torch.cat((probability,decision[:,None].to(logits.dtype)),1)}


def actual_wires(model,x):
    return [own_wire(model.encode(i,x),model.head[i],model.class_orders[i]) for i in range(2)]


def load_checkpoint(path,expected_sha256=None):
    p=Path(path)
    if expected_sha256 and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=expected_sha256:raise ValueError('checkpoint hash differs')
    payload=torch.load(p,map_location='cpu',weights_only=False)
    if payload.get('schema')!='TCR1-trainable-own-head-checkpoint-v1':raise ValueError('checkpoint schema differs')
    if len(payload['feature_names'])!=83 or len(set(payload['feature_names']))!=83 or payload['class_orders']!=[[0,1],list(range(6))]:raise ValueError('model/schema/class order mismatch')
    model=UsefulModel(payload['feature_names']);model.load_state_dict(payload['model'],strict=True)
    model.class_orders=payload['class_orders'];model.source_bindings=payload['source_bindings'];model.eval()
    return model,payload

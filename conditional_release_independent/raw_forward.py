"""Independent frozen raw83 encoder, task heads, and fine-cell routing.

Only third-party numerical libraries are used. No source deployment loader
is invoked, so COMPLETE's all-role archive is never opened or hashed.
"""
from __future__ import annotations
import json
import numpy as np
from .reconstruction import digest


def frozen_raw_probabilities(pins, raw):
    import torch
    import joblib
    torch.set_num_threads(1)
    if np.asarray(raw).ndim!=2 or np.asarray(raw).shape[1]!=83:
        raise ValueError('raw83 required')
    for name in ('model','head0','head1','record'):
        if digest(pins[name]['path'])!=pins[name]['sha256']:
            raise PermissionError('frozen model/head source mismatch')
    state=torch.load(pins['model']['path'],map_location='cpu',weights_only=True)
    x=torch.from_numpy(np.asarray(raw,dtype=np.float32))
    probabilities=[]
    with torch.no_grad():
        for recipient in range(2):
            hidden=x
            for layer in (0,2,4):
                hidden=torch.nn.functional.linear(hidden,state[f'enc.{recipient}.{layer}.weight'],state[f'enc.{recipient}.{layer}.bias'])
                if layer!=4:
                    hidden=torch.relu(hidden)
            latent=hidden.to(torch.float64).numpy()
            head=joblib.load(pins[f'head{recipient}']['path'])
            # A frozen third-party estimator prediction is permitted. It is
            # neither runner endpoint arithmetic nor an independent refit.
            probabilities.append(np.asarray(head.predict_proba(latent),dtype=np.float64))
    return probabilities


def route_fine(policy_json, probability):
    fine=policy_json['fine']
    classes=np.asarray(fine['cell_class'],dtype=np.int64)
    prototypes=np.asarray(fine['centroid'],dtype=np.float64)
    token=np.asarray(policy_json['cell_token'],dtype=np.int64)
    decisions=np.argmax(probability,axis=1)
    codes=np.full(len(probability),-1,dtype=np.int64)
    for i, (row, decision) in enumerate(zip(probability,decisions,strict=True)):
        allowed=np.flatnonzero(classes==decision)
        if not len(allowed):
            raise ValueError('frozen router lacks predicted class')
        costs=[]
        for cell in allowed:
            cost=0.
            for value, reference in zip(row,prototypes[cell],strict=True):
                if value>0:
                    cost+=float(value)*(float(np.log(value))-float(np.log(reference)))
            costs.append(cost)
        codes[i]=token[allowed[int(np.argmin(costs))]]
    return codes


def raw_p0(pins, bank, raw, feature_names):
    if list(feature_names)!=bank['feature_names'] or not np.all(np.isfinite(raw)):
        raise ValueError('forbidden or reordered raw schema')
    if digest(pins['policy']['path'])!=pins['policy']['sha256']:
        raise PermissionError('frozen router differs')
    pair=json.loads(open(pins['policy']['path']).read())
    scores=frozen_raw_probabilities(pins,raw)
    result=[]
    for recipient in (1,2):
        codes=route_fine(pair[f'p{recipient}'],scores[recipient-1])
        probabilities=np.asarray(bank['recipients'][str(recipient)]['probability'],dtype=np.float64)[codes]
        decisions=np.argmax(probabilities,axis=1)
        if not np.array_equal(decisions,np.argmax(scores[recipient-1],axis=1)):
            raise ValueError('frozen calibrated decision differs')
        result.append({'codes':codes,'probabilities':probabilities,'decisions':decisions})
    return result,scores

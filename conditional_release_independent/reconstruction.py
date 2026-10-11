"""Independent frozen finite-law reconstruction from serialized numbers.

The architecture and source feature definitions are pinned written formats.
No source policy/deployment or runner implementation is imported.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
from zipfile import ZipFile
import numpy as np


def digest(path):
    p=Path(path)
    lowered=str(p).lower()
    if p.is_symlink() or any(s in lowered for s in ('head_validation','osf_development_assessment','all_role','old_assessment')):
        raise PermissionError('barred independent replay source')
    if p.name=='release.npz' and any(s in lowered for s in ('teacher','rel__s')):
        raise PermissionError('all-role teacher archive forbidden even for hash')
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def admitted_arrays(entry, names):
    if digest(entry['path']) != entry['sha256']:
        raise PermissionError('independent source hash mismatch')
    with ZipFile(entry['path']) as archive:
        members=archive.namelist()
        if len(members)!=len(names) or set(members)!={name+'.npy' for name in names}:
            raise PermissionError('unexpected source member before deserialization')
    with np.load(entry['path'],allow_pickle=False) as archive:
        return {name:archive[name].copy() for name in names}


def folds(groups):
    return np.array([int.from_bytes(hashlib.sha256(('IVR1|FIT-crossfold|20261010|'+str(g)).encode()).digest(),'big')%3 for g in groups],dtype=np.int64)


def feature_summaries(probability):
    entropy = -np.sum(probability*np.log(np.maximum(probability,1e-12)),axis=1)
    ordered=np.sort(probability,axis=1)
    return np.column_stack((entropy,ordered[:,-1]-ordered[:,-2]))


def features(p1,p2):
    return np.column_stack((p1,p2,feature_summaries(p1),feature_summaries(p2)))


def gate(features_, weights, valid):
    hidden=np.tanh(np.dot(features_,weights[0])+weights[1])
    logits=np.dot(hidden,weights[2])+weights[3]
    logits=np.where(valid,logits,-np.inf)
    mass=np.exp(logits-np.max(logits,axis=1,keepdims=True))
    return mass/np.sum(mass,axis=1,keepdims=True)


def source_joint_law(arrays, codes1, codes2, eta):
    """V-J donor from exact archived gate weights; eta=1 supplies donor."""
    p1=arrays['probability1'][codes1];p2=arrays['probability2'][codes2]
    actions1=arrays['actions1'][codes1];actions2=arrays['actions2'][codes2]
    c1=np.repeat(actions1,4,axis=1);c2=np.tile(actions2,(1,4))
    valid=np.repeat(arrays['valid1'][codes1],4,axis=1)&np.tile(arrays['valid2'][codes2],(1,4))
    weights=[arrays[f'gate0_{i}'] for i in range(4)]
    mass=gate(features(p1,p2),weights,valid)*eta
    mass[:,0]+=1-eta
    return c1,c2,mass


def source_seq21_law(arrays,codes1,codes2,eta1,eta2):
    """Historical freeze-first order occupation then income."""
    p1=arrays['probability1'][codes1];p2=arrays['probability2'][codes2]
    acts1=arrays['actions1'][codes1];acts2=arrays['actions2'][codes2]
    first=gate(features(p1,p2),[arrays[f'gate0_{i}'] for i in range(4)],arrays['valid2'][codes2])*eta1
    first[:,0]+=1-eta1
    mass=np.zeros((len(codes1),4,4))
    for occupation_slot in range(4):
        emitted=arrays['probability2'][acts2[:,occupation_slot]]
        later=gate(features(p1,emitted),[arrays[f'gate1_{i}'] for i in range(4)],arrays['valid1'][codes1])*eta2
        later[:,0]+=1-eta2
        mass[:,:,occupation_slot]=first[:,occupation_slot,None]*later
    return np.repeat(acts1,4,axis=1),np.tile(acts2,(1,4)),mass.reshape(len(codes1),16)


def merge_rows(codes1,codes2,mass):
    rows=[]
    for c1,c2,q in zip(codes1,codes2,mass,strict=True):
        law={}
        for a,b,weight in zip(c1,c2,q,strict=True):
            if weight:
                law[int(a),int(b)]=law.get((int(a),int(b)),0.)+float(weight)
        rows.append(law)
    return rows


def validate_support(arrays,codes1,codes2,mass):
    maximum_error=float(np.max(np.abs(np.sum(mass,axis=1)-1),initial=0))
    if np.min(mass)<0 or maximum_error>1e-12:
        raise ValueError('invalid source mass')
    for recipient,codes in ((1,codes1),(2,codes2)):
        p=arrays[f'probability{recipient}']
        if np.any(np.argmax(p[codes],axis=-1)!=np.argmax(p[codes[:,0]],axis=-1)[:,None]):
            raise ValueError('source class change')
    return {'maximum_mass_error':maximum_error,'same_class':'PASS'}

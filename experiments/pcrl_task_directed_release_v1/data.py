"""Explicit owned-input loader. No global historical fallbacks and no evaluation lookahead."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import OUT, configuration

POOLS = ('representation_fit','source_validation','downstream_fit','downstream_validation',
         'attacker_fit','attacker_validation','test')
FIXED = 'results/redesign_20260909_acs_fixed_predictions_v1'
RAW = 'data/folktables/2018/1-Year/psam_p06.csv'
LABEL_COLUMNS = ['SEX','RAC1P','PUBCOV','PINCP','ESR','MIG','JWMNP']


def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


@dataclass(frozen=True)
class RuntimeInputs:
    x_a: np.ndarray
    h_a: np.ndarray

    def __post_init__(self):
        x,h=np.asarray(self.x_a),np.asarray(self.h_a)
        if x.ndim!=2 or x.shape[1]!=32 or h.shape!=(len(x),4):
            raise ValueError('Only PCA32 A inputs and four H_A coordinates are permitted')
        if not np.isfinite(x).all() or not np.isfinite(h).all():
            raise ValueError('Nonfinite runtime inputs')

    def features(self):
        return np.column_stack((self.x_a,self.h_a))


def household_roles(serials):
    cfg=configuration()['split']; salt=cfg['salt']
    u=np.array([int(hashlib.sha256(f'{salt}|{s}'.encode()).hexdigest()[:12],16)/16**12
                for s in np.asarray(serials).astype(str)])
    # .48 fit, .12 internal validation/code fitting, .40 mechanism; all by household.
    return {'teacher_fit':np.flatnonzero(u<.48),
            'teacher_internal_validation':np.flatnonzero((u>=.48)&(u<.60)),
            'mechanism':np.flatnonzero(u>=.60)}


def read_rows(raw_path, rows, columns, *, pool, evaluation_permit=None):
    if pool=='test':
        if evaluation_permit is None or not Path(evaluation_permit).is_file():
            raise PermissionError('Current-run evaluation is sealed until selection freeze')
        permit=json.loads(Path(evaluation_permit).read_text())
        if not permit.get('selection_frozen'):
            raise PermissionError('Evaluation permit does not freeze selection')
    rows=np.asarray(rows,dtype=np.int64)
    if len(np.unique(rows))!=len(rows) or np.any(rows<0):
        raise ValueError('Duplicate or invalid raw row indices')
    if len(rows)>1 and np.any(np.diff(rows)<=0):
        raise ValueError('Read rows must be strictly ordered to preserve identities')
    selected=set((rows+1).tolist())
    frame=pd.read_csv(raw_path,usecols=columns,dtype={'SERIALNO':str},
                      skiprows=lambda i:i!=0 and i not in selected)
    if len(frame)!=len(rows):raise ValueError('Missing requested people')
    return frame


def labels_from_frame(frame):
    def coded(key,k):
        v=frame[key].to_numpy(float)
        return np.where(np.isin(v,np.arange(1,k+1)),v-1,-1).astype(np.int64)
    sex,race=coded('SEX',2),coded('RAC1P',9)
    pub=coded('PUBCOV',2); mig=coded('MIG',3); esr=coded('ESR',6)
    inc=frame.PINCP.to_numpy(float); commute=frame.JWMNP.to_numpy(float)
    return {'SEX':sex,'RAC1P':race,
            'public_coverage':np.where(pub>=0,(pub==0).astype(int),-1),
            'same_residence':np.where(mig>=0,(mig==0).astype(int),-1),
            'civilian_at_work':np.where(esr>=0,(esr==0).astype(int),-1),
            'income_binary':np.where(np.isfinite(inc)&(inc>=-19998)&(inc<=4209995),(inc>50000).astype(int),-1),
            'commute_over20':np.where(np.isfinite(commute)&(commute>=1)&(commute<=200)&(commute==np.floor(commute)),(commute>20).astype(int),-1)}


def load_anchor(anchor, pools=POOLS[:-1], *, inputs_root=None, evaluation_permit=None):
    import torch
    from torch import nn
    inputs_root=Path(inputs_root or OUT/'private/inputs').resolve()
    if not set(pools).issubset(POOLS):raise ValueError('Unknown pool')
    if 'test' in pools and evaluation_permit is None:
        raise PermissionError('Evaluation arrays stay sealed until selection freeze')
    base=inputs_root/FIXED/f'seed_{anchor}'
    # npz members are read lazily. Evaluation feature/label members are not accessed here.
    with np.load(base/'split_rows.npz') as z: rows={p:z[p] for p in pools}
    with np.load(base/'pca.npz') as z: pca={p:z[p] for p in pools}
    with np.load(base/'anchors.npz') as z:
        ha={p:z[p+'/A'] for p in pools};hb={p:z[p+'/B'] for p in pools}
    checkpoint=torch.load(base/'training/J/final.pt',map_location='cpu',weights_only=False)['model_state']
    mean=checkpoint['input_mean'].numpy();scale=checkpoint['input_scale'].numpy()
    x={p:np.asarray((np.asarray(pca[p],np.float64)-mean)/scale,np.float32) for p in pools}
    channels={}; parity={}
    for arm in ('A0','J'):
        state=torch.load(base/f'training/{arm}/final.pt',map_location='cpu',weights_only=False)['model_state']
        if not np.array_equal(state['input_mean'].numpy(),mean) or not np.array_equal(state['input_scale'].numpy(),scale):
            raise ValueError('Historical standardizers disagree')
        mapper=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,16))
        mapper.load_state_dict({k[len('branch.mapper.'):]:v for k,v in state.items() if k.startswith('branch.mapper.')})
        mapper.eval();channels[arm]={}; parity[arm]={}
        with np.load(base/f'training/{arm}/releases.npz') as z,torch.no_grad():
            for p in pools:
                wire=z[f'wire/A/{p}']
                if wire.dtype!=ha[p].dtype or not np.array_equal(wire[:,:4],ha[p]):raise ValueError('H parity failure')
                stored=wire[:,4:];calc=mapper(torch.from_numpy(x[p])).numpy()
                err=float(np.max(np.abs(calc-stored)))
                if err>1e-5:raise ValueError(f'{arm} portability exceeds 1e-5: {err}')
                channels[arm][p]=stored.copy();parity[arm][p]=err
    data={}
    for p in pools:
        f=read_rows(inputs_root/RAW,rows[p],['SERIALNO','SPORDER','PWGTP',*LABEL_COLUMNS],
                    pool=p,evaluation_permit=evaluation_permit)
        ids=(f.SERIALNO.astype(str)+':'+f.SPORDER.astype(str)).to_numpy()
        if len(set(ids))!=len(ids):raise ValueError('Duplicate person IDs')
        weights=f.PWGTP.to_numpy(float)
        if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Invalid PWGTP')
        data[p]={'x':x[p],'ha':ha[p],'hb':hb[p],'J':channels['J'][p],
                 'A0':channels['A0'][p],'ids':ids,'households':f.SERIALNO.to_numpy(),
                 'raw_rows':rows[p],'weights':weights,'labels':labels_from_frame(f)}
        RuntimeInputs(x[p],ha[p])
    return {'anchor':anchor,'pools':data,'portability':parity,
            'input_contract':'saved PCA32 standardized float64 then float32; H_A 4; H_B excluded from encoder'}

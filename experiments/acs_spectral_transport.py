"""Outcome-free admission of a public ACS transport release.

No fitted model, preprocessor, anchor, or scorer is imported. Person rows stay
local. Household partition ordering depends only on public release identifiers.
"""
from __future__ import annotations
import hashlib
import re
import numpy as np
import pandas as pd
from experiments.acs_transfer_data import FEATURES, CATEGORY_CODES, array_hash

REQUIRED = (*FEATURES,'PINCP','ESR','PUBCOV','MIG','JWMNP','SEX','RAC1P',
            'SERIALNO','SPORDER','PWGTP','ST','RT')
POOLS = ('attacker_fit','attacker_validation','task_fit','task_validation','final_evaluation')
FRACTIONS = (.25,.15,.25,.15,.20)
PARTITION_SALT = 'PCRL-residual-spectral-transport-v1-20260910'
CLASSES = {'income_binary':2,'civilian_at_work':2,'public_coverage':2,
           'same_residence':2,'commute_over20':2,'SEX':2,'RAC1P':9}


def dictionary_block(text, name):
    match = re.search(r'^'+re.escape(name)+r'\s+(?:Numeric|Character).*?'
                      r'(?=\n[A-Z0-9]+\s+(?:Numeric|Character)|\Z)', text, re.M|re.S)
    if match is None: raise ValueError('Missing dictionary entry: '+name)
    return match.group(0).strip()


def validate_frame(frame, year):
    if year != 2017: raise ValueError('Only the dictionary-audited 2017 release is implemented')
    missing = set(REQUIRED)-set(frame)
    if missing: raise ValueError('Missing columns: '+str(sorted(missing)))
    if not frame.RT.eq('P').all() or not frame.ST.eq(6).all():
        raise ValueError('Wrong record type or California state')
    pattern = rf'{year}[0-9]{{9}}'
    if not frame.SERIALNO.astype('string').str.fullmatch(pattern).fillna(False).all():
        raise ValueError('Invalid SERIALNO release identifier')
    sp = pd.to_numeric(frame.SPORDER,errors='coerce').to_numpy(float)
    if not (np.isfinite(sp)&(sp>=1)&(sp<=20)&(sp==np.floor(sp))).all():
        raise ValueError('Invalid SPORDER')
    w=frame.PWGTP.to_numpy(float)
    if not (np.isfinite(w)&(w>=1)&(w<=9999)&(w==np.floor(w))).all():
        raise ValueError('Invalid PWGTP')
    codes={**{k:v for k,v in CATEGORY_CODES.items() if k in FEATURES},
           'ESR':range(1,7),'PUBCOV':(1,2),'MIG':(1,2,3),'SEX':(1,2),'RAC1P':range(1,10)}
    stats={}
    for c in (*FEATURES,'PINCP','ESR','PUBCOV','MIG','JWMNP','SEX','RAC1P'):
        v=frame[c].to_numpy(float); nonmissing=np.isfinite(v)
        if c in codes: valid=np.isin(v,list(codes[c]))
        else:
            lo,hi={'AGEP':(0,99),'WKHP':(1,99),'PINCP':(-19998,4209995),'JWMNP':(1,200)}[c]
            valid=(v>=lo)&(v<=hi)&(v==np.floor(v))
        if ((~valid)&nonmissing).any() or np.isinf(v).any():
            raise ValueError('Out-of-dictionary values: '+c)
        if c in ('AGEP','RELP','MAR','CIT','DIS','DEAR','DEYE','PUBCOV','SEX','RAC1P') and not nonmissing.all():
            raise ValueError('Unexpected missing values: '+c)
        stats[c]={'missing':int((~nonmissing).sum()),'valid':int(nonmissing.sum()),
                  'observed_codes':sorted(np.unique(v[nonmissing]).astype(int).tolist()) if c in codes else None}
    return stats


def eligible_cohort(raw,year):
    validate_frame(raw,year)
    raw=raw.copy(); raw['_raw_row']=np.arange(len(raw),dtype=np.int64)
    # Canonical numeric SPORDER removes formatting-only identifier differences.
    raw['SPORDER']=pd.to_numeric(raw.SPORDER).astype(int).astype(str)
    eligible=raw.AGEP.between(19,34)&raw.PWGTP.gt(0)
    f=raw.loc[eligible].copy(); duplicates=f.duplicated(['SERIALNO','SPORDER'],keep=False)
    for _,g in f.loc[duplicates].groupby(['SERIALNO','SPORDER']):
        if len(g.drop(columns='_raw_row').drop_duplicates())!=1:
            raise ValueError('Conflicting duplicate person keys')
    before=len(f); f=f.drop_duplicates(['SERIALNO','SPORDER']).reset_index(drop=True)
    return f, {'raw_rows':len(raw),'eligible_rows_before_dedup':before,
               'duplicate_rows_removed':before-len(f),'eligible_rows':len(f),
               'eligible_households_or_gq_persons':int(f.SERIALNO.nunique()),
               'raw_row_hash':array_hash(f._raw_row.to_numpy()),'cap':None,
               'group_quarters_person_rows':int(f.RELP.isin([16,17]).sum())}


def partition_households(frame,year):
    groups=sorted(frame.SERIALNO.unique(),key=lambda s:(
        hashlib.sha256(f'{PARTITION_SALT}|{year}|06|{s}'.encode()).hexdigest(),s))
    boundaries=np.rint(np.cumsum((0.,)+FRACTIONS)*len(groups)).astype(int)
    out={name:np.flatnonzero(frame.SERIALNO.isin(groups[boundaries[i]:boundaries[i+1]]))
         for i,name in enumerate(POOLS)}
    if len(np.unique(np.concatenate(list(out.values()))))!=len(frame):
        raise ValueError('Incomplete household partitions')
    return out


def fixed_labels(frame):
    out={}
    for target,column,valid_codes in [('income_binary','PINCP',None),
        ('civilian_at_work','ESR',range(1,7)),('public_coverage','PUBCOV',(1,2)),
        ('same_residence','MIG',(1,2,3)),('commute_over20','JWMNP',None),
        ('SEX','SEX',(1,2)),('RAC1P','RAC1P',range(1,10))]:
        v=frame[column].to_numpy(float)
        if target=='income_binary': valid=np.isfinite(v)&(v>=-19998)&(v<=4209995); y=v>50000
        elif target=='commute_over20': valid=np.isfinite(v)&(v>=1)&(v<=200)&(v==np.floor(v));y=v>20
        else:
            valid=np.isin(v,list(valid_codes))
            y=v-1 if target in ('SEX','RAC1P') else v==1
        result=np.full(len(frame),-1,dtype=np.int64); result[valid]=np.asarray(y[valid],dtype=np.int64)
        out[target]=result
    return out


def support(frame):
    weights=frame.PWGTP.to_numpy(float)
    return {k:{'valid':int((v>=0).sum()),'missing_or_inapplicable':int((v<0).sum()),
               'class_counts':np.bincount(v[v>=0],minlength=CLASSES[k]).tolist(),
               'class_weight_sums':np.bincount(v[v>=0],weights=weights[v>=0],minlength=CLASSES[k]).tolist()}
            for k,v in fixed_labels(frame).items()}

#!/usr/bin/env python3
"""Independent historical ACS exposure replay; no models or outcome scoring.

Raw identifiers are written only below the new study's ignored local/ directory.
Running again verifies identical completed units and never overwrites evidence.
The selection implementation deliberately does not import the new evaluator.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NAMESPACE='ACS/2018/1-Year/CA/person'
SALT='PCRL_LOCKED_EVAL_20260910_V1'
RAW='data/folktables/2018/1-Year/psam_p06.csv'
RAW_SHA='dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0'
CACHE='data/folktables/acs_2018_CA_v2_ffb_binarized.parquet'
CACHE_SHA='b1d08f98f9a95ac163d0b8fd56edec2218c40f4995de47aa55650b5dd3476f04'
TRANSFER='results/redesign_20260907_acs_transfer_v1'


def canonical(*parts):
    return json.dumps(list(parts),ensure_ascii=True,separators=(',',':'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def save(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:raise RuntimeError('Incompatible completed evidence: '+str(path))
    else:
        tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(content);tmp.replace(path)
    return sha(path)


def json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def jsonl_bytes(values):
    return ''.join(json.dumps(v,ensure_ascii=True,separators=(',',':'))+'\n' for v in values).encode()


def independent_selection(raw,used_raw_indices,cap=30000):
    """Historical non-target members exclude their complete household."""
    r=raw.copy()
    for key in ('SERIALNO','SPORDER'):
        if not r[key].map(lambda v:isinstance(v,str) and len(v)>0).all():
            raise ValueError('Missing or non-string identifier')
    r['_raw_row']=np.arange(len(r),dtype=np.int64)
    duplicated=r.duplicated(['SERIALNO','SPORDER'],keep=False)
    for _,g in r.loc[duplicated].groupby(['SERIALNO','SPORDER']):
        if len(g.drop(columns='_raw_row').drop_duplicates())!=1:
            raise ValueError('Conflicting duplicate person records')
    r['household_id']=[canonical(NAMESPACE,h) for h in r.SERIALNO]
    r['person_id']=[canonical(NAMESPACE,h,p) for h,p in zip(r.SERIALNO,r.SPORDER)]
    excluded=set(r.iloc[np.asarray(used_raw_indices,dtype=np.int64)].household_id)
    age=r.AGEP;weight=r.PWGTP
    target=r.loc[age.ge(19)&age.le(34)&age.mod(1).eq(0)&weight.gt(0)].drop_duplicates(['SERIALNO','SPORDER']).copy()
    available=target.loc[~target.household_id.isin(excluded)]
    sizes=available.groupby('household_id').size().to_dict()
    order=sorted(sizes,key=lambda h:(hashlib.sha256(canonical(SALT,h).encode()).digest(),h))
    chosen=[];total=0
    for h in order:
        if total+int(sizes[h])>cap:break
        chosen.append({'household_id':h,'eligible_persons':int(sizes[h]),
                       'ordering_sha256':hashlib.sha256(canonical(SALT,h).encode()).hexdigest()})
        total+=int(sizes[h])
    rank={v['household_id']:i for i,v in enumerate(chosen)}
    sample=available.loc[available.household_id.isin(rank)].copy()
    sample['_household_order']=sample.household_id.map(rank)
    sample=sample.sort_values(['_household_order','person_id']).reset_index(drop=True)
    return target,excluded,sample,chosen


def replay(out):
    started=time.perf_counter();created=dt.datetime.now(dt.timezone.utc).isoformat()
    # Verify raw identity before parsing or consulting historical cache values.
    assert sha(ROOT/RAW)==RAW_SHA,'Original ACS identity mismatch'
    assert sha(ROOT/CACHE)==CACHE_SHA,'Historical cache identity mismatch'
    cache=pd.read_parquet(ROOT/CACHE)
    columns=list(dict.fromkeys(list(cache.columns)+['SERIALNO','SPORDER','PWGTP']))
    raw=pd.read_csv(ROOT/RAW,usecols=columns,dtype={'SERIALNO':str,'SPORDER':str},low_memory=False)
    mask=raw.AGEP.ge(16)&raw[['AGEP','SEX','RAC1P']].notna().all(axis=1)
    historical=raw.loc[mask,cache.columns].reset_index(drop=True)
    for c in ('AGEP','SEX','RAC1P'):historical[c]=historical[c].astype(int)
    pd.testing.assert_frame_equal(cache,historical,check_dtype=False,check_exact=True)
    v2rows=np.flatnonzero(mask);n=len(v2rows);perm=np.random.RandomState(42).permutation(n)
    na=int(.8*n);nb=int(.1*n)
    reasons={'v2_train':v2rows[perm[:na]],'v2_validation':v2rows[perm[na:na+nb]],'v2_test':v2rows[perm[na+nb:]]}
    # Reconstruct the original transfer cohort independently of its loader.
    target,_,_,_=independent_selection(raw,[],cap=30000)
    sizes=target.groupby('SERIALNO',sort=True).size()
    cfg=json.loads((ROOT/TRANSFER/'config.json').read_text())
    ordered=np.random.default_rng(cfg['sample_seed']).permutation(sizes.index.to_numpy())
    chosen=ordered[sizes.loc[ordered].cumsum().to_numpy()<=cfg['sample_cap']]
    cohort=target.loc[target.SERIALNO.isin(chosen)].sort_values('_raw_row').reset_index(drop=True)
    reasons['complete_transfer_cohort']=cohort._raw_row.to_numpy()
    schema=json.loads((ROOT/TRANSFER/'schema_support.json').read_text())
    assert array_hash(reasons['complete_transfer_cohort'])==schema['cohort']['raw_row_hash']
    pool_names=list(cfg['pool_fractions']);fractions=list(cfg['pool_fractions'].values());pool_checks={}
    for seed in cfg['seeds']:
        households=np.random.default_rng(1210000+seed).permutation(np.asarray(sorted(cohort.SERIALNO.unique())))
        edges=np.rint(np.cumsum([0.]+fractions)*len(households)).astype(int)
        with np.load(ROOT/TRANSFER/f'seed_{seed}/split_rows.npz') as saved:
            for i,pool in enumerate(pool_names):
                rows=cohort.loc[cohort.SERIALNO.isin(households[edges[i]:edges[i+1]]),'_raw_row'].to_numpy()
                assert np.array_equal(saved[pool],rows)
                assert array_hash(rows)==schema['seeds'][str(seed)]['pools'][pool]['raw_row_hash']
                pool_checks[f'seed_{seed}/{pool}']={'rows':len(rows),'raw_row_hash':array_hash(rows),'saved_npz_exact':True}
    # Legacy ACSIncome historical use is dominated by v2 but included explicitly.
    legacy=(raw.AGEP.gt(16)&raw.PINCP.gt(100)&raw.WKHP.gt(0)&raw.PWGTP.ge(1)&
            raw[['AGEP','COW','SCHL','MAR','WKHP','SEX','RAC1P','PINCP']].notna().all(axis=1))
    reasons['legacy_acsincome_population']=np.flatnonzero(legacy)
    used=np.unique(np.concatenate(list(reasons.values())))
    target,excluded,sample,ordering=independent_selection(raw,used)
    assert len(sample)==0,'Known exhaustive historical exposure changed; review instead of scoring'
    local_files={};reason_counts={}
    def emit(name,records):
        path=out/'local'/name;local_files[str(path.relative_to(ROOT))]=save(path,jsonl_bytes(records))
    emit('exclusion_households.jsonl',sorted(excluded))
    target_excluded=target.loc[target.household_id.isin(excluded)]
    emit('excluded_target_people.jsonl',target_excluded.sort_values('person_id')[['person_id','household_id','_raw_row']].to_dict('records'))
    historical_people=raw.iloc[used]
    emit('historical_used_people.jsonl',({'person_id':canonical(NAMESPACE,h,p),'raw_row':int(i)} for h,p,i in zip(historical_people.SERIALNO,historical_people.SPORDER,used)))
    emit('sample_people.jsonl',sample[['person_id','household_id','_raw_row']].to_dict('records'))
    emit('sample_households_ordered.jsonl',ordering)
    for reason,indices in reasons.items():
        people=raw.iloc[indices];houses={canonical(NAMESPACE,h) for h in people.SERIALNO}
        affected=target.loc[target.household_id.isin(houses)]
        emit(reason+'_people.jsonl',({'person_id':canonical(NAMESPACE,h,p),'raw_row':int(i)} for h,p,i in zip(people.SERIALNO,people.SPORDER,indices)))
        emit(reason+'_households.jsonl',sorted(houses))
        reason_counts[reason]={'historical_rows':len(indices),'historical_households':len(houses),
                              'direct_eligible_rows':int(target._raw_row.isin(indices).sum()),
                              'excluded_eligible_rows':len(affected),'excluded_eligible_households':affected.SERIALNO.nunique()}
    source_paths=[RAW,CACHE,'pcrl/data/folktables.py','pcrl/data/folktables_legacy.py',
                  'experiments/run_v2_dataset.py','experiments/run_folktables.py',
                  'experiments/acs_transfer_data.py','experiments/run_acs_transfer.py',
                  'scripts/folktables_round1_verdict.py','scripts/folktables_round2_verdict.py',
                  'results/V2_FOLKTABLES_ROUND1_VERDICT.md','results/V2_FOLKTABLES_ROUND2_VERDICT.md',
                  'results/v2_folktables_FOLKTABLES_FIXED_PROBE/per_seed_results.json',
                  'results/v2_folktables_ROUND2/verdict_raw.json',TRANSFER+'/config.json',TRANSFER+'/schema_support.json',
                  'scripts/verify_acs_locked_independence.py']
    source_paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'results').glob('redesign*acs*/config.json')) if str(p.relative_to(ROOT)) not in source_paths and p.parent!=out]
    inventory={'sample_rows':len(sample),'sample_households':sample.SERIALNO.nunique(),
               'independence_verified':False,'independence_audit_complete':True,
               'status':'blocked_no_unused_same_population_households',
               'eligible_rows':len(target),'eligible_households':target.SERIALNO.nunique(),
               'excluded_eligible_rows':len(target_excluded),'excluded_eligible_households':target_excluded.SERIALNO.nunique(),
               'excluded_households_all_ages':len(excluded),'raw_rows':len(raw),'historical_v2_rows':len(v2rows),
               'historical_cache_exact_values_and_order':True,'raw_identity_verified':True,
               'raw_sha256':RAW_SHA,'historical_cache_sha256':CACHE_SHA,
               'transfer_only_remainder_rows':int((~target.SERIALNO.isin(cohort.SERIALNO)).sum()),
               'transfer_only_remainder_households':target.loc[~target.SERIALNO.isin(cohort.SERIALNO),'SERIALNO'].nunique(),
               'reason_counts':reason_counts,'reason_counts_overlap':True,'transfer_pool_checks':pool_checks,
               'namespace':NAMESPACE,'identifier_encoding':'JSON compact ASCII arrays; household=[namespace,SERIALNO]; person=[namespace,SERIALNO,SPORDER]; original CSV strings',
               'manifest_encoding':'UTF-8 JSON Lines; sorted canonical IDs for union; empty manifests have zero bytes',
               'sampling':{'salt':SALT,'order':'SHA256(JSON compact [salt,canonical household ID]); digest ties use canonical ID',
                           'rule':'longest complete-household prefix totaling at most 30000 eligible persons; no skipped groups','cap':30000,
                           'sample_code_path':'scripts/verify_acs_locked_independence.py','sample_code_sha256':sha(Path(__file__))},
               'local_files':local_files,'source_sha256':{p:sha(ROOT/p) for p in source_paths},
               'fitting_exposure':{'retained_transfer_preprocessor':'representation_fit only, separately by seed',
                                   'retained_transfer_pca':'representation_fit only, separately by seed',
                                   'historical_v2_numeric_preprocessing':'all 248720 historical training rows',
                                   'historical_v2_all_population':'union of fitting, validation selection, test metrics and representation diagnostics'},
               'prior_aggregate_exposure':['Original-file loading/filtering and household counts are not model fitting.',
                                          'Application screen records ACS inventory-only; no ACS outcome analysis in that screen.',
                                          'Transfer schema/support preparation covered its complete selected 30000-person cohort.'],
               'coverage_questions':['Earlier ACSIncome processed cache unavailable; exact source predicate and retained results imply a subset dominated by verified v2 exposure.',
                                     'Unrecorded exploratory uses cannot be reconstructed; any additional exposure can only enlarge this already exhaustive exclusion.',
                                     'Individual executed jobs lack raw identifier manifests; v2 cache exact raw-row reconstruction and preserved executed fit/test evidence recover the population.',
                                     'No claim is made that retained redesigned models fitted on the entire eligible file. Broad historical research use is the blocker.'],
               'scientific_model_fits':0,'optimizer_updates':0,'new_scored_rows':0}
    path=out/'DATA_USE_INVENTORY.json';save(path,json_bytes(inventory))
    # Keep completion timestamp on compatible resumes rather than inventing a new lock.
    completion=out/'DATA_INDEPENDENCE_COMPLETION.json'
    if completion.exists():
        previous=json.loads(completion.read_text());assert previous['inventory_sha256']==sha(path)
    else:save(completion,json_bytes({'created_utc':created,'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                                    'elapsed_seconds':time.perf_counter()-started,'inventory_sha256':sha(path),
                                    'status':inventory['status'],'new_scored_rows':0}))
    print(json.dumps({k:inventory[k] for k in ('status','eligible_rows','eligible_households','sample_rows','sample_households')}))
    return inventory


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260910_acs_locked_evaluation_v1')
    replay(parser.parse_args().out.resolve())

"""Metadata-only post-seal audit specs; ROOT owns all numerical execution."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .prepare_specs import ROOT,PRIVATE,PUBLIC,metadata_pin,receipt_pin,write
from .prepare_postnomination import CAMPAIGN_PRIVATE,CAMPAIGN_PUBLIC,ARMS,SERVICES,read,context,emit,packages


def refit_scores(ctx,index_path,seal_path,replace):
    index,entries=packages(index_path);seal=read(seal_path)
    freeze_path=PUBLIC/'REFIT_TARGET_FREEZE_V1.json';freeze=read(freeze_path)
    registry_path=PRIVATE/'targets/FREEZE_V1/TARGETS.json';registry=read(registry_path)
    cfg={'unit_id':'CIS_INDEPENDENT_REFIT_SCORE_AUDIT_V1','public_output':str(PUBLIC/'REFIT_SCORE_AUDIT_V1.json'),
         'reporting_lock':{'path':seal['path'],'sha256':seal['sha256']},'truth':seal['roles']['INNER_SELECTION']['sex'],
         'target_registry':receipt_pin(registry_path,freeze['target_registry_sha256']),'attempt_receipts':[],'attempts':[]}
    pins=[metadata_pin(index_path),metadata_pin(seal_path),metadata_pin(freeze_path),cfg['reporting_lock'],cfg['truth'],cfg['target_registry']]
    targets={t['target_id']:t for t in registry['targets']};deps=[]
    for p in entries:
        tag=f'{p["service"]}_S{p["seed"]}';rp=PUBLIC/'refits'/f'{tag}.json';receipt=read(rp);rpin=metadata_pin(rp)
        cfg['attempt_receipts'].append(rpin);pins.append(rpin);deps.append(f'CIS_INDEPENDENT_REFIT_{tag}_V1')
        for record in receipt['records']:
            tid=record['target_id'];t=targets[tid];directory=PRIVATE/'refits'/tag
            item={'target_id':tid,'attempt_receipt':rpin,'model':receipt_pin(directory/(tid+'.joblib'),record['model_sha256']),
                  'score':receipt_pin(directory/(tid+'.npz'),record['score_sha256']),
                  'runner_att_scores':p['att_scores'],'runner_inner_scores':p['scores']}
            cfg['attempts'].append(item);pins += [item[k] for k in ('model','score','runner_att_scores','runner_inner_scores')]
            pins += [t[k] for k in ('att_release','inner_release','freeze')]
    if len(cfg['attempts'])!=54:raise ValueError('exact54 actual completed attempts')
    return emit(ctx,cfg,'refit_score_audit_worker',pins,deps=deps,wall=600,replace=replace)


def unit_census(ctx,index_path,replace):
    index,entries=packages(index_path)
    cfg={'unit_id':'CIS_INDEPENDENT_UNIT_CENSUS_V1','public_output':str(PUBLIC/'UNIT_CENSUS_V1.json'),
         'banks':[],'frozen_sources':[],'continuation_checkpoints':[]}
    pins=[metadata_pin(index_path)];deps=[]
    identities=[(a,s,e) for a in ARMS for s in (0,1) for e in (0,5,10,20)]
    identities += [(a,s,None) for a in SERVICES[4:] for s in (0,1)]
    for service,seed,epoch in identities:
        tag=service+(f'_E{epoch}' if epoch is not None else '')+f'_S{seed}';directory=CAMPAIGN_PRIVATE/'readers'/tag
        fp=directory/'FROZEN_COMPLETE.json';f=read(fp);item={'service':service,'seed':seed,'epoch':epoch,'freeze':metadata_pin(fp),
              'models':{name:receipt_pin(directory/name,digest) for name,digest in f['model_files'].items()}}
        for key,name,hashkey in (('fit_release','FIT_SELECTION_RELEASE.npz','FIT_selection_release_sha256'),
                                ('att_release','ATT_RELEASE.npz','att_release_sha256'),('inner_release','INNER_RELEASE.npz','inner_release_sha256'),
                                ('fit_scores','FIT_SELECTION_SCORES.npz','FIT_selection_scores_sha256'),
                                ('att_scores','ATT_SCORES.npz','att_scores_sha256'),('inner_scores','INNER_SCORES.npz','score_sha256')):
            item[key]=receipt_pin(directory/name,f[hashkey]);pins.append(item[key])
        cfg['banks'].append(item);pins += [item['freeze'],*item['models'].values()]
    for seed in (0,1):
        source=next(p for p in entries if p['service']=='RAW_HYBRID' and p['seed']==seed)
        cfg['frozen_sources'].append({'seed':seed,'checkpoint':source['checkpoint'],'geometry':source['geometry']});pins += [source['checkpoint'],source['geometry']]
        for arm in ARMS:
            directory=CAMPAIGN_PRIVATE/'training'/f'S{seed}'/arm;summary_path=directory/'SUMMARY.json';summary=read(summary_path);pins.append(metadata_pin(summary_path))
            for checkpoint in summary['checkpoints']:
                epoch=checkpoint['epoch'];pin=receipt_pin(directory/f'epoch_{epoch}.pt',checkpoint['checkpoint_sha256'])
                cfg['continuation_checkpoints'].append({'service':arm,'seed':seed,'epoch':epoch,'checkpoint':pin});pins.append(pin)
        deps.append(f'CIS_TRAIN_S{seed}_V1')
    return emit(ctx,cfg,'unit_census_worker',pins,deps=deps,wall=600,output_bytes=8*1024**2,replace=replace)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('refit_scores','unit_census','all'))
    p.add_argument('--package-index',default=str(CAMPAIGN_PRIVATE/'r4/FINAL_PACKAGE_INDEX_V2.json'))
    p.add_argument('--seal',default=str(CAMPAIGN_PRIVATE/'root/REPORTING_OPERATIONAL_SEAL.json'))
    p.add_argument('--amendment');p.add_argument('--replace',action='store_true');args=p.parse_args();ctx=context(args.amendment);units=[]
    if args.phase in ('refit_scores','all'):units.append(refit_scores(ctx,args.package_index,args.seal,args.replace))
    if args.phase in ('unit_census','all'):units.append(unit_census(ctx,args.package_index,args.replace))
    result={'schema':'CIS1-additional-independent-audit-preparation-v1','status':'PREPARED_NOT_ENQUEUED','units':units,
            'numerical_artifact_bytes_read_or_hashed':False,'numerical_jobs_launched':0,'amendment':ctx['revision']}
    write(PRIVATE/'GENERATED_ADDITIONAL_AUDITS_V1.json',result,replace=args.replace);print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()

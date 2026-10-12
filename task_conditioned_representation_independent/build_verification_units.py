"""Metadata-only preparation of independently implemented queued verification.

No package importing numerical code, person-array deserialization, dispatch or
label exploration occurs in this builder. The root executes every queue unit.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def pin(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('preserve existing unit preparation')
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def build_refits(root,index_path):
    index=json.loads(index_path.read_text());selected=[];sources=[pin(index_path)]
    for unit in index['units']:
        if unit['arm_id'] not in ('JOINT_CONDITIONAL','U_NEW'):continue
        directory=Path(unit['private_output']);receipt=pin(directory/'FROZEN_COMPLETE.json');frozen=json.loads(Path(receipt['path']).read_text())
        if frozen['status']!='FROZEN_COMPLETE' or frozen['INNER_SEX_opened']:raise ValueError('source reader bank not prospectively frozen')
        selected.append({'family':unit['arm_id'],'seed':unit['seed'],'directory':str(directory),'receipt_sha256':receipt['sha256']})
        sources.append(receipt)
        for file,key in [('ATT_RELEASE.npz','att_release_sha256'),('INNER_RELEASE.npz','inner_release_sha256')]:sources.append({'path':str(directory/file),'sha256':frozen[key]})
    if len(selected)!=4:raise ValueError('four independent reader refits required')
    unit_id='R6_INDEPENDENT_PAIR_REFITS_V1';public=root/'results/pcrl_task_conditioned_representations_v1/r6/INDEPENDENT_PAIR_REFITS_V1.json';private=root/'.private/task_conditioned_representation/r6/refits_v1'
    config=root/f'.private/task_conditioned_representation/r6/configs/{unit_id}.json'
    cfg={'banks':selected,'private_output':str(private),'output':str(public)};write(config,cfg);sources.append(pin(config))
    for name in ('__init__','science','readers','refit_worker'):sources.append(pin(root/f'task_conditioned_representation_independent/{name}.py'))
    deps=[u['unit_id'] for u in index['units']+index['legacy_units']]
    spec={'unit_id':unit_id,'stage':'verification','argv':['/Users/nathansamson/PCRL/.venv/bin/python','-m','task_conditioned_representation_independent.refit_worker','--spec',str(config)],'cwd':str(root),'sources':sources,'outputs':[str(public)],'deps':deps,'cpu_reserve_s':300,'max_wall_s':300,'max_rss_gib':2,'projected_output_bytes':10485760}
    destination=root/f'.private/task_conditioned_representation/r6/specs/{unit_id}.json';write(destination,spec)
    return {'unit_id':unit_id,'config':pin(config),'spec':pin(destination),'public_output':str(public),'private_output':str(private),'numerical_jobs_launched':0,'INNER_SEX_opened_or_hashed':False}

def build_final(root,index_path,operational_path,reporting_config_path):
    index=json.loads(index_path.read_text());operational=json.loads(operational_path.read_text());reporting=json.loads(reporting_config_path.read_text())
    lock={'path':reporting['reporting_lock'],'sha256':reporting['reporting_lock_sha256']};sealed=json.loads(Path(lock['path']).read_text())
    if sealed.get('status')!='ALL_READERS_FROZEN' or len(sealed.get('reader_freezes',[]))<26:raise PermissionError('all service freezes required before preparation')
    refit_index=json.loads((root/'.private/task_conditioned_representation/r6/INDEPENDENT_REFIT_UNIT_INDEX_V1.json').read_text())
    refit_pin=pin(refit_index['public_output']);refit=json.loads(Path(refit_pin['path']).read_text())
    if refit.get('status')!='FROZEN_COMPLETE' or refit.get('numerical_fit_count')!=4:raise PermissionError('four independent refits must be frozen')
    sources=[pin(index_path),pin(operational_path),pin(reporting_config_path),lock,operational['nomination_lock'],operational['manifest'],refit_pin,reporting['inner_sex']]
    banks=[];historical=[];legacy=[]
    for unit in index['units']:
        entry=next(i for i in operational['instances'] if (i['arm_id'],i['seed'])==(unit['arm_id'],unit['seed']))
        directory=Path(unit['private_output']);freeze_pin=pin(directory/'FROZEN_COMPLETE.json');frozen=json.loads(Path(freeze_pin['path']).read_text());sources.append(freeze_pin)
        bank={'family':entry['arm_id'],'seed':entry['seed'],'directory':str(directory),'receipt_sha256':freeze_pin['sha256'],'checkpoint':entry['checkpoint'],'checkpoint_sha256':entry['checkpoint_sha256'],'nominee_checkpoint':entry['nominee_checkpoint'],'nominee_checkpoint_sha256':entry['nominee_checkpoint_sha256'],'nominee_point_eligible':entry['nominee_point_eligible'],'point_eligible_of_actualmodel':entry['actual_model_point_eligible'],'diagnostic_only':entry['diagnostic_only']}
        (historical if entry['arm_id']=='H_RAW_CSTAR' else banks).append(bank)
        for field in ('checkpoint','nominee_checkpoint'):sources.append({'path':entry[field],'sha256':entry[field+'_sha256']})
        for file,key in [('ATT_RELEASE.npz','att_release_sha256'),('INNER_RELEASE.npz','inner_release_sha256'),('INNER_SCORES.npz','score_sha256')]:sources.append({'path':str(directory/file),'sha256':frozen[key]})
        sources.extend({'path':str(directory/file),'sha256':sha} for file,sha in frozen['model_files'].items())
    for unit in index['legacy_units']:
        cfg=json.loads(Path(unit['config']['path']).read_text());directory=Path(unit['private_output']);freeze_pin=pin(directory/'FROZEN_COMPLETE.json');frozen=json.loads(Path(freeze_pin['path']).read_text())
        legacy.append({'family':unit['arm_id'],'seed':unit['seed'],'directory':str(directory),'receipt_sha256':freeze_pin['sha256'],'bank':cfg['bank'],'inner_law':cfg['inner_law']})
        sources.extend([freeze_pin,cfg['bank'],cfg['inner_law']])
        for file,key in [('TRAINING_INPUTS.npz','input_sha256'),('PUBLIC_UNIFORM_PRIOR.npz','public_uniform_prior_sha256'),('FROZEN_SCORE_TABLES.npz','score_table_sha256')]:sources.append({'path':str(directory/file),'sha256':frozen[key]})
        sources.extend({'path':str(directory/file),'sha256':sha} for file,sha in frozen['model_files'].items())
    source=json.loads(Path(operational['manifest']['path']).read_text());sources.append(source['roles']['ATTACK_FIT_NEW']['features'])
    sources.extend(source['roles']['INNER_SELECTION'][k] for k in ('features','income','occupation','references_s0','references_s1'))
    historical_pin=pin(root/'.private/task_conditioned_representation/r1/HISTORICAL_RAW_JOINT_SOURCE_PATHS.json');sources.append(historical_pin)
    hm=json.loads(Path(historical_pin['path']).read_text())
    for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION'):sources.append(hm['roles'][role]['features'])
    for seed in ('0','1'):sources.extend(hm['seeds'][seed]['pins'][k] for k in ('complete','model','head0','head1'))
    for r in refit['fits']:
        directory=Path(refit_index['private_output'])/f"{r['family']}__s{r['seed']}"
        sources.extend([{'path':str(directory/'MODEL.joblib'),'sha256':r['model_sha256']},{'path':str(directory/'INNER_SCORE.npz'),'sha256':r['score_sha256']}])
    sources.extend(sealed['reader_freezes'])
    unit_id='R6_FULL_MODEL_READER_ENDPOINT_REPLAY_V1';output=root/'results/pcrl_task_conditioned_representations_v1/r6/FULL_REPLAY_V1.json';config=root/f'.private/task_conditioned_representation/r6/configs/{unit_id}.json'
    cfg={'source_manifest':operational['manifest'],'nomination_lock':operational['nomination_lock'],'reader_lock':lock,'all_reader_banks_frozen':True,'inner_sex':reporting['inner_sex'],'banks':banks,'historical_banks':historical,'historical_manifest':historical_pin,'legacy_banks':legacy,'independent_refits':refit_pin,'independent_refit_root':refit_index['private_output'],'output':str(output)}
    write(config,cfg);sources.append(pin(config))
    for name in ('__init__','science','checkpoint_io','readers','endpoint_math','legacy_replay','historical_replay','replay_worker'):sources.append(pin(root/f'task_conditioned_representation_independent/{name}.py'))
    unique={}
    for source_pin in sources:
        path=str(Path(source_pin['path']).resolve());sha=source_pin['sha256']
        if path in unique and unique[path]!=sha:raise ValueError('conflicting exact source pins')
        unique[path]=sha
    sources=[{'path':path,'sha256':sha} for path,sha in sorted(unique.items())]
    deps=[u['unit_id'] for u in index['units']+index['legacy_units']]+[refit_index['unit_id']]
    spec={'unit_id':unit_id,'stage':'verification','argv':['/Users/nathansamson/PCRL/.venv/bin/python','-m','task_conditioned_representation_independent.replay_worker','--spec',str(config)],'cwd':str(root),'sources':sources,'outputs':[str(output)],'deps':deps,'cpu_reserve_s':1800,'max_wall_s':1800,'max_rss_gib':2,'projected_output_bytes':20971520}
    destination=root/f'.private/task_conditioned_representation/r6/specs/{unit_id}.json';write(destination,spec)
    return {'unit_id':unit_id,'config':pin(config),'spec':pin(destination),'public_output':str(output),'numerical_jobs_launched':0,'reporting_label_array_deserializations':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reader-index',required=True);p.add_argument('--phase',choices=('refits','final'),default='refits');p.add_argument('--operational');p.add_argument('--reporting-config');a=p.parse_args()
    root=Path(a.root).resolve()
    if a.phase=='refits':result=build_refits(root,Path(a.reader_index).resolve());name='INDEPENDENT_REFIT_UNIT_INDEX_V1.json'
    else:result=build_final(root,Path(a.reader_index).resolve(),Path(a.operational).resolve(),Path(a.reporting_config).resolve());name='FULL_REPLAY_UNIT_INDEX_V1.json'
    write(root/'.private/task_conditioned_representation/r6'/name,result)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()

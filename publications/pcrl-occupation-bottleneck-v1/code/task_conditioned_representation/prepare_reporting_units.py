"""Metadata-only preparation of globally frozen reporting and inference units."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .prepare_evaluation_units import BASE,PRIVATE,PUBLIC,COMMON,MODEL,LEGACY,pin,read,write,make_unit,source


def clean(item):
    # Reporting SEX metadata is admitted only after check_global below; no
    # hashing or deserialization of the source or target archive happens here.
    return {'path':str(Path(item['path']).resolve()),'sha256':item['sha256']}


def check_global(index):
    rp=source(index['reporting_lock']);lock=read(rp)
    if lock.get('status')!='ALL_READERS_FROZEN' or lock.get('K_primary')!=51 or lock.get('K_secondary')!=33 or lock.get('K_utility')!=240 or lock.get('fitted_service_banks')!=26:raise PermissionError('global science/reader freeze absent')
    if not lock.get('four_independent_refits_frozen') or not lock.get('all_release_and_reader_parameters_frozen_before_reporting_SEX_bytes'):raise PermissionError('refits/parameters not globally frozen')
    if len(lock.get('reader_freezes',[]))!=28:raise PermissionError('26 banks plus oracle/refit aggregate required')
    for item in lock['reader_freezes']:
        if read(source(item)).get('status')!='FROZEN_COMPLETE':raise PermissionError('a final freeze is incomplete')
    target=clean(index['inner_sex_target']);original=clean(index['inner_sex_source'])
    if target['sha256']!=lock['inner_sex_sha256'] or original['sha256']!=target['sha256']:raise PermissionError('prospective reporting custody SHA differs')
    public=source(index['public_reporting_lock']);read(public)
    return rp,lock,target,original,public


def make(config,module,inputs,code,python,amendment,wall,deps=(),phase=None):
    # make_unit's generic source filter deliberately bars INNER SEX in earlier
    # stages. For the globally frozen stage, add only its declared existing
    # metadata pin after preparing the otherwise identical source list.
    protected=[i for i in inputs if 'inner' in i['path'].lower() and Path(i['path']).name.lower().startswith(('sex','inner_selection_sex'))]
    ordinary=[i for i in inputs if i not in protected]
    result=make_unit(config,module,inputs=ordinary,code=code,python=python,amendment_pin=amendment,wall=wall,deps=deps,phase=phase)
    path=Path(result['spec']['path']);spec=json.loads(path.read_text());by_path={i['path']:i['sha256'] for i in spec['sources']}
    for item in protected:
        if item['path'] in by_path and by_path[item['path']]!=item['sha256']:raise PermissionError('conflicting protected source pin')
        by_path[item['path']]=item['sha256']
    spec['sources']=[{'path':p,'sha256':h} for p,h in sorted(by_path.items())]
    # This metadata file is still unregistered; write final bytes before pin.
    path.write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n');result['spec']=pin(path)
    return result


def prepare(input_path,output,python,attacks_amendment=None):
    index=read(pin(input_path));rp,lock,target,original,public=check_global(index)
    ap=source(attacks_amendment) if attacks_amendment else public;read(ap)
    readers=read(source(index['reader_index']));operational=read(source(index['reader_operational']));manifest=source(operational['manifest'])
    nominal={(i['arm_id'],i['seed']):i for i in operational['instances']}
    refit=read(source(index['independent_refits']));refit_root=BASE/'.private/task_conditioned_representation/r6/refits_v1'
    refit_models=[{'family':r['family'],'seed':r['seed'],'path':str(refit_root/f"{r['family']}__s{r['seed']}"/'MODEL.joblib'),'sha256':r['model_sha256']} for r in refit['fits']]
    for item in refit_models:
        if pin(item['path'])['sha256']!=item['sha256']:raise PermissionError('independent refit model bytes differ before custody')
    frozen=[source(i) for i in lock['reader_freezes']];base=[pin(input_path),rp,public,source(index['nomination_lock']),*frozen,*[source(i) for i in refit_models]]
    custody_uid='R5_REPORTING_LABEL_CUSTODY_V1';custody_cfg={'unit_id':custody_uid,'reporting_lock':rp['path'],'reporting_lock_sha256':rp['sha256'],'inner_sex':target,'inner_sex_source':original,'public_reporting_lock':public,'refit_models':refit_models,
         'private_output':str(PRIVATE/'reporting_label_custody/CUSTODY_RECEIPT.json'),'public_output':str(PUBLIC/'REPORTING_LABEL_CUSTODY_V1.json')}
    custody=make(custody_cfg,'task_conditioned_representation.reporting_custody_worker',[*base,original,source(index['sex_custody_source_metadata'])],(*COMMON,'task_conditioned_representation/reporting_custody_worker.py'),python,ap,120)
    oracle_freeze=read(source(index['oracle_freeze']));oracle_score={'path':str(Path(index['oracle_freeze']['path']).parent/'INNER_LABEL_ORACLE.npz'),'sha256':oracle_freeze['score_sha256']}
    report_units=[];inference_instances=[]
    for entry in readers['units']:
        meta=nominal[(entry['arm_id'],entry['seed'])];directory=Path(entry['private_output']);freeze=pin(directory/'FROZEN_COMPLETE.json');body=read(freeze)
        scores={'path':str(directory/'INNER_SCORES.npz'),'sha256':body['score_sha256']};release={'path':str(directory/'INNER_RELEASE.npz'),'sha256':body['inner_release_sha256']}
        uid=f"R5_REPORT_{entry['arm_id']}_S{entry['seed']}_V1";cfg={'unit_id':uid,'arm_id':entry['arm_id'],'seed':entry['seed'],'point_eligible':meta['point_eligible'],'reporting_lock':rp['path'],'reporting_lock_sha256':rp['sha256'],'inner_sex':target,'freeze':freeze,'scores':scores,'release':release,'oracle_scores':oracle_score,'FIT_surrogate_diagnostic':meta.get('FIT_surrogate_diagnostic'),'public_output':str(PUBLIC/f'{uid}.json'),'private_output':str(PRIVATE/'reports'/uid)}
        report=make(cfg,'task_conditioned_representation.attack_worker',[*base,target,freeze,scores,release,oracle_score],COMMON,python,ap,180,deps=[custody_uid],phase='report');report.update(arm_id=entry['arm_id'],seed=entry['seed']);report_units.append(report)
        inference_instances.append({'arm_id':entry['arm_id'],'seed':entry['seed'],'point_eligible':meta['point_eligible'],'is_nominee_or_exact_alias':meta['is_nominee_or_exact_alias'],'contract':'new_continuous','release':release,'scores':scores})
    for entry in readers['legacy_units']:
        directory=Path(entry['private_output']);fit_cfg=read(source(entry['config']));freeze=pin(directory/'FROZEN_COMPLETE.json');body=read(freeze)
        law=pin(directory/'INNER_LAW.npz');tables={'path':str(directory/'FROZEN_SCORE_TABLES.npz'),'sha256':body['score_table_sha256']}
        uid=f"R5_LEGACY_REPORT_{entry['arm_id']}_S{entry['seed']}_V1";cfg={'unit_id':uid,'arm_id':entry['arm_id'],'seed':entry['seed'],'reporting_lock':rp['path'],'reporting_lock_sha256':rp['sha256'],'inner_sex':target,'freeze':freeze,'score_tables':tables,'inner_law':law,'bank':fit_cfg['bank'],'manifest':manifest['path'],'manifest_sha256':manifest['sha256'],'public_output':str(PUBLIC/f'{uid}.json'),'private_output':str(PRIVATE/'legacy_reports'/uid)}
        report=make(cfg,'task_conditioned_representation.legacy_attack_worker',[*base,target,freeze,tables,law,fit_cfg['bank'],manifest],(*COMMON,*LEGACY),python,ap,180,deps=[custody_uid],phase='report');report.update(arm_id=entry['arm_id'],seed=entry['seed'],contract='legacy_complete_score_wire');report_units.append(report)
        inference_instances.append({'arm_id':entry['arm_id'],'seed':entry['seed'],'point_eligible':True,'is_nominee_or_exact_alias':False,'contract':'legacy_complete_score_wire','bank':fit_cfg['bank'],'inner_law':law,'score_tables':tables})
    utility_dir=PRIVATE/'utility/R5_ORIGINAL_UTILITY_ALL40_V1';utility_receipt=read(pin(utility_dir/'UTILITY_COMPLETE.json'))
    utility_inputs={'path':str(utility_dir/'UTILITY_INPUTS.npz'),'sha256':utility_receipt['utility_input_sha256']};endpoint=pin(BASE/'results/pcrl_task_conditioned_representations_v1/ENDPOINT_REGISTRY.json')
    if endpoint['sha256']!=lock['endpoint_registry_sha256']:raise PermissionError('global frozen endpoint registry differs')
    uid='R5_FINAL_PAIRED_INFERENCE_V1';cfg={'unit_id':uid,'reporting_lock':rp['path'],'reporting_lock_sha256':rp['sha256'],'inner_sex':target,'endpoint_registry':endpoint,'utility_inputs':utility_inputs,'instances':inference_instances,'private_output':str(PRIVATE/'inference'/uid),'public_output':str(PUBLIC/'FINAL_PAIRED_INFERENCE_V1.json')}
    inference_sources=[*base,target,endpoint,utility_inputs]
    for item in inference_instances:
        inference_sources.extend(item[k] for k in ('release','scores','bank','inner_law','score_tables') if k in item)
    infer=make(cfg,'task_conditioned_representation.inference_worker',inference_sources,(*COMMON,'task_conditioned_representation/inference_worker.py',*LEGACY,'conditional_release/inference.py'),python,ap,3600,deps=[custody_uid,*[i['unit_id'] for i in report_units]])
    return write(output,{'schema':'TCR1-prepared-custody-report-inference-v1','reporting_lock':rp,'public_reporting_lock':public,'custody':custody,'reports':report_units,'inference':infer,'continuous_report_count':22,'legacy_report_count':4,'expected_primary_slots':51,'expected_utility_slots':240,'expected_secondary_slots':33,'INNER_SEX_source_or_target_opened_or_hashed_by_preparation':False,'executed':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--python',required=True);p.add_argument('--attacks-amendment');a=p.parse_args()
    override=pin(a.attacks_amendment) if a.attacks_amendment else None
    print(json.dumps(prepare(a.input,a.output,a.python,override)))
if __name__=='__main__':main()

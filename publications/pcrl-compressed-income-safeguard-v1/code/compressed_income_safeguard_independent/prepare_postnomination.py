"""Metadata-only independent phase specs; ROOT alone enqueues and runs them."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .prepare_specs import ROOT,PRIVATE,PUBLIC,metadata_pin,receipt_pin,write

CAMPAIGN_PRIVATE=ROOT/'.private/compressed_income_safeguard'
CAMPAIGN_PUBLIC=ROOT/'results/pcrl_compressed_income_safeguard_v1'
ARMS=('CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY')
SERVICES=ARMS+('RAW_FULL','RAW_HYBRID','RAW_PROJ','U0_FULL','U0_PROJ')
FAMILY='compressed_income_safeguard'


def read(path):
    path=Path(path)
    if path.suffix!='.json':raise PermissionError('metadata helper reads JSON only')
    return json.loads(path.read_text())


def context(amendment=None):
    state=read(CAMPAIGN_PRIVATE/'compute/state.json');revisions=[r for r in state['amendments'] if 'verification' in r['stages']]
    revision=revisions[-1]
    if amendment:
        requested=metadata_pin(amendment);matches=[r for r in revisions if r['path']==requested['path'] and r['sha256']==requested['sha256']]
        if len(matches)!=1:raise ValueError('ROOT must register the chosen verification amendment')
        revision=matches[0]
    code=[metadata_pin(p) for p in sorted((ROOT/'compressed_income_safeguard_independent').glob('*.py'))]
    code += [metadata_pin(ROOT/'compressed_income_safeguard'/name) for name in ('__init__.py','scheduler.py','admission.py','source.py')]
    return {'state':state,'revision':{'path':revision['path'],'sha256':revision['sha256']},'code':code,
            'python':state['units']['CIS_TRAIN_S0_V1']['spec']['argv'][0],
            'science':metadata_pin(CAMPAIGN_PUBLIC/'SCIENCE_LOCK.json')}


def emit(ctx,cfg,module,sources,*,deps=(),wall=300,output_bytes=16*1024**2,extra_outputs=(),replace=False):
    uid=cfg['unit_id']
    if uid in ctx['state']['units']:raise PermissionError('queued unit ID is immutable: '+uid)
    path=PRIVATE/'configs'/f'{uid}.json';write(path,cfg,replace=replace)
    allpins=[metadata_pin(path),ctx['science'],ctx['revision'],*ctx['code'],*sources];unique={}
    for pin in allpins:
        if not isinstance(pin.get('sha256'),str) or len(pin['sha256'])!=64:raise ValueError('source receipt hash missing')
        if pin['path'] in unique and unique[pin['path']]!=pin['sha256']:raise ValueError('conflicting pinned input')
        unique[pin['path']]=pin['sha256']
    outputs=[cfg['public_output'],*map(str,extra_outputs)]
    spec={'unit_id':uid,'stage':'verification','cwd':str(ROOT),
          'argv':[ctx['python'],'-m','compressed_income_safeguard_independent.'+module,'--spec',str(path)],
          'sources':[{'path':p,'sha256':h} for p,h in sorted(unique.items())],'outputs':outputs,
          'deps':list(dict.fromkeys(deps)),'env':{},'max_wall_s':wall,'cpu_reserve_s':wall,
          'max_rss_gib':2,'projected_output_bytes':output_bytes,'amendment_sha256':ctx['revision']['sha256']}
    spec_path=PRIVATE/'specs'/f'{uid}.json';write(spec_path,spec,replace=replace)
    return {'unit_id':uid,'config':metadata_pin(path),'spec':metadata_pin(spec_path),'outputs':outputs}


def packages(path):
    index=read(path);entries=index['entries']
    if len(entries)!=18 or {(p['service'],p['seed']) for p in entries}!={(a,s) for a in SERVICES for s in (0,1)}:
        raise ValueError('exact eighteen final service/source packages required')
    return index,entries


def _fit_pin(trajectory):
    summary=read(trajectory['path']);service=summary['service'];seed=summary['seed'];epoch=summary['epoch']
    tag=service+(f'_E{epoch}' if epoch is not None else '')+f'_S{seed}'
    return summary,receipt_pin(CAMPAIGN_PRIVATE/'trajectory_v2'/tag/'FIT2_TASK_PROBABILITIES.npz',summary['task_probability_sha256'])


def nomination(ctx,input_path,runner_lock,replace):
    source=read(input_path);cfg={'unit_id':'CIS_INDEPENDENT_NOMINATION_V1','public_output':str(PUBLIC/'NOMINATION_REPLAY_V1.json'),'starts':[],'records':[]}
    pins=[metadata_pin(input_path)];deps=[]
    for item in source['records']:
        trajectory=item['trajectory'];summary,fit=_fit_pin(trajectory);freeze=item['freeze'];bank=read(freeze['path']);directory=Path(freeze['path']).parent
        att=receipt_pin(directory/'ATT_RELEASE.npz',bank['att_release_sha256']);scores=receipt_pin(directory/'ATT_SCORES.npz',bank['att_scores_sha256'])
        cfg['records'].append({'arm_id':summary['arm_id'],'seed':summary['seed'],'epoch':summary['epoch'],
                               'fit_inputs':fit,'probability_keys':{'income':'p1','occupation':'p2'},'att_release':att,'att_scores':scores})
        pins += [trajectory,fit,freeze,att,scores];deps.append(summary['unit_id'])
    for trajectory in source['starts']:
        summary,fit=_fit_pin(trajectory)
        if summary['service']!='RAW_HYBRID':raise ValueError('exact RAW_HYBRID starts required')
        cfg['starts'].append({'seed':summary['seed'],'fit_inputs':fit,'probability_keys':{'income':'p1','occupation':'p2'}})
        pins += [trajectory,fit];deps.append(summary['unit_id'])
    if len(cfg['records'])!=32 or len(cfg['starts'])!=2:raise ValueError('all32 checkpoints plus2 starts required')
    if runner_lock and Path(runner_lock).exists():
        wrapper=read(runner_lock)
        if 'nomination' in wrapper:
            # Preserve the public lock; a metadata-only copy exposes its exact
            # immutable nested result to the established independent adapter.
            target=PRIVATE/'inputs'/'RUNNER_NOMINATION_RESULT.json';write(target,wrapper['nomination'],replace=replace)
            cfg['runner_nomination']=metadata_pin(target);pins += [metadata_pin(runner_lock),cfg['runner_nomination']]
        else:cfg['runner_nomination']=metadata_pin(runner_lock);pins.append(cfg['runner_nomination'])
    return [emit(ctx,cfg,'nomination_worker',pins,deps=deps,wall=300,replace=replace)]


def freeze_targets(ctx,index_path,replace):
    if (CAMPAIGN_PUBLIC/'REPORTING_LOCK.json').exists():raise PermissionError('new target selection is closed after reporting freeze')
    index,entries=packages(index_path);cfg={'unit_id':'CIS_INDEPENDENT_TARGET_FREEZE_V1','operation':'freeze_targets',
        'public_output':str(PUBLIC/'REFIT_TARGET_FREEZE_V1.json'),'private_output':str(PRIVATE/'targets/FREEZE_V1'),'banks':[]}
    pins=[metadata_pin(index_path),index['nomination_lock']]
    for p in entries:
        bank={'arm_id':p['service'],'seed':p['seed'],'att_release':p['att_release'],'att_scores':p['att_scores'],
              'inner_release':p['release'],'freeze':p['freeze']}
        cfg['banks'].append(bank);pins += [p[k] for k in ('att_release','att_scores','release','freeze')]
    return [emit(ctx,cfg,'refit_worker',pins,deps=('CIS_INDEPENDENT_NOMINATION_V1',),wall=300,
                 extra_outputs=(Path(cfg['private_output'])/'TARGETS.json',),replace=replace)]


def refit(ctx,receipt_path,registry_path,replace):
    receipt=read(receipt_path);registry_pin=receipt_pin(registry_path,receipt['target_registry_sha256']);registry=read(registry_path)
    if receipt['status']!='FROZEN' or registry['registered_attempts']!=54 or len(registry['targets'])!=54:raise ValueError('queued54-target freeze receipt required')
    result=[]
    for service in SERVICES:
        for seed in (0,1):
            targets=[t for t in registry['targets'] if t['arm_id']==service and t['seed']==seed]
            if len(targets)!=3:raise ValueError('three exact registered attempts per service/source seed')
            uid=f'CIS_INDEPENDENT_REFIT_{service}_S{seed}_V1';out=PRIVATE/'refits'/f'{service}_S{seed}'
            cfg={'unit_id':uid,'operation':'refit','public_output':str(PUBLIC/'refits'/f'{service}_S{seed}.json'),
                 'private_output':str(out),'target_registry':registry_pin,'targets':targets}
            pins=[metadata_pin(receipt_path),registry_pin]
            for t in targets:pins += [t[k] for k in ('att_release','inner_release','freeze')]
            outputs=[out/(t['target_id']+suffix) for t in targets for suffix in ('.joblib','.npz')]
            result.append(emit(ctx,cfg,'refit_worker',pins,deps=('CIS_INDEPENDENT_TARGET_FREEZE_V1',),wall=600,
                               output_bytes=32*1024**2,extra_outputs=outputs,replace=replace))
    return result


def release_replay(ctx,index_path,replace):
    index,entries=packages(index_path);roster_path=CAMPAIGN_PRIVATE/'r1/ROSTER_METADATA.json';roster={r['unit_id']:r for r in read(roster_path)['entries']}
    manifest=read(index['manifest']['path']);cfg={'unit_id':'CIS_INDEPENDENT_FINAL_RELEASE_FORWARDS_V1',
        'public_output':str(PUBLIC/'FINAL_RELEASE_FORWARDS_V1.json'),'manifest':index['manifest'],'packages':[],'aliases':[]}
    pins=[metadata_pin(index_path),metadata_pin(roster_path),index['manifest'],index['nomination_lock']]
    pins += [manifest['roles'][role]['features'] for role in ('OSF_DEFENSE_FIT','ATTACK_FIT_NEW','INNER_SELECTION')]
    for p in entries:
        service=p['service'];rules=('full','full') if service in ('RAW_FULL','U0_FULL') else ('head_row_space','head_row_space') if service in ('RAW_PROJ','U0_PROJ') else ('head_row_space','full')
        item={'arm_id':service,'seed':p['seed'],'checkpoint':p['checkpoint'],'geometry_binding':p['geometry'],
              'wire_contract':dict(zip(('income','occupation'),rules)),
              'fit_release':p['fit_release'],'att_release':p['att_release'],'inner_release':p['release']}
        if service in ARMS:
            source=roster[f'RAW_JOINT_R03_E5_S{p["seed"]}'];item['frozen_source_checkpoint']=receipt_pin(source['checkpoint'],source['checkpoint_sha256']);pins.append(item['frozen_source_checkpoint'])
        cfg['packages'].append(item);pins += [p[k] for k in ('checkpoint','geometry','fit_release','att_release','release')]
        if 'rule' not in cfg:cfg['rule']=read(p['geometry']['path'])['heads'][0]['rule']
    for seed in (0,1):
        for service in (*ARMS,'RAW_PROJ'):cfg['aliases'].append({'left':service,'right':'RAW_HYBRID','seed':seed,'view':'r1'})
        cfg['aliases'].append({'left':'RAW_FULL','right':'RAW_HYBRID','seed':seed,'view':'r2'})
    for p in entries:
        if p['service'] in ARMS and p['epoch']==0:
            cfg['aliases'].append({'left':p['service'],'right':'RAW_HYBRID','seed':p['seed'],'view':'r2'})
    return [emit(ctx,cfg,'release_replay_worker',pins,wall=600,replace=replace)]


def final(ctx,index_path,report_index_path,replace):
    index,entries=packages(index_path);report=read(report_index_path);seal=report['reporting_lock'];manifest=read(index['manifest']['path'])
    cfg={'unit_id':'CIS_INDEPENDENT_FINAL_ENDPOINTS_V1','public_output':str(PUBLIC/'FINAL_ENDPOINTS_V1.json'),
        'private_output':str(PRIVATE/'final/METRIC_VECTORS_V1'),'manifest':index['manifest'],
        'reporting_lock':{'path':seal['path'],'sha256':seal['sha256']},'truth':seal['roles']['INNER_SELECTION'],'banks':[],
        'runner_control_gates':report['arrays']['control_gates']}
    pins=[metadata_pin(index_path),metadata_pin(report_index_path),index['manifest'],cfg['reporting_lock'],cfg['runner_control_gates']]
    pins += [cfg['truth'][t] for t in ('income','occupation','sex')]
    pins += [manifest['roles']['INNER_SELECTION'][f'references_s{s}'] for s in (0,1)]
    outputs=[]
    for p in entries:
        service=p['service'];seed=p['seed'];vectors=report['arrays'][f'metrics__{service}__S{seed}']
        bank={'arm_id':service,'seed':seed,'release':p['release'],'scores':p['scores'],'freeze':p['freeze'],'metric_vectors':vectors}
        if service in ('U0_FULL','RAW_FULL'):
            bank.update(att_release=p['att_release'],att_scores=p['att_scores']);pins += [p['att_release'],p['att_scores']]
        cfg['banks'].append(bank)
        pins += [p[k] for k in ('release','scores','freeze')]+[vectors]
        outputs.append(Path(cfg['private_output'])/f'METRICS_{service}_S{seed}.npz')
    outputs.append(Path(cfg['private_output'])/'CONTROL_GATES.npz')
    return [emit(ctx,cfg,'final_worker',pins,deps=('CIS_INNER_REPORT_V1','CIS_INDEPENDENT_FINAL_RELEASE_FORWARDS_V1'),wall=600,
                 extra_outputs=outputs,output_bytes=64*1024**2,replace=replace)]


def k240(ctx,report_index_path,runner_summary,replace):
    report=read(report_index_path);seal=report['reporting_lock'];final_uid='CIS_INDEPENDENT_FINAL_ENDPOINTS_V1'
    completed=ctx['state']['units'].get(final_uid)
    if not completed or completed['status']!='done':raise ValueError('independent row metrics must finish before own K240 replay')
    arrays=dict(report['arrays']);private=PRIVATE/'final/METRIC_VECTORS_V1'
    for service in SERVICES:
        for seed in (0,1):
            path=private/f'METRICS_{service}_S{seed}.npz';arrays[f'metrics__{service}__S{seed}']=receipt_pin(path,completed['output_sha256'][str(path)])
    gate_path=private/'CONTROL_GATES.npz';arrays['control_gates']=receipt_pin(gate_path,completed['output_sha256'][str(gate_path)])
    cfg={'unit_id':'CIS_INDEPENDENT_K240_REPLAY_V1','operation':'assessment','public_output':str(PUBLIC/'K240_REPLAY_V1.json'),
         'private_output':str(PRIVATE/'assessment/K240_V1'),'reporting_lock':{'path':seal['path'],'sha256':seal['sha256']},
         'endpoint_registry':report['endpoint_registries'][FAMILY],'unit_input':report['unit_input'],'arrays':arrays}
    pins=[metadata_pin(report_index_path),cfg['reporting_lock'],cfg['endpoint_registry'],cfg['unit_input'],*arrays.values()]
    if runner_summary:
        cfg['runner_summary']=metadata_pin(runner_summary);pins.append(cfg['runner_summary'])
    return [emit(ctx,cfg,'replay',pins,deps=(final_uid,'CIS_INNER_INFERENCE_V1'),wall=1800,output_bytes=128*1024**2,
                 extra_outputs=(Path(cfg['private_output'])/'BOOTSTRAP.npz',Path(cfg['private_output'])/'WEIGHTS.npz'),replace=replace)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('nomination','freeze_targets','refit','release_replay','final','K240'))
    parser.add_argument('--package-index',default=str(CAMPAIGN_PRIVATE/'r4/FINAL_PACKAGE_INDEX_V2.json'))
    parser.add_argument('--nomination-input',default=str(CAMPAIGN_PRIVATE/'r4/configs/CIS_NOMINATION_V2.json'))
    parser.add_argument('--nomination-lock',default=str(CAMPAIGN_PUBLIC/'NOMINATION_LOCK.json'))
    parser.add_argument('--target-registry',default=str(PRIVATE/'targets/FREEZE_V1/TARGETS.json'))
    parser.add_argument('--freeze-receipt',default=str(PUBLIC/'REFIT_TARGET_FREEZE_V1.json'))
    parser.add_argument('--report-index',default=str(CAMPAIGN_PRIVATE/'reporting/RUNNER_INNER_REPORT_V1/REPORT_INPUT_INDEX.json'))
    parser.add_argument('--runner-summary',default=str(CAMPAIGN_PUBLIC/'assessment/COMBINED_INFERENCE.json'))
    parser.add_argument('--amendment');parser.add_argument('--replace',action='store_true');args=parser.parse_args();ctx=context(args.amendment)
    if args.phase=='nomination':units=nomination(ctx,args.nomination_input,args.nomination_lock,args.replace)
    elif args.phase=='freeze_targets':units=freeze_targets(ctx,args.package_index,args.replace)
    elif args.phase=='refit':units=refit(ctx,args.freeze_receipt,args.target_registry,args.replace)
    elif args.phase=='release_replay':units=release_replay(ctx,args.package_index,args.replace)
    elif args.phase=='final':units=final(ctx,args.package_index,args.report_index,args.replace)
    else:units=k240(ctx,args.report_index,args.runner_summary,args.replace)
    result={'schema':'CIS1-independent-phase-preparation-v1','status':'PREPARED_NOT_ENQUEUED','phase':args.phase,
            'unit_count':len(units),'units':units,'numerical_artifact_bytes_read_or_hashed':False,'numerical_jobs_launched':0}
    write(PRIVATE/f'GENERATED_{args.phase.upper()}_V1.json',result,replace=args.replace)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()

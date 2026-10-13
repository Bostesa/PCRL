"""Metadata-only specifications; coordinator alone registers and launches units.

Numeric artifact hashes are carried from admission receipts, never recomputed
here. Only code and JSON metadata are read or hashed by this builder.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from .admission import sha_metadata,read_metadata
from .source import write_json
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'.private/compressed_income_safeguard/r5'
PUBLIC=ROOT/'results/pcrl_compressed_income_safeguard_v1/r5'


def metadata_pin(p):
    p=Path(p).resolve();return {'path':str(p),'sha256':sha_metadata(p)}


def exact(items):
    out={}
    for i in items:
        p=str(Path(i['path']).resolve())
        if p in out and out[p]!=i['sha256']:raise ValueError('conflicting exact pins')
        out[p]=i['sha256']
    return [{'path':p,'sha256':h} for p,h in sorted(out.items())]


def code_pins(base=ROOT):
    # Code discovery only; neither source datasets nor private trees are walked.
    return [metadata_pin(p) for ns in ('compressed_income_safeguard','task_conditioned_representation','conditional_release') for p in sorted((Path(base)/ns).glob('*.py'))]


def write_unit(cfg,sources,*,python=sys.executable,cwd=ROOT,deps=(),max_wall_s=180,output_bytes=20000000):
    config=PRIVATE/'configs'/f"{cfg['unit_id']}.json";write_json(config,cfg,private=True)
    spec={'unit_id':cfg['unit_id'],'stage':cfg['stage'],'cwd':str(cwd),'argv':[python,'-m','compressed_income_safeguard.'+('custody_worker' if cfg['operation'] in ('plan','snapshot','off_device') else 'deployment_worker'),'--spec',str(config)],'max_wall_s':max_wall_s,'cpu_reserve_s':max_wall_s,'max_rss_gib':2,'projected_output_bytes':output_bytes,'sources':exact([metadata_pin(config),*sources]),'deps':list(deps),'outputs':[cfg['public_output']]}
    # Custody uses restore while deployment restore uses package fields.
    if cfg['operation']=='restore' and 'package' not in cfg:spec['argv'][2]='compressed_income_safeguard.custody_worker'
    path=PRIVATE/'specs'/f"{cfg['unit_id']}.json";write_json(path,spec,private=True);return {'config':metadata_pin(config),'spec':metadata_pin(path),'public_output':cfg['public_output']}


def prepare_live(index,output):
    data=read_metadata(index);manifest_pin=data['manifest'];manifest=read_metadata(manifest_pin['path']);role=data.get('role','INNER_SELECTION');units=[]
    services={'CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY','RAW_FULL','RAW_HYBRID','RAW_PROJ','U0_FULL','U0_PROJ'}
    identities=[(x['arm_id'],x['seed']) for x in data['instances']]
    if len(identities)!=18 or set(identities)!={(arm,seed) for arm in services for seed in (0,1)}:
        raise PermissionError('exact nine services by two admitted source seeds required')
    for item in data['instances']:
        arm=item['arm_id'];seed=item['seed'];tag=f'{arm}_S{seed}';uid='R5_DEPLOY_'+tag.upper()+'_V1'
        geometry=item['geometry'];bank=item['bank_wire'];geometry_meta=read_metadata(geometry['path'])
        if geometry_meta['binding_sha256']!=item['geometry_binding_sha256'] or geometry_meta['source_seed']!=seed:
            raise PermissionError('per-seed admitted geometry metadata differs')
        role=item.get('role',data.get('role','INNER_SELECTION'))
        cfg={'unit_id':uid,'stage':'verification','operation':'export_live','arm_id':arm,'seed':seed,
             'service_variant':arm,'geometry':geometry,'geometry_binding_sha256':item['geometry_binding_sha256'],
             'bank_wire':bank,'bank_slice':item.get('bank_slice','whole_role'),
             'manifest':manifest_pin['path'],'manifest_sha256':manifest_pin['sha256'],
             'checkpoint':item['checkpoint'],'checkpoint_sha256':item['checkpoint_sha256'],'role':role,
             'package':str(PRIVATE/'packages'/tag),'input':str(PRIVATE/'inputs'/f'{tag}.npz'),
             'expected':str(PRIVATE/'expected'/f'{tag}.npz'),'public_output':str(PUBLIC/f'{uid}.json')}
        for name in ('science_lock','diagnostic_lock'):
            if name in data:cfg[name]=data[name]['path'];cfg[name+'_sha256']=data[name]['sha256']
        sources=[manifest_pin,{'path':item['checkpoint'],'sha256':item['checkpoint_sha256']},geometry,bank,
                 manifest['roles'][role]['features'],*code_pins(),
                 *[data[k] for k in ('science_lock','diagnostic_lock') if k in data]]
        unit=write_unit(cfg,sources,python=data.get('python',sys.executable),deps=item.get('deps',data.get('deps',[])))
        unit.update({k:cfg[k] for k in ('arm_id','seed','service_variant','package','input','expected')});units.append(unit)
    write_json(output,{'schema':'CIS1-prepared-live-deployment-v1','units':units,'unit_count':len(units),'executed':False},private=True)
    return len(units)


def prepare_restore(live_index,restore_root,reader_index,output,tag='LOCAL',python=sys.executable):
    live=read_metadata(live_index);root=Path(restore_root).resolve();plan=read_metadata(root/'files/campaign/r5/CUSTODY_PLAN.private.json') if (root/'files/campaign/r5/CUSTODY_PLAN.private.json').exists() else read_metadata(root/'PLAN.private.json')
    remapping={e['path']:root/'files'/e['alias'] for e in plan['entries']}
    def remap(path):
        if path not in remapping:raise PermissionError('needed file absent from exact snapshot plan: '+Path(path).name)
        return remapping[path]
    readers=read_metadata(reader_index);units=[]
    for item in live['units']:
        receipt=read_metadata(remap(item['public_output']));variant=item['service_variant'];identity=(item['arm_id'],item['seed'])
        choices=[r for r in readers['instances'] if (r['arm_id'],r['seed'])==identity]
        if len(choices)!=1 or not choices[0]['readers']:raise ValueError('unique frozen representative reader records required')
        package=remap(str(Path(item['package'])/'DEPLOYMENT.json')).parent;files=[{'path':str(package/'DEPLOYMENT.json'),'sha256':receipt['package_manifest_sha256']},{'path':str(package/'model_state.pt'),'sha256':receipt['model_sha256']}]
        files.append({'path':str(package/'geometry.json'),'sha256':receipt['geometry']['sha256']})
        rr=[]
        for record in choices[0]['readers']:
            record=dict(record)
            for key in ('path','expected'):record[key]=str(remap(record[key]))
            rr.append(record)
        uid=f"R5_RESTORE_{tag}_{item['arm_id']}_S{item['seed']}_{variant}_V1".upper()
        cfg={'unit_id':uid,'stage':'custody','operation':'restore','restore_root':str(root),'arm_id':item['arm_id'],'seed':item['seed'],'service_variant':variant,'package':str(package),'package_files':files,'package_manifest_sha256':receipt['package_manifest_sha256'],'input':str(remap(item['input'])),'input_sha256':receipt['input_sha256'],'expected':str(remap(item['expected'])),'expected_sha256':receipt['expected_sha256'],'environment_pins':receipt['environment_pins'],'readers':rr,'public_output':str(PUBLIC/f'{uid}.json')}
        sources=[*files,{'path':cfg['input'],'sha256':cfg['input_sha256']},{'path':cfg['expected'],'sha256':cfg['expected_sha256']},*code_pins(root/'files/code')]
        for r in rr:sources.extend({'path':r[k],'sha256':r[k+'_sha256']} for k in ('path','expected'))
        unit=write_unit(cfg,sources,python=python,cwd=root/'files/code');unit.update(arm_id=item['arm_id'],seed=item['seed'],service_variant=variant);units.append(unit)
    write_json(output,{'schema':'CIS1-prepared-backup-only-numerical-replay-v1','units':units,'unit_count':len(units),'executed':False,'original_numerical_inputs_used':False},private=True);return len(units)


def prepare_custody(config,output,python=sys.executable):
    cfg=read_metadata(config);operation=cfg['operation'];sources=code_pins();total=1
    if operation=='plan':sources.extend({'path':e['path'],'sha256':e['sha256']} for e in cfg['entries'])
    elif operation=='snapshot':
        plan=read_metadata(cfg['plan']);sources.extend([{'path':cfg['plan'],'sha256':cfg['plan_sha256']},*[{'path':e['path'],'sha256':e['sha256']} for e in plan['entries']]]);total=sum(e['size_bytes'] for e in plan['entries'])
    elif operation in ('restore','off_device'):
        root=Path(cfg['snapshot']);receipt=read_metadata(root/'SNAPSHOT_RECEIPT.json');sources.extend([{'path':str(root/'SNAPSHOT_RECEIPT.json'),'sha256':cfg['snapshot_receipt_sha256']},{'path':str(root/'PLAN.private.json'),'sha256':receipt['plan_sha256']},*[{'path':str(root/'files'/e['alias']),'sha256':e['sha256']} for e in receipt['entries']]]);total=sum(e['size_bytes'] for e in receipt['entries'])
    else:raise ValueError('unregistered custody operation')
    unit=write_unit(cfg,sources,python=python,deps=cfg.get('deps',[]),max_wall_s=1200,output_bytes=total);write_json(output,unit,private=True);return unit

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True);a=sub.add_parser('live');a.add_argument('index');a.add_argument('output');a=sub.add_parser('restore');a.add_argument('index');a.add_argument('restore_root');a.add_argument('reader_index');a.add_argument('output');a.add_argument('--tag',default='LOCAL');a=sub.add_parser('custody');a.add_argument('config');a.add_argument('output');a=p.parse_args()
    result=prepare_live(a.index,a.output) if a.action=='live' else prepare_restore(a.index,a.restore_root,a.reader_index,a.output,a.tag) if a.action=='restore' else prepare_custody(a.config,a.output);print(json.dumps(result))


def prepare_reader_index(instances,output):
    """Bind LR and ATT-frozen envelope representatives, without opening scores.

    Each instance names a frozen bank. `alias_directories` maps actual view to
    the old bank directory for exact-wire reused models, if needed. The saved
    expected score archive remains the new instance's frozen INNER scores.
    """
    result=[]
    for instance in instances:
        root=Path(instance['directory']);freeze=read_metadata(root/'FROZEN_COMPLETE.json')
        if freeze.get('status')!='FROZEN_COMPLETE' or freeze['source_seed']!=instance['seed'] or not freeze.get('score_sha256'):raise PermissionError('frozen label-free reader score bank required')
        records=[]
        for reported_view in ('r1','r2','pair'):
            summary=freeze['ATT_validation'][reported_view]
            primary=summary.get('contracts',{}).get('primary')
            strongest=primary['strongest_reader'] if primary else summary.get('strongest_reader','LR_C1')
            for reported_name in dict.fromkeys(('LR_C1',strongest)):
                view=reported_view;name=reported_name
                if name.startswith('IGNORE_'):
                    ignored,name=name.split('__',1)
                    if view!='pair' or ignored not in ('IGNORE_r1','IGNORE_r2'):raise ValueError('invalid pair ignore alias')
                    view='r2' if ignored=='IGNORE_r1' else 'r1'
                filename=f'MODEL_{view}__{name}.joblib';model_dir=root;model_sha=freeze['model_files'].get(filename)
                if model_sha is None:
                    directory=instance.get('alias_directories',{}).get(view)
                    if not directory:raise ValueError('exact reused model directory missing for '+view+'/'+name)
                    model_dir=Path(directory);original=read_metadata(model_dir/'FROZEN_COMPLETE.json');entry=freeze.get('reused_readers',{}).get(view,{}).get(name)
                    if entry is None or metadata_pin(model_dir/'FROZEN_COMPLETE.json')['sha256']!=entry['source_freeze_sha256']:raise PermissionError('reader alias freeze differs')
                    model_sha=original['model_files'][filename]
                    if model_sha!=entry['model_sha256']:raise PermissionError('reader alias model differs')
                records.append({'path':str(model_dir/filename),'path_sha256':model_sha,'expected':str(root/'INNER_SCORES.npz'),'expected_sha256':freeze['score_sha256'],'view':view,'reader_name':name,'orientation':freeze['validation'][view][name]['orientation'],'score_field':reported_view+'__'+reported_name,'purpose':'representative_linear' if reported_name=='LR_C1' else 'ATT_frozen_primary_envelope_strongest'})
        result.append({'arm_id':instance['arm_id'],'seed':instance['seed'],'service_variant':instance['arm_id'],'readers':records})
    write_json(output,{'schema':'CIS1-frozen-restore-reader-index-v1','instances':result,'reader_records':sum(len(x['readers']) for x in result),'scientific_outputs_or_labels_read':False},private=True);return result

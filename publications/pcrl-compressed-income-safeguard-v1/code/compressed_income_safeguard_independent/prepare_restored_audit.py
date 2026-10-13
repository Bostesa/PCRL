"""Independent metadata-only restored representative model/reader queue spec."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .prepare_specs import ROOT,PRIVATE,PUBLIC,metadata_pin,write
from .prepare_postnomination import CAMPAIGN_PRIVATE,read

REPRESENTATIVES=('CONSTRAINED','ORDINARY','RAW_HYBRID','RAW_PROJ')


def prepare(restore_root,tag,*,replace=False):
    root=Path(restore_root).absolute()
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):raise PermissionError('restore root symlink refused')
    root=root.resolve(strict=True);plan_path=root/'PLAN.private.json';plan=read(plan_path)
    mapping={e['path']:{'path':str(root/'files'/e['alias']),'sha256':e['sha256']} for e in plan['entries']}
    if len(mapping)!=len(plan['entries']):raise ValueError('duplicate original snapshot path')
    def remap(path,digest):
        item=mapping.get(str(path))
        if not item or item['sha256']!=digest:raise PermissionError('restored snapshot pin missing or different '+str(path))
        p=Path(item['path'])
        if not p.is_file() or not p.resolve(strict=True).is_relative_to(root):raise PermissionError('restored snapshot file absent/outside')
        return item
    state=read(CAMPAIGN_PRIVATE/'compute/state.json');amendments=[a for a in state['amendments'] if 'custody' in a['stages']]
    revision=amendments[-1];uid=f'CIS_INDEPENDENT_RESTORED_REPRESENTATIVES_{tag}_V1'
    if uid in state['units']:raise PermissionError('queued unit identity immutable')
    cfg={'unit_id':uid,'operation':'restore','restore_root':str(root),'packages':[],
         'public_output':str(PUBLIC/f'RESTORED_REPRESENTATIVES_{tag}_V1.json')}
    sources=[metadata_pin(plan_path)]
    live=read(CAMPAIGN_PRIVATE/'r5/GENERATED_LIVE.json')
    for unit in live['units']:
        arm=unit['arm_id'];seed=unit['seed']
        if arm not in REPRESENTATIVES:continue
        receipt=read(unit['public_output']);directory=Path(unit['package'])
        package={'arm_id':arm,'seed':seed,'deployment':remap(directory/'DEPLOYMENT.json',receipt['package_manifest_sha256']),
                 'model_state':remap(directory/'model_state.pt',receipt['model_sha256']),
                 'geometry':remap(directory/'geometry.json',receipt['geometry']['sha256']),
                 'input':remap(unit['input'],receipt['input_sha256']),'expected_wire':remap(unit['expected'],receipt['expected_sha256']),
                 'readers':[]}
        sources += [v for v in package.values() if isinstance(v,dict)]
        refit_path=PUBLIC/'refits'/f'{arm}_S{seed}.json';refit=read(refit_path)
        if refit['status']!='PASS' or len(refit['records'])!=3:raise ValueError('actual immutable refit receipt required')
        for record in refit['records']:
            tid=record['target_id'];base=PRIVATE/'refits'/f'{arm}_S{seed}'
            reader={'model':remap(base/(tid+'.joblib'),record['model_sha256']),
                    'expected':remap(base/(tid+'.npz'),record['score_sha256']),
                    'reader':record['reader'],'view':record['view'],'orientation':record['orientation']}
            package['readers'].append(reader);sources += [reader['model'],reader['expected']]
        cfg['packages'].append(package)
    if len(cfg['packages'])!=8 or {(p['arm_id'],p['seed']) for p in cfg['packages']}!={(a,s) for a in REPRESENTATIVES for s in (0,1)}:
        raise ValueError('exact requested eight representative services/source seeds')
    code=[{'path':str(root/'files'/e['alias']),'sha256':e['sha256']} for e in plan['entries'] if e['alias'].startswith('code/') and e['alias'].endswith('.py')]
    config_path=PRIVATE/'configs'/f'{uid}.json';write(config_path,cfg,replace=replace)
    pins=[metadata_pin(config_path),*code,*sources];unique={}
    for item in pins:
        if item['path'] in unique and unique[item['path']]!=item['sha256']:raise ValueError('conflicting restored source pin')
        unique[item['path']]=item['sha256']
    spec={'unit_id':uid,'stage':'custody','cwd':str(root/'files/code'),
          'argv':[state['units']['CIS_TRAIN_S0_V1']['spec']['argv'][0],'-m','compressed_income_safeguard_independent.deployment_replay_worker','--spec',str(config_path)],
          'sources':[{'path':p,'sha256':h} for p,h in sorted(unique.items())],'outputs':[cfg['public_output']],
          'max_wall_s':300,'cpu_reserve_s':300,'max_rss_gib':2,'projected_output_bytes':4*1024**2,
          'deps':[],'env':{},'amendment_sha256':revision['sha256']}
    spec_path=PRIVATE/'specs'/f'{uid}.json';write(spec_path,spec,replace=replace)
    result={'schema':'CIS1-independent-restored-representative-spec-v1','status':'PREPARED_NOT_ENQUEUED','unit_id':uid,
            'config':metadata_pin(config_path),'spec':metadata_pin(spec_path),'representative_package_count':8,'frozen_refit_reader_replays':24,
            'bound_custody_amendment':{'path':revision['path'],'sha256':revision['sha256']},
            'numerical_artifact_bytes_read_or_hashed':False,'numerical_jobs_launched':0}
    write(PRIVATE/f'GENERATED_RESTORED_REPRESENTATIVES_{tag}_V1.json',result,replace=replace);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--restore-root',required=True);p.add_argument('--tag',choices=('LOCAL','OFF_DEVICE'),required=True)
    p.add_argument('--replace',action='store_true');args=p.parse_args();print(json.dumps(prepare(args.restore_root,args.tag,replace=args.replace),sort_keys=True))

if __name__=='__main__':main()

"""Metadata-only preparation using exclusively the restored snapshot tree."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path

FAMILIES=('U_NEW','JOINT_CONDITIONAL','LOCAL_TWIN','SEQ12','SEQ21','H_RAW_CSTAR')

def pin(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def prepare(restore_root,output_root,deps):
    restore_root=Path(restore_root).resolve();output_root=Path(output_root).resolve();code=restore_root/'files/code';campaign=restore_root/'files/campaign';packages=[];sources=[]
    def metadata(path):
        if not path.resolve().is_relative_to(restore_root):raise PermissionError('restored metadata only')
        sources.append(pin(path));return json.loads(path.read_text())
    for family in FAMILIES:
        for seed in (0,1):
            package=campaign/f'r7/packages/{family}_s{seed}';manifest=package/'DEPLOYMENT.json';m=metadata(manifest)
            directory=campaign/f'r5/readers/R5_FIT_{family}_S{seed}_V1';freeze=metadata(directory/'FROZEN_COMPLETE.json')
            report=metadata(code/f'results/pcrl_task_conditioned_representations_v1/r5/R5_REPORT_{family}_S{seed}_V1.json')
            inp=campaign/f'r7/inputs/{family}_s{seed}.npz';expected=campaign/f'r7/expected/{family}_s{seed}.npz';reader_target={'path':str(directory/'INNER_SCORES.npz'),'sha256':freeze['score_sha256']};readers=[]
            for view in ('r1','r2','pair'):
                for role,selected in [('LR_C1','LR_C1'),('actual_strongest',report['views'][view]['strongest_reader'])]:
                    if selected.startswith('IGNORE_'):
                        other,name=selected.split('__',1);model_view='r1' if other=='IGNORE_r2' else 'r2' if other=='IGNORE_r1' else None
                        if model_view is None:raise ValueError('unknown pair ignore-other alias')
                    else:model_view,name=view,selected
                    filename=f'MODEL_{model_view}__{name}.joblib';model_pin={'path':str(directory/filename),'sha256':freeze['model_files'][filename]}
                    readers.append({'view':view,'model_view':model_view,'name':name,'role':role,'model':model_pin,'orientation':freeze['validation'][model_view][name]['orientation'],'target':reader_target,'score_field':f'{view}__{selected}'})
                    sources.extend([model_pin,reader_target])
            entry={'family':family,'seed':seed,'manifest':pin(manifest),'input':pin(inp),'expected':pin(expected),'readers':readers};packages.append(entry)
            sources.extend([entry['manifest'],entry['input'],entry['expected'],{'path':str(package/'model_state.pt'),'sha256':m['model_sha256']}])
    aggregate=metadata(code/'results/pcrl_task_conditioned_representations_v1/r6/INDEPENDENT_PAIR_REFITS_V1.json');refits=[]
    for record in aggregate['fits']:
        directory=campaign/f"r6/refits_v1/{record['family']}__s{record['seed']}";model={'path':str(directory/'MODEL.joblib'),'sha256':record['model_sha256']};target={'path':str(directory/'INNER_SCORE.npz'),'sha256':record['score_sha256']}
        refits.append({'family':record['family'],'seed':record['seed'],'name':record['reader'],'orientation':record['orientation'],'model':model,'target':target});sources.extend([model,target])
    unit='R6_INDEPENDENT_BACKUP_ONLY_REPLAY_V1';config=output_root/'configs'/f'{unit}.json';spec_path=output_root/'specs'/f'{unit}.json';public=output_root/'BACKUP_ONLY_REPLAY_V1.json'
    for path in (config,spec_path):
        if path.exists():raise FileExistsError('preserve existing restore unit')
        path.parent.mkdir(parents=True,exist_ok=True)
    cfg={'restore_root':str(restore_root),'packages':packages,'independent_refits':refits,'output':str(public)};config.write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n');sources.append(pin(config))
    for name in ('__init__','science','readers','backup_replay_worker'):sources.append(pin(code/f'task_conditioned_representation_independent/{name}.py'))
    sources.append(pin(code/'task_conditioned_representation/scheduler.py'))
    unique={r['path']:r['sha256'] for r in sources};sources=[{'path':p,'sha256':sha} for p,sha in sorted(unique.items())]
    spec={'unit_id':unit,'stage':'verification','cwd':str(code),'argv':[sys.executable,'-m','task_conditioned_representation_independent.backup_replay_worker','--spec',str(config)],'sources':sources,'outputs':[str(public)],'deps':deps,'cpu_reserve_s':120,'max_wall_s':120,'max_rss_gib':2,'projected_output_bytes':1048576};spec_path.write_text(json.dumps(spec,indent=2,sort_keys=True)+'\n')
    result={'unit_id':unit,'config':pin(config),'spec':pin(spec_path),'public_output':str(public),'numerical_jobs_launched':0,'original_sources_read':False,'restored_packages':12,'independent_refits':4};(output_root/'BACKUP_ONLY_REPLAY_UNIT_INDEX_V1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--restore-root',required=True);p.add_argument('--output-root',required=True);p.add_argument('--deps',default='');a=p.parse_args();print(json.dumps(prepare(a.restore_root,a.output_root,list(filter(None,a.deps.split(','))))))

if __name__=='__main__':main()

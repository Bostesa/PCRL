"""Queue-only duplicate cache premises; final INNER SEX remains closed."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .source import load_manifest,load_allowed_role,sha_file,write_once,ROLES
from .scheduler import assert_active_from_env


def run(manifest_path,manifest_sha256,output):
    active=assert_active_from_env(stage='admission')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    if pins.get(str(Path(manifest_path).absolute()))!=manifest_sha256:raise PermissionError('unqueued manifest')
    manifest=load_manifest(manifest_path,manifest_sha256)
    def check(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:
                if pins.get(x['path'])!=x['sha256']:raise PermissionError('source unqueued')
            else:
                for z in x.values():check(z)
    check(manifest)
    rows=[]
    for seed in (0,1):
        for role in ROLES:
            fields={'features'}|({'sex'} if role!='INNER_SELECTION' else set())|({'income','occupation'} if role!='ATTACK_FIT_NEW' else set())
            values=load_allowed_role(manifest,seed,role,fields);wire=values['wire'];unit=wire['unit'];unique,count=np.unique(unit,return_counts=True)
            duplicate=unique[count>1];stats={'seed':seed,'role':role,'duplicate_groups':len(duplicate),'duplicate_rows':int(count[count>1].sum()),
                'raw83_float32_consistent':True,'p0_public_wire_consistent':True,'sex_consistent':None if role=='INNER_SELECTION' else True,
                'task_labels_consistent':None if role=='ATTACK_FIT_NEW' else True}
            for group in duplicate:
                mask=unit==group
                raw=values['features']['X'][mask].astype(np.float32)
                stats['raw83_float32_consistent'] &= bool(np.all(raw==raw[0]))
                for key in ('code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'):
                    arr=wire[key][mask];stats['p0_public_wire_consistent'] &= bool(np.all(arr==arr[0]))
                if 'sex' in values:
                    sex=values['sex']['sex'][mask];stats['sex_consistent'] &= bool(np.all(sex==sex[0]))
                if 'income' in values:
                    for task in ('income','occupation'):
                        y=values[task][task][mask];stats['task_labels_consistent'] &= bool(np.all(y==y[0]))
            rows.append(stats)
    report={'schema':'CBR1-duplicate-cache-source-premise-v1','status':'PASS' if all(x['raw83_float32_consistent'] and x['p0_public_wire_consistent'] and x['sex_consistent'] is not False for x in rows) else 'PREMISE_FAILURE',
            'source_manifest_sha256':manifest_sha256,'by_role_seed':rows,'inner_sex_read':False,
            'inner_sex_remaining':'final reporting owner must verify duplicate-group SEX consistency after reader/report seal'}
    write_once(output,report,False);print(json.dumps({'status':report['status'],'receipt_sha256':sha_file(output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.manifest,a.manifest_sha256,a.output)

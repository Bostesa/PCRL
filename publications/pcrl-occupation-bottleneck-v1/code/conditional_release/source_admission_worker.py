"""Queued exact source admission; no protected-label reader or optimizer fit."""
import argparse,json,math,os
from pathlib import Path
import numpy as np
from .source import load_manifest,load_allowed_role,load_bank,load_historical_policy,donor_law,sha_file,write_once,ROLES
from .deployment import load_p0_service


def run(manifest_path,manifest_sha256,output):
    from .scheduler import assert_active_from_env
    active=assert_active_from_env(stage="admission")
    queue=json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    registered={p["path"]:p["sha256"] for p in queue["units"][active["unit_id"]]["spec"]["sources"]}
    if registered.get(str(Path(manifest_path).absolute()))!=manifest_sha256:raise PermissionError("source manifest not queued")
    manifest=load_manifest(manifest_path,manifest_sha256)
    def check(value):
        if isinstance(value,dict):
            if "path" in value and "sha256" in value:
                if registered.get(value["path"])!=value["sha256"]:raise PermissionError("source pin not queued")
            else:
                for item in value.values():check(item)
    check(manifest)
    reports=[];pairings={}
    for seed in (0,1):
        bank=load_bank(manifest,seed);service=load_p0_service(manifest,seed)
        donor=load_historical_policy(manifest,seed,'V_J');seq=load_historical_policy(manifest,seed,'V_S21')
        for role in ROLES:
            fields={'features'} | ({'income','occupation','references'} if role!='ATTACK_FIT_NEW' else set())
            loaded=load_allowed_role(manifest,seed,role,fields);wire=loaded['wire'];features=loaded['features']
            if role in pairings:
                if not all(np.array_equal(wire[k],pairings[role][k]) for k in ('row_id','unit')):raise AssertionError('seed pairing differs')
            else:pairings[role]=wire
            outputs,scores=service.route_raw(features['X'],features['feature_names'])
            for r in (1,2):
                if not all(np.array_equal(outputs[r-1][j],wire[f'{name}_r{r}']) for j,name in enumerate(('code','prob','decision'))):
                    raise AssertionError('full raw83 P0 source forward differs')
            prepared=donor.space.prepare(wire['code_r1'],wire['code_r2'])
            q=donor.law_prepared(prepared);R=donor_law(manifest,seed,wire['code_r1'],wire['code_r2'])
            original=donor.eta1*R;original[:,0]+=1-donor.eta1
            if not np.array_equal(original,q):raise AssertionError('direct frozen donor reconstruction differs')
            s=seq.law(wire['code_r1'],wire['code_r2'])
            if not np.allclose(s.sum(1),1,atol=1e-12,rtol=0):raise AssertionError('historical sequential mass invalid')
            item={'seed':seed,'role':role,'rows':len(wire['row_id']),'groups':len(np.unique(wire['unit'])),
                  'full_raw83_p0_forward':'EXACT','selected_vj_direct_donor_law':'EXACT','eta':donor.eta1,
                  'historical_s21_mass':'PASS','fit_folds':None,'reference_forward':None}
            if role=='OSF_DEFENSE_FIT':
                f=loaded['fold'];item['fit_folds']={'rows':{str(i):int((f==i).sum()) for i in range(3)},
                    'groups':{str(i):len(np.unique(wire['unit'][f==i])) for i in range(3)}}
            if role!='ATTACK_FIT_NEW':
                refs=loaded['references'];gaps={}
                for r in (1,2):
                    gap=float(np.max(np.abs(scores[r-1]-refs[f'U0_{r}'])))
                    if gap>1e-12:raise AssertionError('U0 frozen teacher forward differs')
                    gaps[str(r)]=gap
                item['reference_forward']={'U0_max_absolute_gap':gaps,'Ucal':'hash-pinned role-only stored frozen reference; no refit'}
            reports.append(item)
    overlap=[]
    keys=list(ROLES)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            rows=len(np.intersect1d(pairings[a]['row_id'],pairings[b]['row_id']))
            groups=len(np.intersect1d(pairings[a]['unit'],pairings[b]['unit']))
            if rows or groups:raise AssertionError('permitted roles overlap')
            overlap.append({'a':a,'b':b,'row_overlap':rows,'group_overlap':groups})
    result={'schema':'CBR1-full-source-forward-admission-v1','status':'PASS','source_manifest_sha256':manifest_sha256,
            'seeds':reports,'role_overlap':overlap,'label_reads':['FIT:income','FIT:occupation','INNER:income','INNER:occupation'],
            'sex_labels_read':False,'barred_archive_read':False,'new_fits':0}
    write_once(output,result,False)
    return {'status':'PASS','output_sha256':sha_file(output),'source_seed_role_forwards':len(reports)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.manifest,a.manifest_sha256,a.output)))

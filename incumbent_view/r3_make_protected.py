"""Create hash-pinned R3 FIT/INNER scoring specs from a completed R2 unit.

This builder reads only public receipts and file hashes, never role arrays.
It runs after a selected R2 policy/release exists. R0 alone enqueues units.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .r3_family import ARMS


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build(arm,seed,r2_unit_id,seal_path,amendment_path=None):
    if arm not in ARMS or arm=='P0' or seed not in (0,1) or not r2_unit_id.startswith(f'R2_{arm}_S{seed}_FIT') or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for c in r2_unit_id):
        raise ValueError('selected R2 unit identity differs')
    root=Path(__file__).resolve().parents[1];owner=root/'results/pcrl_incumbent_view_risk_v1/r3';private=owner/'.private';r1=root/'results/pcrl_incumbent_view_risk_v1/r1';r2=root/'results/pcrl_incumbent_view_risk_v1/r2'
    source=json.loads((r1/'.private/P0_SOURCE_PATHS.json').read_text());bankdoc=source['seeds'][str(seed)];unit_root=r2/'.private/units'/r2_unit_id;policy_dir=unit_root/'policy'
    policy_path=policy_dir/'POLICY_RECEIPT.json';package_path=policy_dir/'policy.npz';policy=json.loads(policy_path.read_text())
    if policy['schema']!='IVR1-selected-policy-v1' or policy['arm_id']!=arm or policy['source_seed']!=seed or policy['policy_package_sha256']!=sha(package_path):raise PermissionError('R2 selected policy source differs')
    currentbank=Path(bankdoc['public_bank']['path'])
    if sha(currentbank)!=bankdoc['public_bank']['sha256'] or policy['public_bank_sha256']!=sha(currentbank):raise PermissionError('original P0 bank source differs')
    if policy['source_lock_sha256']!=sha(seal_path):raise PermissionError('R2 policy selected under different science seal')
    if 'source_binding_sha256' not in policy:raise PermissionError('R2 source binding absent')
    outputs=[]
    code=[root/f'incumbent_view/r3_{name}.py' for name in ('worker','utility','family','make_protected')]+[root/'results/pcrl_incumbent_view_risk_v1/r0/compute_queue.py']
    manifest=r1/'.private/P0_SOURCE_PATHS.json'
    for role,tag in (('OSF_DEFENSE_FIT','FIT'),('INNER_SELECTION','INNER')):
        releases={}
        for d in (0,1,2):
            folder=unit_root/f'{role}_R{d}';recp=folder/'RELEASE_RECEIPT.json';rec=json.loads(recp.read_text())
            if (rec.get('schema')!='IVR1-selected-role-release-v1' or rec.get('arm_id')!=arm or rec.get('source_seed')!=seed or rec.get('role')!=role or rec.get('realization')!=d
                or rec.get('policy_receipt_sha256')!=sha(policy_path) or rec.get('source_binding_sha256')!=policy['source_binding_sha256']
                or rec.get('law_file_sha256')!=sha(folder/'law.npz') or rec.get('codes_file_sha256')!=sha(folder/'release_codes.npz')):
                raise PermissionError('R2 selected release receipt differs')
            releases[d]=(folder,recp,rec)
        law_hashes={r['law_content_sha256'] for _,_,r in releases.values()}
        if len(law_hashes)!=1:raise PermissionError('R2 law changes between virtual deployments')
        uid=f'R3_{arm}_S{seed}_{tag}_PROTECTED_V1'
        labels=source['roles'][role];p0=bankdoc['roles'][role]
        pinpaths={'seal':Path(seal_path),'role_manifest':manifest,'public_bank':currentbank,'p0_admission_receipt':r1/f'R1_P0_S{seed}_ADMISSION_RECEIPT.json',
                  'p0_wire':Path(p0['path']),'income':Path(labels['income']['path']),'occupation':Path(labels['occupation']['path']),
                  'references':Path(labels[f'references_s{seed}']['path']),'policy_receipt':policy_path,'policy_package':package_path,
                  'release_receipt':releases[0][1],'law':releases[0][0]/'law.npz'}
        for d in (0,1,2):pinpaths[f'draw{d}']=releases[d][0]/'release_codes.npz';pinpaths[f'draw{d}_receipt']=releases[d][1]
        pins={k:{'path':str(p),'sha256':sha(p)} for k,p in pinpaths.items()}
        for field in ('income','occupation',f'references_s{seed}'):
            key='references' if field.startswith('references') else field
            if pins[key]['sha256']!=labels[field]['sha256']:raise PermissionError('role task archive differs from R1 manifest')
        if pins['p0_wire']['sha256']!=p0['sha256']:raise PermissionError('P0 role wire differs from R1 manifest')
        identity=json.loads((r1/f'R1_P0_S{seed}_ADMISSION_RECEIPT.json').read_text())['roles'][role]
        paths={'csv':owner/f'{uid}.csv','private_vectors':private/f'{uid}_VECTORS.npz','receipt':owner/f'{uid}_RECEIPT.json'}
        cfg={'schema':'pcrl-incumbent-r3-utility-config-v1','unit_id':uid,'stage':'protection','role':role,'arm_id':arm,'source_seed':seed,
             'rows':identity['rows'],'groups':identity['groups'],'row_unit_pair_sha256':identity['row_unit_pair_sha256'],
             'source_unit_id':r2_unit_id,'pins':pins,'outputs':{k:str(v) for k,v in paths.items()}}
        cfgp=private/f'{uid}_CONFIG.json'
        with os.fdopen(os.open(cfgp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(cfg,f,sort_keys=True,indent=2);f.write('\n')
        sources=[{'path':str(p),'sha256':sha(p)} for p in dict.fromkeys(list(pinpaths.values())+code+[cfgp])]
        spec={'unit_id':uid,'stage':'protection','argv':[str(root.parents[1]/'.venv/bin/python'),'-m','incumbent_view.r3_worker','--config',str(cfgp)],'cwd':str(root),
              'max_wall_s':300 if tag=='FIT' else 180,'cpu_reserve_s':300 if tag=='FIT' else 180,'max_rss_gib':2.5,'projected_output_bytes':10000000,
              'deps':[r2_unit_id], 'outputs':[str(v) for v in paths.values()],'sources':sources,'env':{'PYTHONPATH':str(root)}}
        if amendment_path is not None:spec['amendment_sha256']=sha(amendment_path);spec['sources'].append({'path':str(amendment_path),'sha256':sha(amendment_path)})
        specp=private/f'{uid}_QUEUE_SPEC.json'
        with os.fdopen(os.open(specp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(spec,f,sort_keys=True,indent=2);f.write('\n')
        outputs.append({'unit_id':uid,'config_sha256':sha(cfgp),'spec_sha256':sha(specp),'private_config_label':f'r3/.private/{cfgp.name}','private_spec_label':f'r3/.private/{specp.name}','source_count':len(sources),'selected_r2_unit_id':r2_unit_id,'selected_policy_receipt_sha256':sha(policy_path),'law_content_sha256':next(iter(law_hashes))})
    index=owner/f'R3_{arm}_S{seed}_PROTECTED_UNIT_INDEX_V1.json'
    with index.open('x') as f:json.dump({'schema':'pcrl-incumbent-r3-protected-unit-index-v1','units':outputs,'role_arrays_deserialized':False},f,sort_keys=True,indent=2);f.write('\n')
    return {'index_sha256':sha(index),'units':outputs}


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--r2-unit-id',required=True);p.add_argument('--seal',required=True);p.add_argument('--amendment');a=p.parse_args()
    print(json.dumps(build(a.arm,a.seed,a.r2_unit_id,Path(a.seal),Path(a.amendment) if a.amendment else None),sort_keys=True))

if __name__=='__main__':main()

"""Create hash-pinned P0 witness configs, never deserialize role arrays."""
import hashlib
import json
import os
from pathlib import Path


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build():
    root=Path(__file__).resolve().parents[1];owner=root/'results/pcrl_incumbent_view_risk_v1/r3';private=owner/'.private';private.mkdir(exist_ok=True);private.chmod(0o700)
    prior=root.parent/'pcrl-utility-reference-repair-v1/results/pcrl_utility_reference_repair_v1'
    manifest_path=prior/'m1/DATA_ACCESS_MANIFEST.json';manifest=json.loads(manifest_path.read_text())
    oldroot=root.parent/'pcrl-bank-anchored-release-v1/results/pcrl_bank_anchored_release_v1'
    r1=root/'results/pcrl_incumbent_view_risk_v1/r1'
    seal=root/'results/pcrl_incumbent_view_risk_v1/r0/ADMISSION_LOCK.json'
    code=[root/'incumbent_view/r3_worker.py',root/'incumbent_view/r3_utility.py',root/'incumbent_view/r3_family.py',root/'results/pcrl_incumbent_view_risk_v1/r0/compute_queue.py',Path(__file__).resolve()]
    units=[]
    for seed in (0,1):
        r1cfg=json.loads((r1/f'.private/R1_P0_S{seed}_ADMISSION_CONFIG.json').read_text())
        oldcfg=json.loads((oldroot/f'r1/.private/R1_BANK_S{seed}_FIT_SAMPLE32_CONFIG.json').read_text())
        r1rec=json.loads((r1/f'R1_P0_S{seed}_ADMISSION_RECEIPT.json').read_text())
        for role,tag in (('OSF_DEFENSE_FIT','FIT'),('INNER_SELECTION','INNER')):
            uid=f'R3_P0_S{seed}_{tag}_WITNESS_V1'
            files={'seal':seal,'public_bank':r1/f'P0_PUBLIC_BANK_S{seed}.json','p0_admission_receipt':r1/f'R1_P0_S{seed}_ADMISSION_RECEIPT.json',
                   'p0_wire':r1/f'.private/R1_P0_S{seed}_ADMISSION/{role}/P0_RELEASE.npz','role_manifest':manifest_path}
            for name in ('income','occupation','references'):
                field=f'references_s{seed}' if name=='references' else name;key=f'{role}:{field}'
                path=prior/manifest['role_file_relative_path'][key]
                if sha(path)!=manifest['role_file_sha256'][key]:raise ValueError('source role file hash differs')
                files[name]=path
            if tag=='FIT':
                files['old_b0_wire']=Path(oldcfg['pins']['saved_b0']);files['old_b0_table']=Path(oldcfg['pins']['public_table'])
            pins={k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()}
            outputs={'csv':str(owner/f'{uid}.csv'),'private_vectors':str(private/f'{uid}_VECTORS.npz'),'receipt':str(owner/f'{uid}_RECEIPT.json')}
            identity=r1rec['roles'][role]
            cfg={'schema':'pcrl-incumbent-r3-utility-config-v1','unit_id':uid,'stage':'witness','role':role,'arm_id':'P0','source_seed':seed,
                 'rows':identity['rows'],'groups':identity['groups'],'row_unit_pair_sha256':identity['row_unit_pair_sha256'],
                 'source_unit_id':f'R1_P0_S{seed}_ADMISSION','pins':pins,'outputs':outputs}
            cfgpath=private/f'{uid}_CONFIG.json'
            with os.fdopen(os.open(cfgpath,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(cfg,f,sort_keys=True,indent=2);f.write('\n')
            sources=[{'path':str(v),'sha256':sha(v)} for v in list(files.values())+code+[cfgpath]]
            spec={'unit_id':uid,'stage':'witness','argv':[r1cfg['python'],'-m','incumbent_view.r3_worker','--config',str(cfgpath)],'cwd':str(root),
                  'max_wall_s':180,'cpu_reserve_s':180,'max_rss_gib':2,'projected_output_bytes':4000000,'deps':[f'R1_P0_S{seed}_ADMISSION'],
                  'outputs':list(outputs.values()),'sources':sources,'env':{'PYTHONPATH':str(root)}}
            specpath=private/f'{uid}_QUEUE_SPEC.json'
            with os.fdopen(os.open(specpath,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump(spec,f,sort_keys=True,indent=2);f.write('\n')
            units.append({'unit_id':uid,'config_sha256':sha(cfgpath),'queue_spec_sha256':sha(specpath),'private_config_label':f'r3/.private/{cfgpath.name}','private_queue_spec_label':f'r3/.private/{specpath.name}','source_count':len(sources),'code_sha256':{str(p.relative_to(root)):sha(p) for p in code}})
    index={'schema':'pcrl-incumbent-r3-witness-unit-index-v1','units':units,'role_arrays_deserialized':False}
    out=owner/'WITNESS_UNIT_INDEX_V1.json'
    with out.open('x') as f:json.dump(index,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'units':len(units),'index_sha256':sha(out)}))

if __name__=='__main__':build()

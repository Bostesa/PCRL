"""Queue-only FIT raw83 deployment/cache parity, no label arrays opened."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from .source import load_manifest,load_allowed_role,load_historical_policy,sha_file,write_once
from .deployment import (load_p0_service,CachedRelease,identity_law,historical_law,sample_tuple,validate_law,load_control_package)
from .scheduler import assert_active_from_env


def load_model(item,manifest):
    kind=item['kind'];seed=item['seed']
    if kind=='P0':return identity_law,'deterministic',manifest['seeds'][str(seed)]['public_bank']['sha256']
    if kind in ('V_J','V_S21'):
        policy=load_historical_policy(manifest,seed,kind)
        return historical_law(policy),'joint' if kind=='V_J' else 'seq21',manifest['seeds'][str(seed)]['policies'][kind]['receipt']['sha256']
    if kind=='ALLOCATION':
        from .fit import load_allocation_policy
        policy=load_allocation_policy(item['policy_dir'],manifest,expected_receipt_sha256=item['receipt_sha256']);sampler='joint'
    elif kind=='TEMPLATE':
        from .secondary import load_template_policy
        policy=load_template_policy(item['policy_dir'],manifest,expected_receipt_sha256=item['receipt_sha256']);sampler='joint'
    elif kind=='CONTROL':
        policy=load_control_package(item['policy_dir'],manifest,item['receipt_sha256'])
        sampler={'LOCAL':'product','PRODUCT':'product','DETERMINISTIC':'deterministic','SEQUENTIAL':'seq12' if policy.first_recipient==1 else 'seq21'}[policy.kind]
    else:raise PermissionError('unknown deployed model kind')
    receipt=json.loads((Path(item['policy_dir'])/'POLICY_RECEIPT.json').read_text())
    if receipt['seed']!=seed:raise PermissionError('deployed source seed differs')
    return policy.law_codes,sampler,item['receipt_sha256']


def run(config):
    active=assert_active_from_env(stage='verification')
    cfg=json.loads(Path(config).read_text())
    if cfg['unit_id']!=active['unit_id']:raise PermissionError('wrong deployment unit')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    if pins.get(str(Path(config).absolute()))!=sha_file(config):raise PermissionError('deployment config unqueued')
    manifest=load_manifest(cfg['source_manifest'],cfg['source_manifest_sha256'])
    def check(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:
                if pins.get(x['path'])!=x['sha256']:raise PermissionError('deployed source unqueued')
            else:
                for z in x.values():check(z)
    check(manifest)
    private=Path(cfg['private_output'])
    if private.exists():raise FileExistsError('new deployment output version required')
    private.mkdir(parents=True,mode=0o700)
    secret=os.urandom(32);root=private/'sampling_root.bin';root.write_bytes(secret);root.chmod(0o600)
    services={};inputs={};results=[]
    for item in cfg['models']:
        seed=item['seed'];arm=item['arm_id']
        if seed not in services:
            services[seed]=load_p0_service(manifest,seed)
            fit=load_allowed_role(manifest,seed,'OSF_DEFENSE_FIT',{'features'})
            # Include a forced duplicate of an actual allowed fitting person.
            ix=np.r_[np.arange(32),0,0];x=fit['features']['X'][ix]
            keys=fit['wire']['unit'][ix].tolist();names=fit['features']['feature_names']
            with os.fdopen(os.open(private/f'raw_s{seed}.npz',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
                np.savez_compressed(f,X=x,feature_names=names)
            write_once(private/f'keys_s{seed}.json',keys)
            inputs[seed]=(x,keys,names)
        service=services[seed];x,keys,names=inputs[seed]
        law,sampling,binding=load_model(item,manifest)
        cache=private/f'{arm}_s{seed}.sqlite'
        api=CachedRelease(service,law,policy_sha256=binding,cache_path=cache,secret=secret,sampling_kind=sampling)
        wires=api.release(x,names,keys)
        again=api.release(x,names,keys)
        if wires!=again:raise AssertionError('same cached query changes public wire')
        api.close()
        api=CachedRelease(service,law,policy_sha256=binding,cache_path=cache,secret=secret,sampling_kind=sampling)
        if api.release(x,names,keys)!=wires:raise AssertionError('reopened cache changes public wire')
        outputs,_=service.route_raw(x,names);c1,c2=outputs[0][0],outputs[1][0]
        a,b,q=validate_law(service.bank,c1,c2,*law(c1,c2))
        direct=[sample_tuple(a[i],b[i],q[i],api.uniform(key),api.uniform(key,1),api.uniform(key,2),sampling) for i,key in enumerate(keys)]
        for recipient,view in enumerate(wires,1):
            if not all(w['code_id']==direct[i][recipient-1] for i,w in enumerate(view)):raise AssertionError('wire and actual finite sampler disagree')
            if not(view[0]==view[32]==view[33]):raise AssertionError('duplicate person not cached identically')
            for w in view:
                if any(k in w or k in w['public_metadata'] for k in ('person_key','unit','row_id','context','coefficient','fallback','uniform','source_code','fine_action','sampling_root')):
                    raise AssertionError('private component transmitted')
            out=private/f'{arm}_s{seed}_r{recipient}.jsonl'
            with os.fdopen(os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:
                for wire in view:f.write(json.dumps(wire,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        refusal={}
        for name,raw,n in (('reordered',x,names[::-1]),('extra_column',np.c_[x,np.zeros(len(x))],names),('nonfinite',np.full(x.shape,np.nan),names)):
            try:api.release(raw,n,keys)
            except ValueError:refusal[name]=True
            else:raise AssertionError('raw input refusal missing')
        different=x[:1].copy();different[0,0]+=1
        try:api.release(different,names,keys[:1])
        except ValueError:refusal['same_person_changed_raw']=True
        else:raise AssertionError('same key altered input not refused')
        api.close()
        cli1=private/f'{arm}_s{seed}_cli_r1.jsonl';cli2=private/f'{arm}_s{seed}_cli_r2.jsonl'
        argv=[sys.executable,'-m','conditional_release.deployment','--source-manifest',cfg['source_manifest'],
              '--source-manifest-sha256',cfg['source_manifest_sha256'],'--seed',str(seed),'--kind',item['kind'],
              '--input',str(private/f'raw_s{seed}.npz'),'--input-sha256',sha_file(private/f'raw_s{seed}.npz'),
              '--person-keys',str(private/f'keys_s{seed}.json'),'--person-keys-sha256',sha_file(private/f'keys_s{seed}.json'),
              '--sampling-root',str(root),'--cache',str(cache),'--output-r1',str(cli1),'--output-r2',str(cli2)]
        if item['kind'] in ('ALLOCATION','TEMPLATE','CONTROL'):
            argv+=['--policy-dir',item['policy_dir'],'--policy-receipt-sha256',item['receipt_sha256']]
        child=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=90)
        if any(sha_file(path)!=sha_file(private/f'{arm}_s{seed}_r{r}.jsonl') for r,path in enumerate((cli1,cli2),1)):
            raise AssertionError('CLI and API public wires differ')
        write_once(private/f'{arm}_s{seed}_EXECUTED_COMMAND.json',{'argv':argv,'stdout':child.stdout,'stderr':child.stderr})
        results.append({'seed':seed,'arm_id':arm,'kind':item['kind'],'sampling_kind':sampling,'rows':len(x),
          'law_mass_class_support':'PASS','cached_repeat':'EXACT','reopened_cache':'EXACT','duplicate_wire':'EXACT',
          'sampler_wire_parity':'EXACT','cli_api_byte_parity':'EXACT','private_fields_excluded':'PASS','refusal':refusal,
          'wire_sha256':[sha_file(private/f'{arm}_s{seed}_r{r}.jsonl') for r in (1,2)]})
    result={'schema':'CBR1-raw83-cache-deployment-replay-v1','unit_id':cfg['unit_id'],'status':'PASS','model_replays':results,
            'scientific_labels_read':False,'independent_replay':False,'source_manifest_sha256':cfg['source_manifest_sha256'],
            'private_sampling_root_sha256':hashlib.sha256(secret).hexdigest()}
    write_once(cfg['public_output'],result,False)
    print(json.dumps({'status':'PASS','models':len(results),'rows_each':34}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)

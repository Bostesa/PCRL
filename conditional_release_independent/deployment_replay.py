"""Independent raw83/law/HMAC/CDF and complete emitted-wire replay."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .reconstruction import digest,admitted_arrays
from .raw_forward import raw_p0
from .policy_replay import source_numbers
from .policy_laws import checked_law
from .sampling import uniform,sampled_pair


def expected_wire(bank,r,code):
    b=bank['recipients'][str(r)];p=np.asarray(b['probability'][code],float)
    return {'schema':'SCV1-reference-score-release-v1','recipient':r,'probability':p.tolist(),'decision':int(np.argmax(p)),'release_id':bank['release_id'],'code_id':int(code),'public_metadata':{'model_id':bank['release_id'],'source_binding_sha256':bank['source_binding_sha256'],'bank_sha256':b['bank_sha256'],'code_length_bits':(len(b['probability'])-1).bit_length()}}


def run(config):
    from conditional_release.scheduler import assert_active_from_env
    active=assert_active_from_env(stage='verification');state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text());pins={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    if pins.get(str(Path(config).absolute()))!=digest(config):raise PermissionError('unqueued deployment config')
    cfg=json.loads(Path(config).read_text());restore=Path(cfg['restore_root']).resolve() if cfg.get('restore_root') else None
    def check(e):
        if restore and not Path(e['path']).resolve().is_relative_to(restore):raise PermissionError('restore replay fell through to original')
        if pins.get(e['path'])!=e['sha256'] or digest(e['path'])!=e['sha256']:raise PermissionError('unqueued deployment input')
    check(cfg['manifest']);source=json.loads(Path(cfg['manifest']['path']).read_text())
    def validate_source(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:check(x)
            else:
                for v in x.values():validate_source(v)
    validate_source(source);results=[]
    for item in cfg['models']:
        seed=item['seed'];check(item['raw']);check(item['keys']);check(item['secret']);raw=admitted_arrays(item['raw'],{'X','feature_names'});keys=json.loads(Path(item['keys']['path']).read_text());secret=Path(item['secret']['path']).read_bytes()
        if len(secret)!=32 or len(keys)!=len(raw['X']):raise ValueError('private cache sample alignment')
        entry=source['seeds'][str(seed)]['public_bank'];bank=json.loads(Path(entry['path']).read_text());p0,_=raw_p0(source['seeds'][str(seed)]['pins'],bank,raw['X'],raw['feature_names']);c1,c2=p0[0]['codes'],p0[1]['codes']
        numbers,historical=source_numbers(source,seed,'V_S21' if item['kind']=='V_S21' else 'V_J');receipt=historical;package=None
        if item.get('receipt'):
            check(item['receipt']);check(item['package']);receipt=json.loads(Path(item['receipt']['path']).read_text())
            with ZipFile(item['package']['path']) as z:names={n[:-4] for n in z.namelist()}
            package=admitted_arrays(item['package'],names);binding=item['receipt']['sha256']
        elif item['kind']=='P0':binding=entry['sha256']
        else:binding=source['seeds'][str(seed)]['policies'][item['kind']]['receipt']['sha256']
        a,b,q=checked_law(numbers,c1,c2,item['kind'],receipt,package);kind=item['sampling_kind'];cache={};draws=[]
        for i,key in enumerate(keys):
            rowbind=digest_input(raw['X'][i]);old=cache.get(str(key))
            if old:
                if old[0]!=rowbind:raise ValueError('same person changed raw input in fixture')
                draw=old[1]
            else:
                coins=[uniform(secret,kind,binding,bank['source_binding_sha256'],seed,key,r) for r in (None,1,2)];draw=sampled_pair(a[i],b[i],q[i],kind,*coins);cache[str(key)]=(rowbind,draw)
            draws.append(draw)
        for r in (1,2):
            expected=[expected_wire(bank,r,z[r-1]) for z in draws]
            for label in ('wire','cli'):
                entry=item[f'{label}_r{r}'];check(entry);actual=[json.loads(line) for line in Path(entry['path']).read_text().splitlines()]
                if actual!=expected:raise ValueError('independent raw-to-wire sampling replay differs')
                canonical=''.join(json.dumps(w,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for w in expected)
                if Path(entry['path']).read_bytes()!=canonical.encode():raise ValueError('wire canonical byte replay differs')
        results.append({'arm_id':item['arm_id'],'seed':seed,'kind':item['kind'],'sampling_kind':kind,'raw_rows':len(draws),'unique_cached_people':len(cache),'raw83_routing':'EXACT','serialized_law_class_support':'PASS','HMAC_CDF_wire':'EXACT','complete_metadata_and_CLI_bytes':'EXACT','snapshot_only':restore is not None})
    report={'schema':'CBR1-independent-deployment-replay-v1','status':'PASS','unit_id':active['unit_id'],'models':results,'snapshot_only':restore is not None,'scientific_labels_read':False,'scope':'Independent raw83 frozen encoder/head/router, serialized release law, policy-bound HMAC domains, recipient/joint coins, cached actual code tuple and every public wire/CLI byte. Cache refusal and transactional behavior are separately tested by actual deployment engineering. Ideal continuous-law endpoints and 53-bit cached PRF deployment are distinct contracts.'}
    with Path(cfg['output']).open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','models':len(results),'snapshot_only':restore is not None}))


def digest_input(row):
    import hashlib
    return hashlib.sha256(np.asarray(row,np.float32).tobytes()).digest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)

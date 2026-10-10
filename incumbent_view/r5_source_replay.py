"""Independent P0 wire/support and scalar cost replay; queue admission only."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, fields):
    with ZipFile(path) as archive:
        if len(archive.infolist()) != len(fields) or {x.filename for x in archive.infolist()} != {x+'.npy' for x in fields}:
            raise PermissionError('unexpected archive fields before deserialization')
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in fields}


def wire_hash(bank, j, release):
    info = bank['recipients'][str(j)]
    sha = hashlib.sha256()
    for code, p, decision in zip(release[f'code_r{j}'],release[f'prob_r{j}'],release[f'decision_r{j}']):
        wire = {'schema':'SCV1-reference-score-release-v1','recipient':j,
                'probability':p.tolist(),'decision':int(decision),'release_id':bank['release_id'],
                'code_id':int(code),'public_metadata':{'model_id':bank['release_id'],
                'source_binding_sha256':bank['source_binding_sha256'],'bank_sha256':info['bank_sha256'],
                'code_length_bits':(len(info['probability'])-1).bit_length()}}
        sha.update(json.dumps(wire,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
    return sha.hexdigest()


def scalar_cost(p,y,metric,decimal=False):
    if metric=='accuracy': return Decimal(int(max(range(len(p)),key=lambda k:p[k])==y)) if decimal else float(max(range(len(p)),key=lambda k:p[k])==y)
    if decimal:
        p=[Decimal.from_float(float(x)) for x in p]
        return -max(p[y],Decimal.from_float(1e-12)).ln() if metric=='nll' else sum(((x-Decimal(int(k==y)))**2 for k,x in enumerate(p)),Decimal(0))
    return -math.log(max(float(p[y]),1e-12)) if metric=='nll' else math.fsum((float(x)-int(k==y))**2 for k,x in enumerate(p))


def run(config_path):
    active=assert_active_from_env(stage='admission')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(path):
        if registered.get(str(path))!=digest(path): raise PermissionError('unregistered source')
        return Path(path)
    cfg=json.loads(pinned(config_path).read_text())
    if cfg['unit_id']!=active['unit_id']: raise PermissionError('wrong replay unit')
    bank=json.loads(pinned(cfg['bank']).read_text())
    source=json.loads(pinned(cfg['source_config']).read_text())
    report={'schema':'pcrl-incumbent-independent-source-cost-replay-v1','unit_id':active['unit_id'],
            'seed':cfg['seed'],'status':'PASS','label_fields_read':['FIT:income','FIT:occupation','INNER:income','INNER:occupation'],
            'wire_checks':{},'utility':[],'precision':{},'source_lock_sha256':active['seal_sha256']}
    wire_fields={'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'}
    for role,item in cfg['roles'].items():
        r=load(pinned(item['release']),wire_fields)
        rec=json.loads(pinned(cfg['source_receipt']).read_text())['roles'][role]
        for j in (1,2):
            table=np.asarray(bank['recipients'][str(j)]['probability'],np.float64)
            assert np.array_equal(table[r[f'code_r{j}']],r[f'prob_r{j}'])
            assert np.array_equal(r[f'prob_r{j}'].argmax(1),r[f'decision_r{j}'])
            assert wire_hash(bank,j,r)==rec['wire_sha256_by_recipient'][f'r{j}']
        if role!='OSF_DEFENSE_FIT':
            historical=load(pinned(source['roles'][role]['historical']['path']),wire_fields)
            assert all(np.array_equal(r[k],historical[k]) for k in wire_fields)
        report['wire_checks'][role]={'rows':len(r['row_id']),'groups':len(np.unique(r['unit'])),'all_fields':'PASS','wire_hashes':'PASS'}
        if role=='ATTACK_FIT_NEW': continue
        refs=load(pinned(item['references']),{'row_id','unit','U0_1','U0_2','Ucal_1','Ucal_2','d_1','d_2'})
        baseline=load(pinned(cfg['old_b0']),{'row_id','unit','code0','code1','prob0','prob1','decision0','decision1'}) if role=='OSF_DEFENSE_FIT' else None
        for j,task in enumerate(('income','occupation'),1):
            labels=load(pinned(item[task]),{'row_id','unit',task})
            assert all(np.array_equal(r[k],labels[k]) and np.array_equal(r[k],refs[k]) for k in ('row_id','unit'))
            y=labels[task]
            if baseline is not None: assert all(np.array_equal(r[k],baseline[k]) for k in ('row_id','unit'))
            choices={'OLD_B0':baseline[f'prob{j-1}']} if baseline is not None else {name:refs[f'{name}_{j}'] for name in ('U0','Ucal')}
            with localcontext() as ctx:
                ctx.prec=80
                for metric in ('accuracy','nll','brier'):
                    cache={}
                    def value(p,y,high):
                        key=(tuple(p),int(y),high)
                        if key not in cache: cache[key]=scalar_cost(p,int(y),metric,high)
                        return cache[key]
                    primary=[value(p,z,False) for p,z in zip(r[f'prob_r{j}'],y)]
                    precise=[value(p,z,True) for p,z in zip(r[f'prob_r{j}'],y)]
                    for ref,p in choices.items():
                        other=[value(v,z,False) for v,z in zip(p,y)]
                        high=[value(v,z,True) for v,z in zip(p,y)]
                        margin=math.fsum(a-b for a,b in zip(primary,other))/len(y)
                        exact=sum((a-b for a,b in zip(precise,high)),Decimal(0))/Decimal(len(y))
                        error=float(abs(Decimal.from_float(margin)-exact))
                        cap=({'accuracy':-.002,'nll':.001,'brier':.0005} if baseline is not None else {'accuracy':-.010,'nll':.010,'brier':.005})[metric]
                        passed=margin>=cap if metric=='accuracy' else margin<=cap
                        report['utility'].append({'role':role,'task':task,'reference':ref,'metric':metric,'margin':margin,'decimal80_margin':str(exact),'arithmetic_error':error,'cap':cap,'strict_point_status':'PASS' if passed else 'FAIL'})
                    if role=='OSF_DEFENSE_FIT':
                        table=bank['recipients'][str(j)]['probability']; maxerror=0.
                        # Fixed uniform supported action law is an arithmetic check, never a fitted candidate.
                        for k,p0 in enumerate(table):
                            support=bank['recipients'][str(j)]['support'][k]
                            for z in range(len(p0)):
                                w=1./len(support)
                                got=math.fsum(w*value(table[c],z,False) for c in support)-value(p0,z,False)
                                exact=sum((Decimal.from_float(w)*value(table[c],z,True) for c in support),Decimal(0))-value(p0,z,True)
                                maxerror=max(maxerror,float(abs(Decimal.from_float(got)-exact)))
                        assert maxerror<1e-10
                        report['precision'][f'{task}:{metric}']={'max_all_state_label_uniform_action_margin_error':maxerror,'envelope':1e-10,'status':'PASS'}
    report['inner_24_point_gate']=all(x['strict_point_status']=='PASS' for x in report['utility'] if x['role']=='INNER_SELECTION')
    report['scope']='Exact P0 identity and allowed source utility; frozen uniform action arithmetic only. No policy or attacker fit; no SEX/HEAD/assessment labels.'
    out=Path(cfg['output'])
    with os.fdopen(os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),'w') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();run(args.config)

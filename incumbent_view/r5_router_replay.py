"""Independent original-ID/support/router diagnostics and black-box raw CLI."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile
import numpy as np
from qpc import deploy as QD
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_source_replay import digest,load,wire_hash


def divergence(a,b):
    a=[max(float(x),1e-12) for x in a];b=[max(float(x),1e-12) for x in b]
    sa,sb=math.fsum(a),math.fsum(b);a=[x/sa for x in a];b=[x/sb for x in b]
    return .5*(math.fsum(x*math.log(x/y) for x,y in zip(a,b))+math.fsum(y*math.log(y/x) for x,y in zip(a,b)))


def collide(scores,codes):
    seen={}
    for x,c in zip(scores,codes):seen.setdefault(np.ascontiguousarray(x).tobytes(),set()).add(tuple(np.atleast_1d(c).tolist()))
    bad=[v for v in seen.values() if len(v)>1]
    return {'rows':len(scores),'unique_inputs':len(seen),'repeated_rows':len(scores)-len(seen),
            'conflicting_inputs':len(bad),'required_output_states_in_conflicts':sum(map(len,bad))}


def run(config_path):
    active=assert_active_from_env(stage='admission')
    state=json.loads((Path(os.environ['PCRL_QUEUE_ROOT'])/'state.json').read_text())
    registered={x['path']:x['sha256'] for x in state['units'][active['unit_id']]['spec']['sources']}
    def pinned(p):
        if registered.get(str(p))!=digest(p):raise PermissionError('router source unpinned')
        return Path(p)
    cfg=json.loads(pinned(config_path).read_text());src=json.loads(pinned(cfg['source_config']).read_text())
    bank=json.loads(pinned(cfg['bank']).read_text());admission=json.loads(pinned(cfg['source_receipt']).read_text())
    assert cfg['unit_id']==active['unit_id'] and bank['seed']==cfg['seed']==src['seed']
    for item in src['pins'].values():pinned(item['path'])
    decoder=json.loads(pinned(src['pins']['decoder']['path']).read_text())
    checks={};old=Path(src['old_package']['path']);pinned(old)
    with ZipFile(old) as archive:
        names={x.filename for x in archive.infolist()}
        assert {'bank1_probability.npy','bank2_probability.npy'}.issubset(names)
        assert not any(x.startswith(('sex','label','income','occupation')) for x in names)
    with np.load(old,allow_pickle=False) as a:oldbanks=[a[f'bank{j}_probability'].copy() for j in (1,2)]
    for j in (1,2):
        table=np.asarray(bank['recipients'][str(j)]['probability'])
        assert np.array_equal(table,np.asarray(decoder[f'r{j}']['q']))
        assert bank['recipients'][str(j)]['global_ids']==list(range(len(table)))
        expected=[]
        for k,p in enumerate(table):
            d=int(p.argmax());others=[i for i,row in enumerate(table) if i!=k and row.argmax()==d]
            others.sort(key=lambda i:(divergence(p,table[i]),i));expected.append([k,*others[:3]])
        assert expected==bank['recipients'][str(j)]['support']
        missing=sum(not any(np.array_equal(p,q) for q in oldbanks[j-1]) for p in table)
        assert missing==admission['reachability']['old_global_bank_missing_states'][j-1]
        checks[f'r{j}']={'table_source':'EXACT_HISTORICAL_DECODER','original_ids':'PASS','same_class_fixed_support':'PASS','old_probability_states_missing':missing,'states':len(table)}
    # Only fixed source prediction functions are used; no candidate endpoints.
    model,heads,_=QD.load_teacher(str(Path(src['pins']['model']['path']).parent),cfg['seed'])
    diagnostics={}
    fields={'row_id','unit','code_r1','code_r2','prob_r1','prob_r2','decision_r1','decision_r2'}
    for role,info in cfg['roles'].items():
        features=load(pinned(src['roles'][role]['features']['path']),{'row_id','unit','X','feature_names'})
        release=load(pinned(info['release']),fields)
        assert np.array_equal(features['row_id'],release['row_id']) and np.array_equal(features['unit'],release['unit'])
        score=QD.teacher_probs(model,heads,features['X'].astype(np.float32))
        pair=np.c_[release['code_r1'],release['code_r2']]
        c=collide(np.c_[score[0],score[1]],pair)
        local=[collide(score[i],release[f'code_r{i+1}']) for i in (0,1)]
        expected=admission['reachability'][role]
        assert c==expected['central_old_score_inputs'] and local==expected['local_old_score_inputs']
        # Every decoder state absent the global bank implies absent each old action support.
        assert all(checks[f'r{j}']['old_probability_states_missing']==checks[f'r{j}']['states'] for j in (1,2))
        assert expected['old_per_row_support_missing']==[len(pair)]*2
        diagnostics[role]={'central':c,'local':local,'old_per_row_support_missing':[len(pair)]*2}
    private=Path(cfg['private_output']);private.mkdir(mode=0o700)
    fit=Path(src['private_output_dir'])/'OSF_DEFENSE_FIT'
    sample=pinned(str(fit/'sample_input.npz'))
    oldsample=load(pinned(str(fit/'historical_hcal_sample.npz')),{'tokens_1','probs_1','decision_1','tokens_2','probs_2','decision_2'})
    cli_checks=[]
    for j in (1,2):
        output=private/f'recipient{j}.jsonl'
        argv=[cfg['python'],'-m','incumbent_view.r1_deploy','--source-manifest',cfg['source_paths'],'--source-manifest-sha256',digest(pinned(cfg['source_paths'])),'--seed',str(cfg['seed']),'--recipient',str(j),'--input',str(sample),'--output',str(output)]
        done=subprocess.run(argv,cwd=cfg['cwd'],capture_output=True,text=True,timeout=120)
        if done.returncode:
            with os.fdopen(os.open(private/f'failure{j}.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as f:json.dump({'returncode':done.returncode,'stderr':done.stderr},f)
            raise ValueError('raw CLI failed; private stderr preserved')
        rows=[json.loads(s) for s in output.read_text().splitlines()]
        assert len(rows)==32
        assert [x['code_id'] for x in rows]==oldsample[f'tokens_{j}'].tolist()
        assert np.array_equal(np.asarray([x['probability'] for x in rows]),oldsample[f'probs_{j}'])
        assert [x['decision'] for x in rows]==oldsample[f'decision_{j}'].tolist()
        assert digest(output)==digest(pinned(str(fit/f'recipient{j}.jsonl')))
        cli_checks.append({'recipient':j,'rows':len(rows),'actual_cli_and_historical_raw_wire':'PASS','wire_sha256':digest(output)})
    report={'schema':'pcrl-incumbent-independent-router-replay-v1','status':'PASS','unit_id':active['unit_id'],'seed':cfg['seed'],'support_checks':checks,'old_input_diagnostics':diagnostics,'raw_cli_checks':cli_checks,
            'scope':'Allowed role features only; no labels. Observed zero input conflicts is not proof of a population routing function. Exact probability absence proves old finite bank cannot emit exact P0 probabilities; not a global objective-cause claim.'}
    with os.fdopen(os.open(cfg['output'],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),'w') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)

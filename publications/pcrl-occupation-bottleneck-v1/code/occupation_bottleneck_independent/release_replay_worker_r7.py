"""Independent all72 actual-checkpoint release replay using feature roles only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

COUNTS={'OSF_DEFENSE_FIT':(15434,15428),'ATTACK_FIT_NEW':(4064,4061),'INNER_SELECTION':(2235,2234)}
RANK_RULE={'relative_rank_cutoff':1e-12,'epsilon_multiplier':32.,'near_tie_absolute_logit_gap':1e-10}


def features(inputs,manifest,independent):
    import numpy as np
    if manifest['schema']!='OB1-role-safe-private-source-v1' or set(manifest['roles'])!=set(COUNTS):raise ValueError('role-safe source manifest differs')
    roles={};names=None
    forbidden={'sex','race','income','occupation','fnlwgt','row_id','unit','id','person_id','sampling_id','group_id','hidden_flag','y','label','target','teacher','teacher_score'}
    for role,(rows,groups) in COUNTS.items():
        data=inputs.arrays(manifest['roles'][role]['features'],{'row_id','unit','X','feature_names'});current=data['feature_names'].tolist()
        if len(current)!=83 or len(set(current))!=83 or any(not isinstance(n,str) or not n or n.lower().split('=')[0] in forbidden for n in current):raise ValueError('raw83 feature names differ')
        if names is None:names=current
        if current!=names or data['X'].shape!=(rows,83) or data['X'].dtype!=np.float32 or not np.isfinite(data['X']).all():raise ValueError('actual raw83 precision/schema differs')
        if data['row_id'].shape!=(rows,) or data['unit'].shape!=(rows,) or data['row_id'].dtype.kind not in 'iu' or data['unit'].dtype.kind not in 'iu' or len(np.unique(data['row_id']))!=rows or len(np.unique(data['unit']))!=groups:raise ValueError('actual role row/group counts differ')
        roles[role]=data
    for a in roles:
        for b in roles:
            if a<b and (np.intersect1d(roles[a]['row_id'],roles[b]['row_id']).size or np.intersect1d(roles[a]['unit'],roles[b]['unit']).size):raise ValueError('source role people/groups overlap')
    fit=roles['OSF_DEFENSE_FIT'];mask=independent.folds(fit['unit'])==2
    if np.intersect1d(fit['unit'][mask],fit['unit'][~mask]).size:raise ValueError('FIT2 duplicate group leakage')
    roles['OSF_DEFENSE_FIT']={key:value[mask] for key,value in fit.items() if key!='feature_names'}|{'feature_names':fit['feature_names']}
    return roles,names


def checkpoint(payload,package,names):
    import torch
    if payload.get('schema')!='TCR1-trainable-own-head-checkpoint-v1' or payload.get('class_orders')!=[[0,1],list(range(6))] or payload.get('seed')!=package['seed'] or list(payload.get('feature_names',[]))!=names:raise ValueError('checkpoint seed/features/head class order differs')
    shapes={}
    for i,k in ((0,2),(1,6)):
        for layer,ins,outs in ((0,83,64),(2,64,64),(4,64,16)):
            shapes[f'enc.{i}.{layer}.weight']=(outs,ins);shapes[f'enc.{i}.{layer}.bias']=(outs,)
        shapes[f'head.{i}.weight']=(k,16);shapes[f'head.{i}.bias']=(k,)
    state=payload['model']
    if set(state)!=set(shapes) or any(not isinstance(v,torch.Tensor) or tuple(v.shape)!=shapes[name] or v.dtype!=(torch.float32 if name.startswith('enc.') else torch.float64) or not torch.isfinite(v).all() for name,v in state.items()):raise ValueError('actual parameter shape/precision/finiteness differs')
    if sum(v.numel() for v in state.values())!=21288:raise ValueError('actual parameter count differs')


def wire_members(inputs,pin,role):
    import numpy as np
    path=inputs.path(pin);wanted={'row_id','unit','r1','r2'};unopened={'sex','train','validation'} if role=='ATTACK_FIT_NEW' else set()
    with ZipFile(path) as archive:
        names=[entry.filename for entry in archive.infolist()]
        if len(names)!=len(set(names)) or set(names)!={name+'.npy' for name in wanted|unopened}:raise ValueError('exact saved wire archive schema differs')
    # No sex/train/validation member is opened. The complete authorized ATT
    # archive is still hashed as its registered file pin requires.
    with np.load(path,allow_pickle=False) as archive:return {name:archive[name].copy() for name in wanted}


def project(independent,payload,wires):
    import numpy as np
    results=[];ranks=[]
    for i,wire in enumerate(wires):
        weight=payload['model'][f'head.{i}.weight'].numpy();bias=payload['model'][f'head.{i}.bias'].numpy();g=independent.geometry(weight,bias,wire[:,:16],RANK_RULE)
        if g['decision_mismatch_rows'] or g['AP_max_absolute_residual']>1e-10 or g['max_absolute_logit_difference']>1e-10 or g['max_absolute_probability_difference']>1e-10:raise AssertionError('independent current-head projection parity differs')
        protected=g['projected_z'];logits=protected@weight.T+bias;logits-=logits.mean(1,keepdims=True);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        results.append(np.column_stack((protected,logits,p,p.argmax(1))));ranks.append(g['rank'])
    return results,ranks


def run(inputs,independent):
    import numpy as np
    cfg=inputs.cfg;manifest=inputs.json(cfg['manifest']);roles,names=features(inputs,manifest,independent);packages=cfg['packages'];records=[]
    identity={(p['arm_id'],p['seed']) for p in packages};arms={p['arm_id'] for p in packages}
    if len(packages)!=72 or len(identity)!=72 or len(arms)!=36 or any({p['seed'] for p in packages if p['arm_id']==arm}!={0,1} for arm in arms) or sum(p['release_variant']=='full' for p in packages)!=66 or sum(p['release_variant']=='head_row_space' for p in packages)!=6:raise ValueError('all72 canonical packages/36 paired configurations required')
    if 'canonical_packages' not in cfg:raise ValueError('canonical package roster pin required')
    if 'canonical_packages' in cfg:
        canonical=inputs.json(cfg['canonical_packages'])['entries'];expected={(p['arm_id'],p['seed']):p for p in canonical}
        if len(canonical)!=72 or set(expected)!=identity:raise ValueError('canonical roster differs')
        for package in packages:
            expected_package=expected[package['arm_id'],package['seed']]
            if package['checkpoint']!={'path':expected_package['checkpoint'],'sha256':expected_package['checkpoint_sha256']} or package['release_variant']!=expected_package['release_variant']:raise ValueError('canonical checkpoint/variant binding differs')
    if 'rule' in cfg and any(cfg['rule'][name]!=value for name,value in RANK_RULE.items()):raise ValueError('frozen SVD rank/tie rule differs')
    for package in packages:
        inputs.guard(stage='verification');payload=independent.torch_load(inputs.path(package['checkpoint']));checkpoint(payload,package,names)
        for role,key in (('OSF_DEFENSE_FIT','fit_release'),('ATTACK_FIT_NEW','att_release'),('INNER_SELECTION','inner_release')):
            inputs.guard(stage='verification');data=roles[role];saved=wire_members(inputs,package[key],role)
            if any(not np.array_equal(saved[name],data[name]) for name in ('row_id','unit')):raise AssertionError('actual source and frozen wire people/order differ')
            actual=independent.forward(payload,data['X']);ranks=None
            if package['release_variant']=='head_row_space':actual,ranks=project(independent,payload,actual)
            recipients=[]
            for i,k in ((0,2),(1,6)):
                expected=saved[f'r{i+1}'];wire=actual[i]
                if expected.shape!=wire.shape or not np.isfinite(expected).all() or not np.isfinite(wire).all():raise ValueError('saved complete wire shape/finiteness differs')
                errors={'representation':float(abs(wire[:,:16]-expected[:,:16]).max()),'centered_logits':float(abs(wire[:,16:16+k]-expected[:,16:16+k]).max()),'probabilities':float(abs(wire[:,16+k:16+2*k]-expected[:,16+k:16+2*k]).max())}
                representation_atol=0. if package['release_variant']=='full' else 1e-10;head_atol=2e-12 if package['release_variant']=='full' else 1e-10
                if errors['representation']>representation_atol or errors['centered_logits']>head_atol or errors['probabilities']>head_atol or not np.array_equal(wire[:,-1],expected[:,-1]):raise AssertionError('independent actual checkpoint release component differs')
                recipients.append({'recipient':i+1,'max_absolute_difference':errors,'representation_atol':representation_atol,'head_atol':head_atol,'decision_mismatch_rows':0,'status':'PASS'})
            records.append({'arm_id':package['arm_id'],'seed':package['seed'],'release_variant':package['release_variant'],'role':role,'rows':len(data['X']),'groups':len(np.unique(data['unit'])),'checkpoint_sha256':package['checkpoint']['sha256'],'release_sha256':package[key]['sha256'],'ranks':ranks,'recipients':recipients,'status':'PASS'})
    maximum={component:max(recipient['max_absolute_difference'][component] for record in records for recipient in record['recipients']) for component in ('representation','centered_logits','probabilities')}
    return {'maximum_absolute_component_differences':maximum,'schema':'OB1-R7-independent-all72-actual-release-replay-v1','status':'PASS','canonical_packages':72,'paired_configurations':36,'package_role_checks':216,'recipient_functional_forwards':432,'records':records,'scientific_runner_imports':False,'feature_exports_only':True,'true_label_arrays_materialized':False,'authorized_ATT_SEX_archive_bytes_hashed':True,'ATT_label_or_split_members_opened':False,'INNER_label_bytes_read_or_hashed':False,'new_fits':0}


def main():
    from occupation_bottleneck_independent import replay
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=replay.Inputs(args.spec)
    inputs.path(Path(__file__));inputs.path(Path(replay.__file__))
    result=run(inputs,replay);output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'actual_checkpoint_release_checks':216}))


if __name__=='__main__':main()

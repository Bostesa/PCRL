"""Finite independent fixtures: numerical entry only under the root queue."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def fixtures():
    import numpy as np
    import torch
    from .registry import membership,score_keys,common_epoch,strongest_control,SERVICES,CONTROLS
    from .replay import forward,geometry,weighted_auc,vectors,reader_table
    from .safeguard import project,finite_step
    torch.set_num_threads(1);tests=[]
    def passed(name):tests.append({'test':name,'status':'PASS'})
    family=membership();assert len(family[:216])==216 and len(family[216:])==24
    assert all(sum(e.get('service')==s for e in family)==24 for s in SERVICES)
    assert len(score_keys('r1'))==18 and len(score_keys('pair'))==54
    passed('K240 derives from nine complete services and eight contrasts')
    base={'income':{'accuracy':.8,'nll':.4,'brier':.2},'occupation':{'accuracy':.5,'nll':1.2,'brier':.6}}
    starts={0:base,1:base};records=[]
    for epoch in (0,5,10,20):
        for seed in (0,1):
            tasks={task:dict(value) for task,value in base.items()}
            if epoch==20:tasks['occupation']['nll']+=.1
            records.append({'arm_id':'CONSTRAINED','epoch':epoch,'seed':seed,'FIT2':tasks,
                            'ATT_validation_reader_auc':{k:.9-.002*epoch for k in score_keys('pair')}})
    assert common_epoch(records,starts,'CONSTRAINED')['epoch']==10
    assert common_epoch(records[:-1],starts,'CONSTRAINED')['epoch'] is None
    for r in records:
        if r['epoch']==10:r['ATT_validation_reader_auc']['pair__IGNORE_r2__LR_C1']=.99
    assert common_epoch(records,starts,'CONSTRAINED')['epoch']==5
    passed('paired common epoch eligibility and all54 ATT selection')
    for r in records:r['FIT2']['occupation']['nll']+=.02
    assert common_epoch(records,starts,'CONSTRAINED')['status']=='NOMINATED_PROXY_INELIGIBLE'
    passed('proxy-ineligible fallback preserves original rule')
    utility=[{**e,'point':0.} for e in family[:216]];pair={s:{0:.8,1:.8} for s in CONTROLS}
    assert strongest_control(utility,pair)['control']=='RAW_PROJ'
    for e in utility:
        if e['service']=='RAW_PROJ':e['point']=.1
    assert strongest_control(utility,pair)['control']=='RAW_HYBRID'
    passed('strongest control retains RAW_PROJ and prelocked tie order')
    rule={'relative_rank_cutoff':1e-12,'epsilon_multiplier':32,'near_tie_absolute_logit_gap':1e-10}
    w=np.zeros((2,16));w[:,0]=[-1,1];b=np.zeros(2);z=np.zeros((4,16));z[:,1]=[0,1,2,3]
    g=geometry(w,b,z,rule);assert g['rank']==1 and not g['projected_z'].any() and g['near_tie_rows']==4 and g['decision_mismatch_rows']==0
    zero=geometry(w*0,b,z,rule);assert zero['rank']==0 and zero['decision_mismatch_rows']==0
    passed('factorized income discards residual and first-class near ties')
    generator=torch.Generator().manual_seed(471);state={}
    for i,k in enumerate((2,6)):
        for layer,inn,out in ((0,83,64),(2,64,64),(4,64,16)):
            state[f'enc.{i}.{layer}.weight']=torch.randn((out,inn),generator=generator)*.02
            state[f'enc.{i}.{layer}.bias']=torch.randn((out,),generator=generator)*.02
        state[f'head.{i}.weight']=torch.randn((k,16),generator=generator,dtype=torch.float64)*.02
        state[f'head.{i}.bias']=torch.zeros(k,dtype=torch.float64)
    payload={'schema':'TCR1-trainable-own-head-checkpoint-v1','class_orders':[[0,1],list(range(6))],'model':state}
    x=np.zeros((1027,83),dtype=np.float32);raw=forward(payload,x)
    basis=geometry(state['head.0.weight'].numpy(),state['head.0.bias'].numpy(),raw[0][:,:16],rule)['B']
    binding={'heads':[{'B':basis.tolist()},{'B':[]}]};contract={'income':'head_row_space','occupation':'full'}
    hybrid=forward(payload,x,wire_contract=contract,geometry_binding=binding)
    assert hybrid[0].shape==(1027,21) and hybrid[1].shape==(1027,29)
    assert np.array_equal(hybrid[1],raw[1]);assert np.max(abs(hybrid[0][:,:16]-raw[0][:,:16]))>0
    for _arm in ('CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY'):
        assert np.array_equal(forward(payload,x,wire_contract=contract,geometry_binding=binding)[0],hybrid[0])
    passed('hybrid income projection is actual wire across continuation arms')
    assert np.array_equal(reader_table(*hybrid,'DECISION_LR_C1','pair'),np.column_stack((hybrid[0][:,-1],hybrid[1][:,-1])))
    assert np.array_equal(reader_table(*hybrid,'CONFIDENCE_LR_C1','r1'),hybrid[0][:,18:20].max(1,keepdims=True))
    tiny=np.array([[0.],[1e-150],[2e-150]]);sd=tiny.std(0);assert sd[0]>0 and np.isfinite((tiny-tiny.mean(0))/sd).all()
    passed('reader feature subsets and small positive variances retained')
    assert weighted_auc([0,1],[.5,.5])[0]==.5 and weighted_auc([0,1],[0.,1.])[0]==1.
    assert abs(vectors(np.array([0]),np.array([[.5,.5]]))['brier'][0]-.5)<1e-15
    passed('tie-block AUC and summed Brier convention')
    d=project(np.array([1.,-1.]),np.array([[1e-200,0.],[0.,1.]]),.01)
    assert abs(d[0])<1e-10 and d[1]<0 and np.linalg.norm(d)<=.01
    assert not project(np.zeros(2),np.eye(2)).any()
    passed('QP actual-displacement sign, zero feasibility and tiny nonzero guard')
    arrays=[np.array([.3],np.float32),np.array([-.2],np.float64)];saved=[a.copy() for a in arrays]
    result=finite_step(arrays,np.array([.01,0.]),np.array([[0.,1.],[0.,1.]]),
                       lambda:{'nll':float(arrays[0][0]),'brier':float(arrays[0][0])})
    assert result['status']=='REJECTED_ZERO_STEP' and all(a.tobytes()==b.tobytes() for a,b in zip(arrays,saved))
    passed('failed finite checks restore mixed-dtype parameter bytes')
    arrays=[np.array([100.],np.float32),np.array([-.2],np.float64)]
    result=finite_step(arrays,np.array([1e-15,0.]),np.zeros((2,2)),lambda:{'nll':1.,'brier':.5})
    assert result['status']=='ROUNDING_ZERO_STEP' and result['accepted_fraction']==1.
    passed('feasible accepted rounding-zero distinguished from nonzero accepted')
    return {'schema':'CIS1-independent-preflight-v1','status':'PASS','tests':tests,
            'test_count':len(tests),'scientific_runner_imports':False,'source_artifacts_opened':False,
            'engineering_fixtures_only':True,'Adult_benefit_established':False}


def main():
    from .replay import Inputs
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args()
    inputs=Inputs(args.spec);result=fixtures();inputs.guard(stage='verification')
    if inputs.cfg.get('endpoint_registry'):
        from .registry import validate_registry
        validate_registry(inputs.json(inputs.cfg['endpoint_registry']))
        result['frozen_K240_registry_validation']='PASS'
    output=Path(inputs.cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'test_count':result['test_count']}))

if __name__=='__main__':main()

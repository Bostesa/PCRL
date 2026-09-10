import copy,json
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits
from tests.test_acs_coalition_training import fixture_data
from experiments import acs_fixed_predictions_training as t
from experiments import acs_fixed_predictions_audits as a
from experiments import acs_bottleneck_training as old
from experiments import acs_coalition_training as hist
from experiments.acs_transfer_heads import FittedCandidate,InputStandardizer,metrics,_network
from scripts.acs_coalition_strength_comparisons import evaluate_pair

def anchors(n):
    rng=np.random.default_rng(18);p=rng.uniform(.05,.95,(n,3));return {'A':np.column_stack((1-p[:,0],p[:,0],1-p[:,1],p[:,1])),'B':np.column_stack((1-p[:,2],p[:,2]))}

def test_binary_full_vector_lossless_wire():
    p=np.array([[.1,.9]],dtype=np.float32).astype(np.float64)
    assert p[0,0]!=1-p[0,1]
    an={'A':np.column_stack((p,p)),'B':p.copy()};w,d=t.array_wires(np.ones((1,16),np.float32),an)
    assert w['AB'].dtype==np.float64 and np.array_equal(w['A'][:,:2],p)
    assert metrics(np.array([0]),w['A'][:,:2],2)==metrics(np.array([0]),p,2)


def test_initial_heads_parameters_and_gradient_routes():
    raw,s,y,pre,state,*_=fixture_data();m=t.Auxiliary(pre,state);x=m.standardize(raw);an={k:torch.tensor(v,dtype=torch.float32) for k,v in anchors(len(raw)).items()};o,_=t.make_observers(0)
    assert sum(p.numel() for p in m.parameters())==3186
    np.testing.assert_allclose(m(x)[0].detach().numpy(),raw[:,:16],atol=1e-5,rtol=1e-5)
    for name,h in m.branch.heads.items():
        for k,v in h.state_dict().items():assert torch.equal(v,state[f'heads.{name}.{k}'])
    ss=old._labels(s,len(raw),hist.SOURCE_SCHEMA,'s');yy=old._labels(y,len(raw),hist.ATTRIBUTE_SCHEMA,'y');priors=hist.label_priors(ss,yy)
    d=t.diagnostic(m,o,x,an,ss,yy,priors,'J');assert d['B_gradient_zero'];assert d['gradients']['mapper']['source']['raw_norm']>0
    assert d['gradients']['heads']['coalition']['raw_norm']==0
    c,detail=t.components(m,o,x,an,ss,yy,priors)
    ce=detail['ce'];expected=sum(ce[v+'__'+target]/priors[target]['entropy']/2 for v in ('A','B') for target in ('SEX','RAC1P'))
    assert torch.allclose(c['extra_local'],expected)
    m0=copy.deepcopy(m);opt=old._adam(m.parameters());before=old.tree_digest(an)
    t.forward_step(m,o,opt,x,an,ss,yy,priors,'A0');assert before==old.tree_digest(an);assert old.tree_digest(m.state_dict())!=old.tree_digest(m0.state_dict())


def test_raw_projection_precedes_preprocessing():
    with torch.random.fork_rng(devices=[]):net=_network(2,[64,32],2).eval()
    c=FittedCandidate('mlp',net,InputStandardizer(np.array([3.,5.]),np.array([2.,4.])),2,{})
    x=np.array([[11.,99.,17.],[2.,88.,1.]])
    witness=a.AuditCandidate(c,'wire',(0,2),{})
    assert np.array_equal(witness.predict_proba({'wire':x}),c.predict_proba(np.ascontiguousarray(x[:,[0,2]])))
    with pytest.raises(ValueError):witness.predict_proba({'hidden':x})


def test_B_alias_preserves_full_optimizer_slots():
    o,_=t.make_observers(0,True);q,_=t.make_observers(0,False);oo=old._adam(o.parameters());qo=old._adam(q.parameters());an=anchors(12);aw={k:torch.tensor(v,dtype=torch.float32) for k,v in an.items()};w=t.tensor_wires(torch.zeros(12,16),aw)
    labels={k:torch.arange(12)%(9 if k=='RAC1P' else 2) for k in (*hist.SOURCE_SCHEMA,*hist.ATTRIBUTE_SCHEMA)};priors=hist.label_priors({k:labels[k] for k in hist.SOURCE_SCHEMA},{k:labels[k] for k in hist.ATTRIBUTE_SCHEMA})
    t.observer_step(o,oo,w,labels,priors);t.copy_b(q,qo,o,oo)
    for r in o:
        if r.startswith('B__'):
            assert old.tree_digest(o[r].state_dict())==old.tree_digest(q[r].state_dict())
            for p,p0 in zip(q[r].parameters(),o[r].parameters()):assert old.tree_digest(qo.state[p])==old.tree_digest(oo.state[p0])


def test_minature_full_training_forks_static_and_B(tmp_path):
    raw,s,y,pre,state,*_=fixture_data();torch.set_num_threads(1)
    with threadpool_limits(limits=1):arms,meta=t.train_seed(raw,s,y,pre,state,anchors(len(raw)),raw[:,:16]*.7,0,tmp_path/'training',miniature=True)
    assert len(arms)==6 and len(set(v['fork_sha256'] for k,v in meta['arms'].items() if k in t.ARMS))==1
    hashes=meta['B_final_identity'];assert all(v==hashes['A0'] for v in hashes.values())
    n=(len(raw)+15)//16
    assert meta['arms']['H']['counts']['observer_steps']==7*n
    cps=[torch.load(tmp_path/'training'/c/'fork.pt',weights_only=True) for c in t.ARMS]
    assert all(old.tree_digest(cp)==old.tree_digest(cps[0]) for cp in cps)
    assert meta['reserved_labels_received'] is False


def test_source_native_is_not_probe_admission_and_no_mean_match():
    u={'income_binary':.2,'civilian_at_work':.3,'public_coverage':.4,'same_residence':.5,'commute_over20':.6};p={'utility':u,'gains':{'AB/SEX':.1}};q=copy.deepcopy(p);q['utility']['income_binary']+=.02;q['utility']['public_coverage']-=.02
    r=evaluate_pair(p,q,u,list(u),.001)
    assert not r['both_source_feasible'] and not r['utility_close']
    q=copy.deepcopy(p);q['utility']['same_residence']-=.005
    r=evaluate_pair(q,p,u,list(u),.001);assert r['utility_directional'] and not r['utility_close']


def test_complete_miniature_audits_ancestor_and_selected_replay(tmp_path):
    torch.set_num_threads(1);n=32;aa=anchors(n);rng=np.random.default_rng(72)
    wire={v:{} for v in a.VIEWS};aug={v:{} for v in a.VIEWS};derived={v:{} for v in a.VIEWS}
    raw,s,y,pre,state,*_=fixture_data()
    trained,_=t.train_seed(raw,s,y,pre,state,anchors(len(raw)),raw[:,:16]*.7,0,tmp_path/'mini_training',miniature=True)
    model=trained['J'][0]
    for pool in a.FIT_POOLS:
        h=model(model.standardize(raw[:n]))[0].detach().numpy();w,_=t.array_wires(None,aa);ww,dd=t.array_wires(h,aa,model)
        for v in a.VIEWS:wire[v][pool]=w[v];aug[v][pool]=ww[v];derived[v][pool]=dd[v]
    labels={p:{target:np.arange(n)%(9 if target=='RAC1P' else 2) for target in a.TARGETS} for p in a.FIT_POOLS};ix={t:np.arange(n) for t in a.TARGETS}
    o=trained['H'][1];q=trained['J'][1]
    for r in o:
        if r.startswith('B__'):q[r].load_state_dict(o[r].state_dict())
    budget={'mlp':{'epochs':2,'validation_interval':1},'histgb':{'max_iter':2},'logistic':{'max_iter':20}}
    with threadpool_limits(limits=1):
        base=a.fit_condition_audits(wire,None,labels,ix,o,0,tmp_path/'H',interface='H',epochs=2,nested_epochs=1,miniature=True,budget=budget)
        result=a.fit_condition_audits(aug,derived,labels,ix,q,0,tmp_path/'aug',interface='learned',epochs=2,nested_epochs=1,miniature=True,budget=budget,ancestor=base)
    # Finish the miniature pipeline with correctly routed readouts and saved scored predictions.
    from experiments.run_acs_fixed_predictions import utilities
    from experiments import run_acs_coalition as runner
    def full(views):
        return {v:{**arr,'downstream_fit':arr['attacker_fit'],'downstream_validation':arr['attacker_validation']} for v,arr in views.items()}
    labs={**labels,'downstream_fit':labels['attacker_fit'],'downstream_validation':labels['attacker_validation']}
    ti={task:np.arange(n) for task in runner.TASKS};weights={p:np.ones(n) for p in labs};hd=tmp_path/'H_scored';jd=tmp_path/'J_scored';hd.mkdir();jd.mkdir()
    with threadpool_limits(limits=1):
        hu=utilities(full(wire),labs,ti,0,hd)
        (hd/'selection_before_test.json').write_text(json.dumps({'utility':hu['selection'],'utility_metadata':hu['metadata']}))
        ju=utilities(full(aug),labs,ti,0,jd,hd)
        runner.score_condition(0,'H',full(wire),None,{v:wire[v]['attacker_validation'] for v in a.VIEWS},None,hu,base,labs,labels['attacker_validation'],weights,np.ones(n),hd)
        runner.score_condition(0,'J',full(aug),full(derived),{v:aug[v]['attacker_validation'] for v in a.VIEWS},{v:derived[v]['attacker_validation'] for v in a.VIEWS},ju,result,labs,labels['attacker_validation'],weights,np.ones(n),jd)
    assert (jd/'predictions.npz').exists() and json.loads((jd/'projection_parity.json').read_text())['all_exact']
    assert base['metadata']['counts']['new_five_candidate_roles']==11
    assert result['metadata']['counts']['new_five_candidate_roles']==12
    assert result['metadata']['counts']['own_catchup_trajectories']==5
    for b,roles in result['candidates'].items():
        for role,cs in roles.items():
            v,target=role.split('/');bundle={'wire':aug[v]['attacker_validation'],'derived':derived[v]['attacker_validation']}
            for cid,c in cs.items():
                assert metrics(labels['attacker_validation'][target],c.predict_proba(bundle),a.CLASSES[target])==c.metadata['validation_scores']
                if cid.startswith('anchor__'):
                    bc=base['candidates'][b][role][cid.removeprefix('anchor__')]
                    assert np.array_equal(c.predict_proba(bundle),bc.predict_proba({'wire':wire[v]['attacker_validation']}))
            for scope in a.SCOPES:
                chosen=cs[result['selection'][b][role][scope]]
                assert all(chosen.metadata['validation_scores']['log_loss']<=cs[cid].metadata['validation_scores']['log_loss'] for cid in result['selection_pools'][b][role][scope])


def test_configuration_and_reserved_schema_boundary(tmp_path):
    from pathlib import Path
    cfg=json.loads((Path(__file__).resolve().parents[1]/'results/redesign_20260909_acs_fixed_predictions_v1/config.json').read_text())
    assert cfg['objectives']=={k:list(v) for k,v in t.COEF.items()}
    assert cfg['wire_dimensions']=={'H':[4,2,6],'E':[20,2,22],'learned':[20,2,22],'public_auxiliary_probability':[8,2,10]}
    raw,s,y,pre,state,*_=fixture_data();bad={**s,'same_residence':np.zeros(len(raw),dtype=int)}
    with pytest.raises(ValueError):t.train_seed(raw,bad,y,pre,state,anchors(len(raw)),raw[:,:16],0,tmp_path/'forbidden',miniature=True)

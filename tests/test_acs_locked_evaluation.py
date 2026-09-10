"""Behavioral checks for the inference-only, household-locked evaluation."""
import importlib
import numpy as np
import pandas as pd
import pytest


def api():
    return importlib.import_module('experiments.acs_locked_evaluation')


def fixture():
    return pd.DataFrame({'SERIALNO':['0001','0001','0002','0003','0003','0003'],
                         'SPORDER':['01','02','01','01','02','03'],
                         'AGEP':[19,34,25,20,22,30], 'PWGTP':[2,3,1,1,4,5]})


def test_household_exclusion_catches_unsampled_member_and_preserves_strings():
    m=api(); f=m.canonical_people(fixture()); h=f.iloc[0].household_id
    new=m.select_households(f,{h},cap=30)
    assert len(new)==4 and not new.household_id.eq(h).any()
    assert '0001' in f.iloc[0].person_id and '01' in f.iloc[0].person_id


def test_conflicting_duplicate_is_error_and_identical_deduplicates():
    m=api(); f=fixture(); assert len(m.canonical_people(pd.concat([f,f.iloc[:1]])))==6
    bad=f.iloc[:1].copy();bad['PWGTP']=9
    with pytest.raises(ValueError,match='Conflicting'):m.canonical_people(pd.concat([f,bad]))


def test_integral_age_and_missing_keys():
    m=api();f=fixture();f['AGEP']=f.AGEP.astype(float);f.loc[0,'AGEP']=19.5
    assert len(m.canonical_people(f))==5
    f.loc[1,'SPORDER']=None
    with pytest.raises(ValueError,match='key'):m.canonical_people(f)


def test_deterministic_whole_household_prefix_stops_without_gap_filling():
    m=api();f=m.canonical_people(fixture());order=m.household_order(f.household_id.unique())
    sizes=f.groupby('household_id').size();cap=int(sizes[order[0]]+sizes[order[1]]-1)
    a=m.select_households(f,set(),cap);b=m.select_households(f.sample(frac=1,random_state=6),set(),cap)
    assert set(a.household_id)=={order[0]};assert a.person_id.tolist()==b.person_id.tolist()
    assert len(m.select_households(f,set(f.household_id),30000))==0


def test_raw_projection_precedes_frozen_standardization():
    m=api();x=np.array([[.1,.9,10.,20.,.4,.6]])
    class Model:
        def predict_proba(self,z):return (z-np.array([.2,.8,.5,.5]))/np.array([2,3,4,5])
    assert np.array_equal(m.project_predict(Model(),x,[0,1,4,5]),Model().predict_proba(x[:,[0,1,4,5]]))


def test_signed_recovery_when_fixed_winner_loses_to_available_h_witness():
    m=api();y=np.array([0,1]);h=np.array([[.8,.2],[.2,.8]]);chosen=np.array([[.6,.4],[.4,.6]])
    prior=m.losses(y,np.full((2,2),.5)); hg=(prior-m.losses(y,h)).mean();ag=(prior-m.losses(y,chosen)).mean()
    assert ag-hg<0;assert np.array_equal(m.project_predict(type('M',(),{'predict_proba':lambda _,x:x})(),np.c_[h,chosen],[0,1]),h)


def test_fresh_source_reference_and_directional_not_close():
    m=api();ref={'income':.4};j={'income':.3,'residence':.4};c={'income':.301,'residence':.42}
    z=m.match(j,c,ref,['income','residence'],.001)
    assert z['both_source_feasible'] and z['utility_directional'] and not z['utility_close']
    assert not m.match(j,c,{'income':.28},['income'],.001)['both_source_feasible']


def test_household_bootstrap_matches_explicit_duplicates_shared_across_seeds():
    m=api();groups=np.array([0,1,1,1,2]);w=np.array([1,2,3,7,4.]);mask=np.array([1,1,0,1,1],bool)
    differences=np.arange(30,dtype=float).reshape(3,5,2)/50
    sums,den=m.household_sums(differences,groups,np.column_stack([mask,mask*w]))
    draws=np.array([[1,1,0],[2,0,2],[0,1,2]])
    actual=m.bootstrap_values(sums,den,draws)
    for b,draw in enumerate(draws):
        ix=np.concatenate([np.flatnonzero(groups==g) for g in draw])
        expected=[]
        for j,weights in enumerate([mask,mask*w]):
            expected.append(np.mean([np.sum(differences[s,ix,j]*weights[ix])/weights[ix].sum() for s in range(3)]))
        np.testing.assert_allclose(actual[b],expected,rtol=1e-14,atol=1e-14)


def test_bootstrap_degenerate_is_not_success():
    m=api();r=m.intervals(np.zeros(6),np.zeros((2000,6)))
    assert all(x['status']=='degenerate' and x['simultaneous'] is None for x in r['contrasts'])


def test_frozen_categories_and_numeric_statistics_do_not_change():
    m=api();meta={'numeric':{'AGEP':[25,25,4],'WKHP':[40,35,9]},
        'categories':{k:[1] for k in ('SCHL','MAR','RELP','CIT','DIS','DEAR','DEYE','DREM')},'feature_names':[]}
    f=fixture();f['WKHP']=[np.nan,40,35,20,99,0]
    for k in meta['categories']:f[k]=[1,2,np.nan,1,1,1]
    import copy
    before=copy.deepcopy(meta);pre=m.frozen_preprocessor(meta);x=pre.transform(f)
    assert meta==before and x.dtype==np.float32
    assert x[1,6]==1 and x[2,5]==1  # unseen valid and missing are distinct
    assert x[0,2]==pytest.approx((40-35)/9) and x[0,3]==1


def test_complete_graphs_and_all_selected_outputs_synthetic(tmp_path):
    """Synthetic members split across old/new proposals; full graph and scoring."""
    m=api();import torch
    f=m.canonical_people(fixture());sample=m.select_households(f,{f.iloc[0].household_id})
    n=len(sample);rng=np.random.default_rng(19);pca=rng.normal(size=(n,32)).astype(np.float32)
    state={'input_mean':torch.zeros(32,dtype=torch.float64),'input_scale':torch.ones(32,dtype=torch.float64)}
    for key,shape in [('branch.mapper.0.weight',(64,32)),('branch.mapper.0.bias',(64,)),
                      ('branch.mapper.2.weight',(16,64)),('branch.mapper.2.bias',(16,))]:
        state[key]=torch.tensor(rng.normal(0,.01,shape),dtype=torch.float32)
    for t in m.TASKS[:2]:
        state[f'branch.heads.{t}.weight']=torch.ones((1,16))*.01;state[f'branch.heads.{t}.bias']=torch.zeros(1)
    checkpoint={'model_state':state};before={k:v.clone() for k,v in state.items()}
    anchors={'A':np.tile([.2,.8,.4,.6],(n,1)),'B':np.tile([.7,.3],(n,1))}
    class Teacher:
        def apply(self,x):return x.copy()
    class Reader:
        def predict_proba(self,x):return x[:,:2]
    all_b=[]
    for seed in range(3):
        for arm in m.ARMS:
            graph=m.inference_graph(pca,anchors,Teacher(),checkpoint,arm)
            assert graph['wire']['A'].shape==(n,4 if arm=='H' else 20)
            assert graph['wire']['AB'].dtype==np.float64
            assert np.array_equal(graph['wire']['A'][:,:4],anchors['A'])
            witness=m.project_predict(Reader(),graph['wire']['AB'],[0,1])
            assert np.array_equal(witness,anchors['A'][:,:2])
            y=np.array([0,1,-1,0]);scores,ll=m.score_predictions(y,witness,sample.PWGTP.to_numpy())
            assert scores['unweighted']['n']==3 and np.isnan(ll[2])
            all_b.append(graph['wire']['B'])
    assert all(np.array_equal(b,all_b[0]) for b in all_b)
    assert all(torch.equal(v,before[k]) for k,v in state.items())
    assert all(not v.requires_grad for v in state.values())


def test_no_fitting_or_selection_in_evaluation_primitives():
    m=api();import ast,inspect
    tree=ast.parse(inspect.getsource(m))
    forbidden={'fit','fit_transform','partial_fit','backward','step','train','fit_candidates','fit_prior','argmin'}
    calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
           for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not forbidden.intersection(calls)
    with pytest.raises(RuntimeError,match='blocked'):m.require_independence({'independence_verified':False,'sample_rows':0})


def test_nonfinite_output_is_not_replaced_with_prior():
    with pytest.raises(ValueError,match='Nonfinite'):api().losses(np.array([0]),np.array([[np.nan,.5]]))


def test_completion_record_refuses_incompatible_resume(tmp_path):
    m=api();p=tmp_path/'done.json';m.atomic_json(p,{'count':1});m.atomic_json(p,{'count':1})
    with pytest.raises(RuntimeError,match='incompatible'):m.atomic_json(p,{'count':2})


def test_saved_inference_completion_detects_modified_predictions(tmp_path,monkeypatch):
    r=importlib.import_module('scripts.run_acs_locked_evaluation');monkeypatch.setattr(r,'ROOT',tmp_path)
    out=tmp_path/'study';out.mkdir();(out/'LOCK.json').write_text('{}')
    m=api();files={}
    for s in range(3):
        for c in m.ARMS:
            d=out/'local'/f'seed_{s}'/c;d.mkdir(parents=True);p=d/'predictions.npz';p.write_bytes(b'fixture')
            unit={'seed':s,'condition':c,'lock_sha256':r.sha_file(out/'LOCK.json'),'files':{str(p.relative_to(tmp_path)):r.sha_file(p)}}
            m.atomic_json(d/'inference_complete.json',unit);files[s,c]=unit
    m.atomic_json(out/'INFERENCE_COMPLETION.json',{'complete':True,'systems':list(files.values())})
    assert r.verify_inference_completion(out)==18
    p.write_bytes(b'changed')
    with pytest.raises(AssertionError):r.verify_inference_completion(out)


def test_deterministic_bootstrap_resume_rejects_changed_draws(tmp_path):
    r=importlib.import_module('scripts.run_acs_locked_evaluation');p=tmp_path/'draws.npz'
    original=r.bootstrap_draws(p,4);assert original.shape==(2000,4)
    assert np.array_equal(r.bootstrap_draws(p,4),original)
    with p.open('wb') as f:np.savez_compressed(f,draws=np.zeros_like(original))
    with pytest.raises(AssertionError):r.bootstrap_draws(p,4)

"""Synthetic table and release contracts; no survey loaders or outcomes."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logit


def module():
    return importlib.import_module('experiments.pcrl_task_directed_release_v1.mechanisms')


class Code:
    def n_states(self, name):
        return 2 if name == 'T0' else 4

    def parents(self, name):
        return np.arange(2) if name == 'T0' else np.repeat(np.arange(2), 2)


class Partitions:
    centers = np.zeros((2, 2))

    def assign(self, ha, hb):
        a = (ha[:, 0] > .5).astype(int)
        return a, 2*a+(hb[:, 0] > .5).astype(int)


def fixture():
    pools, encoded = {}, {}
    for pool, n in [('representation_fit', 8), ('downstream_validation', 4)]:
        t = np.arange(n) % 2
        ha = np.column_stack((t*.7+.1, np.full(n, .2), np.full(n, .3), np.full(n, .4))).astype(np.float32)
        hb = np.column_stack((np.arange(n) % 3 > 0, np.full(n, .4))).astype(np.float32)
        labels = {'SEX': np.arange(n) % 2, 'RAC1P': np.arange(n) % 3,
                  'same_residence': np.arange(n) % 2}
        if n == 8:
            labels['same_residence'][2] = -1
            labels['RAC1P'][3] = -1
        pools[pool] = {'x': np.arange(n*32).reshape(n,32)/100, 'ha': ha, 'hb': hb,
                       'J': np.ones((n, 3)), 'weights': np.arange(1., n+1), 'labels': labels}
        b = np.linspace(.2, .6, n)
        encoded[pool] = {'b': b, 'p': np.linspace(.1, .9, n),
            'codes': {'T0': t, 'Ttask': 2*t+(np.arange(n)//2)%2, 'Trisk': 2*t+(np.arange(n)//3)%2},
            'actions': {17: expit(logit(b)[:,None]+np.array([0.,-1.,1.]))}}
    encoder = SimpleNamespace(code=Code(), partitions=Partitions(),
                              dictionaries={17:{'offsets':np.array([0.,-1.,1.]),'zero_action':0}})
    return {'anchor':0, 'pools':pools}, encoder, encoded, {'mechanism':np.arange(2,8)}


def test_tables_use_independent_label_masks_and_normalize_each_measure():
    m=module();ctx,e,enc,split=fixture()
    tables=m.estimate_tables(ctx,e,enc,split)
    assert set(tables)=={'T0','Ttask','Trisk'}
    t=tables['T0']
    assert_array_equal(t['state_mass'],[3,3])
    assert len(t['roles'])==8
    assert t['support']['A/SEX']['count'].sum()==6
    assert t['support']['A/RAC1P']['count'].sum()==5
    assert t['population']['utility_valid_rows']==5
    for law in t['roles'].values():
        assert law.sum()==pytest.approx(1)
    assert t['support']['A/SEX']['sumw'].sum()==33
    assert t['support']['A/SEX']['sumw2'].sum()==199
    rows=np.array([3,4,5,6,7]);y=ctx['pools']['representation_fit']['labels']['same_residence'][rows]
    p=enc['representation_fit']['actions'][17][rows]
    loss=-y[:,None]*np.log(p)-(1-y[:,None])*np.log1p(-p)
    assert_allclose(t['cost_U'].sum(0),loss.mean(0))
    assert_allclose(t['cost_W'].sum(0),(loss*np.arange(4.,9)[:,None]).sum(0)/30)
    assert_allclose(t['cost'],.5*t['cost_U']+.5*t['cost_W'])


def test_all_refined_tables_aggregate_to_one_coarse_table():
    m=module();ctx,e,enc,split=fixture();ts=m.estimate_tables(ctx,e,enc,split)
    for family in ('Ttask','Trisk'):
        fine=ts[family]
        for key in ('cost_U','cost_W','cost'):
            assert_allclose(fine[key].reshape(2,2,3).sum(1),ts['T0'][key],atol=1e-12)
        for key,p in fine['roles'].items():
            assert_allclose(p.reshape(p.shape[0],p.shape[1],2,2).sum(3),ts['T0']['roles'][key],atol=1e-12)
        assert fine['aggregation']['maximum_error']<=1e-12


def test_fit_map_saves_tables_and_independently_checks_parent_embedding(tmp_path):
    m=module();ctx,e,enc,split=fixture();ts=m.estimate_tables(ctx,e,enc,split)
    base={'input':'T0','policy':'L','budget':0.,'max_actions':17,'configuration':'T0_L_0_a17'}
    coarse=m.fit_map(ctx,e,enc,ts,base,tmp_path/'coarse')
    refined=m.fit_map(ctx,e,enc,ts,{**base,'input':'Trisk','configuration':'Trisk_L_0_a17'},tmp_path/'fine',coarse_solution=coarse)
    assert coarse['feasible'] and refined['feasible']
    assert len(coarse['constrained_roles'])==4
    assert refined['embedding']['maximum_error']<=1e-12
    assert refined['objective']<=coarse['objective']+1e-7
    assert (tmp_path/'fine'/'Q.npz').is_file()
    assert (tmp_path/'fine'/'tables.npz').is_file()
    meta=json.loads((tmp_path/'fine'/'metadata.json').read_text())
    assert 'ids' not in meta and meta['embedding']['maximum_error']<=1e-12
    coalition=m.fit_map(ctx,e,enc,ts,{**base,'policy':'C','configuration':'T0_C_0_a17'},tmp_path/'coalition')
    assert len(coalition['constrained_roles'])==8


def base_mechanism():
    return {'input':'T0','policy':'U','budget':None,'configuration':'T0_U_unconstrained_a17',
            'Q':np.array([[1.,0.,0.],[0.,0.,1.]]),'actions':17,'constant_action':2,'feasible':True}


@pytest.mark.parametrize('rho',[0.,.25,.5,.75,1.])
def test_withholding_and_rr_are_actual_token_mixtures(rho):
    m=module();ctx,e,enc,_=fixture();mech=base_mechanism()
    pub=m.build_release(ctx,e,enc,f'T0_withhold_{rho:g}',mechanism=mech)
    rr=m.build_release(ctx,e,enc,f'T0_rr_{rho:g}',mechanism=mech)
    for pool in ctx['pools']:
        q=mech['Q'][enc[pool]['codes']['T0']]
        assert_allclose(pub[pool]['token_probs'][:,:3],rho*q)
        assert_allclose(pub[pool]['token_probs'][:,3],1-rho)
        assert_allclose(pub[pool]['fixed_probabilities'][:,3],enc[pool]['b'])
        assert_allclose(rr[pool]['token_probs'],rho*q+(1-rho)/3)
        g=enc[pool]['actions'][17]
        expected=(q*-np.log(g)).sum(1)
        got=(pub[pool]['token_probs']*-np.log(pub[pool]['fixed_probabilities'])).sum(1)
        assert_allclose(got,rho*expected+(1-rho)*-np.log(enc[pool]['b']))


def test_control_shapes_and_fixed_predictions_do_not_release_channel_rows():
    m=module();ctx,e,enc,_=fixture()
    for name in ('H','J','continuous_task','constant_best','independent_token','T0_code','Ttask_code','Trisk_code'):
        release=m.build_release(ctx,e,enc,name,mechanism=base_mechanism())
        for pool,arm in release.items():
            n=len(ctx['pools'][pool]['ha'])
            assert arm['token_probs'].shape[0]==n
            assert_allclose(arm['token_probs'].sum(1),1)
            assert_allclose(arm['global_offsets'],enc[pool]['actions'][17])
            if name=='continuous_task':
                assert_allclose(arm['aux'],enc[pool]['p'][:,None])
            elif name=='constant_best':
                assert arm['aux'] is None
                assert_allclose(arm['fixed_probabilities'],enc[pool]['actions'][17][:,[2]])
            elif name.endswith('_code'):
                assert_array_equal(arm['token_probs'].argmax(1),enc[pool]['codes'][name[:-5]])
    finite=m.build_release(ctx,e,enc,'T0_U_unconstrained_a17',mechanism=base_mechanism())
    assert finite['representation_fit']['aux'] is None


def test_historical_release_requires_exact_h_bytes(tmp_path):
    m=module();ctx,e,enc,_=fixture()
    path=tmp_path/'results/pcrl_invariant_baselines_v1/seed_0/releases/leace_A0'
    path.mkdir(parents=True)
    wires={f'wire/A/{pool}':np.column_stack((values['ha'],np.full((len(values['ha']),2),.25,dtype=np.float32))) for pool,values in ctx['pools'].items()}
    np.savez(path/'releases.npz',**wires)
    result=m.build_release(ctx,e,enc,'leace_A0',inputs_root=tmp_path)
    assert_allclose(result['representation_fit']['aux'],.25)
    wires['wire/A/representation_fit'][0,0]+=.01
    np.savez(path/'releases.npz',**wires)
    with pytest.raises(ValueError,match='H'):
        m.build_release(ctx,e,enc,'leace_A0',inputs_root=tmp_path)


def test_supervised_eraser_receives_only_pca32_and_teacher_logit():
    m=module();ctx,e,enc,_=fixture()
    class Eraser:
        def transform(self,x):
            assert x.shape[1]==33
            return x+1
    before={p:v['ha'].copy() for p,v in ctx['pools'].items()}
    result=m.build_release(ctx,e,enc,'leace_supervised',erasers={'leace_supervised':Eraser()})
    for pool in ctx['pools']:
        assert_allclose(result[pool]['aux'][:,-1],logit(enc[pool]['p'])+1)
        assert_array_equal(ctx['pools'][pool]['ha'],before[pool])


def test_corrupt_probabilities_or_mismatched_input_family_are_rejected():
    m=module();ctx,e,enc,_=fixture();bad=base_mechanism()
    bad['Q']=np.ones((2,3))
    with pytest.raises(ValueError):m.build_release(ctx,e,enc,'T0_U_unconstrained_a17',mechanism=bad)
    with pytest.raises(ValueError):m.build_release(ctx,e,enc,'Trisk_withhold_0.5',mechanism=base_mechanism())


def test_positive_budget_and_unconstrained_maps_share_saved_empirical_cost(tmp_path):
    m=module();ctx,e,enc,split=fixture();ts=m.estimate_tables(ctx,e,enc,split)
    maps=[]
    for policy,budget in [('C',.002),('U',None)]:
        spec={'input':'T0','policy':policy,'budget':budget,'max_actions':17,
              'configuration':f'T0_{policy}_check_a17'}
        result=m.fit_map(ctx,e,enc,ts,spec,tmp_path/policy)
        assert result['feasible']
        assert result['objective']==pytest.approx(np.sum(ts['T0']['cost']*result['Q']))
        maps.append(result)
        if budget is not None:
            assert max(result['independent_cmi'].values())<=budget+1e-7
        with np.load(tmp_path/policy/'tables.npz') as saved:
            assert_array_equal(saved['support/A/SEX/count'],ts['T0']['support']['A/SEX']['count'])
            assert_allclose(saved['joint/AB/RAC1P/W'],ts['T0']['roles']['AB/RAC1P/W'])
    assert maps[1]['objective']<=maps[0]['objective']+1e-7


def test_empty_refined_child_deploys_supported_sibling_distribution(tmp_path):
    m=module();ctx,e,enc,split=fixture()
    for pool in enc:
        enc[pool]['codes']['Trisk']=2*enc[pool]['codes']['T0']
    ts=m.estimate_tables(ctx,e,enc,split)
    spec={'input':'Trisk','policy':'U','budget':None,'max_actions':17,
          'configuration':'Trisk_U_unconstrained_a17'}
    mechanism=m.fit_map(ctx,e,enc,ts,spec,tmp_path/'absent')
    assert_allclose(mechanism['Q'][[1,3]],mechanism['Q'][[0,2]])
    # A new deployment person can occupy a previously empty child.
    enc['downstream_validation']['codes']['Trisk']+=1
    release=m.build_release(ctx,e,enc,spec['configuration'],mechanism=mechanism)
    assert_allclose(release['downstream_validation']['token_probs'],
                    mechanism['Q'][enc['downstream_validation']['codes']['Trisk']])


def test_all_control_releases_retain_common_calibration_library():
    m=module();ctx,e,enc,split=fixture()
    for values in enc.values():
        values['global_offsets']=np.column_stack((values['actions'][17],values['b']*.8))
    for name in ('H','continuous_task','T0_code','Trisk_code','independent_token'):
        release=m.build_release(ctx,e,enc,name)
        for pool,arm in release.items():
            assert_array_equal(arm['global_offsets'],enc[pool]['global_offsets'])

"""Independent mathematical replay tests; synthetic arrays/artifacts only."""
import importlib
import copy
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy.special import expit,logit


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.math_replay')


def test_information_replay_handles_empty_contexts_and_exact_private_fixture():
    m=module();p=np.zeros((2,3,4));p[0,0,:2]=.25;p[1,0,2:]=.25
    private=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    revealing=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    assert m.conditional_information(p,private)==pytest.approx(0,abs=1e-14)
    assert m.conditional_information(p,revealing)==pytest.approx(np.log(2))
    assert m.conditional_information(p,np.full((4,2),.5))==pytest.approx(0,abs=1e-14)


def test_information_replay_does_not_call_optimizer_information(monkeypatch):
    m=module()
    from experiments.pcrl_task_directed_release_v1 import finite
    monkeypatch.setattr(finite,'cmi',lambda *a:pytest.fail('production cmi called'))
    p=np.zeros((2,1,2));p[0,0,0]=p[1,0,1]=.5
    assert m.conditional_information(p,np.eye(2))==pytest.approx(np.log(2))


class Code:
    def n_states(self,name):return 3 if name=='T0' else 6
    def parents(self,name):return np.arange(3) if name=='T0' else np.repeat(np.arange(3),2)


def prepared():
    n=8;i=np.arange(n);t=i%2;b=np.linspace(.2,.6,n)
    ha=np.column_stack((np.zeros(n),t,np.zeros(n),np.zeros(n)))
    hb=np.column_stack((np.zeros(n),i%3))
    labels={'SEX':t.copy(),'RAC1P':i%3,'same_residence':t.copy()}
    labels['same_residence'][2]=-1;labels['RAC1P'][3]=-1
    data={'x':np.zeros((n,32)),'ha':ha,'hb':hb,'weights':np.arange(1.,n+1),'labels':labels}
    encoder=SimpleNamespace(code=Code(),partitions=SimpleNamespace(centers=np.array([[0.,0.],[1.,0.]]),b_median=.5),
             dictionaries={17:{'offsets':np.array([0.,1.,-1.]),'zero_action':0}})
    enc={'b':b,'p':b,'codes':{'T0':t,'Ttask':2*t,'Trisk':2*t+(i//3)%2},
         'actions':{17:expit(logit(b)[:,None]+np.array([0.,1.,-1.]))}}
    return {'ctx':{'anchor':0,'pools':{'representation_fit':data}},'encoder':encoder,
            'encoded':{'representation_fit':enc},'roles':{'mechanism':np.arange(2,8)}}


def test_table_replay_uses_independent_masks_joint_mass_and_raw_weight_support(monkeypatch):
    m=module();p=prepared()
    from experiments.pcrl_task_directed_release_v1 import mechanisms,finite
    monkeypatch.setattr(mechanisms,'estimate_tables',lambda *a,**k:pytest.fail('production tables called'))
    monkeypatch.setattr(finite,'joint_table',lambda *a,**k:pytest.fail('production joint called'))
    monkeypatch.setattr(finite,'cost_table',lambda *a,**k:pytest.fail('production cost called'))
    table=m.reconstruct_tables(p,'T0',17)
    assert table['state_mass'].tolist()==[3,3,0]
    assert table['support']['A/SEX']['count'].sum()==6
    assert table['support']['A/RAC1P']['count'].sum()==5
    assert table['support']['A/SEX']['sumw2'].sum()==199
    rows=np.arange(3,8);y=p['ctx']['pools']['representation_fit']['labels']['same_residence'][rows]
    g=p['encoded']['representation_fit']['actions'][17][rows]
    ce=-y[:,None]*np.log(g)-(1-y[:,None])*np.log1p(-g)
    np.testing.assert_allclose(table['cost_U'].sum(0),ce.mean(0))
    np.testing.assert_allclose(table['cost_W'].sum(0),(ce*np.arange(4.,9.)[:,None]).sum(0)/30.)


def test_support_ties_absent_parents_and_pool_frequencies_are_replayed():
    m=module();mass=np.array([3.,0.,1.,1.,0.,0.]);parents=np.repeat(np.arange(3),2)
    q=np.array([[.2,.8],[.2,.8],[1.,0.],[0.,1.],[1.,0.],[1.,0.]])
    result=m.support_diagnostics(q,mass,parents,0)
    assert result['maximum_error']<1e-14 and result['unsupported_states']==3
    bad=q.copy();bad[1]=[.8,.2]
    assert m.support_diagnostics(bad,mass,parents,0)['maximum_error']==pytest.approx(.6)
    rates=m.deployment_support(np.array([0,1,4,5]),np.array([1.,2.,3.,4.]),mass,parents)
    assert rates['unsupported_count']==3 and rates['absent_parent_count']==2
    assert rates['unsupported_weight_fraction']==pytest.approx(.9)


def test_refinement_aggregation_and_embedding_compare_actual_deployment_laws():
    m=module();p=prepared();coarse=m.reconstruct_tables(p,'T0',17);fine=m.reconstruct_tables(p,'Trisk',17)
    assert m.refinement_errors(coarse,fine)['maximum_error']<1e-12
    q=np.array([[.2,.3,.5],[.6,.2,.2],[1.,0.,0.]])
    result=m.embedding_errors(coarse,fine,q,p['encoded'],'Trisk',17)
    assert result['maximum_error']<1e-12
    corrupt={**fine,'cost':fine['cost'].copy()};corrupt['cost'][0,0]+=.01
    assert m.refinement_errors(coarse,corrupt)['maximum_error']>=.0099


def test_zero_rank_nullity_and_constant_vector_are_independent():
    m=module();p=np.zeros((2,1,2));p[0,0,0]=p[1,0,1]=.5
    out=m.zero_diagnostics({'role':p},np.tile([.3,.7],(2,1)))
    assert out['rank']==1 and out['nullity']==1
    assert out['constant_vector_residual']<1e-14 and out['normalized_equation_error']<1e-14
    product=np.outer([.3,.7],[.2,.3,.5])[:,None,:]
    assert m.zero_diagnostics({'role':product},np.eye(3))['rank']==0


def test_channel_replay_detects_privacy_violation_and_double_state_mass():
    m=module();p=np.zeros((2,1,2));p[0,0,0]=p[1,0,1]=.5
    cost=np.array([[.2,.5],[.3,.1]])
    table={'cost':cost,'state_mass':np.array([1,1]),'roles':{'A/SEX/U':p}}
    result={'Q':np.eye(2),'objective':.15,'status':'optimal','optimal':True,
            'budget':0.,'constrained_roles':['A/SEX/U'],'rank':1}
    report=m.inspect_channel(table,result,np.arange(2),0)
    assert not report['passed']
    assert report['objective']==pytest.approx(.3)
    assert report['objective_error']==pytest.approx(.15)
    assert report['maximum_cmi_excess']>0.69


def test_ordering_is_qualified_when_solvers_are_approximate():
    m=module();left={'objective':1.,'status':'optimal','optimal':True};right={'objective':.8,'status':'optimal','optimal':True}
    assert m.ordering_check(left,right)['warning']
    right['status']='optimal_inaccurate';right['optimal']=False
    result=m.ordering_check(left,right)
    assert not result['solver_status_supports_optimum_comparison'] and result['ordering_inversion']


def test_resource_and_evaluation_gates_precede_any_artifact_loading(tmp_path,monkeypatch):
    m=module();from experiments.pcrl_task_directed_release_v1 import run
    monkeypatch.setattr(run,'OUT',tmp_path)
    monkeypatch.setattr(run,'prepare',lambda *a,**k:pytest.fail('artifact loaded before gate'))
    with pytest.raises((FileNotFoundError,PermissionError)):m.verify_anchor(0)
    run.atomic(tmp_path/'RESOURCE_SCHEDULE.json',{'frozen_before_comparative_outcomes':True,'config_hash':run.digest(run.configuration())})
    with pytest.raises((FileNotFoundError,PermissionError)):m.verify_anchor(0,include_evaluation=True)


def test_h_parity_is_byte_exact_and_does_not_load_test_member(tmp_path):
    m=module();p=prepared();data=p['ctx']['pools']['representation_fit']
    path=tmp_path/'results/redesign_20260909_acs_fixed_predictions_v1/seed_0';path.mkdir(parents=True)
    np.savez(path/'anchors.npz',**{'representation_fit/A':data['ha'],'representation_fit/B':data['hb'],
                                 'test/A':np.array([np.nan])})
    report=m.h_parity(p['ctx'],tmp_path)
    assert report['representation_fit']['A_byte_identical']
    data['ha']=data['ha'].copy();data['ha'][0,0]=-0.
    assert not m.h_parity(p['ctx'],tmp_path)['representation_fit']['A_byte_identical']


def test_fine_conditioning_reconstructs16_laws_without_changing_costs():
    m=module();p=prepared();primary=m.reconstruct_tables(p,'Trisk',17)
    fine=SimpleNamespace(centers=np.array([[0.,0.],[.25,0.],[.75,0.],[1.,0.]]),b_median=.5)
    expanded=m.reconstruct_tables(p,'Trisk',17,fine_partition=fine)
    assert len(expanded['roles'])==16
    np.testing.assert_array_equal(expanded['cost'],primary['cost'])
    for key,law in primary['roles'].items():
        np.testing.assert_array_equal(expanded['roles'][key],law)
        assert expanded['roles'][key+'/fineC'].shape[1]==2*law.shape[1]


class FrozenEncoder:
    def __init__(self,source,record):
        self.code=source.code;self.partitions=source.partitions
        self.dictionaries=source.dictionaries;self.record=record
    def encode(self,inputs):return copy.deepcopy(self.record)


@pytest.mark.parametrize('corruption',['none','dropped_role','double_mass'])
def test_anchor_replay_writes_only_aggregate_public_and_full_support_private(tmp_path,monkeypatch,corruption):
    m=module();from experiments.pcrl_task_directed_release_v1 import run
    monkeypatch.setattr(run,'OUT',tmp_path)
    run.atomic(tmp_path/'RESOURCE_SCHEDULE.json',{'frozen_before_comparative_outcomes':True,'config_hash':run.digest(run.configuration())})
    p=prepared();enc=p['encoded']['representation_fit'];enc['r']=np.zeros(8);enc['risk']=np.full((8,11),1/11)
    p['ctx']['pools']['representation_fit']['ids']=np.array([f'SENTINEL-PERSON-{i}' for i in range(8)])
    source=FrozenEncoder(p['encoder'],enc);p['encoder']=source
    def prep(anchor,*,allow_fit):
        assert allow_fit is False;return p
    monkeypatch.setattr(run,'prepare',prep)
    monkeypatch.setattr(run,'_prepared_receipt',lambda anchor:{'cache_sha256':'synthetic-prepared'})
    monkeypatch.setattr(run,'_map_receipt',lambda *a:{'coarse_configuration':None})
    base=run.anchor_dir(0);(base/'encoder').mkdir(parents=True)
    joblib.dump(source,base/'encoder/encoder.joblib')
    table=m.reconstruct_tables(p,'T0',17);q=np.tile([1.,0.,0.],(3,1));roles=sorted(table['roles'])
    result={'Q':q,'objective':float(table['cost'][:,0].sum()),'status':'feasible_witness',
            'optimal':False,'feasible':True,'input':'T0','actions':17,'budget':.002,
            'constrained_roles':roles,'rank':m.zero_diagnostics(table['roles'],q)['rank'],
            'tolerances':dict(m.TOLERANCES)}
    if corruption=='dropped_role':result['constrained_roles']=roles[:-1]
    if corruption=='double_mass':result['objective']*=.5
    folder=base/'maps/T0_C_0.002_a17';folder.mkdir(parents=True)
    run.atomic(folder/'ACCEPTED.json',{'synthetic':True});joblib.dump(result,folder/'solution.joblib')
    arrays={key:table[key] for key in ('cost_U','cost_W','cost','state_mass')}
    arrays.update({'joint/'+key:value for key,value in table['roles'].items()})
    arrays.update({'support/'+role+'/'+field:value for role,fields in table['support'].items() for field,value in fields.items()})
    np.savez(folder/'tables.npz',**arrays)
    raw=tmp_path/'private/inputs/results/redesign_20260909_acs_fixed_predictions_v1/seed_0';raw.mkdir(parents=True)
    rows=p['ctx']['pools']['representation_fit']
    np.savez(raw/'anchors.npz',**{'representation_fit/A':rows['ha'],'representation_fit/B':rows['hb']})
    report=m.verify_anchor(0)
    assert report['passed']==(corruption=='none') and report['repairs_performed'] is False
    public=(tmp_path/'MATH_REPLAY_anchor_0.json').read_text()
    assert 'SENTINEL-PERSON' not in public
    if corruption!='dropped_role':
        assert json.loads(public)['maps']['T0_C_0.002_a17']['solver_status']=='feasible_witness'
    files=list((tmp_path/'private/math_replay').rglob('*_support.npz'))
    assert len(files)==(0 if corruption=='dropped_role' else 1)
    if files:
        with np.load(files[0]) as saved:np.testing.assert_array_equal(saved['A/SEX/count'],table['support']['A/SEX']['count'])
    assert not list(tmp_path.glob('*_support.npz'))

"""Prospective branch registration tests: synthetic receipts and rows only."""
import copy
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit,logit

from experiments.pcrl_task_directed_release_v1 import run


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.branches')


@pytest.fixture
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUT',tmp_path)
    run.atomic(tmp_path/'RESOURCE_SCHEDULE.json',{'frozen_before_comparative_outcomes':True,
                                               'config_hash':run.digest(run.configuration())})
    return tmp_path


def seed_gaps(monkeypatch,gaps):
    receipts=[]
    def receipt(anchor,name):
        receipts.append((anchor,name));return {'registry_sha256':'synthetic'}
    monkeypatch.setattr(run,'_audit_receipt',receipt)
    for anchor in (0,1,2):
        for family in ('continuous_task','T0','Ttask','Trisk'):
            name=family if family=='continuous_task' else run.mechanism_id(family,'U',None)
            base=run.anchor_dir(anchor)/'audits'/name
            run.atomic(base/'COMPLETE.json',{'synthetic':True})
            value=.3 if family=='continuous_task' else .3+gaps[family][anchor]
            run.atomic(base/'summary.json',{'utility:A/same_residence':{
                'fixed_decoder':{'balanced':value},'validation':{'balanced':1000.}}})
    return receipts


def test_trigger_uses_direct_decoder_and_equal_three_anchor_family_mean(isolated,monkeypatch):
    b=module();receipts=seed_gaps(monkeypatch,{'T0':[.004,-.002,-.002],
                                            'Ttask':[.002,.002,.002],'Trisk':[0.,0.,0.]})
    evidence=b.action33_trigger()
    assert evidence['triggered'] and evidence['triggering_families']==['Ttask']
    assert evidence['family_gaps']['T0']['mean_balanced_gap']==pytest.approx(0)
    assert evidence['family_gaps']['Ttask']['mean_balanced_gap']==pytest.approx(.002)
    mandatory=[name for anchor,name in receipts if name.startswith('T0_') and '_U_' not in name]
    assert len(mandatory)==18


def test_missing_mandatory_completion_blocks_gap_inspection(isolated,monkeypatch):
    b=module()
    monkeypatch.setattr(run,'_audit_receipt',lambda *a:(_ for _ in ()).throw(FileNotFoundError('mandatory missing')))
    with pytest.raises(FileNotFoundError):b.action33_trigger()


def register(isolated,monkeypatch):
    b=module();seed_gaps(monkeypatch,{'T0':[.002]*3,'Ttask':[0.]*3,'Trisk':[0.]*3})
    return b,b.register_action33()


def test_action33_registration_is_separate_frozen_complete_matrix(isolated,monkeypatch):
    original=run.digest(run.configuration());b,record=register(isolated,monkeypatch)
    assert len(record['maps'])==63 and sum(s['policy']=='U' for s in record['maps'])==9
    assert len(record['controls'])==60 and record['nominal_extra_units']==123
    assert all(s['label_matched'] for s in record['controls'])
    assert {s['max_actions'] for s in record['maps']}=={33}
    assert run.digest(run.configuration())==original
    frozen=(isolated/'EXTRA_CONFIGS.json').read_bytes()
    assert b.register_action33()==record
    assert (isolated/'EXTRA_CONFIGS.json').read_bytes()==frozen
    spec=b.lookup_spec('Trisk_C_0.002_a33',2)
    assert spec['branch']=='A' and spec['extra_registration_sha256']==run.sha(isolated/'EXTRA_CONFIGS.json')
    assert run.map_spec('T0_C_0.002_a17',0)['max_actions']==17
    assert run.map_spec('Trisk_C_0.002_a33',2)['max_actions']==33


def test_no_trigger_does_not_register_branch(isolated,monkeypatch):
    b=module();seed_gaps(monkeypatch,{k:[0.]*3 for k in ('T0','Ttask','Trisk')})
    result=b.register_action33()
    assert not result['registered'] and not (isolated/'EXTRA_CONFIGS.json').exists()


def test_extra_schedule_requires_coarse_witness_units_and_is_frozen(isolated,monkeypatch):
    b,registration=register(isolated,monkeypatch)
    fine='Trisk_C_0.002_a33/anchor_0'
    with pytest.raises(ValueError,match='coarse'):b.freeze_action33_schedule([fine],resource_decision={'capacity':'synthetic'})
    units=['T0_C_0.002_a33/anchor_0',fine]
    record=b.freeze_action33_schedule(units,resource_decision={'capacity':'synthetic'})
    assert record['unit_ids']==units and record['frozen_before_affected_outcomes']
    assert b.require_scheduled(b.lookup_spec('Trisk_C_0.002_a33',0))
    with pytest.raises(ValueError):b.require_scheduled(b.lookup_spec('Ttask_C_0.002_a33',0))
    with pytest.raises(FileExistsError):b.freeze_action33_schedule(units,resource_decision={'capacity':'changed'})


class Code:
    def n_states(self,name):return 2 if name=='T0' else 4
    def parents(self,name):return np.arange(2) if name=='T0' else np.repeat(np.arange(2),2)


class Partitions:
    centers=np.zeros((2,2))
    def assign(self,ha,hb):
        c=np.arange(len(ha))%2
        return c,2*c+(np.arange(len(ha))//2)%2


def prepared():
    n=8;t=np.arange(n)%2;b=np.linspace(.2,.6,n)
    data={'x':np.zeros((n,32)),'ha':np.zeros((n,4)),'hb':np.zeros((n,2)),
          'weights':np.arange(1.,n+1),'labels':{'SEX':t,'RAC1P':np.arange(n)%3,'same_residence':t}}
    enc={'b':b,'p':expit(logit(b)+.2),'codes':{'T0':t,'Ttask':2*t+(np.arange(n)//2)%2,'Trisk':2*t+(np.arange(n)//3)%2},
         'actions':{k:expit(logit(b)[:,None]+offsets) for k,offsets in [(17,np.array([0.,1.])),(33,np.array([0.,1.,-1.]))]}}
    enc['global_offsets']=np.column_stack((enc['actions'][17],enc['actions'][33][:,1:]))
    encoder=SimpleNamespace(code=Code(),partitions=Partitions(),dictionaries={17:{'offsets':[0.,1.],'zero_action':0},33:{'offsets':[0.,1.,-1.],'zero_action':0}})
    p={'ctx':{'anchor':0,'pools':{'representation_fit':data}},'encoder':encoder,
       'encoded':{'representation_fit':enc},'roles':{'mechanism':np.arange(n)},'erasers':{}}
    from experiments.pcrl_task_directed_release_v1.mechanisms import estimate_tables
    p['tables']=estimate_tables(p['ctx'],encoder,p['encoded'],p['roles'])
    return p


def test_expanded_tables_preserve_primary_rows_laws_and_cache(isolated,monkeypatch):
    b,reg=register(isolated,monkeypatch);unit='T0_C_0.002_a33/anchor_0'
    b.freeze_action33_schedule([unit],resource_decision={'capacity':'synthetic'})
    monkeypatch.setattr(run,'_prepared_receipt',lambda anchor:{'cache_sha256':'frozen-primary'})
    p=prepared();primary=copy.deepcopy(p['tables']);spec=b.lookup_spec(unit.split('/')[0],0)
    expanded=b.prepared_for_spec(0,p,spec)
    assert expanded['encoder'] is p['encoder'] and expanded['ctx'] is p['ctx']
    for family in primary:
        assert p['tables'][family]['actions']==17 and expanded['tables'][family]['actions']==33
        np.testing.assert_array_equal(p['tables'][family]['cost'],primary[family]['cost'])
        for role in primary[family]['roles']:
            np.testing.assert_array_equal(expanded['tables'][family]['roles'][role],primary[family]['roles'][role])
    from experiments.pcrl_task_directed_release_v1 import mechanisms
    monkeypatch.setattr(mechanisms,'estimate_tables',lambda *a,**k:pytest.fail('cached tables rebuilt'))
    again=b.prepared_for_spec(0,p,spec,allow_fit=False)
    np.testing.assert_array_equal(again['tables']['T0']['cost'],expanded['tables']['T0']['cost'])


def test_branch_evaluation_never_builds_tables_or_fits_missing_map(isolated,monkeypatch):
    b,reg=register(isolated,monkeypatch)
    b.freeze_action33_schedule(['T0_C_0.002_a33/anchor_0'],resource_decision={'capacity':'synthetic'})
    monkeypatch.setattr(run,'ensure_map',lambda *a,**k:pytest.fail('evaluation fitted'))
    monkeypatch.setattr(b,'prepared_for_spec',lambda *a,**k:pytest.fail('evaluation prepared tables'))
    with pytest.raises(FileNotFoundError):
        run.release_for('T0_C_0.002_a33',{'ctx':{'pools':{'test':{}}}},0,allow_fit=False)


def test_extra_spec_tampering_is_rejected(isolated,monkeypatch):
    b,reg=register(isolated,monkeypatch)
    reg['maps'][0]['budget']=.5;run.atomic(isolated/'EXTRA_CONFIGS.json',reg)
    with pytest.raises(ValueError):b.lookup_spec('T0_C_0.002_a33',0)


def test_matched_controls_route_canonical_names_and_larger_action_dictionary(isolated,monkeypatch):
    b,reg=register(isolated,monkeypatch)
    units=['T0_U_unconstrained_a33/anchor_0','T0_withhold_0.25_a33/anchor_0',
           'T0_rr_0.5_a33/anchor_0','constant_best_a33/anchor_0','independent_token_a33/anchor_0']
    b.freeze_action33_schedule(units,resource_decision={'capacity':'synthetic'})
    seen=[];mechanism={'synthetic':True}
    monkeypatch.setattr(run,'_load_map',lambda anchor,name:seen.append(('map',name)) or mechanism)
    from experiments.pcrl_task_directed_release_v1 import mechanisms
    def build(ctx,encoder,encoded,name,**kwargs):
        seen.append((name,kwargs));return {'sentinel':True}
    monkeypatch.setattr(mechanisms,'build_release',build)
    p={'ctx':{'pools':{'test':{}}},'encoder':None,'encoded':{},'erasers':{}}
    for unit in units[1:]:
        name=unit.split('/')[0]
        assert run.release_for(name,p,0,allow_fit=False)=={'sentinel':True}
        canonical,kwargs=seen[-1]
        assert canonical==name[:-4] and kwargs['actions']==33
        assert kwargs['mechanism'] is (None if name=='independent_token_a33' else mechanism)
    assert run.map_spec('T0_withhold_0.25_a33',0) is None
    assert b.lookup_release('T0_withhold_0.25_a33',0)['baseline_family']=='withhold'


def test_real_finite_branch_maps_embed_matching33_coarse_without_primary_mutation(isolated,monkeypatch):
    b,reg=register(isolated,monkeypatch)
    units=['T0_U_unconstrained_a33/anchor_0','Trisk_U_unconstrained_a33/anchor_0',
           'Trisk_withhold_0.5_a33/anchor_0']
    b.freeze_action33_schedule(units,resource_decision={'capacity':'synthetic'})
    monkeypatch.setattr(run,'_prepared_receipt',lambda anchor:{'cache_sha256':'frozen-primary'})
    p=prepared();before=p['tables']['T0']['cost'].copy()
    coarse=run.ensure_map(0,'T0_U_unconstrained_a33',p)
    fine=run.ensure_map(0,'Trisk_U_unconstrained_a33',p)
    assert fine['feasible'] and fine['actions']==33 and fine['Q'].shape==(4,3)
    assert fine['embedding']['maximum_error']<=1e-12
    assert fine['objective']<=coarse['objective']+1e-7
    np.testing.assert_array_equal(p['tables']['T0']['cost'],before)
    release=run.release_for('Trisk_withhold_0.5_a33',p,0)
    token=release['representation_fit']['token_probs']
    assert token.shape==(8,4)
    np.testing.assert_allclose(token[:,-1],.5)
    np.testing.assert_array_equal(release['representation_fit']['global_offsets'],
                                  p['encoded']['representation_fit']['global_offsets'])

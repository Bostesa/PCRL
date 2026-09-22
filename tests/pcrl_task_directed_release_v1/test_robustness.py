"""Finer-conditioning branch tests use synthetic people and receipt files."""
import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit,logit

from experiments.pcrl_task_directed_release_v1 import run
from experiments.pcrl_task_directed_release_v1.encoding import ServicePartitions


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.robustness')


@pytest.fixture
def isolated(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUT',tmp_path)
    run.atomic(tmp_path/'RESOURCE_SCHEDULE.json',{'frozen_before_comparative_outcomes':True,
                                               'config_hash':run.digest(run.configuration())})
    return tmp_path


def seed_trigger(monkeypatch,budget=.002,weight='weighted',recovery=.005):
    monkeypatch.setattr(run,'_audit_receipt',lambda *a:{'registry_sha256':'synthetic-audit'})
    monkeypatch.setattr(run,'_map_receipt',lambda *a:{'sha256':'synthetic-map'})
    name=run.mechanism_id('T0','L',budget)
    base=run.anchor_dir(0)
    score={f'attack:{view}/{target}':{'validation':{'unweighted':.5,'weighted':.5}}
           for view in ('A','AB') for target in ('SEX','RAC1P')}
    run.atomic(base/'audits/H/summary.json',score)
    candidate=copy.deepcopy(score);candidate['attack:AB/SEX']['validation'][weight]-=recovery
    run.atomic(base/'audits'/name/'summary.json',candidate)
    run.atomic(base/'audits'/name/'COMPLETE.json',{'synthetic':True})
    run.atomic(base/'maps'/name/'ACCEPTED.json',{'synthetic':True})
    roles=[f'A/{target}/{weight}' for target in ('SEX','RAC1P') for weight in ('U','W')]
    run.atomic(base/'maps'/name/'metadata.json',{'feasible':True,'budget':budget,
        'constrained_roles':roles,'independent_cmi':{role:budget for role in roles}})
    return name


def test_trigger_retains_budget_threshold_and_fixed_target_family(isolated,monkeypatch):
    c=module();seed_trigger(monkeypatch)
    trigger=c.robustness_trigger()
    assert trigger['triggered'] and trigger['predetermined_input']=='Trisk'
    event=trigger['violations'][0]
    assert event['role']=='AB/SEX' and event['weighting']=='weighted'
    assert event['threshold']==.004 and event['recovery']==pytest.approx(.005)
    assert event['configuration'].startswith('T0_')


def test_infeasible_or_subthreshold_channels_do_not_trigger(isolated,monkeypatch):
    c=module();name=seed_trigger(monkeypatch,recovery=.003)
    assert not c.robustness_trigger()['triggered']
    file=run.anchor_dir(0)/'maps'/name/'metadata.json'
    run.atomic(file,{'feasible':False,'budget':.002})
    assert not c.robustness_trigger()['triggered']


def register(isolated,monkeypatch):
    c=module();seed_trigger(monkeypatch)
    return c,c.register_robustness()


def test_registry_has18_matched_fine_constraints_specs_without_changing_A(isolated,monkeypatch):
    (isolated/'EXTRA_CONFIGS.json').write_bytes(b'untouched branch A sentinel')
    config_hash=run.digest(run.configuration());c,record=register(isolated,monkeypatch)
    assert len(record['maps'])==18 and record['controls']==[]
    assert {(s['input'],s['max_actions']) for s in record['maps']}=={('Trisk',17)}
    spec=c.lookup_spec('Trisk_C_0.002_a17_fineC',1)
    assert spec['paired_local_configuration']=='Trisk_L_0.002_a17_fineC'
    assert spec['conditioning_family']=='primary_intersect_fineC4'
    assert run.map_spec(spec['configuration'],1)==spec
    assert run.digest(run.configuration())==config_hash
    assert (isolated/'EXTRA_CONFIGS.json').read_bytes()==b'untouched branch A sentinel'
    assert c.register_robustness()==record


def test_robustness_schedule_freezes_matched_blocks_before_fits(isolated,monkeypatch):
    c,reg=register(isolated,monkeypatch)
    with pytest.raises(ValueError,match='matched'):
        c.freeze_robustness_schedule([reg['maps'][0]['id']],resource_decision={'capacity':'synthetic'})
    schedule=c.freeze_robustness_schedule(resource_decision={'capacity':'synthetic'})
    assert len(schedule['unit_ids'])==18 and schedule['frozen_before_affected_outcomes']
    assert c.require_scheduled(c.lookup_spec('Trisk_L_0.002_a17_fineC',0))
    with pytest.raises(FileExistsError):c.freeze_robustness_schedule(resource_decision={'capacity':'changed'})


class Code:
    def n_states(self,name):return 2 if name=='T0' else 4
    def parents(self,name):return np.arange(2) if name=='T0' else np.repeat(np.arange(2),2)


def prepared():
    n=24;i=np.arange(n);t=i%2;b=np.linspace(.2,.6,n)
    ha=np.column_stack((np.full(n,.1),(i%4)/4,np.full(n,.3),(i//4%4)/4))
    hb=np.column_stack((np.full(n,.3),(i%3)/3))
    roles={'teacher_fit':np.arange(8),'teacher_internal_validation':np.arange(8,16),'mechanism':np.arange(16,24)}
    partition=ServicePartitions.fit(ha[:16],hb[:16],0)
    ha[16:,1]+=4;ha[16:,3]+=4
    encoder=SimpleNamespace(code=Code(),partitions=partition,dictionaries={17:{'offsets':np.array([0.,1.,-1.]),'zero_action':0}})
    data={'x':np.zeros((n,32)),'ha':ha,'hb':hb,'raw_rows':i,'weights':np.arange(1.,n+1),
          'labels':{'SEX':t,'RAC1P':i%3,'same_residence':i//2%2}}
    enc={'b':b,'p':expit(logit(b)+.2),'codes':{'T0':t,'Ttask':2*t+(i//2)%2,'Trisk':2*t+(i//3)%2},
         'actions':{17:expit(logit(b)[:,None]+np.array([0.,1.,-1.]))}}
    p={'ctx':{'anchor':0,'pools':{'representation_fit':data}},'encoder':encoder,
       'encoded':{'representation_fit':enc},'roles':roles,'erasers':{}}
    from experiments.pcrl_task_directed_release_v1.mechanisms import estimate_tables
    p['tables']=estimate_tables(p['ctx'],encoder,p['encoded'],roles)
    return p


def test_four_cell_fit_uses_teacher_pool_only_and_preserves_original_laws(isolated,monkeypatch):
    c,reg=register(isolated,monkeypatch);c.freeze_robustness_schedule(resource_decision={'capacity':'synthetic'})
    monkeypatch.setattr(run,'_prepared_receipt',lambda a:{'cache_sha256':'primary-frozen'})
    p=prepared();original=copy.deepcopy(p['tables']);centers=p['encoder'].partitions.centers.copy()
    # Mechanism H lies far outside teacher support in this fixture; fitting
    # centers must equal the teacher/codebook-only four-cell partition.
    expected=ServicePartitions.fit(p['ctx']['pools']['representation_fit']['ha'][:16],
                                   p['ctx']['pools']['representation_fit']['hb'][:16],0,n_local=4)
    q=c.prepared_for_spec(0,p,c.lookup_spec('Trisk_C_0.002_a17_fineC',0))
    assert q['encoder'] is not p['encoder']
    np.testing.assert_array_equal(q['encoder'].partitions.centers,expected.centers)
    assert q['encoder'].partitions.b_median==p['encoder'].partitions.b_median
    np.testing.assert_array_equal(p['encoder'].partitions.centers,centers)
    for family in original:
        assert len(q['tables'][family]['roles'])==16
        np.testing.assert_array_equal(q['tables'][family]['cost'],original[family]['cost'])
        for role,law in original[family]['roles'].items():
            np.testing.assert_array_equal(q['tables'][family]['roles'][role],law)
            assert q['tables'][family]['roles'][role+'/fineC'].shape[1]==2*law.shape[1]
    cached=c.prepared_for_spec(0,p,c.lookup_spec('Trisk_L_0.0005_a17_fineC',0),allow_fit=False)
    assert len(cached['tables']['Trisk']['roles'])==16


def test_runner_uses_augmented_constraints_without_original_T0_witness(isolated,monkeypatch):
    c,reg=register(isolated,monkeypatch);c.freeze_robustness_schedule(resource_decision={'capacity':'synthetic'})
    monkeypatch.setattr(run,'_prepared_receipt',lambda a:{'cache_sha256':'primary-frozen'})
    # Restore real _map_receipt for independently verified acceptance.
    monkeypatch.undo()
    monkeypatch.setattr(run,'OUT',isolated)
    monkeypatch.setattr(run,'_prepared_receipt',lambda a:{'cache_sha256':'primary-frozen'})
    p=prepared()
    coarse_path=run.anchor_dir(0)/'maps/T0_L_0.002_a17'
    coarse_before=(coarse_path/'ACCEPTED.json').read_bytes()
    result=run.ensure_map(0,'Trisk_L_0.002_a17_fineC',p)
    assert result['feasible'] and len(result['constrained_roles'])==8
    assert result['embedding'] is None
    assert (coarse_path/'ACCEPTED.json').read_bytes()==coarse_before
    assert not (coarse_path/'solution.joblib').exists()
    coalition=run.ensure_map(0,'Trisk_C_0.002_a17_fineC',p)
    assert coalition['feasible'] and len(coalition['constrained_roles'])==16
    assert coalition['objective']>=result['objective']-1e-7
    assert coalition['actions']==17


def test_robust_evaluation_is_reuse_only(isolated,monkeypatch):
    c,reg=register(isolated,monkeypatch);c.freeze_robustness_schedule(resource_decision={'capacity':'synthetic'})
    monkeypatch.setattr(run,'ensure_map',lambda *a,**k:pytest.fail('evaluation fitted Q'))
    monkeypatch.setattr(c,'prepared_for_spec',lambda *a,**k:pytest.fail('evaluation fitted conditioning'))
    with pytest.raises(FileNotFoundError):
        run.release_for('Trisk_C_0.002_a17_fineC',{'ctx':{'pools':{'test':{}}}},0,allow_fit=False)

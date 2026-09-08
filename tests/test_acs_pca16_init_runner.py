"""Artificial complete PCA16-init runner check; no actual ACS outcomes."""
import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments import run_acs_bottleneck as runner
from experiments.acs_bottleneck_training import BottleneckModel
from experiments.acs_transfer_data import array_hash, sha_file
from tests.test_acs_protection_runner import artificial_parent, one_thread


def test_new_snapshots_remain_fixed_and_utility_only_with_evaluation_sealed(
        artificial_parent,monkeypatch,tmp_path):
    fixture=artificial_parent;frame,pools=fixture['frame'],fixture['pools']
    out=tmp_path/'new';out.mkdir();directory=out/'seed_97'
    cfg={'parent_results':'parent','head_budget':16,'attacker_budget':32,'head_mlp_epochs':40,
         'catchup_epochs':120,'mapper_initialization':'pca16','bottleneck_reference_results':'old_bottleneck',
         'pca16_reference_results':'old_pca16','evaluation_status':'artificial fixture only'}
    original_cfg=copy.deepcopy(cfg)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'load_cohort',lambda *a:(frame.copy(),{'artificial':True}))
    monkeypatch.setattr(runner,'split_households',lambda *a:pools)
    initial_source=runner._state_hash(fixture['model']);pairs=[];accesses=[]
    class FixedReferences:
        def __init__(self,*args):
            self.directory=tmp_path/'fixed_reference';self.directory.mkdir()
            self.source=SimpleNamespace(used_files={},selection=runner.read(fixture['parent']/'seed_97/source_selection.json'))
            self.releases={n:{p:outputs[n.removesuffix('_leace')].copy() for p,outputs in fixture['all_releases'].items() if p!='test'}
                           for n in runner.REFERENCES}
            self.previous={'support_by_pool':{p:{'raw_row_sha256':array_hash(frame.iloc[i]._raw_row.to_numpy())} for p,i in pools.items()},
                'release_metadata':{n:{'fixture':True} for n in runner.REFERENCES},'raw_metrics':[]}
            runner.write_json(self.directory/'metrics.json',self.previous)
            self.selection={k:{} for k in ('fitting_records','head_selections','family_selections','auroc_selections')}
            self.used_files={};self.checked_candidates=0;self.initial=self.fingerprint()
        def fingerprint(self):
            return {'arrays':runner.frozen_digest(self.releases),'source':runner._state_hash(fixture['model'])}
        def verify_reuse(self,*args):
            assert (directory/'release_freeze.json').exists()
        def evaluation_releases(self,part):
            selection=runner.read(directory/'selection_before_test.json')
            assert len(selection['head_selections'])==24
            assert len(list((directory/'fitted').rglob('selection.json')))==24
            assert not any(k.startswith(('audit/I/','audit/W/')) for k in selection['head_selections'])
            assert runner.require_test_selection(directory,list(selection['head_selections']))
            np.testing.assert_array_equal(part._raw_row.to_numpy(),pools['test'])
            accesses.append('evaluation')
            return {n:fixture['all_releases']['test'][n.removesuffix('_leace')] for n in runner.REFERENCES}
    monkeypatch.setattr(runner,'SavedReferences',FixedReferences)
    real_train,real_history,real_tasks=runner.train_pair,runner.initialization_history,runner.binary_tasks
    def fit_pair(*args,**kwargs):
        assert not accesses and kwargs['initialization']=='pca16'
        assert not (directory/'release_freeze.json').exists()
        result=real_train(*args,**kwargs);pairs.append(result)
        return result
    def fixture_history(config,seed,pair,releases):
        assert not accesses and set(releases)=={'I','W','C_init','D_init'}
        # Artificial saved references exercise the actual identity/history code.
        # Constructor follows the original seed before overwriting its mapper.
        pca=fixture['all_releases']['representation_fit']['E_pca'].astype(np.float64)
        mean,std=pca.mean(0),pca.std(0);scale=np.where(std>1e-12,std,1.)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1270000+100*seed)
            old_initial=BottleneckModel(mean,scale)
        for config_key in ('bottleneck_reference_results','pca16_reference_results'):
            historical=tmp_path/config[config_key]/f'seed_{seed}';historical.mkdir(parents=True)
            np.savez_compressed(historical/'predictions.npz',fixture=np.zeros(1))
            runner.write_json(historical/'metrics.json',{'raw_metrics':[]})
            selected={k:{} for k in ('head_selections','family_selections','auroc_selections','fitting_records')}
            def indices(labels,pool,base,limit):
                result={}
                for j,(target,y) in enumerate(labels.items()):
                    i=runner.subset_indices(y,limit,base+100*seed+j)
                    result[target]={'n':len(i),'pool_indices_sha256':array_hash(i),
                        'raw_rows_sha256':array_hash(frame.iloc[pools[pool][i]]._raw_row.to_numpy())}
                return result
            edges=[10000,20000,30000,40000,50000,60000,70000]
            selected['task_fit_indices']=indices(real_tasks(frame.iloc[pools['downstream_fit']],edges),'downstream_fit',1230000,16)
            selected['audit_fit_indices']=indices(runner.audit_labels(frame.iloc[pools['attacker_fit']]),'attacker_fit',1240000,32)
            runner.write_json(historical/'selection_before_test.json',selected)
            if config_key=='bottleneck_reference_results':
                (historical/'training').mkdir()
                torch.save({'model_state':old_initial.state_dict()},historical/'training/initialization.pt')
                runner.write_json(historical/'training/training.json',pair['metadata'])
            else:
                np.savez_compressed(historical/'release_PCA16.npz',**{p:a['E_pca'][:,:16] for p,a in fixture['all_releases'].items()})
            runner.write_json(historical/'local_artifacts.json',[{'path':str(p.relative_to(tmp_path)),
                'sha256':sha_file(p)} for p in historical.rglob('*') if p.suffix in ('.npz','.pt')])
        return real_history(config,seed,pair,releases)
    def guarded_tasks(part,edges):
        assert (directory/'release_freeze.json').exists()
        frozen=runner.read(directory/'release_freeze.json')
        assert frozen['scientific_training_completed_utc']<=frozen['created_utc']
        accesses.append('reserved')
        return real_tasks(part,edges)
    monkeypatch.setattr(runner,'train_pair',fit_pair)
    monkeypatch.setattr(runner,'initialization_history',fixture_history)
    monkeypatch.setattr(runner,'binary_tasks',guarded_tasks)
    result=runner.run_seed(out,cfg,97,miniature=True)
    assert cfg==original_cfg and len(pairs)==1
    assert runner._state_hash(fixture['model'])==initial_source
    assert len(result['raw_metrics'])==68
    assert all(v for k,v in result['integrity'].items() if k.endswith('_unchanged'))
    pair=pairs[0];frozen=runner.read(directory/'release_freeze.json')
    for name in ('I','W'):
        snapshot=pair['snapshots'][name]
        assert not snapshot.training and not any(p.requires_grad for p in snapshot.parameters())
        assert runner._state_hash(snapshot)==frozen['learned_state'][name]['model']
        assert {r['role'] for r in result['raw_metrics'] if r['release']==name}=={'transfer'}
    with np.load(directory/'predictions.npz') as p:assert len(p.files)==136
    with np.load(directory/'release_I.npz') as values:
        for pool in pools:np.testing.assert_allclose(values[pool],fixture['all_releases'][pool]['E_pca'][:,:16],atol=1e-5,rtol=1e-5)
    assert runner.read(directory/'evaluation_parity.json')['selected_before_access']==result['integrity']['selection_sha256']
    assert all(frozen['initialization_history']['original_initial_nonmapper_exact'].values())
    assert (directory/'fitted/transfer/I/same_residence/selection.json').exists()
    assert (directory/'fitted/audit/C_init/RAC1P/catchup/model.pt').exists()

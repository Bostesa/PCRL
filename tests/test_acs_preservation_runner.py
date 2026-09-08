"""Artificial complete unit: reserved-label gate and frozen final audit boundary."""
from types import SimpleNamespace
import numpy as np
from experiments import run_acs_preservation as runner
from tests.test_acs_protection_runner import artificial_parent, one_thread


def test_complete_unit_seals_reserved_labels_until_all_four_final_maps(artificial_parent,monkeypatch,tmp_path):
    f=artificial_parent;frame,pools=f['frame'],f['pools'];out=tmp_path/'study';out.mkdir()
    directory=out/'beta_0p1/seed_97';events=[]
    cfg={'parent_results':'parent','init_reference_results':'historical','pca16_reference_results':'pca16',
         'head_budget':16,'attacker_budget':32,'head_mlp_epochs':40,'catchup_epochs':120,
         'evaluation_status':'artificial fixture only'}
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'load_cohort',lambda *a:(frame.copy(),{'artificial':True}))
    monkeypatch.setattr(runner,'split_households',lambda *a:pools)
    fit=f['all_releases']['representation_fit']['E_pca'].astype(float)
    historical=tmp_path/'historical/seed_97/training';historical.mkdir(parents=True)
    runner.write_json(historical/'training.json',{'preprocessing':{'mean':fit.mean(0).tolist(),'scale':np.where(fit.std(0)>1e-12,fit.std(0),1.).tolist()}})
    p16=tmp_path/'pca16/seed_97';p16.mkdir(parents=True)
    np.savez_compressed(p16/'release_PCA16.npz',**{p:v['E_pca'][:,:16] for p,v in f['all_releases'].items()})
    class References:
        def __init__(self,*args):
            self.directory=tmp_path/'refs';self.directory.mkdir()
            self.source=SimpleNamespace(used_files={},selection=runner.read(f['parent']/'seed_97/source_selection.json'))
            self.releases={n:{p:v[n.removesuffix('_leace')].copy() for p,v in f['all_releases'].items() if p!='test'} for n in runner.REFERENCES}
            self.previous={'support_by_pool':{p:{'raw_row_sha256':runner.array_hash(frame.iloc[i]._raw_row.to_numpy())} for p,i in pools.items()},'release_metadata':{n:{} for n in runner.REFERENCES},'raw_metrics':[]}
            runner.write_json(self.directory/'metrics.json',self.previous)
            self.selection={k:{} for k in ('fitting_records','head_selections','family_selections','auroc_selections')}
            self.used_files={};self.checked_candidates=0;self.initial=self.fingerprint()
        def fingerprint(self):return runner.frozen_digest(self.releases)
        def verify_reuse(self,*args):assert (directory/'release_freeze.json').exists()
        def evaluation_releases(self,part):
            assert len(list((directory/'training').glob('*/final.pt')))==4
            assert len(list((directory/'fitted').rglob('selection.json')))==33
            selected=runner.read(directory/'selection_before_test.json')
            assert not any(k.startswith(('audit/I/','audit/W/','transfer/I/')) for k in selected['head_selections'])
            assert (directory/'affine_freeze.json').exists()
            events.append('development')
            return {n:f['all_releases']['test'][n.removesuffix('_leace')] for n in runner.REFERENCES}
    monkeypatch.setattr(runner,'SavedReferences',References)
    def history(cfg,seed,pair,releases):
        assert len(pair['arms'])==4 and not events
        return {'raw':[],'records':{},'used_files':{},'identities':{}}
    monkeypatch.setattr(runner,'preservation_history',history)
    actual=runner.binary_tasks
    def guarded(part,edges):
        assert (directory/'release_freeze.json').exists()
        assert len(list((directory/'training').glob('*/final.pt')))==4
        events.append('reserved')
        return actual(part,edges)
    monkeypatch.setattr(runner,'binary_tasks',guarded)
    result=runner.run_unit(out,cfg,.1,97,miniature=True)
    assert len(result['raw_metrics'])==106
    assert all(v for k,v in result['integrity'].items() if k.endswith('_unchanged'))
    assert events.index('development')>=2
    diagnostics=runner.read(directory/'preservation_diagnostics.json')
    assert len(diagnostics['snapshots'])==6
    assert all('development_evaluation' in v for v in diagnostics['snapshots'].values())


def test_separately_named_config_does_not_relax_old_runner():
    from experiments.run_acs_bottleneck import check_config as old_check
    import pytest
    cfg=runner.read(runner.DEFAULT_OUT/'config.json')
    runner.check_config(cfg)
    with pytest.raises(AssertionError):old_check(cfg)

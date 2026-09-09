"""New orchestration boundaries, identities and immutable completion fixtures."""
import copy,inspect,json,tempfile
from pathlib import Path
import pytest
from experiments import run_acs_source_guard as run


def test_matrix_shared_T_and_exact_historical_aliases():
    cfg=run.read(run.DEFAULT_OUT/'config.json');run.check_config(cfg);m=run.matrix(cfg,run.DEFAULT_OUT)
    assert len(m)==48 and sum(not e['reused'] for e in m)==24
    assert len({e['forward_alias'] for e in m if not e['reused']})==21
    for seed in (0,1,2):
        t=[e for e in m if e['seed']==seed and e['arm']=='T']
        assert len(t)==2 and len({e['forward_alias'] for e in t})==1
        assert {e['interface'] for e in t}=={'F','P'}
        assert all(e['family']=='T' and e['beta']==0 and not e['guarded'] for e in t)
    assert len(run.ordered_training(run.DEFAULT_OUT))==21
    assert [e['condition'] for e in run.ordered_training(run.DEFAULT_OUT)[:7]]==['F_T','F_G_J','F_G_L025','F_G_L20','P_G_J','P_G_L025','P_G_L20']


def test_config_rejects_recipe_budget_loss_mutation():
    cfg=run.read(run.DEFAULT_OUT/'config.json')
    for key,value in [('audit_mlp_epochs',360),('extended_audit_epochs',120),('maximum_scientific_seconds',6000),('new_unique_forward_continuations',24),('condition_order',cfg['condition_order'][::-1])]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(AssertionError):run.check_config(bad)
    bad=copy.deepcopy(cfg);bad['objectives']['F_T']['individual']=-.1
    with pytest.raises(AssertionError):run.check_config(bad)
    bad=copy.deepcopy(cfg);bad['objectives']['F_G_L20']['extra_local']=-.1
    with pytest.raises(AssertionError):run.check_config(bad)


def test_reserved_gate_and_no_training_after_freeze(tmp_path,monkeypatch):
    cfg=run.read(run.DEFAULT_OUT/'config.json');e=run.ordered_new(run.DEFAULT_OUT)[0]
    monkeypatch.setattr(run,'freeze_all_releases',lambda out:(_ for _ in ()).throw(RuntimeError('all new releases not frozen')))
    monkeypatch.setattr(run,'load_context',lambda *args:pytest.fail('data reached before freeze'))
    with pytest.raises(RuntimeError,match='not frozen'):run.evaluate_unit(tmp_path,cfg,e)
    (tmp_path/'RELEASE_MANIFEST.json').write_text('{}')
    with pytest.raises(AssertionError,match='permanently closed'):run.train_unit(tmp_path,cfg,e)
    code=inspect.getsource(run.train_unit)
    assert 'all_labels' not in code and 'source_binaries' in code and 'audit_labels' in code


def test_completion_hash_and_no_overwrite():
    with tempfile.TemporaryDirectory(dir=run.ROOT) as folder:
        p=Path(folder);(p/'a.json').write_text('{}')
        run.complete_record(p,'complete.json',{'condition':'disposable'})
        run.verify_complete(p/'complete.json')
        with pytest.raises(FileExistsError):run.complete_record(p,'complete.json',{})
        (p/'a.json').write_text('{"changed":1}')
        with pytest.raises(AssertionError):run.verify_complete(p/'complete.json')

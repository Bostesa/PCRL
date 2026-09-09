"""Focused orchestration risks; no historical/scientific fitting."""
import copy
import inspect
from pathlib import Path
import pytest
from experiments import run_acs_coalition_strength as run


def config():return run.read(run.DEFAULT_OUT/'config.json')

def test_exact_unique_reuse_and_new_matrix():
    cfg=config();run.check_config(cfg);m=run.matrix(cfg,run.DEFAULT_OUT)
    assert len(m)==54 and sum(e['reused'] for e in m)==18
    assert len({(e['seed'],e['condition']) for e in m})==54
    assert sum(e['beta']==0 for e in m)==6
    assert sum(e['beta']==.1 for e in m)==12
    assert all(e['historical_fork'].endswith(e['interface']+'_I/fork.pt') for e in m)
    assert len(run.ordered_new(run.DEFAULT_OUT))==36


def test_fixed_contract_and_actual_nested_budget():
    cfg=config()
    for key,value in [('audit_mlp_epochs',360),('catchup_epochs',360),('extended_audit_epochs',120),('ordinary_individual_coefficient',.2),('new_betas',[.025,.05,.1,.2])]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(AssertionError):run.check_config(bad)
    bad=copy.deepcopy(cfg);bad['objectives']['F_Iplus_b0p025']['extra_local']=-.0125
    with pytest.raises(AssertionError):run.check_config(bad)


def test_reserved_access_has_global_release_gate(tmp_path,monkeypatch):
    cfg=config();e=run.matrix(cfg,run.DEFAULT_OUT)[1]
    monkeypatch.setattr(run,'freeze_all_releases',lambda out:(_ for _ in ()).throw(RuntimeError('54 not frozen')))
    monkeypatch.setattr(run,'load_context',lambda *a:pytest.fail('Data loading before global freeze'))
    with pytest.raises(RuntimeError,match='54 not frozen'):run.evaluate_unit(tmp_path,cfg,e)
    source=inspect.getsource(run.train_unit)
    assert 'all_labels' not in source and 'binary_tasks' not in source
    assert 'source_binaries' in source and 'audit_labels' in source


def test_completed_units_cannot_silently_change(tmp_path):
    dest=tmp_path/'unit';dest.mkdir();artifact=dest/'state.json';artifact.write_text('{}')
    # Complete record paths are repository-relative, so use a local fixture beneath ROOT.
    import tempfile
    with tempfile.TemporaryDirectory(dir=run.ROOT) as d:
        local=Path(d);p=local/'value.json';p.write_text('{}')
        run.complete_record(local,'complete.json',{'training_complete':True})
        run.verify_complete(local/'complete.json')
        p.write_text('{"changed":true}')
        with pytest.raises(AssertionError):run.verify_complete(local/'complete.json')

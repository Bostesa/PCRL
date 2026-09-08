"""Focused new-study orchestration boundaries; no ACS fitting."""
import ast
import copy
from pathlib import Path
import pytest
from experiments.run_acs_selective import check_config, read, DEFAULT_OUT, unit_name, unit_path, fit_static_auditors


def test_frozen_matrix_order_and_historical_recipe():
    cfg=read(DEFAULT_OUT/'config.json');check_config(cfg)
    assert len(cfg['execution_order'])==18
    assert sum(u[0]!='static' for u in cfg['execution_order'])==15
    assert ['R',.1,0] not in cfg['execution_order']
    assert unit_path(DEFAULT_OUT,['E',.1,2])==DEFAULT_OUT/'E_rho0p1/seed_2'
    for field,value in [('head_budget',999),('extended_audit_epochs',720),('preservation_betas',[.1,1.]),('permutation_seed_base',1)]:
        altered=copy.deepcopy(cfg);altered[field]=value
        with pytest.raises(AssertionError): check_config(altered)


def test_static_auditors_do_not_fit_duplicate_mlp(monkeypatch):
    import experiments.run_acs_selective as runner
    calls=[]
    class Candidate:
        metadata={}
    def fit(*args, families, budget):
        calls.append((families,budget['histgb']['min_samples_leaf']))
        return {'candidates':{families[0]:Candidate()}}
    monkeypatch.setattr(runner,'fit_candidates',fit)
    result=fit_static_auditors(None,None,None,None,2,3)
    assert list(result)==['logistic','hist_gb_20','hist_gb_5']
    assert calls==[(('logistic',),20),(('histgb',),20),(('histgb',),5)]


def test_reserved_targets_and_development_follow_frozen_units():
    source=Path('experiments/run_acs_selective.py').read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_unit')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
    def named(name):
        return sorted(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id==name)
    freezes=[n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='write_json' and n.args and any(isinstance(v,ast.Constant) and v.value=='release_freeze.json' for v in ast.walk(n.args[0]))]
    assert named('train_selective_unit')[0]<freezes[0]<named('binary_tasks')[0]
    assert named('fit_snapshots')[0]<named('binary_tasks')[0]
    selection=[n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='write_json' and n.args and any(isinstance(v,ast.Constant) and v.value=='selection_before_test.json' for v in ast.walk(n.args[0]))]
    evaluation=[n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='evaluation_releases']
    assert selection[0]<evaluation[0]

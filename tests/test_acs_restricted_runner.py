import copy
import pytest
from experiments.run_acs_restricted import ROOT, read, check_config, unit_path


def test_fixed_matrix_and_historical_contract():
    cfg=read(ROOT/'results/redesign_20260908_acs_restricted_inputs_v1/config.json')
    check_config(cfg)
    assert len(cfg['execution_order'])==18
    assert sum(1 if u[1]=='bank' else 2 for u in cfg['execution_order'])==30
    for key,value in [('head_budget',1024),('extended_audit_epochs',120),('source_bank_preservation_beta',1.),('orthogonal_seed_base',20260910),('mapper_initialization','pca16')]:
        bad=copy.deepcopy(cfg);bad[key]=value
        with pytest.raises(AssertionError):check_config(bad)
    bad=copy.deepcopy(cfg);bad['training']['protection_weight']=.2
    with pytest.raises(AssertionError):check_config(bad)


def test_distinct_units_and_no_historical_output_paths(tmp_path):
    cfg=read(ROOT/'results/redesign_20260908_acs_restricted_inputs_v1/config.json')
    paths=[unit_path(tmp_path,u) for u in cfg['execution_order']]
    assert len(set(paths))==18
    assert paths[0]==tmp_path/'E_F/seed_0'
    assert paths[2]==tmp_path/'E_bank/seed_0'

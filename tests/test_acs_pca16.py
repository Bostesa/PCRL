"""Exact fixed-coordinate PCA16 and fitting/evaluation boundary checks."""
import copy
import inspect

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from experiments import run_acs_pca16 as runner
from experiments.acs_transfer_data import array_hash, require_test_selection, write_json
from experiments.acs_transfer_heads import _state_hash


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_slice_preserves_original_component_order_and_cannot_be_mutated():
    x = np.arange(7*32, dtype=np.float32).reshape(7,32)
    original = x.copy()
    released = runner.slice_pca16(x)
    np.testing.assert_array_equal(released, original[:,:16])
    assert released.dtype == np.float32 and released.shape == (7,16)
    assert not released.flags.writeable and not np.shares_memory(x,released)
    with pytest.raises(ValueError):
        released.setflags(write=True)
    with pytest.raises(ValueError):
        released[0,0] = 100
    x[:] = -999
    np.testing.assert_array_equal(released, original[:,:16])


@pytest.mark.parametrize('invalid', [np.zeros((5,16),dtype=np.float32),
    np.zeros((5,32),dtype=np.float64), np.zeros(32,dtype=np.float32),
    np.full((5,32),np.nan,dtype=np.float32)])
def test_slice_refuses_changed_schema_dtype_and_nonfinite_input(invalid):
    with pytest.raises((ValueError,TypeError,AssertionError)):
        runner.slice_pca16(invalid)


@pytest.fixture
def fitting_data():
    rng = np.random.default_rng(7115)
    arrays = {}
    for pool in ('downstream_fit','downstream_validation','attacker_fit','attacker_validation'):
        x = rng.normal(size=(48,32)).astype(np.float32)
        if pool.endswith('validation'):
            x += 7
        x[:,15] = 5 if pool.endswith('fit') else 25
        arrays[pool] = runner.slice_pca16(x)
    utility = {target: (np.arange(48)+j)%2 for j,target in enumerate(runner.TASKS)}
    utility['same_residence'][0] = -1
    utility['commute_over20'][::7] = -1
    race = np.arange(48)%9
    race[race==3] = 4
    audit = {'SEX':np.arange(48)%2,'RAC1P':race}
    return arrays,utility,copy.deepcopy(utility),audit,copy.deepcopy(audit)


def test_full_tiny_fitting_uses_only_fitting_stats_and_keeps_evaluation_sealed(
        fitting_data,monkeypatch,tmp_path):
    arrays,fy,vy,ay,av = fitting_data
    originals = {p:array_hash(x) for p,x in arrays.items()}
    labels_before = [{t:y.copy() for t,y in d.items()} for d in (fy,vy,ay,av)]
    def forbidden_fit(*args,**kwargs):
        raise AssertionError('A PCA fit is forbidden for this fixed-coordinate diagnostic')
    monkeypatch.setattr(PCA,'fit',forbidden_fit)
    assert not any('test' in key for key in inspect.signature(runner.fit_release).parameters)
    with pytest.raises(RuntimeError,match='sealed'):
        require_test_selection(tmp_path,[])
    fitted,record = runner.fit_release(arrays,fy,vy,ay,av,97,tmp_path,miniature=True)
    assert len(fitted) == len(record['head_selections']) == 7
    assert sum(len(v) for v in fitted.values()) == 20
    assert len(list(tmp_path.rglob('selection.json'))) == 7
    with pytest.raises(RuntimeError,match='sealed'):
        require_test_selection(tmp_path,list(record['head_selections']))
    for key,candidates in fitted.items():
        role,release,target = key.split('/')
        pool = 'downstream_fit' if role == 'transfer' else 'attacker_fit'
        indices = record['task_fit_indices' if role == 'transfer' else 'audit_fit_indices'][target]
        truth = fy[target] if role == 'transfer' else ay[target]
        assert (truth[indices]>=0).all()
        expected = arrays[pool][indices].astype(np.float64)
        std = expected.std(0)
        for candidate in candidates.values():
            np.testing.assert_array_equal(candidate.preprocessing.mean,expected.mean(0))
            np.testing.assert_array_equal(candidate.preprocessing.scale,np.where(std>1e-12,std,1.))
            assert candidate.preprocessing.mean[15] == 5
            assert candidate.preprocessing.scale[15] == 1
            if target == 'RAC1P':
                assert candidate.n_classes == 9
                assert candidate.metadata['fit_support'][3] == 0
                assert candidate.metadata['fit_coverage_complete'] is False
                assert candidate.metadata['validation_scores']['auroc'] is None
            if candidate.family == 'mlp':
                assert candidate.metadata['parameters']['epochs'] == 1
                assert candidate.metadata['optimizer_steps'] == 1
                assert not any(p.requires_grad for p in candidate.model.parameters())
        chosen = min(candidates,key=lambda c:(candidates[c].metadata['validation_scores']['log_loss'],c))
        assert record['head_selections'][key] == chosen
        assert record['fitting_records'][key]['selected_family'] == chosen
    assert {p:array_hash(x) for p,x in arrays.items()} == originals
    for actual,before in zip((fy,vy,ay,av),labels_before):
        for name in actual:
            np.testing.assert_array_equal(actual[name],before[name])
    write_json(tmp_path/'selection_before_test.json',{'head_selections':{}})
    with pytest.raises(RuntimeError,match='Incomplete'):
        require_test_selection(tmp_path,list(record['head_selections']))
    write_json(tmp_path/'selection_before_test.json',{'head_selections':record['head_selections']})
    assert require_test_selection(tmp_path,list(record['head_selections']))
    evaluation = runner.slice_pca16(np.random.default_rng(911).normal(size=(8,32)).astype(np.float32))
    for candidates in fitted.values():
        for candidate in candidates.values():
            before = _state_hash(candidate.model) if candidate.family == 'mlp' else None
            probability = candidate.predict_proba(evaluation)
            assert probability.shape == (8,candidate.n_classes)
            if before is not None:
                assert _state_hash(candidate.model) == before


def test_fitting_rejects_final_evaluation_arrays(fitting_data,tmp_path):
    arrays,fy,vy,ay,av = fitting_data
    arrays = {**arrays,'test':arrays['attacker_validation']}
    with pytest.raises((ValueError,AssertionError)):
        runner.fit_release(arrays,fy,vy,ay,av,97,tmp_path,miniature=True)
    assert not list(tmp_path.rglob('model.pt'))

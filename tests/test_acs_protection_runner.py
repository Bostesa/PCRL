"""Small artificial-data checks of frozen-parent protection orchestration."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from experiments import run_acs_protection as runner
from experiments.acs_protection_maps import FrozenAffineMap
from experiments.acs_transfer_data import CovariatePreprocessor, POOLS, array_hash, sha_file
from experiments.acs_transfer_models import state_digest


_SCHEMA = {'income_binary': 2, 'income_bins': 8, 'esr': 6, 'pubcov': 2, 'joint': 8}


def fixture_probabilities(x, reverse=False):
    x = np.asarray(x, dtype=np.float32)
    if reverse:
        x = x[:, ::-1]
    output = {}
    offset = 0
    for name, size in _SCHEMA.items():
        logits = x[:, offset:offset+size]
        logits = logits-logits.max(1, keepdims=True)
        probabilities = np.exp(logits)
        output[name] = probabilities/probabilities.sum(1, keepdims=True)
        offset += size
    return output


class FixtureSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.linspace(.8, 1.2, 32))
        self.eval()

    def probabilities(self, x):
        return fixture_probabilities(x)

    def release(self, x):
        return np.asarray(x[:, :32]*self.scale.numpy(), dtype=np.float32)


class FixtureTree:
    def probabilities(self, x):
        return fixture_probabilities(x, reverse=True)


class FixtureMap:
    def __init__(self, dimension):
        self.dimension = dimension

    def transform(self, x):
        return np.asarray(x[:, :self.dimension], dtype=np.float32).copy()


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def artificial_parent(tmp_path, monkeypatch):
    """Saved fixed arrays/models: no source fitting and no actual ACS access."""
    n = 36*len(POOLS)
    index = np.arange(n)
    frame = pd.DataFrame({
        '_raw_row': index, 'SERIALNO': [f'fixture_{i//2}' for i in index], 'SPORDER': 1+index%2,
        'PWGTP': 1+index%13, 'AGEP': 19+index%16, 'WKHP': 1+index%65,
        'SCHL': 1+index%24, 'MAR': 1+index%5, 'RELP': index%18, 'CIT': 1+index%5,
        'DIS': 1+index%2, 'DEAR': 1+(index//2)%2, 'DEYE': 1+(index//3)%2,
        'DREM': 1+(index//5)%2, 'PINCP': (index*751)%100000,
        'ESR': 1+index%6, 'PUBCOV': 1+index%2, 'MIG': 1+index%3,
        'JWMNP': 1+index%60, 'SEX': 1+index%2, 'RAC1P': 1+index%9,
    })
    for field in ('PINCP', 'ESR', 'PUBCOV', 'MIG', 'JWMNP', 'SEX', 'RAC1P'):
        frame[field] = frame[field].astype(float)
    frame.loc[index%23 == 0, 'PINCP'] = np.nan
    frame.loc[index%17 == 0, 'ESR'] = np.nan
    frame.loc[index%19 == 0, 'PUBCOV'] = np.nan
    frame.loc[index%29 == 0, 'MIG'] = np.nan
    frame.loc[index%13 == 0, 'JWMNP'] = np.nan
    frame.loc[11, 'SEX'] = np.nan
    frame.loc[22, 'RAC1P'] = np.nan
    pools = {name: index[i*36:(i+1)*36] for i, name in enumerate(POOLS)}
    parent = tmp_path/'parent'
    directory = parent/'seed_97'
    directory.mkdir(parents=True)
    raw = tmp_path/'fixture.csv'
    frame.drop(columns='_raw_row').to_csv(raw, index=False)
    write = runner.write_json
    write(parent/'config.json', {'raw_path': 'fixture.csv', 'sample_cap': n, 'sample_seed': 5})
    write(parent/'schema_support.json', {'raw_sha256': sha_file(raw)})
    pre = CovariatePreprocessor().fit(frame.iloc[pools['representation_fit']])
    write(directory/'preprocessing.json', pre.metadata())
    model, tree = FixtureSource(), FixtureTree()
    maps = {'pca': FixtureMap(32), 'compression': FixtureMap(26)}
    selection = {'selected_lr': .001, 'selected_tree_leaves': 15,
                 'income_edges': np.arange(10000, 80000, 10000).tolist(),
                 'bank_keys': list(_SCHEMA), 'selected_encoder_state_sha256': state_digest(model.state_dict())}
    write(directory/'source_selection.json', selection)
    source_file = directory/'source/encoder_0.001/selected.pt'
    source_file.parent.mkdir(parents=True)
    torch.save({'fixture_state': model.state_dict()}, source_file)
    tree_file = directory/'source/tree_15/source_tree_bank.joblib'
    tree_file.parent.mkdir(parents=True)
    joblib.dump(tree, tree_file)
    joblib.dump(maps, directory/'release_maps.joblib')
    np.savez_compressed(directory/'split_rows.npz', **pools)

    def releases(part):
        x = pre.transform(part)
        p, q = model.probabilities(x), tree.probabilities(x)
        return {
            'A_binary_bank': np.column_stack([p['income_binary'][:, 1], p['esr'][:, 0], p['pubcov'][:, 0]]),
            'B_rich_bank': np.concatenate([p[k] for k in _SCHEMA], axis=1),
            'C_tree_bank': np.concatenate([q[k] for k in _SCHEMA], axis=1),
            'D_compressed': maps['compression'].transform(model.release(x)),
            'E_pca': maps['pca'].transform(x), 'F_covariates': x,
        }

    all_releases = {pool: releases(frame.iloc[idx]) for pool, idx in pools.items()}
    metadata = {}
    for name in (*runner.PARENTS, 'F_covariates'):
        development = {pool: outputs[name] for pool, outputs in all_releases.items() if pool != 'test'}
        np.savez_compressed(directory/f'release_{name}.npz', **development)
        metadata[name] = {'development_hashes': {pool: array_hash(x) for pool, x in development.items()},
                          'test_output_sha256': array_hash(all_releases['test'][name])}
    write(directory/'metrics.json', {'release_metadata': metadata})
    artifact_paths = [p for p in directory.rglob('*') if p.suffix in ('.npz', '.pt', '.joblib')]
    write(directory/'local_artifacts.json', [{'path': str(p.relative_to(tmp_path)),
                                             'sha256': sha_file(p), 'bytes': p.stat().st_size}
                                            for p in artifact_paths])
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.SourceEncoder, 'load', lambda path: model)
    return {'frame': frame, 'pools': pools, 'parent': parent, 'model': model,
            'all_releases': all_releases, 'tree_file': tree_file, 'raw': raw}


def test_binary_source_and_reserved_task_masks_are_not_reinterpreted():
    frame = pd.DataFrame({'PINCP': [50000., 50001., np.nan, -19999., -19998.],
                          'ESR': [1., 2., np.nan, 0., 6.], 'PUBCOV': [1., 2., np.nan, 0., 2.],
                          'MIG': [1., 2., 3., np.nan, 0.], 'JWMNP': [20., 21., 0., np.nan, 201.]})
    result = runner.binary_tasks(frame, [0., 50000.])
    assert tuple(result) == runner.TASKS
    assert result['income_binary'].tolist() == [0, 1, -1, -1, 0]
    assert result['civilian_at_work'].tolist() == [1, 0, -1, -1, 0]
    assert result['public_coverage'].tolist() == [1, 0, -1, -1, 0]
    assert result['same_residence'].tolist() == [1, 0, 0, -1, -1]
    assert result['commute_over20'].tolist() == [0, 1, -1, -1, -1]


def test_parent_loader_uses_recorded_tree_artifact_and_exact_saved_outputs(artificial_parent):
    fixture = artificial_parent
    bundle = runner.ParentBundle(fixture['parent'], 97, fixture['frame'], fixture['pools'])
    assert any(path.endswith('source_tree_bank.joblib') for path in bundle.used_files)
    assert bundle.fingerprint() == bundle.initial
    recreated = bundle.evaluation_releases(fixture['frame'].iloc[fixture['pools']['test']])
    for name in (*runner.PARENTS, 'F_covariates'):
        np.testing.assert_array_equal(recreated[name], fixture['all_releases']['test'][name])
        for pool, array in bundle.releases[name].items():
            assert array.flags.writeable is False
            np.testing.assert_array_equal(array, fixture['all_releases'][pool][name])
    fixture['tree_file'].write_bytes(b'changed fixture artifact')
    with pytest.raises(RuntimeError, match='Missing/changed saved parent artifact'):
        runner.ParentBundle(fixture['parent'], 97, fixture['frame'], fixture['pools'])


def test_complete_runner_fits_only_calibration_and_opens_evaluation_after_all_saved_choices(
        artificial_parent, monkeypatch, tmp_path):
    fixture = artificial_parent
    frame, pools = fixture['frame'], fixture['pools']
    out = tmp_path/'protection'
    out.mkdir()
    runner.write_json(out/'protocol_freeze.json', {'fixture_only': True})
    cfg = {'parent_results': 'parent', 'head_budget': 16, 'attacker_budget': 32,
           'head_mlp_epochs': 40, 'evaluation_status': 'artificial fixture; no ACS outcomes'}
    cfg_before = copy.deepcopy(cfg)
    seed_dir = out/'seed_97'
    monkeypatch.setattr(runner, 'verify_freeze', lambda directory: None)
    monkeypatch.setattr(runner, 'load_cohort', lambda *args: (frame.copy(), {'fixture_rows': len(frame)}))
    monkeypatch.setattr(runner, 'split_households', lambda f, seed: pools)
    fit_calls, audit_calls, calibration_calls, task_accesses = [], [], [], []
    original_fit, original_audit = runner.fit_candidates, runner.fit_primary_auditors
    original_leace, original_tasks = runner.fit_joint_leace, runner.binary_tasks
    original_evaluate = runner.ParentBundle.evaluation_releases
    original_score = runner.score_and_save

    def fit_fixture(x, y, xv, yv, k, seed, **kwargs):
        assert (seed_dir/'release_freeze.json').exists()
        assert not (seed_dir/'selection_before_test.json').exists()
        assert (y >= 0).all() and (yv >= 0).all()
        fit_calls.append((y.copy(), yv.copy(), seed))
        # Scientific config stays unchanged; only this artificial fitting seam is tiny.
        budget = {'mlp': {'epochs': 1, 'hidden': [8, 4], 'batch_size': 16},
                  'logistic': {'max_iter': 15}}
        return original_fit(x, y, xv, yv, k, seed, budget=budget)

    def audit_fixture(x, y, xv, yv, k, seed, **kwargs):
        assert (seed_dir/'release_freeze.json').exists()
        assert not (seed_dir/'selection_before_test.json').exists()
        assert (y >= 0).all() and (yv >= 0).all()
        audit_calls.append((y.copy(), yv.copy(), seed))
        budget = {'mlp': {'epochs': 1, 'hidden': [8, 4], 'batch_size': 16},
                  'logistic': {'max_iter': 15}, 'histgb': {'max_iter': 1}}
        return original_audit(x, y, xv, yv, k, seed, budget=budget)

    def calibration_only(x, labels, **kwargs):
        assert not (seed_dir/'release_freeze.json').exists()
        assert not task_accesses
        assert kwargs == {'fit_split': 'representation_fit'}
        assert set(labels) == {'SEX', 'RAC1P'}
        expected = runner.audit_labels(frame.iloc[pools['representation_fit']])
        for name in expected:
            np.testing.assert_array_equal(labels[name], expected[name])
        parent_name = runner.PARENTS[len(calibration_calls)]
        np.testing.assert_array_equal(x, fixture['all_releases']['representation_fit'][parent_name])
        calibration_calls.append(parent_name)
        return original_leace(x, labels, **kwargs)

    def reserved_labels_after_freeze(part, edges):
        assert (seed_dir/'release_freeze.json').exists()
        task_accesses.append(part._raw_row.to_numpy().copy())
        return original_tasks(part, edges)

    def evaluation_after_saved_choices(bundle, part):
        selected = runner.read(seed_dir/'selection_before_test.json')
        assert len(selected['head_selections']) == 86
        assert len(list((seed_dir/'fitted').rglob('selection.json'))) == 86
        assert runner.require_test_selection(seed_dir, selected['head_selections'])
        np.testing.assert_array_equal(part._raw_row.to_numpy(), pools['test'])
        return original_evaluate(bundle, part)

    def scoring_after_saved_choices(*args, **kwargs):
        assert (seed_dir/'selection_before_test.json').exists()
        return original_score(*args, **kwargs)

    monkeypatch.setattr(runner, 'fit_candidates', fit_fixture)
    monkeypatch.setattr(runner, 'fit_primary_auditors', audit_fixture)
    monkeypatch.setattr(runner, 'fit_joint_leace', calibration_only)
    monkeypatch.setattr(runner, 'binary_tasks', reserved_labels_after_freeze)
    monkeypatch.setattr(runner.ParentBundle, 'evaluation_releases', evaluation_after_saved_choices)
    monkeypatch.setattr(runner, 'score_and_save', scoring_after_saved_choices)
    result = runner.run_seed(out, cfg, 97)
    assert cfg == cfg_before
    assert calibration_calls == list(runner.PARENTS)
    assert len(fit_calls) == 55 and len(audit_calls) == 24
    assert len(result['release_metadata']) == 11
    assert len(result['raw_metrics']) == 237
    for name in ('source_unchanged', 'parent_files_unchanged', 'erasers_unchanged',
                 'development_releases_unchanged', 'selection_unchanged',
                 'evaluation_parent_outputs_match_historical'):
        assert result['integrity'][name] is True
    for name in runner.PARENTS:
        with np.load(seed_dir/f'release_{name}.npz') as saved:
            for pool in pools:
                np.testing.assert_array_equal(saved[pool], fixture['all_releases'][pool][name])
        fitted_map = FrozenAffineMap.load(seed_dir/f'map_{name}.npz')
        with np.load(seed_dir/f'release_{name}_leace.npz') as saved:
            for pool in pools:
                np.testing.assert_array_equal(saved[pool], fitted_map.apply(fixture['all_releases'][pool][name]))
    # The original source cache still has no evaluation arrays inserted by reuse.
    with np.load(fixture['parent']/'seed_97/release_A_binary_bank.npz') as parent_cache:
        assert 'test' not in parent_cache.files
    with np.load(seed_dir/'predictions.npz') as saved:
        assert len(saved.files) == 474
    selection = runner.read(seed_dir/'selection_before_test.json')
    for task_index, target in enumerate(runner.TASKS):
        expected = fit_calls[task_index]
        for release_index in range(11):
            actual = fit_calls[release_index*5+task_index]
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            assert actual[2] == expected[2]
        assert selection['task_fit_indices'][target]['n'] == len(expected[0])

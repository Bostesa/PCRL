"""Patched runner integration on artificial rows; no ACS models are fitted."""
import json

import numpy as np
import pandas as pd
import torch

from experiments import run_acs_transfer as runner
from experiments import acs_transfer_data as data
from experiments.acs_transfer_heads import fit_prior, metrics


class StubSource(torch.nn.Module):
    def __init__(self, width, schema, tag):
        super().__init__()
        self.register_buffer('tag', torch.tensor(float(tag)))
        self.schema, self.repr_dim = schema, 32
        self.matrix = np.random.default_rng(61).normal(size=(width, 32)).astype(np.float32)
        self.eval()

    def release(self, x):
        return np.asarray(x @ self.matrix, dtype=np.float32)

    def probabilities(self, x):
        result = {}
        for name, k in self.schema.items():
            logits = np.clip(x[:, :1], -2, 2)*np.linspace(-.3, .3, k)[None, :]
            p = np.exp(logits-logits.max(1, keepdims=True))
            result[name] = np.asarray(p/p.sum(1, keepdims=True), dtype=np.float32)
        return result


class SpyPreprocessor(data.CovariatePreprocessor):
    context = None

    def fit(self, frame):
        assert set(frame._raw_row) == self.context['representation_rows']
        return super().fit(frame)

    def transform(self, frame):
        if set(frame._raw_row).intersection(self.context['test_rows']):
            assert self.context['guard_open'], 'Test features accessed before all selections were saved'
            self.context['test_transforms'] += 1
        return super().transform(frame)


def artificial_frame(n=840):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'SERIALNO': [f'fixture_{i//2}' for i in range(n)],
                          'SPORDER': np.arange(n) % 2 + 1, 'PWGTP': rng.integers(1, 50, n),
                          '_raw_row': np.arange(n), 'AGEP': rng.integers(19, 35, n),
                          'WKHP': rng.integers(1, 99, n), 'PINCP': rng.integers(0, 100001, n),
                          'ESR': rng.integers(1, 7, n), 'PUBCOV': rng.integers(1, 3, n),
                          'MIG': rng.integers(1, 4, n), 'JWMNP': rng.integers(1, 201, n),
                          'SEX': rng.integers(1, 3, n), 'RAC1P': rng.integers(1, 10, n)})
    for name in data.CATEGORICAL:
        frame[name] = rng.choice(list(data.CATEGORY_CODES[name]), n)
    frame.loc[::11, 'JWMNP'] = np.nan
    return frame


def test_runner_freezes_every_selection_before_test_and_matches_access(tmp_path, monkeypatch):
    frame = artificial_frame()
    pools = data.split_households(frame, 0)
    context = {'guard_open': False, 'test_transforms': 0,
               'representation_rows': set(frame.iloc[pools['representation_fit']]._raw_row),
               'test_rows': set(frame.iloc[pools['test']]._raw_row)}
    SpyPreprocessor.context = context
    monkeypatch.setattr(runner, 'CovariatePreprocessor', SpyPreprocessor)
    monkeypatch.setattr(runner, 'load_cohort', lambda *a, **kw: (frame.copy(), {'fixture': True}))
    source_calls, head_calls = [], []

    def fake_source(x, labels, xv, yv, schema, *, seed, **kwargs):
        assert not context['guard_open']
        assert set(labels) == set(yv) == set(data.SOURCE_KEYS)
        assert len(x) == len(pools['representation_fit'])
        assert len(xv) == len(pools['source_validation'])
        source_calls.append((x.copy(), {k: v.copy() for k, v in labels.items()}, seed, kwargs))
        tag = kwargs.get('lr', kwargs.get('max_leaf_nodes'))
        return StubSource(x.shape[1], schema, tag), {'validation_loss': float(tag), 'fixture': True}

    def fake_heads(x, y, xv, yv, k, seed, families=('logistic', 'mlp'), budget=None):
        assert not context['guard_open'], 'Fitting called after final-test barrier'
        head_calls.append((seed, len(x), y.copy(), yv.copy(), tuple(families)))
        candidates = {}
        for family in families:
            candidate = fit_prior(y, k)
            candidate.metadata.update(family=family, validation_scores=metrics(yv, candidate.predict_proba(xv), k))
            candidates[family] = candidate
        return {'candidates': candidates, 'selected_family': families[0],
                'metadata': {'fixture': True, 'candidates': {k: v.metadata for k, v in candidates.items()}}}

    original_guard = runner.require_test_selection

    def open_guard(directory, expected):
        selection_path = directory/'selection_before_test.json'
        assert (directory/'source_selection.json').exists()
        assert selection_path.exists()
        saved = json.loads(selection_path.read_text())
        assert sorted(saved['head_selections']) == sorted(expected)
        for key in expected:
            directory_for_key = directory/'fitted'/key
            assert directory_for_key.exists()
            assert list(directory_for_key.rglob('metadata.json'))
        digest = original_guard(directory, expected)
        context['guard_open'] = True
        return digest

    monkeypatch.setattr(runner, 'fit_source_encoder', fake_source)
    monkeypatch.setattr(runner, 'fit_source_tree_bank', fake_source)
    monkeypatch.setattr(runner, 'fit_candidates', fake_heads)
    monkeypatch.setattr(runner, 'require_test_selection', open_guard)
    cfg = {'raw_path': 'unused_fixture.csv', 'sample_cap': 840, 'sample_seed': 1,
           'threads': 1, 'source_lr_grid': [.001, .003], 'source_epochs': 60,
           'source_batch_size': 256, 'source_tree_leaf_grid': [15, 31],
           'source_tree_iterations': 150, 'head_budget': 32, 'attacker_budget': 64,
           'head_mlp_epochs': 40, 'head_mlp_batch_size': 256, 'head_mlp_lr': .001,
           'attacker_tree_iterations': 150, 'attacker_tree_leaves': 15}
    result = runner.run_seed(tmp_path, cfg, 0, verify=False)
    assert context['guard_open'] and context['test_transforms'] == 1
    assert len(source_calls) == 4
    for x, labels, seed, _ in source_calls[1:]:
        np.testing.assert_array_equal(x, source_calls[0][0])
        assert seed == source_calls[0][2]
        for key in labels:
            np.testing.assert_array_equal(labels[key], source_calls[0][1][key])
    # The same task/audit seed identifies exactly the same labeled rows across interfaces.
    for seed in {row[0] for row in head_calls}:
        rows = [row for row in head_calls if row[0] == seed]
        for row in rows[1:]:
            np.testing.assert_array_equal(row[2], rows[0][2])
            np.testing.assert_array_equal(row[3], rows[0][3])
            assert row[4] == rows[0][4]
    assert result['source_selection']['lr'] == .001
    assert result['source_selection']['tree_leaves'] == 15
    assert result['release_metadata']['A_binary_bank']['dimension'] == 3
    assert result['release_metadata']['B_rich_bank']['dimension'] == result['release_metadata']['C_tree_bank']['dimension']
    assert result['release_metadata']['D_compressed']['dimension'] == result['release_metadata']['B_rich_bank']['dimension']
    assert result['integrity']['source_unchanged']
    assert result['integrity']['development_releases_unchanged']
    assert result['integrity']['selection_still_unchanged']


def test_source_scores_match_common_cross_entropy_floor_and_renormalization():
    from experiments.acs_transfer_models import source_scores
    schema = {'income_binary': 2, 'income_bins': 8, 'esr': 6, 'pubcov': 2, 'joint': 8}
    probabilities = {name: np.eye(k, dtype=np.float32)[[0, 1, 0]] for name, k in schema.items()}
    labels = {name: np.array([1, 0, -1]) for name in schema}
    score = source_scores(probabilities, labels, schema)
    for name, k in schema.items():
        expected = metrics(labels[name][:2], probabilities[name][:2], k)['log_loss']
        assert score['heads'][name]['log_loss'] == expected

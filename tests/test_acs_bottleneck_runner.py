"""Artificial-data checks for frozen-reference ACS bottleneck orchestration."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments import run_acs_bottleneck as runner
from experiments.acs_transfer_data import array_hash
from experiments.acs_transfer_heads import _state_hash, load_candidate
from tests.test_acs_protection_runner import artificial_parent, one_thread


def test_source_training_receives_only_source_columns_and_preserves_missing_masks(monkeypatch):
    frame = pd.DataFrame({'PINCP': [50000., 50001., np.nan, -19999., -19998.],
                          'ESR': [1., 2., np.nan, 0., 6.], 'PUBCOV': [1., 2., np.nan, 0., 2.],
                          'MIG': [1., 2., 3., np.nan, 0.], 'JWMNP': [20., 21., 0., np.nan, 201.]})
    original = runner.source_labels
    calls = []

    def guarded(part, edges):
        assert list(part.columns) == ['PINCP', 'ESR', 'PUBCOV']
        calls.append(True)
        return original(part, edges)

    monkeypatch.setattr(runner, 'source_labels', guarded)
    labels = runner.source_binaries(frame, [0., 50000.])
    assert tuple(labels) == runner.SOURCE_TASKS and calls == [True]
    assert labels['income_binary'].tolist() == [0, 1, -1, -1, 0]
    assert labels['civilian_at_work'].tolist() == [1, 0, -1, -1, 0]
    assert labels['public_coverage'].tolist() == [1, 0, -1, -1, 0]


def test_full_miniature_preserves_reference_states_and_saves_selections_before_evaluation(
        artificial_parent, monkeypatch, tmp_path):
    fixture = artificial_parent
    frame, pools = fixture['frame'], fixture['pools']
    out = tmp_path/'bottleneck'
    out.mkdir()
    seed_dir = out/'seed_97'
    cfg = {'parent_results': 'parent', 'reference_results': 'reference',
           'head_budget': 16, 'attacker_budget': 32, 'head_mlp_epochs': 40,
           'catchup_epochs': 120, 'evaluation_status': 'artificial fixture; no ACS outcomes'}
    cfg_before = copy.deepcopy(cfg)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load_cohort', lambda *args: (frame.copy(), {'fixture_rows': len(frame)}))
    monkeypatch.setattr(runner, 'split_households', lambda *args: pools)
    source_before = _state_hash(fixture['model'])
    accesses, training, catches, references = [], [], [], []

    class FixtureReferences:
        """Previously fixed outputs/records; no reference model is fitted here."""
        def __init__(self, config, seed, actual_frame, actual_pools):
            assert seed == 97
            self.directory = tmp_path/'reference/seed_97'
            self.directory.mkdir(parents=True)
            self.source = SimpleNamespace(selection=runner.read(fixture['parent']/'seed_97/source_selection.json'),
                                          used_files={})
            self.used_files = {}
            self.checked_candidates = 0
            self.releases, self.test_outputs = {}, {}
            for name in runner.REFERENCES:
                parent = name.removesuffix('_leace')
                fixed = {}
                for pool, outputs in fixture['all_releases'].items():
                    array = outputs[parent].copy()
                    if name.endswith('_leace'):
                        # A predeclared fixture map; no eraser fitting in this stage.
                        array[:, :2] = 0
                    array.setflags(write=False)
                    if pool == 'test':
                        self.test_outputs[name] = array
                    else:
                        fixed[pool] = array
                self.releases[name] = fixed
            self.selection = {k: {} for k in ('fitting_records', 'head_selections',
                                               'family_selections', 'auroc_selections')}
            raw = []
            for role, targets, names in (
                    ('transfer', runner.TASKS, (*runner.REFERENCES, 'prior')),
                    ('audit', runner.AUDITS, (*runner.REFERENCES, 'prior', 'exposed'))):
                for name in names:
                    for target in targets:
                        key = f'{role}/{name}/{target}'
                        self.selection['fitting_records'][key] = {'candidates': {'fixture': {'frozen': True}}}
                        self.selection['head_selections'][key] = 'fixture'
                        self.selection['family_selections'][key] = {'fixture': 'fixture'}
                        self.selection['auroc_selections'][key] = None
                        raw.append({'release': name, 'role': role, 'target': target,
                                    'candidate_id': 'fixture', 'family': 'fixture',
                                    'selected': True, 'fixture_record': key})
            self.previous = {
                'support_by_pool': {p: {'raw_row_sha256': array_hash(frame.iloc[i]._raw_row.to_numpy())}
                                    for p, i in pools.items()},
                'release_metadata': {n: {'dimension': a['representation_fit'].shape[1], 'fixture': True}
                                     for n, a in self.releases.items()},
                'raw_metrics': raw}
            runner.write_json(self.directory/'metrics.json', self.previous)
            self.initial = self.fingerprint()
            references.append(self)

        def fingerprint(self):
            return {'outputs': runner.frozen_digest(self.releases),
                    'source': _state_hash(fixture['model'])}

        def verify_reuse(self, fy, vy, ay, av, ti, ai, records):
            assert (seed_dir/'release_freeze.json').exists()
            assert not (seed_dir/'selection_before_test.json').exists()
            assert all((fy[t][i] >= 0).all() for t, i in ti.items())
            assert all((ay[t][i] >= 0).all() for t, i in ai.items())
            assert all(r['n'] <= cfg['head_budget'] for r in records['task_fit_indices'].values())
            self.checked_candidates = len(self.previous['raw_metrics'])

        def evaluation_releases(self, part):
            selected = runner.read(seed_dir/'selection_before_test.json')
            assert len(selected['head_selections']) == 65
            assert len(list((seed_dir/'fitted').rglob('selection.json'))) == 14
            assert runner.require_test_selection(seed_dir, list(selected['head_selections']))
            np.testing.assert_array_equal(part._raw_row.to_numpy(), pools['test'])
            accesses.append(('test_outputs', part._raw_row.to_numpy().copy()))
            return self.test_outputs

    monkeypatch.setattr(runner, 'SavedReferences', FixtureReferences)
    original_train, original_tasks, original_catch = runner.train_pair, runner.binary_tasks, runner.fit_catchup

    def train_only(x, y, attributes, xv, yv, seed, directory, **kwargs):
        assert not accesses and not (seed_dir/'release_freeze.json').exists()
        assert tuple(y) == tuple(yv) == runner.SOURCE_TASKS
        assert set(attributes) == set(runner.AUDITS)
        np.testing.assert_array_equal(x, fixture['all_releases']['representation_fit']['E_pca'])
        np.testing.assert_array_equal(xv, fixture['all_releases']['source_validation']['E_pca'])
        expected = runner.source_binaries(frame.iloc[pools['representation_fit']], [0., 50000.])
        for target in y:
            np.testing.assert_array_equal(y[target], expected[target])
        result = original_train(x, y, attributes, xv, yv, seed, directory, **kwargs)
        training.append(result)
        return result

    def reserved_after_freeze(part, edges):
        assert (seed_dir/'release_freeze.json').exists()
        accesses.append(('reserved', part._raw_row.to_numpy().copy()))
        return original_tasks(part, edges)

    def catch_in_original_coordinates(model, x, y, xv, yv, k, seed, **kwargs):
        assert (seed_dir/'release_freeze.json').exists()
        assert not (seed_dir/'selection_before_test.json').exists()
        before = _state_hash(model)
        info = kwargs['inherited_exposure']
        assert info['target'] == ('SEX' if k == 2 else 'RAC1P')
        assert info['fit_pool'] == 'representation_fit' and info['fit_rows'] == 36
        assert info['total_row_passes_equivalent'] == 7
        assert info['total_row_exposures'] == 36*7
        assert info['total_known_label_exposures'] == info['fit_coverage']['known']*7
        assert info['common_warm_adversary_optimizer_steps'] == 3
        assert info['continuation_adversary_optimizer_steps'] == 18
        assert info['total_optimizer_steps'] == 21
        assert info['final_adversary_state_hash'] == before
        assert info['fit_raw_row_sha256'] == array_hash(pools['representation_fit'])
        result = original_catch(model, x, y, xv, yv, k, seed, **kwargs)
        assert before == _state_hash(model)
        for candidate in (result['saved'], result['catchup']):
            np.testing.assert_array_equal(candidate.preprocessing.mean, np.zeros(16))
            np.testing.assert_array_equal(candidate.preprocessing.scale, np.ones(16))
            assert candidate.metadata['inherited_exposure'] == info
        assert result['saved'].metadata['fit_support'] is None
        assert result['saved'].metadata['optimizer_steps'] == 0
        assert result['catchup'].metadata['optimizer']['initial_state_entries'] == 0
        catches.append(result)
        return result

    monkeypatch.setattr(runner, 'train_pair', train_only)
    monkeypatch.setattr(runner, 'binary_tasks', reserved_after_freeze)
    monkeypatch.setattr(runner, 'fit_catchup', catch_in_original_coordinates)
    result = runner.run_seed(out, cfg, 97, miniature=True)
    assert cfg == cfg_before and len(training) == 1 and len(catches) == 4
    assert source_before == _state_hash(fixture['model'])
    assert references[0].fingerprint() == references[0].initial
    assert all(value for key, value in result['integrity'].items() if key.endswith('_unchanged'))
    assert len(result['raw_metrics']) == 99
    with np.load(seed_dir/'predictions.npz') as predictions:
        assert len(predictions.files) == 96
    saved = runner.read(seed_dir/'selection_before_test.json')
    for key, record in saved['fitting_records'].items():
        if key.split('/')[1] not in runner.LEARNED:
            continue
        persisted = runner.read(seed_dir/'fitted'/key/'selection.json')
        assert persisted == record
        assert persisted['selected_family'] == saved['head_selections'][key]
        assert persisted['independent_selection']['selected_family'] == saved['independent_selections'][key]
        assert 'same five' not in persisted['auroc_scope']
        if key.startswith('audit/'):
            assert len(persisted['candidate_ids']) == 7
            assert len(persisted['independent_candidate_ids']) == 5
            assert len(persisted['primary_selection_candidates']) == 6
            assert 'saved_adversary' not in persisted['primary_selection_candidates']
            assert persisted['total_new_mlp_optimizer_steps'] == persisted['independent_total_mlp_optimizer_steps']+1
            before = load_candidate(seed_dir/'fitted'/key/'saved_adversary')
            assert before.metadata['selected_state_hash'] == before.metadata['inherited_exposure']['final_adversary_state_hash']
        else:
            assert persisted['selection_split'] == 'downstream_validation'
    old_rows = references[0].previous['raw_metrics']
    copied_rows = [v for v in result['raw_metrics'] if v['reused_reference']]
    assert [{k: v for k, v in row.items() if k not in ('independent_selected', 'reused_reference')}
            for row in copied_rows] == old_rows
    for name in runner.LEARNED:
        with np.load(seed_dir/f'release_{name}.npz') as z:
            assert all(z[p].shape == (36, 16) for p in pools)
    summary = {'fixture': 'artificial seed97; 252 generated rows; no ACS outcomes',
               'new_candidate_rows': 48, 'reused_mock_reference_rows': 51,
               'prediction_arrays': 96, 'new_fitting_suites': 14, 'catchup_suites': 4,
               'integrity': result['integrity'], 'runtime': result['runtime']}
    runner.write_json(tmp_path/'PIPELINE_CHECK.json', summary)

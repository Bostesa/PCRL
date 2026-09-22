"""Independent artifact replay must detect altered weights, rows and choices."""
import copy
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest
import torch

from experiments.pcrl_task_directed_release_v1 import audits, replay


def score(loss, w):
    u, weighted = float(loss.mean()), float(np.dot(w/w.sum(), loss))
    return {'unweighted': u, 'weighted': weighted, 'balanced': (u+weighted)/2}


@pytest.fixture
def artifacts(tmp_path):
    n, y, weights = 4, np.zeros(4, dtype=int), np.arange(1., 5.)
    rows = {'ha': np.zeros((n, 4)), 'hb': np.zeros((n, 2)),
            'ids': np.array(['p0', 'p1', 'p2', 'p3']),
            'households': np.array(['hh0', 'hh0', 'hh1', 'hh2']),
            'weights': weights, 'labels': {'SEX': y}}
    p = np.array([[.25, .75], [.5, .5], [1., 0.], [0., 1.]])
    release = {'aux': None, 'token_probs': p}
    net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net[0].weight[0, 4] = np.log(9.)
        net[2].weight[0, 0] = 1.
        net[4].weight[0, 0] = 1.
    model = audits.TokenCandidate('mlp', net.eval(), np.zeros(4), np.ones(4), 2, 2, False,
        {'class_order': [0, 1], 'input_dim': 6, 'probability_floor': 1e-9})
    prior = audits.TokenCandidate('prior', np.array([.2, .8]), np.zeros(4), np.ones(4), 1, 2, False,
        {'class_order': [0, 1], 'input_dim': 4, 'probability_floor': 1e-9})
    model.save(tmp_path/'model'); prior.save(tmp_path/'prior')
    q = model.predict_token_proba(rows['ha'], 2)
    loss = np.sum(p * -np.log(q[:, :, 0]), axis=1)
    prior_q = prior.predict_proba(rows['ha'])
    prior_loss = -np.log(prior_q[:, 0])
    validation = tmp_path/'val_losses.npz'
    np.savez_compressed(validation, ids=rows['ids'], households=rows['households'], weights=weights,
                        y=y, candidate_ids=['model', 'prior'], losses=np.column_stack((loss, prior_loss)))
    role = {'view': 'A', 'target': 'SEX', 'kind': 'attack', 'validation_pool': 'attacker_validation',
            'selection': 'model', 'independent_selection': 'model',
            'validation_scores': {'model': score(loss, weights), 'prior': score(prior_loss, weights)},
            'validation_loss_path': str(validation),
            'candidates': {
                'model': {'candidate': model, 'model_path': str(tmp_path/'model'), 'origin': 'independent',
                          'route': {'kind': 'model', 'source_view': 'A', 'wire': 'release'}},
                'prior': {'candidate': prior, 'model_path': str(tmp_path/'prior'), 'origin': 'independent',
                          'route': {'kind': 'model', 'source_view': 'A', 'wire': 'H'}}}}
    registry = {'schema': 1, 'name': 'fixture', 'anchor': 0, 'roles': {'attack:A/SEX': role},
                'same_predictions_for_both_weightings': True, 'selection_frozen': True, 'is_H_release': False}
    joblib.dump(registry, tmp_path/'registry.joblib')
    evaluation_dir = tmp_path/'evaluation'
    (evaluation_dir/'test').mkdir(parents=True)
    np.savez_compressed(evaluation_dir/'test'/'attack__A__SEX.npz', ids=rows['ids'],
                        households=rows['households'], weights=weights, y=y,
                        probabilities=q, token_probs=p, loss=loss, accuracy=np.ones(n), independent_loss=loss)
    summary = {'test': {'attack:A/SEX': {'ce': score(loss, weights),
                'accuracy': {'unweighted': 1., 'weighted': 1.}, 'support': 4, 'weight_sum': 10., 'ess': 100/30,
                'selection': 'model', 'independent_selection': 'model', 'independent_ce': score(loss, weights),
                'per_class': [{'class': 0, 'support': 4, 'weight_sum': 10., 'ess': 100/30, 'ce': score(loss, weights)},
                              {'class': 1, 'support': 0, 'weight_sum': 0., 'ess': 0.,
                               'ce': {'unweighted': None, 'weighted': None, 'balanced': None}}]}}}
    (evaluation_dir/'summary.json').write_text(json.dumps(summary))
    return registry, rows, release, tmp_path


def test_validation_replay_loads_weights_instead_of_trusting_embedded_objects(artifacts):
    registry, rows, release, out = artifacts
    registry['roles']['attack:A/SEX']['candidates']['model']['candidate'].model[0].weight.data.zero_()
    report = replay.verify_validation(registry, {'anchor': 0, 'pools': {'attacker_validation': rows}},
                                      {'attacker_validation': release}, out_dir=out/'replay')
    assert report['passed'] is True
    assert report['roles']['attack:A/SEX']['replayed_candidates'] == 2
    assert report['roles']['attack:A/SEX']['original_households'] == 3


def test_validation_replay_detects_changed_loss_or_weight_specific_selection(artifacts):
    registry, rows, release, _ = artifacts
    wrong = copy.deepcopy(registry)
    wrong['roles']['attack:A/SEX']['selection'] = 'prior'
    with pytest.raises(replay.ReplayError, match='selection'):
        replay.verify_validation(wrong, {'anchor': 0, 'pools': {'attacker_validation': rows}},
                                  {'attacker_validation': release})
    with np.load(registry['roles']['attack:A/SEX']['validation_loss_path']) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['losses'][0, 0] += .1
    np.savez_compressed(registry['roles']['attack:A/SEX']['validation_loss_path'], **arrays)
    with pytest.raises(replay.ReplayError):
        replay.verify_validation(registry, {'anchor': 0, 'pools': {'attacker_validation': rows}},
                                  {'attacker_validation': release})


def test_frozen_evaluation_replay_and_sampled_wire_keep_original_households(artifacts):
    registry, rows, release, out = artifacts
    report = replay.verify_evaluation(registry, {'anchor': 0, 'pools': {'test': rows}}, {'test': release},
                                      out/'evaluation', out_dir=out/'eval_replay', wire_repetitions=4096)
    role = report['pools']['test']['attack:A/SEX']
    assert role['original_people'] == 4 and role['original_households'] == 3
    assert role['wire_check']['passed'] is True
    with np.load(out/'eval_replay'/'test'/'attack__A__SEX_wire.npz') as wire:
        np.testing.assert_array_equal(wire['household_counts'], [2, 1, 1])
        assert wire['first_tokens'].shape == (4,)
        assert wire['mean_sampled_loss'].shape == (4,)


def test_household_sampling_keeps_whole_households_and_still_checks_full_selection(artifacts):
    registry, rows, release, _ = artifacts
    report = replay.verify_validation(registry, {'anchor': 0, 'pools': {'attacker_validation': rows}},
                                      {'attacker_validation': release}, max_households=1)
    row = report['roles']['attack:A/SEX']
    assert row['original_households'] == 3
    assert row['replayed_households'] == 1
    assert row['replayed_people'] in (1, 2)
    assert row['selection_verified_on_all_saved_validation_rows'] is True


def test_corrupt_class_order_and_saved_probabilities_fail(artifacts):
    registry, rows, release, out = artifacts
    path = out/'model'/'metadata.json'
    metadata = json.loads(path.read_text()); metadata['class_order'] = [1, 0]
    path.write_text(json.dumps(metadata))
    with pytest.raises(replay.ReplayError, match='class'):
        replay.verify_validation(registry, {'anchor': 0, 'pools': {'attacker_validation': rows}},
                                  {'attacker_validation': release})
    metadata['class_order'] = [0, 1]; path.write_text(json.dumps(metadata))
    path = out/'evaluation'/'test'/'attack__A__SEX.npz'
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays['probabilities'][0, 0] = [-.1, 1.1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(replay.ReplayError, match='probabilit'):
        replay.verify_evaluation(registry, {'anchor': 0, 'pools': {'test': rows}}, {'test': release},
                                  out/'evaluation', out_dir=out/'bad_replay', wire_repetitions=0)


def test_sampled_wire_handles_deterministic_token_law_exactly(tmp_path):
    q = np.array([[[.8, .2]], [[.3, .7]]])
    result = replay.sampled_wire_check(q, np.ones((2, 1)), [0, 1], [1., 2.],
                                       ['p0', 'p1'], ['same', 'same'],
                                       tmp_path/'wire.npz', repetitions=16, seed=31)
    assert result['passed'] is True
    assert result['metrics']['weighted']['conditional_mc_se'] == 0.
    assert result['metrics']['weighted']['absolute_error'] < 1e-12


def test_wire_check_rejects_fractional_labels_instead_of_truncating(tmp_path):
    with pytest.raises(replay.ReplayError, match='label'):
        replay.sampled_wire_check(np.full((2, 1, 2), .5), np.ones((2, 1)), [.5, 1.], [1., 1.],
                                   ['p0', 'p1'], ['h0', 'h1'], tmp_path/'bad_labels.npz', repetitions=8)


def test_archive_relocation_uses_only_restored_paths_and_preserves_original_registry(artifacts):
    registry, rows, release, source = artifacts
    restored = source.parent/(source.name+'_restored')
    shutil.copytree(source, restored)
    original_model_path = registry['roles']['attack:A/SEX']['candidates']['model']['model_path']
    relocated = replay.relocate_registry(registry, original_root=source, artifact_root=restored)
    assert registry['roles']['attack:A/SEX']['candidates']['model']['model_path'] == original_model_path
    assert relocated['roles']['attack:A/SEX']['candidates']['model']['model_path'] == str(restored/'model')
    shutil.rmtree(source/'model'); shutil.rmtree(source/'prior'); (source/'val_losses.npz').unlink()
    result = replay.verify_validation(source/'registry.joblib', {'anchor': 0, 'pools': {'attacker_validation': rows}},
        {'attacker_validation': release}, original_root=source, artifact_root=restored)
    assert result['passed'] is True


def test_archive_relocation_rejects_escape_and_missing_restored_files(artifacts):
    registry, _, _, source = artifacts
    restored = source.parent/(source.name+'_restored')
    shutil.copytree(source, restored)
    shutil.rmtree(restored/'model')
    (restored/'model').symlink_to(source/'model', target_is_directory=True)
    with pytest.raises(replay.ReplayError, match='contained'):
        replay.relocate_registry(registry, original_root=source, artifact_root=restored)
    (restored/'model').unlink()
    with pytest.raises(replay.ReplayError, match='missing'):
        replay.relocate_registry(registry, original_root=source, artifact_root=restored)


def test_archive_relocation_rejects_model_file_symlink_escape(artifacts):
    registry, _, _, source = artifacts
    restored = source.parent/(source.name+'_restored')
    shutil.copytree(source, restored)
    (restored/'model'/'candidate.joblib').unlink()
    (restored/'model'/'candidate.joblib').symlink_to(source/'model'/'candidate.joblib')
    with pytest.raises(replay.ReplayError, match='contained'):
        replay.relocate_registry(registry, original_root=source, artifact_root=restored)


def test_H_B_and_actual_singleton_parity_detects_missing_ancestor(artifacts):
    registry, rows, release, out = artifacts
    template = registry['roles']['attack:A/SEX']
    a_h = copy.deepcopy(template['candidates']['prior'])
    a_h['origin'] = 'independent'

    def prior_record(name, width, k=1):
        model = audits.TokenCandidate('prior', np.array([.6, .4]), np.zeros(width), np.ones(width), k, 2, False,
            {'class_order': [0, 1], 'input_dim': width if k == 1 else width+k, 'probability_floor': 1e-9})
        model.save(out/name)
        return {'candidate': model, 'model_path': str(out/name), 'origin': 'independent',
                'route': {'kind': 'model', 'source_view': 'B' if width == 2 else 'AB', 'wire': 'H' if k == 1 else 'release'}}

    def role(view, candidates, selected):
        return {**copy.deepcopy(template), 'view': view, 'candidates': candidates,
                'selection': selected, 'independent_selection': selected}

    b_h, ab_h = prior_record('B', 2), prior_record('AB', 6)
    baseline = {**registry, 'name': 'H', 'is_H_release': True, 'roles': {
        'attack:A/SEX': role('A', {'base': a_h}, 'base'),
        'attack:B/SEX': role('B', {'base': b_h}, 'base'),
        'attack:AB/SEX': role('AB', {'base': ab_h, 'ancestor_A__base': a_h,
                                     'ancestor_B__base': b_h}, 'base')}}
    actual_a = {'model': template['candidates']['model'], 'H__base': a_h}
    actual_ab = {'model': prior_record('AB_Q', 6, 2), 'H__base': ab_h,
                 'H__ancestor_A__base': a_h, 'H__ancestor_B__base': b_h,
                 'ancestor_A__model': actual_a['model'], 'ancestor_A__H__base': a_h,
                 'ancestor_B__base': b_h}
    current = {**registry, 'roles': {'attack:A/SEX': role('A', actual_a, 'model'),
                'attack:B/SEX': baseline['roles']['attack:B/SEX'],
                'attack:AB/SEX': role('AB', actual_ab, 'model')}}
    ctx = {'anchor': 0, 'pools': {'attacker_validation': rows}}
    h_release = {'attacker_validation': {'aux': None, 'token_probs': np.ones((4, 1))}}
    q_release = {'attacker_validation': release}
    report = replay.verify_ancestor_parity(baseline, current, ctx, h_release, q_release)
    assert report['passed'] is True
    assert report['roles']['attack:AB/SEX']['attacker_validation']['candidate_parity_comparisons'] == 6
    del current['roles']['attack:AB/SEX']['candidates']['ancestor_A__model']
    with pytest.raises(replay.ReplayError, match='singleton ancestor is missing'):
        replay.verify_ancestor_parity(baseline, current, ctx, h_release, q_release)

"""The audit estimand is expected log loss over actual categorical tokens."""
import json

import numpy as np
import pytest
import torch

from experiments.pcrl_task_directed_release_v1 import audits


def test_expected_token_loss_is_not_loss_of_averaged_predictions():
    q = np.array([[[.9, .1], [.1, .9]], [[.8, .2], [.3, .7]]])
    p = np.array([[.5, .5], [.25, .75]])
    y = np.array([0, 1])
    got = audits.expected_token_loss(q, p, y)
    np.testing.assert_allclose(got, [-.5*np.log(.9)-.5*np.log(.1),
                                   -.25*np.log(.2)-.75*np.log(.7)])
    assert got[0] > -np.log(.5)
    duplicated_q = np.repeat(q, 2, axis=1)
    duplicated_p = np.repeat(p/2, 2, axis=1)
    np.testing.assert_allclose(audits.expected_token_loss(duplicated_q, duplicated_p, y), got)


def test_exact_expansion_preserves_each_person_weight_and_original_ids():
    p = np.array([[.75, .25, 0.], [0., .2, .8]])
    person, token, weight = audits.expand_person_tokens(p, np.array([2., 7.]))
    np.testing.assert_array_equal(person, [0, 0, 1, 1])
    np.testing.assert_array_equal(token, [0, 1, 1, 2])
    np.testing.assert_allclose(weight, [1.5, .5, 1.4, 5.6])
    np.testing.assert_allclose(np.bincount(person, weights=weight), [2., 7.])


def test_original_person_normalization_does_not_change_when_tokens_duplicate():
    logits = torch.tensor([[2., 0.], [0., 2.], [1., 0.]], requires_grad=True)
    y = torch.tensor([0, 0, 1])
    w = torch.tensor([.75, .25, 3.])
    got = audits.weighted_person_cross_entropy(logits, y, w, n_persons=2)
    hand = (.75*np.log1p(np.exp(-2)) + .25*np.log1p(np.exp(2)) +
            3*np.log1p(np.exp(1))) / 2
    assert got.item() == pytest.approx(hand)
    repeated = audits.weighted_person_cross_entropy(
        logits.repeat_interleave(3, dim=0), y.repeat_interleave(3),
        w.repeat_interleave(3)/3, n_persons=2)
    assert repeated.item() == pytest.approx(got.item())
    got.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('bad', [np.array([[np.nan, 0.]]), np.array([[-.1, 1.1]]),
                                np.array([[.2, .2]]), np.array([[np.inf, 0.]])])
def test_corrupt_token_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        audits.validate_inputs(np.ones((1, 1)), bad, [0], [1.], 2)


@pytest.mark.parametrize('changes', [
    {'h': [[np.nan], [1.]]}, {'y': [0, 2]}, {'y': [0., .5]},
    {'weights': [1., -1.]}, {'weights': [0., 0.]},
    {'ids': ['duplicate', 'duplicate']}, {'ids': ['ok', None]},
    {'ids': [1, 1.]},
    {'ids': [1]}, {'class_order': [1, 0]},
])
def test_corrupt_aligned_inputs_rejected(changes):
    values = dict(h=np.ones((2, 1)), token_probs=np.ones((2, 1)),
                  y=[0, 1], weights=[1., 1.], n_classes=2,
                  ids=['a', 'b'], class_order=[0, 1])
    values.update(changes)
    with pytest.raises(ValueError):
        audits.validate_inputs(**values)


def test_same_prediction_selection_uses_balanced_weighting_and_lexical_ties():
    losses = {'z': np.array([.1, 1.]), 'a': np.array([.1, 1.]),
              'weighted_only_winner': np.array([3., .01])}
    result = audits.select_losses(losses, [1., 9.])
    assert result['selection'] == 'a'
    assert result['scores']['a']['unweighted'] == pytest.approx(.55)
    assert result['scores']['a']['weighted'] == pytest.approx(.91)
    assert result['scores']['a']['balanced'] == pytest.approx(.73)
    assert result['selection'] == audits.select_losses(losses, [100., 900.])['selection']
    np.testing.assert_allclose(audits.balanced_person_weights([1., 9.]), [.6, 1.4])


def test_weight_scale_invariance_avoids_intermediate_overflow_and_underflow():
    score = audits.loss_scores([10., 20.], [1e307, 1e308])
    assert score['weighted'] == pytest.approx(210/11)
    tiny = np.nextafter(0., 1.)
    np.testing.assert_allclose(audits.balanced_person_weights([tiny, 0., 0.]), [2., .5, .5])


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    rng = np.random.default_rng(88)
    h = rng.normal(size=(35, 2))
    y = np.where(h[:, 0] > 0, 2, 0)  # class 1 is absent from fitting.
    p = np.column_stack([np.linspace(.05, .95, len(h)), np.linspace(.95, .05, len(h))])
    w = np.linspace(1., 5., len(h))
    hv = rng.normal(size=(12, 2))
    pv = np.full((12, 2), .5)
    yv = np.arange(12) % 3
    wv = np.arange(1., 13.)
    out = tmp_path_factory.mktemp('weighted_audits')
    result = audits.fit_slate(h, p, y, w, hv, pv, yv, wv, 3, 99, out/'models')
    return result, (h, p, y, w, hv, pv, yv, wv), out


def test_all_audit_families_keep_full_schema_and_reload_exactly(fitted):
    result, (_, _, _, _, hv, pv, yv, _), _ = fitted
    assert set(result['candidates']) == {'logistic', 'mlp_120', 'hist_gb_20', 'hist_gb_5',
                                         'sampled_hist_gb_20', 'sampled_hist_gb_5'}
    for cid, candidate in result['candidates'].items():
        q = audits.predict_token_proba(candidate, hv, 2)
        assert q.shape == (12, 2, 3)
        assert np.isfinite(q).all() and (q > 0).all()
        np.testing.assert_allclose(q.sum(2), 1., atol=1e-12)
        loaded = audits.load_candidate(candidate.metadata['directory'])
        np.testing.assert_array_equal(audits.predict_token_proba(loaded, hv, 2), q)
        np.testing.assert_allclose(audits.expected_loss(candidate, hv, pv, yv),
                                   audits.expected_token_loss(q, pv, yv))
        assert candidate.metadata['class_order'] == [0, 1, 2]
    json.dumps(result['metadata'], allow_nan=False)


def test_fitting_scaler_uses_original_people_and_mlp_schedule_counts_people(fitted):
    result, (h, _, _, _, _, _, _, _), _ = fitted
    for candidate in result['candidates'].values():
        np.testing.assert_allclose(candidate.mean, h.mean(0))
    mlp = result['candidates']['mlp_120']
    assert mlp.metadata['optimizer_steps'] == 120
    assert mlp.metadata['training_person_exposures'] == 120*35
    assert mlp.metadata['training_expanded_exposures'] == 120*70
    assert mlp.metadata['batch_size_persons'] == 256
    curve = mlp.metadata['validation_curve']
    chosen = min(curve, key=lambda row: (row['balanced'], row['epoch']))
    assert mlp.metadata['selected_epoch'] == chosen['epoch']
    assert len(list((__import__('pathlib').Path(mlp.metadata['directory']).parent/
                     'mlp_trajectory').glob('epoch_*.pt'))) == len(curve)
    tree = result['candidates']['hist_gb_20']
    assert tree.model.min_samples_leaf == 40
    assert tree.metadata['minimum_distinct_persons_per_leaf'] == 20


def test_sampled_trees_use_one_fixed_original_person_draw_and_exact_validation(fitted):
    result, (h, p, _, _, hv, pv, yv, wv), out = fitted
    with np.load(out/'models'/'histgb_training_draw.npz') as draw:
        assert draw['tokens'].shape == (35,)
        assert draw['uniforms'].shape == (35,)
        expected_uniforms = np.random.default_rng(20260921+99).random(35)
        np.testing.assert_array_equal(draw['uniforms'], expected_uniforms)
        np.testing.assert_array_equal(draw['tokens'], (expected_uniforms >= p[:, 0]).astype(int))
    for leaf in (5, 20):
        model = result['candidates'][f'sampled_hist_gb_{leaf}']
        assert model.model.min_samples_leaf == leaf
        assert model.metadata['training_expanded_exposures'] == len(h)
        assert model.metadata['sampling_seed'] == 20260921+99
        expected = audits.loss_scores(audits.expected_loss(model, hv, pv, yv), wv)
        assert model.metadata['validation_scores'] == expected


def test_catchup_keeps_standard_prefix_and_actual_final_state(tmp_path):
    h = np.array([[-1.], [1.], [-2.], [2.]])
    p = np.ones((4, 1))
    y = np.array([0, 1, 0, 1])
    result = audits.fit_slate(h, p, y, np.ones(4), h, p, y, np.ones(4),
                              2, 7, tmp_path/'catchup', slate='catchup')
    short, long = (result['candidates'][key] for key in ('mlp_120', 'mlp_360'))
    assert long.metadata['validation_curve'][:len(short.metadata['validation_curve'])] == short.metadata['validation_curve']
    assert long.metadata['optimizer_steps'] == 360
    assert long.metadata['validation_scores']['balanced'] <= short.metadata['validation_scores']['balanced']
    for epoch in (120, 360):
        checkpoint = torch.load(tmp_path/'catchup'/'mlp_trajectory'/f'boundary_{epoch:04d}.pt',
                                map_location='cpu', weights_only=False)
        assert checkpoint['optimizer_steps'] == epoch
        assert checkpoint['epoch'] == epoch
        assert checkpoint['optimizer_state']['state']
        assert checkpoint['schedule_rng_state']
    for leaf in (5, 20):
        exact = result['candidates'][f'hist_gb_{leaf}']
        sampled = result['candidates'][f'sampled_hist_gb_{leaf}']
        assert sampled.metadata['reused_deterministic_exact_tree'] is True
        np.testing.assert_array_equal(sampled.predict_proba(h), exact.predict_proba(h))


def test_teacher_slate_selects_saved_deterministic_predictor(tmp_path):
    x = np.linspace(-2., 2., 18)[:, None]
    y = (x[:, 0] > 0).astype(int)
    result = audits.fit_predictor_slate(x, y, np.ones(18), x, y, np.ones(18),
                                        2, 91, tmp_path/'teacher')
    assert set(result['candidates']) == {'logistic_C0.03', 'logistic_C0.3', 'logistic_C3',
                                         'mlp_wd0_200', 'mlp_wd0.0001_200'}
    selected = result['candidates'][result['selection']]
    q = selected.predict_proba(x)
    assert q.shape == (18, 2)
    assert np.mean(q.argmax(1) == y) > .9
    np.testing.assert_array_equal(audits.load_candidate(selected.metadata['directory']).predict_proba(x), q)


@pytest.mark.parametrize('bad', [
    np.array([[[np.nan, 0.], [.5, .5]]]),
    np.array([[[-.1, 1.1], [.5, .5]]]),
    np.array([[[.2, .2], [.5, .5]]]),
])
def test_corrupt_conditional_predictions_rejected(bad):
    with pytest.raises(ValueError):
        audits.expected_token_loss(bad, [[.5, .5]], [0])


def test_fit_rejects_validation_schema_mismatch_and_existing_artifacts(tmp_path):
    h, y, w = np.ones((4, 1)), np.array([0, 1, 0, 1]), np.ones(4)
    with pytest.raises(ValueError):
        audits.fit_slate(h, np.ones((4, 1)), y, w, h, np.full((4, 2), .5), y, w,
                         2, 1, tmp_path/'schema')
    out = tmp_path/'existing'
    out.mkdir()
    (out/'protected').write_text('keep')
    with pytest.raises(FileExistsError):
        audits.fit_slate(h, np.ones((4, 1)), y, w, h, np.ones((4, 1)), y, w,
                         2, 1, out)
    assert (out/'protected').read_text() == 'keep'

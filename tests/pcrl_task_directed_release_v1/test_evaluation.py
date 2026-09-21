"""Release routing, inherited-candidate closure, and frozen scoring contracts."""
from pathlib import Path

import numpy as np
import pytest

from experiments.pcrl_task_directed_release_v1 import audits, evaluation
from experiments.pcrl_task_directed_release_v1.config import PRIMARY, SECONDARY, UTILITY


class ConditionalModel:
    """Known conditional probabilities exercise the real TokenCandidate route."""
    def __init__(self, n_classes, p0, n_tokens):
        self.classes_ = np.arange(n_classes)
        self.p0, self.n_tokens = p0, n_tokens

    def predict_proba(self, x):
        if self.n_tokens == 1:
            p0 = np.full(len(x), self.p0)
        else:
            p0 = self.p0 + .1*x[:, -1]
        q = np.repeat(((1-p0)/(len(self.classes_)-1))[:, None], len(self.classes_), axis=1)
        q[:, 0] = p0
        return q


def tiny_context():
    pools = {}
    targets = {role.split('/')[1] for role in PRIMARY+SECONDARY+UTILITY}
    for i, pool in enumerate(('downstream_fit', 'attacker_fit', 'attacker_validation',
                               'downstream_validation', 'test')):
        n = 6
        pools[pool] = {'ha': np.ones((n, 4)), 'hb': np.ones((n, 2)),
                       'ids': np.array([f'{i}:person{j}' for j in range(n)]),
                       'households': np.array([f'{i}:house{j//2}' for j in range(n)]),
                       'weights': np.arange(1., n+1),
                       'labels': {target: np.zeros(n, dtype=int) for target in targets}}
    return {'anchor': 1, 'pools': pools}


def release_for(ctx, k):
    return {pool: {'aux': None, 'token_probs': np.full((len(data['ids']), k), 1/k)}
            for pool, data in ctx['pools'].items()}


@pytest.fixture
def fast_fitter(monkeypatch):
    calls = []

    def fit(h, p, y, w, hv, pv, yv, wv, n_classes, seed, out_dir, *, slate):
        calls.append((h.shape[1], p.shape[1], seed))
        p0 = {2: .7, 4: .6, 6: .5}[h.shape[1]] if p.shape[1] == 1 else (.85 if h.shape[1] == 4 else .2)
        candidates = {}
        for cid in ('logistic', 'mlp_120', 'mlp_360', 'hist_gb_20', 'hist_gb_5',
                    'sampled_hist_gb_20', 'sampled_hist_gb_5'):
            candidate = audits.TokenCandidate('logistic', ConditionalModel(n_classes, p0, p.shape[1]),
                np.zeros(h.shape[1]), np.ones(h.shape[1]), p.shape[1], n_classes, False,
                {'class_order': list(range(n_classes)), 'input_dim': h.shape[1]+p.shape[1],
                 'candidate_id': cid})
            candidate.save(Path(out_dir)/cid)
            candidates[cid] = candidate
        return {'candidates': candidates}

    monkeypatch.setattr(evaluation, 'fit_slate', fit)
    return calls


def test_actual_singletons_and_H_candidates_are_routed_and_B_fits_reused(tmp_path, fast_fitter):
    ctx = tiny_context()
    ctx['pools'].pop('test')
    base = evaluation.fit_release_audits('H', ctx, release_for(ctx, 1), tmp_path/'H')
    assert len(fast_fitter) == 16
    release = release_for(ctx, 2)
    new = evaluation.fit_release_audits('Q', ctx, release, tmp_path/'Q', h_baseline=base, b_baseline=base)
    assert len(fast_fitter) == 25  # all seven unchanged B roles reused.
    role = new['registry']['roles']['attack:AB/SEX']
    selected = role['candidates'][role['selection']]
    assert selected['route']['source_view'] == 'A'
    assert selected['route']['wire'] == 'release'
    assert selected['origin'] == 'A_ancestor'
    assert len(role['candidates']) >= 25
    assert role['selection'] != role['independent_selection']
    assert new['summary']['attack:B/same_residence']['reused_B'] is True
    assert all('J' not in record['route'].values() for record in role['candidates'].values())
    with np.load(role['validation_loss_path']) as saved:
        assert saved['losses'].shape == (6, len(role['candidates']))
        np.testing.assert_array_equal(saved['ids'], ctx['pools']['attacker_validation']['ids'])
    h_record = next(record for cid, record in role['candidates'].items() if cid.startswith('H__'))
    q, p = evaluation.routed_probabilities(h_record, ctx['pools']['attacker_validation'],
                                           release['attacker_validation'])
    assert q.shape[1] == p.shape[1] == 1


def test_frozen_evaluation_never_fits_or_reselects_and_preserves_expected_loss(tmp_path, fast_fitter, monkeypatch):
    ctx = tiny_context()
    test = ctx['pools'].pop('test')
    trained = evaluation.fit_release_audits('Q', ctx, release_for(ctx, 2), tmp_path/'fit')
    frozen = {'anchor': 1, 'pools': {'test': test}}
    monkeypatch.setattr(evaluation, 'fit_slate', lambda *a, **k: pytest.fail('Frozen evaluation fit a model'))
    monkeypatch.setattr(evaluation, 'select_losses', lambda *a, **k: pytest.fail('Frozen evaluation reselected'))
    result = evaluation.evaluate_frozen_audits(trained['registry_path'], frozen,
                                               release_for(frozen, 2), tmp_path/'eval')
    path = result['artifacts']['test']['attack:AB/SEX']
    with np.load(path) as saved:
        q, p = saved['probabilities'], saved['token_probs']
        hand = (p * -np.log(q[:, :, 0])).sum(1)
        np.testing.assert_allclose(saved['loss'], hand)
        assert np.all(hand > -np.log((p*q[:, :, 0]).sum(1)))
        np.testing.assert_allclose(saved['accuracy'], 1.)
    summary = result['summary']['test']['attack:AB/SEX']
    assert summary['per_class'][0]['support'] == 6
    assert summary['per_class'][1]['ce']['unweighted'] is None
    assert summary['ess'] == pytest.approx(21**2/91)


def test_fixed_decoder_and_H_offset_candidates_are_distinct_and_compatible(tmp_path, fast_fitter):
    ctx = tiny_context()
    ctx['pools'].pop('test')
    release = release_for(ctx, 2)
    offsets = {}
    for pool, value in release.items():
        value['fixed_probabilities'] = np.tile([.2, .3], (6, 1))
        offsets[pool] = np.tile([.1, .001], (6, 1))
    fitted = evaluation.fit_release_audits('Q', ctx, release, tmp_path/'Q', global_offsets=offsets)
    residence = fitted['registry']['roles']['utility:A/same_residence']
    assert residence['selection'] == 'global_offset_0001'
    assert 'fixed_decoder' in residence['candidates']
    assert fitted['summary']['utility:A/same_residence']['fixed_decoder']['balanced'] > .2
    denied = fitted['registry']['roles']['attack:B/same_residence']['candidates']
    assert not any('offset' in key or 'decoder' in key for key in denied)


def test_fit_rejects_household_overlap_or_corrupt_release_before_models(tmp_path, fast_fitter):
    ctx = tiny_context()
    ctx['pools'].pop('test')
    ctx['pools']['attacker_validation']['households'][0] = ctx['pools']['downstream_fit']['households'][0]
    with pytest.raises(ValueError, match='household'):
        evaluation.fit_release_audits('bad', ctx, release_for(ctx, 1), tmp_path/'bad')
    assert not fast_fitter
    ctx = tiny_context()
    ctx['pools'].pop('test')
    release = release_for(ctx, 2)
    release['attacker_fit']['token_probs'][0] = [.1, .1]
    with pytest.raises(ValueError):
        evaluation.fit_release_audits('bad', ctx, release, tmp_path/'bad')
    assert not fast_fitter


def test_missing_target_labels_are_masked_with_ids_weights_and_probabilities(tmp_path, fast_fitter):
    ctx = tiny_context()
    ctx['pools'].pop('test')
    ctx['pools']['attacker_validation']['labels']['SEX'][1] = -1
    fitted = evaluation.fit_release_audits('H', ctx, release_for(ctx, 1), tmp_path/'masked')
    role = fitted['registry']['roles']['attack:A/SEX']
    with np.load(role['validation_loss_path']) as saved:
        np.testing.assert_array_equal(saved['weights'], [1., 3., 4., 5., 6.])
        assert len(saved['ids']) == len(saved['y']) == len(saved['losses']) == 5


def test_identifier_serialization_cannot_collapse_distinct_people(tmp_path, fast_fitter):
    ctx = tiny_context()
    ctx['pools'].pop('test')
    ctx['pools']['downstream_fit']['ids'] = np.array([1, '1', 'x', 'y', 'z', 'w'], dtype=object)
    with pytest.raises(ValueError):
        evaluation.fit_release_audits('bad_ids', ctx, release_for(ctx, 1), tmp_path/'ids')
    assert not fast_fitter

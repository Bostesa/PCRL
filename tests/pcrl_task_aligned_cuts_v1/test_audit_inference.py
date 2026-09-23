"""Focused independent scorer and registered-family fixtures."""
import numpy as np
import pytest
import hashlib
import shutil

from experiments.pcrl_task_aligned_cuts_v1.audit import (
    assert_household_disjoint, expected_token_loss, score_weightings,
    select_validation, validate_token_probs, view_features,
)
from experiments.pcrl_task_aligned_cuts_v1.inference import (
    capability_endpoints, contrasts_from_scores, enumerate_endpoints,
    evaluate_family, family_manifest,
)


def test_expected_token_loss_is_not_loss_of_averaged_predictions():
    p = np.array([[.5, .5]])
    q = np.array([[[.9, .1], [.1, .9]]])
    actual = expected_token_loss(q, p, np.array([1]))[0]
    assert actual == pytest.approx(-.5 * np.log(.1) - .5 * np.log(.9))
    assert actual > -np.log(.5)


def test_token_relabeling_is_invariant_when_predictor_is_relabelled():
    p = np.array([[.1, .3, .6], [.7, .2, .1]])
    q = np.array([[[.4, .6], [.8, .2], [.1, .9]],
                  [[.6, .4], [.2, .8], [.9, .1]]])
    y = np.array([1, 0])
    order = [2, 0, 1]
    np.testing.assert_allclose(expected_token_loss(q, p, y),
                               expected_token_loss(q[:, order], p[:, order], y), atol=0, rtol=1e-14)


def test_zero_token_row_and_incomplete_class_schema_fail():
    with pytest.raises(ValueError):
        validate_token_probs(np.zeros((2, 17)))
    with pytest.raises(ValueError):
        expected_token_loss(np.ones((1, 2, 1)), np.array([[.5, .5]]), np.array([0]))


def test_recipient_views_do_not_give_hb_to_local_a():
    ha = np.arange(8).reshape(2, 4)
    hb = np.array([[10., 11.], [12., 13.]])
    assert view_features(ha, hb, 'A').shape == (2, 4)
    assert view_features(ha, hb, 'AB').shape == (2, 6)
    np.testing.assert_array_equal(view_features(ha, hb, 'B'), hb)


def test_continuous_aux_is_on_released_a_and_ab_wires_only():
    from experiments.pcrl_task_aligned_cuts_v1.audit import routed_features
    rows = {'ha': np.zeros((2, 4)), 'hb': np.ones((2, 2)),
            'aux': np.full((2, 3), 7.)}
    assert routed_features(rows, 'A', 'release').shape == (2, 7)
    assert routed_features(rows, 'AB', 'release').shape == (2, 9)
    assert routed_features(rows, 'A', 'H').shape == (2, 4)
    assert routed_features(rows, 'B', 'H').shape == (2, 2)
    with pytest.raises(ValueError):
        routed_features(rows, 'B', 'release')


def test_household_disjointness_and_validation_selection():
    with pytest.raises(ValueError):
        assert_household_disjoint(np.array(['h1', 'h2']), np.array(['h2', 'h3']))
    answer = select_validation({'b': np.array([.2, .2]), 'a': np.array([.2, .2])}, np.ones(2))
    assert answer['selected'] == 'a'
    assert score_weightings(np.array([0., 1.]), np.array([1., 3.]))['PWGTP'] == pytest.approx(.75)


def test_primary_family_signs_counts_and_aliases():
    slots = [{'id': 'M', 'comparators': ['D17', 'D_U1', 'D17_alias', 'det', 'grad']}]
    family = enumerate_endpoints(slots, alias_of={'D17_alias': 'D17'})
    assert len(family) == 40  # 4 distinct comparators x 5 roles x 2 weights
    assert family_manifest(slots, alias_of={'D17_alias': 'D17'})['n_endpoints'] == 40
    task = next(e for e in family if e['comparator'] == 'D17' and e['clause'] == 'task')
    race = next(e for e in family if e['comparator'] == 'D17' and e['clause'] == 'target')
    assert (task['plus'], task['minus'], task['threshold']) == ('M', 'D17', .001)
    assert (race['plus'], race['minus'], race['threshold']) == ('D17', 'M', -.002)
    assert race['comparator_names'] == ['D17', 'D17_alias']
    assert len(capability_endpoints(['M'])) == 2


def test_identical_prediction_pairs_give_zero_household_difference():
    role = 'utility:A/same_residence'
    endpoint = {'id': 'x', 'plus': 'M', 'minus': 'D17', 'role': role, 'weighting': 'U',
                'candidate': 'M', 'comparator': 'D17', 'threshold': .001}
    record = {'ids': np.array(['a', 'b', 'c', 'd']),
              'households': np.array(['h1', 'h1', 'h2', 'h3']),
              'weights': np.array([1., 2., 1., 1.]), 'loss': np.array([.2, .3, .4, .1])}
    scores = {(name, anchor, role): record for name in ('M', 'D17') for anchor in (0, 1, 2)}
    contrasts = contrasts_from_scores([endpoint], scores)
    assert all(np.array_equal(a['difference'], np.zeros(4)) for a in contrasts[0]['anchors'])
    result = evaluate_family([endpoint], scores, n_boot=100, seed=17)
    assert result['rows'][0]['estimate'] == 0
    assert result['rows'][0]['bootstrap_se'] == 0


def test_paired_masks_must_match():
    role = 'attack:A/SEX'
    endpoint = {'id': 'x', 'plus': 'D17', 'minus': 'M', 'role': role, 'weighting': 'U'}
    a = {'ids': np.array(['a', 'b']), 'households': np.array(['h1', 'h2']),
         'weights': np.ones(2), 'loss': np.ones(2)}
    b = {**a, 'ids': np.array(['b', 'a'])}
    scores = {(name, anchor, role): a if name == 'D17' else b
              for name in ('D17', 'M') for anchor in (0, 1, 2)}
    with pytest.raises(ValueError):
        contrasts_from_scores([endpoint], scores)


def test_full_view_xor_can_reveal_what_coarse_or_local_view_misses():
    # One person for every (S,H) cell; T=S xor H and Z=T. Marginal S,Z are
    # independent, yet H+Z reveals S exactly. A coalition version assigns H
    # to H_B so A sees no increment while AB does.
    s = np.array([0, 0, 1, 1])
    h = np.array([0, 1, 0, 1])
    z = np.bitwise_xor(s, h)
    p = np.eye(2)[z]
    marginal = np.full((4, 2, 2), .5)
    local_full = np.zeros((4, 2, 2))
    for i in range(4):
        for token in range(2):
            local_full[i, token, h[i] ^ token] = 1
    assert expected_token_loss(marginal, p, s).mean() == pytest.approx(np.log(2))
    assert expected_token_loss(local_full, p, s).mean() == pytest.approx(0, abs=1e-12)
    # Reinterpreting h as H_B, the same tensor is legal only to AB, not A.
    ha = np.zeros((4, 4))
    hb = np.column_stack((h, np.zeros(4)))
    assert view_features(ha, hb, 'A').shape[1] == 4
    assert view_features(ha, hb, 'AB').shape[1] == 6


def test_identical_task_and_sensitive_labels_have_no_selective_gain():
    y = np.array([0, 1])
    p = np.array([[.8, .2], [.4, .6]])
    q = np.array([[[.9, .1], [.6, .4]], [[.2, .8], [.4, .6]]])
    task = expected_token_loss(q, p, y)
    sensitive = expected_token_loss(q, p, y.copy())
    np.testing.assert_array_equal(task, sensitive)


def test_constant_token_is_no_increment_under_exact_law():
    s = np.array([0, 1, 0, 1])
    p = np.tile(np.array([.2, .8]), (4, 1))
    q = np.full((4, 2, 2), .5)
    loss = expected_token_loss(q, p, s)
    np.testing.assert_allclose(loss, np.log(2), atol=1e-15)


def test_exact_candidate_alias_remains_in_family_and_fails_material_target():
    family = enumerate_endpoints([{'id': 'M', 'comparators': ['D17']}],
                                 alias_of={'M': 'same-release', 'D17': 'same-release'})
    assert len(family) == 10
    assert any(e['clause'] == 'target' and e['threshold'] < 0 for e in family)


def test_independent_score_matches_pinned_historical_scorer():
    from experiments.pcrl_task_directed_release_v1.audits import expected_token_loss as historical
    rng = np.random.default_rng(29)
    p = rng.dirichlet(np.ones(17), size=13)
    q = rng.dirichlet(np.ones(9), size=(13, 17))
    y = rng.integers(0, 9, size=13)
    np.testing.assert_allclose(expected_token_loss(q, p, y), historical(q, p, y), atol=1e-14, rtol=0)


def test_fixed_binary_decoder_keeps_expected_token_loss():
    from experiments.pcrl_task_aligned_cuts_v1.audit import score_fixed_binary_decoder
    p = np.array([[.5, .5]])
    positive = np.array([[.1, .9]])
    result = score_fixed_binary_decoder(positive, p, np.array([1]), np.ones(1))
    assert result['loss'][0] == pytest.approx(-.5 * np.log(.1) - .5 * np.log(.9))


def test_fixed_decoder_route_reads_only_public_a_predictions():
    from experiments.pcrl_task_aligned_cuts_v1.audit import predict_frozen_route, _legal_route
    route = {'kind': 'fixed_decoder', 'target': 'same_residence', 'source_view': 'A',
             'wire': 'release', 'source_release_id': 'M'}
    rows = {'ha': np.zeros((1, 4)), 'hb': np.ones((1, 2)),
            'fixed_probabilities': np.array([[.1, .9]])}
    p = np.array([[.5, .5]])
    _legal_route('utility:A/same_residence', route, release_id='M')
    q, actual_p = predict_frozen_route(route, rows, p)
    np.testing.assert_array_equal(actual_p, p)
    assert expected_token_loss(q, actual_p, np.array([1]))[0] == pytest.approx(
        -.5*np.log(.1)-.5*np.log(.9))
    with pytest.raises(ValueError):
        _legal_route('attack:A/SEX', route, release_id='M')


def test_declared_race_schema_reports_unsupported_classes():
    from experiments.pcrl_task_aligned_cuts_v1.audit import role_arrays
    rows = {'ha': np.zeros((2, 4)), 'hb': np.zeros((2, 2)),
            'ids': np.array(['p1', 'p2']), 'households': np.array(['h1', 'h2']),
            'weights': np.ones(2), 'labels': {'RAC1P': np.array([0, 1])}}
    p = np.ones((2, 1))
    result = role_arrays(rows, p, 'attack:A/RAC1P')
    assert result['n_classes'] == 9
    assert result['missing_classes'] == list(range(2, 9))
    with pytest.raises(ValueError, match='Missing fitting support'):
        role_arrays(rows, p, 'attack:A/RAC1P', require_full_fit_support=True)


def test_inherited_static_auditor_keeps_absent_race_class_coordinate():
    from sklearn.linear_model import LogisticRegression
    from experiments.pcrl_task_directed_release_v1.audits import TokenCandidate
    x = np.array([[-2.], [-1.], [1.], [2.]])
    y = np.array([0, 0, 1, 1])  # class 3, and seven others, absent
    model = LogisticRegression().fit(x, y)
    candidate = TokenCandidate('logistic', model, np.zeros(1), np.ones(1), 1, 9,
                               False, {'class_order': list(range(9))})
    prediction = candidate.predict_token_proba(x, 1)
    assert prediction.shape == (4, 1, 9)
    assert np.all(prediction[:, 0, 3] > 0)
    np.testing.assert_allclose(prediction.sum(axis=2), 1., atol=1e-14)


def test_equal_alphabet_does_not_license_cross_release_predictor_transfer():
    from experiments.pcrl_task_aligned_cuts_v1.audit import _legal_route
    route = {'kind': 'model', 'target': 'SEX', 'source_view': 'A',
             'wire': 'release', 'source_release_id': 'Q', 'model_sha256': 'a'*64}
    with pytest.raises(ValueError, match='Token semantics differ'):
        _legal_route('attack:A/SEX', route, release_id='D17')
    _legal_route('attack:A/SEX', route, release_id='Q')


def test_complete_coalition_route_bank_requires_legal_ancestors():
    from experiments.pcrl_task_aligned_cuts_v1.audit import build_role_route_bank
    def reg(role, release, view, wire):
        return {'role': role, 'release_id': release, 'slate': 'catchup', 'models': {
                'logistic': {'kind': 'model', 'target': 'SEX', 'source_view': view,
                         'wire': wire, 'source_release_id': release,
                         'model_sha256': 'a'*64,
                         'model_directory': '/private/frozen/logistic'}}}
    own = reg('attack:AB/SEX', 'Q', 'AB', 'release')
    h = reg('attack:AB/SEX', 'H', 'AB', 'H')
    a = reg('attack:A/SEX', 'Q', 'A', 'release')
    b = reg('attack:B/SEX', 'H', 'B', 'H')
    bank = build_role_route_bank('attack:AB/SEX', 'Q', own, h, a_same=a, b_h_only=b)
    assert set(bank) == {'own/logistic', 'H/logistic', 'A/logistic', 'B/logistic'}
    with pytest.raises(ValueError, match='coalition ancestors'):
        build_role_route_bank('attack:AB/SEX', 'Q', own, h)
    with pytest.raises(ValueError, match='A coalition ancestor role or release differs'):
        build_role_route_bank('attack:AB/SEX', 'Q', own, h,
                              a_same=reg('attack:A/SEX', 'D17', 'A', 'release'), b_h_only=b)
    different_slate = reg('attack:B/SEX', 'H', 'B', 'H')
    different_slate['slate'] = 'standard'
    with pytest.raises(ValueError, match='same declared slate'):
        build_role_route_bank('attack:AB/SEX', 'Q', own, h,
                              a_same=a, b_h_only=different_slate)


def test_frozen_predictor_replay_rejects_changed_model_bytes(tmp_path):
    from experiments.pcrl_task_aligned_cuts_v1.audit import model_directory_hash, predict_model_route
    directory = tmp_path / 'candidate'
    directory.mkdir()
    path = directory / 'model.joblib'
    path.write_bytes(b'first frozen bytes')
    pinned = model_directory_hash(directory)
    path.write_bytes(b'changed frozen bytes')
    rows = {'ha': np.zeros((1, 4)), 'hb': np.zeros((1, 2))}
    with pytest.raises(ValueError, match='hash mismatch'):
        predict_model_route(directory, rows, np.ones((1, 1)),
                            source_view='A', wire='H', expected_model_sha256=pinned)


def test_keyed_one_release_replays_across_descriptive_channel_aliases():
    from experiments.pcrl_task_aligned_cuts_v1.release import ChannelArtifact, OneReleaseSession
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
    class Code:
        def n_states(self, name):
            assert name == 'T0'
            return 32
    class Encoder:
        code = Code()
        def encode(self, inputs):
            return {'codes': {'T0': np.zeros(len(inputs.h_a), dtype=int)}}
    q = np.full((32, 17), 1/17)
    first = ChannelArtifact(q, '0'*64, 'primary-name')
    alias = ChannelArtifact(q.copy(), '0'*64, 'same-map-alias')
    a = OneReleaseSession(first, Encoder(), replay_key=b'k'*32)
    b = OneReleaseSession(alias, Encoder(), replay_key=b'k'*32)
    assert a._channel_digest == b._channel_digest
    h = np.array([[1., 2., 3., 4.]])
    inputs = RuntimeInputs(np.zeros((1, 32)), h)
    one = a.emit(inputs, ['person-1'])
    two = b.emit(inputs, ['person-1'])
    np.testing.assert_array_equal(one['token'], two['token'])
    assert one['h_a'].tobytes() == h.tobytes()


def test_complete_audit_slate_reuses_exact_inputs_and_rejects_tamper(tmp_path, monkeypatch):
    from experiments.pcrl_task_aligned_cuts_v1.audit import fit_role_slate
    from experiments.pcrl_task_directed_release_v1 import audits as inherited
    def rows(prefix):
        return {'ha': np.zeros((2, 4)), 'hb': np.zeros((2, 2)),
                'labels': {'SEX': np.array([0, 1])}, 'weights': np.ones(2),
                'ids': np.array([f'{prefix}1', f'{prefix}2']),
                'households': np.array([f'{prefix}h1', f'{prefix}h2'])}
    fit_rows, val_rows = rows('fit'), rows('val')
    token = np.ones((2, 1))
    calls = []
    def fake_fit_slate(*args, slate, **kwargs):
        calls.append(slate)
        out_dir = args[-1]
        path = out_dir / 'logistic'
        path.mkdir(parents=True)
        (path / 'model.joblib').write_bytes(b'pinned synthetic model')
        (out_dir / 'slate.json').write_text('{"selection":"logistic"}')
        candidate = type('Candidate', (), {'metadata': {'directory': str(path)}})()
        return {'candidates': {'logistic': candidate}, 'selection': 'logistic',
                'metadata': {'validation_scores': {'logistic': {'balanced': .1}}}}
    monkeypatch.setattr(inherited, 'fit_slate', fake_fit_slate)
    output = tmp_path / 'private' / 'slate'
    one = fit_role_slate(fit_rows, token, val_rows, token,
                         'attack:A/SEX', output, 13, release_id='H', slate='standard')
    two = fit_role_slate(fit_rows, token, val_rows, token,
                         'attack:A/SEX', output, 13, release_id='H', slate='standard')
    assert one == two
    assert calls == ['standard']
    restored = tmp_path / 'private' / 'restored-slate'
    shutil.copytree(output, restored)
    replayed = fit_role_slate(fit_rows, token, val_rows, token,
                              'attack:A/SEX', restored, 13, release_id='H', slate='standard')
    assert replayed['models']['logistic']['model_directory'] == str((restored / 'models' / 'logistic').resolve())
    assert calls == ['standard']
    changed = rows('fit')
    changed['labels']['SEX'][0] = 1
    with pytest.raises(ValueError, match='fit_input_sha256'):
        fit_role_slate(changed, token, val_rows, token,
                       'attack:A/SEX', output, 13, release_id='H', slate='standard')
    (output / 'models' / 'logistic' / 'model.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='model hash differs'):
        fit_role_slate(fit_rows, token, val_rows, token,
                       'attack:A/SEX', output, 13, release_id='H', slate='standard')


def test_public_synthetic_fit_to_release_to_exact_audit():
    from experiments.pcrl_task_aligned_cuts_v1.audit import _synthetic_fit_release_audit
    result = _synthetic_fit_release_audit()
    assert result['status'] == 'pass'
    assert result['scope'] == 'public synthetic data only'
    assert result['fitted_channel_shape'] == [32, 17]
    assert result['fixed_bank_objective'] == pytest.approx(.5, abs=1e-8)
    assert result['fixed_bank_dual_lower_bound'] == pytest.approx(.5, abs=1e-7)
    assert result['service_byte_parity']
    assert result['n_original_people'] == 4
    assert 0 < result['independent_exact_task_loss_U'] < 2


def test_visible_missing_token_is_eighteenth_category():
    p = np.zeros((2, 18)); p[0, 17] = 1.; p[1, 3] = 1.
    q = np.full((2, 18, 2), .5)
    np.testing.assert_allclose(expected_token_loss(q, p, np.array([0, 1])), np.log(2))
    assert validate_token_probs(p).shape[1] == 18


def test_global_validation_split_is_label_blind_and_cross_anchor_consistent():
    from experiments.pcrl_task_aligned_cuts_v1.data import global_validation_split
    households = {
        0: np.array(['h0', 'h1', 'h2', 'h3']),
        1: np.array(['h1', 'h2', 'h4', 'h5']),
        2: np.array(['h0', 'h4', 'h6', 'h7']),
    }

    def prepared(label_shift):
        return {anchor: {'ctx': {'pools': {'downstream_validation': {
            'households': hh,
            'labels': {
                'same_residence': np.arange(len(hh)) % 2,
                'SEX': (np.arange(len(hh)) + label_shift) % 2,
                'RAC1P': (np.arange(len(hh)) + label_shift) % 9,
            },
        }}}} for anchor, hh in households.items()}

    original = global_validation_split(prepared(0))
    shifted = global_validation_split(prepared(3))
    assert original['assignment_sha256'] == shifted['assignment_sha256']
    assert original['selection_households_global'] == 4
    assert original['pilot_households_global'] == 4
    all_households = set().union(*(set(hh) for hh in households.values()))
    ordered = sorted(all_households,
                     key=lambda h: (hashlib.sha256(h.encode('utf-8')).hexdigest(), h))
    selection = set(ordered[:4])
    for anchor, hh in households.items():
        rows = original['anchors'][anchor]['rows']
        assert set(rows['inner_selection']).isdisjoint(rows['inner_pilot'])
        assert set(rows['inner_selection']) | set(rows['inner_pilot']) == set(range(len(hh)))
        assert set(hh[rows['inner_selection']]) == set(hh) & selection
        np.testing.assert_array_equal(rows['inner_selection'],
                                      shifted['anchors'][anchor]['rows']['inner_selection'])
    # A household recurring under another anchor cannot move between halves.
    for h in all_households:
        assignments = [h in set(households[a][original['anchors'][a]['rows']['inner_selection']])
                       for a in households if h in households[a]]
        assert len(set(assignments)) == 1

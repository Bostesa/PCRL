"""Small independent inference and witness-selection failure probes."""
import numpy as np
import pytest
import torch

from scripts.compose_acs_restricted_attacks import choose_witnesses
from scripts.verify_acs_restricted import SOURCES, composition_comparison, literal_release


def identity_state():
    state = {'input_mean': torch.zeros(32, dtype=torch.float64),
             'input_scale': torch.ones(32, dtype=torch.float64),
             'mapper.0.weight': torch.zeros(64, 48), 'mapper.0.bias': torch.zeros(64),
             'mapper.2.weight': torch.zeros(16, 64), 'mapper.2.bias': torch.zeros(16)}
    state['mapper.0.weight'][:16, :16] = torch.eye(16)
    state['mapper.0.weight'][16:32, :16] = -torch.eye(16)
    state['mapper.2.weight'][:, :16] = torch.eye(16)
    state['mapper.2.weight'][:, 16:32] = -torch.eye(16)
    for index, target in enumerate(SOURCES):
        state[f'heads.{target}.weight'] = torch.eye(16)[index:index+1]
        state[f'heads.{target}.bias'] = torch.zeros(1)
    return state


def test_literal_teacher_boundary_and_three_probability_bank():
    state = identity_state()
    teacher = np.arange(5*16, dtype=np.float32).reshape(5, 16)/20-2
    teacher_before = teacher.copy()
    release, native = literal_release(state, teacher, access='K')
    np.testing.assert_array_equal(release, teacher)
    np.testing.assert_array_equal(teacher, teacher_before)
    bank, _ = literal_release(state, teacher, access='K', bank=True)
    assert bank.shape == (5, 3)
    np.testing.assert_array_equal(bank, np.column_stack([native[target][:, 1] for target in SOURCES]))
    np.testing.assert_allclose(bank, 1/(1+np.exp(-teacher[:, :3])), atol=1e-7)
    with pytest.raises(ValueError, match='rejects raw'):
        literal_release(state, teacher, access='K', raw=np.zeros((5, 32), np.float32))
    with pytest.raises(ValueError, match='requires aligned'):
        literal_release(state, teacher, access='F')


def test_literal_F_raw_route_is_exposed_and_K_stays_teacher_only():
    state = identity_state()
    state['mapper.0.weight'][0, 16] = 2.
    teacher, raw = np.ones((4, 16), np.float32), np.ones((4, 32), np.float32)
    full, _ = literal_release(state, teacher, access='F', raw=raw)
    restricted, _ = literal_release(state, teacher, access='K')
    assert np.all(full[:, 0] == 3.)
    np.testing.assert_array_equal(restricted, teacher)


def test_composition_checks_full_prediction_array_and_hash():
    probability = np.array([[.25, .75], [.625, .375]])
    record = composition_comparison(probability, probability.copy())
    assert record['bitwise_equal'] and record['max_abs'] == record['rms'] == 0.
    assert record['composed_probability_sha256'] == record['direct_probability_sha256']
    with pytest.raises(AssertionError):
        composition_comparison(probability, probability[::-1])


def test_global_validation_choice_excludes_saved_but_allows_epoch_zero():
    base = {'teacher': 'E', 'seed': 0, 'target': 'SEX', 'audit_budget': 360}
    witnesses = [{**base, 'witness_id': name, 'candidate_id': cid, 'validation_log_loss': loss,
                  'selected_epoch': epoch, 'test': {'log_loss': 100-loss}}
                 for name, cid, loss, epoch in [('z', 'logistic', .4, None), ('a', 'mlp_0', .4, 20),
                                                ('b', 'catchup', .3, 0), ('s', 'saved_adversary', .1, 0)]]
    selected = choose_witnesses(witnesses)
    assert selected == {'E/seed_0/SEX/budget360': {'independent': 'a', 'catchup': 'b', 'pooled': 'b'}}
    for row in witnesses:
        row['test'] = {'log_loss': -10000. if row['witness_id'] == 'z' else 10000.}
    assert choose_witnesses(witnesses[::-1]) == selected

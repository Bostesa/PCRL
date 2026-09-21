"""Token-release scoring and attacker-fitting invariances (METHOD.md §5).

These target consequential risks: averaging the wrong thing, and inflating one person into K people.
"""
import numpy as np
import pytest

from experiments.pcrl_stochastic_replacement_overnight_v1 import tokens as T


def _probe(coef, bias=0.0):
    """A fixed logistic predictor over [h, onehot(token)]."""
    def predict(x):
        z = np.asarray(x, np.float64) @ coef + bias
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])
    return predict


# ---------------------------------------------------------------- averaging the right thing
def test_expected_loss_averages_losses_not_predictions():
    """`E[loss]` and `loss(E[prediction])` are different estimands; only the former is used."""
    rng = np.random.default_rng(0)
    n, k = 200, 3
    h = rng.normal(size=(n, 2))
    t = rng.integers(0, 4, n)
    q = rng.random((4, k)) + 0.05
    q /= q.sum(1, keepdims=True)
    y = rng.integers(0, 2, n)
    probe = _probe(np.array([0.8, -0.5, 2.0, 0.0, -2.0]))

    exact = T.expected_loss(probe, h, t, q, y)['per_person_expected_loss']
    wrong = T.loss_of_expected_prediction(probe, h, t, q, y)
    assert not np.allclose(exact, wrong)
    # Jensen: log loss is convex in the probability, so averaging predictions understates the loss.
    assert wrong.mean() <= exact.mean() + 1e-12


def test_expected_loss_is_exact_for_a_degenerate_channel():
    """A deterministic Q must reproduce the plain per-row loss."""
    rng = np.random.default_rng(1)
    n, k = 50, 3
    h = rng.normal(size=(n, 2))
    t = rng.integers(0, 4, n)
    q = np.zeros((4, k))
    q[:, 1] = 1.0
    y = rng.integers(0, 2, n)
    probe = _probe(np.array([0.5, 0.5, 1.0, -1.0, 0.25]))
    got = T.expected_loss(probe, h, t, q, y)['per_person_expected_loss']

    onehot = np.zeros((n, k))
    onehot[:, 1] = 1.0
    p = probe(np.column_stack([h, onehot]))
    want = -np.log(p[np.arange(n), y])
    assert np.allclose(got, want)


def test_expected_loss_respects_person_weights():
    rng = np.random.default_rng(2)
    n, k = 80, 2
    h = rng.normal(size=(n, 1))
    t = rng.integers(0, 3, n)
    q = np.array([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    y = rng.integers(0, 2, n)
    probe = _probe(np.array([0.3, 1.0, -1.0]))
    flat = T.expected_loss(probe, h, t, q, y)
    doubled = T.expected_loss(probe, h, t, q, y, weight=np.full(n, 2.0))
    assert doubled['mean'] == pytest.approx(flat['mean'], rel=1e-12)


# ---------------------------------------------------------------- K rows are one person
def test_expansion_weights_sum_to_the_person_weight():
    rng = np.random.default_rng(3)
    n, k = 40, 4
    h = rng.normal(size=(n, 3))
    t = rng.integers(0, 5, n)
    q = rng.random((5, k)) + 0.01
    q /= q.sum(1, keepdims=True)
    exp = T.expand_exact(h, t, q)
    chk = T.check_expansion_weights(exp)
    assert chk['ok'] and chk['max_abs_weight_error'] < 1e-12
    assert exp['n_rows'] == n * k and exp['n_people'] == n
    # The decisive invariant: total weight is the number of PEOPLE, not the number of rows.
    assert chk['total_expanded_weight'] == pytest.approx(float(n))
    assert exp['n_rows'] == 4 * n            # rows inflated ...
    assert chk['total_expanded_weight'] != exp['n_rows']   # ... weight is not


def test_expansion_preserves_pwgtp_multiplication():
    rng = np.random.default_rng(4)
    n, k = 30, 3
    h = rng.normal(size=(n, 2))
    t = rng.integers(0, 4, n)
    q = rng.random((4, k)) + 0.01
    q /= q.sum(1, keepdims=True)
    w = rng.uniform(5, 500, n)
    exp = T.expand_exact(h, t, q, weight=w)
    chk = T.check_expansion_weights(exp, weight=w)
    assert chk['ok']
    assert chk['total_expanded_weight'] == pytest.approx(float(w.sum()))


def test_expansion_carries_household_and_label_on_every_row():
    n, k = 5, 3
    h = np.zeros((n, 1))
    t = np.arange(n) % 2
    q = np.full((2, k), 1 / k)
    serial = np.array([10, 10, 11, 12, 12])
    y = np.array([0, 1, 0, 1, 1])
    exp = T.expand_exact(h, t, q, y=y, serial=serial)
    for i in range(n):
        rows = exp['person'] == i
        assert set(np.unique(exp['serial'][rows])) == {serial[i]}
        assert set(np.unique(exp['y'][rows])) == {y[i]}
        assert sorted(exp['token'][rows]) == list(range(k))


def test_constant_token_expansion_is_the_unexpanded_problem():
    """Identity fixture: a constant token must give a weighted problem identical to no expansion.

    This is the guard against regularization or epoch normalization treating K expanded rows as K
    independent people -- with a constant token, K-1 of them carry zero weight.
    """
    rng = np.random.default_rng(5)
    n, k = 25, 4
    h = rng.normal(size=(n, 2))
    t = rng.integers(0, 3, n)
    q = np.zeros((3, k))
    q[:, 2] = 1.0
    exp = T.expand_exact(h, t, q)
    live = exp['weight'] > 0
    assert live.sum() == n                                  # exactly one live row per person
    assert np.allclose(exp['weight'][live], 1.0)
    assert np.allclose(np.sort(exp['person'][live]), np.arange(n))
    assert set(np.unique(exp['token'][live])) == {2}


def test_weighted_gradient_identity_under_duplication():
    """Duplicating a row with halved weights must leave a weighted gradient unchanged.

    If an implementation ignored weights it would double this person's influence.
    """
    x = np.array([[1.0, 2.0], [1.0, 2.0]])
    y = np.array([1, 1])
    w_half = np.array([0.5, 0.5])
    single_x, single_y, single_w = np.array([[1.0, 2.0]]), np.array([1]), np.array([1.0])

    def grad(xx, yy, ww, beta):
        p = 1.0 / (1.0 + np.exp(-xx @ beta))
        return (ww[:, None] * (p - yy)[:, None] * xx).sum(0) / ww.sum()

    beta = np.array([0.3, -0.7])
    assert np.allclose(grad(x, y, w_half, beta), grad(single_x, single_y, single_w, beta))


# ---------------------------------------------------------------- the released object
def test_sampled_wire_token_block_is_one_hot_not_a_probability_row():
    rng = np.random.default_rng(6)
    n, k = 60, 3
    h = rng.normal(size=(n, 4))
    t = rng.integers(0, 2, n)
    q = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    wire = T.sampled_wire(h, t, q, rng)
    assert wire.shape == (n, 4 + k)
    block = wire[:, 4:]
    assert T.is_sampled_token_block(block, k) is True
    assert np.allclose(wire[:, :4], h)                      # H_A preserved exactly
    # The forbidden alternative: releasing the row of Q.
    assert T.is_sampled_token_block(q[t], k) is False


def test_sampling_is_unbiased_for_the_channel():
    rng = np.random.default_rng(7)
    q = np.array([[0.25, 0.75], [0.9, 0.1]])
    t = np.repeat([0, 1], 20000)
    z = T.sample_tokens(t, q, rng)
    assert np.mean(z[:20000]) == pytest.approx(0.75, abs=0.01)
    assert np.mean(z[20000:]) == pytest.approx(0.10, abs=0.01)


def test_degenerate_row_samples_deterministically():
    rng = np.random.default_rng(8)
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    t = np.array([0, 0, 1, 1])
    assert list(T.sample_tokens(t, q, rng)) == [0, 0, 1, 1]

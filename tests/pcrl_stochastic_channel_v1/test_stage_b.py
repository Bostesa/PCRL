"""Fixtures for the Stage B capacity diagnostic. No data needed."""
import numpy as np
import pytest

from experiments.pcrl_stochastic_channel_v1 import stage_b as sb


def test_onehot_is_a_one_hot():
    t = np.array([0, 3, 1, 3])
    o = sb.onehot(t, 4)
    assert o.shape == (4, 4) and np.all(o.sum(1) == 1)
    assert np.array_equal(np.argmax(o, 1), t)


def test_declared_resolutions_are_fixed():
    """The declared resolution and the single predeclared fallback, pinned against drift."""
    assert sb.N_CODES == 64 and sb.N_CODES_FALLBACK == 256
    assert sb.RESIDENCE == 'same_residence'


def test_household_fold_never_splits_a_household():
    serials = np.array([10, 10, 11, 12, 12, 12, 13])
    fold = sb._household_fold(serials, 5)
    for s in np.unique(serials):
        assert len(np.unique(fold[serials == s])) == 1


def test_household_fold_is_deterministic():
    serials = np.arange(50) // 3
    assert np.array_equal(sb._household_fold(serials, 5), sb._household_fold(serials, 5))


def test_bin_view_is_dense_and_respects_the_reference_quantiles():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 3))
    cells = sb._bin_view(v, 3, v)
    assert cells.min() == 0 and cells.max() == len(np.unique(cells)) - 1
    assert len(np.unique(cells)) > 1


def test_ceiling_is_zero_when_the_code_is_constant():
    """A constant code can add nothing, so the matched tables must agree exactly."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    base = rng.integers(0, 5, 400)
    fold = np.arange(400) % 5
    out = sb.ceiling_from_code(y, base, np.zeros(400, dtype=np.int64), fold)
    assert out['ceiling_reduction_nats'] == pytest.approx(0.0, abs=1e-12)


def test_ceiling_is_positive_when_the_code_determines_the_label():
    """A code that reveals y must show a large reduction, with enough rows per cell to estimate it."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 4000)
    base = np.zeros(4000, dtype=np.int64)
    fold = np.arange(4000) % 5
    out = sb.ceiling_from_code(y, base, y.copy(), fold)
    assert out['ceiling_reduction_nats'] > 0.5


def test_ceiling_reports_its_own_scope():
    y = np.array([0, 1] * 50)
    out = sb.ceiling_from_code(y, np.zeros(100, np.int64), np.zeros(100, np.int64), np.arange(100) % 5)
    assert 'not a population bound' in out['scope']


def test_paired_interval_resamples_households_not_rows():
    """Two rows in one household must not count as two independent units."""
    n = 300
    serial = np.repeat(np.arange(n // 3), 3)
    rng = np.random.default_rng(3)
    # The DIFFERENCE must be household-correlated; a shared term in both arms would cancel.
    shared = rng.normal(size=n // 3) * 0.5
    right = rng.normal(size=n)
    left = right + np.repeat(shared, 3) + 1.0
    tight = sb.paired_household_interval(serial, left, right, n_boot=800, seed=1)
    loose = sb.paired_household_interval(np.arange(n), left, right, n_boot=800, seed=1)
    assert tight['households'] == n // 3 and loose['households'] == n
    # The point estimate does not depend on the resampling unit, only its spread does.
    assert tight['estimate'] == pytest.approx(loose['estimate'], abs=1e-12)
    assert tight['estimate'] == pytest.approx(1.0, abs=0.2)
    # Treating correlated rows as independent understates the SE; the household unit is larger.
    assert tight['bootstrap_se'] > loose['bootstrap_se']


def test_paired_interval_reports_the_unit_and_both_bounds():
    serial = np.repeat(np.arange(40), 2)
    rng = np.random.default_rng(4)
    left, right = rng.normal(size=80), rng.normal(size=80)
    out = sb.paired_household_interval(serial, left, right, n_boot=500)
    assert out['unit'] == 'household' and out['households'] == 40 and out['rows'] == 80
    assert out['lower_one_sided'] < out['estimate'] < out['upper_one_sided']
    assert out['upper_one_sided'] == pytest.approx(
        out['estimate'] + out['critical_value'] * out['bootstrap_se'], abs=1e-12)


def test_paired_interval_is_not_withheld_for_a_near_zero_estimate():
    """AMENDMENT_1 section 4: an interval is reported on its own merits, whatever its width."""
    serial = np.repeat(np.arange(60), 2)
    out = sb.paired_household_interval(serial, np.zeros(120), np.zeros(120), n_boot=400)
    assert out['estimate'] == 0.0 and out['bootstrap_se'] == 0.0
    assert 'upper_one_sided' in out and out['upper_one_sided'] == 0.0


def test_extra_draws_do_not_add_households():
    """Replicating every row leaves the household count unchanged."""
    serial = np.repeat(np.arange(30), 2)
    rng = np.random.default_rng(5)
    left, right = rng.normal(size=60), rng.normal(size=60)
    once = sb.paired_household_interval(serial, left, right, n_boot=400, seed=2)
    twice = sb.paired_household_interval(np.concatenate([serial, serial]),
                                        np.concatenate([left, left]),
                                        np.concatenate([right, right]), n_boot=400, seed=2)
    assert once['households'] == twice['households'] == 30
    assert twice['rows'] == 2 * once['rows']
    assert twice['bootstrap_se'] == pytest.approx(once['bootstrap_se'], rel=1e-9)

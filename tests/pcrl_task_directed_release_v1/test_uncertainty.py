"""Registered household inference contracts using small synthetic populations."""
import numpy as np
import pytest
from scipy.stats import norm

from experiments.pcrl_task_directed_release_v1 import uncertainty


def anchor(household, difference, weights=None):
    difference = np.asarray(difference, dtype=float)
    return {'household': np.asarray(household, dtype=str), 'difference': difference,
            'weights': np.ones_like(difference) if weights is None else np.asarray(weights, dtype=float)}


def contrast(ident, *anchors):
    return {'id': ident, 'anchors': list(anchors)}


def test_replicate_weighted_denominator_matches_direct_oracle():
    """A fixed population denominator gives the wrong survey-ratio bootstrap variance."""
    a = anchor(['a', 'b', 'c', 'd'], [0, 1, 3, 7], [1, 2, 20, 1])
    n, seed = 4000, 31
    result = uncertainty.paired_household_bounds([contrast('weighted', a, a, a)], n_boot=n, seed=seed)
    counts = np.random.default_rng(seed).multinomial(4, np.full(4, .25), size=n)
    numerator = counts @ (a['weights'] * a['difference'])
    denominator = counts @ a['weights']
    true_ratio = numerator / denominator
    fixed_denominator = numerator / a['weights'].sum()
    bound = result['bounds']['weighted']
    assert bound['estimate'] == pytest.approx(69 / 24)
    assert bound['bootstrap_se'] == pytest.approx(true_ratio.std(ddof=1), abs=1e-13)
    assert abs(bound['bootstrap_se'] - fixed_denominator.std(ddof=1)) > .5


def test_overlapping_anchor_covariance_is_retained():
    """Resampling identical anchors independently would incorrectly divide their SE by sqrt3."""
    a = anchor(['a', 'a', 'b', 'c', 'd'], [0, 2, -1, 3, 8])
    n, seed = 3000, 42
    result = uncertainty.paired_household_bounds([contrast('shared', a, a, a)], n_boot=n, seed=seed)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(4, np.full(4, .25), size=n)
    one_anchor = (counts @ np.array([2., -1., 3., 8.])) / (counts @ np.array([2., 1., 1., 1.]))
    independent = [one_anchor]
    for _ in range(2):
        count = rng.multinomial(4, np.full(4, .25), size=n)
        independent.append((count @ np.array([2., -1., 3., 8.])) / (count @ np.array([2., 1., 1., 1.])))
    actual = result['bounds']['shared']['bootstrap_se']
    assert actual == pytest.approx(one_anchor.std(ddof=1), abs=1e-13)
    assert actual > 1.5 * np.mean(independent, axis=0).std(ddof=1)
    assert result['households_union'] == 4


def test_family_uses_joint_draws_and_list_size_for_two_sided_bonferroni():
    a = anchor(['a', 'b', 'c'], [-1, 2, 5])
    opposite = anchor(['a', 'b', 'c'], [1, -2, -5])
    family = [contrast('forward', a, a, a), contrast('reverse', opposite, opposite, opposite)]
    result = uncertainty.paired_household_bounds(family, n_boot=333, seed=17)
    assert result['family_size'] == 2
    assert result['critical_value_adjusted'] == pytest.approx(norm.isf(.05 / 4))
    f, r = (result['bounds'][k] for k in ('forward', 'reverse'))
    assert f['bootstrap_se'] == r['bootstrap_se']
    assert f['estimate'] == -r['estimate']
    assert f['upper'] == -r['lower']
    assert f['lower_one_sided_adjusted'] == f['lower']
    assert f['upper_one_sided_adjusted'] == f['upper']


def test_identical_pairs_keep_exact_zero_variance():
    a = anchor(['a', 'b', 'c'], [0, 0, 0], [1, 100, 2])
    result = uncertainty.paired_household_bounds([contrast('identical', a, a, a)], n_boot=177)
    bound = result['bounds']['identical']
    assert bound['estimate'] == bound['bootstrap_se'] == bound['lower'] == bound['upper'] == 0.
    assert bound['identical_pairs'] is True


def test_zero_denominators_reject_whole_shared_draw_deterministically():
    """Never average only2 surviving anchors or give different contrasts different draws."""
    a = anchor(['a'], [1])
    b = anchor(['b'], [3])
    c = anchor(['c'], [5])
    second = anchor(['a', 'b', 'c'], [0, 1, 6])
    family = [contrast('sparse', a, b, c), contrast('full', second, second, second)]
    result = uncertainty.paired_household_bounds(family, n_boot=250, seed=71)
    boot = result['bootstrap']
    assert boot['accepted'] == 250
    assert boot['rejected_zero_denominator'] > 0
    # Drawing3 households and requiring a,b,c leaves precisely one of each.
    assert result['bounds']['sparse']['estimate'] == 3
    assert result['bounds']['full']['bootstrap_se'] == pytest.approx(0, abs=1e-15)
    assert result == uncertainty.paired_household_bounds(family, n_boot=250, seed=71)


def test_three_anchors_are_averaged_equally_not_by_total_weight():
    a = anchor(['a', 'b'], [1, 1], [1, 1])
    b = anchor(['a', 'b'], [2, 2], [100, 100])
    c = anchor(['a', 'b'], [9, 9], [1000, 1000])
    result = uncertainty.paired_household_bounds([contrast('equal', a, b, c)], n_boot=20)
    assert result['bounds']['equal']['estimate'] == 4


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_exactly_three_anchor_records_are_required(count):
    a = anchor(['a', 'b'], [1, 2])
    with pytest.raises(ValueError, match='three|3'):
        uncertainty.paired_household_bounds([contrast('bad', *([a] * count))], n_boot=20)


def test_duplicate_ids_and_malformed_arrays_are_rejected():
    a = anchor(['a', 'b'], [1, 2])
    c = contrast('duplicate', a, a, a)
    with pytest.raises(ValueError, match='unique|duplicate'):
        uncertainty.paired_household_bounds([c, c], n_boot=20)
    for field, bad in [('difference', np.array([1., np.nan])),
                       ('weights', np.array([1., np.inf])),
                       ('weights', np.array([1., -1.])),
                       ('weights', np.array([0., 0.])),
                       ('difference', np.array([1.])),
                       ('household', np.array(['a', None], dtype=object))]:
        broken = {**a, field: bad}
        with pytest.raises(ValueError):
            uncertainty.paired_household_bounds([contrast('bad', broken, a, a)], n_boot=20)


def test_empty_family_and_invalid_bootstrap_parameters_rejected():
    a = anchor(['a', 'b'], [1, 2])
    family = [contrast('one', a, a, a)]
    with pytest.raises(ValueError):
        uncertainty.paired_household_bounds([], n_boot=20)
    for kwargs in [{'n_boot': 1}, {'n_boot': 2.5}, {'alpha': 0}, {'alpha': 1}, {'alpha': np.nan}, {'alpha': 'bad'}, {'alpha': True}]:
        with pytest.raises(ValueError):
            uncertainty.paired_household_bounds(family, **kwargs)

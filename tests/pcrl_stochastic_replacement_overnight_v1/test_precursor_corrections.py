"""Fixtures pinning the precursor's interpretation and routing errors (PRECURSOR_CORRECTIONS.md).

These are mathematical fixtures, not re-runs. They read no data and they do not reverse the
precursor's operational FAIL at G2.
"""
import numpy as np
import pytest

from experiments.pcrl_direct_adversarial_v1 import inputs as dax


# ---------------------------------------------------------------- C1: what "J" actually was
def test_j_baseline_is_twenty_columns_not_the_service_predictions():
    """The precursor's 'same-host J' is H_A plus a 16-coordinate auxiliary channel."""
    assert dax.H_A_WIDTH == 4 and dax.A0_WIDTH == 16
    assert dax.H_A_WIDTH + dax.A0_WIDTH == 20
    # A claim about "the released service predictions" is a claim about 4 columns, not 20.
    assert dax.H_A_WIDTH != dax.H_A_WIDTH + dax.A0_WIDTH


# ---------------------------------------------------------------- C3: the min() identity
def inclusion_respecting_advantage(ref_loss: float, code_loss: float) -> float:
    """The precursor's statistic: ref - min(code, ref)."""
    return ref_loss - min(code_loss, ref_loss)


def test_inclusion_respecting_advantage_is_identically_zero_when_code_is_not_better():
    """Exactly 0.00000 across anchors is a tautology, not corroborating evidence."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        ref = float(rng.uniform(0.3, 0.8))
        code = ref + float(rng.uniform(0.0, 0.2))        # code no better
        assert inclusion_respecting_advantage(ref, code) == 0.0


def test_the_statistic_has_zero_variance_by_construction_not_by_precision():
    """Repeating it over anchors cannot produce evidence; the spread is structurally zero."""
    vals = [inclusion_respecting_advantage(r, r + 0.01) for r in (0.50, 0.52, 0.55)]
    assert vals == [0.0, 0.0, 0.0] and np.std(vals) == 0.0


def test_the_statistic_is_also_uninformative_about_disclosure():
    """The same identity holds for a sensitive endpoint.

    Attackers are selected by MINIMUM validation log loss, and measured recovery rises as that loss
    falls. So an ancestor-inclusive slate can never score a HIGHER loss than the ancestor, i.e. can
    never show less recovery than the ancestor. Equality is the best case, not an improvement.
    """
    ancestor_loss = 0.62
    slate = [ancestor_loss, 0.70, 0.66]                   # ancestor plus weaker attacks
    assert min(slate) == ancestor_loss                    # selection cannot exceed the ancestor
    assert min(slate) - ancestor_loss == 0.0              # so the increment is identically zero
    stronger = [ancestor_loss, 0.55]                      # a genuinely stronger attack exists
    assert min(stronger) < ancestor_loss                  # recovery can only go UP, never down


# ---------------------------------------------------------------- C4: label dependence
def per_row_loss(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exactly the precursor's `_per_row`."""
    return -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12))


def test_per_row_loss_depends_on_the_true_label_so_it_is_not_a_predicted_probability():
    """The withdrawn 'probability decile' partition was built from a label-dependent quantity."""
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    a = per_row_loss(probs, np.array([0, 0]))
    b = per_row_loss(probs, np.array([1, 1]))
    # Identical features and identical predictions, different labels -> different partition
    # coordinate. A conditioning partition may never behave this way.
    assert not np.allclose(a, b)
    assert np.allclose(a, -np.log(0.9)) and np.allclose(b, -np.log(0.1))


def test_a_label_free_partition_coordinate_is_invariant_to_the_label():
    """The admissible alternative: partition on the prediction itself."""
    probs = np.array([[0.9, 0.1], [0.3, 0.7]])
    coord = probs[:, 1]
    assert np.allclose(coord, [0.1, 0.7])                 # no y anywhere


# ---------------------------------------------------------------- C5: not a ceiling
def test_a_table_predictor_loss_difference_can_be_negative_so_it_bounds_no_information_quantity():
    """Adding a fine code to a partition can make a smoothed table predictor WORSE.

    A conditional mutual information is nonnegative. A quantity that can go negative therefore
    cannot be an upper bound on one, which is why the 'ceiling' name is withdrawn.
    """
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    base = np.zeros(n, dtype=np.int64)
    noise_code = rng.integers(0, 64, n)                   # pure noise, independent of y
    fold = np.arange(n) % 5

    def table_loss(cells):
        loss = np.zeros(n)
        for f in range(5):
            tr, te = fold != f, fold == f
            _, key = np.unique(cells, return_inverse=True)
            tab = np.full((key.max() + 1, 2), 0.5)
            np.add.at(tab, (key[tr], y[tr]), 1.0)
            tab /= tab.sum(1, keepdims=True)
            loss[te] = -np.log(np.maximum(tab[key[te]][np.arange(te.sum()), y[te]], 1e-12))
        return loss.mean()

    difference = table_loss(base) - table_loss(base * 64 + noise_code)
    assert difference < 0.0                               # "ceiling" would have to be >= 0
    # And the true added information is zero, so the estimator is not tracking it either way.


# ---------------------------------------------------------------- C7: extension vs replacement
def test_an_ancestor_inclusive_maximum_cannot_beat_its_ancestor():
    """A J-retaining extension cannot show strictly lower optimal disclosure than J."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        ancestor = float(rng.uniform(0.4, 0.9))
        others = rng.uniform(0.4, 1.2, 5).tolist()
        # An attacker on the extension may ignore the appended block, so the ancestor is available.
        best_attack = max([ancestor] + others)
        assert best_attack >= ancestor


def test_replacement_can_strictly_remove_information_that_an_extension_cannot():
    """The decisive counterexample, in exact arithmetic.

    Constant H; independent fair bits Y (useful) and S (sensitive); J = (Y, S); T = Y.
    Adding T to J gains nothing -- J already determines Y. Replacing J by T keeps Y and removes S.
    This is explanatory, not an ACS finding and not a novelty claim.
    """
    from fractions import Fraction
    log2 = np.log(2)

    # Joint over (Y, S), both fair and independent.
    cells = [(y, s) for y in (0, 1) for s in (0, 1)]
    p = {c: Fraction(1, 4) for c in cells}
    assert sum(p.values()) == 1

    def mi(view):
        """I(view ; S) in nats, exactly, for a deterministic view of (Y,S)."""
        joint = {}
        for (y, s), q in p.items():
            joint[(view(y, s), s)] = joint.get((view(y, s), s), Fraction(0)) + q
        pv, ps = {}, {}
        for (v, s), q in joint.items():
            pv[v] = pv.get(v, Fraction(0)) + q
            ps[s] = ps.get(s, Fraction(0)) + q
        return sum(float(q) * np.log(float(q) / (float(pv[v]) * float(ps[s])))
                   for (v, s), q in joint.items() if q > 0)

    def util(view):
        """I(view ; Y) in nats, exactly."""
        joint = {}
        for (y, s), q in p.items():
            joint[(view(y, s), y)] = joint.get((view(y, s), y), Fraction(0)) + q
        pv, py = {}, {}
        for (v, y), q in joint.items():
            pv[v] = pv.get(v, Fraction(0)) + q
            py[y] = py.get(y, Fraction(0)) + q
        return sum(float(q) * np.log(float(q) / (float(pv[v]) * float(py[y])))
                   for (v, y), q in joint.items() if q > 0)

    J = lambda y, s: (y, s)
    T = lambda y, s: y
    J_plus_T = lambda y, s: (y, s, y)                     # the extension

    # Utility: J already carries Y, so appending T adds exactly nothing.
    assert util(J_plus_T) == pytest.approx(util(J), abs=1e-12)
    assert util(J_plus_T) - util(J) == pytest.approx(0.0, abs=1e-12)

    # Disclosure: the extension cannot remove S. The replacement removes it entirely.
    assert mi(J) == pytest.approx(log2, abs=1e-12)
    assert mi(J_plus_T) == pytest.approx(log2, abs=1e-12)
    assert mi(T) == pytest.approx(0.0, abs=1e-12)

    # And the replacement keeps the useful bit in full.
    assert util(T) == pytest.approx(log2, abs=1e-12)

    # So "T adds no utility over J" is compatible with "T is a strictly better release than J".
    # The precursor's screen measured the former and concluded about the latter.


def test_zero_added_utility_does_not_imply_zero_information_in_the_code():
    """Numeric guard on the inference the corrections withdraw."""
    added_utility_over_j = 0.0
    # Nothing about the code's own information content follows.
    assert added_utility_over_j == 0.0
    for code_information in (0.0, 0.1, 0.693):
        assert code_information >= 0.0                    # unconstrained by the line above

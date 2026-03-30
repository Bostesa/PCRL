"""Tests for the multi-class impossibility theorem.

Verifies:
1. fano_mi_lower_bound works correctly for K-class attributes
2. multiclass_impossibility_bound gives valid bounds for K=2,6,30
3. min_representations_needed computes correct chromatic number
4. Concrete checks on HAR's 30-class subject_id and 6-class activity
"""

import math

import pytest

from pcrl.purposes.spec import PurposeSpec
from pcrl.purposes.verification import (
    fano_mi_lower_bound,
    fano_mi_lower_bound_multiclass,
    impossibility_bound,
    multiclass_impossibility_bound,
    min_representations_needed,
    find_conflicting_attributes,
)


# ── fano_mi_lower_bound: multi-class cases ──────────────────────────────


class TestFanoMILowerBoundMulticlass:
    """Verify Fano MI lower bound for various K."""

    def test_binary_perfect_accuracy(self):
        """Perfect binary accuracy => MI = H(A)."""
        mi = fano_mi_lower_bound(0.999, 2)
        assert mi > 0.6, f"Expected near log(2)=0.693, got {mi}"

    def test_binary_chance_accuracy(self):
        """Chance accuracy => MI = 0."""
        mi = fano_mi_lower_bound(0.5, 2)
        assert mi < 0.01, f"Expected ~0, got {mi}"

    def test_6class_high_accuracy(self):
        """90% accuracy on 6-class => significant MI."""
        mi = fano_mi_lower_bound(0.90, 6)
        # H(A) = log(6) = 1.79 nats
        # pe = 0.1, H_b(0.1) ≈ 0.325, 0.1*log(5) ≈ 0.161
        # MI >= 1.79 - 0.325 - 0.161 = 1.304
        assert mi > 1.0, f"Expected MI > 1.0 for 90% on 6-class, got {mi}"

    def test_30class_high_accuracy(self):
        """90% accuracy on 30-class => significant MI."""
        mi = fano_mi_lower_bound(0.90, 30)
        # H(A) = log(30) = 3.40 nats
        # pe = 0.1, H_b(0.1) ≈ 0.325, 0.1*log(29) ≈ 0.337
        # MI >= 3.40 - 0.325 - 0.337 = 2.74
        assert mi > 2.0, f"Expected MI > 2.0 for 90% on 30-class, got {mi}"

    def test_30class_chance_accuracy(self):
        """Chance accuracy on 30-class => MI = 0."""
        mi = fano_mi_lower_bound(1.0 / 30, 30)
        assert mi < 0.01, f"Expected ~0, got {mi}"

    def test_monotonic_in_accuracy(self):
        """Higher accuracy => higher MI lower bound."""
        accs = [0.2, 0.4, 0.6, 0.8, 0.95]
        mis = [fano_mi_lower_bound(a, 6) for a in accs]
        for i in range(len(mis) - 1):
            assert mis[i] <= mis[i + 1] + 1e-10, (
                f"Non-monotonic: MI({accs[i]})={mis[i]} > MI({accs[i+1]})={mis[i+1]}"
            )

    def test_multiclass_alias_matches(self):
        """fano_mi_lower_bound_multiclass should match fano_mi_lower_bound."""
        for k in [2, 6, 30]:
            for acc in [0.3, 0.7, 0.95]:
                a = fano_mi_lower_bound(acc, k)
                b = fano_mi_lower_bound_multiclass(acc, k)
                assert abs(a - b) < 1e-15

    def test_with_nonuniform_entropy(self):
        """Non-uniform entropy should tighten the bound."""
        # Uniform entropy for 6 classes
        mi_uniform = fano_mi_lower_bound(0.9, 6)
        # Skewed entropy (one class dominates)
        entropy_skewed = -0.7 * math.log(0.7) - 5 * 0.06 * math.log(0.06)
        mi_skewed = fano_mi_lower_bound(0.9, 6, entropy_a=entropy_skewed)
        # Skewed entropy is lower, so MI bound should be lower
        assert mi_skewed < mi_uniform + 0.01


# ── multiclass_impossibility_bound ───────────────────────────────────────


class TestMulticlassImpossibilityBound:
    """Test the combined accuracy -> MI -> impossibility pipeline."""

    def test_binary_95_percent(self):
        """95% accuracy on binary attribute."""
        bound = multiclass_impossibility_bound(0.95, 2)
        # Should be well above 50% (actual: ~82%)
        assert bound > 0.80, f"Expected > 80%, got {bound:.1%}"

    def test_6class_90_percent(self):
        """90% accuracy on 6-class attribute."""
        bound = multiclass_impossibility_bound(0.90, 6)
        # MI is high, but spread over 6 classes
        assert bound > 1.0 / 6, f"Expected above chance (16.7%), got {bound:.1%}"

    def test_30class_90_percent(self):
        """90% accuracy on 30-class subject_id."""
        bound = multiclass_impossibility_bound(0.90, 30)
        # With 30 classes, even high MI gives modest per-class accuracy
        assert bound > 1.0 / 30, f"Expected above chance (3.3%), got {bound:.1%}"

    def test_chance_accuracy_gives_chance_bound(self):
        """Chance accuracy => bound at chance level."""
        for k in [2, 6, 30]:
            bound = multiclass_impossibility_bound(1.0 / k, k)
            assert abs(bound - 1.0 / k) < 0.01, (
                f"K={k}: expected ~{1/k:.3f}, got {bound:.3f}"
            )

    def test_perfect_accuracy_gives_high_bound(self):
        """Near-perfect accuracy => bound near 1.0."""
        for k in [2, 6, 30]:
            bound = multiclass_impossibility_bound(0.999, k)
            assert bound > 0.9, f"K={k}: expected > 90%, got {bound:.1%}"

    def test_har_subject_id_concrete(self):
        """HAR: if activity_recognition achieves 91% on 6-class activity,
        and activity is also a 6-class sensitive attr for health_monitoring,
        any single-rep must leak activity above chance."""
        bound = multiclass_impossibility_bound(0.91, 6)
        chance = 1.0 / 6
        assert bound > chance, (
            f"Expected single-rep leak > {chance:.1%}, got {bound:.1%}"
        )

    def test_har_30class_subject(self):
        """HAR: 30-class subject_id with hypothetical 80% recognition accuracy."""
        bound = multiclass_impossibility_bound(0.80, 30)
        chance = 1.0 / 30
        assert bound > chance, (
            f"Expected single-rep leak > {chance:.1%}, got {bound:.1%}"
        )


# ── min_representations_needed ───────────────────────────────────────────


class TestMinRepresentationsNeeded:
    """Test the corollary on minimum representation count."""

    def test_no_conflicts(self):
        """Disjoint purposes => 1 representation suffices."""
        purposes = [
            PurposeSpec(
                name="p1", allowed_tasks=["task_a"],
                disallowed_attrs=["attr_b"],
                allowed_task_dims={"task_a": 2},
                disallowed_attr_dims={"attr_b": 2},
            ),
            PurposeSpec(
                name="p2", allowed_tasks=["task_c"],
                disallowed_attrs=["attr_d"],
                allowed_task_dims={"task_c": 2},
                disallowed_attr_dims={"attr_d": 2},
            ),
        ]
        assert min_representations_needed(purposes) == 1

    def test_single_conflict(self):
        """Two purposes with one conflict => need 2 representations."""
        purposes = [
            PurposeSpec(
                name="p1", allowed_tasks=["income"],
                disallowed_attrs=["sex"],
                allowed_task_dims={"income": 2},
                disallowed_attr_dims={"sex": 2},
            ),
            PurposeSpec(
                name="p2", allowed_tasks=["sex"],
                disallowed_attrs=["income"],
                allowed_task_dims={"sex": 2},
                disallowed_attr_dims={"income": 2},
            ),
        ]
        assert min_representations_needed(purposes) == 2

    def test_adult_three_purposes(self):
        """Adult dataset 3 purposes have conflicts => need > 1."""
        from pcrl.data.adult import get_adult_purposes
        purposes = get_adult_purposes()
        n = min_representations_needed(purposes)
        # income is task for income_prediction but disallowed for education_assessment
        assert n >= 2, f"Expected >= 2, got {n}"

    def test_har_two_purposes(self):
        """HAR 2 purposes: activity is task for one, disallowed for other => 2."""
        from pcrl.data.har import get_har_purposes
        purposes = get_har_purposes()
        n = min_representations_needed(purposes)
        # activity is task for activity_recognition, disallowed for health_monitoring
        assert n == 2, f"Expected 2, got {n}"

    def test_single_purpose(self):
        """Single purpose => 1 representation."""
        purposes = [
            PurposeSpec(
                name="p1", allowed_tasks=["income"],
                disallowed_attrs=["sex"],
                allowed_task_dims={"income": 2},
                disallowed_attr_dims={"sex": 2},
            ),
        ]
        assert min_representations_needed(purposes) == 1

    def test_three_way_conflict(self):
        """Three purposes forming a triangle => need 3."""
        purposes = [
            PurposeSpec(
                name="p1", allowed_tasks=["A"],
                disallowed_attrs=["B"],
                allowed_task_dims={"A": 2},
                disallowed_attr_dims={"B": 2},
            ),
            PurposeSpec(
                name="p2", allowed_tasks=["B"],
                disallowed_attrs=["C"],
                allowed_task_dims={"B": 2},
                disallowed_attr_dims={"C": 2},
            ),
            PurposeSpec(
                name="p3", allowed_tasks=["C"],
                disallowed_attrs=["A"],
                allowed_task_dims={"C": 2},
                disallowed_attr_dims={"A": 2},
            ),
        ]
        n = min_representations_needed(purposes)
        # Triangle conflict graph => chromatic number = 3
        # But wait — p1 uses A and p3 disallows A => conflict (1,3)
        # p2 uses B and p1 disallows B => conflict (1,2)
        # p3 uses C and p2 disallows C => conflict (2,3)
        # Triangle => 3 colors
        assert n == 3, f"Expected 3 for triangle, got {n}"


# ── Consistency checks ──────────────────────────────────────────────────


class TestConsistency:
    """Verify internal consistency between functions."""

    def test_fano_then_impossibility_matches_multiclass(self):
        """multiclass_impossibility_bound should equal fano -> impossibility."""
        for k in [2, 6, 30]:
            for acc in [0.5, 0.8, 0.95]:
                direct = multiclass_impossibility_bound(acc, k)
                mi = fano_mi_lower_bound(acc, k)
                indirect = impossibility_bound(mi, k)
                assert abs(direct - indirect) < 1e-10, (
                    f"Mismatch for K={k}, acc={acc}: {direct} != {indirect}"
                )

    def test_bound_never_below_chance(self):
        """Impossibility bound should never be below chance level."""
        for k in [2, 6, 30]:
            for acc in [1.0 / k, 0.3, 0.5, 0.8, 0.99]:
                if acc < 1.0 / k:
                    continue
                bound = multiclass_impossibility_bound(acc, k)
                assert bound >= 1.0 / k - 1e-10, (
                    f"K={k}, acc={acc}: bound {bound} < chance {1/k}"
                )

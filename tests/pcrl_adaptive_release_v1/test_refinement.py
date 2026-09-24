"""Exact Branch B counterexamples and deployable nested-state invariants."""

from fractions import Fraction

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import refinement


def test_b1_task_posterior_is_sufficient_but_t_only_privacy_loses_task_information():
    result = refinement.fixture_b1()
    assert result["input_masses"] == tuple(map(Fraction, ("2/5", "1/10", "1/10", "2/5")))
    assert result["private_stochastic_z1"] == tuple(map(Fraction, ("0", "0", "1", "1/4")))
    assert result["p_z1_given_s"] == (Fraction(1, 5), Fraction(1, 5))
    assert result["p_y1_given_z"] == (Fraction(2, 5), Fraction(9, 10))
    assert result["t_only_private_implies_row_equality"] is True
    assert result["partitions_enumerated"] == 15
    assert result["deterministic_private_partitions"] == 2
    assert result["max_deterministic_private_task_information"] == pytest.approx(0)
    assert result["stochastic_task_information_nats"] == pytest.approx(0.0897212523, abs=1e-10)


def test_b2_context_helps_even_when_deterministic_and_task_code_has_no_extra_y_signal():
    result = refinement.fixture_b2()
    assert result["exact_sensitive_independence"] is True
    assert result["conditional_sensitive_information_nats"] == pytest.approx(0, abs=1e-15)
    assert result["conditional_task_information_nats"] == pytest.approx(np.log(2) / 2)
    assert result["context_blind_private_implies_constant"] is True


def test_refinement_copies_every_parent_row_and_routes_only_on_public_features():
    partition = refinement.NestedPartition.base(32).split(3, "h_a_0", 0.5)
    parent = np.array([3, 3, 4, 31])
    features = {"h_a_0": np.array([0.2, 0.8, 0.9, 0.1])}
    child = partition.route(parent, features)
    assert child.tolist() == [3, 32, 4, 31]
    assert np.array_equal(partition.parent_of_leaf[child], parent)
    q = np.eye(17, dtype=np.float64)[np.arange(32) % 17]
    assert np.array_equal(refinement.copy_parent_kernel(q, partition.parent_of_leaf)[child], q[parent])
    with pytest.raises(ValueError, match="forbidden"):
        partition.split(3, "observed_SEX", 0.5)
    with pytest.raises(ValueError, match="missing"):
        partition.route(parent, {})


def test_nested_partition_roundtrips_a_second_child_split_without_parent_drift():
    original = (refinement.NestedPartition.base(32)
                .split(3, "h_a_0", 0.5)
                .split(32, "sex_risk_1", 0.7))
    restored = refinement.NestedPartition.from_record(original.to_record())
    parent = np.array([3, 3, 3, 4])
    features = {"h_a_0": np.array([0.1, 0.8, 0.8, 0.8]),
                "sex_risk_1": np.array([0.9, 0.1, 0.9, 0.9])}
    assert restored.route(parent, features).tolist() == [3, 32, 33, 4]
    assert np.array_equal(restored.parent_of_leaf[restored.route(parent, features)], parent)
    bad = original.to_record()
    bad["rules"][0]["feature_name"] = "observed_RAC1P"
    with pytest.raises(ValueError, match="forbidden"):
        refinement.NestedPartition.from_record(bad)


def test_fixed_price_score_and_support_are_computed_on_original_households():
    task = np.array([[0., 2.], [2., 0.], [1., 3.], [3., 1.]])
    attack = np.ones_like(task)
    priced = refinement.priced_contributions(task, [attack], [0.25])
    assert np.array_equal(priced, task - 0.25)
    assert refinement.fixed_price_gain(priced, [True, False, True, False]) == pytest.approx(4)
    features = {"h_a_0": np.array([0., 1., 0., 1.])}
    picks = refinement.rank_splits(
        leaf_ids=np.zeros(4, dtype=np.int64), leaf=0, features=features,
        priced_rows=priced, households=np.array([1, 2, 3, 4]),
        weights=np.ones(4), min_households=2, min_effective_weight=2,
        feature_names=("h_a_0",), quantiles=(0.5,), max_candidates_per_feature=1,
    )
    assert len(picks) == 1
    assert picks[0].feature_name == "h_a_0"
    assert picks[0].gain == pytest.approx(4)
    assert picks[0].children_households == (2, 2)
    assert not refinement.rank_splits(
        leaf_ids=np.zeros(4, dtype=np.int64), leaf=0, features=features,
        priced_rows=priced, households=np.array([1, 1, 1, 2]),
        weights=np.ones(4), min_households=2, min_effective_weight=2,
        feature_names=("h_a_0",), quantiles=(0.5,), max_candidates_per_feature=1,
    )


def test_split_gain_is_zero_if_children_choose_same_action():
    g = np.array([[0., 1.], [0., 2.]])
    assert refinement.fixed_price_gain(g, [True, False]) == pytest.approx(0)


def test_checking_split_is_measured_on_disjoint_households():
    features = {"h_a_0": np.array([0., 0., 1., 1.])}
    g = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    common = dict(leaf_ids=np.zeros(4, dtype=int), leaf=0, features=features,
                  priced_rows=g, households=np.array([1, 2, 3, 4]),
                  weights=np.ones(4), min_households=2,
                  min_effective_weight=2, feature_names=("h_a_0",),
                  quantiles=(0.5,))
    candidates = refinement.rank_splits(**common, checking=(
        np.zeros(4, dtype=int), features, g, np.array([5, 6, 7, 8])))
    assert candidates[0].checking_gain == pytest.approx(2)
    with pytest.raises(ValueError, match="overlap"):
        refinement.rank_splits(**common, checking=(
            np.zeros(4, dtype=int), features, g, np.array([1, 6, 7, 8])))


def test_weighted_support_uses_households_not_repeated_people():
    features = {"h_a_0": np.array([0., 0., 1., 1.])}
    candidates = refinement.rank_splits(
        leaf_ids=np.zeros(4, dtype=int), leaf=0, features=features,
        priced_rows=np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]]),
        households=np.array([1, 1, 2, 3]), weights=np.ones(4),
        min_households=1, min_effective_weight=1.5,
        feature_names=("h_a_0",), quantiles=(0.5,))
    assert not candidates  # left child has two people but one effective household


def test_deployable_feature_builder_requires_local_runtime_inputs_and_frozen_nuisance():
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs

    class Encoder:
        def encode(self, inputs):
            assert isinstance(inputs, RuntimeInputs)
            return {"codes": {"T0": np.array([0, 1])},
                    "r": np.array([0.1, -0.2]), "p": np.array([0.4, 0.6])}

    class Nuisance:
        def probabilities(self, inputs):
            assert isinstance(inputs, RuntimeInputs)
            return {"SEX": np.tile([0.4, 0.6], (2, 1)),
                    "RAC1P": np.tile(np.ones(9)/9, (2, 1))}

    local = RuntimeInputs(np.zeros((2, 32)), np.zeros((2, 4)))
    t, features = refinement.build_deployable_features(local, Encoder(), Nuisance())
    assert t.tolist() == [0, 1]
    assert set(features) == refinement.ALLOWED_FEATURES
    assert "observed_SEX" not in features
    partition = refinement.NestedPartition.base(32).split(0, "sex_risk_1", 0.5)
    assert partition.route_runtime(local, Encoder(), Nuisance()).tolist() == [32, 1]
    with pytest.raises(TypeError, match="RuntimeInputs"):
        refinement.build_deployable_features({"x_a": local.x_a, "h_a": local.h_a,
                                              "s": np.array([0, 1])}, Encoder(), Nuisance())

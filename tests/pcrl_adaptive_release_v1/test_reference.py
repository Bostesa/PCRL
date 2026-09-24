import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1.reference import (
    calibrate_reference,
    solve_calibrated,
)


def _cut(name, loss, *, role="AB/RAC1P", weighting="W", pool="same"):
    # Token-invariant H-only loss. Every channel incurs this exact loss.
    return {"id": name, "role": role, "weighting": weighting,
            "coeff": np.full((2, 2), loss / 2),
            "coefficient_pool_sha256": pool,
            "class_order": list(range(9)),
            "weight_normalization": "PWGTP/sum(PWGTP)"}


def test_cross_pool_reference_ranking_reversal_is_repaired_on_coefficient_rows():
    q_ref = np.eye(2)
    selected_on_other_pool = _cut("selected_elsewhere", 1.0)
    stronger_here = _cut("stronger_here", 0.8)
    # Historical selected floor 1.0 makes the H-only 0.8 loss impossible.
    assert np.sum(stronger_here["coeff"] * q_ref) < 1.0
    result = calibrate_reference(q_ref, [selected_on_other_pool, stronger_here], 0.0)
    assert result["rho"]["AB/RAC1P|W"]["value"] == pytest.approx(0.8)
    assert result["rho"]["AB/RAC1P|W"]["attaining_attack_id"] == "stronger_here"
    assert all(cut["floor"] == pytest.approx(0.8) for cut in result["cuts"])
    assert result["witness"]["maximum_cut_violation"] == pytest.approx(0.0)
    cost = np.array([[0.0, 0.1], [0.2, 0.0]])
    solved = solve_calibrated(cost, q_ref, [selected_on_other_pool, stronger_here], 0.0)
    assert solved["solution"]["status"] == "optimal"
    assert solved["solution"]["feasible"]


def test_expanding_bank_rebases_every_old_floor_and_keeps_reference_feasible():
    q_ref = np.eye(2)
    old = [_cut("a", 1.0), _cut("b", 0.8)]
    before = calibrate_reference(q_ref, old, 0.001)
    after = calibrate_reference(q_ref, old + [_cut("c", 0.7)], 0.001)
    assert {c["id"]: c["floor"] for c in before["cuts"]} == {
        "a": pytest.approx(0.799), "b": pytest.approx(0.799)}
    assert {c["id"]: c["floor"] for c in after["cuts"]} == {
        "a": pytest.approx(0.699), "b": pytest.approx(0.699), "c": pytest.approx(0.699)}
    assert after["witness"]["maximum_cut_violation"] == pytest.approx(0.0)


def test_role_weighting_reference_risks_are_separate_and_negative_allowance_rejected():
    q_ref = np.eye(2)
    bank = [_cut("w", 0.8),
            {**_cut("u", 0.6, weighting="U"), "weight_normalization": "1/n"},
            {**_cut("sex", 0.4, role="A/SEX"), "class_order": [0, 1]}]
    result = calibrate_reference(q_ref, bank, {"AB/RAC1P|W": 0.001,
                                               "AB/RAC1P|U": 0.003,
                                               "A/SEX|W": 0.0})
    assert result["rho"]["AB/RAC1P|W"]["value"] == pytest.approx(0.8)
    assert result["rho"]["AB/RAC1P|U"]["value"] == pytest.approx(0.6)
    assert result["rho"]["A/SEX|W"]["value"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="nonnegative"):
        calibrate_reference(q_ref, bank, -0.001)
    with pytest.raises(ValueError, match="coefficient pool"):
        calibrate_reference(q_ref, bank + [_cut("other_pool", 0.7, pool="different")], 0.0)


def test_reference_feasibility_uses_all_attacks_not_only_selected_one():
    q_ref = np.array([[0.1, 0.9], [0.8, 0.2]])
    bank = [
        {"id": "a", "role": "A/SEX", "weighting": "U", "coeff": np.array([[1.0, 0.2], [0.3, 0.4]])},
        {"id": "b", "role": "A/SEX", "weighting": "U", "coeff": np.array([[0.4, 0.5], [0.2, 0.6]])},
    ]
    result = calibrate_reference(q_ref, bank, 0.0, require_provenance=False)
    rho = min(np.sum(c["coeff"] * q_ref) for c in bank)
    assert result["rho"]["A/SEX|U"]["value"] == pytest.approx(rho)
    assert all(np.sum(c["coeff"] * q_ref) >= c["floor"] - 1e-12 for c in result["cuts"])


def test_production_reference_rejects_missing_coefficient_provenance():
    q_ref = np.eye(2)
    bank = [{"id": "untracked", "role": "A/SEX", "weighting": "U",
             "coeff": np.full((2, 2), 0.25)}]
    with pytest.raises(ValueError, match="provenance"):
        calibrate_reference(q_ref, bank, 0.0)


def test_u_and_weighted_role_use_same_original_coefficient_people():
    q_ref = np.eye(2)
    w = _cut("w", 0.8, pool="population_A")
    u = {**_cut("u", 0.7, weighting="U", pool="population_B"),
         "weight_normalization": "1/n"}
    with pytest.raises(ValueError, match="same original coefficient people"):
        calibrate_reference(q_ref, [w, u], 0.0)

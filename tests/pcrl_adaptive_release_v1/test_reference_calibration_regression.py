"""Independent exact fixtures for reference calibration and fixed-bank claims.

These use hand-constructed laws and small deterministic enumerations. They do
not load ACS rows or refit any empirical predictor.
"""
from fractions import Fraction
import itertools
import math

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import fit_b, reference, refinement
from experiments.pcrl_task_aligned_cuts_v1 import controls, solver


def _cut(name, coefficient, weighting="U"):
    return {"id": name, "role": "A/SEX", "weighting": weighting,
            "coeff": np.asarray(coefficient, dtype=float),
            "coefficient_pool_sha256": "same-two-original-people",
            "class_order": [0, 1],
            "weight_normalization": "1/n" if weighting == "U" else "PWGTP/sum(PWGTP)"}


def test_cross_pool_selected_reference_can_make_invariant_h_only_cut_impossible():
    q_ref = np.eye(2)
    # Selection-pool ranking is A=.5 < H=.6; on coefficient rows it reverses.
    selection_losses = {"A": .5, "H": .6}
    assert min(selection_losses, key=selection_losses.get) == "A"
    attack_a = _cut("A", [[.7, 0], [0, 0]])
    h_only = _cut("H", np.full((2, 2), .2))
    assert np.sum(attack_a["coeff"] * q_ref) == pytest.approx(.7)
    for actions in itertools.product(range(2), repeat=2):
        channel = np.eye(2)[list(actions)]
        assert np.sum(h_only["coeff"] * channel) == pytest.approx(.4)
        assert np.sum(h_only["coeff"] * channel) < .7  # old selected-rho floor
    calibrated = reference.calibrate_reference(q_ref, [attack_a, h_only], 0.)
    assert calibrated["rho"]["A/SEX|U"]["value"] == pytest.approx(.4)
    assert [cut["floor"] for cut in calibrated["cuts"]] == pytest.approx([.4, .4])
    assert calibrated["witness"]["maximum_cut_violation"] == 0


def test_expanding_bank_rebases_every_floor_and_preserves_d17_witness():
    q_ref = np.eye(2)
    initial = [_cut("A/U", [[.7, 0], [0, 0]]),
               _cut("H/U", np.full((2, 2), .2)),
               _cut("A/W", [[.9, 0], [0, 0]], "W"),
               _cut("H/W", np.full((2, 2), .25), "W")]
    allowance = {"A/SEX|U": .05, "A/SEX|W": .05}
    first = reference.calibrate_reference(q_ref, initial, allowance)
    assert first["rho"]["A/SEX|U"]["value"] == pytest.approx(.4)
    assert first["rho"]["A/SEX|W"]["value"] == pytest.approx(.5)
    expanded = initial + [_cut("new_H/U", np.full((2, 2), .15)),
                          _cut("new_H/W", np.full((2, 2), .2), "W")]
    second = reference.calibrate_reference(q_ref, expanded, allowance)
    assert second["rho"]["A/SEX|U"]["value"] == pytest.approx(.3)
    assert second["rho"]["A/SEX|W"]["value"] == pytest.approx(.4)
    by_id = {cut["id"]: cut for cut in second["cuts"]}
    assert by_id["A/U"]["floor"] == pytest.approx(.25)
    assert by_id["A/W"]["floor"] == pytest.approx(.35)
    assert by_id["new_H/U"]["floor"] == pytest.approx(.25)
    assert second["witness"]["maximum_cut_violation"] == 0
    assert second["bank_sha256"] != first["bank_sha256"]


def test_one_frozen_bank_has_strict_lp_vs_certified_deterministic_gap():
    cost = np.full((1, 17), 2.)
    cost[0, 0], cost[0, 1] = 0., 1.
    q_ref = np.eye(17)[[1]]
    attack = np.zeros_like(cost)
    attack[0, 1] = 1.
    solved = reference.solve_calibrated(cost, q_ref, [_cut("risk", attack)], .5)
    lp = solved["solution"]
    milp = controls.solve_deterministic_p1(cost, solved["cuts"], time_limit_seconds=10.)
    assert lp["feasible"] and milp["incumbent_valid"]
    assert lp["bank_sha256"] == solver.bank_sha256(solved["cuts"], cost.shape)
    assert milp["fixed_bank_cut_ids"] == [cut["id"] for cut in solved["cuts"]]
    assert np.sum(cost * lp["Q"]) == pytest.approx(.5, abs=1e-8)
    assert np.sum(attack * lp["Q"]) == pytest.approx(.5, abs=1e-8)
    assert lp["dual_lower_bound"] == pytest.approx(.5, abs=1e-8)
    assert milp["incumbent_objective"] == pytest.approx(1., abs=1e-8)
    assert milp["lower_bound"] == pytest.approx(1., abs=1e-8)
    # Exhaustion, independent of the MILP incumbent, confirms the 17 maps.
    assert min(cost[0, z] for z in range(17) if attack[0, z] >= .5) == 1.


def test_b1_exact_private_fractions_and_b2_contextual_counterexample():
    b1 = refinement.fixture_b1()
    assert b1["partitions_enumerated"] == 15
    assert b1["p_z1_given_s"] == (Fraction(1, 5), Fraction(1, 5))
    assert b1["p_y1_given_z"] == (Fraction(2, 5), Fraction(9, 10))
    assert b1["deterministic_private_partitions"] == 2
    assert b1["max_deterministic_private_task_information"] == pytest.approx(0., abs=1e-14)

    def entropy(p):
        p = float(p)
        return -p * math.log(p) - (1-p) * math.log(1-p)

    expected_mi = math.log(2) - Fraction(1, 5) * entropy(Fraction(9, 10)) - Fraction(4, 5) * entropy(Fraction(2, 5))
    assert b1["stochastic_task_information_nats"] == pytest.approx(expected_mi, abs=1e-12)
    assert expected_mi == pytest.approx(.0897212523, abs=1e-10)

    b2 = refinement.fixture_b2()
    assert b2["exact_sensitive_independence"] is True
    assert b2["conditional_sensitive_information_nats"] == pytest.approx(0., abs=1e-14)
    assert b2["conditional_task_information_nats"] == pytest.approx(math.log(2)/2, abs=1e-14)
    # At H=1, S=T and privacy forces any context-blind q(Z|T=0) and
    # q(Z|T=1) to coincide. At H=0, Y=T, so that law then carries zero task MI.
    assert b2["context_blind_private_implies_constant"] is True


def test_privacy_floor_dual_has_minus_sign_in_priced_split_rows():
    cost = np.full((1, 17), 2.)
    cost[0, 0], cost[0, 1] = 0., 1.
    attack = np.zeros_like(cost)
    attack[0, 1] = 1.
    dual = fit_b.fixed_bank_dual_prices(cost, [{"id": "a", "coeff": attack, "floor": .5}])
    assert dual["status"] == "optimal"
    assert dual["multipliers"]["a"] == pytest.approx(1., abs=1e-8)
    assert dual["objective"] == dual["dual_lower_bound"] == pytest.approx(.5, abs=1e-8)
    task_rows = np.array([[0., 2.], [2., 0.]])
    attack_rows = np.array([[0., 1.], [0., 1.]])
    priced = refinement.priced_contributions(task_rows, [attack_rows], [1.])
    assert np.array_equal(priced, np.array([[0., 1.], [2., -1.]]))
    assert refinement.fixed_price_gain(priced, [True, False]) == pytest.approx(1.)

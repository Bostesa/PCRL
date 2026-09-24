"""The optional privacy-first LP uses the same frozen rows and 17-token wire."""

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import privacy_first


ROLES = ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P")


def tiny_problem(*, h_only_ab_sex=False):
    d17 = np.eye(17, dtype=np.float64)[[0]]
    cost_u = np.zeros((1, 17)); cost_u[0, 1:] = 1.
    cost_w = np.zeros((1, 17)); cost_w[0, 1:] = 2.
    cuts = []
    for role in ROLES:
        for weighting in ("U", "W"):
            coeff = np.full((1, 17), .5)
            if role == "AB/SEX":
                coeff[0, 1] = .6 if h_only_ab_sex else 1.
                coeff[0, 0] = .5 if h_only_ab_sex else 0.
            rho = float(np.sum(coeff*d17))
            cuts.append({"id": f"{role}/{weighting}/main", "role": role,
                         "weighting": weighting, "coeff": coeff,
                         "rho": rho, "delta": .001, "floor": rho-.001})
            if h_only_ab_sex and role == "AB/SEX":
                h_only = np.full((1, 17), .5)
                cuts.append({"id": f"{role}/{weighting}/H", "role": role,
                             "weighting": weighting, "coeff": h_only,
                             "rho": .5, "delta": .001, "floor": .499})
    return {"U": cost_u, "W": cost_w}, cuts, d17


def test_privacy_first_both_task_caps_and_positive_loss_floor_sign():
    pair, cuts, d17 = tiny_problem()
    result = privacy_first.solve_privacy_first(pair, cuts, d17, time_limit_seconds=5)
    assert result["status"] == "OPTIMAL"
    assert result["witness"]["feasible"] is True
    assert result["phase_one"]["status"] == "OPTIMAL"
    assert result["phase_one"]["minimum_common_violation"] == pytest.approx(0, abs=1e-9)
    assert result["tau"] == pytest.approx(.0005, abs=1e-8)  # W task cap binds
    assert result["Q"][0, 1] == pytest.approx(.0005, abs=1e-8)
    assert result["replay"]["feasible"] is True
    assert result["replay"]["task_loss"]["U"] <= .001 + 1e-8
    assert result["replay"]["task_loss"]["W"] <= .001 + 1e-8
    assert result["dual_tau_upper_bound"] >= result["tau"] - 1e-8
    assert result["dual_gap"] <= 1e-7


def test_h_only_ab_sex_attack_can_force_exact_zero_tau():
    pair, cuts, d17 = tiny_problem(h_only_ab_sex=True)
    result = privacy_first.solve_privacy_first(pair, cuts, d17, time_limit_seconds=5)
    assert result["status"] == "OPTIMAL"
    assert result["tau"] == pytest.approx(0, abs=1e-9)
    assert result["replay"]["feasible"] is True


def test_corrupt_reference_floor_is_detected_before_optimization():
    pair, cuts, d17 = tiny_problem()
    cut = next(item for item in cuts if item["role"] == "A/SEX")
    cut["rho"] += .01
    cut["floor"] += .01
    witness = privacy_first.replay_privacy_first(d17, 0., pair, cuts, d17)
    assert witness["maximum_cut_violation"] == pytest.approx(.009)
    phase = privacy_first.phase_one_privacy_first(pair, cuts, d17, time_limit_seconds=5)
    assert phase["minimum_common_violation"] > .004
    with pytest.raises(ValueError, match="D17.*witness"):
        privacy_first.solve_privacy_first(pair, cuts, d17, time_limit_seconds=5)


def test_missing_role_or_bad_calibration_rejected():
    pair, cuts, d17 = tiny_problem()
    with pytest.raises(ValueError, match="role.*weighting"):
        privacy_first.solve_privacy_first(pair, cuts[:-1], d17)
    altered = [{**cut} for cut in cuts]
    altered[0]["floor"] += .002
    with pytest.raises(ValueError, match="floor.*rho"):
        privacy_first.solve_privacy_first(pair, altered, d17)

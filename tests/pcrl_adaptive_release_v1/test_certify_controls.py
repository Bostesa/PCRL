"""Same-final-bank LP/MILP certificates, without ACS rows."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import certify_controls
from experiments.pcrl_adaptive_release_v1 import fit_controls


def _toy():
    cost = np.full((1, 17), 3.0)
    cost[0, 0] = 0.0
    cost[0, 1] = 1.0
    cost[0, 2] = 2.0
    coeff = np.zeros_like(cost)
    coeff[0, 1] = coeff[0, 2] = 1.0
    cut = {"id": "A/SEX/U/one", "coeff": coeff, "floor": 0.5}
    selected = np.zeros_like(cost)
    selected[0, 0] = selected[0, 1] = 0.5
    incumbent = np.zeros_like(cost)
    incumbent[0, 1] = 1.0
    return cost, [cut], selected, incumbent


def test_exact_shared_bank_gap_and_selected_channel_provenance():
    cost, cuts, selected, incumbent = _toy()
    result = certify_controls.certify_fixed_bank(
        cost, cuts, selected,
        {"lower_bound": 1.0, "incumbent_objective": 1.0,
         "incumbent_valid": True, "status_code": 0}, incumbent)
    assert result["status"] == "NUMERICAL_FIXED_BANK_INTEGRALITY_GAP"
    assert result["lp"]["objective"] == pytest.approx(0.5)
    assert result["milp"]["lower_bound"] == pytest.approx(1.0)
    assert result["deterministic_lower_bound_minus_stochastic_feasible"] == pytest.approx(0.5)
    assert result["selected_channel"]["Q_array_sha256"] == certify_controls._array_sha(selected)
    assert result["lp"]["bank_sha256"] == result["selected_channel"]["bank_sha256"]
    assert result["oracle_error"] == "UNRESOLVED"
    assert result["scope"] == "fixed bank and frozen decoder only"


def test_rejects_milp_bound_or_incumbent_inconsistent_with_same_bank():
    cost, cuts, selected, incumbent = _toy()
    with pytest.raises(ValueError, match="lower bound exceeds"):
        certify_controls.certify_fixed_bank(
            cost, cuts, selected,
            {"lower_bound": 1.1, "incumbent_objective": 1.0,
             "incumbent_valid": True}, incumbent)
    wrong = incumbent.copy()
    wrong[0, :] = 0.0
    wrong[0, 0] = 1.0
    with pytest.raises(ValueError, match="MILP incumbent"):
        certify_controls.certify_fixed_bank(
            cost, cuts, selected,
            {"lower_bound": 1.0, "incumbent_objective": 1.0,
             "incumbent_valid": True}, wrong)
    with pytest.raises(ValueError, match="cut IDs"):
        certify_controls.certify_fixed_bank(
            cost, cuts, selected,
            {"lower_bound": 1.0, "incumbent_objective": 1.0,
             "incumbent_valid": True, "fixed_bank_cut_ids": ["wrong-bank-cut"]},
            incumbent)


def test_selected_channel_may_be_feasible_but_suboptimal_on_final_bank():
    cost, cuts, _, incumbent = _toy()
    selected = incumbent.copy()
    result = certify_controls.certify_fixed_bank(
        cost, cuts, selected,
        {"lower_bound": 1.0, "incumbent_objective": 1.0,
         "incumbent_valid": True}, incumbent)
    assert result["selected_channel"]["objective"] == pytest.approx(1.0)
    assert result["lp"]["objective"] == pytest.approx(0.5)
    assert result["status"] == "NUMERICAL_FIXED_BANK_INTEGRALITY_GAP"


def test_unresolved_milp_lower_bound_is_not_reported_as_gap():
    cost, cuts, selected, _ = _toy()
    result = certify_controls.certify_fixed_bank(
        cost, cuts, selected,
        {"lower_bound": None, "incumbent_objective": None,
         "incumbent_valid": False, "status_code": 1}, None)
    assert result["status"] == "DETERMINISTIC_LOWER_BOUND_UNRESOLVED"
    assert result["deterministic_lower_bound_minus_stochastic_feasible"] is None


def test_completed_control_receipt_replay_and_tamper_rejection(tmp_path, monkeypatch):
    cost, cuts, selected, incumbent = _toy()
    source = {"branch": "A", "anchor": 0, "delta": 0.0,
              "center_sha256": "synthetic-center", "selected_bank_sha256": "synthetic-bank"}
    controls_dir = tmp_path / "private" / "controls"
    fit_controls.fit_controls_from_frozen(
        {"U": cost, "W": cost}, cuts, incumbent, selected,
        controls_dir, source=source, milp_seconds=5., heuristic_seconds=.001,
        gradient_steps=2, gradient_penalties=(10.,), gradient_seeds=(17,))
    problem = {"cost_pair": {"U": cost, "W": cost}, "cuts": cuts,
               "selected_channel": selected, "source": source}
    monkeypatch.setattr(fit_controls, "load_final_problem", lambda *args, **kwargs: problem)
    output = tmp_path / "private" / "certificate"
    receipt = certify_controls.certify_from_completed_controls(
        "A", tmp_path / "private" / "center", controls_dir, output,
        anchor=0, delta=0.0)
    assert receipt["certificate"]["status"] == "NUMERICAL_FIXED_BANK_INTEGRALITY_GAP"
    assert receipt["artifact_sha256"] == fit_controls._inventory(output)
    assert certify_controls.certify_from_completed_controls(
        "A", tmp_path / "private" / "center", controls_dir, output,
        anchor=0, delta=0.0) == receipt
    with (controls_dir / "BANK.json").open("a") as stream:
        stream.write("tamper")
    with pytest.raises(ValueError, match="artifact inventory"):
        certify_controls.certify_from_completed_controls(
            "A", tmp_path / "private" / "center", controls_dir, output,
            anchor=0, delta=0.0)

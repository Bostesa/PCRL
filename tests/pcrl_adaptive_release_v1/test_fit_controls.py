"""The control runner must replay one frozen bank and retain immutable channels."""

import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import fit_controls


def _tiny_problem():
    cost = np.full((1, 17), 3.0)
    cost[0, 0] = 1.0
    cost[0, 1] = 0.0
    attack = np.zeros_like(cost)
    attack[0, 0] = 1.0
    d17 = np.eye(17)[[0]]
    historical_q = np.eye(17)[[1]]
    cuts = [{"id": "A/SEX/H/U", "role": "A/SEX", "weighting": "U",
             "coeff": attack, "floor": 0.5}]
    return {"U": cost, "W": cost.copy()}, cuts, d17, historical_q


def test_frozen_bank_runner_keeps_d17_and_milp_with_hash_replayed_simple_curves(tmp_path):
    """Wrong sign, token width, or output overwrite must break this fixture."""
    pair, cuts, d17, historical_q = _tiny_problem()
    output = tmp_path / "private" / "controls"
    result = fit_controls.fit_controls_from_frozen(
        pair, cuts, d17, historical_q, output,
        source={"branch": "A", "anchor": 0, "center_sha256": "a"*64},
        milp_seconds=2, heuristic_seconds=0.01, gradient_steps=3,
        gradient_penalties=(1.0,), gradient_seeds=(17,))
    assert result["status"] == "COMPLETE"
    receipt = json.loads((output / "COMPLETE.json").read_text())
    assert receipt["source"]["center_sha256"] == "a"*64
    assert receipt["cost_pair_sha256"]["U"] == receipt["cost_pair_sha256"]["W"]
    assert receipt["fixed_bank_sha256"]
    assert receipt["mip_lower_bound"] <= 1.0 + 1e-8
    assert receipt["mip_incumbent_objective"] == pytest.approx(1.0)
    assert receipt["mip_incumbent_valid"] is True
    assert len(receipt["control_ids"]) >= 20
    for cid in ("D17", "historical_Q", "D_task", "MILP",
                "D17_constant_replace_publish_050", "D17_randomized_response_publish_050"):
        assert cid in receipt["control_ids"]
        with np.load(output / "channels" / cid / "Q.npz", allow_pickle=False) as archive:
            assert archive["Q"].shape == (1, 17)
    assert receipt["constant_token"] == 0  # cheapest action is 1 in this fixture
    with np.load(output / "channels" / "D_task_constant_replace_publish_050" / "Q.npz",
                 allow_pickle=False) as archive:
        assert archive["Q"][0, 0] == pytest.approx(.5)
        assert archive["Q"][0, 1] == pytest.approx(.5)
    assert fit_controls.fit_controls_from_frozen(
        pair, cuts, d17, historical_q, output,
        source={"branch": "A", "anchor": 0, "center_sha256": "a"*64},
        milp_seconds=2, heuristic_seconds=0.01, gradient_steps=3,
        gradient_penalties=(1.0,), gradient_seeds=(17,))["status"] == "COMPLETE"


def test_completed_control_unit_refuses_changed_source_or_tampered_channel(tmp_path):
    pair, cuts, d17, historical_q = _tiny_problem()
    output = tmp_path / "private" / "controls"
    kwargs = dict(milp_seconds=1, heuristic_seconds=0.01,
                  gradient_steps=2, gradient_penalties=(1.0,),
                  gradient_seeds=(17,))
    fit_controls.fit_controls_from_frozen(
        pair, cuts, d17, historical_q, output,
        source={"branch": "A", "anchor": 0, "center_sha256": "a"*64}, **kwargs)
    with pytest.raises(ValueError, match="source|unit"):
        fit_controls.fit_controls_from_frozen(
            pair, cuts, d17, historical_q, output,
            source={"branch": "A", "anchor": 0, "center_sha256": "b"*64}, **kwargs)
    altered_role = [{**cuts[0], "role": "AB/SEX"}]
    with pytest.raises(ValueError, match="source|unit"):
        fit_controls.fit_controls_from_frozen(
            pair, altered_role, d17, historical_q, output,
            source={"branch": "A", "anchor": 0, "center_sha256": "a"*64}, **kwargs)
    path = output / "channels" / "D17" / "Q.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash|inventory|artifact"):
        fit_controls.fit_controls_from_frozen(
            pair, cuts, d17, historical_q, output,
            source={"branch": "A", "anchor": 0, "center_sha256": "a"*64}, **kwargs)


def test_b_loader_replays_selected_q_on_final_child_bank_and_rejects_tamper(tmp_path):
    """A selected earlier Q is accepted only on the last stored bank/cost."""
    from experiments.pcrl_adaptive_release_v1 import fit_a, fit_b, refinement
    from experiments.pcrl_task_aligned_cuts_v1 import solver

    root = tmp_path / "private" / "B"
    round_root = root / "round_r00"
    round_root.mkdir(parents=True)
    partition = refinement.NestedPartition.base(32).split(0, "h_a_0", 0.0)
    (root / "PARTITION.json").write_text(json.dumps(partition.to_record()))
    q = np.eye(17)[np.zeros(33, dtype=int)]
    cost = np.full((33, 17), 2.0); cost[:, 0] = 0.1
    coeff = np.zeros_like(cost); coeff[:, 0] = 1/33
    cut = {"id": "A/SEX/H/U", "role": "A/SEX", "weighting": "U", "floor": .5,
           "coeff": coeff}
    np.savez_compressed(round_root / "Q.npz", Q=q)
    np.savez_compressed(round_root / "COSTS.npz", cost_U=cost, cost_W=cost)
    np.savez_compressed(round_root / "CUTS.npz", cut_0000=coeff)
    bank_sha = solver.replay_p1(q, cost, [cut])["bank_sha256"]
    (round_root / "BANK.json").write_text(json.dumps({
        "schema": 1, "bank_sha256": bank_sha,
        "cut_coefficients_sha256": fit_a._sha_file(round_root / "CUTS.npz"),
        "cuts": [{key: val for key, val in cut.items() if key != "coeff"}]}))
    (root / "FINAL_BANK_SELECTION.json").write_text(json.dumps({
        "selected_round": 0, "final_bank_sha256": bank_sha,
        "final_bank_checks": [{"round": 0, "feasible": True}]}))
    (root / "SELECTED.json").write_text(json.dumps({
        "selected_round": 0, "final_bank_sha256": bank_sha,
        "final_bank_selection_sha256": fit_a._sha_file(root / "FINAL_BANK_SELECTION.json"),
        "selected_channel_relative": "round_r00/Q.npz"}))
    complete = {"schema": "pcrl-adaptive-B-center-v1", "anchor": 0,
                "delta": .001, "status": "COMPLETE", "selected_round": 0,
                "selected_channel_relative": "round_r00/Q.npz",
                "final_bank_sha256": bank_sha,
                "partition_sha256": fit_a._sha_file(root / "PARTITION.json"),
                "rounds": [{"round": 0, "bank_sha256": bank_sha,
                            "channel_array_sha256": fit_b._array_sha256(q),
                            "channel_file_sha256": fit_a._sha_file(round_root / "Q.npz")}],
                "amendment_01_sha256": fit_b.AMENDMENT_01_SHA256,
                "artifact_sha256": fit_a._inventory(root)}
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    loaded = fit_controls.load_final_problem("B", root, anchor=0, delta=.001)
    assert loaded["parent_of_leaf"].shape == (33,)
    assert np.array_equal(loaded["selected_channel"], q)
    assert loaded["source"]["selected_bank_sha256"] == bank_sha
    assert loaded["cuts"][0]["floor"] == .5
    (round_root / "CUTS.npz").write_bytes((round_root / "CUTS.npz").read_bytes()+b"bad")
    with pytest.raises(ValueError, match="hash|inventory"):
        fit_controls.load_final_problem("B", root, anchor=0, delta=.001)


def test_control_cli_exposes_only_frozen_center_and_private_output_arguments():
    run = subprocess.run(
        [sys.executable, "-m", "experiments.pcrl_adaptive_release_v1.fit_controls", "--help"],
        capture_output=True, text=True, check=True)
    for option in ("--branch", "--center-dir", "--index", "--anchor",
                   "--delta", "--output-dir", "--a-center-dir"):
        assert option in run.stdout


@pytest.mark.parametrize("alias_status", ["SUPPORT_LIMITED_ALIAS_A",
                                           "NO_ACCEPTED_SPLIT_ALIAS_A"])
def test_b_alias_loader_accepts_exact_frozen_a_channel(tmp_path, monkeypatch, alias_status):
    from experiments.pcrl_adaptive_release_v1 import fit_b

    root = tmp_path / "private" / "b_alias"
    root.mkdir(parents=True)
    q = np.eye(17)[np.zeros(32, dtype=int)]
    np.savez_compressed(root / "Q.npz", Q=q)
    complete = {"schema": "pcrl-adaptive-B-center-v1", "anchor": 0,
                "delta": .001, "status": alias_status,
                "a_complete_receipt_sha256": "a"*64,
                "selected_channel_relative": "Q.npz",
                "artifact_sha256": fit_controls._inventory(root)}
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    cost = np.ones_like(q)
    monkeypatch.setattr(fit_b, "load_a_selected", lambda *args, **kwargs: {
        "cost": {"U": cost, "W": cost}, "cuts": [], "Q": q,
        "complete_receipt_sha256": "a"*64,
        "selected_bank_sha256": "b"*64})
    result = fit_controls.load_final_problem(
        "B", root, anchor=0, delta=.001,
        a_center_dir=tmp_path / "private" / "a")
    assert result["source"]["branch"] == "B_ALIAS_A"
    assert np.array_equal(result["selected_channel"], q)

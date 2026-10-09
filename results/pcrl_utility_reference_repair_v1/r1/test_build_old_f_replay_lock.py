"""Data-free predecessor checks for the optional independent old-F replay."""

import pytest

from results.pcrl_utility_reference_repair_v1.r1.build_old_f_replay_lock import (
    validate_old_diagnostic_predecessor,
)


def _records():
    outputs = {}
    units = {}
    for seed in (0, 1):
        folder = f"/study/.private/r1/old_f_inner/seed_{seed}"
        output = {f"{folder}/DIAGNOSTIC_RECEIPT.json": str(seed) * 64,
                  f"{folder}/OLD_F_INNER_PREDICTIONS.npz": str(seed + 2) * 64}
        outputs[seed] = output
        units[f"R1_OLD_F_DIAGNOSTIC_S{seed}"] = {
            "status": "DONE", "charged_cpu_s": 2.0, "output_sha256": output}
    previous = {"archived_queue_relative": "queue_g20",
                "archived_queue_sha256": "c" * 64,
                "charged_cpu_s": 100.0, "unit_count": 2}
    state = {"units": units, "carry_forward": previous, "used_cpu_s": 104.0}
    lock = {"schema": "pcrl-utility-campaign-lock-v1",
            "stage": "optional_old_validation_f_same_inner_diagnostic",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "required_queue_carry_forward": previous,
            "full_input_gate_sha256": "g" * 64,
            "old_f_diagnostic_inputs_sha256": "m" * 64}
    return state, lock, outputs


def test_old_diagnostic_done_binds_outputs_and_cpu():
    state, lock, outputs = _records()
    assert validate_old_diagnostic_predecessor(
        state, lock, 21, "q" * 64, outputs, "g" * 64, "m" * 64) == {
            "archived_queue_relative": "queue_g21",
            "archived_queue_sha256": "q" * 64,
            "charged_cpu_s": 104.0, "unit_count": 2}


@pytest.mark.parametrize("change", ("unfinished", "wrong_output", "wrong_gate",
                                    "cpu_mismatch"))
def test_old_diagnostic_predecessor_fails_closed(change):
    state, lock, outputs = _records()
    if change == "unfinished":
        state["units"]["R1_OLD_F_DIAGNOSTIC_S0"]["status"] = "FAILED"
    elif change == "wrong_output":
        state["units"]["R1_OLD_F_DIAGNOSTIC_S1"]["output_sha256"] = {}
    elif change == "wrong_gate":
        lock["full_input_gate_sha256"] = "0" * 64
    else:
        state["used_cpu_s"] = 103.0
    with pytest.raises(PermissionError):
        validate_old_diagnostic_predecessor(
            state, lock, 21, "q" * 64, outputs, "g" * 64, "m" * 64)

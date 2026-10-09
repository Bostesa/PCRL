"""Data-free checks for the post-interval independent replay seal."""

import pytest

from results.pcrl_utility_reference_repair_v1.r1.build_utility_interval_replay_lock import (
    validate_interval_predecessor,
)


def _records(output_path="/study/.private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"):
    state = {
        "units": {"R1_UTILITY_INTERVALS": {
            "status": "DONE", "charged_cpu_s": 3.0,
            "output_sha256": {output_path: "d" * 64}}},
        "carry_forward": {"archived_queue_relative": "queue_g13",
                          "charged_cpu_s": 100.0, "unit_count": 2},
        "used_cpu_s": 103.0,
    }
    lock = {"schema": "pcrl-utility-campaign-lock-v1",
            "stage": "r1_full_input_utility_intervals",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "required_queue_carry_forward": state["carry_forward"],
            "full_input_gate_sha256": "g" * 64,
            "interval_unit_proposal_sha256": "p" * 64,
            "source_map_sha256": "m" * 64}
    return state, lock


def test_done_interval_binds_exact_output_and_carry_forward():
    path = "/study/.private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"
    state, lock = _records(path)
    receipt = validate_interval_predecessor(
        state, lock, "a" * 64, path, "d" * 64,
        "g" * 64, "p" * 64, "m" * 64)
    assert receipt == {"archived_queue_relative": "queue_g14",
                       "archived_queue_sha256": "a" * 64,
                       "charged_cpu_s": 103.0, "unit_count": 1}


@pytest.mark.parametrize("change", ("unfinished", "output_mismatch", "cpu_mismatch",
                                    "source_lock_mismatch"))
def test_interval_predecessor_fails_closed(change):
    path = "/study/.private/r1/intervals/SIMULTANEOUS_UTILITY_INTERVALS.json"
    state, lock = _records(path)
    if change == "unfinished":
        state["units"]["R1_UTILITY_INTERVALS"]["status"] = "RUNNING"
    elif change == "output_mismatch":
        state["units"]["R1_UTILITY_INTERVALS"]["output_sha256"][path] = "0" * 64
    elif change == "cpu_mismatch":
        state["used_cpu_s"] = 102.0
    else:
        lock["source_map_sha256"] = "0" * 64
    with pytest.raises(PermissionError):
        validate_interval_predecessor(
            state, lock, "a" * 64, path, "d" * 64,
            "g" * 64, "p" * 64, "m" * 64)

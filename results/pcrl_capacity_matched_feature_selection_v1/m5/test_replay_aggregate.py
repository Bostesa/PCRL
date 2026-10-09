"""Data-free checks of the frozen aggregate replay admission contract."""

import argparse

import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5 import replay_aggregate


def test_wrong_phase_output_denied_before_running_gate(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(replay_aggregate, "STUDY", study)
    monkeypatch.setattr(replay_aggregate, "assert_running_unit", lambda *a: (_ for _ in ()).throw(
        AssertionError("running gate must not be reached")))
    args = argparse.Namespace(
        phase="SCREEN", unit_id="M5-REPLAY-AGGREGATE-SCREEN",
        queue_root=str(study / ".private/queue"),
        output_report=str(study / ".private/m5/aggregates/FINALIST/AGGREGATE_REPORT.json"),
    )
    with pytest.raises(PermissionError, match="frozen"):
        replay_aggregate._pre_role_gate(args)


def test_phase_roster_is_fixed():
    assert replay_aggregate.PHASE_SEEDS == {"SCREEN": (0, 1),
                                            "FINALIST": (0, 1, 2)}

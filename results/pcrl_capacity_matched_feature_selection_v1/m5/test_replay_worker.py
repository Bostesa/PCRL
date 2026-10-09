"""Data-free admission tests for the independent pair replay worker."""

import argparse

import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5 import replay_worker


def test_wrong_unit_or_output_denied_before_running_gate(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(replay_worker, "STUDY", study)
    monkeypatch.setattr(replay_worker, "assert_running_unit", lambda *a: (_ for _ in ()).throw(
        AssertionError("running gate must not be reached")))
    args = argparse.Namespace(
        unit_id="M5-REPLAY-S0-WRONG", seed=0,
        pair_id="M_DROP_NATIVE_COUNTRY__M_DROP_NATIVE_COUNTRY",
        queue_root=str(study / ".private/queue"),
        output_report=str(study / ".private/m5/replays/seed_0/WRONG/REPLAY_REPORT.json"),
        output_values=str(study / ".private/m5/replays/seed_0/WRONG/REPLAY_VALUES.npz"),
    )
    with pytest.raises(PermissionError, match="frozen"):
        replay_worker._pre_role_gate(args)


def test_role_contract_excludes_fit_protected_label():
    fields = replay_worker.role_keys(2)
    assert "OSF_DEFENSE_FIT:sex" not in fields
    assert "OSF_DEFENSE_FIT:features" in fields
    assert "OSF_DEFENSE_FIT:references_s2" in fields
    assert "INNER_SELECTION:sex" in fields
    assert len(fields) == 11

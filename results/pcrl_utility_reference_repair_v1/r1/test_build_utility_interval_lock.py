"""Data-free checks for the interval stage's sealed predecessor gate."""

import pytest

from results.pcrl_utility_reference_repair_v1.r1.build_utility_interval_lock import (
    bind_inner_task_role,
    launcher_path,
    validate_predecessor,
)


def _records():
    state = {
        "units": {"R2_PAIR_SELECT_S0": {"status": "DONE", "charged_cpu_s": 4.0},
                  "R2_PAIR_SELECT_S1": {"status": "DONE", "charged_cpu_s": 5.0}},
        "carry_forward": {"archived_queue_relative": "queue_g12",
                          "archived_queue_sha256": "c" * 64,
                          "charged_cpu_s": 90.0, "unit_count": 2},
        "used_cpu_s": 99.0,
        "epoch_start_utc": "2026-10-09T11:36:10+00:00",
    }
    audit = {"schema": "pcrl-utility-r3-pair-selection-audit-v1",
             "status": "PASS_AGGREGATE_SELECTION_REPLAY",
             "pair_search_queue_state_sha256": "a" * 64,
             "pair_search_lock_sha256": "b" * 64}
    return state, audit


def test_prior_pair_search_carry_forward_is_exact():
    state, audit = _records()
    receipt = validate_predecessor(state, audit, "a" * 64, "b" * 64)
    assert receipt == {"archived_queue_relative": "queue_g13",
                       "archived_queue_sha256": "a" * 64,
                       "charged_cpu_s": 99.0, "unit_count": 2}


def test_launcher_path_keeps_venv_symlink(tmp_path):
    real = tmp_path / "base-python"
    real.write_text("base")
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    launcher = venv_bin / "python"
    launcher.symlink_to(real)
    assert launcher_path(launcher) == launcher
    assert launcher_path(launcher).name == "python"
    assert launcher_path(launcher).parent.parent.name == ".venv"


def test_role_binding_uses_manifest_hash_without_opening_archive(tmp_path, monkeypatch):
    study = tmp_path / "study"
    role = study / ".private/role_export_v1/INNER_SELECTION/income.npz"
    role.parent.mkdir(parents=True)
    role.touch()
    relative = ".private/role_export_v1/INNER_SELECTION/income.npz"
    manifest = {"role_file_relative_path": {"INNER_SELECTION:income": relative},
                "role_file_sha256": {"INNER_SELECTION:income": "a" * 64}}
    proposal = {"shared_inputs_sha256_by_study_relative_path": {relative: "a" * 64}}
    original_open = type(role).open

    def denied_open(path, *args, **kwargs):
        if path == role:
            raise AssertionError("unsealed role archive must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(role), "open", denied_open)
    inputs = {}
    record = bind_inner_task_role(study, manifest, proposal, "income", inputs)
    assert record == {"path": str(role), "sha256": "a" * 64}
    assert inputs == {str(role): "a" * 64}


@pytest.mark.parametrize("change", ("unfinished", "audit_mismatch", "cpu_mismatch",
                                    "missing_unit_cpu"))
def test_prior_pair_search_fails_closed(change):
    state, audit = _records()
    if change == "unfinished":
        state["units"]["R2_PAIR_SELECT_S1"]["status"] = "RUNNING"
    elif change == "audit_mismatch":
        audit["pair_search_queue_state_sha256"] = "0" * 64
    elif change == "missing_unit_cpu":
        del state["units"]["R2_PAIR_SELECT_S1"]["charged_cpu_s"]
    else:
        state["used_cpu_s"] = 98.0
    with pytest.raises(PermissionError):
        validate_predecessor(state, audit, "a" * 64, "b" * 64)

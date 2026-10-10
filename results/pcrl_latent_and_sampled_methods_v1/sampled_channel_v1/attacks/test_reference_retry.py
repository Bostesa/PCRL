"""Data-free checks for the dated 12-attempt reference attack correction."""

import sys

import pytest

from . import run_reference_retry as worker


@pytest.mark.parametrize("ref,expected", [
    ("P_STAR", "b" * 64),
    ("D0_LOCAL", "b" * 64),
    ("D0_SEQ12", "b" * 64),
    ("D0_SEQ21", "b" * 64),
    ("T_STAR", "a" * 64),
    ("A_IDENTITY", "a" * 64),
    ("A_CLASS_ONLY", "a" * 64),
])
def test_fixed_map_audit_uses_the_actual_source_family(ref, expected):
    amendment = {"source_sha256": {
        "r01/SOURCE_AUDIT.json": "a" * 64,
        "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json": "b" * 64,
    }}
    assert worker._fixed_map_expected_audit_sha(ref, amendment) == expected


def test_fixed_map_audit_refuses_missing_historical_pin():
    amendment = {"source_sha256": {"r01/SOURCE_AUDIT.json": "a" * 64}}
    with pytest.raises(PermissionError, match="not pinned"):
        worker._fixed_map_expected_audit_sha("P_STAR", amendment)
    with pytest.raises(ValueError, match="no fixed-map"):
        worker._fixed_map_expected_audit_sha("U0", amendment)


def test_retry_worker_refuses_missing_slot_before_any_file_read(monkeypatch):
    def no_slot(**_):
        raise PermissionError("synthetic slot absent")
    monkeypatch.setattr(worker, "assert_active_from_env", no_slot)
    with pytest.raises(PermissionError, match="synthetic slot absent"):
        worker.admit_reference_retry(
            unit_id="A-REFERENCE-ATTACK-U0-S0-R1",
            original_attack_unit_id="A-REFERENCE-ATTACK-U0-S0",
            correction_path="unread", correction_sha256="a" * 64,
            downstream_manifest_path="unread", downstream_manifest_sha256="b" * 64,
            downstream_registration_path="unread",
            downstream_registration_sha256="c" * 64,
            original_lock_path="unread", original_lock_sha256="d" * 64,
            a_amendment_path="unread", a_amendment_sha256="e" * 64,
            role_manifest_path="unread", role_manifest_sha256="f" * 64,
            att_release_dir="unread", inner_release_dir="unread",
            fixed_map_receipt=None, fixed_map_npz=None, q_table_dir=None,
            private_root="unread")


def test_retry_command_refuses_different_script_flags_or_thread_budget(
        monkeypatch, tmp_path):
    module = ("results.pcrl_latent_and_sampled_methods_v1."
              "sampled_channel_v1.attacks.run_reference_retry")
    monkeypatch.setattr(sys, "argv", [str(worker.__file__), "--unit-id", "physical-R1"])
    spec = {
        "argv": [sys.executable, "-m", module, "--unit-id", "physical-R1"],
        "cwd": str(tmp_path),
        "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"},
    }
    monkeypatch.chdir(tmp_path)
    worker._checked_retry_command(spec)
    for changed in (
        {**spec, "argv": spec["argv"][:-1] + ["wrong"]},
        {**spec, "argv": spec["argv"][:2] + ["wrong"] + spec["argv"][3:]},
        {**spec, "env": {**spec["env"], "OMP_NUM_THREADS": "2"}},
    ):
        with pytest.raises(PermissionError, match="command differs"):
            worker._checked_retry_command(changed)

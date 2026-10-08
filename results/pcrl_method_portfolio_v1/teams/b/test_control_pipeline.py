"""Synthetic full-arm orchestration; run only through the S3 numerical queue."""

import stat

import numpy as np
import pytest

from bottleneck import model_fingerprint
from control_artifacts import critic_fingerprint
from control_pipeline import FitAdmissionSeal, fit_arm
from test_control_artifacts import fixtures


def admitted(data):
    rows = np.asarray(data.row_ids, dtype=np.int64)
    groups = np.asarray(data.group_ids, dtype=np.int64)
    return FitAdmissionSeal(
        source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
        role_only_export_sha256="e" * 64,
        reference_export_sha256="d" * 64,
        source_npz_sha256="c" * 64,
        admission_manifest_sha256="f" * 64,
        admitted_rows=len(rows), admitted_groups=np.unique(groups).size,
        admitted_row_id_sha256=FitAdmissionSeal._sha_arr(np.sort(rows)),
        admitted_group_id_sha256=FitAdmissionSeal._sha_arr(np.unique(groups)),
        admitted_row_unit_pair_sha256=FitAdmissionSeal._sha_arr(
            np.column_stack((rows, groups)).astype(np.int64)),
        data_sha256=data.sha256, row_order_sha256=data.row_order_sha256,
        group_order_sha256=data.group_order_sha256,
        feature_schema_sha256=data.feature_schema_sha256,
        teacher_identity=tuple(sorted(data.teacher_identity.items())),
    )


@pytest.mark.parametrize("arm,stage_count", [
    ("B-JVIB", 1), ("B-DVIB", 1), ("B-TASK", 1),
    ("B-RAW-TASK", 1), ("B-LOCAL", 2),
    ("B-SEQ12", 2), ("B-SEQ21", 2), ("B-RAWJ", 1),
])
def test_registered_arm_completes_with_private_provenance(tmp_path, arm, stage_count):
    base, critics, data, hyper = fixtures()
    root = tmp_path / ".private" / arm
    result = fit_arm(
        arm, 0, data=data, base_model=base, base_critics=critics,
        hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
        admitted_fit=admitted(data),
        private_attempt_dir=root,
        expected_base_model_sha256=model_fingerprint(base),
        expected_base_critics_sha256=critic_fingerprint(critics),
        sampler_secret=b"s" * 32,
        sampler_plan_sha256="a" * 64,
    )
    assert result.arm == arm
    assert len(result.stage_receipts) == stage_count
    assert all(receipt["status"] == "COMPLETED" for receipt in result.stage_receipts)
    assert result.base_model_sha256 == model_fingerprint(base)
    assert result.private_artifact_sha256
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    if arm == "B-LOCAL":
        assert "assembled_local_pair" in result.private_artifact_sha256
    if arm.startswith("B-SEQ"):
        assert "first_fixed_draw_file" in result.private_artifact_sha256


def test_wrong_common_seed_identity_refuses_before_creating_attempt(tmp_path):
    base, critics, data, hyper = fixtures()
    root = tmp_path / ".private" / "bad"
    with pytest.raises(ValueError, match="common per-seed"):
        fit_arm(
            "B-TASK", 0, data=data, base_model=base, base_critics=critics,
            hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
            admitted_fit=admitted(data),
            private_attempt_dir=root,
            expected_base_model_sha256="0" * 64,
            expected_base_critics_sha256=critic_fingerprint(critics),
        )


def test_wrong_fit_membership_refuses_before_creating_attempt(tmp_path):
    base, critics, data, hyper = fixtures()
    root = tmp_path / ".private" / "wrong-membership"
    original = admitted(data)
    tampered = FitAdmissionSeal(
        **{**vars(original), "row_order_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="differs from S2"):
        fit_arm(
            "B-TASK", 0, data=data, base_model=base, base_critics=critics,
            hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
            admitted_fit=tampered, private_attempt_dir=root,
            expected_base_model_sha256=model_fingerprint(base),
            expected_base_critics_sha256=critic_fingerprint(critics),
        )
    assert not root.exists()

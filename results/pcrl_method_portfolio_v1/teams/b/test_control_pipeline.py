"""Synthetic full-arm orchestration; run only through the S3 numerical queue."""

import stat
import hashlib
import json

import numpy as np
import pytest

from bottleneck import model_fingerprint
from control_artifacts import critic_fingerprint
import control_pipeline
from control_pipeline import FitAdmissionSeal, fit_arm
from test_control_artifacts import fixtures
from train_bottleneck import FitData


def task_only_data(data):
    return FitData(
        x=data.x, feature_names=data.feature_names,
        permitted_feature_names=data.feature_names,
        expected_schema_sha256=data.feature_schema_sha256,
        labels=data.labels, teacher_probabilities=data.teacher_probabilities,
        sex=None, row_ids=data.row_ids, group_ids=data.group_ids,
        teacher_identity=data.teacher_identity,
    )


def admitted(data, tmp_path, *, include_sex=True):
    rows = np.asarray(data.row_ids, dtype=np.int64)
    groups = np.asarray(data.group_ids, dtype=np.int64)
    row_hash = FitAdmissionSeal._sha_arr(np.sort(rows))
    group_hash = FitAdmissionSeal._sha_arr(np.unique(groups))
    pair_hash = FitAdmissionSeal._sha_arr(np.column_stack((rows, groups)).astype(np.int64))
    root = tmp_path / "admission"
    root.mkdir(exist_ok=True)
    specs = {}
    fields = (["features", "income", "occupation", "sex", "references_s0"]
              if include_sex else
              ["features", "income", "occupation", "references_s0"])
    for field in fields:
        path = root / f"{field}.npz"
        path.write_bytes(f"synthetic admitted {field}".encode())
        specs[f"OSF_DEFENSE_FIT:{field}"] = {
            "role": "OSF_DEFENSE_FIT", "field": field, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_unit_pair_sha256": pair_hash,
        }
    manifest = {
        "schema": "pcrl-method-portfolio-data-access-v1",
        "status": "ROLE_FILES_ADMITTED_S4_VERIFIED",
        "source": {
            "source_npz_sha256": "c" * 64,
            "feature_names_sha256_newline_join": hashlib.sha256(
                "\n".join(data.feature_names).encode()).hexdigest(),
        },
        "roles": {"OSF_DEFENSE_FIT": {
            "rows": len(rows), "groups": np.unique(groups).size,
            "row_id_sha256": row_hash, "group_id_sha256": group_hash,
        }},
        "real_file_specs": specs,
    }
    manifest_path = root / "DATA_ACCESS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return FitAdmissionSeal(
        source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
        reference_seed=0, admission_manifest_path=str(manifest_path),
        role_only_export_sha256=specs["OSF_DEFENSE_FIT:features"]["sha256"],
        reference_export_sha256=specs["OSF_DEFENSE_FIT:references_s0"]["sha256"],
        source_npz_sha256="c" * 64,
        admission_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        admitted_rows=len(rows), admitted_groups=np.unique(groups).size,
        admitted_row_id_sha256=row_hash,
        admitted_group_id_sha256=group_hash,
        admitted_row_unit_pair_sha256=pair_hash,
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
    task_only = arm in ("B-TASK", "B-RAW-TASK")
    if task_only:
        data = task_only_data(data)
    root = tmp_path / ".private" / arm
    result = fit_arm(
        arm, 0, data=data, base_model=base, base_critics=critics,
        hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
        admitted_fit=admitted(data, tmp_path, include_sex=not task_only),
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
    data = task_only_data(data)
    root = tmp_path / ".private" / "bad"
    with pytest.raises(ValueError, match="common per-seed"):
        fit_arm(
            "B-TASK", 0, data=data, base_model=base, base_critics=critics,
            hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
            admitted_fit=admitted(data, tmp_path, include_sex=False),
            private_attempt_dir=root,
            expected_base_model_sha256="0" * 64,
            expected_base_critics_sha256=critic_fingerprint(critics),
        )


def test_wrong_fit_membership_refuses_before_creating_attempt(tmp_path):
    base, critics, data, hyper = fixtures()
    data = task_only_data(data)
    root = tmp_path / ".private" / "wrong-membership"
    original = admitted(data, tmp_path, include_sex=False)
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


@pytest.mark.parametrize("arm", ["B-TASK", "B-RAW-TASK"])
def test_task_only_arm_never_hashes_sex_file(tmp_path, monkeypatch, arm):
    base, critics, full_data, hyper = fixtures()
    data = task_only_data(full_data)
    seal = admitted(data, tmp_path, include_sex=True)
    original_hash = control_pipeline.file_sha256

    def refuse_sex(path):
        if str(path).endswith("sex.npz"):
            raise AssertionError("Task-only arm read the SEX role-only file")
        return original_hash(path)

    monkeypatch.setattr(control_pipeline, "file_sha256", refuse_sex)
    fit_arm(
        arm, 0, data=data, base_model=base, base_critics=critics,
        hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
        admitted_fit=seal, private_attempt_dir=tmp_path / ".private" / arm,
        expected_base_model_sha256=model_fingerprint(base),
        expected_base_critics_sha256=critic_fingerprint(critics),
    )


def test_task_only_arm_rejects_sex_materialized_by_caller(tmp_path):
    base, critics, data, hyper = fixtures()
    with pytest.raises(ValueError, match="SEX fitting access"):
        fit_arm(
            "B-TASK", 0, data=data, base_model=base, base_critics=critics,
            hyper=hyper, source_id="synthetic-source", role_id="OSF_DEFENSE_FIT",
            admitted_fit=admitted(data, tmp_path),
            private_attempt_dir=tmp_path / ".private" / "wrong-sex",
            expected_base_model_sha256=model_fingerprint(base),
            expected_base_critics_sha256=critic_fingerprint(critics),
        )

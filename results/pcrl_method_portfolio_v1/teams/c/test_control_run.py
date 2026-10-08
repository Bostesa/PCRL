"""Synthetic admission and custody tests; no Adult values or inherited loader."""

import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import control_run
from control_run import (
    ArmOutput, RoleData, _check_seal, _fit_or_restore_calibrator, _fit_or_restore_head, _sha_spec,
    _team_c_fit_lock, _write_json_once, admit_control_roles, run_controls,
    run_task_only_control,
)
from prototype import AdultGroupedSchema
from selector import TaskDelta
from training import HeadSpec


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_draft_campaign_lock_cannot_admit_science_fit(tmp_path):
    lock_path = tmp_path / "CAMPAIGN_LOCK.json"
    lock_path.write_text(json.dumps({
        "schema": "pcrl-method-portfolio-campaign-lock-v1",
        "status": "BLOCKING_DRAFT_CONTINUATION_NOT_SEALED",
        "adult_fit_permission": False,
    }))
    with pytest.raises(PermissionError, match="not effective"):
        _check_seal(
            lock_path, _sha(lock_path),
            expected_manifest_sha256="a" * 64,
            expected_head_spec_sha256="b" * 64,
            local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
        )


def test_runner_rejects_draft_before_queue_or_role_read(tmp_path):
    lock_path = tmp_path / "CAMPAIGN_LOCK.json"
    lock_path.write_text(json.dumps({
        "schema": "pcrl-method-portfolio-campaign-lock-v1",
        "status": "BLOCKING_DRAFT_CONTINUATION_NOT_SEALED",
        "adult_fit_permission": False,
    }))

    class NoReads:
        def read(self, *args):
            raise AssertionError("forbidden role read")

    def no_call(*args, **kwargs):
        raise AssertionError("fit, save, or queue call before science seal")

    spec = HeadSpec()
    with pytest.raises(PermissionError, match="not effective"):
        run_controls(
            NoReads(), seed=0, head_spec=spec,
            expected_head_spec_sha256=_sha_spec(spec),
            expected_manifest_sha256="a" * 64,
            campaign_lock_path=lock_path,
            campaign_lock_sha256=_sha(lock_path),
            local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
            fit_calibrator=no_call, save_calibrator=no_call,
            load_calibrator=no_call, apply_global_temperature=no_call,
            attack_view_type=no_call,
            heavy_slot=no_call, private_checkpoint_root=tmp_path / ".private",
        )


def test_pending_role_manifest_rejects_before_any_role_read(tmp_path):
    manifest_path = tmp_path / "DATA_ACCESS_MANIFEST.json"
    manifest = {
        "schema": "pcrl-method-portfolio-data-access-v1",
        "status": "PENDING_INDEPENDENT_ROLE_AUDIT",
    }
    manifest_path.write_text(json.dumps(manifest))

    class DeniedStore:
        def __init__(self):
            self.manifest_path = manifest_path
            self.manifest = manifest
            self.read_calls = 0

        def read(self, *args):
            self.read_calls += 1
            raise AssertionError("role read occurred before admission")

    store = DeniedStore()
    with pytest.raises(PermissionError, match="not independently admitted"):
        admit_control_roles(store, expected_manifest_sha256=_sha(manifest_path))
    assert store.read_calls == 0


def test_private_receipt_rejects_changed_identity(tmp_path):
    path = tmp_path / "private" / "unit.json"
    _write_json_once(path, {"identity": "first"})
    original = path.read_bytes()
    with pytest.raises(PermissionError, match="identity collision"):
        _write_json_once(path, {"identity": "second"})
    assert path.read_bytes() == original


def test_exact_head_identity_restores_without_second_fit(tmp_path, monkeypatch):
    spec = HeadSpec()
    calls = {"fit": 0, "restore": 0}
    fake_head = SimpleNamespace(
        model_id="synthetic-head", purpose="income", selected_atoms=("age",),
        spec=spec, receipt={"test": "synthetic"},
    )

    def fake_fit(*args, **kwargs):
        calls["fit"] += 1
        return fake_head

    def fake_save(head, path, *, private_root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint")
        return _sha(path)

    def fake_restore(path, *, expected_sha256, private_root):
        calls["restore"] += 1
        assert _sha(path) == expected_sha256
        return fake_head

    monkeypatch.setattr(control_run, "fit_task_head", fake_fit)
    monkeypatch.setattr(control_run, "save_head", fake_save)
    monkeypatch.setattr(control_run, "restore_head", fake_restore)
    fitting = RoleData(
        "OSF_DEFENSE_FIT", np.zeros((1, 83)), ("unused",),
        np.array([1]), np.array([1]), {"income": np.array([0])},
        {"features": "a" * 64, "income": "b" * 64},
    )
    kwargs = dict(
        seed=0, purpose="income", field="income", selected_atoms=("age",),
        spec=spec, model_id="synthetic-head", campaign_lock_sha256="c" * 64,
        data_manifest_sha256="d" * 64,
        private_checkpoint_root=tmp_path / ".private",
    )
    first = _fit_or_restore_head(fitting, **kwargs)
    second = _fit_or_restore_head(fitting, **kwargs)
    assert (first[3], second[3]) == ("COMPLETED", "RESTORED_EXACT_IDENTITY")
    assert calls == {"fit": 1, "restore": 1}
    assert first[1] == second[1]


def test_calibrator_refuses_wrong_heldout_identity_before_save(tmp_path):
    schema = AdultGroupedSchema()
    head = SimpleNamespace(
        purpose="income", model_id="synthetic-head", spec=HeadSpec(),
        predict_proba=lambda selected: np.tile([0.6, 0.4], (len(selected), 1)),
    )
    role = RoleData(
        "CALIBRATION_HELDOUT", np.zeros((2, 83)), schema.feature_names,
        np.array([7, 8], dtype=np.int64), np.array([70, 80], dtype=np.int64),
        {"income": np.array([0, 1], dtype=np.int64)},
        {"features": "a" * 64, "income": "b" * 64},
    )
    save_calls = []

    def wrong_record(raw, labels, **identity):
        return {
            "schema": "pcrl-portfolio-global-temp-v1",
            "task": identity["task"], "arm_id": identity["arm_id"],
            "model_seed": identity["model_seed"],
            "fit_role": "CALIBRATION_HELDOUT", "draw_id": "deterministic",
            "fit_row_unit_pair_sha256": "0" * 64,
        }

    with pytest.raises(PermissionError, match="differs from admitted role/head"):
        _fit_or_restore_calibrator(
            head, role, field="income", seed=0, selected_atoms=("age",),
            schema=schema, campaign_lock_sha256="c" * 64,
            data_manifest_sha256="d" * 64,
            private_checkpoint_root=tmp_path / ".private",
            fit_calibrator=wrong_record,
            save_calibrator=lambda *args: save_calls.append(args),
            load_calibrator=lambda *args: None,
        )
    assert save_calls == []


def test_task_only_worker_never_requests_sensitive_label(tmp_path, monkeypatch):
    roles = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
             "ATTACK_FIT_NEW", "INNER_SELECTION")
    fields = {
        "OSF_DEFENSE_FIT": ("features", "income", "occupation"),
        "CALIBRATION_HELDOUT": ("features", "income", "occupation"),
        "ATTACK_FIT_NEW": ("features",),
        "INNER_SELECTION": ("features",),
    }
    manifest = {
        "schema": "pcrl-method-portfolio-data-access-v1",
        "status": "ROLE_FILES_ADMITTED_S4_VERIFIED",
        "custodian_export": {"s4_report_sha256": "b" * 64},
        "source": {"source_npz_sha256": (
            "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
        )},
        "real_file_specs": {
            f"{role}:{field}": {"sha256": "a" * 64}
            for role in roles for field in fields[role]
        },
    }
    manifest_path = tmp_path / "DATA_ACCESS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    schema = AdultGroupedSchema()

    class SentinelStore:
        def __init__(self):
            self.manifest_path = manifest_path
            self.manifest = manifest
            self.read_calls = []

        def read(self, role, field, procedure=None):
            self.read_calls.append((role, field, procedure))
            if field == "sex":
                raise AssertionError("task-only worker requested SEX")
            if role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
                raise AssertionError("task-only fitting worker opened screen rows")
            base = 1000 * roles.index(role)
            row_id = base + np.arange(12, dtype=np.int64)
            unit = row_id.copy()
            if field == "features":
                return {
                    "row_id": row_id, "unit": unit,
                    "X": np.zeros((12, 83), dtype=np.float32),
                    "feature_names": schema.feature_names,
                }
            return {
                "row_id": row_id, "unit": unit,
                field: (np.arange(12, dtype=np.int64) %
                        (2 if field == "income" else 6)),
            }

    store = SentinelStore()
    atoms = tuple(sorted(schema.atoms))
    task_deltas = {
        task: {atom: TaskDelta(0.0, 0.0, 0.0) for atom in atoms}
        for task in ("income", "occupation_group")
    }
    monkeypatch.setattr(control_run, "_check_seal", lambda *a, **k: "b" * 64)
    monkeypatch.setattr(control_run, "build_task_scorecard", lambda *a, **k:
        SimpleNamespace(optional_atoms=atoms, task_deltas=task_deltas,
                        fit_counts={"task_proxy": 20, "sensitive_proxy": 0, "total": 20},
                        split_receipt={"sensitive_label_input": False}))

    def no_fit(admission, plan, **kwargs):
        assert "sex" not in admission.roles["OSF_DEFENSE_FIT"].labels
        assert "sex" not in admission.roles["OSF_DEFENSE_FIT"].file_hashes
        assert set(admission.roles) == {"OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"}
        assert tuple(a.arm_id for a in plan.arms) == ("C-SIMPLE-TASK-SHARED1",)
        assert kwargs["forward"] is False
        arm = plan.arms[0]
        return {arm.arm_id: ArmOutput(
            arm, "FROZEN_PACKAGE_READY_FOR_FORWARD", None, None, None, None,
            ("head1", "head2"), ("cal1", "cal2"),
            {"package_units": ({"purpose": "income"},
                               {"purpose": "occupation_group"})},
        )}

    monkeypatch.setattr(control_run, "_fit_and_calibrate", no_fit)
    monkeypatch.setattr(control_run, "_publish_fit_handoff", lambda *a, **k: (
        tmp_path / ".private" / "task-handoff.json", "d" * 64,
    ))
    spec = HeadSpec()
    no_call = lambda *args, **kwargs: None
    result = run_task_only_control(
        store, seed=7, head_spec=spec,
        expected_head_spec_sha256=_sha_spec(spec),
        expected_manifest_sha256=_sha(manifest_path),
        campaign_lock_path=tmp_path / "CAMPAIGN_LOCK.json",
        campaign_lock_sha256="c" * 64,
        local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
        fit_calibrator=no_call, save_calibrator=no_call,
        load_calibrator=no_call, apply_global_temperature=no_call,
        attack_view_type=no_call, heavy_slot=lambda _: nullcontext(),
        private_checkpoint_root=tmp_path / ".private",
    )
    assert result.scorecard_receipt["worker_scope"] == "TASK_ONLY_NO_SEX_READ"
    assert result.scorecard_receipt["fit_counts"]["total"] == 20
    assert not any(field == "sex" for _, field, _ in store.read_calls)
    assert len(store.read_calls) == 6
    assert {role for role, _, _ in store.read_calls} == {
        "OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
    }
    assert not any(key.endswith(":sex") for key in result.role_file_hashes)
    assert result.scorecard_receipt["worker_pid"] > 0


def test_privacy_fit_admission_defers_attack_and_inner_features(tmp_path):
    roles = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
             "ATTACK_FIT_NEW", "INNER_SELECTION")
    fields = {
        "OSF_DEFENSE_FIT": ("features", "income", "occupation", "sex"),
        "CALIBRATION_HELDOUT": ("features", "income", "occupation"),
        "ATTACK_FIT_NEW": ("features",),
        "INNER_SELECTION": ("features",),
    }
    manifest = {
        "schema": "pcrl-method-portfolio-data-access-v1",
        "status": "ROLE_FILES_ADMITTED_S4_VERIFIED",
        "custodian_export": {"s4_report_sha256": "b" * 64},
        "source": {"source_npz_sha256": (
            "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
        )},
        "real_file_specs": {
            f"{role}:{field}": {"sha256": "a" * 64}
            for role in roles for field in fields[role]
        },
    }
    path = tmp_path / "DATA_ACCESS_MANIFEST.json"
    path.write_text(json.dumps(manifest))
    schema = AdultGroupedSchema()
    reads = []

    def read(role, field, procedure=None):
        reads.append((role, field, procedure))
        if role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            raise AssertionError("screen-role feature opened before fit handoff")
        row_id = np.arange(12, dtype=np.int64) + 1000 * roles.index(role)
        if field == "features":
            return {"row_id": row_id, "unit": row_id.copy(),
                    "X": np.zeros((12, 83), dtype=np.float32),
                    "feature_names": schema.feature_names}
        return {"row_id": row_id, "unit": row_id.copy(),
                field: np.arange(12, dtype=np.int64) %
                (6 if field == "occupation" else 2)}

    store = SimpleNamespace(manifest_path=path, manifest=manifest, read=read)
    admission = admit_control_roles(
        store, expected_manifest_sha256=_sha(path), scope="privacy",
    )
    assert set(admission.roles) == {"OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT"}
    assert "sex" in admission.roles["OSF_DEFENSE_FIT"].labels
    assert len(reads) == 7
    assert not any(role in ("ATTACK_FIT_NEW", "INNER_SELECTION") for role, _, _ in reads)


def test_fit_lock_refuses_a_second_identity_writer(tmp_path):
    root = tmp_path / ".private"
    lock_sha = "a" * 64
    path = root / "team_c" / lock_sha / "fit.lock"
    probe = (
        "import fcntl, os, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "try:\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(11)\n"
        "finally:\n"
        "    os.close(fd)\n"
    )
    with _team_c_fit_lock(root, lock_sha):
        blocked = subprocess.run([sys.executable, "-c", probe, str(path)],
                                 check=False, capture_output=True)
        assert blocked.returncode == 11
    acquired = subprocess.run([sys.executable, "-c", probe, str(path)],
                              check=False, capture_output=True)
    assert acquired.returncode == 0

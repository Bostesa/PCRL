"""Synthetic admission and custody tests; no Adult values or inherited loader."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import control_run
from control_run import (
    RoleData, _check_seal, _fit_or_restore_calibrator, _fit_or_restore_head, _sha_spec,
    _write_json_once, admit_control_roles, run_controls,
)
from prototype import AdultGroupedSchema
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

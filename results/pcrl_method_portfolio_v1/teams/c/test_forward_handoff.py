"""Synthetic staged C forward test; no Adult source or SEX read."""

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import control_run
from control_run import restore_forward_for_s2, run_forward_from_handoff
from prototype import AdultGroupedSchema
from test_s2_bridge import S2AttackView, public_releases
from view_custody import restore_view_bundle


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_forward_opens_only_screen_features_after_frozen_fit_receipt(tmp_path, monkeypatch):
    lock_sha = "c" * 64
    manifest_sha_placeholder = "d" * 64
    private = tmp_path / ".private"
    arm_id = "C-SIMPLE-INDEP1"
    seed = 3
    sources = {
        "OSF_DEFENSE_FIT:features": "1" * 64,
        "OSF_DEFENSE_FIT:income": "2" * 64,
        "OSF_DEFENSE_FIT:occupation": "3" * 64,
        "OSF_DEFENSE_FIT:sex": "4" * 64,
        "CALIBRATION_HELDOUT:features": "5" * 64,
        "CALIBRATION_HELDOUT:income": "6" * 64,
        "CALIBRATION_HELDOUT:occupation": "7" * 64,
        "ATTACK_FIT_NEW:features": "8" * 64,
        "INNER_SELECTION:features": "9" * 64,
    }
    manifest = {
        "schema": "pcrl-method-portfolio-data-access-v1",
        "status": "ROLE_FILES_ADMITTED_S4_VERIFIED",
        "custodian_export": {"s4_report_sha256": "b" * 64},
        "source": {"source_npz_sha256": (
            "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
        )},
        "real_file_specs": {key: {"sha256": value}
                            for key, value in sources.items()},
    }
    manifest_path = tmp_path / "DATA_ACCESS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = _sha(manifest_path)
    releases = public_releases()
    package_hashes = {"income": "a" * 64, "occupation_group": "e" * 64}
    package_units = []
    for purpose, label in (("income", "income"),
                           ("occupation_group", "occupation")):
        package_units.append({
            "purpose": purpose,
            "package_relative_path": f"team_c/{lock_sha}/packages/{purpose}.json",
            "package_sha256": package_hashes[purpose],
            "source_role_hashes": {
                "fit_features": sources["OSF_DEFENSE_FIT:features"],
                "fit_label": sources[f"OSF_DEFENSE_FIT:{label}"],
                "calibration_features": sources["CALIBRATION_HELDOUT:features"],
                "calibration_label": sources[f"CALIBRATION_HELDOUT:{label}"],
            },
        })
    handoff = {
        "schema": "team-c-fit-handoff-v1",
        "worker_scope": "PRIVACY_FIT_CAL_NO_SCREEN_ROWS",
        "worker_pid": 1,
        "seed": seed, "arm_id": arm_id,
        "campaign_lock_sha256": lock_sha,
        "data_manifest_sha256": manifest_sha,
        "head_spec_sha256": "f" * 64,
        "s4_audit_receipt_sha256": "b" * 64,
        "selected_atoms": {
            "income": list(releases["income"].selected_atoms),
            "occupation_group": list(releases["occupation_group"].selected_atoms),
        },
        "fit_cal_role_file_hashes": {key: value for key, value in sources.items()
                                     if not key.startswith(("ATTACK", "INNER"))},
        "arm_receipt": {"package_units": package_units},
        "head_ids": [releases["income"].head_id,
                     releases["occupation_group"].head_id],
        "calibrator_ids": [releases["income"].calibrator_id,
                           releases["occupation_group"].calibrator_id],
    }
    fit_path = (private / "team_c" / lock_sha / "fit_handoff" /
                f"{arm_id}__s{seed}.json")
    fit_path.parent.mkdir(parents=True)
    fit_path.write_text(json.dumps(handoff))
    fit_sha = _sha(fit_path)
    restored = []

    def fake_restore(path, **kwargs):
        purpose = Path(path).stem
        restored.append(purpose)
        return releases[purpose]

    monkeypatch.setattr(control_run, "_check_seal", lambda *a, **k: "b" * 64)
    monkeypatch.setattr(control_run, "restore_release_package", fake_restore)
    schema = AdultGroupedSchema()

    class SentinelStore:
        def __init__(self):
            self.manifest_path = manifest_path
            self.manifest = manifest
            self.calls = []

        def read(self, role, field, procedure=None):
            self.calls.append((role, field, procedure))
            if len(restored) != 2:
                raise AssertionError("screen features read before frozen packages restored")
            if role not in ("ATTACK_FIT_NEW", "INNER_SELECTION") or field != "features":
                raise AssertionError("forward worker opened a label or fitting row")
            offset = 10 if role == "ATTACK_FIT_NEW" else 100
            row_id = offset + np.arange(4, dtype=np.int64)
            return {"row_id": row_id, "unit": row_id.copy(),
                    "X": np.zeros((4, 83), dtype=np.float32),
                    "feature_names": schema.feature_names}

    store = SentinelStore()
    with pytest.raises(PermissionError, match="fit handoff changed"):
        run_forward_from_handoff(
            store, arm_id=arm_id, seed=seed,
            fit_handoff_sha256="0" * 64,
            expected_head_spec_sha256="f" * 64,
            expected_manifest_sha256=manifest_sha,
            campaign_lock_path=tmp_path / "CAMPAIGN_LOCK.json",
            campaign_lock_sha256=lock_sha,
            local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
            load_calibrator=lambda *a: {},
            apply_global_temperature=lambda x, _: x,
            attack_view_type=S2AttackView,
            heavy_slot=lambda _: nullcontext(),
            private_checkpoint_root=private,
        )
    assert store.calls == []
    assert restored == []
    original_handoff_bytes = fit_path.read_bytes()
    fit_path.write_text(json.dumps({**handoff, "worker_pid": os.getpid()}))
    with pytest.raises(PermissionError, match="wrong locked identity"):
        run_forward_from_handoff(
            store, arm_id=arm_id, seed=seed,
            fit_handoff_sha256=_sha(fit_path),
            expected_head_spec_sha256="f" * 64,
            expected_manifest_sha256=manifest_sha,
            campaign_lock_path=tmp_path / "CAMPAIGN_LOCK.json",
            campaign_lock_sha256=lock_sha,
            local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
            load_calibrator=lambda *a: {},
            apply_global_temperature=lambda x, _: x,
            attack_view_type=S2AttackView,
            heavy_slot=lambda _: nullcontext(),
            private_checkpoint_root=private,
        )
    assert store.calls == []
    assert restored == []
    fit_path.write_bytes(original_handoff_bytes)
    result = run_forward_from_handoff(
        store, arm_id=arm_id, seed=seed,
        fit_handoff_sha256=fit_sha,
        expected_head_spec_sha256="f" * 64,
        expected_manifest_sha256=manifest_sha,
        campaign_lock_path=tmp_path / "CAMPAIGN_LOCK.json",
        campaign_lock_sha256=lock_sha,
        local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
        load_calibrator=lambda *a: {},
        apply_global_temperature=lambda x, _: x,
        attack_view_type=S2AttackView,
        heavy_slot=lambda _: nullcontext(),
        private_checkpoint_root=private,
    )
    assert result["status"] == "PRIVATE_VIEWS_FROZEN_FOR_S2_ATTACK"
    assert restored == ["income", "occupation_group"]
    assert store.calls == [("ATTACK_FIT_NEW", "features", None),
                           ("INNER_SELECTION", "features", None)]
    for role, item in result["roles"].items():
        views, wire = restore_view_bundle(
            private / item["receipt_relative_path"],
            expected_receipt_sha256=item["receipt_sha256"],
            private_root=private,
            expected_campaign_lock_sha256=lock_sha,
            expected_data_manifest_sha256=manifest_sha,
            expected_role=role, expected_arm_id=arm_id, expected_seed=seed,
            expected_role_feature_sha256=sources[f"{role}:features"],
            expected_package_sha256_by_purpose=package_hashes,
            public_releases=releases, attack_view_type=S2AttackView,
            return_release=True,
        )
        assert set(views) == {"income", "occupation_group", "coalition"}
        assert wire["coalition"]["selected_values"].shape[0] == 4
    restored_for_s2 = restore_forward_for_s2(
        arm_id=arm_id, seed=seed,
        fit_handoff_sha256=fit_sha,
        forward_handoff_sha256=result["forward_receipt_sha256"],
        expected_head_spec_sha256="f" * 64,
        expected_manifest_sha256=manifest_sha,
        manifest_path=manifest_path,
        campaign_lock_path=tmp_path / "CAMPAIGN_LOCK.json",
        campaign_lock_sha256=lock_sha,
        local_seal_receipt_path=tmp_path / "LOCAL_SEAL_RECEIPT.json",
        load_calibrator=lambda *a: {},
        apply_global_temperature=lambda x, _: x,
        attack_view_type=S2AttackView,
        private_checkpoint_root=private,
    )
    assert set(restored_for_s2["roles"]) == {"ATTACK_FIT_NEW", "INNER_SELECTION"}
    assert restored_for_s2["roles"]["INNER_SELECTION"]["wire"]["income"][
        "probabilities"
    ].shape == (4, 2)

"""Data-free custody checks for the historical validation-F diagnostic."""

import hashlib
import importlib
import json

import pytest


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_old_f_custody_uses_only_validation_package_components(tmp_path, monkeypatch):
    diagnostic = importlib.import_module("old_f_inner_diagnostic")
    old = tmp_path / "old"
    source_role = {"fit_features": "c" * 64, "fit_label": "d" * 64,
                   "calibration_features": "e" * 64,
                   "calibration_label": "f" * 64}
    entries = {}
    inherited = {}
    for task in ("income", "occupation_group"):
        home = old / "cache" / task
        home.mkdir(parents=True)
        (home / "head.npz").write_bytes(f"synthetic-{task}-head".encode())
        (home / "calibrator.json").write_text("{}")
        (home / "release.npz").write_bytes(b"forbidden-assessment-sentinel")
        prefix = f"cache/{task}"
        package = {"schema": "team-c-private-release-package-v1",
                   "purpose": task, "seed": 0,
                   "head_path": f"{prefix}/head.npz",
                   "head_sha256": _sha(home / "head.npz"),
                   "calibrator_path": f"{prefix}/calibrator.json",
                   "calibrator_sha256": _sha(home / "calibrator.json"),
                   "campaign_lock_sha256": "a" * 64,
                   "data_manifest_sha256": "b" * 64,
                   "source_role_hashes": source_role}
        (home / "release_package.json").write_text(json.dumps(package))
        head_receipt = {"cache_identity": {
            "cache_key": task, "mask_id": "F_FULL83", "purpose": task,
            "seed": 0, "selected_columns": list(range(83)),
            "campaign_lock_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64,
            "role_file_sha256": source_role},
            "fit_rows": 11574, "calibration_rows": 2000}
        (home / "HEAD_POOL_RECEIPT.json").write_text(json.dumps(head_receipt))
        entries[task] = {"stage": "validation", "mask_id": "F_FULL83",
            "relative_package_path": f"{prefix}/release_package.json",
            "package_sha256": _sha(home / "release_package.json"),
            "relative_head_pool_receipt_path": f"{prefix}/HEAD_POOL_RECEIPT.json",
            "head_pool_receipt_sha256": _sha(home / "HEAD_POOL_RECEIPT.json"),
            "relative_head_path": f"{prefix}/head.npz",
            "head_sha256": _sha(home / "head.npz"),
            "relative_calibrator_path": f"{prefix}/calibrator.json",
            "calibrator_sha256": _sha(home / "calibrator.json")}
        e = entries[task]
        inherited[task] = {"F_FULL83": {
            "stage": "validation", "mask_id": "F_FULL83",
            "relative_package_path": e["relative_package_path"],
            "package_sha256": e["package_sha256"],
            "head_pool_receipt_relative_path": e["relative_head_pool_receipt_path"],
            "head_pool_receipt_sha256": e["head_pool_receipt_sha256"],
            "head_sha256": e["head_sha256"],
            "calibrator_sha256": e["calibrator_sha256"]}}
    fit_receipt = {"seed": 0, "status": "NO_FEASIBLE_J",
                   "campaign_lock_sha256": "a" * 64,
                   "data_manifest_sha256": "b" * 64,
                   "validation_head_packages": inherited}
    fit_path = old / "m2/fit/seed_0/FIT_RECEIPT.json"
    fit_path.parent.mkdir(parents=True)
    fit_path.write_text(json.dumps(fit_receipt))
    inputs = {"seed": {"0": {
        "fit_receipt_relative_path": "m2/fit/seed_0/FIT_RECEIPT.json",
        "fit_receipt_sha256": _sha(fit_path),
        "tasks": entries}}}
    real_sha = diagnostic.sha_file
    touched = []

    def checked_sha(path):
        touched.append(path.name)
        if path.name == "release.npz":
            raise AssertionError("old assessment release was read")
        return real_sha(path)

    monkeypatch.setattr(diagnostic, "sha_file", checked_sha)
    restored = diagnostic.validate_old_f_inputs(old, inputs, 0)
    assert set(restored) == {"income", "occupation_group"}
    assert "release.npz" not in touched
    assert set(touched) == {"FIT_RECEIPT.json", "release_package.json",
                            "HEAD_POOL_RECEIPT.json", "head.npz",
                            "calibrator.json"}


def test_old_f_custody_rejects_nonvalidation_entry(tmp_path):
    diagnostic = importlib.import_module("old_f_inner_diagnostic")
    with pytest.raises(PermissionError, match="validation"):
        diagnostic.validate_old_f_inputs(tmp_path, {"seed": {"0": {
            "fit_receipt_relative_path": "anything",
            "fit_receipt_sha256": "0" * 64,
            "tasks": {task: {"stage": "final", "mask_id": "F_FULL83"}
                      for task in ("income", "occupation_group")}}}}, 0)

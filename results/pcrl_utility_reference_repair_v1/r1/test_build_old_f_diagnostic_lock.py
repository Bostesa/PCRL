"""Data-free custody and budget checks for the optional old-F diagnostic."""

import hashlib
import json

import pytest

from results.pcrl_utility_reference_repair_v1.r1 import build_old_f_diagnostic_lock as builder


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_queue_carries_failed_unit_cost_without_claiming_success():
    state = {"units": {"SOME_PRIVACY_UNIT": {"status": "FAILED",
                                             "charged_cpu_s": 3.0}},
             "carry_forward": {"charged_cpu_s": 100.0},
             "used_cpu_s": 103.0}
    assert builder.validate_prior_queue(state, "a" * 64, 20) == {
        "archived_queue_relative": "queue_g20",
        "archived_queue_sha256": "a" * 64,
        "charged_cpu_s": 103.0, "unit_count": 1}
    state["units"]["SOME_PRIVACY_UNIT"]["status"] = "RUNNING"
    with pytest.raises(PermissionError):
        builder.validate_prior_queue(state, "a" * 64, 20)
    state["units"]["SOME_PRIVACY_UNIT"]["status"] = "UNKNOWN"
    with pytest.raises(PermissionError):
        builder.validate_prior_queue(state, "a" * 64, 20)


def test_old_component_inventory_does_not_touch_release_archive(tmp_path, monkeypatch):
    old = tmp_path / "old"
    entries, inherited = {}, {}
    for task in ("income", "occupation_group"):
        folder = old / "cache" / task
        folder.mkdir(parents=True)
        (folder / "release.npz").write_bytes(b"forbidden")
        for name in ("release_package.json", "HEAD_POOL_RECEIPT.json",
                     "head.npz", "calibrator.json"):
            (folder / name).write_bytes((task + name).encode())
        prefix = f"cache/{task}"
        entries[task] = {"stage": "validation", "mask_id": "F_FULL83"}
        for key, name in (("package", "release_package.json"),
                          ("head_pool_receipt", "HEAD_POOL_RECEIPT.json"),
                          ("head", "head.npz"),
                          ("calibrator", "calibrator.json")):
            entries[task][f"relative_{key}_path"] = f"{prefix}/{name}"
            entries[task][f"{key}_sha256"] = _sha(folder / name)
        inherited[task] = {"F_FULL83": {
            "stage": "validation", "mask_id": "F_FULL83",
            "relative_package_path": entries[task]["relative_package_path"],
            "package_sha256": entries[task]["package_sha256"],
            "head_pool_receipt_relative_path": entries[task]["relative_head_pool_receipt_path"],
            "head_pool_receipt_sha256": entries[task]["head_pool_receipt_sha256"],
            "head_sha256": entries[task]["head_sha256"],
            "calibrator_sha256": entries[task]["calibrator_sha256"]}}
    fit = old / "m2/fit/seed_0/FIT_RECEIPT.json"
    fit.parent.mkdir(parents=True)
    fit.write_text(json.dumps({"seed": 0, "status": "NO_FEASIBLE_J",
                               "validation_head_packages": inherited}))
    catalog = {"seed": {"0": {"fit_receipt_relative_path":
                               "m2/fit/seed_0/FIT_RECEIPT.json",
                               "fit_receipt_sha256": _sha(fit),
                               "tasks": entries}}}
    original = builder.sha

    def denied(path):
        if path.name == "release.npz":
            raise AssertionError("old assessment archive must stay closed")
        return original(path)

    monkeypatch.setattr(builder, "sha", denied)
    inventory = {}
    builder.collect_old_components(old, catalog, 0, inventory)
    assert len(inventory) == 9
    assert all(not name.endswith("release.npz") for name in inventory)


def test_role_binding_does_not_open_preseal_archive(tmp_path, monkeypatch):
    study = tmp_path / "study"
    archive = study / ".private/role_export_v1/INNER_SELECTION/references_s0.npz"
    archive.parent.mkdir(parents=True)
    archive.touch()
    manifest = {"role_file_relative_path": {
        "INNER_SELECTION:references_s0":
        ".private/role_export_v1/INNER_SELECTION/references_s0.npz"},
        "role_file_sha256": {"INNER_SELECTION:references_s0": "f" * 64}}
    original = type(archive).open

    def denied(path, *args, **kwargs):
        if path == archive:
            raise AssertionError("role archive opened before diagnostic seal")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(archive), "open", denied)
    inventory = {}
    assert builder.bind_inner_role(study, manifest, "references_s0", inventory) == {
        "path": str(archive), "sha256": "f" * 64}

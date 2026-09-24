"""Synthetic, private-only merge fixtures; no ACS rows or refits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import audit, evaluate, fit_controls
from experiments.pcrl_adaptive_release_v1 import merge_inner_panels


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n")


def _fixture(tmp_path):
    private = tmp_path / "private"
    private.mkdir(parents=True)
    index = private / "REUSABLE_INPUTS_PINNED.json"
    _json(index, {"synthetic": True})
    fit = private / "a0_privacy_first_d001"
    (fit / "channel").mkdir(parents=True)
    q = np.full((32, 17), 1 / 17)
    np.savez_compressed(fit / "channel" / "Q.npz", Q=q)
    main_source = private / "main_Q.npz"
    main_q = np.zeros((32, 17))
    main_q[:, 0] = 1.
    np.savez_compressed(main_source, Q=main_q)
    _json(fit / "RELEASE_SPEC.json", {
        "schema": "pcrl-adaptive-privacy-first-T0-spec-v1", "router": "T0",
        "channel_relative_path": "channel/Q.npz",
        "channel_file_sha256": _sha(fit / "channel" / "Q.npz"),
        "channel_array_sha256": fit_controls._array_sha(q)})
    _json(fit / "COMPLETE.json", {
        "schema": "pcrl-adaptive-privacy-first-fit-v1", "status": "COMPLETE",
        "inputs": {"anchor": 0, "branch": "A", "delta": .001},
        "artifact_sha256": {str(p.relative_to(fit)): _sha(p)
                            for p in fit.rglob("*") if p.is_file()}})
    source_code = {name: "a" * 64 for name in merge_inner_panels.CORE_SOURCE_FILES}

    def panel(name: str, release_id: str, source_path: Path):
        root = private / name
        for h_role in evaluate.H_ROLES:
            slug = h_role.replace(":", "_").replace("/", "_")
            (root / "H" / slug).mkdir(parents=True)
            _json(root / "H" / slug / "own_registry.json",
                  {"model_sha256": "a" * 64, "selected": "H"})
        (root / release_id / "utility_A_same_residence").mkdir(parents=True)
        (root / release_id / "utility_A_same_residence" / "model.bin").write_bytes(b"same fitted model")
        prefix = hashlib.sha256(release_id.encode()).hexdigest()[:12]
        contributions = {}
        role_results = {}
        for role in audit.ROLES:
            score = {"ids": np.array(["synthetic-person-1", "synthetic-person-2"]),
                     "households": np.array(["hh1", "hh2"]),
                     "weights": np.array([1., 2.]),
                     "loss": np.array([.4, .6])}
            h_score = {**score, "loss": np.array([.7, .8])}
            contributions.update(evaluate._private_contributions(prefix, role, score, h_score))
            role_results[role] = {
                "H": {"U": .75, "PWGTP": .766666666666},
                "H_selected_candidate": "H:fixed",
                "H_validation_scores": {"H:fixed": {"U": .75}},
                "H_selected_route": {"model_sha256": "a" * 64},
                "candidate": {"U": .5, "PWGTP": .533333333333},
            }
        np.savez_compressed(root / "PANEL_CONTRIBUTIONS.npz", **contributions)
        with np.load(source_path, allow_pickle=False) as channel:
            source_q = channel["Q"].copy()
        report = {
            "schema": "pcrl-adaptive-inner-audit-v1", "anchor": 0,
            "not_confirmation": True, "outer_pool_opened": False,
            "fit_role": "audit_fit", "selection_role": "inner_selection",
            "score_role": "inner_check", "slate": "standard",
            "probability_floor": audit.FLOOR, "index_sha256": _sha(index),
            "source_code_sha256": source_code,
            "contributions_relative_path": "PANEL_CONTRIBUTIONS.npz",
            "contributions_sha256": _sha(root / "PANEL_CONTRIBUTIONS.npz"),
            "releases": {release_id: {
                "roles": role_results, "private_contribution_prefix": prefix,
                "source": {"channel_artifact": {
                    "relative_to_index_directory": str(source_path.relative_to(private)),
                    "sha256": _sha(source_path)},
                    "channel_array_sha256": evaluate._array_sha(source_q),
                    "router_kind": "T0"}}},
        }
        _json(root / "INNER_AUDIT.json", report)
        _json(root / "COMPLETE.json", {
            "schema": "pcrl-adaptive-inner-audit-complete-v1", "anchor": 0,
            "release_ids": [release_id], "index_sha256": _sha(index),
            "artifacts": evaluate._inventory(root)})
        return root

    main = panel("a0_inner_panel_d001", "A_selected", main_source)
    privacy = panel("a0_privacy_first_inner_d001", "PrivacyFirst_selected",
                    fit / "channel" / "Q.npz")
    main_spec = private / "a0_INNER_SPECS.json"
    _json(main_spec, {
        "schema": "pcrl-inner-audit-spec-index-v1", "anchor": 0,
        "delta": .001, "slate": "standard", "input_index_sha256": _sha(index),
        "canonical_ids": ["A_selected"], "aliases": {"A_selected": "A_selected"},
        "source_receipts": {"A_complete_sha256": "b" * 64},
        "source_module_sha256": "c" * 64, "canonical_release_count": 1,
        "declared_name_count": 1,
    })
    _json(main_spec.with_suffix(".binding.json"), {
        "schema": "pcrl-inner-panel-binding-v1", "anchor": 0,
        "slate": "standard", "canonical_ids": ["A_selected"],
        "source_receipts": {"A_complete_sha256": "b" * 64},
        "spec_index_sha256": _sha(main_spec),
        "inner_complete_sha256": _sha(main / "COMPLETE.json"),
        "inner_audit_sha256": _sha(main / "INNER_AUDIT.json")})
    privacy_spec = private / "a0_privacy_first_specs.json"
    _json(privacy_spec, {
        "schema": "pcrl-privacy-first-inner-spec-index-v1", "anchor": 0,
        "delta": .001, "slate": "standard",
        "canonical_ids": ["PrivacyFirst_selected"],
        "input_index_sha256": _sha(index),
        "privacy_fit_complete_sha256": _sha(fit / "COMPLETE.json"),
        "privacy_fit_release_spec_sha256": _sha(fit / "RELEASE_SPEC.json"),
        "channel_artifact_sha256": _sha(fit / "channel" / "Q.npz"),
        "main_inner_complete_sha256": _sha(main / "COMPLETE.json"),
        "main_inner_audit_sha256": _sha(main / "INNER_AUDIT.json"),
        "H_slates": {},
    })
    _json(privacy_spec.with_suffix(".binding.json"), {
        "schema": "pcrl-privacy-first-inner-binding-v1", "anchor": 0,
        "canonical_ids": ["PrivacyFirst_selected"],
        "spec_index_sha256": _sha(privacy_spec),
        "main_inner_complete_sha256": _sha(main / "COMPLETE.json"),
        "privacy_fit_complete_sha256": _sha(fit / "COMPLETE.json"),
        "inner_complete_sha256": _sha(privacy / "COMPLETE.json"),
        "inner_audit_sha256": _sha(privacy / "INNER_AUDIT.json"),
    })
    output = private / "a0_inner_panel_plus_privacy"
    merged_spec = private / "a0_INNER_PLUS_PRIVACY_SPECS.json"
    return index, fit, main, privacy, main_spec, privacy_spec, output, merged_spec


def _run(paths):
    index, fit, main, privacy, main_spec, privacy_spec, output, merged_spec = paths
    return merge_inner_panels.merge_anchor(
        0, index, main, privacy, fit, main_spec, privacy_spec, output, merged_spec)


def test_merge_preserves_inputs_and_replays_complete_receipt(tmp_path):
    paths = _fixture(tmp_path)
    _, _, main, privacy, _, _, output, merged_spec = paths
    input_bytes = {p: p.read_bytes() for root in (main, privacy)
                   for p in root.rglob("*") if p.is_file()}
    input_mtimes = {p: p.stat().st_mtime_ns for p in input_bytes}
    report = _run(paths)
    assert sorted(report["releases"]) == ["A_selected", "PrivacyFirst_selected"]
    assert all(p.read_bytes() == value and p.stat().st_mtime_ns == input_mtimes[p]
               for p, value in input_bytes.items())
    with np.load(output / "PANEL_CONTRIBUTIONS.npz", allow_pickle=False) as archive:
        prefixes = {item["private_contribution_prefix"] for item in report["releases"].values()}
        assert all(any(key.startswith(prefix + "__") for key in archive.files)
                   for prefix in prefixes)
    assert merged_spec.is_file()
    assert merged_spec.with_suffix(".binding.json").is_file()
    assert _run(paths) == report
    assert (output / "COMPLETE.json").is_file()
    result = merge_inner_panels.main([
        "--anchor", "0", "--index", str(paths[0]),
        "--main-panel", str(paths[2]), "--privacy-panel", str(paths[3]),
        "--privacy-fit-dir", str(paths[1]), "--main-spec", str(paths[4]),
        "--privacy-spec", str(paths[5]), "--output-dir", str(paths[6]),
        "--merged-spec", str(paths[7])])
    assert result["release_ids"] == ["A_selected", "PrivacyFirst_selected"]


def test_merge_rejects_h_route_drift_and_source_tamper(tmp_path):
    paths = _fixture(tmp_path)
    privacy = paths[3]
    report_path = privacy / "INNER_AUDIT.json"
    report = json.loads(report_path.read_text())
    report["releases"]["PrivacyFirst_selected"]["roles"][audit.ROLES[0]]["H_selected_candidate"] = "different"
    _json(report_path, report)
    receipt_path = privacy / "COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"] = evaluate._inventory(privacy)
    _json(receipt_path, receipt)
    binding_path = paths[5].with_suffix(".binding.json")
    binding = json.loads(binding_path.read_text())
    binding["inner_complete_sha256"] = _sha(receipt_path)
    binding["inner_audit_sha256"] = _sha(report_path)
    _json(binding_path, binding)
    with pytest.raises(ValueError, match="H"):
        _run(paths)
    report["releases"]["PrivacyFirst_selected"]["roles"][audit.ROLES[0]]["H_selected_candidate"] = "H:fixed"
    _json(report_path, report)
    receipt["artifacts"] = evaluate._inventory(privacy)
    _json(receipt_path, receipt)
    binding["inner_complete_sha256"] = _sha(receipt_path)
    binding["inner_audit_sha256"] = _sha(report_path)
    _json(binding_path, binding)
    (paths[1] / "channel" / "Q.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source|SHA|fit"):
        _run(paths)


def test_merge_rejects_partial_output_and_release_collision(tmp_path):
    paths = _fixture(tmp_path)
    output = paths[6]
    output.mkdir()
    (output / "partial").write_text("incomplete")
    with pytest.raises(FileExistsError, match="partial"):
        _run(paths)
    (output / "partial").unlink()
    output.rmdir()
    privacy = paths[3]
    report_path = privacy / "INNER_AUDIT.json"
    report = json.loads(report_path.read_text())
    report["releases"]["A_selected"] = report["releases"].pop("PrivacyFirst_selected")
    _json(report_path, report)
    receipt_path = privacy / "COMPLETE.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["release_ids"] = ["A_selected"]
    receipt["artifacts"] = evaluate._inventory(privacy)
    _json(receipt_path, receipt)
    with pytest.raises(ValueError, match="PrivacyFirst|collision|release"):
        _run(paths)


def test_merge_resumes_only_byte_verified_partial_copy(tmp_path):
    paths = _fixture(tmp_path)
    main, output = paths[2], paths[6]
    member = Path("H") / "utility_A_same_residence" / "own_registry.json"
    target = output / member
    target.parent.mkdir(parents=True)
    shutil.copy2(main / member, target)
    original_mtime = target.stat().st_mtime_ns
    _run(paths)
    assert target.stat().st_mtime_ns == original_mtime
    assert target.read_bytes() == (main / member).read_bytes()

    other = _fixture(tmp_path / "second")
    bad = other[6] / member
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"changed")
    with pytest.raises(ValueError, match="partial merged artifact"):
        _run(other)

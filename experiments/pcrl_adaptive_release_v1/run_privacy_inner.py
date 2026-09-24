"""Audit a frozen PrivacyFirst T0 channel with verified, reused H-only slates.

The main common inner panel is immutable. Its complete H-only predictor trees
are copied byte-for-byte into a separate private panel before the ordinary
standard-slate inner audit runs. This runner neither opens outer outcomes nor
refits valid H predictors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit
from . import evaluate, privacy_first_fit, run_inner


RELEASE_ID = "PrivacyFirst_selected"


def _sha(path: str | Path) -> str:
    return run_inner._sha(path)


def _private(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("private audit path cannot be a symlink")
    result = raw.resolve()
    if "private" not in result.parts:
        raise ValueError("inner audit evidence must remain private")
    return result


def _h_directory(root: Path, role: str) -> Path:
    return root / "H" / role.replace(":", "_").replace("/", "_")


def _verify_h_slate(root: Path, role: str, anchor: int) -> dict:
    registry_path = root / "own_registry.json"
    if not registry_path.is_file():
        raise ValueError(f"partial H slate: {role}")
    if root.is_symlink():
        raise ValueError("H slate is a symlink")
    registry = json.loads(registry_path.read_text())
    expected = {"schema": 1, "role": role, "release_id": "H",
                "seed": 26000 + 1000*anchor + evaluate.H_ROLES.index(role),
                "slate": "standard"}
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ValueError(f"H slate {role} {key} differs")
    models = registry.get("models")
    if not isinstance(models, dict) or not models or registry.get("own_selection") not in models:
        raise ValueError(f"partial H slate models: {role}")
    model_root = (root / "models").resolve()
    for model in models.values():
        relative = model.get("model_relative_directory")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("H model has unsafe relative directory")
        location = (root / relative).resolve()
        if not location.is_relative_to(model_root):
            raise ValueError("H model escapes slate directory")
        if inherited_audit.model_directory_hash(location) != model.get("model_sha256"):
            raise ValueError("H model hash differs")
    artifacts = inherited_audit._artifact_inventory(root)
    if artifacts != registry.get("artifacts_sha256"):
        raise ValueError(f"H slate artifact inventory differs: {role}")
    return {"registry_sha256": _sha(registry_path),
            "artifact_sha256": artifacts}


def _verify_main(root: Path, index: Path, anchor: int) -> dict:
    complete_path = root / "COMPLETE.json"
    report_path = root / "INNER_AUDIT.json"
    if not complete_path.is_file() or not report_path.is_file():
        raise ValueError("partial main common inner panel")
    complete = json.loads(complete_path.read_text())
    report = json.loads(report_path.read_text())
    index_sha = _sha(index)
    if (complete.get("schema") != "pcrl-adaptive-inner-audit-complete-v1" or
            complete.get("anchor") != anchor or
            complete.get("index_sha256") != index_sha or
            complete.get("artifacts") != evaluate._inventory(root) or
            sorted(complete.get("release_ids", [])) != sorted(report.get("releases", {})) or
            report.get("schema") != "pcrl-adaptive-inner-audit-v1" or
            report.get("anchor") != anchor or report.get("slate") != "standard" or
            report.get("index_sha256") != index_sha or
            report.get("outer_pool_opened") is not False or
            report.get("contributions_sha256") != _sha(root / "PANEL_CONTRIBUTIONS.npz")):
        raise ValueError("main common inner panel receipt or artifact inventory differs")
    h_hashes = {}
    for role in evaluate.H_ROLES:
        h_hashes[role] = _verify_h_slate(_h_directory(root, role), role, anchor)
    h_subdirs = {p.name for p in (root / "H").iterdir()}
    expected = {_h_directory(root, role).name for role in evaluate.H_ROLES}
    if h_subdirs != expected:
        raise ValueError("main H slate set differs from declared roles")
    return {"main_inner_complete_sha256": _sha(complete_path),
            "main_inner_audit_sha256": _sha(report_path),
            "H_slates": h_hashes}


def _verify_copied_h(source: Path, destination: Path, anchor: int) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("partial copied H slates")
    if {p.name for p in destination.iterdir()} != {
            _h_directory(source, role).name for role in evaluate.H_ROLES}:
        raise ValueError("partial or extra copied H slates")
    for role in evaluate.H_ROLES:
        left = _h_directory(source, role)
        right = _h_directory(destination.parent, role)
        if (_verify_h_slate(left, role, anchor) !=
                _verify_h_slate(right, role, anchor)):
            raise ValueError(f"copied H slate differs: {role}")


def _copy_h(source: Path, output: Path, anchor: int) -> None:
    target = output / "H"
    if target.exists():
        _verify_copied_h(source, target, anchor)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("partial inner panel has no complete copied H slates")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    shutil.copytree(source / "H", target, symlinks=False)
    evaluate._seal_private_permissions(target)
    _verify_copied_h(source, target, anchor)


def _verify_complete(output: Path, index: Path, anchor: int, spec: dict) -> dict:
    receipt_path = output / "COMPLETE.json"
    report_path = output / "INNER_AUDIT.json"
    if not receipt_path.is_file() or not report_path.is_file():
        raise ValueError("partial PrivacyFirst inner report")
    receipt = json.loads(receipt_path.read_text())
    report = json.loads(report_path.read_text())
    index_sha = _sha(index)
    if (receipt.get("schema") != "pcrl-adaptive-inner-audit-complete-v1" or
            receipt.get("anchor") != anchor or
            receipt.get("release_ids") != [RELEASE_ID] or
            receipt.get("index_sha256") != index_sha or
            receipt.get("artifacts") != evaluate._inventory(output) or
            report.get("schema") != "pcrl-adaptive-inner-audit-v1" or
            report.get("anchor") != anchor or report.get("slate") != "standard" or
            report.get("index_sha256") != index_sha or
            report.get("outer_pool_opened") is not False or
            sorted(report.get("releases", {})) != [RELEASE_ID] or
            report["releases"][RELEASE_ID].get("source", {}).get(
                "channel_artifact", {}).get("sha256") != spec["channel_artifact_sha256"] or
            report.get("contributions_sha256") != _sha(output / "PANEL_CONTRIBUTIONS.npz")):
        raise ValueError("completed PrivacyFirst panel identity or inventory differs")
    return report


def dispatch_privacy_inner(anchor: int, index_path: str | Path,
                           privacy_fit_dir: str | Path, main_inner_dir: str | Path,
                           output_dir: str | Path, spec_index: str | Path) -> dict:
    """Verify frozen evidence, reuse H slates, and run one independent inner audit."""
    if anchor not in (0, 1, 2):
        raise ValueError("registered anchor required")
    index = Path(index_path).resolve()
    source = _private(main_inner_dir)
    fit = _private(privacy_fit_dir)
    output = _private(output_dir)
    spec_path = _private(spec_index)
    if (output.is_relative_to(source) or source.is_relative_to(output) or
            output.is_relative_to(fit) or fit.is_relative_to(output) or
            any(spec_path.is_relative_to(root) for root in (source, fit, output))):
        raise ValueError("PrivacyFirst audit output and spec index must be separate from source evidence")
    source_evidence = _verify_main(source, index, anchor)
    spec = privacy_first_fit.load_release_spec(fit)
    fit_complete = json.loads((fit / "COMPLETE.json").read_text())
    fit_inputs = fit_complete.get("inputs", {})
    if (fit_complete.get("status") != "COMPLETE" or
            fit_inputs.get("anchor") != anchor or fit_inputs.get("branch") != "A" or
            fit_inputs.get("delta") != .001):
        raise ValueError("PrivacyFirst fit does not match registered anchor/A center")
    manifest = {"schema": "pcrl-privacy-first-inner-spec-index-v1",
                "anchor": anchor, "delta": .001, "slate": "standard",
                "canonical_ids": [RELEASE_ID],
                "input_index_sha256": _sha(index),
                "privacy_fit_complete_sha256": _sha(fit / "COMPLETE.json"),
                "privacy_fit_release_spec_sha256": _sha(fit / "RELEASE_SPEC.json"),
                "channel_artifact_sha256": spec["channel_artifact_sha256"],
                "source_module_sha256": _sha(__file__),
                "evaluate_module_sha256": _sha(evaluate.__file__),
                **source_evidence}
    spec_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_inner._write_or_verify(spec_path, manifest)
    if (output / "COMPLETE.json").is_file():
        _verify_copied_h(source, output / "H", anchor)
        report = _verify_complete(output, index, anchor, spec)
    else:
        _copy_h(source, output, anchor)
        report = evaluate.audit_panel(
            anchor, {RELEASE_ID: spec}, index, output, resume=True,
            slate="standard")
        _verify_copied_h(source, output / "H", anchor)
        report = _verify_complete(output, index, anchor, spec)
    binding = {"schema": "pcrl-privacy-first-inner-binding-v1",
               "anchor": anchor, "canonical_ids": [RELEASE_ID],
               "spec_index_sha256": _sha(spec_path),
               "main_inner_complete_sha256": source_evidence["main_inner_complete_sha256"],
               "privacy_fit_complete_sha256": manifest["privacy_fit_complete_sha256"],
               "inner_complete_sha256": _sha(output / "COMPLETE.json"),
               "inner_audit_sha256": _sha(output / "INNER_AUDIT.json")}
    run_inner._write_or_verify(spec_path.with_suffix(".binding.json"), binding)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--privacy-fit-dir", required=True)
    parser.add_argument("--main-inner-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec-index", required=True)
    args = parser.parse_args(argv)
    report = dispatch_privacy_inner(
        args.anchor, args.index, args.privacy_fit_dir, args.main_inner_dir,
        args.output_dir, args.spec_index)
    print(json.dumps({"anchor": args.anchor,
                      "canonical_releases": sorted(report["releases"]),
                      "outer_pool_opened": report["outer_pool_opened"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()

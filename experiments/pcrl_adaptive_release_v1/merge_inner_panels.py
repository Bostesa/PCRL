"""Derive a combined inner panel from two completed, immutable audits.

No row loader, model fitter, route selector, or outer-access function is used.
Inputs are verified by their COMPLETE inventories and source bindings, then
copied into an independent private panel. Partial copies can be resumed only
when every existing byte matches an already verified source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np

from . import audit, evaluate, privacy_first_fit


PRIVACY_ID = "PrivacyFirst_selected"
NPZ_NAME = "PANEL_CONTRIBUTIONS.npz"
REPORT_NAME = "INNER_AUDIT.json"
COMPLETE_NAME = "COMPLETE.json"
FIELDS = ("ids", "households", "weights", "candidate_loss", "H_loss",
          "household", "household_count", "household_weight",
          "household_candidate_U_num", "household_H_U_num",
          "household_candidate_W_num", "household_H_W_num")
H_FIELDS = tuple(key for key in FIELDS if key != "candidate_loss"
                 and not key.startswith("household_candidate_"))
CORE_SOURCE_FILES = ("evaluate.py", "audit.py", "roles.py",
                     "inherited_audit.py", "data.py")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("verified receipt must be a JSON object")
    return value


def _private(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("private panel path cannot be a symlink")
    result = raw.resolve()
    if "private" not in result.parts:
        raise ValueError("audit panel and derived evidence must remain private")
    return result


def _hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        letter in "0123456789abcdef" for letter in value)


def _verify_source_artifacts(source: dict, index_dir: Path) -> None:
    seen = 0
    for value in source.values():
        if not isinstance(value, dict) or "relative_to_index_directory" not in value:
            continue
        relative = value["relative_to_index_directory"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or not _hex(value.get("sha256")):
            raise ValueError("release source artifact path or SHA is malformed")
        path = (index_dir / relative).resolve()
        if ("private" not in path.parts or path.is_symlink() or not path.is_file()
                or _sha(path) != value["sha256"]):
            raise ValueError("release source artifact SHA differs")
        seen += 1
    if seen == 0:
        raise ValueError("release has no verifiable source artifact")


def _prefix(release_id: str, release: dict) -> str:
    prefix = hashlib.sha256(release_id.encode()).hexdigest()[:12]
    if release.get("private_contribution_prefix") != prefix:
        raise ValueError("private contribution prefix differs from release identity")
    return prefix


def _role_key(prefix: str, role: str, field: str) -> str:
    return f"{prefix}__{role.replace(':', '_').replace('/', '_')}_{field}"


def _verify_contributions(root: Path, report: dict) -> None:
    path = root / NPZ_NAME
    if (report.get("contributions_relative_path") != NPZ_NAME
            or not path.is_file() or _sha(path) != report.get("contributions_sha256")):
        raise ValueError("private panel contribution SHA differs")
    expected = {_role_key(_prefix(name, release), role, field)
                for name, release in report["releases"].items()
                for role in audit.ROLES for field in FIELDS}
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != expected:
            raise ValueError("private contribution keys collide or differ from release roster")
        for key in expected:
            if archive[key].dtype.hasobject:
                raise ValueError("private contribution archive contains object arrays")


def _verify_panel(root: Path, index: Path, anchor: int, *, privacy: bool,
                  merged: bool = False) -> tuple[dict, dict]:
    complete_path, report_path = root / COMPLETE_NAME, root / REPORT_NAME
    if not complete_path.is_file() or not report_path.is_file():
        raise ValueError("partial completed inner panel input")
    complete, report = _read_json(complete_path), _read_json(report_path)
    ids = sorted(report.get("releases", {}))
    if (complete.get("schema") != "pcrl-adaptive-inner-audit-complete-v1"
            or complete.get("anchor") != anchor or complete.get("release_ids") != ids
            or complete.get("index_sha256") != _sha(index)
            or complete.get("artifacts") != evaluate._inventory(root)
            or report.get("schema") != "pcrl-adaptive-inner-audit-v1"
            or report.get("anchor") != anchor or report.get("index_sha256") != _sha(index)
            or report.get("slate") != "standard"
            or report.get("outer_pool_opened") is not False
            or report.get("fit_role") != "audit_fit"
            or report.get("selection_role") != "inner_selection"
            or report.get("score_role") != "inner_check"
            or not ids or (privacy and ids != [PRIVACY_ID])
            or (not privacy and not merged and PRIVACY_ID in ids)
            or (merged and PRIVACY_ID not in ids)):
        raise ValueError("completed inner panel receipt, index, slate or release identity differs")
    code = report.get("source_code_sha256")
    if (not isinstance(code, dict) or any(not _hex(code.get(name))
                                          for name in CORE_SOURCE_FILES)):
        raise ValueError("inner panel source code hashes are incomplete")
    for name, item in report["releases"].items():
        if (set(item.get("roles", {})) != set(audit.ROLES)
                or not isinstance(item.get("source"), dict)):
            raise ValueError("inner panel role or release source roster differs")
        _verify_source_artifacts(item["source"], index.parent)
        _prefix(name, item)
    expected_top = {"H", REPORT_NAME, COMPLETE_NAME, NPZ_NAME, *ids}
    if {item.name for item in root.iterdir()} != expected_top:
        raise ValueError("inner panel has partial or unexpected top-level artifacts")
    h_roles = {role.replace(":", "_").replace("/", "_") for role in evaluate.H_ROLES}
    if {item.name for item in (root / "H").iterdir()} != h_roles:
        raise ValueError("inner panel H role subtrees differ")
    _verify_contributions(root, report)
    return complete, report


def _verify_main_spec(path: Path, main: Path, index: Path, anchor: int,
                      complete: dict, report: dict) -> dict:
    spec = _read_json(path)
    binding = _read_json(path.with_suffix(".binding.json"))
    ids = sorted(report["releases"])
    aliases = spec.get("aliases")
    sources = spec.get("source_receipts")
    if (spec.get("schema") != "pcrl-inner-audit-spec-index-v1"
            or spec.get("anchor") != anchor or spec.get("delta") != .001
            or spec.get("slate") != "standard"
            or spec.get("input_index_sha256") != _sha(index)
            or spec.get("canonical_ids") != ids
            or not isinstance(aliases, dict) or set(aliases.values()) != set(ids)
            or not isinstance(sources, dict) or not sources
            or any(not _hex(value) for value in sources.values())
            or not _hex(spec.get("source_module_sha256"))
            or binding.get("schema") != "pcrl-inner-panel-binding-v1"
            or binding.get("anchor") != anchor or binding.get("slate") != "standard"
            or binding.get("canonical_ids") != ids
            or binding.get("source_receipts") != sources
            or binding.get("spec_index_sha256") != _sha(path)
            or binding.get("inner_complete_sha256") != _sha(main / COMPLETE_NAME)
            or binding.get("inner_audit_sha256") != _sha(main / REPORT_NAME)):
        raise ValueError("main spec/binding source or panel hashes differ")
    return spec


def _verify_privacy_spec(path: Path, privacy: Path, fit: Path, main: Path,
                         index: Path, anchor: int, report: dict) -> dict:
    spec = _read_json(path)
    binding = _read_json(path.with_suffix(".binding.json"))
    fit_spec = privacy_first_fit.load_release_spec(fit)
    fit_complete = _read_json(fit / COMPLETE_NAME)
    inputs = fit_complete.get("inputs", {})
    if (inputs.get("anchor") != anchor or inputs.get("branch") != "A"
            or inputs.get("delta") != .001):
        raise ValueError("PrivacyFirst fitted source anchor or registered allowance differs")
    source = report["releases"][PRIVACY_ID]["source"]
    if (source.get("channel_artifact", {}).get("sha256") != fit_spec["channel_artifact_sha256"]
            or source.get("channel_array_sha256") != evaluate._array_sha(fit_spec["Q"])):
        raise ValueError("PrivacyFirst panel channel differs from completed fitted source")
    if (spec.get("schema") != "pcrl-privacy-first-inner-spec-index-v1"
            or spec.get("anchor") != anchor or spec.get("delta") != .001
            or spec.get("slate") != "standard"
            or spec.get("canonical_ids") != [PRIVACY_ID]
            or spec.get("input_index_sha256") != _sha(index)
            or spec.get("privacy_fit_complete_sha256") != _sha(fit / COMPLETE_NAME)
            or spec.get("privacy_fit_release_spec_sha256") != _sha(fit / "RELEASE_SPEC.json")
            or spec.get("channel_artifact_sha256") != fit_spec["channel_artifact_sha256"]
            or spec.get("main_inner_complete_sha256") != _sha(main / COMPLETE_NAME)
            or spec.get("main_inner_audit_sha256") != _sha(main / REPORT_NAME)
            or binding.get("schema") != "pcrl-privacy-first-inner-binding-v1"
            or binding.get("anchor") != anchor
            or binding.get("canonical_ids") != [PRIVACY_ID]
            or binding.get("spec_index_sha256") != _sha(path)
            or binding.get("main_inner_complete_sha256") != _sha(main / COMPLETE_NAME)
            or binding.get("privacy_fit_complete_sha256") != _sha(fit / COMPLETE_NAME)
            or binding.get("inner_complete_sha256") != _sha(privacy / COMPLETE_NAME)
            or binding.get("inner_audit_sha256") != _sha(privacy / REPORT_NAME)):
        raise ValueError("PrivacyFirst spec/binding source or panel hashes differ")
    return spec


def _verify_shared_h(main: Path, privacy: Path, main_report: dict,
                     privacy_report: dict) -> None:
    left = {name: digest for name, digest in evaluate._inventory(main).items()
            if name.startswith("H/")}
    right = {name: digest for name, digest in evaluate._inventory(privacy).items()
             if name.startswith("H/")}
    if not left or left != right:
        raise ValueError("H slate/model files are not byte-identical across panels")
    baseline = next(iter(main_report["releases"].values()))
    for item in [*main_report["releases"].values(),
                 privacy_report["releases"][PRIVACY_ID]]:
        for role in audit.ROLES:
            for key in ("H", "H_selected_candidate", "H_validation_scores",
                        "H_selected_route"):
                if item["roles"][role].get(key) != baseline["roles"][role].get(key):
                    raise ValueError("H route, model, validation or aggregate score differs")
    paths = [(main, main_report), (privacy, privacy_report)]
    first_root, first_report = paths[0]
    first_name = next(iter(first_report["releases"]))
    first_prefix = _prefix(first_name, first_report["releases"][first_name])
    with np.load(first_root / NPZ_NAME, allow_pickle=False) as baseline_npz:
        for root, report in paths:
            with np.load(root / NPZ_NAME, allow_pickle=False) as archive:
                for name, item in report["releases"].items():
                    prefix = _prefix(name, item)
                    for role in audit.ROLES:
                        for field in H_FIELDS:
                            if not np.array_equal(
                                    archive[_role_key(prefix, role, field)],
                                    baseline_npz[_role_key(first_prefix, role, field)]):
                                raise ValueError("H private score rows or household contributions differ")


def _source_files(main: Path, privacy: Path) -> dict[str, Path]:
    paths = {}
    for name in evaluate._inventory(main):
        if name not in (REPORT_NAME, COMPLETE_NAME, NPZ_NAME):
            paths[name] = main / name
    for name in evaluate._inventory(privacy):
        if name.startswith(PRIVACY_ID + "/"):
            if name in paths:
                raise ValueError("colliding panel model artifact")
            paths[name] = privacy / name
    if not any(name.startswith(PRIVACY_ID + "/") for name in paths):
        raise ValueError("PrivacyFirst fitted model directory missing")
    return paths


def _copy_or_verify(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_symlink() or not target.is_file() or _sha(target) != _sha(source):
            raise ValueError("partial merged artifact differs from verified source")
        return
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    if _sha(temporary) != _sha(source):
        raise ValueError("copied private artifact hash differs")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def _write_or_verify(path: Path, record: dict) -> None:
    payload = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"existing derived {path.name} differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(payload)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _merge_npz(main: Path, privacy: Path, output: Path) -> None:
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    seen = set()
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as merged:
            for source in (main / NPZ_NAME, privacy / NPZ_NAME):
                with zipfile.ZipFile(source) as archive:
                    for name in sorted(archive.namelist()):
                        if name in seen or not name.endswith(".npy") or "/" in name or ".." in name:
                            raise ValueError("colliding or unsafe private contribution key")
                        seen.add(name)
                        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = 0o600 << 16
                        with archive.open(name) as reader, merged.open(info, "w") as writer:
                            shutil.copyfileobj(reader, writer, length=1 << 20)
        if output.exists():
            if _sha(output) != _sha(temporary):
                raise ValueError("partial merged contribution archive differs")
        else:
            os.chmod(temporary, 0o600)
            os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def merge_anchor(anchor: int, index_path: str | Path, main_panel_dir: str | Path,
                 privacy_panel_dir: str | Path, privacy_fit_dir: str | Path,
                 main_spec_path: str | Path, privacy_spec_path: str | Path,
                 output_dir: str | Path, merged_spec_path: str | Path) -> dict:
    """Verify, copy, and seal a derived standard-slate inner panel for one anchor."""
    if anchor not in (0, 1, 2):
        raise ValueError("registered anchor 0, 1, or 2 required")
    index = Path(index_path).resolve()
    main, privacy, fit = map(_private, (main_panel_dir, privacy_panel_dir,
                                       privacy_fit_dir))
    main_spec, privacy_spec = map(_private, (main_spec_path, privacy_spec_path))
    output, merged_spec = map(_private, (output_dir, merged_spec_path))
    roots = (main, privacy, fit, output)
    if (len(set(roots)) != 4 or any(a.is_relative_to(b) for a in roots for b in roots if a != b)
            or merged_spec.is_relative_to(output) or main_spec.is_relative_to(output)
            or privacy_spec.is_relative_to(output)):
        raise ValueError("input, output, fit and spec paths must be disjoint")
    main_complete, main_report = _verify_panel(main, index, anchor, privacy=False)
    privacy_complete, privacy_report = _verify_panel(privacy, index, anchor, privacy=True)
    main_spec_record = _verify_main_spec(main_spec, main, index, anchor,
                                         main_complete, main_report)
    privacy_spec_record = _verify_privacy_spec(
        privacy_spec, privacy, fit, main, index, anchor, privacy_report)
    if any(main_report["source_code_sha256"][name] !=
           privacy_report["source_code_sha256"][name] for name in CORE_SOURCE_FILES):
        raise ValueError("main and PrivacyFirst audit scoring source hashes differ")
    if main_report["probability_floor"] != privacy_report["probability_floor"]:
        raise ValueError("main and PrivacyFirst audit probability floors differ")
    privacy_source = privacy_report["releases"][PRIVACY_ID]["source"]
    if any(item["source"].get("router_kind") == "T0"
           and item["source"].get("channel_array_sha256") ==
           privacy_source.get("channel_array_sha256")
           for item in main_report["releases"].values()):
        raise ValueError("PrivacyFirst release is an exact T0 channel alias of a main release")
    _verify_shared_h(main, privacy, main_report, privacy_report)
    files = _source_files(main, privacy)
    expected_names = set(files) | {NPZ_NAME, REPORT_NAME, COMPLETE_NAME}
    if output.exists():
        if any(path.is_symlink() for path in output.rglob("*")):
            raise ValueError("symlink in partial merged panel")
        existing = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
        unknown = existing - expected_names
        if unknown:
            raise FileExistsError("partial merged output contains unregistered files")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    for name, source in files.items():
        _copy_or_verify(source, output / name)
    _merge_npz(main, privacy, output / NPZ_NAME)
    releases = {**main_report["releases"], **privacy_report["releases"]}
    if len(releases) != len(main_report["releases"]) + 1:
        raise ValueError("PrivacyFirst release ID collides with main panel")
    source_pins = {
        "main_complete_sha256": _sha(main / COMPLETE_NAME),
        "main_audit_sha256": _sha(main / REPORT_NAME),
        "main_spec_sha256": _sha(main_spec),
        "privacy_complete_sha256": _sha(privacy / COMPLETE_NAME),
        "privacy_audit_sha256": _sha(privacy / REPORT_NAME),
        "privacy_spec_sha256": _sha(privacy_spec),
        "privacy_fit_complete_sha256": _sha(fit / COMPLETE_NAME),
        "merge_source_sha256": _sha(__file__),
    }
    merged_report = {**main_report, "releases": releases,
                     "contributions_relative_path": NPZ_NAME,
                     "contributions_sha256": _sha(output / NPZ_NAME),
                     "derived_from": source_pins,
                     "no_refit_or_reselection": True}
    _write_or_verify(output / REPORT_NAME, merged_report)
    # A completed rerun preserves the original completion timestamp exactly.
    receipt_path = output / COMPLETE_NAME
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if (receipt.get("schema") != "pcrl-adaptive-inner-audit-complete-v1"
                or receipt.get("anchor") != anchor
                or receipt.get("release_ids") != sorted(releases)
                or receipt.get("index_sha256") != _sha(index)
                or receipt.get("artifacts") != evaluate._inventory(output)):
            raise ValueError("completed derived panel receipt or SHA inventory differs")
    else:
        receipt = {"schema": "pcrl-adaptive-inner-audit-complete-v1",
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "anchor": anchor, "release_ids": sorted(releases),
                   "index_sha256": _sha(index),
                   "artifacts": evaluate._inventory(output)}
        _write_or_verify(receipt_path, receipt)
    aliases = dict(main_spec_record["aliases"])
    if PRIVACY_ID in aliases:
        raise ValueError("PrivacyFirst declared alias collides with main spec")
    aliases[PRIVACY_ID] = PRIVACY_ID
    sources = dict(main_spec_record["source_receipts"])
    sources["PrivacyFirst_complete_sha256"] = source_pins["privacy_fit_complete_sha256"]
    merged_spec_record = {**main_spec_record,
                          "canonical_ids": sorted(releases),
                          "aliases": aliases,
                          "source_receipts": sources,
                          "canonical_release_count": len(releases),
                          "declared_name_count": len(aliases),
                          "derived_merge": source_pins,
                          "privacy_first_input_spec_sha256": _sha(privacy_spec),
                          "privacy_first_h_slate_count": len(privacy_spec_record.get("H_slates", {}))}
    _write_or_verify(merged_spec, merged_spec_record)
    binding = {"schema": "pcrl-inner-panel-binding-v1", "anchor": anchor,
               "slate": "standard", "canonical_ids": sorted(releases),
               "source_receipts": sources,
               "spec_index_sha256": _sha(merged_spec),
               "inner_complete_sha256": _sha(receipt_path),
               "inner_audit_sha256": _sha(output / REPORT_NAME)}
    _write_or_verify(merged_spec.with_suffix(".binding.json"), binding)
    _verify_panel(output, index, anchor, privacy=False, merged=True)
    return merged_report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--main-panel", required=True)
    parser.add_argument("--privacy-panel", required=True)
    parser.add_argument("--privacy-fit-dir", required=True)
    parser.add_argument("--main-spec", required=True)
    parser.add_argument("--privacy-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merged-spec", required=True)
    args = parser.parse_args(argv)
    report = merge_anchor(args.anchor, args.index, args.main_panel,
                          args.privacy_panel, args.privacy_fit_dir,
                          args.main_spec, args.privacy_spec,
                          args.output_dir, args.merged_spec)
    result = {"anchor": args.anchor,
              "release_ids": sorted(report["releases"]),
              "complete_sha256": _sha(Path(args.output_dir) / COMPLETE_NAME),
              "merged_spec_sha256": _sha(args.merged_spec),
              "no_refit_or_reselection": True}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()

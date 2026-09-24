"""Replay locked three-anchor outer scores with one paired household family.

Only completed, hash-verified private score archives are read. This module has
no outer-row loader, fitter, attacker selector, or release selector.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from . import audit, evaluate, inference


BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260923
ALPHA = .05
ANCHORS = (0, 1, 2)


def _verify_remote_lock(path: str | Path, expected_sha256: str) -> None:
    from . import outer_access
    outer_access.verify_outer_unlock(path, expected_sha256)


def validate_locked_resolution(lock: Mapping) -> dict[int, dict[str, str]]:
    """Resolve every frozen endpoint on each anchor before reading outcomes."""
    if (lock.get("schema") != "pcrl-adaptive-selection-lock-v1" or
            lock.get("status") != "LOCKED" or
            lock.get("assessment_year") != 2018 or
            set(lock.get("anchors", {})) != {"0", "1", "2"}):
        raise ValueError("registered three-anchor 2018 selection lock required")
    expected_manifest = inference.family_manifest(
        lock["slots"], alias_of=lock.get("alias_of", {}))
    if lock.get("family_manifest") != expected_manifest:
        raise ValueError("locked endpoint family differs from registered slots")
    required = {name for item in expected_manifest["endpoints"]
                for name in (item["plus"], item["minus"]) if name != "H"}
    resolved = {}
    for anchor in ANCHORS:
        record = lock["anchors"][str(anchor)]
        mapping = record.get("logical_to_canonical")
        releases = record.get("releases")
        if not isinstance(mapping, Mapping) or not isinstance(releases, Mapping) or not releases:
            raise ValueError("anchor logical release map or scored panel absent")
        if any(name not in mapping or mapping[name] not in releases
               for name in required):
            raise ValueError("endpoint logical release is unmapped or unscored on an anchor")
        resolved[anchor] = {name: mapping[name] for name in sorted(required)}
    return resolved


def _score_record(archive: Mapping, prefix: str, role: str,
                  source: str) -> dict[str, np.ndarray]:
    stem = f"{prefix}__{role.replace(':', '_').replace('/', '_')}"
    suffix = "candidate" if source == "candidate" else "H"
    required = {"ids": f"{stem}_ids", "households": f"{stem}_households",
                "weights": f"{stem}_weights", "loss": f"{stem}_{suffix}_loss"}
    if any(key not in archive for key in required.values()):
        raise ValueError("private contribution archive lacks a locked role record")
    result = {key: np.asarray(archive[value]).copy() for key, value in required.items()}
    if (result["loss"].ndim != 1 or len(result["loss"]) == 0 or
            any(np.asarray(value).shape != result["loss"].shape
                for value in result.values()) or
            not np.isfinite(result["loss"]).all() or
            not np.isfinite(result["weights"]).all()):
        raise ValueError("private contribution arrays misalign original people")
    return result


def _same_score(a: Mapping, b: Mapping) -> bool:
    return all(np.array_equal(a[key], b[key]) for key in
               ("ids", "households", "weights", "loss"))


def _load_completed_outer(root: str | Path, anchor: int, lock_sha256: str,
                          locked_releases: Mapping) -> tuple[dict, dict, dict]:
    """Read only an immutable completed outer score and its private loss rows."""
    from . import outer_audit
    directory = Path(root).resolve()
    if "private" not in directory.parts:
        raise ValueError("outer contribution directory must remain private")
    complete_path = directory / "COMPLETE.json"
    report_path = directory / "OUTER_AUDIT.json"
    complete = json.loads(complete_path.read_text())
    report = json.loads(report_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-outer-audit-complete-v1" or
            complete.get("anchor") != anchor or
            complete.get("selection_lock_sha256") != lock_sha256 or
            complete.get("artifacts") != evaluate._inventory(directory) or
            set(complete.get("release_ids", [])) != set(locked_releases) or
            report.get("schema") != "pcrl-adaptive-outer-audit-v1" or
            report.get("anchor") != anchor or
            report.get("selection_lock_sha256") != lock_sha256 or
            report.get("assessment_year") != 2018 or
            report.get("development_only") is not True or
            report.get("no_outer_fit_or_selection") is not True or
            set(report.get("releases", {})) != set(locked_releases)):
        raise ValueError("outer artifact inventory, schema or locked release panel differs")
    member = Path(report.get("contributions_relative_path", ""))
    contributions_path = (directory / member).resolve()
    if (member.is_absolute() or ".." in member.parts or
            not contributions_path.is_relative_to(directory) or
            not contributions_path.is_file() or
            evaluate._sha(contributions_path) != report.get("contributions_sha256")):
        raise ValueError("outer contribution archive SHA or path differs")
    scores = {}
    h_by_role = {}
    with np.load(contributions_path, allow_pickle=False) as archive:
        for release_id in sorted(locked_releases):
            payload = report["releases"][release_id]
            if (payload.get("source") != locked_releases[release_id] or
                    set(payload.get("roles", {})) != set(audit.ROLES)):
                raise ValueError("outer release source descriptor or role family differs")
            prefix = payload.get("private_contribution_prefix")
            if not isinstance(prefix, str) or not prefix:
                raise ValueError("outer release has no private contribution prefix")
            for role in audit.ROLES:
                candidate = _score_record(archive, prefix, role, "candidate")
                h = _score_record(archive, prefix, role, "H")
                declared = payload["roles"][role]
                for key, item in (("candidate", candidate), ("H", h)):
                    replay = audit.score_weightings(item["loss"], item["weights"])
                    if any(abs(replay[w]-declared[key][w]) > 1e-10
                           for w in ("U", "PWGTP")):
                        raise ValueError("outer aggregate score differs from private person replay")
                if role in h_by_role and not _same_score(h_by_role[role], h):
                    raise ValueError("shared H ancestor differs across scored releases")
                h_by_role[role] = h
                scores[(release_id, anchor, role)] = candidate
    for role, score in h_by_role.items():
        scores[("H", anchor, role)] = score
    provenance = {"directory": str(directory),
                  "complete_sha256": evaluate._sha(complete_path),
                  "outer_audit_sha256": evaluate._sha(report_path),
                  "contributions_sha256": evaluate._sha(contributions_path),
                  "contributions_relative_path": str(member),
                  "canonical_release_ids": sorted(locked_releases),
                  "outer_scorer_source_sha256": evaluate._sha(outer_audit.__file__)}
    return scores, provenance, report


def _write_new(path: Path, value: Mapping, *, private: bool) -> None:
    if path.exists():
        raise FileExistsError("completed inference artifact is immutable")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private and "private" not in path.parts:
        raise ValueError("private replay index must remain in private directory")
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(encoded)
    os.chmod(temporary, 0o600 if private else 0o644)
    os.replace(temporary, path)


def run_inference(selection_lock_path: str | Path,
                  expected_lock_sha256: str,
                  outer_dirs: Mapping[int, str | Path],
                  output_path: str | Path,
                  private_index_path: str | Path) -> dict:
    """Run exactly the locked Bonferroni family on three completed anchors."""
    output = Path(output_path).resolve()
    private_index = Path(private_index_path).resolve()
    if output.exists() or private_index.exists():
        raise FileExistsError("inference results or private replay index already exist")
    if set(outer_dirs) != set(ANCHORS):
        raise ValueError("exactly three locked outer anchor directories required")
    lock_file = evaluate._pin(selection_lock_path, expected_lock_sha256)
    lock = json.loads(lock_file.read_text())
    resolution = validate_locked_resolution(lock)
    _verify_remote_lock(lock_file, expected_lock_sha256)
    scores = {}
    provenance = {}
    for anchor in ANCHORS:
        locked = lock["anchors"][str(anchor)]
        current, source, report = _load_completed_outer(
            outer_dirs[anchor], anchor, expected_lock_sha256, locked["releases"])
        if (report.get("inner_panel_complete_sha256") !=
                locked.get("inner_panel_complete_sha256") or
                report.get("inner_audit_sha256") != locked.get("inner_audit_sha256")):
            raise ValueError("outer score did not use the locked inner predictor panel")
        scores.update(current)
        provenance[str(anchor)] = source
    endpoints = lock["family_manifest"]["endpoints"]
    for anchor in ANCHORS:
        for logical, canonical in resolution[anchor].items():
            for role in audit.ROLES:
                scores[(logical, anchor, role)] = scores[(canonical, anchor, role)]
    result = inference.evaluate_family(
        endpoints, scores, n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED, alpha=ALPHA)
    if (result["family_size"] != lock["family_manifest"]["n_endpoints"] or
            {row["id"] for row in result["rows"]} !=
            {endpoint["id"] for endpoint in endpoints}):
        raise ValueError("inference endpoint family differs from committed lock")
    private_record = {"schema": "pcrl-adaptive-inference-replay-index-v1",
                      "selection_lock_sha256": expected_lock_sha256,
                      "outer_artifacts": provenance,
                      "endpoint_ids": [item["id"] for item in endpoints],
                      "bootstrap_seed": BOOTSTRAP_SEED,
                      "bootstrap_draws": BOOTSTRAP_DRAWS,
                      "inference_run_source_sha256": evaluate._sha(__file__),
                      "inference_source_sha256": evaluate._sha(inference.__file__)}
    _write_new(private_index, private_record, private=True)
    report = {"schema": "pcrl-adaptive-outer-inference-v1",
              "assessment_year": 2018, "development_only": True,
              "selection_lock_sha256": expected_lock_sha256,
              "family_manifest": lock["family_manifest"],
              "anchor_resolution": {str(a): resolution[a] for a in ANCHORS},
              "private_replay_index_sha256": evaluate._sha(private_index),
              **result}
    _write_new(output, report, private=False)
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--outer-dir", action="append", required=True,
                        help="repeat 0=PATH, 1=PATH, 2=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--private-replay-index", required=True)
    args = parser.parse_args(argv)
    outer = {}
    for raw in args.outer_dir:
        anchor, sep, path = raw.partition("=")
        if not sep or anchor not in ("0", "1", "2") or not path or int(anchor) in outer:
            raise ValueError("each outer-dir must be one unique anchor=PATH")
        outer[int(anchor)] = path
    report = run_inference(args.selection_lock, args.lock_sha256,
                           outer, args.output, args.private_replay_index)
    print(json.dumps({"schema": report["schema"],
                      "family_size": report["family_size"],
                      "selection_lock_sha256": report["selection_lock_sha256"],
                      "inference_sha256": evaluate._sha(args.output)}, sort_keys=True))
    return report


if __name__ == "__main__":
    main()

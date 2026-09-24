"""One locked 2018 outer-development replay of previously selected predictors.

This module has no fitting or route-selection operation. The coordinator owns
the lock, verified remote publication, and the separate outer-label gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import audit, evaluate, roles
from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit
from experiments.pcrl_task_aligned_cuts_v1 import data


def current_inner_code_hashes(*, include_task_baselines: bool = False) -> dict[str, str]:
    """Hash only code paths used by the inner scorer and this replay."""
    answer = {"evaluate.py": evaluate._sha(evaluate.__file__),
              "audit.py": evaluate._sha(audit.__file__),
              "roles.py": evaluate._sha(roles.__file__),
              "inherited_audit.py": evaluate._sha(inherited_audit.__file__),
              "data.py": evaluate._sha(data.__file__)}
    if include_task_baselines:
        from . import task_baselines
        answer["task_baselines.py"] = evaluate._sha(task_baselines.__file__)
    return answer


def verify_scoring_provenance(inner_report: dict, index_path: str | Path) -> None:
    """Refuse a different prepared index or changed token scorer before unlock."""
    index_file = Path(index_path).resolve()
    if (not index_file.is_file() or
            evaluate._sha(index_file) != inner_report.get("index_sha256")):
        raise ValueError("locked inner prepared index SHA differs")
    include_task = any(payload.get("source", {}).get("law_kind") ==
                       "task_only_original_person_17_token"
                       for payload in inner_report.get("releases", {}).values())
    if inner_report.get("source_code_sha256") != current_inner_code_hashes(
            include_task_baselines=include_task):
        raise ValueError("inner scoring source SHA differs from frozen panel")


def release_descriptor(spec: Mapping[str, Any], index_path: str | Path) -> dict:
    """Recreate the exact inner-panel artifact descriptor without any labels."""
    index_dir = Path(index_path).resolve().parent
    if "task_only_model_dir" in spec:
        from . import task_baselines
        receipt_path = evaluate._pin(
            Path(spec["task_only_model_dir"]) / "TASK_ONLY.json",
            spec["task_only_receipt_sha256"])
        _, model_receipt = task_baselines.load_task_only(receipt_path.parent)
        return {
            "law_kind": "task_only_original_person_17_token",
            "task_only_receipt_artifact": evaluate._artifact_record(receipt_path, index_dir),
            "task_only_model_artifact": evaluate._artifact_record(
                receipt_path.parent / "task_only.joblib", index_dir),
            "task_only_model_sha256": model_receipt["model_sha256"],
            "mode": spec["mode"], "publish": spec["publish"],
            "constant_token": 0 if spec["mode"] == "constant_replacement" else None,
            "recipient_observes_person_law": False,
        }
    q, source = evaluate._channel_artifact(spec)
    descriptor = {
        "channel_artifact": evaluate._artifact_record(source, index_dir),
        "channel_array_sha256": evaluate._array_sha(q),
        "router_kind": "T0" if spec["router"] == "T0" else "callable",
    }
    if callable(spec["router"]):
        route_source = evaluate._pin(spec["router_artifact_path"], spec["router_sha256"])
        descriptor["router_artifact"] = evaluate._artifact_record(route_source, index_dir)
        descriptor["router_entrypoint"] = spec["router_entrypoint"]
    else:
        descriptor["router_source"] = "stored Linux x86 T0 codes in pinned prepared object"
    return descriptor


def verify_locked_inner(inner_panel_dir: str | Path, selection_lock_path: str | Path,
                        expected_lock_sha256: str, *, anchor: int,
                        release_ids: set[str]) -> tuple[dict, dict, dict]:
    """Verify every frozen predictor and release pin before outer-label access."""
    root = Path(inner_panel_dir).resolve()
    if "private" not in root.parts or anchor not in (0, 1, 2) or not release_ids:
        raise ValueError("private completed panel, declared anchor and releases required")
    lock_path = evaluate._pin(selection_lock_path, expected_lock_sha256)
    lock = json.loads(lock_path.read_text())
    if (lock.get("schema") != "pcrl-adaptive-selection-lock-v1" or
            lock.get("status") != "LOCKED" or
            lock.get("assessment_year") != 2018):
        raise ValueError("registered 2018 selection lock is absent")
    anchor_record = lock.get("anchors", {}).get(str(anchor))
    if (not isinstance(anchor_record, dict) or
            set(anchor_record.get("releases", {})) != release_ids):
        raise ValueError("selection lock release identities differ for anchor")
    complete_path = root / "COMPLETE.json"
    report_path = root / "INNER_AUDIT.json"
    if (evaluate._sha(complete_path) != anchor_record.get("inner_panel_complete_sha256") or
            evaluate._sha(report_path) != anchor_record.get("inner_audit_sha256")):
        raise ValueError("inner report or completion SHA differs from selection lock")
    complete = json.loads(complete_path.read_text())
    report = json.loads(report_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-inner-audit-complete-v1" or
            complete.get("anchor") != anchor or
            complete.get("artifacts") != evaluate._inventory(root) or
            report.get("schema") != "pcrl-adaptive-inner-audit-v1" or
            report.get("anchor") != anchor or
            report.get("outer_pool_opened") is not False or
            set(report.get("releases", {})) != set(complete.get("release_ids", [])) or
            not release_ids <= set(report.get("releases", {}))):
        raise ValueError("inner panel receipt or report differs from locked complete panel")
    for release_id in release_ids:
        if (report["releases"][release_id].get("source") !=
                anchor_record["releases"][release_id]):
            raise ValueError("inner release descriptor differs from selection lock")
    return lock, report, complete


def _load_registry(inner_root: Path, release_id: str, role: str,
                   declared_slate: str) -> dict:
    path = inner_root / release_id / role.replace(":", "_").replace("/", "_")
    registry_path = path / "own_registry.json"
    registry = json.loads(registry_path.read_text())
    if (registry.get("role") != role or registry.get("release_id") != release_id or
            registry.get("slate") != declared_slate or
            registry.get("artifacts_sha256") != inherited_audit._artifact_inventory(path)):
        raise ValueError("frozen inner predictor slate differs")
    for route in registry.get("models", {}).values():
        relative = route.get("model_relative_directory")
        if (not isinstance(relative, str) or Path(relative).is_absolute() or
                ".." in Path(relative).parts):
            raise ValueError("frozen model lacks safe restore-relative path")
        model_dir = (path / relative).resolve()
        if not model_dir.is_relative_to((path / "models").resolve()):
            raise ValueError("frozen model escaped its slate directory")
        if inherited_audit.model_directory_hash(model_dir) != route.get("model_sha256"):
            raise ValueError("frozen audit model hash differs")
        route["model_directory"] = str(model_dir)
    if not registry.get("models"):
        raise ValueError("frozen audit slate contains no models")
    return registry


def reconstruct_selected_routes(inner_root: str | Path, report: dict,
                                release_id: str) -> tuple[dict, dict]:
    """Select exactly saved route IDs; never recompute validation selection."""
    root = Path(inner_root).resolve()
    slate = report.get("slate")
    if slate not in ("standard", "catchup") or release_id not in report["releases"]:
        raise ValueError("frozen audit slate or release absent")
    h_roles = (*audit.ROLES, "attack:B/SEX", "attack:B/RAC1P")
    h_regs = {role: _load_registry(root, "H", role, slate) for role in h_roles}
    own_regs = {role: _load_registry(root, release_id, role, slate)
                for role in audit.ROLES}
    own_locks, h_locks = {}, {}
    for role in audit.ROLES:
        record = report["releases"][release_id]["roles"][role]
        own_routes = evaluate._registry_routes(
            role, release_id, own_regs[role], h_regs[role], own_regs, h_regs)
        h_routes = evaluate._registry_routes(
            role, "H", h_regs[role], h_regs[role], h_regs, h_regs)
        for field, metadata, routes, identity, sink in (
            ("selected_candidate", "selected_route", own_routes, release_id, own_locks),
            ("H_selected_candidate", "H_selected_route", h_routes, "H", h_locks),
        ):
            chosen = record[field]
            if chosen not in routes:
                raise ValueError("saved validation-selected route absent from frozen bank")
            route = routes[chosen]
            brief = {key: route.get(key) for key in
                     ("source_view", "wire", "model_sha256")}
            if brief != record[metadata]:
                raise ValueError("frozen route differs from saved inner selection")
            sink[role] = {"role": role, "release_id": identity,
                          "selected": chosen, "route": route}
        if len(own_routes) != record["candidate_count"]:
            raise ValueError("frozen route candidate count differs")
    return own_locks, h_locks


def _load_unlocked_rows(index_path: str | Path, anchor: int,
                        selection_lock_path: str | Path,
                        expected_lock_sha256: str) -> dict:
    from . import outer_access
    return outer_access.load_locked_outer_role(index_path, anchor,
                                               selection_lock_path,
                                               expected_lock_sha256)


def score_locked_outer(anchor: int, releases: Mapping[str, Mapping[str, Any]],
                       index_path: str | Path, inner_panel_dir: str | Path,
                       selection_lock_path: str | Path,
                       expected_lock_sha256: str,
                       output_dir: str | Path) -> dict:
    """Score one anchor once, with all routes fixed before gate invocation."""
    root = Path(output_dir).resolve()
    if "private" not in root.parts:
        raise ValueError("outer person and household contributions must remain private")
    if root.exists():
        raise FileExistsError("outer scoring output already exists; preserve first record")
    if not isinstance(releases, Mapping) or not releases:
        raise ValueError("locked nonempty release panel required")
    release_ids = set(releases)
    lock, inner, complete = verify_locked_inner(
        inner_panel_dir, selection_lock_path, expected_lock_sha256,
        anchor=anchor, release_ids=release_ids)
    index_file = Path(index_path).resolve()
    verify_scoring_provenance(inner, index_file)
    route_locks = {}
    for release_id, spec in releases.items():
        evaluate._slug(release_id)
        if release_descriptor(spec, index_file) != inner["releases"][release_id]["source"]:
            raise ValueError("supplied release artifact differs from locked inner descriptor")
        route_locks[release_id] = reconstruct_selected_routes(
            inner_panel_dir, inner, release_id)
    # This is the sole outer-label access path. All selection, model and
    # release checks above precede it; no route is changed afterward.
    rows = _load_unlocked_rows(index_file, anchor, selection_lock_path,
                                expected_lock_sha256)
    h_law = np.ones((len(rows["ids"]), 1), dtype=np.float64)
    results, contributions = {}, {}
    for release_id in sorted(releases):
        law = evaluate.token_law_for_release(releases[release_id], rows)
        own_locks, h_locks = route_locks[release_id]
        role_results = {}
        prefix = inner["releases"][release_id]["private_contribution_prefix"]
        for role in audit.ROLES:
            score = audit.score_frozen_route(rows, law, own_locks[role])
            h_score = audit.score_frozen_route(rows, h_law, h_locks[role])
            contributions.update(evaluate._private_contributions(
                prefix, role, score, h_score))
            role_results[role] = {
                "n_people": len(score["loss"]),
                "n_households": len(set(score["households"])),
                "candidate": score["scores"], "H": h_score["scores"],
                "H_minus_candidate": {w: h_score["scores"][w]-score["scores"][w]
                                      for w in ("U", "PWGTP")},
                "selected_candidate": own_locks[role]["selected"],
                "H_selected_candidate": h_locks[role]["selected"],
            }
        results[release_id] = {"source": inner["releases"][release_id]["source"],
                               "roles": role_results,
                               "private_contribution_prefix": prefix}
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    contributions_path = root / "OUTER_CONTRIBUTIONS.npz"
    np.savez_compressed(contributions_path, **contributions)
    report = {"schema": "pcrl-adaptive-outer-audit-v1", "anchor": anchor,
              "assessment_year": 2018, "development_only": True,
              "no_outer_fit_or_selection": True,
              "selection_lock_sha256": expected_lock_sha256,
              "inner_panel_complete_sha256": lock["anchors"][str(anchor)]["inner_panel_complete_sha256"],
              "inner_audit_sha256": lock["anchors"][str(anchor)]["inner_audit_sha256"],
              "index_sha256": evaluate._sha(index_file),
              "releases": results,
              "contributions_relative_path": contributions_path.name,
              "contributions_sha256": evaluate._sha(contributions_path)}
    evaluate._json_atomic(root / "OUTER_AUDIT.json", report)
    evaluate._seal_private_permissions(root)
    receipt = {"schema": "pcrl-adaptive-outer-audit-complete-v1",
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "anchor": anchor, "release_ids": sorted(releases),
               "selection_lock_sha256": expected_lock_sha256,
               "artifacts": evaluate._inventory(root)}
    evaluate._json_atomic(root / "COMPLETE.json", receipt)
    (root / "COMPLETE.json").chmod(0o600)
    return report


def main(argv: list[str] | None = None) -> dict:
    """Coordinator-dispatched reconstruction from pinned fitted sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--delta", type=float, choices=(0., .001, .003), required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--inner-panel-dir", required=True)
    parser.add_argument("--selection-lock", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--a-center-dir", required=True)
    parser.add_argument("--b-center-dir")
    parser.add_argument("--b-parity-receipt")
    parser.add_argument("--a-controls-dir")
    parser.add_argument("--b-controls-dir")
    parser.add_argument("--task-only-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    from . import release_specs

    lock_path = evaluate._pin(args.selection_lock, args.lock_sha256)
    lock = json.loads(lock_path.read_text())
    if (lock.get("schema") != "pcrl-adaptive-selection-lock-v1" or
            lock.get("status") != "LOCKED" or
            lock.get("assessment_year") != 2018 or
            str(args.anchor) not in lock.get("anchors", {})):
        raise ValueError("registered selection lock required")
    required = set(lock["anchors"][str(args.anchor)].get("releases", {}))
    if not required:
        raise ValueError("selection lock has no scored release IDs")
    bundle = release_specs.build_release_specs(
        args.anchor, args.delta, args.a_center_dir,
        b_center_dir=args.b_center_dir,
        b_parity_receipt=args.b_parity_receipt,
        a_controls_dir=args.a_controls_dir,
        b_controls_dir=args.b_controls_dir,
        task_only_dir=args.task_only_dir)
    available = bundle["releases"]
    if not required <= set(available):
        raise ValueError("locked release ID is absent from verified fitted sources")
    releases = {key: available[key] for key in sorted(required)}
    report = score_locked_outer(
        args.anchor, releases, args.index, args.inner_panel_dir,
        args.selection_lock, args.lock_sha256, args.output_dir)
    print(json.dumps({"schema": report["schema"], "anchor": args.anchor,
                      "release_count": len(releases),
                      "complete_sha256": evaluate._sha(Path(args.output_dir) / "COMPLETE.json")},
                     sort_keys=True))
    return report


if __name__ == "__main__":
    main()

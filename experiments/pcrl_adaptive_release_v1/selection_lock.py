"""Build the one 2018 development assessment lock from validation-only scores.

The builder verifies completed inner panels and their per-anchor alias indexes.
It never loads assessment rows or inner-check person contributions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import evaluate, inference, run_inner, selection

FAMILIES = {
    "simple": (
        "A_control_D17_constant_replace_publish_",
        "A_control_D17_randomized_response_publish_"),
    "task_only": ("TaskOnly", "A_control_D_task"),
    "gradient": ("A_control_GRADIENT_",),
    "deterministic": ("A_control_MILP", "A_control_deterministic_search"),
}


def _sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def global_exact_aliases(anchor_aliases: dict[int, dict[str, str]],
                         names: set[str]) -> dict[str, str]:
    """Collapse only laws with equal canonical identities on all three anchors."""
    if set(anchor_aliases) != set(selection.ANCHORS):
        raise ValueError("three anchor alias maps required")
    vectors = {}
    for name in sorted(names):
        vector = tuple(anchor_aliases[anchor].get(
            "A_control_D17" if name == "D17" else name)
            for anchor in selection.ANCHORS)
        if any(value is None for value in vector):
            raise ValueError(f"logical release {name} absent on an anchor")
        vectors.setdefault(vector, []).append(name)
    aliases = {}
    for group in vectors.values():
        representative = min(group, key=lambda name: (
            name != "D17", name != "A_selected", name))
        aliases.update({name: representative for name in group
                        if name != representative})
    return aliases


def _family_members(names: set[str], family: str) -> list[str]:
    prefixes = FAMILIES[family]
    return sorted(name for name in names if any(name.startswith(prefix)
                                               for prefix in prefixes))


def _verified_anchor(private_root: Path, anchor: int, index_sha: str) -> tuple[dict, dict, dict]:
    panel = private_root / f"a{anchor}_inner_panel_d001"
    spec_path = private_root / f"a{anchor}_INNER_SPECS.json"
    binding_path = spec_path.with_suffix(".binding.json")
    complete = json.loads((panel / "COMPLETE.json").read_text())
    report = json.loads((panel / "INNER_AUDIT.json").read_text())
    spec = json.loads(spec_path.read_text())
    binding = json.loads(binding_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-inner-audit-complete-v1" or
            complete.get("anchor") != anchor or
            complete.get("artifacts") != evaluate._inventory(panel) or
            report.get("schema") != "pcrl-adaptive-inner-audit-v1" or
            report.get("anchor") != anchor or
            report.get("selection_role") != "inner_selection" or
            report.get("score_role") != "inner_check" or
            report.get("outer_pool_opened") is not False or
            set(report.get("releases", {})) != set(complete.get("release_ids", [])) or
            report.get("index_sha256") != index_sha or
            spec.get("schema") != "pcrl-inner-audit-spec-index-v1" or
            spec.get("anchor") != anchor or spec.get("delta") != .001 or
            spec.get("slate") != "standard" or
            spec.get("input_index_sha256") != index_sha or
            set(spec.get("canonical_ids", [])) != set(report["releases"]) or
            binding.get("schema") != "pcrl-inner-panel-binding-v1" or
            binding.get("anchor") != anchor or
            binding.get("spec_index_sha256") != _sha(spec_path) or
            binding.get("inner_complete_sha256") != _sha(panel / "COMPLETE.json") or
            binding.get("inner_audit_sha256") != _sha(panel / "INNER_AUDIT.json") or
            binding.get("canonical_ids") != spec["canonical_ids"]):
        raise ValueError(f"anchor {anchor} completed inner panel or binding differs")
    aliases = spec.get("aliases")
    if (not isinstance(aliases, dict) or
            set(aliases.values()) != set(report["releases"])):
        raise ValueError("pinned logical alias roster differs from inner panel")
    scores = selection.expand_named_scores(
        selection.inner_validation_from_report(report), aliases)
    return report, spec, {
        "scores": scores, "aliases": aliases,
        "inner_panel_complete_sha256": _sha(panel / "COMPLETE.json"),
        "inner_audit_sha256": _sha(panel / "INNER_AUDIT.json"),
        "inner_spec_index_sha256": _sha(spec_path),
        "inner_binding_sha256": _sha(binding_path),
    }


def build_lock(private_root: str | Path, index_path: str | Path,
               protocol_path: str | Path, *,
               external_j: bool = False) -> tuple[dict, dict]:
    root = Path(private_root).resolve()
    if "private" not in root.parts:
        raise ValueError("verified fitted panels must remain private")
    index_sha = _sha(index_path)
    reports, specifications, records = {}, {}, {}
    for anchor in selection.ANCHORS:
        reports[anchor], specifications[anchor], records[anchor] = _verified_anchor(
            root, anchor, index_sha)
    names = set(records[0]["scores"]) - {"H", "D17"}
    if any(set(records[a]["scores"]) - {"H", "D17"} != names
           for a in selection.ANCHORS):
        raise ValueError("anchor logical release rosters differ")
    scores = {a: records[a]["scores"] for a in selection.ANCHORS}
    aliases = global_exact_aliases(
        {a: records[a]["aliases"] for a in selection.ANCHORS},
        names | {"D17"})
    family_selections = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        members = _family_members(names, family)
        if not members:
            raise ValueError(f"mandatory fitted {family} family absent")
        for route in ("U", "P"):
            family_selections[family][route] = selection.select_family_representative(
                scores, family, members, route)
    candidate_names = [name for name in ("A_selected", "B_selected") if name in names]
    if len(candidate_names) != 2:
        raise ValueError("both registered A and B candidate names required")
    # An exact B=A alias on all anchors is one physical candidate, while the
    # B branch's recorded support-limited status remains in its fitted receipt.
    candidate_names = [name for name in candidate_names
                       if name == "A_selected" or aliases.get(name, name) != "A_selected"]
    choices = {route: selection.select_route_candidate(
        scores, candidate_names, route) for route in ("U", "P")}
    slots = []
    for route in ("U", "P"):
        comparator_names = ["D17", *(family_selections[family][route]["selected"]
                                       for family in FAMILIES),
                            "A_control_historical_Q"]
        slots.append({"id": f"{route}_candidate", "arm": route,
                      "source_name": choices[route]["selected"],
                      "selection_status": choices[route]["status"],
                      "comparators": list(dict.fromkeys(comparator_names))})
    selected_comparators = {name for slot in slots for name in slot["comparators"]}
    endpoint_aliases = {name: canonical for name, canonical in aliases.items()
                        if name in selected_comparators}
    manifest = inference.family_manifest(slots, alias_of=endpoint_aliases)
    anchor_locks = {}
    for anchor in selection.ANCHORS:
        mapping = dict(records[anchor]["aliases"])
        mapping["D17"] = mapping["A_control_D17"]
        for slot in slots:
            mapping[slot["id"]] = mapping[slot["source_name"]]
        needed = {name for endpoint in manifest["endpoints"]
                  for name in (endpoint["plus"], endpoint["minus"])}
        needed.update(name for slot in slots for name in slot["comparators"])
        needed.update(slot["source_name"] for slot in slots)
        canonical = {mapping[name] for name in needed}
        anchor_locks[str(anchor)] = {
            key: records[anchor][key] for key in (
                "inner_panel_complete_sha256", "inner_audit_sha256",
                "inner_spec_index_sha256", "inner_binding_sha256")}
        anchor_locks[str(anchor)]["logical_to_canonical"] = {
            name: mapping[name] for name in sorted(needed)}
        anchor_locks[str(anchor)]["releases"] = {
            name: reports[anchor]["releases"][name]["source"]
            for name in sorted(canonical)}
    selection_source = {
        "role": "inner_selection", "scores_only": "validation-selected predictors",
        "outer_pool_opened": False,
        "selection_code_sha256": _sha(selection.__file__),
        "inference_code_sha256": _sha(inference.__file__),
        "candidate_choices": choices,
        "family_choices": family_selections,
        "candidate_names": candidate_names,
        "family_members": {family: _family_members(names, family)
                           for family in FAMILIES},
    }
    from . import outer_access
    lock = {
        "schema": "pcrl-adaptive-selection-lock-v1", "status": "LOCKED",
        "study": "pcrl_adaptive_release_v1", "assessment_year": 2018,
        "development_only": True, "outer_assessment_authorized": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha(protocol_path),
        "input_index_sha256": index_sha,
        "scoring_code_sha256": outer_access.scoring_code_hashes(),
        "outer_population": {
            "census_sha256": _sha(outer_access.CENSUS_PATH),
            "pools": list(outer_access.data.POOLS),
            "role": "outer_assessment"},
        "selection_source": selection_source,
        "slots": slots, "alias_of": endpoint_aliases,
        "family_representatives": {family: {
            route: family_selections[family][route]["selected"]
            for route in ("U", "P")} for family in FAMILIES},
        "family_manifest": manifest,
        "capability_manifest": {
            "schema": 1,
            "endpoints": inference.capability_endpoints(
                [slot["id"] for slot in slots]),
            "n_endpoints": 2 * len(slots),
            "multiplicity": "separate two-sided Bonferroni H-capability family",
            "scope": "2018 development; not a primary-clause rescue",
        },
        "anchors": anchor_locks,
    }
    if external_j:
        external = {}
        for anchor in selection.ANCHORS:
            directory = root / f"a{anchor}_external_inner_J"
            complete_path = directory / "COMPLETE.json"
            report_path = directory / "INNER_AUDIT.json"
            complete = json.loads(complete_path.read_text())
            report = json.loads(report_path.read_text())
            if (complete.get("schema") != "pcrl-adaptive-continuous-inner-complete-v1" or
                    complete.get("anchor") != anchor or
                    complete.get("release_ids") != ["J"] or
                    complete.get("artifacts") != evaluate._inventory(directory) or
                    report.get("schema") != "pcrl-adaptive-continuous-inner-audit-v1" or
                    report.get("anchor") != anchor or
                    report.get("outer_pool_opened") is not False or
                    report.get("slate") != "standard" or
                    report.get("input_index_sha256") != index_sha or
                    set(report.get("releases", {})) != {"J"}):
                raise ValueError(f"anchor {anchor} contextual J inner audit differs")
            external[str(anchor)] = {
                "inner_complete_sha256": _sha(complete_path),
                "inner_audit_sha256": _sha(report_path),
                "source": report["releases"]["J"]["source"],
            }
        lock["external_context"] = {"J": external,
            "comparison_scope": "contextual continuous 16-coordinate A auxiliary; not a matched 17-token baseline"}
    # Enforce the same pre-unlock structural validation without creating an
    # unlock receipt or touching any assessment rows.
    outer_access.verify_lock_structure(lock)
    summary = {"schema": "pcrl-adaptive-inner-selection-v1",
               "source_role": "inner_selection", "development_only": True,
               "candidate_choices": choices,
               "family_choices": family_selections,
               "named_validation_scores": scores,
               "global_exact_aliases": aliases,
               "outer_pool_opened": False}
    return lock, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--lock-out", required=True)
    parser.add_argument("--selection-out", required=True)
    parser.add_argument("--external-j", action="store_true",
                        help="pin three already completed same-host continuous J inner audits")
    args = parser.parse_args(argv)
    lock, summary = build_lock(args.private_root, args.index, args.protocol,
                               external_j=args.external_j)
    for name, value in ((args.lock_out, lock),
                        (args.selection_out, summary)):
        path = Path(name)
        if path.exists():
            raise FileExistsError(f"preserve existing selection artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n")
    print(json.dumps({"lock_sha256": _sha(args.lock_out),
                      "selection_sha256": _sha(args.selection_out),
                      "n_endpoints": lock["family_manifest"]["n_endpoints"],
                      "slot_status": {slot["arm"]: slot["selection_status"]
                                      for slot in lock["slots"]}}, sort_keys=True))


if __name__ == "__main__":
    main()

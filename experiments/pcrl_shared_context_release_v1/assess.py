"""PROTOCOL section 7: locked assessment from completed outer score archives.

Reads only the immutable outputs of `audit_panel.score_outer` (and
`score_outer_j` for the J secondary endpoints), after the coordinator's gate
(`audit_panel.verify_outer_gate`) accepts the lock and the remote-verified
unlock. It fits nothing and selects nothing: every endpoint, slot and
comparator comes from the committed lock, whose families are regenerated and
hash-checked first.

Uncertainty is AR's (`TAC/inference.evaluate_family` ->
`TDR/uncertainty.paired_household_bounds`): 10,000 draws, seed 20260923, one
multinomial over the union of household IDs across the three anchors, per-anchor
weighted ratios averaged equally, normal-approximation Bonferroni bounds within
each family. Primary, secondary and H-capability families are corrected
separately. A route passes only if all of its primary clauses pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.pcrl_adaptive_release_v1 import evaluate, inference_run
from experiments.pcrl_task_aligned_cuts_v1 import inference as paired

from . import audit_panel, lock as lock_module

ANCHORS = lock_module.ANCHORS
ROLES = lock_module.ROLES
FAMILIES = ("family_manifest", "secondary_manifest", "capability_manifest")
CSV_FIELDS = ("family", "id", "candidate", "candidate_release", "arm", "comparator", "role",
              "weighting", "clause", "threshold", "estimate", "bootstrap_se", "lower", "upper",
              "passed_upper_bound", "point_screen_passed", "demonstrated_adverse",
              "exact_zero_same_route", "anchor_estimate_0", "anchor_estimate_1", "anchor_estimate_2")


def _load_outer(directory: str | Path, anchor: int, lock_sha: str, *, j: bool = False) -> dict:
    """Scores {(canonical, anchor, role): person record} from one completed outer archive."""
    root = Path(directory).resolve()
    if "private" not in root.parts:
        raise ValueError("outer contributions must stay private")
    complete = json.loads((root / "COMPLETE.json").read_text())
    report = json.loads((root / "OUTER_AUDIT.json").read_text())
    schema = ("pcrl-sc-outer-J-complete-v1" if j else "pcrl-sc-outer-audit-complete-v1")
    if (complete.get("schema") != schema or complete.get("anchor") != anchor or
            complete.get("artifacts") != evaluate._inventory(root) or
            report.get("anchor") != anchor or report.get("selection_lock_sha256") != lock_sha or
            report.get("no_outer_fit_or_selection") is not True or
            evaluate._sha(root / "OUTER_CONTRIBUTIONS.npz") != report.get("contributions_sha256")):
        raise ValueError(f"outer archive for anchor {anchor} is incomplete or not from this lock")
    scores, h_by_role = {}, {}
    with np.load(root / "OUTER_CONTRIBUTIONS.npz", allow_pickle=False) as archive:
        for release, payload in report["releases"].items():
            for role in ROLES:
                candidate = inference_run._score_record(
                    archive, payload["private_contribution_prefix"], role, "candidate")
                h = inference_run._score_record(
                    archive, payload["private_contribution_prefix"], role, "H")
                if role in h_by_role and not inference_run._same_score(h_by_role[role], h):
                    raise ValueError("shared H differs across scored releases")
                h_by_role[role] = h
                scores[(release, anchor, role)] = candidate
    for role, h in h_by_role.items():
        scores[("H", anchor, role)] = h
    return scores


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"{path.name} is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def assess(selection_lock_path: str | Path, expected_lock_sha256: str,
           outer_dirs: Mapping[int, str | Path], output_dir: str | Path, *,
           j_outer_dirs: Mapping[int, str | Path] | None = None,
           n_boot: int = lock_module.BOOTSTRAP["draws"], _gate=None) -> dict:
    gate = _gate or audit_panel.verify_outer_gate
    lock = gate(selection_lock_path, expected_lock_sha256)
    lock_module.verify_lock_structure(lock)
    if set(outer_dirs) != set(ANCHORS):
        raise ValueError("three completed outer anchors required")
    out = Path(output_dir)
    scores, h_match = {}, {}
    for anchor in ANCHORS:
        scores.update(_load_outer(outer_dirs[anchor], anchor, expected_lock_sha256))
        mapping = lock["anchors"][str(anchor)]["logical_to_canonical"]
        for logical, canonical in mapping.items():
            for role in ROLES:
                if (canonical, anchor, role) in scores:
                    scores[(logical, anchor, role)] = scores[(canonical, anchor, role)]
        if j_outer_dirs is not None:
            j_scores = _load_outer(j_outer_dirs[anchor], anchor, expected_lock_sha256, j=True)
            for role in ROLES:
                scores[("J", anchor, role)] = j_scores[("J", anchor, role)]
                h_match[f"{anchor}|{role}"] = inference_run._same_score(
                    j_scores[("H", anchor, role)], scores[("H", anchor, role)])
    needed = {name for fam in FAMILIES for e in lock[fam]["endpoints"]
              for name in (e["plus"], e["minus"])
              if not (name == "J" and j_outer_dirs is None)}
    missing = sorted({f"{n}@a{a}" for n in needed for a in ANCHORS for r in ROLES
                      if (n, a, r) not in scores})
    if missing:
        raise ValueError(f"outer archives lack locked releases: {missing[:10]}")
    seed, alpha = lock["bootstrap"]["seed"], lock["bootstrap"]["alpha"]
    results, rows_out = {}, []
    for name in FAMILIES:
        endpoints = lock[name]["endpoints"]
        if name == "secondary_manifest" and j_outer_dirs is None:
            endpoints = [e for e in endpoints if "J" not in (e["plus"], e["minus"])]
        if not endpoints:
            continue
        clean = [{k: v for k, v in e.items() if k != "exact_zero_same_route"} for e in endpoints]
        result = paired.evaluate_family(clean, scores, n_boot=n_boot, seed=seed, alpha=alpha)
        flags = {e["id"]: e["exact_zero_same_route"] for e in endpoints}
        for row in result["rows"]:
            row["exact_zero_same_route"] = flags[row["id"]]
            rows_out.append(row)
        results[name] = result
    primary = results["family_manifest"]
    by_comparator = {}
    for slot in lock["slots"]:
        rows = [r for r in primary["rows"] if r["candidate"] == slot["id"]]
        by_comparator[slot["id"]] = {
            comp: all(r["passed_upper_bound"] for r in rows if r["comparator"] == comp)
            for comp in sorted({r["comparator"] for r in rows})}
    decisions = {slot["id"]: {"nominee": slot["source_name"], "label": slot["label"],
                              "route_passed_all_primary_clauses":
                                  primary["decisions"][slot["id"]]["all_primary_clauses_passed"],
                              "clauses_passed": primary["decisions"][slot["id"]]["passed"],
                              "clauses_total": primary["decisions"][slot["id"]]["total"],
                              "all_clauses_by_comparator": by_comparator[slot["id"]],
                              "note": "a pass against D17 alone is not a competitive advantage"}
                 for slot in lock["slots"]}
    report = {"schema": "pcrl-sc-inference-v1", "study": audit_panel.STUDY,
              "assessment_year": 2018, "development_only": True,
              "selection_lock_sha256": expected_lock_sha256,
              "no_outer_fit_or_selection": True, "bootstrap_draws": n_boot,
              "decisions": decisions,
              "families": {name: {k: v for k, v in res.items() if k != "rows"}
                           for name, res in results.items()},
              "j_H_identical_to_main_H": h_match or None,
              "scope": ("conditional on the fitted, selected objects; not the adaptive research "
                        "history nor the ACS survey design")}
    out.mkdir(parents=True, exist_ok=True)
    _write_new(out / "INFERENCE.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_new(out / "ENDPOINT_TABLE.json",
               json.dumps({"selection_lock_sha256": expected_lock_sha256, "rows": rows_out},
                          indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [",".join(CSV_FIELDS)]
    for row in rows_out:
        values = {**row, **{f"anchor_estimate_{i}": v for i, v in enumerate(row["anchor_estimates"])}}
        lines.append(",".join(_csv(values.get(field)) for field in CSV_FIELDS))
    _write_new(out / "FULL_RESULTS.csv", "\n".join(lines) + "\n")
    return report


def _csv(value) -> str:
    if value is None:
        return ""
    text = repr(value) if isinstance(value, float) else str(value)
    return f'"{text}"' if "," in text else text


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lock", default=str(audit_panel.LOCK_PATH))
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--outer", nargs=3, required=True, help="outer score dirs a0 a1 a2")
    parser.add_argument("--outer-j", nargs=3, help="outer J score dirs a0 a1 a2")
    parser.add_argument("--out", default=str(audit_panel.RESULTS))
    args = parser.parse_args(argv)
    report = assess(args.lock, args.lock_sha256, dict(zip(ANCHORS, args.outer)), args.out,
                    j_outer_dirs=dict(zip(ANCHORS, args.outer_j)) if args.outer_j else None)
    print(json.dumps(report["decisions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

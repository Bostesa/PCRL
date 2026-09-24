"""EXPLORATORY post-hoc family, chosen AFTER the locked outcomes were seen.

Status `EXPLORATORY_POST_HOC_NOT_REGISTERED`. It changes no locked result: it
reads only the outer score archives that were already produced under the lock
(`audit_panel.score_outer` outputs verified against the lock SHA), refits
nothing, scores nothing new, and writes to its own directory.

Family: DET_SEL4, DET_SEL1, RD_TASK and NM1_P, each versus D17, over 5 roles x
2 weightings (task + 8 recovery contrasts per candidate) with the primary
family's D17 sign conventions (task = CE_cand - CE_D17; recovery =
CE_D17 - CE_cand). Clause thresholds are the primary ones for the unit's form
(P for NM1_P, U otherwise) and are shown for description only.

Uncertainty uses the locked machinery (TAC `evaluate_family` -> TDR
`paired_household_bounds`) with 10,000 draws but a DIFFERENT seed (20260925),
so the result is not presented as the locked draw, and Bonferroni over this
family's own size.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.pcrl_task_aligned_cuts_v1 import inference as paired

from . import assess, audit_panel, laws, lock as lock_module

STATUS = "EXPLORATORY_POST_HOC_NOT_REGISTERED"
SCHEMA = "pcrl-sc-posthoc-exploratory-v1"
CANDIDATES = {"DET_SEL4": "U", "DET_SEL1": "U", "RD_TASK": "U", "NM1_P": "P"}
COMPARATOR = "D17"
SEED = 20260925
DRAWS = 10000
ANCHORS = lock_module.ANCHORS
ROLES = lock_module.ROLES


def endpoints() -> list[dict]:
    rows = []
    for candidate, arm in CANDIDATES.items():
        slot = {"id": f"posthoc_{candidate}", "arm": arm, "source_name": candidate,
                "comparators": [COMPARATOR]}
        rows += lock_module.enumerate_family([slot], alias_of={}, family="posthoc")
    return rows


def load_scores(lock: Mapping, lock_sha256: str, outer_dirs: Mapping[int, str | Path],
                names: Sequence[str]) -> tuple[dict, dict]:
    """Existing locked outer archives only; refuses if any needed release is unscored."""
    scores, provenance, missing = {}, {}, []
    for anchor in ANCHORS:
        current, report = assess._load_outer(outer_dirs[anchor], anchor, lock_sha256)
        mapping = lock["anchors"][str(anchor)]["logical_to_canonical"]
        confirmed = report.get("outer_alias_confirmed", {})
        broken = report.get("outer_alias_breaks", {})
        provenance[str(anchor)] = {
            "outer_dir": str(Path(outer_dirs[anchor]).resolve()),
            "complete_sha256": laws.sha256_file(Path(outer_dirs[anchor]) / "COMPLETE.json"),
            "outer_audit_sha256": laws.sha256_file(Path(outer_dirs[anchor]) / "OUTER_AUDIT.json")}
        for name in names:
            canonical = mapping.get(name)
            if canonical is None:
                missing.append(f"{name}@a{anchor}: not in the locked panel")
                continue
            if name in report["releases"]:        # canonical, or an alias scored separately
                source = name
            elif canonical in report["releases"] and confirmed.get(name) == canonical:
                source = canonical
            else:
                missing.append(f"{name}@a{anchor}: no outer score under the lock")
                continue
            if name in broken:
                provenance[str(anchor)].setdefault("outer_alias_breaks", {})[name] = broken[name]
            for role in ROLES:
                scores[(name, anchor, role)] = current[(source, anchor, role)]
        for role in ROLES:
            scores[("H", anchor, role)] = current[("H", anchor, role)]
    if missing:
        raise FileNotFoundError("post-hoc family needs existing outer scores; missing: "
                                + "; ".join(missing) + ". Nothing new is scored.")
    return scores, provenance


def run(lock_path: str | Path, lock_sha256: str, outer_dirs: Mapping[int, str | Path],
        output_dir: str | Path, *, unlock_path: str | Path | None = None,
        n_boot: int = DRAWS, remote_check: bool = True) -> dict:
    lock_file = Path(lock_path)
    if laws.sha256_file(lock_file) != lock_sha256:
        raise PermissionError("lock bytes differ from the expected SHA-256")
    lock = json.loads(lock_file.read_text())
    lock_module.verify_lock_structure(lock)
    if unlock_path is not None:
        unlock = json.loads(Path(unlock_path).read_text())
        if (unlock.get("schema") != audit_panel.UNLOCK_SCHEMA or unlock.get("lock_sha256") != lock_sha256
                or unlock.get("remote_verified") is not True):
            raise PermissionError("outer unlock receipt does not match this lock")
        if remote_check:
            audit_panel.remote_lock_check(unlock["remote_commit_sha"], lock_sha256)
    out = Path(output_dir)
    for name in ("POSTHOC_EXPLORATORY.json", "POSTHOC_EXPLORATORY.csv"):
        if (out / name).exists():
            raise FileExistsError(f"{name} is write-once")
    family = endpoints()
    scores, provenance = load_scores(lock, lock_sha256, outer_dirs, [*CANDIDATES, COMPARATOR])
    result = paired.evaluate_family(family, scores, n_boot=n_boot, seed=SEED, alpha=lock_module.ALPHA)
    if result["family_size"] != len(family):
        raise AssertionError("post-hoc family size changed")
    exact_zero = {}
    for row in result["rows"]:
        row["exact_zero_same_law_on_all_anchors"] = all(
            lock["anchors"][str(a)]["logical_to_canonical"][row["candidate_release"]] ==
            lock["anchors"][str(a)]["logical_to_canonical"][COMPARATOR]
            and row["candidate_release"] not in provenance[str(a)].get("outer_alias_breaks", {})
            for a in ANCHORS)
        exact_zero[row["candidate_release"]] = row["exact_zero_same_law_on_all_anchors"]
    report = {"schema": SCHEMA, "status": STATUS, "study": audit_panel.STUDY,
              "chosen_after_locked_outcomes_were_seen": True,
              "changes_locked_results": False, "no_refit_no_new_outer_scoring": True,
              "selection_lock_sha256": lock_sha256, "candidates": CANDIDATES,
              "comparator": COMPARATOR, "bootstrap_seed": SEED, "bootstrap_draws": n_boot,
              "locked_bootstrap_seed_not_used": lock["bootstrap"]["seed"],
              "family_size": result["family_size"],
              "critical_value_two_sided": result["critical_value_two_sided"],
              "multiplicity": "two-sided Bonferroni over this post-hoc family only",
              "thresholds_descriptive_only": True,
              "candidate_identical_to_D17_on_all_anchors": exact_zero,
              "outer_archives": provenance, "bootstrap": result["bootstrap"],
              "rows": result["rows"],
              "scope": ("exploratory; conditional on the fitted, selected objects; "
                        "not part of any registered claim")}
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / f".POSTHOC_EXPLORATORY.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, out / "POSTHOC_EXPLORATORY.json")
    fields = ("status", "id", "candidate_release", "arm", "comparator", "role", "weighting",
              "clause", "threshold", "estimate", "bootstrap_se", "lower", "upper",
              "passed_upper_bound", "exact_zero_same_law_on_all_anchors",
              "anchor_estimate_0", "anchor_estimate_1", "anchor_estimate_2")
    lines = [",".join(fields)]
    for row in result["rows"]:
        values = {**row, "status": STATUS,
                  **{f"anchor_estimate_{i}": v for i, v in enumerate(row["anchor_estimates"])}}
        lines.append(",".join(assess._csv(values.get(field)) for field in fields))
    (out / "POSTHOC_EXPLORATORY.csv").write_text("\n".join(lines) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lock", required=True, help="the committed SELECTION_LOCK.json")
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--outer", nargs=3, required=True, help="existing outer score dirs a0 a1 a2")
    parser.add_argument("--unlock", help="OUTER_UNLOCK.json used to open the outer role")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = run(args.lock, args.lock_sha256, dict(zip(ANCHORS, args.outer)), args.out,
                 unlock_path=args.unlock)
    print(json.dumps({"status": report["status"], "family_size": report["family_size"],
                      "z": report["critical_value_two_sided"],
                      "identical_to_D17": report["candidate_identical_to_D17_on_all_anchors"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()

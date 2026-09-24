"""Post-lock steps, run from a SEPARATE checkout of the pushed lock commit.

Every step calls `lockgate.verify_outer_gate` first (lock SHA, write-once
remote-verified unlock, commit and lock bytes re-checked on origin).

    prepare         copy the sanitized (label-stripped) anchors from the work tree into
                    this checkout (SC `postlock.prepare`)
    restore-outer   fetch AR's three ORIGINAL prepared objects (carry outer labels) by
                    key/version/SHA-256 into this study's private restore root
    score           SC `audit_panel.score_outer` per anchor: frozen inner routes, no refit,
                    no reselection; then label-free outer decision variation
    assess          families from the lock; paired household bootstrap (seed 20260926);
                    registered decision labels; INFERENCE.json, ENDPOINT_TABLE.json,
                    FULL_RESULTS.csv
Fitted units and inner panels are read from the work tree by explicit path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from experiments.pcrl_adaptive_release_v1 import outer_pool
from experiments.pcrl_shared_context_release_v1 import assess as sc_assess, audit_panel, channel, laws, postlock
from experiments.pcrl_task_aligned_cuts_v1 import data, inference as paired

from . import host, lockgate

ROOT = lockgate.ROOT
RESULTS = lockgate.RESULTS
PRIVATE_OUT = RESULTS / "private"
RESTORE_ROOT = PRIVATE_OUT / "original_2018_restore"
WORK = Path(os.environ.get("PFS_WORK_ROOT", "/opt/pcrl/work"))
WORK_PRIVATE = WORK / "results" / lockgate.STUDY / "private"
CENSUS_PATH = ROOT / "results/pcrl_adaptive_release_v1/DATA_ROLE_COUNTS.json"
ANCHORS = lockgate.ANCHORS
FAMILIES = ("family_manifest", "secondary_manifest", "capability_manifest")
CSV_FIELDS = ("family", "id", "candidate", "candidate_release", "arm", "comparator", "role",
              "weighting", "clause", "threshold", "estimate", "bootstrap_se", "lower", "upper",
              "passed_upper_bound", "point_screen_passed", "demonstrated_adverse",
              "exact_zero_same_route", "anchor_estimate_0", "anchor_estimate_1", "anchor_estimate_2")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _gate(lock_sha256):
    return lockgate.verify_outer_gate(lockgate.LOCK_PATH, lock_sha256)


def work_sources(anchor: int) -> dict:
    """The locked panel's logical releases, resolved in the work tree (declaration order)."""
    index = WORK / data.INDEX_RELATIVE
    value = {"D17": {"kind": "historical_map", "map_name": "D17", "anchor": anchor,
                     "index_path": str(index)},
             "DET_SEL4": str(WORK_PRIVATE / "prior_units" / f"a{anchor}_DET_SEL4")}
    for name in host.DECLARED[2:]:
        value[name] = str(WORK_PRIVATE / "units" / f"a{anchor}_{name}")
    return value


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"{path.name} is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def restore_outer(lock_sha256: str, *, fetch=None) -> dict:
    _gate(lock_sha256)
    value = data.index(WORK / data.INDEX_RELATIVE)
    records = []
    for anchor, pin in postlock.ORIGINAL_OBJECTS.items():
        member = data.member_record(value, anchor, "prepared")
        if member["sha256"] != pin["sha256"] or member["archive"]["member_sha256"] != pin["sha256"]:
            raise ValueError(f"anchor {anchor}: S3 pin differs from the pinned index")
        relative = Path(member["archive"]["member_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe archive member path")
        destination = RESTORE_ROOT / relative
        if destination.is_file() and laws.sha256_file(destination) == pin["sha256"]:
            records.append({"anchor": anchor, "action": "already_present_verified", **pin})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        partial = destination.with_name(destination.name + ".partial")
        (fetch or postlock._s3_get)(pin["key"], pin["version_id"], partial)
        if partial.stat().st_size != pin["bytes"] or laws.sha256_file(partial) != pin["sha256"]:
            partial.unlink()
            raise ValueError(f"anchor {anchor}: restored original differs from its SHA-256 pin")
        os.chmod(partial, 0o600)
        os.replace(partial, destination)
        records.append({"anchor": anchor, "action": "restored_verified", **pin})
    receipt = {"schema": "pcrl-pfs-original-restore-v1", "lock_sha256": lock_sha256, "utc": _utc(),
               "restore_root": str(RESTORE_ROOT), "records": records, "all_verified": True}
    _write_new(PRIVATE_OUT / "ORIGINAL_RESTORE.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _outer_rows(anchor: int):
    census = json.loads(CENSUS_PATH.read_text())
    return outer_pool.load_verified_outer(data.index(WORK / data.INDEX_RELATIVE), anchor,
                                          census["anchors"][str(anchor)]["outer_assessment"], RESTORE_ROOT)


def score(lock_sha256: str, anchors=ANCHORS) -> dict:
    lock = _gate(lock_sha256)
    done = {}
    for anchor in anchors:
        out = PRIVATE_OUT / "outer" / f"a{anchor}"
        if (out / "COMPLETE.json").is_file():
            done[str(anchor)] = "already_complete"
            continue
        audit_panel.score_outer(anchor, work_sources(anchor), WORK / data.INDEX_RELATIVE,
                                WORK_PRIVATE / "inner_panels" / f"a{anchor}", lockgate.LOCK_PATH,
                                lock_sha256, out, _gate=lambda path, sha: lockgate.verify_outer_gate(path, sha),
                                _load_outer_rows=lambda a=anchor: _outer_rows(a))
        done[str(anchor)] = str(out)
    variation = {"schema": "pcrl-pfs-decision-variation-outer-v1", "utc": _utc(),
                 "label_free": True, "lock_sha256": lock_sha256, "anchors": {}}
    value = data.index(WORK / data.INDEX_RELATIVE)
    for anchor in ANCHORS:
        rows = _outer_rows(anchor)
        q_ref = data.load_map(value, anchor, "D17")
        per = {}
        for name, source in work_sources(anchor).items():
            q = laws.load(source)(laws.legal_inputs(rows))
            diag, _ = channel.nonalias_diagnostic(q, rows["token_codes"], q_ref, rows["weights"], rows["households"])
            within = {k: v for k, v in diag["within_t32"].items() if k != "per_state"}
            per[name] = {"people": diag["people"], "households": diag["households"],
                         "tv_to_d17": diag["tv_to_d17"], "within_t32": within,
                         "stochastic_weight_fraction": diag["stochastic_weight_fraction"],
                         "law_sha256": laws.law_identity(q)}
        variation["anchors"][str(anchor)] = per
        del rows
    _write_new(RESULTS / "DECISION_VARIATION_OUTER.json", json.dumps(variation, indent=2, sort_keys=True) + "\n")
    return done


def _load_all(lock, lock_sha256):
    scores, breaks, provenance = {}, {}, {}
    for anchor in ANCHORS:
        directory = PRIVATE_OUT / "outer" / f"a{anchor}"
        current, report = sc_assess._load_outer(directory, anchor, lock_sha256)
        scores.update(current)
        provenance[str(anchor)] = {"complete_sha256": laws.sha256_file(directory / "COMPLETE.json"),
                                   "outer_audit_sha256": laws.sha256_file(directory / "OUTER_AUDIT.json")}
        mapping = lock["anchors"][str(anchor)]["logical_to_canonical"]
        confirmed = report.get("outer_alias_confirmed", {})
        broken = report.get("outer_alias_breaks", {})
        for logical, canonical in mapping.items():
            if logical == canonical:
                continue
            if logical in broken:
                if broken[logical]["canonical"] != canonical or logical not in report["releases"]:
                    raise ValueError(f"{logical}: outer alias break record is inconsistent")
                breaks.setdefault(logical, {})[str(anchor)] = broken[logical]
                continue
            if confirmed.get(logical) != canonical:
                raise ValueError(f"{logical}: outer alias to {canonical} was not verified on outer rows")
            for role in sc_assess.ROLES:
                scores[(logical, anchor, role)] = scores[(canonical, anchor, role)]
    return scores, breaks, provenance


def labels(primary_rows) -> dict:
    by_slot = {}
    for row in primary_rows:
        by_slot.setdefault(row["candidate"], []).append(row)
    passed = {slot: all(r["passed_upper_bound"] for r in rows) for slot, rows in by_slot.items()}
    d = {(r["role"], r["weighting"]): r["estimate"] for r in by_slot["D"]}
    target = "attack:AB/SEX"
    lead = (all(d[(target, w)] <= -.002 for w in ("U", "PWGTP")) and
            all(d[("utility:A/same_residence", w)] <= .001 for w in ("U", "PWGTP")) and
            all(v <= .001 for (role, w), v in d.items() if role not in (target, "utility:A/same_residence")))
    return {"ARM_D_MEETS_CRITERIA": passed["D"], "ARM_R_MEETS_CRITERIA": passed["R"],
            "RANDOMIZATION_ADDS": passed["R_vs_D"], "LEAD_REPRODUCED": bool(lead),
            "clauses_passed": {slot: sum(r["passed_upper_bound"] for r in rows) for slot, rows in by_slot.items()},
            "clauses_total": {slot: len(rows) for slot, rows in by_slot.items()}}


def assess(lock_sha256: str, *, n_boot: int | None = None) -> dict:
    lock = _gate(lock_sha256)
    scores, breaks, provenance = _load_all(lock, lock_sha256)
    boot = lock["bootstrap"]
    n_boot = n_boot or boot["draws"]
    results, rows_out = {}, []
    for name in FAMILIES:
        endpoints = lock[name]["endpoints"]
        clean = [{k: v for k, v in e.items() if k != "exact_zero_same_route"} for e in endpoints]
        result = paired.evaluate_family(clean, scores, n_boot=n_boot, seed=boot["seed"], alpha=boot["alpha"])
        flags = {e["id"]: e["exact_zero_same_route"] for e in endpoints}
        for row in result["rows"]:
            row["exact_zero_same_route"] = flags[row["id"]]
            rows_out.append(row)
        results[name] = result
    supplement = []
    for name in FAMILIES:
        for e in lock[name]["endpoints"]:
            for other in e["comparator_names"]:
                if other != e["comparator"] and other in breaks:
                    supplement.append({**{k: v for k, v in e.items() if k != "exact_zero_same_route"},
                                       "id": f"outer_alias_supplement|{e['id']}|{other}",
                                       "family": "outer_alias_supplement", "comparator": other,
                                       "comparator_names": [other],
                                       "plus": e["candidate_release"] if e["clause"] in ("task", "capability") else other,
                                       "minus": other if e["clause"] in ("task", "capability") else e["candidate_release"]})
    if supplement:
        result = paired.evaluate_family(supplement, scores, n_boot=n_boot, seed=boot["seed"], alpha=boot["alpha"])
        for row in result["rows"]:
            row["exact_zero_same_route"] = False
            rows_out.append(row)
        results["outer_alias_supplement"] = result
    decision = labels(results["family_manifest"]["rows"])
    met = [k for k in ("ARM_D_MEETS_CRITERIA", "ARM_R_MEETS_CRITERIA", "RANDOMIZATION_ADDS") if decision[k]]
    report = {"schema": "pcrl-pfs-inference-v1", "study": lockgate.STUDY, "assessment_year": 2018,
              "development_only": True, "registered_statement": lockgate.STATEMENT,
              "status": ("MEETS_PREDECLARED_CRITERIA (" + ", ".join(met) + "); exploratory development evidence"
                         if met else "NOT_ESTABLISHED"),
              "selection_lock_sha256": lock_sha256, "no_outer_fit_or_selection": True,
              "bootstrap_draws": n_boot, "bootstrap_seed": boot["seed"], "decision_labels": decision,
              "decision_rules": lock["decision_rules"],
              "families": {name: {k: v for k, v in res.items() if k != "rows"} for name, res in results.items()},
              "outer_alias_breaks": breaks, "outer_archives": provenance,
              "positive_controls": lock["positive_controls"], "audit_uninformative": lock["audit_uninformative"],
              "scope": ("conditional on the fitted, selected objects; not the adaptive research history "
                        "nor the ACS survey design")}
    _write_new(RESULTS / "INFERENCE.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write_new(RESULTS / "ENDPOINT_TABLE.json",
               json.dumps({"selection_lock_sha256": lock_sha256, "rows": rows_out},
                          indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [",".join(CSV_FIELDS)]
    for row in rows_out:
        values = {**row, **{f"anchor_estimate_{i}": v for i, v in enumerate(row["anchor_estimates"])}}
        lines.append(",".join(sc_assess._csv(values.get(field)) for field in CSV_FIELDS))
    _write_new(RESULTS / "FULL_RESULTS.csv", "\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    for name in ("restore-outer", "score", "assess"):
        item = sub.add_parser(name); item.add_argument("--lock-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        value = postlock.prepare(WORK)
    elif args.action == "restore-outer":
        value = {k: v for k, v in restore_outer(args.lock_sha256).items() if k != "records"}
    elif args.action == "score":
        value = score(args.lock_sha256)
    else:
        report = assess(args.lock_sha256)
        value = {"status": report["status"], "labels": report["decision_labels"]}
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

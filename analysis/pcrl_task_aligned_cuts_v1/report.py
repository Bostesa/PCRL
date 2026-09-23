"""Render aggregate-only crossed-arm 2018 development status from receipts.

This module never opens person-level contributions or outer assessment labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/pcrl_task_aligned_cuts_v1"
ROLES = ("utility:A/same_residence", "attack:A/SEX", "attack:A/RAC1P",
         "attack:AB/SEX", "attack:AB/RAC1P")
WEIGHTINGS = ("U", "PWGTP")
FIELDS = ("anchor", "configuration", "arm", "budget", "role", "weighting",
          "fit_status", "audit_status", "registered_feasible", "fallback",
          "release_sha256", "alias_of", "n_people", "n_households",
          "release_loss", "H_loss", "H_minus_release", "audit_selected_route")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(unit: dict, queue_sha: str) -> bool:
    directory = ROOT / unit["output_dir"]
    marker = directory / "COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = json.loads(marker.read_text())
    if receipt.get("unit_id") != unit["id"] or receipt.get("queue_sha256") != queue_sha:
        return False
    files = receipt.get("artifacts", {})
    return bool(files) and all((directory / rel).is_file() and
                               sha(directory / rel) == digest
                               for rel, digest in files.items())


def rows() -> tuple[list[dict], dict]:
    queue_file = STUDY / "RUN_QUEUE.json"
    queue = json.loads(queue_file.read_text())
    lock = json.loads((STUDY / "PROTOCOL_LOCK.json").read_text())
    queue_sha = sha(queue_file)
    if queue_sha != lock["run_queue_sha256"] or sha(STUDY / "PROTOCOL.md") != lock["protocol_sha256"]:
        raise ValueError("Registered queue/protocol changed")
    units = {unit["id"]: unit for unit in queue["units"]}
    state_path = STUDY / "private/QUEUE_STATUS.json"
    unit_states = json.loads(state_path.read_text()).get("units", {}) if state_path.exists() else {}
    candidate_units = [unit for unit in queue["units"]
                       if re.fullmatch(r"a[012]_u[01]p[01]_[a-z0-9]+", unit["id"])]
    if len(candidate_units) != 36:
        raise ValueError("Nominal four-arm three-budget three-anchor panel changed")
    aliases: dict[tuple[int, str], str] = {}
    output = []
    counts = {"nominal_configurations": 36, "fit_complete": 0,
              "registered_feasible": 0, "infeasible_fallback": 0,
              "audit_complete": 0, "not_triggered": 0,
              "audit_running": 0, "audit_failed": 0, "exact_aliases": 0}
    for unit in candidate_units:
        name = unit["id"]
        anchor = int(name[1])
        arm = name.split("_")[1].upper()
        budget = name.split("_")[2]
        fit_ok = complete(unit, queue_sha)
        audit_unit = units[name + "_audit"]
        audit_ok = complete(audit_unit, queue_sha)
        fit_file = ROOT / unit["output_dir"] / ("FIT_P0_ARM.json" if arm.endswith("P0") else "FIT_P1_ARM.json")
        audit_file = ROOT / audit_unit["output_dir"] / "INNER_PANEL.json"
        fit = json.loads(fit_file.read_text()) if fit_ok else None
        audit = json.loads(audit_file.read_text()) if audit_ok else None
        if fit_ok:
            counts["fit_complete"] += 1
            feasible = fit.get("feasible", fit.get("registered_feasible"))
            if feasible:
                counts["registered_feasible"] += 1
            elif fit.get("status") == "DESCRIPTIVE_PHASE_I_FALLBACK":
                counts["infeasible_fallback"] += 1
        else:
            counts["not_triggered"] += 1
            feasible = None
        if audit_ok:
            counts["audit_complete"] += 1
        elif fit_ok:
            audit_state = unit_states.get(name + "_audit", {}).get("status")
            if audit_state == "running":
                counts["audit_running"] += 1
            elif audit_state in ("incident", "scheduler_error", "deadline_stop"):
                counts["audit_failed"] += 1
        release_sha = fit.get("channel_sha256") if fit else None
        alias = None
        if release_sha:
            key = anchor, release_sha
            if key in aliases:
                alias = aliases[key]
                counts["exact_aliases"] += 1
            else:
                aliases[key] = name
        for role in ROLES:
            for weighting in WEIGHTINGS:
                item = audit["roles"].get(role) if audit and "roles" in audit else None
                output.append({
                    "anchor": anchor, "configuration": name, "arm": arm,
                    "budget": budget, "role": role, "weighting": weighting,
                    "fit_status": fit.get("status") if fit else "not_triggered",
                    "audit_status": "complete" if item else unit_states.get(name + "_audit", {}).get("status", "not_triggered"),
                    "registered_feasible": feasible,
                    "fallback": bool(fit and fit.get("status") == "DESCRIPTIVE_PHASE_I_FALLBACK"),
                    "release_sha256": release_sha, "alias_of": alias,
                    "n_people": item.get("n_scored_people") if item else None,
                    "n_households": item.get("n_scored_households") if item else None,
                    "release_loss": item["candidate"][weighting] if item else None,
                    "H_loss": item["H"][weighting] if item else None,
                    "H_minus_release": item["H_minus_candidate"][weighting] if item else None,
                    "audit_selected_route": item["candidate_selection"]["selected"] if item else None,
                })
    if len(output) != 360:
        raise AssertionError("endpoint enumeration changed")
    return output, counts


def main() -> None:
    output, counts = rows()
    csv_path = STUDY / "CORE_FACTORIAL.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    status = {"schema": "pcrl-core-status-v1", "scope": "2018 development aggregate only",
              "protocol_lock_sha256": sha(STUDY / "PROTOCOL_LOCK.json"),
              "queue_sha256": sha(STUDY / "RUN_QUEUE.json"),
              "core_factorial_sha256": sha(csv_path), "counts": counts}
    (STUDY / "RUN_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()

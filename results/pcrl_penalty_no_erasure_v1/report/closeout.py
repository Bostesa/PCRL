"""Write HANDOFF.json and the final RUN_STATUS.json from the committed aggregate files.

Run from the worktree root after INDEPENDENT_VERIFICATION.json exists:
    python3 results/pcrl_penalty_no_erasure_v1/report/closeout.py
"""
import csv
import json
import subprocess
import time
from pathlib import Path

OUT = Path("results/pcrl_penalty_no_erasure_v1")


def git(*a):
    return subprocess.check_output(["git", *a], text=True).strip()


def primary():
    rows = {}
    with open(OUT / "PRIMARY_ENDPOINTS.csv") as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = {k: r[k] for k in ("stat", "point", "lower", "upper", "decision", "alias_of")}
    return rows


def secondary():
    keep = {}
    with open(OUT / "SECONDARY_ENDPOINTS.csv") as f:
        for r in csv.DictReader(f):
            keep[r["id"]] = {k: r.get(k, "") for k in ("stat", "point", "lower", "upper", "decision")}
    return keep


def main():
    ver = json.loads((OUT / "INDEPENDENT_VERIFICATION.json").read_text())
    backup = json.loads((OUT / "MODEL_AND_PRIVATE_BACKUP_INDEX.json").read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    locks = {"protocol": git("rev-parse", "f1d1b76"), "selection": git("rev-parse", "b0b9256")}
    verification = {k: ver[k] for k in ("verdict", "summary", "counts") if k in ver}
    handoff = {
        "schema": "pnx-handoff-v1",
        "branch": "research/pcrl-penalty-no-erasure-v1",
        "parent": git("rev-parse", "568f970"),
        "locks": locks,
        "status": "EXPLORATORY DEVELOPMENT evidence on reused Adult rows; not fresh confirmation",
        "model_label": "EXPERIMENTAL_NO_ADVANTAGE",
        "outcome": {
            "claim_A_PN_vs_LN": "NOT_ESTABLISHED (P02 income-recipient local guard fails: +0.010 [0.001, 0.019] vs 0.01; seed-0 PN is NO_FEASIBLE_NOMINEE)",
            "claim_B_PN_vs_Cstar": "NOT_ESTABLISHED (C* = LN on every seed, so identical to claim A)",
            "clauses_passed": "8 of 9 per claim; a missed conjunction is not a success",
            "selection": "LN b0.1 NOMINEE on all seeds; PN b0.1 NOMINEE on seeds 1, 2; seed 0 PN v2 inner recovery 0.827 > LN 0.815 + 0.01",
            "erasure_on_off": "PN - JP beta-matched: income acc +0.043 [0.034, 0.052] ABOVE, occupation acc +0.017 [0.010, 0.024] ABOVE; coalition SEX AUC +0.003 [-0.004, 0.010] NOT_RESOLVED; coalition OLS R^2 0 -> 0.23 (fitting rows)",
            "critic_gap": "fresh same-architecture critics beat online critics on identical frozen views in 45/45 cells, mean +0.047 nats; at beta >= 1 online CE 0.60-0.62 vs prior 0.63",
            "descriptive": {
                "PN_coalition_vs_U": "-0.033",
                "PN_coalition_vs_F0": "-0.034",
                "PN_coalition_vs_E": "-0.021",
                "PN_coalition_vs_JP_reference": "+0.067 (frozen per-seed JP reference: b0.1 seed 0, b1 seeds 1-2; fails utility gates); +0.023 at matched b0.1",
                "PN_coalition_vs_F": "+0.123 (F fails occupation utility gate)",
                "decomposition_vs_LN": "+0.029 = worse-local-view +0.019 + synergy +0.010",
            },
            "guarantees_lost": "LEACE fitted-row linear guardedness of each release, the primary view and the coalition; evidence is attacker-based only",
            "bottleneck": "income-recipient local leakage under joint training; online-critic tracking at beta >= 1",
        },
        "primary": primary(),
        "secondary": secondary(),
        "verification": verification,
        "backup": {k: backup[k] for k in ("local", "drive_copy", "files", "SHA256SUMS_sha256", "verification") if k in backup},
        "spend": "$0 cloud; about 0.7 CPU-h; about 1.5 h elapsed; peak RSS 0.94 GB",
        "incomplete_units": [],
        "deploy": "see QUICKSTART.md (tested bitwise against stored releases)",
    }
    (OUT / "HANDOFF.json").write_text(json.dumps(handoff, indent=1) + "\n")
    status = {
        "schema": "pnx-run-status-v1",
        "started_at": "2026-10-03T23:34:59Z",
        "stage": "COMPLETE",
        "updated_at": now,
        "locks": locks,
        "claims": {"A": "NOT_ESTABLISHED", "B": "NOT_ESTABLISHED"},
        "model_label": "EXPERIMENTAL_NO_ADVANTAGE",
        "units": "UNIT_MANIFEST.csv 122/122 rows complete: 18 new fits, 18 inner audits, 39 outer scorings, 18 critic-gap units, 1 parity receipt, 1 controls unit, and 27 rows of hash-checked reused or alias units (117 alias records pinned in LOCK.json)",
        "verification": "INDEPENDENT_VERIFICATION.json: {PASS} PASS / {FAIL} FAIL / {WARN} WARN / {INFO} INFO".format(
            **ver["summary"]["counts"]) + f" ({ver['summary']['n_checks']} checks); claims A/B "
            f"{ver['claims']['A']['decision']}/{ver['claims']['B']['decision']}",
        "resources": "$0 cloud; about 0.7 of 12 CPU-h; about 1.5 of 8 h elapsed; at most 2 workers; peak RSS 0.94 GB",
        "backup": "561/561 private files on <drive>/private_pnx_v1_20261003, uncached read-back verified; PN, LN and E restored exactly",
        "owned_processes": "none running",
        "next_safe_command": "none; study closed. Deployment: see QUICKSTART.md",
    }
    (OUT / "RUN_STATUS.json").write_text(json.dumps(status, indent=1) + "\n")
    print("wrote HANDOFF.json and RUN_STATUS.json")


if __name__ == "__main__":
    main()

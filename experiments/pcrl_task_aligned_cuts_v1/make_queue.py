"""Deterministically register, then hash-lock, the 2018 development job graph.

This writes metadata only. It never opens private arrays or starts a fit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/pcrl_task_aligned_cuts_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"registered file already exists: {path}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def unit(ident: str, command: str, anchor: int, *, arm: str | None = None,
         budget: str | None = None, deps: list[str] | None = None,
         tier: str = "A") -> dict:
    out = f"results/pcrl_task_aligned_cuts_v1/private/run/{ident}"
    argv = [sys.executable, "-m", "experiments.pcrl_task_aligned_cuts_v1.pipeline",
            command, "--anchor", str(anchor), "--index",
            "results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json",
            "--output-dir", out]
    if arm is not None:
        argv += ["--arm", arm]
    if budget is not None:
        argv += ["--budget", budget]
    return {"id": ident, "tier": tier, "dependencies": deps or [],
            "argv": argv, "output_dir": out, "complete_marker": "COMPLETE.json"}


def queue() -> dict:
    units: list[dict] = []
    for anchor in (0, 1, 2):
        tier = "A" if anchor == 0 else "B"
        admit = f"a{anchor}_admit"
        decoder = f"a{anchor}_u1_decoder"
        bank = f"a{anchor}_reference_bank"
        units += [unit(admit, "admit", anchor, tier=tier),
                  unit(decoder, "fit-u1", anchor, deps=[admit], tier=tier),
                  unit(bank, "fit-bank", anchor, deps=[admit], tier=tier)]
        # The center point is first in the registered order, so it can be
        # fitted/audited before the factorial or deterministic control finishes.
        arms_budgets = [("U1P1", "0"),
                        *[(arm, budget) for arm in ("U0P0", "U1P0", "U0P1", "U1P1")
                          for budget in (("0.005", "0.01", "0.02") if arm.endswith("P0")
                                         else ("-0.002", "+0.002", "0"))
                          if (arm, budget) != ("U1P1", "0")]]
        for arm, budget in arms_budgets:
            budget_tag = {"-0.002":"m002", "0":"z000", "+0.002":"p002",
                          "0.005":"b005", "0.01":"b010", "0.02":"b020"}[budget]
            ident = f"a{anchor}_{arm.lower()}_{budget_tag}"
            deps = [admit, bank] + ([decoder] if arm.startswith("U1") else [])
            units.append(unit(ident, "fit-arm", anchor, arm=arm, budget=budget,
                              deps=deps, tier=tier))
            units.append(unit(ident + "_audit", "audit-inner", anchor, arm=arm,
                              budget=budget, deps=[ident], tier=tier))
    return {"schema": "pcrl-task-aligned-queue-v1", "created_utc":
            dt.datetime.now(dt.timezone.utc).isoformat(),
            "frozen_before_comparative_outcomes": True,
            "source_baseline_commit": "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb",
            "protocol_sha256": sha(STUDY / "PROTOCOL.md"),
            "nominal_core_configurations": 36,
            "core_audit_units": 36,
            "units": units}


def main() -> None:
    run_queue = STUDY / "RUN_QUEUE.json"
    lock = STUDY / "PROTOCOL_LOCK.json"
    write_new(run_queue, queue())
    write_new(lock, {"schema": "pcrl-task-aligned-protocol-lock-v1",
                     "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "protocol_sha256": sha(STUDY / "PROTOCOL.md"),
                     "run_queue_sha256": sha(run_queue),
                     "status": "pre-candidate-outcome"})
    print(json.dumps({"queue_sha256": sha(run_queue),
                      "protocol_sha256": sha(STUDY / "PROTOCOL.md"),
                      "units": len(json.loads(run_queue.read_text())["units"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()

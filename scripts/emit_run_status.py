#!/usr/bin/env python3
"""Emit a compact plain-text STATUS file for a LAFTR hard-R² AWS run.

The point of this file is durability: even if the result blobs
(per_seed_results.json, checkpoints) are later lost — e.g. to an S3
lifecycle expiration — the headline numbers survive as a tiny text file
that is cheap to archive and trivial to read months later.

Reads, for each dataset that produced output:
    results/laftr_hard_r2_<ds>_LAFTR_HARD_R2/summary.json
    results/laftr_hard_r2_<ds>_LAFTR_HARD_R2/per_seed_results.json

Writes a plain-text status with: instance id, completion time, per-dataset
strict-pass count, mean R² (all cells), mean R² on passing cells, and task
accuracy. Datasets with no output are reported as MISSING rather than
crashing the run.

Usage:
    python scripts/emit_run_status.py --instance-id i-0abc --out STATUS.txt
    python scripts/emit_run_status.py --instance-id i-0abc   # prints to stdout
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["adult", "hmda", "diabetes"]
TAU = 0.05


def _dataset_line(root: Path, ds: str) -> str:
    d = root / "results" / f"laftr_hard_r2_{ds}_LAFTR_HARD_R2"
    per_seed_path = d / "per_seed_results.json"
    summary_path = d / "summary.json"
    if not per_seed_path.exists():
        return f"  {ds:10s} MISSING (no {per_seed_path.name})"

    per_seed = json.loads(per_seed_path.read_text())["per_seed"]
    cells = [c for s in per_seed for c in s["attribute_results"]]
    n_total = len(cells)
    passing = [c for c in cells if c["linear_r2"] < TAU]
    n_pass = len(passing)
    mean_r2_all = sum(c["linear_r2"] for c in cells) / n_total if n_total else float("nan")
    mean_r2_pass = (sum(c["linear_r2"] for c in passing) / n_pass) if n_pass else None

    status = "?"
    task_acc_str = "?"
    if summary_path.exists():
        summ = json.loads(summary_path.read_text())
        status = summ.get("STATUS", "?")
        accs = summ.get("task_acc_mean", {})
        if accs:
            task_acc_str = ", ".join(f"{k}={v * 100:.1f}%" for k, v in accs.items())

    r2_pass_str = f"{mean_r2_pass:.4f}" if mean_r2_pass is not None else "—"
    return (
        f"  {ds:10s} strict-pass {n_pass}/{n_total}  "
        f"meanR²(all)={mean_r2_all:.4f}  meanR²(pass)={r2_pass_str}  "
        f"health={status}  task_acc[{task_acc_str}]"
    )


def build_status(root: Path, instance_id: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "LAFTR HARD-R² PILOT — RUN STATUS",
        "=" * 60,
        f"instance_id:     {instance_id}",
        f"completed_utc:   {now}",
        f"strict_threshold: linear R²(h, A) < {TAU}",
        "",
    ]
    for ds in DATASETS:
        lines.append(_dataset_line(root, ds))
    lines.append("")
    lines.append("This file is the durable headline. Full per-cell data (if retained)")
    lines.append("is in results/laftr_hard_r2_<ds>_LAFTR_HARD_R2/per_seed_results.json.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-id", default="unknown")
    p.add_argument("--root", default=str(DEFAULT_REPO_ROOT))
    p.add_argument("--out", default=None, help="Output path; prints to stdout if omitted")
    args = p.parse_args()

    text = build_status(Path(args.root).resolve(), args.instance_id)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

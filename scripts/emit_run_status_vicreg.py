#!/usr/bin/env python3
"""Emit a compact plain-text STATUS file for the erase-layer VICReg sweep.

Mirrors scripts/emit_run_status.py from the LAFTR-hard-R² branch
(c23e113), adapted for the vicreg sweep's result paths and metrics. The
point is durability: even if the result blobs are later lost — e.g. to
an S3 lifecycle expiration — the headline numbers survive as a tiny
text file that is cheap to archive and trivial to read months later.

The vicreg sweep's specific headline question is whether scaling
``lambda_vicreg`` 1.0 → 5.0 pushes per_dim_std above the 0.5
cleanly-compliant threshold while preserving the strict R² ≤ 0.05
certificate. So this STATUS includes per-dataset per_dim_std_mean and
effective_rank_mean alongside the strict-pass count + mean R².

Reads, for each dataset that produced output:
    results/v2_<ds>_ERASE_VICREG5/summary.json
    results/v2_<ds>_ERASE_VICREG5/per_seed_results.json

Writes a plain-text status. Datasets with no output are reported as
MISSING rather than crashing the run.

Usage:
    python scripts/emit_run_status_vicreg.py --instance-id i-0abc --out STATUS.txt
    python scripts/emit_run_status_vicreg.py --instance-id i-0abc   # stdout
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["adult", "hmda", "diabetes"]
TAU = 0.05
PER_DIM_STD_THRESHOLD = 0.5
EFF_RANK_THRESHOLD = 2.0


def _dataset_line(root: Path, ds: str) -> str:
    d = root / "results" / f"v2_{ds}_ERASE_VICREG5"
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

    # Per-purpose health aggregated across all (purpose, seed) pairs. This is
    # the metric the vicreg sweep is trying to move.
    std_vals: list[float] = []
    rank_vals: list[float] = []
    clean_count = 0
    for s in per_seed:
        h = s.get("per_purpose_health", {})
        for ph in h.values():
            std_vals.append(ph.get("per_dim_std_mean", 0.0))
            rank_vals.append(ph.get("effective_rank", 0.0))
        for c in s["attribute_results"]:
            ph = h.get(c["purpose"], {})
            if (c["linear_r2"] < TAU
                    and ph.get("per_dim_std_mean", 0.0) >= PER_DIM_STD_THRESHOLD
                    and ph.get("effective_rank", 0.0) >= EFF_RANK_THRESHOLD):
                clean_count += 1
    std_mean = sum(std_vals) / len(std_vals) if std_vals else float("nan")
    rank_mean = sum(rank_vals) / len(rank_vals) if rank_vals else float("nan")

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
        f"  {ds:10s} strict-pass {n_pass}/{n_total}  clean {clean_count}/{n_total}  "
        f"meanR²(all)={mean_r2_all:.4f}  meanR²(pass)={r2_pass_str}  "
        f"per_dim_std={std_mean:.3f}  eff_rank={rank_mean:.2f}  "
        f"health={status}  task_acc[{task_acc_str}]"
    )


def build_status(root: Path, instance_id: str) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "ERASE-LAYER VICREG SWEEP — RUN STATUS",
        "=" * 60,
        f"instance_id:     {instance_id}",
        f"completed_utc:   {now}",
        f"out_tag:         _ERASE_VICREG5  (--lambda-vicreg=5.0 vs original pilot's 1.0)",
        f"strict_threshold: linear R²(h, A) < {TAU}",
        f"clean_threshold:  R² < {TAU} AND per_dim_std_mean ≥ {PER_DIM_STD_THRESHOLD} AND eff_rank ≥ {EFF_RANK_THRESHOLD}",
        "",
    ]
    for ds in DATASETS:
        lines.append(_dataset_line(root, ds))
    lines.append("")
    lines.append("This file is the durable headline. Full per-cell data (if retained)")
    lines.append("is in results/v2_<ds>_ERASE_VICREG5/per_seed_results.json.")
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

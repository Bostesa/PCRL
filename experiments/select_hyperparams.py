#!/usr/bin/env python3
"""Held-out hyperparameter selection for the FAccT resubmission Ablation 3.

Reads every candidate config's VAL-split results
(``results/v2_<dataset><tag>/per_seed_results.json`` produced by
``run_v2_dataset.py --eval-split val``), applies the selection rule
registered in ``results/rebuttal/ablations_facct/PREDICTIONS.md`` BEFORE any
candidate was trained, and writes ``selection.json`` naming the winner.

Selection rule (deterministic; registered 2026-07-24):
    1. Maximise total strict-pass count (linear R² < 0.05) on the val grid,
       summed over seeds.
    2. Tie-break 1: maximise clean-compliance count on val (strict pass AND
       per_dim_std_mean ≥ 0.5 AND eff_rank ≥ 2.0 for the cell's purpose).
    3. Tie-break 2: maximise mean task accuracy on val (mean over tasks and
       seeds).
    4. Tie-break 3 (simplicity): lower LoRA rank, then fewer warmup epochs,
       then lower λ_min.

The script refuses candidates whose results were computed on the test split —
that is the whole point of the protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TAU = 0.05
PER_DIM_STD_MIN = 0.5
EFF_RANK_MIN = 2.0


def score_candidate(result_path: Path) -> dict:
    data = json.loads(result_path.read_text())
    per_seed = data["per_seed"]
    summary = data.get("summary", {})
    run_config = summary.get("run_config", {})

    for s in per_seed:
        if s.get("eval_split") != "val":
            raise SystemExit(
                f"PROTOCOL VIOLATION: {result_path} seed={s.get('seed')} was "
                f"evaluated on split {s.get('eval_split')!r}, not 'val'. "
                "Selection must only ever see val-grid numbers."
            )

    strict = 0
    clean = 0
    total = 0
    for s in per_seed:
        health = s.get("per_purpose_health", {})
        for c in s["attribute_results"]:
            total += 1
            if c["linear_r2"] < TAU:
                strict += 1
                ph = health.get(c["purpose"], {})
                if (ph.get("per_dim_std_mean", 0.0) >= PER_DIM_STD_MIN
                        and ph.get("effective_rank", 0.0) >= EFF_RANK_MIN):
                    clean += 1

    accs = [v for s in per_seed for v in s["task_accuracies"].values()]
    mean_acc = sum(accs) / len(accs) if accs else float("nan")

    lora_rank = run_config.get("lora_rank")
    warmup = run_config.get("warmup_epochs")
    lambda_min = run_config.get("lambda_min")

    return {
        "path": str(result_path),
        "tag": result_path.parent.name,
        "n_seeds": len(per_seed),
        "strict_pass": strict,
        "clean_pass": clean,
        "total_cells": total,
        "mean_task_acc": mean_acc,
        "lora_rank": lora_rank,
        "warmup_epochs": warmup,
        "lambda_min": lambda_min,
        "run_config": run_config,
    }


def sort_key(c: dict):
    # Higher strict, then higher clean, then higher acc, then SIMPLER:
    # lower rank, fewer warmup epochs, lower lambda_min. None sorts last.
    def low(v):
        return float("inf") if v is None else float(v)
    return (
        -c["strict_pass"],
        -c["clean_pass"],
        -c["mean_task_acc"],
        low(c["lora_rank"]),
        low(c["warmup_epochs"]),
        low(c["lambda_min"]),
        c["tag"],  # final deterministic fallback
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pattern", default=None,
                    help="Glob for candidate result dirs "
                         "(default: results/v2_<dataset>_ABL3_*)")
    ap.add_argument("--out", default=None,
                    help="Output path (default: "
                         "results/rebuttal/ablations_facct/selection_<dataset>.json)")
    args = ap.parse_args()

    pattern = args.pattern or str(ROOT / "results" / f"v2_{args.dataset}_ABL3_*")
    dirs = sorted(glob.glob(pattern))
    candidates = []
    for d in dirs:
        p = Path(d) / "per_seed_results.json"
        if not p.exists():
            print(f"WARN: skipping {d} (no per_seed_results.json)", file=sys.stderr)
            continue
        candidates.append(score_candidate(p))

    if not candidates:
        raise SystemExit(f"no candidates matched {pattern}")

    ranked = sorted(candidates, key=sort_key)
    winner = ranked[0]

    out_path = Path(args.out) if args.out else (
        ROOT / "results" / "rebuttal" / "ablations_facct" / f"selection_{args.dataset}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "rule": "strict_pass > clean_pass > mean_task_acc > lower rank > fewer warmup > lower lambda_min",
        "selected": winner,
        "ranked": ranked,
    }, indent=2))

    print(f"dataset={args.dataset}  candidates={len(ranked)}")
    for c in ranked:
        marker = "→ SELECTED" if c is winner else ""
        print(f"  {c['tag']:<40} strict {c['strict_pass']}/{c['total_cells']}  "
              f"clean {c['clean_pass']}  acc {c['mean_task_acc']:.4f}  "
              f"rank={c['lora_rank']} wu={c['warmup_epochs']} lm={c['lambda_min']} {marker}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

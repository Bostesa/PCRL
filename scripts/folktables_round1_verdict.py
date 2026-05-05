#!/usr/bin/env python3
"""Folktables Round 1 verdict generator.

Reads ``results/v2_folktables_ROUND1/per_seed_results.json`` and the
checkpoint dir. Runs ``generate_report`` on ``final.pt`` for each seed,
aggregates into a 24-cell (3 seeds × 8 pairs) matrix, and writes the
verdict to ``results/V2_FOLKTABLES_ROUND1_VERDICT.md``.

Folktables uses BINARY race (White vs non-White) by construction, so the
"Framework D dominant-axis" convention is identical to the standard
audit; no separate dominant-axis pass is needed.

Verdict thresholds:
  GREEN  ≥ 18/24 strict (R² < 0.05)
  YELLOW 15-17/24 strict
  RED    <15/24 strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.folktables import FolktablesACSDataset, get_folktables_purposes
from pcrl.evaluation.certificates import generate_report
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.independence.vclub import VCLUB

LORA_RANK = 8
LORA_ALPHA = 16.0
REPR_DIM = 64
HIDDEN = [128, 128]
DROPOUT = 0.3
SEEDS = [0, 1, 2]
OUT_TAG = "_ROUND1"
DATASET = "folktables"
STRICT_R2 = 0.05
DELTA_THR = 0.02


def build_encoder(input_dim: int, n_purposes: int):
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN, repr_dim=REPR_DIM, dropout=DROPOUT,
    )
    return PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=n_purposes,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0,
    )


def evaluate_seed(seed: int, train_ds, test_ds, purposes, registry, device: str):
    ckpt_dir = ROOT / "checkpoints" / f"v2_{DATASET}{OUT_TAG}_s{seed}"
    final_pt = ckpt_dir / "final.pt"
    if not final_pt.exists():
        raise FileNotFoundError(f"missing checkpoint: {final_pt}")

    encoder = build_encoder(input_dim=train_ds.info.num_features, n_purposes=len(purposes))
    ckpt = torch.load(final_pt, map_location=device, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    encoder.eval().to(device)

    train_loader = DataLoader(
        train_ds, batch_size=512, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )

    reports = generate_report(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )

    rows = []
    for r in reports:
        delta = float(r.empirical_best_acc - r.majority_proportion)
        rows.append({
            "seed": seed,
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "linear_r2": float(r.linear_r2),
            "empirical_best_acc": float(r.empirical_best_acc),
            "majority": float(r.majority_proportion),
            "delta": delta,
            "strict_pass": bool(r.linear_r2 < STRICT_R2),
            "combined_pass": bool(r.linear_r2 < STRICT_R2 and delta < DELTA_THR),
        })
    return rows


def aggregate(per_seed_rows: list[dict]) -> dict:
    n = len(per_seed_rows)
    strict_pass = sum(1 for r in per_seed_rows if r["strict_pass"])
    combined_pass = sum(1 for r in per_seed_rows if r["combined_pass"])
    mean_r2 = float(np.mean([r["linear_r2"] for r in per_seed_rows]))
    mean_delta = float(np.mean([r["delta"] for r in per_seed_rows]))

    purposes = sorted({r["purpose"] for r in per_seed_rows})
    per_purpose: dict[str, dict] = {}
    for p in purposes:
        cells = [r for r in per_seed_rows if r["purpose"] == p]
        per_purpose[p] = {
            "n": len(cells),
            "strict_pass": sum(1 for r in cells if r["strict_pass"]),
            "combined_pass": sum(1 for r in cells if r["combined_pass"]),
            "mean_r2": float(np.mean([r["linear_r2"] for r in cells])),
        }

    if strict_pass >= 18:
        verdict = "GREEN"
    elif strict_pass >= 15:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    return {
        "n_cells": n,
        "strict_pass": strict_pass,
        "combined_pass": combined_pass,
        "mean_r2": mean_r2,
        "mean_delta": mean_delta,
        "verdict": verdict,
        "per_purpose": per_purpose,
    }


def fmt_table(rows: list[dict]) -> str:
    out = ["| seed | purpose | attribute | R² | delta | strict | combined |",
           "|------|---------|-----------|------|--------|--------|----------|"]
    for r in sorted(rows, key=lambda x: (x["seed"], x["purpose"], x["attribute"])):
        out.append(
            f"| {r['seed']} | {r['purpose']} | {r['attribute']} | "
            f"{r['linear_r2']:.4f} | {r['delta']:+.4f} | "
            f"{'✓' if r['strict_pass'] else '✗'} | "
            f"{'✓' if r['combined_pass'] else '✗'} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-tag", default=OUT_TAG)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Folktables Round 1 verdict on device={device}")

    purposes = get_folktables_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = FolktablesACSDataset(
        purposes=purposes, root="data", split="train", download=True,
    )
    test_ds = FolktablesACSDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )

    all_rows: list[dict] = []
    for seed in SEEDS:
        print(f"  seed={seed}")
        rows = evaluate_seed(seed, train_ds, test_ds, purposes, registry, device)
        all_rows.extend(rows)

    agg = aggregate(all_rows)

    summary_lines = [
        "# V2 Folktables Round 1 — Verdict",
        "",
        f"Dataset: California 2018 1-Year ACS PUMS (N_train={len(train_ds)}, "
        f"N_test={len(test_ds)}, D={train_ds.info.num_features})",
        f"Selection: final.pt for seeds {SEEDS}",
        f"Convention: binary RAC1P (White vs non-White) and binary SEX (FFB ICLR 2024). "
        "Folktables is binary by construction so Framework D dominant-axis is identical.",
        "",
        "## Headline",
        "",
        f"- **Verdict: {agg['verdict']}**",
        f"- Strict pass (R² < {STRICT_R2}): **{agg['strict_pass']}/{agg['n_cells']}**",
        f"- Combined pass (R² < {STRICT_R2} AND delta < {DELTA_THR}): "
        f"**{agg['combined_pass']}/{agg['n_cells']}**",
        f"- Mean R²: {agg['mean_r2']:.4f}",
        f"- Mean post-hoc delta: {agg['mean_delta']:+.4f}",
        "",
        "## Per-purpose breakdown",
        "",
        "| purpose | n_cells | strict_pass | combined_pass | mean_r2 |",
        "|---------|---------|-------------|---------------|---------|",
    ]
    for p, s in agg["per_purpose"].items():
        summary_lines.append(
            f"| {p} | {s['n']} | {s['strict_pass']}/{s['n']} | "
            f"{s['combined_pass']}/{s['n']} | {s['mean_r2']:.4f} |"
        )
    summary_lines += [
        "",
        "## Per-cell table (24 = 3 seeds × 8 pairs)",
        "",
        fmt_table(all_rows),
        "",
        "## Recommendation for paper integration",
        "",
    ]
    if agg["verdict"] == "GREEN":
        summary_lines.append(
            "Folktables clears the strict 18/24 threshold — eligible to enter the "
            "paper's main results table alongside Adult / HMDA / Diabetes (all under "
            "FFB binary-race convention). No retrain needed for paper inclusion."
        )
    elif agg["verdict"] == "YELLOW":
        summary_lines.append(
            "Folktables is on the boundary (15-17/24 strict). Decide on inclusion "
            "based on which pairs fail and whether a Round 2 sweep (e.g. lambda_min "
            "tuning, longer training, or rank bump) is worth the GPU spend."
        )
    else:
        summary_lines.append(
            "Folktables underperforms (<15/24 strict). Diagnose dominant failure "
            "purpose first; then decide between Round 2 retrain or excluding from "
            "the paper's main table."
        )

    out_path = ROOT / "results" / "V2_FOLKTABLES_ROUND1_VERDICT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(summary_lines) + "\n")

    raw_out = ROOT / "results" / f"v2_{DATASET}{args.out_tag}" / "verdict_raw.json"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text(json.dumps(
        {"aggregate": agg, "per_cell": all_rows}, indent=2,
    ))

    print()
    print("=" * 60)
    print(f"VERDICT: {agg['verdict']}")
    print(
        f"  strict pass {agg['strict_pass']}/{agg['n_cells']}  "
        f"combined {agg['combined_pass']}/{agg['n_cells']}  "
        f"mean R²={agg['mean_r2']:.4f}"
    )
    print(f"  saved: {out_path}")
    print(f"  saved: {raw_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()

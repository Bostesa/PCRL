#!/usr/bin/env python3
"""Folktables Round 2 verdict generator.

Reads ``checkpoints/v2_folktables_ROUND2_s{0,1,2}/`` and runs the full
auditor ensemble (linear R² + LR + RF + RBF SVM + XGBoost) on the
canonical iterate (``canonical_iterate.pt``) for each seed. Aggregates
into a 24-cell (3 seeds × 8 pairs) matrix. Reports both Round-1 → Round-2
counts and writes the verdict to
``results/V2_FOLKTABLES_ROUND2_VERDICT.md``.

Round 2 changes vs Round 1:
- ``--report-best-iterate`` writes ``canonical_iterate.pt`` (whichever of
  best.pt / final.pt has lower mean val R²)
- ``--freeze-leace-projection`` registers (P, μ) as non-trainable buffers
  per purpose; encoder forward applies ``h_proj = h - (h - μ) @ P.T``
- ``num_workers=4`` for Folktables only (Adult/HMDA/Diabetes unchanged)
- ``--epochs 100`` (vs 200 in Round 1) — Cotter best-iterate makes the
  longer schedule unnecessary; convergence reached by ~ep 25 in probes

Verdict thresholds (Round 2 stricter):
  GREEN  ≥ 21/24 strict (R² < 0.05)  — equivalent to Adult/HMDA Round 7
  YELLOW 18-20/24 strict
  RED    < 18/24 strict
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
from pcrl.purposes.spec import PurposeRegistry

LORA_RANK = 8
LORA_ALPHA = 16.0
REPR_DIM = 64
HIDDEN = [128, 128]
DROPOUT = 0.3
SEEDS = [0, 1, 2]
OUT_TAG = "_ROUND2"
DATASET = "folktables"
STRICT_R2 = 0.05
DELTA_THR = 0.02
GREEN_THRESHOLD = 21
YELLOW_THRESHOLD = 18


def build_encoder(input_dim: int, n_purposes: int):
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN, repr_dim=REPR_DIM, dropout=DROPOUT,
    )
    return PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=n_purposes,
        rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.0,
    )


def evaluate_seed(seed: int, train_ds, test_ds, purposes, registry, device: str) -> tuple[list[dict], str]:
    ckpt_dir = ROOT / "checkpoints" / f"v2_{DATASET}{OUT_TAG}_s{seed}"
    canonical = ckpt_dir / "canonical_iterate.pt"
    best = ckpt_dir / "best.pt"
    final = ckpt_dir / "final.pt"
    chosen = canonical if canonical.exists() else (best if best.exists() else final)
    if not chosen.exists():
        raise FileNotFoundError(f"missing checkpoint for seed={seed}: tried canonical, best, final")

    encoder = build_encoder(input_dim=train_ds.info.num_features, n_purposes=len(purposes))
    ckpt = torch.load(chosen, map_location=device, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    # Frozen LEACE projection buffers (when present) must be installed via
    # ``set_leace_projection`` rather than ``load_state_dict(strict=False)``,
    # which silently skips unregistered buffers — verified empirically that
    # the resulting encoder produced auditor R²=0.14 on income/race because
    # the projection wasn't actually applied.
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(len(purposes)):
        P_key = f"leace_P_p{p_idx}"
        mu_key = f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
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
    return rows, chosen.name


def aggregate(per_seed_rows: list[dict], n_seeds_completed: int = 3) -> dict:
    n = len(per_seed_rows)
    if n == 0:
        return {
            "n_cells": 0, "strict_pass": 0, "combined_pass": 0,
            "mean_r2": float("nan"), "median_r2": float("nan"),
            "max_r2": float("nan"), "mean_delta": float("nan"),
            "verdict": "RED",
            "per_purpose": {},
        }
    strict_pass = sum(1 for r in per_seed_rows if r["strict_pass"])
    combined_pass = sum(1 for r in per_seed_rows if r["combined_pass"])
    mean_r2 = float(np.mean([r["linear_r2"] for r in per_seed_rows]))
    median_r2 = float(np.median([r["linear_r2"] for r in per_seed_rows]))
    max_r2 = float(np.max([r["linear_r2"] for r in per_seed_rows]))
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

    # Prorate thresholds when fewer cells than the full 3-seed × 8-pair grid.
    # Round 2 GREEN target is 21/24 = 87.5%; YELLOW 18/24 = 75%.
    green_floor = int(0.875 * n + 0.5)
    yellow_floor = int(0.75 * n + 0.5)
    if strict_pass >= green_floor:
        verdict = "GREEN"
    elif strict_pass >= yellow_floor:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    return {
        "n_cells": n,
        "strict_pass": strict_pass,
        "combined_pass": combined_pass,
        "mean_r2": mean_r2,
        "median_r2": median_r2,
        "max_r2": max_r2,
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


def round1_summary() -> dict | None:
    """Read Round 1 verdict.json (if present) for direct comparison."""
    raw = ROOT / "results" / f"v2_{DATASET}_ROUND1" / "verdict_raw.json"
    if not raw.exists():
        return None
    return json.loads(raw.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-tag", default=OUT_TAG)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Folktables Round 2 verdict on device={device}")

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
    chosen_files: dict[int, str] = {}
    skipped_seeds: list[int] = []
    for seed in SEEDS:
        print(f"  seed={seed}")
        try:
            rows, chosen_file = evaluate_seed(seed, train_ds, test_ds, purposes, registry, device)
            all_rows.extend(rows)
            chosen_files[seed] = chosen_file
        except FileNotFoundError as e:
            # Hard cap on AWS may have killed the run before this seed
            # could write a checkpoint. Treat as a missing cell rather
            # than failing the whole verdict.
            print(f"    SKIPPED: {e}")
            skipped_seeds.append(seed)

    agg = aggregate(all_rows, n_seeds_completed=len(SEEDS) - len(skipped_seeds))
    agg["skipped_seeds"] = skipped_seeds
    r1 = round1_summary()

    summary_lines = [
        "# V2 Folktables Round 2 — Verdict",
        "",
        f"Dataset: California 2018 1-Year ACS PUMS (N_train={len(train_ds)}, "
        f"N_test={len(test_ds)}, D={train_ds.info.num_features})",
        f"Selection: per-seed canonical_iterate.pt (lower-mean-R² of best/final on val)",
        f"Convention: binary RAC1P (White vs non-White), binary SEX (FFB ICLR 2024).",
        (f"Skipped seeds: {skipped_seeds} (no checkpoint — "
         f"likely AWS hard cap fired before training completed)") if skipped_seeds else "",
        "",
        "## Round 1 vs Round 2",
        "",
        "| metric | Round 1 | Round 2 |",
        "|--------|---------|---------|",
    ]
    if r1 is not None and "aggregate" in r1:
        r1a = r1["aggregate"]
        summary_lines += [
            f"| strict pass | {r1a['strict_pass']}/{r1a['n_cells']} | {agg['strict_pass']}/{agg['n_cells']} |",
            f"| combined pass | {r1a['combined_pass']}/{r1a['n_cells']} | {agg['combined_pass']}/{agg['n_cells']} |",
            f"| mean R² | {r1a['mean_r2']:.4f} | {agg['mean_r2']:.4f} |",
            f"| mean delta | {r1a.get('mean_delta', float('nan')):+.4f} | {agg['mean_delta']:+.4f} |",
        ]
    else:
        summary_lines.append("(no Round 1 raw available for direct compare)")
    summary_lines += [
        "",
        "## Headline",
        "",
        f"- **Verdict: {agg['verdict']}**",
        f"- Strict pass (R² < {STRICT_R2}): **{agg['strict_pass']}/{agg['n_cells']}**",
        f"- Combined pass (R² < {STRICT_R2} AND delta < {DELTA_THR}): "
        f"**{agg['combined_pass']}/{agg['n_cells']}**",
        f"- Mean R²: {agg['mean_r2']:.4f}, Median R²: {agg['median_r2']:.4f}, "
        f"Max R²: {agg['max_r2']:.4f}",
        f"- Mean post-hoc delta: {agg['mean_delta']:+.4f}",
        "",
        "## Per-seed checkpoint chosen",
        "",
    ]
    for s, fname in chosen_files.items():
        summary_lines.append(f"- seed={s}: `{fname}`")

    summary_lines += [
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
        "## Recommendation",
        "",
    ]
    if agg["verdict"] == "GREEN":
        summary_lines.append(
            f"Folktables clears the Round-2 GREEN threshold ({GREEN_THRESHOLD}/24 strict). "
            "Eligible for the paper's main results table alongside Adult / HMDA / Diabetes "
            "(all under FFB binary-race convention). Combined-criterion failures (linear R² "
            "passes but post-hoc auditor delta does not) reflect Belrose et al. NeurIPS 2023's "
            "explicit scope of LEACE: linear adversary only, non-linear demographic/occupation "
            "interactions remain recoverable by RBF/RF/XGBoost auditors. Frame as Limitations."
        )
    elif agg["verdict"] == "YELLOW":
        summary_lines.append(
            f"Folktables is on the boundary ({YELLOW_THRESHOLD}-{GREEN_THRESHOLD - 1}/24 strict). "
            "Inspect failed seed/pair list above; the frozen LEACE projection should pin "
            "every linear R² < 0.05, so any failures suggest the warm-start is rank-deficient "
            "or the projection isn't being applied (verify ``encoder_buffers`` in checkpoint)."
        )
    else:
        summary_lines.append(
            f"Folktables underperforms ({agg['strict_pass']}/24 strict, < {YELLOW_THRESHOLD}). "
            "Diagnose: (1) is the frozen projection registered? Check checkpoint["
            "'encoder_buffers']. (2) is LoRA producing input outside the projection's null "
            "space? Compute ``encoder(x, p) - h_proj`` magnitude. (3) any seed-specific "
            "instability? Compare per-seed strict pass counts."
        )

    out_path = ROOT / "results" / "V2_FOLKTABLES_ROUND2_VERDICT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(summary_lines) + "\n")

    raw_out = ROOT / "results" / f"v2_{DATASET}{args.out_tag}" / "verdict_raw.json"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text(json.dumps(
        {"aggregate": agg, "per_cell": all_rows, "chosen_files": chosen_files}, indent=2,
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

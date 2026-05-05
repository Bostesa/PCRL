#!/usr/bin/env python3
"""Re-evaluate Round 4 Adult final.pt with binarized race (FFB ICLR 2024).

Computes, per (purpose, race, seed) cell:
  - linear R² with 5-class one-hot race (current convention)
  - linear R² with binarized race (1 if "White" else 0)
  - post-hoc auditor delta with 5-class race target
  - post-hoc auditor delta with binarized race target

Then re-aggregates the 24 cells (8 pairs × 3 seeds) under the FFB combined
convention (binarized race for race pairs, multinomial for the rest) and
emits a verdict.

Race index mapping (alphabetical sort of the 5 categories):
    0 Amer-Indian-Eskimo, 1 Asian-Pac-Islander, 2 Black,
    3 Other,             4 White
Binarized: 1 iff race==4.

Usage:
    .venv/bin/python scripts/binarize_race_reeval_adult.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.certificates import _extract_representations_and_labels  # noqa: E402
from pcrl.models.auditor import PostHocAuditorSuite  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry  # noqa: E402
from pcrl.purposes.verification import LinearComplianceCertificate  # noqa: E402

SEEDS = [0, 1, 2]
WHITE_IDX = 4  # alphabetical sort places "White" last in the 5-class race list
DEVICE = "cpu"
RACE_PAIRS = [
    ("income_prediction", "race"),
    ("employment_analysis", "race"),
    ("education_assessment", "race"),
]


# ──────────────────────────────────────────────────────────────────────
# Subsample a large train set for the post-hoc auditor (matches the
# existing eval pipeline's max_fit_samples = 20_000 behaviour).
# ──────────────────────────────────────────────────────────────────────
def subsample(reprs: np.ndarray, labels: np.ndarray, max_n: int, seed: int = 42):
    n = labels.shape[0]
    if n <= max_n:
        return reprs, labels
    try:
        from sklearn.model_selection import train_test_split
        rs, _, ls, _ = train_test_split(
            reprs, labels, train_size=max_n, stratify=labels, random_state=seed,
        )
        return rs, ls
    except ValueError:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        return reprs[idx], labels[idx]


def post_hoc_delta(
    train_reprs: np.ndarray, train_labels: np.ndarray,
    test_reprs: np.ndarray, test_labels: np.ndarray,
    seed: int = 42,
):
    """Best-of-suite accuracy minus majority-class proportion (= delta)."""
    fit_r, fit_l = subsample(train_reprs, train_labels, 20_000, seed=seed)
    suite = PostHocAuditorSuite(random_state=seed)
    suite.fit(fit_r, fit_l)
    results = suite.evaluate(test_reprs, test_labels)
    best_acc = float(max(m["accuracy"] for m in results.values()))
    all_labels = np.concatenate([train_labels, test_labels])
    _, counts = np.unique(all_labels, return_counts=True)
    maj = float(counts.max() / len(all_labels))
    return best_acc, maj, best_acc - maj


def linear_r2(reprs: np.ndarray, labels: np.ndarray) -> float:
    cert = LinearComplianceCertificate(epsilon=0.05)
    return cert.check(reprs, labels).r_squared


def build_encoder(input_dim: int, n_purposes: int) -> PerPurposeLoRAEncoder:
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    return PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=n_purposes, rank=8, alpha=16.0, dropout=0.0,
    )


def load_seed(seed: int, train_ds, val_ds, test_ds, registry, purposes):
    """Load encoder weights from final.pt for `seed` and return the encoder."""
    encoder = build_encoder(train_ds.info.num_features, len(purposes))
    ckpt_path = ROOT / "checkpoints" / f"v2_adult_s{seed}" / "final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    encoder.eval()
    return encoder


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading Adult dataset (train/val/test) …", flush=True)
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=False)
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )
    train_loader = DataLoader(
        train_ds, batch_size=512, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    purpose_idx = {p.name: i for i, p in enumerate(purposes)}

    rows = []  # per (seed, purpose, attr) cell, race pairs only
    for seed in SEEDS:
        print(f"\n=== seed {seed}: loading final.pt encoder …", flush=True)
        encoder = load_seed(seed, train_ds, None, test_ds, registry, purposes)

        # Cache reprs+race per purpose (single-pass extraction → no row misalignment)
        reps_cache = {}
        for p_name, _ in RACE_PAIRS:
            idx = purpose_idx[p_name]
            tr_r, tr_lab = _extract_representations_and_labels(
                encoder, train_loader, idx, ["race"], DEVICE,
            )
            te_r, te_lab = _extract_representations_and_labels(
                encoder, test_loader, idx, ["race"], DEVICE,
            )
            reps_cache[p_name] = (tr_r, tr_lab["race"], te_r, te_lab["race"])
            print(
                f"  [{p_name}] train_n={tr_r.shape[0]} test_n={te_r.shape[0]} "
                f"race_classes={sorted(np.unique(np.concatenate([tr_lab['race'], te_lab['race']])).tolist())}",
                flush=True,
            )

        for p_name, attr in RACE_PAIRS:
            tr_r, tr_race5, te_r, te_race5 = reps_cache[p_name]

            # Binarize: White (idx=4) → 1, else 0.
            tr_race_bin = (tr_race5 == WHITE_IDX).astype(np.int64)
            te_race_bin = (te_race5 == WHITE_IDX).astype(np.int64)

            r2_5class = linear_r2(te_r, te_race5)
            r2_binarized = linear_r2(te_r, te_race_bin)

            print(
                f"  [{p_name}] linear R² → 5-class={r2_5class:.4f} "
                f"binarized={r2_binarized:.4f}",
                flush=True,
            )

            t0 = time.time()
            best5, maj5, delta5 = post_hoc_delta(tr_r, tr_race5, te_r, te_race5)
            print(
                f"  [{p_name}] post-hoc 5-class: best={best5:.4f} maj={maj5:.4f} "
                f"delta={delta5:+.4f} ({time.time()-t0:.0f}s)",
                flush=True,
            )

            t0 = time.time()
            bestB, majB, deltaB = post_hoc_delta(tr_r, tr_race_bin, te_r, te_race_bin)
            print(
                f"  [{p_name}] post-hoc binarized: best={bestB:.4f} maj={majB:.4f} "
                f"delta={deltaB:+.4f} ({time.time()-t0:.0f}s)",
                flush=True,
            )

            rows.append({
                "seed": seed,
                "purpose": p_name,
                "attribute": attr,
                "linear_r2_5class": r2_5class,
                "linear_r2_binarized": r2_binarized,
                "phoc_best_5class": best5,
                "phoc_majority_5class": maj5,
                "phoc_delta_5class": delta5,
                "phoc_best_binarized": bestB,
                "phoc_majority_binarized": majB,
                "phoc_delta_binarized": deltaB,
                "binarized_R2_passes": r2_binarized < 0.05,
                "binarized_combined_passes": (r2_binarized < 0.05) and (deltaB < 0.02),
            })

    # ── Aggregation under combined FFB convention ─────────────────────
    # Pull the existing 5-class results from final_vs_best.json for non-race
    # cells; replace the race rows with the binarized ones we just computed.
    fvb_path = ROOT / "results" / "v2_adult_ROUND4" / "final_vs_best.json"
    fvb = json.loads(fvb_path.read_text())
    by_seed_rows = {int(s): d["final"]["rows"] for s, d in fvb["per_seed"].items()}

    cells_strict = []  # (seed, purpose, attr, r2_used, delta_used, passes)
    for seed in SEEDS:
        for r in by_seed_rows[seed]:
            if r["attribute"] == "race":
                bin_row = next(
                    x for x in rows if x["seed"] == seed and x["purpose"] == r["purpose"]
                )
                cells_strict.append({
                    "seed": seed,
                    "purpose": r["purpose"],
                    "attribute": "race",
                    "r2": bin_row["linear_r2_binarized"],
                    "delta": bin_row["phoc_delta_binarized"],
                    "convention": "binarized",
                })
            else:
                cells_strict.append({
                    "seed": seed,
                    "purpose": r["purpose"],
                    "attribute": r["attribute"],
                    "r2": r["linear_r2"],
                    "delta": r["delta"],
                    "convention": "multinomial" if r["attribute"] != "sex" else "binary",
                })

    pass24_combined = sum(1 for c in cells_strict if (c["r2"] < 0.05 and c["delta"] < 0.02))
    below05_combined = sum(1 for c in cells_strict if c["r2"] < 0.05)
    mean_r2_combined = float(np.mean([c["r2"] for c in cells_strict]))

    # Race-only stats under each convention
    n_race_below_05_5 = sum(1 for r in rows if r["linear_r2_5class"] < 0.05)
    n_race_below_05_b = sum(1 for r in rows if r["linear_r2_binarized"] < 0.05)
    mean_r2_race_5 = float(np.mean([r["linear_r2_5class"] for r in rows]))
    mean_r2_race_b = float(np.mean([r["linear_r2_binarized"] for r in rows]))

    # ── Verdict ───────────────────────────────────────────────────────
    if below05_combined >= 22:
        verdict = "GREEN"
        rec = "Adopt FFB binarization for race; report multi-class in appendix."
    elif below05_combined >= 21:
        verdict = "YELLOW"
        rec = "Binarization helps but income/race fails 1-2 seeds; consider Fix 1 (joint multivariate LEACE)."
    else:
        verdict = "RED"
        rec = "Metric artifact wasn't dominant; commit to Fix 1 implementation."

    # ── Persist & report ──────────────────────────────────────────────
    out_dir = ROOT / "results" / "v2_adult_ROUND4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md = []
    md.append("# Round 4 Adult — binarized race re-evaluation (FFB ICLR 2024)")
    md.append("")
    md.append("Re-evaluates `final.pt` for all 3 seeds with binarized race (1=White, 0=non-White).")
    md.append("No retraining — encoders are loaded from disk and only the audit metric changes.")
    md.append("")
    md.append("## Per-seed race-pair table")
    md.append("")
    md.append("| Pair | seed | 5-class R² | binarized R² | binarized passes (R²<0.05)? | binarized post-hoc delta | binarized combined passes? |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['purpose']}/{r['attribute']} | {r['seed']} | "
            f"{r['linear_r2_5class']:.4f} | {r['linear_r2_binarized']:.4f} | "
            f"{'YES' if r['binarized_R2_passes'] else 'NO'} | "
            f"{r['phoc_delta_binarized']:+.4f} | "
            f"{'YES' if r['binarized_combined_passes'] else 'NO'} |"
        )
    md.append("")
    md.append("## Race-pair aggregates (n=9 across 3 seeds × 3 race pairs)")
    md.append("")
    md.append(f"- 5-class: mean R² = **{mean_r2_race_5:.4f}**, n<0.05 = **{n_race_below_05_5}/9**")
    md.append(f"- Binarized: mean R² = **{mean_r2_race_b:.4f}**, n<0.05 = **{n_race_below_05_b}/9**")
    md.append("")
    md.append("## Combined-convention 24-cell aggregate")
    md.append("")
    md.append("Race pairs use binarized R² + binarized post-hoc delta; non-race pairs unchanged from `final_vs_best.json`.")
    md.append("")
    md.append(f"- Pass count under combined convention (R²<0.05 AND delta<0.02): **{pass24_combined}/24**")
    md.append(f"- Cells with R² < 0.05 under combined convention: **{below05_combined}/24**")
    md.append(f"- Mean R² under combined convention: **{mean_r2_combined:.4f}**")
    md.append("")
    md.append("## Per-cell breakdown (combined convention)")
    md.append("")
    md.append("| seed | purpose | attribute | convention | R² | delta | combined pass |")
    md.append("|---|---|---|---|---|---|---|")
    for c in cells_strict:
        passes = (c["r2"] < 0.05 and c["delta"] < 0.02)
        md.append(
            f"| {c['seed']} | {c['purpose']} | {c['attribute']} | "
            f"{c['convention']} | {c['r2']:.4f} | {c['delta']:+.4f} | "
            f"{'YES' if passes else 'NO'} |"
        )
    md.append("")
    md.append(f"## Verdict: **{verdict}**")
    md.append("")
    md.append(rec)
    md.append("")
    (out_dir / "binarized_race_evaluation.md").write_text("\n".join(md))

    # JSON snapshot
    snap = {
        "convention": {"race": "binarized (FFB ICLR 2024)", "non_race": "unchanged"},
        "per_race_cell": rows,
        "combined_24cell": {
            "pass_count": pass24_combined,
            "n_below_05": below05_combined,
            "mean_linear_r2": mean_r2_combined,
            "cells": cells_strict,
        },
        "race_only": {
            "5class": {"mean_r2": mean_r2_race_5, "n_below_05": n_race_below_05_5, "n_total": 9},
            "binarized": {"mean_r2": mean_r2_race_b, "n_below_05": n_race_below_05_b, "n_total": 9},
        },
        "verdict": verdict,
        "recommendation": rec,
    }
    (out_dir / "binarized_race_evaluation.json").write_text(
        json.dumps(snap, indent=2)
    )

    # Update summary.json with the FFB combined headline
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["ffb_combined_convention"] = {
        "convention": "race binarized (White vs non-White, FFB ICLR 2024); other attrs unchanged",
        "pass_count_24": pass24_combined,
        "n_below_05_24": below05_combined,
        "mean_linear_r2_24": round(mean_r2_combined, 4),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # Final stdout line
    print(
        f"\nBinarized race re-eval complete. Verdict: {verdict}. "
        f"Pass count under FFB convention: {pass24_combined}/24. "
        f"Recommendation: {rec}",
        flush=True,
    )


if __name__ == "__main__":
    main()

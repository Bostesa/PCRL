#!/usr/bin/env python3
"""LEACE post-processing on trained PCRL CelebA encoder.

Loads the trained PCRL CelebA encoder checkpoint, extracts purpose-conditioned
representations, applies LEACE projection to zero out linear leakage of each
disallowed attribute, and re-runs compliance audit on the post-processed
representations.

For each (purpose, disallowed_attr) pair:
  1. Extract train/test representations with the encoder for this purpose
  2. Fit LEACE on train representations to erase the disallowed attribute
  3. Transform both train and test representations
  4. Run linear + empirical compliance audit on post-processed representations
  5. Compare before/after LEACE

Saves to results/celeba/pcrl_leace_postprocess.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcrl.data.celeba import CelebADataset, get_celeba_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    _extract_representations_and_labels,
)
from pcrl.models.baselines import LEACEEraser
from pcrl.models.cnn_encoder import CNNEncoder
from pcrl.purposes.spec import PurposeRegistry

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

# Architecture must match run_celeba.py
REPR_DIM = 256
PURPOSE_EMB_DIM = 32
CONV_CHANNELS = (32, 64, 128, 256)
DROPOUT = 0.3


def run_audit_on_representations(
    train_reprs: np.ndarray,
    train_labels: np.ndarray,
    test_reprs: np.ndarray,
    test_labels: np.ndarray,
) -> ComplianceReport:
    """Run linear + empirical audit on given representations."""
    linear_audit = LinearAudit(epsilon=0.01)
    empirical_audit = EmpiricalAudit(random_state=42)

    linear_result, null_result = linear_audit.audit(test_reprs, test_labels)

    all_labels = np.concatenate([train_labels, test_labels])
    unique_classes, class_counts = np.unique(all_labels, return_counts=True)
    num_classes = len(unique_classes)
    majority_proportion = float(class_counts.max() / len(all_labels))

    best_acc, emp_results = empirical_audit.audit(
        train_reprs, train_labels, test_reprs, test_labels,
    )

    chance_acc = 1.0 / max(num_classes, 1)
    empirical_ok = (best_acc - chance_acc) < 0.05
    certified = linear_result.certified and empirical_ok

    return ComplianceReport(
        purpose_name="",
        attr_name="",
        linear_r2=linear_result.r_squared,
        linear_certified=linear_result.certified,
        null_space_r2=null_result.r_squared,
        variance_preserved=null_result.variance_preserved or 1.0,
        empirical_best_acc=best_acc,
        empirical_chance_acc=chance_acc,
        empirical_results=emp_results,
        certified=certified,
        majority_proportion=majority_proportion,
        num_classes=num_classes,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    # Load checkpoint
    checkpoint_path = project_root / "checkpoints" / "celeba" / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = project_root / "checkpoints" / "celeba" / "final.pt"
    if not checkpoint_path.exists():
        print("ERROR: No checkpoint found at checkpoints/celeba/")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    encoder = CNNEncoder(
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conv_channels=CONV_CHANNELS,
        dropout=DROPOUT,
    )
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.to(device).eval()
    print("  Encoder loaded successfully")

    # Load data
    print("Loading CelebA data...")
    train_ds = CelebADataset(
        purposes, root="data/celeba", split="train", max_samples=10000,
    )
    test_ds = CelebADataset(
        purposes, root="data/celeba", split="test", max_samples=3000,
    )
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Run audit for each (purpose, disallowed_attr) pair — before and after LEACE
    rows = []

    for purpose_idx, purpose in enumerate(purposes):
        for attr_name in purpose.disallowed_attrs:
            print(f"\n[{purpose_idx}] {purpose.name} / {attr_name}")

            # Extract representations
            train_reprs, train_labels = _extract_representations_and_labels(
                encoder, train_loader, purpose_idx, attr_name, device,
            )
            test_reprs, test_labels = _extract_representations_and_labels(
                encoder, test_loader, purpose_idx, attr_name, device,
            )

            # --- Before LEACE ---
            print("  Before LEACE...")
            before = run_audit_on_representations(
                train_reprs, train_labels, test_reprs, test_labels,
            )
            print(f"    R²={before.linear_r2:.4f}, "
                  f"EmpAcc={before.empirical_best_acc:.1%}, "
                  f"Majority={before.majority_proportion:.1%}")

            # --- Apply LEACE ---
            print("  Fitting LEACE...")
            eraser = LEACEEraser(regularization=1e-6)
            eraser.fit(train_reprs, train_labels)
            train_reprs_leace = eraser.transform(train_reprs)
            test_reprs_leace = eraser.transform(test_reprs)

            # --- After LEACE ---
            print("  After LEACE...")
            after = run_audit_on_representations(
                train_reprs_leace, train_labels, test_reprs_leace, test_labels,
            )

            delta_before = before.empirical_best_acc - before.majority_proportion
            delta_after = after.empirical_best_acc - after.majority_proportion
            adj_pass_before = delta_before < 0.02 and before.linear_r2 < 0.05
            adj_pass_after = delta_after < 0.02 and after.linear_r2 < 0.05

            print(f"    R²={after.linear_r2:.4f}, "
                  f"EmpAcc={after.empirical_best_acc:.1%}, "
                  f"Majority={after.majority_proportion:.1%}, "
                  f"Pass={'YES' if adj_pass_after else 'NO'}")

            rows.append({
                "purpose": purpose.name,
                "attribute": attr_name,
                "before_r2": round(before.linear_r2, 6),
                "before_emp_acc": round(before.empirical_best_acc, 4),
                "before_majority": round(before.majority_proportion, 4),
                "before_delta": round(delta_before, 4),
                "before_pass": adj_pass_before,
                "after_r2": round(after.linear_r2, 6),
                "after_emp_acc": round(after.empirical_best_acc, 4),
                "after_majority": round(after.majority_proportion, 4),
                "after_delta": round(delta_after, 4),
                "after_pass": adj_pass_after,
            })

    # Print summary table
    print(f"\n{'=' * 110}")
    print("LEACE Post-Processing Results on PCRL CelebA Encoder")
    print(f"{'=' * 110}")
    print(f"{'Purpose':<28} {'Attribute':<12} "
          f"{'Before R²':>10} {'Before Δ':>9} {'Before':>7} "
          f"{'After R²':>10} {'After Δ':>9} {'After':>7}")
    print("-" * 110)

    before_pass = 0
    after_pass = 0
    for r in rows:
        b_status = "PASS" if r["before_pass"] else "FAIL"
        a_status = "PASS" if r["after_pass"] else "FAIL"
        print(f"{r['purpose']:<28} {r['attribute']:<12} "
              f"{r['before_r2']:>10.4f} {r['before_delta']:>8.1%} {b_status:>7} "
              f"{r['after_r2']:>10.4f} {r['after_delta']:>8.1%} {a_status:>7}")
        before_pass += r["before_pass"]
        after_pass += r["after_pass"]

    print("-" * 110)
    print(f"Compliance: Before LEACE {before_pass}/{len(rows)}, "
          f"After LEACE {after_pass}/{len(rows)}")
    print(f"{'=' * 110}")

    # Save CSV
    output_path = project_root / "results" / "celeba" / "pcrl_leace_postprocess.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "purpose", "attribute",
        "before_r2", "before_emp_acc", "before_majority", "before_delta", "before_pass",
        "after_r2", "after_emp_acc", "after_majority", "after_delta", "after_pass",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

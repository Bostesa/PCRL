#!/usr/bin/env python3
"""Run MINE on the trained PCRL CelebA encoder from run_celeba.py.

The MINE estimator works correctly (verified on synthetic data), but previously
failed because the impossibility script's small encoder (repr_dim=64) collapsed.

This script loads the GOOD encoder checkpoint (repr_dim=256, conv=(32,64,128,256))
from run_celeba.py, extracts purpose-conditioned representations, and runs MINE
to estimate MI between representations and conflicting attributes.

Saves to results/celeba/mine_on_good_encoder.csv.
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
from pcrl.evaluation.mine import MINEstimator
from pcrl.models.cnn_encoder import CNNEncoder
from pcrl.purposes.verification import fano_mi_lower_bound, impossibility_bound

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Architecture must match run_celeba.py
REPR_DIM = 256
PURPOSE_EMB_DIM = 32
CONV_CHANNELS = (32, 64, 128, 256)
DROPOUT = 0.3


def extract_representations(
    encoder: torch.nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract representations, task labels, and sensitive attrs."""
    encoder.eval()
    all_reprs, all_tasks, all_sens = [], {}, {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reprs.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
    return (
        np.concatenate(all_reprs),
        {k: np.concatenate(v) for k, v in all_tasks.items()},
        {k: np.concatenate(v) for k, v in all_sens.items()},
    )


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    purposes = get_celeba_purposes()

    # Load checkpoint
    checkpoint_path = project_root / "checkpoints" / "celeba" / "best.pt"
    if not checkpoint_path.exists():
        # Try final
        checkpoint_path = project_root / "checkpoints" / "celeba" / "final.pt"
    if not checkpoint_path.exists():
        print(f"ERROR: No checkpoint found at checkpoints/celeba/")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct encoder
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

    # Load test data
    print("Loading CelebA test data...")
    test_ds = CelebADataset(
        purposes, root="data/celeba", split="test", max_samples=5000,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    print(f"  Test samples: {len(test_ds)}")

    # Extract representations for all purposes
    print("\nExtracting representations...")
    reprs = {}
    tasks = {}
    sens = {}
    for idx, p in enumerate(purposes):
        r, t, s = extract_representations(encoder, test_loader, idx, device)
        reprs[p.name] = r
        tasks[p.name] = t
        sens[p.name] = s
        print(f"  [{idx}] {p.name}: repr shape = {r.shape}")

    # Build combined attribute labels
    all_labels: dict[str, np.ndarray] = {}
    ref_sens = sens[purposes[0].name]
    ref_tasks = tasks[purposes[0].name]
    for k, v in ref_sens.items():
        all_labels[k] = v
    for k, v in ref_tasks.items():
        all_labels[k] = v

    # Find conflicts
    conflicts = []
    for p_need in purposes:
        for task_attr in p_need.allowed_tasks:
            for p_forbid in purposes:
                if p_need.name != p_forbid.name and task_attr in p_forbid.disallowed_attrs:
                    conflicts.append((task_attr, p_need.name, p_forbid.name))

    print(f"\nFound {len(conflicts)} conflicting triples")

    # Run MINE on all unique (purpose, attribute) pairs
    mine = MINEstimator(hidden_dim=256, num_steps=2000, batch_size=512)
    mi_cache: dict[str, float] = {}

    def get_mi(repr_array: np.ndarray, labels: np.ndarray, key: str) -> float:
        if key not in mi_cache:
            result = mine.estimate(repr_array, labels, device=device)
            mi_cache[key] = result.mi_bits
            print(f"    MINE({key}): {result.mi_bits:.4f} bits")
        return mi_cache[key]

    print("\nComputing MI estimates (2000 steps each)...")
    rows = []

    for attr, need, forbid in conflicts:
        if attr not in all_labels:
            continue
        labels = all_labels[attr]
        num_classes = int(labels.max()) + 1

        # Entropy H(A)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy_a = -float(np.sum(probs * np.log(probs + 1e-12)))

        # MI for purpose that NEEDS this attribute
        mi_need_bits = get_mi(reprs[need], labels, f"{need}_{attr}")

        # MI for purpose that FORBIDS this attribute (PCRL's leakage)
        mi_forbid_bits = get_mi(reprs[forbid], labels, f"{forbid}_{attr}")

        row = {
            "attribute": attr,
            "needed_by": need,
            "forbidden_by": forbid,
            "num_classes": num_classes,
            "entropy_a_bits": entropy_a / np.log(2),
            "mi_needed_bits": mi_need_bits,
            "mi_forbidden_bits": mi_forbid_bits,
            "sr_min_leak": impossibility_bound(
                mi_need_bits * np.log(2), num_classes, entropy_a,
            ),
            "pcrl_leak_bound": impossibility_bound(
                mi_forbid_bits * np.log(2), num_classes, entropy_a,
            ),
        }
        rows.append(row)

    # Print results
    print(f"\n{'=' * 100}")
    print("MINE Results on Trained CelebA Encoder (repr_dim=256)")
    print(f"{'=' * 100}")
    print(f"{'Attribute':<12} {'Needed by':<28} {'Forbidden by':<28} "
          f"{'MI need':>8} {'MI forbid':>10} {'SR Min':>8} {'PCRL':>8}")
    print("-" * 100)

    for r in rows:
        print(f"{r['attribute']:<12} {r['needed_by']:<28} {r['forbidden_by']:<28} "
              f"{r['mi_needed_bits']:>7.3f}b {r['mi_forbidden_bits']:>9.3f}b "
              f"{r['sr_min_leak']:>7.1%} {r['pcrl_leak_bound']:>7.1%}")

    print(f"{'=' * 100}")

    # Save CSV
    output_path = project_root / "results" / "celeba" / "mine_on_good_encoder.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attribute", "needed_by", "forbidden_by", "num_classes",
        "entropy_a_bits", "mi_needed_bits", "mi_forbidden_bits",
        "sr_min_leak", "pcrl_leak_bound",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items()
            })
    print(f"\nSaved {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()

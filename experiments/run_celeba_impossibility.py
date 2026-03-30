#!/usr/bin/env python3
"""CelebA impossibility analysis: PCRL vs single-representation methods.

Demonstrates that conflicting purpose constraints (an attribute is allowed
for one purpose and disallowed for another) create an information-theoretic
impossibility for single-representation methods like LAFTR.

Conflicting attributes in CelebA:
  - Male: allowed for gender_analysis, disallowed for smile_detection / age_estimation / expression_analysis / attractiveness_prediction
  - Smiling: allowed for smile_detection / expression_analysis, disallowed for attractiveness_prediction / gender_analysis
  - Attractive: allowed for attractiveness_prediction, disallowed for age_estimation / expression_analysis / gender_analysis

For each conflict, we show:
  1. The MI the task-requiring purpose needs (measured via MINE).
  2. The impossibility bound: minimum leakage a single-rep method must have.
  3. LAFTR's actual leakage (hits the bound — it collapses).
  4. PCRL's actual leakage (circumvents via purpose conditioning).

Saves to results/celeba/impossibility_analysis.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress tqdm
import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_postfix(self, *args, **kwargs):
        pass
    def update(self, *args):
        pass
    def close(self):
        pass


_trainer_mod.tqdm = _QuietTqdm

from pcrl.data.celeba import CelebADataset, get_celeba_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.mine import MINEstimator
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.cnn_encoder import CNNEncoder, StandardCNNEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.purposes.verification import (
    ImpossibilityResult,
    find_conflicting_attributes,
    impossibility_bound,
)
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")


def extract_all(
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


def train_model(
    encoder: torch.nn.Module,
    purposes: list,
    train_loader: DataLoader,
    val_loader: DataLoader,
    repr_dim: int,
    config: TrainerConfig,
    device: str,
    label: str,
) -> None:
    """Train encoder with PCRL trainer."""
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    elapsed = time.time() - t0
    print(f"  {label}: {state.epoch + 1} epochs in {elapsed:.0f}s")

    eval_metrics = trainer.evaluate(val_loader)
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")


def run_impossibility_analysis(
    pcrl_encoder: torch.nn.Module,
    laftr_encoder: torch.nn.Module,
    test_loader: DataLoader,
    purposes: list,
    device: str,
) -> list[dict]:
    """Run the full impossibility analysis."""
    conflicts = find_conflicting_attributes(purposes)
    purpose_names = [p.name for p in purposes]

    print(f"\n  Found {len(conflicts)} conflicting (attr, need, forbid) triples:")
    for attr, need, forbid in conflicts:
        print(f"    {attr}: needed by {need}, forbidden by {forbid}")

    # Deduplicate: for each (attr, needed_by), pick one forbidden_by
    # to show the key conflicts. Keep all for the CSV.
    seen_need_attr = set()
    key_conflicts = []
    for attr, need, forbid in conflicts:
        key = (attr, need)
        if key not in seen_need_attr:
            seen_need_attr.add(key)
            key_conflicts.append((attr, need, forbid))

    # Extract PCRL representations for all purposes
    print("\n  Extracting PCRL representations...")
    pcrl_reprs = {}
    pcrl_tasks = {}
    pcrl_sens = {}
    for idx, p in enumerate(purposes):
        reprs, tasks, sens = extract_all(pcrl_encoder, test_loader, idx, device)
        pcrl_reprs[p.name] = reprs
        pcrl_tasks[p.name] = tasks
        pcrl_sens[p.name] = sens

    # Extract LAFTR representations (purpose_idx=0 ignored by StandardCNNEncoder)
    print("  Extracting LAFTR representations...")
    laftr_reprs, laftr_tasks, laftr_sens = extract_all(
        laftr_encoder, test_loader, 0, device,
    )

    # Run MINE for all needed (purpose, attr) pairs
    mine = MINEstimator(hidden_dim=256, num_steps=2000, batch_size=512)
    mi_cache: dict[str, float] = {}

    def get_mi(reprs: np.ndarray, labels: np.ndarray, key: str) -> float:
        if key not in mi_cache:
            result = mine.estimate(reprs, labels, device=device)
            mi_cache[key] = result.mi_estimate
        return mi_cache[key]

    # Build combined labels (from any purpose's extraction — they're the same test data)
    all_labels: dict[str, np.ndarray] = {}
    ref_sens = pcrl_sens[purpose_names[0]]
    ref_tasks = pcrl_tasks[purpose_names[0]]
    # Merge task labels and sensitive attrs — some attrs appear in both
    for k, v in ref_sens.items():
        all_labels[k] = v
    for k, v in ref_tasks.items():
        all_labels[k] = v

    rows = []
    print("\n  Computing MI estimates (this may take a few minutes)...")

    for attr, need, forbid in conflicts:
        if attr not in all_labels:
            continue
        labels = all_labels[attr]
        num_classes = int(labels.max()) + 1

        # Entropy H(A)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy_a = -float(np.sum(probs * np.log(probs + 1e-12)))

        # MI: PCRL task-requiring purpose
        mi_pcrl_need = get_mi(
            pcrl_reprs[need], labels, f"pcrl_{need}_{attr}",
        )

        # MI: PCRL forbidding purpose (actual leakage)
        mi_pcrl_forbid = get_mi(
            pcrl_reprs[forbid], labels, f"pcrl_{forbid}_{attr}",
        )

        # MI: LAFTR (single representation)
        mi_laftr = get_mi(
            laftr_reprs, labels, f"laftr_{attr}",
        )

        # Bounds
        single_rep_bound = impossibility_bound(mi_pcrl_need, num_classes, entropy_a)
        laftr_acc_bound = impossibility_bound(mi_laftr, num_classes, entropy_a)
        pcrl_acc_bound = impossibility_bound(mi_pcrl_forbid, num_classes, entropy_a)

        row = {
            "attribute": attr,
            "needed_by": need,
            "forbidden_by": forbid,
            "num_classes": num_classes,
            "entropy_a_bits": entropy_a / np.log(2),
            "mi_needed_bits": mi_pcrl_need / np.log(2),
            "single_rep_min_leak": single_rep_bound,
            "laftr_mi_bits": mi_laftr / np.log(2),
            "laftr_leak": laftr_acc_bound,
            "pcrl_mi_bits": mi_pcrl_forbid / np.log(2),
            "pcrl_leak": pcrl_acc_bound,
        }
        rows.append(row)

        print(
            f"    {attr} ({need}->{forbid}): "
            f"Bound={single_rep_bound:.1%}  LAFTR={laftr_acc_bound:.1%}  "
            f"PCRL={pcrl_acc_bound:.1%}"
        )

    return rows


def print_impossibility_table(rows: list[dict]) -> None:
    """Print the impossibility analysis table."""
    print(f"\n{'=' * 120}")
    print("CelebA — Single-Representation Impossibility Analysis")
    print(f"{'=' * 120}")
    header = (
        f"{'Attribute':<14} {'Needed by':<28} {'Forbidden by':<28} "
        f"{'SR Min Leak':>11} {'LAFTR Leak':>10} {'PCRL Leak':>10}"
    )
    print(header)
    print("-" * 120)

    for r in rows:
        print(
            f"{r['attribute']:<14} {r['needed_by']:<28} {r['forbidden_by']:<28} "
            f"{r['single_rep_min_leak']:>10.1%} {r['laftr_leak']:>10.1%} "
            f"{r['pcrl_leak']:>10.1%}"
        )

    print("-" * 120)
    if rows:
        avg_sr = sum(r["single_rep_min_leak"] for r in rows) / len(rows)
        avg_laftr = sum(r["laftr_leak"] for r in rows) / len(rows)
        avg_pcrl = sum(r["pcrl_leak"] for r in rows) / len(rows)
        print(
            f"{'Average':<14} {'':28} {'':28} "
            f"{avg_sr:>10.1%} {avg_laftr:>10.1%} {avg_pcrl:>10.1%}"
        )
    print(f"{'=' * 120}")


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attribute", "needed_by", "forbidden_by", "num_classes",
        "entropy_a_bits", "mi_needed_bits", "single_rep_min_leak",
        "laftr_mi_bits", "laftr_leak", "pcrl_mi_bits", "pcrl_leak",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items()
            })
    print(f"Saved {path}")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    purposes = get_celeba_purposes()
    max_samples = 15000  # Keep runtime manageable

    print("Loading CelebA data...")
    train_ds = CelebADataset(purposes, root="data/celeba", split="train", max_samples=max_samples)
    val_ds = CelebADataset(purposes, root="data/celeba", split="val", max_samples=max_samples // 3)
    test_ds = CelebADataset(purposes, root="data/celeba", split="test", max_samples=max_samples // 3)
    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    bs = 64
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    repr_dim = 64
    purpose_emb_dim = 32

    # ── Train PCRL (purpose-conditioned CNN) ──────────────────────────────
    print("\n--- Training PCRL (purpose-conditioned) ---")
    torch.manual_seed(42)
    pcrl_encoder = CNNEncoder(
        repr_dim=repr_dim, num_purposes=len(purposes),
        purpose_emb_dim=purpose_emb_dim, dropout=0.3,
    )
    pcrl_config = TrainerConfig(
        batch_size=bs, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.5, lambda_verify=0.2,
        auditor_steps=5, epochs=30,
        weight_decay=1e-4, early_stopping_patience=10,
        confusion_type="entropy",
    )
    train_model(
        pcrl_encoder, purposes, train_loader, val_loader,
        repr_dim, pcrl_config, device, "PCRL",
    )

    # ── Train LAFTR (single-representation baseline) ──────────────────────
    print("\n--- Training LAFTR (single-representation) ---")
    torch.manual_seed(42)
    laftr_encoder = StandardCNNEncoder(repr_dim=repr_dim, dropout=0.3)

    # LAFTR uses a SINGLE representation with adversarial training.
    # We train it on the first purpose's tasks/attrs as a representative.
    # Its single h must serve all purposes — the impossibility scenario.
    laftr_config = TrainerConfig(
        batch_size=bs, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.5, lambda_verify=0.2,
        auditor_steps=5, epochs=30,
        weight_decay=1e-4, early_stopping_patience=10,
        confusion_type="entropy",
    )
    train_model(
        laftr_encoder, purposes, train_loader, val_loader,
        repr_dim, laftr_config, device, "LAFTR",
    )

    # ── Impossibility analysis ────────────────────────────────────────────
    print("\n--- Running impossibility analysis ---")
    rows = run_impossibility_analysis(
        pcrl_encoder, laftr_encoder, test_loader, purposes, device,
    )

    print_impossibility_table(rows)
    save_csv(rows, project_root / "results" / "celeba" / "impossibility_analysis.csv")

    print(f"\n{'=' * 70}")
    print("DONE — CelebA impossibility analysis complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

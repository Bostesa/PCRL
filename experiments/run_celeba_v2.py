#!/usr/bin/env python3
"""CelebA experiment v2 — per-purpose projection heads.

FiLM conditioning fails on CelebA because it scales entire CNN channels
uniformly, so it can't selectively suppress attributes (like Male) that
are entangled with task-relevant features (like Smiling) in the same
channels. When 4 out of 5 purposes try to suppress Male, the encoder
collapses to constant output.

Fix: CNNPurposeProjectionEncoder uses a shared conv backbone with
separate MLP projection heads per purpose. Each head learns independent
linear combinations of all conv features, enabling purpose-specific
information routing without entanglement.

Results saved to results/celeba/baseline_v2.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.celeba import CelebADataset, get_celeba_purposes
from pcrl.evaluation.certificates import generate_report, print_compliance_table
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.cnn_encoder import CNNPurposeProjectionEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ── Config ──────────────────────────────────────────────────────────────
REPR_DIM = 128
CONV_CHANNELS = (32, 64, 128)
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3

LAMBDA_ADV = 0.5
LAMBDA_VERIFY = 0.3
AUDITOR_STEPS = 5
EPOCHS = 50
WARMUP_EPOCHS = 5

AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3

MAX_TRAIN = 10000
MAX_VAL = 3000
MAX_TEST = 3000


class MultiTaskHead(nn.Module):
    """Task head returning a dict for multi-task purposes."""

    def __init__(self, heads: dict[str, nn.Module]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Purposes ─────────────────────────────────────────────────────────
    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    print(f"Purposes ({len(purposes)}):")
    for p in purposes:
        print(f"  {p.name}: tasks={p.allowed_tasks}, disallowed={p.disallowed_attrs}")

    # ── Data ─────────────────────────────────────────────────────────────
    print("\nLoading CelebA dataset...")
    train_ds = CelebADataset(purposes, root="data/celeba", split="train", max_samples=MAX_TRAIN)
    val_ds = CelebADataset(purposes, root="data/celeba", split="val", max_samples=MAX_VAL)
    test_ds = CelebADataset(purposes, root="data/celeba", split="test", max_samples=MAX_TEST)
    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    majority = {}
    for attr_name, labels in test_ds.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        majority[attr_name] = counts.max().item() / len(labels)
    print("\nMajority baselines:")
    for attr, baseline in sorted(majority.items()):
        print(f"  {attr}: {baseline:.1%}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    # ── Model ────────────────────────────────────────────────────────────
    # Per-purpose projection heads instead of FiLM — each purpose gets its
    # own MLP from flattened conv features, enabling selective routing.
    encoder = CNNPurposeProjectionEncoder(
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        conv_channels=CONV_CHANNELS,
        dropout=DROPOUT,
        backbone_grad_scale=1.0,  # no scaling — lambda=0.5 is gentle enough
    )
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder: {total_params:,} parameters")

    task_heads: dict[str, nn.Module] = {}
    for p in purposes:
        if len(p.allowed_tasks) == 1:
            task_name = p.allowed_tasks[0]
            output_dim = p.allowed_task_dims.get(task_name, 2)
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        else:
            sub = {}
            for task_name in p.allowed_tasks:
                output_dim = p.allowed_task_dims.get(task_name, 2)
                sub[task_name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
            task_heads[p.name] = MultiTaskHead(sub)

    auditors: dict[str, nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=AUDITOR_HIDDEN,
            num_layers=AUDITOR_LAYERS,
        )

    # ── Training ─────────────────────────────────────────────────────────
    # Standard minimax, all purposes per batch — no sequential needed since
    # each purpose has its own projection head (no shared representation
    # that would collapse under conflicting gradients).
    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR,
        lr_auditor=LR,
        lambda_adv=LAMBDA_ADV,
        lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=None,
        log_interval=100,
        confusion_type="entropy",
        warmup_epochs=WARMUP_EPOCHS,
        gradient_reversal=False,
        sequential_purposes=False,
        checkpoint_dir=str(project_root / "checkpoints" / "celeba_v2"),
    )

    trainer = PCRLTrainer(
        encoder=encoder,
        task_heads=task_heads,
        auditors=auditors,
        config=config,
        purpose_registry=registry,
        device=device,
    )

    print("\n" + "=" * 60)
    print("TRAINING PCRL (CelebA v2 — per-purpose projections)")
    print(f"  repr_dim={REPR_DIM}, lambda_adv={LAMBDA_ADV}, lambda_verify={LAMBDA_VERIFY}")
    print(f"  K={AUDITOR_STEPS}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"  mode=minimax, all purposes per batch")
    print("=" * 60)

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    print(f"\nTraining complete — epoch {state.epoch + 1}, {train_time:.0f}s")

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    eval_metrics = trainer.evaluate(test_loader)
    print(f"\nTest loss: {eval_metrics.loss:.4f}")
    print(f"Task accuracies:")
    for task, acc in sorted(eval_metrics.task_accuracy.items()):
        print(f"  {task}: {acc:.1%}")
    print(f"\nAuditor accuracies (training auditors):")
    for attr, acc in sorted(eval_metrics.auditor_accuracy.items()):
        baseline = majority.get(attr, 0.5)
        delta = acc - baseline
        print(f"  {attr}: {acc:.1%} (baseline={baseline:.1%}, delta={delta:+.1%})")

    # ── Compliance audit ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPLIANCE AUDIT")
    print("=" * 60)

    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )
    print_compliance_table(reports)

    num_pass = 0
    total = len(reports)
    print("\nAdjusted compliance (delta < 2%, R² < 5%):")
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        adj_pass = delta < 0.02 and r.linear_r2 < 0.05
        status = "PASS" if adj_pass else "FAIL"
        if adj_pass:
            num_pass += 1
        print(f"  {r.purpose_name:30s} / {r.attr_name:12s}  "
              f"delta={delta:+.1%}  R²={r.linear_r2:.4f}  {status}")

    print(f"\nOverall: {num_pass}/{total} pairs pass adjusted compliance")

    # ── Save results ─────────────────────────────────────────────────────
    out_dir = project_root / "results" / "celeba"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_v2.csv"

    fieldnames = [
        "purpose", "attribute", "linear_r2", "linear_pass",
        "empirical_best_acc", "majority_baseline", "delta",
        "adj_pass", "nonlinear_bound", "train_time_s",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            delta = r.empirical_best_acc - r.majority_proportion
            writer.writerow({
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "linear_r2": round(r.linear_r2, 6),
                "linear_pass": r.linear_certified,
                "empirical_best_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(r.majority_proportion, 4),
                "delta": round(delta, 4),
                "adj_pass": delta < 0.02 and r.linear_r2 < 0.05,
                "nonlinear_bound": round(r.nonlinear_bound, 4) if r.nonlinear_bound else None,
                "train_time_s": round(train_time, 1),
            })

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

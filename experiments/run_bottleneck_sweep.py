#!/usr/bin/env python3
"""HAR bottleneck sweep: vary repr_dim and measure privacy-utility tradeoff.

Trains PCRL on UCI HAR with repr_dim in [16, 32, 64, 128].
For each, records:
  - activity accuracy (6-class)
  - subject_id delta (empirical best acc - majority baseline)
  - subject_id R² (linear certificate)

Plots: x = subject_id delta, y = activity accuracy, with repr_dim labeled.
Shows how the information bottleneck controls the privacy-utility tradeoff.

Saves to results/har_real/bottleneck_sweep.csv and .png.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass
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

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.har import HARDataset, get_har_purposes, N_SUBJECTS, N_ACTIVITIES
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

# Architecture (same as run_har_real.py except repr_dim varies)
HIDDEN_DIMS = [128, 128]
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 2.0
LAMBDA_VERIFY = 1.0
AUDITOR_STEPS = 20
EPOCHS = 100
PATIENCE = None  # No early stopping (same as run_har_real.py)
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3

REPR_DIMS = [16, 32, 64, 128]


@dataclass
class BottleneckResult:
    repr_dim: int
    activity_accuracy: float
    is_active_accuracy: float
    subject_delta: float
    subject_r2: float
    subject_emp_acc: float
    subject_majority: float
    compliance_pass: int
    compliance_total: int
    train_time: float


def run_one_dim(
    repr_dim: int,
    purposes,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_dataset: HARDataset,
    input_dim: int,
    device: str,
) -> BottleneckResult:
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=repr_dim,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=AUDITOR_HIDDEN,
            num_layers=AUDITOR_LAYERS,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=PATIENCE,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    eval_metrics = trainer.evaluate(test_loader)
    reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader, purpose_registry=registry, device=device,
    )

    # Majority baselines
    majority_baselines = {}
    for attr_name, labels in test_dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        majority_baselines[attr_name] = counts.max().item() / len(labels)

    # Find subject_id results (worst case across purposes)
    subject_delta = -1.0
    subject_r2 = 0.0
    subject_emp = 0.0
    subject_majority = majority_baselines.get("subject_id", 1.0 / N_SUBJECTS)
    pass_count = 0
    total = 0

    for r in reports:
        baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
        delta = r.empirical_best_acc - baseline
        adj_pass = delta < 0.02 and r.linear_r2 < 0.05
        total += 1
        if adj_pass:
            pass_count += 1

        if r.attr_name == "subject_id":
            if delta > subject_delta:
                subject_delta = delta
                subject_r2 = r.linear_r2
                subject_emp = r.empirical_best_acc

    act_acc = eval_metrics.task_accuracy.get("activity", 0.0)
    ia_acc = eval_metrics.task_accuracy.get("is_active", 0.0)

    print(f"  repr_dim={repr_dim:>3}: activity={act_acc:.1%}, "
          f"subj_delta={subject_delta:+.1%}, R²={subject_r2:.4f}, "
          f"pass={pass_count}/{total}, epoch={state.epoch + 1}, {train_time:.0f}s")

    return BottleneckResult(
        repr_dim=repr_dim,
        activity_accuracy=act_acc,
        is_active_accuracy=ia_acc,
        subject_delta=subject_delta,
        subject_r2=subject_r2,
        subject_emp_acc=subject_emp,
        subject_majority=subject_majority,
        compliance_pass=pass_count,
        compliance_total=total,
        train_time=train_time,
    )


def save_csv(results: list[BottleneckResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "repr_dim", "activity_accuracy", "is_active_accuracy",
            "subject_delta", "subject_r2", "subject_emp_acc",
            "subject_majority", "compliance_pass", "compliance_total",
            "train_time_s",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "repr_dim": r.repr_dim,
                "activity_accuracy": round(r.activity_accuracy, 4),
                "is_active_accuracy": round(r.is_active_accuracy, 4),
                "subject_delta": round(r.subject_delta, 4),
                "subject_r2": round(r.subject_r2, 6),
                "subject_emp_acc": round(r.subject_emp_acc, 4),
                "subject_majority": round(r.subject_majority, 4),
                "compliance_pass": r.compliance_pass,
                "compliance_total": r.compliance_total,
                "train_time_s": round(r.train_time, 1),
            })
    print(f"Saved {path}")


def plot_bottleneck(results: list[BottleneckResult], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    deltas = [r.subject_delta * 100 for r in results]
    accs = [r.activity_accuracy * 100 for r in results]
    dims = [r.repr_dim for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deltas, accs, "o-", color="#2196F3", linewidth=2, markersize=12, zorder=5)

    for d, a, dim in zip(deltas, accs, dims):
        ax.annotate(f"d={dim}", (d, a), textcoords="offset points",
                    xytext=(10, 8), fontsize=10, fontweight="bold")

    ax.set_xlabel("Subject ID Delta (emp_acc - majority, %)", fontsize=12)
    ax.set_ylabel("Activity Recognition Accuracy (%)", fontsize=12)
    ax.set_title("Information Bottleneck: repr_dim vs Privacy-Utility", fontsize=13, pad=12)
    ax.grid(True, alpha=0.3)

    ax.axvline(x=2.0, color="#4CAF50", linestyle="--", alpha=0.7, label="2% compliance threshold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Bottleneck sweep: repr_dim = {REPR_DIMS}")
    print(f"Config: λ_adv={LAMBDA_ADV}, λ_ver={LAMBDA_VERIFY}, K={AUDITOR_STEPS}, E={EPOCHS}")

    purposes = get_har_purposes()
    print("Loading UCI HAR dataset...")
    train_ds = HARDataset(purposes=purposes, split="train")
    val_ds = HARDataset(purposes=purposes, split="val")
    test_ds = HARDataset(purposes=purposes, split="test")

    input_dim = train_ds.info.num_features
    print(f"HAR: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
          f"Features={input_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    results = []
    for dim in REPR_DIMS:
        print(f"\n--- repr_dim = {dim} ---")
        r = run_one_dim(dim, purposes, train_loader, val_loader, test_loader,
                        test_ds, input_dim, device)
        results.append(r)

    # Summary table
    print("\n" + "=" * 100)
    print("BOTTLENECK SWEEP RESULTS (HAR)")
    print("=" * 100)
    print(f"{'repr_dim':>10} {'Activity':>10} {'IsActive':>10} "
          f"{'Subj Δ':>10} {'Subj R²':>10} {'Pass':>8} {'Time':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r.repr_dim:>10} {r.activity_accuracy:>9.1%} {r.is_active_accuracy:>9.1%} "
              f"{r.subject_delta:>+9.1%} {r.subject_r2:>10.4f} "
              f"{r.compliance_pass:>4}/{r.compliance_total} {r.train_time:>7.0f}s")
    print("=" * 100)

    results_dir = project_root / "results" / "har_real"
    save_csv(results, results_dir / "bottleneck_sweep.csv")
    plot_bottleneck(results, results_dir / "bottleneck_sweep.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

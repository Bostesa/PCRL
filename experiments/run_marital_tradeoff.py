#!/usr/bin/env python3
"""Marital status tradeoff curve: sweep lambda_adv and plot the Pareto frontier.

Trains PCRL on Adult with income_prediction AND employment_analysis.
Sweeps lambda_adv in [10, 25, 50, 100, 200] and for each records:
  - income task accuracy
  - marital_status delta (empirical best acc - majority baseline)
  - all other compliance pairs

Plots: x = marital_status delta, y = income accuracy, showing the exact
tradeoff between suppressing marital_status and maintaining task performance.

Saves to results/adult/marital_status_tradeoff.csv and .png.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass, field
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

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

# Architecture matches run_adult.py
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 256
AUDITOR_STEPS = 10
EPOCHS = 200
PATIENCE = 15

LAMBDA_VALUES = [10, 25, 50, 100, 200]


def get_two_purposes() -> list[PurposeSpec]:
    """income_prediction + employment_analysis (marital_status is disallowed for employment)."""
    all_p = get_adult_purposes()
    return [all_p[0], all_p[1]]  # income_prediction, employment_analysis


@dataclass
class TradeoffResult:
    lambda_adv: float
    income_accuracy: float
    occupation_accuracy: float
    marital_delta: float
    marital_r2: float
    marital_emp_acc: float
    marital_majority: float
    all_pairs: list[dict] = field(default_factory=list)
    train_time: float = 0.0


def run_one_lambda(
    lambda_adv: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_dataset: AdultDataset,
    input_dim: int,
    device: str,
) -> TradeoffResult:
    purposes = get_two_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
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
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR, lr_auditor=LR,
        lambda_adv=lambda_adv,
        lambda_verify=lambda_adv,  # keep lambda_verify = lambda_adv
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

    # Compute majority baselines
    majority_baselines = {}
    for attr_name, labels in test_dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        majority_baselines[attr_name] = counts.max().item() / len(labels)

    # Extract marital_status result
    marital_report = None
    all_pairs = []
    for r in reports:
        baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
        delta = r.empirical_best_acc - baseline
        adj_pass = delta < 0.02 and r.linear_r2 < 0.05
        pair = {
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "emp_acc": r.empirical_best_acc,
            "majority": baseline,
            "delta": delta,
            "r2": r.linear_r2,
            "pass": adj_pass,
        }
        all_pairs.append(pair)
        if r.attr_name == "marital_status":
            marital_report = r

    income_acc = eval_metrics.task_accuracy.get("income", 0.0)
    occ_acc = eval_metrics.task_accuracy.get("occupation_group", 0.0)

    marital_majority = majority_baselines.get("marital_status", 0.5)
    if marital_report is not None:
        marital_delta = marital_report.empirical_best_acc - marital_majority
        marital_r2 = marital_report.linear_r2
        marital_emp = marital_report.empirical_best_acc
    else:
        marital_delta = 0.0
        marital_r2 = 0.0
        marital_emp = marital_majority

    print(f"  lambda={lambda_adv:>3}: income={income_acc:.1%}, "
          f"occ={occ_acc:.1%}, marital_delta={marital_delta:+.1%}, "
          f"R²={marital_r2:.4f}, epoch={state.epoch + 1}, {train_time:.0f}s")

    return TradeoffResult(
        lambda_adv=lambda_adv,
        income_accuracy=income_acc,
        occupation_accuracy=occ_acc,
        marital_delta=marital_delta,
        marital_r2=marital_r2,
        marital_emp_acc=marital_emp,
        marital_majority=marital_majority,
        all_pairs=all_pairs,
        train_time=train_time,
    )


def save_csv(results: list[TradeoffResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # One row per (lambda, purpose, attribute) pair
    rows = []
    for r in results:
        for pair in r.all_pairs:
            rows.append({
                "lambda_adv": r.lambda_adv,
                "income_accuracy": round(r.income_accuracy, 4),
                "occupation_accuracy": round(r.occupation_accuracy, 4),
                "marital_delta": round(r.marital_delta, 4),
                "marital_r2": round(r.marital_r2, 6),
                "purpose": pair["purpose"],
                "attribute": pair["attribute"],
                "emp_acc": round(pair["emp_acc"], 4),
                "majority": round(pair["majority"], 4),
                "delta": round(pair["delta"], 4),
                "r2": round(pair["r2"], 6),
                "adj_pass": pair["pass"],
                "train_time_s": round(r.train_time, 1),
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def plot_tradeoff(results: list[TradeoffResult], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    deltas = [r.marital_delta * 100 for r in results]
    incomes = [r.income_accuracy * 100 for r in results]
    lambdas = [r.lambda_adv for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deltas, incomes, "o-", color="#2196F3", linewidth=2, markersize=10, zorder=5)

    for d, inc, lam in zip(deltas, incomes, lambdas):
        ax.annotate(f"λ={lam}", (d, inc), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold")

    ax.set_xlabel("Marital Status Delta (emp_acc - majority, %)", fontsize=12)
    ax.set_ylabel("Income Prediction Accuracy (%)", fontsize=12)
    ax.set_title("Privacy-Utility Tradeoff: Marital Status Suppression", fontsize=13, pad=12)
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier shading
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
    print(f"Architecture: hidden={HIDDEN_DIMS}, repr_dim={REPR_DIM}")
    print(f"Lambda sweep: {LAMBDA_VALUES}")

    purposes = get_two_purposes()
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False)

    input_dim = train_ds.info.num_features
    print(f"Adult: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
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
    for lam in LAMBDA_VALUES:
        print(f"\n--- lambda_adv = {lam} ---")
        r = run_one_lambda(lam, train_loader, val_loader, test_loader, test_ds, input_dim, device)
        results.append(r)

    # Summary table
    print("\n" + "=" * 90)
    print("MARITAL STATUS TRADEOFF RESULTS")
    print("=" * 90)
    print(f"{'Lambda':>8} {'Income Acc':>12} {'Occ Acc':>10} "
          f"{'Marital Δ':>11} {'Marital R²':>11} {'Time':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r.lambda_adv:>8} {r.income_accuracy:>11.1%} {r.occupation_accuracy:>9.1%} "
              f"{r.marital_delta:>+10.1%} {r.marital_r2:>11.4f} {r.train_time:>7.0f}s")

    # All compliance pairs
    print(f"\n{'Lambda':>8} {'Purpose':<22} {'Attribute':<16} "
          f"{'Emp Acc':>9} {'Majority':>9} {'Delta':>8} {'R²':>8} {'Pass':>6}")
    print("-" * 90)
    for r in results:
        for p in r.all_pairs:
            status = "PASS" if p["pass"] else "FAIL"
            print(f"{r.lambda_adv:>8} {p['purpose']:<22} {p['attribute']:<16} "
                  f"{p['emp_acc']:>8.1%} {p['majority']:>8.1%} "
                  f"{p['delta']:>+7.1%} {p['r2']:>8.4f} {status:>6}")
    print("=" * 90)

    results_dir = project_root / "results" / "adult"
    save_csv(results, results_dir / "marital_status_tradeoff.csv")
    plot_tradeoff(results, results_dir / "marital_status_tradeoff.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

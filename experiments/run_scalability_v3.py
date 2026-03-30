#!/usr/bin/env python3
"""Scalability experiment v3: lower lambdas to fix v2's 0% compliance.

v2 used lambda_adv=50, lambda_verify=50 which was too aggressive for the
256-dim encoder, resulting in 0% compliance and ~50% task accuracy.

v3 uses lambda_adv=10, lambda_verify=10, epochs=100, patience=20 — giving
the encoder more room to learn task-relevant features before the adversarial
pressure dominates.

Saves to results/adult/scalability_v3.csv and results/adult/scalability_v3.png.
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

# ── Architecture: SAME as main experiment (run_adult.py) ──────────────────
HIDDEN_DIMS = [256, 256]
REPR_DIM = 128
PURPOSE_EMB_DIM = 64
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 256
# ── Re-tuned hyperparameters (lower lambda, more epochs) ─────────────────
LAMBDA_ADV = 10.0
LAMBDA_VERIFY = 10.0
AUDITOR_STEPS = 10
EPOCHS = 100
PATIENCE = 20


# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE SETS
# ═══════════════════════════════════════════════════════════════════════════

def get_purposes_2() -> list[PurposeSpec]:
    return [
        PurposeSpec(
            name="income_prediction",
            allowed_tasks=["income"], disallowed_attrs=["race", "sex"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 5, "sex": 2},
        ),
        PurposeSpec(
            name="employment_analysis",
            allowed_tasks=["occupation_group"],
            disallowed_attrs=["race", "age_group", "marital_status"],
            task_type="classification",
            allowed_task_dims={"occupation_group": 6},
            disallowed_attr_dims={"race": 5, "age_group": 4, "marital_status": 2},
        ),
    ]


def get_purposes_3() -> list[PurposeSpec]:
    return get_adult_purposes()


def get_purposes_5() -> list[PurposeSpec]:
    return [
        *get_purposes_2(),
        PurposeSpec(
            name="education_assessment",
            allowed_tasks=["education_level"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"sex": 2, "race": 5, "income": 2},
        ),
        PurposeSpec(
            name="fair_lending",
            allowed_tasks=["income"],
            disallowed_attrs=["race", "sex", "age_group"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 5, "sex": 2, "age_group": 4},
        ),
        PurposeSpec(
            name="education_equity",
            allowed_tasks=["education_level"],
            disallowed_attrs=["sex", "race", "marital_status"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"sex": 2, "race": 5, "marital_status": 2},
        ),
    ]


def get_purposes_8() -> list[PurposeSpec]:
    return [
        *get_purposes_5(),
        PurposeSpec(
            name="gender_blind_employment",
            allowed_tasks=["occupation_group"],
            disallowed_attrs=["sex"],
            task_type="classification",
            allowed_task_dims={"occupation_group": 6},
            disallowed_attr_dims={"sex": 2},
        ),
        PurposeSpec(
            name="insurance_risk",
            allowed_tasks=["income"],
            disallowed_attrs=["marital_status", "race"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"marital_status": 2, "race": 5},
        ),
        PurposeSpec(
            name="age_blind_education",
            allowed_tasks=["education_level"],
            disallowed_attrs=["age_group", "race"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"age_group": 4, "race": 5},
        ),
    ]


PURPOSE_SETS = {2: get_purposes_2, 3: get_purposes_3, 5: get_purposes_5, 8: get_purposes_8}


@dataclass
class ScalabilityResult:
    num_purposes: int
    avg_task_accuracy: float
    compliance_rate: float
    total_pairs: int
    passing_pairs: int
    train_time: float


def run_scalability(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: str,
) -> list[ScalabilityResult]:
    results = []

    for n_purposes in [2, 3, 5, 8]:
        print(f"\n--- {n_purposes} purposes ---")
        purposes = PURPOSE_SETS[n_purposes]()
        registry = PurposeRegistry()
        for p in purposes:
            registry.register(p)

        for i, p in enumerate(purposes):
            print(f"  [{i}] {p.name}: tasks={p.allowed_tasks}, "
                  f"disallowed={p.disallowed_attrs}")

        torch.manual_seed(42)
        encoder = PurposeConditionedEncoder(
            input_dim=input_dim,
            hidden_dims=HIDDEN_DIMS,
            repr_dim=REPR_DIM,
            num_purposes=n_purposes,
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
            lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
            auditor_steps=AUDITOR_STEPS,
            epochs=EPOCHS, weight_decay=1e-4,
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
        print(f"  Trained {state.epoch + 1} epochs in {train_time:.0f}s")

        task_accs = list(eval_metrics.task_accuracy.values())
        avg_task_acc = sum(task_accs) / len(task_accs) if task_accs else 0.0
        for task, acc in eval_metrics.task_accuracy.items():
            print(f"    {task}: {acc:.1%}")

        reports = generate_report(
            encoder=encoder, train_loader=train_loader,
            test_loader=test_loader, purpose_registry=registry,
            device=device,
        )

        passing = sum(1 for r in reports
                       if (r.empirical_best_acc - r.majority_proportion) < 0.02
                       and r.linear_r2 < 0.05)
        total = len(reports)
        rate = passing / total if total > 0 else 0.0

        print(f"  Avg task acc: {avg_task_acc:.1%}, "
              f"Compliance: {passing}/{total} ({rate:.0%})")

        results.append(ScalabilityResult(
            num_purposes=n_purposes,
            avg_task_accuracy=avg_task_acc,
            compliance_rate=rate,
            total_pairs=total,
            passing_pairs=passing,
            train_time=train_time,
        ))

    return results


def save_csv(results: list[ScalabilityResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_purposes", "avg_task_accuracy", "compliance_rate",
            "passing_pairs", "total_pairs", "train_time_s",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "num_purposes": r.num_purposes,
                "avg_task_accuracy": round(r.avg_task_accuracy, 4),
                "compliance_rate": round(r.compliance_rate, 4),
                "passing_pairs": r.passing_pairs,
                "total_pairs": r.total_pairs,
                "train_time_s": round(r.train_time, 1),
            })
    print(f"Saved {path}")


def plot_scalability(results: list[ScalabilityResult], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_purposes = [r.num_purposes for r in results]
    task_accs = [r.avg_task_accuracy * 100 for r in results]
    compliance = [r.compliance_rate * 100 for r in results]
    times = [r.train_time for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Number of Purposes", fontsize=12)
    ax1.set_ylabel("Accuracy / Compliance (%)", fontsize=12)

    line1, = ax1.plot(n_purposes, task_accs, "o-", color="#2196F3",
                       linewidth=2, markersize=8, label="Avg Task Accuracy")
    line2, = ax1.plot(n_purposes, compliance, "s--", color="#4CAF50",
                       linewidth=2, markersize=8, label="Compliance Rate")

    ax1.set_ylim(0, 105)
    ax1.set_xticks(n_purposes)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Time (s)", fontsize=12, color="#FF9800")
    line3, = ax2.plot(n_purposes, times, "^:", color="#FF9800",
                       linewidth=2, markersize=8, label="Training Time")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=10)

    ax1.set_title("PCRL Scalability v3 (256-dim, lambda=10)", fontsize=13, pad=12)
    ax1.grid(True, alpha=0.3)

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
    print(f"Architecture: hidden={HIDDEN_DIMS}, repr_dim={REPR_DIM}, "
          f"purpose_emb_dim={PURPOSE_EMB_DIM}")
    print(f"Lambdas: adv={LAMBDA_ADV}, verify={LAMBDA_VERIFY}, "
          f"epochs={EPOCHS}, patience={PATIENCE}")

    purposes_full = get_purposes_8()
    train_ds = AdultDataset(purposes=purposes_full, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes_full, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=purposes_full, root="data", split="test", download=False)

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

    results_dir = project_root / "results" / "adult"

    results = run_scalability(train_loader, val_loader, test_loader, input_dim, device)

    print("\n" + "=" * 80)
    print("SCALABILITY v3 RESULTS (Adult, 256-dim encoder, lambda=10)")
    print("=" * 80)
    print(f"{'Purposes':>10} {'Avg Task Acc':>14} {'Compliance':>14} "
          f"{'Pass/Total':>12} {'Train Time':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r.num_purposes:>10} {r.avg_task_accuracy:>13.1%} "
              f"{r.compliance_rate:>13.0%} "
              f"{r.passing_pairs:>5}/{r.total_pairs:<5} "
              f"{r.train_time:>11.0f}s")
    print("=" * 80)

    save_csv(results, results_dir / "scalability_v3.csv")
    plot_scalability(results, results_dir / "scalability_v3.png")

    print("\nDone!")


if __name__ == "__main__":
    main()

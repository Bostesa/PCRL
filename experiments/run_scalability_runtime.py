#!/usr/bin/env python3
"""Scalability and runtime experiments for PCRL.

1. SCALABILITY: Train PCRL with 2, 3, 5, 8 purposes on Adult.
   Report avg task accuracy, avg compliance pass rate, training time.
   Plot scalability curves.

2. RUNTIME: Compare inference cost of PCRL (1 model, purpose tokens)
   vs separate models (N encoders) vs LAFTR (1 model, retrain per purpose set).

Results saved to results/adult/scalability.{csv,png} and
results/runtime_comparison.csv.
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
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

# ── Architecture constants ───────────────────────────────────────────────
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 256
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 10
EPOCHS = 80
PATIENCE = 15


# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE SETS (2, 3, 5, 8)
# ═══════════════════════════════════════════════════════════════════════════

def get_purposes_2() -> list[PurposeSpec]:
    """2 purposes: income + employment."""
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
    """3 purposes: the standard Adult set."""
    return get_adult_purposes()


def get_purposes_5() -> list[PurposeSpec]:
    """5 purposes: split income + add demographic analysis."""
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
        PurposeSpec(
            name="education_assessment",
            allowed_tasks=["education_level"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"sex": 2, "race": 5, "income": 2},
        ),
        # New: income prediction hiding age too
        PurposeSpec(
            name="fair_lending",
            allowed_tasks=["income"],
            disallowed_attrs=["race", "sex", "age_group"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 5, "sex": 2, "age_group": 4},
        ),
        # New: education hiding marital status
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
    """8 purposes: full set with fine-grained privacy requirements."""
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
        # New: employment hiding only sex
        PurposeSpec(
            name="gender_blind_employment",
            allowed_tasks=["occupation_group"],
            disallowed_attrs=["sex"],
            task_type="classification",
            allowed_task_dims={"occupation_group": 6},
            disallowed_attr_dims={"sex": 2},
        ),
        # New: income hiding marital status (insurance use case)
        PurposeSpec(
            name="insurance_risk",
            allowed_tasks=["income"],
            disallowed_attrs=["marital_status", "race"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"marital_status": 2, "race": 5},
        ),
        # New: education hiding age (age discrimination)
        PurposeSpec(
            name="age_blind_education",
            allowed_tasks=["education_level"],
            disallowed_attrs=["age_group", "race"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"age_group": 4, "race": 5},
        ),
    ]


PURPOSE_SETS = {
    2: get_purposes_2,
    3: get_purposes_3,
    5: get_purposes_5,
    8: get_purposes_8,
}


# ═══════════════════════════════════════════════════════════════════════════
# SCALABILITY EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScalabilityResult:
    num_purposes: int
    avg_task_accuracy: float
    compliance_rate: float  # fraction of pairs passing
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
    """Run PCRL with increasing numbers of purposes."""
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

        # Task accuracies
        task_accs = list(eval_metrics.task_accuracy.values())
        avg_task_acc = sum(task_accs) / len(task_accs) if task_accs else 0.0
        for task, acc in eval_metrics.task_accuracy.items():
            print(f"    {task}: {acc:.1%}")

        # Compliance
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


def print_scalability_table(results: list[ScalabilityResult]) -> None:
    print("\n" + "=" * 80)
    print("SCALABILITY RESULTS (Adult)")
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


def save_scalability_csv(results: list[ScalabilityResult], path: Path) -> None:
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

    color1 = "#2196F3"
    color2 = "#4CAF50"
    color3 = "#FF9800"

    ax1.set_xlabel("Number of Purposes", fontsize=12)
    ax1.set_ylabel("Accuracy / Compliance (%)", fontsize=12, color="black")

    line1, = ax1.plot(n_purposes, task_accs, "o-", color=color1,
                       linewidth=2, markersize=8, label="Avg Task Accuracy")
    line2, = ax1.plot(n_purposes, compliance, "s--", color=color2,
                       linewidth=2, markersize=8, label="Compliance Rate")

    ax1.set_ylim(0, 105)
    ax1.set_xticks(n_purposes)
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Time (s)", fontsize=12, color=color3)
    line3, = ax2.plot(n_purposes, times, "^:", color=color3,
                       linewidth=2, markersize=8, label="Training Time")
    ax2.tick_params(axis="y", labelcolor=color3)

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=10)

    ax1.set_title("PCRL Scalability: Task Accuracy & Compliance vs Number of Purposes",
                   fontsize=13, pad=12)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeResult:
    method: str
    num_params: int
    inference_time_ms: float
    num_calls: int
    supports_new_purpose: bool
    training_time_per_purpose: float = 0.0


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_runtime_comparison(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: str,
) -> list[RuntimeResult]:
    """Compare inference cost of PCRL vs separate models vs LAFTR."""
    purposes = get_purposes_3()
    n_purposes = len(purposes)
    num_calls = 1000

    # Get a batch of data for inference timing
    sample_batch = next(iter(train_loader))
    x_sample = sample_batch["features"][:32].to(device)  # 32 samples

    results = []

    # ── 1. PCRL: one model, switch purpose tokens ────────────────────────
    print("\n--- PCRL inference ---")
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=n_purposes, purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film", dropout=DROPOUT,
    )

    pcrl_task_heads = {}
    pcrl_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        pcrl_task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        pcrl_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    pcrl_config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS, epochs=EPOCHS,
        weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
    )

    pcrl_trainer = PCRLTrainer(
        encoder=pcrl_encoder, task_heads=pcrl_task_heads,
        auditors=pcrl_auditors, config=pcrl_config,
        purpose_registry=registry, device=device,
    )

    t0 = time.time()
    pcrl_trainer.train(train_loader, val_loader=val_loader)
    pcrl_train_time = time.time() - t0

    pcrl_encoder.to(device).eval()
    pcrl_params = count_params(pcrl_encoder)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            pcrl_encoder(x_sample, 0)

    # Time inference
    t0 = time.time()
    with torch.no_grad():
        for i in range(num_calls):
            purpose_idx = i % n_purposes
            pcrl_encoder(x_sample, purpose_idx)
    pcrl_inf_time = (time.time() - t0) * 1000  # ms

    print(f"  Params: {pcrl_params:,}, Inference: {pcrl_inf_time:.1f}ms "
          f"({num_calls} calls), Train: {pcrl_train_time:.0f}s")

    results.append(RuntimeResult(
        method="PCRL",
        num_params=pcrl_params,
        inference_time_ms=pcrl_inf_time,
        num_calls=num_calls,
        supports_new_purpose=True,
        training_time_per_purpose=pcrl_train_time / n_purposes,
    ))

    # ── 2. Separate models: N independent encoders ───────────────────────
    print("\n--- Separate models inference ---")
    separate_encoders = []
    total_separate_params = 0
    total_separate_train_time = 0.0

    for i, p in enumerate(purposes):
        torch.manual_seed(42 + i)
        enc = StandardEncoder(
            input_dim=input_dim, hidden_dims=HIDDEN_DIMS,
            repr_dim=REPR_DIM, dropout=DROPOUT,
        )

        sep_registry = PurposeRegistry()
        sep_registry.register(p)

        task_heads = {p.name: TaskHead(
            repr_dim=REPR_DIM,
            output_dim=p.allowed_task_dims.get(p.allowed_tasks[0], 2),
        )}
        auditors = {p.name: MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )}

        config = TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
            auditor_steps=AUDITOR_STEPS, epochs=EPOCHS,
            weight_decay=1e-4, early_stopping_patience=PATIENCE,
            confusion_type="entropy",
        )

        trainer = PCRLTrainer(
            encoder=enc, task_heads=task_heads, auditors=auditors,
            config=config, purpose_registry=sep_registry, device=device,
        )

        t0 = time.time()
        trainer.train(train_loader, val_loader=val_loader)
        total_separate_train_time += time.time() - t0

        enc.to(device).eval()
        separate_encoders.append(enc)
        total_separate_params += count_params(enc)

    # Warm up
    with torch.no_grad():
        for enc in separate_encoders:
            for _ in range(5):
                enc(x_sample)

    # Time inference: cycle through encoders
    t0 = time.time()
    with torch.no_grad():
        for i in range(num_calls):
            enc_idx = i % n_purposes
            separate_encoders[enc_idx](x_sample)
    separate_inf_time = (time.time() - t0) * 1000

    print(f"  Total params: {total_separate_params:,} "
          f"({n_purposes} models x {count_params(separate_encoders[0]):,}), "
          f"Inference: {separate_inf_time:.1f}ms, "
          f"Train: {total_separate_train_time:.0f}s")

    results.append(RuntimeResult(
        method=f"Separate ({n_purposes} models)",
        num_params=total_separate_params,
        inference_time_ms=separate_inf_time,
        num_calls=num_calls,
        supports_new_purpose=False,
        training_time_per_purpose=total_separate_train_time / n_purposes,
    ))

    # ── 3. LAFTR: single model, retrain for each purpose set ─────────────
    print("\n--- LAFTR inference ---")
    torch.manual_seed(42)
    laftr_encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM, dropout=DROPOUT,
    )

    laftr_task_heads = {}
    laftr_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        laftr_task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        laftr_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    laftr_config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS, epochs=EPOCHS,
        weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
    )

    laftr_registry = PurposeRegistry()
    for p in purposes:
        laftr_registry.register(p)

    laftr_trainer = PCRLTrainer(
        encoder=laftr_encoder, task_heads=laftr_task_heads,
        auditors=laftr_auditors, config=laftr_config,
        purpose_registry=laftr_registry, device=device,
    )

    t0 = time.time()
    laftr_trainer.train(train_loader, val_loader=val_loader)
    laftr_train_time = time.time() - t0

    laftr_encoder.to(device).eval()
    laftr_params = count_params(laftr_encoder)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            laftr_encoder(x_sample)

    # Time inference (same representation for all purposes)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_calls):
            laftr_encoder(x_sample)
    laftr_inf_time = (time.time() - t0) * 1000

    print(f"  Params: {laftr_params:,}, Inference: {laftr_inf_time:.1f}ms, "
          f"Train: {laftr_train_time:.0f}s "
          f"(must retrain for new purpose sets)")

    results.append(RuntimeResult(
        method="LAFTR",
        num_params=laftr_params,
        inference_time_ms=laftr_inf_time,
        num_calls=num_calls,
        supports_new_purpose=False,
        training_time_per_purpose=laftr_train_time,
    ))

    return results


def print_runtime_table(results: list[RuntimeResult]) -> None:
    print("\n" + "=" * 105)
    print("RUNTIME COMPARISON")
    print("=" * 105)
    print(f"{'Method':<28} {'Parameters':>14} {'Inference (1000 calls)':>22} "
          f"{'Train/Purpose':>15} {'New Purpose w/o Retrain?':>25}")
    print("-" * 105)
    for r in results:
        new_p = "Yes" if r.supports_new_purpose else "No"
        print(f"{r.method:<28} {r.num_params:>14,} {r.inference_time_ms:>18.1f} ms "
              f"{r.training_time_per_purpose:>14.0f}s {new_p:>25}")
    print("=" * 105)


def save_runtime_csv(results: list[RuntimeResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "num_params", "inference_time_ms", "num_calls",
            "training_time_per_purpose_s", "supports_new_purpose",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "method": r.method,
                "num_params": r.num_params,
                "inference_time_ms": round(r.inference_time_ms, 1),
                "num_calls": r.num_calls,
                "training_time_per_purpose_s": round(r.training_time_per_purpose, 1),
                "supports_new_purpose": r.supports_new_purpose,
            })
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Load Adult data
    purposes_full = get_purposes_8()  # superset for data loading
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
    results_dir.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════
    # 1. SCALABILITY
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 70)
    print("# SCALABILITY EXPERIMENT")
    print("#" * 70)

    scalability_results = run_scalability(
        train_loader, val_loader, test_loader, input_dim, device,
    )
    print_scalability_table(scalability_results)
    save_scalability_csv(scalability_results, results_dir / "scalability.csv")
    plot_scalability(scalability_results, results_dir / "scalability.png")

    # ═════════════════════════════════════════════════════════════════════
    # 2. RUNTIME
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 70)
    print("# RUNTIME COMPARISON")
    print("#" * 70)

    runtime_results = run_runtime_comparison(
        train_loader, val_loader, input_dim, device,
    )
    print_runtime_table(runtime_results)
    save_runtime_csv(runtime_results, project_root / "results" / "runtime_comparison.csv")

    print("\n" + "=" * 60)
    print("DONE — scalability and runtime experiments complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run baseline comparisons for the Adult dataset.

Trains 3 baselines and compares against PCRL (loaded from checkpoint):
1. Standard (no privacy): Regular encoder, no adversarial training
2. Adversarial-only (LAFTR): Regular encoder WITH adversarial training
3. Purpose-only: Purpose-conditioned encoder, no adversarial training

Produces a comparison table and saves to results/adult/baseline_comparison.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress tqdm
import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    """No-op replacement for tqdm to suppress progress bars during baselines."""

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
from pcrl.evaluation.certificates import ComplianceReport, generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Results from a single baseline run."""

    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def compute_majority_baselines(test_dataset: AdultDataset) -> dict[str, float]:
    """Compute majority-class accuracy baselines for each sensitive attribute."""
    baselines: dict[str, float] = {}
    for attr_name, labels in test_dataset.sensitive_attrs.items():
        n = len(labels)
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / n
    return baselines


def make_task_heads(purposes, repr_dim: int = 64) -> dict[str, torch.nn.Module]:
    """Create task heads for each purpose."""
    task_heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
    return task_heads


def make_auditors(purposes, repr_dim: int = 64) -> dict[str, torch.nn.Module]:
    """Create multi-attribute auditors for each purpose."""
    auditors: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256,
            num_layers=3,
        )
    return auditors


def run_baseline(
    name: str,
    encoder: torch.nn.Module,
    purposes,
    registry: PurposeRegistry,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainerConfig,
    device: str,
) -> BaselineResult:
    """Train and evaluate a single baseline."""
    task_heads = make_task_heads(purposes)
    auditors = make_auditors(purposes)

    trainer = PCRLTrainer(
        encoder=encoder,
        task_heads=task_heads,
        auditors=auditors,
        config=config,
        purpose_registry=registry,
        device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    eval_metrics = trainer.evaluate(test_loader)
    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )

    print(f"  Completed: epoch {state.epoch + 1}, {train_time:.0f}s")
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    return BaselineResult(
        name=name,
        task_accuracies=eval_metrics.task_accuracy,
        reports=reports,
        train_time=train_time,
    )


def load_pcrl_result(
    purposes,
    registry: PurposeRegistry,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: str,
) -> BaselineResult:
    """Load PCRL results from the existing checkpoint."""
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=64,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
    )
    task_heads = make_task_heads(purposes)
    auditors = make_auditors(purposes)

    config = TrainerConfig(
        lambda_adv=50.0,
        lambda_verify=50.0,
        auditor_steps=10,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "adult"),
    )

    trainer = PCRLTrainer(
        encoder=encoder,
        task_heads=task_heads,
        auditors=auditors,
        config=config,
        purpose_registry=registry,
        device=device,
    )

    checkpoint_path = project_root / "checkpoints" / "adult" / "final.pt"
    trainer.load_checkpoint(checkpoint_path)

    eval_metrics = trainer.evaluate(test_loader)
    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )

    return BaselineResult(
        name="PCRL (ours)",
        task_accuracies=eval_metrics.task_accuracy,
        reports=reports,
        train_time=0.0,
    )


def print_comparison_table(
    results: list[BaselineResult],
    majority_baselines: dict[str, float],
) -> None:
    """Print the comparison table."""
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON")
    print("=" * 100)

    header = (
        f"{'Method':<28} {'Income Acc':>10} {'Race Δ':>9} {'Sex Δ':>9} "
        f"{'MarStat Δ':>10} {'AgeGrp Δ':>9} {'Inc Δ':>9} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        income_acc = result.task_accuracies.get("income", 0.0)

        # Compute deltas and count passes for all attribute pairs
        deltas: dict[str, float] = {}
        pass_count = 0
        total = 0
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            deltas[r.attr_name] = max(deltas.get(r.attr_name, -1.0), delta)
            total += 1
            if delta < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        race_d = deltas.get("race", 0.0)
        sex_d = deltas.get("sex", 0.0)
        marital_d = deltas.get("marital_status", 0.0)
        age_d = deltas.get("age_group", 0.0)
        inc_d = deltas.get("income", 0.0)

        print(
            f"{result.name:<28} {income_acc:>9.1%} "
            f"{race_d:>+8.1%} {sex_d:>+8.1%} "
            f"{marital_d:>+9.1%} {age_d:>+8.1%} {inc_d:>+8.1%} "
            f"{pass_count:>3}/{total}"
        )

    print("=" * 100)

    # Detailed per-pair breakdown
    print("\n" + "=" * 100)
    print("DETAILED PER-PAIR BREAKDOWN")
    print("=" * 100)

    detail_hdr = (
        f"{'Method':<28} {'Purpose':<22} {'Attribute':<16} "
        f"{'Best Acc':>9} {'Baseline':>9} {'Delta':>8} {'R²':>8} {'Status':>8}"
    )
    print(detail_hdr)
    print("-" * len(detail_hdr))

    for result in results:
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            adj_ok = delta < 0.02 and r.linear_r2 < 0.05
            status = "PASS" if adj_ok else "FAIL"
            print(
                f"{result.name:<28} {r.purpose_name:<22} {r.attr_name:<16} "
                f"{r.empirical_best_acc:>8.1%} {baseline:>8.1%} "
                f"{delta:>+7.1%} {r.linear_r2:>8.4f} {status:>8}"
            )
        print()

    print("=" * 100)


def save_comparison_csv(
    results: list[BaselineResult],
    majority_baselines: dict[str, float],
    output_path: Path,
) -> None:
    """Save detailed comparison to CSV."""
    rows = []
    for result in results:
        income_acc = result.task_accuracies.get("income", 0.0)

        # Per-pair details
        pass_count = 0
        total = 0
        pair_details = []
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            adj_ok = delta < 0.02 and r.linear_r2 < 0.05
            total += 1
            if adj_ok:
                pass_count += 1
            pair_details.append({
                "method": result.name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "task_acc": round(income_acc, 4),
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(baseline, 4),
                "delta": round(delta, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": adj_ok,
                "pairs_passing": f"{pass_count}/{total}" if r == result.reports[-1] else "",
                "train_time_s": round(result.train_time, 1),
            })
        rows.extend(pair_details)

    fieldnames = [
        "method", "purpose", "attribute", "task_acc", "best_emp_acc",
        "majority_baseline", "delta", "linear_r2", "adj_pass",
        "pairs_passing", "train_time_s",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {output_path}")


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data (shared across all baselines) ─────────────────────────
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    print("Loading Adult dataset...")
    train_dataset = AdultDataset(
        purposes=purposes, root="data", split="train", download=True
    )
    val_dataset = AdultDataset(
        purposes=purposes, root="data", split="val", download=False
    )
    test_dataset = AdultDataset(
        purposes=purposes, root="data", split="test", download=False
    )

    input_dim = train_dataset.info.num_features
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}, Features: {input_dim}"
    )

    majority_baselines = compute_majority_baselines(test_dataset)
    print("Majority-class baselines:")
    for attr, baseline in majority_baselines.items():
        print(f"  {attr}: {baseline:.1%}")

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch,
    )

    all_results: list[BaselineResult] = []

    # ── Baseline 1: Standard (no privacy) ────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE 1: Standard (no privacy)")
    print("=" * 60)

    torch.manual_seed(42)
    result_standard = run_baseline(
        name="Standard (no privacy)",
        encoder=StandardEncoder(
            input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
        ),
        purposes=purposes,
        registry=registry,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=TrainerConfig(
            batch_size=256,
            lr_encoder=1e-3,
            lr_auditor=1e-3,
            lambda_adv=0.0,
            lambda_verify=0.0,
            auditor_steps=1,
            epochs=200,
            weight_decay=1e-4,
            early_stopping_patience=15,
            confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "baseline_standard"),
        ),
        device=device,
    )
    all_results.append(result_standard)

    # ── Baseline 2: Adversarial-only (LAFTR) ─────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE 2: Adversarial-only (LAFTR)")
    print("=" * 60)

    torch.manual_seed(42)
    result_adversarial = run_baseline(
        name="Adversarial-only (LAFTR)",
        encoder=StandardEncoder(
            input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
        ),
        purposes=purposes,
        registry=registry,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=TrainerConfig(
            batch_size=256,
            lr_encoder=1e-3,
            lr_auditor=1e-3,
            lambda_adv=50.0,
            lambda_verify=50.0,
            auditor_steps=10,
            epochs=200,
            weight_decay=1e-4,
            early_stopping_patience=15,
            confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "baseline_adversarial"),
        ),
        device=device,
    )
    all_results.append(result_adversarial)

    # ── Baseline 3: Purpose-only (no adversarial) ────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE 3: Purpose-only (no adversarial)")
    print("=" * 60)

    torch.manual_seed(42)
    result_purpose = run_baseline(
        name="Purpose-only (no adv.)",
        encoder=PurposeConditionedEncoder(
            input_dim=input_dim,
            hidden_dims=[128, 128],
            repr_dim=64,
            num_purposes=len(purposes),
            purpose_emb_dim=32,
            conditioning="film",
            dropout=0.3,
        ),
        purposes=purposes,
        registry=registry,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=TrainerConfig(
            batch_size=256,
            lr_encoder=1e-3,
            lr_auditor=1e-3,
            lambda_adv=0.0,
            lambda_verify=0.0,
            auditor_steps=1,
            epochs=200,
            weight_decay=1e-4,
            early_stopping_patience=15,
            confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "baseline_purpose_only"),
        ),
        device=device,
    )
    all_results.append(result_purpose)

    # ── Load PCRL from checkpoint ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PCRL (ours) - loading from checkpoint")
    print("=" * 60)

    result_pcrl = load_pcrl_result(
        purposes=purposes,
        registry=registry,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=input_dim,
        device=device,
    )
    all_results.append(result_pcrl)

    # ── Print comparison ─────────────────────────────────────────────────
    print_comparison_table(all_results, majority_baselines)

    # ── Save CSV ─────────────────────────────────────────────────────────
    results_dir = project_root / "results" / "adult"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_csv(all_results, majority_baselines, results_dir / "baseline_comparison.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()

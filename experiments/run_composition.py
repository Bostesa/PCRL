#!/usr/bin/env python3
"""Compositional purpose experiment on the Adult dataset.

Composes income_prediction AND employment_analysis:
  - Allowed tasks: income + occupation_group
  - Disallowed attrs: race, sex, age_group, marital_status (union)

Compares two approaches:
1. Algebraic composition: additive embedding (emb_0 + emb_1) on pretrained PCRL
2. Retrained union: train a new purpose from scratch with combined constraints

Shows whether algebraic composition works without retraining.
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

# Add project root to path
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
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.composition import compose_and
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CompositionResult:
    """Results from one composition approach."""

    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def compute_majority_baselines(test_dataset: AdultDataset) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for attr_name, labels in test_dataset.sensitive_attrs.items():
        n = len(labels)
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / n
    return baselines


def extract_composed_representations(
    encoder: PurposeConditionedEncoder,
    loader: DataLoader,
    composed_emb: torch.Tensor,
    attr_name: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract representations using a composed embedding."""
    encoder.eval()
    all_reprs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder.forward_with_embedding(x, composed_emb)
            all_reprs.append(h.cpu().numpy())
            all_labels.append(batch["sensitive_attrs"][attr_name].numpy())

    return np.concatenate(all_reprs), np.concatenate(all_labels)


def evaluate_task_with_embedding(
    encoder: PurposeConditionedEncoder,
    task_head: torch.nn.Module,
    loader: DataLoader,
    composed_emb: torch.Tensor,
    task_name: str,
    device: str,
) -> float:
    """Evaluate task accuracy using a composed embedding."""
    encoder.eval()
    task_head.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder.forward_with_embedding(x, composed_emb)
            logits = task_head(h)
            preds = logits.argmax(dim=-1)
            targets = batch["task_labels"][task_name].to(device)
            correct += (preds == targets).sum().item()
            total += targets.shape[0]

    return correct / total if total > 0 else 0.0


def run_composed_embedding(
    purposes,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: str,
) -> CompositionResult:
    """Approach 1: Additive embedding composition on pretrained PCRL."""
    # Reconstruct model architecture and load checkpoint
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=64,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
    )

    task_heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)

    auditors: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=64,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256,
            num_layers=3,
        )

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

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
    encoder.to(device).eval()

    # Compose embeddings: purpose 0 (income) + purpose 1 (employment)
    with torch.no_grad():
        emb_0 = encoder.get_purpose_embedding(0).to(device)
        emb_1 = encoder.get_purpose_embedding(1).to(device)
        composed_emb = emb_0 + emb_1

    # Composed purpose spec
    composed = compose_and(purposes[0], purposes[1])
    print(f"  Composed: {composed.name}")
    print(f"    allowed_tasks = {composed.allowed_tasks}")
    print(f"    disallowed_attrs = {composed.disallowed_attrs}")

    # Evaluate tasks using composed embedding
    # income uses income_prediction's task head, occ uses employment_analysis's
    income_acc = evaluate_task_with_embedding(
        encoder,
        trainer.task_heads["income_prediction"].to(device),
        test_loader,
        composed_emb,
        "income",
        device,
    )
    occ_acc = evaluate_task_with_embedding(
        encoder,
        trainer.task_heads["employment_analysis"].to(device),
        test_loader,
        composed_emb,
        "occupation_group",
        device,
    )

    # Run compliance audit manually for composed representation
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    reports: list[ComplianceReport] = []

    for attr_name in composed.disallowed_attrs:
        train_reprs, train_labels = extract_composed_representations(
            encoder, train_loader, composed_emb, attr_name, device,
        )
        test_reprs, test_labels = extract_composed_representations(
            encoder, test_loader, composed_emb, attr_name, device,
        )

        linear_result, null_result = linear_audit.audit(test_reprs, test_labels)
        num_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
        chance_acc = 1.0 / max(num_classes, 1)
        best_acc, emp_results = empirical_audit.audit(
            train_reprs, train_labels, test_reprs, test_labels,
        )

        certified = linear_result.certified and (best_acc - chance_acc) < 0.05
        reports.append(ComplianceReport(
            purpose_name=composed.name,
            attr_name=attr_name,
            linear_r2=linear_result.r_squared,
            linear_certified=linear_result.certified,
            null_space_r2=null_result.r_squared,
            variance_preserved=null_result.variance_preserved or 0.0,
            empirical_best_acc=best_acc,
            empirical_chance_acc=chance_acc,
            empirical_results=emp_results,
            certified=certified,
        ))

    return CompositionResult(
        name="Composed (emb₀ + emb₁)",
        task_accuracies={"income": income_acc, "occupation_group": occ_acc},
        reports=reports,
        train_time=0.0,
    )


def run_union_retrained(
    purposes,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: str,
) -> CompositionResult:
    """Approach 2: Train a new model from scratch with union constraints.

    Uses two purpose specs that share the same disallowed attributes (the union)
    but have separate task heads, so both income and occupation_group are trained.
    """
    union_disallowed = ["age_group", "marital_status", "race", "sex"]
    union_attr_dims = {"race": 5, "sex": 2, "age_group": 4, "marital_status": 2}

    income_union = PurposeSpec(
        name="income_union",
        allowed_tasks=["income"],
        disallowed_attrs=union_disallowed,
        task_type="classification",
        allowed_task_dims={"income": 2},
        disallowed_attr_dims=union_attr_dims,
    )
    employment_union = PurposeSpec(
        name="employment_union",
        allowed_tasks=["occupation_group"],
        disallowed_attrs=union_disallowed,
        task_type="classification",
        allowed_task_dims={"occupation_group": 6},
        disallowed_attr_dims=union_attr_dims,
    )

    union_purposes = [income_union, employment_union]
    registry = PurposeRegistry()
    for p in union_purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=64,
        num_purposes=2,
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
    )

    task_heads: dict[str, torch.nn.Module] = {
        "income_union": TaskHead(repr_dim=64, output_dim=2),
        "employment_union": TaskHead(repr_dim=64, output_dim=6),
    }

    auditors: dict[str, torch.nn.Module] = {}
    for p in union_purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=64,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256,
            num_layers=3,
        )

    config = TrainerConfig(
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
        checkpoint_dir=str(project_root / "checkpoints" / "composition_union"),
    )

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
    print(f"  Completed: epoch {state.epoch + 1}, {train_time:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    income_acc = eval_metrics.task_accuracy.get("income", 0.0)
    occ_acc = eval_metrics.task_accuracy.get("occupation_group", 0.0)

    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )

    # Consolidate reports: take worst delta per attr across the two purposes
    consolidated: list[ComplianceReport] = []
    seen_attrs: set[str] = set()
    for attr in union_disallowed:
        attr_reports = [r for r in reports if r.attr_name == attr]
        if not attr_reports:
            continue
        # Take worst case (highest empirical accuracy, highest R²)
        worst = max(attr_reports, key=lambda r: r.empirical_best_acc)
        worst_r2 = max(r.linear_r2 for r in attr_reports)
        consolidated.append(ComplianceReport(
            purpose_name="income_AND_employment",
            attr_name=attr,
            linear_r2=worst_r2,
            linear_certified=worst.linear_certified,
            null_space_r2=worst.null_space_r2,
            variance_preserved=worst.variance_preserved,
            empirical_best_acc=worst.empirical_best_acc,
            empirical_chance_acc=worst.empirical_chance_acc,
            empirical_results=worst.empirical_results,
            certified=worst.certified,
        ))

    return CompositionResult(
        name="Retrained union",
        task_accuracies={"income": income_acc, "occupation_group": occ_acc},
        reports=consolidated,
        train_time=train_time,
    )


def print_results(
    results: list[CompositionResult],
    majority_baselines: dict[str, float],
) -> None:
    print("\n" + "=" * 110)
    print("COMPOSITION EXPERIMENT RESULTS")
    print("=" * 110)

    header = (
        f"{'Method':<28} {'Income':>8} {'OccGrp':>8} "
        f"{'Race Δ':>9} {'Sex Δ':>9} {'AgeGrp Δ':>10} {'MarStat Δ':>10} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        income = result.task_accuracies.get("income", 0.0)
        occ = result.task_accuracies.get("occupation_group", 0.0)

        deltas: dict[str, float] = {}
        pass_count = 0
        total = 0
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            deltas[r.attr_name] = delta
            total += 1
            if delta < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        print(
            f"{result.name:<28} {income:>7.1%} {occ:>7.1%} "
            f"{deltas.get('race', 0):>+8.1%} {deltas.get('sex', 0):>+8.1%} "
            f"{deltas.get('age_group', 0):>+9.1%} "
            f"{deltas.get('marital_status', 0):>+9.1%} "
            f"{pass_count:>3}/{total}"
        )

    print("=" * 110)

    # Detailed breakdown
    print("\n" + "=" * 110)
    print("DETAILED PER-ATTRIBUTE BREAKDOWN")
    print("=" * 110)
    detail_hdr = (
        f"{'Method':<28} {'Attribute':<16} "
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
                f"{result.name:<28} {r.attr_name:<16} "
                f"{r.empirical_best_acc:>8.1%} {baseline:>8.1%} "
                f"{delta:>+7.1%} {r.linear_r2:>8.4f} {status:>8}"
            )
        print()

    print("=" * 110)


def save_csv(
    results: list[CompositionResult],
    majority_baselines: dict[str, float],
    output_path: Path,
) -> None:
    rows = []
    for result in results:
        income = result.task_accuracies.get("income", 0.0)
        occ = result.task_accuracies.get("occupation_group", 0.0)
        pass_count = 0
        total = 0
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            adj_ok = delta < 0.02 and r.linear_r2 < 0.05
            total += 1
            if adj_ok:
                pass_count += 1
            rows.append({
                "method": result.name,
                "attribute": r.attr_name,
                "income_acc": round(income, 4),
                "occ_group_acc": round(occ, 4),
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(baseline, 4),
                "delta": round(delta, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": adj_ok,
                "pairs_passing": f"{pass_count}/{total}" if r == result.reports[-1] else "",
                "train_time_s": round(result.train_time, 1),
            })

    fieldnames = [
        "method", "attribute", "income_acc", "occ_group_acc",
        "best_emp_acc", "majority_baseline", "delta", "linear_r2",
        "adj_pass", "pairs_passing", "train_time_s",
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

    # Load data
    purposes = get_adult_purposes()
    print("Loading Adult dataset...")
    train_dataset = AdultDataset(
        purposes=purposes, root="data", split="train", download=True,
    )
    val_dataset = AdultDataset(
        purposes=purposes, root="data", split="val", download=False,
    )
    test_dataset = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
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

    all_results: list[CompositionResult] = []

    # ── Approach 1: Algebraic composition ────────────────────────────────
    print("\n" + "=" * 60)
    print("APPROACH 1: Algebraic composition (emb₀ + emb₁)")
    print("=" * 60)

    result_composed = run_composed_embedding(
        purposes, train_loader, test_loader, input_dim, device,
    )
    all_results.append(result_composed)

    # ── Approach 2: Retrained union purpose ──────────────────────────────
    print("\n" + "=" * 60)
    print("APPROACH 2: Retrained union purpose")
    print("=" * 60)

    result_union = run_union_retrained(
        purposes, train_loader, val_loader, test_loader, input_dim, device,
    )
    all_results.append(result_union)

    # ── Print and save results ───────────────────────────────────────────
    print_results(all_results, majority_baselines)

    results_dir = project_root / "results" / "adult"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_csv(all_results, majority_baselines, results_dir / "composition_experiment.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()

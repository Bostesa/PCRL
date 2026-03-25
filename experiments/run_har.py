#!/usr/bin/env python3
"""Full HAR experiment: PCRL training, baselines, and composition.

Runs the complete pipeline on the synthetic Human Activity Recognition
dataset to demonstrate PCRL on IoT sensor data:

A. PCRL training with adversarial + verification regularizer
B. Baselines: standard (no privacy) and adversarial-only (LAFTR)
C. Composition: activity_recognition AND health_monitoring
D. Comparison tables and CSV output

Results saved to results/har/.
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

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.har import HARDataset, get_har_purposes
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.composition import compose_and
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Shared constants ─────────────────────────────────────────────────────
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 20
EPOCHS = 50
PATIENCE = None  # No early stopping — adversarial needs full training
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3


@dataclass
class MethodResult:
    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def compute_majority_baselines(dataset: HARDataset) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for attr_name, labels in dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / len(labels)
    return baselines


def make_task_heads(purposes) -> dict[str, torch.nn.Module]:
    heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
    return heads


def make_auditors(purposes) -> dict[str, torch.nn.Module]:
    auds: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auds[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=AUDITOR_HIDDEN,
            num_layers=AUDITOR_LAYERS,
        )
    return auds


def train_and_evaluate(
    name: str,
    encoder: torch.nn.Module,
    purposes,
    registry: PurposeRegistry,
    config: TrainerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> MethodResult:
    """Train a model and run full compliance audit."""
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

    print(f"  {name}: epoch {state.epoch + 1}, {train_time:.0f}s")
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    return MethodResult(
        name=name,
        task_accuracies=eval_metrics.task_accuracy,
        reports=reports,
        train_time=train_time,
    )


def print_baseline_table(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
) -> None:
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON")
    print("=" * 100)

    # Collect all disallowed attrs across all reports
    all_attrs = sorted({r.attr_name for res in results for r in res.reports})

    # Header
    hdr_parts = [f"{'Method':<28}", f"{'Activity':>9}", f"{'IsActive':>9}"]
    for attr in all_attrs:
        hdr_parts.append(f"{attr + ' Δ':>12}")
    hdr_parts.append(f"{'Pass':>6}")
    header = " ".join(hdr_parts)
    print(header)
    print("-" * len(header))

    for result in results:
        act_acc = result.task_accuracies.get("activity", 0.0)
        ia_acc = result.task_accuracies.get("is_active", 0.0)

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

        parts = [f"{result.name:<28}", f"{act_acc:>8.1%}", f"{ia_acc:>8.1%}"]
        for attr in all_attrs:
            d = deltas.get(attr, 0.0)
            parts.append(f"{d:>+11.1%}")
        parts.append(f"{pass_count:>3}/{total}")
        print(" ".join(parts))

    print("=" * 100)

    # Detailed breakdown
    print("\n" + "=" * 100)
    print("DETAILED PER-PAIR BREAKDOWN")
    print("=" * 100)
    detail_hdr = (
        f"{'Method':<28} {'Purpose':<22} {'Attribute':<12} "
        f"{'Best Acc':>9} {'Baseline':>9} {'Delta':>8} {'R²':>8} {'Status':>8}"
    )
    print(detail_hdr)
    print("-" * len(detail_hdr))

    for result in results:
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            ok = delta < 0.02 and r.linear_r2 < 0.05
            print(
                f"{result.name:<28} {r.purpose_name:<22} {r.attr_name:<12} "
                f"{r.empirical_best_acc:>8.1%} {baseline:>8.1%} "
                f"{delta:>+7.1%} {r.linear_r2:>8.4f} {'PASS' if ok else 'FAIL':>8}"
            )
        print()
    print("=" * 100)


def save_baseline_csv(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
    path: Path,
) -> None:
    rows = []
    for result in results:
        act_acc = result.task_accuracies.get("activity", 0.0)
        ia_acc = result.task_accuracies.get("is_active", 0.0)
        pass_count = 0
        total = 0
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            ok = delta < 0.02 and r.linear_r2 < 0.05
            total += 1
            if ok:
                pass_count += 1
            rows.append({
                "method": result.name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "activity_acc": round(act_acc, 4),
                "is_active_acc": round(ia_acc, 4),
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(baseline, 4),
                "delta": round(delta, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": ok,
                "pairs_passing": f"{pass_count}/{total}" if r == result.reports[-1] else "",
                "train_time_s": round(result.train_time, 1),
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


# ── Composition helpers ──────────────────────────────────────────────────

def extract_composed_representations(
    encoder: PurposeConditionedEncoder,
    loader: DataLoader,
    composed_emb: torch.Tensor,
    attr_name: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    all_reprs, all_labels = [], []
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
    encoder.eval()
    task_head.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder.forward_with_embedding(x, composed_emb)
            preds = task_head(h).argmax(dim=-1)
            targets = batch["task_labels"][task_name].to(device)
            correct += (preds == targets).sum().item()
            total += targets.shape[0]
    return correct / total if total > 0 else 0.0


def run_composed_embedding(
    encoder: PurposeConditionedEncoder,
    trainer: PCRLTrainer,
    purposes,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> MethodResult:
    """Additive embedding composition on pretrained PCRL."""
    encoder.to(device).eval()

    with torch.no_grad():
        emb_0 = encoder.get_purpose_embedding(0).to(device)
        emb_1 = encoder.get_purpose_embedding(1).to(device)
        composed_emb = emb_0 + emb_1

    composed = compose_and(purposes[0], purposes[1])
    print(f"  Composed: {composed.name}")
    print(f"    disallowed_attrs = {composed.disallowed_attrs}")

    act_acc = evaluate_task_with_embedding(
        encoder, trainer.task_heads["activity_recognition"].to(device),
        test_loader, composed_emb, "activity", device,
    )
    ia_acc = evaluate_task_with_embedding(
        encoder, trainer.task_heads["health_monitoring"].to(device),
        test_loader, composed_emb, "is_active", device,
    )

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

    return MethodResult(
        name="Composed (emb₀ + emb₁)",
        task_accuracies={"activity": act_acc, "is_active": ia_acc},
        reports=reports,
    )


def run_union_retrained(
    purposes,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    device: str,
) -> MethodResult:
    """Train union purpose from scratch."""
    union_disallowed = ["activity", "subject_id"]
    union_attr_dims = {"subject_id": 10, "activity": 6}

    act_union = PurposeSpec(
        name="activity_union",
        allowed_tasks=["activity"],
        disallowed_attrs=union_disallowed,
        task_type="classification",
        allowed_task_dims={"activity": 6},
        disallowed_attr_dims=union_attr_dims,
    )
    health_union = PurposeSpec(
        name="health_union",
        allowed_tasks=["is_active"],
        disallowed_attrs=union_disallowed,
        task_type="classification",
        allowed_task_dims={"is_active": 2},
        disallowed_attr_dims=union_attr_dims,
    )

    union_purposes = [act_union, health_union]
    registry = PurposeRegistry()
    for p in union_purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=2,
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )

    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR,
        lr_auditor=LR,
        lambda_adv=LAMBDA_ADV,
        lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "har_union"),
    )

    result = train_and_evaluate(
        "Retrained union", encoder, union_purposes, registry, config,
        train_loader, val_loader, test_loader, device,
    )

    # Consolidate: worst case per attr across both purposes
    consolidated: list[ComplianceReport] = []
    for attr in union_disallowed:
        attr_reports = [r for r in result.reports if r.attr_name == attr]
        if not attr_reports:
            continue
        worst = max(attr_reports, key=lambda r: r.empirical_best_acc)
        worst_r2 = max(r.linear_r2 for r in attr_reports)
        consolidated.append(ComplianceReport(
            purpose_name="activity_AND_health",
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

    result.reports = consolidated
    return result


def print_composition_table(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
) -> None:
    print("\n" + "=" * 100)
    print("COMPOSITION EXPERIMENT")
    print("=" * 100)

    header = (
        f"{'Method':<28} {'Activity':>9} {'IsActive':>9} "
        f"{'SubjID Δ':>10} {'Activity Δ':>11} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        act = result.task_accuracies.get("activity", 0.0)
        ia = result.task_accuracies.get("is_active", 0.0)
        deltas: dict[str, float] = {}
        pc = total = 0
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            deltas[r.attr_name] = d
            total += 1
            if d < 0.02 and r.linear_r2 < 0.05:
                pc += 1
        print(
            f"{result.name:<28} {act:>8.1%} {ia:>8.1%} "
            f"{deltas.get('subject_id', 0):>+9.1%} "
            f"{deltas.get('activity', 0):>+10.1%} "
            f"{pc:>3}/{total}"
        )

    print("=" * 100)

    # Detail
    detail_hdr = (
        f"{'Method':<28} {'Attribute':<12} "
        f"{'Best Acc':>9} {'Baseline':>9} {'Delta':>8} {'R²':>8} {'Status':>8}"
    )
    print(detail_hdr)
    print("-" * len(detail_hdr))
    for result in results:
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            print(
                f"{result.name:<28} {r.attr_name:<12} "
                f"{r.empirical_best_acc:>8.1%} {bl:>8.1%} "
                f"{d:>+7.1%} {r.linear_r2:>8.4f} {'PASS' if ok else 'FAIL':>8}"
            )
        print()
    print("=" * 100)


def save_composition_csv(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
    path: Path,
) -> None:
    rows = []
    for result in results:
        act = result.task_accuracies.get("activity", 0.0)
        ia = result.task_accuracies.get("is_active", 0.0)
        pc = total = 0
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            total += 1
            if ok:
                pc += 1
            rows.append({
                "method": result.name,
                "attribute": r.attr_name,
                "activity_acc": round(act, 4),
                "is_active_acc": round(ia, 4),
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(majority_baselines.get(r.attr_name, 0), 4),
                "delta": round(d, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": ok,
                "pairs_passing": f"{pc}/{total}" if r == result.reports[-1] else "",
                "train_time_s": round(result.train_time, 1),
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    print("Generating HAR dataset...")
    train_dataset = HARDataset(purposes=purposes, split="train")
    val_dataset = HARDataset(purposes=purposes, split="val")
    test_dataset = HARDataset(purposes=purposes, split="test")

    input_dim = train_dataset.info.num_features
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}, Features: {input_dim}"
    )

    majority_baselines = compute_majority_baselines(test_dataset)
    print("Majority-class baselines:")
    for attr, bl in majority_baselines.items():
        print(f"  {attr}: {bl:.1%}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_pcrl_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch,
    )

    results_dir = project_root / "results" / "har"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════
    # A. PCRL TRAINING
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A. PCRL TRAINING")
    print("=" * 60)

    torch.manual_seed(42)
    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )

    pcrl_config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR,
        lr_auditor=LR,
        lambda_adv=LAMBDA_ADV,
        lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "har"),
    )

    pcrl_task_heads = make_task_heads(purposes)
    pcrl_auditors = make_auditors(purposes)

    pcrl_trainer = PCRLTrainer(
        encoder=pcrl_encoder,
        task_heads=pcrl_task_heads,
        auditors=pcrl_auditors,
        config=pcrl_config,
        purpose_registry=registry,
        device=device,
    )

    t0 = time.time()
    state = pcrl_trainer.train(train_loader, val_loader=val_loader)
    pcrl_time = time.time() - t0

    eval_metrics = pcrl_trainer.evaluate(test_loader)
    pcrl_reports = generate_report(
        encoder=pcrl_encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )

    pcrl_result = MethodResult(
        name="PCRL (ours)",
        task_accuracies=eval_metrics.task_accuracy,
        reports=pcrl_reports,
        train_time=pcrl_time,
    )

    print(f"  PCRL: epoch {state.epoch + 1}, {pcrl_time:.0f}s")
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    # ═════════════════════════════════════════════════════════════════════
    # B. BASELINES
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. BASELINES")
    print("=" * 60)

    # B1: Standard (no privacy)
    print("\n--- Standard (no privacy) ---")
    torch.manual_seed(42)
    standard_result = train_and_evaluate(
        name="Standard (no privacy)",
        encoder=StandardEncoder(
            input_dim=input_dim, hidden_dims=HIDDEN_DIMS,
            repr_dim=REPR_DIM, dropout=DROPOUT,
        ),
        purposes=purposes,
        registry=registry,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
            epochs=EPOCHS, weight_decay=1e-4,
            early_stopping_patience=PATIENCE, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "har_standard"),
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    # B2: Adversarial-only (LAFTR)
    print("\n--- Adversarial-only (LAFTR) ---")
    torch.manual_seed(42)
    laftr_result = train_and_evaluate(
        name="Adversarial-only (LAFTR)",
        encoder=StandardEncoder(
            input_dim=input_dim, hidden_dims=HIDDEN_DIMS,
            repr_dim=REPR_DIM, dropout=DROPOUT,
        ),
        purposes=purposes,
        registry=registry,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
            auditor_steps=AUDITOR_STEPS,
            epochs=EPOCHS, weight_decay=1e-4,
            early_stopping_patience=PATIENCE, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "har_laftr"),
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    all_baseline_results = [standard_result, laftr_result, pcrl_result]
    print_baseline_table(all_baseline_results, majority_baselines)
    save_baseline_csv(
        all_baseline_results, majority_baselines,
        results_dir / "baseline_comparison.csv",
    )

    # ═════════════════════════════════════════════════════════════════════
    # C. COMPOSITION
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C. COMPOSITION (activity_recognition AND health_monitoring)")
    print("=" * 60)

    # C1: Algebraic composition
    print("\n--- Algebraic composition (emb₀ + emb₁) ---")
    composed_result = run_composed_embedding(
        pcrl_encoder, pcrl_trainer, purposes,
        train_loader, test_loader, device,
    )

    # C2: Retrained union
    print("\n--- Retrained union ---")
    torch.manual_seed(42)
    union_result = run_union_retrained(
        purposes, train_loader, val_loader, test_loader, input_dim, device,
    )

    composition_results = [composed_result, union_result]
    print_composition_table(composition_results, majority_baselines)
    save_composition_csv(
        composition_results, majority_baselines,
        results_dir / "composition_experiment.csv",
    )

    # ═════════════════════════════════════════════════════════════════════
    # D. SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DONE — all results saved to results/har/")
    print("=" * 60)


if __name__ == "__main__":
    main()

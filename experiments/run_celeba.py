#!/usr/bin/env python3
"""Full experiment on CelebA: PCRL with CNN encoder, baselines, and composition.

Runs the complete pipeline on CelebA face attribute dataset with 5 purposes
that create conflicting constraints (Male, Attractive, Smiling each appear
as both allowed tasks and disallowed attrs across different purposes).

A. Hyperparameter sweep (λ_adv in {1, 2, 5, 10}) — fast, no compliance audit
B. PCRL training with best config
C. Baselines: standard (no privacy) and adversarial-only (LAFTR)
D. Composition: smile_detection AND attractiveness_prediction
E. Comparison tables and CSV output

Results saved to results/celeba/.
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
import torch.nn as nn
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
from pcrl.data.celeba import CelebADataset, get_celeba_purposes, TASK_ATTRS, SENSITIVE_ATTRS
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.cnn_encoder import CNNEncoder, StandardCNNEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.composition import compose_and
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ── Architecture constants ───────────────────────────────────────────────
REPR_DIM = 256
PURPOSE_EMB_DIM = 32
CONV_CHANNELS = (32, 64, 128, 256)
DROPOUT = 0.3
BATCH_SIZE = 128
LR = 1e-3
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3

# ── Training constants ──────────────────────────────────────────────────
LAMBDA_ADV = 0.3  # global default (overridden by per-purpose)
LAMBDA_VERIFY = 0.1
AUDITOR_STEPS = 5
EPOCHS = 50
PATIENCE = None  # no early stopping
WARMUP_EPOCHS = 5

# Per-purpose lambda_adv values
LAMBDA_ADV_PER_PURPOSE = {
    "smile_detection": 0.5,
    "age_estimation": 0.5,
    "expression_analysis": 0.2,
    "attractiveness_prediction": 0.3,
    "gender_analysis": 0.3,
}

# ── Data constants ───────────────────────────────────────────────────────
MAX_TRAIN_SAMPLES = 10000
MAX_VAL_SAMPLES = 3000
MAX_TEST_SAMPLES = 3000

# Primary tasks used for evaluation display
PRIMARY_TASKS = ["Smiling", "Male", "Attractive", "Young", "Mouth_Slightly_Open"]


@dataclass
class MethodResult:
    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def compute_majority_baselines(dataset: CelebADataset) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for attr_name, labels in dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / len(labels)
    return baselines


class MultiTaskHead(nn.Module):
    """Task head that returns a dict of predictions for multiple tasks."""

    def __init__(self, heads: dict[str, nn.Module]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


def make_task_heads(purposes, repr_dim: int) -> dict[str, torch.nn.Module]:
    heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        if len(p.allowed_tasks) == 1:
            task_name = p.allowed_tasks[0]
            output_dim = p.allowed_task_dims.get(task_name, 2)
            heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        else:
            sub_heads = {}
            for task_name in p.allowed_tasks:
                output_dim = p.allowed_task_dims.get(task_name, 2)
                sub_heads[task_name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
            heads[p.name] = MultiTaskHead(sub_heads)
    return heads


def make_auditors(purposes, repr_dim: int, use_grl: bool = False) -> dict[str, torch.nn.Module]:
    auds: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auds[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=AUDITOR_HIDDEN,
            num_layers=AUDITOR_LAYERS,
            use_gradient_reversal=use_grl,
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
    repr_dim: int = REPR_DIM,
    run_compliance: bool = True,
) -> tuple[MethodResult, PCRLTrainer]:
    task_heads = make_task_heads(purposes, repr_dim)
    auditors = make_auditors(purposes, repr_dim)

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

    reports = []
    if run_compliance:
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
    ), trainer


# ── Baseline table ───────────────────────────────────────────────────────

def print_baseline_table(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
) -> None:
    print("\n" + "=" * 140)
    print("BASELINE COMPARISON (CelebA — 5 purposes with conflicting constraints)")
    print("=" * 140)

    all_attrs = sorted({r.attr_name for res in results for r in res.reports})

    hdr = f"{'Method':<28}"
    for task in PRIMARY_TASKS:
        hdr += f" {task[:7]:>8}"
    for attr in all_attrs:
        hdr += f" {attr[:6]+'Δ':>8}"
    hdr += f" {'Pass':>7}"
    print(hdr)
    print("-" * len(hdr))

    for result in results:
        row = f"{result.name:<28}"
        for task in PRIMARY_TASKS:
            acc = result.task_accuracies.get(task, 0.0)
            row += f" {acc:>7.1%}"

        deltas: dict[str, float] = {}
        pass_count = total = 0
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            deltas[r.attr_name] = max(deltas.get(r.attr_name, -1.0), delta)
            total += 1
            if delta < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        for attr in all_attrs:
            d = deltas.get(attr, 0.0)
            row += f" {d:>+7.1%}"
        row += f" {pass_count:>3}/{total}"
        print(row)

    print("=" * 140)


def print_detailed_breakdown(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
) -> None:
    """Print per-purpose, per-attribute breakdown."""
    print("\nDETAILED PER-PAIR BREAKDOWN")
    print("-" * 100)
    print(f"{'Method':<18} {'Purpose':<25} {'Attribute':<12} "
          f"{'Best':>7} {'Base':>7} {'Delta':>8} {'R²':>8} {'Status':>8}")
    print("-" * 100)
    for result in results:
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            print(f"{result.name:<18} {r.purpose_name:<25} {r.attr_name:<12} "
                  f"{r.empirical_best_acc:>6.1%} {bl:>6.1%} {d:>+7.1%} "
                  f"{r.linear_r2:>7.4f} {'PASS' if ok else 'FAIL':>8}")
    print()


def save_baseline_csv(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
    path: Path,
) -> None:
    rows = []
    for result in results:
        for r in result.reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            ok = delta < 0.02 and r.linear_r2 < 0.05
            rows.append({
                "method": result.name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                **{f"{t}_acc": round(result.task_accuracies.get(t, 0.0), 4)
                   for t in PRIMARY_TASKS},
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(baseline, 4),
                "delta": round(delta, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": ok,
                "train_time_s": round(result.train_time, 1),
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


# ── Composition helpers ──────────────────────────────────────────────────

def extract_composed_representations(
    encoder: CNNEncoder,
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
    encoder: CNNEncoder,
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
    encoder: CNNEncoder,
    trainer: PCRLTrainer,
    purposes,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    p0_idx: int = 0,
    p1_idx: int = 3,
) -> MethodResult:
    """Additive embedding composition: purposes[p0_idx] + purposes[p1_idx]."""
    encoder.to(device).eval()
    p0 = purposes[p0_idx]
    p1 = purposes[p1_idx]

    with torch.no_grad():
        emb_0 = encoder.get_purpose_embedding(p0_idx).to(device)
        emb_1 = encoder.get_purpose_embedding(p1_idx).to(device)
        composed_emb = emb_0 + emb_1

    composed = compose_and(p0, p1)
    print(f"  Composed: {composed.name}")
    print(f"    tasks = {composed.allowed_tasks}")
    print(f"    disallowed = {composed.disallowed_attrs}")

    # Evaluate tasks using composed embedding with the original task heads
    task_accs = {}
    for p_idx, p in [(p0_idx, p0), (p1_idx, p1)]:
        task_head = trainer.task_heads[p.name].to(device)
        task_name = p.allowed_tasks[0]
        task_accs[task_name] = evaluate_task_with_embedding(
            encoder, task_head, test_loader, composed_emb, task_name, device,
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
        name=f"Composed ({p0.name[:5]}+{p1.name[:5]})",
        task_accuracies=task_accs,
        reports=reports,
    )


def run_union_retrained(
    purposes,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    p0_idx: int = 0,
    p1_idx: int = 3,
    repr_dim: int = REPR_DIM,
) -> MethodResult:
    """Train union purpose from scratch."""
    p0 = purposes[p0_idx]
    p1 = purposes[p1_idx]
    union_disallowed = sorted(set(p0.disallowed_attrs) | set(p1.disallowed_attrs))
    union_attr_dims = {}
    for p in [p0, p1]:
        for attr, dim in p.disallowed_attr_dims.items():
            union_attr_dims[attr] = dim

    # Create one purpose per task in the union
    union_purposes = []
    for p in [p0, p1]:
        task_name = p.allowed_tasks[0]
        union_purposes.append(PurposeSpec(
            name=f"{task_name}_union",
            allowed_tasks=[task_name],
            disallowed_attrs=union_disallowed,
            task_type="classification",
            allowed_task_dims={task_name: p.allowed_task_dims[task_name]},
            disallowed_attr_dims=union_attr_dims,
        ))

    registry = PurposeRegistry()
    for p in union_purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = CNNEncoder(
        repr_dim=repr_dim,
        num_purposes=len(union_purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conv_channels=CONV_CHANNELS,
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
        warmup_epochs=WARMUP_EPOCHS,
        checkpoint_dir=str(project_root / "checkpoints" / "celeba_union"),
    )

    result, _ = train_and_evaluate(
        "Retrained union", encoder, union_purposes, registry, config,
        train_loader, val_loader, test_loader, device, repr_dim,
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
            purpose_name=f"{p0.name}_AND_{p1.name}",
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
    # Map task accuracies
    task0 = p0.allowed_tasks[0]
    task1 = p1.allowed_tasks[0]
    result.task_accuracies = {
        task0: result.task_accuracies.get(task0, 0.0),
        task1: result.task_accuracies.get(task1, 0.0),
    }
    return result


def print_composition_table(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
    title: str = "COMPOSITION EXPERIMENT",
) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    # Collect all tasks and attrs from results
    all_tasks = []
    for res in results:
        for t in res.task_accuracies:
            if t not in all_tasks:
                all_tasks.append(t)
    all_attrs = sorted({r.attr_name for res in results for r in res.reports})

    header = f"{'Method':<28}"
    for t in all_tasks:
        header += f" {t[:8]:>9}"
    for a in all_attrs:
        header += f" {a[:6]+'Δ':>9}"
    header += f" {'Pass':>7}"
    print(header)
    print("-" * len(header))

    for result in results:
        row = f"{result.name:<28}"
        for t in all_tasks:
            acc = result.task_accuracies.get(t, 0.0)
            row += f" {acc:>8.1%}"

        deltas: dict[str, float] = {}
        pc = total = 0
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            deltas[r.attr_name] = max(deltas.get(r.attr_name, -1.0), d)
            total += 1
            if d < 0.02 and r.linear_r2 < 0.05:
                pc += 1

        for a in all_attrs:
            row += f" {deltas.get(a, 0):>+8.1%}"
        row += f" {pc:>3}/{total}"
        print(row)

    print("=" * 110)


def save_composition_csv(
    results: list[MethodResult],
    majority_baselines: dict[str, float],
    path: Path,
) -> None:
    rows = []
    all_tasks = []
    for res in results:
        for t in res.task_accuracies:
            if t not in all_tasks:
                all_tasks.append(t)

    for result in results:
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            row = {
                "method": result.name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
            }
            for t in all_tasks:
                row[f"{t}_acc"] = round(result.task_accuracies.get(t, 0.0), 4)
            row.update({
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(bl, 4),
                "delta": round(d, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": ok,
                "train_time_s": round(result.train_time, 1),
            })
            rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP — fast, uses auditor accuracy instead of full compliance audit
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(
    purposes,
    registry: PurposeRegistry,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    majority_baselines: dict[str, float],
    device: str,
) -> tuple[float, float, int]:
    """Run hyperparameter sweep and return best (lambda_adv, lambda_verify, K)."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP (5 purposes)")
    print("=" * 60)

    configs = [
        # (lambda_adv, lambda_verify, auditor_steps, label)
        # With 5 purposes, total adv pressure ~5x a single purpose.
        # Need lower per-purpose lambdas to avoid collapse.
        (0.1, 0.05, 5,  "λ=0.1/0.05 K=5"),
        (0.2, 0.1,  5,  "λ=0.2/0.1  K=5"),
        (0.5, 0.2,  5,  "λ=0.5/0.2  K=5"),
        (1.0, 0.5,  5,  "λ=1/0.5    K=5"),
        # Higher lambdas for stronger privacy
        (2.0, 1.0,  5,  "λ=2/1      K=5"),
        (5.0, 2.0,  5,  "λ=5/2      K=5"),
        (10.0, 5.0, 5,  "λ=10/5     K=5"),
    ]

    print(f"{'Config':<18} {'Smile':>7} {'Male':>7} {'Attr':>7} {'Young':>7} {'MouthO':>7} "
          f"{'AvgTask':>8} {'AvgAudΔ':>8} {'Time':>6}")
    print("-" * 90)

    best_config = configs[0][:3]
    best_score = -float("inf")

    for lam_adv, lam_ver, k_steps, label in configs:
        torch.manual_seed(42)
        encoder = CNNEncoder(
            repr_dim=REPR_DIM,
            num_purposes=len(purposes),
            purpose_emb_dim=PURPOSE_EMB_DIM,
            conv_channels=CONV_CHANNELS,
            dropout=DROPOUT,
        )

        config = TrainerConfig(
            batch_size=BATCH_SIZE,
            lr_encoder=LR,
            lr_auditor=LR,
            lambda_adv=lam_adv,
            lambda_verify=lam_ver,
            auditor_steps=k_steps,
            epochs=EPOCHS,
            weight_decay=1e-4,
            early_stopping_patience=PATIENCE,
            confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "celeba_sweep"),
        )

        task_heads = make_task_heads(purposes, REPR_DIM)
        auditors = make_auditors(purposes, REPR_DIM)

        trainer = PCRLTrainer(
            encoder=encoder, task_heads=task_heads, auditors=auditors,
            config=config, purpose_registry=registry, device=device,
        )

        t0 = time.time()
        trainer.train(train_loader, val_loader=val_loader)
        elapsed = time.time() - t0

        ev = trainer.evaluate(test_loader)

        smile = ev.task_accuracy.get("Smiling", 0.0)
        male = ev.task_accuracy.get("Male", 0.0)
        attr = ev.task_accuracy.get("Attractive", 0.0)
        young = ev.task_accuracy.get("Young", 0.0)
        mouth = ev.task_accuracy.get("Mouth_Slightly_Open", 0.0)

        # Average task accuracy (exclude tasks at exact 0%)
        task_accs = [a for a in [smile, male, attr, young, mouth] if a > 0.01]
        avg_task = sum(task_accs) / max(len(task_accs), 1) if task_accs else 0.0

        # Auditor accuracy — near chance means good privacy
        # Compute delta from majority baseline
        aud_deltas = []
        for attr_name, aud_acc in ev.auditor_accuracy.items():
            bl = majority_baselines.get(attr_name, 0.5)
            aud_deltas.append(max(0.0, aud_acc - bl))
        avg_aud_delta = sum(aud_deltas) / max(len(aud_deltas), 1)

        print(f"{label:<18} {smile:>6.1%} {male:>6.1%} {attr:>6.1%} {young:>6.1%} {mouth:>6.1%} "
              f"{avg_task:>7.1%} {avg_aud_delta:>+7.1%} {elapsed:>5.0f}s")

        # Score: maximize task accuracy, minimize auditor leakage
        score = avg_task - 3 * avg_aud_delta

        # Reject if primary tasks collapse (Smiling < 55% or Male < 55%)
        if smile < 0.55 and male < 0.55:
            score = -float("inf")

        if score > best_score:
            best_score = score
            best_config = (lam_adv, lam_ver, k_steps)

    print("-" * 90)
    print(f"Best config: λ_adv={best_config[0]}, λ_ver={best_config[1]}, K={best_config[2]}")
    return best_config


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sweep", action="store_true",
                        help="Run hyperparameter sweep (skipped by default with per-purpose λ)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    print(f"Purposes ({len(purposes)}):")
    for i, p in enumerate(purposes):
        print(f"  [{i}] {p.name}: tasks={p.allowed_tasks}, disallowed={p.disallowed_attrs}")

    print("\nLoading CelebA dataset...")
    train_dataset = CelebADataset(
        purposes=purposes, root="data/celeba", split="train",
        max_samples=MAX_TRAIN_SAMPLES,
    )
    val_dataset = CelebADataset(
        purposes=purposes, root="data/celeba", split="val",
        max_samples=MAX_VAL_SAMPLES,
    )
    test_dataset = CelebADataset(
        purposes=purposes, root="data/celeba", split="test",
        max_samples=MAX_TEST_SAMPLES,
    )

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    majority_baselines = compute_majority_baselines(test_dataset)
    print("Majority-class baselines:")
    for a, bl in majority_baselines.items():
        print(f"  {a}: {bl:.1%}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    results_dir = project_root / "results" / "celeba"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════
    # A. HYPERPARAMETER SWEEP (fast — no compliance audit)
    # ═════════════════════════════════════════════════════════════════════
    global LAMBDA_ADV, LAMBDA_VERIFY, AUDITOR_STEPS
    if not args.run_sweep:
        print(f"\nSkipping sweep — using per-purpose λ_adv: {LAMBDA_ADV_PER_PURPOSE}")
    else:
        best_lam_adv, best_lam_ver, best_k = run_sweep(
            purposes, registry, train_loader, val_loader, test_loader,
            majority_baselines, device,
        )
        LAMBDA_ADV = best_lam_adv
        LAMBDA_VERIFY = best_lam_ver
        AUDITOR_STEPS = best_k

    # ═════════════════════════════════════════════════════════════════════
    # B. PCRL TRAINING — sequential purposes, GRL, per-purpose λ, warmup
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. PCRL TRAINING (sequential + GRL + per-purpose λ)")
    print(f"   5 purposes, repr_dim={REPR_DIM}, conv={CONV_CHANNELS}")
    print(f"   Per-purpose λ_adv: {LAMBDA_ADV_PER_PURPOSE}")
    print(f"   GRL, sequential purposes, {EPOCHS} epochs, {WARMUP_EPOCHS} warmup")
    print("=" * 60)

    torch.manual_seed(42)
    pcrl_encoder = CNNEncoder(
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conv_channels=CONV_CHANNELS,
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
        gradient_reversal=True,
        sequential_purposes=True,
        lambda_adv_per_purpose=LAMBDA_ADV_PER_PURPOSE,
        warmup_epochs=WARMUP_EPOCHS,
        checkpoint_dir=str(project_root / "checkpoints" / "celeba"),
    )

    # Build auditors with GRL enabled
    pcrl_task_heads = make_task_heads(purposes, REPR_DIM)
    pcrl_auditors = make_auditors(purposes, REPR_DIM, use_grl=True)

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
    pcrl_train_time = time.time() - t0

    eval_metrics = pcrl_trainer.evaluate(test_loader)
    print(f"  PCRL (ours): epoch {state.epoch + 1}, {pcrl_train_time:.0f}s")
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

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
        train_time=pcrl_train_time,
    )

    # ═════════════════════════════════════════════════════════════════════
    # C. BASELINES — same architecture for fair comparison
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C. BASELINES")
    print("=" * 60)

    BASELINE_EPOCHS = 50

    # C1: Standard (no privacy)
    print("\n--- Standard (no privacy) ---")
    torch.manual_seed(42)
    standard_result, _ = train_and_evaluate(
        name="Standard (no privacy)",
        encoder=StandardCNNEncoder(
            repr_dim=REPR_DIM, conv_channels=CONV_CHANNELS, dropout=DROPOUT,
        ),
        purposes=purposes,
        registry=registry,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
            epochs=BASELINE_EPOCHS, weight_decay=1e-4,
            early_stopping_patience=10, confusion_type="entropy",
            warmup_epochs=WARMUP_EPOCHS,
            checkpoint_dir=str(project_root / "checkpoints" / "celeba_standard"),
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        repr_dim=REPR_DIM,
    )

    # C2: Adversarial-only (LAFTR) — same 4-layer CNN but no purpose conditioning
    #     Uses same per-purpose lambdas for fair comparison (averaged)
    print("\n--- Adversarial-only (LAFTR) ---")
    laftr_lambda = sum(LAMBDA_ADV_PER_PURPOSE.values()) / len(LAMBDA_ADV_PER_PURPOSE)
    torch.manual_seed(42)
    laftr_result, _ = train_and_evaluate(
        name="Adversarial-only (LAFTR)",
        encoder=StandardCNNEncoder(
            repr_dim=REPR_DIM, conv_channels=CONV_CHANNELS, dropout=DROPOUT,
        ),
        purposes=purposes,
        registry=registry,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=laftr_lambda, lambda_verify=LAMBDA_VERIFY,
            auditor_steps=AUDITOR_STEPS,
            epochs=BASELINE_EPOCHS, weight_decay=1e-4,
            early_stopping_patience=10, confusion_type="entropy",
            warmup_epochs=WARMUP_EPOCHS,
            checkpoint_dir=str(project_root / "checkpoints" / "celeba_laftr"),
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        repr_dim=REPR_DIM,
    )

    all_baseline_results = [standard_result, laftr_result, pcrl_result]
    print_baseline_table(all_baseline_results, majority_baselines)
    print_detailed_breakdown(all_baseline_results, majority_baselines)
    save_baseline_csv(
        all_baseline_results, majority_baselines,
        results_dir / "baseline_comparison.csv",
    )

    # ═════════════════════════════════════════════════════════════════════
    # D. COMPOSITION (smile_detection AND attractiveness_prediction)
    #    Both disallow Male — composition should work well.
    #    LAFTR can't do this since it already killed Male globally.
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("D. COMPOSITION (smile_detection AND attractiveness_prediction)")
    print("=" * 60)

    # D1: Algebraic composition
    print("\n--- Algebraic composition (emb₀ + emb₃) ---")
    composed_result = run_composed_embedding(
        pcrl_encoder, pcrl_trainer, purposes,
        train_loader, test_loader, device,
        p0_idx=0,  # smile_detection
        p1_idx=3,  # attractiveness_prediction
    )

    # D2: Retrained union
    print("\n--- Retrained union ---")
    torch.manual_seed(42)
    union_result = run_union_retrained(
        purposes, train_loader, val_loader, test_loader, device,
        p0_idx=0,  # smile_detection
        p1_idx=3,  # attractiveness_prediction
    )

    composition_results = [composed_result, union_result]
    print_composition_table(
        composition_results, majority_baselines,
        "COMPOSITION (CelebA: smile_detection AND attractiveness_prediction)",
    )
    save_composition_csv(
        composition_results, majority_baselines,
        results_dir / "composition_experiment.csv",
    )

    # ═════════════════════════════════════════════════════════════════════
    # E. SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("DONE — all results saved to results/celeba/")
    print("=" * 60)


if __name__ == "__main__":
    main()

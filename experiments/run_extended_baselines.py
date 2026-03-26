#!/usr/bin/env python3
"""Extended baselines: INLP and LEACE on Adult and HAR datasets.

Compares PCRL against post-hoc debiasing methods (INLP, LEACE) that
erase linear information about disallowed attributes after training a
standard encoder.

Key limitation: INLP/LEACE produce ONE erased representation. They
can't handle different disallowed attrs for different purposes without
separate projections, and can't handle conflicting constraints.

Results saved to results/adult/extended_baselines.csv and
results/har_real/extended_baselines.csv.
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
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.baselines import INLPProjector, LEACEEraser
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.purposes.verification import certified_accuracy_bound
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


# ── Shared utilities ─────────────────────────────────────────────────────


@dataclass
class MethodResult:
    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0
    supports_multi_purpose: bool = False


def extract_representations(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: str,
    purpose_idx: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract representations, task labels, and sensitive attrs from a loader."""
    encoder.eval()
    all_reprs, all_tasks, all_sens = [], {}, {}

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            if purpose_idx is not None:
                h = encoder(x, purpose_idx)
            else:
                h = encoder(x)
            all_reprs.append(h.cpu().numpy())

            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())

    reprs = np.concatenate(all_reprs)
    tasks = {k: np.concatenate(v) for k, v in all_tasks.items()}
    sens = {k: np.concatenate(v) for k, v in all_sens.items()}
    return reprs, tasks, sens


def evaluate_task_accuracy(
    train_reprs: np.ndarray,
    train_labels: np.ndarray,
    test_reprs: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Train a logistic regression classifier and return test accuracy."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(train_reprs, train_labels)
    return float(clf.score(test_reprs, test_labels))


def compute_majority_baselines(
    labels: np.ndarray,
) -> tuple[float, int]:
    """Return majority proportion and num_classes."""
    unique, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels)), len(unique)


def run_compliance_on_reprs(
    train_reprs: np.ndarray,
    test_reprs: np.ndarray,
    train_sens: dict[str, np.ndarray],
    test_sens: dict[str, np.ndarray],
    purpose_name: str,
    disallowed_attrs: list[str],
) -> list[ComplianceReport]:
    """Run linear + empirical audit on pre-computed representations."""
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    reports = []

    for attr_name in disallowed_attrs:
        train_labels = train_sens[attr_name]
        test_labels = test_sens[attr_name]

        linear_result, null_result = linear_audit.audit(test_reprs, test_labels)

        all_labels = np.concatenate([train_labels, test_labels])
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        num_classes = len(unique_classes)
        chance_acc = 1.0 / max(num_classes, 1)
        majority_proportion = float(class_counts.max() / len(all_labels))

        best_acc, emp_results = empirical_audit.audit(
            train_reprs, train_labels, test_reprs, test_labels,
        )

        empirical_ok = (best_acc - chance_acc) < 0.05
        certified = linear_result.certified and empirical_ok

        reports.append(ComplianceReport(
            purpose_name=purpose_name,
            attr_name=attr_name,
            linear_r2=linear_result.r_squared,
            linear_certified=linear_result.certified,
            null_space_r2=null_result.r_squared,
            variance_preserved=null_result.variance_preserved or 1.0,
            empirical_best_acc=best_acc,
            empirical_chance_acc=chance_acc,
            empirical_results=emp_results,
            certified=certified,
            majority_proportion=majority_proportion,
            num_classes=num_classes,
        ))

    return reports


# ── Standard encoder training ───────────────────────────────────────────


def train_standard_encoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    purposes: list[PurposeSpec],
    input_dim: int,
    hidden_dims: list[int],
    repr_dim: int,
    device: str,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> tuple[StandardEncoder, float]:
    """Train a standard encoder (no privacy) and return it with training time."""
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = StandardEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        repr_dim=repr_dim,
        dropout=0.3,
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
            hidden_dim=64, num_layers=2,
        )

    config = TrainerConfig(
        batch_size=batch_size,
        lr_encoder=lr,
        lr_auditor=lr,
        lambda_adv=0.0,
        lambda_verify=0.0,
        auditor_steps=1,
        epochs=epochs,
        weight_decay=1e-4,
        early_stopping_patience=15,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    return encoder, train_time


def train_pcrl_encoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    purposes: list[PurposeSpec],
    input_dim: int,
    hidden_dims: list[int],
    repr_dim: int,
    device: str,
    lambda_adv: float = 50.0,
    lambda_verify: float = 50.0,
    auditor_steps: int = 10,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> tuple[PurposeConditionedEncoder, PurposeRegistry, float]:
    """Train a PCRL encoder and return it with registry and training time."""
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        repr_dim=repr_dim,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
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
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=batch_size,
        lr_encoder=lr,
        lr_auditor=lr,
        lambda_adv=lambda_adv,
        lambda_verify=lambda_verify,
        auditor_steps=auditor_steps,
        epochs=epochs,
        weight_decay=1e-4,
        early_stopping_patience=15,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    return encoder, registry, train_time


# ── INLP and LEACE runners ──────────────────────────────────────────────


def run_inlp(
    encoder: StandardEncoder,
    purpose: PurposeSpec,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> MethodResult:
    """Run INLP on a single purpose: erase all disallowed attrs."""
    t0 = time.time()
    train_reprs, train_tasks, train_sens = extract_representations(
        encoder, train_loader, device,
    )
    test_reprs, test_tasks, test_sens = extract_representations(
        encoder, test_loader, device,
    )

    # Apply INLP for each disallowed attribute sequentially
    proj_train = train_reprs.copy()
    proj_test = test_reprs.copy()
    for attr_name in purpose.disallowed_attrs:
        projector = INLPProjector(max_iters=35, min_accuracy=0.52)
        projector.fit(proj_train, train_sens[attr_name])
        proj_train = projector.transform(proj_train)
        proj_test = projector.transform(proj_test)
        print(f"    INLP {attr_name}: {projector.num_iters_used} iters")

    elapsed = time.time() - t0

    # Evaluate task accuracy on projected representations
    task_name = purpose.allowed_tasks[0]
    task_acc = evaluate_task_accuracy(
        proj_train, train_tasks[task_name],
        proj_test, test_tasks[task_name],
    )

    # Compliance audit on projected representations
    reports = run_compliance_on_reprs(
        proj_train, proj_test, train_sens, test_sens,
        purpose.name, purpose.disallowed_attrs,
    )

    return MethodResult(
        name="INLP",
        task_accuracies={task_name: task_acc},
        reports=reports,
        train_time=elapsed,
        supports_multi_purpose=False,
    )


def run_leace(
    encoder: StandardEncoder,
    purpose: PurposeSpec,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> MethodResult:
    """Run LEACE on a single purpose: erase all disallowed attrs."""
    t0 = time.time()
    train_reprs, train_tasks, train_sens = extract_representations(
        encoder, train_loader, device,
    )
    test_reprs, test_tasks, test_sens = extract_representations(
        encoder, test_loader, device,
    )

    # Apply LEACE for each disallowed attribute sequentially
    proj_train = train_reprs.copy()
    proj_test = test_reprs.copy()
    for attr_name in purpose.disallowed_attrs:
        eraser = LEACEEraser()
        eraser.fit(proj_train, train_sens[attr_name])
        proj_train = eraser.transform(proj_train)
        proj_test = eraser.transform(proj_test)

    elapsed = time.time() - t0

    # Evaluate task accuracy on projected representations
    task_name = purpose.allowed_tasks[0]
    task_acc = evaluate_task_accuracy(
        proj_train, train_tasks[task_name],
        proj_test, test_tasks[task_name],
    )

    # Compliance audit on projected representations
    reports = run_compliance_on_reprs(
        proj_train, proj_test, train_sens, test_sens,
        purpose.name, purpose.disallowed_attrs,
    )

    return MethodResult(
        name="LEACE",
        task_accuracies={task_name: task_acc},
        reports=reports,
        train_time=elapsed,
        supports_multi_purpose=False,
    )


def run_standard_baseline(
    encoder: StandardEncoder,
    purpose: PurposeSpec,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    train_time: float = 0.0,
) -> MethodResult:
    """Evaluate standard encoder (no privacy) on a single purpose."""
    train_reprs, train_tasks, train_sens = extract_representations(
        encoder, train_loader, device,
    )
    test_reprs, test_tasks, test_sens = extract_representations(
        encoder, test_loader, device,
    )

    task_name = purpose.allowed_tasks[0]
    task_acc = evaluate_task_accuracy(
        train_reprs, train_tasks[task_name],
        test_reprs, test_tasks[task_name],
    )

    reports = run_compliance_on_reprs(
        train_reprs, test_reprs, train_sens, test_sens,
        purpose.name, purpose.disallowed_attrs,
    )

    return MethodResult(
        name="Standard (no privacy)",
        task_accuracies={task_name: task_acc},
        reports=reports,
        train_time=train_time,
        supports_multi_purpose=False,
    )


def run_pcrl_baseline(
    encoder: PurposeConditionedEncoder,
    registry: PurposeRegistry,
    purpose: PurposeSpec,
    purpose_idx: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    train_time: float = 0.0,
) -> MethodResult:
    """Evaluate PCRL encoder on a single purpose."""
    train_reprs, train_tasks, train_sens = extract_representations(
        encoder, train_loader, device, purpose_idx=purpose_idx,
    )
    test_reprs, test_tasks, test_sens = extract_representations(
        encoder, test_loader, device, purpose_idx=purpose_idx,
    )

    task_name = purpose.allowed_tasks[0]
    task_acc = evaluate_task_accuracy(
        train_reprs, train_tasks[task_name],
        test_reprs, test_tasks[task_name],
    )

    reports = run_compliance_on_reprs(
        train_reprs, test_reprs, train_sens, test_sens,
        purpose.name, purpose.disallowed_attrs,
    )

    return MethodResult(
        name="PCRL (ours)",
        task_accuracies={task_name: task_acc},
        reports=reports,
        train_time=train_time,
        supports_multi_purpose=True,
    )


# ── Printing and saving ─────────────────────────────────────────────────


def print_comparison_table(
    results: list[MethodResult],
    task_name: str,
    dataset_name: str,
) -> None:
    print("\n" + "=" * 120)
    print(f"EXTENDED BASELINE COMPARISON ({dataset_name})")
    print("=" * 120)

    all_attrs = []
    for res in results:
        for r in res.reports:
            if r.attr_name not in all_attrs:
                all_attrs.append(r.attr_name)

    hdr = f"{'Method':<24} {task_name[:12]:>13}"
    for attr in all_attrs:
        hdr += f" {attr[:10]+'Δ':>12}"
    hdr += f" {'Pass':>7} {'Multi-P?':>9} {'Time':>7}"
    print(hdr)
    print("-" * len(hdr))

    for result in results:
        task_acc = result.task_accuracies.get(task_name, 0.0)
        row = f"{result.name:<24} {task_acc:>12.1%}"

        deltas: dict[str, float] = {}
        pass_count = total = 0
        for r in result.reports:
            delta = r.empirical_best_acc - r.majority_proportion
            deltas[r.attr_name] = max(deltas.get(r.attr_name, -1.0), delta)
            total += 1
            if delta < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        for attr in all_attrs:
            d = deltas.get(attr, 0.0)
            row += f" {d:>+11.1%}"

        multi = "Yes" if result.supports_multi_purpose else "No"
        row += f" {pass_count:>3}/{total} {multi:>9} {result.train_time:>6.0f}s"
        print(row)

    print("=" * 120)


def print_detailed_table(results: list[MethodResult]) -> None:
    print("\nDETAILED PER-ATTRIBUTE BREAKDOWN")
    print("-" * 110)
    print(f"{'Method':<24} {'Purpose':<22} {'Attribute':<14} "
          f"{'Best':>7} {'MajBL':>7} {'Delta':>8} {'R²':>8} {'Bound':>8} {'Status':>8}")
    print("-" * 110)
    for result in results:
        for r in result.reports:
            d = r.empirical_best_acc - r.majority_proportion
            ok = d < 0.02 and r.linear_r2 < 0.05
            bound = certified_accuracy_bound(
                r.linear_r2, r.majority_proportion, r.num_classes,
            )
            print(f"{result.name:<24} {r.purpose_name:<22} {r.attr_name:<14} "
                  f"{r.empirical_best_acc:>6.1%} {r.majority_proportion:>6.1%} "
                  f"{d:>+7.1%} {r.linear_r2:>7.4f} {bound:>7.1%} "
                  f"{'PASS' if ok else 'FAIL':>8}")
    print()


def save_csv(
    results: list[MethodResult],
    path: Path,
) -> None:
    rows = []
    for result in results:
        for r in result.reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            bound = certified_accuracy_bound(
                r.linear_r2, r.majority_proportion, r.num_classes,
            )
            task_accs = {f"{t}_acc": round(v, 4)
                         for t, v in result.task_accuracies.items()}
            rows.append({
                "method": result.name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                **task_accs,
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(r.majority_proportion, 4),
                "delta": round(delta, 4),
                "linear_r2": round(r.linear_r2, 4),
                "bound": round(bound, 4),
                "adj_pass": ok,
                "supports_multi_purpose": result.supports_multi_purpose,
                "train_time_s": round(result.train_time, 1),
            })

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ADULT EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run_adult_experiment(device: str) -> None:
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    print("\n" + "#" * 70)
    print("# ADULT DATASET — Extended Baselines")
    print("#" * 70)

    purposes = get_adult_purposes()
    # Focus on income_prediction (purpose 0): disallow race, sex
    purpose = purposes[0]
    print(f"\nPurpose: {purpose.name}")
    print(f"  tasks={purpose.allowed_tasks}, disallowed={purpose.disallowed_attrs}")

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False)

    input_dim = train_ds.info.num_features
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, "
          f"Features: {input_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    hidden_dims = [128, 128]
    repr_dim = 64

    # 1. Train standard encoder (shared by Standard, INLP, LEACE)
    print("\n--- Training standard encoder ---")
    std_encoder, std_time = train_standard_encoder(
        train_loader, val_loader, purposes, input_dim,
        hidden_dims, repr_dim, device, epochs=100,
    )
    print(f"  Standard encoder trained in {std_time:.0f}s")

    # 2. Standard baseline (no privacy)
    print("\n--- Standard (no privacy) ---")
    standard_result = run_standard_baseline(
        std_encoder, purpose, train_loader, test_loader, device, std_time,
    )
    print(f"  {purpose.allowed_tasks[0]} acc: "
          f"{standard_result.task_accuracies[purpose.allowed_tasks[0]]:.1%}")

    # 3. INLP
    print("\n--- INLP ---")
    inlp_result = run_inlp(
        std_encoder, purpose, train_loader, test_loader, device,
    )
    print(f"  {purpose.allowed_tasks[0]} acc: "
          f"{inlp_result.task_accuracies[purpose.allowed_tasks[0]]:.1%}")

    # 4. LEACE
    print("\n--- LEACE ---")
    leace_result = run_leace(
        std_encoder, purpose, train_loader, test_loader, device,
    )
    print(f"  {purpose.allowed_tasks[0]} acc: "
          f"{leace_result.task_accuracies[purpose.allowed_tasks[0]]:.1%}")

    # 5. PCRL
    print("\n--- PCRL ---")
    pcrl_encoder, pcrl_registry, pcrl_time = train_pcrl_encoder(
        train_loader, val_loader, purposes, input_dim,
        hidden_dims, repr_dim, device,
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
        epochs=100,
    )
    pcrl_result = run_pcrl_baseline(
        pcrl_encoder, pcrl_registry, purpose, 0,
        train_loader, test_loader, device, pcrl_time,
    )
    print(f"  {purpose.allowed_tasks[0]} acc: "
          f"{pcrl_result.task_accuracies[purpose.allowed_tasks[0]]:.1%}")

    # 6. Print and save
    all_results = [standard_result, inlp_result, leace_result, pcrl_result]
    print_comparison_table(all_results, purpose.allowed_tasks[0], "Adult — income_prediction")
    print_detailed_table(all_results)

    results_dir = project_root / "results" / "adult"
    save_csv(all_results, results_dir / "extended_baselines.csv")


# ═══════════════════════════════════════════════════════════════════════════
# HAR EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run_har_experiment(device: str) -> None:
    from pcrl.data.har import HARDataset, get_har_purposes, N_SUBJECTS, N_ACTIVITIES

    print("\n" + "#" * 70)
    print("# HAR DATASET — Extended Baselines")
    print("#" * 70)

    purposes = get_har_purposes()
    # Focus on activity_recognition (purpose 0): disallow subject_id
    purpose = purposes[0]
    print(f"\nPurpose: {purpose.name}")
    print(f"  tasks={purpose.allowed_tasks}, disallowed={purpose.disallowed_attrs}")

    train_ds = HARDataset(purposes=purposes, root="data", split="train")
    val_ds = HARDataset(purposes=purposes, root="data", split="val")
    test_ds = HARDataset(purposes=purposes, root="data", split="test")

    input_dim = train_ds.info.num_features
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, "
          f"Features: {input_dim}")

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    hidden_dims = [256, 128]
    repr_dim = 16  # Small repr_dim acts as bottleneck (best from HAR sweep)

    # 1. Train standard encoder
    print("\n--- Training standard encoder ---")
    std_encoder, std_time = train_standard_encoder(
        train_loader, val_loader, purposes, input_dim,
        hidden_dims, repr_dim, device, epochs=100,
    )
    print(f"  Standard encoder trained in {std_time:.0f}s")

    # 2. Standard baseline
    print("\n--- Standard (no privacy) ---")
    standard_result = run_standard_baseline(
        std_encoder, purpose, train_loader, test_loader, device, std_time,
    )
    print(f"  activity acc: "
          f"{standard_result.task_accuracies.get('activity', 0):.1%}")

    # 3. INLP
    print("\n--- INLP ---")
    inlp_result = run_inlp(
        std_encoder, purpose, train_loader, test_loader, device,
    )
    print(f"  activity acc: "
          f"{inlp_result.task_accuracies.get('activity', 0):.1%}")

    # 4. LEACE
    print("\n--- LEACE ---")
    leace_result = run_leace(
        std_encoder, purpose, train_loader, test_loader, device,
    )
    print(f"  activity acc: "
          f"{leace_result.task_accuracies.get('activity', 0):.1%}")

    # 5. PCRL
    print("\n--- PCRL ---")
    pcrl_encoder, pcrl_registry, pcrl_time = train_pcrl_encoder(
        train_loader, val_loader, purposes, input_dim,
        hidden_dims, repr_dim, device,
        lambda_adv=2.0, lambda_verify=1.0, auditor_steps=20,
        epochs=100,
    )
    pcrl_result = run_pcrl_baseline(
        pcrl_encoder, pcrl_registry, purpose, 0,
        train_loader, test_loader, device, pcrl_time,
    )
    print(f"  activity acc: "
          f"{pcrl_result.task_accuracies.get('activity', 0):.1%}")

    # 6. Print and save
    all_results = [standard_result, inlp_result, leace_result, pcrl_result]
    print_comparison_table(all_results, "activity", "HAR — activity_recognition")
    print_detailed_table(all_results)

    results_dir = project_root / "results" / "har_real"
    save_csv(all_results, results_dir / "extended_baselines.csv")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    run_adult_experiment(device)
    run_har_experiment(device)

    print("\n" + "=" * 60)
    print("DONE — extended baselines complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

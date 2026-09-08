#!/usr/bin/env python3
"""Compare empirical leakage under a covariate shift.

Differences in least-squares scores and attacker accuracies are descriptive;
the retired R²-to-classification bounds cannot certify privacy under shift.
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
from torch.utils.data import DataLoader, Subset

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
    print_compliance_table,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


# ── Constants (matching run_adult.py best config) ────────────────────────
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 10
EPOCHS = 80
PATIENCE = 15


def extract_all_representations(
    encoder: torch.nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract representations, task labels, and sensitive attrs."""
    encoder.eval()
    all_reprs: list[np.ndarray] = []
    all_tasks: dict[str, list[np.ndarray]] = {}
    all_sens: dict[str, list[np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reprs.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())

    reprs = np.concatenate(all_reprs)
    tasks = {k: np.concatenate(v) for k, v in all_tasks.items()}
    sens = {k: np.concatenate(v) for k, v in all_sens.items()}
    return reprs, tasks, sens


def run_compliance_audit(
    train_reprs: np.ndarray,
    test_reprs: np.ndarray,
    train_sens: dict[str, np.ndarray],
    test_sens: dict[str, np.ndarray],
    purpose_name: str,
    disallowed_attrs: list[str],
    disallowed_attr_dims: dict[str, int],
) -> list[ComplianceReport]:
    """Run empirical least-squares and classification audits on representations."""
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    reports = []

    for attr_name in disallowed_attrs:
        train_labels = train_sens[attr_name]
        test_labels = test_sens[attr_name]

        # Check we have enough classes in the shifted subset
        unique_test = np.unique(test_labels)
        if len(unique_test) < 2:
            print(f"    Skipping {attr_name}: only {len(unique_test)} class(es) in test")
            continue

        linear_result, null_result = linear_audit.audit(test_reprs, test_labels)

        all_labels = np.concatenate([train_labels, test_labels])
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        num_classes = len(unique_classes)
        chance_acc = 1.0 / max(num_classes, 1)
        majority_proportion = float(class_counts.max() / len(all_labels))

        # Majority proportion for the test set specifically
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        test_majority = float(test_counts.max() / len(test_labels))

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
            majority_proportion=test_majority,
            num_classes=len(test_unique),
        ))

    return reports


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    # ── Load data ─────────────────────────────────────────────────────────
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False)

    print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
          f"Features={train_ds.info.num_features}")

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

    # ── Create distribution-shifted subset ────────────────────────────────
    # Education level 3 = Bachelor's, Master's, Doctorate, Prof-school
    edu_labels = test_ds.task_labels["education_level"]
    shifted_mask = edu_labels == 3
    shifted_indices = torch.where(shifted_mask)[0].tolist()

    shifted_ds = Subset(test_ds, shifted_indices)
    shifted_loader = DataLoader(
        shifted_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    print(f"\nDistribution shift: education >= Bachelor's")
    print(f"  Full test set:    {len(test_ds)} samples")
    print(f"  Shifted subset:   {len(shifted_ds)} samples "
          f"({len(shifted_ds)/len(test_ds):.1%} of test)")

    # Show demographic differences
    print("\n  Demographic comparison (full test vs shifted):")
    for attr in ["sex", "race", "income", "age_group", "marital_status"]:
        full_vals = test_ds.sensitive_attrs[attr]
        shift_vals = full_vals[shifted_indices]
        full_maj = float(torch.bincount(full_vals.long()).max() / len(full_vals))
        shift_maj = float(torch.bincount(shift_vals.long()).max() / len(shift_vals))
        print(f"    {attr:<18} majority: {full_maj:.1%} -> {shift_maj:.1%}")

    # Income distribution shift (key metric)
    full_income = test_ds.task_labels["income"]
    shift_income = full_income[shifted_indices]
    full_high = float((full_income == 1).sum() / len(full_income))
    shift_high = float((shift_income == 1).sum() / len(shift_income))
    print(f"\n  Income >50K rate:  {full_high:.1%} -> {shift_high:.1%}")

    # ── Train PCRL ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING PCRL")
    print("=" * 70)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=train_ds.info.num_features,
        hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=len(purposes), purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film", dropout=DROPOUT,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS, epochs=EPOCHS,
        weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    print(f"Trained {state.epoch + 1} epochs in {train_time:.0f}s")

    # ── Extract representations for all purposes ──────────────────────────
    print("\n" + "=" * 70)
    print("EXTRACTING REPRESENTATIONS")
    print("=" * 70)

    full_reports: list[ComplianceReport] = []
    shifted_reports: list[ComplianceReport] = []

    for purpose_idx, purpose in enumerate(purposes):
        print(f"\n  Purpose: {purpose.name}")
        print(f"    Tasks: {purpose.allowed_tasks}, "
              f"Disallowed: {purpose.disallowed_attrs}")

        # Extract for full train + full test
        train_reprs, train_tasks, train_sens = extract_all_representations(
            encoder, train_loader, purpose_idx, device,
        )
        full_reprs, full_tasks, full_sens = extract_all_representations(
            encoder, test_loader, purpose_idx, device,
        )
        shifted_reprs, shifted_tasks, shifted_sens = extract_all_representations(
            encoder, shifted_loader, purpose_idx, device,
        )

        print(f"    Full test reprs: {full_reprs.shape}")
        print(f"    Shifted reprs:   {shifted_reprs.shape}")

        # Run compliance on full test
        print("    Auditing full test set...")
        full_r = run_compliance_audit(
            train_reprs, full_reprs, train_sens, full_sens,
            purpose.name, purpose.disallowed_attrs,
            purpose.disallowed_attr_dims,
        )
        full_reports.extend(full_r)

        # Run compliance on shifted test
        print("    Auditing shifted subset...")
        shifted_r = run_compliance_audit(
            train_reprs, shifted_reprs, train_sens, shifted_sens,
            purpose.name, purpose.disallowed_attrs,
            purpose.disallowed_attr_dims,
        )
        shifted_reports.extend(shifted_r)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL TEST SET — Compliance Audit")
    print("=" * 70)
    print_compliance_table(full_reports)

    print("\n" + "=" * 70)
    print("SHIFTED SUBSET (Bachelor's+) — Compliance Audit")
    print("=" * 70)
    print_compliance_table(shifted_reports)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("DISTRIBUTION SHIFT COMPARISON")
    print("=" * 110)
    header = (
        f"{'Purpose':<25} {'Attribute':<14} "
        f"{'Full R²':>8} {'Shift R²':>9} {'Delta R²':>9} "
        f"{'Full Bound':>11} {'Shift Bound':>12} "
        f"{'Full NL':>8} {'Shift NL':>9} "
        f"{'Full Emp':>9} {'Shift Emp':>10} {'Holds?':>7}"
    )
    print(header)
    print("-" * 110)

    comparison_rows = []
    for fr, sr in zip(full_reports, shifted_reports):
        delta_r2 = sr.linear_r2 - fr.linear_r2

        # The classification-accuracy guarantee is retired as invalid.
        holds = None
        holds_str = "retired"

        f_nl = f"{fr.nonlinear_bound:.1%}" if fr.nonlinear_bound is not None else "N/A"
        s_nl = f"{sr.nonlinear_bound:.1%}" if sr.nonlinear_bound is not None else "N/A"

        print(
            f"{fr.purpose_name:<25} {fr.attr_name:<14} "
            f"{fr.linear_r2:>8.4f} {sr.linear_r2:>9.4f} {delta_r2:>+9.4f} "
            f"{'retired':>10} {'retired':>11} "
            f"{f_nl:>8} {s_nl:>9} "
            f"{fr.empirical_best_acc:>8.1%} {sr.empirical_best_acc:>9.1%} "
            f"{holds_str:>7}"
        )

        comparison_rows.append({
            "purpose": fr.purpose_name,
            "attribute": fr.attr_name,
            "full_r2": round(fr.linear_r2, 6),
            "shifted_r2": round(sr.linear_r2, 6),
            "delta_r2": round(delta_r2, 6),
            "full_linear_bound": None,
            "shifted_linear_bound": None,
            "full_nonlinear_bound": round(fr.nonlinear_bound, 4) if fr.nonlinear_bound is not None else "",
            "shifted_nonlinear_bound": round(sr.nonlinear_bound, 4) if sr.nonlinear_bound is not None else "",
            "full_empirical_acc": round(fr.empirical_best_acc, 4),
            "shifted_empirical_acc": round(sr.empirical_best_acc, 4),
            "full_majority": round(fr.majority_proportion, 4),
            "shifted_majority": round(sr.majority_proportion, 4),
            "bound_holds": holds,
        })

    print("-" * 110)
    print("Classification guarantees under shift: unavailable (invalid bound retired)")
    print("=" * 110)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = project_root / "results" / "adult" / "distribution_shift.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"\nSaved {csv_path}")

    print("\n" + "=" * 70)
    print("DONE — distribution shift experiment complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retired experiment: its R²-to-classification-accuracy premise is false.

The entry point fails before training. Historical outputs are retained only
as artifacts and must not be interpreted as mathematical guarantees.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
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
from pcrl.data.har import HARDataset, get_har_purposes, N_SUBJECTS, N_ACTIVITIES
from pcrl.evaluation.certificates import (
    ComplianceReport,
    generate_report,
    print_compliance_table,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.purposes.verification import certified_accuracy_bound
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


def save_nonlinear_csv(reports: list[ComplianceReport], path: Path) -> None:
    """Save compliance reports with nonlinear bounds to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "purpose", "attribute", "linear_r2", "linear_certified",
            "linear_bound", "nonlinear_bound", "nonlinear_sigma",
            "empirical_best_acc", "majority_proportion", "num_classes",
            "certified",
        ])
        writer.writeheader()
        for r in reports:
            lin_bound = certified_accuracy_bound(
                r.linear_r2, r.majority_proportion, r.num_classes,
            )
            writer.writerow({
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "linear_r2": round(r.linear_r2, 6),
                "linear_certified": r.linear_certified,
                "linear_bound": round(lin_bound, 4),
                "nonlinear_bound": round(r.nonlinear_bound, 4) if r.nonlinear_bound is not None else "",
                "nonlinear_sigma": r.nonlinear_best_sigma if r.nonlinear_best_sigma is not None else "",
                "empirical_best_acc": round(r.empirical_best_acc, 4),
                "majority_proportion": round(r.majority_proportion, 4),
                "num_classes": r.num_classes,
                "certified": r.certified,
            })
    print(f"Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# ADULT EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_adult(device: str) -> list[ComplianceReport]:
    """Train PCRL on Adult and generate compliance reports with nonlinear bounds."""
    print("\n" + "=" * 70)
    print("ADULT DATASET — Nonlinear Compliance Certificates")
    print("=" * 70)

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False)

    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
          f"Features={train_ds.info.num_features}")

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

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=train_ds.info.num_features,
        hidden_dims=[128, 128], repr_dim=64,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="film", dropout=0.3,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=64, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=50.0, lambda_verify=50.0,
        auditor_steps=10, epochs=80,
        weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    print(f"  Trained {state.epoch + 1} epochs in {train_time:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    print("\n  Generating compliance reports with nonlinear bounds...")
    reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader, purpose_registry=registry,
        device=device,
    )

    print()
    print_compliance_table(reports)
    return reports


# ═══════════════════════════════════════════════════════════════════════════
# HAR EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_har(device: str) -> list[ComplianceReport]:
    """Train PCRL on HAR and generate compliance reports with nonlinear bounds."""
    print("\n" + "=" * 70)
    print("HAR DATASET — Nonlinear Compliance Certificates")
    print("=" * 70)

    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HARDataset(purposes=purposes, root="data", split="train")
    val_ds = HARDataset(purposes=purposes, root="data", split="val")
    test_ds = HARDataset(purposes=purposes, root="data", split="test")

    input_dim = train_ds[0]["features"].shape[0]
    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
          f"Features={input_dim}")

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

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128], repr_dim=16,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="film", dropout=0.3,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=16, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=16, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=2.0, lambda_verify=1.0,
        auditor_steps=20, epochs=100,
        weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    print(f"  Trained {state.epoch + 1} epochs in {train_time:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    print("\n  Generating compliance reports with nonlinear bounds...")
    reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader, purpose_registry=registry,
        device=device,
    )

    print()
    print_compliance_table(reports)
    return reports


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    raise NotImplementedError("Retired invalid accuracy certificate; see docs/ACCURACY_CERTIFICATE_RETIREMENT.md")
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Adult
    adult_reports = run_adult(device)
    adult_dir = project_root / "results" / "adult"
    save_nonlinear_csv(adult_reports, adult_dir / "nonlinear_certificates.csv")

    # HAR
    har_reports = run_har(device)
    har_dir = project_root / "results" / "har_real"
    save_nonlinear_csv(har_reports, har_dir / "nonlinear_certificates.csv")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for dataset, reports in [("Adult", adult_reports), ("HAR", har_reports)]:
        print(f"\n  {dataset}:")
        for r in reports:
            lin_bound = certified_accuracy_bound(
                r.linear_r2, r.majority_proportion, r.num_classes,
            )
            nl_str = f"{r.nonlinear_bound:.1%}" if r.nonlinear_bound is not None else "N/A"
            print(
                f"    {r.purpose_name}/{r.attr_name}: "
                f"LinBound={lin_bound:.1%}, NLBound={nl_str}, "
                f"EmpAcc={r.empirical_best_acc:.1%}, "
                f"sigma={r.nonlinear_best_sigma}"
            )

    print("\n" + "=" * 70)
    print("DONE — nonlinear certificate experiments complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

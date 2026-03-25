#!/usr/bin/env python3
"""Fix marital_status suppression on Adult dataset.

The marital_status pair (in employment_analysis purpose) is the only
consistent failure. This script tries targeted approaches:

1. Baseline: original config (λ_adv=50, λ_verify=50 globally)
2. Per-attribute weighting: λ_adv=200, λ_verify=100 for marital_status
3. Larger auditor for marital_status: 4 layers, 512 hidden, K=30

Reports whether marital_status delta drops below 5% and task accuracy impact.
Saves to results/adult/marital_status_fix.csv.
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

logging.basicConfig(level=logging.WARNING)

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import generate_report, ComplianceReport
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

# ── Base config (same as best Adult) ─────────────────────────────────────
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 200
PATIENCE = 15


@dataclass
class FixResult:
    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def compute_majority_baselines(dataset: AdultDataset) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for attr_name, labels in dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / len(labels)
    return baselines


def run_config(
    name: str,
    purposes,
    registry: PurposeRegistry,
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    config: TrainerConfig,
    auditor_hidden: int = 256,
    auditor_layers: int = 3,
) -> FixResult:
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

    task_heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        tn = p.allowed_tasks[0]
        od = p.allowed_task_dims.get(tn, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=od)

    auditors: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=auditor_hidden,
            num_layers=auditor_layers,
        )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    elapsed = time.time() - t0

    ev = trainer.evaluate(test_loader)
    reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader, purpose_registry=registry, device=device,
    )

    print(f"  {name}: epoch {state.epoch + 1}, {elapsed:.0f}s")
    for task, acc in ev.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    return FixResult(
        name=name,
        task_accuracies=ev.task_accuracy,
        reports=reports,
        train_time=elapsed,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False)

    input_dim = train_ds.info.num_features
    print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, Feat={input_dim}")

    majority_baselines = compute_majority_baselines(test_ds)
    print("Majority baselines:", {k: f"{v:.1%}" for k, v in majority_baselines.items()})

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    results: list[FixResult] = []

    # ── 1. Baseline: original config ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Config 1: Baseline (λ_adv=50, λ_verify=50 global)")
    print(f"{'='*60}")
    results.append(run_config(
        "Baseline (λ=50/50)",
        purposes, registry, input_dim,
        train_loader, val_loader, test_loader, device,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
            epochs=EPOCHS, weight_decay=1e-4,
            early_stopping_patience=PATIENCE, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "adult_marstat_baseline"),
        ),
    ))

    # ── 2. Per-attribute weighting: boost marital_status ─────────────────
    print(f"\n{'='*60}")
    print("Config 2: Per-attr λ (marital_status: λ_adv=200, λ_ver=100)")
    print(f"{'='*60}")
    results.append(run_config(
        "Per-attr λ (mar=200/100)",
        purposes, registry, input_dim,
        train_loader, val_loader, test_loader, device,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
            epochs=EPOCHS, weight_decay=1e-4,
            early_stopping_patience=PATIENCE, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "adult_marstat_perattr"),
            lambda_adv_per_attr={"marital_status": 200.0},
            lambda_verify_per_attr={"marital_status": 100.0},
        ),
    ))

    # ── 3. Larger auditor for all + per-attr lambda ──────────────────────
    print(f"\n{'='*60}")
    print("Config 3: Larger auditor (4L/512H) + per-attr λ + K=30")
    print(f"{'='*60}")
    results.append(run_config(
        "Big aud + per-attr λ",
        purposes, registry, input_dim,
        train_loader, val_loader, test_loader, device,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=50.0, lambda_verify=50.0, auditor_steps=30,
            epochs=EPOCHS, weight_decay=1e-4,
            early_stopping_patience=PATIENCE, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "adult_marstat_bigaud"),
            lambda_adv_per_attr={"marital_status": 200.0},
            lambda_verify_per_attr={"marital_status": 100.0},
        ),
        auditor_hidden=512,
        auditor_layers=4,
    ))

    # ── 4. Even more aggressive: λ=300/150 + big auditor + K=30 ─────────
    print(f"\n{'='*60}")
    print("Config 4: Very aggressive (mar λ=300/150 + big aud + K=30)")
    print(f"{'='*60}")
    results.append(run_config(
        "Aggressive (mar=300/150)",
        purposes, registry, input_dim,
        train_loader, val_loader, test_loader, device,
        config=TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=50.0, lambda_verify=50.0, auditor_steps=30,
            epochs=EPOCHS, weight_decay=1e-4,
            early_stopping_patience=PATIENCE, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "adult_marstat_aggressive"),
            lambda_adv_per_attr={"marital_status": 300.0},
            lambda_verify_per_attr={"marital_status": 150.0},
        ),
        auditor_hidden=512,
        auditor_layers=4,
    ))

    # ── Print comparison table ───────────────────────────────────────────
    print("\n" + "=" * 110)
    print("MARITAL STATUS FIX COMPARISON (Adult Dataset)")
    print("=" * 110)

    header = (
        f"{'Config':<28} {'Income':>8} {'OccGrp':>8} {'EduLvl':>8} "
        f"{'Race Δ':>8} {'Sex Δ':>8} {'MarStat Δ':>10} {'AgeGrp Δ':>9} "
        f"{'Pass':>6} {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    csv_rows = []
    for result in results:
        income = result.task_accuracies.get("income", 0.0)
        occ = result.task_accuracies.get("occupation_group", 0.0)
        edu = result.task_accuracies.get("education_level", 0.0)

        deltas: dict[str, float] = {}
        pass_count = 0
        total = 0
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            deltas[r.attr_name] = max(deltas.get(r.attr_name, -1.0), d)
            total += 1
            if d < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        race_d = deltas.get("race", 0.0)
        sex_d = deltas.get("sex", 0.0)
        mar_d = deltas.get("marital_status", 0.0)
        age_d = deltas.get("age_group", 0.0)

        print(
            f"{result.name:<28} {income:>7.1%} {occ:>7.1%} {edu:>7.1%} "
            f"{race_d:>+7.1%} {sex_d:>+7.1%} {mar_d:>+9.1%} {age_d:>+8.1%} "
            f"{pass_count:>3}/{total} {result.train_time:>5.0f}s"
        )

        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            csv_rows.append({
                "config": result.name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "income_acc": round(income, 4),
                "occupation_group_acc": round(occ, 4),
                "education_level_acc": round(edu, 4),
                "best_emp_acc": round(r.empirical_best_acc, 4),
                "majority_baseline": round(bl, 4),
                "delta": round(d, 4),
                "linear_r2": round(r.linear_r2, 4),
                "adj_pass": ok,
                "train_time_s": round(result.train_time, 1),
            })

    print("=" * 110)

    # Focus on marital_status results
    print("\n" + "=" * 80)
    print("MARITAL STATUS DETAIL")
    print("=" * 80)
    detail_hdr = (
        f"{'Config':<28} {'Purpose':<22} "
        f"{'Best':>7} {'Base':>7} {'Delta':>8} {'R²':>8} {'Status':>8}"
    )
    print(detail_hdr)
    print("-" * len(detail_hdr))

    for result in results:
        for r in result.reports:
            if r.attr_name == "marital_status":
                bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
                d = r.empirical_best_acc - bl
                ok = d < 0.02 and r.linear_r2 < 0.05
                print(
                    f"{result.name:<28} {r.purpose_name:<22} "
                    f"{r.empirical_best_acc:>6.1%} {bl:>6.1%} "
                    f"{d:>+7.1%} {r.linear_r2:>8.4f} {'PASS' if ok else 'FAIL':>8}"
                )
    print("=" * 80)

    # ── Save CSV ─────────────────────────────────────────────────────────
    results_dir = project_root / "results" / "adult"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "marital_status_fix.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()

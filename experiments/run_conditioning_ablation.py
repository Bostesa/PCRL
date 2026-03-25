#!/usr/bin/env python3
"""Conditioning ablation on Adult dataset.

Runs the same PCRL training with all three conditioning types:
FiLM, concat, and attention. Everything else is identical.

Prints comparison table and saves to results/adult/conditioning_ablation.csv.
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

# ── Best Adult config from previous experiments ──────────────────────────
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 10
EPOCHS = 200
PATIENCE = 15
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3


@dataclass
class AblationResult:
    conditioning: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def compute_majority_baselines(dataset: AdultDataset) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for attr_name, labels in dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / len(labels)
    return baselines


def run_with_conditioning(
    conditioning: str,
    purposes,
    registry: PurposeRegistry,
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> AblationResult:
    torch.manual_seed(42)

    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning=conditioning,
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
            hidden_dim=AUDITOR_HIDDEN,
            num_layers=AUDITOR_LAYERS,
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
        checkpoint_dir=str(project_root / "checkpoints" / f"adult_cond_{conditioning}"),
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

    print(f"  {conditioning}: epoch {state.epoch + 1}, {elapsed:.0f}s")
    for task, acc in ev.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    return AblationResult(
        conditioning=conditioning,
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

    # ── Run all three conditioning types ─────────────────────────────────
    conditioning_types = ["film", "concat", "attention"]
    results: list[AblationResult] = []

    for cond in conditioning_types:
        print(f"\n{'='*60}")
        print(f"Training with conditioning: {cond}")
        print(f"{'='*60}")
        result = run_with_conditioning(
            cond, purposes, registry, input_dim,
            train_loader, val_loader, test_loader, device,
        )
        results.append(result)

    # ── Print comparison table ───────────────────────────────────────────
    print("\n" + "=" * 100)
    print("CONDITIONING ABLATION (Adult Dataset)")
    print("=" * 100)

    header = (
        f"{'Conditioning':<14} {'Income':>8} {'OccGrp':>8} {'EduLvl':>8} "
        f"{'Race Δ':>8} {'Sex Δ':>8} {'MarStat Δ':>10} {'AgeGrp Δ':>9} "
        f"{'Pass':>6} {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    rows = []
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
            f"{result.conditioning:<14} {income:>7.1%} {occ:>7.1%} {edu:>7.1%} "
            f"{race_d:>+7.1%} {sex_d:>+7.1%} {mar_d:>+9.1%} {age_d:>+8.1%} "
            f"{pass_count:>3}/{total} {result.train_time:>5.0f}s"
        )

        # CSV rows
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            rows.append({
                "conditioning": result.conditioning,
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

    print("=" * 100)

    # ── Detailed per-pair breakdown ──────────────────────────────────────
    print("\nDETAILED PER-PAIR BREAKDOWN")
    print("-" * 90)
    detail_hdr = (
        f"{'Cond':<10} {'Purpose':<22} {'Attribute':<16} "
        f"{'Best':>7} {'Base':>7} {'Delta':>8} {'R²':>8} {'Status':>8}"
    )
    print(detail_hdr)
    print("-" * len(detail_hdr))

    for result in results:
        for r in result.reports:
            bl = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            d = r.empirical_best_acc - bl
            ok = d < 0.02 and r.linear_r2 < 0.05
            print(
                f"{result.conditioning:<10} {r.purpose_name:<22} {r.attr_name:<16} "
                f"{r.empirical_best_acc:>6.1%} {bl:>6.1%} "
                f"{d:>+7.1%} {r.linear_r2:>8.4f} {'PASS' if ok else 'FAIL':>8}"
            )
        print()

    # ── Save CSV ─────────────────────────────────────────────────────────
    results_dir = project_root / "results" / "adult"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "conditioning_ablation.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CelebA LAFTR ensemble baseline — one StandardCNNEncoder per purpose.

LAFTR doesn't support multiple purposes natively (single shared representation
must suppress all disallowed attrs simultaneously). For a fair comparison with
PCRL's 4/13, we train 5 independent LAFTR models, each protecting only its
own purpose's disallowed attributes. The ensemble pass count aggregates
across all 5 models' purpose-attribute pairs (13 total).

This gives LAFTR every advantage: each model only needs to suppress 2-3
attributes (vs PCRL handling all 13 pairs with one encoder).

Config matches PCRL v2: lambda_adv=0.5, lambda_verify=0.3, 50 epochs,
batch=256, repr_dim=128, conv=(32,64,128), same max_samples.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress tqdm
import pcrl.training.trainer as _trainer_mod
class _Q:
    def __init__(self, iterable=None, *a, **k): self.iterable = iterable
    def __iter__(self): return iter(self.iterable) if self.iterable else iter([])
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_postfix(self, *a, **k): pass
    def update(self, *a): pass
    def close(self): pass
_trainer_mod.tqdm = _Q

import warnings
warnings.filterwarnings("ignore")

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.celeba import CelebADataset, get_celeba_purposes
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.cnn_encoder import StandardCNNEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

import logging
logging.basicConfig(level=logging.WARNING)

# ── Config (matches PCRL v2) ───────────────────────────────────────────
REPR_DIM = 128
CONV_CHANNELS = (32, 64, 128)
DROPOUT = 0.3
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_ADV = 0.5
LAMBDA_VERIFY = 0.3
AUDITOR_STEPS = 5
EPOCHS = 50
WARMUP_EPOCHS = 5
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3
MAX_TRAIN = 10000
MAX_VAL = 3000
MAX_TEST = 3000
SEED = 0

WALL_CAP_SECONDS = 8 * 3600  # 8 hours


class MultiTaskHead(nn.Module):
    """Task head returning a dict for multi-task purposes."""
    def __init__(self, heads: dict[str, nn.Module]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(heads)
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data once ──────────────────────────────────────────────────
    all_purposes = get_celeba_purposes()
    print(f"\nPurposes ({len(all_purposes)}):")
    for p in all_purposes:
        print(f"  {p.name}: tasks={p.allowed_tasks}, disallowed={p.disallowed_attrs}")

    print("\nLoading CelebA dataset...")
    train_ds = CelebADataset(all_purposes, root="data/celeba", split="train", max_samples=MAX_TRAIN)
    val_ds = CelebADataset(all_purposes, root="data/celeba", split="val", max_samples=MAX_VAL)
    test_ds = CelebADataset(all_purposes, root="data/celeba", split="test", max_samples=MAX_TEST)
    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    # Majority baselines from test set
    majority = {}
    for attr_name, labels in test_ds.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        majority[attr_name] = counts.max().item() / len(labels)
    print("\nMajority baselines:")
    for attr, baseline in sorted(majority.items()):
        print(f"  {attr}: {baseline:.1%}")

    # ── Train one LAFTR per purpose ─────────────────────────────────────
    all_rows = []
    out_path = project_root / "results" / "celeba" / "laftr_per_purpose.csv"
    purpose_summaries = []

    for purpose in all_purposes:
        if time.time() - t_start > WALL_CAP_SECONDS:
            print(f"\nWALL CAP REACHED ({WALL_CAP_SECONDS/3600:.0f}h), stopping early")
            break

        print(f"\n{'='*60}")
        print(f"LAFTR for purpose: {purpose.name}")
        print(f"  tasks={purpose.allowed_tasks}, disallowed={purpose.disallowed_attrs}")
        print(f"{'='*60}")

        try:
            # Fresh seed for each purpose
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            # Single-purpose registry
            registry = PurposeRegistry()
            registry.register(purpose)

            # StandardCNNEncoder — ignores purpose_idx
            encoder = StandardCNNEncoder(
                repr_dim=REPR_DIM,
                conv_channels=CONV_CHANNELS,
                dropout=DROPOUT,
            )
            total_params = sum(p.numel() for p in encoder.parameters())
            print(f"  Encoder: {total_params:,} parameters")

            # Task head(s) for this purpose
            task_heads: dict[str, nn.Module] = {}
            if len(purpose.allowed_tasks) == 1:
                task_name = purpose.allowed_tasks[0]
                output_dim = purpose.allowed_task_dims.get(task_name, 2)
                task_heads[purpose.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
            else:
                sub = {}
                for task_name in purpose.allowed_tasks:
                    output_dim = purpose.allowed_task_dims.get(task_name, 2)
                    sub[task_name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
                task_heads[purpose.name] = MultiTaskHead(sub)

            # Auditor for this purpose's disallowed attributes
            auditors: dict[str, nn.Module] = {}
            auditors[purpose.name] = MultiAttributeAuditor(
                repr_dim=REPR_DIM,
                attr_output_dims=purpose.disallowed_attr_dims,
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
                early_stopping_patience=None,
                log_interval=100,
                confusion_type="entropy",
                warmup_epochs=WARMUP_EPOCHS,
                gradient_reversal=False,
                checkpoint_dir=str(project_root / "checkpoints" / f"celeba_laftr_{purpose.name}"),
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
            print(f"  Training complete — epoch {state.epoch + 1}, {train_time:.0f}s")

            # Evaluate task accuracy
            eval_metrics = trainer.evaluate(test_loader)
            task_accs = eval_metrics.task_accuracy
            print(f"  Task accuracies: {task_accs}")

            # Compliance audit using fixed generate_report
            reports = generate_report(
                encoder=encoder,
                train_loader=train_loader,
                test_loader=test_loader,
                purpose_registry=registry,
                device=device,
            )

            pass_count = 0
            purpose_pairs = len(reports)
            for r in reports:
                delta = r.empirical_best_acc - r.majority_proportion
                adj_pass = delta < 0.02 and r.linear_r2 < 0.05
                if adj_pass:
                    pass_count += 1

                # Get the primary task accuracy for this purpose
                primary_task = purpose.allowed_tasks[0]
                task_acc = task_accs.get(primary_task, 0.0)

                row = {
                    "purpose": r.purpose_name,
                    "attribute": r.attr_name,
                    "task_accuracy": round(task_acc, 4),
                    "empirical_best_acc": round(r.empirical_best_acc, 4),
                    "majority_baseline": round(r.majority_proportion, 4),
                    "delta": round(delta, 4),
                    "r_squared": round(r.linear_r2, 6),
                    "adj_pass": adj_pass,
                    "train_time_s": round(train_time, 1),
                    "stopped_epoch": state.epoch + 1,
                }
                all_rows.append(row)

                status = "PASS" if adj_pass else "FAIL"
                print(f"    {r.attr_name:15s}: delta={delta:+.1%}, R2={r.linear_r2:.4f}, {status}")

            print(f"  Purpose pass: {pass_count}/{purpose_pairs}")
            purpose_summaries.append({
                "purpose": purpose.name,
                "pass_count": pass_count,
                "total_pairs": purpose_pairs,
                "primary_task_acc": task_accs.get(purpose.allowed_tasks[0], 0.0),
                "train_time": train_time,
            })

            # Save intermediate results
            save_csv(all_rows, out_path)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            purpose_summaries.append({
                "purpose": purpose.name,
                "pass_count": 0,
                "total_pairs": len(purpose.disallowed_attrs),
                "primary_task_acc": 0.0,
                "train_time": 0.0,
                "error": str(e),
            })
            continue

    # ── Final save ──────────────────────────────────────────────────────
    save_csv(all_rows, out_path)

    # ── Load PCRL v2 results for comparison ─────────────────────────────
    pcrl_path = project_root / "results" / "celeba" / "baseline_v2.csv"
    pcrl_rows = []
    if pcrl_path.exists():
        import csv as csv_mod
        with open(pcrl_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                pcrl_rows.append(row)

    # ── Comparison CSV ──────────────────────────────────────────────────
    comparison_rows = []
    for purpose in all_purposes:
        # LAFTR results for this purpose
        laftr_purpose_rows = [r for r in all_rows if r["purpose"] == purpose.name]
        laftr_pass = sum(1 for r in laftr_purpose_rows if r["adj_pass"])
        laftr_total = len(laftr_purpose_rows)
        laftr_task_acc = laftr_purpose_rows[0]["task_accuracy"] if laftr_purpose_rows else 0.0

        # PCRL results for this purpose
        pcrl_purpose_rows = [r for r in pcrl_rows if r["purpose"] == purpose.name]
        pcrl_pass = sum(1 for r in pcrl_purpose_rows if r.get("adj_pass", "False") == "True")
        pcrl_total = len(pcrl_purpose_rows)

        comparison_rows.append({
            "purpose": purpose.name,
            "laftr_task_acc": laftr_task_acc,
            "laftr_pass": laftr_pass,
            "laftr_total": laftr_total,
            "pcrl_pass": pcrl_pass,
            "pcrl_total": pcrl_total,
        })

    comp_path = project_root / "results" / "celeba" / "pcrl_vs_laftr_ensemble.csv"
    save_csv(comparison_rows, comp_path)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_laftr_pass = sum(r.get("pass_count", 0) for r in purpose_summaries)
    total_laftr_pairs = sum(r.get("total_pairs", 0) for r in purpose_summaries)
    total_pcrl_pass = sum(1 for r in pcrl_rows if r.get("adj_pass", "False") == "True")

    print(f"\n{'='*70}")
    print(f"LAFTR ENSEMBLE vs PCRL — CelebA COMPARISON")
    print(f"{'='*70}")
    print(f"\nPer-purpose breakdown:")
    print(f"{'Purpose':<30s} {'LAFTR Pass':>12s} {'PCRL Pass':>11s} {'LAFTR TaskAcc':>14s}")
    print("-" * 70)

    for ps in purpose_summaries:
        pname = ps["purpose"]
        pcrl_p = [r for r in pcrl_rows if r["purpose"] == pname]
        pcrl_pass = sum(1 for r in pcrl_p if r.get("adj_pass", "False") == "True")
        pcrl_total = len(pcrl_p)
        laftr_acc_str = f"{ps['primary_task_acc']:.1%}" if ps['primary_task_acc'] > 0 else "ERROR"
        print(f"{pname:<30s} {ps['pass_count']:>3d}/{ps['total_pairs']:<8d} "
              f"{pcrl_pass:>3d}/{pcrl_total:<7d} {laftr_acc_str:>14s}")

    print("-" * 70)
    print(f"{'TOTAL':<30s} {total_laftr_pass:>3d}/{total_laftr_pairs:<8d} "
          f"{total_pcrl_pass:>3d}/{len(pcrl_rows):<7d}")
    print(f"\nLAFTR ensemble: {total_laftr_pass}/{total_laftr_pairs} pairs pass")
    print(f"PCRL (single model): {total_pcrl_pass}/{len(pcrl_rows)} pairs pass")
    print(f"\nTotal time: {elapsed/60:.0f} minutes ({elapsed/3600:.1f} hours)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

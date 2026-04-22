#!/usr/bin/env python3
"""Final 3-seed test-set evaluation for sweep winners.

Winners (from validation-only sweep):
  PCRL Adult:  lambda_adv=50, K=20
  LAFTR Adult: lambda_adv=25, K=10
  PCRL HAR:    lambda_adv=100, K=20

Test set touched exactly once per method — this is the final evaluation.
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
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)

SEEDS = [0, 1, 2]


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


def run_adult_final(method, lam, K, device):
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    print(f"\n{'='*60}")
    print(f"FINAL: {method} Adult, lambda={lam}, K={K}")
    print(f"{'='*60}")

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False, norm_stats=train_ds.norm_stats)
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    all_rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if method == "PCRL":
            encoder = PurposeConditionedEncoder(
                input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64,
                num_purposes=len(purposes), purpose_emb_dim=32, conditioning="film", dropout=0.3,
            )
        else:
            encoder = StandardEncoder(
                input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
            )

        task_heads, auditors = {}, {}
        for p in purposes:
            task_name = p.allowed_tasks[0]
            output_dim = p.allowed_task_dims.get(task_name, 2)
            task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=64, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3,
            )

        config = TrainerConfig(
            batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
            lambda_adv=float(lam), lambda_verify=float(lam),
            auditor_steps=int(K), epochs=300, weight_decay=1e-4,
            early_stopping_patience=30, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / f"final_{method.lower()}_adult_s{seed}"),
        )

        trainer = PCRLTrainer(
            encoder=encoder, task_heads=task_heads, auditors=auditors,
            config=config, purpose_registry=registry, device=device,
        )

        t0 = time.time()
        state = trainer.train(train_loader, val_loader=val_loader)
        train_time = time.time() - t0

        test_eval = trainer.evaluate(test_loader)
        test_reports = generate_report(
            encoder=encoder, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )

        pass_count = sum(1 for r in test_reports
                         if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)

        for r in test_reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            all_rows.append({
                "method": method, "lambda_adv": lam, "K": K, "seed": seed,
                "purpose": r.purpose_name, "attribute": r.attr_name,
                "best_emp_acc": round(r.empirical_best_acc, 6),
                "majority_baseline": round(r.majority_proportion, 6),
                "delta": round(delta, 6), "linear_r2": round(r.linear_r2, 6),
                "adj_pass": ok, "pass_count": pass_count, "total_pairs": len(test_reports),
                "train_time_s": round(train_time, 1),
                "income_acc": round(test_eval.task_accuracy.get("income", 0), 4),
                "stopped_epoch": state.epoch + 1,
            })

        # Per-attr deltas for quick display
        sex_d = max((r.empirical_best_acc - r.majority_proportion for r in test_reports if r.attr_name == "sex"), default=0)
        mar_d = max((r.empirical_best_acc - r.majority_proportion for r in test_reports if r.attr_name == "marital_status"), default=0)
        print(f"  seed={seed}: income={test_eval.task_accuracy.get('income',0):.1%}, "
              f"pass={pass_count}/8, sex_d={sex_d:+.1%}, mar_d={mar_d:+.1%}, "
              f"epoch={state.epoch+1}, {train_time:.0f}s")

    return all_rows


def run_har_final(lam, K, device):
    from pcrl.data.har import HARDataset, get_har_purposes

    print(f"\n{'='*60}")
    print(f"FINAL: PCRL HAR, lambda={lam}, K={K}")
    print(f"{'='*60}")

    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HARDataset(purposes=purposes, split="train")
    val_ds = HARDataset(purposes=purposes, split="val")
    test_ds = HARDataset(purposes=purposes, split="test")
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    all_rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        encoder = PurposeConditionedEncoder(
            input_dim=input_dim, hidden_dims=[128, 128], repr_dim=16,
            num_purposes=len(purposes), purpose_emb_dim=32, conditioning="film", dropout=0.3,
        )
        task_heads, auditors = {}, {}
        for p in purposes:
            task_name = p.allowed_tasks[0]
            output_dim = p.allowed_task_dims.get(task_name, 2)
            task_heads[p.name] = TaskHead(repr_dim=16, output_dim=output_dim)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=16, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3,
            )

        config = TrainerConfig(
            batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
            lambda_adv=float(lam), lambda_verify=float(lam),
            auditor_steps=int(K), epochs=200, weight_decay=1e-4,
            early_stopping_patience=25, confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / f"final_pcrl_har_s{seed}"),
        )

        trainer = PCRLTrainer(
            encoder=encoder, task_heads=task_heads, auditors=auditors,
            config=config, purpose_registry=registry, device=device,
        )

        t0 = time.time()
        state = trainer.train(train_loader, val_loader=val_loader)
        train_time = time.time() - t0

        test_eval = trainer.evaluate(test_loader)
        test_reports = generate_report(
            encoder=encoder, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )

        pass_count = sum(1 for r in test_reports
                         if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)

        for r in test_reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            all_rows.append({
                "method": "PCRL", "lambda_adv": lam, "K": K, "seed": seed,
                "purpose": r.purpose_name, "attribute": r.attr_name,
                "best_emp_acc": round(r.empirical_best_acc, 6),
                "majority_baseline": round(r.majority_proportion, 6),
                "delta": round(delta, 6), "linear_r2": round(r.linear_r2, 6),
                "adj_pass": ok, "pass_count": pass_count, "total_pairs": len(test_reports),
                "train_time_s": round(train_time, 1),
                "activity_acc": round(test_eval.task_accuracy.get("activity", 0), 4),
                "stopped_epoch": state.epoch + 1,
            })

        subj_d = max((r.empirical_best_acc - r.majority_proportion for r in test_reports if r.attr_name == "subject_id"), default=0)
        print(f"  seed={seed}: activity={test_eval.task_accuracy.get('activity',0):.1%}, "
              f"pass={pass_count}/3, subj_d={subj_d:+.1%}, "
              f"epoch={state.epoch+1}, {train_time:.0f}s")

    return all_rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. PCRL Adult: lambda=50, K=20
    pcrl_rows = run_adult_final("PCRL", 50, 20, device)
    save_csv(pcrl_rows, project_root / "results" / "adult" / "pcrl_selected_seeds.csv")

    # 2. LAFTR Adult: lambda=25, K=10
    laftr_rows = run_adult_final("LAFTR", 25, 10, device)
    save_csv(laftr_rows, project_root / "results" / "adult" / "laftr_selected_seeds.csv")

    # 3. PCRL HAR: lambda=100, K=20
    har_rows = run_har_final(100, 20, device)
    save_csv(har_rows, project_root / "results" / "har_real" / "pcrl_selected_seeds.csv")

    print("\nAll final evaluations complete!")


if __name__ == "__main__":
    main()

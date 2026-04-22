#!/usr/bin/env python3
"""Overnight hyperparameter sweep: PCRL Adult, LAFTR Adult, PCRL HAR.

Validation-only selection. Test set touched exactly once for final 3-seed
evaluation of each winner.

Saves intermediate CSVs after each config for crash recovery.
"""

from __future__ import annotations

import csv
import logging
import signal
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
from pcrl.models.baselines import INLPProjector, LEACEEraser
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)

WALL_START = time.time()
WALL_LIMIT = 10 * 3600  # 10 hours
RUN_TIMEOUT = 45 * 60   # 45 minutes per run


def wall_remaining():
    return WALL_LIMIT - (time.time() - WALL_START)


def save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


def compute_val_compliance(reports) -> dict:
    """Compute pass count and mean delta/R2 from compliance reports."""
    pass_count = 0
    deltas, r2s = [], []
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        if ok:
            pass_count += 1
        deltas.append(delta)
        r2s.append(r.linear_r2)
    return {
        "val_pass_count": pass_count,
        "val_total_pairs": len(reports),
        "val_mean_delta": round(float(np.mean(deltas)), 6),
        "val_mean_r2": round(float(np.mean(r2s)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP: PCRL on Adult
# ═══════════════════════════════════════════════════════════════════════════

def sweep_pcrl_adult(device: str) -> list[dict]:
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    print("\n" + "#" * 60)
    print("# SWEEP 1: PCRL on Adult")
    print("#" * 60)

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    lambdas = [25, 50, 100, 200]
    Ks = [10, 20]
    results = []
    out_path = project_root / "results" / "adult" / "pcrl_sweep.csv"

    for lam in lambdas:
        for K in Ks:
            if wall_remaining() < RUN_TIMEOUT + 300:
                print(f"  WALL LIMIT approaching, stopping sweep")
                return results

            tag = f"lam={lam}_K={K}"
            ckpt_dir = str(project_root / "checkpoints" / f"sweep_pcrl_adult_{tag}")
            print(f"\n  Config: lambda_adv={lam}, K={K}")

            torch.manual_seed(0)
            np.random.seed(0)

            encoder = PurposeConditionedEncoder(
                input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64,
                num_purposes=len(purposes), purpose_emb_dim=32, conditioning="film", dropout=0.3,
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
                auditor_steps=K, epochs=300, weight_decay=1e-4,
                early_stopping_patience=30, confusion_type="entropy",
                checkpoint_dir=ckpt_dir,
            )

            trainer = PCRLTrainer(
                encoder=encoder, task_heads=task_heads, auditors=auditors,
                config=config, purpose_registry=registry, device=device,
            )

            t0 = time.time()
            try:
                state = trainer.train(train_loader, val_loader=val_loader)
                train_time = time.time() - t0
                if train_time > RUN_TIMEOUT:
                    print(f"  WARNING: run took {train_time:.0f}s (>{RUN_TIMEOUT}s)")
            except Exception as e:
                print(f"  CRASHED: {e}")
                results.append({
                    "lambda_adv": lam, "K": K, "val_pass_count": -1,
                    "val_mean_delta": -1, "val_mean_r2": -1,
                    "checkpoint_path": ckpt_dir, "train_time_s": time.time() - t0,
                    "stopped_epoch": -1, "error": str(e),
                })
                save_csv(results, out_path)
                continue

            # Evaluate on VALIDATION set
            val_eval = trainer.evaluate(val_loader)
            val_reports = generate_report(
                encoder=encoder, train_loader=train_loader, test_loader=val_loader,
                purpose_registry=registry, device=device,
            )
            metrics = compute_val_compliance(val_reports)

            row = {
                "lambda_adv": lam, "K": K,
                **metrics,
                "checkpoint_path": ckpt_dir,
                "train_time_s": round(train_time, 1),
                "stopped_epoch": state.epoch + 1,
                "income_acc": round(val_eval.task_accuracy.get("income", 0), 4),
                "error": "",
            }
            results.append(row)
            save_csv(results, out_path)

            print(f"    epoch={state.epoch+1}, {train_time:.0f}s, "
                  f"pass={metrics['val_pass_count']}/{metrics['val_total_pairs']}, "
                  f"mean_delta={metrics['val_mean_delta']:.4f}, "
                  f"income_acc={val_eval.task_accuracy.get('income', 0):.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP: LAFTR on Adult
# ═══════════════════════════════════════════════════════════════════════════

def sweep_laftr_adult(device: str) -> list[dict]:
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    print("\n" + "#" * 60)
    print("# SWEEP 2: LAFTR on Adult")
    print("#" * 60)

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    lambdas = [25, 50, 100, 200]
    Ks = [10, 20]
    results = []
    out_path = project_root / "results" / "adult" / "laftr_sweep.csv"

    for lam in lambdas:
        for K in Ks:
            if wall_remaining() < RUN_TIMEOUT + 300:
                print(f"  WALL LIMIT approaching, stopping sweep")
                return results

            tag = f"lam={lam}_K={K}"
            ckpt_dir = str(project_root / "checkpoints" / f"sweep_laftr_adult_{tag}")
            print(f"\n  Config: lambda_adv={lam}, K={K}")

            torch.manual_seed(0)
            np.random.seed(0)

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
                auditor_steps=K, epochs=300, weight_decay=1e-4,
                early_stopping_patience=30, confusion_type="entropy",
                checkpoint_dir=ckpt_dir,
            )

            trainer = PCRLTrainer(
                encoder=encoder, task_heads=task_heads, auditors=auditors,
                config=config, purpose_registry=registry, device=device,
            )

            t0 = time.time()
            try:
                state = trainer.train(train_loader, val_loader=val_loader)
                train_time = time.time() - t0
            except Exception as e:
                print(f"  CRASHED: {e}")
                results.append({
                    "lambda_adv": lam, "K": K, "val_pass_count": -1,
                    "val_mean_delta": -1, "val_mean_r2": -1,
                    "checkpoint_path": ckpt_dir, "train_time_s": time.time() - t0,
                    "stopped_epoch": -1, "error": str(e),
                })
                save_csv(results, out_path)
                continue

            val_eval = trainer.evaluate(val_loader)
            val_reports = generate_report(
                encoder=encoder, train_loader=train_loader, test_loader=val_loader,
                purpose_registry=registry, device=device,
            )
            metrics = compute_val_compliance(val_reports)

            row = {
                "lambda_adv": lam, "K": K,
                **metrics,
                "checkpoint_path": ckpt_dir,
                "train_time_s": round(train_time, 1),
                "stopped_epoch": state.epoch + 1,
                "income_acc": round(val_eval.task_accuracy.get("income", 0), 4),
                "error": "",
            }
            results.append(row)
            save_csv(results, out_path)

            print(f"    epoch={state.epoch+1}, {train_time:.0f}s, "
                  f"pass={metrics['val_pass_count']}/{metrics['val_total_pairs']}, "
                  f"mean_delta={metrics['val_mean_delta']:.4f}, "
                  f"income_acc={val_eval.task_accuracy.get('income', 0):.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SWEEP: PCRL on HAR
# ═══════════════════════════════════════════════════════════════════════════

def sweep_pcrl_har(device: str) -> list[dict]:
    from pcrl.data.har import HARDataset, get_har_purposes

    print("\n" + "#" * 60)
    print("# SWEEP 3: PCRL on HAR")
    print("#" * 60)

    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HARDataset(purposes=purposes, split="train")
    val_ds = HARDataset(purposes=purposes, split="val")
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    lambdas = [25, 50, 100]
    Ks = [10, 20]
    results = []
    out_path = project_root / "results" / "har_real" / "pcrl_sweep.csv"

    for lam in lambdas:
        for K in Ks:
            if wall_remaining() < RUN_TIMEOUT + 300:
                print(f"  WALL LIMIT approaching, stopping sweep")
                return results

            tag = f"lam={lam}_K={K}"
            ckpt_dir = str(project_root / "checkpoints" / f"sweep_pcrl_har_{tag}")
            print(f"\n  Config: lambda_adv={lam}, K={K}")

            torch.manual_seed(0)
            np.random.seed(0)

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
                auditor_steps=K, epochs=200, weight_decay=1e-4,
                early_stopping_patience=25, confusion_type="entropy",
                checkpoint_dir=ckpt_dir,
            )

            trainer = PCRLTrainer(
                encoder=encoder, task_heads=task_heads, auditors=auditors,
                config=config, purpose_registry=registry, device=device,
            )

            t0 = time.time()
            try:
                state = trainer.train(train_loader, val_loader=val_loader)
                train_time = time.time() - t0
            except Exception as e:
                print(f"  CRASHED: {e}")
                results.append({
                    "lambda_adv": lam, "K": K, "val_pass_count": -1,
                    "val_mean_delta": -1, "val_mean_r2": -1,
                    "checkpoint_path": ckpt_dir, "train_time_s": time.time() - t0,
                    "stopped_epoch": -1, "error": str(e),
                })
                save_csv(results, out_path)
                continue

            val_eval = trainer.evaluate(val_loader)
            val_reports = generate_report(
                encoder=encoder, train_loader=train_loader, test_loader=val_loader,
                purpose_registry=registry, device=device,
            )
            metrics = compute_val_compliance(val_reports)

            row = {
                "lambda_adv": lam, "K": K,
                **metrics,
                "checkpoint_path": ckpt_dir,
                "train_time_s": round(train_time, 1),
                "stopped_epoch": state.epoch + 1,
                "activity_acc": round(val_eval.task_accuracy.get("activity", 0), 4),
                "error": "",
            }
            results.append(row)
            save_csv(results, out_path)

            print(f"    epoch={state.epoch+1}, {train_time:.0f}s, "
                  f"pass={metrics['val_pass_count']}/{metrics['val_total_pairs']}, "
                  f"mean_delta={metrics['val_mean_delta']:.4f}, "
                  f"activity_acc={val_eval.task_accuracy.get('activity', 0):.1%}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION: 3-seed test-set run for winners
# ═══════════════════════════════════════════════════════════════════════════

def pick_winner(results: list[dict]) -> dict | None:
    """Pick config with highest val_pass_count, break ties by lower delta, then R2."""
    valid = [r for r in results if r.get("val_pass_count", -1) >= 0]
    if not valid:
        return None
    valid.sort(key=lambda r: (-r["val_pass_count"], r["val_mean_delta"], r["val_mean_r2"]))
    return valid[0]


def final_eval_adult(method: str, winner: dict, device: str) -> list[dict]:
    """Run 3-seed test-set evaluation for the winning Adult config."""
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    lam = winner["lambda_adv"]
    K = winner["K"]
    print(f"\n  FINAL EVAL: {method} Adult, lambda={lam}, K={K}, seeds={{0,1,2}}")

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
    for seed in [0, 1, 2]:
        if wall_remaining() < RUN_TIMEOUT + 300:
            print(f"  WALL LIMIT, stopping at seed {seed}")
            break

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

        # TEST SET — touched exactly once
        test_eval = trainer.evaluate(test_loader)
        test_reports = generate_report(
            encoder=encoder, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )

        pass_count = 0
        for r in test_reports:
            delta = r.empirical_best_acc - r.majority_proportion
            if delta < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        for r in test_reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            row = {
                "method": method, "lambda_adv": lam, "K": K, "seed": seed,
                "purpose": r.purpose_name, "attribute": r.attr_name,
                "best_emp_acc": round(r.empirical_best_acc, 6),
                "majority_baseline": round(r.majority_proportion, 6),
                "delta": round(delta, 6), "linear_r2": round(r.linear_r2, 6),
                "adj_pass": ok, "pass_count": pass_count, "total_pairs": len(test_reports),
                "train_time_s": round(train_time, 1),
                "income_acc": round(test_eval.task_accuracy.get("income", 0), 4),
                "stopped_epoch": state.epoch + 1,
            }
            all_rows.append(row)

        print(f"    seed={seed}: income_acc={test_eval.task_accuracy.get('income',0):.1%}, "
              f"pass={pass_count}/{len(test_reports)}, epoch={state.epoch+1}, {train_time:.0f}s")

    return all_rows


def final_eval_har(winner: dict, device: str) -> list[dict]:
    """Run 3-seed test-set evaluation for winning HAR PCRL config."""
    from pcrl.data.har import HARDataset, get_har_purposes

    lam = winner["lambda_adv"]
    K = winner["K"]
    print(f"\n  FINAL EVAL: PCRL HAR, lambda={lam}, K={K}, seeds={{0,1,2}}")

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
    for seed in [0, 1, 2]:
        if wall_remaining() < RUN_TIMEOUT + 300:
            print(f"  WALL LIMIT, stopping at seed {seed}")
            break

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

        pass_count = 0
        for r in test_reports:
            delta = r.empirical_best_acc - r.majority_proportion
            if delta < 0.02 and r.linear_r2 < 0.05:
                pass_count += 1

        for r in test_reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            row = {
                "method": "PCRL", "lambda_adv": lam, "K": K, "seed": seed,
                "purpose": r.purpose_name, "attribute": r.attr_name,
                "best_emp_acc": round(r.empirical_best_acc, 6),
                "majority_baseline": round(r.majority_proportion, 6),
                "delta": round(delta, 6), "linear_r2": round(r.linear_r2, 6),
                "adj_pass": ok, "pass_count": pass_count, "total_pairs": len(test_reports),
                "train_time_s": round(train_time, 1),
                "activity_acc": round(test_eval.task_accuracy.get("activity", 0), 4),
                "stopped_epoch": state.epoch + 1,
            }
            all_rows.append(row)

        print(f"    seed={seed}: activity_acc={test_eval.task_accuracy.get('activity',0):.1%}, "
              f"pass={pass_count}/{len(test_reports)}, epoch={state.epoch+1}, {train_time:.0f}s")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Wall limit: {WALL_LIMIT/3600:.0f}h")
    print(f"Run timeout: {RUN_TIMEOUT/60:.0f}min")

    # ── Sweep 1: PCRL Adult ──────────────────────────────────────────────
    pcrl_adult_results = sweep_pcrl_adult(device)
    pcrl_adult_winner = pick_winner(pcrl_adult_results)
    if pcrl_adult_winner:
        print(f"\n  PCRL Adult winner: lambda={pcrl_adult_winner['lambda_adv']}, "
              f"K={pcrl_adult_winner['K']}, pass={pcrl_adult_winner['val_pass_count']}")

    # ── Sweep 2: LAFTR Adult ─────────────────────────────────────────────
    laftr_adult_results = sweep_laftr_adult(device)
    laftr_adult_winner = pick_winner(laftr_adult_results)
    if laftr_adult_winner:
        print(f"\n  LAFTR Adult winner: lambda={laftr_adult_winner['lambda_adv']}, "
              f"K={laftr_adult_winner['K']}, pass={laftr_adult_winner['val_pass_count']}")

    # ── Sweep 3: PCRL HAR ────────────────────────────────────────────────
    pcrl_har_results = sweep_pcrl_har(device)
    pcrl_har_winner = pick_winner(pcrl_har_results)
    if pcrl_har_winner:
        print(f"\n  PCRL HAR winner: lambda={pcrl_har_winner['lambda_adv']}, "
              f"K={pcrl_har_winner['K']}, pass={pcrl_har_winner['val_pass_count']}")

    # ── Final 3-seed test evaluations ────────────────────────────────────
    print("\n" + "#" * 60)
    print("# FINAL 3-SEED TEST-SET EVALUATIONS")
    print("#" * 60)

    if pcrl_adult_winner and wall_remaining() > RUN_TIMEOUT * 3 + 600:
        pcrl_adult_final = final_eval_adult("PCRL", pcrl_adult_winner, device)
        save_csv(pcrl_adult_final, project_root / "results" / "adult" / "pcrl_selected_seeds.csv")
    else:
        print("  Skipping PCRL Adult final (time)")

    if laftr_adult_winner and wall_remaining() > RUN_TIMEOUT * 3 + 600:
        laftr_adult_final = final_eval_adult("LAFTR", laftr_adult_winner, device)
        save_csv(laftr_adult_final, project_root / "results" / "adult" / "laftr_selected_seeds.csv")
    else:
        print("  Skipping LAFTR Adult final (time)")

    if pcrl_har_winner and wall_remaining() > RUN_TIMEOUT * 3 + 600:
        pcrl_har_final = final_eval_har(pcrl_har_winner, device)
        save_csv(pcrl_har_final, project_root / "results" / "har_real" / "pcrl_selected_seeds.csv")
    else:
        print("  Skipping PCRL HAR final (time)")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - WALL_START
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE — {elapsed/3600:.1f}h elapsed")
    print(f"{'='*60}")

    print("\nSweep winners:")
    if pcrl_adult_winner:
        print(f"  PCRL Adult:  lambda={pcrl_adult_winner['lambda_adv']}, K={pcrl_adult_winner['K']}, "
              f"val_pass={pcrl_adult_winner['val_pass_count']}")
    if laftr_adult_winner:
        print(f"  LAFTR Adult: lambda={laftr_adult_winner['lambda_adv']}, K={laftr_adult_winner['K']}, "
              f"val_pass={laftr_adult_winner['val_pass_count']}")
    if pcrl_har_winner:
        print(f"  PCRL HAR:    lambda={pcrl_har_winner['lambda_adv']}, K={pcrl_har_winner['K']}, "
              f"val_pass={pcrl_har_winner['val_pass_count']}")

    print("\nAll done!")


if __name__ == "__main__":
    main()

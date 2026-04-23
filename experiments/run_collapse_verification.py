#!/usr/bin/env python3
"""Collapse verification: 3 configs × 3 seeds + Attention seed 1 diagnostic.

Trains models and computes confusion matrix + representation statistics
to distinguish genuine privacy from representation collapse.

Configs:
  A: LAFTR default (λ=50, K=10) — StandardEncoder
  B: LAFTR swept  (λ=25, K=10) — StandardEncoder
  C: PCRL matched (λ=50, K=10) — PurposeConditionedEncoder (FiLM)
  D: Attention seed 1 diagnostic (λ=50, K=20) — PurposeConditionedEncoder (attention)
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter
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

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

import logging
logging.basicConfig(level=logging.WARNING)

WALL_CAP_SECONDS = 90 * 60  # 90 minutes


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


def compute_collapse_metrics(encoder, test_loader, purpose_idx, device):
    """Compute income confusion matrix and representation statistics."""
    encoder.eval()
    all_reprs = []
    all_preds = []
    all_labels = []

    # We need task heads to get predictions — but we need the trainer for that.
    # Instead, just get representations and labels here; predictions come from trainer.
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reprs.append(h.cpu().numpy())

    reprs = np.concatenate(all_reprs)

    # Representation statistics
    per_dim_std = reprs.std(axis=0)  # std per dimension across samples
    repr_std_mean = float(per_dim_std.mean())
    repr_std_max = float(per_dim_std.max())
    repr_l2_norm_mean = float(np.linalg.norm(reprs, axis=1).mean())

    return repr_std_mean, repr_std_max, repr_l2_norm_mean


def get_income_predictions(trainer, encoder, test_loader, purpose_idx, device):
    """Get income predictions and labels."""
    encoder.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            task_head = trainer.task_heads["income_prediction"]
            logits = task_head(h)
            preds = logits.argmax(dim=1)
            labels = batch["task_labels"]["income"]
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0

    class1_preds = tp + fp
    class1_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    class1_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return acc, class1_preds, class1_recall, class1_precision, total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t_start = time.time()

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

    # Majority baseline
    all_income = []
    for batch in test_loader:
        all_income.extend(batch["task_labels"]["income"].numpy().tolist())
    counts = Counter(all_income)
    total_test = len(all_income)
    majority_baseline = counts[0] / total_test  # class 0 = <=50K is majority
    print(f"Majority baseline: {majority_baseline:.4%} ({counts[0]}/{total_test})")

    # ── Configurations ──────────────────────────────────────────────────
    configs = [
        {"config": "LAFTR_default", "lambda_adv": 50.0, "K": 10, "encoder_type": "standard", "conditioning": None},
        {"config": "LAFTR_swept",   "lambda_adv": 25.0, "K": 10, "encoder_type": "standard", "conditioning": None},
        {"config": "PCRL_matched",  "lambda_adv": 50.0, "K": 10, "encoder_type": "pcrl",     "conditioning": "film"},
    ]

    all_rows = []
    out_path = project_root / "results" / "adult" / "collapse_verification.csv"

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"{cfg['config']} (λ={cfg['lambda_adv']}, K={cfg['K']})")
        print(f"{'='*60}")

        for seed in [0, 1, 2]:
            if time.time() - t_start > WALL_CAP_SECONDS:
                print(f"  WALL CAP REACHED ({WALL_CAP_SECONDS/60:.0f} min), stopping early")
                save_csv(all_rows, out_path)
                print_summary(all_rows, majority_baseline, total_test)
                return

            torch.manual_seed(seed)
            np.random.seed(seed)

            if cfg["encoder_type"] == "standard":
                encoder = StandardEncoder(
                    input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
                )
            else:
                encoder = PurposeConditionedEncoder(
                    input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64,
                    num_purposes=len(purposes), purpose_emb_dim=32,
                    conditioning=cfg["conditioning"], dropout=0.3,
                )

            task_heads, auditors = {}, {}
            for p in purposes:
                task_name = p.allowed_tasks[0]
                output_dim = p.allowed_task_dims.get(task_name, 2)
                task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)
                auditors[p.name] = MultiAttributeAuditor(
                    repr_dim=64, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3,
                )

            ckpt_dir = f"collapse_{cfg['config']}_s{seed}"
            config = TrainerConfig(
                batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
                lambda_adv=cfg["lambda_adv"], lambda_verify=cfg["lambda_adv"],
                auditor_steps=cfg["K"],
                epochs=200, weight_decay=1e-4, early_stopping_patience=20,
                confusion_type="entropy",
                checkpoint_dir=str(project_root / "checkpoints" / ckpt_dir),
            )

            trainer = PCRLTrainer(
                encoder=encoder, task_heads=task_heads, auditors=auditors,
                config=config, purpose_registry=registry, device=device,
            )

            t0 = time.time()
            state = trainer.train(train_loader, val_loader=val_loader)
            train_time = time.time() - t0

            # Collapse metrics
            repr_std_mean, repr_std_max, repr_l2_norm_mean = compute_collapse_metrics(
                encoder, test_loader, 0, device)

            acc, class1_preds, class1_recall, class1_precision, total = get_income_predictions(
                trainer, encoder, test_loader, 0, device)

            row = {
                "config": cfg["config"],
                "lambda_adv": cfg["lambda_adv"],
                "K": cfg["K"],
                "seed": seed,
                "income_accuracy": round(acc, 6),
                "majority_baseline": round(majority_baseline, 6),
                "pct_majority_predictions": round(1.0 - class1_preds / total, 6) if total > 0 else 0,
                "class1_recall": round(class1_recall, 6),
                "class1_precision": round(class1_precision, 6),
                "class1_num_predictions": class1_preds,
                "repr_std_mean": round(repr_std_mean, 8),
                "repr_std_max": round(repr_std_max, 8),
                "repr_l2_norm_mean": round(repr_l2_norm_mean, 6),
            }
            all_rows.append(row)

            print(f"  seed={seed}: acc={acc:.4%}, class1_preds={class1_preds}/{total}, "
                  f"repr_std={repr_std_mean:.6f}, epoch={state.epoch+1}, {train_time:.0f}s")

        # Save after each config
        save_csv(all_rows, out_path)

    # ── Task 2: Attention seed 1 diagnostic ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"Attention seed 1 diagnostic (λ=50, K=20)")
    print(f"{'='*60}")

    if time.time() - t_start > WALL_CAP_SECONDS:
        print(f"  WALL CAP REACHED, skipping Attention diagnostic")
        save_csv(all_rows, out_path)
        print_summary(all_rows, majority_baseline, total_test)
        return

    torch.manual_seed(1)
    np.random.seed(1)

    encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="attention", dropout=0.3,
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
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=20,
        epochs=200, weight_decay=1e-4, early_stopping_patience=20,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "collapse_attention_s1"),
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    repr_std_mean, repr_std_max, repr_l2_norm_mean = compute_collapse_metrics(
        encoder, test_loader, 0, device)

    acc, class1_preds, class1_recall, class1_precision, total = get_income_predictions(
        trainer, encoder, test_loader, 0, device)

    row = {
        "config": "Attention_seed1_conditioning",
        "lambda_adv": 50.0,
        "K": 20,
        "seed": 1,
        "income_accuracy": round(acc, 6),
        "majority_baseline": round(majority_baseline, 6),
        "pct_majority_predictions": round(1.0 - class1_preds / total, 6) if total > 0 else 0,
        "class1_recall": round(class1_recall, 6),
        "class1_precision": round(class1_precision, 6),
        "class1_num_predictions": class1_preds,
        "repr_std_mean": round(repr_std_mean, 8),
        "repr_std_max": round(repr_std_max, 8),
        "repr_l2_norm_mean": round(repr_l2_norm_mean, 6),
    }
    all_rows.append(row)

    print(f"  seed=1: acc={acc:.4%}, class1_preds={class1_preds}/{total}, "
          f"repr_std={repr_std_mean:.6f}, epoch={state.epoch+1}, {train_time:.0f}s")

    save_csv(all_rows, out_path)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\nAll runs complete! Total: {elapsed/60:.0f} minutes")
    print_summary(all_rows, majority_baseline, total_test)


def print_summary(rows, majority_baseline, total_test):
    print(f"\n{'='*60}")
    print(f"=== Collapse Verification Summary ===")
    print(f"{'='*60}")

    # Group by config
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["config"]].append(r)

    for config_name, config_rows in grouped.items():
        accs = [r["income_accuracy"] for r in config_rows]
        c1_preds = [r["class1_num_predictions"] for r in config_rows]
        repr_stds = [r["repr_std_mean"] for r in config_rows]

        if len(config_rows) > 1:
            acc_str = f"{np.mean(accs):.2%}±{np.std(accs):.2%}"
            c1_str = f"{np.mean(c1_preds):.0f}±{np.std(c1_preds):.0f}/{total_test}"
            std_str = f"{np.mean(repr_stds):.6f}"
        else:
            acc_str = f"{accs[0]:.2%}"
            c1_str = f"{c1_preds[0]}/{total_test}"
            std_str = f"{repr_stds[0]:.6f}"

        lam = config_rows[0]["lambda_adv"]
        k = config_rows[0]["K"]
        print(f"{config_name:30s} (λ={lam}, K={k}): acc={acc_str}, class1_preds={c1_str}, repr_std={std_str}")

    print(f"Majority baseline: {majority_baseline:.2%}")

    # Verdict
    print(f"\nVerdict:")
    for config_name, config_rows in grouped.items():
        mean_std = np.mean([r["repr_std_mean"] for r in config_rows])
        mean_c1 = np.mean([r["class1_num_predictions"] for r in config_rows])
        mean_recall = np.mean([r["class1_recall"] for r in config_rows])

        if mean_std < 0.001:
            status = "COLLAPSED (repr_std < 0.001)"
        elif mean_c1 < 10:
            status = "COLLAPSED (near-zero class 1 predictions)"
        elif mean_recall < 0.01:
            status = "COLLAPSED (class 1 recall ≈ 0)"
        else:
            status = f"GENUINE (repr_std={mean_std:.4f}, recall={mean_recall:.4f})"

        print(f"  {config_name}: {status}")


if __name__ == "__main__":
    main()

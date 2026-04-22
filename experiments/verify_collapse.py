#!/usr/bin/env python3
"""Verify LAFTR collapse claim: confusion matrices for income prediction."""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter

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

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig


def get_confusion_matrix(trainer, encoder, test_loader, purpose_idx, device):
    """Get predictions and compute confusion matrix for a binary task."""
    encoder.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)

            # Get task head for income_prediction (purpose 0)
            task_head = trainer.task_heads["income_prediction"]
            logits = task_head(h)
            preds = logits.argmax(dim=1)
            labels = batch["task_labels"]["income"]

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Confusion matrix: label=0 is income<=50K (majority), label=1 is income>50K
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    return preds, labels, tp, tn, fp, fn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False, norm_stats=train_ds.norm_stats)
    input_dim = train_ds.info.num_features

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    # ── Exact majority baseline ─────────────────────────────────────────
    all_income = []
    for batch in test_loader:
        all_income.extend(batch["task_labels"]["income"].numpy().tolist())

    counts = Counter(all_income)
    total = len(all_income)
    print(f"TEST SET INCOME DISTRIBUTION:")
    print(f"  Class 0 (<=50K): {counts[0]} ({counts[0]/total:.6f})")
    print(f"  Class 1 (>50K):  {counts[1]} ({counts[1]/total:.6f})")
    print(f"  Total:           {total}")
    print(f"  Majority baseline: {counts[0]/total:.6f} ({counts[0]/total:.4%})")

    # ── LAFTR (lambda=25, K=10, seed=0) ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LAFTR (lambda=25, K=10, seed=0)")
    print(f"{'='*60}")

    torch.manual_seed(0)
    np.random.seed(0)

    laftr_encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    laftr_task_heads, laftr_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        laftr_task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)
        laftr_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=64, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3,
        )

    laftr_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=25.0, lambda_verify=25.0, auditor_steps=10,
        epochs=300, weight_decay=1e-4, early_stopping_patience=30,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "final_laftr_adult_s0"),
    )

    laftr_trainer = PCRLTrainer(
        encoder=laftr_encoder, task_heads=laftr_task_heads, auditors=laftr_auditors,
        config=laftr_config, purpose_registry=registry, device=device,
    )

    ckpt_path = project_root / "checkpoints" / "final_laftr_adult_s0" / "final.pt"
    if ckpt_path.exists():
        laftr_trainer.load_checkpoint(ckpt_path)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  WARNING: No checkpoint at {ckpt_path}, training fresh...")
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
        val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
        laftr_trainer.train(train_loader, val_loader=val_loader)

    # income_prediction is purpose index 0
    preds, labels, tp, tn, fp, fn = get_confusion_matrix(
        laftr_trainer, laftr_encoder, test_loader, 0, device)

    acc = (tp + tn) / (tp + tn + fp + fn)
    pred_counts = Counter(preds.tolist())
    print(f"\n  Confusion matrix:")
    print(f"                  Predicted 0   Predicted 1")
    print(f"    Actual 0       {tn:>6}        {fp:>6}")
    print(f"    Actual 1       {fn:>6}        {tp:>6}")
    print(f"\n  Accuracy: {acc:.4%}")
    print(f"  Predictions: class 0 = {pred_counts.get(0,0)}, class 1 = {pred_counts.get(1,0)}")
    print(f"  Pct predicting majority class: {pred_counts.get(0,0)/(tp+tn+fp+fn):.4%}")
    if (tp + fp) > 0:
        print(f"  Precision (class 1): {tp/(tp+fp):.4f}")
    if (tp + fn) > 0:
        print(f"  Recall (class 1):    {tp/(tp+fn):.4f}")

    # Also check representation statistics
    all_reprs = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            h = laftr_encoder(x, 0)
            all_reprs.append(h.cpu().numpy())
    reprs = np.concatenate(all_reprs)
    print(f"\n  Representation stats:")
    print(f"    Shape: {reprs.shape}")
    print(f"    Mean norm: {np.linalg.norm(reprs, axis=1).mean():.4f}")
    print(f"    Std across samples: {reprs.std(axis=0).mean():.6f}")
    print(f"    Max abs value: {np.abs(reprs).max():.4f}")

    # ── PCRL (lambda=50, K=20, seed=0) ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PCRL (lambda=50, K=20, seed=0)")
    print(f"{'='*60}")

    torch.manual_seed(0)
    np.random.seed(0)

    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64,
        num_purposes=len(purposes), purpose_emb_dim=32, conditioning="film", dropout=0.3,
    )
    pcrl_task_heads, pcrl_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        pcrl_task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)
        pcrl_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=64, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3,
        )

    pcrl_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=20,
        epochs=300, weight_decay=1e-4, early_stopping_patience=30,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "final_pcrl_adult_s0"),
    )

    pcrl_trainer = PCRLTrainer(
        encoder=pcrl_encoder, task_heads=pcrl_task_heads, auditors=pcrl_auditors,
        config=pcrl_config, purpose_registry=registry, device=device,
    )

    ckpt_path = project_root / "checkpoints" / "final_pcrl_adult_s0" / "final.pt"
    if ckpt_path.exists():
        pcrl_trainer.load_checkpoint(ckpt_path)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  WARNING: No checkpoint at {ckpt_path}, training fresh...")
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
        val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
        pcrl_trainer.train(train_loader, val_loader=val_loader)

    preds, labels, tp, tn, fp, fn = get_confusion_matrix(
        pcrl_trainer, pcrl_encoder, test_loader, 0, device)

    acc = (tp + tn) / (tp + tn + fp + fn)
    pred_counts = Counter(preds.tolist())
    print(f"\n  Confusion matrix:")
    print(f"                  Predicted 0   Predicted 1")
    print(f"    Actual 0       {tn:>6}        {fp:>6}")
    print(f"    Actual 1       {fn:>6}        {tp:>6}")
    print(f"\n  Accuracy: {acc:.4%}")
    print(f"  Predictions: class 0 = {pred_counts.get(0,0)}, class 1 = {pred_counts.get(1,0)}")
    print(f"  Pct predicting majority class: {pred_counts.get(0,0)/(tp+tn+fp+fn):.4%}")
    if (tp + fp) > 0:
        print(f"  Precision (class 1): {tp/(tp+fp):.4f}")
    if (tp + fn) > 0:
        print(f"  Recall (class 1):    {tp/(tp+fn):.4f}")

    all_reprs = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            h = pcrl_encoder(x, 0)
            all_reprs.append(h.cpu().numpy())
    reprs = np.concatenate(all_reprs)
    print(f"\n  Representation stats:")
    print(f"    Shape: {reprs.shape}")
    print(f"    Mean norm: {np.linalg.norm(reprs, axis=1).mean():.4f}")
    print(f"    Std across samples: {reprs.std(axis=0).mean():.6f}")
    print(f"    Max abs value: {np.abs(reprs).max():.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick HAR lambda sweep: find the best PCRL config for subject_id suppression.

Tests a few (lambda_adv, lambda_verify, epochs) configs and reports
activity accuracy + subject_id delta for each.  No baselines — just PCRL.
"""
from __future__ import annotations

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


class _Q:
    def __init__(self, it=None, *a, **kw):
        self.iterable = it

    def __iter__(self):
        return iter(self.iterable) if self.iterable else iter([])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def set_postfix(self, *a, **kw):
        pass

    def update(self, *a):
        pass

    def close(self):
        pass


_trainer_mod.tqdm = _Q

import logging

logging.basicConfig(level=logging.WARNING)

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.har import HARDataset, get_har_purposes
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

# ── Fixed architecture ───────────────────────────────────────────────────
HIDDEN_DIMS = [128, 128]
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH_SIZE = 256
AUDITOR_HIDDEN = 256
AUDITOR_LAYERS = 3

# ── Configs to sweep (real UCI HAR data) ─────────────────────────────────
# Fine-tune around best config: r=16 with low lambda
CONFIGS = [
    # (lambda_adv, lambda_verify, K, epochs, repr_dim, lr_enc, lr_aud, label)
    (2,  1,  20, 100, 16, 1e-3, 1e-3, "r=16 λ=2/1  E=100"),
    (3,  1,  20, 100, 16, 1e-3, 1e-3, "r=16 λ=3/1  E=100"),
    (3,  2,  20, 100, 16, 1e-3, 1e-3, "r=16 λ=3/2  E=100"),
    (2,  1,  20, 200, 16, 1e-3, 1e-3, "r=16 λ=2/1  E=200"),
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HARDataset(purposes=purposes, split="train")
    val_ds = HARDataset(purposes=purposes, split="val")
    test_ds = HARDataset(purposes=purposes, split="test")

    input_dim = train_ds.info.num_features
    print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, Feat={input_dim}")

    # Majority baselines
    maj: dict[str, float] = {}
    for attr, labels in test_ds.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        maj[attr] = counts.max().item() / len(labels)
    print(f"Majority baselines: {', '.join(f'{k}={v:.1%}' for k, v in maj.items())}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    # Header
    print(f"{'Config':<24} {'Act':>6} {'IsAct':>6} {'SubjΔ':>7} {'R²':>7} {'SubjPass':>9} {'Time':>6}")
    print("-" * 72)

    for lam_adv, lam_ver, k_steps, epochs, repr_dim, lr_enc, lr_aud, label in CONFIGS:
        torch.manual_seed(42)
        encoder = PurposeConditionedEncoder(
            input_dim=input_dim,
            hidden_dims=HIDDEN_DIMS,
            repr_dim=repr_dim,
            num_purposes=len(purposes),
            purpose_emb_dim=PURPOSE_EMB_DIM,
            conditioning="film",
            dropout=DROPOUT,
        )
        task_heads: dict[str, torch.nn.Module] = {}
        auditors: dict[str, torch.nn.Module] = {}
        for p in purposes:
            tn = p.allowed_tasks[0]
            od = p.allowed_task_dims.get(tn, 2)
            task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=od)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=repr_dim,
                attr_output_dims=p.disallowed_attr_dims,
                hidden_dim=AUDITOR_HIDDEN,
                num_layers=AUDITOR_LAYERS,
            )

        config = TrainerConfig(
            batch_size=BATCH_SIZE,
            lr_encoder=lr_enc,
            lr_auditor=lr_aud,
            lambda_adv=lam_adv,
            lambda_verify=lam_ver,
            auditor_steps=k_steps,
            epochs=epochs,
            weight_decay=1e-4,
            early_stopping_patience=None,   # no early stopping
            confusion_type="entropy",
            checkpoint_dir=str(project_root / "checkpoints" / "har_sweep"),
        )
        trainer = PCRLTrainer(
            encoder=encoder, task_heads=task_heads, auditors=auditors,
            config=config, purpose_registry=registry, device=device,
        )

        t0 = time.time()
        trainer.train(train_loader, val_loader=val_loader)
        elapsed = time.time() - t0

        # Evaluate
        ev = trainer.evaluate(test_loader)
        reports = generate_report(
            encoder=encoder,
            train_loader=train_loader,
            test_loader=test_loader,
            purpose_registry=registry,
            device=device,
        )

        act_acc = ev.task_accuracy.get("activity", 0.0)
        ia_acc = ev.task_accuracy.get("is_active", 0.0)

        # Get worst subject_id delta across purposes
        worst_subj_delta = -1.0
        worst_subj_r2 = 0.0
        subj_pass = True
        for r in reports:
            if r.attr_name == "subject_id":
                bl = maj.get("subject_id", r.empirical_chance_acc)
                d = r.empirical_best_acc - bl
                if d > worst_subj_delta:
                    worst_subj_delta = d
                    worst_subj_r2 = r.linear_r2
                if not (d < 0.02 and r.linear_r2 < 0.05):
                    subj_pass = False

        status = "PASS" if subj_pass else "FAIL"
        print(
            f"{label:<24} {act_acc:>5.1%} {ia_acc:>5.1%} "
            f"{worst_subj_delta:>+6.1%} {worst_subj_r2:>6.4f} "
            f"{status:>9} {elapsed:>5.0f}s"
        )

    print("-" * 72)
    print("Target: Activity ≥ 90%, Subject Δ < 5% (ideally < 2%), R² < 0.05")


if __name__ == "__main__":
    main()

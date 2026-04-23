#!/usr/bin/env python3
"""Three reconciliation re-runs with fixed generate_report().

1. Conditioning ablation (FiLM, Concat, Attention) — 3 seeds each
2. Nonlinear cross-purpose attack — LogReg, MLP, XGBoost
3. Distribution shift — Female/Male subgroups

All use the fixed single-pass extraction in generate_report().
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

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
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

import logging
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


def load_adult():
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False, norm_stats=train_ds.norm_stats)

    return purposes, registry, train_ds, val_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
# RE-RUN 1: Conditioning Ablation
# ═══════════════════════════════════════════════════════════════════════════

def run_conditioning_ablation(device):
    print("\n" + "#" * 60)
    print("# RE-RUN 1: Conditioning Ablation (FiLM, Concat, Attention)")
    print("#" * 60)

    purposes, registry, train_ds, val_ds, test_ds = load_adult()
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    all_rows = []
    out_path = project_root / "results" / "adult" / "conditioning_ablation_fixed.csv"

    for cond_type in ["film", "concat", "attention"]:
        seed_pass_counts = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            encoder = PurposeConditionedEncoder(
                input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64,
                num_purposes=len(purposes), purpose_emb_dim=32,
                conditioning=cond_type, dropout=0.3,
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
                lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
                epochs=200, weight_decay=1e-4, early_stopping_patience=20,
                confusion_type="entropy",
                checkpoint_dir=str(project_root / "checkpoints" / f"ablation_{cond_type}_s{seed}"),
            )

            trainer = PCRLTrainer(
                encoder=encoder, task_heads=task_heads, auditors=auditors,
                config=config, purpose_registry=registry, device=device,
            )

            t0 = time.time()
            state = trainer.train(train_loader, val_loader=val_loader)
            train_time = time.time() - t0

            # Evaluate on VALIDATION set
            val_eval = trainer.evaluate(val_loader)
            val_reports = generate_report(
                encoder=encoder, train_loader=train_loader, test_loader=val_loader,
                purpose_registry=registry, device=device,
            )

            pass_count = sum(1 for r in val_reports
                             if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)
            seed_pass_counts.append(pass_count)

            for r in val_reports:
                delta = r.empirical_best_acc - r.majority_proportion
                ok = delta < 0.02 and r.linear_r2 < 0.05
                all_rows.append({
                    "conditioning": cond_type, "seed": seed,
                    "purpose": r.purpose_name, "attribute": r.attr_name,
                    "best_emp_acc": round(r.empirical_best_acc, 6),
                    "majority_baseline": round(r.majority_proportion, 6),
                    "delta": round(delta, 6), "linear_r2": round(r.linear_r2, 6),
                    "adj_pass": ok, "pass_count": pass_count, "total_pairs": len(val_reports),
                    "income_acc": round(val_eval.task_accuracy.get("income", 0), 4),
                    "train_time_s": round(train_time, 1),
                    "stopped_epoch": state.epoch + 1,
                })

            print(f"  {cond_type} seed={seed}: income={val_eval.task_accuracy.get('income',0):.1%}, "
                  f"val_pass={pass_count}/8, epoch={state.epoch+1}, {train_time:.0f}s")

        save_csv(all_rows, out_path)
        mean_pass = np.mean(seed_pass_counts)
        std_pass = np.std(seed_pass_counts, ddof=0)
        print(f"  {cond_type} SUMMARY: val_pass = {mean_pass:.1f} +/- {std_pass:.1f}")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# RE-RUN 2: Nonlinear Cross-Purpose Attack
# ═══════════════════════════════════════════════════════════════════════════

class MLPAuditor(nn.Module):
    """2-layer MLP for cross-purpose attack."""
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_mlp_auditor(train_X, train_y, test_X, test_y, num_classes, device, seed=42,
                       epochs=100, lr=1e-3, batch_size=256):
    """Train a 2-layer MLP auditor and return test accuracy."""
    torch.manual_seed(seed)
    input_dim = train_X.shape[1]
    model = MLPAuditor(input_dim, num_classes, hidden_dim=256, dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_X_t = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.long, device=device)
    test_X_t = torch.tensor(test_X, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.long, device=device)

    n = len(train_X_t)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(train_X_t[idx])
            loss = nn.functional.cross_entropy(logits, train_y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(test_X_t).argmax(dim=1)
        acc = (preds == test_y_t).float().mean().item()
    return acc


def run_cross_purpose_attack(device):
    print("\n" + "#" * 60)
    print("# RE-RUN 2: Nonlinear Cross-Purpose Attack")
    print("#" * 60)

    purposes, registry, train_ds, val_ds, test_ds = load_adult()
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    # Load fixed seed 0 PCRL checkpoint
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
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10, confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "fix_pcrl_adult_0"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    ckpt = project_root / "checkpoints" / "fix_pcrl_adult_0" / "final.pt"
    trainer.load_checkpoint(ckpt)
    print(f"  Loaded checkpoint: {ckpt}")

    # Extract per-purpose representations (non-shuffled for alignment)
    # Use a non-shuffled train loader for extraction
    train_extract_loader = DataLoader(train_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    purpose_names = [p.name for p in purposes]
    encoder.eval()

    train_reps, test_reps = {}, {}
    train_sens, test_sens = {}, {}

    with torch.no_grad():
        # Train
        all_reps = {pname: [] for pname in purpose_names}
        all_sens = {}
        for batch in train_extract_loader:
            x = batch["features"].to(device)
            for idx, pname in enumerate(purpose_names):
                all_reps[pname].append(encoder(x, idx).cpu().numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
        for pname in purpose_names:
            train_reps[pname] = np.concatenate(all_reps[pname])
        train_sens = {k: np.concatenate(v) for k, v in all_sens.items()}

        # Test
        all_reps = {pname: [] for pname in purpose_names}
        all_sens = {}
        for batch in test_loader:
            x = batch["features"].to(device)
            for idx, pname in enumerate(purpose_names):
                all_reps[pname].append(encoder(x, idx).cpu().numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
        for pname in purpose_names:
            test_reps[pname] = np.concatenate(all_reps[pname])
        test_sens = {k: np.concatenate(v) for k, v in all_sens.items()}

    # Concatenated representations
    train_concat = np.concatenate([train_reps[p] for p in purpose_names], axis=1)
    test_concat = np.concatenate([test_reps[p] for p in purpose_names], axis=1)
    print(f"  Concatenated shape: {train_concat.shape}")

    # Majority baselines
    from collections import Counter
    majority = {}
    for attr in ["race", "sex", "age_group", "marital_status"]:
        all_labels = np.concatenate([train_sens[attr], test_sens[attr]])
        counts = Counter(all_labels.tolist())
        majority[attr] = max(counts.values()) / len(all_labels)

    # Attack with 3 auditor types × 3 auditor seeds
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    all_rows = []
    auditor_seeds = [0, 1, 2]
    attrs = ["race", "sex", "age_group", "marital_status"]

    for attr in attrs:
        train_y = train_sens[attr]
        test_y = test_sens[attr]
        num_classes = int(max(train_y.max(), test_y.max())) + 1

        # Best single-purpose linear (for reference)
        best_single = 0.0
        for pname in purpose_names:
            lr = LogisticRegression(max_iter=2000, random_state=42)
            lr.fit(train_reps[pname], train_y)
            acc = lr.score(test_reps[pname], test_y)
            best_single = max(best_single, acc)

        # Concatenated attacks
        logreg_accs, mlp_accs, xgb_accs = [], [], []

        for aseed in auditor_seeds:
            # Linear
            lr = LogisticRegression(max_iter=2000, random_state=aseed)
            lr.fit(train_concat, train_y)
            logreg_accs.append(lr.score(test_concat, test_y))

            # MLP (2 layers, 256 hidden, dropout 0.3)
            mlp_acc = train_mlp_auditor(
                train_concat, train_y, test_concat, test_y,
                num_classes, device, seed=aseed, epochs=100,
            )
            mlp_accs.append(mlp_acc)

            # XGBoost
            xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=aseed,
                                eval_metric="logloss", verbosity=0)
            xgb.fit(train_concat, train_y)
            xgb_accs.append(xgb.score(test_concat, test_y))

        row = {
            "attribute": attr,
            "num_classes": num_classes,
            "majority_baseline": round(majority[attr], 4),
            "best_single_linear": round(best_single, 4),
            "concat_logreg_max": round(max(logreg_accs), 4),
            "concat_logreg_mean": round(float(np.mean(logreg_accs)), 4),
            "concat_mlp_max": round(max(mlp_accs), 4),
            "concat_mlp_mean": round(float(np.mean(mlp_accs)), 4),
            "concat_xgb_max": round(max(xgb_accs), 4),
            "concat_xgb_mean": round(float(np.mean(xgb_accs)), 4),
        }
        all_rows.append(row)

        print(f"  {attr:20s}: single_lin={best_single:.1%}, "
              f"concat_logreg={max(logreg_accs):.1%}, "
              f"concat_mlp={max(mlp_accs):.1%}, "
              f"concat_xgb={max(xgb_accs):.1%}, "
              f"majority={majority[attr]:.1%}")

    out_path = project_root / "results" / "adult" / "cross_purpose_nonlinear_fixed.csv"
    save_csv(all_rows, out_path)
    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# RE-RUN 3: Distribution Shift (Female / Male subgroups)
# ═══════════════════════════════════════════════════════════════════════════

def run_distribution_shift(device):
    print("\n" + "#" * 60)
    print("# RE-RUN 3: Distribution Shift (Female/Male subgroups)")
    print("#" * 60)

    purposes, registry, train_ds, val_ds, test_ds = load_adult()
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    # Train PCRL seed 0 (or load checkpoint)
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
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
        epochs=200, weight_decay=1e-4, early_stopping_patience=20,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "distshift_pcrl_s0"),
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    # Try loading existing checkpoint first
    ckpt = project_root / "checkpoints" / "fix_pcrl_adult_0" / "final.pt"
    if ckpt.exists():
        trainer.load_checkpoint(ckpt)
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  Training fresh (no checkpoint at {ckpt})...")
        trainer.train(train_loader, val_loader=val_loader)

    # Split test set by sex attribute
    sex_labels = test_ds.sensitive_attrs["sex"]  # 0=Female, 1=Male (or vice versa)
    female_indices = (sex_labels == 0).nonzero(as_tuple=True)[0].tolist()
    male_indices = (sex_labels == 1).nonzero(as_tuple=True)[0].tolist()

    print(f"  Test set: {len(test_ds)} total, {len(female_indices)} Female (sex=0), {len(male_indices)} Male (sex=1)")

    female_ds = Subset(test_ds, female_indices)
    male_ds = Subset(test_ds, male_indices)
    full_test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    female_loader = DataLoader(female_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    male_loader = DataLoader(male_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    all_rows = []

    for subset_name, subset_loader in [("full", full_test_loader), ("female", female_loader), ("male", male_loader)]:
        print(f"\n  Evaluating on {subset_name} subset...")

        reports = generate_report(
            encoder=encoder, train_loader=train_loader, test_loader=subset_loader,
            purpose_registry=registry, device=device,
        )

        pass_count = sum(1 for r in reports
                         if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)

        for r in reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            all_rows.append({
                "subset": subset_name,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "best_emp_acc": round(r.empirical_best_acc, 6),
                "majority_baseline": round(r.majority_proportion, 6),
                "delta": round(delta, 6),
                "linear_r2": round(r.linear_r2, 6),
                "adj_pass": ok,
                "pass_count": pass_count,
                "total_pairs": len(reports),
            })

        print(f"    {subset_name}: pass={pass_count}/{len(reports)}")
        for r in reports:
            delta = r.empirical_best_acc - r.majority_proportion
            print(f"      {r.purpose_name:25s} / {r.attr_name:15s}: delta={delta:+.1%}, R2={r.linear_r2:.4f}")

    out_path = project_root / "results" / "adult" / "distribution_shift_fixed.csv"
    save_csv(all_rows, out_path)
    return all_rows


# ═══════════════════════════════════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    t0 = time.time()

    run_conditioning_ablation(device)
    run_cross_purpose_attack(device)
    run_distribution_shift(device)

    elapsed = time.time() - t0
    print(f"\nAll re-runs complete! Total: {elapsed/60:.0f} minutes")


if __name__ == "__main__":
    main()

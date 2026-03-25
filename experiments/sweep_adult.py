#!/usr/bin/env python3
"""Adult-specific hyperparameter sweep for PCRL income_prediction.

Sweeps two training modes:
  - Minimax: lambda_adv × lambda_verify × auditor_steps
  - GRL (gradient reversal): lambda_adv × lambda_verify

Model: [128,128] hidden, 64 repr, 0.3 dropout, weight_decay=1e-4.
Early stopping with patience 15. Max 200 epochs.

Selection: income accuracy >70%, race AND sex empirical best acc <58%.
Saves results to results/adult/hyperparam_sweep.csv.
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress tqdm
import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    """No-op replacement for tqdm to suppress progress bars during sweep."""

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
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_config(
    mode: str,
    lambda_adv: float,
    lambda_verify: float,
    auditor_steps: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    purposes: list,
    registry: PurposeRegistry,
    device: str,
    input_dim: int,
    epochs: int = 200,
) -> dict:
    """Train and evaluate one configuration. Returns metrics dict."""
    torch.manual_seed(42)

    repr_dim = 64
    use_grl = mode == "grl"

    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=repr_dim,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
    )

    task_heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        task_heads[p.name] = TaskHead(
            repr_dim=repr_dim,
            output_dim=p.allowed_task_dims.get(p.allowed_tasks[0], 2),
        )

    # Stronger auditors: 3-layer MLP with 256 hidden units
    auditors: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256,
            num_layers=3,
            use_gradient_reversal=use_grl,
        )

    config = TrainerConfig(
        batch_size=256,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=lambda_adv,
        lambda_verify=lambda_verify,
        auditor_steps=auditor_steps if not use_grl else 0,
        epochs=epochs,
        weight_decay=1e-4,
        early_stopping_patience=15,
        log_interval=10000,  # effectively silent
        confusion_type="entropy",
        gradient_reversal=use_grl,
        checkpoint_dir=str(project_root / "checkpoints" / "sweep_adult_tmp"),
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
    epochs_completed = state.epoch + 1

    # Evaluate task accuracy
    eval_metrics = trainer.evaluate(test_loader)

    # Compliance report for income_prediction only (race, sex)
    # Build a mini-registry with just income_prediction for faster report
    income_purpose = purposes[0]  # income_prediction is first
    mini_registry = PurposeRegistry()
    mini_registry.register(income_purpose)

    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=mini_registry,
        device=device,
    )

    row: dict = {
        "mode": mode,
        "lambda_adv": lambda_adv,
        "lambda_verify": lambda_verify,
        "auditor_steps": auditor_steps,
        "epochs_completed": epochs_completed,
        "train_time_s": round(train_time, 1),
    }

    # Income task accuracy
    row["income_acc"] = round(
        eval_metrics.task_accuracy.get("income", 0.0), 4
    )

    # Other task accuracies (for reference)
    row["occ_group_acc"] = round(
        eval_metrics.task_accuracy.get("occupation_group", 0.0), 4
    )
    row["edu_level_acc"] = round(
        eval_metrics.task_accuracy.get("education_level", 0.0), 4
    )

    # Compliance metrics for income_prediction
    for r in reports:
        row[f"emp_best_{r.attr_name}"] = round(r.empirical_best_acc, 4)
        row[f"r2_{r.attr_name}"] = round(r.linear_r2, 4)
        row[f"certified_{r.attr_name}"] = r.certified

    # Training auditor accuracies
    for attr in ["race", "sex"]:
        row[f"train_aud_{attr}"] = round(
            eval_metrics.auditor_accuracy.get(attr, 0.0), 4
        )

    return row


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Adult dataset
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    print("Loading Adult dataset...")
    train_dataset = AdultDataset(
        purposes=purposes, root="data", split="train", download=True
    )
    val_dataset = AdultDataset(
        purposes=purposes, root="data", split="val", download=False
    )
    test_dataset = AdultDataset(
        purposes=purposes, root="data", split="test", download=False
    )

    input_dim = train_dataset.info.num_features
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}, Features: {input_dim}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch
    )

    # ── Sweep grid ────────────────────────────────────────────────────────
    lambda_advs = [10, 20, 50, 100]
    lambda_verifys = [10, 20, 50]
    auditor_steps_list = [10, 20]

    configs: list[tuple[str, float, float, int]] = []

    # Minimax configs
    for la in lambda_advs:
        for lv in lambda_verifys:
            for ast in auditor_steps_list:
                configs.append(("minimax", la, lv, ast))

    # GRL configs (auditor_steps irrelevant)
    for la in lambda_advs:
        for lv in lambda_verifys:
            configs.append(("grl", la, lv, 0))

    print(f"\n{len(configs)} configurations, max 200 epochs each (early stopping patience=15)")
    print("=" * 120)

    results: list[dict] = []
    total_t0 = time.time()

    for i, (mode, la, lv, ast) in enumerate(configs):
        ast_str = f"K={ast:>2}" if mode == "minimax" else "K=GRL"
        print(
            f"[{i + 1:>2}/{len(configs)}] {mode:<7} λ_adv={la:>3}  λ_verify={lv:>2}  {ast_str}",
            end="  →  ",
            flush=True,
        )

        row = run_config(
            mode, la, lv, ast,
            train_loader, val_loader, test_loader,
            purposes, registry, device, input_dim,
        )
        results.append(row)

        # Compact one-line summary
        income = row.get("income_acc", 0)
        race_emp = row.get("emp_best_race", 0)
        sex_emp = row.get("emp_best_sex", 0)
        epochs = row.get("epochs_completed", 0)
        t = row.get("train_time_s", 0)
        print(
            f"{t:>5.0f}s  ep={epochs:>3}  "
            f"income={income:.1%}  race_emp={race_emp:.1%}  sex_emp={sex_emp:.1%}"
        )

    total_time = time.time() - total_t0
    print(f"\nTotal sweep time: {total_time / 60:.1f} min")

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir = project_root / "results" / "adult"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hyperparam_sweep.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

    # ── Find best config ──────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("BEST CONFIGURATION (income_prediction focus)")
    print("=" * 120)

    # Target: income >70%, race AND sex emp_best <58%
    target_acc = 0.70
    target_aud = 0.58

    compliant = []
    for r in results:
        income_ok = r.get("income_acc", 0) >= target_acc
        race_ok = r.get("emp_best_race", 1.0) <= target_aud
        sex_ok = r.get("emp_best_sex", 1.0) <= target_aud
        if income_ok and race_ok and sex_ok:
            compliant.append((r["income_acc"], r))

    if compliant:
        compliant.sort(key=lambda x: -x[0])
        best_acc, best = compliant[0]
        print(f"  FOUND {len(compliant)} compliant config(s)!")
        print(
            f"  Best: mode={best['mode']}, λ_adv={best['lambda_adv']}, "
            f"λ_verify={best['lambda_verify']}, "
            f"aud_steps={best['auditor_steps']}"
        )
        print(f"  Income accuracy: {best_acc:.3f}")
        print(
            f"  Race emp best: {best.get('emp_best_race', 0):.3f}, "
            f"Sex emp best: {best.get('emp_best_sex', 0):.3f}"
        )
    else:
        print("  No config meets target (income>70%, race<58%, sex<58%).")
        print("  Closest configs by max auditor accuracy:")
        # Sort by max of (race, sex) emp best acc
        ranked = sorted(
            results,
            key=lambda r: max(
                r.get("emp_best_race", 1.0),
                r.get("emp_best_sex", 1.0),
            ),
        )
        for r in ranked[:5]:
            print(
                f"    {r['mode']:<7} λ_adv={r['lambda_adv']:>3} λ_verify={r['lambda_verify']:>2} "
                f"K={r['auditor_steps']:>2}  "
                f"income={r.get('income_acc', 0):.1%}  "
                f"race={r.get('emp_best_race', 0):.1%}  "
                f"sex={r.get('emp_best_sex', 0):.1%}  "
                f"ep={r.get('epochs_completed', 0)}"
            )

    # ── Full results table ────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("FULL SWEEP RESULTS")
    print("=" * 120)
    hdr = (
        f"{'Mode':<7} {'λ_adv':>5} {'λ_ver':>5} {'K':>3} {'Ep':>3} | "
        f"{'Income':>7} {'OccGrp':>7} {'EduLev':>7} | "
        f"{'Race emp':>8} {'Sex emp':>8} | "
        f"{'R² race':>7} {'R² sex':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['mode']:<7} {r['lambda_adv']:>5} {r['lambda_verify']:>5} "
            f"{r['auditor_steps']:>3} {r.get('epochs_completed', 0):>3} | "
            f"{r.get('income_acc', 0):>6.1%} "
            f"{r.get('occ_group_acc', 0):>6.1%} "
            f"{r.get('edu_level_acc', 0):>6.1%} | "
            f"{r.get('emp_best_race', 0):>7.1%} "
            f"{r.get('emp_best_sex', 0):>7.1%} | "
            f"{r.get('r2_race', 0):>7.4f} "
            f"{r.get('r2_sex', 0):>7.4f}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()

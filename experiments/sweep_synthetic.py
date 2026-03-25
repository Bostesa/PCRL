#!/usr/bin/env python3
"""Hyperparameter sweep for PCRL on synthetic data.

Sweeps lambda_adv in [1, 5, 10, 20], lambda_verify in [0, 1, 5],
auditor_steps in [5, 10] with stronger auditors (3-layer, 256 hidden).
Trains 200 epochs per config. Saves results to results/synthetic/hyperparam_sweep.csv.
Generates task-privacy tradeoff (Pareto frontier) plot.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress tqdm in trainer by monkey-patching
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

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.synthetic import SyntheticPCRLDataset, get_synthetic_purposes
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
    lambda_adv: float,
    lambda_verify: float,
    auditor_steps: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    purposes: list,
    registry: PurposeRegistry,
    device: str,
    epochs: int = 200,
) -> dict:
    """Train and evaluate one configuration. Returns metrics dict."""
    torch.manual_seed(42)

    input_dim = 20
    repr_dim = 64

    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=repr_dim,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
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
        )

    config = TrainerConfig(
        batch_size=256,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=lambda_adv,
        lambda_verify=lambda_verify,
        auditor_steps=auditor_steps,
        epochs=epochs,
        early_stopping_patience=None,
        log_interval=10000,  # effectively silent
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "sweep_tmp"),
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
    trainer.train(train_loader)  # skip val for speed
    train_time = time.time() - t0

    # Evaluate task accuracy + training auditor accuracy
    eval_metrics = trainer.evaluate(test_loader)

    # Full compliance report: linear R² + PostHocAuditorSuite
    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )

    row: dict = {
        "lambda_adv": lambda_adv,
        "lambda_verify": lambda_verify,
        "auditor_steps": auditor_steps,
        "train_time_s": round(train_time, 1),
    }

    for p in purposes:
        task_name = p.allowed_tasks[0]
        row[f"task_acc_{p.name}"] = round(
            eval_metrics.task_accuracy.get(task_name, 0.0), 4
        )

        for attr in p.disallowed_attrs:
            row[f"train_aud_{p.name}_{attr}"] = round(
                eval_metrics.auditor_accuracy.get(attr, 0.0), 4
            )
            match = [
                r
                for r in reports
                if r.purpose_name == p.name and r.attr_name == attr
            ]
            if match:
                row[f"r2_{p.name}_{attr}"] = round(match[0].linear_r2, 4)
                row[f"emp_best_{p.name}_{attr}"] = round(
                    match[0].empirical_best_acc, 4
                )
                row[f"certified_{p.name}_{attr}"] = match[0].certified

    return row


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    purposes = get_synthetic_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    dataset = SyntheticPCRLDataset(n_samples=5000, seed=42)
    n_train = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch
    )

    # ── Sweep grid ────────────────────────────────────────────────────────
    lambda_advs = [1, 5, 10, 20]
    lambda_verifys = [0, 1, 5]
    auditor_steps_list = [5, 10]

    configs = [
        (la, lv, ast)
        for la in lambda_advs
        for lv in lambda_verifys
        for ast in auditor_steps_list
    ]

    print(f"\n{len(configs)} configurations, 200 epochs each")
    print("=" * 110)

    results: list[dict] = []
    total_t0 = time.time()

    for i, (la, lv, ast) in enumerate(configs):
        print(
            f"[{i + 1:>2}/{len(configs)}] λ_adv={la:>2}  λ_verify={lv}  K={ast:>2}",
            end="  →  ",
            flush=True,
        )

        row = run_config(
            la, lv, ast, train_loader, test_loader, purposes, registry, device
        )
        results.append(row)

        # Compact one-line summary
        parts = [f"{row['train_time_s']:>5.0f}s"]
        for p in purposes:
            ta = row.get(f"task_acc_{p.name}", 0)
            parts.append(f"{p.name}={ta:.1%}")
        for p in purposes:
            for a in p.disallowed_attrs:
                emp = row.get(f"emp_best_{p.name}_{a}", 0)
                parts.append(f"emp_{a}={emp:.1%}")
        print("  ".join(parts))

    total_time = time.time() - total_t0
    print(f"\nTotal sweep time: {total_time / 60:.1f} min")

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_dir = project_root / "results" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hyperparam_sweep.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

    # ── Find best config ──────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("BEST CONFIGURATION")
    print("=" * 110)

    chance = 0.5
    threshold = chance + 0.05  # auditor accuracy must be ≤ 55%

    # Find configs where ALL empirical auditor accuracies are within threshold
    compliant = []
    for r in results:
        all_ok = all(
            r.get(f"emp_best_{p.name}_{a}", 1.0) <= threshold
            for p in purposes
            for a in p.disallowed_attrs
        )
        if all_ok:
            avg_task = sum(
                r.get(f"task_acc_{p.name}", 0) for p in purposes
            ) / len(purposes)
            compliant.append((avg_task, r))

    if compliant:
        compliant.sort(key=lambda x: -x[0])
        best_task_acc, best = compliant[0]
        print(
            f"  FOUND: λ_adv={best['lambda_adv']}, λ_verify={best['lambda_verify']}, "
            f"aud_steps={best['auditor_steps']}"
        )
        print(f"  Avg task accuracy: {best_task_acc:.3f}")
    else:
        print("  No config meets strict ≤55% threshold. Best tradeoff:")
        best = min(
            results,
            key=lambda r: max(
                r.get(f"emp_best_{p.name}_{a}", 1.0)
                for p in purposes
                for a in p.disallowed_attrs
            ),
        )
        print(
            f"  λ_adv={best['lambda_adv']}, λ_verify={best['lambda_verify']}, "
            f"aud_steps={best['auditor_steps']}"
        )

    for p in purposes:
        ta = best.get(f"task_acc_{p.name}", 0)
        print(f"  {p.name}: task_acc={ta:.3f}")
        for a in p.disallowed_attrs:
            emp = best.get(f"emp_best_{p.name}_{a}", 0)
            r2 = best.get(f"r2_{p.name}_{a}", 0)
            cert = best.get(f"certified_{p.name}_{a}", False)
            print(f"    {a}: emp={emp:.3f}, R²={r2:.4f}, certified={cert}")

    # ── Pareto frontier plot ──────────────────────────────────────────────
    print("\nGenerating task-privacy tradeoff plot...")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from pcrl.evaluation.visualize import plot_task_privacy_tradeoff

        pareto: dict[float, dict[str, tuple[float, float]]] = {}
        for la in lambda_advs:
            at_lambda = [r for r in results if r["lambda_adv"] == la]
            # Pick config with lowest max empirical accuracy at this lambda
            best_at = min(
                at_lambda,
                key=lambda r: max(
                    r.get(f"emp_best_{p.name}_{a}", 1.0)
                    for p in purposes
                    for a in p.disallowed_attrs
                ),
            )
            purpose_data: dict[str, tuple[float, float]] = {}
            for p in purposes:
                ta = best_at.get(f"task_acc_{p.name}", 0)
                max_cvr = max(
                    max(0, best_at.get(f"emp_best_{p.name}_{a}", 0) - chance)
                    for a in p.disallowed_attrs
                )
                purpose_data[p.name] = (ta, max_cvr)
            pareto[la] = purpose_data

        plot_task_privacy_tradeoff(
            pareto, save_path=out_dir / "task_privacy_tradeoff.png"
        )
        print(f"Saved {out_dir / 'task_privacy_tradeoff.png'}")
    except Exception as e:
        print(f"Plot failed: {e}")
        import traceback

        traceback.print_exc()

    # ── Full results table ────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("FULL SWEEP RESULTS")
    print("=" * 120)
    hdr = (
        f"{'λ_adv':>6} {'λ_ver':>5} {'K':>3} | "
        f"{'task_A':>7} {'task_B':>7} | "
        f"{'emp z1/A':>8} {'emp z2/A':>8} {'emp z1/B':>8} | "
        f"{'R² z1/A':>7} {'R² z2/A':>7} {'R² z1/B':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['lambda_adv']:>6} {r['lambda_verify']:>5} {r['auditor_steps']:>3} | "
            f"{r.get('task_acc_task_A', 0):>6.1%} {r.get('task_acc_task_B', 0):>6.1%} | "
            f"{r.get('emp_best_task_A_z_1', 0):>7.1%} "
            f"{r.get('emp_best_task_A_z_2', 0):>7.1%} "
            f"{r.get('emp_best_task_B_z_1', 0):>7.1%} | "
            f"{r.get('r2_task_A_z_1', 0):>7.4f} "
            f"{r.get('r2_task_A_z_2', 0):>7.4f} "
            f"{r.get('r2_task_B_z_1', 0):>7.4f}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()

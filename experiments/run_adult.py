#!/usr/bin/env python3
"""Run PCRL experiment on the Adult/Census dataset.

End-to-end experiment:
1. Load Adult dataset (downloads from UCI or falls back to synthetic)
2. Create encoder, task heads, auditors for 3 purposes
3. Train with adversarial + verification regularizer
4. Run compliance audit and print results
5. Generate visualizations

Best config from Adult sweep: λ_adv=50, λ_verify=50, K=10
with smaller model (128x128 hidden, 64 repr, 0.3 dropout).
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import generate_report, print_compliance_table
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_majority_baselines(
    test_dataset: AdultDataset,
) -> dict[str, float]:
    """Compute majority-class accuracy baselines for each sensitive attribute."""
    baselines: dict[str, float] = {}
    for attr_name, labels in test_dataset.sensitive_attrs.items():
        n = len(labels)
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / n
    return baselines


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ── 1. Purposes ──────────────────────────────────────────────────────
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    logger.info(f"Purposes: {registry.names}")
    for p in purposes:
        logger.info(
            f"  {p.name}: tasks={p.allowed_tasks}, "
            f"disallowed={p.disallowed_attrs}"
        )

    # ── 2. Load dataset ──────────────────────────────────────────────────
    logger.info("Loading Adult dataset...")
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
    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}, Features: {input_dim}"
    )

    # Compute majority-class baselines for proper evaluation
    majority_baselines = compute_majority_baselines(test_dataset)
    print("\nClass distribution baselines (majority-class accuracy):")
    for attr, baseline in majority_baselines.items():
        print(f"  {attr}: {baseline:.1%}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_pcrl_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_pcrl_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_pcrl_batch,
    )

    # ── 3. Build models ──────────────────────────────────────────────────
    # Smaller model to combat overfitting on Adult
    hidden_dims = [128, 128]
    repr_dim = 64
    purpose_emb_dim = 32
    num_purposes = len(purposes)

    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        repr_dim=repr_dim,
        num_purposes=num_purposes,
        purpose_emb_dim=purpose_emb_dim,
        conditioning="film",
        dropout=0.3,
    )

    task_heads: dict[str, torch.nn.Module] = {}
    for purpose in purposes:
        task_name = purpose.allowed_tasks[0]
        output_dim = purpose.allowed_task_dims.get(task_name, 2)
        task_heads[purpose.name] = TaskHead(
            repr_dim=repr_dim, output_dim=output_dim
        )

    # Stronger auditors: 3-layer MLP with 256 hidden units
    auditors: dict[str, torch.nn.Module] = {}
    for purpose in purposes:
        auditors[purpose.name] = MultiAttributeAuditor(
            repr_dim=repr_dim,
            attr_output_dims=purpose.disallowed_attr_dims,
            hidden_dim=256,
            num_layers=3,
        )

    # ── 4. Train ─────────────────────────────────────────────────────────
    # Best config from Adult sweep: λ_adv=50, λ_verify=50, K=10
    config = TrainerConfig(
        batch_size=256,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=50.0,
        lambda_verify=50.0,
        auditor_steps=10,
        epochs=200,
        weight_decay=1e-4,
        early_stopping_patience=15,
        log_interval=50,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "adult"),
    )

    trainer = PCRLTrainer(
        encoder=encoder,
        task_heads=task_heads,
        auditors=auditors,
        config=config,
        purpose_registry=registry,
        device=device,
    )

    logger.info("Starting training...")
    state = trainer.train(train_loader, val_loader=val_loader)
    logger.info(f"Training completed at epoch {state.epoch + 1}")

    # ── 5. Evaluation ────────────────────────────────────────────────────
    logger.info("Running final evaluation on test set...")
    eval_metrics = trainer.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS")
    print("=" * 60)
    print(f"  Total loss:       {eval_metrics.loss:.4f}")
    print(f"  Task loss:        {eval_metrics.task_loss:.4f}")
    print(f"  Adversarial loss: {eval_metrics.adversarial_loss:.4f}")
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"  Task '{task}' accuracy: {acc:.4f}")
    for attr, acc in eval_metrics.auditor_accuracy.items():
        baseline = majority_baselines.get(attr, 0.0)
        delta = acc - baseline
        print(f"  Auditor '{attr}' accuracy: {acc:.4f} (baseline={baseline:.4f}, delta={delta:+.4f})")

    # ── 6. Compliance audit ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPLIANCE AUDIT")
    print("=" * 60)

    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )
    print_compliance_table(reports)

    # ── 7. Majority-class-adjusted compliance ────────────────────────────
    print("\n" + "=" * 80)
    print("ADJUSTED COMPLIANCE (using majority-class baseline)")
    print("=" * 80)
    print("NOTE: For imbalanced datasets, chance_acc = 1/num_classes is misleading.")
    print("      The true baseline is the majority-class voting rate.\n")

    adj_header = (
        f"{'Purpose':<22} {'Attribute':<16} {'Best Acc':>9} "
        f"{'Majority':>9} {'Delta':>8} {'R²':>8} {'Status':>8}"
    )
    print(adj_header)
    print("-" * len(adj_header))

    adj_pass_count = 0
    adj_total = 0
    for r in reports:
        baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
        delta = r.empirical_best_acc - baseline
        # Adjusted pass: delta < 2% AND R² < 0.05
        adj_ok = delta < 0.02 and r.linear_r2 < 0.05
        status = "PASS" if adj_ok else "FAIL"
        if adj_ok:
            adj_pass_count += 1
        adj_total += 1
        print(
            f"{r.purpose_name:<22} {r.attr_name:<16} "
            f"{r.empirical_best_acc:>8.1%} {baseline:>8.1%} "
            f"{delta:>+7.1%} {r.linear_r2:>8.4f} {status:>8}"
        )

    print("-" * len(adj_header))
    print(f"Adjusted: {adj_pass_count}/{adj_total} pairs pass (delta<2%, R²<0.05)")
    print("=" * 80)

    # ── 8. Visualizations ────────────────────────────────────────────────
    results_dir = project_root / "results" / "adult"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")

        from pcrl.evaluation.visualize import (
            plot_certificate_heatmap,
            plot_cvr_comparison,
            plot_purpose_separation,
        )

        # Certificate heatmap
        plot_certificate_heatmap(
            reports, save_path=results_dir / "certificate_heatmap.png"
        )
        logger.info("Saved certificate_heatmap.png")

        # CVR comparison using majority-class baseline
        cvr_dict: dict[str, float] = {}
        for r in reports:
            key = f"{r.purpose_name}/{r.attr_name}"
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            cvr_dict[key] = max(0.0, r.empirical_best_acc - baseline)
        plot_cvr_comparison(
            {"PCRL": cvr_dict},
            save_path=results_dir / "cvr_comparison.png",
        )
        logger.info("Saved cvr_comparison.png")

        # t-SNE (use first batch of test data for speed)
        batch = next(iter(test_loader))
        plot_purpose_separation(
            encoder=encoder,
            features=batch["features"],
            sensitive_attrs=batch["sensitive_attrs"],
            purpose_names=[p.name for p in purposes],
            save_path=results_dir / "purpose_separation.png",
            n_samples=500,
        )
        logger.info("Saved purpose_separation.png")

    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    # ── 9. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = (
        f"{'Purpose':<22} {'Task Acc':>10} {'Best Emp':>10} "
        f"{'Baseline':>10} {'Delta':>8} {'Max R²':>8} {'Adj Cert':>10}"
    )
    print(header)
    print("-" * len(header))

    for purpose in purposes:
        task_name = purpose.allowed_tasks[0]
        task_acc = eval_metrics.task_accuracy.get(task_name, 0.0)

        purpose_reports = [
            r for r in reports if r.purpose_name == purpose.name
        ]

        # Find worst-case delta (highest above baseline)
        max_delta = 0.0
        max_emp = 0.0
        max_baseline = 0.0
        for r in purpose_reports:
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            if delta > max_delta:
                max_delta = delta
                max_emp = r.empirical_best_acc
                max_baseline = baseline

        max_r2 = max((r.linear_r2 for r in purpose_reports), default=0.0)
        all_adj_cert = all(
            (r.empirical_best_acc - majority_baselines.get(r.attr_name, r.empirical_chance_acc)) < 0.02
            and r.linear_r2 < 0.05
            for r in purpose_reports
        )
        cert_str = "PASS" if all_adj_cert else "FAIL"

        print(
            f"{purpose.name:<22} {task_acc:>9.1%} {max_emp:>9.1%} "
            f"{max_baseline:>9.1%} {max_delta:>+7.1%} "
            f"{max_r2:>8.4f} {cert_str:>10}"
        )

    print("=" * 90)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()

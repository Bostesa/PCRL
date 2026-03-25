#!/usr/bin/env python3
"""Run PCRL on synthetic dataset.

End-to-end experiment that:
1. Generates synthetic data with known structure
2. Creates encoder, task heads, and auditors
3. Trains for 50 epochs with adversarial + verification regularizer
4. Verifies different purposes produce different representations (cosine similarity, CKA)
5. Runs compliance audit (linear certificates + empirical auditors)
6. Computes CVR with probe suite
7. Tests composition: AND purpose hides union of disallowed attrs
8. Generates all visualizations and saves to results/
9. Prints final summary table
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.synthetic import SyntheticPCRLDataset, get_synthetic_purposes
from pcrl.evaluation.certificates import generate_report, print_compliance_table
from pcrl.evaluation.cvr import ProbeSuite, compliance_violation_rate
from pcrl.evaluation.probes import (
    cosine_similarity_batch,
    verify_purpose_differentiation,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.composition import compose_and
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ── 1. Create purposes and registry ──────────────────────────────────
    purposes = get_synthetic_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    logger.info(f"Purposes: {registry.names}")
    for p in purposes:
        logger.info(
            f"  {p.name}: tasks={p.allowed_tasks}, "
            f"disallowed={p.disallowed_attrs}"
        )

    # ── 2. Create dataset and loaders ────────────────────────────────────
    dataset = SyntheticPCRLDataset(n_samples=5000, seed=42)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_pcrl_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_pcrl_batch,
    )

    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ── 3. Build models ──────────────────────────────────────────────────
    input_dim = 20
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
    )

    # Task heads: one per purpose
    task_heads: dict[str, torch.nn.Module] = {}
    for purpose in purposes:
        task_name = purpose.allowed_tasks[0]
        output_dim = purpose.allowed_task_dims.get(task_name, 2)
        task_heads[purpose.name] = TaskHead(
            repr_dim=repr_dim,
            output_dim=output_dim,
        )

    # Auditors: one MultiAttributeAuditor per purpose
    auditors: dict[str, torch.nn.Module] = {}
    for purpose in purposes:
        auditors[purpose.name] = MultiAttributeAuditor(
            repr_dim=repr_dim,
            attr_output_dims=purpose.disallowed_attr_dims,
        )

    # ── 4. Training ──────────────────────────────────────────────────────
    config = TrainerConfig(
        batch_size=256,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=1.0,
        lambda_verify=0.1,
        auditor_steps=5,
        epochs=50,
        early_stopping_patience=None,
        log_interval=100,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "synthetic"),
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
    trainer.train(train_loader, val_loader=test_loader)

    # ── 5. Final evaluation ──────────────────────────────────────────────
    logger.info("Running final evaluation...")
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
        print(f"  Auditor '{attr}' accuracy: {acc:.4f}")

    # ── 6. Verify purpose differentiation (cosine similarity + CKA) ─────
    print("\n" + "=" * 60)
    print("PURPOSE DIFFERENTIATION")
    print("=" * 60)

    # Collect test features
    test_features_list = []
    test_sensitive_list: dict[str, list[torch.Tensor]] = {}
    for batch in test_loader:
        test_features_list.append(batch["features"])
        for attr_name, labels in batch["sensitive_attrs"].items():
            if attr_name not in test_sensitive_list:
                test_sensitive_list[attr_name] = []
            test_sensitive_list[attr_name].append(labels)

    test_features = torch.cat(test_features_list, dim=0)
    test_sensitive = {k: torch.cat(v, dim=0) for k, v in test_sensitive_list.items()}

    diff_results = verify_purpose_differentiation(
        encoder=encoder,
        purposes=purposes,
        test_features=test_features,
        device=device,
    )

    for pair, cka in diff_results["pairwise_cka"].items():
        cos = diff_results["pairwise_cosine"][pair]
        print(f"  {pair[0]} vs {pair[1]}:")
        print(f"    CKA: {cka:.4f}, Cosine: {cos:.4f}")

    if diff_results["representations_differ"]:
        print("  PASS: Representations differ across purposes")
    else:
        print(f"  WARNING: Similar pairs found: {diff_results['similar_pairs']}")

    # ── 7. Compliance audit ──────────────────────────────────────────────
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

    # ── 8. CVR with probe suite ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CVR (COMPLIANCE VIOLATION RATE)")
    print("=" * 60)

    probe_suite = ProbeSuite(
        epochs=50,
        patience=10,
        num_seeds=2,
        device=device,
    )

    cvr_results: dict[str, dict[str, float]] = {}
    for p_idx, purpose in enumerate(purposes):
        cvr_result = compliance_violation_rate(
            encoder=encoder,
            purpose_idx=p_idx,
            probe_suite=probe_suite,
            test_data=(test_features, test_sensitive),
            disallowed_attrs=purpose.disallowed_attrs,
            device=device,
        )
        print(f"\n  Purpose: {purpose.name}")
        print(f"    Overall CVR: {cvr_result['overall_cvr']:.4f}")
        for attr, cvr in cvr_result["per_attr_cvr"].items():
            acc = cvr_result["per_attr_accuracy"].get(attr, 0)
            chance = cvr_result["chance_levels"].get(attr, 0)
            print(f"    {attr}: CVR={cvr:.4f} (acc={acc:.4f}, chance={chance:.4f})")

        cvr_results[purpose.name] = cvr_result["per_attr_cvr"]

    # ── 9. Composition test (AND purpose) ────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPOSITION TEST (AND)")
    print("=" * 60)

    composed = compose_and(purposes[0], purposes[1])
    print(f"  {composed.name}:")
    print(f"    allowed_tasks = {composed.allowed_tasks}")
    print(f"    disallowed_attrs = {composed.disallowed_attrs}")

    # Verify semantics: AND should hide union of disallowed attrs
    expected_disallowed = sorted(
        set(purposes[0].disallowed_attrs) | set(purposes[1].disallowed_attrs)
    )
    actual_disallowed = composed.disallowed_attrs
    assert actual_disallowed == expected_disallowed, (
        f"AND composition mismatch: expected {expected_disallowed}, "
        f"got {actual_disallowed}"
    )
    print(f"    Union check: PASS (disallowed = {expected_disallowed})")

    # Evaluate composed purpose: use additive embedding composition
    # Register composed purpose and evaluate its CVR
    composed_spec = composed.to_purpose_spec()
    composed_registry = PurposeRegistry()
    composed_registry.register(composed_spec)

    # For composed purpose, use additive embedding approach
    # Get embeddings for both base purposes and add them
    encoder.eval()
    with torch.no_grad():
        emb_0 = encoder.get_purpose_embedding(0)
        emb_1 = encoder.get_purpose_embedding(1)
        composed_emb = emb_0 + emb_1

        # Get composed representations
        x_test = test_features.to(device)
        h_composed = encoder.forward_with_embedding(x_test, composed_emb).cpu()

    # Check that composed representation hides all disallowed attrs
    print("\n  Composed representation audit:")
    for attr_name in composed.disallowed_attrs:
        if attr_name not in test_sensitive:
            continue
        labels = test_sensitive[attr_name]
        num_classes = int(labels.max().item()) + 1
        chance = 1.0 / num_classes

        # Train a linear probe
        from pcrl.evaluation.cvr import LinearProbe, train_probe

        probe = LinearProbe(repr_dim, num_classes)
        trained_probe, _ = train_probe(
            probe=probe,
            train_repr=h_composed,
            train_labels=labels,
            epochs=50,
            lr=1e-3,
            device=device,
        )
        trained_probe.eval()
        with torch.no_grad():
            logits = trained_probe(h_composed.to(device))
            preds = logits.argmax(dim=-1)
            acc = (preds == labels.to(device)).float().mean().item()
        cvr_val = max(0.0, acc - chance)
        print(f"    {attr_name}: acc={acc:.4f}, chance={chance:.4f}, CVR={cvr_val:.4f}")

    # ── 10. Visualizations ───────────────────────────────────────────────
    results_dir = project_root / "results" / "synthetic"
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

        # CVR comparison
        cvr_plot_dict: dict[str, dict[str, float]] = {"PCRL": {}}
        for r in reports:
            key = f"{r.purpose_name}/{r.attr_name}"
            cvr_plot_dict["PCRL"][key] = max(
                0.0, r.empirical_best_acc - r.empirical_chance_acc
            )
        plot_cvr_comparison(
            cvr_plot_dict,
            save_path=results_dir / "cvr_comparison.png",
        )
        logger.info("Saved cvr_comparison.png")

        # t-SNE purpose separation
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

    # ── 11. Final summary table ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    # Build header with per-attribute columns
    all_sensitive = sorted(
        set(a for p in purposes for a in p.disallowed_attrs)
    )
    attr_cols = "".join(f" {'Aud ' + a:>12}" for a in all_sensitive)
    header = f"{'Purpose':<15} {'Task Acc':>10} {'CVR':>8} {'Lin Cert':>10}{attr_cols}"
    print(header)
    print("-" * len(header))

    for purpose in purposes:
        # Task accuracy
        task_name = purpose.allowed_tasks[0]
        task_acc = eval_metrics.task_accuracy.get(task_name, 0.0)

        # CVR (average across disallowed attrs for this purpose)
        purpose_cvr_dict = cvr_results.get(purpose.name, {})
        avg_cvr = (
            sum(purpose_cvr_dict.values()) / len(purpose_cvr_dict)
            if purpose_cvr_dict
            else 0.0
        )

        # Certificate status
        purpose_reports = [r for r in reports if r.purpose_name == purpose.name]
        all_certified = all(r.certified for r in purpose_reports)
        cert_status = "PASS" if all_certified else "FAIL"

        # Per-attribute auditor accuracy from reports
        attr_accs = ""
        for attr in all_sensitive:
            matching = [
                r for r in purpose_reports if r.attr_name == attr
            ]
            if matching:
                attr_accs += f" {matching[0].empirical_best_acc:>11.1%}"
            else:
                attr_accs += f" {'N/A':>12}"

        print(
            f"{purpose.name:<15} {task_acc:>9.1%} {avg_cvr:>8.4f} {cert_status:>10}{attr_accs}"
        )

    print("=" * 90)
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()

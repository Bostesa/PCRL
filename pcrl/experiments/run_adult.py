#!/usr/bin/env python3
"""Run PCRL experiment on the Adult/Census dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pcrl import (
    AdultDataset,
    PCRLConfig,
    PurposeSpec,
    TrainingConfig,
    ExperimentConfig,
    create_trainer,
    evaluate_all_purposes,
    load_config_from_yaml,
    get_adult_purposes,
)
from pcrl.evaluation import (
    run_full_probe_evaluation,
    evaluate_pcrl,
    verify_purpose_differentiation,
    compliance_violation_rate,
    ProbeSuite,
)
from pcrl.data.base import collate_pcrl_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PCRL experiment on Adult dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/adult_example.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store/load data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=None,
        help="Override adversarial loss weight",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation",
    )
    return parser.parse_args()


def create_default_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create default configuration if no config file provided."""
    purposes = get_adult_purposes()

    model_config = PCRLConfig(
        input_dim=108,  # Adult dataset feature dimension
        hidden_dims=[256, 128],
        repr_dim=64,
        purpose_emb_dim=32,
        purposes=purposes,
        conditioning="film",
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size or 256,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=args.lambda_adv or 1.0,
        auditor_steps=5,
        epochs=args.epochs or 100,
        seed=args.seed,
    )

    return ExperimentConfig(
        model=model_config,
        training=training_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiment_name="adult_pcrl",
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load or create config
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        config = load_config_from_yaml(config_path)
    else:
        logger.info("Using default configuration")
        config = create_default_config(args)

    # Override config with command line args
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lambda_adv is not None:
        config.training.lambda_adv = args.lambda_adv
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir

    logger.info(f"Using device: {args.device}")
    logger.info(f"Configuration: {config}")

    # Load datasets
    logger.info("Loading Adult dataset...")
    purposes = config.model.purposes

    train_dataset = AdultDataset(
        purposes=purposes,
        root=config.data_dir,
        split="train",
        download=True,
    )
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Feature dimension: {train_dataset.info.num_features}")

    val_dataset = AdultDataset(
        purposes=purposes,
        root=config.data_dir,
        split="val",
        download=False,
    )
    logger.info(f"Validation samples: {len(val_dataset)}")

    test_dataset = AdultDataset(
        purposes=purposes,
        root=config.data_dir,
        split="test",
        download=False,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # Update input_dim in config based on actual data
    config.model.input_dim = train_dataset.info.num_features

    # Create trainer
    trainer = create_trainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=args.device,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Training
    if not args.eval_only:
        logger.info("Starting training...")
        state = trainer.train()
        logger.info(f"Training completed. Best validation loss: {state.best_val_loss:.4f}")

    # Evaluation
    logger.info("Running evaluation...")

    # Load best checkpoint for evaluation
    best_checkpoint = Path(config.training.checkpoint_dir) / "best.pt"
    if best_checkpoint.exists() and not args.checkpoint:
        trainer.load_checkpoint(best_checkpoint)

    # CVR evaluation
    logger.info("Computing Compliance Violation Rate (CVR)...")
    cvr_results = evaluate_all_purposes(
        encoder=trainer.encoder,
        config=config.model,
        test_dataset=test_dataset,
        device=args.device,
    )

    for purpose_name, result in cvr_results.items():
        logger.info(f"\n=== CVR Results for {purpose_name} ===")
        logger.info(f"Overall CVR: {result.overall_cvr:.4f}")
        for attr, cvr in result.per_attr_cvr.items():
            logger.info(f"  {attr}: CVR={cvr:.4f}, Accuracy={result.auditor_accuracies[attr]:.4f}")

    # Post-hoc probe evaluation
    logger.info("\nRunning post-hoc probe evaluation...")
    purposes_list = [(i, p.name) for i, p in enumerate(config.model.purposes)]

    probe_results = run_full_probe_evaluation(
        encoder=trainer.encoder,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        purposes=purposes_list,
        probe_type="mlp",
        device=args.device,
    )

    for purpose_name, result in probe_results.items():
        logger.info(f"\n=== Probe Results for {purpose_name} ===")
        logger.info("Task probes:")
        for task_name, probe in result.task_probes.items():
            logger.info(f"  {task_name}: Accuracy={probe.accuracy:.4f}")
        logger.info("Sensitive attribute probes:")
        for attr_name, probe in result.sensitive_probes.items():
            logger.info(f"  {attr_name}: Accuracy={probe.accuracy:.4f}")

    # Representation analysis - verify purposes produce different representations
    logger.info("\nAnalyzing representation differentiation...")

    # Get test features for analysis
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_pcrl_batch,
    )

    # Collect all test features
    all_features = []
    for batch in test_loader:
        all_features.append(batch["features"])
    test_features = torch.cat(all_features, dim=0)

    differentiation_results = verify_purpose_differentiation(
        encoder=trainer.encoder,
        purposes=config.model.purposes,
        test_features=test_features,
        device=args.device,
    )

    logger.info(f"Representations differ adequately: {differentiation_results['representations_differ']}")
    logger.info("Pairwise CKA similarities:")
    for pair, cka in differentiation_results["pairwise_cka"].items():
        logger.info(f"  {pair[0]} <-> {pair[1]}: CKA={cka:.4f}")

    if differentiation_results["similar_pairs"]:
        logger.warning(f"Similar purpose pairs (CKA > 0.95): {differentiation_results['similar_pairs']}")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "config": {
            "model": {
                "input_dim": config.model.input_dim,
                "hidden_dims": config.model.hidden_dims,
                "repr_dim": config.model.repr_dim,
                "purpose_emb_dim": config.model.purpose_emb_dim,
                "conditioning": config.model.conditioning,
            },
            "training": {
                "batch_size": config.training.batch_size,
                "lr_encoder": config.training.lr_encoder,
                "lr_auditor": config.training.lr_auditor,
                "lambda_adv": config.training.lambda_adv,
                "auditor_steps": config.training.auditor_steps,
                "epochs": config.training.epochs,
                "seed": config.training.seed,
            },
        },
        "cvr_results": {
            purpose_name: {
                "overall_cvr": result.overall_cvr,
                "per_attr_cvr": result.per_attr_cvr,
                "auditor_accuracies": result.auditor_accuracies,
            }
            for purpose_name, result in cvr_results.items()
        },
        "probe_results": {
            purpose_name: {
                "task_probes": {
                    task_name: probe.accuracy
                    for task_name, probe in result.task_probes.items()
                },
                "sensitive_probes": {
                    attr_name: probe.accuracy
                    for attr_name, probe in result.sensitive_probes.items()
                },
            }
            for purpose_name, result in probe_results.items()
        },
        "representation_analysis": {
            "representations_differ": differentiation_results["representations_differ"],
            "pairwise_cka": {
                f"{p[0]}_vs_{p[1]}": v
                for p, v in differentiation_results["pairwise_cka"].items()
            },
            "pairwise_cosine": {
                f"{p[0]}_vs_{p[1]}": v
                for p, v in differentiation_results["pairwise_cosine"].items()
            },
        },
    }

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save final model
    model_path = output_dir / f"model_{timestamp}.pt"
    torch.save({
        "encoder_state_dict": trainer.encoder.state_dict(),
        "task_heads_state_dict": {k: v.state_dict() for k, v in trainer.task_heads.items()},
        "auditors_state_dict": {k: v.state_dict() for k, v in trainer.auditors.items()},
        "config": results["config"],
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # Generate plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Plot CVR comparison across purposes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # CVR bar chart
        ax1 = axes[0]
        purpose_names = list(cvr_results.keys())
        cvr_values = [cvr_results[p].overall_cvr for p in purpose_names]
        ax1.bar(purpose_names, cvr_values, color='steelblue')
        ax1.set_ylabel("Overall CVR")
        ax1.set_title("Compliance Violation Rate by Purpose")
        ax1.set_ylim(0, 1)
        for i, v in enumerate(cvr_values):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha='center')

        # Task accuracy vs sensitive leakage scatter
        ax2 = axes[1]
        for purpose_name, result in probe_results.items():
            task_acc = sum(p.accuracy for p in result.task_probes.values()) / max(len(result.task_probes), 1)
            sens_acc = sum(p.accuracy for p in result.sensitive_probes.values()) / max(len(result.sensitive_probes), 1)
            ax2.scatter(task_acc, sens_acc, s=100, label=purpose_name)
            ax2.annotate(purpose_name, (task_acc, sens_acc), textcoords="offset points", xytext=(5, 5))

        ax2.set_xlabel("Task Accuracy")
        ax2.set_ylabel("Sensitive Attr Probe Accuracy")
        ax2.set_title("Task Performance vs Information Leakage")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.3, label='Chance level (binary)')
        ax2.legend()

        plt.tight_layout()
        plot_path = output_dir / f"analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Plots saved to {plot_path}")
        plt.close()

    except ImportError:
        logger.info("matplotlib not available, skipping plots")

    logger.info("\nExperiment completed!")


if __name__ == "__main__":
    main()

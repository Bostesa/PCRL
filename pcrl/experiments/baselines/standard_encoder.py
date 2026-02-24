#!/usr/bin/env python3
"""Standard encoder baseline - no adversarial training.

This baseline trains the same encoder architecture as PCRL but without
adversarial training to remove sensitive information. It measures what's
recoverable from vanilla representations.

This serves as an upper bound on information leakage - the PCRL model
should leak less sensitive information than this baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pcrl import (
    AdultDataset,
    get_adult_purposes,
)
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.training.losses import task_loss
from pcrl.evaluation import (
    run_full_probe_evaluation,
    verify_purpose_differentiation,
    ProbeSuite,
    compliance_violation_rate,
)
from pcrl.data.base import collate_pcrl_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StandardEncoder(nn.Module):
    """Standard encoder without purpose conditioning or adversarial training."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        repr_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(hidden_dims[-1], repr_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, purpose_idx: int | None = None) -> torch.Tensor:
        """Forward pass. purpose_idx is ignored (included for API compatibility)."""
        return self.network(x)


def train_standard_encoder(
    encoder: nn.Module,
    task_heads: dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_names: list[str],
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 10,
) -> dict[str, list[float]]:
    """Train standard encoder on task prediction only.

    Args:
        encoder: Encoder network.
        task_heads: Dictionary of task prediction heads.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        task_names: Names of tasks to train on.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to use.
        patience: Early stopping patience.

    Returns:
        Dictionary with training history.
    """
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    for head in task_heads.values():
        head.to(device_obj)

    # Single optimizer for encoder and all task heads
    params = list(encoder.parameters())
    for head in task_heads.values():
        params.extend(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        encoder.train()
        for head in task_heads.values():
            head.train()

        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device_obj)
            task_labels = {k: v.to(device_obj) for k, v in batch["task_labels"].items()}

            optimizer.zero_grad()

            # Get representations (no purpose conditioning)
            representations = encoder(features)

            # Compute task losses
            total_loss = torch.tensor(0.0, device=device_obj)
            for task_name in task_names:
                if task_name in task_labels and task_name in task_heads:
                    preds = task_heads[task_name](representations)
                    loss = task_loss(preds, task_labels[task_name], task_type="classification")
                    total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        encoder.eval()
        for head in task_heads.values():
            head.eval()

        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device_obj)
                task_labels = {k: v.to(device_obj) for k, v in batch["task_labels"].items()}

                representations = encoder(features)

                batch_loss = torch.tensor(0.0, device=device_obj)
                for task_name in task_names:
                    if task_name in task_labels and task_name in task_heads:
                        preds = task_heads[task_name](representations)
                        loss = task_loss(preds, task_labels[task_name], task_type="classification")
                        batch_loss = batch_loss + loss

                val_loss += batch_loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        history["val_loss"].append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "task_heads": {
                    name: {k: v.cpu().clone() for k, v in head.state_dict().items()}
                    for name, head in task_heads.items()
                },
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

    # Restore best model
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device_obj) for k, v in best_state["encoder"].items()})
        for name, head in task_heads.items():
            head.load_state_dict({k: v.to(device_obj) for k, v in best_state["task_heads"][name].items()})

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard encoder baseline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/baselines/standard")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--repr-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    purposes = get_adult_purposes()
    logger.info("Loading Adult dataset...")

    train_dataset = AdultDataset(purposes=purposes, root=args.data_dir, split="train")
    val_dataset = AdultDataset(purposes=purposes, root=args.data_dir, split="val")
    test_dataset = AdultDataset(purposes=purposes, root=args.data_dir, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pcrl_batch)

    input_dim = train_dataset.info.num_features
    logger.info(f"Input dimension: {input_dim}")

    # Create model
    encoder = StandardEncoder(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        repr_dim=args.repr_dim,
    )

    # Create task heads for all tasks
    task_names = list(train_dataset.info.task_labels.keys())
    task_heads = {
        name: TaskHead(
            repr_dim=args.repr_dim,
            output_dim=train_dataset.info.task_labels[name],
            hidden_dim=64,
        )
        for name in task_names
    }

    logger.info(f"Training standard encoder on tasks: {task_names}")

    # Train
    history = train_standard_encoder(
        encoder=encoder,
        task_heads=task_heads,
        train_loader=train_loader,
        val_loader=val_loader,
        task_names=task_names,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )

    # Evaluate
    logger.info("\n=== Evaluation ===")

    # Collect test features
    all_features = []
    all_task_labels = {name: [] for name in task_names}
    all_sensitive_attrs = {name: [] for name in train_dataset.info.sensitive_attrs.keys()}

    for batch in test_loader:
        all_features.append(batch["features"])
        for name in task_names:
            if name in batch["task_labels"]:
                all_task_labels[name].append(batch["task_labels"][name])
        for name in all_sensitive_attrs.keys():
            if name in batch["sensitive_attrs"]:
                all_sensitive_attrs[name].append(batch["sensitive_attrs"][name])

    test_features = torch.cat(all_features, dim=0)
    test_task_labels = {name: torch.cat(labels, dim=0) for name, labels in all_task_labels.items() if labels}
    test_sensitive_attrs = {name: torch.cat(labels, dim=0) for name, labels in all_sensitive_attrs.items() if labels}

    # Task accuracy
    encoder.eval()
    encoder.to(args.device)
    for head in task_heads.values():
        head.eval()
        head.to(args.device)

    with torch.no_grad():
        representations = encoder(test_features.to(args.device))

        task_accuracies = {}
        for name, head in task_heads.items():
            if name in test_task_labels:
                preds = head(representations).argmax(dim=-1)
                acc = (preds.cpu() == test_task_labels[name]).float().mean().item()
                task_accuracies[name] = acc
                logger.info(f"Task {name} accuracy: {acc:.4f}")

    # Sensitive attribute leakage (CVR)
    logger.info("\nMeasuring sensitive attribute leakage...")
    probe_suite = ProbeSuite(device=args.device)

    sensitive_attr_names = list(test_sensitive_attrs.keys())
    cvr_results = compliance_violation_rate(
        encoder=encoder,
        purpose_idx=0,  # Doesn't matter for standard encoder
        probe_suite=probe_suite,
        test_data=(test_features, test_sensitive_attrs),
        disallowed_attrs=sensitive_attr_names,
        device=args.device,
    )

    logger.info(f"Overall CVR: {cvr_results['overall_cvr']:.4f}")
    for attr, cvr in cvr_results["per_attr_cvr"].items():
        logger.info(f"  {attr}: CVR={cvr:.4f}, Accuracy={cvr_results['per_attr_accuracy'][attr]:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "baseline": "standard_encoder",
        "timestamp": timestamp,
        "config": {
            "hidden_dims": args.hidden_dims,
            "repr_dim": args.repr_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
        },
        "task_accuracies": task_accuracies,
        "cvr_results": {
            "overall_cvr": cvr_results["overall_cvr"],
            "per_attr_cvr": cvr_results["per_attr_cvr"],
            "per_attr_accuracy": cvr_results["per_attr_accuracy"],
        },
        "training_history": history,
    }

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save model
    model_path = output_dir / f"model_{timestamp}.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "task_heads_state_dict": {k: v.state_dict() for k, v in task_heads.items()},
    }, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

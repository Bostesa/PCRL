#!/usr/bin/env python3
"""Adversarial fair representation baseline (LAFTR-style).

This baseline implements standard adversarial fair representation learning
with fixed sensitive attributes and no purpose conditioning. It demonstrates:
1. Traditional fairness approach removes ALL specified sensitive info
2. Cannot dynamically switch what to protect based on purpose
3. May unnecessarily remove useful information for some applications

This serves as a comparison showing why purpose-conditioned representations
are more flexible than static fair representations.
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
from pcrl.models.task_head import TaskHead
from pcrl.models.auditor import Auditor
from pcrl.training.losses import task_loss, adversarial_loss
from pcrl.evaluation import ProbeSuite, compliance_violation_rate
from pcrl.data.base import collate_pcrl_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FairEncoder(nn.Module):
    """Fair encoder with adversarial training on fixed sensitive attributes."""

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
        """Forward pass. purpose_idx is ignored (fixed fairness constraints)."""
        return self.network(x)


def train_fair_encoder(
    encoder: nn.Module,
    task_heads: dict[str, nn.Module],
    auditors: dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    protected_attrs: list[str],
    task_names: list[str],
    epochs: int = 100,
    lr_encoder: float = 1e-3,
    lr_auditor: float = 1e-3,
    lambda_adv: float = 1.0,
    auditor_steps: int = 5,
    device: str = "cpu",
    patience: int = 10,
) -> dict[str, list[float]]:
    """Train fair encoder with adversarial debiasing.

    Uses alternating optimization:
    1. Train auditors to predict protected attributes
    2. Train encoder to maximize auditor confusion while minimizing task loss

    Args:
        encoder: Encoder network.
        task_heads: Task prediction heads.
        auditors: Auditor networks for protected attributes.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        protected_attrs: List of attributes to protect (remove from representation).
        task_names: Names of tasks to train on.
        epochs: Number of training epochs.
        lr_encoder: Learning rate for encoder and task heads.
        lr_auditor: Learning rate for auditors.
        lambda_adv: Adversarial loss weight.
        auditor_steps: Number of auditor updates per encoder update.
        device: Device to use.
        patience: Early stopping patience.

    Returns:
        Dictionary with training history.
    """
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    for head in task_heads.values():
        head.to(device_obj)
    for auditor in auditors.values():
        auditor.to(device_obj)

    # Optimizers
    encoder_params = list(encoder.parameters())
    for head in task_heads.values():
        encoder_params.extend(head.parameters())
    encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr_encoder)

    auditor_params = []
    for auditor in auditors.values():
        auditor_params.extend(auditor.parameters())
    auditor_optimizer = torch.optim.Adam(auditor_params, lr=lr_auditor)

    history = {
        "train_task_loss": [],
        "train_adv_loss": [],
        "val_task_loss": [],
        "auditor_loss": [],
    }
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        encoder.train()
        for head in task_heads.values():
            head.train()
        for auditor in auditors.values():
            auditor.train()

        train_task_loss = 0.0
        train_adv_loss = 0.0
        auditor_loss_total = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device_obj)
            task_labels = {k: v.to(device_obj) for k, v in batch["task_labels"].items()}
            sensitive_attrs = {k: v.to(device_obj) for k, v in batch["sensitive_attrs"].items()}

            # Step 1: Update auditors K times
            for _ in range(auditor_steps):
                auditor_optimizer.zero_grad()

                with torch.no_grad():
                    representations = encoder(features)

                auditor_batch_loss = torch.tensor(0.0, device=device_obj)
                for attr_name in protected_attrs:
                    if attr_name in sensitive_attrs and attr_name in auditors:
                        preds = auditors[attr_name](representations.detach())
                        loss = adversarial_loss(preds, sensitive_attrs[attr_name], maximize=False)
                        auditor_batch_loss = auditor_batch_loss + loss

                auditor_batch_loss.backward()
                auditor_optimizer.step()

            # Step 2: Update encoder and task heads
            encoder_optimizer.zero_grad()

            representations = encoder(features)

            # Task loss
            total_task_loss = torch.tensor(0.0, device=device_obj)
            for task_name in task_names:
                if task_name in task_labels and task_name in task_heads:
                    preds = task_heads[task_name](representations)
                    loss = task_loss(preds, task_labels[task_name], task_type="classification")
                    total_task_loss = total_task_loss + loss

            # Adversarial loss (confusion) - try to maximize auditor error
            total_adv_loss = torch.tensor(0.0, device=device_obj)
            for attr_name in protected_attrs:
                if attr_name in sensitive_attrs and attr_name in auditors:
                    preds = auditors[attr_name](representations)
                    # Use entropy maximization for confusion
                    loss = adversarial_loss(
                        preds, sensitive_attrs[attr_name],
                        maximize=True, confusion_type="entropy"
                    )
                    total_adv_loss = total_adv_loss + loss

            total_loss = total_task_loss + lambda_adv * total_adv_loss
            total_loss.backward()
            encoder_optimizer.step()

            train_task_loss += total_task_loss.item()
            train_adv_loss += total_adv_loss.item()
            auditor_loss_total += auditor_batch_loss.item()
            num_batches += 1

        avg_train_task = train_task_loss / num_batches
        avg_train_adv = train_adv_loss / num_batches
        avg_auditor = auditor_loss_total / num_batches

        history["train_task_loss"].append(avg_train_task)
        history["train_adv_loss"].append(avg_train_adv)
        history["auditor_loss"].append(avg_auditor)

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
        history["val_task_loss"].append(avg_val_loss)

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
            logger.info(
                f"Epoch {epoch + 1}: task={avg_train_task:.4f}, "
                f"adv={avg_train_adv:.4f}, val={avg_val_loss:.4f}"
            )

    # Restore best model
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device_obj) for k, v in best_state["encoder"].items()})
        for name, head in task_heads.items():
            head.load_state_dict({k: v.to(device_obj) for k, v in best_state["task_heads"][name].items()})

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial fair representation baseline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/baselines/adversarial_fair")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-encoder", type=float, default=1e-3)
    parser.add_argument("--lr-auditor", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--auditor-steps", type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--repr-dim", type=int, default=64)
    parser.add_argument("--protected-attrs", type=str, nargs="+", default=["sex", "race"])
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
    logger.info(f"Protected attributes: {args.protected_attrs}")

    # Create model
    encoder = FairEncoder(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        repr_dim=args.repr_dim,
    )

    # Create task heads for ALL tasks (single representation for all)
    task_names = list(train_dataset.info.task_labels.keys())
    task_heads = {
        name: TaskHead(
            repr_dim=args.repr_dim,
            output_dim=train_dataset.info.task_labels[name],
            hidden_dim=64,
        )
        for name in task_names
    }
    logger.info(f"Task heads: {task_names}")

    # Create auditors for protected attributes
    auditors = {}
    for attr_name in args.protected_attrs:
        if attr_name in train_dataset.info.sensitive_attrs:
            auditors[attr_name] = Auditor(
                repr_dim=args.repr_dim,
                output_dim=train_dataset.info.sensitive_attrs[attr_name],
                hidden_dim=64,
                num_layers=2,
            )
    logger.info(f"Auditors for: {list(auditors.keys())}")

    # Train
    logger.info("\nTraining fair encoder...")
    history = train_fair_encoder(
        encoder=encoder,
        task_heads=task_heads,
        auditors=auditors,
        train_loader=train_loader,
        val_loader=val_loader,
        protected_attrs=args.protected_attrs,
        task_names=task_names,
        epochs=args.epochs,
        lr_encoder=args.lr_encoder,
        lr_auditor=args.lr_auditor,
        lambda_adv=args.lambda_adv,
        auditor_steps=args.auditor_steps,
        device=args.device,
    )

    # Evaluate
    logger.info("\n=== Evaluation ===")

    encoder.eval()
    encoder.to(args.device)
    for head in task_heads.values():
        head.eval()
        head.to(args.device)

    # Collect test data
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
    logger.info("\nTask accuracies:")
    task_accuracies = {}
    with torch.no_grad():
        representations = encoder(test_features.to(args.device))

        for name, head in task_heads.items():
            if name in test_task_labels:
                preds = head(representations).argmax(dim=-1)
                acc = (preds.cpu() == test_task_labels[name]).float().mean().item()
                task_accuracies[name] = acc
                logger.info(f"  {name}: {acc:.4f}")

    # CVR for protected attributes
    logger.info("\nCVR for protected attributes:")
    probe_suite = ProbeSuite(device=args.device)

    cvr_results = compliance_violation_rate(
        encoder=encoder,
        purpose_idx=0,  # Doesn't matter for fixed fair encoder
        probe_suite=probe_suite,
        test_data=(test_features, test_sensitive_attrs),
        disallowed_attrs=args.protected_attrs,
        device=args.device,
    )

    logger.info(f"Overall CVR: {cvr_results['overall_cvr']:.4f}")
    for attr, cvr in cvr_results["per_attr_cvr"].items():
        logger.info(f"  {attr}: CVR={cvr:.4f}, Accuracy={cvr_results['per_attr_accuracy'][attr]:.4f}")

    # Also measure leakage of NON-protected attributes (to show we might remove too much)
    non_protected = [a for a in test_sensitive_attrs.keys() if a not in args.protected_attrs]
    if non_protected:
        logger.info("\nCVR for NON-protected attributes (should be high if useful):")
        non_protected_cvr = compliance_violation_rate(
            encoder=encoder,
            purpose_idx=0,
            probe_suite=probe_suite,
            test_data=(test_features, test_sensitive_attrs),
            disallowed_attrs=non_protected,
            device=args.device,
        )
        for attr, cvr in non_protected_cvr["per_attr_cvr"].items():
            logger.info(f"  {attr}: CVR={cvr:.4f}")

    # Key insight: Show that we can't dynamically change what's protected
    logger.info("\n=== Key Limitation: No Dynamic Purpose Switching ===")
    logger.info("This model always protects the same attributes (sex, race).")
    logger.info("It cannot adapt to different use cases that might need different")
    logger.info("fairness constraints. PCRL addresses this by conditioning on purpose.")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "baseline": "adversarial_fair",
        "timestamp": timestamp,
        "config": {
            "hidden_dims": args.hidden_dims,
            "repr_dim": args.repr_dim,
            "epochs": args.epochs,
            "lr_encoder": args.lr_encoder,
            "lr_auditor": args.lr_auditor,
            "lambda_adv": args.lambda_adv,
            "protected_attrs": args.protected_attrs,
            "seed": args.seed,
        },
        "task_accuracies": task_accuracies,
        "protected_cvr": {
            "overall": cvr_results["overall_cvr"],
            "per_attr": cvr_results["per_attr_cvr"],
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

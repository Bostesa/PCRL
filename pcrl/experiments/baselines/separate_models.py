#!/usr/bin/env python3
"""Separate models baseline - one encoder per purpose.

This baseline trains independent encoder models for each purpose,
without any shared representation. It demonstrates:
1. The compute cost of training separate models
2. The fairness achievable when models can be fully specialized
3. Why purpose-conditioned single models are more practical

This serves as a comparison point for PCRL's parameter efficiency
while achieving similar fairness properties.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from pcrl.utils.config import PurposeSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PurposeSpecificEncoder(nn.Module):
    """Encoder trained for a single specific purpose."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_purpose_specific_model(
    encoder: nn.Module,
    task_heads: dict[str, nn.Module],
    auditors: dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    purpose: PurposeSpec,
    epochs: int = 100,
    lr_encoder: float = 1e-3,
    lr_auditor: float = 1e-3,
    lambda_adv: float = 1.0,
    auditor_steps: int = 5,
    device: str = "cpu",
    patience: int = 10,
) -> dict[str, list[float]]:
    """Train an encoder specific to one purpose with adversarial training.

    Args:
        encoder: Encoder network.
        task_heads: Task prediction heads for allowed tasks.
        auditors: Auditor networks for disallowed attributes.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        purpose: Purpose specification.
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
    auditor_optimizer = torch.optim.Adam(auditor_params, lr=lr_auditor) if auditor_params else None

    history = {"train_loss": [], "val_loss": [], "auditor_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        encoder.train()
        for head in task_heads.values():
            head.train()
        for auditor in auditors.values():
            auditor.train()

        train_loss = 0.0
        auditor_loss_total = 0.0
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device_obj)
            task_labels = {k: v.to(device_obj) for k, v in batch["task_labels"].items()}
            sensitive_attrs = {k: v.to(device_obj) for k, v in batch["sensitive_attrs"].items()}

            # Step 1: Update auditors K times
            for _ in range(auditor_steps):
                if auditor_optimizer is not None:
                    auditor_optimizer.zero_grad()

                    with torch.no_grad():
                        representations = encoder(features)

                    auditor_batch_loss = torch.tensor(0.0, device=device_obj)
                    for attr_name in purpose.disallowed_attrs:
                        if attr_name in sensitive_attrs and attr_name in auditors:
                            preds = auditors[attr_name](representations.detach())
                            loss = adversarial_loss(preds, sensitive_attrs[attr_name], maximize=False)
                            auditor_batch_loss = auditor_batch_loss + loss

                    if auditor_batch_loss.requires_grad:
                        auditor_batch_loss.backward()
                        auditor_optimizer.step()

            # Step 2: Update encoder and task heads
            encoder_optimizer.zero_grad()

            representations = encoder(features)

            # Task loss
            total_task_loss = torch.tensor(0.0, device=device_obj)
            for task_name in purpose.allowed_tasks:
                if task_name in task_labels and task_name in task_heads:
                    preds = task_heads[task_name](representations)
                    loss = task_loss(preds, task_labels[task_name], task_type="classification")
                    total_task_loss = total_task_loss + loss

            # Adversarial loss (confusion)
            total_adv_loss = torch.tensor(0.0, device=device_obj)
            for attr_name in purpose.disallowed_attrs:
                if attr_name in sensitive_attrs and attr_name in auditors:
                    preds = auditors[attr_name](representations)
                    loss = adversarial_loss(preds, sensitive_attrs[attr_name], maximize=True)
                    total_adv_loss = total_adv_loss + loss

            total_loss = total_task_loss + lambda_adv * total_adv_loss
            total_loss.backward()
            encoder_optimizer.step()

            train_loss += total_task_loss.item()
            auditor_loss_total += auditor_batch_loss.item() if auditor_optimizer else 0.0
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        avg_auditor_loss = auditor_loss_total / num_batches
        history["train_loss"].append(avg_train_loss)
        history["auditor_loss"].append(avg_auditor_loss)

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
                for task_name in purpose.allowed_tasks:
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
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch + 1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")

    # Restore best model
    if best_state is not None:
        encoder.load_state_dict({k: v.to(device_obj) for k, v in best_state["encoder"].items()})
        for name, head in task_heads.items():
            head.load_state_dict({k: v.to(device_obj) for k, v in best_state["task_heads"][name].items()})

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate models baseline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/baselines/separate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-encoder", type=float, default=1e-3)
    parser.add_argument("--lr-auditor", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--auditor-steps", type=int, default=5)
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
    logger.info(f"Number of purposes: {len(purposes)}")

    # Train separate model for each purpose
    all_results = {}
    total_params = 0
    total_train_time = 0.0

    for purpose in purposes:
        logger.info(f"\n=== Training model for purpose: {purpose.name} ===")
        logger.info(f"  Allowed tasks: {purpose.allowed_tasks}")
        logger.info(f"  Disallowed attrs: {purpose.disallowed_attrs}")

        # Create encoder for this purpose
        encoder = PurposeSpecificEncoder(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            repr_dim=args.repr_dim,
        )

        # Count parameters
        encoder_params = sum(p.numel() for p in encoder.parameters())

        # Create task heads for allowed tasks
        task_heads = {}
        for task_name in purpose.allowed_tasks:
            if task_name in train_dataset.info.task_labels:
                task_heads[task_name] = TaskHead(
                    repr_dim=args.repr_dim,
                    output_dim=train_dataset.info.task_labels[task_name],
                    hidden_dim=64,
                )

        # Create auditors for disallowed attributes
        auditors = {}
        for attr_name in purpose.disallowed_attrs:
            if attr_name in train_dataset.info.sensitive_attrs:
                auditors[attr_name] = Auditor(
                    repr_dim=args.repr_dim,
                    output_dim=train_dataset.info.sensitive_attrs[attr_name],
                    hidden_dim=64,
                    num_layers=2,
                )

        # Count total params for this purpose
        purpose_params = encoder_params
        for head in task_heads.values():
            purpose_params += sum(p.numel() for p in head.parameters())
        total_params += purpose_params

        logger.info(f"  Parameters for this purpose: {purpose_params:,}")

        # Train
        start_time = time.time()
        history = train_purpose_specific_model(
            encoder=encoder,
            task_heads=task_heads,
            auditors=auditors,
            train_loader=train_loader,
            val_loader=val_loader,
            purpose=purpose,
            epochs=args.epochs,
            lr_encoder=args.lr_encoder,
            lr_auditor=args.lr_auditor,
            lambda_adv=args.lambda_adv,
            auditor_steps=args.auditor_steps,
            device=args.device,
        )
        train_time = time.time() - start_time
        total_train_time += train_time
        logger.info(f"  Training time: {train_time:.1f}s")

        # Evaluate
        encoder.eval()
        encoder.to(args.device)
        for head in task_heads.values():
            head.eval()
            head.to(args.device)

        # Collect test data
        all_features = []
        all_task_labels = {name: [] for name in train_dataset.info.task_labels.keys()}
        all_sensitive_attrs = {name: [] for name in train_dataset.info.sensitive_attrs.keys()}

        for batch in test_loader:
            all_features.append(batch["features"])
            for name in all_task_labels.keys():
                if name in batch["task_labels"]:
                    all_task_labels[name].append(batch["task_labels"][name])
            for name in all_sensitive_attrs.keys():
                if name in batch["sensitive_attrs"]:
                    all_sensitive_attrs[name].append(batch["sensitive_attrs"][name])

        test_features = torch.cat(all_features, dim=0)
        test_task_labels = {name: torch.cat(labels, dim=0) for name, labels in all_task_labels.items() if labels}
        test_sensitive_attrs = {name: torch.cat(labels, dim=0) for name, labels in all_sensitive_attrs.items() if labels}

        # Task accuracy
        with torch.no_grad():
            representations = encoder(test_features.to(args.device))

            task_accuracies = {}
            for name, head in task_heads.items():
                if name in test_task_labels:
                    preds = head(representations).argmax(dim=-1)
                    acc = (preds.cpu() == test_task_labels[name]).float().mean().item()
                    task_accuracies[name] = acc
                    logger.info(f"  Task {name} accuracy: {acc:.4f}")

        # CVR for this purpose
        probe_suite = ProbeSuite(device=args.device, num_seeds=2)

        # Wrapper for API compatibility
        class EncoderWrapper(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.enc = enc

            def forward(self, x, purpose_idx=None):
                return self.enc(x)

        wrapped_encoder = EncoderWrapper(encoder)

        cvr_results = compliance_violation_rate(
            encoder=wrapped_encoder,
            purpose_idx=0,
            probe_suite=probe_suite,
            test_data=(test_features, test_sensitive_attrs),
            disallowed_attrs=purpose.disallowed_attrs,
            device=args.device,
        )

        logger.info(f"  Overall CVR: {cvr_results['overall_cvr']:.4f}")
        for attr, cvr in cvr_results["per_attr_cvr"].items():
            logger.info(f"    {attr}: CVR={cvr:.4f}")

        all_results[purpose.name] = {
            "task_accuracies": task_accuracies,
            "cvr": cvr_results["overall_cvr"],
            "per_attr_cvr": cvr_results["per_attr_cvr"],
            "training_time": train_time,
            "parameters": purpose_params,
        }

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("=== SUMMARY ===")
    logger.info(f"Total parameters (all models): {total_params:,}")
    logger.info(f"Total training time: {total_train_time:.1f}s")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "baseline": "separate_models",
        "timestamp": timestamp,
        "config": {
            "hidden_dims": args.hidden_dims,
            "repr_dim": args.repr_dim,
            "epochs": args.epochs,
            "lr_encoder": args.lr_encoder,
            "lr_auditor": args.lr_auditor,
            "lambda_adv": args.lambda_adv,
            "seed": args.seed,
        },
        "total_parameters": total_params,
        "total_training_time": total_train_time,
        "per_purpose_results": all_results,
    }

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

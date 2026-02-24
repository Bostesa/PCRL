"""Compliance Violation Rate (CVR) evaluation for PCRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pcrl.data.base import PCRLDataset, collate_pcrl_batch
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.utils.config import PCRLConfig, PurposeSpec


@dataclass
class ProbeConfig:
    """Configuration for a single probe architecture.

    Attributes:
        name: Name identifier for this probe config.
        probe_type: Type of probe ("linear" or "mlp").
        hidden_dims: Hidden dimensions for MLP probe.
        dropout: Dropout probability.
    """
    name: str
    probe_type: Literal["linear", "mlp"] = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [64])
    dropout: float = 0.1


@dataclass
class CVRResult:
    """Results from CVR evaluation.

    Attributes:
        purpose_name: Name of the evaluated purpose.
        overall_cvr: Overall compliance violation rate.
        per_attr_cvr: CVR per sensitive attribute.
        auditor_accuracies: Accuracy of trained auditors per attribute.
        baseline_accuracies: Baseline (majority class) accuracy per attribute.
        random_baseline: Expected random baseline accuracy per attribute.
    """
    purpose_name: str
    overall_cvr: float
    per_attr_cvr: dict[str, float]
    auditor_accuracies: dict[str, float]
    baseline_accuracies: dict[str, float]
    random_baseline: dict[str, float]


class LinearProbe(nn.Module):
    """Linear probe classifier."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MLPProbe(nn.Module):
    """MLP probe classifier with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64]

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_probe(
    probe: nn.Module,
    train_repr: torch.Tensor,
    train_labels: torch.Tensor,
    val_repr: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[nn.Module, list[float]]:
    """Train a probe classifier with early stopping.

    Args:
        probe: Probe network to train.
        train_repr: Training representations.
        train_labels: Training labels.
        val_repr: Validation representations (for early stopping).
        val_labels: Validation labels (for early stopping).
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        patience: Early stopping patience (epochs without improvement).
        device: Device to use.

    Returns:
        Tuple of (trained_probe, loss_history).
    """
    device = torch.device(device)
    probe = probe.to(device)

    # If no validation set provided, split training data
    if val_repr is None or val_labels is None:
        n = len(train_repr)
        indices = torch.randperm(n)
        split_idx = int(0.8 * n)
        val_repr = train_repr[indices[split_idx:]]
        val_labels = train_labels[indices[split_idx:]]
        train_repr = train_repr[indices[:split_idx]]
        train_labels = train_labels[indices[:split_idx]]

    train_repr = train_repr.to(device)
    train_labels = train_labels.to(device)
    val_repr = val_repr.to(device)
    val_labels = val_labels.to(device)

    # Create data loader
    train_dataset = TensorDataset(train_repr, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    loss_history = []

    probe.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_repr, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_repr)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # Validation for early stopping
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_repr)
            val_loss = criterion(val_logits, val_labels).item()
        probe.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    probe.eval()
    return probe, loss_history


class ProbeSuite:
    """Suite of probe classifiers with varying architectures.

    Trains multiple probe architectures to get a robust estimate of
    information content in representations.
    """

    DEFAULT_CONFIGS = [
        ProbeConfig("linear", probe_type="linear"),
        ProbeConfig("mlp_32", probe_type="mlp", hidden_dims=[32]),
        ProbeConfig("mlp_64", probe_type="mlp", hidden_dims=[64]),
        ProbeConfig("mlp_128", probe_type="mlp", hidden_dims=[128]),
        ProbeConfig("mlp_64_32", probe_type="mlp", hidden_dims=[64, 32]),
    ]

    def __init__(
        self,
        configs: list[ProbeConfig] | None = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        patience: int = 10,
        num_seeds: int = 3,
        device: str = "cpu",
    ) -> None:
        """Initialize probe suite.

        Args:
            configs: List of probe configurations. Uses defaults if None.
            epochs: Maximum training epochs per probe.
            lr: Learning rate.
            batch_size: Batch size.
            patience: Early stopping patience.
            num_seeds: Number of random seeds per config.
            device: Device to use.
        """
        self.configs = configs or self.DEFAULT_CONFIGS
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.num_seeds = num_seeds
        self.device = device

    def _create_probe(
        self,
        config: ProbeConfig,
        input_dim: int,
        num_classes: int,
    ) -> nn.Module:
        """Create a probe from config."""
        if config.probe_type == "linear":
            return LinearProbe(input_dim, num_classes)
        else:
            return MLPProbe(
                input_dim,
                num_classes,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )

    def train_all(
        self,
        train_repr: torch.Tensor,
        train_labels: torch.Tensor,
        val_repr: torch.Tensor | None = None,
        val_labels: torch.Tensor | None = None,
    ) -> list[tuple[nn.Module, ProbeConfig, float]]:
        """Train all probes in the suite.

        Args:
            train_repr: Training representations.
            train_labels: Training labels.
            val_repr: Validation representations.
            val_labels: Validation labels.

        Returns:
            List of (probe, config, final_loss) tuples.
        """
        input_dim = train_repr.shape[1]
        num_classes = int(train_labels.max().item()) + 1

        trained_probes = []

        for config in self.configs:
            for seed in range(self.num_seeds):
                torch.manual_seed(seed)
                probe = self._create_probe(config, input_dim, num_classes)

                trained_probe, loss_history = train_probe(
                    probe=probe,
                    train_repr=train_repr,
                    train_labels=train_labels,
                    val_repr=val_repr,
                    val_labels=val_labels,
                    epochs=self.epochs,
                    lr=self.lr,
                    batch_size=self.batch_size,
                    patience=self.patience,
                    device=self.device,
                )

                final_loss = loss_history[-1] if loss_history else float("inf")
                trained_probes.append((trained_probe, config, final_loss))

        return trained_probes

    def evaluate_all(
        self,
        trained_probes: list[tuple[nn.Module, ProbeConfig, float]],
        test_repr: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate all trained probes.

        Args:
            trained_probes: List of (probe, config, loss) tuples.
            test_repr: Test representations.
            test_labels: Test labels.

        Returns:
            Dictionary with accuracy metrics.
        """
        device = torch.device(self.device)
        test_repr = test_repr.to(device)
        test_labels = test_labels.to(device)

        accuracies = []
        per_config_accuracies: dict[str, list[float]] = {}

        for probe, config, _ in trained_probes:
            probe.eval()
            with torch.no_grad():
                logits = probe(test_repr)
                preds = logits.argmax(dim=-1)
                acc = (preds == test_labels).float().mean().item()

            accuracies.append(acc)

            if config.name not in per_config_accuracies:
                per_config_accuracies[config.name] = []
            per_config_accuracies[config.name].append(acc)

        return {
            "best_accuracy": max(accuracies),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "per_config_best": {
                name: max(accs) for name, accs in per_config_accuracies.items()
            },
        }


def compliance_violation_rate(
    encoder: nn.Module,
    purpose_idx: int,
    probe_suite: ProbeSuite,
    test_data: tuple[torch.Tensor, dict[str, torch.Tensor]],
    disallowed_attrs: list[str],
    train_data: tuple[torch.Tensor, dict[str, torch.Tensor]] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute Compliance Violation Rate for a purpose.

    Freezes encoder and trains fresh probe classifiers to measure how much
    disallowed sensitive information can be extracted from representations.

    CVR = best_accuracy - chance_level

    Args:
        encoder: Trained encoder (will be frozen).
        purpose_idx: Index of the purpose to evaluate.
        probe_suite: Suite of probe classifiers to use.
        test_data: Tuple of (features, {attr_name: labels}) for testing.
        disallowed_attrs: List of sensitive attribute names to evaluate.
        train_data: Optional separate training data. Uses test_data if None.
        device: Device to use.

    Returns:
        Dictionary containing:
            - overall_cvr: Mean CVR across all disallowed attributes.
            - per_attr_cvr: CVR for each attribute.
            - per_attr_accuracy: Best probe accuracy for each attribute.
            - chance_levels: Chance level for each attribute.
    """
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    encoder.eval()

    # Use test_data for training if no separate train_data
    if train_data is None:
        train_data = test_data

    # Extract representations
    train_features, train_attrs = train_data
    test_features, test_attrs = test_data

    with torch.no_grad():
        train_features = train_features.to(device_obj)
        test_features = test_features.to(device_obj)
        train_repr = encoder(train_features, purpose_idx).cpu()
        test_repr = encoder(test_features, purpose_idx).cpu()

    # Evaluate each disallowed attribute
    per_attr_cvr = {}
    per_attr_accuracy = {}
    chance_levels = {}

    for attr_name in disallowed_attrs:
        if attr_name not in test_attrs:
            continue

        train_labels = train_attrs[attr_name]
        test_labels = test_attrs[attr_name]

        # Compute chance level (uniform random baseline)
        num_classes = int(test_labels.max().item()) + 1
        chance_level = 1.0 / num_classes
        chance_levels[attr_name] = chance_level

        # Train all probes in suite
        trained_probes = probe_suite.train_all(
            train_repr=train_repr,
            train_labels=train_labels,
        )

        # Evaluate and get best accuracy
        eval_results = probe_suite.evaluate_all(
            trained_probes=trained_probes,
            test_repr=test_repr,
            test_labels=test_labels,
        )

        best_accuracy = eval_results["best_accuracy"]
        per_attr_accuracy[attr_name] = best_accuracy

        # Compute CVR
        cvr = max(0.0, best_accuracy - chance_level)
        per_attr_cvr[attr_name] = cvr

    # Overall CVR is mean across attributes
    overall_cvr = (
        sum(per_attr_cvr.values()) / len(per_attr_cvr) if per_attr_cvr else 0.0
    )

    return {
        "overall_cvr": overall_cvr,
        "per_attr_cvr": per_attr_cvr,
        "per_attr_accuracy": per_attr_accuracy,
        "chance_levels": chance_levels,
    }


class CVRMetric:
    """Compliance Violation Rate metric.

    CVR measures how well sensitive attributes can be predicted from
    learned representations. Lower CVR indicates better privacy/fairness.

    CVR = (auditor_accuracy - random_baseline) / (1 - random_baseline)

    A CVR of 0 means the auditor does no better than random guessing.
    A CVR of 1 means perfect prediction of sensitive attributes.
    """

    def __init__(
        self,
        auditor_hidden_dims: list[int] | None = None,
        auditor_epochs: int = 50,
        auditor_lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        """Initialize CVR metric.

        Args:
            auditor_hidden_dims: Hidden dimensions for auditor networks.
            auditor_epochs: Number of epochs to train auditors.
            auditor_lr: Learning rate for auditor training.
            batch_size: Batch size for auditor training.
            device: Device to use.
        """
        self.auditor_hidden_dims = auditor_hidden_dims or [64, 32]
        self.auditor_epochs = auditor_epochs
        self.auditor_lr = auditor_lr
        self.batch_size = batch_size
        self.device = torch.device(device)

    def compute(
        self,
        encoder: PurposeConditionedEncoder,
        dataset: PCRLDataset,
        purpose: PurposeSpec,
        purpose_idx: int,
    ) -> CVRResult:
        """Compute CVR for a specific purpose.

        Trains fresh auditors on the learned representations and measures
        how well they can predict disallowed sensitive attributes.

        Args:
            encoder: Trained encoder.
            dataset: Dataset to evaluate on.
            purpose: Purpose specification.
            purpose_idx: Index of the purpose.

        Returns:
            CVR evaluation results.
        """
        encoder.eval()
        encoder.to(self.device)

        # Extract representations
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_pcrl_batch,
        )

        all_repr = []
        all_attrs: dict[str, list[torch.Tensor]] = {
            attr: [] for attr in purpose.disallowed_attrs
        }

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                repr = encoder(features, purpose_idx)
                all_repr.append(repr.cpu())

                for attr in purpose.disallowed_attrs:
                    if attr in batch["sensitive_attrs"]:
                        all_attrs[attr].append(batch["sensitive_attrs"][attr])

        representations = torch.cat(all_repr, dim=0)
        attr_labels = {
            attr: torch.cat(labels, dim=0)
            for attr, labels in all_attrs.items()
            if labels
        }

        # Compute per-attribute CVR
        per_attr_cvr = {}
        auditor_accuracies = {}
        baseline_accuracies = {}
        random_baselines = {}

        for attr in purpose.disallowed_attrs:
            if attr not in attr_labels:
                continue

            labels = attr_labels[attr]
            num_classes = dataset.info.sensitive_attrs.get(attr, 2)

            # Train auditor
            accuracy = self._train_and_evaluate_auditor(
                representations, labels, num_classes
            )
            auditor_accuracies[attr] = accuracy

            # Compute baselines
            majority_acc = self._compute_majority_baseline(labels)
            baseline_accuracies[attr] = majority_acc

            random_acc = 1.0 / num_classes
            random_baselines[attr] = random_acc

            # Compute CVR
            cvr = (accuracy - random_acc) / (1 - random_acc) if (1 - random_acc) > 0 else 0.0
            cvr = max(0.0, min(1.0, cvr))  # Clamp to [0, 1]
            per_attr_cvr[attr] = cvr

        # Overall CVR is average across attributes
        overall_cvr = sum(per_attr_cvr.values()) / len(per_attr_cvr) if per_attr_cvr else 0.0

        return CVRResult(
            purpose_name=purpose.name,
            overall_cvr=overall_cvr,
            per_attr_cvr=per_attr_cvr,
            auditor_accuracies=auditor_accuracies,
            baseline_accuracies=baseline_accuracies,
            random_baseline=random_baselines,
        )

    def _train_and_evaluate_auditor(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> float:
        """Train an auditor and return its accuracy.

        Args:
            representations: Encoded representations.
            labels: Sensitive attribute labels.
            num_classes: Number of classes.

        Returns:
            Auditor accuracy.
        """
        # Split into train/test (80/20)
        n = len(representations)
        indices = torch.randperm(n)
        split_idx = int(0.8 * n)

        train_repr = representations[indices[:split_idx]].to(self.device)
        train_labels = labels[indices[:split_idx]].to(self.device)
        test_repr = representations[indices[split_idx:]].to(self.device)
        test_labels = labels[indices[split_idx:]].to(self.device)

        # Create and train auditor using MLPProbe
        auditor = MLPProbe(
            input_dim=representations.shape[1],
            num_classes=num_classes,
            hidden_dims=self.auditor_hidden_dims,
        ).to(self.device)

        optimizer = torch.optim.Adam(auditor.parameters(), lr=self.auditor_lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        auditor.train()
        for _ in range(self.auditor_epochs):
            # Mini-batch training
            perm = torch.randperm(len(train_repr))
            for i in range(0, len(train_repr), self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                batch_repr = train_repr[batch_idx]
                batch_labels = train_labels[batch_idx]

                optimizer.zero_grad()
                logits = auditor(batch_repr)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

        # Evaluate
        auditor.eval()
        with torch.no_grad():
            test_logits = auditor(test_repr)
            predictions = test_logits.argmax(dim=-1)
            accuracy = (predictions == test_labels).float().mean().item()

        return accuracy

    def _compute_majority_baseline(self, labels: torch.Tensor) -> float:
        """Compute majority class baseline accuracy."""
        unique, counts = labels.unique(return_counts=True)
        majority_count = counts.max().item()
        return majority_count / len(labels)


def evaluate_all_purposes(
    encoder: PurposeConditionedEncoder,
    config: PCRLConfig,
    test_dataset: PCRLDataset,
    device: str = "cpu",
) -> dict[str, CVRResult]:
    """Evaluate CVR for all purposes.

    Args:
        encoder: Trained encoder.
        config: Model configuration.
        test_dataset: Test dataset.
        device: Device to use.

    Returns:
        Dictionary mapping purpose name to CVR results.
    """
    cvr_metric = CVRMetric(device=device)
    results = {}

    for idx, purpose in enumerate(config.purposes):
        result = cvr_metric.compute(encoder, test_dataset, purpose, idx)
        results[purpose.name] = result

    return results

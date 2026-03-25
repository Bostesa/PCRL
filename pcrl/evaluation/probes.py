"""Post-hoc auditor probes for PCRL evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pcrl.data.base import PCRLDataset, collate_pcrl_batch
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.utils.config import PurposeSpec


@dataclass
class ProbeResult:
    """Results from a single probe evaluation.

    Attributes:
        attribute_name: Name of the probed attribute.
        accuracy: Classification accuracy.
        auc_roc: Area under ROC curve (for binary).
        f1_score: F1 score.
        train_loss_history: Training loss over epochs.
    """
    attribute_name: str
    accuracy: float
    auc_roc: float | None = None
    f1_score: float | None = None
    train_loss_history: list[float] | None = None


@dataclass
class ProbeResults:
    """Aggregated results from all probes.

    Attributes:
        purpose_name: Name of the purpose being evaluated.
        purpose_idx: Index of the purpose.
        task_probes: Probe results for allowed tasks.
        sensitive_probes: Probe results for disallowed attributes.
    """
    purpose_name: str
    purpose_idx: int
    task_probes: dict[str, ProbeResult]
    sensitive_probes: dict[str, ProbeResult]


class LinearProbe(nn.Module):
    """Linear probe for evaluating representation quality.

    A simple linear classifier trained on frozen representations
    to assess how much information they contain about specific attributes.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        """Initialize linear probe.

        Args:
            input_dim: Dimension of input representations.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class MLPProbe(nn.Module):
    """MLP probe for evaluating representation quality.

    A small MLP trained on frozen representations to assess
    information content with more modeling capacity than linear probes.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize MLP probe.

        Args:
            input_dim: Dimension of input representations.
            num_classes: Number of output classes.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64]

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
        """Forward pass."""
        return self.network(x)


class PostHocProber:
    """Post-hoc probing for representation evaluation.

    Trains probe classifiers on frozen representations to measure
    how much information they contain about various attributes.
    """

    def __init__(
        self,
        probe_type: Literal["linear", "mlp"] = "mlp",
        hidden_dims: list[int] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int | None = 5,
        device: str = "cpu",
    ) -> None:
        """Initialize post-hoc prober.

        Args:
            probe_type: Type of probe ("linear" or "mlp").
            hidden_dims: Hidden dimensions for MLP probe.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            early_stopping_patience: Patience for early stopping.
            device: Device to use.
        """
        self.probe_type = probe_type
        self.hidden_dims = hidden_dims or [64]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)

    def probe_representations(
        self,
        encoder: PurposeConditionedEncoder,
        train_dataset: PCRLDataset,
        test_dataset: PCRLDataset,
        purpose_idx: int,
        purpose_name: str,
        task_names: list[str] | None = None,
        sensitive_names: list[str] | None = None,
    ) -> ProbeResults:
        """Run probes on representations for a specific purpose.

        Args:
            encoder: Trained encoder.
            train_dataset: Training dataset for probe training.
            test_dataset: Test dataset for probe evaluation.
            purpose_idx: Index of the purpose.
            purpose_name: Name of the purpose.
            task_names: Task names to probe (default: all).
            sensitive_names: Sensitive attribute names to probe (default: all).

        Returns:
            Probe results for all probed attributes.
        """
        encoder.eval()
        encoder.to(self.device)

        # Extract representations
        train_repr, train_labels = self._extract_representations(
            encoder, train_dataset, purpose_idx
        )
        test_repr, test_labels = self._extract_representations(
            encoder, test_dataset, purpose_idx
        )

        # Get attribute names to probe
        if task_names is None:
            task_names = list(train_dataset.info.task_labels.keys())
        if sensitive_names is None:
            sensitive_names = list(train_dataset.info.sensitive_attrs.keys())

        # Probe tasks
        task_probes = {}
        for task_name in task_names:
            if task_name in train_labels["task_labels"]:
                num_classes = train_dataset.info.task_labels[task_name]
                result = self._train_and_evaluate_probe(
                    train_repr,
                    train_labels["task_labels"][task_name],
                    test_repr,
                    test_labels["task_labels"][task_name],
                    num_classes,
                    task_name,
                )
                task_probes[task_name] = result

        # Probe sensitive attributes
        sensitive_probes = {}
        for attr_name in sensitive_names:
            if attr_name in train_labels["sensitive_attrs"]:
                num_classes = train_dataset.info.sensitive_attrs[attr_name]
                result = self._train_and_evaluate_probe(
                    train_repr,
                    train_labels["sensitive_attrs"][attr_name],
                    test_repr,
                    test_labels["sensitive_attrs"][attr_name],
                    num_classes,
                    attr_name,
                )
                sensitive_probes[attr_name] = result

        return ProbeResults(
            purpose_name=purpose_name,
            purpose_idx=purpose_idx,
            task_probes=task_probes,
            sensitive_probes=sensitive_probes,
        )

    def _extract_representations(
        self,
        encoder: PurposeConditionedEncoder,
        dataset: PCRLDataset,
        purpose_idx: int,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Extract representations from dataset.

        Returns:
            Tuple of (representations, labels_dict).
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_pcrl_batch,
        )

        all_repr = []
        all_task_labels: dict[str, list[torch.Tensor]] = {}
        all_sensitive_labels: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                repr = encoder(features, purpose_idx)
                all_repr.append(repr.cpu())

                for task_name, labels in batch["task_labels"].items():
                    if task_name not in all_task_labels:
                        all_task_labels[task_name] = []
                    all_task_labels[task_name].append(labels)

                for attr_name, labels in batch["sensitive_attrs"].items():
                    if attr_name not in all_sensitive_labels:
                        all_sensitive_labels[attr_name] = []
                    all_sensitive_labels[attr_name].append(labels)

        representations = torch.cat(all_repr, dim=0)

        labels = {
            "task_labels": {
                k: torch.cat(v, dim=0) for k, v in all_task_labels.items()
            },
            "sensitive_attrs": {
                k: torch.cat(v, dim=0) for k, v in all_sensitive_labels.items()
            },
        }

        return representations, labels

    def _train_and_evaluate_probe(
        self,
        train_repr: torch.Tensor,
        train_labels: torch.Tensor,
        test_repr: torch.Tensor,
        test_labels: torch.Tensor,
        num_classes: int,
        attribute_name: str,
    ) -> ProbeResult:
        """Train and evaluate a single probe.

        Args:
            train_repr: Training representations.
            train_labels: Training labels.
            test_repr: Test representations.
            test_labels: Test labels.
            num_classes: Number of classes.
            attribute_name: Name of the attribute being probed.

        Returns:
            Probe evaluation result.
        """
        input_dim = train_repr.shape[1]

        # Create probe
        if self.probe_type == "linear":
            probe = LinearProbe(input_dim, num_classes)
        else:
            probe = MLPProbe(input_dim, num_classes, self.hidden_dims)

        probe.to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(train_repr, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Training
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        patience_counter = 0
        loss_history = []

        probe.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for batch_repr, batch_labels in train_loader:
                batch_repr = batch_repr.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = probe(batch_repr)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)

            # Early stopping
            if self.early_stopping_patience is not None:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

        # Evaluation
        probe.eval()
        test_repr = test_repr.to(self.device)
        test_labels = test_labels.to(self.device)

        with torch.no_grad():
            logits = probe(test_repr)
            predictions = logits.argmax(dim=-1)

            accuracy = (predictions == test_labels).float().mean().item()

            # Compute AUC-ROC for binary classification
            auc_roc = None
            if num_classes == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    auc_roc = roc_auc_score(test_labels.cpu().numpy(), probs)
                except ImportError:
                    pass

            # Compute F1 score
            f1_score = None
            try:
                from sklearn.metrics import f1_score as sklearn_f1
                f1_score = sklearn_f1(
                    test_labels.cpu().numpy(),
                    predictions.cpu().numpy(),
                    average="macro" if num_classes > 2 else "binary",
                )
            except ImportError:
                pass

        return ProbeResult(
            attribute_name=attribute_name,
            accuracy=accuracy,
            auc_roc=auc_roc,
            f1_score=f1_score,
            train_loss_history=loss_history,
        )


def run_full_probe_evaluation(
    encoder: PurposeConditionedEncoder,
    train_dataset: PCRLDataset,
    test_dataset: PCRLDataset,
    purposes: list[tuple[int, str]],
    probe_type: Literal["linear", "mlp"] = "mlp",
    device: str = "cpu",
) -> dict[str, ProbeResults]:
    """Run probe evaluation for multiple purposes.

    Args:
        encoder: Trained encoder.
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        purposes: List of (purpose_idx, purpose_name) tuples.
        probe_type: Type of probe to use.
        device: Device to use.

    Returns:
        Dictionary mapping purpose name to probe results.
    """
    prober = PostHocProber(probe_type=probe_type, device=device)
    results = {}

    for purpose_idx, purpose_name in purposes:
        result = prober.probe_representations(
            encoder=encoder,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            purpose_idx=purpose_idx,
            purpose_name=purpose_name,
        )
        results[purpose_name] = result

    return results


def cosine_similarity_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute mean cosine similarity between two batches of representations.

    Args:
        x: First batch of representations (N, D).
        y: Second batch of representations (N, D).

    Returns:
        Mean cosine similarity (scalar).
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return (x_norm * y_norm).sum(dim=-1).mean()


def centered_kernel_alignment(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute Centered Kernel Alignment (CKA) between representations.

    CKA is a similarity measure that is invariant to orthogonal transformations
    and isotropic scaling, making it useful for comparing neural representations.

    Args:
        x: First batch of representations (N, D1).
        y: Second batch of representations (N, D2).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center the representations
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Compute Gram matrices (linear kernel)
    k_x = x @ x.T
    k_y = y @ y.T

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    def hsic(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
        n = k1.shape[0]
        # Center the kernel matrices
        h = torch.eye(n, device=k1.device) - torch.ones(n, n, device=k1.device) / n
        k1_c = h @ k1 @ h
        k2_c = h @ k2 @ h
        return (k1_c * k2_c).sum() / ((n - 1) ** 2)

    hsic_xy = hsic(k_x, k_y)
    hsic_xx = hsic(k_x, k_x)
    hsic_yy = hsic(k_y, k_y)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    cka = hsic_xy / denom
    return cka.item()


@dataclass
class RepresentationSimilarity:
    """Similarity between representations for different purposes.

    Attributes:
        purpose_pair: Tuple of (purpose1_name, purpose2_name).
        cka: Centered Kernel Alignment score.
        mean_cosine: Mean cosine similarity.
    """
    purpose_pair: tuple[str, str]
    cka: float
    mean_cosine: float


@dataclass
class TaskPerformance:
    """Task performance metrics.

    Attributes:
        task_name: Name of the task.
        purpose_name: Name of the purpose.
        accuracy: Classification accuracy (for classification tasks).
        mse: Mean squared error (for regression tasks).
        f1_score: F1 score (for classification tasks).
    """
    task_name: str
    purpose_name: str
    accuracy: float | None = None
    mse: float | None = None
    f1_score: float | None = None


@dataclass
class PCRLEvaluation:
    """Comprehensive PCRL evaluation results.

    Attributes:
        task_performance: Task performance per purpose.
        sensitive_probe_accuracy: Sensitive attribute probe accuracy per purpose.
        representation_similarity: Pairwise representation similarity.
        cvr_per_purpose: CVR per purpose (if computed).
    """
    task_performance: dict[str, list[TaskPerformance]]
    sensitive_probe_accuracy: dict[str, dict[str, float]]
    representation_similarity: list[RepresentationSimilarity]
    cvr_per_purpose: dict[str, float] = field(default_factory=dict)


def evaluate_pcrl(
    encoder: nn.Module,
    purposes: list[PurposeSpec],
    test_data: tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]],
    task_heads: dict[str, nn.Module] | None = None,
    train_data: tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None,
    probe_type: Literal["linear", "mlp"] = "mlp",
    probe_epochs: int = 50,
    batch_size: int = 256,
    device: str = "cpu",
) -> PCRLEvaluation:
    """Comprehensive PCRL evaluation.

    Evaluates:
    1. Task performance for each purpose (if task_heads provided)
    2. Sensitive attribute probe accuracy for each purpose
    3. Representation similarity between purposes (CKA, cosine)
    4. Verifies representations differ across purposes

    Args:
        encoder: Trained encoder.
        purposes: List of purpose specifications.
        test_data: Tuple of (features, task_labels_dict, sensitive_attrs_dict).
        task_heads: Optional task prediction heads for performance evaluation.
        train_data: Optional training data for probe training.
        probe_type: Type of probe for sensitive attribute evaluation.
        probe_epochs: Number of epochs for probe training.
        batch_size: Batch size for evaluation.
        device: Device to use.

    Returns:
        Comprehensive evaluation results.
    """
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    encoder.eval()

    test_features, test_task_labels, test_sensitive_attrs = test_data

    # Use test data for probe training if no train data provided
    if train_data is None:
        train_data = test_data

    train_features, train_task_labels, train_sensitive_attrs = train_data

    # Extract representations for all purposes
    purpose_representations: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        test_features_device = test_features.to(device_obj)
        for idx, purpose in enumerate(purposes):
            repr = encoder(test_features_device, idx).cpu()
            purpose_representations[purpose.name] = repr

    # 1. Evaluate task performance (if task_heads provided)
    task_performance: dict[str, list[TaskPerformance]] = {}

    if task_heads is not None:
        for idx, purpose in enumerate(purposes):
            task_performance[purpose.name] = []

            for task_name in purpose.allowed_tasks:
                if task_name not in task_heads or task_name not in test_task_labels:
                    continue

                head = task_heads[task_name].to(device_obj)
                head.eval()

                repr = purpose_representations[purpose.name].to(device_obj)
                labels = test_task_labels[task_name].to(device_obj)

                with torch.no_grad():
                    preds = head(repr)

                    # Compute metrics based on task type
                    if preds.dim() > 1 and preds.shape[-1] > 1:
                        # Classification
                        pred_classes = preds.argmax(dim=-1)
                        accuracy = (pred_classes == labels).float().mean().item()

                        # F1 score
                        f1 = None
                        try:
                            from sklearn.metrics import f1_score as sklearn_f1
                            num_classes = preds.shape[-1]
                            f1 = sklearn_f1(
                                labels.cpu().numpy(),
                                pred_classes.cpu().numpy(),
                                average="macro" if num_classes > 2 else "binary",
                            )
                        except ImportError:
                            pass

                        task_performance[purpose.name].append(
                            TaskPerformance(
                                task_name=task_name,
                                purpose_name=purpose.name,
                                accuracy=accuracy,
                                f1_score=f1,
                            )
                        )
                    else:
                        # Regression
                        preds = preds.squeeze(-1) if preds.dim() > 1 else preds
                        mse = F.mse_loss(preds, labels.float()).item()

                        task_performance[purpose.name].append(
                            TaskPerformance(
                                task_name=task_name,
                                purpose_name=purpose.name,
                                mse=mse,
                            )
                        )

    # 2. Evaluate sensitive attribute probe accuracy
    sensitive_probe_accuracy: dict[str, dict[str, float]] = {}

    prober = PostHocProber(
        probe_type=probe_type,
        epochs=probe_epochs,
        batch_size=batch_size,
        device=device,
    )

    # Extract train representations
    train_purpose_representations: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        train_features_device = train_features.to(device_obj)
        for idx, purpose in enumerate(purposes):
            repr = encoder(train_features_device, idx).cpu()
            train_purpose_representations[purpose.name] = repr

    for idx, purpose in enumerate(purposes):
        sensitive_probe_accuracy[purpose.name] = {}

        train_repr = train_purpose_representations[purpose.name]
        test_repr = purpose_representations[purpose.name]

        # Probe disallowed attributes
        for attr_name in purpose.disallowed_attrs:
            if attr_name not in test_sensitive_attrs:
                continue

            train_labels = train_sensitive_attrs[attr_name]
            test_labels = test_sensitive_attrs[attr_name]
            num_classes = int(test_labels.max().item()) + 1

            result = prober._train_and_evaluate_probe(
                train_repr=train_repr,
                train_labels=train_labels,
                test_repr=test_repr,
                test_labels=test_labels,
                num_classes=num_classes,
                attribute_name=attr_name,
            )

            sensitive_probe_accuracy[purpose.name][attr_name] = result.accuracy

    # 3. Compute representation similarity between purposes
    representation_similarity: list[RepresentationSimilarity] = []

    purpose_names = [p.name for p in purposes]
    for i in range(len(purpose_names)):
        for j in range(i + 1, len(purpose_names)):
            name_i, name_j = purpose_names[i], purpose_names[j]
            repr_i = purpose_representations[name_i]
            repr_j = purpose_representations[name_j]

            # CKA similarity
            cka = centered_kernel_alignment(repr_i, repr_j)

            # Mean cosine similarity
            mean_cos = cosine_similarity_batch(repr_i, repr_j).item()

            representation_similarity.append(
                RepresentationSimilarity(
                    purpose_pair=(name_i, name_j),
                    cka=cka,
                    mean_cosine=mean_cos,
                )
            )

    return PCRLEvaluation(
        task_performance=task_performance,
        sensitive_probe_accuracy=sensitive_probe_accuracy,
        representation_similarity=representation_similarity,
    )


def verify_purpose_differentiation(
    encoder: nn.Module,
    purposes: list[PurposeSpec],
    test_features: torch.Tensor,
    device: str = "cpu",
    cka_threshold: float = 0.95,
) -> dict[str, Any]:
    """Verify that representations differ across purposes.

    A key property of PCRL is that different purposes should produce
    meaningfully different representations.

    Args:
        encoder: Trained encoder.
        purposes: List of purpose specifications.
        test_features: Test features tensor.
        device: Device to use.
        cka_threshold: CKA threshold above which purposes are considered
            too similar (default 0.95).

    Returns:
        Dictionary containing:
            - pairwise_cka: CKA scores between all purpose pairs.
            - pairwise_cosine: Mean cosine similarity between pairs.
            - similar_pairs: List of purpose pairs that are too similar.
            - representations_differ: Whether representations adequately differ.
    """
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    encoder.eval()

    # Extract representations for all purposes
    representations: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        test_features = test_features.to(device_obj)
        for idx, purpose in enumerate(purposes):
            repr = encoder(test_features, idx).cpu()
            representations[purpose.name] = repr

    # Compute pairwise similarities
    pairwise_cka: dict[tuple[str, str], float] = {}
    pairwise_cosine: dict[tuple[str, str], float] = {}
    similar_pairs: list[tuple[str, str]] = []

    purpose_names = [p.name for p in purposes]
    for i in range(len(purpose_names)):
        for j in range(i + 1, len(purpose_names)):
            name_i, name_j = purpose_names[i], purpose_names[j]
            repr_i = representations[name_i]
            repr_j = representations[name_j]

            cka = centered_kernel_alignment(repr_i, repr_j)
            cos = cosine_similarity_batch(repr_i, repr_j).item()

            pair = (name_i, name_j)
            pairwise_cka[pair] = cka
            pairwise_cosine[pair] = cos

            if cka > cka_threshold:
                similar_pairs.append(pair)

    representations_differ = len(similar_pairs) == 0

    return {
        "pairwise_cka": pairwise_cka,
        "pairwise_cosine": pairwise_cosine,
        "similar_pairs": similar_pairs,
        "representations_differ": representations_differ,
    }

"""Adversarial auditors for PCRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training.

    During forward pass, acts as identity.
    During backward pass, reverses gradient and scales by lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Forward pass (identity)."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass (reverse and scale gradient)."""
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal.

    Args:
        x: Input tensor.
        lambda_: Gradient scaling factor.

    Returns:
        Same tensor with reversed gradients during backprop.
    """
    return GradientReversalFunction.apply(x, lambda_)


class Auditor(nn.Module):
    """Adversarial auditor.

    MLP that tries to predict disallowed attributes from representation.
    Used adversarially to encourage the encoder to remove sensitive information.
    """

    def __init__(
        self,
        repr_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gradient_reversal: bool = False,
    ) -> None:
        """Initialize the auditor.

        Args:
            repr_dim: Dimension of input representation.
            output_dim: Number of classes for the sensitive attribute.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            use_gradient_reversal: Whether to apply gradient reversal for end-to-end training.
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gradient_reversal = use_gradient_reversal

        # Build MLP
        layers: list[nn.Module] = []

        # Input layer
        layers.extend([
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        h: torch.Tensor,
        lambda_: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Input representation of shape (batch_size, repr_dim).
            lambda_: Gradient reversal scaling factor (only used if use_gradient_reversal=True).

        Returns:
            Logits of shape (batch_size, output_dim).
        """
        if self.use_gradient_reversal:
            h = gradient_reversal(h, lambda_)
        return self.network(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels.

        Args:
            h: Input representation.

        Returns:
            Predicted class labels of shape (batch_size,).
        """
        with torch.no_grad():
            logits = self.network(h)  # Skip gradient reversal for prediction
        return logits.argmax(dim=-1)

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """Get predicted probabilities.

        Args:
            h: Input representation.

        Returns:
            Predicted probabilities of shape (batch_size, output_dim).
        """
        with torch.no_grad():
            logits = self.network(h)
        return torch.softmax(logits, dim=-1)

    def get_accuracy(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute accuracy on given data.

        Args:
            h: Input representation.
            labels: True labels.

        Returns:
            Accuracy as a float.
        """
        predictions = self.predict(h)
        return (predictions == labels).float().mean().item()


@dataclass
class AuditorConfig:
    """Configuration for a single auditor in the pool."""
    hidden_dim: int
    num_layers: int
    seed: int


class AuditorPool(nn.Module):
    """Pool of auditors with varying architectures.

    Manages multiple auditors with different:
    - architectures (varying depth/width)
    - random seeds

    This is important for robust CVR measurement, as a single auditor
    might not fully exploit information in the representation.
    """

    def __init__(
        self,
        repr_dim: int,
        output_dim: int,
        pool_size: int = 5,
        hidden_dims: list[int] | None = None,
        num_layers_options: list[int] | None = None,
        base_seed: int = 42,
        dropout: float = 0.1,
        use_gradient_reversal: bool = False,
    ) -> None:
        """Initialize the auditor pool.

        Args:
            repr_dim: Dimension of input representation.
            output_dim: Number of classes for the sensitive attribute.
            pool_size: Number of auditors in the pool.
            hidden_dims: List of hidden dimensions to sample from. Defaults to [32, 64, 128].
            num_layers_options: List of layer counts to sample from. Defaults to [1, 2, 3].
            base_seed: Base random seed for reproducibility.
            dropout: Dropout probability for all auditors.
            use_gradient_reversal: Whether to apply gradient reversal.
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.use_gradient_reversal = use_gradient_reversal

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        if num_layers_options is None:
            num_layers_options = [1, 2, 3]

        # Generate diverse configurations
        self.configs: list[AuditorConfig] = []
        torch.manual_seed(base_seed)

        for i in range(pool_size):
            config = AuditorConfig(
                hidden_dim=hidden_dims[i % len(hidden_dims)],
                num_layers=num_layers_options[i % len(num_layers_options)],
                seed=base_seed + i,
            )
            self.configs.append(config)

        # Create auditors
        self.auditors = nn.ModuleList()
        for config in self.configs:
            torch.manual_seed(config.seed)
            auditor = Auditor(
                repr_dim=repr_dim,
                output_dim=output_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=dropout,
                use_gradient_reversal=use_gradient_reversal,
            )
            self.auditors.append(auditor)

    def forward(
        self,
        h: torch.Tensor,
        lambda_: float = 1.0,
    ) -> list[torch.Tensor]:
        """Forward pass through all auditors.

        Args:
            h: Input representation of shape (batch_size, repr_dim).
            lambda_: Gradient reversal scaling factor.

        Returns:
            List of logits from each auditor, each of shape (batch_size, output_dim).
        """
        return [auditor(h, lambda_) for auditor in self.auditors]

    def forward_mean(
        self,
        h: torch.Tensor,
        lambda_: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with averaged predictions.

        Args:
            h: Input representation.
            lambda_: Gradient reversal scaling factor.

        Returns:
            Averaged logits of shape (batch_size, output_dim).
        """
        all_logits = self.forward(h, lambda_)
        return torch.stack(all_logits).mean(dim=0)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Get ensemble prediction via majority voting.

        Args:
            h: Input representation.

        Returns:
            Predicted class labels of shape (batch_size,).
        """
        all_preds = torch.stack([auditor.predict(h) for auditor in self.auditors])
        # Majority voting
        return torch.mode(all_preds, dim=0).values

    def get_individual_predictions(self, h: torch.Tensor) -> list[torch.Tensor]:
        """Get predictions from each auditor individually.

        Args:
            h: Input representation.

        Returns:
            List of predictions from each auditor.
        """
        return [auditor.predict(h) for auditor in self.auditors]

    def get_individual_accuracies(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[float]:
        """Get accuracy of each auditor.

        Args:
            h: Input representation.
            labels: True labels.

        Returns:
            List of accuracies for each auditor.
        """
        return [auditor.get_accuracy(h, labels) for auditor in self.auditors]

    def get_best_accuracy(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Get the best accuracy among all auditors.

        This is used for CVR measurement - the best auditor represents
        the upper bound on information leakage.

        Args:
            h: Input representation.
            labels: True labels.

        Returns:
            Best accuracy among all auditors.
        """
        accuracies = self.get_individual_accuracies(h, labels)
        return max(accuracies)

    def get_mean_accuracy(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Get the mean accuracy across all auditors.

        Args:
            h: Input representation.
            labels: True labels.

        Returns:
            Mean accuracy across all auditors.
        """
        accuracies = self.get_individual_accuracies(h, labels)
        return sum(accuracies) / len(accuracies)

    def __iter__(self) -> Iterator[Auditor]:
        """Iterate over auditors in the pool."""
        return iter(self.auditors)

    def __len__(self) -> int:
        """Return the number of auditors."""
        return len(self.auditors)

    def __getitem__(self, idx: int) -> Auditor:
        """Get a specific auditor by index."""
        return self.auditors[idx]


class MultiAttributeAuditor(nn.Module):
    """Auditor for multiple sensitive attributes.

    Manages separate auditors (or auditor pools) for each sensitive attribute.
    """

    def __init__(
        self,
        repr_dim: int,
        attr_output_dims: dict[str, int],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gradient_reversal: bool = False,
        use_pool: bool = False,
        pool_size: int = 5,
    ) -> None:
        """Initialize multi-attribute auditor.

        Args:
            repr_dim: Dimension of input representation.
            attr_output_dims: Dictionary mapping attribute name to number of classes.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            use_gradient_reversal: Whether to use gradient reversal.
            use_pool: Whether to use auditor pools instead of single auditors.
            pool_size: Size of auditor pool (if use_pool=True).
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.attr_names = list(attr_output_dims.keys())
        self.use_gradient_reversal = use_gradient_reversal
        self.use_pool = use_pool

        self.auditors = nn.ModuleDict()
        for name, output_dim in attr_output_dims.items():
            if use_pool:
                self.auditors[name] = AuditorPool(
                    repr_dim=repr_dim,
                    output_dim=output_dim,
                    pool_size=pool_size,
                    dropout=dropout,
                    use_gradient_reversal=use_gradient_reversal,
                )
            else:
                self.auditors[name] = Auditor(
                    repr_dim=repr_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_gradient_reversal=use_gradient_reversal,
                )

    def forward(
        self,
        h: torch.Tensor,
        attr_names: list[str] | None = None,
        lambda_: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for specified attributes.

        Args:
            h: Input representation.
            attr_names: List of attribute names to audit. If None, audits all.
            lambda_: Gradient reversal scaling factor.

        Returns:
            Dictionary mapping attribute name to logits.
            For pools, returns the mean logits.
        """
        if attr_names is None:
            attr_names = self.attr_names

        outputs = {}
        for name in attr_names:
            if name in self.auditors:
                auditor = self.auditors[name]
                if isinstance(auditor, AuditorPool):
                    outputs[name] = auditor.forward_mean(h, lambda_)
                else:
                    outputs[name] = auditor(h, lambda_)

        return outputs

    def predict(
        self,
        h: torch.Tensor,
        attr_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get predictions for specified attributes.

        Args:
            h: Input representation.
            attr_names: List of attribute names.

        Returns:
            Dictionary mapping attribute name to predicted labels.
        """
        if attr_names is None:
            attr_names = self.attr_names

        predictions = {}
        for name in attr_names:
            if name in self.auditors:
                predictions[name] = self.auditors[name].predict(h)

        return predictions

    def get_accuracies(
        self,
        h: torch.Tensor,
        labels: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Get accuracy for each attribute.

        Args:
            h: Input representation.
            labels: Dictionary mapping attribute name to labels.

        Returns:
            Dictionary mapping attribute name to accuracy.
        """
        accuracies = {}
        for name in self.attr_names:
            if name in labels and name in self.auditors:
                auditor = self.auditors[name]
                if isinstance(auditor, AuditorPool):
                    accuracies[name] = auditor.get_best_accuracy(h, labels[name])
                else:
                    accuracies[name] = auditor.get_accuracy(h, labels[name])
        return accuracies


class PurposeAuditors(nn.Module):
    """Auditors organized by purpose.

    Each purpose has auditors for its disallowed attributes.
    """

    def __init__(
        self,
        repr_dim: int,
        purpose_attr_specs: dict[str, dict[str, int]],
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gradient_reversal: bool = False,
        use_pool: bool = False,
        pool_size: int = 5,
    ) -> None:
        """Initialize purpose-specific auditors.

        Args:
            repr_dim: Dimension of input representation.
            purpose_attr_specs: Nested dict: purpose_name -> attr_name -> output_dim.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            use_gradient_reversal: Whether to use gradient reversal.
            use_pool: Whether to use auditor pools.
            pool_size: Size of auditor pools.
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.purpose_names = list(purpose_attr_specs.keys())
        self.use_gradient_reversal = use_gradient_reversal

        self.purpose_auditors = nn.ModuleDict()
        for purpose_name, attr_specs in purpose_attr_specs.items():
            if attr_specs:
                self.purpose_auditors[purpose_name] = MultiAttributeAuditor(
                    repr_dim=repr_dim,
                    attr_output_dims=attr_specs,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_gradient_reversal=use_gradient_reversal,
                    use_pool=use_pool,
                    pool_size=pool_size,
                )

    def forward(
        self,
        h: torch.Tensor,
        purpose_name: str,
        lambda_: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a specific purpose.

        Args:
            h: Input representation.
            purpose_name: Name of the purpose.
            lambda_: Gradient reversal scaling factor.

        Returns:
            Dictionary mapping attribute name to logits.
        """
        if purpose_name not in self.purpose_auditors:
            return {}

        return self.purpose_auditors[purpose_name](h, lambda_=lambda_)

    def forward_all_purposes(
        self,
        representations: dict[str, torch.Tensor],
        lambda_: float = 1.0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass for all purposes with their representations.

        Args:
            representations: Dictionary mapping purpose_name to representation.
            lambda_: Gradient reversal scaling factor.

        Returns:
            Nested dictionary: purpose_name -> attr_name -> logits.
        """
        outputs = {}
        for purpose_name in self.purpose_names:
            if purpose_name in representations and purpose_name in self.purpose_auditors:
                outputs[purpose_name] = self.forward(
                    representations[purpose_name], purpose_name, lambda_
                )
        return outputs

    def predict(
        self,
        h: torch.Tensor,
        purpose_name: str,
    ) -> dict[str, torch.Tensor]:
        """Get predictions for a specific purpose.

        Args:
            h: Input representation.
            purpose_name: Name of the purpose.

        Returns:
            Dictionary mapping attribute name to predicted labels.
        """
        if purpose_name not in self.purpose_auditors:
            return {}

        return self.purpose_auditors[purpose_name].predict(h)

    def get_accuracies(
        self,
        h: torch.Tensor,
        purpose_name: str,
        labels: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Get accuracy for each attribute of a purpose.

        Args:
            h: Input representation.
            purpose_name: Name of the purpose.
            labels: Dictionary mapping attribute name to labels.

        Returns:
            Dictionary mapping attribute name to accuracy.
        """
        if purpose_name not in self.purpose_auditors:
            return {}

        return self.purpose_auditors[purpose_name].get_accuracies(h, labels)

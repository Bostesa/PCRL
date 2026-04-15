"""Task prediction heads for PCRL."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Task prediction head.

    MLP that predicts allowed task output from representation.
    Supports both classification and regression tasks.
    """

    def __init__(
        self,
        repr_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        task_type: Literal["classification", "regression"] = "classification",
        dropout: float = 0.1,
    ) -> None:
        """Initialize the task head.

        Args:
            repr_dim: Dimension of input representation.
            output_dim: Number of output classes (for classification) or 1 (for regression).
            hidden_dim: Dimension of hidden layer.
            task_type: Type of prediction task ("classification" or "regression").
            dropout: Dropout probability.
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.task_type = task_type

        self.network = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim if task_type == "classification" else 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Kaiming for hidden layers (ReLU), Xavier for output."""
        modules = list(self.network)
        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                if i == len(modules) - 1:
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Input representation of shape (batch_size, repr_dim).

        Returns:
            Predictions of shape (batch_size, output_dim) for classification
            or (batch_size, 1) for regression.
        """
        return self.network(h)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Get predictions (class labels for classification, values for regression).

        Args:
            h: Input representation.

        Returns:
            Predicted labels of shape (batch_size,) for classification
            or values of shape (batch_size,) for regression.
        """
        logits = self.forward(h)
        if self.task_type == "classification":
            return logits.argmax(dim=-1)
        return logits.squeeze(-1)

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """Get predicted probabilities (classification only).

        Args:
            h: Input representation.

        Returns:
            Predicted probabilities of shape (batch_size, output_dim).
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        logits = self.forward(h)
        return torch.softmax(logits, dim=-1)


class MultiTaskHead(nn.Module):
    """Multi-task prediction head.

    Manages multiple task heads for different prediction tasks.
    """

    def __init__(
        self,
        repr_dim: int,
        task_specs: dict[str, tuple[int, str]],
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """Initialize multi-task head.

        Args:
            repr_dim: Dimension of input representation.
            task_specs: Dictionary mapping task name to (output_dim, task_type).
            hidden_dim: Dimension of hidden layers in task heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.task_names = list(task_specs.keys())

        self.heads = nn.ModuleDict()
        for name, (output_dim, task_type) in task_specs.items():
            self.heads[name] = TaskHead(
                repr_dim=repr_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                task_type=task_type,
                dropout=dropout,
            )

    def forward(
        self,
        h: torch.Tensor,
        task_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for specified tasks.

        Args:
            h: Input representation of shape (batch_size, repr_dim).
            task_names: List of task names to predict. If None, predicts all.

        Returns:
            Dictionary mapping task name to predictions.
        """
        if task_names is None:
            task_names = self.task_names

        outputs = {}
        for name in task_names:
            if name in self.heads:
                outputs[name] = self.heads[name](h)

        return outputs

    def predict(
        self,
        h: torch.Tensor,
        task_names: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get predictions for specified tasks.

        Args:
            h: Input representation.
            task_names: List of task names.

        Returns:
            Dictionary mapping task name to predicted labels/values.
        """
        if task_names is None:
            task_names = self.task_names

        predictions = {}
        for name in task_names:
            if name in self.heads:
                predictions[name] = self.heads[name].predict(h)

        return predictions


class PurposeTaskHeads(nn.Module):
    """Task heads organized by purpose.

    Each purpose has its own set of task heads for allowed tasks.
    """

    def __init__(
        self,
        repr_dim: int,
        purpose_task_specs: dict[str, dict[str, tuple[int, str]]],
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """Initialize purpose-specific task heads.

        Args:
            repr_dim: Dimension of input representation.
            purpose_task_specs: Nested dict: purpose_name -> task_name -> (output_dim, task_type).
            hidden_dim: Dimension of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.purpose_names = list(purpose_task_specs.keys())

        self.purpose_heads = nn.ModuleDict()
        for purpose_name, task_specs in purpose_task_specs.items():
            if task_specs:
                self.purpose_heads[purpose_name] = MultiTaskHead(
                    repr_dim=repr_dim,
                    task_specs=task_specs,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )

    def forward(
        self,
        h: torch.Tensor,
        purpose_name: str,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a specific purpose.

        Args:
            h: Input representation of shape (batch_size, repr_dim).
            purpose_name: Name of the purpose.

        Returns:
            Dictionary mapping task name to predictions.
        """
        if purpose_name not in self.purpose_heads:
            raise ValueError(f"Unknown purpose: {purpose_name}")

        return self.purpose_heads[purpose_name](h)

    def forward_all_purposes(
        self,
        representations: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass for all purposes with their representations.

        Args:
            representations: Dictionary mapping purpose_name to representation.

        Returns:
            Nested dictionary: purpose_name -> task_name -> predictions.
        """
        outputs = {}
        for purpose_name in self.purpose_names:
            if purpose_name in representations and purpose_name in self.purpose_heads:
                outputs[purpose_name] = self.purpose_heads[purpose_name](
                    representations[purpose_name]
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
            Dictionary mapping task name to predicted labels/values.
        """
        if purpose_name not in self.purpose_heads:
            raise ValueError(f"Unknown purpose: {purpose_name}")

        return self.purpose_heads[purpose_name].predict(h)

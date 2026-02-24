"""Loss functions for PCRL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    """Breakdown of loss components for logging.

    Attributes:
        total_loss: Combined total loss.
        task_loss: Total task prediction loss.
        adversarial_loss: Total adversarial loss.
        task_losses: Per-task losses.
        auditor_losses: Per-attribute auditor losses.
    """
    total_loss: float
    task_loss: float
    adversarial_loss: float
    task_losses: dict[str, float] = field(default_factory=dict)
    auditor_losses: dict[str, float] = field(default_factory=dict)


def task_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: Literal["classification", "regression"] = "classification",
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute task prediction loss.

    Args:
        predictions: Model predictions.
            - For classification: logits of shape (batch_size, num_classes).
            - For regression: values of shape (batch_size,) or (batch_size, 1).
        targets: Ground truth labels.
            - For classification: class indices of shape (batch_size,).
            - For regression: values of shape (batch_size,).
        task_type: Type of prediction task ("classification" or "regression").
        label_smoothing: Label smoothing factor for classification.
        class_weights: Optional class weights for imbalanced classification.

    Returns:
        Scalar loss tensor.
    """
    if task_type == "classification":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        return criterion(predictions, targets)
    else:
        # Regression
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        targets = targets.float()
        return F.mse_loss(predictions, targets)


def adversarial_loss(
    auditor_preds: torch.Tensor,
    true_attrs: torch.Tensor,
    maximize: bool = False,
    confusion_type: Literal["negative_ce", "entropy", "uniform_kl"] = "negative_ce",
) -> torch.Tensor:
    """Compute adversarial loss for sensitive attribute prediction.

    Args:
        auditor_preds: Auditor predictions (logits) of shape (batch_size, num_classes).
        true_attrs: Ground truth sensitive attribute labels of shape (batch_size,).
        maximize: If False, standard CE loss for training auditor.
                  If True, confusion loss for training encoder (maximize auditor error).
        confusion_type: Type of confusion loss when maximize=True:
            - "negative_ce": Negative cross-entropy (simple negation).
            - "entropy": Maximize prediction entropy.
            - "uniform_kl": Push predictions toward uniform distribution.

    Returns:
        Scalar loss tensor.
    """
    if not maximize:
        # Standard cross-entropy for training auditor to predict correctly
        return F.cross_entropy(auditor_preds, true_attrs)

    # Confusion loss for training encoder to confuse auditor
    if confusion_type == "negative_ce":
        # Simple negative cross-entropy
        return -F.cross_entropy(auditor_preds, true_attrs)

    elif confusion_type == "entropy":
        # Maximize entropy of predictions
        probs = F.softmax(auditor_preds, dim=-1)
        log_probs = F.log_softmax(auditor_preds, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        # Return negative because we want to maximize entropy (minimize negative entropy)
        return -entropy

    elif confusion_type == "uniform_kl":
        # Push predictions toward uniform distribution
        num_classes = auditor_preds.shape[-1]
        log_probs = F.log_softmax(auditor_preds, dim=-1)
        uniform = torch.ones_like(log_probs) / num_classes
        # KL(uniform || predicted) - minimize this to push toward uniform
        kl_div = F.kl_div(log_probs, uniform, reduction="batchmean")
        return kl_div

    else:
        raise ValueError(f"Unknown confusion_type: {confusion_type}")


def pcrl_loss(
    task_loss_val: torch.Tensor,
    adv_losses: dict[str, torch.Tensor],
    lambda_adv: float = 1.0,
    adv_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, LossBreakdown]:
    """Compute combined PCRL loss.

    Combined loss: L_total = L_task + lambda_adv * sum(L_adv)

    Note: The adversarial losses should already be computed with maximize=True
    for the encoder step (so they are confusion losses that should be minimized).

    Args:
        task_loss_val: Total task prediction loss.
        adv_losses: Dictionary mapping attribute name to adversarial loss.
        lambda_adv: Weight for adversarial loss term.
        adv_weights: Optional weights for each attribute's adversarial loss.

    Returns:
        Tuple of (total_loss, breakdown) where breakdown contains per-component losses.
    """
    adv_weights = adv_weights or {}

    # Sum adversarial losses
    total_adv_loss = torch.tensor(0.0, device=task_loss_val.device)
    adv_losses_float = {}

    for attr_name, loss in adv_losses.items():
        weight = adv_weights.get(attr_name, 1.0)
        total_adv_loss = total_adv_loss + weight * loss
        adv_losses_float[attr_name] = loss.item()

    # Combined loss
    total_loss = task_loss_val + lambda_adv * total_adv_loss

    breakdown = LossBreakdown(
        total_loss=total_loss.item(),
        task_loss=task_loss_val.item(),
        adversarial_loss=total_adv_loss.item(),
        task_losses={},  # Can be filled by caller if needed
        auditor_losses=adv_losses_float,
    )

    return total_loss, breakdown


def compute_multi_task_loss(
    task_predictions: dict[str, torch.Tensor],
    task_targets: dict[str, torch.Tensor],
    task_types: dict[str, str],
    task_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute loss for multiple tasks.

    Args:
        task_predictions: Dictionary mapping task name to predictions.
        task_targets: Dictionary mapping task name to targets.
        task_types: Dictionary mapping task name to type ("classification" or "regression").
        task_weights: Optional weights for each task.

    Returns:
        Tuple of (total_task_loss, per_task_losses).
    """
    task_weights = task_weights or {}
    per_task_losses = {}
    total_loss = None

    for task_name, preds in task_predictions.items():
        if task_name not in task_targets:
            continue

        targets = task_targets[task_name]
        task_type = task_types.get(task_name, "classification")

        loss = task_loss(preds, targets, task_type)
        per_task_losses[task_name] = loss

        weight = task_weights.get(task_name, 1.0)
        weighted_loss = weight * loss

        if total_loss is None:
            total_loss = weighted_loss
        else:
            total_loss = total_loss + weighted_loss

    if total_loss is None:
        # No tasks computed, return zero loss
        device = next(iter(task_predictions.values())).device if task_predictions else "cpu"
        total_loss = torch.tensor(0.0, device=device)

    return total_loss, per_task_losses


def compute_multi_auditor_loss(
    auditor_predictions: dict[str, torch.Tensor],
    auditor_targets: dict[str, torch.Tensor],
    maximize: bool = False,
    attr_weights: dict[str, float] | None = None,
    confusion_type: Literal["negative_ce", "entropy", "uniform_kl"] = "negative_ce",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute loss for multiple auditors.

    Args:
        auditor_predictions: Dictionary mapping attribute name to predictions.
        auditor_targets: Dictionary mapping attribute name to targets.
        maximize: Whether to compute confusion loss (for encoder training).
        attr_weights: Optional weights for each attribute.
        confusion_type: Type of confusion loss when maximize=True.

    Returns:
        Tuple of (total_auditor_loss, per_attr_losses).
    """
    attr_weights = attr_weights or {}
    per_attr_losses = {}
    total_loss = None

    for attr_name, preds in auditor_predictions.items():
        if attr_name not in auditor_targets:
            continue

        targets = auditor_targets[attr_name]

        loss = adversarial_loss(preds, targets, maximize=maximize, confusion_type=confusion_type)
        per_attr_losses[attr_name] = loss

        weight = attr_weights.get(attr_name, 1.0)
        weighted_loss = weight * loss

        if total_loss is None:
            total_loss = weighted_loss
        else:
            total_loss = total_loss + weighted_loss

    if total_loss is None:
        device = next(iter(auditor_predictions.values())).device if auditor_predictions else "cpu"
        total_loss = torch.tensor(0.0, device=device)

    return total_loss, per_attr_losses


class PCRLLossModule(nn.Module):
    """Module-based PCRL loss for convenience.

    Wraps the functional loss interface in an nn.Module.
    """

    def __init__(
        self,
        task_types: dict[str, str],
        lambda_adv: float = 1.0,
        task_weights: dict[str, float] | None = None,
        attr_weights: dict[str, float] | None = None,
        confusion_type: Literal["negative_ce", "entropy", "uniform_kl"] = "negative_ce",
    ) -> None:
        """Initialize PCRL loss module.

        Args:
            task_types: Dictionary mapping task name to type.
            lambda_adv: Weight for adversarial loss term.
            task_weights: Optional weights for each task.
            attr_weights: Optional weights for each attribute.
            confusion_type: Type of confusion loss for encoder training.
        """
        super().__init__()
        self.task_types = task_types
        self.lambda_adv = lambda_adv
        self.task_weights = task_weights
        self.attr_weights = attr_weights
        self.confusion_type = confusion_type

    def forward(
        self,
        task_predictions: dict[str, torch.Tensor],
        task_targets: dict[str, torch.Tensor],
        auditor_predictions: dict[str, torch.Tensor],
        auditor_targets: dict[str, torch.Tensor],
        maximize_confusion: bool = True,
    ) -> tuple[torch.Tensor, LossBreakdown]:
        """Compute combined PCRL loss.

        Args:
            task_predictions: Dictionary mapping task name to predictions.
            task_targets: Dictionary mapping task name to targets.
            auditor_predictions: Dictionary mapping attribute name to predictions.
            auditor_targets: Dictionary mapping attribute name to targets.
            maximize_confusion: Whether to compute confusion loss for adversarial term.

        Returns:
            Tuple of (total_loss, breakdown).
        """
        # Compute task loss
        total_task_loss, per_task_losses = compute_multi_task_loss(
            task_predictions, task_targets, self.task_types, self.task_weights
        )

        # Compute adversarial loss
        total_adv_loss, per_attr_losses = compute_multi_auditor_loss(
            auditor_predictions, auditor_targets,
            maximize=maximize_confusion,
            attr_weights=self.attr_weights,
            confusion_type=self.confusion_type,
        )

        # Combined loss
        total_loss = total_task_loss + self.lambda_adv * total_adv_loss

        breakdown = LossBreakdown(
            total_loss=total_loss.item(),
            task_loss=total_task_loss.item(),
            adversarial_loss=total_adv_loss.item(),
            task_losses={k: v.item() for k, v in per_task_losses.items()},
            auditor_losses={k: v.item() for k, v in per_attr_losses.items()},
        )

        return total_loss, breakdown

    def auditor_loss(
        self,
        auditor_predictions: dict[str, torch.Tensor],
        auditor_targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute auditor loss for auditor training step.

        Args:
            auditor_predictions: Dictionary mapping attribute name to predictions.
            auditor_targets: Dictionary mapping attribute name to targets.

        Returns:
            Total auditor loss (to be minimized by auditors).
        """
        total_loss, _ = compute_multi_auditor_loss(
            auditor_predictions, auditor_targets,
            maximize=False,  # Auditor wants to minimize CE
            attr_weights=self.attr_weights,
        )
        return total_loss


class FairnessPenalty(nn.Module):
    """Additional fairness penalties for PCRL.

    Implements various fairness constraints that can be added to the loss.
    """

    def __init__(
        self,
        penalty_type: Literal["demographic_parity", "equalized_odds"] = "demographic_parity",
        strength: float = 1.0,
    ) -> None:
        """Initialize fairness penalty.

        Args:
            penalty_type: Type of fairness penalty.
            strength: Strength of the penalty.
        """
        super().__init__()
        self.penalty_type = penalty_type
        self.strength = strength

    def forward(
        self,
        predictions: torch.Tensor,
        sensitive_attrs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute fairness penalty.

        Args:
            predictions: Model predictions (logits).
            sensitive_attrs: Sensitive attribute values.
            targets: Ground truth labels (required for equalized odds).

        Returns:
            Scalar penalty tensor.
        """
        if self.penalty_type == "demographic_parity":
            return self._demographic_parity(predictions, sensitive_attrs)
        elif self.penalty_type == "equalized_odds":
            if targets is None:
                raise ValueError("targets required for equalized_odds")
            return self._equalized_odds(predictions, sensitive_attrs, targets)
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")

    def _demographic_parity(
        self,
        predictions: torch.Tensor,
        sensitive_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute demographic parity penalty."""
        probs = torch.softmax(predictions, dim=-1)
        if probs.shape[-1] > 1:
            probs = probs[:, 1]  # Probability of positive class for binary

        unique_attrs = sensitive_attrs.unique()
        if len(unique_attrs) < 2:
            return torch.tensor(0.0, device=predictions.device)

        rates = []
        for attr_val in unique_attrs:
            mask = sensitive_attrs == attr_val
            if mask.sum() > 0:
                rates.append(probs[mask].mean())

        if len(rates) < 2:
            return torch.tensor(0.0, device=predictions.device)

        rates_tensor = torch.stack(rates)
        penalty = rates_tensor.var()

        return self.strength * penalty

    def _equalized_odds(
        self,
        predictions: torch.Tensor,
        sensitive_attrs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute equalized odds penalty."""
        probs = torch.softmax(predictions, dim=-1)
        if probs.shape[-1] > 1:
            probs = probs[:, 1]
        preds = (probs > 0.5).long()

        unique_attrs = sensitive_attrs.unique()
        if len(unique_attrs) < 2:
            return torch.tensor(0.0, device=predictions.device)

        tpr_list, fpr_list = [], []

        for attr_val in unique_attrs:
            mask = sensitive_attrs == attr_val

            pos_mask = mask & (targets == 1)
            if pos_mask.sum() > 0:
                tpr = (preds[pos_mask] == 1).float().mean()
                tpr_list.append(tpr)

            neg_mask = mask & (targets == 0)
            if neg_mask.sum() > 0:
                fpr = (preds[neg_mask] == 1).float().mean()
                fpr_list.append(fpr)

        penalty = torch.tensor(0.0, device=predictions.device)

        if len(tpr_list) >= 2:
            tpr_tensor = torch.stack(tpr_list)
            penalty = penalty + tpr_tensor.var()

        if len(fpr_list) >= 2:
            fpr_tensor = torch.stack(fpr_list)
            penalty = penalty + fpr_tensor.var()

        return self.strength * penalty

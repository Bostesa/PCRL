"""Compositional purpose algebra (AND, OR, hierarchy).

Key novelty: purposes can compose algebraically.

AND (p1 & p2): support tasks from both, hide attrs disallowed by either.
  allowed_tasks = UNION, disallowed_attrs = UNION

OR (p1 | p2): support tasks from either, hide only attrs disallowed by both.
  allowed_tasks = UNION, disallowed_attrs = INTERSECTION

Hierarchy: child inherits all parent constraints plus its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcrl.purposes.spec import PurposeSpec


# ── Composed purpose specification ───────────────────────────────────────


@dataclass
class ComposedPurpose:
    """A purpose formed by composing two base purposes.

    Supports Python operators: ``p1 & p2`` (AND), ``p1 | p2`` (OR).

    Attributes:
        name: Derived name (e.g. "p1_AND_p2").
        left: Left operand purpose spec.
        right: Right operand purpose spec.
        op: Composition operator ("and" or "or").
        allowed_tasks: Resolved set of allowed tasks.
        disallowed_attrs: Resolved set of disallowed attributes.
        allowed_task_dims: Merged output dims (union, conflicts resolved by left).
        disallowed_attr_dims: Merged attr dims (union/intersection matching op).
    """

    name: str
    left: PurposeSpec
    right: PurposeSpec
    op: Literal["and", "or"]
    allowed_tasks: list[str] = field(default_factory=list)
    disallowed_attrs: list[str] = field(default_factory=list)
    allowed_task_dims: dict[str, int] = field(default_factory=dict)
    disallowed_attr_dims: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        left_tasks = set(self.left.allowed_tasks)
        right_tasks = set(self.right.allowed_tasks)
        left_attrs = set(self.left.disallowed_attrs)
        right_attrs = set(self.right.disallowed_attrs)

        # Both AND and OR take the union of allowed tasks
        merged_tasks = left_tasks | right_tasks

        if self.op == "and":
            # AND: hide everything disallowed by *either* purpose
            merged_attrs = left_attrs | right_attrs
        elif self.op == "or":
            # OR: hide only what is disallowed by *both* purposes
            merged_attrs = left_attrs & right_attrs
        else:
            raise ValueError(f"Unknown composition op: {self.op}")

        self.allowed_tasks = sorted(merged_tasks)
        self.disallowed_attrs = sorted(merged_attrs)

        # Merge dims — left takes priority on conflicts
        merged_task_dims: dict[str, int] = {}
        for src in (self.right, self.left):
            for t in src.allowed_task_dims:
                if t in merged_tasks:
                    merged_task_dims[t] = src.allowed_task_dims[t]
        self.allowed_task_dims = merged_task_dims

        merged_attr_dims: dict[str, int] = {}
        for src in (self.right, self.left):
            for a in src.disallowed_attr_dims:
                if a in merged_attrs:
                    merged_attr_dims[a] = src.disallowed_attr_dims[a]
        self.disallowed_attr_dims = merged_attr_dims

    def to_purpose_spec(self) -> PurposeSpec:
        """Convert back to a flat PurposeSpec."""
        return PurposeSpec(
            name=self.name,
            allowed_tasks=self.allowed_tasks,
            disallowed_attrs=self.disallowed_attrs,
            task_type=self.left.task_type,
            allowed_task_dims=self.allowed_task_dims,
            disallowed_attr_dims=self.disallowed_attr_dims,
        )


def compose_and(p1: PurposeSpec, p2: PurposeSpec) -> ComposedPurpose:
    """AND composition: support tasks from both, hide attrs disallowed by either."""
    return ComposedPurpose(
        name=f"{p1.name}_AND_{p2.name}",
        left=p1,
        right=p2,
        op="and",
    )


def compose_or(p1: PurposeSpec, p2: PurposeSpec) -> ComposedPurpose:
    """OR composition: support tasks from either, hide only attrs disallowed by both."""
    return ComposedPurpose(
        name=f"{p1.name}_OR_{p2.name}",
        left=p1,
        right=p2,
        op="or",
    )


def compose_hierarchy(
    parent: PurposeSpec,
    child: PurposeSpec,
) -> ComposedPurpose:
    """Hierarchical composition: child inherits all parent constraints plus its own.

    Equivalent to AND but with clearer semantics for parent-child relationships.
    """
    return ComposedPurpose(
        name=f"{child.name}_inherits_{parent.name}",
        left=child,
        right=parent,
        op="and",
    )


# ── Embedding composition ────────────────────────────────────────────────


def compose_embeddings_additive(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> torch.Tensor:
    """AND composition of embeddings: additive.

    h_{p1 AND p2} = embed(p1) + embed(p2)

    Args:
        emb1: Purpose embedding (batch_size, emb_dim) or (emb_dim,).
        emb2: Purpose embedding (batch_size, emb_dim) or (emb_dim,).

    Returns:
        Composed embedding of same shape.
    """
    return emb1 + emb2


def compose_embeddings_max(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> torch.Tensor:
    """OR composition of embeddings: element-wise max.

    h_{p1 OR p2} = max(embed(p1), embed(p2))

    Args:
        emb1: Purpose embedding (batch_size, emb_dim) or (emb_dim,).
        emb2: Purpose embedding (batch_size, emb_dim) or (emb_dim,).

    Returns:
        Composed embedding of same shape.
    """
    return torch.max(emb1, emb2)


class LearnedComposer(nn.Module):
    """Learned embedding composition via MLP.

    embed(p1 OP p2) = MLP([embed(p1); embed(p2)])
    """

    def __init__(self, emb_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or emb_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compose two embeddings via learned MLP.

        Args:
            emb1: (batch_size, emb_dim) or (emb_dim,).
            emb2: (batch_size, emb_dim) or (emb_dim,).

        Returns:
            Composed embedding of shape matching emb1.
        """
        combined = torch.cat([emb1, emb2], dim=-1)
        return self.mlp(combined)


# ── Composition consistency loss ─────────────────────────────────────────


class CompositionLoss(nn.Module):
    """Loss that enforces compositional consistency.

    For AND: the composed representation h_{p1 AND p2} must hide everything
    disallowed by either p1 OR p2. We measure this by training auditors on
    the composed representation and penalizing high accuracy.

    For OR: the composed representation h_{p1 OR p2} must hide everything
    disallowed by both p1 AND p2.

    This loss takes auditor logits on the composed representation and
    pushes them toward uniform (maximum entropy), enforcing that
    disallowed attributes cannot be predicted.
    """

    def __init__(self, confusion_type: Literal["entropy", "uniform_kl"] = "entropy") -> None:
        super().__init__()
        self.confusion_type = confusion_type

    def forward(
        self,
        auditor_logits: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute composition consistency loss.

        Args:
            auditor_logits: Dict mapping disallowed attr name to logits
                tensor of shape (batch_size, num_classes) from auditors
                applied to the composed representation.

        Returns:
            Scalar loss (lower is better — representations hide disallowed attrs).
        """
        if not auditor_logits:
            return torch.tensor(0.0)

        total_loss = torch.tensor(0.0, device=next(iter(auditor_logits.values())).device)

        for attr_name, logits in auditor_logits.items():
            if self.confusion_type == "entropy":
                # Maximize prediction entropy → push toward uniform
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                # Negative entropy: minimizing this maximizes entropy
                total_loss = total_loss - entropy
            elif self.confusion_type == "uniform_kl":
                num_classes = logits.shape[-1]
                log_probs = F.log_softmax(logits, dim=-1)
                uniform = torch.ones_like(log_probs) / num_classes
                total_loss = total_loss + F.kl_div(
                    log_probs, uniform, reduction="batchmean"
                )

        return total_loss

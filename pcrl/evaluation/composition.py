"""Compositional purpose evaluation: additive vs sequential FiLM.

Two ways to obtain a representation that simultaneously honours a set of
purposes (e.g. p1 AND p2 AND p3):

1. Additive embedding composition (the original paper rule)
       e_composed = sum_k e_p_k
       h_composed = encoder.forward_with_embedding(x, e_composed)

   Implemented in :func:`extract_composed_reprs_additive`. Because the FiLM
   gamma generator is linear, gamma_composed = sum_k gamma_p_k - (K - 1) b
   where b is the learned gamma bias. The resulting gate behaviour depends
   on the bias and is not multiplicatively suppressing.

2. Sequential FiLM composition (the principled rule)
       h_layer_i <- linear_i(h_layer_{i-1})
       for k in 1..K:  h_layer_i <- film_i(h_layer_i, e_p_k)
       h_layer_i <- norm; relu; dropout

   Implemented in :func:`extract_composed_reprs_sequential`, which calls
   :meth:`PurposeConditionedEncoder.forward_sequential_composition`. The
   effective per-layer gamma is the elementwise product of per-purpose
   gammas, so a zero gate in any single purpose suppresses that feature
   in the composed output regardless of the other purposes.

Audit utilities here pair both representations with the existing linear
and empirical compliance audits (R^2, delta vs majority baseline, adjusted
pass flag) so the two composition rules can be benchmarked head-to-head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pcrl.evaluation.certificates import EmpiricalAudit, LinearAudit
from pcrl.models.encoder import PurposeConditionedEncoder


# ─────────────────────────────────────────────────────────────────────────
# Representation extraction
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def extract_composed_reprs_additive(
    encoder: PurposeConditionedEncoder,
    loader: DataLoader,
    purpose_embs: list[torch.Tensor],
    device: str,
) -> np.ndarray:
    """Extract representations using additively-composed embeddings.

    The list of purpose embeddings is summed and passed through
    ``forward_with_embedding`` once per batch.
    """
    encoder.eval()
    composed = torch.stack([e.to(device) for e in purpose_embs], dim=0).sum(dim=0)
    out: list[np.ndarray] = []
    for batch in loader:
        x = batch["features"].to(device)
        if composed.dim() == 1:
            composed_b = composed.unsqueeze(0).expand(x.shape[0], -1)
        else:
            composed_b = composed
        h = encoder.forward_with_embedding(x, composed_b)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def extract_composed_reprs_sequential(
    encoder: PurposeConditionedEncoder,
    loader: DataLoader,
    purpose_embs: list[torch.Tensor],
    device: str,
) -> np.ndarray:
    """Extract representations by applying each FiLM head sequentially.

    Equivalent to multiplying per-purpose gates layer-by-layer.
    """
    encoder.eval()
    embs_dev = [e.to(device) for e in purpose_embs]
    out: list[np.ndarray] = []
    for batch in loader:
        x = batch["features"].to(device)
        embs_b = []
        for e in embs_dev:
            if e.dim() == 1:
                embs_b.append(e.unsqueeze(0).expand(x.shape[0], -1))
            else:
                embs_b.append(e)
        h = encoder.forward_sequential_composition(x, embs_b)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def evaluate_task_acc_with_reprs(
    test_reprs: np.ndarray,
    test_labels: np.ndarray,
    train_reprs: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
) -> float:
    """Fit a logistic regression on the composed reps and report test accuracy.

    Uses train reps if provided, else fits on test (in-sample, less reliable
    but useful for quick sanity checks).
    """
    from sklearn.linear_model import LogisticRegression
    if train_reprs is None or train_labels is None:
        train_reprs, train_labels = test_reprs, test_labels
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(train_reprs, train_labels)
    return float(clf.score(test_reprs, test_labels))


# ─────────────────────────────────────────────────────────────────────────
# Per-attribute compliance audit
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class AttrAuditResult:
    attr_name: str
    delta: float            # empirical_best_acc - majority_proportion
    r_squared: float        # linear audit R^2
    empirical_best_acc: float
    majority_proportion: float
    num_classes: int
    adj_pass: bool          # delta < 0.02 AND r_squared < 0.05


def audit_attr_on_reprs(
    train_reprs: np.ndarray,
    train_labels: np.ndarray,
    test_reprs: np.ndarray,
    test_labels: np.ndarray,
    attr_name: str,
    delta_threshold: float = 0.02,
    r2_threshold: float = 0.05,
) -> AttrAuditResult:
    """Linear audit + empirical audit for a single attribute on a fixed rep."""
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()

    linear_result, _ = linear_audit.audit(test_reprs, test_labels)
    all_labels = np.concatenate([train_labels, test_labels])
    _, counts = np.unique(all_labels, return_counts=True)
    majority = float(counts.max() / len(all_labels))
    num_classes = int(len(np.unique(all_labels)))
    best_acc, _ = empirical_audit.audit(train_reprs, train_labels, test_reprs, test_labels)

    delta = float(best_acc) - majority
    return AttrAuditResult(
        attr_name=attr_name,
        delta=delta,
        r_squared=float(linear_result.r_squared),
        empirical_best_acc=float(best_acc),
        majority_proportion=majority,
        num_classes=num_classes,
        adj_pass=(delta < delta_threshold and float(linear_result.r_squared) < r2_threshold),
    )


def audit_all_attrs(
    train_reprs: np.ndarray,
    test_reprs: np.ndarray,
    train_attrs: dict[str, np.ndarray],
    test_attrs: dict[str, np.ndarray],
    attr_names: Iterable[str],
    delta_threshold: float = 0.02,
    r2_threshold: float = 0.05,
) -> dict[str, AttrAuditResult]:
    """Audit each attribute against a single fixed representation set.

    Returns a dict keyed by attribute name.
    """
    results: dict[str, AttrAuditResult] = {}
    for name in attr_names:
        results[name] = audit_attr_on_reprs(
            train_reprs=train_reprs,
            train_labels=train_attrs[name],
            test_reprs=test_reprs,
            test_labels=test_attrs[name],
            attr_name=name,
            delta_threshold=delta_threshold,
            r2_threshold=r2_threshold,
        )
    return results

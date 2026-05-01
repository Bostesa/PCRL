"""Tests for the Round 6 per-class OvR linear-R² constraint.

Diabetes Round 5 failure on quality_research/age_bucket (joint multi-output
R²≈0.06 driven by a single class with R²≈0.6) motivates the per-class fix:
attributes with cardinality K >= per_class_constraint_threshold get K
independent OvR binary R² constraints instead of one joint constraint.
"""

from __future__ import annotations

import torch

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.independence.vclub import VCLUB
from pcrl.training.losses import VerificationRegularizer
from pcrl.training.proxy_lagrangian import Constraint
from pcrl.training.v2_trainer import V2Trainer, V2TrainerConfig, _per_class_key


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────


def _build_trainer(
    attr_dims: dict[str, int],
    per_class_threshold: int = 6,
    repr_dim: int = 16,
    n_features: int = 8,
) -> V2Trainer:
    """Build a minimal V2Trainer with one purpose and the requested attr dims."""
    purpose = PurposeSpec(
        name="task_purpose",
        task_type="classification",
        allowed_tasks=["primary_task"],
        disallowed_attrs=list(attr_dims.keys()),
        allowed_task_dims={"primary_task": 2},
        disallowed_attr_dims=dict(attr_dims),
    )
    registry = PurposeRegistry()
    registry.register(purpose)

    backbone = StandardEncoder(
        input_dim=n_features, hidden_dims=[16], repr_dim=repr_dim, dropout=0.0,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=1, rank=4, alpha=8.0, dropout=0.0,
    )
    task_heads = {"task_purpose": TaskHead(repr_dim=repr_dim, output_dim=2)}
    vclubs = {}
    for attr, K in attr_dims.items():
        vclubs[f"task_purpose__{attr}"] = VCLUB(
            x_dim=repr_dim, z_dim=K, hidden_dim=16, z_categorical=True, l2=1e-1,
        )

    config = V2TrainerConfig(
        epochs=1, warmup_epochs=0, leace_init=False,
        lambda_min=5.0,
        per_class_constraint_threshold=per_class_threshold,
    )
    return V2Trainer(
        encoder=encoder, task_heads=task_heads, vclubs=vclubs,
        purpose_registry=registry, config=config, device="cpu",
    )


# ───────────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────────


def test_per_class_constraint_count_K_10_threshold_6():
    """K=10 >= threshold 6 → K=10 OvR constraints (one per class)."""
    trainer = _build_trainer({"age_bucket": 10}, per_class_threshold=6)
    pair = ("task_purpose", "age_bucket")
    assert pair in trainer.high_k_pairs
    assert trainer.high_k_pairs[pair] == 10
    names = trainer.pair_constraint_keys[pair]
    assert len(names) == 10
    expected = {_per_class_key("task_purpose", "age_bucket", k) for k in range(10)}
    assert set(names) == expected
    # Every constraint must live inside the proxy optimiser as well.
    for name in names:
        assert name in trainer.proxy.constraints


def test_low_cardinality_keeps_joint_constraint():
    """K=5 < default threshold 6 → 1 joint multi-output constraint (legacy)."""
    trainer = _build_trainer({"race": 5}, per_class_threshold=6)
    pair = ("task_purpose", "race")
    assert pair not in trainer.high_k_pairs
    names = trainer.pair_constraint_keys[pair]
    assert len(names) == 1
    assert names[0] == "task_purpose__race"
    assert "task_purpose__race" in trainer.proxy.constraints
    # No per-class constraint names should be registered.
    for k in range(5):
        assert _per_class_key("task_purpose", "race", k) not in trainer.proxy.constraints


def test_threshold_boundary_K_equals_threshold_activates_per_class():
    """K==threshold goes per-class (>= boundary). K==threshold-1 stays joint."""
    # threshold=6, K=6 → per-class
    trainer_a = _build_trainer({"x_6": 6}, per_class_threshold=6)
    assert ("task_purpose", "x_6") in trainer_a.high_k_pairs
    assert len(trainer_a.pair_constraint_keys[("task_purpose", "x_6")]) == 6
    # threshold=8, K=6 → joint (single constraint)
    trainer_b = _build_trainer({"x_6": 6}, per_class_threshold=8)
    assert ("task_purpose", "x_6") not in trainer_b.high_k_pairs
    assert len(trainer_b.pair_constraint_keys[("task_purpose", "x_6")]) == 1


def test_per_class_r2_matches_independent_ridge_per_class():
    """forward_per_class[k] equals the R² of an independent ridge solve on
    the binary indicator (Z == k).float(). This is the closed-form OvR
    one-vs-rest metric we claim to be enforcing.
    """
    torch.manual_seed(0)
    n, d, K = 200, 12, 5
    H = torch.randn(n, d)
    # Inject linear dependence so several per-class R² are >> 0.
    Z = torch.randint(0, K, (n,))
    # Make class k=2 clearly leakable.
    H[:, 0] += (Z == 2).float() * 1.5
    # Make class k=4 mildly leakable.
    H[:, 1] += (Z == 4).float() * 0.7

    verifier = VerificationRegularizer(regularization=1e-4)
    r2_per_k = verifier.forward_per_class(H, Z)
    assert r2_per_k.shape == (K,)

    # Independently compute single-output ridge per class.
    H_c = H - H.mean(dim=0, keepdim=True)
    gram = H_c.T @ H_c + 1e-4 * torch.eye(d)
    for k in range(K):
        z_k = (Z == k).float()
        z_k_c = z_k - z_k.mean()
        w = torch.linalg.solve(gram, H_c.T @ z_k_c)
        z_pred = H_c @ w
        ss_res = ((z_k_c - z_pred) ** 2).sum()
        ss_tot = (z_k_c ** 2).sum().clamp(min=1e-12)
        expected_r2 = max(0.0, float(1.0 - ss_res / ss_tot))
        assert float(r2_per_k[k].item()) == \
            __import__("pytest").approx(expected_r2, rel=1e-4, abs=1e-5)

    # The leaky classes should genuinely show > 0 R²; pathology check.
    assert float(r2_per_k[2].item()) > 0.05
    assert float(r2_per_k[4].item()) > 0.0


def test_per_class_dual_independence():
    """Per-class duals are independent: updating one constraint's lambda
    does not move any other constraint's lambda or its gradient direction.
    """
    trainer = _build_trainer({"age_bucket": 10}, per_class_threshold=6)
    pair = ("task_purpose", "age_bucket")
    names = trainer.pair_constraint_keys[pair]
    # Snapshot lambdas
    pre = {n: trainer.proxy.constraints[n].lambda_value for n in names}
    # Push one constraint hard via a violating value
    trainer.proxy.dual_step({names[3]: 1.0})  # value=1.0 vs threshold=0.05
    post = {n: trainer.proxy.constraints[n].lambda_value for n in names}
    # Only names[3]'s lambda should have changed.
    for n in names:
        if n == names[3]:
            assert post[n] > pre[n]
        else:
            assert post[n] == pre[n]


def test_joint_R2_dominated_by_max_per_class_R2():
    """Sanity check on the math the fix relies on: per_class R²<τ for every
    k implies joint multi-output R² < τ. (joint = ss_tot-weighted average
    of per-class R², so it is bounded above by max_k R²_k.)
    """
    torch.manual_seed(0)
    n, d, K = 500, 16, 10
    H = torch.randn(n, d)
    Z = torch.randint(0, K, (n,))
    # Light leak on k=7 only.
    H[:, 0] += (Z == 7).float() * 0.3

    verifier = VerificationRegularizer(regularization=1e-4)
    joint = float(verifier(H, Z).item())
    per_k = verifier.forward_per_class(H, Z)
    max_k = float(per_k.max().item())
    assert joint <= max_k + 1e-6, (
        f"joint R² {joint:.4f} should be <= max_k R² {max_k:.4f}"
    )

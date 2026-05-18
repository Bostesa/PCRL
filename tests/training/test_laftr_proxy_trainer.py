"""Tests for LAFTR + proxy-Lagrangian trainer."""
from __future__ import annotations

import torch

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.laftr_proxy_trainer import (
    LAFTRProxyTrainer,
    LAFTRProxyTrainerConfig,
)


def _build_trainer(toy_purposes, *, lambda_adv: float = 0.1, lr_lambda: float = 0.02,
                   lambda_min: float = 5.0, leace_init: bool = False) -> LAFTRProxyTrainer:
    registry = PurposeRegistry()
    for p in toy_purposes:
        registry.register(p)
    backbone = StandardEncoder(input_dim=16, hidden_dims=[128, 128], repr_dim=64, dropout=0.3)
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(toy_purposes), rank=8, alpha=16.0, dropout=0.0,
    )
    task_heads = {p.name: TaskHead(repr_dim=64, output_dim=2) for p in toy_purposes}
    config = LAFTRProxyTrainerConfig(
        lambda_adv=lambda_adv, lr_lambda=lr_lambda, r2_threshold=0.05,
        lambda_min=lambda_min, lambda_hsic_init=1.0, r2_lambda_max=1000.0,
        lambda_vicreg=1.0, epochs=1, batch_size=32, leace_init=leace_init,
    )
    return LAFTRProxyTrainer(
        encoder=encoder, task_heads=task_heads,
        purpose_registry=registry, config=config, device="cpu",
    )


def test_constraints_registered_one_per_pair(toy_purposes):
    trainer = _build_trainer(toy_purposes)
    # 2 purposes × 1 attr each = 2 constraints
    assert len(trainer.proxy.constraints) == 2
    assert "purpose_0__attr_a" in trainer.proxy.constraints
    assert "purpose_1__attr_b" in trainer.proxy.constraints


def test_constraint_hyperparameters_match_spec(toy_purposes):
    trainer = _build_trainer(toy_purposes)
    c = trainer.proxy.constraints["purpose_0__attr_a"]
    assert c.threshold == 0.05
    assert c.eta_lambda == 0.02
    assert c.lambda_value == 5.0  # max(lambda_init=1.0, lambda_min=5.0)
    assert c.lambda_max == 1000.0
    assert c.lambda_min == 5.0
    assert c.direction == "<="


def test_discriminator_registered_per_pair(toy_purposes):
    trainer = _build_trainer(toy_purposes)
    assert set(trainer.discriminators.keys()) == {
        "purpose_0__attr_a", "purpose_1__attr_b",
    }
    for d in trainer.discriminators.values():
        # LAFTRDiscriminator: Linear(64,64) → ReLU → Linear(64,32) → ReLU → Linear(32,K)
        layers = list(d.net.children())
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == 64


def test_discriminator_step_detaches_encoder(toy_purposes, toy_loaders):
    train, _ = toy_loaders
    trainer = _build_trainer(toy_purposes)
    # Snapshot LoRA adapter weights before the disc step.
    before = {
        n: p.detach().clone() for n, p in trainer.encoder.adapters.named_parameters()
    }
    batch = next(iter(train))
    batch = {
        "features": batch["features"],
        "task_labels": {k: v for k, v in batch["task_labels"].items()},
        "sensitive_attrs": {k: v for k, v in batch["sensitive_attrs"].items()},
    }
    loss = trainer._discriminator_step(batch)
    assert loss > 0  # discriminator should produce a positive CE
    for n, p in trainer.encoder.adapters.named_parameters():
        assert torch.allclose(before[n], p), f"LoRA param {n} moved during disc step"
    # Discriminator weights, by contrast, must move.
    moved = False
    for d in trainer.discriminators.values():
        for p in d.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                moved = True
                break
    assert moved, "discriminator parameters had no gradient after step"


def test_primal_loss_formula(toy_purposes, toy_loaders, monkeypatch):
    """The primal loss must equal:
        L_task + λ_vicreg·L_vicreg − λ_adv·Σ CE(disc, attr) + Σ λ_pa·(R² − τ)
    Reconstruct using ``stats["pre_dual_lambdas"]`` — primal_loss was computed
    with those values; ``trainer.proxy.constraints[n].lambda_value`` reflects
    the *post* dual-step state.
    """
    train, _ = toy_loaders
    trainer = _build_trainer(toy_purposes, lambda_adv=0.1)
    # Capture the loss the trainer constructs.
    batch = trainer._to_device(next(iter(train)))
    stats = trainer._primal_step_components(batch)
    # Expected reconstruction
    base = stats["L_task"] + trainer.config.lambda_vicreg * stats["L_vicreg"] \
           - trainer.config.lambda_adv * stats["L_adv"]
    lagrangian = sum(
        stats["pre_dual_lambdas"][n] * (stats["constraint_scalars"][n] - trainer.config.r2_threshold)
        for n in stats["constraint_scalars"]
    )
    assert abs(stats["primal_loss"] - (base + lagrangian)) < 1e-4

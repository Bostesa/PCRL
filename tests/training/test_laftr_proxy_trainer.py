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

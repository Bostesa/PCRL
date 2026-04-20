"""Regression test: generate_report() must align representations with labels.

Catches the shuffle-misalignment bug from commit a50810e (April 2026), where
generate_report() was refactored to extract representations and labels in
separate iterations of a shuffled DataLoader.  With shuffle=True, each
iteration produces a different sample ordering, so the PostHocAuditorSuite
was trained on misaligned (representation, label) pairs and could not beat
majority accuracy regardless of actual attribute leakage.

The test trains a Standard encoder (no adversarial training) on Adult Census
data.  The encoder's representation should trivially leak Sex (R² ≈ 0.6).
We verify that generate_report() reports Sex empirical accuracy well above
the majority baseline — specifically, delta > 15%.  If the extraction is
misaligned, the auditors learn nothing and delta ≈ 0%.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

# Suppress tqdm during test
import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args):
        pass

    def close(self):
        pass


_trainer_mod.tqdm = _QuietTqdm


@pytest.fixture(scope="module")
def adult_reports():
    """Train a Standard encoder and run generate_report() once for all tests."""
    torch.manual_seed(42)

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(
        purposes=purposes, root="data", split="val", download=False,
        norm_stats=train_ds.norm_stats,
    )
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )

    input_dim = train_ds.info.num_features

    # Shuffled train loader — this is what triggers the bug if extraction
    # is done in separate passes.
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )

    encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    task_heads, auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=64, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
        epochs=30, weight_decay=1e-4, early_stopping_patience=10,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device="cpu",
    )
    trainer.train(train_loader, val_loader=val_loader)

    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device="cpu",
    )
    return reports


def test_standard_sex_delta_above_15_percent(adult_reports):
    """Standard encoder must leak Sex — delta > 15% confirms aligned extraction."""
    sex_reports = [r for r in adult_reports if r.attr_name == "sex"]
    assert len(sex_reports) > 0, "No Sex attribute reports found"

    for r in sex_reports:
        delta = r.empirical_best_acc - r.majority_proportion
        assert delta > 0.15, (
            f"Standard encoder Sex delta = {delta:+.1%} for purpose "
            f"'{r.purpose_name}' — expected > +15%. This likely means "
            f"generate_report() is extracting representations and labels "
            f"in separate passes over a shuffled DataLoader, causing "
            f"row-level misalignment. See commit a50810e."
        )


def test_standard_race_delta_positive(adult_reports):
    """Standard encoder must leak Race — delta should be positive."""
    race_reports = [r for r in adult_reports if r.attr_name == "race"]
    assert len(race_reports) > 0, "No Race attribute reports found"

    for r in race_reports:
        delta = r.empirical_best_acc - r.majority_proportion
        assert delta > 0.01, (
            f"Standard encoder Race delta = {delta:+.1%} for purpose "
            f"'{r.purpose_name}' — expected > +1%."
        )


def test_standard_marital_delta_above_30_percent(adult_reports):
    """Standard encoder must leak Marital Status — delta > 30%."""
    marital_reports = [r for r in adult_reports if r.attr_name == "marital_status"]
    assert len(marital_reports) > 0, "No Marital Status attribute reports found"

    for r in marital_reports:
        delta = r.empirical_best_acc - r.majority_proportion
        assert delta > 0.30, (
            f"Standard encoder Marital delta = {delta:+.1%} for purpose "
            f"'{r.purpose_name}' — expected > +30%."
        )


def test_r2_consistent_with_empirical(adult_reports):
    """High R² should correlate with high empirical accuracy, not near-majority."""
    for r in adult_reports:
        if r.linear_r2 > 0.3:
            delta = r.empirical_best_acc - r.majority_proportion
            assert delta > 0.02, (
                f"R² = {r.linear_r2:.3f} but empirical delta = {delta:+.1%} "
                f"for {r.purpose_name}/{r.attr_name}. High R² with near-zero "
                f"delta indicates misaligned extraction."
            )

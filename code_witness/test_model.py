"""Data-free wire and training-contract checks for the repaired code learner."""

import numpy as np
import pytest
import torch

from code_witness.model import FixedScaler, HardCodeModel, HardCodeRelease
from code_witness.learner import FitData, TrainConfig, fit_task_only, task_loss


@pytest.mark.parametrize("width", [10, 16])
def test_hard_forward_has_exact_wire_width_and_gradient(width):
    torch.manual_seed(17)
    model = HardCodeModel(16, (2, 6), width)
    x = torch.randn(7, 16)
    code, logits = model(1, x, True)
    assert code.shape == (7, width)
    assert logits.shape == (7, 6)
    assert bool(torch.all((code == 0) | (code == 1)))
    logits.sum().backward()
    assert model.mappers[1].layer2.weight.grad is not None
    assert torch.isfinite(model.mappers[1].layer2.weight.grad).all()
    with pytest.raises(ValueError, match="hard binary"):
        model.logits_from_code(1, torch.full((7, width), 0.5))


def test_release_probabilities_are_reconstructed_from_code_only():
    torch.manual_seed(3)
    model = HardCodeModel(16, (2, 6), 10)
    scaler = FixedScaler(np.zeros((2, 16), np.float32), np.ones((2, 16), np.float32))
    release = HardCodeRelease(model, scaler, "test", "a" * 64)
    h = np.random.default_rng(1).normal(size=(9, 16)).astype(np.float32)
    out = release.recipient(1, h)
    assert set(out) == {"code", "probabilities", "decision", "metadata"}
    assert out["code"].shape == (9, 10)
    assert np.array_equal(out["probabilities"], release.probabilities_from_code(1, out["code"]))
    assert np.array_equal(out["decision"], out["probabilities"].argmax(1))
    assert out["metadata"]["capacity_states"] == 1024
    with pytest.raises(ValueError):
        release.probabilities_from_code(1, np.full((9, 10), 0.5))


def _synthetic_data(n=40):
    rng = np.random.default_rng(9)
    h0 = rng.normal(size=(n, 16)).astype(np.float32)
    h1 = rng.normal(size=(n, 16)).astype(np.float32)
    y0 = (h0[:, 0] > 0).astype(np.int64)
    y1 = np.arange(n, dtype=np.int64) % 6
    t0 = np.full((n, 2), 0.2, np.float32)
    t0[np.arange(n), y0] = 0.8
    t1 = np.full((n, 6), 0.04, np.float32)
    t1[np.arange(n), y1] = 0.8
    return FitData(h0, h1, y0, y1, t0, t1, None, np.arange(n), "SYNTHETIC")


def test_task_loss_uses_real_labels_and_teacher_kl():
    logits = (torch.tensor([[1., 0.], [0., 1.]]), torch.zeros(2, 6))
    y = (torch.tensor([0, 1]), torch.tensor([0, 1]))
    teachers = (torch.tensor([[0.7, 0.3], [0.3, 0.7]]), torch.full((2, 6), 1/6))
    got = task_loss(logits, y, teachers, 0.5)
    expected = sum(torch.nn.functional.cross_entropy(logits[i], y[i]) +
                   0.5 * torch.nn.functional.kl_div(
                       torch.nn.functional.log_softmax(logits[i], dim=1),
                       teachers[i], reduction="batchmean") for i in (0, 1)) / 2
    assert torch.allclose(got, expected)


def test_task_only_fold_selects_hard_validation_and_refits_full_role():
    data = _synthetic_data()
    config = TrainConfig(epochs=3, patience=2, batch_size=8)
    result = fit_task_only(data, config, seed=0, bits=16,
                           train_idx=np.arange(30), val_idx=np.arange(30, 40))
    assert 1 <= result.selected_epoch <= 3
    assert result.fold_updates == 4 * len(result.fold_history)
    assert result.full_updates == 5 * result.selected_epoch
    assert len(result.full_history) == result.selected_epoch
    assert result.model.bits_count == 16
    assert all(np.isfinite(row["validation_nll"]) for row in result.fold_history)


def test_task_only_refuses_sensitive_labels():
    data = _synthetic_data()
    sensitive = FitData(data.h0, data.h1, data.y0, data.y1,
                        data.teacher0, data.teacher1, np.zeros(data.rows, np.int64),
                        data.row_ids, "SYNTHETIC")
    with pytest.raises(ValueError, match="SEX"):
        fit_task_only(sensitive, TrainConfig(epochs=1, patience=1), seed=0,
                      bits=10, train_idx=np.arange(30), val_idx=np.arange(30, 40))

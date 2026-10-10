"""F reference head receives only unchanged latents and task supervision."""

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.fit_f import (
    fit_f_reference,
    fit_f_stage,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import FitData, TrainConfig
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler


def _data(sex):
    rng = np.random.default_rng(41)
    n = 18
    h0 = rng.normal(size=(n, 3)).astype(np.float32)
    h1 = rng.normal(size=(n, 3)).astype(np.float32)
    y0 = np.arange(n) % 2
    y1 = np.arange(n) % 6
    t0 = np.full((n, 2), 0.4, dtype=np.float32)
    t0[np.arange(n), y0] = 0.6
    t1 = np.full((n, 6), 0.1, dtype=np.float32)
    t1[np.arange(n), y1] = 0.5
    return FitData(h0, h1, y0, y1, t0, t1, sex,
                   np.arange(n), "SYNTHETIC")


def test_f_stage_ignores_sex_and_updates_both_heads_equally():
    sex = np.arange(18) % 2
    data = _data(sex)
    permuted = _data(1 - sex)
    scaler = FixedScaler.fit(data.h0, data.h1)
    config = TrainConfig(epochs=2, batch_size=8, learning_rate=0.001,
                         distill_weight=0.5, privacy_weight=0)
    first = fit_f_stage(data, config, 13, scaler, np.arange(18))
    second = fit_f_stage(permuted, config, 13, scaler, np.arange(18))
    assert first.optimizer_updates == 6
    assert len(first.history) == 2
    for purpose in (0, 1):
        for name, value in first.heads[purpose].state_dict().items():
            assert torch.equal(value, second.heads[purpose].state_dict()[name])


def test_f_selects_each_task_epoch_on_fit_fold_then_refits_full_fit():
    data = _data(np.arange(18) % 2)
    scaler = FixedScaler.fit(data.h0, data.h1)
    config = TrainConfig(epochs=2, batch_size=8, learning_rate=0.001,
                         distill_weight=0.5, privacy_weight=0,
                         checkpoints=(1, 2))
    result = fit_f_reference(data, config, 13, scaler,
                             train_idx=np.arange(14), val_idx=np.arange(14, 18),
                             source_sha256="c" * 64)
    assert len(result.split_checkpoint_nll) == 2
    assert all(epoch in (1, 2) for epoch in result.selected_epochs)
    assert result.full_fit_updates_by_purpose == tuple(
        3 * epoch for epoch in result.selected_epochs)
    assert result.release.recipient(0, data.h0)["features"].shape == (18, 3)

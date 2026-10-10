"""Matched ordinary adversarial control training checks on tiny synthetic rows."""

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.fit_c import fit_c_stage
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import (
    FitData, TrainConfig, fit_arm, make_batch_plan,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler, HardCodeModel


def _data():
    rng = np.random.default_rng(22)
    n = 20
    h0 = rng.normal(size=(n, 4)).astype(np.float32)
    h1 = rng.normal(size=(n, 4)).astype(np.float32)
    y0 = (h0[:, 0] > 0).astype(np.int64)
    y1 = np.arange(n, dtype=np.int64) % 6
    t0 = np.full((n, 2), 0.15, dtype=np.float32)
    t0[np.arange(n), y0] = 0.85
    t1 = np.full((n, 6), 0.02, dtype=np.float32)
    t1[np.arange(n), y1] = 0.90
    return FitData(h0, h1, y0, y1, t0, t1, np.arange(n) % 2,
                   np.arange(n), "SYNTHETIC")


def test_zero_adversarial_coefficient_matches_task_only_continuation():
    data = _data()
    scaler = FixedScaler.fit(data.h0, data.h1)
    torch.manual_seed(7)
    initial = HardCodeModel(4, (2, 6)).state_dict()
    initial = {name: value.clone() for name, value in initial.items()}
    config = TrainConfig(epochs=1, batch_size=8, learning_rate=0.001,
                         distill_weight=0.5, privacy_weight=0.0, checkpoints=(1,))
    train, val = np.arange(16), np.arange(16, 20)
    task_data = FitData(data.h0, data.h1, data.y0, data.y1,
                        data.teacher0, data.teacher1, None,
                        data.row_ids, "SYNTHETIC")
    plan = make_batch_plan(train, seed=11, epochs=1, batch_size=8, sex=data.sex)
    task = fit_arm(task_data, config, 11, "T", initial, scaler, train, val,
                   batch_plan=plan)
    adversarial = fit_c_stage(data, config, 11, initial, scaler,
                              train_idx=train, coefficient=0.0,
                              batch_plan=plan)
    assert adversarial.optimizer_updates == task.optimizer_updates
    for name, value in task.state_dict.items():
        assert torch.equal(value, adversarial.model.state_dict()[name]), name


def test_c_fits_critics_and_keeps_same_mapper_update_count():
    data = _data()
    scaler = FixedScaler.fit(data.h0, data.h1)
    torch.manual_seed(3)
    initial = {name: value.clone() for name, value in HardCodeModel(4, (2, 6)).state_dict().items()}
    config = TrainConfig(epochs=2, batch_size=8, learning_rate=0.001,
                         distill_weight=0.5, privacy_weight=1.0)
    trained = fit_c_stage(data, config, 11, initial, scaler,
                          train_idx=np.arange(20), coefficient=1.0)
    assert trained.optimizer_updates == 6
    assert trained.critic_updates == 12
    assert len(trained.history) == 2
    assert any(not torch.equal(value, trained.model.state_dict()[name])
               for name, value in initial.items())

"""Data-free matched continuation tests; no real role export is opened."""

import numpy as np
import pytest
import torch

from code_repair.protection import (fit_continue, fit_sequential, fit_adversarial,
                                    calibrate_kernel_strength)
from code_witness.model import FixedScaler, HardCodeModel
from code_witness.learner import FitData, TrainConfig, make_batch_plan


def synthetic_inputs(bits):
    rng = np.random.default_rng(440)
    n = 24
    h0, h1 = (rng.normal(size=(n, 4)).astype(np.float32) for _ in range(2))
    y0 = np.arange(n, dtype=np.int64) % 2
    y1 = np.arange(n, dtype=np.int64) % 6
    t0 = np.full((n, 2), 0.5, dtype=np.float32)
    t1 = np.full((n, 6), 1 / 6, dtype=np.float32)
    sex = np.arange(n, dtype=np.int64) % 2
    protected = FitData(h0, h1, y0, y1, t0, t1, sex, np.arange(n), "SYNTHETIC")
    task = FitData(h0, h1, y0, y1, t0, t1, None, np.arange(n), "SYNTHETIC")
    scaler = FixedScaler.fit(h0, h1)
    torch.manual_seed(9)
    state = {k: v.clone() for k, v in HardCodeModel(4, (2, 6), bits).state_dict().items()}
    config = TrainConfig(epochs=1, patience=1, batch_size=8)
    plan = make_batch_plan(np.arange(n), seed=7, epochs=1, batch_size=8, sex=sex)
    return protected, task, scaler, state, config, plan


@pytest.mark.parametrize("bits", [10, 16])
def test_zero_kernel_strength_matches_task_only_for_same_hard_code_updates(bits):
    protected, task, scaler, state, config, plan = synthetic_inputs(bits)
    common = dict(config=config, seed=7, bits=bits, init_state=state,
                  scaler=scaler, batch_plan=plan)
    task_result = fit_continue(task, mode="T+", privacy_weight=0, **common)
    joint_result = fit_continue(protected, mode="J", privacy_weight=0, **common)
    local_result = fit_continue(protected, mode="L", privacy_weight=0, **common)
    assert task_result.optimizer_updates == joint_result.optimizer_updates == local_result.optimizer_updates == 3
    for name, expected in task_result.model.state_dict().items():
        assert torch.equal(expected, joint_result.model.state_dict()[name]), name
        assert torch.equal(expected, local_result.model.state_dict()[name]), name


@pytest.mark.parametrize("order,first", [("S12", 0), ("S21", 1)])
def test_sequential_first_public_model_and_calibrator_are_frozen(order, first):
    protected, _, scaler, state, config, plan = synthetic_inputs(16)
    calls = []

    def calibrate_first(model, purpose):
        calls.append((purpose, {k: v.clone() for k, v in model.state_dict().items()}))
        return {"schema": "synthetic-global-temp", "task": purpose, "alpha": 1.0}

    result = fit_sequential(protected, config=config, seed=7, bits=16,
                            init_state=state, scaler=scaler, mode=order,
                            privacy_weight=0.2, batch_plan=plan,
                            calibrate_first=calibrate_first)
    assert [purpose for purpose, _ in calls] == [first]
    for name, expected in calls[0][1].items():
        if name.startswith((f"mappers.{first}.", f"heads.{first}.")):
            assert torch.equal(expected, result.model.state_dict()[name])
    assert result.first_calibrator["alpha"] == 1.0
    assert result.first_code.shape == (24, 16)
    assert result.updates_by_purpose == (3, 3)


def test_adversarial_has_equal_mapper_steps_and_two_critic_steps_per_batch():
    protected, task, scaler, state, config, plan = synthetic_inputs(10)
    common = dict(config=config, seed=7, bits=10, init_state=state,
                  scaler=scaler, batch_plan=plan)
    task_result = fit_continue(task, mode="T+", privacy_weight=0, **common)
    c_result = fit_adversarial(protected, coefficient=0, **common)
    assert c_result.optimizer_updates == 3
    assert c_result.critic_updates == 6
    for name, expected in task_result.model.state_dict().items():
        assert torch.equal(expected, c_result.model.state_dict()[name]), name


def test_real_protection_data_cannot_fit_without_r0_gate():
    protected, _, scaler, state, config, plan = synthetic_inputs(10)
    object.__setattr__(protected, "role", "OSF_DEFENSE_FIT")
    with pytest.raises(PermissionError):
        fit_continue(protected, config=config, seed=7, bits=10,
                     init_state=state, scaler=scaler, mode="J",
                     privacy_weight=0.2, batch_plan=plan)


def test_fixed_batch_gradient_ratio_has_no_updates_and_clips_zero_signal():
    protected, _, scaler, state, config, plan = synthetic_inputs(10)
    # Identical latents yield the same hard code in both SEX groups and no
    # sensitive kernel gradient, while task labels still vary.
    protected.h0[:] = protected.h0[0]
    protected.h1[:] = protected.h1[0]
    before = {name: value.clone() for name, value in state.items()}
    receipt = calibrate_kernel_strength(protected, config=config, seed=7,
                                        bits=10, init_state=state, scaler=scaler,
                                        first_batch=plan[0][0])
    assert receipt["privacy_weight"] == 10.0
    assert receipt["parameter_updates"] == 0
    assert receipt["batch_rows"] == 8
    assert all(torch.equal(value, state[name]) for name, value in before.items())

"""L and freeze-first sequential design use the actual aligned public bits."""

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.local_sequential import (
    fit_local_or_sequential,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import (
    FitData, TrainConfig, make_batch_plan,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler, HardCodeModel


def _inputs():
    rng = np.random.default_rng(37)
    n = 20
    h0, h1 = (rng.normal(size=(n, 4)).astype(np.float32) for _ in range(2))
    y0 = np.arange(n) % 2
    y1 = np.arange(n) % 6
    t0 = np.full((n, 2), 0.5, dtype=np.float32)
    t1 = np.full((n, 6), 1 / 6, dtype=np.float32)
    data = FitData(h0, h1, y0, y1, t0, t1, np.arange(n) % 2,
                   np.arange(n), "SYNTHETIC")
    scaler = FixedScaler.fit(h0, h1)
    torch.manual_seed(2)
    warm = {name: value.clone() for name, value in HardCodeModel(4, (2, 6)).state_dict().items()}
    config = TrainConfig(epochs=1, batch_size=8, privacy_weight=0.1,
                         checkpoints=(1,))
    return data, scaler, warm, config


def test_both_sequential_orders_freeze_first_model_and_use_its_actual_codes():
    data, scaler, warm, config = _inputs()
    for mode, first in (("S12", 0), ("S21", 1)):
        result = fit_local_or_sequential(data, config, 5, warm, scaler,
                                         mode, "a" * 64)
        assert result.first_release is not None
        expected = result.first_release.recipient(first, (data.h0, data.h1)[first])["code"]
        np.testing.assert_array_equal(result.first_fit_codes, expected)
        for prefix in (f"mappers.{first}.", f"heads.{first}."):
            for name, value in result.first_release.model.state_dict().items():
                if name.startswith(prefix):
                    assert torch.equal(value, result.fit.model.state_dict()[name])
        assert result.updates_by_purpose == (3, 3)


def test_local_arm_has_same_per_purpose_updates_and_no_pair_anchor():
    data, scaler, warm, config = _inputs()
    result = fit_local_or_sequential(data, config, 5, warm, scaler, "L", "a" * 64)
    assert result.first_release is None
    assert result.first_fit_codes is None
    assert result.updates_by_purpose == (3, 3)


def test_sequential_uses_one_shared_paired_batch_plan():
    data, scaler, warm, config = _inputs()
    plan = make_batch_plan(np.arange(data.rows), seed=5, epochs=1,
                           batch_size=8, sex=data.sex)
    result = fit_local_or_sequential(data, config, 5, warm, scaler,
                                     "S12", "a" * 64, batch_plan=plan)
    assert result.updates_by_purpose == (len(plan[0]), len(plan[0]))
    wrong = (tuple(reversed(plan[0])),)
    try:
        fit_local_or_sequential(data, config, 5, warm, scaler,
                                "S12", "a" * 64, batch_plan=wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("changed FIT batch plan accepted")

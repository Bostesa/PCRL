"""Data-free correctness checks for hard transport and equal task continuation."""

import numpy as np
import pytest
import torch

from .fit import FitData, TrainConfig, calibrate_privacy_weight, fit_arm, make_batch_plan
from .model import FixedScaler, HardCodeModel, HardCodeRelease
from .service import load_service, save_service


def _data(sex: bool) -> FitData:
    rng = np.random.default_rng(7)
    n = 32
    h0 = rng.normal(size=(n, 16)).astype(np.float32)
    h1 = rng.normal(size=(n, 16)).astype(np.float32)
    y0 = np.arange(n) % 2
    y1 = np.arange(n) % 6
    p0 = np.eye(2)[y0] * 0.7 + 0.3 / 2
    p1 = np.eye(6)[y1] * 0.7 + 0.3 / 6
    return FitData(h0, h1, y0, y1, p0, p1,
                   (np.arange(n) % 2) if sex else None,
                   np.arange(n, dtype=np.int64), "SYNTHETIC")


def test_hard_st_forward_gradient_and_code_only_wire(tmp_path):
    torch.manual_seed(2)
    model = HardCodeModel(16, (2, 6))
    scaler = FixedScaler.fit(_data(False).h0, _data(False).h1)
    x = torch.randn(5, 16)
    code_st, logits = model(0, x, True)
    code_hard, _ = model(0, x, False)
    assert torch.equal(code_st, code_hard)
    assert torch.equal(code_hard, (model.mappers[0](x) >= 0.5).float())
    logits.sum().backward()
    assert model.mappers[0].layer1.weight.grad.abs().sum() > 0

    cal = ({"schema": "pcrl-portfolio-global-temp-v1", "task": "income", "alpha": 1.3},
           {"schema": "pcrl-portfolio-global-temp-v1", "task": "occupation", "alpha": 0.8})
    release = HardCodeRelease(model, scaler, "test-v1", "a" * 64, cal)
    wire = release.recipient(0, _data(False).h0[:5])
    assert set(wire) == {"code", "probabilities", "decision", "metadata"}
    assert wire["code"].dtype == np.uint8 and wire["code"].shape == (5, 10)
    assert wire["probabilities"].dtype == np.float64
    np.testing.assert_allclose(wire["probabilities"].sum(1), 1, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(wire["probabilities"], release.probabilities_from_code(0, wire["code"]))
    assert wire["metadata"]["bit_order"] == "c0_least_significant"
    packed = (wire["code"].astype(np.uint16) * (1 << np.arange(10))).sum(axis=1)
    assert np.all(packed < 1024)

    package = tmp_path / "private_model"
    receipt = save_service(package, release, source_seed=0, arm_id="T",
                           role_manifest_sha256="b" * 64, fit_receipt={"rows": 32})
    restored = load_service(package / "manifest.json",
                            expected_manifest_sha256=receipt["manifest_sha256"],
                            expected_source_sha256="a" * 64)
    replay = restored.recipient(0, _data(False).h0[:5])
    for key in ("code", "probabilities", "decision"):
        np.testing.assert_array_equal(wire[key], replay[key])
    with pytest.raises(ValueError):
        load_service(package / "manifest.json", expected_manifest_sha256=receipt["manifest_sha256"],
                     expected_source_sha256="c" * 64)


def test_zero_privacy_same_task_continuation_and_gradient_receipt():
    task = _data(False)
    protected = _data(True)
    train, val = np.arange(24), np.arange(24, 32)
    plan = make_batch_plan(train, seed=3, epochs=5, batch_size=8, sex=protected.sex)
    config_t = TrainConfig(epochs=5, batch_size=8, privacy_weight=0, checkpoints=(5,))
    torch.manual_seed(9)
    state = HardCodeModel(16, (2, 6)).state_dict()
    scaler = FixedScaler.fit(task.h0, task.h1)
    t = fit_arm(task, config_t, 3, "T", state, scaler, train, val, batch_plan=plan)
    k0 = fit_arm(protected, config_t, 3, "D-K", state, scaler, train, val, batch_plan=plan)
    assert t.optimizer_updates == k0.optimizer_updates == 15
    assert t.selected_epoch == k0.selected_epoch == 5
    assert all(torch.equal(t.state_dict[key], k0.state_dict[key]) for key in t.state_dict)
    receipt = calibrate_privacy_weight(protected, t.state_dict, scaler, 3,
                                       train, val, batch_size=8)
    assert 0.1 <= receipt["privacy_weight"] <= 10
    assert receipt["parameter_updates"] == 0
    with pytest.raises(ValueError):
        fit_arm(protected, config_t, 3, "T", state, scaler, train, val)

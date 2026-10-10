"""F reference checks: full unchanged scaled latent is the public feature."""

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.unchanged import (
    ContinuousHead,
    UnchangedLatentRelease,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler


def test_unchanged_release_exposes_scaled_latent_and_derived_probabilities():
    h0 = np.array([[0., 2.], [1., 2.], [2., 2.]], dtype=np.float32)
    h1 = np.array([[3., 4.], [4., 4.], [5., 4.]], dtype=np.float32)
    scaler = FixedScaler.fit(h0, h1)
    torch.manual_seed(5)
    heads = (ContinuousHead(2, 2), ContinuousHead(2, 3))
    release = UnchangedLatentRelease(heads, scaler, "f-v1", "a" * 64)
    wire = release.recipient(0, h0)
    np.testing.assert_array_equal(wire["features"], scaler.transform(0, h0))
    np.testing.assert_array_equal(wire["decision"], wire["probabilities"].argmax(1))
    np.testing.assert_allclose(wire["probabilities"].sum(1), 1, atol=1e-6)
    assert set(wire) == {"features", "probabilities", "decision", "metadata"}
    assert "code" not in wire
    assert release.heads[0].training is False


def test_unchanged_release_rejects_bad_source_or_nonfinite_latent():
    scaler = FixedScaler.fit(np.array([[0.], [1.]]), np.array([[2.], [3.]]))
    heads = (ContinuousHead(1, 2), ContinuousHead(1, 2))
    release = UnchangedLatentRelease(heads, scaler, "f-v1", "b" * 64)
    try:
        release.recipient(0, np.array([[np.nan]]))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite latent accepted")


def test_f_calibration_is_applied_to_the_public_head_probability():
    scaler = FixedScaler.fit(np.array([[0.], [1.]]), np.array([[2.], [3.]]))
    heads = (ContinuousHead(1, 2), ContinuousHead(1, 6))
    records = ({"schema": "pcrl-portfolio-global-temp-v1", "task": "income", "alpha": 2.0},
               {"schema": "pcrl-portfolio-global-temp-v1", "task": "occupation", "alpha": 1.0})
    raw = UnchangedLatentRelease(heads, scaler, "f-v1", "c" * 64)
    calibrated = UnchangedLatentRelease(heads, scaler, "f-v1", "c" * 64,
                                        calibrators=records)
    h = np.array([[0.], [1.]], dtype=np.float32)
    p = raw.recipient(0, h)["probabilities"]
    q = calibrated.recipient(0, h)["probabilities"]
    expected = p ** 2
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(q, expected, atol=1e-12)
    assert calibrated.recipient(0, h)["metadata"]["calibration_family"] == "H-GLOBAL-TEMP"

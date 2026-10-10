"""Synthetic restore tests for the private deployment package."""

import numpy as np
import pytest
import torch

from code_witness.model import FixedScaler, HardCodeModel, HardCodeRelease
from code_witness.service import load_service, save_service


def _release(bits=16):
    torch.manual_seed(13)
    model = HardCodeModel(16, (2, 6), bits)
    scaler = FixedScaler(np.zeros((2, 16), np.float32), np.ones((2, 16), np.float32))
    cal = tuple({"schema": "pcrl-portfolio-global-temp-v1", "task": task,
                 "alpha": 1.0} for task in ("income", "occupation"))
    return HardCodeRelease(model, scaler, "test", "a" * 64, cal)


def test_private_package_restores_code_derived_release(tmp_path):
    release = _release()
    saved = save_service(tmp_path / "package", release, source_seed=0,
                         arm_id="T0", role_manifest_sha256="b" * 64,
                         source_schema_sha256="c" * 64,
                         fit_receipt={"selected_epoch": 4})
    restored = load_service(tmp_path / "package/manifest.json",
                            expected_manifest_sha256=saved["manifest_sha256"],
                            expected_source_sha256="a" * 64, expected_bits=16)
    x = np.random.default_rng(1).normal(size=(3, 16)).astype(np.float32)
    a, b = release.recipient(1, x), restored.recipient(1, x)
    for key in ("code", "probabilities", "decision"):
        assert np.array_equal(a[key], b[key])
    assert saved["model_sha256"] == restored.model_sha256
    assert len(saved["scaler_sha256"]) == 64
    assert saved["scaler_sha256"] == restored.scaler_sha256
    with pytest.raises(ValueError):
        load_service(tmp_path / "package/manifest.json",
                     expected_manifest_sha256=saved["manifest_sha256"],
                     expected_source_sha256="a" * 64, expected_bits=10)

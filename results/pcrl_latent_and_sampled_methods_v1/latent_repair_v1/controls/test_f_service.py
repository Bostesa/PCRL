"""Owner-only F diagnostic package roundtrip without per-person payloads."""

import json

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.f_service import (
    load_f_service,
    save_f_service,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.unchanged import (
    ContinuousHead,
    UnchangedLatentRelease,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler


def test_f_service_roundtrip_is_exact_and_contains_no_people(tmp_path):
    h0 = np.array([[0., 1.], [1., 2.]], dtype=np.float32)
    h1 = np.array([[2., 3.], [3., 4.]], dtype=np.float32)
    scaler = FixedScaler.fit(h0, h1)
    torch.manual_seed(31)
    records = ({"schema": "pcrl-portfolio-global-temp-v1", "task": "income", "alpha": 1.5},
               {"schema": "pcrl-portfolio-global-temp-v1", "task": "occupation", "alpha": 0.9})
    release = UnchangedLatentRelease((ContinuousHead(2, 2), ContinuousHead(2, 6)),
                                     scaler, "F-s0-v1", "a" * 64, records)
    receipt = save_f_service(tmp_path / "F", release, source_seed=0,
                             role_manifest_sha256="b" * 64,
                             fit_receipt={"selected_epochs": [5, 10]})
    restored = load_f_service(tmp_path / "F" / "manifest.json",
                              expected_manifest_sha256=receipt["manifest_sha256"],
                              expected_source_sha256="a" * 64)
    for purpose, h in ((0, h0), (1, h1)):
        left, right = release.recipient(purpose, h), restored.recipient(purpose, h)
        for field in ("features", "probabilities", "decision"):
            np.testing.assert_array_equal(left[field], right[field])
    manifest = json.loads((tmp_path / "F" / "manifest.json").read_text())
    assert not any(key in manifest for key in ("row_id", "unit", "SEX", "labels", "teacher"))

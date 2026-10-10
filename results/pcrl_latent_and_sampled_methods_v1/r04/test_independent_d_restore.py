"""Tiny end-to-end R04 restore fixture made without candidate code."""

import hashlib
import json

import numpy as np
import pytest
import torch

from independent_d_restore import verify_d_raw_to_wire


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_raw_encoder_hard_threshold_head_and_wire(tmp_path):
    source = {}
    for purpose in (0, 1):
        for layer, width_in, width_out in ((0, 83, 64), (2, 64, 64), (4, 64, 16)):
            source[f"enc.{purpose}.{layer}.weight"] = torch.zeros((width_out, width_in))
            source[f"enc.{purpose}.{layer}.bias"] = torch.zeros(width_out)
    source_path = tmp_path / "source.pt"
    torch.save(source, source_path)
    source_sha = _sha(source_path)
    arrays = {"scaler.center": np.zeros((2, 16), dtype=np.float32),
              "scaler.divisor": np.ones((2, 16), dtype=np.float32)}
    for purpose in (0, 1):
        classes = (2, 6)[purpose]
        for kind, first_in, second_out in (("mappers", 16, 10), ("heads", 10, classes)):
            arrays[f"{kind}.{purpose}.layer1.weight"] = np.zeros((64, first_in), dtype=np.float32)
            arrays[f"{kind}.{purpose}.layer1.bias"] = np.zeros(64, dtype=np.float32)
            arrays[f"{kind}.{purpose}.layer2.weight"] = np.zeros((second_out, 64), dtype=np.float32)
            arrays[f"{kind}.{purpose}.layer2.bias"] = np.zeros(second_out, dtype=np.float32)
    weights = tmp_path / "weights.npz"
    np.savez_compressed(weights, **arrays)
    calibrators = [{"schema": "pcrl-portfolio-global-temp-v1", "task": task, "alpha": 1.0}
                   for task in ("income", "occupation")]
    manifest = {"schema": "latent-repair-service-package-v1", "source_sha256": source_sha,
                "classes": [2, 6], "latent_dim": 16, "bit_count": 10,
                "bit_order": "c0_least_significant", "calibration_family": "H-GLOBAL-TEMP",
                "weights_sha256": _sha(weights), "weights_keys": sorted(arrays),
                "calibrators": calibrators, "version": "fixture",
                "model_sha256": "m" * 64, "head_sha256": ["h" * 64, "j" * 64]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    names = tuple(f"f{index}" for index in range(83))
    name_sha = hashlib.sha256("\n".join(names).encode()).hexdigest()
    cal_sha = hashlib.sha256((json.dumps(calibrators[0], sort_keys=True,
        separators=(",", ":")) + "\n").encode()).hexdigest()
    wire = {"code": np.ones((2, 10), dtype=np.uint8),
            "probabilities": np.full((2, 2), .5), "decision": np.zeros(2, dtype=np.int64),
            "metadata": {"schema": "latent-repair-hardbits-v1", "purpose": 0,
                         "bits": 10, "bit_order": "c0_least_significant",
                         "capacity_states": 1024, "version": "fixture",
                         "source_sha256": source_sha, "model_sha256": "m" * 64,
                         "head_sha256": "h" * 64,
                         "calibration_family": "H-GLOBAL-TEMP", "calibrator_sha256": cal_sha}}
    args = dict(manifest_path=manifest_path, manifest_sha256=_sha(manifest_path),
                source_checkpoint=source_path, source_sha256=source_sha,
                x83=np.zeros((2, 83), dtype=np.float32), feature_names=names,
                feature_names_sha256=name_sha, purpose=0, actual_wire=wire)
    assert verify_d_raw_to_wire(**args)["status"] == "INDEPENDENT_RAW_TO_WIRE_PASS"
    with pytest.raises(ValueError):
        verify_d_raw_to_wire(**(args | {"feature_names": names[::-1]}))
    with pytest.raises(ValueError):
        verify_d_raw_to_wire(**(args | {"actual_wire": wire | {"latent": np.zeros((2, 16))}}))

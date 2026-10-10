"""Independent frozen-U to continuous F wire reconstruction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .independent_d_restore import _affine, _sha, _softmax64, _source_latent


def verify_f_fit_scaler(*, manifest_path: str | Path, manifest_sha256: str,
                        source_checkpoint: str | Path, source_sha256: str,
                        fit_x83: np.ndarray, tolerance: float = 2e-6) -> dict:
    path = Path(manifest_path)
    if _sha(path) != manifest_sha256:
        raise ValueError("F manifest byte SHA changed")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "latent-repair-f-service-package-v1"
            or manifest.get("source_sha256") != source_sha256):
        raise ValueError("F FIT scaler package/source differs")
    weights = path.with_name("weights.npz")
    if _sha(weights) != manifest["weights_sha256"]:
        raise ValueError("F FIT scaler weights SHA changed")
    with np.load(weights, allow_pickle=False) as archive:
        center = np.asarray(archive["scaler.center"])
        divisor = np.asarray(archive["scaler.divisor"])
    if center.shape != (2, 16) or divisor.shape != (2, 16):
        raise ValueError("F FIT scaler dimensions differ")
    discrepancy = []
    for purpose in (0, 1):
        latent = _source_latent(Path(source_checkpoint), source_sha256,
                                purpose, fit_x83)
        expected_center = latent.astype(np.float64).mean(axis=0).astype(np.float32)
        expected_divisor = np.maximum(latent.astype(np.float64).std(axis=0, ddof=0),
                                      1.0).astype(np.float32)
        mismatch = max(float(np.max(np.abs(center[purpose] - expected_center))),
                       float(np.max(np.abs(divisor[purpose] - expected_divisor))))
        if mismatch > tolerance:
            raise ValueError("F scaler differs from FIT-only population moments")
        discrepancy.append(mismatch)
    return {"status": "F_FIT_ONLY_SCALER_PASS", "fit_rows": len(fit_x83),
            "max_abs_difference_by_purpose": discrepancy}


def verify_f_raw_to_wire(*, manifest_path: str | Path, manifest_sha256: str,
                         source_checkpoint: str | Path, source_sha256: str,
                         x83: np.ndarray, purpose: int, actual_features: np.ndarray,
                         actual_probabilities: np.ndarray,
                         actual_decision: np.ndarray) -> dict:
    """Rebuild scaled continuous latent, new head, CAL and exact task wire."""
    if purpose not in (0, 1):
        raise ValueError("F purpose outside two frozen tasks")
    path = Path(manifest_path)
    if _sha(path) != manifest_sha256:
        raise ValueError("F manifest byte SHA changed")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "latent-repair-f-service-package-v1"
            or manifest.get("arm_id") != "F"
            or manifest.get("source_sha256") != source_sha256
            or manifest.get("latent_dim") != 16
            or manifest.get("classes") != [2, 6]
            or manifest.get("calibration_family") != "H-GLOBAL-TEMP"):
        raise ValueError("F manifest/source contract changed")
    weights_path = path.with_name("weights.npz")
    if _sha(weights_path) != manifest["weights_sha256"]:
        raise ValueError("F weights byte SHA changed")
    expected = {f"head{i}.{layer}.{part}"
                for i in (0, 1) for layer in ("layer1", "layer2")
                for part in ("weight", "bias")}
    expected |= {"scaler.center", "scaler.divisor"}
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != expected or manifest.get("weights_keys") != sorted(expected):
            raise ValueError("F tensor inventory changed")
        arrays = {key: np.asarray(archive[key]).copy() for key in expected}
    shapes = {"scaler.center": (2, 16), "scaler.divisor": (2, 16)}
    for i in (0, 1):
        shapes.update({f"head{i}.layer1.weight": (64, 16),
                       f"head{i}.layer1.bias": (64,),
                       f"head{i}.layer2.weight": ((2, 6)[i], 64),
                       f"head{i}.layer2.bias": ((2, 6)[i],)})
    for key, shape in shapes.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"F tensor invalid: {key}")
    if np.any(arrays["scaler.divisor"] < 1):
        raise ValueError("F scaler amplifies frozen latent")
    latent = _source_latent(Path(source_checkpoint), source_sha256, purpose, x83)
    scaled = ((latent - arrays["scaler.center"][purpose])
              / arrays["scaler.divisor"][purpose]).astype(np.float32)
    transmitted = np.asarray(actual_features)
    if transmitted.shape != scaled.shape or not np.array_equal(transmitted, scaled):
        raise ValueError("F transmitted feature differs from frozen scaled latent")
    with torch.no_grad():
        one = _affine(torch.from_numpy(scaled), arrays, f"head{purpose}.layer1")
        logits = _affine(F.gelu(one), arrays, f"head{purpose}.layer2").numpy()
    probability = _softmax64(logits)
    calibration = manifest["calibrators"][purpose]
    if (calibration.get("schema") != "pcrl-portfolio-global-temp-v1"
            or calibration.get("task") != ("income", "occupation")[purpose]):
        raise ValueError("F CAL record changed")
    alpha = float(calibration["alpha"])
    if not 0.25 <= alpha <= 4:
        raise ValueError("F CAL alpha outside inherited bounds")
    clipped = np.maximum(probability, 1e-12)
    clipped /= clipped.sum(axis=1, keepdims=True)
    calibrated = _softmax64(alpha * np.log(clipped))
    actual = np.asarray(actual_probabilities, dtype=np.float64)
    maximum_difference = float(np.max(np.abs(calibrated - actual)))
    if (actual.shape != calibrated.shape or maximum_difference > 2e-7
            or not np.array_equal(np.asarray(actual_decision), np.argmax(calibrated, axis=1))):
        raise ValueError("F code-head/CAL task output differs")
    return {"status": "INDEPENDENT_F_RAW_TO_WIRE_PASS", "rows": len(x83),
            "manifest_sha256": manifest_sha256, "source_sha256": source_sha256,
            "maximum_probability_difference": maximum_difference,
            "features_sha256": hashlib.sha256(np.ascontiguousarray(scaled).tobytes()).hexdigest()}

"""R04 reconstruction of frozen-encoder D wire from sealed tensors.

No candidate model/deployment or runner endpoint module is imported. PyTorch
is used as the ordinary tensor serialization/float32 math framework. Actual
role arrays and model files may be opened only in an R03 verification unit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .independent_release import verify_d_hard_code_wire
except ImportError:  # The data-free pytest files are also run as loose modules.
    from independent_release import verify_d_hard_code_wire


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _affine(x: torch.Tensor, arrays: dict, prefix: str) -> torch.Tensor:
    return F.linear(x, torch.from_numpy(arrays[f"{prefix}.weight"]),
                    torch.from_numpy(arrays[f"{prefix}.bias"]))


def _softmax64(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values -= np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _manifest_and_tensors(manifest_path: Path, manifest_sha256: str,
                          source_sha256: str) -> tuple[dict, dict]:
    if _sha(manifest_path) != manifest_sha256:
        raise ValueError("D service manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "latent-repair-service-package-v1"
            or manifest.get("source_sha256") != source_sha256
            or manifest.get("classes") != [2, 6]
            or manifest.get("latent_dim") != 16 or manifest.get("bit_count") != 10
            or manifest.get("bit_order") != "c0_least_significant"
            or manifest.get("calibration_family") != "H-GLOBAL-TEMP"):
        raise ValueError("D service manifest conflicts with source/contract")
    weights = manifest_path.with_name("weights.npz")
    if _sha(weights) != manifest["weights_sha256"]:
        raise ValueError("D weights SHA mismatch")
    expected = {f"{kind}.{purpose}.{layer}.{part}"
                for kind in ("mappers", "heads") for purpose in (0, 1)
                for layer in ("layer1", "layer2") for part in ("weight", "bias")}
    expected |= {"scaler.center", "scaler.divisor"}
    with np.load(weights, allow_pickle=False) as archive:
        if set(archive.files) != expected or manifest.get("weights_keys") != sorted(expected):
            raise ValueError("D tensor inventory mismatch")
        arrays = {name: np.asarray(archive[name]).copy() for name in expected}
    shapes = {"scaler.center": (2, 16), "scaler.divisor": (2, 16)}
    for purpose in (0, 1):
        classes = (2, 6)[purpose]
        shapes.update({
            f"mappers.{purpose}.layer1.weight": (64, 16),
            f"mappers.{purpose}.layer1.bias": (64,),
            f"mappers.{purpose}.layer2.weight": (10, 64),
            f"mappers.{purpose}.layer2.bias": (10,),
            f"heads.{purpose}.layer1.weight": (64, 10),
            f"heads.{purpose}.layer1.bias": (64,),
            f"heads.{purpose}.layer2.weight": (classes, 64),
            f"heads.{purpose}.layer2.bias": (classes,),
        })
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"D tensor shape/dtype/finite mismatch: {name}")
    if np.any(arrays["scaler.divisor"] < 1):
        raise ValueError("D scaler would amplify a latent coordinate")
    return manifest, arrays


def _source_latent(source_checkpoint: Path, source_sha256: str,
                   purpose: int, x83: np.ndarray) -> np.ndarray:
    if _sha(source_checkpoint) != source_sha256:
        raise ValueError("frozen source checkpoint changed")
    x = np.asarray(x83)
    if x.ndim != 2 or x.shape[1] != 83 or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("pinned finite float32 83-predictor input required")
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("source checkpoint state missing")
    with torch.no_grad():
        value = torch.from_numpy(np.ascontiguousarray(x))
        for index in (0, 2, 4):
            prefix = f"enc.{purpose}.{index}"
            weight, bias = state[f"{prefix}.weight"], state[f"{prefix}.bias"]
            if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                raise ValueError("source encoder dtype differs")
            value = F.linear(value, weight, bias)
            if index != 4:
                value = F.relu(value)
    result = value.numpy()
    if result.shape != (len(x), 16) or not np.isfinite(result).all():
        raise ValueError("source latent shape/finite mismatch")
    return result


def _code_from_latent(arrays: dict, purpose: int, latent: np.ndarray) -> np.ndarray:
    scaled = ((latent - arrays["scaler.center"][purpose])
              / arrays["scaler.divisor"][purpose]).astype(np.float32)
    with torch.no_grad():
        value = F.relu(_affine(torch.from_numpy(scaled), arrays,
                               f"mappers.{purpose}.layer1"))
        soft = torch.sigmoid(_affine(value, arrays, f"mappers.{purpose}.layer2"))
    return (soft.numpy() >= 0.5).astype(np.uint8)


def _code_head_table(arrays: dict, purpose: int, alpha: float) -> np.ndarray:
    code = np.asarray([[(index >> bit) & 1 for bit in range(10)]
                       for index in range(1024)], dtype=np.float32)
    with torch.no_grad():
        first = _affine(torch.from_numpy(code), arrays, f"heads.{purpose}.layer1")
        second = _affine(F.gelu(first), arrays, f"heads.{purpose}.layer2")
    probability = _softmax64(second.numpy())
    if alpha != 1:
        clipped = np.maximum(probability, 1e-12)
        clipped /= clipped.sum(axis=1, keepdims=True)
        probability = _softmax64(alpha * np.log(clipped))
    return probability


def verify_d_raw_to_wire(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    source_checkpoint: str | Path,
    source_sha256: str,
    x83: np.ndarray,
    feature_names: tuple[str, ...],
    feature_names_sha256: str,
    purpose: int,
    actual_wire: dict,
) -> dict:
    """Independently replay actual code, code-only output and fixed metadata."""
    if purpose not in (0, 1):
        raise ValueError("invalid purpose")
    if (len(feature_names) != 83 or len(set(feature_names)) != 83
            or hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
            != feature_names_sha256):
        raise ValueError("wrong or reordered deployment schema")
    path = Path(manifest_path)
    manifest, arrays = _manifest_and_tensors(path, manifest_sha256, source_sha256)
    latent = _source_latent(Path(source_checkpoint), source_sha256, purpose, x83)
    code = _code_from_latent(arrays, purpose, latent)
    if not np.array_equal(code, np.asarray(actual_wire.get("code"))):
        raise ValueError("restored source/scaler/mapper produces different hard code")
    calibrator = manifest["calibrators"][purpose]
    if (calibrator.get("schema") != "pcrl-portfolio-global-temp-v1"
            or calibrator.get("task") != ("income", "occupation")[purpose]
            or not 0.25 <= float(calibrator.get("alpha", float("nan"))) <= 4):
        raise ValueError("calibrator identity/value invalid")
    table = _code_head_table(arrays, purpose, float(calibrator["alpha"]))
    expected_meta = {
        "schema": "latent-repair-hardbits-v1", "purpose": purpose, "bits": 10,
        "bit_order": "c0_least_significant", "capacity_states": 1024,
        "version": manifest["version"], "source_sha256": source_sha256,
        "model_sha256": manifest["model_sha256"],
        "head_sha256": manifest["head_sha256"][purpose],
        "calibration_family": "H-GLOBAL-TEMP",
        "calibrator_sha256": hashlib.sha256((json.dumps(calibrator, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest(),
    }
    if actual_wire.get("metadata") != expected_meta:
        raise ValueError("public model/version/calibration metadata mismatch")
    report = verify_d_hard_code_wire(actual_wire, table, bit_order="little", tolerance=2e-7)
    return {"status": "INDEPENDENT_RAW_TO_WIRE_PASS", "rows": report["rows"],
            "occupied_states": report["occupied_states"],
            "entropy_bits": report["entropy_bits"],
            "maximum_probability_difference": report["max_output_difference"],
            "manifest_sha256": manifest_sha256, "source_sha256": source_sha256}


def verify_fit_role_scaler(
    *, manifest_path: str | Path, manifest_sha256: str,
    source_checkpoint: str | Path, source_sha256: str,
    fit_x83: np.ndarray, tolerance: float = 2e-6,
) -> dict:
    """Recompute both FIT-only center and non-amplifying population scales."""
    _, arrays = _manifest_and_tensors(Path(manifest_path), manifest_sha256, source_sha256)
    x = np.asarray(fit_x83)
    if x.shape != (15434, 83) or x.dtype != np.float32:
        raise ValueError("exact admitted FIT feature shape/dtype required")
    discrepancy = []
    for purpose in (0, 1):
        latent = _source_latent(Path(source_checkpoint), source_sha256, purpose, x)
        center = latent.astype(np.float64).mean(axis=0).astype(np.float32)
        scale = np.maximum(latent.astype(np.float64).std(axis=0, ddof=0), 1.0).astype(np.float32)
        mismatch = max(float(np.max(np.abs(center - arrays["scaler.center"][purpose]))),
                       float(np.max(np.abs(scale - arrays["scaler.divisor"][purpose]))))
        if mismatch > tolerance:
            raise ValueError("service scaler differs from FIT-only population moments")
        discrepancy.append(mismatch)
    return {"status": "FIT_ONLY_SCALER_PASS", "fit_rows": len(x),
            "max_abs_difference_by_purpose": discrepancy}

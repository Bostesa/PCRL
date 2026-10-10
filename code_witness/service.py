"""Hash-pinned private package for exact hard-code deployment and restore."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .model import FixedScaler, HardCodeModel, HardCodeRelease


SCHEMA = "code-witness-service-package-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, writer) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def save_service(directory: Path, release: HardCodeRelease,
                 *, source_seed: int, arm_id: str, role_manifest_sha256: str,
                 fit_receipt: dict) -> dict:
    """Save fixed parameters only; no per-person arrays, labels or teacher scores."""
    path = Path(directory)
    if path.exists():
        raise FileExistsError(path)
    if (not path.parent.is_dir() or not arm_id or source_seed not in (0, 1)
            or len(role_manifest_sha256) != 64 or release.calibrators is None):
        raise ValueError("pinned source, CAL and private parent required")
    path.mkdir(mode=0o700)
    state = release.model.state_dict()
    arrays = {name: value.detach().cpu().numpy().copy() for name, value in state.items()}
    arrays["scaler.center"] = release.scaler.center.copy()
    arrays["scaler.divisor"] = release.scaler.divisor.copy()
    weights = path / "weights.npz"
    _write_once(weights, lambda stream: np.savez_compressed(stream, **arrays))
    manifest = {
        "schema": SCHEMA, "source_seed": source_seed, "arm_id": arm_id,
        "version": release.version, "source_sha256": release.source_sha256,
        "model_sha256": release.model_sha256,
        "head_sha256": list(release.head_sha256),
        "role_manifest_sha256": role_manifest_sha256,
        "weights_sha256": sha256_file(weights), "weights_keys": sorted(arrays),
        "classes": list(release.model.classes),
        "latent_dim": release.model.latent_dim,
        "bit_count": release.model.bits_count,
        "bit_order": "c0_least_significant",
        "scaler_formula": "(h-mean)/max(population_std,1.0) on OSF_DEFENSE_FIT",
        "calibration_family": "H-GLOBAL-TEMP",
        "calibrators": list(release.calibrators),
        "fit_receipt": fit_receipt,
        "payload_scope": "fixed model/scaler/calibrators and aggregate fit receipt only",
    }
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    manifest_path = path / "manifest.json"
    _write_once(manifest_path, lambda stream: stream.write(payload))
    return {"manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "weights_sha256": manifest["weights_sha256"],
            "model_sha256": release.model_sha256}


def load_service(manifest_path: Path, *, expected_manifest_sha256: str,
                 expected_source_sha256: str, expected_bits: int) -> HardCodeRelease:
    path = Path(manifest_path)
    if sha256_file(path) != expected_manifest_sha256 or path.name != "manifest.json":
        raise ValueError("service manifest byte pin/name mismatch")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != SCHEMA
            or manifest.get("source_sha256") != expected_source_sha256
            or manifest.get("latent_dim") != 16
            or manifest.get("classes") != [2, 6]
            or manifest.get("bit_count") != expected_bits
            or expected_bits not in (10, 16)
            or manifest.get("bit_order") != "c0_least_significant"
            or manifest.get("calibration_family") != "H-GLOBAL-TEMP"):
        raise ValueError("service package/source identity mismatch")
    weights = path.with_name("weights.npz")
    if sha256_file(weights) != manifest.get("weights_sha256"):
        raise ValueError("service weight byte pin mismatch")
    model = HardCodeModel(16, (2, 6), expected_bits)
    expected = set(model.state_dict()) | {"scaler.center", "scaler.divisor"}
    with np.load(weights, allow_pickle=False) as archive:
        if set(archive.files) != expected or sorted(expected) != manifest.get("weights_keys"):
            raise ValueError("service tensor inventory differs")
        arrays = {name: np.asarray(archive[name]).copy() for name in expected}
    state = model.state_dict()
    for name, template in state.items():
        value = arrays[name]
        if value.shape != tuple(template.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("service tensor shape/dtype/value differs")
    model.load_state_dict({name: torch.from_numpy(arrays[name]) for name in state}, strict=True)
    scaler = FixedScaler(arrays["scaler.center"], arrays["scaler.divisor"])
    release = HardCodeRelease(model, scaler, manifest["version"],
                              expected_source_sha256, tuple(manifest["calibrators"]))
    if (release.model_sha256 != manifest.get("model_sha256")
            or list(release.head_sha256) != manifest.get("head_sha256")):
        raise ValueError("service model/head identity mismatch")
    return release

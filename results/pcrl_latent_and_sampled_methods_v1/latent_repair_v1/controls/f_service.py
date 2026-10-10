"""Hash-pinned, owner-only package for the continuous F diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.service import (
    _write_once,
    sha256_file,
)
from .unchanged import ContinuousHead, UnchangedLatentRelease


SCHEMA = "latent-repair-f-service-package-v1"


def save_f_service(directory: Path, release: UnchangedLatentRelease,
                   *, source_seed: int, role_manifest_sha256: str,
                   fit_receipt: dict) -> dict:
    """Persist fixed heads, scaler and CAL records, without any person rows."""
    path = Path(directory)
    if path.exists():
        raise FileExistsError(path)
    if (not path.parent.is_dir() or source_seed not in (0, 1)
            or len(role_manifest_sha256) != 64
            or release.calibrators is None or not isinstance(fit_receipt, dict)):
        raise ValueError("F package needs pinned source, CAL and private parent")
    path.mkdir(mode=0o700)
    arrays = {f"head{purpose}.{name}": value.detach().cpu().numpy().copy()
              for purpose, head in enumerate(release.heads)
              for name, value in head.state_dict().items()}
    arrays["scaler.center"] = release.scaler.center.copy()
    arrays["scaler.divisor"] = release.scaler.divisor.copy()
    weights = path / "weights.npz"
    _write_once(weights, lambda stream: np.savez_compressed(stream, **arrays))
    manifest = {
        "schema": SCHEMA,
        "source_seed": source_seed,
        "arm_id": "F",
        "version": release.version,
        "source_sha256": release.source_sha256,
        "role_manifest_sha256": role_manifest_sha256,
        "weights_sha256": sha256_file(weights),
        "weights_keys": sorted(arrays),
        "classes": [release.heads[0].layer2.out_features,
                    release.heads[1].layer2.out_features],
        "latent_dim": release.scaler.dim,
        "scaler_formula": "(h-mean)/max(population_std,1.0) on OSF_DEFENSE_FIT",
        "calibration_family": "H-GLOBAL-TEMP",
        "calibrators": list(release.calibrators),
        "fit_receipt": fit_receipt,
        "payload_scope": "fixed heads/scaler/calibrators and aggregate fit receipt only",
    }
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    manifest_path = path / "manifest.json"
    _write_once(manifest_path, lambda stream: stream.write(payload))
    return {"manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "weights_sha256": manifest["weights_sha256"]}


def load_f_service(manifest_path: Path, *, expected_manifest_sha256: str,
                   expected_source_sha256: str) -> UnchangedLatentRelease:
    """Restore the exact F wire after checking package and source identities."""
    path = Path(manifest_path)
    if path.name != "manifest.json" or sha256_file(path) != expected_manifest_sha256:
        raise ValueError("F manifest hash/name mismatch")
    manifest = json.loads(path.read_text())
    dim = manifest.get("latent_dim")
    if (manifest.get("schema") != SCHEMA
            or manifest.get("arm_id") != "F"
            or manifest.get("source_sha256") != expected_source_sha256
            or manifest.get("classes") != [2, 6]
            or not isinstance(dim, int) or dim < 1
            or manifest.get("calibration_family") != "H-GLOBAL-TEMP"):
        raise ValueError("F package/source identity mismatch")
    weights = path.with_name("weights.npz")
    if sha256_file(weights) != manifest.get("weights_sha256"):
        raise ValueError("F weights byte hash mismatch")
    heads = (ContinuousHead(dim, 2), ContinuousHead(dim, 6))
    expected = {f"head{purpose}.{name}"
                for purpose, head in enumerate(heads)
                for name in head.state_dict()}
    expected.update(("scaler.center", "scaler.divisor"))
    with np.load(weights, allow_pickle=False) as archive:
        if set(archive.files) != expected or manifest.get("weights_keys") != sorted(expected):
            raise ValueError("F tensor inventory mismatch")
        arrays = {name: np.asarray(archive[name]).copy() for name in expected}
    for purpose, head in enumerate(heads):
        state = head.state_dict()
        for name, template in state.items():
            value = arrays[f"head{purpose}.{name}"]
            if (value.shape != tuple(template.shape) or value.dtype != np.float32
                    or not np.isfinite(value).all()):
                raise ValueError("F head tensor shape/dtype/value mismatch")
        head.load_state_dict({name: torch.from_numpy(arrays[f"head{purpose}.{name}"])
                              for name in state}, strict=True)
    scaler = FixedScaler(arrays["scaler.center"], arrays["scaler.divisor"])
    if scaler.dim != dim:
        raise ValueError("F scaler latent dimension mismatch")
    return UnchangedLatentRelease(heads, scaler, manifest["version"],
                                  expected_source_sha256,
                                  tuple(manifest["calibrators"]))

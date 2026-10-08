"""Immutable, owner-only custody for D's shared normalization/KMeans cache.

This holds no row-level arrays, labels, latents or release predictions. The
initializer bytes and receipt bind exact source/seed/fit-row identities;
callers must additionally run ``cache.validate`` before every arm fit.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from checkpoint import _private_target, _sha256, _write_json
from controls_fit import DControlCache
from prototype import PurposeNormalizer


SCHEMA = "team-d-private-control-cache-v1"
ARRAY_KEYS = {"normalizer_mean", "normalizer_scale", "pooled_centroids",
              "purpose_centroids_0", "purpose_centroids_1"}
BINDING_KEYS = {"campaign_lock_sha256", "data_manifest_sha256",
                "controls_fit_sha256", "encoder_adapter_sha256"}


def _check_binding(binding: dict) -> dict:
    if set(binding) != BINDING_KEYS or any(
        not isinstance(binding[key], str)
        or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None
        for key in BINDING_KEYS
    ):
        raise ValueError("D cache requires exact lock/data/code binding hashes")
    return dict(binding)


def save_private_cache(cache: DControlCache, directory: Path, binding: dict,
                       *, private_root: Path, source_encoder_receipt: dict | None = None) -> dict:
    """Create a private version once; never overwrite an existing cache."""
    target = _private_target(directory, private_root)
    binding = _check_binding(binding)
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError("private cache parent must already exist")
    arrays = {
        "normalizer_mean": np.asarray(cache.normalizer.mean).copy(),
        "normalizer_scale": np.asarray(cache.normalizer.scale).copy(),
        "pooled_centroids": np.asarray(cache.pooled_centroids).copy(),
        "purpose_centroids_0": cache.purpose_centroids[0].detach().cpu().numpy().copy(),
        "purpose_centroids_1": cache.purpose_centroids[1].detach().cpu().numpy().copy(),
    }
    if set(arrays) != ARRAY_KEYS or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("invalid finite initializer arrays")
    target.mkdir(mode=0o700)
    arrays_path = target / "arrays.npz"
    fd = os.open(arrays_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {"schema": SCHEMA, "binding": binding,
                "initializer_receipt": cache.receipt,
                "source_encoder_receipt": source_encoder_receipt,
                "arrays_sha256": _sha256(arrays_path),
                "payload_scope": "normalization and three initializer codebooks only; no person arrays or labels"}
    manifest_path = target / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(target / "COMPLETE.json", {
        "schema": SCHEMA,
        "manifest_sha256": _sha256(manifest_path),
        "arrays_sha256": manifest["arrays_sha256"],
    })
    return manifest


def load_private_cache(directory: Path, expected_binding: dict,
                       *, private_root: Path) -> DControlCache:
    """Restore only after immutable byte and source-binding checks."""
    target = _private_target(directory, private_root)
    expected_binding = _check_binding(expected_binding)
    manifest_path, arrays_path = target / "manifest.json", target / "arrays.npz"
    try:
        complete = json.loads((target / "COMPLETE.json").read_text())
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("incomplete or unreadable private initializer cache") from exc
    if (complete.get("schema") != SCHEMA or manifest.get("schema") != SCHEMA
            or complete.get("manifest_sha256") != _sha256(manifest_path)
            or complete.get("arrays_sha256") != _sha256(arrays_path)
            or manifest.get("arrays_sha256") != complete["arrays_sha256"]
            or manifest.get("binding") != expected_binding):
        raise ValueError("private initializer cache hash or binding mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise ValueError("private initializer cache has unexpected array fields")
        arrays = {key: np.asarray(archive[key]).copy() for key in ARRAY_KEYS}
    if any(not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
           for a in arrays.values()):
        raise ValueError("private initializer cache has invalid arrays")
    normalizer = PurposeNormalizer(arrays["normalizer_mean"], arrays["normalizer_scale"])
    pooled = arrays["pooled_centroids"]
    purposes = (torch.from_numpy(arrays["purpose_centroids_0"]),
                torch.from_numpy(arrays["purpose_centroids_1"]))
    if (pooled.ndim != 2 or pooled.shape[1] != normalizer.dim
            or any(p.shape != pooled.shape for p in purposes)):
        raise ValueError("private initializer cache dimensions mismatch")
    return DControlCache(normalizer, pooled, purposes, manifest["initializer_receipt"])

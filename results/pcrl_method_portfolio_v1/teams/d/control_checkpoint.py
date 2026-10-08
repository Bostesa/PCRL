"""Private, immutable checkpoints for D's matched frozen-latent controls.

Only fitted parameters and aggregate receipts are saved. A completed fit is
uncalibrated; S2's held-out temperature records are separate private objects.
"""

from __future__ import annotations

import os
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from checkpoint import _check_binding, _private_target, _sha256, _write_json
from controls_fit import (ControlFitConfig, ControlFitResult, ContinuousRelease,
                          TwoCodeRelease)
from prototype import PurposeNormalizer
from trainer import _Head


SCHEMA = "team-d-private-control-fit-v1"
STATUS = "UNCALIBRATED_CONTROL_FIT"
MODES = {"LOCAL", "SEQ12", "SEQ21", "UNCHANGED", "LEACE"}
LEACE_AUDIT_KEYS = {"mean_z", "sigma_xx_used", "sigma_xz",
                    "singular_values_whitened_xz"}


def _head_arrays(heads: tuple[_Head, _Head]) -> dict[str, np.ndarray]:
    arrays = {}
    for purpose, head in enumerate(heads):
        for layer in ("l1", "l2"):
            module = getattr(head, layer)
            arrays[f"head{purpose}_{layer}_w"] = module.weight.detach().cpu().numpy().copy()
            arrays[f"head{purpose}_{layer}_b"] = module.bias.detach().cpu().numpy().copy()
    return arrays


def _arrays(fit: ControlFitResult) -> dict[str, np.ndarray]:
    release = fit.release
    arrays = {"normalizer_mean": release.normalizer.mean.copy(),
              "normalizer_scale": release.normalizer.scale.copy()}
    arrays.update(_head_arrays(release.heads))
    if isinstance(release, TwoCodeRelease):
        arrays.update({f"codebook{i}": release.codebooks[i].detach().cpu().numpy().copy()
                       for i in (0, 1)})
    elif isinstance(release, ContinuousRelease) and fit.mode == "LEACE":
        for i, mapping in enumerate(release.maps):
            if mapping is None or set(mapping.arrays) != LEACE_AUDIT_KEYS:
                raise ValueError("official LEACE map and complete audit arrays are required")
            arrays[f"map{i}_left"] = np.asarray(mapping.proj_left).copy()
            arrays[f"map{i}_right"] = np.asarray(mapping.proj_right).copy()
            arrays[f"map{i}_mean"] = np.asarray(mapping.mean_x).copy()
            for name in sorted(LEACE_AUDIT_KEYS):
                arrays[f"map{i}_audit_{name}"] = np.asarray(mapping.arrays[name]).copy()
    elif isinstance(release, ContinuousRelease):
        if release.maps != (None, None):
            raise ValueError("UNCHANGED control must have no fitted map")
    else:
        raise ValueError("control release/mode mismatch")
    if any(not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
           for a in arrays.values()):
        raise ValueError("control checkpoint requires finite numeric parameters")
    return arrays


def save_private_control_checkpoint(
    fit: ControlFitResult, directory: Path, binding: dict, *, private_root: Path,
    config: ControlFitConfig = ControlFitConfig(),
) -> dict:
    """Save one completed uncalibrated fit under an explicit owner-only root."""
    target = _private_target(directory, private_root)
    binding = _check_binding(binding)
    if fit.mode not in MODES or fit.release.kind != fit.mode:
        raise ValueError("control mode/release identity mismatch")
    if fit.seed < 0 or fit.fit_rows < 1:
        raise ValueError("invalid control fit receipt")
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError("private control parent must already exist")
    arrays = _arrays(fit)
    target.mkdir(mode=0o700)
    path = target / "weights.npz"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    map_metadata = ([dict(m.metadata) for m in fit.release.maps]
                    if fit.mode == "LEACE" else None)
    manifest = {"schema": SCHEMA, "status": STATUS, "binding": binding,
                "mode": fit.mode, "seed": int(fit.seed), "fit_rows": int(fit.fit_rows),
                "config": asdict(config), "history": fit.history,
                "map_metadata": fit.map_metadata,
                "leace_map_metadata": map_metadata,
                "weight_fields": sorted(arrays), "weights_sha256": _sha256(path),
                "payload_scope": "model and map parameters plus aggregate receipts; no person rows or labels"}
    manifest_path = target / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(target / "COMPLETE.json", {
        "schema": SCHEMA, "status": STATUS,
        "manifest_sha256": _sha256(manifest_path),
        "weights_sha256": manifest["weights_sha256"],
    })
    return manifest


def _restore_heads(arrays: dict[str, np.ndarray], dim: int,
                   hidden: int) -> tuple[_Head, _Head]:
    heads = []
    for i in (0, 1):
        w1, b1 = arrays[f"head{i}_l1_w"], arrays[f"head{i}_l1_b"]
        w2, b2 = arrays[f"head{i}_l2_w"], arrays[f"head{i}_l2_b"]
        if (w1.shape != (hidden, dim) or b1.shape != (hidden,)
                or w2.ndim != 2 or w2.shape[1] != hidden or w2.shape[0] < 2
                or b2.shape != (w2.shape[0],)):
            raise ValueError("control checkpoint head dimensions mismatch")
        head = _Head(dim, hidden, w2.shape[0])
        with torch.no_grad():
            head.l1.weight.copy_(torch.from_numpy(w1))
            head.l1.bias.copy_(torch.from_numpy(b1))
            head.l2.weight.copy_(torch.from_numpy(w2))
            head.l2.bias.copy_(torch.from_numpy(b2))
        heads.append(head)
    return heads[0], heads[1]


def load_private_control_checkpoint(
    directory: Path, expected_binding: dict, *, private_root: Path,
    expected_mode: str,
) -> ControlFitResult:
    """Verify exact bytes/binding, then restore a local or continuous control."""
    target = _private_target(directory, private_root)
    binding = _check_binding(expected_binding)
    manifest_path, weight_path = target / "manifest.json", target / "weights.npz"
    try:
        manifest = json.loads(manifest_path.read_text())
        complete = json.loads((target / "COMPLETE.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("incomplete or unreadable control checkpoint") from exc
    if (expected_mode not in MODES or manifest.get("mode") != expected_mode
            or manifest.get("schema") != SCHEMA or complete.get("schema") != SCHEMA
            or manifest.get("status") != STATUS or complete.get("status") != STATUS
            or manifest.get("binding") != binding
            or complete.get("manifest_sha256") != _sha256(manifest_path)
            or complete.get("weights_sha256") != _sha256(weight_path)
            or manifest.get("weights_sha256") != complete["weights_sha256"]):
        raise ValueError("control checkpoint hash, mode or binding mismatch")
    config = ControlFitConfig(**manifest["config"])
    with np.load(weight_path, allow_pickle=False) as source:
        if sorted(source.files) != manifest.get("weight_fields"):
            raise ValueError("control checkpoint weight fields mismatch")
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    if any(not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
           for a in arrays.values()):
        raise ValueError("invalid control checkpoint numeric arrays")
    normalizer = PurposeNormalizer(arrays["normalizer_mean"], arrays["normalizer_scale"])
    heads = _restore_heads(arrays, normalizer.dim, config.hidden)
    if expected_mode in ("LOCAL", "SEQ12", "SEQ21"):
        required = {"normalizer_mean", "normalizer_scale", "codebook0", "codebook1"}
        required.update(_head_arrays(heads))
        if set(arrays) != required:
            raise ValueError("unexpected VQ control parameters")
        books = (arrays["codebook0"], arrays["codebook1"])
        if any(b.shape != (config.codes, normalizer.dim) for b in books):
            raise ValueError("control codebook dimensions mismatch")
        release = TwoCodeRelease(normalizer, tuple(torch.from_numpy(b) for b in books),
                                 heads, expected_mode)
    else:
        maps = (None, None)
        if expected_mode == "LEACE":
            from stored_model_eval.defenses import LeaceMap
            metadata = manifest.get("leace_map_metadata")
            if not isinstance(metadata, list) or len(metadata) != 2:
                raise ValueError("official LEACE provenance missing")
            restored = []
            for i in (0, 1):
                left = arrays[f"map{i}_left"]
                right = arrays[f"map{i}_right"]
                mean = arrays[f"map{i}_mean"]
                if (left.ndim != 2 or right.ndim != 2 or
                        left.shape[0] != normalizer.dim or
                        right.shape != (left.shape[1], normalizer.dim) or
                        mean.shape != (normalizer.dim,)):
                    raise ValueError("LEACE map dimensions mismatch")
                audit = {name: arrays[f"map{i}_audit_{name}"]
                         for name in LEACE_AUDIT_KEYS}
                restored.append(LeaceMap(left, right, mean, metadata[i], audit))
            maps = (restored[0], restored[1])
        required = {"normalizer_mean", "normalizer_scale"}
        required.update(_head_arrays(heads))
        if expected_mode == "LEACE":
            for i in (0, 1):
                required.update({f"map{i}_left", f"map{i}_right", f"map{i}_mean"})
                required.update({f"map{i}_audit_{name}" for name in LEACE_AUDIT_KEYS})
        if set(arrays) != required:
            raise ValueError("unexpected continuous control parameters")
        release = ContinuousRelease(normalizer, heads, maps, expected_mode)
    return ControlFitResult(release, expected_mode, int(manifest["seed"]),
                            int(manifest["fit_rows"]), manifest["history"],
                            manifest["map_metadata"])

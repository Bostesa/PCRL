"""Versioned private Team D training checkpoints without row-level payloads.

Callers choose an authorized private directory and bind its science lock, role
export, source model and trainer hashes. This module refuses overwrites and
restores only after streaming SHA-256 checks. It stores an uncalibrated completed
fit, not a full deployed release or mid-epoch optimizer state.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from prototype import PurposeNormalizer
from trainer import DFitResult, DTrainConfig, TorchDRelease, _Head


SCHEMA = "team-d-private-uncalibrated-fit-v1"
STATUS = "UNCALIBRATED_TRAINING_FIT"
BINDING_KEYS = {"campaign_lock_sha256", "role_export_sha256",
                "source_model_sha256", "trainer_sha256"}
WEIGHT_KEYS = {
    "codebook", "normalizer_mean", "normalizer_scale",
    "head0_l1_w", "head0_l1_b", "head0_l2_w", "head0_l2_b",
    "head1_l1_w", "head1_l1_b", "head1_l2_w", "head1_l2_b",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _check_binding(binding: dict) -> dict:
    if set(binding) != BINDING_KEYS or any(
        not isinstance(binding[key], str) or len(binding[key]) != 64
        or any(ch not in "0123456789abcdef" for ch in binding[key])
        for key in BINDING_KEYS
    ):
        raise ValueError("four lowercase 64-hex source/lock binding hashes required")
    return dict(binding)


def _private_target(directory: Path, private_root: Path) -> Path:
    """Require an explicit owner-only root and a resolved descendant target."""
    try:
        root = Path(private_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError("approved private checkpoint root is unavailable") from exc
    if not root.is_dir() or (root.stat().st_mode & 0o077):
        raise ValueError("private checkpoint root must be an owner-only directory")
    target = Path(directory).resolve(strict=False)
    if target == root or not target.is_relative_to(root):
        raise ValueError("checkpoint target must be inside the approved private root")
    return target


def _write_json(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _weight_arrays(fit: DFitResult) -> dict[str, np.ndarray]:
    release = fit.release
    arrays = {
        "codebook": release.codebook.detach().cpu().numpy().copy(),
        "normalizer_mean": release.normalizer.mean.copy(),
        "normalizer_scale": release.normalizer.scale.copy(),
    }
    for i, head in enumerate(release.heads):
        for layer in ("l1", "l2"):
            module = getattr(head, layer)
            arrays[f"head{i}_{layer}_w"] = module.weight.detach().cpu().numpy().copy()
            arrays[f"head{i}_{layer}_b"] = module.bias.detach().cpu().numpy().copy()
    return arrays


def save_private_checkpoint(
    fit: DFitResult, directory: Path, binding: dict, *, private_root: Path,
) -> dict:
    """Create one new private version; never replace a prior checkpoint."""
    directory = _private_target(directory, private_root)
    binding = _check_binding(binding)
    if directory.exists():
        raise FileExistsError(directory)
    if not directory.parent.is_dir():
        raise FileNotFoundError("private checkpoint parent must already exist")
    directory.mkdir(mode=0o700)
    weights_path = directory / "weights.npz"
    fd = os.open(weights_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **_weight_arrays(fit))
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "binding": binding,
        "weights_sha256": _sha256(weights_path),
        "config": asdict(fit.config),
        "role": fit.role,
        "seed": fit.seed,
        "protected": fit.protected,
        "fit_rows": fit.fit_rows,
        "history": fit.history,
        "fit_code_change": fit.fit_code_change,
        "payload_scope": "model parameters and aggregate receipts only; no row arrays or labels",
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(directory / "COMPLETE.json", {
        "schema": SCHEMA,
        "status": STATUS,
        "manifest_sha256": _sha256(manifest_path),
        "weights_sha256": manifest["weights_sha256"],
    })
    return manifest


def load_private_checkpoint(
    directory: Path, expected_binding: dict, *, private_root: Path,
) -> DFitResult:
    """Restore an uncalibrated fit after binding and file-hash validation."""
    directory = _private_target(directory, private_root)
    expected_binding = _check_binding(expected_binding)
    complete_path = directory / "COMPLETE.json"
    manifest_path = directory / "manifest.json"
    weights_path = directory / "weights.npz"
    try:
        complete = json.loads(complete_path.read_text())
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("incomplete or unreadable private checkpoint") from exc
    if (complete.get("schema") != SCHEMA or manifest.get("schema") != SCHEMA
            or complete.get("status") != STATUS or manifest.get("status") != STATUS
            or complete.get("manifest_sha256") != _sha256(manifest_path)
            or complete.get("weights_sha256") != _sha256(weights_path)
            or manifest.get("weights_sha256") != complete["weights_sha256"]
            or manifest.get("binding") != expected_binding):
        raise ValueError("private checkpoint hash or source binding mismatch")
    with np.load(weights_path, allow_pickle=False) as source:
        if set(source.files) != WEIGHT_KEYS:
            raise ValueError("unexpected private checkpoint weight fields")
        arrays = {key: np.asarray(source[key]).copy() for key in WEIGHT_KEYS}
    if not all(np.issubdtype(a.dtype, np.number) and np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite or nonnumeric checkpoint weights")
    config = DTrainConfig(**manifest["config"])
    codebook = arrays["codebook"]
    if codebook.ndim != 2 or codebook.shape[0] != config.codes:
        raise ValueError("checkpoint codebook/config dimension mismatch")
    normalizer = PurposeNormalizer(arrays["normalizer_mean"], arrays["normalizer_scale"])
    if codebook.shape[1] != normalizer.dim:
        raise ValueError("checkpoint codebook/normalizer dimension mismatch")
    heads = []
    for i in (0, 1):
        w1, b1 = arrays[f"head{i}_l1_w"], arrays[f"head{i}_l1_b"]
        w2, b2 = arrays[f"head{i}_l2_w"], arrays[f"head{i}_l2_b"]
        if (w1.shape != (config.hidden, normalizer.dim) or b1.shape != (config.hidden,)
                or w2.ndim != 2 or w2.shape[1] != config.hidden or w2.shape[0] < 2
                or b2.shape != (w2.shape[0],)):
            raise ValueError("checkpoint head shape mismatch")
        head = _Head(normalizer.dim, config.hidden, w2.shape[0])
        with torch.no_grad():
            head.l1.weight.copy_(torch.from_numpy(w1))
            head.l1.bias.copy_(torch.from_numpy(b1))
            head.l2.weight.copy_(torch.from_numpy(w2))
            head.l2.bias.copy_(torch.from_numpy(b2))
        heads.append(head)
    release = TorchDRelease(normalizer, torch.from_numpy(codebook), (heads[0], heads[1]))
    return DFitResult(
        release=release,
        history=manifest["history"],
        role=manifest["role"],
        seed=int(manifest["seed"]),
        protected=bool(manifest["protected"]),
        fit_rows=int(manifest["fit_rows"]),
        fit_code_change=manifest["fit_code_change"],
        config=config,
    )

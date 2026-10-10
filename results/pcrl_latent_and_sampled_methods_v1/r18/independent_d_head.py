"""Independent code-only reconstruction from one hash-pinned D service package.

This module reads fixed model tensors and two CAL records, never an Adult row.
It does not import the candidate D model or its release method.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .acceptance_contract import sha256_file


def _sha_record(value: dict) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha_tensors(tensors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name])
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(str(array.shape).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


class IndependentDHead:
    """A separately loaded, byte-checked two-purpose public code head."""

    def __init__(self, manifest_path: Path, *, manifest_sha256: str,
                 weights_sha256: str, source_sha256: str, model_sha256: str):
        path = Path(manifest_path)
        if path.name != "manifest.json" or sha256_file(path) != manifest_sha256:
            raise ValueError("D service manifest SHA differs from postfit receipt")
        manifest = json.loads(path.read_text())
        if (manifest.get("schema") != "latent-repair-service-package-v1"
                or manifest.get("source_sha256") != source_sha256
                or manifest.get("model_sha256") != model_sha256
                or manifest.get("weights_sha256") != weights_sha256
                or manifest.get("classes") != [2, 6]
                or manifest.get("latent_dim") != 16
                or manifest.get("bit_count") != 10
                or manifest.get("bit_order") != "c0_least_significant"
                or manifest.get("calibration_family") != "H-GLOBAL-TEMP"
                or len(manifest.get("calibrators", [])) != 2
                or len(manifest.get("head_sha256", [])) != 2):
            raise ValueError("D service manifest identity differs")
        weights_path = path.with_name("weights.npz")
        if sha256_file(weights_path) != weights_sha256:
            raise ValueError("D service weights SHA differs from postfit receipt")
        expected_keys = {
            f"heads.{purpose}.{suffix}"
            for purpose in (0, 1)
            for suffix in ("layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias")
        }
        with np.load(weights_path, allow_pickle=False) as saved:
            if not expected_keys.issubset(saved.files):
                raise ValueError("D code head tensor inventory differs")
            heads = {
                purpose: {name: np.asarray(saved[name]).copy()
                          for name in sorted(expected_keys)
                          if name.startswith(f"heads.{purpose}.")}
                for purpose in (0, 1)
            }
        for purpose in (0, 1):
            prefix = f"heads.{purpose}."
            state = heads[purpose]
            classes = (2, 6)[purpose]
            shapes = {prefix + "layer1.weight": (64, 10),
                      prefix + "layer1.bias": (64,),
                      prefix + "layer2.weight": (classes, 64),
                      prefix + "layer2.bias": (classes,)}
            if (any(state[name].shape != shape or state[name].dtype != np.float32
                    or not np.isfinite(state[name]).all()
                    for name, shape in shapes.items())
                    or _sha_tensors(state) != manifest["head_sha256"][purpose]):
                raise ValueError("D code head source hash or tensor shape differs")
            record = manifest["calibrators"][purpose]
            if (record.get("schema") != "pcrl-portfolio-global-temp-v1"
                    or record.get("task") != ("income", "occupation")[purpose]
                    or not 0.25 <= float(record.get("alpha", float("nan"))) <= 4.0):
                raise ValueError("D global calibrator differs")
        self.manifest = manifest
        self.heads = heads

    def expected_metadata(self, purpose: int) -> dict:
        if purpose not in (0, 1):
            raise ValueError("D purpose differs")
        manifest = self.manifest
        return {
            "schema": "latent-repair-hardbits-v1", "purpose": purpose,
            "bits": 10, "bit_order": "c0_least_significant",
            "capacity_states": 1024, "version": manifest["version"],
            "source_sha256": manifest["source_sha256"],
            "model_sha256": manifest["model_sha256"],
            "head_sha256": manifest["head_sha256"][purpose],
            "calibration_family": "H-GLOBAL-TEMP",
            "calibrator_sha256": _sha_record(manifest["calibrators"][purpose]),
        }

    def probabilities(self, purpose: int, code: np.ndarray) -> np.ndarray:
        if purpose not in (0, 1):
            raise ValueError("D purpose differs")
        bits = np.asarray(code)
        if (bits.shape != (1, 10) or bits.dtype != np.uint8
                or not np.isin(bits, (0, 1)).all()):
            raise ValueError("D public code must be one ten-bit row")
        tensors = self.heads[purpose]
        prefix = f"heads.{purpose}."
        with torch.no_grad():
            x = torch.from_numpy(bits.astype(np.float32))
            hidden = F.gelu(F.linear(x,
                                     torch.from_numpy(tensors[prefix + "layer1.weight"]),
                                     torch.from_numpy(tensors[prefix + "layer1.bias"])))
            logits = F.linear(hidden,
                              torch.from_numpy(tensors[prefix + "layer2.weight"]),
                              torch.from_numpy(tensors[prefix + "layer2.bias"])
                              ).numpy().astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        probability = values / values.sum(axis=1, keepdims=True)
        alpha = float(self.manifest["calibrators"][purpose]["alpha"])
        if alpha != 1.0:
            clipped = np.maximum(probability, 1e-12)
            clipped /= clipped.sum(axis=1, keepdims=True)
            scores = alpha * np.log(clipped)
            scores -= scores.max(axis=1, keepdims=True)
            values = np.exp(scores)
            probability = values / values.sum(axis=1, keepdims=True)
        return probability

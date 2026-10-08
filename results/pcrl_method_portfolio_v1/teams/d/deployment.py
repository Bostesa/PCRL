"""Recipient-local calibrated D release from one admitted 83-column input.

The deployed head sees only the hard public code's centroid. Neither this
adapter nor its frozen encoder accepts SEX, labels, the other party's latent,
or an original teacher output at runtime.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Callable, Protocol

import numpy as np

from attack_views import validate_recipient_release
from encoder_adapter import FrozenUEncoder


def normalize_head_probabilities(probability: np.ndarray) -> np.ndarray:
    """Convert float32 Torch softmax to S2's row-normalized float64 input."""
    p = np.asarray(probability, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p.sum(axis=1) <= 0)):
        raise ValueError("D head probabilities must be finite and nonnegative")
    return p / p.sum(axis=1, keepdims=True)


def calibrator_record_hash(record: dict) -> str:
    """Exactly S2 save_calibrator's canonical JSON-plus-newline bytes."""
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


class DCalibratedDeployment:
    """Fixed head-on-code and H-GLOBAL-TEMP wire for one purpose at a time."""

    def __init__(self, encoder: FrozenUEncoder, release: "RecipientRelease",
                 calibrators: tuple[dict, dict], calibrator_sha256: tuple[str, str],
                 arm_id: str, seed: int,
                 temperature_apply: Callable[[np.ndarray, dict], np.ndarray]):
        if not arm_id or len(calibrators) != 2 or len(calibrator_sha256) != 2:
            raise ValueError("two calibrated D heads and an arm ID are required")
        for purpose, (record, sha) in enumerate(zip(calibrators, calibrator_sha256)):
            if (record.get("task") != ("income", "occupation")[purpose]
                    or record.get("arm_id") != arm_id
                    or record.get("model_seed") != int(seed)
                    or not isinstance(sha, str)
                    or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                    or calibrator_record_hash(record) != sha):
                raise ValueError("D calibrator task/arm/seed/hash binding mismatch")
        if encoder.seed != int(seed):
            raise ValueError("frozen source and D head model seeds differ")
        if not callable(temperature_apply):
            raise ValueError("a sealed S2 temperature callback is required")
        self.encoder = encoder
        self.release = release
        self.calibrators = tuple(copy.deepcopy(record) for record in calibrators)
        self.calibrator_sha256 = tuple(calibrator_sha256)
        self.arm_id = arm_id
        self.seed = int(seed)
        self.temperature_apply = temperature_apply

    def recipient(self, purpose: int, permitted_x: np.ndarray,
                  columns: tuple[str, ...], *, role: str = "DEPLOYMENT") -> dict:
        """Compute one person's party-local wire using only that party's path."""
        if purpose not in (0, 1):
            raise ValueError("purpose must be 0 or 1")
        if calibrator_record_hash(self.calibrators[purpose]) != self.calibrator_sha256[purpose]:
            raise ValueError("D calibrator mutated after binding")
        latent = self.encoder.encode_one(purpose, permitted_x, columns, role)
        raw = self.release.recipient(purpose, latent)
        head_probability = normalize_head_probabilities(raw["probabilities"])
        probability = self.temperature_apply(head_probability,
                                             self.calibrators[purpose])
        if (not np.array_equal(np.argmax(head_probability, axis=1), raw["decision"])
                or not np.array_equal(np.argmax(probability, axis=1), raw["decision"])):
            raise ValueError("float64 normalization or calibration changed head decision")
        metadata = dict(raw["metadata"])
        metadata.update({"kind": self.arm_id, "calibration": "H-GLOBAL-TEMP",
                         "calibrator_sha256": self.calibrator_sha256[purpose]})
        wire = {"code_ids": raw["code_ids"], "decoded": raw["decoded"],
                "probabilities": probability,
                "decision": np.argmax(probability, axis=1).astype(np.int64),
                "metadata": metadata}
        return validate_recipient_release(wire, len(permitted_x), purpose)


class RecipientRelease(Protocol):
    """Pooled, local, sequential or continuous D head-on-feature release."""

    def recipient(self, purpose: int, latent: np.ndarray) -> dict: ...

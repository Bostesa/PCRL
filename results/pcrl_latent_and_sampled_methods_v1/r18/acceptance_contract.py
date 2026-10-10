"""Data-free checks shared by the later queued deployment acceptance units.

This module never opens role exports, model checkpoints, or saved releases.
Its callers supply one already admitted row and the sealed public artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FEATURE_NAMES_SHA256 = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"
EXCLUDED_SOURCE_COLUMNS = frozenset({"sex", "race", "income", "occupation", "fnlwgt"})
D_FIELDS = frozenset({"code", "probabilities", "decision", "metadata"})
A_FIELDS = frozenset({"schema", "recipient", "code_id", "probability", "decision",
                      "release_id", "public_metadata"})
A_METADATA_FIELDS = frozenset({"model_id", "source_binding_sha256",
                               "canonical_bank_sha256", "mechanism_sha256",
                               "code_length_bits", "fallback", "sampling_contract"})
D_METADATA_FIELDS = frozenset({"schema", "purpose", "bits", "bit_order",
                               "capacity_states", "version", "source_sha256",
                               "model_sha256", "head_sha256",
                               "calibration_family", "calibrator_sha256"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_raw_schema(row: np.ndarray, columns: Sequence[str],
                     pinned_columns: Sequence[str],
                     pinned_sha256: str = FEATURE_NAMES_SHA256) -> None:
    """Require the exact admitted 83-column raw schema before an encoder runs."""
    names = tuple(columns)
    expected = tuple(pinned_columns)
    if len(expected) != 83 or len(set(expected)) != 83 or any(not name for name in expected):
        raise ValueError("pinned schema must have 83 distinct named predictors")
    if hashlib.sha256("\n".join(expected).encode()).hexdigest() != pinned_sha256:
        raise ValueError("pinned predictor-name digest differs")
    if names != expected:
        raise ValueError("predictor order or membership differs")
    if any(EXCLUDED_SOURCE_COLUMNS.intersection(re.split(r"[^a-z0-9]+", name.casefold()))
           for name in names):
        raise ValueError("excluded source column reached deployment")
    x = np.asarray(row)
    if x.shape != (1, 83) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("one finite float32 admitted 83-predictor row required")


def check_source_checkpoint(path: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError("source checkpoint differs from sealed manifest")


def _probability(value: Any, rows: int = 1) -> np.ndarray:
    probability = np.asarray(value)
    if (probability.ndim != 2 or probability.shape[0] != rows
            or probability.shape[1] < 2 or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)
            or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("public probability is invalid")
    return probability


def check_d_public_release(release: Mapping[str, Any], purpose: int,
                           expected_metadata: Mapping[str, Any],
                           independent_code_reconstruction: Callable[[int, np.ndarray], np.ndarray]) -> None:
    """Require ten hard bits and exactly code-derived public task output."""
    if set(release) != D_FIELDS:
        raise ValueError("D release has missing or hidden public fields")
    code = np.asarray(release["code"])
    if code.shape != (1, 10) or code.dtype != np.uint8 or not np.isin(code, (0, 1)).all():
        raise ValueError("D public code must be ten hard bits")
    metadata = release["metadata"]
    if (not isinstance(metadata, Mapping) or set(metadata) != D_METADATA_FIELDS
            or set(expected_metadata) != D_METADATA_FIELDS
            or dict(metadata) != dict(expected_metadata)
            or metadata["schema"] != "latent-repair-hardbits-v1"
            or metadata["purpose"] != purpose or metadata["bits"] != 10
            or metadata["bit_order"] != "c0_least_significant"
            or metadata["capacity_states"] != 1024
            or not isinstance(metadata["version"], str) or not metadata["version"]
            or metadata["calibration_family"] != "H-GLOBAL-TEMP"
            or any(not isinstance(metadata[key], str) or len(metadata[key]) != 64
                   or any(character not in "0123456789abcdef" for character in metadata[key])
                   for key in ("source_sha256", "model_sha256", "head_sha256",
                               "calibrator_sha256"))):
        raise ValueError("D public metadata differs from the source and capacity pin")
    probability = _probability(release["probabilities"])
    reconstructed = _probability(independent_code_reconstruction(purpose, code))
    if not np.array_equal(probability, reconstructed):
        raise ValueError("D probabilities do not come only from the public code")
    decision = np.asarray(release["decision"])
    if (decision.shape != (1,) or not np.issubdtype(decision.dtype, np.integer)
            or not np.array_equal(decision, np.argmax(probability, axis=1))):
        raise ValueError("D decision differs from the code-only public probability")


def check_a_public_release(release: Mapping[str, Any],
                           bank: Mapping[Any, tuple[Sequence[float], int]],
                           expected_recipient: int, expected_release_id: str,
                           expected_public_metadata: Mapping[str, Any]) -> None:
    """Require a globally named sampled bank entry and no fine-token fields."""
    if set(release) != A_FIELDS:
        raise ValueError("A release has missing or hidden public fields")
    if (release["schema"] != "SCV1-score-release-v1"
            or release["recipient"] != expected_recipient
            or expected_recipient not in (1, 2)
            or release["release_id"] != expected_release_id):
        raise ValueError("A release identity differs from the sealed service")
    code_id = release["code_id"]
    if type(code_id) not in (int, np.int64) or code_id < 0 or code_id not in bank:
        raise ValueError("A code ID is not in the frozen global public bank")
    expected_probability, expected_decision = bank[code_id]
    actual = np.asarray(release["probability"], dtype=np.float64)
    expected = np.asarray(expected_probability, dtype=np.float64)
    if (actual.ndim != 1 or len(actual) < 2 or not np.isfinite(actual).all()
            or not np.array_equal(actual, expected) or abs(float(actual.sum()) - 1) > 1e-6):
        raise ValueError("A sampled probability differs from the public bank")
    if release["decision"] != expected_decision:
        raise ValueError("A decision differs from the selected public bank entry")
    metadata = release["public_metadata"]
    if (not isinstance(metadata, Mapping) or set(metadata) != A_METADATA_FIELDS
            or metadata != expected_public_metadata
            or metadata["model_id"] != expected_release_id
            or metadata["sampling_contract"] != "cached_one_joint_draw_v1"
            or metadata["fallback"] is not False
            or not isinstance(metadata["code_length_bits"], int)
            or metadata["code_length_bits"] != (len(bank) - 1).bit_length()
            or any(not isinstance(metadata[key], str) or len(metadata[key]) != 64
                   or any(character not in "0123456789abcdef" for character in metadata[key])
                   for key in ("source_binding_sha256", "canonical_bank_sha256",
                               "mechanism_sha256"))):
        raise ValueError("A metadata is not the sealed global constant")


def check_cached_pair(first: tuple[Mapping[str, Any], Mapping[str, Any]],
                      repeated: tuple[Mapping[str, Any], Mapping[str, Any]]) -> None:
    """One service key must return the identical saved joint realization."""
    if len(first) != 2 or len(repeated) != 2:
        raise ValueError("two aligned recipient releases required")
    if json.dumps(first, sort_keys=True, separators=(",", ":")) != json.dumps(
            repeated, sort_keys=True, separators=(",", ":")):
        raise ValueError("repeat request did not return the cached joint release")


def check_a_cached_service(call_with_input_digest: Callable[[str],
                           tuple[Mapping[str, Any], Mapping[str, Any]]],
                           admitted_digest: str, changed_digest: str) -> None:
    """Require stable requery and refusal of changed input under one cache key."""
    if admitted_digest == changed_digest:
        raise ValueError("cache challenge digests must differ")
    first = call_with_input_digest(admitted_digest)
    check_cached_pair(first, call_with_input_digest(admitted_digest))
    try:
        call_with_input_digest(changed_digest)
    except (ValueError, PermissionError):
        return
    raise ValueError("changed input digest reused or redrew a cached release")

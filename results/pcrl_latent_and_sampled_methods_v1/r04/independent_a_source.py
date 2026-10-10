"""Independent, hash-bound T* source and deterministic identity-channel replay.

Only admitted FIT, ATT and INNER score references enter this module. No
candidate, historical encoder, source-store adapter, or scoring module is
imported. A worker may execute it on real rows only in the R03 queue.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROLES = ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION")
BANK_FIELDS = {"probability", "token_class", "token_id", "input_class", "fine_to_hcal_token"}
TOKEN_FIELDS = {"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"}
REFERENCE_FIELDS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2", "d_1", "d_2"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def bound_json(path: Path, expected: str) -> dict:
    if sha_file(path) != expected:
        raise ValueError(f"JSON source byte hash changed: {path.name}")
    return json.loads(path.read_text())


def bound_npz(path: Path, expected: str, fields: set[str]) -> dict[str, np.ndarray]:
    if sha_file(path) != expected:
        raise ValueError(f"NPZ source byte hash changed: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != fields:
            raise ValueError(f"NPZ member contract changed: {path.name}")
        return {name: np.array(archive[name], copy=True) for name in fields}


def routed_cells(probability: np.ndarray, fine: dict) -> np.ndarray:
    """Rebuild qpc's within-predicted-class KL-nearest cell, ties to first."""
    p = np.asarray(probability, dtype=np.float64)
    k = int(fine["K"])
    cc = np.asarray(fine["cell_class"], dtype=np.int64)
    q = np.asarray(fine["centroid"], dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != k or q.shape != (len(cc), k)
            or np.any(q <= 0) or np.any(p < 0) or not np.isfinite(p).all()
            or np.max(np.abs(p.sum(1) - 1), initial=0) > 1e-9):
        raise ValueError("invalid source probabilities or fine centroids")
    decision = np.argmax(p, axis=1)
    out = np.empty(len(p), dtype=np.int64)
    for cls in range(k):
        rows = np.flatnonzero(decision == cls)
        idx = np.flatnonzero(cc == cls)
        if not len(idx):
            raise ValueError("fine source omitted a class")
        if not len(rows):
            continue
        if len(idx) == 1:
            out[rows] = idx[0]
            continue
        chunk = p[rows]
        # Accumulate in class-index order, matching the inherited KL rule,
        # while deriving it here from the published mathematical definition.
        distance = np.zeros((len(rows), len(idx)), dtype=np.float64)
        logq = np.log(q[idx])
        for j in range(k):
            pj = chunk[:, j:j + 1]
            distance += np.where(pj > 0, pj * (np.log(np.where(pj > 0, pj, 1))
                                                - logq[None, :, j]), 0)
        out[rows] = idx[np.argmin(distance, axis=1)]
    return out


def check_bank(bank: dict, policy: dict, fine: dict, decoder: dict, binding: dict) -> dict:
    """Reconstruct calibrated public actions from independent source records."""
    p = bank["probability"]
    tc = bank["token_class"]
    ids = bank["token_id"]
    ic = bank["input_class"]
    mapping = bank["fine_to_hcal_token"]
    k = int(fine["K"])
    if (p.dtype != np.float64 or p.shape != (len(ids), k)
            or any(a.dtype != np.int64 or a.ndim != 1 for a in (tc, ids, ic, mapping))
            or tc.shape != ids.shape or ic.shape != mapping.shape
            or not np.array_equal(ids, np.arange(len(ids)))
            or not np.array_equal(ic, np.asarray(fine["cell_class"], dtype=np.int64))
            or not np.array_equal(mapping, np.asarray(policy["cell_token"], dtype=np.int64))
            or not np.array_equal(tc, np.asarray(policy["token_class"], dtype=np.int64))
            or not np.array_equal(ic, tc[mapping])
            or not np.array_equal(ic, np.asarray(policy["fine"]["cell_class"], dtype=np.int64))
            or not np.array_equal(np.asarray(fine["centroid"]),
                                  np.asarray(policy["fine"]["centroid"]))):
        raise ValueError("fine/policy/bank identity or class map changed")
    decoded = np.asarray(decoder["q"], dtype=np.float64)
    if not np.array_equal(p, decoded):
        raise ValueError("bank differs bitwise from published calibrated decoder")
    q0 = np.asarray(policy["token_proto"], dtype=np.float64)
    alpha = float(decoder["alpha"])
    logits = alpha * np.log(q0)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    independently_calibrated = exp / np.sum(exp, axis=1, keepdims=True)
    max_temperature_difference = float(np.max(np.abs(independently_calibrated - p)))
    if max_temperature_difference > 2e-15:
        raise ValueError("decoder is not the frozen global-temperature map")
    if (not np.array_equal(np.argmax(p, axis=1), tc)
            or not np.isfinite(p).all() or np.any(p < 0)
            or np.max(np.abs(p.sum(1) - 1)) > 1e-8):
        raise ValueError("calibrated actions or decisions invalid")
    for field, hash_key in (("probability", "bank_probability_sha256"),
                            ("token_class", "token_class_sha256"),
                            ("input_class", "input_class_sha256"),
                            ("fine_to_hcal_token", "fine_to_hcal_token_sha256")):
        if sha_array(bank[field]) != binding[hash_key]:
            raise ValueError(f"bound action array changed: {field}")
    identities = {(int(tc[i]), p[i].tobytes()) for i in range(len(ids))}
    if len(identities) != len(ids):
        raise ValueError("different IDs duplicate one output symbol")
    return {"input_fine_cells": int(len(ic)), "public_actions": int(len(ids)),
            "unique_public_actions": int(len(identities)),
            "max_temperature_difference": max_temperature_difference,
            "alpha": alpha,
            "identity_decision_class_preserved": True}


def role_reference_path(manifest_path: Path, manifest: dict, role: str, seed: int) -> Path:
    key = f"{role}:references_s{seed}"
    return manifest_path.parent.parent / manifest["role_file_relative_path"][key]

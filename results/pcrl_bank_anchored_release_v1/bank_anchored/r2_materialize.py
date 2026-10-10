"""Role-array to fixed-bank law and cached one-release private artifacts.

Caller supplies only admitted role-specific source arrays and a sealed policy.
This module never reads task/SEX labels, a mixed-role source, or hidden data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import (
    offline_uniform,
)
from .r2_laws import BoundaryFactorPolicy, IdentityPolicy, offline_factor_uniforms
from .r2_package import _sha
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.deterministic_policy import (
    DeterministicJointPolicy,
)


LAW_KEYS = frozenset(("row_id", "unit", "code_r1", "code_r2", "mass", "valid",
                      "b0_code_r1", "b0_code_r2"))


def _safe_rows(row_id, unit, score1, score2, cell1, cell2):
    arrays = [np.asarray(v) for v in (row_id, unit, score1, score2, cell1, cell2)]
    n = len(arrays[0])
    if (n < 1 or any(len(v) != n or v.dtype == object for v in arrays) or
            arrays[2].dtype != np.float64 or arrays[2].shape != (n, 2) or
            arrays[3].dtype != np.float64 or arrays[3].shape != (n, 6) or
            arrays[4].dtype != np.int64 or arrays[5].dtype != np.int64 or
            arrays[4].shape != (n,) or arrays[5].shape != (n,)):
        raise ValueError("admitted role rows have wrong task dimensions or alignment")
    seen = {}
    for i, key in enumerate(arrays[1]):
        value = (arrays[2][i].tobytes(), arrays[3][i].tobytes(),
                 int(arrays[4][i]), int(arrays[5][i]))
        if key in seen and seen[key] != value:
            raise ValueError("duplicate provider unit has inconsistent source input")
        seen[key] = value
    return arrays


def materialize_arrays(policy, *, row_id, unit, score1, score2, cell1, cell2,
                       source_seed: int, role: str, realization: int,
                       private_root: bytes):
    """Return exact-law and one cached release arrays, never private scores."""
    if (source_seed not in (0, 1) or role not in ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION") or
            realization not in (0, 1, 2) or type(private_root) is not bytes or
            len(private_root) != 32):
        raise ValueError("release identity/stream differs from sealed roles")
    row_id, unit, score1, score2, cell1, cell2 = _safe_rows(
        row_id, unit, score1, score2, cell1, cell2)
    n = len(row_id)
    ids1 = np.empty((n, 16), dtype=np.int64)
    ids2 = np.empty((n, 16), dtype=np.int64)
    mass = np.zeros((n, 16), dtype=np.float64)
    valid = np.zeros((n, 16), dtype=np.bool_)
    b0_1 = np.empty(n, dtype=np.int64)
    b0_2 = np.empty(n, dtype=np.int64)
    release1 = np.empty(n, dtype=np.int64)
    release2 = np.empty(n, dtype=np.int64)
    cached = {}
    for i in range(n):
        a, b = score1[i], score2[i]
        c1, c2 = int(cell1[i]), int(cell2[i])
        b0 = (int(policy.actions1[c1][0]), int(policy.actions2[c2][0]))
        allowed_pairs = {(x, y) for x in policy.actions1[c1]
                         for y in policy.actions2[c2]}
        law = tuple(policy.sparse_law(a, b, c1, c2))
        if (len(law) < 1 or len(law) > 16 or
                not np.isclose(sum(p for _, _, p in law), 1, rtol=0, atol=1e-12) or
                len({(x, y) for x, y, _ in law}) != len(law) or
                any((x, y) not in allowed_pairs for x, y, _ in law)):
            raise ValueError("policy emits an invalid public-code law")
        first = next((item for item in law if item[:2] == b0), (*b0, 0.0))
        ordered = [first] + \
                  [item for item in law if item[:2] != b0]
        if len(ordered) > 16:
            raise ValueError("public law plus exact B0 action exceeds fixed support")
        ids1[i, :] = b0[0]
        ids2[i, :] = b0[1]
        b0_1[i], b0_2[i] = b0
        for slot, (z1, z2, p) in enumerate(ordered):
            if (z1 < 0 or z1 >= len(policy.bank1.probability) or
                    z2 < 0 or z2 >= len(policy.bank2.probability) or
                    not np.isfinite(p) or p < 0 or
                    policy.bank1.decision[z1] != policy.bank1.decision[b0[0]] or
                    policy.bank2.decision[z2] != policy.bank2.decision[b0[1]]):
                raise ValueError("selected law leaves fixed same-class bank")
            ids1[i, slot], ids2[i, slot] = z1, z2
            mass[i, slot], valid[i, slot] = p, True
        key = unit[i].item() if isinstance(unit[i], np.generic) else unit[i]
        if key not in cached:
            if isinstance(policy, IdentityPolicy):
                sampled = b0
            elif isinstance(policy, DeterministicJointPolicy):
                sampled = policy.sample_public_pair(a, b, c1, c2, 0.5)
            elif isinstance(policy, BoundaryFactorPolicy):
                coins = offline_factor_uniforms(private_root, source_seed=source_seed,
                    role=role, realization=realization, unit=key)
                sampled = policy.sample_public_pair(a, b, c1, c2, coins)
            else:
                coin = offline_uniform(private_root, source_seed=source_seed,
                    role=role, realization=realization, unit=key)
                sampled = policy.sample_public_pair(a, b, c1, c2, coin)
            cached[key] = sampled
        release1[i], release2[i] = cached[key]
        if (release1[i], release2[i]) not in {(x, y) for x, y, p in law if p > 0}:
            raise ValueError("cached sample is outside positive support")
    law_arrays = {"row_id": row_id, "unit": unit, "code_r1": ids1,
                  "code_r2": ids2, "mass": mass, "valid": valid,
                  "b0_code_r1": b0_1, "b0_code_r2": b0_2}
    release_arrays = {"row_id": row_id, "unit": unit,
                      "code_r1": release1, "code_r2": release2}
    return law_arrays, release_arrays


def write_materialization(directory, law_arrays, release_arrays, *,
                          arm_id, source_seed, role, realization,
                          source_binding_sha256, policy_receipt_sha256):
    root = Path(directory)
    if (root.is_symlink() or not root.is_dir() or
            stat.S_IMODE(root.stat().st_mode) & 0o077):
        raise PermissionError("release parent must be owner-only")
    if set(law_arrays) != LAW_KEYS or set(release_arrays) != {
            "row_id", "unit", "code_r1", "code_r2"}:
        raise ValueError("release array schema differs")
    paths = {name: root / name for name in
             ("law.npz", "release_codes.npz", "RELEASE_RECEIPT.json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("role release is write-once")
    for name, arrays in (("law.npz", law_arrays),
                         ("release_codes.npz", release_arrays)):
        with os.fdopen(os.open(paths[name], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
            np.savez_compressed(out, **arrays)
            out.flush(); os.fsync(out.fileno())
    receipt = {"schema": "BAR1-role-release-v1", "arm_id": arm_id,
               "source_seed": source_seed, "role": role,
               "realization": realization,
               "source_binding_sha256": source_binding_sha256,
               "policy_receipt_sha256": policy_receipt_sha256,
               "rows": len(law_arrays["row_id"]),
               "law_sha256": _sha(paths["law.npz"]),
               "release_codes_sha256": _sha(paths["release_codes.npz"])}
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(paths["RELEASE_RECEIPT.json"],
                           os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as out:
        out.write(payload); out.flush(); os.fsync(out.fileno())
    return {**receipt, "receipt_sha256": hashlib.sha256(payload).hexdigest()}

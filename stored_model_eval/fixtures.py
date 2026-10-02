"""Synthetic fixtures (the only data on which this package fits anything without --execute-scientific-fits).

Every fixture returns row-level arrays with explicit row ids, grouping units, roles and record keys, in
the same layout an admitted manifest produces, so the whole pipeline runs end to end on it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROLE_SHARES = (("attacker_fit", 0.5), ("attacker_val", 0.2), ("evaluation", 0.3))


def _roles_for_units(n_units, rng):
    u = rng.permutation(n_units)
    roles = np.empty(n_units, dtype=object)
    start = 0
    for i, (name, share) in enumerate(ROLE_SHARES):
        stop = n_units if i == len(ROLE_SHARES) - 1 else start + int(round(share * n_units))
        roles[u[start:stop]] = name
        start = stop
    return roles.astype(str)


def make_synthetic(kind: str = "direct", n_units: int = 2000, d: int = 8, seed: int = 0,
                   rows_per_unit: int = 1, dup_factor: int = 1, K: int = 2, signal: float = 1.0) -> dict:
    """kind: direct | null | xor | contrast | output_leak.

    direct      S is (noisily) a column of H                       (positive control)
    null        H independent of S                                 (null control)
    xor         S = 1[h0 * h1 > 0]: zero linear cross-covariance, recoverable by trees/MLP
    contrast    K classes; one direction encodes 1[y=a] - 1[y=b] for two minority classes only
    output_leak H carries nothing about S; the task outputs do (surface decomposition)
    dup_factor  every record is emitted dup_factor times with NEW row ids and NEW unit ids but the same
                record key (duplicated records must collapse to one unit in inference)
    """
    rng = np.random.default_rng(seed)
    n = n_units * rows_per_unit
    unit = np.repeat(np.arange(n_units), rows_per_unit)
    H = rng.normal(size=(n, d))
    if kind == "direct":
        S = rng.integers(0, 2, n)
        H[:, 0] = signal * (2 * S - 1) + 0.5 * rng.normal(size=n)
    elif kind == "null":
        S = rng.integers(0, 2, n)
    elif kind == "xor":
        S = (H[:, 0] * H[:, 1] > 0).astype(int)
    elif kind == "contrast":
        p = np.r_[0.03, 0.03, np.full(K - 2, 0.94 / (K - 2))]
        S = rng.choice(K, n, p=p)
        c = (S == 0).astype(float) - (S == 1).astype(float)
        H[:, 0] = signal * c + 0.05 * rng.normal(size=n)
    elif kind == "output_leak":
        S = rng.integers(0, 2, n)
    else:
        raise ValueError(kind)
    T = (H[:, 2] + 0.5 * rng.normal(size=n) > 0).astype(int)  # task label
    logit = 1.5 * H[:, 2]
    if kind == "output_leak":
        logit = logit + 1.2 * (2 * S - 1)
    outputs = np.c_[-logit / 2, logit / 2]
    roles = _roles_for_units(n_units, rng)[unit]
    rec = np.array([hashlib.sha256(H[i].tobytes() + bytes([int(S[i])])).hexdigest()[:16] for i in range(n)])
    if dup_factor > 1:
        H, S, T, outputs, roles, rec = (np.repeat(a, dup_factor, axis=0) for a in (H, S, T, outputs, roles, rec))
        unit = np.arange(len(S))  # duplicates get fresh unit ids: only the record key reveals them
    row_ids = rng.permutation(len(S)) + 10_000  # ids are NOT positions
    return {"H": H, "S": S.astype(np.int64), "T": T.astype(np.int64), "outputs": outputs, "units": unit,
            "roles": roles, "record_keys": rec, "row_ids": row_ids, "kind": kind, "synthetic": True}


def write_manifest(fx: dict, out_dir: str | Path, name: str = "synthetic") -> Path:
    """Write a fixture as an npz + manifest pair (for exercising `admit` and the CLI)."""
    from .admission import sha256_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{name}.npz"
    np.savez(npz, row_id=fx["row_ids"], H=fx["H"], S=fx["S"], T=fx["T"], outputs=fx["outputs"],
             unit=fx["units"], role=fx["roles"], record_key=fx["record_keys"])
    man = {"schema": "stored_model_eval.manifest/v1", "synthetic": True,
           "files": {"data": {"path": npz.name, "sha256": sha256_file(npz)}},
           "arrays": {"representations": {"file": "data", "key": "H", "ids": "row_id"},
                      "outputs": {"file": "data", "key": "outputs", "ids": "row_id"},
                      "labels": {"file": "data", "key": "S", "ids": "row_id"},
                      "task_labels": {"file": "data", "key": "T", "ids": "row_id"},
                      "units": {"file": "data", "key": "unit", "ids": "row_id"},
                      "roles": {"file": "data", "key": "role", "ids": "row_id"},
                      "record_keys": {"file": "data", "key": "record_key", "ids": "row_id"}},
           "min_class_support": 20}
    mp = out / f"{name}_manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    return mp

"""Role-sized data-free timing of M5 mask/endpoint arithmetic.

This deliberately excludes package restoration and role admission; their
custody is verified by the sealed pair worker and predecessor queued smokes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_bank import reconstruct
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_endpoint import (
    replay_pair_endpoint,
)


STUDY = Path(__file__).resolve().parents[1]
READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
           "RAW_LR_C1", "RAW_HGB_15x200")


def _prob(rng: np.random.Generator, rows: int, width: int) -> np.ndarray:
    raw = rng.random((rows, width), dtype=np.float64) + .1
    return raw / raw.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    rng = np.random.default_rng(26001008)
    bank = reconstruct(STUDY / "m1/CAPACITY_SPEC.json")
    pair = bank.pairs[0]
    fit_x = rng.random((15434, 83), dtype=np.float32)
    att_x = rng.random((4064, 83), dtype=np.float32)
    inner_x = rng.random((2235, 83), dtype=np.float32)
    fit_labels = {"income": rng.integers(0, 2, len(fit_x), dtype=np.int64),
                  "occupation": rng.integers(0, 6, len(fit_x), dtype=np.int64)}
    truth = {"row_id": np.arange(len(inner_x), dtype=np.int64),
             "unit": np.r_[np.arange(len(inner_x) - 1, dtype=np.int64),
                            len(inner_x) - 2],
             "income": rng.integers(0, 2, len(inner_x), dtype=np.int64),
             "occupation": rng.integers(0, 6, len(inner_x), dtype=np.int64),
             "sex": np.arange(len(inner_x), dtype=np.int64) % 2}
    selected = {"income": bank.mask(pair.income_mask_id).column_indices,
                "occupation_group": bank.mask(pair.occupation_mask_id).column_indices,
                "coalition": pair.union_columns}
    wires = {role: {view: x[:, columns].copy(order="C")
                    for view, columns in selected.items()}
             for role, x in (("FIT", fit_x), ("ATT", att_x), ("INNER", inner_x))}
    if any(wire.shape[1] != len(selected[view]) for roles in wires.values()
           for view, wire in roles.items()):
        raise RuntimeError("M5 selected raw-wire geometry differs")
    candidate = {task: {"row_id": truth["row_id"], "unit": truth["unit"],
                        "prob": _prob(rng, len(inner_x), width)}
                 for task, width in (("income", 2), ("occupation_group", 6))}
    references = {task: {ref: {"row_id": truth["row_id"],
                               "unit": truth["unit"],
                               "prob": _prob(rng, len(inner_x), width)}
                         for ref in ("U0", "Ucal", "F")}
                  for task, width in (("income", 2), ("occupation_group", 6))}
    readers = {}
    for view in ("income", "occupation_group", "coalition"):
        roster = READERS if view != "coalition" else READERS + tuple(
            alias + reader for alias in ("IGNORE_R1__", "IGNORE_R2__")
            for reader in READERS)
        readers[view] = {reader: rng.random(len(inner_x), dtype=np.float64)
                         for reader in roster}
    endpoint = replay_pair_endpoint(
        seed=0, pair_id=pair.pair_id, truth=truth, fit_labels=fit_labels,
        candidate=candidate, references=references, reader_scores=readers,
        v2_status_by_view={view: "PASS" for view in readers})
    if (endpoint["status"] != "REPLAYED"
            or set(endpoint["recovery"]["coalition"]["readers"]) != set(readers["coalition"])):
        raise RuntimeError("M5 role-sized endpoint replay is incomplete")
    payload = {"schema": "pcrl-focused-m5-pair-smoke-v1", "status": "PASS",
               "scope": "data-free raw slice and endpoint timing; no package restore",
               "fit_rows": len(fit_x), "att_rows": len(att_x),
               "inner_rows": len(inner_x), "reader_count": 30,
               "bank_sha256": bank.bank_sha256,
               "elapsed_monotonic_seconds": time.monotonic() - start,
               "endpoint_sha256": hashlib.sha256(json.dumps(endpoint, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    print(json.dumps({"status": "PASS", "output_sha256": hashlib.sha256(
        path.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()

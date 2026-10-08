"""S3-only synthetic timing of Team D's nine direct hard-code readers.

The fixture has ATTACK_FIT_NEW/INNER_SELECTION row and group counts, a fixed
1024-code alphabet, and no Adult inputs. It times only the three predeclared
CellReader alphas on each of recipient0, recipient1 and the aligned pair;
S2's learned/META/BIT/RAW readers are outside this timing unit.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

import attack_eval
from attack_views import AlignedDViews
from d_cell_slate import fit_d_cell_readers


class _SyntheticS2View:
    def __init__(self, row_id: np.ndarray, unit: np.ndarray, cell_key: np.ndarray):
        self.row_id = row_id
        self.unit = unit
        self.cell_key = cell_key

    def validate(self):
        if len(self.row_id) != len(self.unit) or len(self.row_id) != len(self.cell_key):
            raise ValueError("synthetic attack view identities disagree")
        return self


def _synthetic_views(n: int, groups: int, seed: int) -> tuple[AlignedDViews, dict]:
    if n < groups or groups < 2:
        raise ValueError("invalid synthetic role size")
    rng = np.random.default_rng(seed)
    row = np.arange(n, dtype=np.int64) + 1000000 * seed
    unit = np.arange(n, dtype=np.int64)
    unit[groups:] = np.arange(n - groups, dtype=np.int64)
    codes = rng.integers(0, 1024, size=(n, 2), dtype=np.int64)
    sex = ((codes[:, 0] & 1) ^ ((codes[:, 1] >> 1) & 1)).astype(np.int64)

    def wire(purpose: int) -> dict:
        q = (np.tile([0.6, 0.4], (n, 1)) if purpose == 0 else
             np.tile([0.5, 0.3, 0.2], (n, 1)))
        return {
            "code_ids": codes[:, purpose:purpose + 1],
            "decoded": np.zeros((n, 16), dtype=np.float32),
            "probabilities": q,
            "decision": np.zeros(n, dtype=np.int64),
            "metadata": {"purpose": purpose, "code_length_bits": 10,
                         "block_count": 1, "fallback": False, "stochastic": False,
                         "kind": "D-CELL-SYNTHETIC-SMOKE", "calibration": "H-GLOBAL-TEMP",
                         "calibrator_sha256": "a" * 64},
        }

    aligned = AlignedDViews(row, unit, row, row, wire(0), wire(1), "b" * 64)
    return aligned, {"row_id": row, "unit": unit, "sex": sex}


def _view(aligned: AlignedDViews, name: str) -> _SyntheticS2View:
    codes = aligned.codes(name)
    key = (codes[:, 0].astype(str) if codes.shape[1] == 1 else
           np.asarray([f"{int(a)}|{int(b)}" for a, b in codes]))
    return _SyntheticS2View(aligned.row_id, aligned.unit_id, key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    args = parser.parse_args()
    root = args.private_root.resolve(strict=True)
    if ".private" not in root.parts or (root.stat().st_mode & 0o077):
        raise PermissionError("synthetic timing artifacts require owner-only .private custody")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise PermissionError(f"{variable}=1 is required by the S3 timing contract")
    if not args.unit_id or not all(c.isalnum() or c in "_-" for c in args.unit_id):
        raise ValueError("invalid queued unit ID")
    target = root / args.unit_id
    target.mkdir(mode=0o700)
    att, sex_att = _synthetic_views(4064, 4061, 11)
    inner, _ = _synthetic_views(2235, 2234, 12)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    records = []
    for name in ("recipient0", "recipient1", "pair"):
        readers = fit_d_cell_readers(att, inner, name, _view(att, name),
                                     _view(inner, name), sex_att, attack_eval,
                                     model_dir=target / name, private_root=root)
        for reader in readers:
            receipt = json.loads(Path(reader["completion_receipt_path"]).read_text())
            records.append({key: receipt[key] for key in (
                "name", "view", "fit_rows", "tune_rows", "model_sha256",
                "prediction_sha256", "fit_predict_artifact_wall_seconds",
                "fit_predict_artifact_cpu_seconds")})
    if len(records) != 9:
        raise AssertionError("D specialist smoke must time exactly nine fits")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "schema": "pcrl-d-specialist-synthetic-smoke-v1",
        "unit_id": args.unit_id, "synthetic_only": True,
        "att_rows": 4064, "att_groups": 4061,
        "inner_rows": 2235, "inner_groups": 2234,
        "code_alphabet": 1024, "reader_fits": 9,
        "thread_contract": {name: os.environ.get(name) for name in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_raw_platform_units": rss,
        "per_reader": records,
        "not_timed": "S2 learned, metadata, bit and raw readers; calibration and model training",
    }
    path = target / "SUMMARY.json"
    with path.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.chmod(path, 0o600)
    print(json.dumps({key: summary[key] for key in (
        "schema", "unit_id", "synthetic_only", "reader_fits", "wall_seconds",
        "cpu_seconds", "peak_rss_raw_platform_units")}, sort_keys=True))


if __name__ == "__main__":
    main()

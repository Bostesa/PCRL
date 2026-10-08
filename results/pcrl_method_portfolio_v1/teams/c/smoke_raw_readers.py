"""Queue-only, data-free timing of S2's Team C raw-feature readers.

Each reader/mask runs in a fresh sequential child, so ru_maxrss is a
per-reader process high-water mark. The parent holds S3's heavy slot until all
children exit. This is resource evidence, not a scientific attack result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from prototype import ATOM_ORDER, AdultGroupedSchema, IdentityCalibrator, PurposeRelease, release_pair
from s2_bridge import to_s2_attack_views


READERS = ("RAW_LR_C1", "RAW_HGB_15x200")
MASKS = ("FULL", "REL0")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic(rows: int, seed: int, *, group_duplicates: int = 0):
    rng = np.random.default_rng(seed)
    schema = AdultGroupedSchema()
    X = np.zeros((rows, 83), dtype=np.float32)
    for name in ("age", "education_fact", "capital-gain", "capital-loss",
                 "hours-per-week"):
        index = schema.atoms[name][0]
        X[:, index] = rng.normal(size=rows).astype(np.float32)
    for name in ("workclass", "education_fact", "marital-status",
                 "relationship", "native-country"):
        indices = tuple(i for i in schema.atoms[name]
                        if schema.feature_names[i].startswith(("education=" if name == "education_fact"
                                                              else name + "=")))
        chosen = rng.integers(0, len(indices), size=rows)
        X[np.arange(rows), np.asarray(indices)[chosen]] = 1.0
    row_id = np.arange(seed * 100000, seed * 100000 + rows, dtype=np.int64)
    unit = row_id.copy()
    if group_duplicates:
        unit[-group_duplicates:] = unit[:group_duplicates]
    sex = (X[:, schema.atoms["age"][0]] +
           0.2 * rng.normal(size=rows) > 0).astype(np.int64)
    return X, sex, row_id, unit


class _BinaryHead:
    def predict_proba(self, selected):
        z = np.asarray(selected, dtype=np.float64)[:, 0]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return np.column_stack((1.0 - p, p))


class _SixHead:
    def predict_proba(self, selected):
        x = np.asarray(selected, dtype=np.float64)
        logits = np.column_stack((x[:, :5], np.zeros(len(x))))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


def _views(shared_dir: Path, mask_name: str):
    sys.path.insert(0, str(shared_dir))
    from attack_eval import AttackView

    schema = AdultGroupedSchema()
    selected = (ATOM_ORDER if mask_name == "FULL" else
                tuple(atom for atom in ATOM_ORDER if atom != "relationship"))
    income = PurposeRelease(
        schema, selected, _BinaryHead(), "income", "synthetic-income-head",
        "synthetic-identity-calibrator", IdentityCalibrator(),
    )
    occupation = PurposeRelease(
        schema, selected, _SixHead(), "occupation_group",
        "synthetic-occupation-head", "synthetic-identity-calibrator",
        IdentityCalibrator(),
    )
    outputs = {}
    for role, rows, seed, duplicates in (
        ("ATTACK_FIT_NEW", 4064, 321, 3),
        ("INNER_SELECTION", 2235, 322, 1),
    ):
        X, sex, row_id, unit = _synthetic(rows, seed, group_duplicates=duplicates)
        wire = release_pair(X, schema.feature_names, income, occupation)
        alignment = {
            name: {"row_id": row_id, "unit": unit}
            for name in ("income", "occupation_group", "coalition")
        }
        views = to_s2_attack_views(
            wire, alignment,
            {"income": income, "occupation_group": occupation}, AttackView,
        )
        outputs[role] = (views["coalition"], sex)
    return outputs


def _worker(args) -> None:
    sys.path.insert(0, str(args.shared_dir))
    from attack_eval import _factory, attack_fit_tune_split
    from safe_data_guard import sha_arr

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    outputs = _views(args.shared_dir, args.mask)
    att, y = outputs["ATTACK_FIT_NEW"]
    inner, _ = outputs["INNER_SELECTION"]
    split = attack_fit_tune_split(att.unit)
    fit = split["fit"]
    tune = split["tune"]
    model = _factory(args.reader, 0)
    preparation = {
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
    }
    fit_wall, fit_cpu = time.perf_counter(), time.process_time()
    model.fit(np.asarray(att.x_raw_only, dtype=np.float64)[fit], y[fit])
    tune_prediction = model.predict_proba(np.asarray(att.x_raw_only, dtype=np.float64)[tune])[:, 1]
    inner_prediction = model.predict_proba(np.asarray(inner.x_raw_only, dtype=np.float64))[:, 1]
    elapsed = {
        "wall_seconds": time.perf_counter() - fit_wall,
        "cpu_seconds": time.process_time() - fit_cpu,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    receipt = {
        "schema": "team-c-synthetic-raw-reader-smoke-v1",
        "data_access": "NONE", "science_fit": False,
        "reader_id": args.reader, "mask": args.mask,
        "attack_rows": len(att.row_id), "inner_rows": len(inner.row_id),
        "attack_groups": int(np.unique(att.unit).size),
        "inner_groups": int(np.unique(inner.unit).size),
        "fit_rows": len(fit), "tune_rows": len(tune),
        "raw_only_columns": att.x_raw_only.shape[1],
        "complete_view_columns": att.x.shape[1],
        "defense_aware_columns": att.x_defense_aware.shape[1],
        "raw_only_view_sha256": att.content_hashes()["x_raw_only"],
        "tune_prediction_sha256": sha_arr(tune_prediction),
        "inner_prediction_sha256": sha_arr(inner_prediction),
        "preparation": preparation, "reader_fit_and_forward": elapsed,
        "source_sha256": {
            "script": _sha_file(Path(__file__)),
            "attack_eval": _sha_file(args.shared_dir / "attack_eval.py"),
            "safe_data_guard": _sha_file(args.shared_dir / "safe_data_guard.py"),
            "common_eval": _sha_file(args.shared_dir / "common_eval.py"),
            "prototype": _sha_file(Path(__file__).with_name("prototype.py")),
            "evaluation": _sha_file(Path(__file__).with_name("evaluation.py")),
            "s2_bridge": _sha_file(Path(__file__).with_name("s2_bridge.py")),
        },
        "synthetic_note": "one-hot categorical blocks and five numeric columns; no Adult rows or labels",
    }
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reader", choices=READERS)
    parser.add_argument("--mask", choices=MASKS)
    args = parser.parse_args()
    if args.output.parent.name != ".private":
        raise ValueError("synthetic timing output must be in a .private directory")
    if not (args.shared_dir / "attack_eval.py").is_file():
        raise FileNotFoundError("pinned S2 attack evaluator required")
    if args.worker:
        if args.reader is None or args.mask is None:
            raise ValueError("worker requires one registered reader and mask")
        _worker(args)
        return
    results = []
    for mask in MASKS:
        for reader in READERS:
            child_output = args.output.with_name(
                args.output.stem + f"__{mask}__{reader}.json",
            )
            command = (
                sys.executable, str(Path(__file__)), "--shared-dir",
                str(args.shared_dir), "--output", str(child_output),
                "--worker", "--reader", reader, "--mask", mask,
            )
            subprocess.run(command, check=True)
            results.append(json.loads(child_output.read_text()))
    with args.output.open("x") as stream:
        json.dump({
            "schema": "team-c-synthetic-raw-reader-smoke-aggregate-v1",
            "data_access": "NONE", "science_fit": False,
            "serial_children": True,
            "children": results,
        }, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()

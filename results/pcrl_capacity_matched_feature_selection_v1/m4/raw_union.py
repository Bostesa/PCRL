"""Canonical RAW_ONLY source union and two-reader fit contract.

The raw reader sees only the selected public source columns. This module has
no role loader and does not read INNER sensitive labels. Adult callers must
hold the sealed queue unit before supplying admitted role-only arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace

import joblib
import numpy as np

from views import CanonicalRawUnion


RAW_READER_IDS = ("RAW_LR_C1", "RAW_HGB_15x200")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_json(path: Path, record: dict) -> str:
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_file(path)


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_raw_union(
    source_record: Mapping[str, object], *, source_columns: tuple[int, ...],
    feature_names: tuple[str, ...],
) -> CanonicalRawUnion:
    """Construct the same role-specific union identity as the release checker."""
    x = np.asarray(source_record["X"])
    names = tuple(source_record["feature_names"])
    if (x.dtype != np.dtype("float32") or x.ndim != 2 or x.shape[1] != 83
            or not np.isfinite(x).all() or len(names) != 83):
        raise ValueError("admitted finite float32 83-column source required")
    if (source_columns != tuple(sorted(set(source_columns)))
            or not source_columns or any(type(index) is not int or index < 0 or index >= 83
                                      for index in source_columns)
            or feature_names != tuple(names[index] for index in source_columns)):
        raise ValueError("raw union source columns or feature names differ")
    row_id = np.asarray(source_record["row_id"])
    unit = np.asarray(source_record["unit"])
    if (row_id.dtype != np.dtype("int64") or unit.dtype != np.dtype("int64")
            or row_id.shape != (len(x),) or unit.shape != row_id.shape
            or len(np.unique(row_id)) != len(x)):
        raise ValueError("raw union needs exact ordered int64 people")
    raw = np.ascontiguousarray(x[:, source_columns], dtype=np.float64)
    header = json.dumps({"columns": source_columns, "feature_names": feature_names},
                        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    for block in (header, row_id.tobytes(), unit.tobytes(), raw.tobytes(order="C")):
        digest.update(len(block).to_bytes(8, "big"))
        digest.update(block)
    return CanonicalRawUnion(row_id.copy(), unit.copy(), source_columns,
                             feature_names, raw, digest.hexdigest())


def raw_union_fit_plan(att: CanonicalRawUnion, inner: CanonicalRawUnion) -> dict:
    """Return prefit identity and the fixed S2 group split without fitting."""
    from attack_eval import attack_fit_tune_split

    if (att.source_columns != inner.source_columns
            or att.feature_names != inner.feature_names):
        raise ValueError("ATT and INNER raw union columns differ")
    if np.intersect1d(att.row_id, inner.row_id).size or \
            np.intersect1d(att.unit, inner.unit).size:
        raise ValueError("ATT and INNER people or groups overlap")
    split = attack_fit_tune_split(att.unit)
    return {
        "schema": "pcrl-focused-raw-union-fit-plan-v1",
        "source_columns": list(att.source_columns),
        "att_union_sha256": att.content_sha256,
        "inner_union_sha256": inner.content_sha256,
        "att_row_unit_sha256": _sha_array(np.column_stack((att.row_id, att.unit))),
        "inner_row_unit_sha256": _sha_array(np.column_stack((inner.row_id, inner.unit))),
        "attack_split_sha256": split["tune_group_id_sha256"],
        "fit_rows": int(len(split["fit"])), "tune_rows": int(len(split["tune"])),
        "reader_ids": list(RAW_READER_IDS),
    }


def fit_raw_union_readers(
    att: CanonicalRawUnion, sex_att: Mapping[str, object],
    inner: CanonicalRawUnion, *, seed: int, union_sha256: str,
    union_atoms: tuple[str, ...], source_schema_sha256: str,
    lock_sha256: str, manifest_sha256: str, private_output: str | Path,
) -> dict:
    """Fit two RAW_ONLY readers once for one selected union and seed.

    The caller must establish current S3 science admission *before* reading
    the three role files and invoking this function. There is deliberately no
    INNER SEX argument; the models and orientations use ATT only.
    """
    from attack_eval import _factory, _label_record, attack_fit_tune_split, oriented_auc

    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("raw reader seed is outside frozen study seeds")
    if any(not isinstance(value, str) or len(value) != 64 or
           any(char not in "0123456789abcdef" for char in value)
           for value in (union_sha256, source_schema_sha256,
                         lock_sha256, manifest_sha256)):
        raise ValueError("raw reader source hashes require full SHA-256")
    plan = raw_union_fit_plan(att, inner)
    if (tuple(union_atoms) != tuple(dict.fromkeys(union_atoms))
            or not union_atoms):
        raise ValueError("raw union atoms are missing or duplicated")
    geometry = {"source_schema_sha256": source_schema_sha256,
                "union_atoms": union_atoms,
                "union_column_indices": att.source_columns}
    geometry_sha = hashlib.sha256(json.dumps(
        geometry, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    if geometry_sha != union_sha256:
        raise ValueError("raw union geometry differs from frozen bank")
    out = Path(private_output).resolve()
    if ".private" not in out.parts or out.exists():
        raise PermissionError("raw reader output must be a fresh private directory")
    for field in ("row_id", "unit"):
        identity = np.asarray(sex_att[field])
        if identity.dtype != np.dtype("int64") or identity.shape != att.row_id.shape:
            raise ValueError("ATT SEX needs exact int64 row/group identity")
    y = _label_record(dict(sex_att), SimpleNamespace(
        row_id=att.row_id, unit=att.unit))
    split = attack_fit_tune_split(att.unit)
    if np.unique(y[split["fit"]]).size != 2 or np.unique(y[split["tune"]]).size != 2:
        raise ValueError("raw reader ATT fit/tune lacks both SEX classes")
    out.mkdir(parents=True, mode=0o700)
    readers = {}
    for reader_id in RAW_READER_IDS:
        model = _factory(reader_id, seed)
        model.fit(att.x[split["fit"]], y[split["fit"]])
        model_path = out / f"model__{reader_id}.joblib"
        joblib.dump(model, model_path, compress=3)
        model_path.chmod(0o600)
        model_sha = _sha_file(model_path)
        role_records = {}
        for role_name, union in (("att", att), ("inner", inner)):
            prediction = np.asarray(model.predict_proba(union.x)[:, 1], dtype=np.float64)
            if (prediction.shape != (len(union.row_id),) or
                    not np.isfinite(prediction).all() or
                    np.any((prediction < 0) | (prediction > 1))):
                raise ValueError("raw reader emitted an invalid positive score")
            prediction_path = out / f"prediction__{reader_id}__{role_name}.npy"
            with prediction_path.open("xb") as stream:
                np.save(stream, prediction, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            prediction_path.chmod(0o600)
            receipt = {
                "schema": "pcrl-focused-raw-reader-transport-v1",
                "reader_id": reader_id, "input_kind": "RAW_ONLY",
                "source_union_sha256": union.content_sha256,
                "source_columns": list(union.source_columns),
                "source_prediction_sha256": _sha_array(prediction),
                "attack_split_sha256": plan["attack_split_sha256"],
                "model_artifact_name": model_path.name,
                "model_sha256": model_sha,
            }
            receipt_path = out / f"transport__{reader_id}__{role_name}.json"
            role_records[role_name] = {
                "prediction_file": prediction_path.name,
                "prediction_file_sha256": _sha_file(prediction_path),
                "transport_receipt_file": receipt_path.name,
                "transport_receipt_sha256": _private_json(receipt_path, receipt),
            }
        att_prediction = np.load(out / role_records["att"]["prediction_file"],
                                 allow_pickle=False)
        validation = oriented_auc(y[split["tune"]],
                                  att_prediction[split["tune"]])
        validation["selected_orientation"] = (
            "positive" if validation["positive_alias_auc"] >=
            validation["reverse_alias_auc"] else "reverse"
        )
        readers[reader_id] = {"model_file": model_path.name,
                              "model_sha256": model_sha, "roles": role_records,
                              "att_tune_validation": validation,
                              "attacker_seed": seed}
    index = {
        "schema": "pcrl-focused-raw-union-reader-index-v1",
        "status": "FROZEN", "seed": seed, "union_sha256": union_sha256,
        "lock_sha256": lock_sha256, "manifest_sha256": manifest_sha256,
        "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
        "prediction_roles": ["ATTACK_FIT_NEW", "INNER_SELECTION"],
        "plan": plan, "readers": readers, "fresh_fit_count": len(RAW_READER_IDS),
    }
    _private_json(out / "RAW_UNION_INDEX.json", index)
    return index

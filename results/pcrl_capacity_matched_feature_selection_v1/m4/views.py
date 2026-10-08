"""Independent canonical raw-union checks and zero-fit reader transport.

The functions consume admitted in-memory features and complete public wires.
They neither open role files nor fit a reader. Transport applies only to raw
union models whose original source columns and ordered people match exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np


TRANSPORT_PREDICTION_ATOL = 1e-12
TRANSPORT_RECEIPT_SCHEMA = "pcrl-focused-raw-reader-transport-v1"
RAW_READERS = frozenset(("RAW_LR_C1", "RAW_HGB_15x200"))


@dataclass(frozen=True)
class CanonicalRawUnion:
    row_id: np.ndarray
    unit: np.ndarray
    source_columns: tuple[int, ...]
    feature_names: tuple[str, ...]
    x: np.ndarray
    content_sha256: str


@dataclass(frozen=True)
class TransportedPrediction:
    prediction: np.ndarray
    reader_id: str
    source_union_sha256: str
    target_union_sha256: str
    model_sha256: str
    reader_receipt_sha256: str
    max_abs_replay_delta: float
    tolerance: float


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _int64(value: object, *, name: str, rows: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.dtype("int64") or arr.shape != (rows,):
        raise ValueError(f"{name} needs exact ordered int64 row identity")
    return arr.copy()


def _same_array_bits(actual: np.ndarray, expected: np.ndarray) -> bool:
    return (actual.shape == expected.shape and actual.dtype == expected.dtype and
            actual.tobytes(order="C") == expected.tobytes(order="C"))


def _wire_source_match(
    source_x: np.ndarray, view: Mapping[str, object], schema: object, *, name: str,
) -> tuple[int, ...]:
    groups = tuple(view["selected_groups"])
    if groups != schema.normalize_atoms(groups):
        raise ValueError(f"{name} groups are not in pinned order")
    columns = tuple(schema.column_indices(groups))
    names = tuple(schema.feature_names[j] for j in columns)
    if tuple(view["feature_names"]) != names:
        raise ValueError(f"{name} feature names differ from source columns")
    actual = np.asarray(view["selected_values"])
    expected = source_x[:, columns]
    if not _same_array_bits(actual, expected):
        raise ValueError(f"{name} values differ from source columns")
    return columns


def canonical_union_from_source(
    source_record: Mapping[str, object], release: Mapping[str, Mapping[str, object]],
    schema: object,
) -> CanonicalRawUnion:
    """Reconstruct the disclosed raw union in the original 83-column order."""
    if set(release) != {"income", "occupation_group", "coalition"}:
        raise ValueError("exact two-recipient and coalition release required")
    if tuple(source_record["feature_names"]) != tuple(schema.feature_names):
        raise ValueError("admitted ordered source schema differs")
    x = np.asarray(source_record["X"])
    if (x.ndim != 2 or x.shape[1] != len(schema.feature_names) or
            not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all()):
        raise ValueError("admitted source feature matrix invalid")
    row_id = _int64(source_record["row_id"], name="row_id", rows=len(x))
    unit = _int64(source_record["unit"], name="unit", rows=len(x))
    if np.unique(row_id).size != len(row_id):
        raise ValueError("admitted source row IDs repeat")
    local_columns = {
        name: _wire_source_match(x, release[name], schema, name=name)
        for name in ("income", "occupation_group")
    }
    union_columns = _wire_source_match(x, release["coalition"], schema, name="coalition")
    if set(union_columns) != set(local_columns["income"]) | set(local_columns["occupation_group"]):
        raise ValueError("coalition source columns differ from local union")
    if tuple(release["coalition"]["income_groups"]) != tuple(release["income"]["selected_groups"]) or \
            tuple(release["coalition"]["occupation_group_groups"]) != \
            tuple(release["occupation_group"]["selected_groups"]):
        raise ValueError("coalition declared recipient groups differ")
    raw = np.ascontiguousarray(x[:, union_columns], dtype=np.float64)
    header = json.dumps({"columns": union_columns,
                         "feature_names": tuple(schema.feature_names[j] for j in union_columns)},
                        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    for block in (header, row_id.tobytes(), unit.tobytes(), raw.tobytes(order="C")):
        digest.update(len(block).to_bytes(8, "big"))
        digest.update(block)
    return CanonicalRawUnion(
        row_id=row_id, unit=unit, source_columns=union_columns,
        feature_names=tuple(schema.feature_names[j] for j in union_columns),
        x=raw, content_sha256=digest.hexdigest(),
    )


def build_complete_attack_views(
    source_record: Mapping[str, object],
    release: Mapping[str, Mapping[str, object]],
    public_releases: Mapping[str, object], schema: object, *,
    bridge: object, attack_view_type: type,
) -> tuple[dict[str, object], CanonicalRawUnion]:
    """Validate public head replay and bind every AttackView to admitted raw data.

    ``bridge`` is the inherited C-to-S2 adapter. It checks the full wire and
    head/calibrator replay. We independently reconstruct source columns and
    force canonical C-contiguous matrices before fitting or transport.
    """
    union = canonical_union_from_source(source_record, release, schema)
    alignment = {
        name: {"row_id": union.row_id, "unit": union.unit}
        for name in ("income", "occupation_group", "coalition")
    }
    original = bridge(release, alignment, public_releases, attack_view_type)
    if set(original) != set(alignment):
        raise ValueError("complete S2 view set differs")
    result = {}
    source_x = np.asarray(source_record["X"])
    for name, view in original.items():
        if not isinstance(view, attack_view_type):
            raise ValueError("S2 bridge returned a different view type")
        columns = (union.source_columns if name == "coalition" else
                   tuple(schema.column_indices(tuple(release[name]["selected_groups"]))))
        raw = np.ascontiguousarray(source_x[:, columns], dtype=np.float64)
        if not _same_array_bits(np.asarray(view.x_raw_only), raw):
            raise ValueError(f"{name} S2 raw-only view differs from source columns")
        prepared = replace(
            view,
            x=np.ascontiguousarray(view.x, dtype=np.float64),
            x_defense_aware=np.ascontiguousarray(view.x_defense_aware, dtype=np.float64),
            x_raw_only=raw,
        )
        result[name] = prepared.validate()
    return result, union


def transport_raw_reader(
    source: CanonicalRawUnion, target: CanonicalRawUnion,
    source_prediction: object, reader_receipt_path: str | Path,
    expected_reader_receipt_sha256: str, model_artifact_path: str | Path, *,
    expected_attack_split_sha256: str,
) -> TransportedPrediction:
    """Carry a fitted raw-only reader to an identical union without refitting.

    Replay may differ at the declared machine tolerance, but the returned
    prediction is the original saved score so rank/tie behavior is identical.
    An augmented complete-view model cannot enter this raw-only function.
    """
    receipt_path = Path(reader_receipt_path).resolve(strict=True)
    model_path = Path(model_artifact_path).resolve(strict=True)
    if ".private" not in receipt_path.parts or ".private" not in model_path.parts:
        raise ValueError("transport receipt/model must be in private custody")
    if _sha_file(receipt_path) != expected_reader_receipt_sha256:
        raise ValueError("transport reader receipt hash differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != TRANSPORT_RECEIPT_SCHEMA or \
            receipt.get("input_kind") != "RAW_ONLY" or \
            receipt.get("reader_id") not in RAW_READERS:
        raise ValueError("RAW_ONLY reader receipt required")
    if receipt.get("source_union_sha256") != source.content_sha256 or \
            receipt.get("source_columns") != list(source.source_columns) or \
            receipt.get("attack_split_sha256") != expected_attack_split_sha256:
        raise ValueError("transport source union/split receipt differs")
    if receipt.get("model_artifact_name") != model_path.name or \
            receipt.get("model_sha256") != _sha_file(model_path):
        raise ValueError("transport model artifact differs")
    if source.source_columns != target.source_columns or source.feature_names != target.feature_names:
        raise ValueError("raw reader source columns differ")
    if not np.array_equal(source.row_id, target.row_id) or \
            not np.array_equal(source.unit, target.unit):
        raise ValueError("raw reader ordered people differ")
    if source.content_sha256 != target.content_sha256 or \
            not _same_array_bits(source.x, target.x):
        raise ValueError("raw reader source values differ")
    p = np.asarray(source_prediction, dtype=np.float64)
    if p.shape != (len(source.row_id),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("raw reader saved prediction invalid")
    if receipt.get("source_prediction_sha256") != _sha_array(p):
        raise ValueError("transport saved prediction hash differs")
    model = joblib.load(model_path)
    replay = np.asarray(model.predict_proba(target.x), dtype=np.float64)
    if replay.shape != (len(p), 2) or not np.isfinite(replay).all():
        raise ValueError("raw reader prediction replay differs")
    maximum_delta = float(np.max(np.abs(replay[:, 1] - p)))
    if maximum_delta > TRANSPORT_PREDICTION_ATOL:
        raise ValueError("raw reader prediction replay differs")
    return TransportedPrediction(
        prediction=p.copy(), reader_id=receipt["reader_id"],
        source_union_sha256=source.content_sha256,
        target_union_sha256=target.content_sha256,
        model_sha256=receipt["model_sha256"],
        reader_receipt_sha256=expected_reader_receipt_sha256,
        max_abs_replay_delta=maximum_delta,
        tolerance=TRANSPORT_PREDICTION_ATOL,
    )

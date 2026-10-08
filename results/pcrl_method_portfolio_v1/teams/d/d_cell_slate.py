"""Prelocked direct hard-ID SEX readers for Team D's finite-code views.

This helper reads ATTACK_FIT_NEW SEX only. S2's group split chooses fitting and
tuning rows; INNER labels are never supplied. The caller attaches its three
frozen reader records through S2 `add_external_fixed_readers`, then appends
ignore-recipient aliases and scores INNER under the shared evaluator.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from attack_views import ALPHAS, AlignedDViews, CellReader


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_payload(reader: CellReader, fit_row_id: np.ndarray,
                   fit_codes: np.ndarray, fit_sex: np.ndarray) -> bytes:
    def table(value: dict) -> list:
        return [[list(key) if isinstance(key, tuple) else int(key), list(count)]
                for key, count in sorted(value.items())]

    state = {
        "schema": "pcrl-d-cell-reader-v1", "alpha": reader.alpha,
        "columns": reader.columns, "prior": reader.prior,
        "counts": table(reader.counts),
        "marginal_counts": [table(value) for value in reader.marginal_counts],
        "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
        "fit_rows": int(len(fit_row_id)),
        "fit_row_id_sha256": _array_sha(np.asarray(fit_row_id, dtype=np.int64)),
        "fit_codes_sha256": _array_sha(np.asarray(fit_codes, dtype=np.int64)),
        "fit_sex_sha256": _array_sha(np.asarray(fit_sex, dtype=np.int64)),
    }
    return (json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _private_artifact_path(directory: Path, private_root: Path, name: str,
                           suffix: str = ".json") -> Path:
    root = Path(private_root).resolve(strict=True)
    if ".private" not in root.parts or (root.stat().st_mode & 0o077):
        raise PermissionError("D cell reader root must be owner-only .private custody")
    target = Path(directory).resolve(strict=False)
    if not target.is_relative_to(root) or target == root or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None:
        raise PermissionError("D cell reader artifact escapes private custody")
    if not target.exists():
        if not target.parent.is_dir():
            raise FileNotFoundError("D cell reader private parent must exist")
        target.mkdir(mode=0o700)
    if not target.is_dir() or (target.stat().st_mode & 0o077):
        raise PermissionError("D cell reader directory must be owner-only")
    if suffix not in (".json", ".npz"):
        raise ValueError("unregistered D specialist artifact suffix")
    return target / f"{name}{suffix}"


def _write_private_once(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.is_symlink() or (path.stat().st_mode & 0o077) or path.read_bytes() != payload:
            raise ValueError("existing private D cell artifact differs from deterministic fit")
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return digest


def _save_private_reader(payload: bytes, directory: Path, private_root: Path,
                         name: str) -> tuple[str, str]:
    path = _private_artifact_path(directory, private_root, name)
    return str(path), _write_private_once(path, payload)


def load_private_cell_reader(path: Path, expected_sha256: str, *,
                             private_root: Path) -> CellReader:
    """Restore a frozen specialist after private path, byte and schema checks."""
    declared = Path(path)
    source = declared.resolve(strict=True)
    root = Path(private_root).resolve(strict=True)
    if (".private" not in root.parts or (root.stat().st_mode & 0o077)
            or not source.is_relative_to(root)
            or declared.is_symlink() or (source.stat().st_mode & 0o077)
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        raise PermissionError("D cell reader artifact is outside owner-only custody")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("D cell reader artifact hash mismatch")
    state = json.loads(payload)
    if (state.get("schema") != "pcrl-d-cell-reader-v1"
            or state.get("fit_role") != "ATTACK_FIT_NEW/ATTACK_TRAIN"
            or state.get("alpha") not in ALPHAS
            or state.get("columns") not in (1, 2)
            or not isinstance(state.get("fit_rows"), int)
            or state["fit_rows"] < 1):
        raise ValueError("D cell reader artifact schema mismatch")
    for field in ("fit_row_id_sha256", "fit_codes_sha256", "fit_sex_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", state.get(field, "")) is None:
            raise ValueError("D cell reader fitting identity absent")
    counts = {tuple(int(v) for v in key): tuple(int(v) for v in val)
              for key, val in state["counts"]}
    marginals = tuple({int(key): tuple(int(v) for v in val) for key, val in table}
                      for table in state["marginal_counts"])
    if len(marginals) != state["columns"]:
        raise ValueError("D cell reader marginal width mismatch")
    return CellReader(float(state["alpha"]), int(state["columns"]),
                      float(state["prior"]), counts, marginals)


def fit_d_cell_readers(att_aligned: AlignedDViews, inner_aligned: AlignedDViews,
                       view: str, att_s2_view, inner_s2_view,
                       sex_att: dict, s2_attack_module, *,
                       model_dir: Path, private_root: Path) -> tuple[dict, ...]:
    """Fit exactly three alphas on ATTACK_TRAIN; freeze TUNE/INNER predictions."""
    att_s2_view.validate()
    inner_s2_view.validate()
    if view not in ("recipient0", "recipient1", "pair"):
        raise ValueError("unknown D attack view")
    if any(not np.array_equal(a, b) for a, b in (
            (att_aligned.row_id, att_s2_view.row_id),
            (att_aligned.unit_id, att_s2_view.unit),
            (inner_aligned.row_id, inner_s2_view.row_id),
            (inner_aligned.unit_id, inner_s2_view.unit),
            (att_aligned.row_id, sex_att["row_id"]),
            (att_aligned.unit_id, sex_att["unit"]))):
        raise ValueError("D cell fit row/unit order disagrees with S2 roles")
    y = np.asarray(sex_att["sex"], dtype=np.int64)
    if y.shape != att_aligned.row_id.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("ATTACK_FIT_NEW SEX must be aligned and binary")
    fit_tune = s2_attack_module.attack_fit_tune_split(att_s2_view.unit)
    fit, tune = fit_tune["fit"], fit_tune["tune"]
    if np.unique(y[fit]).size != 2 or np.unique(y[tune]).size != 2:
        raise ValueError("D cell attack fit/tune split lacks both SEX classes")
    codes_att = att_aligned.codes(view)
    codes_inner = inner_aligned.codes(view)
    if codes_att.shape[1] not in (1, 2) or codes_inner.shape[1] != codes_att.shape[1]:
        raise ValueError("D direct cell reader requires the same finite-code view in both roles")
    for codes, s2_view in ((codes_att, att_s2_view), (codes_inner, inner_s2_view)):
        expected_key = (codes[:, 0].astype(str) if codes.shape[1] == 1 else
                        np.asarray([f"{int(a)}|{int(b)}" for a, b in codes]))
        if s2_view.cell_key is None or not np.array_equal(np.asarray(s2_view.cell_key), expected_key):
            raise ValueError("S2 categorical cell key differs from D's released hard IDs")
    out = []
    for alpha in ALPHAS:
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        reader = CellReader.fit(codes_att[fit], y[fit], alpha)
        name = f"D_CELL_A{alpha:g}"
        payload = _state_payload(reader, att_aligned.row_id[fit],
                                 codes_att[fit], y[fit])
        artifact_path, artifact_sha = _save_private_reader(
            payload, model_dir, private_root, f"{view}__{name}")
        prediction_tune = reader.predict_proba(codes_att[tune])[:, 1]
        prediction_inner = reader.predict_proba(codes_inner)[:, 1]
        buffer = io.BytesIO()
        np.savez_compressed(buffer, prediction_tune=prediction_tune,
                            prediction_inner=prediction_inner)
        prediction_path = _private_artifact_path(
            model_dir, private_root, f"predictions__{view}__{name}", ".npz")
        prediction_sha = _write_private_once(prediction_path, buffer.getvalue())
        completed = {
            "schema": "pcrl-d-cell-reader-completed-v1",
            "status": "COMPLETE", "name": name, "view": view,
            "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "fit_rows": int(len(fit)), "tune_rows": int(len(tune)),
            "model_artifact_path": artifact_path, "model_sha256": artifact_sha,
            "prediction_artifact_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "att_row_id_sha256": _array_sha(att_aligned.row_id),
            "inner_row_id_sha256": _array_sha(inner_aligned.row_id),
            "fit_predict_artifact_wall_seconds": time.perf_counter() - wall_start,
            "fit_predict_artifact_cpu_seconds": time.process_time() - cpu_start,
        }
        receipt_path = _private_artifact_path(
            model_dir, private_root, f"completed__{view}__{name}")
        receipt_payload = (json.dumps(completed, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False) + "\n").encode()
        receipt_sha = _write_private_once(receipt_path, receipt_payload)
        out.append({
            "name": name,
            "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "model_sha256": artifact_sha,
            "model_artifact_path": artifact_path,
            "prediction_tune": prediction_tune,
            "prediction_inner": prediction_inner,
            "prediction_artifact_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "completion_receipt_path": str(receipt_path),
            "completion_receipt_sha256": receipt_sha,
        })
    return tuple(out)

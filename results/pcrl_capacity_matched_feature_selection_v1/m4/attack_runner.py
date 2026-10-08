"""One complete-view ATT reader unit with transported RAW_ONLY readers.

The caller first proves the effective queue unit and restores the immutable
release. This module fits the four shared readers on ATTACK_FIT_NEW only;
the two RAW_ONLY models come from one separately queued canonical-union unit.
Every score and model stays in private write-once custody.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from attack_eval import (attack_fit_tune_split, fit_slate, oriented_auc,
                         save_fitted_slate, score_slate)
from raw_union import RAW_READER_IDS, canonical_raw_union
from views import transport_raw_reader
from safe_data_guard import sha_arr


CORE_READER_IDS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")
ALL_READER_IDS = CORE_READER_IDS + RAW_READER_IDS


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once_json(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_file(path)


def _same_raw_view(view: object, union: object) -> None:
    raw = np.asarray(view.x_raw_only)
    if (raw.dtype != union.x.dtype or raw.shape != union.x.shape or
            np.ascontiguousarray(raw).tobytes() != union.x.tobytes() or
            not np.array_equal(view.row_id, union.row_id) or
            not np.array_equal(view.unit, union.unit)):
        raise ValueError("complete view raw disclosure differs from canonical source union")


def _transport_scores(
    *, raw_index_path: Path, expected_raw_index_sha256: str,
    att_union: object, inner_union: object, att_view: object,
    inner_view: object,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    if ".private" not in raw_index_path.resolve(strict=True).parts or \
            _sha_file(raw_index_path) != expected_raw_index_sha256:
        raise PermissionError("RAW_ONLY source index identity differs")
    index = json.loads(raw_index_path.read_text())
    plan = index.get("plan", {})
    if (index.get("schema") != "pcrl-focused-raw-union-reader-index-v1"
            or index.get("status") != "FROZEN"
            or plan.get("att_union_sha256") != att_union.content_sha256
            or plan.get("inner_union_sha256") != inner_union.content_sha256
            or plan.get("source_columns") != list(att_union.source_columns)
            or set(index.get("readers", {})) != set(RAW_READER_IDS)):
        raise PermissionError("RAW_ONLY source union differs from released view")
    split = attack_fit_tune_split(att_view.unit)
    if split["tune_group_id_sha256"] != plan["attack_split_sha256"]:
        raise PermissionError("RAW_ONLY ATT fit/tune split differs")
    outputs, diagnostics = {}, {}
    for reader_id in RAW_READER_IDS:
        reader = index["readers"][reader_id]
        model_path = raw_index_path.parent / reader["model_file"]
        if _sha_file(model_path) != reader["model_sha256"]:
            raise PermissionError("RAW_ONLY model artifact changed")
        predictions, checks = {}, {}
        for label, source, view in (("att", att_union, att_view),
                                    ("inner", inner_union, inner_view)):
            entry = reader["roles"][label]
            path = raw_index_path.parent / entry["prediction_file"]
            if _sha_file(path) != entry["prediction_file_sha256"]:
                raise PermissionError("RAW_ONLY prediction artifact changed")
            saved = np.load(path, allow_pickle=False)
            transported = transport_raw_reader(
                source, source, saved,
                raw_index_path.parent / entry["transport_receipt_file"],
                entry["transport_receipt_sha256"], model_path,
                expected_attack_split_sha256=plan["attack_split_sha256"],
            )
            _same_raw_view(view, source)
            predictions[label] = transported.prediction
            checks[label] = {
                "transport_receipt_sha256": transported.reader_receipt_sha256,
                "max_abs_replay_delta": transported.max_abs_replay_delta,
                "tolerance": transported.tolerance,
            }
        outputs[reader_id] = {
            "tune": predictions["att"][split["tune"]].copy(),
            "inner": predictions["inner"].copy(),
        }
        diagnostics[reader_id] = checks
    return outputs, diagnostics


def _ordered_pair_sha(view: object) -> str:
    row = np.asarray(view.row_id)
    unit = np.asarray(view.unit)
    if (row.dtype != np.dtype("int64") or unit.dtype != np.dtype("int64")
            or row.shape != unit.shape or row.ndim != 1):
        raise ValueError("complete view needs exact int64 row/group identity")
    return sha_arr(np.column_stack((row, unit)))


def load_local_reader_aliases(
    *, local_index_path: str | Path, expected_index_sha256: str,
    expected_view_name: str, expected_pair_id: str,
    att_local_view: object, inner_local_view: object,
    seed: int, lock_sha256: str, manifest_sha256: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Import one local unit's six saved scores for ignore-recipient attacks."""
    path = Path(local_index_path).resolve(strict=True)
    if ".private" not in path.parts or _sha_file(path) != expected_index_sha256:
        raise PermissionError("local reader index custody changed")
    index = json.loads(path.read_text())
    if (index.get("schema") != "pcrl-focused-complete-reader-index-v1"
            or index.get("status") != "FROZEN"
            or index.get("view_name") != expected_view_name
            or index.get("pair_id") != expected_pair_id
            or index.get("seed") != seed
            or index.get("lock_sha256") != lock_sha256
            or index.get("manifest_sha256") != manifest_sha256
            or index.get("att_view_content_sha256") != att_local_view.content_hashes()
            or index.get("inner_view_content_sha256") != inner_local_view.content_hashes()
            or index.get("att_row_unit_sha256") != _ordered_pair_sha(att_local_view)
            or index.get("inner_row_unit_sha256") != _ordered_pair_sha(inner_local_view)
            or index.get("attack_split_sha256") !=
            attack_fit_tune_split(att_local_view.unit)["tune_group_id_sha256"]
            or tuple(index.get("reader_order", ())) != ALL_READER_IDS):
        raise PermissionError("local reader does not match the recipient wire/person/split")
    core_path = path.parent / "core_readers/READERS_RECEIPT.json"
    if _sha_file(core_path) != index["core_readers_receipt_sha256"]:
        raise PermissionError("local core reader custody changed")
    core = json.loads(core_path.read_text())
    for filename, expected in core["files"].items():
        if _sha_file(core_path.parent / filename) != expected:
            raise PermissionError("local fitted model or prediction changed")
    outputs = {}
    for role, filename, hash_key in (
        ("inner", "ALL_READER_PREDICTIONS.npz", "all_reader_predictions_sha256"),
        ("tune", "ALL_READER_TUNE_PREDICTIONS.npz",
         "all_reader_tune_predictions_sha256"),
    ):
        score_path = path.parent / filename
        if _sha_file(score_path) != index[hash_key]:
            raise PermissionError("local saved reader scores changed")
        with np.load(score_path, allow_pickle=False) as archive:
            if set(archive.files) != set(ALL_READER_IDS):
                raise PermissionError("local fixed reader slate is incomplete")
            outputs[role] = {name: np.asarray(archive[name], dtype=np.float64).copy()
                             for name in ALL_READER_IDS}
    expected_lengths = {
        "inner": len(inner_local_view.row_id),
        "tune": len(attack_fit_tune_split(att_local_view.unit)["tune"]),
    }
    for role in ("inner", "tune"):
        for prediction in outputs[role].values():
            if (prediction.shape != (expected_lengths[role],)
                    or not np.isfinite(prediction).all()
                    or np.any((prediction < 0) | (prediction > 1))):
                raise PermissionError("local reader prediction shape or range differs")
    return outputs, {"index_sha256": expected_index_sha256,
                     "core_readers_receipt_sha256":
                     index["core_readers_receipt_sha256"],
                     "source_view_name": expected_view_name,
                     "reader_count": len(ALL_READER_IDS)}


def fit_complete_view_readers(
    *, att_view: object, sex_att: Mapping[str, object],
    inner_view: object, sex_inner: Mapping[str, object],
    att_source: Mapping[str, object], inner_source: Mapping[str, object],
    source_columns: tuple[int, ...], feature_names: tuple[str, ...],
    raw_index_path: str | Path, expected_raw_index_sha256: str,
    release_index_sha256: str, lock_sha256: str, manifest_sha256: str,
    seed: int, pair_id: str, view_name: str, private_output: str | Path,
    coalition_local_dependencies: Mapping[str, tuple] | None = None,
) -> dict:
    """Fit/score the fixed six-reader view with two zero-fit raw transports."""
    if view_name not in ("income", "occupation_group", "coalition"):
        raise ValueError("complete-view name is invalid")
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("paired seed is not frozen")
    if (view_name == "coalition" and
            (coalition_local_dependencies is None or
             set(coalition_local_dependencies) != {"income", "occupation_group"})):
        raise ValueError("coalition must include both recipient reader dependencies")
    if view_name != "coalition" and coalition_local_dependencies is not None:
        raise ValueError("local reader cannot import a coalition dependency")
    out = Path(private_output).resolve()
    if ".private" not in out.parts or out.exists():
        raise PermissionError("complete-reader output must be a fresh private directory")
    att_union = canonical_raw_union(
        att_source, source_columns=source_columns, feature_names=feature_names)
    inner_union = canonical_raw_union(
        inner_source, source_columns=source_columns, feature_names=feature_names)
    _same_raw_view(att_view, att_union)
    _same_raw_view(inner_view, inner_union)
    raw, diagnostics = _transport_scores(
        raw_index_path=Path(raw_index_path),
        expected_raw_index_sha256=expected_raw_index_sha256,
        att_union=att_union, inner_union=inner_union,
        att_view=att_view, inner_view=inner_view,
    )
    # S2 freezes all fitted reader configurations and directions on ATT first.
    fitted = fit_slate(
        att_view, dict(sex_att), inner_view,
        required_readers=CORE_READER_IDS, attacker_seed=seed,
    )
    out.mkdir(parents=True, mode=0o700)
    core = save_fitted_slate(
        fitted, out / "core_readers", att_view=att_view, inner_view=inner_view)
    split = attack_fit_tune_split(att_view.unit)
    y_tune = np.asarray(sex_att["sex"], dtype=np.int64)[split["tune"]]
    joined = {**fitted,
              "predictions_tune": dict(fitted["predictions_tune"]),
              "predictions_inner": dict(fitted["predictions_inner"]),
              "validation": dict(fitted["validation"]),
              "reader_order": list(fitted["reader_order"])}
    for reader_id in RAW_READER_IDS:
        joined["predictions_tune"][reader_id] = raw[reader_id]["tune"]
        joined["predictions_inner"][reader_id] = raw[reader_id]["inner"]
        validation = oriented_auc(y_tune, raw[reader_id]["tune"])
        validation["selected_orientation"] = (
            "positive" if validation["positive_alias_auc"] >=
            validation["reverse_alias_auc"] else "reverse"
        )
        joined["validation"][reader_id] = validation
        joined["reader_order"].append(reader_id)
    local_alias_receipts = {}
    if view_name == "coalition":
        for recipient, ignored in (("income", "R2"),
                                   ("occupation_group", "R1")):
            local_path, local_sha, att_local, inner_local = \
                coalition_local_dependencies[recipient]
            imported, custody = load_local_reader_aliases(
                local_index_path=local_path, expected_index_sha256=local_sha,
                expected_view_name=recipient, expected_pair_id=pair_id,
                att_local_view=att_local, inner_local_view=inner_local,
                seed=seed, lock_sha256=lock_sha256,
                manifest_sha256=manifest_sha256,
            )
            for reader_id in ALL_READER_IDS:
                alias = f"IGNORE_{ignored}__{reader_id}"
                joined["predictions_tune"][alias] = imported["tune"][reader_id]
                joined["predictions_inner"][alias] = imported["inner"][reader_id]
                validation = oriented_auc(y_tune, imported["tune"][reader_id])
                validation["selected_orientation"] = (
                    "positive" if validation["positive_alias_auc"] >=
                    validation["reverse_alias_auc"] else "reverse"
                )
                joined["validation"][alias] = validation
                joined["reader_order"].append(alias)
            local_alias_receipts[recipient] = custody
    winner = joined["reader_order"][0]
    best = -1.0
    for reader_id in joined["reader_order"]:
        value = joined["validation"][reader_id]["fixed_alias_max_auc"]
        if value > best + 1e-12:
            winner, best = reader_id, value
    joined["validation_winner"] = winner
    joined["validation_winner_orientation"] = \
        joined["validation"][winner]["selected_orientation"]
    score = score_slate(joined, dict(sex_inner))
    predictions_path = out / "ALL_READER_PREDICTIONS.npz"
    with predictions_path.open("xb") as stream:
        np.savez_compressed(stream, **joined["predictions_inner"])
        stream.flush()
        os.fsync(stream.fileno())
    predictions_path.chmod(0o600)
    tune_path = out / "ALL_READER_TUNE_PREDICTIONS.npz"
    with tune_path.open("xb") as stream:
        np.savez_compressed(stream, **joined["predictions_tune"])
        stream.flush()
        os.fsync(stream.fileno())
    tune_path.chmod(0o600)
    index = {
        "schema": "pcrl-focused-complete-reader-index-v1",
        "status": "FROZEN", "seed": seed, "pair_id": pair_id,
        "view_name": view_name, "lock_sha256": lock_sha256,
        "manifest_sha256": manifest_sha256,
        "release_index_sha256": release_index_sha256,
        "raw_index_sha256": expected_raw_index_sha256,
        "att_row_unit_sha256": _ordered_pair_sha(att_view),
        "inner_row_unit_sha256": _ordered_pair_sha(inner_view),
        "attack_split_sha256": split["tune_group_id_sha256"],
        "att_view_content_sha256": att_view.content_hashes(),
        "inner_view_content_sha256": inner_view.content_hashes(),
        "core_readers_receipt_sha256": core["private_receipt_sha256"],
        "raw_transport_diagnostics": diagnostics,
        "local_ignore_alias_receipts": local_alias_receipts,
        "reader_order": joined["reader_order"],
        "validation": joined["validation"],
        "validation_winner": joined["validation_winner"],
        "validation_winner_orientation": joined["validation_winner_orientation"],
        "inner_score": score,
        "all_reader_predictions_sha256": _sha_file(predictions_path),
        "all_reader_tune_predictions_sha256": _sha_file(tune_path),
        "fresh_fit_count": 4, "transported_raw_reader_count": 2,
    }
    _write_once_json(out / "ATTACK_INDEX.json", index)
    return index

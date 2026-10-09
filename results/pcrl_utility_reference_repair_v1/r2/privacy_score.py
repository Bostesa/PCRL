"""Score frozen ATT readers on INNER SEX only after all fitting is complete."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    array_sha, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
    CORE, RAW, SCHEMA as ATT_SCHEMA,
)
from results.pcrl_utility_reference_repair_v1.r2.control_fit import SCHEMA as CONTROL_SCHEMA


REPO = Path(__file__).resolve().parents[3]
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from attack_eval import oriented_auc  # noqa: E402
from attack_controls_v2 import (  # noqa: E402
    SHUFFLE_INNER_SALT, groupwise_shuffled_labels,
)


def _private(path: Path, root: Path, expected_sha: str, label: str) -> Path:
    real = path.resolve(strict=True)
    if (not real.is_relative_to(root) or not isinstance(expected_sha, str)
            or len(expected_sha) != 64 or sha_file(real) != expected_sha):
        raise PermissionError(f"{label} private bytes differ")
    return real


def _inner_labels(sex_inner: Mapping[str, object]) -> tuple[np.ndarray, str]:
    row = np.asarray(sex_inner["row_id"])
    unit = np.asarray(sex_inner["unit"])
    y = np.asarray(sex_inner["sex"])
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or y.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or y.shape != row.shape
            or len(row) != 2235 or len(np.unique(unit)) != 2234
            or np.unique(y).tolist() != [0, 1]):
        raise PermissionError("INNER SEX row/group/label identity differs")
    return y, array_sha(np.column_stack((row, unit)))


def _control_scores(index_path: Path, index: dict, sex_inner: dict,
                    private_root: Path) -> dict:
    folder = index_path.parent / "controls"
    receipt_path = _private(
        folder / "CONTROL_FIT_RECEIPT.json", private_root,
        index["control_fit_receipt_sha256"], "ATT control fit receipt")
    receipt = json.loads(receipt_path.read_text())
    y, pair_sha = _inner_labels(sex_inner)
    if (receipt.get("schema") != CONTROL_SCHEMA
            or receipt.get("seed") != index["seed"]
            or receipt.get("inner_row_unit_sha256") != pair_sha
            or receipt.get("inner_view_content_sha256") !=
            index["inner_view_content_sha256"]
            or receipt.get("INNER_SEX_read") is not False
            or receipt.get("fresh_fit_count") != 2
            or receipt.get("INNER_shuffle_salt_for_later_score") !=
            SHUFFLE_INNER_SALT):
        raise PermissionError("ATT control fit lineage differs")
    files = receipt["files_sha256"]
    if set(files) != {"shuffled_model.joblib", "planted_model.joblib",
                      "shuffled_inner.npy"}:
        raise PermissionError("ATT control model inventory differs")
    for name, sha in files.items():
        _private(folder / name, private_root, sha, "ATT control " + name)
    shuffled = np.load(folder / "shuffled_inner.npy", allow_pickle=False)
    if (shuffled.dtype != np.float64 or shuffled.shape != y.shape
            or array_sha(shuffled) != receipt["shuffled_prediction_array_sha256"]):
        raise PermissionError("ATT shuffled prediction bytes differ")
    core_path = _private(
        index_path.parent / "core/READERS_RECEIPT.json", private_root,
        index["core_readers_receipt_sha256"], "core reader custody")
    core = json.loads(core_path.read_text())
    for name, expected_sha in core.get("files", {}).items():
        _private(core_path.parent / name, private_root, expected_sha,
                 "core reader " + name)
    view_file = core_path.parent / "view_inner.npz"
    _private(view_file, private_root, core["files"]["view_inner.npz"],
             "core INNER view")
    with np.load(view_file, allow_pickle=False) as archive:
        x_inner = np.asarray(archive["x"], dtype=np.float64)
        row = np.asarray(archive["row_id"], dtype=np.int64)
        unit = np.asarray(archive["unit"], dtype=np.int64)
    if (x_inner.shape[0] != len(y)
            or not np.array_equal(row, sex_inner["row_id"])
            or not np.array_equal(unit, sex_inner["unit"])):
        raise PermissionError("planted engineering view people differ")
    shuffled_model = joblib.load(folder / "shuffled_model.joblib")
    replay = np.asarray(shuffled_model.predict_proba(x_inner)[:, 1],
                        dtype=np.float64)
    if not np.array_equal(replay, shuffled):
        raise PermissionError("ATT shuffled control model replay differs")
    planted_model = joblib.load(folder / "planted_model.joblib")
    planted = np.asarray(planted_model.predict_proba(
        np.column_stack((x_inner, y)))[:, 1], dtype=np.float64)
    independent = groupwise_shuffled_labels(y, sex_inner["unit"],
                                             salt=SHUFFLE_INNER_SALT)
    null_auc = oriented_auc(y, np.full(len(y), .5))["fixed_alias_max_auc"]
    shuffle_true = oriented_auc(y, shuffled)["fixed_alias_max_auc"]
    shuffle_independent = oriented_auc(
        independent, shuffled)["fixed_alias_max_auc"]
    planted_auc = oriented_auc(y, planted)["fixed_alias_max_auc"]
    gates = {
        "null_exact": null_auc == .5,
        "shuffled_independent_inner_at_most_0.60":
        shuffle_independent <= .60,
        "planted_at_least_0.99": planted_auc >= .99,
    }
    return {
        "schema": "pcrl-utility-control-score-v1",
        "null_auc": null_auc,
        "shuffled_att_true_inner_auc": shuffle_true,
        "shuffled_att_independent_inner_auc": shuffle_independent,
        "planted_auc": planted_auc,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "control_fit_receipt_sha256": index["control_fit_receipt_sha256"],
        "planted_prediction_sha256": array_sha(planted),
    }


def score_saved_attack_view(
    index_path: str | Path, expected_index_sha256: str,
    sex_inner: dict, *, private_root: str | Path,
    seed: int, pair_id: str, view_name: str,
    attack_fit_lock_sha256: str, manifest_sha256: str,
    release_index_sha256: str,
    local_attack_source_lock_sha256: str,
) -> dict:
    """Return reader/control aggregates; no reader fit or selection occurs."""
    root = Path(private_root).resolve(strict=True)
    if root.name != ".private":
        raise PermissionError("privacy scoring requires private custody")
    path = _private(Path(index_path), root, expected_index_sha256,
                    "ATT reader index")
    index = json.loads(path.read_text())
    if (index.get("schema") != ATT_SCHEMA or index.get("status") != "FROZEN"
            or index.get("seed") != seed or index.get("pair_id") != pair_id
            or index.get("view_name") != view_name
            or index.get("lock_sha256") != attack_fit_lock_sha256
            or index.get("local_attack_source_lock_sha256") != (
                local_attack_source_lock_sha256 if view_name == "coalition"
                else None)
            or index.get("manifest_sha256") != manifest_sha256
            or index.get("release_index_sha256") != release_index_sha256
            or index.get("fresh_fit_count") != 6
            or index.get("INNER_SEX_read") is not False):
        raise PermissionError("ATT reader source or fit contract differs")
    y, pair_sha = _inner_labels(sex_inner)
    if index.get("inner_row_unit_sha256") != pair_sha:
        raise PermissionError("ATT reader INNER row/group identity differs")
    score_file = _private(
        path.parent / "ALL_READER_PREDICTIONS.npz", root,
        index["all_prediction_sha256"], "ATT reader saved predictions")
    with np.load(score_file, allow_pickle=False) as archive:
        expected_readers = set(CORE + RAW)
        if view_name == "coalition":
            expected_readers |= {f"IGNORE_R{ignored}__{reader}"
                                 for ignored in (1, 2)
                                 for reader in CORE + RAW}
        if (len(index["reader_order"]) != len(expected_readers)
                or set(archive.files) != expected_readers
                or set(index["reader_order"]) != expected_readers):
            raise PermissionError("ATT saved reader inventory differs")
        predictions = {name: np.asarray(archive[name], dtype=np.float64)
                       for name in archive.files}
    core_path = _private(path.parent / "core/READERS_RECEIPT.json", root,
                         index["core_readers_receipt_sha256"],
                         "core reader receipt")
    core = json.loads(core_path.read_text())
    core_predictions_path = _private(
        core_path.parent / "predictions_inner.npz", root,
        core["files"]["predictions_inner.npz"], "core reader scores")
    with np.load(core_predictions_path, allow_pickle=False) as archive:
        if set(archive.files) != set(CORE):
            raise PermissionError("core reader inventory differs")
        if any(not np.array_equal(archive[name], predictions[name])
               for name in CORE):
            raise PermissionError("core reader saved score transport differs")
    scores = {name: oriented_auc(y, values) for name, values in
              predictions.items()}
    winner = index["validation_winner"]
    if winner not in scores or winner not in index["validation"]:
        raise PermissionError("ATT validation-selected reader missing")
    raw_auc = scores[winner]["positive_alias_auc"]
    direction = index["validation_winner_orientation"]
    if direction not in ("positive", "reverse"):
        raise PermissionError("ATT validation-selected orientation missing")
    controls = _control_scores(path, index, sex_inner, root)
    return {
        "schema": "pcrl-utility-complete-privacy-score-v1",
        "seed": seed, "pair_id": pair_id, "view_name": view_name,
        "attack_index_sha256": expected_index_sha256,
        "inner_row_unit_sha256": pair_sha,
        "reader_order": index["reader_order"],
        "readers": scores,
        "primary_envelope_auc": max(item["fixed_alias_max_auc"]
                                    for item in scores.values()),
        "validation_winner": winner,
        "validation_winner_orientation": direction,
        "validation_winner_inner_auc": raw_auc if direction == "positive"
        else 1 - raw_auc,
        "controls": controls,
        "development_only": True,
    }

"""Independent check of saved fixed-reader artifacts and their predictions.

This module reads private custody only when called by an R03 verification job.
It does not import the inherited attack runner or R02's wrappers. Ordinary
joblib/sklearn restore is disclosed as a dependency, not an independent refit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .independent_replay import weighted_binary_auc


_ATT_TUNE_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_scores(model, keys: np.ndarray) -> np.ndarray:
    """Independent Laplace-smoothed table scoring with unseen-key prior."""
    counts, prior = model
    if not isinstance(counts, dict) or not 0 <= prior <= 1:
        raise ValueError("cell model invalid")
    output = np.empty(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        observed = counts.get(str(key))
        output[index] = prior if observed is None else (observed[1] + 1) / (sum(observed) + 2)
    return output


def _reader_matrix(name: str, view: Any) -> np.ndarray:
    if name == "DA_MLP_64x32":
        key = "x_defense_aware"
    elif name == "META_LR_C1":
        key = "x_metadata"
    elif name == "CODE_BIT_LR_C1":
        key = "x_code_bits"
    elif name in ("RAW_LR_C1", "RAW_HGB_15x200"):
        key = "x_raw_only"
    else:
        key = "x"
    if key not in view:
        raise ValueError(f"required saved public view {key} missing for {name}")
    return np.asarray(view[key], dtype=np.float64)


def verify_reader_custody(
    root: str | Path,
    *,
    expected_row_id: np.ndarray,
    expected_unit: np.ndarray,
    reforward_names: tuple[str, ...] = ("LR_C1", "MLP_64x32", "HGB_15x200", "CELL_A1"),
    tolerance: float = 1e-10,
) -> dict:
    """Hash all custody files and reforward selected full INNER reader arrays."""
    directory = Path(root).resolve()
    if ".private" not in directory.parts:
        raise ValueError("reader custody must remain private")
    receipt_path = directory / "READERS_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "FROZEN" or not isinstance(receipt.get("files"), dict):
        raise ValueError("frozen reader receipt missing")
    for name, expected_sha in receipt["files"].items():
        if Path(name).name != name or _sha256(directory / name) != expected_sha:
            raise ValueError(f"reader artifact hash mismatch: {name}")
    with np.load(directory / "view_inner.npz", allow_pickle=False) as view, \
            np.load(directory / "predictions_inner.npz", allow_pickle=False) as predictions:
        row_id = np.asarray(view["row_id"])
        unit = np.asarray(view["unit"])
        if (not np.array_equal(row_id, np.asarray(expected_row_id))
                or not np.array_equal(unit, np.asarray(expected_unit))):
            raise ValueError("reader INNER role or ordered group identity differs")
        names = tuple(receipt["reader_order"])
        if set(names) != set(predictions.files) or len(set(names)) != len(names):
            raise ValueError("saved prediction bank differs from fixed reader order")
        n = len(row_id)
        for name in names:
            values = np.asarray(predictions[name], dtype=np.float64)
            if values.shape != (n,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"reader prediction invalid: {name}")
        replay = {}
        for name in reforward_names:
            model_path = directory / f"model__{name}.joblib"
            if name not in names or not model_path.exists():
                replay[name] = "NOT_AVAILABLE"
                continue
            model = joblib.load(model_path)
            if name == "CELL_A1":
                if "cell_key" not in view:
                    raise ValueError("cell reader has no public cell keys")
                actual = _cell_scores(model, np.asarray(view["cell_key"]))
            else:
                actual = model.predict_proba(_reader_matrix(name, view))[:, 1]
            maximum = float(np.max(np.abs(actual - np.asarray(predictions[name], dtype=np.float64))))
            if maximum > tolerance:
                raise ValueError(f"reader model/prediction mismatch: {name}")
            replay[name] = maximum
        return {"status": "HASH_AND_FORWARD_PASS", "rows": n,
                "reader_count": len(names), "artifact_count": len(receipt["files"]),
                "reforward_max_abs": replay,
                "receipt_sha256": _sha256(receipt_path)}


def verify_ignore_aliases(
    pair_dir: str | Path, local_r1_dir: str | Path, local_r2_dir: str | Path
) -> dict:
    """Coalition's zero-fit ignore aliases equal the corresponding local bytes."""
    pair_root = Path(pair_dir).resolve()
    report = json.loads((pair_root / "READERS_RECEIPT.json").read_text())
    aliases = report.get("ignore_recipient_aliases", [])
    if not aliases:
        raise ValueError("pair reader receipt lacks ignore-recipient aliases")
    with np.load(pair_root / "predictions_inner.npz", allow_pickle=False) as pair, \
            np.load(Path(local_r1_dir) / "predictions_inner.npz", allow_pickle=False) as r1, \
            np.load(Path(local_r2_dir) / "predictions_inner.npz", allow_pickle=False) as r2:
        for record in aliases:
            source = r1 if record["source_view"] == "r1" else r2
            if record["new_fit"] or not np.array_equal(pair[record["alias"]], source[record["source_reader"]]):
                raise ValueError("zero-fit coalition ignore alias differs from local reader")
    return {"status": "EXACT_IGNORE_ALIAS_PASS", "aliases": len(aliases)}


def verify_tune_selection(reader_dir: str | Path, att_row_id: np.ndarray,
                          att_unit: np.ndarray, sex_att: np.ndarray) -> dict:
    """Recreate the fixed group split, all TUNE AUCs and selected orientation."""
    root = Path(reader_dir).resolve()
    if ".private" not in root.parts:
        raise PermissionError("reader tuning arrays must remain private")
    receipt = json.loads((root / "READERS_RECEIPT.json").read_text())
    row = np.asarray(att_row_id, dtype=np.int64)
    unit = np.asarray(att_unit, dtype=np.int64)
    y = np.asarray(sex_att, dtype=np.int64)
    if (row.shape != unit.shape or row.shape != y.shape or len(row) != 4064
            or np.unique(unit).size != 4061 or not np.isin(y, (0, 1)).all()):
        raise ValueError("ATT tuning role identity or SEX differs")
    ranked = sorted((hashlib.sha256((_ATT_TUNE_SALT + str(int(g))).encode()).hexdigest(), int(g))
                    for g in np.unique(unit))
    tune_groups = np.asarray([group for _, group in ranked[:1015]], dtype=np.int64)
    idx = np.flatnonzero(np.isin(unit, tune_groups))
    digest = hashlib.sha256()
    ordered = np.ascontiguousarray(np.sort(tune_groups))
    digest.update(str(ordered.dtype).encode() + str(ordered.shape).encode())
    digest.update(ordered.tobytes())
    if digest.hexdigest() != receipt["split"]["tune_group_id_sha256"]:
        raise ValueError("reader TUNE group split differs from fixed hash ranking")
    with np.load(root / "view_att.npz", allow_pickle=False) as view:
        if not np.array_equal(view["row_id"], row) or not np.array_equal(view["unit"], unit):
            raise ValueError("reader TUNE view people differ")
    with np.load(root / "predictions_tune.npz", allow_pickle=False) as pred:
        names = tuple(receipt["reader_order"])
        if set(pred.files) != set(names):
            raise ValueError("TUNE prediction bank differs from fixed reader slate")
        scores = {}
        for name in names:
            values = np.asarray(pred[name], dtype=np.float64)
            if values.shape != (len(idx),) or not np.isfinite(values).all():
                raise ValueError("TUNE prediction shape/value differs")
            auc = weighted_binary_auc(values, y[idx])
            selected = "positive" if auc >= 1 - auc else "reverse"
            oriented = max(auc, 1 - auc)
            saved = receipt["validation"][name]
            if (abs(saved["positive_alias_auc"] - auc) > 1e-10
                    or abs(saved["fixed_alias_max_auc"] - oriented) > 1e-10
                    or saved["selected_orientation"] != selected):
                raise ValueError("TUNE validation score/orientation differs")
            scores[name] = {"auc": oriented, "orientation": selected}
    winner = names[0]
    best = -1.0
    for name in names:
        if scores[name]["auc"] > best + 1e-12:
            winner, best = name, scores[name]["auc"]
    if (receipt["validation_winner"] != winner
            or receipt["validation_winner_orientation"] != scores[winner]["orientation"]):
        raise ValueError("frozen TUNE winner differs from independent selection")
    return {"status": "TUNE_SELECTION_PASS", "fit_groups": 3046,
            "tune_groups": 1015, "tune_rows": len(idx), "winner": winner,
            "orientation": scores[winner]["orientation"], "winner_auc": best}


def verify_control_custody(control_dir: str | Path, expected_row_id: np.ndarray,
                           expected_unit: np.ndarray, tolerance: float = 1e-10) -> dict:
    """Independently reforward and score frozen real-role null/planted probes."""
    root = Path(control_dir).resolve()
    if ".private" not in root.parts:
        raise PermissionError("engineering controls must remain private")
    receipt_path = root / "CUSTODY_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "FROZEN" or not isinstance(receipt.get("files"), dict):
        raise ValueError("engineering control custody is not frozen")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or _sha256(root / name) != expected:
            raise ValueError("engineering control artifact hash differs")
    with np.load(root / "probe_replay.npz", allow_pickle=False) as archive:
        z = {key: np.array(archive[key], copy=True) for key in archive.files}
    needed = {"att_fit_row_id", "att_fit_unit", "att_fit_x", "att_fit_true_sex",
              "att_fit_shuffled_sex", "inner_row_id", "inner_unit", "inner_x",
              "inner_true_sex", "inner_shuffled_sex", "shuffled_prediction_inner",
              "planted_prediction_inner"}
    if (set(z) != needed or not np.array_equal(z["inner_row_id"], expected_row_id)
            or not np.array_equal(z["inner_unit"], expected_unit)):
        raise ValueError("engineering probe member/INNER identity differs")
    null = 0.5
    shuffled_model = joblib.load(root / "shuffled_model.joblib")
    planted_model = joblib.load(root / "planted_model.joblib")
    shuffled = shuffled_model.predict_proba(z["inner_x"])[:, 1]
    planted = planted_model.predict_proba(
        np.column_stack((z["inner_x"], z["inner_true_sex"])))[:, 1]
    deviations = {
        "shuffled": float(np.max(np.abs(shuffled - z["shuffled_prediction_inner"]))),
        "planted": float(np.max(np.abs(planted - z["planted_prediction_inner"]))),
    }
    if max(deviations.values()) > tolerance:
        raise ValueError("engineering probe saved predictions differ from model")
    shuffled_auc = weighted_binary_auc(shuffled, z["inner_shuffled_sex"])
    planted_auc = weighted_binary_auc(planted, z["inner_true_sex"])
    independent = max(shuffled_auc, 1 - shuffled_auc)
    positive = max(planted_auc, 1 - planted_auc)
    if independent > 0.60 or positive < 0.99:
        raise ValueError("engineering null/planted control gate failed")
    return {"status": "CONTROL_FORWARD_AND_GATE_PASS", "null_auc": null,
            "shuffled_independent_inner_auc": independent, "planted_auc": positive,
            "maximum_prediction_difference": deviations,
            "receipt_sha256": _sha256(receipt_path)}

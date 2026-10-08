"""Independent replay of the frozen V2 negative-control custody package.

Only NumPy, joblib and saved public view values are used. The campaign's
attacker, control and endpoint modules are deliberately not imported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


ATT_SALT = "pcrl-method-portfolio-v1|shuffled-sex-probe|20261008"
INNER_SALT = "pcrl-method-portfolio-v1|independent-inner-sex-null|20261008|v2"
_SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"
_ARRAY_KEYS = frozenset((
    "att_fit_row_id", "att_fit_unit", "att_fit_x", "att_fit_true_sex",
    "att_fit_shuffled_sex", "inner_row_id", "inner_unit", "inner_x",
    "inner_true_sex", "inner_shuffled_sex", "shuffled_prediction_inner",
    "planted_prediction_inner"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    prefix = (str(array.dtype) + str(array.shape)).encode()
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _equal(actual, expected, reason: str) -> None:
    _require(np.array_equal(np.asarray(actual), np.asarray(expected)), reason)


def group_shuffle(y, groups, salt: str) -> np.ndarray:
    """Independently reconstruct one frozen size-preserving group shuffle."""
    _require(salt in (ATT_SALT, INNER_SALT), "unregistered V2 shuffle salt")
    labels = np.asarray(y, dtype=np.int64)
    unit = np.asarray(groups, dtype=np.int64)
    _require(labels.ndim == 1 and unit.shape == labels.shape,
             "group shuffle identity shape differs")
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    buckets: dict[int, list[np.ndarray]] = {}
    for group in np.unique(unit):
        rows = np.flatnonzero(unit == group)
        buckets.setdefault(len(rows), []).append(rows)
    shuffled = np.empty_like(labels)
    for size in sorted(buckets):
        row_groups = buckets[size]
        for destination, source in zip(row_groups, rng.permutation(len(row_groups))):
            shuffled[destination] = labels[row_groups[int(source)]]
    return shuffled


def _oriented_auc(y, score) -> float:
    """AUC with average-rank ties, independent of the campaign scorer."""
    labels = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    _require(labels.ndim == 1 and values.shape == labels.shape and
             np.all((labels == 0) | (labels == 1)) and np.all(np.isfinite(values)),
             "V2 AUC label or score is invalid")
    positive = int(np.count_nonzero(labels))
    negative = len(labels) - positive
    _require(positive > 0 and negative > 0, "V2 AUC needs both SEX classes")
    order = np.argsort(values, kind="mergesort")
    sorted_score = values[order]
    sorted_label = labels[order]
    favourable = 0.0
    negative_before = 0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        pos_block = int(np.count_nonzero(sorted_label[start:stop]))
        neg_block = stop - start - pos_block
        favourable += pos_block * (negative_before + 0.5 * neg_block)
        negative_before += neg_block
        start = stop
    raw = favourable / (positive * negative)
    return max(raw, 1.0 - raw)


def _fit_mask(unit: np.ndarray) -> np.ndarray:
    _require(unit.ndim == 1 and np.unique(unit).size == 4061,
             "ATT group geometry differs")
    ranked = sorted((hashlib.sha256((_SPLIT_SALT + str(int(group))).encode()).hexdigest(),
                     int(group)) for group in np.unique(unit))
    tune = np.asarray([group for _, group in ranked[:1015]], dtype=np.int64)
    mask = ~np.isin(unit, tune)
    _require(np.unique(unit[mask]).size == 3046 and
             np.unique(unit[~mask]).size == 1015,
             "ATT fit/tune split differs")
    return mask


def _binary(value, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    _require(result.shape == (rows,) and np.all((result == 0) | (result == 1)),
             f"{name} SEX shape or class differs")
    return result


def _model_probability(model, x: np.ndarray, rows: int, name: str) -> np.ndarray:
    full = np.asarray(model.predict_proba(x), dtype=np.float64)
    _require(full.shape == (rows, 2) and np.all(np.isfinite(full)) and
             np.all((full >= 0) & (full <= 1)),
             f"{name} saved model probability shape differs")
    return full[:, 1]


def replay_probe(custody_dir: str | Path, report: dict, att_view: dict,
                 inner_view: dict, y_att, y_inner) -> dict:
    """Verify one local or coalition V2 probe against its exact saved views."""
    root = Path(custody_dir)
    receipt_path = root / "CUSTODY_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    _require(receipt.get("schema") == "pcrl-portfolio-private-control-v2-custody-v1" and
             receipt.get("status") == "FROZEN", "V2 custody schema/status differs")
    if "custody_receipt_sha256" in report:
        _require(report["custody_receipt_sha256"] == _sha_file(receipt_path),
                 "V2 custody receipt hash differs")
    files = receipt.get("files")
    _require(isinstance(files, dict) and set(files) ==
             {"shuffled_model.joblib", "planted_model.joblib", "probe_replay.npz"},
             "V2 custody member set differs")
    if "custody_file_sha256" in report:
        _require(report["custody_file_sha256"] == files,
                 "V2 custody report file hash map differs")
    for filename, expected in files.items():
        _require(_sha_file(root / filename) == expected,
                 f"V2 custody file hash differs: {filename}")
    _require(receipt.get("fit_role") == "ATTACK_FIT_NEW/ATTACK_TRAIN" and
             receipt.get("score_role") == "INNER_SELECTION" and
             receipt.get("att_shuffle_salt") == ATT_SALT and
             receipt.get("inner_shuffle_salt") == INNER_SALT,
             "V2 custody role or shuffle differs")
    with np.load(root / "probe_replay.npz", allow_pickle=False) as package:
        _require(set(package.files) == _ARRAY_KEYS, "V2 custody array members differ")
        arrays = {name: np.asarray(package[name]) for name in _ARRAY_KEYS}
    expected_arrays = receipt.get("arrays_sha256")
    _require(isinstance(expected_arrays, dict) and set(expected_arrays) == _ARRAY_KEYS,
             "V2 custody array hash map differs")
    for name, array in arrays.items():
        _require(_sha_array(array) == expected_arrays[name],
                 f"V2 custody array hash differs: {name}")

    att_id = np.asarray(att_view["row_id"], dtype=np.int64)
    att_unit = np.asarray(att_view["unit"], dtype=np.int64)
    att_x = np.asarray(att_view["x"], dtype=np.float64)
    inner_id = np.asarray(inner_view["row_id"], dtype=np.int64)
    inner_unit = np.asarray(inner_view["unit"], dtype=np.int64)
    inner_x = np.asarray(inner_view["x"], dtype=np.float64)
    _require(att_id.shape == att_unit.shape == (4064,) and
             inner_id.shape == inner_unit.shape == (2235,) and
             np.unique(inner_unit).size == 2234 and
             att_x.ndim == inner_x.ndim == 2 and
             att_x.shape[0] == 4064 and inner_x.shape == (2235, att_x.shape[1]) and
             np.all(np.isfinite(att_x)) and np.all(np.isfinite(inner_x)),
             "V2 public view geometry differs")
    att_y = _binary(y_att, 4064, "ATT")
    inner_y = _binary(y_inner, 2235, "INNER")
    fit = _fit_mask(att_unit)
    for name, value in (
        ("att_fit_row_id", att_id[fit]), ("att_fit_unit", att_unit[fit]),
        ("att_fit_x", att_x[fit]), ("att_fit_true_sex", att_y[fit]),
        ("inner_row_id", inner_id), ("inner_unit", inner_unit),
        ("inner_x", inner_x), ("inner_true_sex", inner_y),
    ):
        _equal(arrays[name], value, f"V2 saved {name} differs from admitted view")
    _equal(arrays["att_fit_shuffled_sex"],
           group_shuffle(att_y[fit], att_unit[fit], ATT_SALT),
           "V2 ATT shuffled labels differ")
    independent_inner = group_shuffle(inner_y, inner_unit, INNER_SALT)
    _equal(arrays["inner_shuffled_sex"], independent_inner,
           "V2 independent INNER shuffled labels differ")
    for field, value in (("att_view_content_sha256", att_x),
                         ("inner_view_content_sha256", inner_x)):
        if field in report:
            _require(report[field] == receipt.get(field),
                     f"V2 {field} report/receipt differs")
        _require(receipt.get(field, {}).get("x") == _sha_array(value),
                 f"V2 {field} numeric matrix differs")
    for role, ids, units in (("att", att_id, att_unit),
                             ("inner", inner_id, inner_unit)):
        field = f"{role}_row_unit_pair_sha256"
        if field in report:
            pair = np.column_stack((ids, units)).astype(np.int64)
            _require(report[field] == _sha_array(pair),
                     f"V2 {field} differs")

    shuffled_model = joblib.load(root / "shuffled_model.joblib")
    planted_model = joblib.load(root / "planted_model.joblib")
    shuffled = _model_probability(shuffled_model, inner_x, 2235, "shuffled")
    planted = _model_probability(planted_model,
                                 np.column_stack((inner_x, inner_y)),
                                 2235, "planted")
    _equal(shuffled, arrays["shuffled_prediction_inner"],
           "V2 shuffled saved model prediction differs")
    _equal(planted, arrays["planted_prediction_inner"],
           "V2 planted saved model prediction differs")
    values = {"null_auc": _oriented_auc(inner_y, np.full(2235, 0.5)),
              "shuffled_att_true_inner_auc": _oriented_auc(inner_y, shuffled),
              "shuffled_att_independent_inner_auc": _oriented_auc(independent_inner, shuffled),
              "planted_auc": _oriented_auc(inner_y, planted)}
    gates = {"null_exact": values["null_auc"] == 0.5,
             "shuffled_independent_inner_at_most_0.60":
                 values["shuffled_att_independent_inner_auc"] <= 0.60,
             "planted_at_least_0.99": values["planted_auc"] >= 0.99}
    expected_status = "PASS" if all(gates.values()) else "FAIL"
    probe = report.get("probe", report)
    _require(probe.get("schema") == "pcrl-portfolio-attack-control-probes-v2" and
             probe.get("fit_role") == "ATTACK_FIT_NEW/ATTACK_TRAIN" and
             probe.get("score_role") == "INNER_SELECTION" and
             probe.get("additional_reader_fits") == 2 and
             probe.get("att_shuffle_salt") == ATT_SALT and
             probe.get("inner_shuffle_salt") == INNER_SALT and
             probe.get("att_shuffle_count") == 1 and
             probe.get("inner_shuffle_count") == 1 and
             probe.get("gates") == gates and
             probe.get("status") == expected_status and
             report.get("status") == expected_status,
             "V2 control gate or identity differs")
    for name, value in values.items():
        reported = probe.get(name)
        _require(isinstance(reported, (int, float)) and
                 abs(float(reported) - value) <= 1e-12,
                 f"V2 reported {name} differs")
    if "att_fit_rows" in report:
        _require(report["att_fit_rows"] == int(np.count_nonzero(fit)),
                 "V2 ATT fit row count differs")
    if "inner_rows" in report:
        _require(report["inner_rows"] == 2235 and
                 report.get("inner_groups") == 2234,
                 "V2 INNER row/group count differs")
    return {"schema": "pcrl-s4-independent-v2-control-replay-v1",
            "status": expected_status, "model_prediction_parity": "PASS",
            "att_shuffle_parity": "PASS", "inner_shuffle_parity": "PASS",
            "att_fit_rows": int(np.count_nonzero(fit)),
            "inner_rows": 2235, "inner_groups": 2234,
            "auc": values, "gates": gates,
            "custody_receipt_sha256": _sha_file(receipt_path)}

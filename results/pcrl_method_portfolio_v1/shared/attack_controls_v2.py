"""Separate V2 engineering probes for complete-view SEX recovery.

The V1 shuffled-fit score against true INNER SEX is retained as a descriptive
number. It is not a valid mandatory null when the disclosed view itself
contains SEX: a random nonzero linear slope ranks that true signal in one of
the two prespecified orientations. V2 scores the same fitted shuffled-label
reader against an independently group-shuffled INNER label vector.

An Adult caller must hold S3's heavy-process slot and a dated effective V2
science seal. This module contains no Adult loader or result migration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from attack_eval import AttackView, _factory, _label_record, attack_fit_tune_split, oriented_auc
from safe_data_guard import sha_arr, sha_file


SCHEMA = "pcrl-portfolio-attack-control-probes-v2"
SHUFFLE_ATT_SALT = "pcrl-method-portfolio-v1|shuffled-sex-probe|20261008"
SHUFFLE_INNER_SALT = "pcrl-method-portfolio-v1|independent-inner-sex-null|20261008|v2"
ATT_SHUFFLE_COUNT = 1
INNER_SHUFFLE_COUNT = 1
NULL_EXPECTED = 0.5
SHUFFLE_CAP = 0.60
PLANTED_FLOOR = 0.99


def groupwise_shuffled_labels(y, unit, *, salt: str) -> np.ndarray:
    """Move whole label vectors within equal-sized group buckets, once."""
    if salt not in (SHUFFLE_ATT_SALT, SHUFFLE_INNER_SALT):
        raise ValueError("unregistered V2 group-shuffle salt")
    labels = np.asarray(y, dtype=np.int64)
    groups = np.asarray(unit, dtype=np.int64)
    if labels.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("group shuffle row shape mismatch")
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    by_size: dict[int, list[np.ndarray]] = {}
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        by_size.setdefault(len(idx), []).append(idx)
    result = np.empty_like(labels)
    for size in sorted(by_size):
        buckets = by_size[size]
        for target, source_index in zip(buckets, rng.permutation(len(buckets))):
            result[target] = labels[buckets[int(source_index)]]
    return result


def run_control_probes_v2(att_view: AttackView, sex_att: dict,
                          inner_view: AttackView, sex_inner: dict, *,
                          return_artifacts: bool = False) -> dict | tuple[dict, dict]:
    """Fit two fixed LR probes on ATT, then score one-shot INNER views."""
    att_view.validate()
    inner_view.validate()
    if (att_view.components != inner_view.components or
            att_view.wire_components != inner_view.wire_components or
            att_view.component_columns != inner_view.component_columns or
            att_view.wire_constants != inner_view.wire_constants):
        raise ValueError("engineering control view schemas differ")
    y_att = _label_record(sex_att, att_view)
    y_inner = _label_record(sex_inner, inner_view)
    if len(inner_view.row_id) != 2235 or np.unique(inner_view.unit).size != 2234:
        raise ValueError("INNER_SELECTION row/group count mismatch")
    split = attack_fit_tune_split(att_view.unit)
    fit = split["fit"]
    if np.unique(y_att[fit]).size != 2 or np.unique(y_inner).size != 2:
        raise ValueError("engineering control needs both SEX classes")
    x_att = np.asarray(att_view.x, dtype=np.float64)
    x_inner = np.asarray(inner_view.x, dtype=np.float64)

    null_auc = oriented_auc(y_inner, np.full(len(y_inner), 0.5))["fixed_alias_max_auc"]
    shuffled_att = groupwise_shuffled_labels(y_att[fit], att_view.unit[fit],
                                             salt=SHUFFLE_ATT_SALT)
    shuffled_inner = groupwise_shuffled_labels(y_inner, inner_view.unit,
                                               salt=SHUFFLE_INNER_SALT)
    if np.unique(shuffled_att).size != 2 or np.unique(shuffled_inner).size != 2:
        raise ValueError("group shuffle lost a SEX class")
    shuffled_model = _factory("LR_C1", 0)
    shuffled_model.fit(x_att[fit], shuffled_att)
    shuffled_prediction = shuffled_model.predict_proba(x_inner)[:, 1]
    descriptive_auc = oriented_auc(y_inner, shuffled_prediction)["fixed_alias_max_auc"]
    independent_auc = oriented_auc(shuffled_inner, shuffled_prediction)["fixed_alias_max_auc"]

    planted_model = _factory("LR_C1", 0)
    planted_model.fit(np.column_stack((x_att[fit], y_att[fit])), y_att[fit])
    planted_prediction = planted_model.predict_proba(np.column_stack((x_inner, y_inner)))[:, 1]
    planted_auc = oriented_auc(y_inner, planted_prediction)["fixed_alias_max_auc"]
    gates = {"null_exact": null_auc == NULL_EXPECTED,
             "shuffled_independent_inner_at_most_0.60": independent_auc <= SHUFFLE_CAP,
             "planted_at_least_0.99": planted_auc >= PLANTED_FLOOR}
    result = {"schema": SCHEMA, "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "score_role": "INNER_SELECTION", "additional_reader_fits": 2,
            "att_shuffle_salt": SHUFFLE_ATT_SALT,
            "inner_shuffle_salt": SHUFFLE_INNER_SALT,
            "att_shuffle_count": ATT_SHUFFLE_COUNT,
            "inner_shuffle_count": INNER_SHUFFLE_COUNT,
            "null_auc": null_auc,
            "shuffled_att_true_inner_auc": descriptive_auc,
            "shuffled_att_independent_inner_auc": independent_auc,
            "planted_auc": planted_auc,
            "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL"}
    if not return_artifacts:
        return result
    artifacts = {
        "shuffled_model": shuffled_model,
        "planted_model": planted_model,
        "att_fit_row_id": np.asarray(att_view.row_id[fit], dtype=np.int64),
        "att_fit_unit": np.asarray(att_view.unit[fit], dtype=np.int64),
        "att_fit_x": x_att[fit],
        "att_fit_true_sex": y_att[fit],
        "att_fit_shuffled_sex": shuffled_att,
        "inner_row_id": np.asarray(inner_view.row_id, dtype=np.int64),
        "inner_unit": np.asarray(inner_view.unit, dtype=np.int64),
        "inner_x": x_inner,
        "inner_true_sex": y_inner,
        "inner_shuffled_sex": shuffled_inner,
        "shuffled_prediction_inner": shuffled_prediction,
        "planted_prediction_inner": planted_prediction,
        "att_view_content_sha256": att_view.content_hashes(),
        "inner_view_content_sha256": inner_view.content_hashes(),
    }
    return result, artifacts


def save_control_probe_v2_custody(artifacts: dict, private_dir: str | Path) -> dict:
    """Save a replayable V2 probe in a new private directory, write once."""
    root = Path(private_dir)
    if ".private" not in root.resolve().parts or root.exists():
        raise ValueError("V2 probe custody requires a new private directory")
    root.mkdir(parents=True, mode=0o700)
    files = {}
    for key in ("shuffled_model", "planted_model"):
        path = root / f"{key}.joblib"
        joblib.dump(artifacts[key], path, compress=3)
        path.chmod(0o600)
        files[path.name] = sha_file(path)
    array_keys = ("att_fit_row_id", "att_fit_unit", "att_fit_x",
                  "att_fit_true_sex", "att_fit_shuffled_sex",
                  "inner_row_id", "inner_unit", "inner_x", "inner_true_sex", "inner_shuffled_sex",
                  "shuffled_prediction_inner", "planted_prediction_inner")
    arrays = {key: np.asarray(artifacts[key]) for key in array_keys}
    path = root / "probe_replay.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    files[path.name] = sha_file(path)
    receipt = {"schema": "pcrl-portfolio-private-control-v2-custody-v1",
               "status": "FROZEN", "files": files,
               "arrays_sha256": {key: sha_arr(value) for key, value in arrays.items()},
               "att_view_content_sha256": artifacts["att_view_content_sha256"],
               "inner_view_content_sha256": artifacts["inner_view_content_sha256"],
               "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN", "score_role": "INNER_SELECTION",
               "att_shuffle_salt": SHUFFLE_ATT_SALT,
               "inner_shuffle_salt": SHUFFLE_INNER_SALT}
    path = root / "CUSTODY_RECEIPT.json"
    with path.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return {"private_receipt_sha256": sha_file(path), "files": files}

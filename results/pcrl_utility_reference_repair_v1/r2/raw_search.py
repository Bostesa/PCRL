"""Source-matched two-reader FIT SEX proxy for the fixed mask/union bank.

This function receives permitted in-memory arrays only after the separate
task-only T unit is frozen. It uses the original finite-bank search readers,
not the later fresh complete-view ATTACK_FIT_NEW attackers.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


READER_IDS = ("LR_C1", "HGB_15x64")


def _one_feature_set(x83: np.ndarray, sex: np.ndarray,
                     fit_rows: np.ndarray, validation_rows: np.ndarray,
                     columns: tuple[int, ...], seed: int) -> dict:
    fit_x = np.ascontiguousarray(x83[np.ix_(fit_rows, columns)], dtype=np.float64)
    val_x = np.ascontiguousarray(x83[np.ix_(validation_rows, columns)], dtype=np.float64)
    readers = (
        make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=300, solver="lbfgs", random_state=seed)),
        HistGradientBoostingClassifier(
            max_iter=64, max_leaf_nodes=15, learning_rate=0.05,
            max_depth=None, random_state=seed),
    )
    scores = {}
    for reader_id, reader in zip(READER_IDS, readers):
        reader.fit(fit_x, sex[fit_rows])
        probability = reader.predict_proba(val_x)[:, 1]
        raw_auc = float(roc_auc_score(sex[validation_rows], probability))
        scores[reader_id] = {"raw_auc": raw_auc,
                             "oriented_auc": max(raw_auc, 1.0 - raw_auc)}
    return {"selected_auc": max(item["oriented_auc"] for item in scores.values()),
            "readers": scores, "fit_count": 2}


def fit_fixed_raw_search(bank, x83: np.ndarray, sex: np.ndarray,
                         fit_rows: np.ndarray, validation_rows: np.ndarray,
                         seed: int) -> tuple[dict, dict, dict]:
    """Run exactly 5 local and 6 unique union calls, two fits each."""
    x = np.asarray(x83)
    y = np.asarray(sex)
    fit = np.asarray(fit_rows)
    val = np.asarray(validation_rows)
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 83
            or not np.isfinite(x).all()
            or y.dtype != np.int64 or y.shape != (len(x),)
            or fit.dtype != np.int64 or val.dtype != np.int64
            or fit.ndim != 1 or val.ndim != 1
            or len(fit) + len(val) != len(x)
            or not np.array_equal(np.sort(np.r_[fit, val]), np.arange(len(x)))
            or not np.array_equal(np.unique(y[fit]), [0, 1])
            or not np.array_equal(np.unique(y[val]), [0, 1])):
        raise ValueError("fixed FIT SEX search split or arrays differ")
    local = {}
    details = {"local": {}, "union": {}}
    for mask in bank.masks:
        result = _one_feature_set(x, y, fit, val,
                                  mask.column_indices, seed)
        local[mask.mask_id] = result["selected_auc"]
        details["local"][mask.mask_id] = result
    unions = {}
    for pair in bank.pairs:
        if pair.union_sha256 in unions:
            continue
        result = _one_feature_set(x, y, fit, val,
                                  pair.union_columns, seed)
        unions[pair.union_sha256] = result["selected_auc"]
        details["union"][pair.union_sha256] = result
    if len(local) != 5 or len(unions) != 6:
        raise ValueError("source bank local/union geometry changed")
    details.update({
        "schema": "pcrl-utility-fixed-raw-search-v1",
        "seed": seed, "fit_rows": len(fit),
        "validation_rows": len(val),
        "local_feature_sets": len(local),
        "unique_union_feature_sets": len(unions),
        "fitted_reader_count": 2 * (len(local) + len(unions)),
        "reader_ids": list(READER_IDS),
        "orientation": "max(AUC,1-AUC) per reader, then max readers",
        "fit_sex_counts": np.bincount(y[fit], minlength=2).tolist(),
        "validation_sex_counts": np.bincount(y[val], minlength=2).tolist(),
    })
    return local, unions, details

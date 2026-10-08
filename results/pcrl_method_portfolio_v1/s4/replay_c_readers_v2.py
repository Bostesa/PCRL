"""Independent C reader math and exact saved-layout RAW_LR_C1 inference.

Adult use requires a separately sealed S4 unit in the global S3 queue.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np

from replay_c_values_v1 import sha_array


READERS = (
    "LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
    "RAW_LR_C1", "RAW_HGB_15x200",
)
SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"


def fixed_split(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Recompute the 3,046/1,015 group split without viewing SEX."""
    unit = np.asarray(groups, dtype=np.int64)
    unique = np.unique(unit)
    if unit.ndim != 1 or len(unique) != 4061:
        raise ValueError("C ATT requires 4,061 distinct groups")
    ranking = sorted((hashlib.sha256((SPLIT_SALT + str(int(u))).encode()).hexdigest(),
                      int(u)) for u in unique)
    tune_groups = np.asarray([group for _, group in ranking[:1015]],
                             dtype=np.int64)
    tune = np.flatnonzero(np.isin(unit, tune_groups))
    fit = np.flatnonzero(~np.isin(unit, tune_groups))
    if (len(np.unique(unit[fit])) != 3046 or
            len(np.unique(unit[tune])) != 1015):
        raise ValueError("C ATT fit/tune group counts differ")
    return fit, tune, {
        "fit_groups": 3046,
        "tune_groups": 1015,
        "tune_group_id_sha256": sha_array(np.sort(tune_groups)),
    }


def oriented_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Mann–Whitney AUC, with half credit for ties and a fixed reverse alias."""
    y = np.asarray(labels, dtype=np.int64)
    value = np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or value.shape != y.shape or not np.isfinite(value).all()
            or not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2):
        raise ValueError("C AUC needs aligned finite scores and two binary classes")
    order = np.argsort(value, kind="stable")
    ordered_scores, ordered_y = value[order], y[order]
    starts = np.r_[0, np.flatnonzero(
        ordered_scores[1:] != ordered_scores[:-1]) + 1]
    positive = np.add.reduceat(ordered_y, starts)
    negative = np.add.reduceat(1 - ordered_y, starts)
    wins = float(np.dot(positive,
                        np.cumsum(negative) - 0.5 * negative))
    auc = wins / (int(positive.sum()) * int(negative.sum()))
    return {"positive_alias_auc": auc,
            "reverse_alias_auc": 1.0 - auc,
            "fixed_alias_max_auc": max(auc, 1.0 - auc)}


def reader_matrix(view: Mapping[str, np.ndarray], reader: str) -> np.ndarray:
    """Choose only the declared C complete, defense aware or raw matrix."""
    if reader not in READERS:
        raise ValueError("C reader is outside the frozen six-reader slate")
    key = ("x_defense_aware" if reader == "DA_MLP_64x32" else
           "x_raw_only" if reader.startswith("RAW_") else "x")
    value = np.asarray(view[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError("C reader matrix is incomplete or nonfinite")
    return value


def replay_model_predictions(
    model_path: str | Path,
    model_sha256: str,
    reader: str,
    att_view: Mapping[str, np.ndarray],
    inner_view: Mapping[str, np.ndarray],
    tune_positions: np.ndarray,
    saved_tune: np.ndarray,
    saved_inner: np.ndarray,
) -> None:
    """Replay one SHA-bound S2 reader model on independently rebuilt matrices."""
    path = Path(model_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != model_sha256:
        raise ValueError("C reader model SHA differs")
    att = reader_matrix(att_view, reader)
    inner = reader_matrix(inner_view, reader)
    if reader == "RAW_LR_C1":
        # The S2 saved INNER matrix is C contiguous. Rebuilding the equal-value
        # raw matrix can yield Fortran strides and change BLAS reduction order.
        # The frozen diagnostic found 899 bit differences for those strides;
        # this copy restored all saved probabilities exactly. Keep the exact
        # prediction check below for every reader.
        inner = np.ascontiguousarray(inner)
    tune = np.asarray(tune_positions, dtype=np.int64)
    if (tune.ndim != 1 or np.any((tune < 0) | (tune >= len(att))) or
            len(np.unique(tune)) != len(tune)):
        raise ValueError("C reader tune positions differ")
    model = joblib.load(path)
    actual_tune = np.asarray(model.predict_proba(att[tune])[:, 1],
                             dtype=np.float64)
    actual_inner = np.asarray(model.predict_proba(inner)[:, 1],
                              dtype=np.float64)
    for name, actual, expected in (("tune", actual_tune, saved_tune),
                                   ("inner", actual_inner, saved_inner)):
        frozen = np.asarray(expected, dtype=np.float64)
        if (actual.shape != frozen.shape or not np.isfinite(actual).all()
                or not np.array_equal(actual, frozen)):
            raise ValueError(f"C reader {name} prediction differs from saved vector")

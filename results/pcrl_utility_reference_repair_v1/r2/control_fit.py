"""ATT-only null, shuffled-label and planted-leak control model fits.

The planted inference column is unavailable until a separate INNER SEX score
unit. This file never accepts that role's protected label.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from results.pcrl_method_portfolio_v1.shared.attack_eval import (
    _factory, _label_record, attack_fit_tune_split,
)
from results.pcrl_method_portfolio_v1.shared.attack_controls_v2 import (
    SHUFFLE_ATT_SALT, SHUFFLE_INNER_SALT, groupwise_shuffled_labels,
)
from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    array_sha, sha_file,
)


SCHEMA = "pcrl-utility-att-control-fit-v1"
FILES = ("shuffled_model.joblib", "planted_model.joblib",
         "shuffled_inner.npy", "CONTROL_FIT_RECEIPT.json")


def fit_control_models(att_view: object, sex_att: Mapping[str, object],
                       inner_view: object, *, seed: int,
                       output_dir: str | Path) -> dict:
    """Fit two source-matched LR probes, with no INNER SEX available."""
    att_view.validate()
    inner_view.validate()
    if (att_view.components != inner_view.components
            or att_view.wire_components != inner_view.wire_components
            or att_view.component_columns != inner_view.component_columns
            or att_view.wire_constants != inner_view.wire_constants):
        raise ValueError("ATT/INNER control complete-view schema differs")
    split = attack_fit_tune_split(att_view.unit)
    fit = split["fit"]
    y = _label_record(dict(sex_att), att_view)
    if np.unique(y[fit]).size != 2:
        raise ValueError("ATT control fit lacks both SEX classes")
    x_att = np.asarray(att_view.x, dtype=np.float64)
    x_inner = np.asarray(inner_view.x, dtype=np.float64)
    shuffled = groupwise_shuffled_labels(y[fit], att_view.unit[fit],
                                         salt=SHUFFLE_ATT_SALT)
    if np.unique(shuffled).size != 2:
        raise ValueError("shuffled ATT control lost a SEX class")
    shuffled_model = _factory("LR_C1", 0)
    shuffled_model.fit(x_att[fit], shuffled)
    planted_model = _factory("LR_C1", 0)
    planted_model.fit(np.column_stack((x_att[fit], y[fit])), y[fit])
    p_inner = np.asarray(shuffled_model.predict_proba(x_inner)[:, 1],
                         dtype=np.float64)
    if (p_inner.shape != (len(inner_view.row_id),)
            or not np.isfinite(p_inner).all()
            or np.any((p_inner < 0) | (p_inner > 1))):
        raise ValueError("shuffled control prediction invalid")
    out = Path(output_dir).resolve()
    if ".private" not in out.parts or out.exists():
        raise PermissionError("ATT control output must be new private custody")
    out.mkdir(parents=True, mode=0o700)
    for name, model in (("shuffled_model", shuffled_model),
                        ("planted_model", planted_model)):
        path = out / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        path.chmod(0o600)
    with (out / "shuffled_inner.npy").open("xb") as stream:
        np.save(stream, p_inner, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema": SCHEMA, "seed": seed,
        "att_view_content_sha256": att_view.content_hashes(),
        "inner_view_content_sha256": inner_view.content_hashes(),
        "att_fit_row_unit_sha256": array_sha(np.column_stack((
            att_view.row_id[fit], att_view.unit[fit]))),
        "inner_row_unit_sha256": array_sha(np.column_stack((
            inner_view.row_id, inner_view.unit))),
        "ATT_shuffle_salt": SHUFFLE_ATT_SALT,
        "INNER_shuffle_salt_for_later_score": SHUFFLE_INNER_SALT,
        "fit_rows": int(len(fit)), "fresh_fit_count": 2,
        "protected_labels_read": ["ATTACK_FIT_NEW:sex"],
        "INNER_SEX_read": False,
        "files_sha256": {name: sha_file(out / name) for name in FILES[:-1]},
        "shuffled_prediction_array_sha256": array_sha(p_inner),
    }
    with (out / "CONTROL_FIT_RECEIPT.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt

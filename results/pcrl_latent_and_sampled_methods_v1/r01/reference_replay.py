"""Replay frozen U0/Ucal on one admitted role after the campaign seal.

This is numerical work for the shared heavy queue. It opens only the chosen
role's feature/reference exports, U model/head files, and the calibration
record. Historical all-role `release.npz` and `calU/u.npz` are never opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from hcal.calib import temp_apply_probs

from .frozen_u import FrozenUEncoder
from .role_store import AccessRefused, RoleOnlyStore, assert_same_people, sha_array, sha_file


TOLERANCE = 2e-7  # inherited predeclared float32 batch-geometry tolerance


def replay_reference_role(store: RoleOnlyStore, frozen_u: FrozenUEncoder,
                          hcal_root: str | Path, role: str) -> tuple[dict, dict]:
    """Return private reconstructed vectors and aggregate-only parity receipt.

    Caller must hold the R00 effective fit seal and a shared numerical slot.
    `hcal_root` is the private `hcal_v1` directory, not an all-role archive.
    """
    seed = frozen_u.seed
    if seed not in (0, 1):
        raise AccessRefused("reference replay seed is not admitted")
    features = store.read(role, "features")
    expected = store.read(role, f"references_s{seed}")
    assert_same_people(features, expected)
    names = tuple(str(item) for item in features["feature_names"])
    if names != frozen_u.columns:
        raise AccessRefused("frozen encoder and export schema order differ")
    root = Path(hcal_root).resolve(strict=True)
    source = root / "admitted" / f"rel__s{seed}__U"
    complete = json.loads((source / "COMPLETE.json").read_text())
    if complete.get("id") != f"rel__s{seed}__U":
        raise AccessRefused("U source unit identity differs")
    for filename in ("model.pt", "head_0.joblib", "head_1.joblib", "record.json"):
        if sha_file(source / filename) != complete["files"].get(filename):
            raise AccessRefused("U source component hash differs")
    if complete["files"]["model.pt"] != frozen_u.model_sha256:
        raise AccessRefused("frozen encoder model differs from deployed U source")
    record = json.loads((source / "record.json").read_text())
    if record.get("seed") != seed or record.get("admitted_from") != f"smf tl__s{seed}__e40":
        raise AccessRefused("U source fit provenance differs")
    heads = tuple(joblib.load(source / f"head_{purpose}.joblib") for purpose in (0, 1))
    for purpose, classes in ((0, 2), (1, 6)):
        if not np.array_equal(heads[purpose].classes_, np.arange(classes)):
            raise AccessRefused("deployed head class order differs")
    cal_dir = root / "run" / "units" / f"calU__s{seed}"
    cal_complete = json.loads((cal_dir / "COMPLETE.json").read_text())
    cal_path = cal_dir / "record.json"
    if (cal_complete.get("id") != f"calU__s{seed}"
            or sha_file(cal_path) != cal_complete.get("files", {}).get("record.json")):
        raise AccessRefused("U calibration record hash differs")
    cal = json.loads(cal_path.read_text())
    if cal.get("schema") != "hcal-calU-v1" or cal.get("seed") != seed:
        raise AccessRefused("U calibration identity differs")
    latents = frozen_u.encode_pair(features["X"], names, role)
    predictions = {}
    comparison = {}
    for purpose, classes in ((0, 2), (1, 6)):
        recipient = purpose + 1
        u0 = np.asarray(heads[purpose].predict_proba(latents[purpose].astype(np.float64)), dtype=np.float64)
        decision = expected[f"d_{recipient}"]
        alpha = cal["tables"][f"H-GLOBAL-TEMP|r{recipient}"]["alpha"]
        ucal = temp_apply_probs(u0, alpha, decision)
        if u0.shape != (len(features["X"]), classes) or ucal.shape != u0.shape:
            raise AccessRefused("source reference probability shape differs")
        for family, actual in (("U0", u0), ("Ucal", ucal)):
            key = f"{family}_{recipient}"
            archived = expected[key]
            delta = np.abs(actual - archived)
            errors = int(np.count_nonzero(delta > TOLERANCE))
            decision_equal = bool(np.array_equal(actual.argmax(axis=1), decision))
            if errors or not decision_equal:
                raise AccessRefused("restored source reference parity failed")
            predictions[key] = actual
            comparison[key] = {"bitwise_equal": bool(np.array_equal(actual, archived)),
                               "max_abs_difference": float(delta.max(initial=0)),
                               "values_above_tolerance": errors,
                               "decision_equal": decision_equal,
                               "restored_sha256": sha_array(actual),
                               "archived_sha256": sha_array(archived)}
    receipt = {"schema": "pcrl-new-campaign-r01-reference-replay-v1",
               "status": "PASS", "role": role, "seed": seed,
               "rows": len(features["row_id"]),
               "groups": len(np.unique(features["unit"])),
               "source_model_sha256": frozen_u.model_sha256,
               "source_head_sha256": [complete["files"][f"head_{i}.joblib"] for i in (0, 1)],
               "calibration_record_sha256": cal_complete["files"]["record.json"],
               "tolerance": TOLERANCE, "parity": comparison,
               "exposure": "U0 fitted on all FIT; Ucal parameter fitted on CAL; Ucal family historically selected using INNER"}
    return predictions, receipt

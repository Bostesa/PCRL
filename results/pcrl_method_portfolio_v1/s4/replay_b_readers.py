"""Independent S4 replay of Team B's frozen ATT/INNER reader slate.

Requires reconstructed actual release matrices from ``replay_b_release``.
No Team B, S2 attack, endpoint, or selection implementation is imported.
Adult execution needs an S3 queue unit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from replay_b_release import ordered_split, sha_array, sha_file


MAIN_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")


def auc(sex: np.ndarray, score: np.ndarray) -> float:
    """Mann–Whitney AUC with half credit for tied predictions."""
    y = np.asarray(sex, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError("AUC labels/predictions differ")
    order = np.argsort(s, kind="stable")
    ss, yy = s[order], y[order]
    start = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    pos = np.add.reduceat(yy, start)
    neg = np.add.reduceat(1 - yy, start)
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        raise ValueError("AUC requires two SEX classes")
    return float(np.dot(pos, np.cumsum(neg) - 0.5 * neg) / (npos * nneg))


def content_hashes(view: dict) -> dict[str, str | None]:
    """Recreate private receipt hashes from independently built matrices."""
    schema = {key: view[key] for key in ("components", "wire_components",
                                         "component_columns", "wire_constants")}
    return {"x": sha_array(np.asarray(view["x"], dtype=np.float64)),
            "x_defense_aware": sha_array(np.asarray(view["x_defense_aware"], dtype=np.float64)),
            "cell_key": None,
            "x_metadata": None if view["x_metadata"] is None else
            sha_array(np.asarray(view["x_metadata"], dtype=np.float64)),
            "x_code_bits": None, "x_raw_only": None,
            "wire_schema": hashlib.sha256(json.dumps(schema, sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest()}


def _frozen_prediction(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    if sha_file(path) != expected_sha:
        raise ValueError("frozen prediction archive SHA differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name], dtype=np.float64).copy()
                for name in archive.files}


def replay_view(attempt: Path, view: str, att: dict, inner: dict,
                att_ids: np.ndarray, att_groups: np.ndarray,
                inner_ids: np.ndarray, inner_groups: np.ndarray,
                sex_att: np.ndarray,
                pair_aliases: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    """Check one local/pair fitted reader package and return frozen predictions."""
    directory = attempt / f"readers__{view}"
    receipt_path = directory / "READERS_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "FROZEN" or receipt.get("schema") != \
            "pcrl-portfolio-private-reader-custody-v1":
        raise ValueError("reader custody is not frozen")
    for name, ids, groups, rebuilt in (("att", att_ids, att_groups, att),
                                       ("inner", inner_ids, inner_groups, inner)):
        rid = np.asarray(ids, dtype=np.int64)
        unit = np.asarray(groups, dtype=np.int64)
        if (receipt[f"{name}_row_id_sha256"] != sha_array(rid)
                or receipt[f"{name}_row_unit_pair_sha256"] !=
                sha_array(np.column_stack((rid, unit)).astype(np.int64))
                or receipt[f"{name}_view_content_sha256"] != content_hashes(rebuilt)):
            raise ValueError("reader row/view identity differs from actual release")
        view_file = directory / f"view_{name}.npz"
        if sha_file(view_file) != receipt["files"][view_file.name]:
            raise ValueError("frozen view file SHA differs")
        with np.load(view_file, allow_pickle=False) as frozen:
            expected = {"row_id", "unit", "x", "x_defense_aware"}
            if rebuilt["x_metadata"] is not None:
                expected.add("x_metadata")
            if set(frozen.files) != expected:
                raise ValueError("frozen view fields differ")
            if not np.array_equal(frozen["row_id"], rid) or not np.array_equal(frozen["unit"], unit):
                raise ValueError("frozen view row/group order differs")
            for key in expected - {"row_id", "unit"}:
                if not np.array_equal(frozen[key], rebuilt[key]):
                    raise ValueError(f"fitted matrix {key} differs from actual wire")
    schema = {key: json.loads(json.dumps(att[key])) for key in
              ("components", "wire_components", "component_columns", "wire_constants")}
    if schema != receipt["view_schema"] or \
            any(json.loads(json.dumps(att[key])) != json.loads(json.dumps(inner[key]))
                for key in schema):
        raise ValueError("reader schema differs from actual complete wire")

    fit, tune, tune_groups = ordered_split(att_groups)
    y_att = np.asarray(sex_att, dtype=np.int64)
    if y_att.shape != np.asarray(att_ids).shape or np.unique(y_att[fit]).size != 2 or \
            np.unique(y_att[tune]).size != 2:
        raise ValueError("ATT split lacks both SEX classes")
    if (receipt["split"]["fit_groups"] != 3046
            or receipt["split"]["tune_groups"] != 1015
            or receipt["split"]["tune_group_id_sha256"] != sha_array(np.sort(tune_groups))):
        raise ValueError("reader split differs from independent group hash")

    saved_tune = _frozen_prediction(directory / "predictions_tune.npz",
                                    receipt["files"]["predictions_tune.npz"])
    saved_inner = _frozen_prediction(directory / "predictions_inner.npz",
                                     receipt["files"]["predictions_inner.npz"])
    if tuple(saved_inner) != tuple(receipt["reader_order"]) or set(saved_tune) != set(saved_inner):
        raise ValueError("reader order/prediction names differ")
    if not set(MAIN_READERS).issubset(saved_inner):
        raise ValueError("fixed common reader slate incomplete")
    for name in MAIN_READERS:
        model_file = directory / f"model__{name}.joblib"
        if sha_file(model_file) != receipt["files"][model_file.name]:
            raise ValueError("fitted reader model SHA differs")
        model = joblib.load(model_file)
        x_att = att["x_defense_aware"] if name == "DA_MLP_64x32" else att["x"]
        x_inner = inner["x_defense_aware"] if name == "DA_MLP_64x32" else inner["x"]
        predicted_tune = np.asarray(model.predict_proba(x_att[tune])[:, 1], dtype=np.float64)
        predicted_inner = np.asarray(model.predict_proba(x_inner)[:, 1], dtype=np.float64)
        if not np.array_equal(predicted_tune, saved_tune[name]) or \
                not np.array_equal(predicted_inner, saved_inner[name]):
            raise ValueError("fitted reader prediction differs from saved vector")
    aliases = receipt.get("ignore_recipient_aliases", [])
    if view == "pair":
        if pair_aliases is None or len(aliases) != 8:
            raise ValueError("coalition ignore-recipient aliases incomplete")
        for alias in aliases:
            name = alias["alias"]
            source = alias["source_view"]
            reader = alias["source_reader"]
            if alias.get("new_fit") is not False or source not in pair_aliases or \
                    name not in saved_inner or reader not in MAIN_READERS:
                raise ValueError("invalid ignore-recipient provenance")
            local_tune, local_inner = pair_aliases[source]
            index = MAIN_READERS.index(reader)
            if not np.array_equal(saved_tune[name], local_tune[index]) or \
                    not np.array_equal(saved_inner[name], local_inner[index]):
                raise ValueError("pair ignore-recipient vector differs from local reader")
    elif aliases:
        raise ValueError("local view has unexpected ignore-recipient alias")

    validation = {name: max(auc(y_att[tune], values),
                            1.0 - auc(y_att[tune], values))
                  for name, values in saved_tune.items()}
    selected = next(iter(saved_tune))
    best = -1.0
    for name in receipt["reader_order"]:
        if validation[name] > best + 1e-12:
            selected, best = name, validation[name]
    orientation = "positive" if auc(y_att[tune], saved_tune[selected]) >= 0.5 else "reverse"
    if selected != receipt["validation_winner"] or \
            orientation != receipt["validation_winner_orientation"]:
        raise ValueError("ATT validation winner/orientation differs")
    return {"reader_receipt_sha256": sha_file(receipt_path),
            "reader_ids": list(receipt["reader_order"]),
            "predictions_inner": saved_inner, "predictions_tune": saved_tune,
            "validation_winner": selected,
            "att_fit_rows": len(fit), "att_tune_rows": len(tune),
            "att_fit_sex_counts": np.bincount(y_att[fit], minlength=2).tolist(),
            "att_tune_sex_counts": np.bincount(y_att[tune], minlength=2).tolist()}

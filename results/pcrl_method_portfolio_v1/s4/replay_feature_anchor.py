"""Independent feature-track raw83+Ucal anchor reconstruction.

Reads only role-specific admitted files inside a separately admitted S3 replay
unit. It imports no shared anchor, attack, scoring, or selection implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from replay_b_readers import auc
from replay_b_release import ordered_split, sha_array, sha_file


ANCHOR_ID = "FEATURE_REF_RAW83_U"
FEATURE_NAMES_SHA256 = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"
READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
           "RAW_LR_C1", "RAW_HGB_15x200")


def source_forward(model_path: Path, head_paths: tuple[Path, Path],
                   expected_hashes: dict[str, str], x: np.ndarray,
                   references: dict) -> dict:
    """Replay the pinned U encoder and fitted public heads on permitted X."""
    paths = {"model.pt": model_path, "head_0.joblib": head_paths[0],
             "head_1.joblib": head_paths[1]}
    if any(sha_file(path) != expected_hashes[name] for name, path in paths.items()):
        raise ValueError("pinned unprotected source model/head bytes differ")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    shapes = {f"enc.{r}.{layer}.{field}": shape
              for r in (0, 1)
              for layer, weight, bias in ((0, (64, 83), (64,)),
                                          (2, (64, 64), (64,)),
                                          (4, (16, 64), (16,)))
              for field, shape in (("weight", weight), ("bias", bias))}
    shapes.update({f"head.{r}.{field}": shape
                   for r, width in ((0, 2), (1, 6))
                   for field, shape in (("weight", (width, 16)), ("bias", (width,)))})
    if set(state) != set(shapes) or any(tuple(state[key].shape) != shape or
                                        state[key].dtype != torch.float32
                                        for key, shape in shapes.items()):
        raise ValueError("unprotected source state differs from pinned architecture")
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    residuals = {}
    with torch.no_grad():
        for r, width in ((0, 2), (1, 6)):
            h = tensor
            for layer in (0, 2, 4):
                h = F.linear(h, state[f"enc.{r}.{layer}.weight"],
                             state[f"enc.{r}.{layer}.bias"])
                if layer != 4:
                    h = F.relu(h)
            head = joblib.load(head_paths[r])
            prob = np.asarray(head.predict_proba(h.double().numpy()), dtype=np.float64)
            expected = np.asarray(references[f"U0_{r + 1}"], dtype=np.float64)
            if prob.shape != expected.shape or prob.shape[1] != width:
                raise ValueError("source U0 class/row schema differs")
            residual = float(np.max(np.abs(prob - expected)))
            if residual > 1e-6 or not np.array_equal(prob.argmax(1), expected.argmax(1)):
                raise ValueError("source U0 probability or decision differs from admitted reference")
            residuals[f"U0_{r + 1}_max_abs_error"] = residual
    return residuals


def build_views(features: dict, references: dict, source_sha: str) -> dict[str, dict]:
    """Independently construct the exact raw values and public score wire."""
    rid = np.asarray(features["row_id"], dtype=np.int64)
    unit = np.asarray(features["unit"], dtype=np.int64)
    if (not np.array_equal(rid, references["row_id"])
            or not np.array_equal(unit, references["unit"])):
        raise ValueError("anchor feature/reference people differ")
    names = tuple(str(name) for name in features["feature_names"])
    if len(names) != 83 or len(set(names)) != 83 or hashlib.sha256(
            "\n".join(names).encode()).hexdigest() != FEATURE_NAMES_SHA256:
        raise ValueError("anchor permitted feature schema differs")
    raw = np.asarray(features["X"], dtype=np.float64)
    if raw.shape != (len(rid), 83) or not np.isfinite(raw).all():
        raise ValueError("raw anchor feature matrix differs")
    fields = {}
    for r, width in ((1, 2), (2, 6)):
        q = np.asarray(references[f"Ucal_{r}"], dtype=np.float64)
        u0 = np.asarray(references[f"U0_{r}"], dtype=np.float64)
        d = np.asarray(references[f"d_{r}"], dtype=np.int64)
        if (q.shape != (len(rid), width) or u0.shape != q.shape
                or d.shape != (len(rid),) or not np.isfinite(q).all()
                or not np.isfinite(u0).all() or np.any(q < 0) or np.any(u0 < 0)
                or not np.allclose(q.sum(1), 1, atol=1e-8, rtol=0)
                or not np.allclose(u0.sum(1), 1, atol=1e-8, rtol=0)
                or not np.array_equal(q.argmax(1), d)):
            raise ValueError("anchor Ucal/U0/decision role arrays differ")
        fields[r] = (q, np.eye(width, dtype=np.float64)[d], u0)

    def one(recipients: tuple[int, ...]) -> dict:
        arrays, bindings, components, constants, u0_parts = [], [], [], [], []
        position = 0
        for r in recipients:
            q, decision, u0 = fields[r]
            for suffix, value in (("selected_values", raw), ("probability", q),
                                  ("decision", decision)):
                name = f"r{r}.{suffix}"
                arrays.append(value)
                bindings.append((name, position, position + value.shape[1]))
                components.append(name)
                position += value.shape[1]
            constants.extend(((f"r{r}.model_id", ANCHOR_ID),
                              (f"r{r}.source_model_sha256", source_sha),
                              (f"r{r}.calibration", "Ucal_H_GLOBAL_TEMP"),
                              (f"r{r}.fallback", "false")))
            u0_parts.append(u0)
        x = np.column_stack(arrays)
        return {"row_id": rid, "unit": unit, "x": x,
                "x_defense_aware": np.column_stack((x, *u0_parts)),
                "x_raw_only": raw,
                "components": tuple(components),
                "wire_components": tuple(components) + tuple(name for name, _ in constants),
                "component_columns": tuple(bindings),
                "wire_constants": tuple(constants)}

    return {"r1": one((1,)), "r2": one((2,)), "pair": one((1, 2))}


def _content_hash(view: dict) -> dict:
    schema = {key: view[key] for key in ("components", "wire_components",
                                         "component_columns", "wire_constants")}
    return {"x": sha_array(view["x"]),
            "x_defense_aware": sha_array(view["x_defense_aware"]),
            "x_raw_only": sha_array(view["x_raw_only"]),
            "x_metadata": None, "x_code_bits": None, "cell_key": None,
            "wire_schema": hashlib.sha256(json.dumps(schema, sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest()}


def replay_readers(root: Path, name: str, att: dict, inner: dict,
                   y_att: np.ndarray, local: dict[str, dict] | None = None) -> dict:
    """Rebuild frozen views and all six fresh readers, then pair aliases."""
    if name not in ("r1", "r2", "pair"):
        raise ValueError("anchor reader view unknown")
    directory = root / f"readers_{name}"
    receipt_path = directory / "READERS_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "FROZEN":
        raise ValueError("anchor reader slate not frozen")
    schema = {key: json.loads(json.dumps(att[key])) for key in
              ("components", "wire_components", "component_columns", "wire_constants")}
    if schema != receipt["view_schema"] or any(
            json.loads(json.dumps(inner[key])) != schema[key] for key in schema):
        raise ValueError("anchor wire schema differs from reader receipt")
    for role, rebuilt in (("att", att), ("inner", inner)):
        if (receipt[f"{role}_row_id_sha256"] != sha_array(rebuilt["row_id"])
                or receipt[f"{role}_row_unit_pair_sha256"] != sha_array(
                    np.column_stack((rebuilt["row_id"], rebuilt["unit"])).astype(np.int64))
                or receipt[f"{role}_view_content_sha256"] != _content_hash(rebuilt)):
            raise ValueError("anchor reader row or complete view hash differs")
        saved_path = directory / f"view_{role}.npz"
        if sha_file(saved_path) != receipt["files"][saved_path.name]:
            raise ValueError("anchor reader view archive bytes differ")
        with np.load(saved_path, allow_pickle=False) as saved:
            if set(saved.files) != {"row_id", "unit", "x", "x_defense_aware", "x_raw_only"}:
                raise ValueError("anchor reader saved matrix fields differ")
            if any(not np.array_equal(saved[key], rebuilt[key]) for key in saved.files):
                raise ValueError("anchor reader saved matrix differs from reconstructed wire")
    fit, tune, groups = ordered_split(att["unit"])
    if (receipt["split"]["tune_group_id_sha256"] != sha_array(np.sort(groups))
            or receipt["split"]["fit_groups"] != 3046
            or receipt["split"]["tune_groups"] != 1015):
        raise ValueError("anchor reader ATT split differs")
    if np.unique(y_att[fit]).size != 2 or np.unique(y_att[tune]).size != 2:
        raise ValueError("anchor ATT split lacks SEX class")
    predictions = {}
    for role in ("tune", "inner"):
        path = directory / f"predictions_{role}.npz"
        if sha_file(path) != receipt["files"][path.name]:
            raise ValueError("anchor frozen prediction archive bytes differ")
        with np.load(path, allow_pickle=False) as saved:
            predictions[role] = {key: np.asarray(saved[key], dtype=np.float64).copy()
                                 for key in saved.files}
    if (tuple(predictions["inner"]) != tuple(receipt["reader_order"])
            or set(predictions["tune"]) != set(predictions["inner"])
            or not set(READERS).issubset(predictions["inner"])):
        raise ValueError("anchor reader order/slate differs")
    for reader in READERS:
        path = directory / f"model__{reader}.joblib"
        if sha_file(path) != receipt["files"][path.name]:
            raise ValueError("anchor fitted reader model bytes differ")
        model = joblib.load(path)
        field = "x_defense_aware" if reader == "DA_MLP_64x32" else \
            "x_raw_only" if reader.startswith("RAW_") else "x"
        for role, matrix in (("tune", att[field][tune]), ("inner", inner[field])):
            actual = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
            if not np.array_equal(actual, predictions[role][reader]):
                raise ValueError("anchor frozen reader prediction differs")
    aliases = receipt.get("ignore_recipient_aliases", [])
    if name == "pair":
        if local is None or len(aliases) != 12:
            raise ValueError("anchor pair ignore aliases incomplete")
        for alias in aliases:
            view = alias["source_view"]
            reader = alias["source_reader"]
            if (view not in local or reader not in READERS or alias.get("new_fit") is not False
                    or alias["alias"] not in predictions["inner"]):
                raise ValueError("anchor pair alias provenance differs")
            for role in ("tune", "inner"):
                if not np.array_equal(predictions[role][alias["alias"]],
                                      local[view]["predictions"][role][reader]):
                    raise ValueError("anchor pair alias differs from local reader")
    elif aliases:
        raise ValueError("anchor local reader has pair aliases")
    winner, best = receipt["reader_order"][0], -1.0
    for reader in receipt["reader_order"]:
        value = max(auc(y_att[tune], predictions["tune"][reader]),
                    1.0 - auc(y_att[tune], predictions["tune"][reader]))
        if value > best + 1e-12:
            winner, best = reader, value
    direction = "positive" if auc(y_att[tune], predictions["tune"][winner]) >= 0.5 else "reverse"
    if (winner != receipt["validation_winner"]
            or direction != receipt["validation_winner_orientation"]):
        raise ValueError("anchor ATT validation winner differs")
    return {"receipt_sha256": sha_file(receipt_path), "predictions": predictions,
            "validation_winner": winner, "att_fit_rows": len(fit),
            "att_tune_rows": len(tune), "att_fit_sex_counts": np.bincount(y_att[fit], minlength=2).tolist(),
            "att_tune_sex_counts": np.bincount(y_att[tune], minlength=2).tolist()}

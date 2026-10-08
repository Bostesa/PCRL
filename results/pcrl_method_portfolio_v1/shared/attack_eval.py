"""Fixed fresh SEX reader slate and group-aware ATTACK_FIT_NEW split.

Pure view construction/scoring functions are testable without private data.
``fit_slate`` is a real fit and must run only after S1's effective seal and
under S3's global heavy-process queue. Inner labels are supplied only to
``score_slate`` after all reader models, configurations and validation-selected
orientations are frozen on ATTACK_FIT_NEW.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common_eval import assert_same_people
from safe_data_guard import sha_arr, sha_file


SCHEMA = "pcrl-portfolio-attack-slate-v1"
SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"
ATTACK_GROUPS = 4061
TUNE_GROUPS = 1015
READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32", "CELL_A1",
           "META_LR_C1", "CODE_BIT_LR_C1", "RAW_LR_C1", "RAW_HGB_15x200")
TIE = 1e-12


def _binary(y, n: int) -> np.ndarray:
    val = np.asarray(y, dtype=np.int64)
    if val.shape != (n,) or np.any((val < 0) | (val > 1)):
        raise ValueError("invalid SEX label shape/range")
    return val


def attack_fit_tune_split(unit) -> dict:
    """Fixed 1,015/3,046 group split, chosen without SEX or view values."""
    grp = np.asarray(unit, dtype=np.int64)
    if grp.ndim != 1 or np.unique(grp).size != ATTACK_GROUPS:
        raise ValueError("ATTACK_FIT_NEW group identity/count mismatch")
    ranked = sorted((hashlib.sha256((SPLIT_SALT + str(int(g))).encode()).hexdigest(), int(g))
                    for g in np.unique(grp))
    tune_groups = np.asarray([g for _, g in ranked[:TUNE_GROUPS]], dtype=np.int64)
    tune = np.isin(grp, tune_groups)
    fit = ~tune
    if not np.any(fit) or not np.any(tune) or np.intersect1d(grp[fit], grp[tune]).size:
        raise ValueError("attack fit/tune group overlap or empty side")
    return {"fit": np.flatnonzero(fit), "tune": np.flatnonzero(tune),
            "fit_groups": ATTACK_GROUPS - TUNE_GROUPS, "tune_groups": TUNE_GROUPS,
            "tune_group_id_sha256": sha_arr(np.sort(tune_groups))}


@dataclass(frozen=True)
class AttackView:
    row_id: np.ndarray
    unit: np.ndarray
    x: np.ndarray
    components: tuple[str, ...]
    # A team view builder must enumerate the actual wire fields and bind every
    # column of x to one. S4 separately reviews the builder against deployment.
    wire_components: tuple[str, ...]
    component_columns: tuple[tuple[str, int, int], ...]
    # Constant public string/scalar fields remain in the wire contract. They
    # have no person-varying numeric column but their literal values are bound.
    wire_constants: tuple[tuple[str, str], ...] = ()
    x_defense_aware: np.ndarray | None = None
    cell_key: np.ndarray | None = None
    x_metadata: np.ndarray | None = None
    x_code_bits: np.ndarray | None = None
    x_raw_only: np.ndarray | None = None

    def validate(self) -> "AttackView":
        rid = np.asarray(self.row_id, dtype=np.int64)
        unit = np.asarray(self.unit, dtype=np.int64)
        x = np.asarray(self.x, dtype=np.float64)
        if rid.ndim != 1 or unit.shape != rid.shape or x.ndim != 2 or x.shape[0] != rid.size or x.shape[1] == 0:
            raise ValueError("attack view row/feature shape mismatch")
        if np.unique(rid).size != rid.size or not np.all(np.isfinite(x)):
            raise ValueError("attack view IDs/nonfinite feature mismatch")
        if not self.components or not all(isinstance(name, str) and name for name in self.components):
            raise ValueError("complete-view component declaration missing")
        constant_names = [name for name, _ in self.wire_constants]
        if len(set(self.components)) != len(self.components) or \
                len(set(self.wire_components)) != len(self.wire_components) or \
                len(set(constant_names)) != len(constant_names) or \
                set(self.components).intersection(constant_names) or \
                set(self.wire_components) != set(self.components).union(constant_names):
            raise ValueError("attack components differ from declared complete wire schema")
        covered = np.zeros(x.shape[1], dtype=np.int8)
        named = set()
        for name, start, end in self.component_columns:
            if name not in self.components or name in named or not 0 <= start < end <= x.shape[1]:
                raise ValueError("invalid attack component column binding")
            covered[start:end] += 1
            named.add(name)
        if named != set(self.components) or np.any(covered != 1):
            raise ValueError("wire component omitted or overlapping in attack matrix")
        if self.x_defense_aware is not None:
            da = np.asarray(self.x_defense_aware, dtype=np.float64)
            if da.ndim != 2 or da.shape[0] != rid.size or da.shape[1] == 0 or not np.all(np.isfinite(da)):
                raise ValueError("defense-aware view shape mismatch")
        if self.cell_key is not None and np.asarray(self.cell_key).shape != rid.shape:
            raise ValueError("cell key shape mismatch")
        for extra in (self.x_metadata, self.x_code_bits, self.x_raw_only):
            if extra is not None:
                arr = np.asarray(extra, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] != rid.size or arr.shape[1] == 0 or not np.all(np.isfinite(arr)):
                    raise ValueError("public metadata/code-bit view shape mismatch")
        return self

    def content_hashes(self) -> dict[str, str | None]:
        self.validate()
        out = {"x": sha_arr(np.asarray(self.x, dtype=np.float64)),
               "x_defense_aware": None, "cell_key": None,
               "x_metadata": None, "x_code_bits": None, "x_raw_only": None}
        for name in ("x_defense_aware", "x_metadata", "x_code_bits", "x_raw_only"):
            value = getattr(self, name)
            if value is not None:
                out[name] = sha_arr(np.asarray(value, dtype=np.float64))
        if self.cell_key is not None:
            # Avoid hashing object-array pointer bytes; canonicalize each
            # categorical key as a UTF-8 string before the array digest.
            out["cell_key"] = sha_arr(np.asarray([str(v) for v in self.cell_key], dtype=np.str_))
        schema = {"components": self.components, "wire_components": self.wire_components,
                  "component_columns": self.component_columns, "wire_constants": self.wire_constants}
        out["wire_schema"] = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return out


def _factory(name: str, seed: int):
    if name in {"LR_C1", "META_LR_C1", "CODE_BIT_LR_C1", "RAW_LR_C1"}:
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"))
    if name in {"MLP_64x32", "DA_MLP_64x32"}:
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                                              solver="adam", alpha=1e-4, max_iter=300,
                                                              early_stopping=False, random_state=seed))
    if name in {"HGB_15x200", "RAW_HGB_15x200"}:
        return HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, max_iter=200,
                                              early_stopping=False, random_state=seed)
    raise KeyError(name)


def _cell_fit(keys, y, alpha: float = 1.0):
    counts = {}
    for key, label in zip(keys, y):
        key = str(key)
        if key not in counts:
            counts[key] = [0, 0]
        counts[key][int(label)] += 1
    prior = float((int(np.sum(y)) + alpha) / (len(y) + 2 * alpha))
    return counts, prior


def _cell_predict(model, keys, alpha: float = 1.0):
    counts, prior = model
    out = np.empty(len(keys), dtype=np.float64)
    for j, key in enumerate(keys):
        pair = counts.get(str(key))
        out[j] = prior if pair is None else (pair[1] + alpha) / (pair[0] + pair[1] + 2 * alpha)
    return out


def oriented_auc(y, prediction) -> dict:
    """AUC for two prespecified zero-fit aliases of one fitted reader."""
    y = _binary(y, len(prediction))
    if np.unique(y).size != 2:
        raise ValueError("AUC needs both SEX classes")
    p = np.asarray(prediction, dtype=np.float64)
    if p.shape != y.shape or not np.all(np.isfinite(p)):
        raise ValueError("reader prediction shape/nonfinite mismatch")
    auc = float(roc_auc_score(y, p))
    return {"positive_alias_auc": auc, "reverse_alias_auc": 1.0 - auc,
            "fixed_alias_max_auc": max(auc, 1.0 - auc)}


def _label_record(record: dict, view: AttackView) -> np.ndarray:
    if not isinstance(record, dict) or "sex" not in record:
        raise ValueError("SEX labels must be an ordered role record")
    assert_same_people(record, {"row_id": view.row_id, "unit": view.unit})
    return _binary(record["sex"], len(view.row_id))


def fit_slate(att_view: AttackView, sex_att: dict, inner_view: AttackView, *,
              required_readers: tuple[str, ...], attacker_seed: int = 0,
              progress_dir: str | Path | None = None) -> dict:
    """Fit fixed readers on ATTACK_TRAIN, orient/select on ATTACK_TUNE.

    No INNER label enters this function. Return private fitted models and frozen
    per-reader prediction arrays; callers must store them privately.
    """
    att_view.validate()
    inner_view.validate()
    if att_view.components != inner_view.components or \
            att_view.wire_components != inner_view.wire_components or \
            att_view.component_columns != inner_view.component_columns or \
            att_view.wire_constants != inner_view.wire_constants:
        raise ValueError("attack/inner component schemas differ")
    if not required_readers or len(set(required_readers)) != len(required_readers) or \
            not set(required_readers).issubset(READERS):
        raise ValueError("fixed required-reader slate missing or unknown")
    if not set(READERS[:3]).issubset(required_readers):
        raise ValueError("common LR/MLP/HGB bank is mandatory")
    if ("DA_MLP_64x32" in required_readers) and \
            (att_view.x_defense_aware is None or inner_view.x_defense_aware is None):
        raise ValueError("required defense-aware reader has no view")
    if ("CELL_A1" in required_readers) and \
            (att_view.cell_key is None or inner_view.cell_key is None):
        raise ValueError("required categorical reader has no cells")
    if ("META_LR_C1" in required_readers) and \
            (att_view.x_metadata is None or inner_view.x_metadata is None):
        raise ValueError("required metadata reader has no public metadata")
    if ("CODE_BIT_LR_C1" in required_readers) and \
            (att_view.x_code_bits is None or inner_view.x_code_bits is None):
        raise ValueError("required code-bit reader has no public bits")
    if any(name in required_readers for name in ("RAW_LR_C1", "RAW_HGB_15x200")) and \
            (att_view.x_raw_only is None or inner_view.x_raw_only is None):
        raise ValueError("required raw-only reader has no selected raw values")
    split = attack_fit_tune_split(att_view.unit)
    y = _label_record(sex_att, att_view)
    fit_idx, tune_idx = split["fit"], split["tune"]
    if np.unique(y[fit_idx]).size != 2 or np.unique(y[tune_idx]).size != 2:
        raise ValueError("attack fit/tune split lacks both SEX classes")
    models, pred_tune, pred_inner, validation = {}, {}, {}, {}
    progress_root = None
    if progress_dir is not None:
        progress_root = Path(progress_dir)
        if ".private" not in progress_root.resolve().parts or progress_root.exists():
            raise ValueError("reader progress directory must be new private custody")
        progress_root.mkdir(parents=True, mode=0o700)
    reader_progress = []
    for name in required_readers:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        if name == "CELL_A1":
            model = _cell_fit(np.asarray(att_view.cell_key)[fit_idx], y[fit_idx])
            p_tune = _cell_predict(model, np.asarray(att_view.cell_key)[tune_idx])
            p_inner = _cell_predict(model, np.asarray(inner_view.cell_key))
        else:
            if name == "DA_MLP_64x32":
                x_att = np.asarray(att_view.x_defense_aware, dtype=np.float64)
                x_inner = np.asarray(inner_view.x_defense_aware, dtype=np.float64)
            elif name == "META_LR_C1":
                x_att = np.asarray(att_view.x_metadata, dtype=np.float64)
                x_inner = np.asarray(inner_view.x_metadata, dtype=np.float64)
            elif name == "CODE_BIT_LR_C1":
                x_att = np.asarray(att_view.x_code_bits, dtype=np.float64)
                x_inner = np.asarray(inner_view.x_code_bits, dtype=np.float64)
            elif name in {"RAW_LR_C1", "RAW_HGB_15x200"}:
                x_att = np.asarray(att_view.x_raw_only, dtype=np.float64)
                x_inner = np.asarray(inner_view.x_raw_only, dtype=np.float64)
            else:
                x_att = np.asarray(att_view.x, dtype=np.float64)
                x_inner = np.asarray(inner_view.x, dtype=np.float64)
            model = _factory(name, attacker_seed)
            model.fit(x_att[fit_idx], y[fit_idx])
            p_tune = model.predict_proba(x_att[tune_idx])[:, 1]
            p_inner = model.predict_proba(x_inner)[:, 1]
        models[name] = model
        pred_tune[name] = np.asarray(p_tune, dtype=np.float64)
        pred_inner[name] = np.asarray(p_inner, dtype=np.float64)
        validation[name] = oriented_auc(y[tune_idx], p_tune)
        validation[name]["selected_orientation"] = ("positive" if validation[name]["positive_alias_auc"] >=
                                                     validation[name]["reverse_alias_auc"] else "reverse")
        progress = {"reader_id": name, "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
                    "fit_rows": int(len(fit_idx)), "tune_rows": int(len(tune_idx)),
                    "att_view_content_sha256": att_view.content_hashes(),
                    "inner_view_content_sha256": inner_view.content_hashes(),
                    "prediction_tune_sha256": sha_arr(pred_tune[name]),
                    "prediction_inner_sha256": sha_arr(pred_inner[name]),
                    "wall_seconds": time.perf_counter() - start_wall,
                    "cpu_seconds": time.process_time() - start_cpu,
                    "attacker_seed": int(attacker_seed)}
        if progress_root is not None:
            model_path = progress_root / f"model__{name}.joblib"
            pred_path = progress_root / f"predictions__{name}.npz"
            joblib.dump(model, model_path, compress=3)
            np.savez_compressed(pred_path, tune=pred_tune[name], inner=pred_inner[name])
            model_path.chmod(0o600)
            pred_path.chmod(0o600)
            progress["model_sha256"] = sha_file(model_path)
            progress["prediction_file_sha256"] = sha_file(pred_path)
            progress["total_through_checkpoint_wall_seconds"] = time.perf_counter() - start_wall
            progress["total_through_checkpoint_cpu_seconds"] = time.process_time() - start_cpu
            receipt_path = progress_root / f"completed__{name}.json"
            fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(progress, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            progress["completion_receipt_sha256"] = sha_file(receipt_path)
        reader_progress.append(progress)
    if not models:
        raise ValueError("empty attack slate")
    # Model and orientation are frozen before any INNER score is observed.
    winner = next(iter(models))
    best = -1.0
    for name in required_readers:
        if name not in models:
            continue
        val = validation[name]["fixed_alias_max_auc"]
        if val > best + TIE:
            winner, best = name, val
    return {"schema": SCHEMA, "view_components": att_view.components,
            "att_row_id_sha256": sha_arr(np.asarray(att_view.row_id, dtype=np.int64)),
            "att_row_unit_pair_sha256": sha_arr(np.column_stack((att_view.row_id, att_view.unit)).astype(np.int64)),
            "att_view_content_sha256": att_view.content_hashes(),
            "inner_row_id_sha256": sha_arr(np.asarray(inner_view.row_id, dtype=np.int64)),
            "inner_row_unit_pair_sha256": sha_arr(np.column_stack((inner_view.row_id, inner_view.unit)).astype(np.int64)),
            "inner_view_content_sha256": inner_view.content_hashes(),
            "split": {k: v for k, v in split.items() if k not in ("fit", "tune")},
            "fit_rows": int(len(fit_idx)), "tune_rows": int(len(tune_idx)),
            "reader_order": list(pred_inner), "required_readers": list(required_readers),
            "validation": validation,
            "validation_winner": winner,
            "validation_winner_orientation": validation[winner]["selected_orientation"],
            "reader_progress": reader_progress,
            "models": models, "predictions_inner": pred_inner, "predictions_tune": pred_tune}


def add_ignore_recipient_aliases(pair_fitted: dict, local1: dict, local2: dict) -> dict:
    """Attach zero-fit local readers to the coalition envelope after identity checks.

    The local and pair readers must use the same ATTACK_FIT_NEW split and the
    same aligned INNER people. An alias carries its original fitted prediction;
    it is never refit using the coalition view or INNER labels.
    """
    joined = {**pair_fitted, "predictions_inner": dict(pair_fitted["predictions_inner"]),
              "predictions_tune": dict(pair_fitted["predictions_tune"]),
              "validation": dict(pair_fitted["validation"]),
              "reader_order": list(pair_fitted["reader_order"]), "ignore_recipient_aliases": []}
    for index, local in ((1, local1), (2, local2)):
        for key in ("att_row_id_sha256", "att_row_unit_pair_sha256", "inner_row_id_sha256",
                    "inner_row_unit_pair_sha256"):
            if pair_fitted[key] != local[key]:
                raise ValueError("ignore-recipient alias has different ordered people")
        if pair_fitted["split"]["tune_group_id_sha256"] != local["split"]["tune_group_id_sha256"]:
            raise ValueError("ignore-recipient alias has different attack tuning split")
        for name in local["reader_order"]:
            alias = f"IGNORE_R{2 if index == 1 else 1}__{name}"
            if alias in joined["predictions_inner"]:
                raise ValueError("ignore-recipient alias collision")
            joined["predictions_inner"][alias] = local["predictions_inner"][name]
            joined["predictions_tune"][alias] = local["predictions_tune"][name]
            joined["validation"][alias] = local["validation"][name]
            joined["reader_order"].append(alias)
            joined["ignore_recipient_aliases"].append({"alias": alias, "source_view": f"r{index}",
                                                        "source_reader": name, "new_fit": False})
    winner = joined["reader_order"][0]
    best = -1.0
    for name in joined["reader_order"]:
        value = joined["validation"][name]["fixed_alias_max_auc"]
        if value > best + TIE:
            winner, best = name, value
    joined["validation_winner"] = winner
    joined["validation_winner_orientation"] = joined["validation"][winner]["selected_orientation"]
    return joined


def add_external_fixed_readers(fitted: dict, att_view: AttackView, sex_att: dict,
                               inner_view: AttackView, readers: tuple[dict, ...]) -> dict:
    """Admit prelocked specialist readers fitted only on ATTACK_TRAIN.

    A team supplies private frozen TUNE/INNER predictions and a model SHA. This
    function scores only TUNE to freeze orientation/winner. S4 must replay the
    specialist fit and verify its training rows and public-view construction.
    """
    att_view.validate()
    inner_view.validate()
    if sha_arr(np.asarray(att_view.row_id, dtype=np.int64)) != fitted["att_row_id_sha256"] or \
            sha_arr(np.column_stack((att_view.row_id, att_view.unit)).astype(np.int64)) != fitted["att_row_unit_pair_sha256"] or \
            sha_arr(np.asarray(inner_view.row_id, dtype=np.int64)) != fitted["inner_row_id_sha256"] or \
            sha_arr(np.column_stack((inner_view.row_id, inner_view.unit)).astype(np.int64)) != fitted["inner_row_unit_pair_sha256"]:
        raise ValueError("external reader view has different ordered people")
    if att_view.content_hashes() != fitted["att_view_content_sha256"] or \
            inner_view.content_hashes() != fitted["inner_view_content_sha256"]:
        raise ValueError("external reader view values differ from fitted view")
    split = attack_fit_tune_split(att_view.unit)
    if split["tune_group_id_sha256"] != fitted["split"]["tune_group_id_sha256"]:
        raise ValueError("external reader attack split mismatch")
    y_tune = _label_record(sex_att, att_view)[split["tune"]]
    joined = {**fitted, "predictions_inner": dict(fitted["predictions_inner"]),
              "predictions_tune": dict(fitted["predictions_tune"]),
              "validation": dict(fitted["validation"]),
              "reader_order": list(fitted["reader_order"]), "external_readers": []}
    for item in readers:
        name = item["name"]
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None or \
                name in joined["predictions_inner"]:
            raise ValueError("external reader name is missing or duplicate")
        if item.get("fit_role") != "ATTACK_FIT_NEW/ATTACK_TRAIN" or \
                item.get("model_sha256") is None or item.get("model_artifact_path") is None:
            raise ValueError("external reader fit identity missing")
        artifact = Path(item["model_artifact_path"])
        if ".private" not in artifact.resolve().parts or sha_file(artifact) != item["model_sha256"]:
            raise ValueError("external reader private model artifact mismatch")
        pt = np.asarray(item["prediction_tune"], dtype=np.float64)
        pi = np.asarray(item["prediction_inner"], dtype=np.float64)
        if pt.shape != y_tune.shape or pi.shape != inner_view.row_id.shape:
            raise ValueError("external reader prediction shape mismatch")
        val = oriented_auc(y_tune, pt)
        val["selected_orientation"] = ("positive" if val["positive_alias_auc"] >= val["reverse_alias_auc"]
                                       else "reverse")
        joined["predictions_tune"][name] = pt
        joined["predictions_inner"][name] = pi
        joined["validation"][name] = val
        joined["reader_order"].append(name)
        joined["external_readers"].append({"name": name, "model_sha256": item["model_sha256"],
                                           "model_artifact_path": str(artifact),
                                           "fit_role": item["fit_role"], "new_fit": True})
    winner = joined["reader_order"][0]
    best = -1.0
    for name in joined["reader_order"]:
        value = joined["validation"][name]["fixed_alias_max_auc"]
        if value > best + TIE:
            winner, best = name, value
    joined["validation_winner"] = winner
    joined["validation_winner_orientation"] = joined["validation"][winner]["selected_orientation"]
    return joined


def score_slate(fitted: dict, sex_inner: dict) -> dict:
    """Primary fixed-reader envelope, plus every reader and validation winner."""
    predictions = fitted["predictions_inner"]
    if not isinstance(sex_inner, dict) or "sex" not in sex_inner:
        raise ValueError("INNER SEX labels must be an ordered role record")
    if sha_arr(np.asarray(sex_inner["row_id"], dtype=np.int64)) != fitted["inner_row_id_sha256"]:
        raise ValueError("INNER SEX row order mismatch")
    pair_sha = sha_arr(np.column_stack((sex_inner["row_id"], sex_inner["unit"])).astype(np.int64))
    if pair_sha != fitted["inner_row_unit_pair_sha256"]:
        raise ValueError("INNER SEX row/group pairing mismatch")
    y = _binary(sex_inner["sex"], len(next(iter(predictions.values()))))
    scores = {name: oriented_auc(y, p) for name, p in predictions.items()}
    envelope = max(v["fixed_alias_max_auc"] for v in scores.values())
    selected = fitted["validation_winner"]
    direction = fitted["validation_winner_orientation"]
    selected_raw = scores[selected]["positive_alias_auc"]
    return {"schema": SCHEMA, "primary_envelope_auc": float(envelope), "readers": scores,
            "validation_winner": selected, "validation_winner_orientation": direction,
            "validation_winner_inner_auc": selected_raw if direction == "positive" else 1 - selected_raw}


def bootstrap_envelope(predictions: dict[str, np.ndarray], y, positions) -> float:
    """Recompute the fixed-slate max inside one paired group bootstrap replicate."""
    pos = np.asarray(positions, dtype=np.int64)
    labels = np.asarray(y, dtype=np.int64)[pos]
    return max(oriented_auc(labels, np.asarray(p)[pos])["fixed_alias_max_auc"] for p in predictions.values())


def save_fitted_slate(fitted: dict, private_dir: str | Path, *,
                      att_view: AttackView, inner_view: AttackView) -> dict:
    """Write replayable fitted readers and predictions to new private custody.

    S4 still reconstructs views from actual releases and independently checks
    each model's ATT/INNER predictions; a SHA by itself is not verification.
    """
    att_view.validate()
    inner_view.validate()
    for view, prefix in ((att_view, "att"), (inner_view, "inner")):
        if sha_arr(np.asarray(view.row_id, dtype=np.int64)) != fitted[f"{prefix}_row_id_sha256"] or \
                sha_arr(np.column_stack((view.row_id, view.unit)).astype(np.int64)) != \
                fitted[f"{prefix}_row_unit_pair_sha256"]:
            raise ValueError("reader custody view identity mismatch")
        if view.content_hashes() != fitted[f"{prefix}_view_content_sha256"]:
            raise ValueError("reader custody view values differ from fitted view")
    root = Path(private_dir)
    if ".private" not in root.resolve().parts or root.exists():
        raise ValueError("attacker custody must be new private directory")
    root.mkdir(parents=True, mode=0o700)
    files = {}
    for name, model in fitted["models"].items():
        path = root / f"model__{name}.joblib"
        joblib.dump(model, path, compress=3)
        path.chmod(0o600)
        files[path.name] = sha_file(path)
    external_custody = []
    for item in fitted.get("external_readers", []):
        source = Path(item["model_artifact_path"])
        if sha_file(source) != item["model_sha256"]:
            raise ValueError("external reader artifact changed before custody")
        path = root / f"model__{item['name']}.bin"
        shutil.copyfile(source, path)
        path.chmod(0o600)
        files[path.name] = sha_file(path)
        external_custody.append({k: v for k, v in item.items() if k != "model_artifact_path"} |
                                {"custody_file": path.name})
    for role, view in (("att", att_view), ("inner", inner_view)):
        path = root / f"view_{role}.npz"
        arrays = {"row_id": view.row_id, "unit": view.unit, "x": view.x}
        for key, value in (("x_defense_aware", view.x_defense_aware),
                           ("cell_key", view.cell_key), ("x_metadata", view.x_metadata),
                           ("x_code_bits", view.x_code_bits), ("x_raw_only", view.x_raw_only)):
            if value is not None:
                arrays[key] = value
        np.savez_compressed(path, **arrays)
        path.chmod(0o600)
        files[path.name] = sha_file(path)
    for role, key in (("tune", "predictions_tune"), ("inner", "predictions_inner")):
        path = root / f"predictions_{role}.npz"
        np.savez_compressed(path, **fitted[key])
        path.chmod(0o600)
        files[path.name] = sha_file(path)
    receipt = {"schema": "pcrl-portfolio-private-reader-custody-v1", "status": "FROZEN",
               "att_row_id_sha256": fitted["att_row_id_sha256"],
               "att_row_unit_pair_sha256": fitted["att_row_unit_pair_sha256"],
               "inner_row_id_sha256": fitted["inner_row_id_sha256"],
               "inner_row_unit_pair_sha256": fitted["inner_row_unit_pair_sha256"],
               "att_view_content_sha256": fitted["att_view_content_sha256"],
               "inner_view_content_sha256": fitted["inner_view_content_sha256"],
               "split": fitted["split"], "reader_order": fitted["reader_order"],
               "validation_winner": fitted["validation_winner"],
               "validation_winner_orientation": fitted["validation_winner_orientation"],
               "validation": fitted["validation"],
               "reader_progress": fitted.get("reader_progress", []),
               "ignore_recipient_aliases": fitted.get("ignore_recipient_aliases", []),
               "external_readers": external_custody,
               "view_schema": {"components": list(att_view.components),
                               "wire_components": list(att_view.wire_components),
                               "component_columns": [list(x) for x in att_view.component_columns],
                               "wire_constants": [list(x) for x in att_view.wire_constants]},
               "files": files,
               "numpy_version": np.__version__, "sklearn_version": __import__("sklearn").__version__}
    path = root / "READERS_RECEIPT.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {"private_receipt_sha256": sha_file(path),
            "model_files": len(fitted["models"]) + len(external_custody),
            "prediction_files": 2, "view_files": 2}

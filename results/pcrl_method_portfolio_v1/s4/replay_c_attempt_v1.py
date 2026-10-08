"""Independent replay of one frozen C arm and seed from saved deployment bytes.

The S4 queue wrapper must verify the S3 parent receipts and admitted role-file
hashes before calling this module. No C or S2 scorer implementation is used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from replay_c_endpoints_v1 import replay_score_receipt
from replay_c_readers_v1 import (READERS, fixed_split, oriented_auc,
                                 replay_model_predictions)
from replay_c_release_v1 import load_packages, predict_release
from replay_c_values_v1 import (atom_indices, load_saved_bundle, sha_array,
                                view_content_hashes)
from replay_v2_controls import replay_probe


VIEWS = {"r1": "income", "r2": "occupation_group", "pair": "coalition"}
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_json(root: Path, relative: str, expected_sha: str) -> tuple[Path, dict]:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("C replay path is not private-relative")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or _sha(path) != expected_sha:
        raise ValueError("C replay file path or SHA differs")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("C replay JSON is not an object")
    return path, data


def _role(role: Mapping[str, object], expected_rows: int,
          feature_names: tuple[str, ...]) -> dict:
    x = np.asarray(role["features"])
    row = np.asarray(role["row_id"], dtype=np.int64)
    unit = np.asarray(role["unit"], dtype=np.int64)
    if (x.shape != (expected_rows, 83) or row.shape != unit.shape or
            row.shape != (expected_rows,) or len(np.unique(row)) != expected_rows or
            tuple(role["feature_names"]) != feature_names or
            not np.isfinite(x).all()):
        raise ValueError("C admitted role features or ordered people differ")
    out = {"features": x, "row_id": row, "unit": unit}
    for name in ("sex", "income", "occupation"):
        if name in role:
            value = np.asarray(role[name], dtype=np.int64)
            if value.shape != row.shape:
                raise ValueError("C admitted role label alignment differs")
            out[name] = value
    return out


def _reader_bank(root: Path, stage: dict, att: dict, inner: dict,
                 sex_att: np.ndarray, arm: str, seed: int, view: str,
                 fit_sha: str, forward_sha: str) -> tuple[dict, dict, str]:
    if (stage.get("schema") != "team-c-reader-stage-v1" or
            stage.get("status") != "FROZEN" or
            (stage.get("arm_id"), stage.get("seed"), stage.get("view")) !=
            (arm, seed, view) or
            stage.get("fit_handoff_sha256") != fit_sha or
            stage.get("forward_handoff_sha256") != forward_sha or
            stage.get("reader_fits") != 6 or
            stage.get("required_readers") != list(READERS) or
            stage.get("att_view_content_sha256") != view_content_hashes(att) or
            stage.get("inner_view_content_sha256") != view_content_hashes(inner)):
        raise ValueError("C frozen reader stage or complete-view identity differs")
    reader_path, receipt = _bound_json(root,
        stage["s2_reader_receipt_relative_path"],
        stage["s2_reader_receipt_sha256"])
    expected_files = {f"model__{name}.joblib" for name in READERS} | {
        "view_att.npz", "view_inner.npz", "predictions_tune.npz",
        "predictions_inner.npz"}
    files = receipt.get("files", {})
    if (receipt.get("schema") != "pcrl-portfolio-private-reader-custody-v1" or
            receipt.get("status") != "FROZEN" or
            receipt.get("reader_order") != list(READERS) or
            set(files) != expected_files or
            receipt.get("att_view_content_sha256") != view_content_hashes(att) or
            receipt.get("inner_view_content_sha256") != view_content_hashes(inner)):
        raise ValueError("C S2 reader custody differs from rebuilt view")
    for name, digest in files.items():
        if _sha(reader_path.parent / name) != digest:
            raise ValueError("C S2 reader file SHA differs")
    progress = receipt.get("reader_progress")
    if (not isinstance(progress, list) or
            [item.get("reader_id") for item in progress] != list(READERS)):
        raise ValueError("C six reader progress records differ")
    progress_dir = (reader_path.parent.parent.parent / "reader_progress" /
                    reader_path.parent.name)
    for item in progress:
        reader = item["reader_id"]
        for filename, key in ((f"model__{reader}.joblib", "model_sha256"),
                              (f"predictions__{reader}.npz", "prediction_file_sha256"),
                              (f"completed__{reader}.json", "completion_receipt_sha256")):
            if _sha(progress_dir / filename) != item.get(key):
                raise ValueError("C reader progress child SHA differs")
    for role, view_data in (("att", att), ("inner", inner)):
        with np.load(reader_path.parent / f"view_{role}.npz", allow_pickle=False) as saved:
            expected = {"row_id", "unit", "x", "x_defense_aware", "x_raw_only"}
            if set(saved.files) != expected or any(
                not np.array_equal(saved[name], view_data[name]) for name in expected
            ):
                raise ValueError("C saved S2 reader matrix differs")
    with np.load(reader_path.parent / "predictions_tune.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(READERS):
            raise ValueError("C reader tune bank is incomplete")
        tune_predictions = {name: np.asarray(saved[name], dtype=np.float64)
                            for name in READERS}
    with np.load(reader_path.parent / "predictions_inner.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(READERS):
            raise ValueError("C reader INNER bank is incomplete")
        inner_predictions = {name: np.asarray(saved[name], dtype=np.float64)
                             for name in READERS}
    for name in READERS:
        with np.load(progress_dir / f"predictions__{name}.npz",
                     allow_pickle=False) as saved:
            if (set(saved.files) != {"tune", "inner"} or
                    not np.array_equal(saved["tune"], tune_predictions[name]) or
                    not np.array_equal(saved["inner"], inner_predictions[name])):
                raise ValueError("C reader progress prediction differs")
    fit, tune, split = fixed_split(att["unit"])
    if receipt.get("split") != split:
        raise ValueError("C frozen reader split differs")
    validation = {}
    for name in READERS:
        replay_model_predictions(reader_path.parent / f"model__{name}.joblib",
                                 files[f"model__{name}.joblib"], name,
                                 att, inner, tune, tune_predictions[name],
                                 inner_predictions[name])
        item = oriented_auc(sex_att[tune], tune_predictions[name])
        item["selected_orientation"] = ("positive" if
            item["positive_alias_auc"] >= item["reverse_alias_auc"]
            else "reverse")
        validation[name] = item
        saved = receipt.get("validation", {}).get(name, {})
        if (set(saved) != set(item) or any(
            (abs(float(saved[key]) - value) > 1e-12 if isinstance(value, float)
             else saved[key] != value)
            for key, value in item.items()
        )):
            raise ValueError("C frozen reader validation differs")
    winner, best = READERS[0], -1.0
    for name in READERS:
        value = validation[name]["fixed_alias_max_auc"]
        if value > best + 1e-12:
            winner, best = name, value
    if (receipt.get("validation_winner") != winner or
            receipt.get("validation_winner_orientation") !=
            validation[winner]["selected_orientation"]):
        raise ValueError("C reader selection used INNER or altered fixed tie order")
    return inner_predictions, validation, stage["s2_reader_receipt_sha256"]


def replay_attempt(*, arm: str, seed: int, lock_sha256: str,
                   manifest_sha256: str, private_root: str | Path,
                   feature_names: tuple[str, ...], role_values: Mapping[str, dict],
                   role_source_sha256: Mapping[str, str], paths: Mapping[str, object],
                   return_artifacts: bool = False) -> dict | tuple[dict, dict]:
    """Replay release, readers, V2 controls and point scores for one arm/seed."""
    root = Path(private_root).resolve(strict=True)
    if root.name != ".private" or set(role_values) != set(ROLES):
        raise ValueError("C S4 private root or admitted role set differs")
    roles = {"ATTACK_FIT_NEW": _role(role_values["ATTACK_FIT_NEW"], 4064,
                                     feature_names),
             "INNER_SELECTION": _role(role_values["INNER_SELECTION"], 2235,
                                      feature_names)}
    if ("sex" not in roles["ATTACK_FIT_NEW"] or
            not {"sex", "income", "occupation"} <= set(roles["INNER_SELECTION"])):
        raise ValueError("C replay needs the admitted ATT and INNER labels")
    fit_path, fit = _bound_json(root, paths["fit_relative_path"],
                                paths["fit_sha256"])
    forward_path, forward = _bound_json(root, paths["forward_relative_path"],
                                        paths["forward_sha256"])
    if (fit.get("arm_id") != arm or fit.get("seed") != seed or
            fit.get("campaign_lock_sha256") != lock_sha256 or
            forward.get("schema") != "team-c-postfit-forward-handoff-v1" or
            (forward.get("arm_id"), forward.get("seed")) != (arm, seed) or
            forward.get("campaign_lock_sha256") != lock_sha256 or
            forward.get("fit_handoff_sha256") != paths["fit_sha256"] or
            forward.get("data_manifest_sha256") != manifest_sha256 or
            set(forward.get("roles", {})) != set(ROLES)):
        raise ValueError("C fit/forward handoff parent chain differs")
    fit_fields = {"OSF_DEFENSE_FIT:features", "OSF_DEFENSE_FIT:income",
                  "OSF_DEFENSE_FIT:occupation", "OSF_DEFENSE_FIT:sex",
                  "CALIBRATION_HELDOUT:features",
                  "CALIBRATION_HELDOUT:income",
                  "CALIBRATION_HELDOUT:occupation"}
    fit_sources = {key: role_source_sha256[key] for key in fit_fields}
    packages = load_packages(fit, root, feature_names, lock_sha256,
                             manifest_sha256, fit_sources)
    atoms = atom_indices(feature_names)
    selected = {purpose: tuple(fit["selected_atoms"][purpose])
                for purpose in ("income", "occupation_group")}
    union = set(selected["income"]) | set(selected["occupation_group"])
    semantic = [len(selected["income"]), len(selected["occupation_group"]),
                len(union)]
    encoded = [sum(len(atoms[atom]) for atom in selected["income"]),
               sum(len(atoms[atom]) for atom in selected["occupation_group"]),
               sum(len(atoms[atom]) for atom in union)]
    budget = {"semantic": semantic, "encoded": encoded,
              "raw_feature_payload_bits_per_person": [32 * width for width in encoded],
              "raw_feature_dtype": "float32"}
    if fit.get("arm_receipt", {}).get("budget") != budget:
        raise ValueError("C independently recomputed selected capacity differs")
    if forward.get("package_sha256_by_purpose") != {
        purpose: packages[purpose]["package_sha256"]
        for purpose in ("income", "occupation_group")
    }:
        raise ValueError("C forward handoff public packages differ from fit")
    role_bundles = {}
    for role_name in ROLES:
        role = roles[role_name]
        raw, calibrated = {}, {}
        for purpose in ("income", "occupation_group"):
            raw[purpose], calibrated[purpose] = predict_release(
                packages[purpose], role["features"], feature_names)
        entry = forward["roles"][role_name]
        if (entry.get("role_feature_sha256") !=
                role_source_sha256[f"{role_name}:features"] or
                entry.get("row_unit_pair_sha256") != sha_array(
                    np.column_stack((role["row_id"], role["unit"])).astype(np.int64))):
            raise ValueError("C forward role feature or ordered person source differs")
        receipt_path, receipt = _bound_json(
            root, entry["receipt_relative_path"], entry["receipt_sha256"])
        if (receipt.get("role") != role_name or
                (receipt.get("arm_id"), receipt.get("seed")) != (arm, seed) or
                receipt.get("campaign_lock_sha256") != lock_sha256 or
                receipt.get("data_manifest_sha256") != manifest_sha256 or
                receipt.get("role_feature_sha256") !=
                role_source_sha256[f"{role_name}:features"] or
                receipt.get("package_sha256_by_purpose") !=
                forward["package_sha256_by_purpose"]):
            raise ValueError("C private complete-view receipt source differs")
        bundle = load_saved_bundle(receipt_path,
                                   receipt_path.with_suffix(".npz"),
                                   role["features"], feature_names, raw,
                                   role["row_id"], role["unit"])
        for purpose in ("income", "occupation_group"):
            wire = bundle["wires"][purpose]
            package = packages[purpose]["package"]
            if (not np.array_equal(wire["probabilities"], calibrated[purpose]) or
                    wire["head_id"] !=
                    f"{package['arm_id']}:{package['head_sha256'][:16]}" or
                    wire["calibrator_id"] !=
                    f"H-GLOBAL-TEMP:{package['calibrator_sha256'][:16]}"):
                raise ValueError("C independently deployed probability differs")
        role_bundles[role_name] = bundle
    readers, validation, controls, s2_receipts = {}, {}, {}, {}
    for view, name in VIEWS.items():
        reader_path, reader_stage = _bound_json(
            root, paths["reader_relative_path_by_view"][view],
            paths["reader_sha256_by_view"][view])
        att = role_bundles["ATTACK_FIT_NEW"]["views"][name]
        inner = role_bundles["INNER_SELECTION"]["views"][name]
        readers[view], validation[view], s2_receipts[view] = _reader_bank(
            root, reader_stage, att, inner,
            roles["ATTACK_FIT_NEW"]["sex"], arm, seed, view,
            paths["fit_sha256"], paths["forward_sha256"])
        v2_path, v2 = _bound_json(root,
            paths["v2_relative_path_by_view"][view],
            paths["v2_sha256_by_view"][view])
        if (v2.get("schema") != "team-c-v2-stage-v1" or
                v2.get("status") != "PASS" or
                (v2.get("arm_id"), v2.get("seed"), v2.get("view")) !=
                (arm, seed, view) or
                v2.get("reader_receipt_sha256") !=
                paths["reader_sha256_by_view"][view] or
                v2.get("fit_handoff_sha256") != paths["fit_sha256"] or
                v2.get("forward_handoff_sha256") != paths["forward_sha256"] or
                v2.get("attack_sex_source_sha256") !=
                role_source_sha256["ATTACK_FIT_NEW:sex"] or
                v2.get("inner_sex_source_sha256") !=
                role_source_sha256["INNER_SELECTION:sex"] or
                v2.get("att_view_content_sha256") != view_content_hashes(att) or
                v2.get("inner_view_content_sha256") != view_content_hashes(inner)):
            raise ValueError("C V2 stage parent, role or view differs")
        custody_path, _ = _bound_json(root, v2["v2_custody_relative_path"],
                                      v2["v2_custody_sha256"])
        controls[view] = replay_probe(
            custody_path.parent, v2["v2_aggregate"], att, inner,
            roles["ATTACK_FIT_NEW"]["sex"], roles["INNER_SELECTION"]["sex"])
        if controls[view]["status"] != "PASS":
            raise ValueError("C independently replayed V2 control failed")
    score_path, score = _bound_json(root, paths["score_relative_path"],
                                    paths["score_sha256"])
    if ((score.get("arm_id"), score.get("seed")) != (arm, seed) or
            score.get("fit_handoff_sha256") != paths["fit_sha256"] or
            score.get("forward_handoff_sha256") != paths["forward_sha256"] or
            score.get("inner_role_label_sha256") != {
                field: role_source_sha256[f"INNER_SELECTION:{field}"]
                for field in ("income", "occupation", "sex")
            } or
            score.get("reader_receipt_sha256") !=
            paths["reader_sha256_by_view"] or
            score.get("v2_receipt_sha256") != paths["v2_sha256_by_view"]):
        raise ValueError("C score-stage parent chain differs")
    point = replay_score_receipt(
        score,
        {"income": role_bundles["INNER_SELECTION"]["wires"]["income"]["probabilities"],
         "occupation": role_bundles["INNER_SELECTION"]["wires"]["occupation_group"]["probabilities"]},
        {"income": roles["INNER_SELECTION"]["income"],
         "occupation": roles["INNER_SELECTION"]["occupation"]},
        roles["INNER_SELECTION"]["sex"], readers, validation)
    result = {"status": "REPLAYED_C_R1_PASS", "arm": arm, "seed": seed,
            "fit_handoff_sha256": _sha(fit_path),
            "forward_handoff_sha256": _sha(forward_path),
            "reader_s2_receipt_sha256_by_view": s2_receipts,
            "control_replay_by_view": controls,
            "point_endpoints": point,
            "capacity": budget,
            "package_sha256_by_purpose": {purpose: packages[purpose]["package_sha256"]
                                           for purpose in ("income", "occupation_group")},
            "score_receipt_sha256": _sha(score_path)}
    if not return_artifacts:
        return result
    inner = roles["INNER_SELECTION"]
    artifacts = {
        "row_id": inner["row_id"], "unit": inner["unit"],
        "income": inner["income"], "occupation": inner["occupation"],
        "sex": inner["sex"],
        "prob_1": role_bundles["INNER_SELECTION"]["wires"]["income"]["probabilities"],
        "prob_2": role_bundles["INNER_SELECTION"]["wires"]["occupation_group"]["probabilities"],
    }
    for view in VIEWS:
        for reader in READERS:
            artifacts[f"reader_{view}__{reader}"] = readers[view][reader]
    return result, artifacts

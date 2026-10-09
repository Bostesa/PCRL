"""Fresh ATT-only complete-view readers, canonical raw transport and probes.

Four shared readers and two engineering controls are fitted for each actual
recipient/coalition view. The two RAW_ONLY readers are fitted once per exact
canonical union and verified here without another fit. Coalition ignore-one
recipient readers are exact local prediction aliases. No INNER SEX enters.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from results.pcrl_utility_reference_repair_v1.r1.role_store import (
    array_sha, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r2.control_fit import (
    FILES as CONTROL_FILES, fit_control_models,
)


REPO = Path(__file__).resolve().parents[3]
M4 = REPO / "results/pcrl_capacity_matched_feature_selection_v1/m4"
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
for source in (M4, SHARED):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
from attack_runner import _transport_scores  # noqa: E402
from attack_eval import (attack_fit_tune_split, fit_slate, oriented_auc,
                         save_fitted_slate)  # noqa: E402


CORE = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")
RAW = ("RAW_LR_C1", "RAW_HGB_15x200")
VIEWS = ("income", "occupation_group", "coalition")
SCHEMA = "pcrl-utility-complete-att-fit-v1"


def expected_outputs(output: Path) -> set[str]:
    parent = output.parent
    names = {"ATTACK_FIT_INDEX.json", "ALL_READER_PREDICTIONS.npz",
             "ALL_READER_TUNE_PREDICTIONS.npz"}
    names.update("core/" + item for item in (
        "READERS_RECEIPT.json", "view_att.npz", "view_inner.npz",
        "predictions_tune.npz", "predictions_inner.npz"))
    names.update(f"core/model__{reader}.joblib" for reader in CORE)
    names.update("controls/" + item for item in CONTROL_FILES)
    return {str(parent / name) for name in names}


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return sha_file(path)


def _write_once_json(path: Path, content: dict) -> str:
    with path.open("x") as stream:
        json.dump(content, stream, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return sha_file(path)


def _pair_sha(view: object) -> str:
    return array_sha(np.column_stack((view.row_id, view.unit)))


def _validated_local_aliases(
    local: str, index_path: Path, expected_sha: str, *,
    seed: int, pair_id: str, release_sha: str,
    att_view: object, inner_view: object,
    coalition_att: object, coalition_inner: object,
    lock_sha: str, manifest_sha: str,
) -> tuple[dict, dict, dict]:
    if (local not in VIEWS[:2] or sha_file(index_path) != expected_sha
            or _pair_sha(att_view) != _pair_sha(coalition_att)
            or _pair_sha(inner_view) != _pair_sha(coalition_inner)):
        raise PermissionError("ignore-recipient local identity differs")
    index = json.loads(index_path.read_text())
    if (index.get("schema") != SCHEMA or index.get("view_name") != local
            or index.get("seed") != seed or index.get("pair_id") != pair_id
            or index.get("release_index_sha256") != release_sha
            or index.get("lock_sha256") != lock_sha
            or index.get("manifest_sha256") != manifest_sha
            or index.get("att_view_content_sha256") != att_view.content_hashes()
            or index.get("inner_view_content_sha256") != inner_view.content_hashes()
            or index.get("att_row_unit_sha256") != _pair_sha(att_view)
            or index.get("inner_row_unit_sha256") != _pair_sha(inner_view)
            or index.get("reader_order") != list(CORE + RAW)):
        raise PermissionError("ignore-recipient local reader source differs")
    outputs = {}
    for role, filename, sha_key in (
        ("inner", "ALL_READER_PREDICTIONS.npz", "all_prediction_sha256"),
        ("tune", "ALL_READER_TUNE_PREDICTIONS.npz", "all_tune_sha256"),
    ):
        path = index_path.parent / filename
        if sha_file(path) != index[sha_key]:
            raise PermissionError("ignore-recipient local scores changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(CORE + RAW):
                raise PermissionError("ignore-recipient local reader set differs")
            outputs[role] = {name: np.asarray(archive[name], dtype=np.float64)
                             for name in CORE + RAW}
    return outputs["tune"], outputs["inner"], {
        "source_view": local, "source_index_sha256": expected_sha,
        "new_fit_count": 0,
    }


def fit_complete_attack_view(
    *, att_view: object, sex_att: Mapping[str, object],
    inner_view: object, att_union: object, inner_union: object,
    raw_index_path: Path, raw_index_sha256: str,
    release_index_sha256: str, seed: int, pair_id: str, view_name: str,
    lock_sha256: str, manifest_sha256: str, private_output: Path,
    local_dependencies: Mapping[str, tuple] | None = None,
) -> dict:
    """Fit, orient on ATT tune, and save predictions without INNER labels."""
    if (view_name not in VIEWS or seed not in (0, 1, 2)
            or (view_name == "coalition") != (local_dependencies is not None)):
        raise ValueError("complete attack view/dependency identity differs")
    out = Path(private_output).resolve()
    if ".private" not in out.parts or out.exists():
        raise PermissionError("complete attack output must be new private custody")
    if (not np.array_equal(att_view.x_raw_only, att_union.x)
            or not np.array_equal(inner_view.x_raw_only, inner_union.x)):
        raise PermissionError("RAW_ONLY source differs from complete release")
    raw, diagnostics = _transport_scores(
        raw_index_path=Path(raw_index_path),
        expected_raw_index_sha256=raw_index_sha256,
        att_union=att_union, inner_union=inner_union,
        att_view=att_view, inner_view=inner_view)
    fitted = fit_slate(att_view, dict(sex_att), inner_view,
                       required_readers=CORE, attacker_seed=seed)
    split = attack_fit_tune_split(att_view.unit)
    y_tune = np.asarray(sex_att["sex"], dtype=np.int64)[split["tune"]]
    joined = {**fitted,
              "predictions_tune": dict(fitted["predictions_tune"]),
              "predictions_inner": dict(fitted["predictions_inner"]),
              "validation": dict(fitted["validation"]),
              "reader_order": list(fitted["reader_order"])}
    for reader in RAW:
        joined["predictions_tune"][reader] = raw[reader]["tune"]
        joined["predictions_inner"][reader] = raw[reader]["inner"]
        validation = oriented_auc(y_tune, raw[reader]["tune"])
        validation["selected_orientation"] = (
            "positive" if validation["positive_alias_auc"] >=
            validation["reverse_alias_auc"] else "reverse")
        joined["validation"][reader] = validation
        joined["reader_order"].append(reader)
    aliases = {}
    if local_dependencies is not None:
        if set(local_dependencies) != set(VIEWS[:2]):
            raise ValueError("coalition ignore-recipient dependencies incomplete")
        for local, ignored in (("income", "R2"),
                               ("occupation_group", "R1")):
            path, sha, att_local, inner_local = local_dependencies[local]
            tune, inner, custody = _validated_local_aliases(
                local, Path(path), sha, seed=seed, pair_id=pair_id,
                release_sha=release_index_sha256,
                att_view=att_local, inner_view=inner_local,
                coalition_att=att_view, coalition_inner=inner_view,
                lock_sha=lock_sha256, manifest_sha=manifest_sha256)
            for reader in CORE + RAW:
                alias = f"IGNORE_{ignored}__{reader}"
                joined["predictions_tune"][alias] = tune[reader]
                joined["predictions_inner"][alias] = inner[reader]
                validation = oriented_auc(y_tune, tune[reader])
                validation["selected_orientation"] = (
                    "positive" if validation["positive_alias_auc"] >=
                    validation["reverse_alias_auc"] else "reverse")
                joined["validation"][alias] = validation
                joined["reader_order"].append(alias)
            aliases[local] = custody
    winner, best = None, -1.0
    for reader in joined["reader_order"]:
        value = joined["validation"][reader]["fixed_alias_max_auc"]
        if value > best + 1e-12:
            winner, best = reader, value
    joined["validation_winner"] = winner
    joined["validation_winner_orientation"] = joined["validation"][winner][
        "selected_orientation"]
    out.mkdir(parents=True, mode=0o700)
    core = save_fitted_slate(fitted, out / "core", att_view=att_view,
                             inner_view=inner_view)
    controls = fit_control_models(att_view, sex_att, inner_view,
                                  seed=seed, output_dir=out / "controls")
    all_prediction_sha = _atomic_npz(
        out / "ALL_READER_PREDICTIONS.npz", joined["predictions_inner"])
    all_tune_sha = _atomic_npz(
        out / "ALL_READER_TUNE_PREDICTIONS.npz", joined["predictions_tune"])
    index = {
        "schema": SCHEMA, "status": "FROZEN", "seed": seed,
        "pair_id": pair_id, "view_name": view_name,
        "lock_sha256": lock_sha256,
        "manifest_sha256": manifest_sha256,
        "release_index_sha256": release_index_sha256,
        "raw_index_sha256": raw_index_sha256,
        "att_row_unit_sha256": _pair_sha(att_view),
        "inner_row_unit_sha256": _pair_sha(inner_view),
        "att_view_content_sha256": att_view.content_hashes(),
        "inner_view_content_sha256": inner_view.content_hashes(),
        "attack_split_sha256": split["tune_group_id_sha256"],
        "reader_order": joined["reader_order"],
        "validation": joined["validation"],
        "validation_winner": winner,
        "validation_winner_orientation": joined[
            "validation_winner_orientation"],
        "core_readers_receipt_sha256": core["private_receipt_sha256"],
        "control_fit_receipt_sha256": sha_file(
            out / "controls/CONTROL_FIT_RECEIPT.json"),
        "control_fit": {key: controls[key] for key in (
            "ATT_shuffle_salt", "INNER_shuffle_salt_for_later_score",
            "fresh_fit_count", "INNER_SEX_read")},
        "raw_transport_diagnostics": diagnostics,
        "ignore_recipient_aliases": aliases,
        "all_prediction_sha256": all_prediction_sha,
        "all_tune_sha256": all_tune_sha,
        "fresh_fit_count": 6,
        "transported_raw_reader_count": 2,
        "INNER_SEX_read": False,
    }
    _write_once_json(out / "ATTACK_FIT_INDEX.json", index)
    return index

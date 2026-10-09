"""Sealed independent replay of selected complete-view SEX recovery.

All fitted models, output vectors and public release wires are frozen before
this stage. R3 computes ATT tuning and INNER endpoints with its own rank
arithmetic and never imports R2 scoring, selection or deployment modules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r3.privacy_math import (
    attack_tune_indices,
)
from results.pcrl_utility_reference_repair_v1.r3.privacy_replay import (
    check_ignore_aliases, control_book, reader_book,
)
from results.pcrl_utility_reference_repair_v1.r3.source_reference import file_sha
from results.pcrl_utility_reference_repair_v1.r3.task_replay import array_sha


STUDY = Path(__file__).resolve().parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
VIEWS = ("income", "occupation_group", "coalition")
CORE = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")
RAW = ("RAW_LR_C1", "RAW_HGB_15x200")
LOCAL_READERS = CORE + RAW
TOL = 1e-12


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise PermissionError(message)


def _require(spec: dict, path: Path) -> str:
    real = path.resolve(strict=True)
    expected = spec.get("inputs", {}).get(str(real))
    if expected is None or file_sha(real) != expected:
        raise PermissionError("privacy replay input absent or altered: " + str(path))
    return expected


def _json(spec: dict, path: Path) -> dict:
    _require(spec, path)
    value = json.loads(path.read_text())
    _check(isinstance(value, dict), "privacy replay JSON is not an object")
    return value


def _npz(spec: dict, path: Path, fields: set[str] | None = None) -> dict:
    _require(spec, path)
    with np.load(path, allow_pickle=False) as archive:
        if fields is not None and set(archive.files) != fields:
            raise PermissionError("privacy replay NPZ member inventory differs")
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _sex(spec: dict, export: Path, manifest: dict, role: str) -> dict:
    path = export / role / "sex.npz"
    key = f"{role}:sex"
    _check(_require(spec, path) == manifest["role_file_sha256"][key],
           "privacy replay SEX role SHA differs")
    data = _npz(spec, path, {"row_id", "unit", "sex"})
    row, unit, label = data["row_id"], data["unit"], data["sex"]
    meta = manifest["roles"][role]
    _check(row.dtype == np.int64 and unit.dtype == np.int64 and
           label.dtype == np.int64 and row.shape == (meta["rows"],) and
           unit.shape == row.shape and label.shape == row.shape and
           len(np.unique(row)) == len(row) and
           len(np.unique(unit)) == meta["groups"] and
           np.all(np.diff(row) > 0) and np.isin(label, (0, 1)).all() and
           np.unique(label).tolist() == [0, 1] and
           array_sha(np.column_stack((row, unit))) ==
           meta["row_unit_pair_sha256"],
           "privacy replay SEX role person/label geometry differs")
    return data


def _metric_same(own: float, saved: float, label: str) -> float:
    a, b = float(own), float(saved)
    difference = abs(a - b)
    if not math.isfinite(difference) or difference > TOL:
        raise PermissionError(label + " independently recomputed value differs")
    return difference


def _semantic_bank_sha(spec: dict, root: Path, seed: int,
                       audit: dict, gate_sha: str) -> str:
    """Bind the bank's semantic digest to R3's frozen q13 receipt audit.

    The public MASK_BANK file SHA is separately bound by the q21 science
    lock. Its semantic SharedBank digest is a distinct quantity.
    """
    path = root / "pair_select" / f"seed{seed}" / "PAIR_SELECT_RECEIPT.json"
    expected = audit["by_seed"][str(seed)]["pair_receipt_sha256"]
    _check(_require(spec, path) == expected,
           "q13 pair-selection receipt differs from independent audit")
    receipt = _json(spec, path)
    digest = receipt.get("bank_sha256")
    _check(receipt.get("schema") == "pcrl-utility-pair-selection-receipt-v1"
           and receipt.get("seed") == seed
           and receipt.get("full_input_gate_sha256") == gate_sha
           and isinstance(digest, str) and len(digest) == 64
           and all(char in "0123456789abcdef" for char in digest),
           "q13 semantic bank identity differs")
    return digest


def _view_equals_release(spec: dict, root: Path, seed: int, pair_id: str,
                         view: str, role: str, frozen: dict) -> None:
    release = root / "releases" / f"seed{seed}" / pair_id / f"{role}.npz"
    saved = _npz(spec, release)
    for field in ("row_id", "unit", "x", "x_defense_aware", "x_raw_only"):
        key = f"{view}__view__{field}"
        _check(key in saved and field in frozen and
               np.array_equal(saved[key], frozen[field]),
               "fitted ATT view differs from actual complete release: " + key)


def _read_reader_files(spec: dict, directory: Path, index: dict,
                       expected_readers: set[str]) -> tuple[dict, dict, dict, dict]:
    inner_path = directory / "ALL_READER_PREDICTIONS.npz"
    tune_path = directory / "ALL_READER_TUNE_PREDICTIONS.npz"
    _check(_require(spec, inner_path) == index["all_prediction_sha256"] and
           _require(spec, tune_path) == index["all_tune_sha256"],
           "reader prediction files differ from frozen index")
    inner = _npz(spec, inner_path, expected_readers)
    tune = _npz(spec, tune_path, expected_readers)
    core_path = directory / "core/READERS_RECEIPT.json"
    _check(_require(spec, core_path) == index["core_readers_receipt_sha256"],
           "core reader receipt differs")
    core = _json(spec, core_path)
    _check(core.get("schema") == "pcrl-portfolio-private-reader-custody-v1" and
           core.get("status") == "FROZEN" and
           set(core.get("files", {})) == {
               *(f"model__{name}.joblib" for name in CORE),
               "view_att.npz", "view_inner.npz", "predictions_tune.npz",
               "predictions_inner.npz"},
           "core reader receipt inventory differs")
    for name, digest in core["files"].items():
        _check(_require(spec, directory / "core" / name) == digest,
               "core reader model/view/prediction bytes differ")
    view_att = _npz(spec, directory / "core/view_att.npz",
                    {"row_id", "unit", "x", "x_defense_aware", "x_raw_only"})
    view_inner = _npz(spec, directory / "core/view_inner.npz",
                      {"row_id", "unit", "x", "x_defense_aware", "x_raw_only"})
    core_tune = _npz(spec, directory / "core/predictions_tune.npz", set(CORE))
    core_inner = _npz(spec, directory / "core/predictions_inner.npz", set(CORE))
    for name in CORE:
        _check(np.array_equal(core_tune[name], tune[name]) and
               np.array_equal(core_inner[name], inner[name]),
               "core reader predictions differ from complete slate")
    return tune, inner, view_att, view_inner


def _reforward_core_models(spec: dict, directory: Path,
                           att_view: dict, inner_view: dict,
                           tune_idx: np.ndarray, tune: dict,
                           inner: dict) -> float:
    """Check that saved core scores come from the frozen fitted estimators."""
    maximum = 0.0
    for name in CORE:
        model_path = directory / "core" / f"model__{name}.joblib"
        _require(spec, model_path)
        model = joblib.load(model_path)
        key = "x_defense_aware" if name == "DA_MLP_64x32" else "x"
        for label, features, target in (
                ("tune", att_view[key][tune_idx], tune[name]),
                ("inner", inner_view[key], inner[name])):
            replayed = np.asarray(model.predict_proba(features)[:, 1],
                                  dtype=np.float64)
            _check(replayed.shape == target.shape and
                   np.isfinite(replayed).all() and
                   np.all((replayed >= 0) & (replayed <= 1)),
                   "core model produced invalid " + label + " predictions")
            delta = float(np.max(np.abs(replayed - target)))
            _check(delta <= TOL,
                   "frozen core model does not reproduce " + name + " " + label)
            maximum = max(maximum, delta)
    return maximum


def _replay_controls(spec: dict, directory: Path, index: dict,
                     y_inner: np.ndarray, group_inner: np.ndarray,
                     x_inner: np.ndarray, reported: dict) -> tuple[dict, float]:
    folder = directory / "controls"
    receipt_path = folder / "CONTROL_FIT_RECEIPT.json"
    _check(_require(spec, receipt_path) == index["control_fit_receipt_sha256"],
           "ATT control fit receipt differs")
    receipt = _json(spec, receipt_path)
    _check(receipt.get("schema") == "pcrl-utility-att-control-fit-v1" and
           receipt.get("fresh_fit_count") == 2 and
           receipt.get("INNER_SEX_read") is False and
           receipt.get("inner_row_unit_sha256") ==
           index["inner_row_unit_sha256"] and
           receipt.get("inner_view_content_sha256") ==
           index["inner_view_content_sha256"] and
           receipt.get("INNER_shuffle_salt_for_later_score") ==
           "pcrl-method-portfolio-v1|independent-inner-sex-null|20261008|v2" and
           set(receipt.get("files_sha256", {})) == {
               "shuffled_model.joblib", "planted_model.joblib",
               "shuffled_inner.npy"},
           "ATT control contract differs")
    for name, digest in receipt["files_sha256"].items():
        _check(_require(spec, folder / name) == digest,
               "ATT control file differs from receipt")
    shuffled = np.load(folder / "shuffled_inner.npy", allow_pickle=False)
    _check(shuffled.dtype == np.float64 and shuffled.shape == y_inner.shape and
           array_sha(shuffled) == receipt["shuffled_prediction_array_sha256"],
           "ATT shuffled control saved vector differs")
    shuffled_model = joblib.load(folder / "shuffled_model.joblib")
    repl = np.asarray(shuffled_model.predict_proba(x_inner)[:, 1], dtype=np.float64)
    _check(np.array_equal(shuffled, repl),
           "ATT shuffled model prediction differs from saved vector")
    planted_model = joblib.load(folder / "planted_model.joblib")
    planted = np.asarray(planted_model.predict_proba(
        np.column_stack((x_inner, y_inner)))[:, 1], dtype=np.float64)
    own = control_book(y_inner, group_inner, shuffled, planted)
    delta = 0.0
    for key in ("null_auc", "shuffled_att_true_inner_auc",
                "shuffled_att_independent_inner_auc", "planted_auc"):
        delta = max(delta, _metric_same(own[key], reported[key], "control " + key))
    _check(own["gates"] == reported["gates"] and
           own["status"] == reported["status"] and
           reported["control_fit_receipt_sha256"] ==
           index["control_fit_receipt_sha256"] and
           reported["planted_prediction_sha256"] == array_sha(planted),
           "independent control gate or identity differs")
    return own, delta


def _one_view(spec: dict, root: Path, seed: int, pair_id: str, view: str,
              index_lock_sha: str, local_lock_sha: str,
              release_sha: str, manifest_sha: str,
              att_sex: dict, inner_sex: dict, reported: dict) -> tuple[dict, dict]:
    folder = root / "attacks" / f"seed{seed}" / pair_id / view
    index_path = folder / "ATTACK_FIT_INDEX.json"
    index_sha = _require(spec, index_path)
    index = _json(spec, index_path)
    expected_readers = set(LOCAL_READERS)
    if view == "coalition":
        expected_readers |= {f"IGNORE_{ignored}__{reader}"
                             for ignored in ("R1", "R2")
                             for reader in LOCAL_READERS}
    _check(index.get("schema") == "pcrl-utility-complete-att-fit-v1" and
           index.get("status") == "FROZEN" and
           index.get("seed") == seed and index.get("pair_id") == pair_id and
           index.get("view_name") == view and
           index.get("lock_sha256") == index_lock_sha and
           index.get("local_attack_source_lock_sha256") ==
           (None if view != "coalition" else local_lock_sha) and
           index.get("manifest_sha256") == manifest_sha and
           index.get("release_index_sha256") == release_sha and
           index.get("fresh_fit_count") == 6 and
           index.get("INNER_SEX_read") is False and
           set(index.get("reader_order", [])) == expected_readers and
           len(index["reader_order"]) == len(expected_readers),
           "fresh complete-view reader index differs")
    tune, inner, att_view, inner_view = _read_reader_files(
        spec, folder, index, expected_readers)
    for role, frozen, sex in (("ATTACK_FIT_NEW", att_view, att_sex),
                              ("INNER_SELECTION", inner_view, inner_sex)):
        _check(np.array_equal(frozen["row_id"], sex["row_id"]) and
               np.array_equal(frozen["unit"], sex["unit"]) and
               array_sha(np.column_stack((frozen["row_id"], frozen["unit"]))) ==
               index["att_row_unit_sha256" if role == "ATTACK_FIT_NEW"
                     else "inner_row_unit_sha256"],
               "fitted complete-view person alignment differs")
        _view_equals_release(spec, root, seed, pair_id, view, role, frozen)
    fit_idx, tune_idx = attack_tune_indices(att_view["unit"])
    _check(len(fit_idx) + len(tune_idx) == len(att_view["unit"]) and
           array_sha(np.sort(np.unique(att_view["unit"][tune_idx]))) ==
           index["attack_split_sha256"],
           "ATT reader tuning split differs")
    core_model_delta = _reforward_core_models(
        spec, folder, att_view, inner_view, tune_idx, tune, inner)
    y_tune = att_sex["sex"][tune_idx]
    y_inner = inner_sex["sex"]
    own = reader_book(y_tune, tune, y_inner, inner,
                      tuple(index["reader_order"]))
    _check(set(index["validation"]) == expected_readers and
           index["validation_winner"] == own["validation_winner"] and
           index["validation_winner_orientation"] ==
           own["validation_winner_orientation"],
           "frozen ATT validation selection differs")
    delta = 0.0
    for name in expected_readers:
        source = index["validation"][name]
        _check(source["selected_orientation"] ==
               own["validation"][name]["selected_orientation"],
               "ATT validation orientation differs")
        for key in ("positive_alias_auc", "reverse_alias_auc",
                    "fixed_alias_max_auc"):
            delta = max(delta, _metric_same(
                own["validation"][name][key], source[key],
                "ATT validation " + name + " " + key))
    _check(reported.get("schema") == "pcrl-utility-complete-privacy-score-v1" and
           reported.get("seed") == seed and reported.get("pair_id") == pair_id and
           reported.get("view_name") == view and
           reported.get("attack_index_sha256") == index_sha and
           reported.get("inner_row_unit_sha256") ==
           index["inner_row_unit_sha256"] and
           reported.get("reader_order") == index["reader_order"] and
           reported.get("validation_winner") == own["validation_winner"] and
           reported.get("validation_winner_orientation") ==
           own["validation_winner_orientation"] and
           set(reported.get("readers", {})) == expected_readers,
           "R2 reported reader inventory or winner differs")
    for name in expected_readers:
        for key in ("positive_alias_auc", "reverse_alias_auc",
                    "fixed_alias_max_auc"):
            delta = max(delta, _metric_same(
                own["readers"][name][key], reported["readers"][name][key],
                "INNER reader " + name + " " + key))
    for key in ("primary_envelope_auc", "validation_winner_inner_auc"):
        delta = max(delta, _metric_same(own[key], reported[key], key))
    controls, control_delta = _replay_controls(
        spec, folder, index, y_inner, inner_sex["unit"],
        inner_view["x"], reported["controls"])
    delta = max(delta, control_delta)
    summary = {
        "reader_count": len(expected_readers),
        "validation_winner": own["validation_winner"],
        "validation_winner_orientation": own["validation_winner_orientation"],
        "validation_winner_inner_auc": own["validation_winner_inner_auc"],
        "primary_envelope_auc": own["primary_envelope_auc"],
        "controls_status": controls["status"],
        "max_abs_core_model_reforward_difference": core_model_delta,
        "max_abs_reported_metric_difference": delta,
    }
    return summary, {"tune": tune, "inner": inner}


def replay(spec: dict, root: Path, export: Path, manifest_path: Path,
           gate_path: Path, bank_path: Path, audit_path: Path,
           release_verification_path: Path, lock: dict) -> dict:
    manifest = _json(spec, manifest_path)
    gate = _json(spec, gate_path)
    bank = _json(spec, bank_path)
    audit = _json(spec, audit_path)
    release_verification = _json(spec, release_verification_path)
    _check(lock.get("source_map_sha256") == file_sha(manifest_path) and
           lock.get("full_input_gate_sha256") == file_sha(gate_path) and
           lock.get("mask_bank_sha256") == file_sha(bank_path) and
           lock.get("pair_selection_audit_sha256") == file_sha(audit_path) and
           lock.get("independent_release_verification_sha256") ==
           file_sha(release_verification_path) and
           gate.get("status") == "PASS_FULL_INPUT_POINT_GATE" and
           audit.get("status") == "PASS_AGGREGATE_SELECTION_REPLAY" and
           release_verification.get("status") ==
           "PASS_INDEPENDENT_COMPLETE_VIEW_REPLAY",
           "privacy replay gate/bank/audit/release verification differs")
    selected = {str(seed): sorted({name for name in audit["by_seed"][str(seed)][
        "selected_rule_pair_ids"].values() if name is not None})
        for seed in (0, 1)}
    _check(lock.get("selected_pairs_by_seed") == selected,
           "privacy replay selected pairs differ from frozen rules")
    att_sex = _sex(spec, export, manifest, "ATTACK_FIT_NEW")
    inner_sex = _sex(spec, export, manifest, "INNER_SELECTION")
    expected_lock = {
        "income": lock["local_attack_source_lock_sha256"],
        "occupation_group": lock["local_attack_source_lock_sha256"],
        "coalition": lock["coalition_attack_source_lock_sha256"],
    }
    maxima = 0.0
    by_seed: dict[str, dict] = {}
    all_controls_pass = True
    for seed in (0, 1):
        semantic_bank_sha = _semantic_bank_sha(
            spec, root, seed, audit, file_sha(gate_path))
        score_dir = root / "privacy_score" / f"seed{seed}"
        score_path = score_dir / "PRIVACY_SCORES.json"
        receipt = _json(spec, score_dir / "PRIVACY_SCORE_RECEIPT.json")
        scores = _json(spec, score_path)
        _check(receipt.get("schema") ==
               "pcrl-utility-seed-privacy-score-receipt-v1" and
               receipt.get("seed") == seed and
               receipt.get("conditional_lock_sha256") ==
               lock["privacy_score_source_lock_sha256"] and
               receipt.get("score_file_sha256") == file_sha(score_path) and
               receipt.get("selected_pair_count") == len(selected[str(seed)]) and
               receipt.get("complete_view_count") == 3 * len(selected[str(seed)]) and
               receipt.get("pair_selection_receipt_sha256") ==
               audit["by_seed"][str(seed)]["pair_receipt_sha256"] and
               scores.get("schema") == "pcrl-utility-seed-privacy-scores-v1" and
               scores.get("seed") == seed and
               scores.get("bank_sha256") == semantic_bank_sha and
               scores.get("full_input_gate_sha256") == file_sha(gate_path) and
               scores.get("pair_selection_receipt_sha256") ==
               audit["by_seed"][str(seed)]["pair_receipt_sha256"] and
               scores.get("attack_source_lock_sha256_by_view") == expected_lock and
               sorted(scores.get("pairs", {})) == selected[str(seed)] and
               scores.get("selected_rule_pair_ids") ==
               audit["by_seed"][str(seed)]["selected_rule_pair_ids"],
               "frozen R2 privacy score seed inventory differs")
        by_seed[str(seed)] = {}
        for pair_id in selected[str(seed)]:
            release_path = root / "releases" / f"seed{seed}" / pair_id / "RELEASE_RECEIPT.json"
            release_sha = _require(spec, release_path)
            release = _json(spec, release_path)
            _check(release.get("schema") ==
                   "pcrl-utility-conditional-complete-release-v1" and
                   release.get("pair_id") == pair_id and release.get("seed") == seed and
                   release.get("lock_sha256") ==
                   lock["release_source_lock_sha256"],
                   "frozen selected release identity differs")
            view_data = {}
            vectors = {}
            for view in VIEWS:
                summary, saved = _one_view(
                    spec, root, seed, pair_id, view, expected_lock[view],
                    expected_lock["income"],
                    release_sha, file_sha(manifest_path), att_sex, inner_sex,
                    scores["pairs"][pair_id][view])
                view_data[view] = summary
                vectors[view] = saved
                maxima = max(maxima, summary["max_abs_reported_metric_difference"])
                all_controls_pass &= summary["controls_status"] == "PASS"
            for role in ("tune", "inner"):
                check_ignore_aliases(
                    vectors["coalition"][role],
                    {name: vectors[name][role] for name in VIEWS[:2]})
            _check(view_data["coalition"]["primary_envelope_auc"] + TOL >=
                   max(view_data[name]["primary_envelope_auc"]
                       for name in VIEWS[:2]),
                   "coalition envelope cannot ignore a recipient")
            by_seed[str(seed)][pair_id] = view_data
        _check(receipt.get("status") ==
               ("SCORED" if all(view_data["controls_status"] == "PASS"
                for pair in by_seed[str(seed)].values()
                for view_data in pair.values()) else "CONTROL_FAILED"),
               "R2 score receipt control status differs")
    return {
        "schema": "pcrl-utility-r3-independent-privacy-replay-v1",
        "status": "REPLAYED" if all_controls_pass else "CONTROL_FAILED_REPLAYED",
        "scope": "selected exploratory reused Adult INNER SEX recovery",
        "role_access": ["ATTACK_FIT_NEW:sex", "INNER_SELECTION:sex"],
        "selected_pairs_by_seed": selected,
        "full_input_gate_sha256": file_sha(gate_path),
        "pair_selection_audit_sha256": file_sha(audit_path),
        "independent_release_verification_sha256":
            file_sha(release_verification_path),
        "max_abs_reported_metric_difference": maxima,
        "all_control_gates_pass": all_controls_pass,
        "by_seed_pair_view": by_seed,
        "interpretation_limit": "post-selection development recovery on repeatedly used Adult people, not confirmation",
    }


def _write_once(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _expected_output(root: Path, unit_id: str) -> Path:
    names = {"R3_PRIVACY_REPLAY": "PRIVACY_REPLAY.json",
             "R3_PRIVACY_REPLAY_RETRY": "PRIVACY_REPLAY_RETRY.json"}
    if unit_id not in names:
        raise PermissionError("independent privacy replay unit identity differs")
    return root / "r3" / names[unit_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "queue-root", "lock", "manifest", "export-root",
                 "private-root", "full-input-gate", "mask-bank",
                 "pair-selection-audit", "release-verification", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "r3_privacy_replay", output)
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("independent privacy replay requires one thread")
    root = Path(args.private_root).resolve(strict=True)
    if (root != STUDY / ".private" or
            output.resolve() != _expected_output(root, args.unit_id) or
            spec["outputs"] != [str(output)] or
            set(spec.get("role_inputs", {})) !=
            {"ATTACK_FIT_NEW:sex", "INNER_SELECTION:sex"}):
        raise PermissionError("independent privacy replay role/output contract differs")
    lock = json.loads(Path(args.lock).read_text())
    result = replay(spec, root, Path(args.export_root), Path(args.manifest),
                    Path(args.full_input_gate), Path(args.mask_bank),
                    Path(args.pair_selection_audit),
                    Path(args.release_verification), lock)
    _write_once(output, result)


if __name__ == "__main__":
    main()

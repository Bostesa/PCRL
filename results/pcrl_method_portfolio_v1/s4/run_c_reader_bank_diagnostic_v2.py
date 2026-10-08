"""One sealed S4 diagnostic of all six frozen C FULL:s0/r1 readers.

Only aggregate numeric differences and array layouts leave this process.  This
does not amend the frozen C replay's exact-prediction requirement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from replay_c_readers_v1 import READERS, fixed_split, reader_matrix
from replay_c_release_v1 import load_packages, predict_release
from replay_c_values_v1 import load_saved_bundle, sha_array
from run_c_adult_replay_v1 import (
    CAMPAIGN, HERE, QUEUE_ROOT, REPO, _git_blob, _need, _one_argument,
    _parent_output, _private_root, _queue_root, _read, _role_file, _sha,
    _write_once,
)


ADMISSION = CAMPAIGN / "S4_C_READER_BANK_DIAGNOSTIC_V2_ADMISSION.json"
REVIEW = CAMPAIGN / "S4_C_READER_BANK_DIAGNOSTIC_V2_PREFIT_REVIEW.json"
SEAL = CAMPAIGN / "S4_C_READER_BANK_DIAGNOSTIC_V2_LOCAL_SEAL_RECEIPT.json"
HISTORICAL_ADMISSION = CAMPAIGN / "S4_C_REPLAY_ADMISSION.json"
HISTORICAL_REVIEW = CAMPAIGN / "S4_C_REPLAY_PREFIT_REVIEW.json"
HISTORICAL_SEAL = CAMPAIGN / "S4_C_REPLAY_LOCAL_SEAL_RECEIPT.json"
PRIOR_ADMISSION = CAMPAIGN / "S4_C_READER_DIAGNOSTIC_ADMISSION.json"
PRIOR_REVIEW = CAMPAIGN / "S4_C_READER_DIAGNOSTIC_PREFIT_REVIEW.json"
PRIOR_SEAL = CAMPAIGN / "S4_C_READER_DIAGNOSTIC_LOCAL_SEAL_RECEIPT.json"
PRIOR_UNIT_ID = "S4-C-R1-LR-INNER-DIAG-FULL-S0-R1"
UNIT_ID = "S4-C-R1-READER-BANK-DIAG-FULL-S0-R1-R2"
FAILED_UNIT_ID = "S4-C-R1-C-SIMPLE-FULL-S0"
PARENT_LABELS = frozenset(("fit", "forward", "reader_r1"))
SOURCE_FILES = frozenset(
    f"results/pcrl_method_portfolio_v1/s4/{stem}.py" for stem in (
        "run_c_reader_bank_diagnostic_v2", "run_c_adult_replay_v1",
        "replay_c_readers_v1", "replay_c_release_v1", "replay_c_values_v1"
    )
)


def matrix_layout(matrix: np.ndarray) -> dict:
    """Expose shape and storage layout, never matrix values."""
    array = np.asarray(matrix)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "strides_bytes": list(array.strides),
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
    }


def _average_ranks(value: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(value, kind="stable")
    sorted_value = value[order]
    starts = np.r_[0, np.flatnonzero(sorted_value[1:] != sorted_value[:-1]) + 1]
    stops = np.r_[starts[1:], len(value)]
    ranks = np.empty(len(value), dtype=np.float64)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
    return ranks, int(np.count_nonzero((stops - starts) > 1))


def compare_predictions(saved: np.ndarray, restored: np.ndarray) -> dict:
    """Compare bit patterns and AUC-determining average ranks without labels."""
    reference = np.asarray(saved, dtype=np.float64)
    actual = np.asarray(restored, dtype=np.float64)
    if (reference.ndim != 1 or reference.shape != actual.shape or
            len(reference) == 0 or not np.isfinite(reference).all() or
            not np.isfinite(actual).all() or
            np.any((reference < 0) | (reference > 1)) or
            np.any((actual < 0) | (actual > 1))):
        raise ValueError("C reader-bank diagnostic needs aligned finite probability vectors")
    different = reference.view(np.uint64) != actual.view(np.uint64)
    delta = np.abs(reference - actual)
    ranks_reference, ties_reference = _average_ranks(reference)
    ranks_actual, ties_actual = _average_ranks(actual)
    rank_changed = int(np.count_nonzero(ranks_reference != ranks_actual))
    return {
        "rows": int(len(reference)),
        "mismatch_count": int(np.count_nonzero(different)),
        "exact_equal": bool(not np.any(different)),
        "absolute_delta_max": float(np.max(delta)),
        "absolute_delta_median": float(np.median(delta)),
        "absolute_delta_p99": float(np.quantile(delta, 0.99)),
        "rank_changed_count": rank_changed,
        "tie_groups_saved": ties_reference,
        "tie_groups_restored": ties_actual,
        "all_binary_label_auc_invariant": bool(rank_changed == 0),
    }


def summarize_reader_bank(frozen: dict[str, np.ndarray],
                          replayed: dict[str, np.ndarray]) -> dict:
    """Find the first exact mismatch across the whole fixed slate."""
    if set(frozen) != set(READERS) or set(replayed) != set(READERS):
        raise ValueError("C diagnostic requires all six frozen readers")
    by_reader = {name: compare_predictions(frozen[name], replayed[name])
                 for name in READERS}
    mismatched = [name for name in READERS
                  if not by_reader[name]["exact_equal"]]
    return {"reader_order": list(READERS),
            "exact_mismatch_readers": mismatched,
            "first_mismatch_reader": mismatched[0] if mismatched else None,
            "by_reader": by_reader}


def _bound_json(private_root: Path, relative: str, digest: str) -> tuple[Path, dict]:
    _need(isinstance(relative, str) and not Path(relative).is_absolute(),
          "C reader-bank diagnostic private path is not relative")
    path = (private_root / relative).resolve(strict=True)
    _need(path.is_relative_to(private_root) and _sha(path) == digest,
          "C reader-bank diagnostic private JSON SHA differs")
    return path, _read(path)


def _admit(args: argparse.Namespace, state: dict) -> tuple[dict, dict, Path, dict]:
    """Verify the new Git seal and every frozen parent before a role read."""
    _need(args.admission.resolve() == ADMISSION and args.local_seal.resolve() == SEAL and
          _sha(ADMISSION) == args.admission_sha256 and
          _sha(SEAL) == args.local_seal_sha256,
          "C reader-bank diagnostic admission or seal file differs")
    admission, review, seal = _read(ADMISSION), _read(REVIEW), _read(SEAL)
    _need(admission.get("schema") == "pcrl-s4-c-reader-bank-diagnostic-admission-v2" and
          admission.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
          admission.get("adult_role_access") is True and
          admission.get("scope") == "S4_C_R1_SIX_READER_INNER_DIAGNOSTIC_V2" and
          review.get("schema") == "pcrl-s4-c-reader-bank-diagnostic-prefit-review-v2" and
          review.get("status") == "PASS" and
          review.get("scope") == admission["scope"] and
          review.get("admission_sha256") == args.admission_sha256 and
          seal.get("schema") == "pcrl-s4-c-reader-bank-diagnostic-local-seal-v2" and
          seal.get("status") == "PASS" and
          seal.get("admission_sha256") == args.admission_sha256 and
          seal.get("review_sha256") == _sha(REVIEW) and
          seal.get("admission_commit_sha") == review.get("admission_commit_sha"),
          "C reader-bank diagnostic review or seal scope differs")
    _git_blob(seal["admission_commit_sha"],
              "results/pcrl_method_portfolio_v1/S4_C_READER_BANK_DIAGNOSTIC_V2_ADMISSION.json",
              args.admission_sha256)
    _need(_sha(HISTORICAL_ADMISSION) == admission.get("historical_s4_admission_sha256") and
          _sha(HISTORICAL_REVIEW) == admission.get("historical_s4_review_sha256") and
          _sha(HISTORICAL_SEAL) == admission.get("historical_s4_seal_sha256"),
          "C reader-bank diagnostic historical S4 admission changed")
    _need(_sha(PRIOR_ADMISSION) == admission.get("prior_single_diagnostic_admission_sha256") and
          _sha(PRIOR_REVIEW) == admission.get("prior_single_diagnostic_review_sha256") and
          _sha(PRIOR_SEAL) == admission.get("prior_single_diagnostic_seal_sha256"),
          "C reader-bank diagnostic prior LR_C1 seal changed")
    prior_receipt_path = QUEUE_ROOT / "units" / PRIOR_UNIT_ID / "RECEIPT.json"
    prior_receipt = _read(prior_receipt_path)
    prior_report_path = HERE / ".private/c_r1_reader_diag_v1" / f"{PRIOR_UNIT_ID}.json"
    prior_report = _read(prior_report_path)
    _need(_sha(prior_receipt_path) == admission.get("prior_single_diagnostic_queue_receipt_sha256") and
          prior_receipt.get("status") == "done" and
          prior_receipt.get("returncode") == 0 and
          prior_receipt.get("output_sha256") == {
              str(prior_report_path): admission.get("prior_single_diagnostic_report_sha256")} and
          _sha(prior_report_path) == admission.get("prior_single_diagnostic_report_sha256") and
          prior_report.get("status") == "DIAGNOSTIC_RECORDED_NO_TOLERANCE_DECISION" and
          prior_report.get("unit_id") == PRIOR_UNIT_ID and
          prior_report.get("saved_vs_rebuilt_model_predictions", {}).get("exact_equal") is True,
          "C reader-bank diagnostic prior LR_C1 result changed")
    historical = _read(HISTORICAL_ADMISSION)
    _git_blob(historical["lock_commit_sha"],
              "results/pcrl_method_portfolio_v1/CAMPAIGN_LOCK.json",
              historical["lock_sha256"])
    fit_lock = json.loads(_git_blob(historical["lock_commit_sha"],
              "results/pcrl_method_portfolio_v1/CAMPAIGN_LOCK.json",
              historical["lock_sha256"]))
    _need(historical.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
          fit_lock.get("families", {}).get("C", {}).get("fit_admission") == "ALLOWED" and
          _sha(CAMPAIGN / "DATA_ACCESS_MANIFEST.json") ==
          historical.get("data_manifest_sha256") ==
          admission.get("data_manifest_sha256") and
          admission.get("historical_fit_lock_sha256") == historical["lock_sha256"] and
          admission.get("historical_fit_commit_sha") == historical["lock_commit_sha"],
          "C reader-bank diagnostic historical FIT or role manifest differs")
    sources = admission.get("source_sha256_by_path", {})
    _need(isinstance(sources, dict) and set(sources) == SOURCE_FILES and
          admission.get("queue_source_sha256") ==
          _sha(CAMPAIGN / "shared/compute_queue.py"),
          "C reader-bank diagnostic source or S3 queue set differs")
    for relative, digest in sources.items():
        source = REPO / relative
        _need(source.is_file() and _sha(source) == digest,
              "C reader-bank diagnostic source bytes changed")
    for relative, digest in historical["source_sha256_by_path"].items():
        _need(_sha(REPO / relative) == digest,
              "C historical S4 independent source changed")
    unit = admission.get("unit_specs", {}).get(UNIT_ID)
    _need(set(admission.get("unit_specs", {})) == {UNIT_ID} and
          isinstance(unit, dict) and unit.get("stage") ==
          "independent_c_r1_reader_bank_diagnostic" and
          unit.get("arm") == "C-SIMPLE-FULL" and unit.get("seed") == 0 and
          unit.get("view") == "r1" and unit.get("readers") == list(READERS) and
          unit.get("data_access") ==
          "S4_C_R1_INNER_FEATURES_SAVED_READER_BANK_DIAGNOSTIC" and
          unit.get("max_wall_s") == unit.get("cpu_reserve_s") == 120 and
          unit.get("max_rss_gib") == 1.5 and
          set(unit.get("parents", {})) == PARENT_LABELS,
          "C reader-bank diagnostic singleton unit declaration differs")
    failed_receipt = QUEUE_ROOT / "units" / FAILED_UNIT_ID / "RECEIPT.json"
    failed = _read(failed_receipt)
    _need(_sha(failed_receipt) == admission.get("failed_s4_queue_receipt_sha256") and
          failed.get("status") == "failed" and failed.get("returncode") == 1 and
          failed.get("output_sha256") == {} and
          state.get("units", {}).get(FAILED_UNIT_ID, {}).get("status") == "failed",
          "C reader-bank diagnostic original failed smoke receipt differs")
    running = state.get("units", {}).get(UNIT_ID, {})
    spec = running.get("spec", {})
    argv = spec.get("argv", [])
    output = args.private_report.resolve()
    canonical = hashlib.sha256(json.dumps(unit, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    _need(state.get("gate", {}).get("safe_inputs") is True and
          state.get("gate", {}).get("occupancy_confirmed") is True and
          state.get("gate", {}).get("manually_closed") is False and
          running.get("status") == "running" and
          running.get("child_pgid") == os.getpgrp() and
          spec.get("unit_id") == UNIT_ID and
          spec.get("mode") == "s4_c_reader_bank_diagnostic" and
          spec.get("team") == spec.get("family") == "S4" and
          spec.get("stage") == unit["stage"] and
          spec.get("arm") == spec.get("arm_id") == unit["arm"] and
          spec.get("seed") == unit["seed"] and
          spec.get("view") == unit["view"] and
          spec.get("readers") == unit["readers"] and
          spec.get("data_access") == unit["data_access"] and
          spec.get("deps") == [unit["parents"]["reader_r1"]["unit_id"]] and
          spec.get("max_wall_s") == unit["max_wall_s"] and
          spec.get("cpu_reserve_s") == unit["cpu_reserve_s"] and
          spec.get("max_rss_gib") == unit["max_rss_gib"] and
          spec.get("admission_unit_sha256") == canonical and
          spec.get("runner_sha256") == sources[
              "results/pcrl_method_portfolio_v1/s4/run_c_reader_bank_diagnostic_v2.py"] and
          spec.get("outputs") == [str(output)] and
          output.name == f"{UNIT_ID}.json" and
          output.parent == (HERE / ".private/c_r1_reader_bank_diag_v2").resolve() and
          not output.exists() and
          len(argv) == 16 and Path(argv[0]).resolve() == Path(sys.executable).resolve() and
          Path(argv[1]).resolve() == Path(__file__).resolve() and
          argv[1:] == sys.argv and
          _one_argument(argv, "--unit-id") == UNIT_ID and
          _one_argument(argv, "--admission-sha256") == args.admission_sha256 and
          _one_argument(argv, "--local-seal-sha256") == args.local_seal_sha256 and
          Path(_one_argument(argv, "--private-report")).resolve() == output and
          Path(_one_argument(argv, "--queue-root")).resolve() == QUEUE_ROOT.resolve(),
          "C reader-bank diagnostic S3 child, argv or write-once output differs")
    historical_unit = historical["unit_specs"][FAILED_UNIT_ID]
    _need({name: unit["parents"][name] for name in PARENT_LABELS} ==
          {name: historical_unit["parents"][name] for name in PARENT_LABELS},
          "C reader-bank diagnostic parents differ from failed S4 arm")
    private_roots, outputs = set(), {}
    for label, entry in unit["parents"].items():
        path, private = _parent_output(QUEUE_ROOT, state, entry, fit_lock,
                                        0, "C-SIMPLE-FULL", label)
        outputs[label] = path
        private_roots.add(private)
    _need(len(private_roots) == 1,
          "C reader-bank diagnostic parent outputs cross private roots")
    return admission, unit, next(iter(private_roots)), outputs


def _npz(path: Path, expected_sha: str, members: set[str]) -> dict[str, np.ndarray]:
    _need(_sha(path) == expected_sha, "C reader-bank diagnostic S2 saved file SHA differs")
    with np.load(path, allow_pickle=False) as saved:
        _need(set(saved.files) == members,
              "C reader-bank diagnostic S2 saved members differ")
        return {name: np.asarray(saved[name]).copy() for name in saved.files}


def _diagnose(admission: dict, private_root: Path, parents: dict) -> dict:
    """Use admitted INNER features and saved hash-bound C/S2 artifacts only."""
    manifest = _read(CAMPAIGN / "DATA_ACCESS_MANIFEST.json")
    _need(manifest.get("status") == "ROLE_FILES_ADMITTED_S4_VERIFIED",
          "C reader-bank diagnostic INNER features role is not admitted")
    arrays, inner_source_sha = _role_file(manifest, "INNER_SELECTION", "features")
    feature_names = tuple(str(item) for item in arrays["feature_names"])
    _need(arrays["X"].shape == (2235, 83) and len(feature_names) == 83,
          "C reader-bank diagnostic INNER feature geometry differs")
    fit_worker, forward_worker, reader_stage = (_read(parents[key])
        for key in ("fit", "forward", "reader_r1"))
    _need(fit_worker.get("schema") == forward_worker.get("schema") ==
          "team-c-s3-worker-receipt-v1" and
          reader_stage.get("schema") == "team-c-reader-stage-v1" and
          reader_stage.get("status") == "FROZEN" and
          (reader_stage.get("arm_id"), reader_stage.get("seed"),
           reader_stage.get("view")) == ("C-SIMPLE-FULL", 0, "r1"),
          "C reader-bank diagnostic C parent output schema differs")
    fit_item = fit_worker["output"]["fit_handoffs"]["C-SIMPLE-FULL"]
    fit_path, fit_handoff = _bound_json(private_root,
        fit_item["relative_path"], fit_item["sha256"])
    forward_item = forward_worker["output"]
    _, forward_handoff = _bound_json(private_root,
        forward_item["forward_receipt_relative_path"],
        forward_item["forward_receipt_sha256"])
    _need(forward_handoff.get("fit_handoff_sha256") == fit_item["sha256"] and
          reader_stage.get("fit_handoff_sha256") == fit_item["sha256"] and
          reader_stage.get("forward_handoff_sha256") ==
          forward_item["forward_receipt_sha256"] and
          forward_handoff.get("campaign_lock_sha256") ==
          admission["historical_fit_lock_sha256"],
          "C reader-bank diagnostic FIT/FORWARD parent chain differs")
    fit_fields = {"OSF_DEFENSE_FIT:features", "OSF_DEFENSE_FIT:income",
                  "OSF_DEFENSE_FIT:occupation", "OSF_DEFENSE_FIT:sex",
                  "CALIBRATION_HELDOUT:features", "CALIBRATION_HELDOUT:income",
                  "CALIBRATION_HELDOUT:occupation"}
    fit_sources = {key: manifest["real_file_specs"][key]["sha256"]
                   for key in fit_fields}
    packages = load_packages(fit_handoff, private_root, feature_names,
        admission["historical_fit_lock_sha256"],
        admission["data_manifest_sha256"], fit_sources)
    raw = {}
    for purpose in ("income", "occupation_group"):
        raw[purpose], _ = predict_release(packages[purpose], arrays["X"],
                                          feature_names)
    role_item = forward_handoff["roles"]["INNER_SELECTION"]
    _need(role_item.get("role_feature_sha256") == inner_source_sha and
          role_item.get("row_unit_pair_sha256") == sha_array(np.column_stack(
              (arrays["row_id"], arrays["unit"])).astype(np.int64)),
          "C reader-bank diagnostic INNER role source differs")
    view_path, view_receipt = _bound_json(private_root,
        role_item["receipt_relative_path"], role_item["receipt_sha256"])
    bundle = load_saved_bundle(view_path, view_path.with_suffix(".npz"),
        arrays["X"], feature_names, raw, arrays["row_id"], arrays["unit"])
    rebuilt = bundle["views"]["income"]
    reader_path, reader_receipt = _bound_json(private_root,
        reader_stage["s2_reader_receipt_relative_path"],
        reader_stage["s2_reader_receipt_sha256"])
    files = reader_receipt.get("files", {})
    _need(reader_receipt.get("schema") ==
          "pcrl-portfolio-private-reader-custody-v1" and
          reader_receipt.get("status") == "FROZEN" and
          reader_receipt.get("reader_order") == list(READERS) and
          reader_receipt.get("inner_view_content_sha256") ==
          reader_stage.get("inner_view_content_sha256") and
          set(files) == {f"model__{name}.joblib" for name in READERS} | {
                        "view_att.npz", "view_inner.npz", "predictions_tune.npz",
                        "predictions_inner.npz"},
          "C reader-bank diagnostic S2 reader custody differs")
    for name, digest in files.items():
        _need(_sha(reader_path.parent / name) == digest,
              "C reader-bank diagnostic S2 reader child SHA differs")
    progress = reader_receipt.get("reader_progress")
    _need(isinstance(progress, list) and
          [item.get("reader_id") for item in progress] == list(READERS),
          "C reader-bank diagnostic six reader progress records differ")
    progress_dir = (reader_path.parent.parent.parent / "reader_progress" /
                    reader_path.parent.name)
    for item in progress:
        reader = item["reader_id"]
        for filename, key in ((f"model__{reader}.joblib", "model_sha256"),
                              (f"predictions__{reader}.npz", "prediction_file_sha256"),
                              (f"completed__{reader}.json", "completion_receipt_sha256")):
            _need(_sha(progress_dir / filename) == item.get(key),
                  "C reader-bank diagnostic reader progress child SHA differs")
    saved_inner = _npz(reader_path.parent / "view_inner.npz",
        files["view_inner.npz"], {"row_id", "unit", "x", "x_defense_aware",
                                  "x_raw_only"})
    saved_att = _npz(reader_path.parent / "view_att.npz",
        files["view_att.npz"], {"row_id", "unit", "x", "x_defense_aware",
                                "x_raw_only"})
    for key in saved_inner:
        _need(np.array_equal(saved_inner[key], rebuilt[key]),
              "C reader-bank diagnostic saved INNER values differ from independent view")
    saved_inner_bank = _npz(reader_path.parent / "predictions_inner.npz",
        files["predictions_inner.npz"], set(READERS))
    saved_tune_bank = _npz(reader_path.parent / "predictions_tune.npz",
        files["predictions_tune.npz"], set(READERS))
    _, tune, split = fixed_split(saved_att["unit"])
    _need(reader_receipt.get("split") == split,
          "C reader-bank diagnostic saved ATT split differs")
    by_reader, replayed_bank = {}, {}
    for index, reader in enumerate(READERS):
        progress_predictions = _npz(progress_dir / f"predictions__{reader}.npz",
            progress[index]["prediction_file_sha256"], {"tune", "inner"})
        _need(np.array_equal(progress_predictions["tune"], saved_tune_bank[reader]) and
              np.array_equal(progress_predictions["inner"], saved_inner_bank[reader]),
              "C reader-bank diagnostic progress and prediction bank differ")
        model_path = reader_path.parent / f"model__{reader}.joblib"
        _need(_sha(model_path) == files[model_path.name],
              "C reader-bank diagnostic model SHA differs")
        model = joblib.load(model_path)
        saved_x = reader_matrix(saved_inner, reader)
        rebuilt_x = reader_matrix(rebuilt, reader)
        att_x = reader_matrix(saved_att, reader)
        variants = {
            "saved_inner": saved_x,
            "rebuilt_inner": rebuilt_x,
            "c_contiguous_inner": np.ascontiguousarray(rebuilt_x),
            "f_contiguous_inner": np.asfortranarray(rebuilt_x),
        }
        predictions = {name: np.asarray(model.predict_proba(matrix)[:, 1],
                                        dtype=np.float64)
                       for name, matrix in variants.items()}
        tune_prediction = np.asarray(model.predict_proba(att_x[tune])[:, 1],
                                     dtype=np.float64)
        replayed_bank[reader] = predictions["rebuilt_inner"]
        by_reader[reader] = {
            "saved_inner_matrix_layout": matrix_layout(saved_x),
            "rebuilt_inner_matrix_layout": matrix_layout(rebuilt_x),
            "saved_att_matrix_layout": matrix_layout(att_x),
            "inner_variants": {
                name: {"layout": matrix_layout(variants[name]),
                       "versus_frozen": compare_predictions(saved_inner_bank[reader], actual)}
                for name, actual in predictions.items()
            },
            "saved_vs_rebuilt_model_predictions": compare_predictions(
                predictions["saved_inner"], predictions["rebuilt_inner"]),
            "tune_saved_matrix_versus_frozen": compare_predictions(
                saved_tune_bank[reader], tune_prediction),
        }
    summary = summarize_reader_bank(saved_inner_bank, replayed_bank)
    return {
        "status": "DIAGNOSTIC_RECORDED_NO_TOLERANCE_DECISION",
        "reader_order": list(READERS), "view": "r1", "arm": "C-SIMPLE-FULL", "seed": 0,
        "inner_role_file_sha256": inner_source_sha,
        "fit_handoff_sha256": fit_item["sha256"],
        "forward_handoff_sha256": forward_item["forward_receipt_sha256"],
        "reader_stage_sha256": _sha(parents["reader_r1"]),
        "s2_reader_receipt_sha256": reader_stage["s2_reader_receipt_sha256"],
        "saved_file_sha256": files,
        "saved_inner_value_equals_rebuilt": True,
        "summary": summary,
        "by_reader": by_reader,
        "labels_or_person_vectors_in_report": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unit-id", "admission", "admission-sha256", "local-seal",
                 "local-seal-sha256", "queue-root", "private-report"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    for name in ("admission", "local_seal", "queue_root", "private_report"):
        setattr(args, name, Path(getattr(args, name)))
    _need(args.unit_id == UNIT_ID, "C reader-bank diagnostic unit ID differs")
    args.queue_root = _queue_root(args.queue_root)
    state = _read(args.queue_root / "QUEUE.json")
    admission, _, private_root, parents = _admit(args, state)
    result = _diagnose(admission, private_root, parents)
    result.update({"schema": "pcrl-s4-c-r1-reader-bank-diagnostic-v2",
                   "unit_id": UNIT_ID,
                   "admission_sha256": args.admission_sha256,
                   "local_seal_sha256": args.local_seal_sha256,
                   "failed_s4_queue_receipt_sha256":
                   admission["failed_s4_queue_receipt_sha256"]})
    _write_once(args.private_report.resolve(), result)
    print(json.dumps({"unit_id": UNIT_ID, "status": result["status"],
                      "report_sha256": _sha(args.private_report.resolve())},
                     sort_keys=True))


if __name__ == "__main__":
    main()

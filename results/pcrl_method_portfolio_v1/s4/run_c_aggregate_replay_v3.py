"""S3-leased independent C R2 paired-inference replay over 16 S4 arm outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from replay_c_inference_v1 import prepare, simultaneous
from replay_c_values_v1 import sha_array
from run_c_adult_replay_v3 import (
    ADMISSION, CAMPAIGN, HERE, REPORT_ROOT, REVIEW, REPO, SEAL, SOURCE_FILES,
    _git_blob, _historical_replay_chain, _need, _one_argument, _queue_root,
    _read, _role_file, _sha,
    _write_once,
)


ARMS = (
    "C-SIMPLE-FULL", "C-SIMPLE-REL0", "C-SIMPLE-TASK-SHARED1",
    "C-SIMPLE-INDEP1", "C-SIMPLE-COMMON1", "C-SIMPLE-SEQ-IO1",
    "C-SIMPLE-SEQ-OI1", "C-SIMPLE-TASK-PAIR1",
)
ARM_RESULT_INDEX = CAMPAIGN / "S4_C_R3_ARM_RESULT_HASH_REGISTRY.json"
ARTIFACT_MEMBERS = {"row_id", "unit", "income", "occupation", "sex",
                    "prob_1", "prob_2"} | {
    f"reader_{view}__{reader}"
    for view in ("r1", "r2", "pair")
    for reader in ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                   "RAW_LR_C1", "RAW_HGB_15x200")
}


def _same(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"C aggregate {path} keys differ")
        for key, value in expected.items():
            _same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"C aggregate {path} length differs")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        if (not isinstance(actual, (int, float)) or
                not math.isfinite(float(actual)) or
                not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-8)):
            raise ValueError(f"C aggregate {path} numerical value differs")
    elif actual != expected:
        raise ValueError(f"C aggregate {path} differs")


def _s3_done(state: dict, queue_root: Path, entry: dict, *, team: str,
             mode: str, expected_id: str) -> tuple[Path, dict]:
    unit_id = entry.get("unit_id")
    _need(unit_id == expected_id, "C aggregate parent unit ID differs")
    unit = state.get("units", {}).get(unit_id, {})
    spec = unit.get("spec", {})
    receipt_path = queue_root / "units" / unit_id / "RECEIPT.json"
    receipt_sha = _sha(receipt_path)
    if entry.get("queue_receipt_sha256") is not None:
        _need(receipt_sha == entry["queue_receipt_sha256"],
              "C aggregate pinned parent S3 receipt SHA differs")
    receipt = _read(receipt_path)
    output = spec.get("outputs")
    output_sha = (receipt.get("output_sha256", {}).get(output[0])
                  if isinstance(output, list) and len(output) == 1 else None)
    if entry.get("output_sha256") is not None:
        _need(output_sha == entry["output_sha256"],
              "C aggregate pinned parent output SHA differs")
    _need(unit.get("status") == receipt.get("status") == "done" and
          unit.get("returncode") == receipt.get("returncode") == 0 and
          receipt.get("unit_id") == unit_id and
          spec.get("mode") == mode and spec.get("team") == team and
          isinstance(output, list) and len(output) == 1 and
          isinstance(output_sha, str) and len(output_sha) == 64 and
          receipt.get("output_sha256") == {output[0]: output_sha} and
          unit.get("output_sha256") == receipt["output_sha256"],
          "C aggregate parent did not complete exact frozen output")
    argv = spec.get("argv", [])
    _need(len(argv) >= 2 and Path(argv[1]).is_absolute() and
          _sha(Path(argv[1])) == spec.get("runner_sha256"),
          "C aggregate parent script source differs")
    path = Path(output[0]).resolve(strict=True)
    _need(path.is_file() and _sha(path) == output_sha,
          "C aggregate parent output changed")
    return path, spec


def _preaccess(args: argparse.Namespace, state: dict, runtime_argv: list[str]) -> tuple[dict, dict, Path, dict, dict]:
    _need(args.admission.resolve() == ADMISSION and
          args.local_seal.resolve() == SEAL and
          _sha(ADMISSION) == args.admission_sha256 and
          _sha(SEAL) == args.local_seal_sha256,
          "C aggregate admission or seal bytes differ")
    admission, review, seal = _read(ADMISSION), _read(REVIEW), _read(SEAL)
    _need(admission.get("schema") == "pcrl-s4-c-r3-replay-admission-v3" and
          admission.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
          admission.get("adult_role_access") is True and
          review.get("schema") == "pcrl-s4-c-r3-replay-prefit-review-v3" and
          review.get("status") == "PASS" and
          review.get("scope") == "S4_C_R3_ADULT_REPLAY" and
          seal.get("schema") == "pcrl-s4-c-r3-replay-local-seal-v3" and
          seal.get("status") == "PASS" and
          seal.get("admission_sha256") == review.get("admission_sha256") ==
          args.admission_sha256 and
          seal.get("review_sha256") == _sha(REVIEW) and
          seal.get("admission_commit_sha") == review.get("admission_commit_sha"),
          "C aggregate separate Git seal does not bind admission")
    _git_blob(seal["admission_commit_sha"],
              "results/pcrl_method_portfolio_v1/S4_C_R3_REPLAY_ADMISSION.json",
              args.admission_sha256)
    _historical_replay_chain(admission, state)
    fit = json.loads(_git_blob(admission["lock_commit_sha"],
        "results/pcrl_method_portfolio_v1/CAMPAIGN_LOCK.json",
        admission["lock_sha256"]))
    _need(fit.get("families", {}).get("C", {}).get("fit_admission") == "ALLOWED" and
          fit.get("families", {}).get("A", {}).get("fit_admission") == "DENIED" and
          fit.get("families", {}).get("D", {}).get("fit_admission") == "DENIED",
          "C aggregate historical R1 fit lock differs")
    seal_name = admission.get("fit_seal_relative_path")
    _need(isinstance(seal_name, str) and Path(seal_name).name == seal_name,
          "C aggregate historical FIT seal path differs")
    fit_seal = CAMPAIGN / seal_name
    review_name = admission.get("fit_review_relative_path")
    _need(isinstance(review_name, str) and Path(review_name).name == review_name,
          "C aggregate historical FIT review path differs")
    fit_review_path = CAMPAIGN / review_name
    fit_seal_obj, fit_review = _read(fit_seal), _read(fit_review_path)
    _need(_sha(fit_seal) == admission.get("fit_seal_sha256") and
          fit_seal_obj.get("status") == "PASS" and
          fit_seal_obj.get("commit_sha") == admission["lock_commit_sha"] and
          fit_seal_obj.get("lock_sha256") == admission["lock_sha256"] and
          fit_seal_obj.get("review_sha256") == admission.get("fit_review_sha256") ==
          _sha(fit_review_path) and
          fit_review.get("schema") == "pcrl-s4-prefit-seal-review-v1" and
          fit_review.get("status") == "PASS" and
          fit_review.get("fit_admission") == "ALL_FAMILIES" and
          fit_review.get("lock_commit_sha") == admission["lock_commit_sha"] and
          fit_review.get("lock_sha256") == admission["lock_sha256"],
          "C aggregate historical FIT seal differs")
    _need(_sha(CAMPAIGN / "DATA_ACCESS_MANIFEST.json") ==
          admission.get("data_manifest_sha256") ==
          fit.get("data_access_manifest_sha256"),
          "C aggregate data manifest differs")
    sources = admission.get("source_sha256_by_path")
    _need(isinstance(sources, dict) and set(sources) == SOURCE_FILES and
          all(_sha(REPO / name) == digest for name, digest in sources.items()) and
          _sha(CAMPAIGN / "shared/compute_queue.py") ==
          admission.get("queue_source_sha256"),
          "C aggregate independent or queue source changed")
    spec = admission.get("unit_specs", {}).get(args.unit_id, {})
    _need(args.unit_id == "S4-C-R3-AGGREGATE" and
          spec.get("stage") == "independent_c_r3_aggregate" and
          spec.get("data_access") == "S4_C_R3_VERIFIED_PRIVATE_AGGREGATE" and
          spec.get("seed") is None and
          spec.get("arm") == "C-AGGREGATE" and
          set(spec.get("parents", {})) == {"c_finalizer", "arm_reports"} and
          set(spec["parents"]["arm_reports"]) == {
              f"S4-C-R3-{arm}-S{seed}" for arm in ARMS for seed in (0, 1)},
          "C aggregate exact 17-parent scope differs")
    running = state.get("units", {}).get(args.unit_id, {})
    queued = running.get("spec", {})
    argv = queued.get("argv", [])
    report = args.private_report.resolve()
    _need(state.get("gate", {}).get("safe_inputs") is True and
          state.get("gate", {}).get("occupancy_confirmed") is True and
          state.get("gate", {}).get("manually_closed") is False and
          running.get("status") == "running" and
          running.get("child_pgid") == os.getpgrp() and
          queued.get("unit_id") == args.unit_id and
          queued.get("mode") == "s4_c_r3_adult_replay" and
          queued.get("team") == queued.get("family") == "S4" and
          queued.get("stage") == spec["stage"] and
          queued.get("arm") == queued.get("arm_id") == spec["arm"] and
          queued.get("seed") is None and
          queued.get("data_access") == spec["data_access"] and
          isinstance(queued.get("deps"), list) and
          len(queued["deps"]) == 17 and
          set(queued["deps"]) == ({spec["parents"]["c_finalizer"]["unit_id"]} |
                                  set(spec["parents"]["arm_reports"])) and
          queued.get("max_wall_s") == spec.get("max_wall_s") and
          queued.get("cpu_reserve_s") == spec.get("cpu_reserve_s") and
          queued.get("max_rss_gib") == spec.get("max_rss_gib") and
          queued.get("runner_sha256") == sources[
              "results/pcrl_method_portfolio_v1/s4/run_c_aggregate_replay_v3.py"] and
          queued.get("outputs") == [str(report)] and
          report.name == f"{args.unit_id}.json" and
          report.parent == REPORT_ROOT.resolve() and
          not report.exists() and
          len(argv) >= 2 and
          Path(argv[0]).resolve() == Path(sys.executable).resolve() and
          Path(argv[1]).resolve() == Path(__file__).resolve() and
          argv[1:] == runtime_argv and
          _one_argument(argv, "--unit-id") == args.unit_id and
          _one_argument(argv, "--admission-sha256") == args.admission_sha256 and
          _one_argument(argv, "--local-seal-sha256") == args.local_seal_sha256 and
          Path(_one_argument(argv, "--private-report")).resolve() == report,
          "C aggregate running S3 child, CLI or output differs")
    queue_root = args.queue_root.resolve(strict=True)
    _need(Path(_one_argument(argv, "--queue-root")).resolve() == queue_root,
          "C aggregate queue root differs")
    _need(args.arm_result_index.resolve() == ARM_RESULT_INDEX and
          _one_argument(argv, "--arm-result-index-sha256") ==
          args.arm_result_index_sha256 and
          Path(_one_argument(argv, "--arm-result-index")).resolve() ==
          ARM_RESULT_INDEX and
          _sha(ARM_RESULT_INDEX) == args.arm_result_index_sha256,
          "C aggregate post-arm registry path/SHA differs")
    registry_blob = subprocess.run([
        "git", "-C", str(REPO), "show",
        "HEAD:results/pcrl_method_portfolio_v1/S4_C_R3_ARM_RESULT_HASH_REGISTRY.json"],
        capture_output=True, timeout=10)
    _need(registry_blob.returncode == 0 and
          hashlib.sha256(registry_blob.stdout).hexdigest() ==
          args.arm_result_index_sha256,
          "C aggregate post-arm registry is not committed")
    registry = _read(ARM_RESULT_INDEX)
    _need(registry.get("schema") == "pcrl-s4-c-r3-arm-result-hash-registry-v3" and
          registry.get("status") == "S4_HASH_CHECKED_16_OF_16" and
          registry.get("admission_sha256") == args.admission_sha256 and
          registry.get("registry_builder_sha256") ==
          admission.get("post_arm_registry_builder_sha256") and
          _sha(HERE / "build_c_arm_result_registry_v3.py") ==
          admission.get("post_arm_registry_builder_sha256") and
          set(registry.get("units", {})) == set(spec["parents"]["arm_reports"]),
          "C aggregate post-arm registry scope differs")
    final_entry = spec["parents"]["c_finalizer"]
    final_id = final_entry.get("unit_id")
    _need(final_id == "C-R1-SCREEN-FINALIZE", "C aggregate R1 finalizer ID differs")
    final_path, final_spec = _s3_done(state, queue_root, final_entry,
        team="C", mode="real_data", expected_id=final_id)
    _need(final_spec.get("stage") == "screen_finalize" and
          final_spec.get("runner_sha256") ==
          fit["families"]["C"]["screen_orchestration"]["source_sha256"],
          "C aggregate finalizer stage/source differs")
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for seed in (0, 1):
            unit_id = f"S4-C-R3-{arm}-S{seed}"
            entry = {"unit_id": unit_id,
                     "queue_receipt_sha256": registry["units"][unit_id][
                         "queue_receipt_sha256"],
                     "output_sha256": registry["units"][unit_id][
                         "report_sha256"]}
            path, parent_spec = _s3_done(state, queue_root, entry,
                team="S4", mode="s4_c_r3_adult_replay", expected_id=unit_id)
            admitted = admission["unit_specs"][unit_id]
            parent_argv = parent_spec.get("argv", [])
            _need(parent_spec.get("stage") == "independent_c_r3_arm_replay" and
                  parent_spec.get("arm") == parent_spec.get("arm_id") == arm and
                  parent_spec.get("seed") == seed and
                  parent_spec.get("data_access") == admitted["data_access"] and
                  parent_spec.get("max_wall_s") == admitted["max_wall_s"] and
                  parent_spec.get("cpu_reserve_s") == admitted["cpu_reserve_s"] and
                  parent_spec.get("max_rss_gib") == admitted["max_rss_gib"] and
                  isinstance(parent_spec.get("deps"), list) and
                  len(parent_spec["deps"]) == 9 and
                  set(parent_spec["deps"]) == {
                      item["unit_id"] for item in admitted["parents"].values()} and
                  parent_spec.get("runner_sha256") ==
                  sources["results/pcrl_method_portfolio_v1/s4/run_c_adult_replay_v3.py"] and
                  _one_argument(parent_argv, "--admission-sha256") ==
                  args.admission_sha256 and
                  _one_argument(parent_argv, "--local-seal-sha256") ==
                  args.local_seal_sha256 and
                  path.parent == REPORT_ROOT.resolve() and
                  path.name == f"{unit_id}.json",
                  "C aggregate S4 arm parent stage/source differs")
            arms[arm][seed] = path
    return admission, fit, final_path, final_spec, arms


def _load_arm(path: Path, arm: str, seed: int) -> tuple[dict, dict]:
    report = _read(path)
    _need(report.get("status") == "REPLAYED_C_R3_PASS" and
          report.get("arm") == arm and report.get("seed") == seed and
          report.get("unit_id") == f"S4-C-R3-{arm}-S{seed}",
          "C aggregate arm replay report differs")
    entry = report.get("verified_inner_artifacts", {})
    _need(entry.get("filename") == path.with_suffix(".npz").name and
          entry.get("sha256") == _sha(path.with_suffix(".npz")) and
          set(entry.get("member_sha256", {})) == ARTIFACT_MEMBERS,
          "C aggregate S4 arm array custody differs")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as npz:
        _need(set(npz.files) == ARTIFACT_MEMBERS,
              "C aggregate S4 arm array members differ")
        arrays = {key: np.asarray(npz[key]).copy() for key in npz.files}
    _need(all(sha_array(value) == entry["member_sha256"][key]
              for key, value in arrays.items()),
          "C aggregate S4 arm array hash differs")
    return report, arrays


def _reference(seed: int, truth: dict, manifest: dict) -> dict:
    arrays, _ = _role_file(manifest, "INNER_SELECTION", f"references_s{seed}")
    _need(np.array_equal(arrays["row_id"], truth["row_id"]) and
          np.array_equal(arrays["unit"], truth["unit"]),
          "C aggregate reference ordered row/unit differs")
    return {key: arrays[key] for key in ("U0_1", "U0_2", "Ucal_1", "Ucal_2")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unit-id", "admission", "admission-sha256", "local-seal",
                 "local-seal-sha256", "queue-root", "private-report",
                 "arm-result-index", "arm-result-index-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    for name in ("admission", "local_seal", "queue_root", "private_report",
                 "arm_result_index"):
        setattr(args, name, Path(getattr(args, name)))
    args.queue_root = _queue_root(args.queue_root)
    state = _read(args.queue_root / "QUEUE.json")
    admission, fit, final_path, final_spec, arm_paths = _preaccess(args, state, sys.argv)
    final = _read(final_path)
    _need(final.get("schema") == "team-c-screen-final-v1" and
          final.get("status") == "AGGREGATED_PENDING_S4_REPLAY" and
          final.get("campaign_lock_sha256") == admission["lock_sha256"] and
          final.get("data_manifest_sha256") == admission["data_manifest_sha256"] and
          final.get("active_arms") == list(ARMS) and
          final.get("paired_seeds") == [0, 1],
          "C aggregate frozen C finalizer receipt differs")
    by_seed, arm_reports = {}, {}
    manifest = _read(CAMPAIGN / "DATA_ACCESS_MANIFEST.json")
    for seed in (0, 1):
        by_seed[seed] = {"arms": {}, "truth": None, "references": None}
    for arm in ARMS:
        arm_reports[arm] = {}
        for seed in (0, 1):
            report, arrays = _load_arm(arm_paths[arm][seed], arm, seed)
            by_seed[seed]["arms"][arm] = arrays
            truth = {key: arrays[key] for key in
                     ("row_id", "unit", "income", "occupation", "sex")}
            if by_seed[seed]["truth"] is None:
                by_seed[seed]["truth"] = truth
            elif any(not np.array_equal(by_seed[seed]["truth"][key], value)
                     for key, value in truth.items()):
                raise ValueError("C aggregate S4 arm truth differs")
            _need(final.get("arm_score_receipt_sha256", {}).get(arm, {}).get(str(seed)) ==
                  report.get("score_receipt_sha256"),
                  "C aggregate score parent differs")
            _same(final.get("capacities", {}).get(arm, {}).get(str(seed)),
                  report["capacity"], f"capacity.{arm}.{seed}")
            arm_reports[arm][seed] = report
    for seed in (0, 1):
        by_seed[seed]["references"] = _reference(seed, by_seed[seed]["truth"], manifest)
    _need(final.get("inner_row_unit_pair_sha256") == sha_array(np.column_stack((
        by_seed[0]["truth"]["row_id"], by_seed[0]["truth"]["unit"])).astype(np.int64)),
          "C aggregate finalizer INNER row/unit differs")
    exact = {}
    for arm in ARMS:
        exact[arm] = []
        for other in ARMS:
            if other != arm and all(
                arm_reports[arm][seed]["capacity"]["semantic"] ==
                arm_reports[other][seed]["capacity"]["semantic"] and
                arm_reports[arm][seed]["capacity"]["encoded"] ==
                arm_reports[other][seed]["capacity"]["encoded"] and
                arm_reports[arm][seed]["package_sha256_by_purpose"] !=
                arm_reports[other][seed]["package_sha256_by_purpose"]
                for seed in (0, 1)):
                exact[arm].append(other)
    _same(final.get("exact_nonalias_comparators"), exact, "exact_nonalias_comparators")
    argv = final_spec["argv"]
    _need(_one_argument(argv, "--input-sha256") ==
          _sha(Path(_one_argument(argv, "--input"))),
          "C finalizer input source changed")
    final_input = _read(Path(_one_argument(argv, "--input")))
    seed_path = Path(final_input["unit"]["private_seed_path"]).resolve(strict=True)
    _need(seed_path.is_relative_to(Path(final_spec["cwd"]).resolve() /
                                    "results/pcrl_method_portfolio_v1/teams/c/.private") and
          _sha(seed_path) == fit["inference_protocol"]["seed_public_sha256"],
          "C aggregate private paired seed differs")
    prepared = prepare(by_seed, ARMS)
    inference = simultaneous(prepared, seed_path.read_bytes())
    _same(final.get("simultaneous_inference"), inference,
          "simultaneous_inference")
    result = {
        "schema": "pcrl-s4-c-independent-aggregate-replay-v3",
        "status": "REPLAYED_C_R3_17_OF_17_PASS",
        "unit_id": args.unit_id,
        "admission_sha256": args.admission_sha256,
        "arm_result_registry_sha256": args.arm_result_index_sha256,
        "fit_lock_sha256": admission["lock_sha256"],
        "finalizer_receipt_sha256": _sha(final_path),
        "arm_report_sha256_by_arm_seed": {
            arm: {str(seed): _sha(arm_paths[arm][seed]) for seed in (0, 1)}
            for arm in ARMS},
        "exact_nonalias_comparators": exact,
        "simultaneous_inference": inference,
        "claim_scope": "exploratory development; independent S4 R2 replay of frozen C R1 outputs",
    }
    _write_once(args.private_report.resolve(), result)
    print(json.dumps({"unit_id": args.unit_id, "status": result["status"],
                      "report_sha256": _sha(args.private_report.resolve())},
                     sort_keys=True))


if __name__ == "__main__":
    main()

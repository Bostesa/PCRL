"""S4-only queued identity gate for one corrected C R2 arm replay.

No admitted person array is opened until the separate Git seal, current S3
child, nine completed C parent receipts, source hashes and role file manifests
have passed. This module imports no Team C or shared scorer implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = CAMPAIGN.parents[1]
ADMISSION = CAMPAIGN / "S4_C_R2_REPLAY_ADMISSION.json"
REVIEW = CAMPAIGN / "S4_C_R2_REPLAY_PREFIT_REVIEW.json"
SEAL = CAMPAIGN / "S4_C_R2_REPLAY_LOCAL_SEAL_RECEIPT.json"
QUEUE_ROOT = CAMPAIGN / "shared/.private/compute"
REPORT_ROOT = HERE / ".private/c_r2_replay_v2"
R1_ADMISSION = CAMPAIGN / "S4_C_REPLAY_ADMISSION.json"
R1_REVIEW = CAMPAIGN / "S4_C_REPLAY_PREFIT_REVIEW.json"
R1_SEAL = CAMPAIGN / "S4_C_REPLAY_LOCAL_SEAL_RECEIPT.json"
DIAG_ADMISSION = CAMPAIGN / "S4_C_READER_BANK_DIAGNOSTIC_V2_ADMISSION.json"
DIAG_REVIEW = CAMPAIGN / "S4_C_READER_BANK_DIAGNOSTIC_V2_PREFIT_REVIEW.json"
DIAG_SEAL = CAMPAIGN / "S4_C_READER_BANK_DIAGNOSTIC_V2_LOCAL_SEAL_RECEIPT.json"
FAILED_R1_ID = "S4-C-R1-C-SIMPLE-FULL-S0"
DIAG_ID = "S4-C-R1-READER-BANK-DIAG-FULL-S0-R1-R2"
SOURCE_FILES = frozenset(
    f"results/pcrl_method_portfolio_v1/s4/{name}.py" for name in (
        "run_c_adult_replay_v2", "replay_c_values_v1", "replay_c_release_v1",
        "replay_c_readers_v2", "replay_c_endpoints_v1",
        "replay_c_attempt_v2", "replay_c_inference_v1",
        "run_c_aggregate_replay_v2", "replay_v2_controls"))
PARENT_NAMES = frozenset(("fit", "forward", "reader_r1", "reader_r2",
                          "reader_pair", "v2_r1", "v2_r2", "v2_pair",
                          "score"))
ROLE_FIELDS = (
    ("ATTACK_FIT_NEW", "features"), ("ATTACK_FIT_NEW", "sex"),
    ("INNER_SELECTION", "features"), ("INNER_SELECTION", "income"),
    ("INNER_SELECTION", "occupation"), ("INNER_SELECTION", "sex"),
)


def _need(ok: bool, why: str) -> None:
    if not ok:
        raise PermissionError(why)


def _queue_root(path: Path) -> Path:
    """Bind the running child to the coordinator's one S3 ledger."""
    root = path.resolve(strict=True)
    _need(root == QUEUE_ROOT.resolve(strict=True),
          "S4 C coordinator queue root differs")
    return root


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    _need(isinstance(value, dict), f"{path.name} is not a JSON object")
    return value


def _git_blob(commit: str, relative: str, digest: str) -> bytes:
    _need(isinstance(commit, str) and len(commit) == 40 and
          all(ch in "0123456789abcdef" for ch in commit),
          "S4 C historical Git commit differs")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True, timeout=10)
    _need(ancestor.returncode == 0, "S4 C Git commit is not an ancestor")
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"],
                          capture_output=True, timeout=10)
    _need(blob.returncode == 0 and
          hashlib.sha256(blob.stdout).hexdigest() == digest,
          "S4 C committed Git blob SHA differs")
    return blob.stdout


def _one_argument(argv: list[str], flag: str) -> str:
    places = [i for i, value in enumerate(argv) if value == flag]
    _need(len(places) == 1 and places[0] + 1 < len(argv),
          f"S4 C running argv lacks one {flag}")
    return argv[places[0] + 1]


def _private_root(path: Path) -> Path:
    path = path.resolve(strict=True)
    matches = [parent for parent in path.parents if parent.name == ".private"]
    _need(len(matches) == 1 and matches[0].parent.name == "c" and
          matches[0].parent.parent.name == "teams" and
          matches[0].is_relative_to(REPO.parent),
          "S4 C parent output is outside Team C private custody")
    return matches[0]


def _historical_replay_chain(admission: dict, state: dict) -> None:
    """Pin the failed R1 attempt and the measured exact-layout correction."""
    files = {
        "prior_r1_admission_sha256": R1_ADMISSION,
        "prior_r1_review_sha256": R1_REVIEW,
        "prior_r1_seal_sha256": R1_SEAL,
        "diagnostic_v2_admission_sha256": DIAG_ADMISSION,
        "diagnostic_v2_review_sha256": DIAG_REVIEW,
        "diagnostic_v2_seal_sha256": DIAG_SEAL,
    }
    _need(all(_sha(path) == admission.get(key) for key, path in files.items()),
          "S4 C R2 historical admission or diagnostic seal changed")
    r1, r1_seal = _read(R1_ADMISSION), _read(R1_SEAL)
    diag, diag_seal = _read(DIAG_ADMISSION), _read(DIAG_SEAL)
    _need(r1.get("status") == diag.get("status") ==
          "LOCAL_GIT_SEALED_EFFECTIVE" and
          r1_seal.get("status") == diag_seal.get("status") == "PASS" and
          r1_seal.get("admission_sha256") == admission["prior_r1_admission_sha256"] and
          r1_seal.get("review_sha256") == admission["prior_r1_review_sha256"] and
          diag_seal.get("admission_sha256") ==
          admission["diagnostic_v2_admission_sha256"] and
          diag_seal.get("review_sha256") ==
          admission["diagnostic_v2_review_sha256"] and
          r1.get("lock_sha256") == admission.get("lock_sha256") and
          r1.get("lock_commit_sha") == admission.get("lock_commit_sha") and
          diag.get("historical_s4_admission_sha256") ==
          admission["prior_r1_admission_sha256"],
          "S4 C R2 historical scope differs")
    _git_blob(r1_seal["admission_commit_sha"],
              "results/pcrl_method_portfolio_v1/S4_C_REPLAY_ADMISSION.json",
              admission["prior_r1_admission_sha256"])
    _git_blob(diag_seal["admission_commit_sha"],
              "results/pcrl_method_portfolio_v1/S4_C_READER_BANK_DIAGNOSTIC_V2_ADMISSION.json",
              admission["diagnostic_v2_admission_sha256"])
    for source, digest in r1["source_sha256_by_path"].items():
        _need(_sha(REPO / source) == digest,
              "S4 C R2 historical R1 source changed")
    failed_path = QUEUE_ROOT / "units" / FAILED_R1_ID / "RECEIPT.json"
    failed = _read(failed_path)
    _need(_sha(failed_path) == admission.get("prior_r1_failed_queue_receipt_sha256") and
          failed.get("status") == "failed" and failed.get("returncode") == 1 and
          failed.get("output_sha256") == {} and
          state.get("units", {}).get(FAILED_R1_ID, {}).get("status") == "failed",
          "S4 C R2 failed R1 attempt changed")
    diag_receipt_path = QUEUE_ROOT / "units" / DIAG_ID / "RECEIPT.json"
    diag_report_path = HERE / ".private/c_r1_reader_bank_diag_v2" / f"{DIAG_ID}.json"
    diag_receipt, report = _read(diag_receipt_path), _read(diag_report_path)
    raw = report.get("by_reader", {}).get("RAW_LR_C1", {})
    summary = report.get("summary", {})
    _need(_sha(diag_receipt_path) ==
          admission.get("diagnostic_v2_queue_receipt_sha256") and
          _sha(diag_report_path) == admission.get("diagnostic_v2_report_sha256") and
          diag_receipt.get("status") == "done" and
          diag_receipt.get("returncode") == 0 and
          diag_receipt.get("output_sha256") == {
              str(diag_report_path): admission["diagnostic_v2_report_sha256"]} and
          report.get("status") == "DIAGNOSTIC_RECORDED_NO_TOLERANCE_DECISION" and
          report.get("saved_inner_value_equals_rebuilt") is True and
          summary.get("exact_mismatch_readers") == ["RAW_LR_C1"] and
          raw.get("inner_variants", {}).get("c_contiguous_inner", {}).get(
              "versus_frozen", {}).get("exact_equal") is True and
          raw.get("inner_variants", {}).get("rebuilt_inner", {}).get(
              "versus_frozen", {}).get("exact_equal") is False,
          "S4 C R2 exact-layout diagnostic evidence changed")


def _parent_output(queue_root: Path, state: dict, entry: dict,
                   lock: dict, seed: int, arm: str, label: str) -> tuple[Path, Path]:
    unit_id = entry.get("unit_id")
    digest = entry.get("queue_receipt_sha256")
    output_sha = entry.get("output_sha256")
    _need(isinstance(unit_id, str) and unit_id.startswith("C-R1-") and
          isinstance(digest, str) and isinstance(output_sha, str),
          "S4 C parent identity map is incomplete")
    receipt_path = queue_root / "units" / unit_id / "RECEIPT.json"
    _need(_sha(receipt_path) == digest, "S4 C parent S3 receipt SHA differs")
    receipt = _read(receipt_path)
    unit = state.get("units", {}).get(unit_id, {})
    spec = unit.get("spec", {})
    outputs = spec.get("outputs")
    stage = ("forward_from_handoff" if label == "forward" else
             "screen_reader" if label.startswith("reader_") else
             "postrelease_control_v2" if label.startswith("v2_") else
             "screen_score" if label == "score" else None)
    view = label.split("_", 1)[1] if label.startswith(("reader_", "v2_")) else None
    if label == "fit":
        _need(unit_id in {
            f"C-R1-FIT-PRIVACY-S{seed}", f"C-R1-FIT-TASK-ONLY-S{seed}",
            f"C-R1-FIT-TASK-PAIR-S{seed}"} and
            spec.get("stage") in {"privacy_fit_cal_batch", "task_only_fit_cal",
                                  "task_pair_fit_cal"},
              "S4 C FIT parent is not one of the frozen R1 fit units")
    else:
        expected_id = (f"C-R1-FORWARD-{arm}-S{seed}" if label == "forward" else
                       f"C-R1-SCREEN-READ-{arm}-S{seed}-{view}" if label.startswith("reader_") else
                       f"C-R1-SCREEN-V2-{arm}-S{seed}-{view}" if label.startswith("v2_") else
                       f"C-R1-SCREEN-SCORE-{arm}-S{seed}")
        _need(unit_id == expected_id and spec.get("stage") == stage and
              spec.get("arm") == arm and spec.get("view") == view,
              "S4 C parent stage, arm or view differs")
    _need(receipt.get("unit_id") == unit_id and receipt.get("status") == "done" and
          receipt.get("returncode") == 0 and unit.get("status") == "done" and
          unit.get("returncode") == 0 and
          spec.get("mode") == "real_data" and spec.get("team") == "C" and
          spec.get("family") == "C" and spec.get("seed") == seed and
          isinstance(outputs, list) and len(outputs) == 1 and
          receipt.get("output_sha256") == {outputs[0]: output_sha} and
          unit.get("output_sha256") == receipt["output_sha256"],
          "S4 C parent unit or output was not frozen done")
    argv = spec.get("argv", [])
    script = Path(argv[1]) if len(argv) > 1 else Path("/nonexistent")
    expected_script = (lock["families"]["C"]["worker_runner_sha256"]
                       if unit_id.startswith("C-R1-FIT-") or
                       unit_id.startswith("C-R1-FORWARD-") else
                       lock["families"]["C"]["screen_orchestration"]["source_sha256"])
    _need(script.is_absolute() and script.is_file() and
          spec.get("runner_sha256") == expected_script and
          _sha(script) == expected_script,
          "S4 C parent script source differs from R1 fit lock")
    output = Path(outputs[0]).resolve(strict=True)
    _need(output.is_file() and _sha(output) == output_sha,
          "S4 C parent output bytes differ")
    return output, _private_root(output)


def admit(args: argparse.Namespace, state: dict, runtime_argv: list[str]) -> tuple[dict, dict, Path, dict]:
    """Complete all pre-role checks; return admission, arm unit and parent outputs."""
    _need(args.admission.resolve() == ADMISSION and args.local_seal.resolve() == SEAL and
          _sha(ADMISSION) == args.admission_sha256 and
          _sha(SEAL) == args.local_seal_sha256,
          "S4 C separate admission or local seal bytes differ")
    admission, review, seal = _read(ADMISSION), _read(REVIEW), _read(SEAL)
    _need(admission.get("schema") == "pcrl-s4-c-r2-replay-admission-v2" and
          admission.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
          admission.get("adult_role_access") is True and
          review.get("schema") == "pcrl-s4-c-r2-replay-prefit-review-v2" and
          review.get("status") == "PASS" and
          review.get("scope") == "S4_C_R2_ADULT_REPLAY" and
          seal.get("schema") == "pcrl-s4-c-r2-replay-local-seal-v2" and
          seal.get("status") == "PASS" and
          seal.get("admission_sha256") == args.admission_sha256 and
          seal.get("admission_commit_sha") == review.get("admission_commit_sha") and
          seal.get("review_sha256") == _sha(REVIEW) and
          review.get("admission_sha256") == args.admission_sha256,
          "S4 C separate review or seal did not bind admission")
    _git_blob(seal["admission_commit_sha"],
              "results/pcrl_method_portfolio_v1/S4_C_R2_REPLAY_ADMISSION.json",
              args.admission_sha256)
    _historical_replay_chain(admission, state)
    fit_blob = _git_blob(admission["lock_commit_sha"],
              "results/pcrl_method_portfolio_v1/CAMPAIGN_LOCK.json",
              admission["lock_sha256"])
    fit = json.loads(fit_blob)
    _need(isinstance(fit, dict) and fit.get("fit_admission") == "ALL_FAMILIES" and
          fit.get("families", {}).get("C", {}).get("fit_admission") == "ALLOWED" and
          fit.get("families", {}).get("A", {}).get("fit_admission") == "DENIED" and
          fit.get("families", {}).get("D", {}).get("fit_admission") == "DENIED",
          "S4 C historical R1 fit scope differs")
    seal_name = admission.get("fit_seal_relative_path")
    _need(isinstance(seal_name, str) and Path(seal_name).name == seal_name,
          "S4 C historical FIT seal path differs")
    fit_seal_path = CAMPAIGN / seal_name
    fit_seal = _read(fit_seal_path)
    review_name = admission.get("fit_review_relative_path")
    _need(isinstance(review_name, str) and Path(review_name).name == review_name,
          "S4 C historical FIT review path differs")
    fit_review_path = CAMPAIGN / review_name
    fit_review = _read(fit_review_path)
    _need(_sha(fit_seal_path) == admission.get("fit_seal_sha256") and
          fit_seal.get("status") == "PASS" and
          fit_seal.get("commit_sha") == admission["lock_commit_sha"] and
          fit_seal.get("lock_sha256") == admission["lock_sha256"] and
          fit_seal.get("review_sha256") == admission.get("fit_review_sha256") ==
          _sha(fit_review_path) and
          fit_review.get("schema") == "pcrl-s4-prefit-seal-review-v1" and
          fit_review.get("status") == "PASS" and
          fit_review.get("fit_admission") == "ALL_FAMILIES" and
          fit_review.get("lock_commit_sha") == admission["lock_commit_sha"] and
          fit_review.get("lock_sha256") == admission["lock_sha256"],
          "S4 C R1 fit seal differs")
    _need(_sha(CAMPAIGN / "DATA_ACCESS_MANIFEST.json") ==
          admission.get("data_manifest_sha256") ==
          fit.get("data_access_manifest_sha256"),
          "S4 C admitted role manifest SHA differs")
    sources = admission.get("source_sha256_by_path", {})
    _need(isinstance(sources, dict) and set(sources) == SOURCE_FILES,
          "S4 C independent source set differs")
    for relative, digest in sources.items():
        source = REPO / relative
        _need(source.is_file() and _sha(source) == digest,
              f"S4 C independent source changed: {relative}")
    _need(admission.get("queue_source_sha256") == _sha(CAMPAIGN / "shared/compute_queue.py"),
          "S4 C queue source differs")
    unit = admission.get("unit_specs", {}).get(args.unit_id)
    _need(isinstance(unit, dict) and unit.get("stage") ==
          "independent_c_r2_arm_replay" and
          args.unit_id == f"S4-C-R2-{unit.get('arm')}-S{unit.get('seed')}" and
          unit.get("data_access") == "S4_C_R2_ATT_INNER_FEATURES_TASK_SEX_REPLAY" and
          unit.get("arm") in fit.get("admitted_arms", {}).get("C", []) and
          unit.get("seed") in (0, 1) and
          set(unit.get("parents", {})) == PARENT_NAMES,
          "S4 C arm/seed, role token or parent set differs")
    running = state.get("units", {}).get(args.unit_id, {})
    spec = running.get("spec", {})
    argv = spec.get("argv", [])
    report = args.private_report.resolve()
    _need(state.get("gate", {}).get("safe_inputs") is True and
          state.get("gate", {}).get("occupancy_confirmed") is True and
          state.get("gate", {}).get("manually_closed") is False and
          running.get("status") == "running" and
          running.get("child_pgid") == os.getpgrp() and
          spec.get("unit_id") == args.unit_id and
          spec.get("mode") == "s4_c_r2_adult_replay" and
          spec.get("team") == "S4" and spec.get("family") == "S4" and
          spec.get("stage") == unit["stage"] and
          spec.get("arm") == spec.get("arm_id") == unit["arm"] and
          spec.get("seed") == unit["seed"] and
          spec.get("data_access") == unit["data_access"] and
          isinstance(spec.get("deps"), list) and
          len(spec["deps"]) == len(PARENT_NAMES) and
          set(spec["deps"]) == {item["unit_id"] for item in unit["parents"].values()} and
          spec.get("max_wall_s") == unit.get("max_wall_s") and
          spec.get("cpu_reserve_s") == unit.get("cpu_reserve_s") and
          spec.get("max_rss_gib") == unit.get("max_rss_gib") and
          spec.get("runner_sha256") == sources[
              "results/pcrl_method_portfolio_v1/s4/run_c_adult_replay_v2.py"] and
          spec.get("outputs") == [str(report)] and
          report.name == f"{args.unit_id}.json" and
          not report.exists() and not report.with_suffix(".npz").exists() and
          report.parent == REPORT_ROOT.resolve() and
          Path(argv[0]).resolve() == Path(sys.executable).resolve() and
          Path(argv[1]).resolve() == Path(__file__).resolve() and
          argv[1:] == runtime_argv and
          _one_argument(argv, "--unit-id") == args.unit_id and
          Path(_one_argument(argv, "--private-report")).resolve() == report and
          _one_argument(argv, "--admission-sha256") == args.admission_sha256 and
          _one_argument(argv, "--local-seal-sha256") == args.local_seal_sha256,
          "S4 C running child, argv or output differs")
    queue_root = args.queue_root.resolve(strict=True)
    _need(Path(_one_argument(argv, "--queue-root")).resolve() == queue_root,
          "S4 C queue path differs")
    parent_outputs, private_roots = {}, set()
    for label, entry in unit["parents"].items():
        output, private = _parent_output(queue_root, state, entry, fit,
                                         unit["seed"], unit["arm"], label)
        parent_outputs[label] = output
        private_roots.add(private)
    forward_spec = state["units"][unit["parents"]["forward"]["unit_id"]]["spec"]
    _need(forward_spec.get("deps") == [unit["parents"]["fit"]["unit_id"]],
          "S4 C forward does not depend on selected fit parent")
    _need(len(private_roots) == 1,
          "S4 C nine parents cross private custody roots")
    return admission, unit, next(iter(private_roots)), parent_outputs


def _role_file(manifest: dict, role: str, field: str) -> tuple[dict, str]:
    spec = manifest["real_file_specs"][f"{role}:{field}"]
    path = (CAMPAIGN / spec["path"]).resolve(strict=True)
    _need(path.is_relative_to(CAMPAIGN.resolve()) and ".private" in path.parts and
          _sha(path) == spec["sha256"],
          "S4 C role file path/SHA differs from admission")
    with np.load(path, allow_pickle=False) as package:
        _need(set(package.files) == set(spec["members"]),
              "S4 C role member set differs")
        arrays = {name: np.asarray(package[name]).copy() for name in package.files}
    row = np.asarray(arrays["row_id"], dtype=np.int64)
    group = np.asarray(arrays["unit"], dtype=np.int64)
    from replay_c_values_v1 import sha_array
    _need(row.ndim == 1 and group.shape == row.shape and
          sha_array(np.column_stack((row, group)).astype(np.int64)) ==
          spec["row_unit_pair_sha256"],
          "S4 C role ordered row/unit differs")
    return arrays, spec["sha256"]


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish the independently replayed INNER inputs for paired inference."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unit-id", "admission", "admission-sha256", "local-seal",
                 "local-seal-sha256", "queue-root", "private-report"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    for name in ("admission", "local_seal", "queue_root", "private_report"):
        setattr(args, name, Path(getattr(args, name)))
    args.queue_root = _queue_root(args.queue_root)
    state = _read(args.queue_root / "QUEUE.json")
    admission, unit, private_root, parents = admit(args, state, sys.argv)
    manifest = _read(CAMPAIGN / "DATA_ACCESS_MANIFEST.json")
    _need(manifest.get("status") == "ROLE_FILES_ADMITTED_S4_VERIFIED",
          "S4 C role manifest is not admitted")
    data, source = {}, {}
    for role, field in ROLE_FIELDS:
        data[f"{role}:{field}"], source[f"{role}:{field}"] = _role_file(
            manifest, role, field)
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        anchor = data[f"{role}:features"]
        for field in ("sex",) if role == "ATTACK_FIT_NEW" else ("income", "occupation", "sex"):
            record = data[f"{role}:{field}"]
            _need(np.array_equal(anchor["row_id"], record["row_id"]) and
                  np.array_equal(anchor["unit"], record["unit"]),
                  "S4 C role label row/unit alignment differs")
    roles = {
        role: {"features": data[f"{role}:features"]["X"],
               "feature_names": data[f"{role}:features"]["feature_names"],
               "row_id": data[f"{role}:features"]["row_id"],
               "unit": data[f"{role}:features"]["unit"],
               **{field: data[f"{role}:{field}"][field]
                  for field in (("sex",) if role == "ATTACK_FIT_NEW"
                                else ("income", "occupation", "sex"))}}
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")
    }
    feature_names = tuple(str(item) for item in roles["ATTACK_FIT_NEW"]["feature_names"])
    _need(tuple(str(item) for item in roles["INNER_SELECTION"]["feature_names"]) ==
          feature_names, "S4 C ATT/INNER feature schema differs")
    fit_fields = {"OSF_DEFENSE_FIT:features", "OSF_DEFENSE_FIT:income",
                  "OSF_DEFENSE_FIT:occupation", "OSF_DEFENSE_FIT:sex",
                  "CALIBRATION_HELDOUT:features",
                  "CALIBRATION_HELDOUT:income",
                  "CALIBRATION_HELDOUT:occupation"}
    expected = {key: manifest["real_file_specs"][key]["sha256"]
                for key in fit_fields}
    source.update(expected)
    fit_worker = _read(parents["fit"])
    forward_worker = _read(parents["forward"])
    _need(fit_worker.get("schema") == forward_worker.get("schema") ==
          "team-c-s3-worker-receipt-v1" and
          fit_worker.get("campaign_lock_sha256") == admission["lock_sha256"] and
          forward_worker.get("campaign_lock_sha256") == admission["lock_sha256"],
          "S4 C FIT/FORWARD worker outputs differ")
    arm = unit["arm"]
    fit_output = fit_worker["output"]
    if "fit_handoffs" in fit_output:
        fit_item = fit_output["fit_handoffs"][arm]
        fit_relative, fit_sha = fit_item["relative_path"], fit_item["sha256"]
    else:
        fit_relative = fit_output["fit_handoff_relative_path"]
        fit_sha = fit_output["fit_handoff_sha256"]
    forward_output = forward_worker["output"]
    paths = {
        "fit_relative_path": fit_relative, "fit_sha256": fit_sha,
        "forward_relative_path": forward_output["forward_receipt_relative_path"],
        "forward_sha256": forward_output["forward_receipt_sha256"],
        "reader_relative_path_by_view": {}, "reader_sha256_by_view": {},
        "v2_relative_path_by_view": {}, "v2_sha256_by_view": {},
    }
    for view in ("r1", "r2", "pair"):
        for stage in ("reader", "v2"):
            path = parents[f"{stage}_{view}"]
            relative = path.relative_to(private_root).as_posix()
            paths[f"{stage}_relative_path_by_view"][view] = relative
            paths[f"{stage}_sha256_by_view"][view] = _sha(path)
    paths["score_relative_path"] = parents["score"].relative_to(private_root).as_posix()
    paths["score_sha256"] = _sha(parents["score"])
    from replay_c_attempt_v2 import replay_attempt
    result, artifacts = replay_attempt(
        arm=arm, seed=unit["seed"], lock_sha256=admission["lock_sha256"],
        manifest_sha256=admission["data_manifest_sha256"],
        private_root=private_root, feature_names=feature_names,
        role_values=roles, role_source_sha256=source, paths=paths,
        return_artifacts=True)
    result["unit_id"] = args.unit_id
    result["parent_queue_receipt_sha256_by_unit"] = {
        item["unit_id"]: item["queue_receipt_sha256"]
        for item in unit["parents"].values()}
    artifact_path = args.private_report.resolve().with_suffix(".npz")
    _need(not artifact_path.exists(), "S4 C replay artifact target already exists")
    _write_npz_once(artifact_path, artifacts)
    from replay_c_values_v1 import sha_array
    result["verified_inner_artifacts"] = {
        "filename": artifact_path.name,
        "sha256": _sha(artifact_path),
        "member_sha256": {name: sha_array(np.asarray(value))
                          for name, value in artifacts.items()},
    }
    _write_once(args.private_report.resolve(), result)
    print(json.dumps({"unit_id": args.unit_id, "status": result["status"],
                      "report_sha256": _sha(args.private_report.resolve())},
                     sort_keys=True))


if __name__ == "__main__":
    main()

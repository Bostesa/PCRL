"""S4-only gate and runner for one independently admitted Adult replay.

All identity, Git, source and S3 checks precede the first role-value read.
The scientific value cores in this directory import no Team B or S2 scorer.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = CAMPAIGN.parents[1]
ADMISSION_NAME = "S4_VERIFIER_ADMISSION.json"
REVIEW_NAME = "S4_VERIFIER_PREFIT_REVIEW.json"
SEAL_NAME = "S4_VERIFIER_LOCAL_SEAL_RECEIPT.json"
MAP_NAME = "S4_ADULT_REPLAY_PARENT_MAP_V2.json"
SCHEMA_NAME = "S4_ADULT_REPLAY_UNIT_SCHEMA_V2.json"
SOURCE_FILES = frozenset(
    f"results/pcrl_method_portfolio_v1/s4/{name}.py" for name in (
        "run_adult_replay_v3", "replay_b_values_v3", "replay_anchor_values_v3",
        "replay_b_attempt", "replay_anchor_attempt", "replay_b_readers",
        "replay_b_release", "replay_feature_anchor", "replay_v2_controls",
        "verifier_identity_v3", "independent_endpoints"))


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise PermissionError(reason)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    _need(isinstance(value, dict), f"{path.name} is not a JSON object")
    return value


def _arg(argv: list[str], name: str) -> str:
    positions = [i for i, value in enumerate(argv) if value == name]
    _need(len(positions) == 1 and positions[0] + 1 < len(argv),
          f"S4 running argv lacks one {name}")
    return str(argv[positions[0] + 1])


def parent_output_root(receipt_path: Path, receipt_sha256: str,
                       output_sha256: str, output_name: str,
                       unit_id: str) -> Path:
    """Recover exactly one hash-bound private parent output from its S3 receipt."""
    receipt_path = Path(receipt_path).resolve()
    _need(_sha(receipt_path) == receipt_sha256, "S4 parent queue receipt SHA differs")
    receipt = _read(receipt_path)
    _need(receipt.get("unit_id") == unit_id and receipt.get("status") == "done" and
          receipt.get("returncode") == 0, "S4 parent queue unit did not finish")
    outputs = receipt.get("output_sha256")
    _need(isinstance(outputs, dict) and len(outputs) == 1,
          "S4 parent queue output must be a singleton")
    raw_path, recorded_sha = next(iter(outputs.items()))
    path = Path(raw_path)
    _need(path.is_absolute() and path.name == output_name and recorded_sha == output_sha256,
          "S4 parent queue output path or SHA differs")
    path = path.resolve()
    _need(".private" in path.parts and path.is_file() and _sha(path) == output_sha256,
          "S4 parent private output bytes differ")
    return path.parent


def running_unit_gate(state: dict, unit_spec: dict, script: Path,
                      argv: list[str], report: Path) -> None:
    """Refuse a direct invocation or a mismatched S3 running child."""
    gate = state.get("gate", {})
    _need(gate.get("safe_inputs") is True and
          gate.get("occupancy_confirmed") is True and
          gate.get("manually_closed") is False,
          "S4 Adult queue gate is closed")
    unit_id = unit_spec["unit_id"]
    running = state.get("units", {}).get(unit_id, {})
    spec = running.get("spec", {})
    stage = ("independent_b_v2_replay" if unit_spec["kind"] == "B_SCREEN"
             else "independent_anchor_v2_replay")
    _need(running.get("status") == "running" and
          running.get("child_pgid") == os.getpgrp() and
          spec.get("unit_id") == unit_id and spec.get("mode") == "s4_adult_replay" and
          spec.get("family") == "S4" and spec.get("team") == "S4" and
          spec.get("stage") == stage and
          spec.get("arm") == unit_spec["arm"] and
          spec.get("seed") == unit_spec["seed"] and
          spec.get("data_access") == unit_spec["data_access"],
          "S4 running queue unit identity differs")
    expected_argv = spec.get("argv")
    _need(isinstance(expected_argv, list) and expected_argv == argv and
          len(argv) >= 2 and Path(argv[1]).resolve() == Path(script).resolve() and
          _arg(argv, "--unit-id") == unit_id and
          Path(_arg(argv, "--private-report")).resolve() == Path(report).resolve() and
          spec.get("outputs") == [str(Path(report).resolve())],
          "S4 running script, argv or output differs")


def _git_blob(commit: str, relative: str, expected_sha256: str) -> bytes:
    ancestor = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                               commit, "HEAD"], capture_output=True, timeout=10)
    _need(ancestor.returncode == 0, "S4 historical lock commit is not an ancestor")
    blob = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"],
                          capture_output=True, timeout=10)
    _need(blob.returncode == 0 and
          hashlib.sha256(blob.stdout).hexdigest() == expected_sha256,
          "S4 committed historical lock blob differs")
    return blob.stdout


def _historical_lock(admission: dict, lock_sha256: str, kind: str) -> dict:
    entry = admission["historical_seal_by_lock_sha256"][lock_sha256]
    _need(entry["kind"] == kind and entry["lock_sha256"] == lock_sha256,
          "S4 historical seal kind or SHA differs")
    paths = {name: (CAMPAIGN / entry[f"{name}_path"]).resolve()
             for name in ("lock", "receipt", "review")}
    _need(all(path.is_relative_to(CAMPAIGN) and path.is_file() for path in paths.values()),
          "S4 historical seal path escapes campaign or is absent")
    for name, path in paths.items():
        _need(_sha(path) == entry[f"{name}_sha256"],
              f"S4 historical {name} file SHA differs")
    _need(entry["commit_sha"] == _read(paths["receipt"]).get("commit_sha"),
          "S4 historical seal commit differs")
    receipt, review = _read(paths["receipt"]), _read(paths["review"])
    _need(receipt.get("status") == "PASS" and review.get("status") == "PASS" and
          receipt.get("review_sha256") == entry["review_sha256"] and
          review.get("lock_sha256") == lock_sha256 and
          review.get("lock_commit_sha") == entry["commit_sha"],
          "S4 historical seal or review is not PASS")
    receipt_lock = (receipt.get("campaign_lock_sha256") if kind == "FIT"
                    else receipt.get("control_lock_sha256"))
    _need(receipt_lock == lock_sha256, "S4 historical seal lock SHA differs")
    original = ("CAMPAIGN_LOCK.json" if kind == "FIT" else "CONTROL_V2_LOCK.json")
    relative = f"results/pcrl_method_portfolio_v1/{original}"
    _git_blob(entry["commit_sha"], relative, lock_sha256)
    lock = _read(paths["lock"])
    _need(lock.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
          (lock.get("fit_admission") == "B_ONLY" if kind == "FIT"
           else lock.get("schema") == "pcrl-control-v2-lock-v1"),
          "S4 historical lock status or scope differs")
    return lock


def _queue_done(state: dict, unit_id: str, output: Path,
                script_sha256: str) -> None:
    unit = state.get("units", {}).get(unit_id, {})
    spec = unit.get("spec", {})
    argv = spec.get("argv", [])
    script = Path(argv[1]) if len(argv) > 1 else Path("/nonexistent")
    _need(unit.get("status") == "done" and unit.get("returncode") == 0 and
          spec.get("unit_id") == unit_id and spec.get("mode") == "real_data" and
          spec.get("outputs") == [str(output)] and
          script.is_absolute() and script.is_file() and
          _sha(script) == script_sha256 and
          unit.get("output_sha256", {}).get(str(output)) == _sha(output),
          "S4 historical S3 unit/spec/script/output differs")


def _admit(args: argparse.Namespace, argv: list[str]) -> tuple[dict, dict, Path, Path, dict]:
    admission_path = CAMPAIGN / ADMISSION_NAME
    review_path = CAMPAIGN / REVIEW_NAME
    seal_path = CAMPAIGN / SEAL_NAME
    _need(Path(args.admission).resolve() == admission_path and
          Path(args.local_seal).resolve() == seal_path and
          _sha(admission_path) == args.admission_sha256 and
          _sha(seal_path) == args.local_seal_sha256,
          "S4 separate Adult admission or local seal bytes differ")
    admission, review, seal = _read(admission_path), _read(review_path), _read(seal_path)
    _need(admission.get("schema") == "pcrl-s4-verifier-admission-v3" and
          admission.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
          review.get("schema") == "pcrl-s4-verifier-prefit-review-v3" and
          review.get("status") == "PASS" and
          review.get("scope") == "S4_ADULT_REPLAY" and
          seal.get("schema") == "pcrl-s4-verifier-local-seal-v3" and
          seal.get("status") == "PASS" and
          seal.get("scope") == "S4_ADULT_REPLAY" and
          seal.get("admission_sha256") == args.admission_sha256 and
          seal.get("review_path") == REVIEW_NAME and
          seal.get("review_sha256") == _sha(review_path) and
          review.get("admission_sha256") == args.admission_sha256 and
          seal.get("commit_sha") == review.get("commit_sha"),
          "S4 separate Adult review/seal scope differs")
    _git_blob(seal["commit_sha"],
              f"results/pcrl_method_portfolio_v1/{ADMISSION_NAME}",
              args.admission_sha256)
    map_path, schema_path = HERE / MAP_NAME, HERE / SCHEMA_NAME
    _need(_sha(map_path) == admission["parent_map_sha256"] and
          _sha(schema_path) == admission["unit_schema_sha256"],
          "S4 exact parent map or unit schema differs")
    parent_map, schema = _read(map_path), _read(schema_path)
    _need(parent_map.get("schema") == "pcrl-s4-adult-replay-parent-map-v2" and
          schema.get("schema") == "pcrl-s4-adult-replay-unit-schema-v2" and
          schema.get("parent_map_sha256") == _sha(map_path) and
          admission.get("unit_specs") == parent_map.get("unit_specs") and
          set(admission["unit_specs"]) == set(parent_map["unit_specs"]) and
          len(admission["unit_specs"]) == 18,
          "S4 admitted unit map differs")
    fit_shas = {unit["fit_lock_sha256"] for unit in admission["unit_specs"].values()}
    control_shas = {unit["control_v2_lock_sha256"]
                    for unit in admission["unit_specs"].values()}
    _need(set(admission["accepted_fit_lock_sha256"]) == fit_shas and
          set(admission["accepted_control_v2_lock_sha256"]) == control_shas and
          set(admission["historical_seal_by_lock_sha256"]) ==
          fit_shas | control_shas and
          set(admission["source_sha256_by_file"]) == SOURCE_FILES,
          "S4 historical/source allowlist differs")
    for relative, expected in admission["source_sha256_by_file"].items():
        path = (REPO / relative).resolve()
        _need(path.is_relative_to(REPO) and path.is_file() and _sha(path) == expected,
              f"S4 source SHA differs: {relative}")
    _need(admission["runner_sha256"] == _sha(Path(__file__)) and
          admission["scheduler_source_sha256"] ==
          _sha(CAMPAIGN / "shared/compute_queue.py"),
          "S4 worker or scheduler source differs")
    expected_queue = (REPO / admission["queue_root_relative_to_coordinator"]).resolve()
    expected_output = (REPO /
                       admission["private_output_root_relative_to_coordinator"]).resolve()
    expected_cache = (REPO /
                      admission["hcal_cache_root_relative_to_coordinator"]).resolve()
    _need(Path(args.queue_root).resolve() == expected_queue and
          Path(args.cache_root).resolve() == expected_cache and
          Path(args.private_report).resolve() ==
          expected_output / args.unit_id / "REPORT.json" and
          expected_output.is_relative_to(CAMPAIGN) and
          ".private" in expected_output.parts,
          "S4 queue, cache or immutable output root differs")
    unit_record = admission["unit_specs"].get(args.unit_id)
    _need(isinstance(unit_record, dict),
          "S4 Adult unit is not admitted")
    unit = dict(unit_record, unit_id=args.unit_id)
    report = Path(args.private_report).resolve()
    _need(report.name == "REPORT.json" and ".private" in report.parts and
          not report.exists(), "S4 private report target is not fresh")
    state = _read(Path(args.queue_root).resolve() / "QUEUE.json")
    running_unit_gate(state, unit, Path(__file__), argv, report)
    parent_id, control_id = unit["parent_unit_id"], unit["control_v2_unit_id"]
    queue_root = Path(args.queue_root).resolve()
    parent_receipt = queue_root / "units" / parent_id / "RECEIPT.json"
    control_receipt = queue_root / "units" / control_id / "RECEIPT.json"
    parent_name = ("ATTEMPT_INDEX.json" if unit["kind"] == "B_SCREEN"
                   else "ANCHOR_RECEIPT.json")
    control_name = ("REAL_CONTROL_V2_INDEX.json" if unit["kind"] == "B_SCREEN"
                    else "V2_ANCHOR_PROBE_RECEIPT.json")
    parent_sha = (unit["attempt_index_sha256"] if unit["kind"] == "B_SCREEN"
                  else unit["anchor_receipt_sha256"])
    control_sha = (unit["control_v2_index_sha256"] if unit["kind"] == "B_SCREEN"
                   else unit["control_v2_receipt_sha256"])
    parent_root = parent_output_root(parent_receipt,
                                     unit["parent_queue_receipt_sha256"],
                                     parent_sha, parent_name, parent_id)
    control_root = parent_output_root(control_receipt,
                                      unit["control_v2_queue_receipt_sha256"],
                                      control_sha, control_name, control_id)
    fit_lock = _historical_lock(admission, unit["fit_lock_sha256"], "FIT")
    control_lock = _historical_lock(admission,
                                    unit["control_v2_lock_sha256"], "CONTROL_V2")
    _need(admission["historical_seal_by_lock_sha256"][unit["fit_lock_sha256"]]
          ["commit_sha"] == unit["fit_lock_commit_sha"] and
          admission["historical_seal_by_lock_sha256"][unit["control_v2_lock_sha256"]]
          ["commit_sha"] == unit["control_v2_lock_commit_sha"],
          "S4 per-parent historical Git commit differs")
    control_spec = control_lock.get("unit_specs", {}).get(control_id, {})
    control_parent = (control_spec.get("screen_unit_id") if unit["kind"] == "B_SCREEN"
                      else control_spec.get("parent_unit_id"))
    parent_fit_seal = control_lock.get("parent_fit_seal_by_unit", {}).get(parent_id, {})
    expected_attempt = (control_lock.get("parent_attempt_index_sha256", {}).get(
        f"{unit['arm']}:s{unit['seed']}") if unit["kind"] == "B_SCREEN"
        else control_lock.get("parent_anchor_receipt_sha256_by_unit", {}).get(parent_id))
    _need(unit["fit_lock_sha256"] in
          control_lock.get("accepted_fit_lock_sha256", []) and
          control_parent == parent_id and
          control_lock.get("parent_queue_receipt_sha256_by_unit", {}).get(parent_id) ==
          unit["parent_queue_receipt_sha256"] and
          parent_fit_seal.get("lock_sha256") == unit["fit_lock_sha256"] and
          parent_fit_seal.get("commit_sha") == unit["fit_lock_commit_sha"] and
          expected_attempt == parent_sha and
          control_spec.get("seed") == unit["seed"] and
          control_spec.get("arm") == unit["arm"] and
          control_spec.get("stage") == "postrelease_control_v2" and
          (unit["kind"] != "B_SCREEN" or
           control_lock.get("probe_unit_ids", {}).get(
               f"{unit['arm']}:s{unit['seed']}") == control_id),
          "S4 historical V2 lock parent/arm/seed differs")
    parent_script_sha = (fit_lock["families"]["B"]["screen_driver_sha256"]
                         if unit["kind"] == "B_SCREEN"
                         else fit_lock["shared_anchor_runner_sha256"])
    control_script_sha = (control_lock["probe_runner_sha256"]
                          if unit["kind"] == "B_SCREEN"
                          else control_lock["anchor_probe_runner_sha256"])
    _queue_done(state, parent_id, parent_root / parent_name, parent_script_sha)
    _queue_done(state, control_id, control_root / control_name,
                control_script_sha)
    return admission, fit_lock, parent_root, control_root, unit


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--local-seal", type=Path, required=True)
    parser.add_argument("--local-seal-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    queue_root = args.queue_root.resolve()
    state = _read(queue_root / "QUEUE.json")
    own_spec = state.get("units", {}).get(args.unit_id, {}).get("spec", {})
    argv = [own_spec.get("argv", [sys.executable])[0], *sys.argv]
    admission, fit_lock, parent_root, control_root, unit = _admit(args, argv)
    if unit["kind"] == "B_SCREEN":
        from replay_b_values_v3 import verify_b_values
        result = verify_b_values(
            CAMPAIGN, parent_root, control_root, arm=unit["arm"], seed=unit["seed"],
            verifier=admission, unit_id=args.unit_id, fit_lock=fit_lock,
            fit_lock_sha256=unit["fit_lock_sha256"],
            control_v2_lock_sha256=unit["control_v2_lock_sha256"])
    else:
        from replay_anchor_values_v3 import verify_anchor_values
        result = verify_anchor_values(
            CAMPAIGN, parent_root, control_root, args.cache_root.resolve(),
            seed=unit["seed"], verifier=admission, unit_id=args.unit_id,
            fit_lock=fit_lock, fit_lock_sha256=unit["fit_lock_sha256"],
            control_v2_lock_sha256=unit["control_v2_lock_sha256"])
    result["parent_queue_receipt_sha256"] = unit["parent_queue_receipt_sha256"]
    result["control_v2_queue_receipt_sha256"] = unit["control_v2_queue_receipt_sha256"]
    result["runtime"] = {
        "python": sys.version.split()[0], "numpy": version("numpy"),
        "torch": version("torch"), "scikit_learn": version("scikit-learn")}
    _write_once(Path(args.private_report).resolve(), result)
    print(json.dumps({"unit_id": args.unit_id, "status": result["status"],
                      "report_sha256": _sha(Path(args.private_report).resolve())},
                     sort_keys=True))


if __name__ == "__main__":
    main()

"""Queued V2 controls on the frozen feature-track anchor's saved complete views.

This unit fits only the six V2 engineering LR probes. It neither refits the
anchor's U source nor any of its main readers. Person arrays remain private.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from attack_controls_v2 import (SHUFFLE_INNER_SALT, run_control_probes_v2,
                                save_control_probe_v2_custody)
from attack_eval import AttackView
from common_eval import assert_same_people
from compute_queue import Queue
from safe_data_guard import SafeRoleStore, sha_file


CAMPAIGN = Path(__file__).resolve().parent.parent
PRIVATE = Path(__file__).resolve().parent / ".private"
ARM = "FEATURE_REF_RAW83_U"
VIEWS = ("r1", "r2", "pair")
ROLE_FIELDS = {"ATTACK_FIT_NEW": ["features", "sex"],
               "INNER_SELECTION": ["features", "sex"]}
GATES = {"null_exact": 0.5,
         "shuffled_independent_inner_at_most_0.60": 0.60,
         "planted_at_least_0.99": 0.99}


def _arg(argv: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise PermissionError("queued V2 anchor argument missing or duplicated")
    return str(argv[positions[0] + 1])


def _write_json_once(path: Path, obj: dict) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def _running_unit(seed: int, argv_identity: dict[str, str], queue_root: Path,
                  parent_anchor_sha: str) -> tuple[dict, dict]:
    """Check the actual running S3 child and its immutable parent bindings."""
    queue = Queue(queue_root)
    unit_id = f"S2-FEATURE-REF-RAW83-U-S{seed}-V2-PROBE"
    parent_id = f"S2-FEATURE-REF-RAW83-U-S{seed}"
    deadline = time.monotonic() + 5.0
    while True:
        state = queue.status()
        unit = state.get("units", {}).get(unit_id)
        parent = state.get("units", {}).get(parent_id)
        if not unit or unit.get("status") != "running" or not parent or parent.get("status") != "done":
            raise PermissionError("V2 anchor or parent lacks an admitted running/done S3 unit")
        spec = unit.get("spec", {})
        argv = spec.get("argv", [])
        if (spec.get("unit_id") != unit_id or spec.get("mode") != "real_data" or
                spec.get("family") != "SHARED" or spec.get("team") != "SHARED" or
                spec.get("stage") != "postrelease_control_v2" or
                spec.get("data_access") != "ATTACK_INNER_FEATURES_AND_SEX_V2_CONTROL" or
                spec.get("arm") != ARM or spec.get("arm_id") != ARM or
                spec.get("seed") != seed or spec.get("parent_unit_id") != parent_id or
                parent_id not in spec.get("deps", []) or
                spec.get("parent_anchor_receipt_sha256") != parent_anchor_sha or
                spec.get("runner_sha256") != sha_file(Path(__file__)) or
                any(_arg(argv, flag) != value for flag, value in argv_identity.items())):
            raise PermissionError("running S3 unit differs from sealed V2 anchor identity")
        parent_receipt = queue_root / "units" / parent_id / "RECEIPT.json"
        if not parent_receipt.is_file() or sha_file(parent_receipt) != spec.get("parent_receipt_sha256"):
            raise PermissionError("queued V2 parent receipt identity differs")
        child_pgid = unit.get("child_pgid")
        if child_pgid is not None:
            if child_pgid != os.getpgrp():
                raise PermissionError("V2 anchor process group differs from S3 unit")
            return state, spec
        if time.monotonic() >= deadline:
            raise PermissionError("S3 V2 child process group receipt unavailable")
        time.sleep(0.02)


def _sealed_inputs(seed: int, control_lock_sha: str, control_seal_sha: str,
                   manifest_sha: str, parent_anchor_sha: str,
                   queue_state: dict, unit_spec: dict) -> tuple[SafeRoleStore, Path, dict, dict]:
    unit_id = f"S2-FEATURE-REF-RAW83-U-S{seed}-V2-PROBE"
    parent_id = f"S2-FEATURE-REF-RAW83-U-S{seed}"
    expected_output = PRIVATE / f"feature_ref_raw83_u_s{seed}_v2_probe" / "V2_ANCHOR_PROBE_RECEIPT.json"
    if unit_spec.get("outputs") != [str(expected_output)]:
        raise PermissionError("V2 anchor output differs from S3 immutable unit")
    lock_path = CAMPAIGN / "CONTROL_V2_LOCK.json"
    seal_path = CAMPAIGN / "CONTROL_V2_LOCAL_SEAL_RECEIPT.json"
    manifest_path = CAMPAIGN / "DATA_ACCESS_MANIFEST.json"
    if (sha_file(lock_path) != control_lock_sha or sha_file(seal_path) != control_seal_sha or
            sha_file(manifest_path) != manifest_sha):
        raise PermissionError("V2 lock, seal, or admitted manifest SHA differs")
    if not Queue._local_v2_seal_ok(queue_state.get("gate", {})):
        raise PermissionError("committed local V2 control seal absent")
    lock = json.loads(lock_path.read_text())
    expected_spec = {"family": "SHARED", "arm": ARM, "seed": seed,
                     "stage": "postrelease_control_v2", "parent_unit_id": parent_id}
    if (lock.get("schema") != "pcrl-control-v2-lock-v1" or
            lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
            lock.get("adult_probe_permission") is not True or
            lock.get("control_version") != 2 or
            lock.get("unit_specs", {}).get(unit_id) != expected_spec or
            lock.get("probe_unit_ids", {}).get(f"{ARM}:s{seed}") != unit_id or
            lock.get("role_fields") != ROLE_FIELDS or
            lock.get("data_access_manifest_sha256") != manifest_sha or
            lock.get("inner_shuffle_salt") != SHUFFLE_INNER_SALT or
            lock.get("gates") != GATES or
            lock.get("attack_controls_v2_sha256") != sha_file(Path(__file__).with_name("attack_controls_v2.py")) or
            lock.get("anchor_probe_runner_sha256") != sha_file(Path(__file__))):
        raise PermissionError("V2 anchor scientific lock fields differ")
    anchor_root = PRIVATE / f"feature_ref_raw83_u_s{seed}_v1"
    parent_path = anchor_root / "ANCHOR_RECEIPT.json"
    if sha_file(parent_path) != parent_anchor_sha:
        raise PermissionError("parent anchor receipt SHA differs")
    parent = json.loads(parent_path.read_text())
    attack_eval_sha = sha_file(Path(__file__).with_name("attack_eval.py"))
    if (lock.get("attack_eval_sha256") != attack_eval_sha or
            parent.get("attack_eval_sha256") != attack_eval_sha):
        raise PermissionError("V2 shared attack implementation differs from historical fit")
    # Historical parents may have used different fit locks. Bind each to an
    # archived seal/review and the actual committed Git blob, independently of
    # whichever fit lock happens to be live during this V2 probe.
    history = lock.get("parent_fit_seal_by_unit", {}).get(parent_id, {})
    required_history = {"lock_sha256", "commit_sha", "receipt_path",
                        "receipt_sha256", "review_path", "review_sha256"}
    if set(history) != required_history:
        raise PermissionError("historical parent fit-seal map is incomplete")
    if (not re.fullmatch(r"[0-9a-f]{40}", history["commit_sha"]) or
            any(not re.fullmatch(r"[0-9a-f]{64}", history[key]) for key in
                ("lock_sha256", "receipt_sha256", "review_sha256"))):
        raise PermissionError("historical parent fit-seal hashes are malformed")
    for name in ("receipt_path", "review_path"):
        rel = Path(history[name])
        if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
            raise PermissionError("historical fit-seal archive path is not local")
    seal_path = CAMPAIGN / history["receipt_path"]
    review_path = CAMPAIGN / history["review_path"]
    if (sha_file(seal_path) != history["receipt_sha256"] or
            sha_file(review_path) != history["review_sha256"]):
        raise PermissionError("historical fit-seal/review archive SHA differs")
    fit_seal = json.loads(seal_path.read_text())
    fit_review = json.loads(review_path.read_text())
    if (fit_seal.get("status") != "PASS" or fit_review.get("status") != "PASS" or
            fit_seal.get("commit_sha") != history["commit_sha"] or
            fit_seal.get("lock_commit_sha") != history["commit_sha"] or
            fit_seal.get("campaign_lock_sha256") != history["lock_sha256"] or
            fit_seal.get("lock_sha256") != history["lock_sha256"] or
            fit_seal.get("review_sha256") != history["review_sha256"] or
            fit_review.get("lock_sha256") != history["lock_sha256"] or
            fit_review.get("committed_lock_blob_sha256") != history["lock_sha256"] or
            (fit_review.get("lock_commit_sha") or fit_review.get("commit_sha")) != history["commit_sha"] or
            fit_review.get("fit_admission") != fit_seal.get("fit_admission") or
            parent.get("campaign_lock_sha256") != history["lock_sha256"] or
            history["lock_sha256"] not in lock.get("accepted_fit_lock_sha256", [])):
        raise PermissionError("historical parent fit-lock/seal identity differs")
    repo = CAMPAIGN.parents[1]
    relative_lock = (CAMPAIGN / "CAMPAIGN_LOCK.json").relative_to(repo)
    blob = subprocess.run(["git", "-C", str(repo), "show",
                           f"{history['commit_sha']}:{relative_lock}"],
                          capture_output=True, check=False, timeout=10)
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                               history["commit_sha"], "HEAD"],
                              capture_output=True, check=False, timeout=10)
    import hashlib
    if (blob.returncode or ancestor.returncode or
            hashlib.sha256(blob.stdout).hexdigest() != history["lock_sha256"]):
        raise PermissionError("historical parent fit-lock committed blob differs")
    historical_fit_lock = json.loads(blob.stdout)
    if (historical_fit_lock.get("attack_protocol", {}).get("source_module_sha256") != attack_eval_sha or
            historical_fit_lock.get("fit_admission") != fit_seal.get("fit_admission")):
        raise PermissionError("historical parent attacker source differs")
    parent_spec = queue_state["units"][parent_id]["spec"]
    if (_arg(parent_spec.get("argv", []), "--campaign-lock-sha256") !=
            parent.get("campaign_lock_sha256")):
        raise PermissionError("historical parent queued fit-lock identity differs")
    if (parent.get("schema") != "pcrl-feature-ref-raw83-u-anchor-receipt-v1" or
            parent.get("arm") != ARM or parent.get("seed") != seed or
            parent.get("campaign_lock_sha256") not in lock.get("accepted_fit_lock_sha256", []) or
            parent.get("data_access_manifest_sha256") != manifest_sha or
            lock.get("parent_anchor_receipt_sha256_by_unit", {}).get(parent_id) != parent_anchor_sha or
            unit_spec.get("reader_receipt_sha256") != parent.get("reader_receipt_sha256")):
        raise PermissionError("parent anchor/source-reader binding differs")
    for view_name in VIEWS:
        reader_path = anchor_root / f"readers_{view_name}" / "READERS_RECEIPT.json"
        if sha_file(reader_path) != parent["reader_receipt_sha256"][view_name]:
            raise PermissionError("parent reader custody SHA differs before role read")
    store = SafeRoleStore(manifest_path)
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("role files are not S4 admitted")
    for role, fields in ROLE_FIELDS.items():
        for field in fields:
            key = f"{role}:{field}"
            if store.manifest["real_file_specs"][key]["sha256"] != parent["role_file_sha256"][key]:
                raise PermissionError("parent role-file identity differs")
    return store, anchor_root, parent, lock


def _restore_view(directory: Path, reader_receipt: dict, role: str) -> AttackView:
    filename = f"view_{role}.npz"
    path = directory / filename
    if sha_file(path) != reader_receipt["files"][filename]:
        raise PermissionError("saved anchor view file hash differs")
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if not {"row_id", "unit", "x"}.issubset(names) or not names.issubset({
                "row_id", "unit", "x", "x_defense_aware", "cell_key", "x_metadata",
                "x_code_bits", "x_raw_only"}):
            raise PermissionError("saved anchor view members differ")
        arrays = {name: np.array(archive[name], copy=True) for name in names}
    schema = reader_receipt["view_schema"]
    view = AttackView(
        row_id=arrays.pop("row_id"), unit=arrays.pop("unit"), x=arrays.pop("x"),
        components=tuple(schema["components"]),
        wire_components=tuple(schema["wire_components"]),
        component_columns=tuple(tuple(item) for item in schema["component_columns"]),
        wire_constants=tuple(tuple(item) for item in schema["wire_constants"]),
        **arrays).validate()
    if view.content_hashes() != reader_receipt[f"{role}_view_content_sha256"]:
        raise PermissionError("saved anchor view content differs from reader fit")
    return view


def run(seed: int, control_lock_sha: str, control_seal_sha: str,
        manifest_sha: str, parent_anchor_sha: str, queue_root: Path) -> dict:
    argv_identity = {"--seed": str(seed), "--control-lock-sha256": control_lock_sha,
                     "--control-seal-sha256": control_seal_sha,
                     "--data-manifest-sha256": manifest_sha,
                     "--parent-anchor-receipt-sha256": parent_anchor_sha,
                     "--queue-root": str(queue_root)}
    queue_state, unit_spec = _running_unit(seed, argv_identity, queue_root, parent_anchor_sha)
    store, anchor_root, parent, lock = _sealed_inputs(
        seed, control_lock_sha, control_seal_sha, manifest_sha,
        parent_anchor_sha, queue_state, unit_spec)
    output = PRIVATE / f"feature_ref_raw83_u_s{seed}_v2_probe"
    if output.exists():
        raise FileExistsError("V2 anchor probe is write-once")
    output.mkdir(parents=True, mode=0o700)
    labels = {}
    for role, procedure in (("ATTACK_FIT_NEW", "attack"), ("INNER_SELECTION", "screen")):
        features = store.read(role, "features")
        sex = store.read(role, "sex", procedure)
        assert_same_people(features, sex)
        labels[role] = (features, sex)
    reports = {}
    report_sha = {}
    custody_sha = {}
    for view_name in VIEWS:
        directory = anchor_root / f"readers_{view_name}"
        reader_path = directory / "READERS_RECEIPT.json"
        if sha_file(reader_path) != parent["reader_receipt_sha256"][view_name]:
            raise PermissionError("saved anchor reader receipt differs")
        reader = json.loads(reader_path.read_text())
        if reader.get("schema") != "pcrl-portfolio-private-reader-custody-v1":
            raise PermissionError("saved anchor reader schema differs")
        att_view = _restore_view(directory, reader, "att")
        inner_view = _restore_view(directory, reader, "inner")
        for role, restored in (("ATTACK_FIT_NEW", att_view),
                               ("INNER_SELECTION", inner_view)):
            features, sex = labels[role]
            if not np.array_equal(restored.row_id, features["row_id"]) or \
                    not np.array_equal(restored.unit, features["unit"]):
                raise PermissionError("saved anchor view role membership/order differs")
            assert_same_people(features, sex)
        report, artifacts = run_control_probes_v2(
            att_view, labels["ATTACK_FIT_NEW"][1],
            inner_view, labels["INNER_SELECTION"][1], return_artifacts=True)
        reports[view_name] = report
        report_sha[view_name] = _write_json_once(output / f"PROBE_{view_name}.json", report)
        custody = save_control_probe_v2_custody(artifacts, output / f"custody_{view_name}")
        custody_sha[view_name] = custody["private_receipt_sha256"]
    receipt = {"schema": "pcrl-feature-ref-raw83-u-v2-probe-receipt-v1",
               "status": "PASS" if all(r["status"] == "PASS" for r in reports.values()) else "FAIL",
               "arm": ARM, "seed": seed,
               "control_lock_sha256": control_lock_sha,
               "control_seal_sha256": control_seal_sha,
               "data_access_manifest_sha256": manifest_sha,
               "parent_anchor_receipt_sha256": parent_anchor_sha,
               "parent_campaign_lock_sha256": parent["campaign_lock_sha256"],
               "runner_sha256": sha_file(Path(__file__)),
               "attack_controls_v2_sha256": sha_file(Path(__file__).with_name("attack_controls_v2.py")),
               "attack_eval_sha256": sha_file(Path(__file__).with_name("attack_eval.py")),
               "reader_receipt_sha256": parent["reader_receipt_sha256"],
               "probe_report_sha256": report_sha,
               "probe_custody_receipt_sha256": custody_sha,
               "reader_fits": 2 * len(VIEWS),
               "null_scorings": len(VIEWS),
               "original_v1_probe_sha256": parent["real_probe_receipt_sha256"]}
    receipt_sha = _write_json_once(output / "V2_ANCHOR_PROBE_RECEIPT.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError("V2 anchor engineering control gate failed")
    return {"arm": ARM, "seed": seed, "status": "PASS",
            "reader_fits": receipt["reader_fits"], "private_receipt_sha256": receipt_sha}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--control-lock-sha256", required=True)
    parser.add_argument("--control-seal-sha256", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--parent-anchor-receipt-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.seed, args.control_lock_sha256, args.control_seal_sha256,
                     args.data_manifest_sha256, args.parent_anchor_receipt_sha256,
                     args.queue_root)
    except BaseException as exc:
        print("FEATURE_REF_V2_PROBE_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))

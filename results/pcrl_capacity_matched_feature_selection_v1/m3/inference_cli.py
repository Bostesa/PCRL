"""Queued, fail-closed M3 inference worker over M4's immutable private index.

The live S3 unit and completed M4 parent are checked before any private
INNER-derived array is opened. This worker has no role loader or fitting API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from inference import (REFERENCES, RULES, TASKS, VIEWS, _digest, point_endpoints,
                       prepare, selected_control_clauses, simultaneous_inference)
from control_rules import strongest_controls


INDEX_SCHEMA = "pcrl-focused-inference-input-index-v1"
AGGREGATE_SCHEMA = "pcrl-focused-aggregate-endpoints-v1"
SEED_FIELDS = {"npz_file", "npz_sha256", "array_sha256", "fit_plan_receipt_path",
               "fit_plan_receipt_sha256", "rules", "reader_ids_by_rule_view"}
INDEX_FIELDS = {"schema", "status", "phase", "lock_sha256", "m1_manifest_sha256",
                "seeds", "aggregate_endpoints_file", "aggregate_endpoints_sha256",
                "source_receipts"}
LOCAL_READER_IDS = {"LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                    "RAW_LR_C1", "RAW_HGB_15x200"}
COALITION_READER_IDS = (LOCAL_READER_IDS |
                        {f"IGNORE_R2__{name}" for name in LOCAL_READER_IDS} |
                        {f"IGNORE_R1__{name}" for name in LOCAL_READER_IDS})


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_json(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _inside(path, root):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(Path(root).resolve(strict=True)):
        raise ValueError("inference artifact escaped private custody root")
    return resolved


def _decode_seed(entry, index_dir, private_root, seed):
    if not isinstance(entry, dict) or set(entry) != SEED_FIELDS:
        raise ValueError("M4 inference seed-index schema differs")
    npz_name = entry["npz_file"]
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise ValueError("M4 seed NPZ name is not a basename")
    path = _inside(index_dir / npz_name, index_dir)
    if _sha_file(path) != entry["npz_sha256"]:
        raise ValueError("M4 seed NPZ SHA differs from index")
    plan_path = _inside(entry["fit_plan_receipt_path"], private_root)
    expected_plan_path = Path(private_root).resolve(strict=True) / "m2" / "fit" / f"seed_{seed}" / "CONTROL_PLAN.json"
    if plan_path != expected_plan_path:
        raise ValueError("FIT plan is outside its exact seed-specific path")
    if _sha_file(plan_path) != entry["fit_plan_receipt_sha256"]:
        raise ValueError("FIT selection plan file SHA differs from index")
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
            or plan.get("selection_receipt_sha256") != _sha_json({
                key: value for key, value in plan.items()
                if key != "selection_receipt_sha256"
            }) or set(plan.get("rules", {})) != set(RULES)):
        raise ValueError("FIT selection plan content identity differs")
    rule_index = entry["rules"]
    reader_ids = entry["reader_ids_by_rule_view"]
    if set(rule_index) != set(RULES) or set(reader_ids) != set(RULES):
        raise ValueError("M4 index omitted a fixed rule")
    expected_keys = {f"truth__{name}" for name in
                     ("row_id", "unit", "income", "occupation", "sex")}
    expected_keys.update(f"reference__{task}__{ref}"
                         for task in TASKS for ref in REFERENCES)
    rules = {}
    for rule_id in RULES:
        indexed = rule_index[rule_id]
        selected = plan["rules"][rule_id]
        if indexed.get("status") != selected["status"] or \
                indexed.get("pair_id") != selected["pair_id"]:
            raise ValueError("M4 indexed rule differs from FIT plan")
        if indexed["status"] != "SELECTED":
            if reader_ids[rule_id] != {}:
                raise ValueError("invalid rule has registered reader arrays")
            rules[rule_id] = {"status": indexed["status"], "pair_id": None}
            continue
        declared = reader_ids[rule_id]
        if set(declared) != set(VIEWS):
            raise ValueError("M4 reader-view declaration differs")
        for view, count in (("income", 6), ("occupation_group", 6), ("coalition", 18)):
            ids = declared[view]
            if (not isinstance(ids, list) or len(ids) != count or
                    len(set(ids)) != count or any(not isinstance(name, str) or not name
                                                 for name in ids)):
                raise ValueError("M4 fixed reader ID slate differs")
            expected_ids = (COALITION_READER_IDS if view == "coalition"
                            else LOCAL_READER_IDS)
            if set(ids) != expected_ids:
                raise ValueError("M4 fixed reader ID slate differs")
        income_ids = set(declared["income"])
        occupation_ids = set(declared["occupation_group"])
        coalition_ids = set(declared["coalition"])
        if (not {f"IGNORE_R2__{name}" for name in income_ids} <= coalition_ids or
                not {f"IGNORE_R1__{name}" for name in occupation_ids} <= coalition_ids):
            raise ValueError("M4 coalition lacks transported local readers")
        expected_keys.update((f"rule__{rule_id}__prob_income",
                              f"rule__{rule_id}__prob_occupation"))
        expected_keys.update(f"rule__{rule_id}__reader__{view}__{reader_id}"
                             for view in VIEWS for reader_id in declared[view])
        rules[rule_id] = {"status": "SELECTED", "pair_id": indexed["pair_id"],
                          "reader_predictions": {view: {} for view in VIEWS}}
    if not isinstance(entry["array_sha256"], dict) or set(entry["array_sha256"]) != expected_keys:
        raise ValueError("M4 array hash registry differs from fixed wire")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("M4 NPZ array keys differ from fixed wire")
        arrays = {key: np.array(archive[key], copy=True) for key in expected_keys}
    for key, array in arrays.items():
        if _digest(array) != entry["array_sha256"][key]:
            raise ValueError("M4 NPZ per-array SHA differs from index")
    for rule_id in RULES:
        if rules[rule_id]["status"] != "SELECTED":
            continue
        rules[rule_id]["prob_income"] = arrays[f"rule__{rule_id}__prob_income"]
        rules[rule_id]["prob_occupation"] = arrays[f"rule__{rule_id}__prob_occupation"]
        for view in VIEWS:
            for reader_id in reader_ids[rule_id][view]:
                rules[rule_id]["reader_predictions"][view][reader_id] = arrays[
                    f"rule__{rule_id}__reader__{view}__{reader_id}"]
    data = {
        "truth": {name: arrays[f"truth__{name}"] for name in
                  ("row_id", "unit", "income", "occupation", "sex")},
        "references": {task: {ref: arrays[f"reference__{task}__{ref}"]
                              for ref in REFERENCES} for task in TASKS},
        "rules": rules,
    }
    return data, plan


def _verify_fit_source(entry, *, seed, private_root, queue, source_receipts,
                       lock_sha256, manifest_sha256):
    """Bind one indexed plan to its completed, seed-specific M2 FIT unit."""
    fit_path = Path(private_root).resolve(strict=True) / "m2" / "fit" / f"seed_{seed}" / "FIT_RECEIPT.json"
    expected_plan = fit_path.parent / "CONTROL_PLAN.json"
    if (not isinstance(source_receipts.get("fit_receipt_sha256_by_seed"), dict) or
            not fit_path.is_file() or not expected_plan.is_file()):
        raise ValueError("seed-specific M2 FIT custody is missing")
    fit_sha = _sha_file(fit_path)
    if source_receipts["fit_receipt_sha256_by_seed"].get(str(seed)) != fit_sha:
        raise ValueError("M4 FIT source receipt SHA differs")
    fit = json.loads(fit_path.read_text())
    if (fit.get("schema") != "pcrl-focused-fit-receipt-v1" or
            fit.get("status") != "FIT_COMPLETE" or fit.get("unit_id") != f"M2-FIT-S{seed}" or
            fit.get("seed") != seed or fit.get("campaign_lock_sha256") != lock_sha256 or
            fit.get("data_manifest_sha256") != manifest_sha256 or
            fit.get("control_plan_relative_path") !=
            f"m2/fit/seed_{seed}/CONTROL_PLAN.json" or
            fit.get("control_plan_sha256") != entry["fit_plan_receipt_sha256"] or
            _sha_file(expected_plan) != entry["fit_plan_receipt_sha256"] or
            Path(entry["fit_plan_receipt_path"]).resolve(strict=True) != expected_plan):
        raise ValueError("seed-specific M2 FIT receipt/plan identity differs")
    unit = queue.get("units", {}).get(f"M2-FIT-S{seed}")
    if (not isinstance(unit, dict) or unit.get("status") != "DONE" or
            unit.get("output_sha256") != {str(fit_path): fit_sha}):
        raise ValueError("M2 FIT receipt lacks its exact DONE output")


def restore_index(index_path, *, private_root, queue_root, lock_sha256, m1_manifest_sha256,
                  phase="SCREEN"):
    """Decode only exact hash-bound M4 arrays and FIT plan receipts."""
    index_path = _inside(index_path, private_root)
    index = json.loads(index_path.read_text())
    if (set(index) != INDEX_FIELDS or index.get("schema") != INDEX_SCHEMA
            or index.get("status") != "FROZEN" or index.get("phase") != phase
            or index.get("lock_sha256") != lock_sha256
            or index.get("m1_manifest_sha256") != m1_manifest_sha256):
        raise ValueError("M4 inference index identity/status differs")
    if not isinstance(index["source_receipts"], dict) or \
            index["source_receipts"].get("campaign_lock_sha256") != lock_sha256 or \
            index["source_receipts"].get("m1_manifest_sha256") != m1_manifest_sha256:
        raise ValueError("M4 source receipts differ from admitted lock/manifest")
    seed_ids = ("0", "1") if phase == "SCREEN" else ("0", "1", "2")
    if set(index["seeds"]) != set(seed_ids):
        raise ValueError("M4 paired seed roster differs from phase")
    queue = json.loads((Path(queue_root) / "QUEUE.json").read_text())
    data, plans = {}, {}
    for key in seed_ids:
        seed = int(key)
        _verify_fit_source(index["seeds"][key], seed=seed, private_root=private_root,
                           queue=queue, source_receipts=index["source_receipts"],
                           lock_sha256=lock_sha256, manifest_sha256=m1_manifest_sha256)
        data[seed], plans[seed] = _decode_seed(
            index["seeds"][key], index_path.parent, private_root, seed)
    aggregate_name = index["aggregate_endpoints_file"]
    if (not isinstance(aggregate_name, str) or Path(aggregate_name).name != aggregate_name):
        raise ValueError("M4 aggregate endpoint path is not a basename")
    aggregate_path = _inside(index_path.parent / aggregate_name, index_path.parent)
    if _sha_file(aggregate_path) != index["aggregate_endpoints_sha256"]:
        raise ValueError("M4 aggregate endpoint SHA differs from index")
    aggregate = json.loads(aggregate_path.read_text())
    if (aggregate.get("schema") != AGGREGATE_SCHEMA or
            aggregate.get("status") != "DEVELOPMENT_POINT_ONLY" or
            aggregate.get("phase") != phase or
            set(aggregate.get("seeds", {})) != set(seed_ids)):
        raise ValueError("M4 aggregate endpoint schema/seed roster differs")
    return data, plans, aggregate


def compare_point_endpoints(independent, aggregate):
    """Require M3 task/reader points to agree with M4 before selection."""
    for seed, rules in independent.items():
        reported = aggregate["seeds"][str(seed)]
        if set(reported) != set(RULES):
            raise ValueError("M4 aggregate dropped a fixed rule")
        for rule_id, own in rules.items():
            other = reported[rule_id]
            if "status" in own:
                if own["status"] != other.get("status"):
                    raise ValueError("M4 aggregate rule status differs")
            elif other.get("status", "SELECTED") != "SELECTED":
                raise ValueError("M4 aggregate selected rule status differs")
            if own.get("pair_id") != other.get("pair_id"):
                raise ValueError("M4 aggregate pair differs")
            if "status" in own:
                continue
            for identity in ("income_mask_id", "occupation_mask_id", "union_sha256",
                             "recipient_capacity_signatures", "union_capacity_signature"):
                if own[identity] != other.get(identity):
                    raise ValueError("M4 aggregate release identity/capacity differs")
            for task in TASKS:
                for metric in ("accuracy", "nll", "brier"):
                    if abs(own["task"][task]["candidate"][metric] -
                           other["task"][task]["candidate"][metric]) > 1e-12:
                        raise ValueError("M4 candidate task endpoint differs")
                for reference in REFERENCES:
                    expected = own["task"][task]["references"][reference]
                    actual = other["task"][task]["references"][reference]
                    for metric in ("accuracy", "nll", "brier",
                                   "accuracy_difference", "nll_excess", "brier_excess"):
                        if abs(expected[metric] - actual[metric]) > 1e-12:
                            raise ValueError("M4 utility reference endpoint differs")
                    if expected["point_eligible"] is not actual["point_eligible"]:
                        raise ValueError("M4 utility point status differs")
            for view in VIEWS:
                if abs(own["recovery"][view]["primary_envelope_auc"] -
                       other["recovery"][view]["primary_envelope_auc"]) > 1e-12:
                    raise ValueError("M4 full-reader envelope differs")


def _write_once_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    temp = path.parent / ("." + path.name + ".tmp." + str(os.getpid()))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "queue-root", "lock-path", "index-path", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    phase = ("SCREEN" if args.unit_id == "M3-INFERENCE-SCREEN" else
             "FINALIST" if args.unit_id == "M3-INFERENCE-FINALIST" else None)
    if phase is None:
        raise ValueError("M3 inference unit ID differs from registered phases")
    study = Path(args.lock_path).resolve(strict=True).parent
    private = study / ".private"
    output = Path(args.output).resolve()
    expected_output = private / "m3" / "inference" / (phase + ".json")
    index_path = Path(args.index_path).resolve()
    expected_index = private / "m4" / "endpoints" / phase / "INFERENCE_INDEX.json"
    if output != expected_output or index_path != expected_index:
        raise ValueError("M3 inference input/output path differs from lock route")
    from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock_path,
                               "simultaneous_inference", expected_output)
    parent_id = "M4-ENDPOINTS-" + phase
    if parent_id not in spec["deps"]:
        raise ValueError("M4 endpoint parent missing from inference dependencies")
    queue = json.loads((Path(args.queue_root) / "QUEUE.json").read_text())
    parent = queue["units"].get(parent_id)
    if (not isinstance(parent, dict) or parent.get("status") != "DONE" or
            parent.get("output_sha256", {}).get(str(expected_index)) != _sha_file(expected_index)):
        raise ValueError("M4 frozen inference index is not a DONE parent output")
    lock_sha = _sha_file(args.lock_path)
    lock = json.loads(Path(args.lock_path).read_text())
    seed_hex = lock.get("inference", {}).get("bootstrap_seed_hex")
    if not _is_sha(seed_hex):
        raise ValueError("locked 32-byte bootstrap seed missing")
    manifest_sha = _sha_file(study / "m1" / "DATA_ACCESS_MANIFEST.json")
    data, plans, aggregate = restore_index(
        index_path, private_root=private, queue_root=args.queue_root, lock_sha256=lock_sha,
        m1_manifest_sha256=manifest_sha, phase=phase)
    seeds = (0, 1) if phase == "SCREEN" else (0, 1, 2)
    prepared = prepare(data, plans, expected_seeds=seeds)
    independent = point_endpoints(prepared, plans)
    compare_point_endpoints(independent, aggregate)
    strongest = strongest_controls(plans, independent)
    interval = simultaneous_inference(data, plans, bytes.fromhex(seed_hex),
                                      expected_seeds=seeds)
    selected = selected_control_clauses(interval, strongest)
    _write_once_json(output, {
        "schema": "pcrl-focused-m3-inference-output-v1", "phase": phase,
        "lock_sha256": lock_sha, "m4_inference_index_sha256": _sha_file(index_path),
        "m4_aggregate_endpoint_sha256":
        _sha_file(index_path.parent / "AGGREGATE_ENDPOINTS.json"),
        "strongest_controls": strongest, "simultaneous_inference": interval,
        "selected_control_clauses": selected,
        "status": "DEVELOPMENT_PENDING_M5_INDEPENDENT_REPLAY",
    })


if __name__ == "__main__":
    main()

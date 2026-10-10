"""R03-only independent six-cell historical D0 INNER utility replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .independent_a_source import (BANK_FIELDS, REFERENCE_FIELDS, TOKEN_FIELDS,
                                   bound_json, bound_npz, role_reference_path,
                                   routed_cells, sha_array, sha_file)
from .independent_replay import task_metrics
from .replay_a_prefit import _role_npz, queue_guard


ARMS = {"LOCAL": "hcal_local_d0", "SEQ12": "hcal_seq12_d0", "SEQ21": "lra_seq21"}
ROLE = "INNER_SELECTION"


def _metric_parity(actual: dict, expected: dict) -> float:
    if (actual["class_counts"] != expected["class_counts"]
            or actual["rows"] != expected["rows"]):
        raise ValueError("D0 task support or rows differ")
    maximum = 0.0
    for metric in ("accuracy", "nll", "brier"):
        difference = abs(actual[metric] - expected[metric])
        maximum = max(maximum, difference)
    for value, target in zip(actual["per_class_recall"], expected["per_class_recall"]):
        if value is None or target is None:
            if value != target:
                raise ValueError("D0 empty-class recall differs")
        else:
            maximum = max(maximum, abs(value - target))
    if maximum > 1e-12:
        raise ValueError("independent D0 task metric differs from R02 receipt")
    return maximum


def _metric(prob: np.ndarray, truth: np.ndarray) -> dict:
    score = task_metrics(prob, truth)
    return {"accuracy": score["accuracy"], "nll": score["nll"],
            "brier": score["brier"], "rows": len(truth),
            "class_counts": score["class_counts"],
            "per_class_recall": score["class_recall"]}


def run(args: argparse.Namespace) -> dict:
    seal = Path(args.seal).resolve(strict=True)
    queue_guard(seal, args.seal_sha256)
    lock = bound_json(seal, args.seal_sha256)
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("D0 independent replay requires effective local seal")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = bound_json(manifest_path, args.role_manifest_sha256)
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise ValueError("historical D0 repair role manifest differs")
    original = bound_json(Path(args.source_audit), args.source_audit_sha256)
    if (original.get("status") != "READ_ONLY_HASH_AND_MEMBERSHIP_PASS"
            or original["source_schema_sha256"] != manifest["source_schema_sha256"]
            or args.source_audit_sha256 != lock["source_files_sha256"]["r01/SOURCE_AUDIT.json"]):
        raise ValueError("D0 historical source not bound to effective source audit")
    audit = bound_json(Path(args.historical_audit), args.historical_audit_sha256)
    if (audit.get("schema") != "pcrl-r01-a-historical-score-source-audit-v1"
            or audit.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED"):
        raise ValueError("historical D0 source map differs")
    portfolio = Path(args.portfolio_package).resolve(strict=True)
    cache_root = Path(args.cache_root).resolve(strict=True)
    if sha_file(portfolio / "DATA_ACCESS_MANIFEST.json") != manifest["portfolio_manifest_sha256"]:
        raise ValueError("historical D0 portfolio admission differs")
    report = {"schema": "pcrl-r04-d0-inner-utility-independent-replay-v1",
              "status": "SIX_D0_INNER_UTILITY_REPLAY_PASS",
              "scope": "six role-only INNER development cells; no assessment or attacker fit",
              "effective_seal_sha256": args.seal_sha256,
              "r01_historical_audit_sha256": args.historical_audit_sha256,
              "role_manifest_sha256": args.role_manifest_sha256,
              "arms": {}}
    labels = {task: _role_npz(manifest, manifest_path, ROLE, task)
              for task in ("income", "occupation")}
    for arm, key in ARMS.items():
        arm_report = {}
        for seed in (0, 1):
            entry = audit["d0"][key][f"s{seed}"]
            receipt_path = (portfolio / entry["receipt_path"]).resolve(strict=True)
            source_receipt = bound_json(receipt_path, entry["receipt_sha256"])
            if (source_receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
                    or source_receipt.get("seed") != seed
                    or source_receipt.get("release_id") != entry["release_id"]
                    or source_receipt.get("preprocessing_only") is not True
                    or source_receipt.get("source_binding_sha256") != entry["source_binding_sha256"]):
                raise ValueError("D0 source bank admission differs")
            bank = {}
            for recipient in (1, 2):
                name = f"bank_r{recipient}.npz"
                source = entry["banks"][f"r{recipient}"]
                if source_receipt["private_output_sha256"][name] != source["sha256"]:
                    raise ValueError("D0 bank hash not admitted")
                bank[recipient] = bound_npz(portfolio / source["path"], source["sha256"], BANK_FIELDS)
                b = bank[recipient]
                if (b["probability"].dtype != np.float64
                        or b["probability"].shape != (source["tokens"], 2 if recipient == 1 else 6)
                        or not np.array_equal(b["token_id"], np.arange(source["tokens"]))
                        or not np.array_equal(np.argmax(b["probability"], axis=1), b["token_class"])
                        or np.max(np.abs(b["probability"].sum(axis=1) - 1)) > 1e-8):
                    raise ValueError("D0 fixed score bank identity differs")
            token_name = f"tokens_{ROLE}.npz"
            role_record = entry["roles"][ROLE]
            if source_receipt["private_output_sha256"][token_name] != role_record["sha256"]:
                raise ValueError("D0 INNER tokens not admitted")
            tokens = bound_npz(portfolio / role_record["path"], role_record["sha256"], TOKEN_FIELDS)
            references = _role_npz(manifest, manifest_path, ROLE,
                                   f"references_s{seed}", REFERENCE_FIELDS)
            reference_path = role_reference_path(manifest_path, manifest, ROLE, seed)
            row_unit = sha_array(np.column_stack((tokens["row_id"], tokens["unit"])))
            if (sha_file(reference_path) != role_record["source_reference_sha256"]
                    or row_unit != role_record["row_unit_pair_sha256"]
                    or not np.array_equal(tokens["row_id"], references["row_id"])
                    or not np.array_equal(tokens["unit"], references["unit"])
                    or any(not np.array_equal(tokens["row_id"], labels[task]["row_id"])
                           or not np.array_equal(tokens["unit"], labels[task]["unit"])
                           for task in labels)):
                raise ValueError("D0 INNER row/person alignment differs")
            fine_record = entry["source_paths_sha256"]["fine.json"]
            fine = bound_json(cache_root / fine_record["path"], fine_record["sha256"])
            q = {}
            d = {}
            for recipient in (1, 2):
                b = bank[recipient]
                cell = routed_cells(references[f"U0_{recipient}"], fine[f"fine{recipient}"])
                code = b["fine_to_hcal_token"][cell]
                if (not np.array_equal(cell, tokens[f"cell_{recipient}"])
                        or not np.array_equal(code, tokens[f"token_{recipient}"])
                        or not np.array_equal(b["token_class"][code], references[f"d_{recipient}"])):
                    raise ValueError("D0 independently routed fine cell/public token differs")
                q[recipient] = b["probability"][code]
                d[recipient] = b["token_class"][code]
            unit_id = f"R02-D0-UTILITY-{arm}-S{seed}"
            aggregate_path = seal.parent / "r02" / f"{unit_id}_RECEIPT.json"
            aggregate = bound_json(aggregate_path, args.receipt_sha256[unit_id])
            if (aggregate.get("unit_id") != unit_id or aggregate.get("seed") != seed
                    or aggregate.get("arm") != arm or aggregate.get("status") != "COMPLETE"
                    or aggregate.get("r01_d0_source_audit_sha256") != args.historical_audit_sha256
                    or aggregate.get("ordered_row_unit_pair_sha256") != row_unit
                    or aggregate["probability_array_sha256"] !=
                       {"r1": sha_array(q[1]), "r2": sha_array(q[2])}
                    or aggregate["decision_array_sha256"] !=
                       {"r1": sha_array(d[1]), "r2": sha_array(d[2])}):
                raise ValueError("D0 aggregate input identity differs")
            maximum = 0.0
            seed_result = {"unit_id": unit_id, "aggregate_receipt_sha256":
                           args.receipt_sha256[unit_id], "tasks": {}}
            all_pass = True
            for task, recipient in (("income", 1), ("occupation", 2)):
                y = labels[task][task]
                candidate = _metric(q[recipient], y)
                saved = aggregate["utility"]["tasks"][task]
                maximum = max(maximum, _metric_parity(candidate, saved["candidate"]))
                task_result = {"candidate": {name: candidate[name]
                                             for name in ("accuracy", "nll", "brier")},
                               "comparisons": {}}
                task_pass = True
                for ref in ("U0", "Ucal"):
                    baseline = _metric(references[f"{ref}_{recipient}"], y)
                    maximum = max(maximum, _metric_parity(baseline, saved["references"][ref]))
                    delta = {name: candidate[name] - baseline[name]
                             for name in ("accuracy", "nll", "brier")}
                    for name, value in delta.items():
                        maximum = max(maximum, abs(value - saved["comparisons"][ref]["delta"][name]))
                    passes = delta["accuracy"] >= -0.010 and delta["nll"] <= 0.010 and delta["brier"] <= 0.005
                    if passes != saved["comparisons"][ref]["point_eligible"]:
                        raise ValueError("D0 reference utility point gate differs")
                    task_pass &= passes
                    task_result["comparisons"][ref] = {"delta": delta, "point_eligible": passes}
                if task_pass != saved["point_eligible_both_refs"]:
                    raise ValueError("D0 task both-reference gate differs")
                all_pass &= task_pass
                seed_result["tasks"][task] = task_result
            if (all_pass != aggregate["utility"]["point_eligible_all_tasks_refs"]
                    or maximum > 1e-12):
                raise ValueError("D0 aggregate utility result differs")
            seed_result["point_eligible"] = all_pass
            seed_result["maximum_absolute_metric_difference"] = maximum
            arm_report[f"s{seed}"] = seed_result
        arm_report["both_seed_point_eligible"] = all(arm_report[f"s{s}"]["point_eligible"]
                                                    for s in (0, 1))
        report["arms"][arm] = arm_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "role-manifest", "role-manifest-sha256",
                 "source-audit", "source-audit-sha256", "historical-audit",
                 "historical-audit-sha256", "portfolio-package", "cache-root", "receipt-map",
                 "receipt-map-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    args.receipt_sha256 = bound_json(Path(args.receipt_map), args.receipt_map_sha256)
    report = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("D0 independent utility receipt already exists")
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()

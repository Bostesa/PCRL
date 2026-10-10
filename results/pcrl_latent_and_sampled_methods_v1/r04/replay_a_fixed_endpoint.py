"""R03-only independent fixed-cell A endpoint and K=6,885 interval replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_a_family_s1 import independent_ids
from .independent_a_intervals import NAMES, STOCHASTIC, _Release, independent_a_intervals
from .independent_a_source import bound_json, sha_array, sha_file
from .replay_a_endpoint import (_cell, _compare_csv, _reader_ids, _truth,
                                METRICS, TASKS, TOLERANCE, VIEWS)
from .replay_a_prefit import queue_guard


FIXED = {"T_STAR", "P_STAR", "U0", "Ucal", "D0_LOCAL", "D0_SEQ12",
         "D0_SEQ21", "A_IDENTITY", "A_CLASS_ONLY"}
ENDPOINT = "R02-A-K6885-FIXED-ENDPOINT-001-R2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _registered_cells(plan: dict) -> dict[tuple[str, int, int], str]:
    cells = {}
    for unit_id, entry in plan.items():
        if entry.get("kind") not in ("attack", "reference_attack", "diagnostic_attack"):
            continue
        arm = entry.get("arm_id", entry.get("reference_id"))
        key = (arm, entry.get("seed"), entry.get("realization"))
        wanted = (0, 1, 2) if arm in STOCHASTIC else (0,)
        if (key in cells or arm not in NAMES or key[1] not in (0, 1)
                or key[2] not in wanted or entry.get("output_id") != unit_id):
            raise ValueError("A sealed attack registry differs")
        cells[key] = unit_id
    expected = {(arm, seed, draw) for arm in NAMES for seed in (0, 1)
                for draw in ((0, 1, 2) if arm in STOCHASTIC else (0,))}
    require(len(cells) == 58 and set(cells) == expected,
            "A sealed 58-cell attack registry differs")
    return cells


def _point_table(records: dict) -> dict[tuple[str, ...], tuple[float, float | None]]:
    table = {}
    for seed in (0, 1):
        truth = records[seed]["truth"]
        count = len(truth["row_id"])
        weight = np.ones(count, np.float64)
        for arm, draws in records[seed]["releases"].items():
            scores = []
            for draw, raw in sorted(draws.items()):
                score = _Release(raw, truth, truth["sex"]).score(weight, float(count))
                scores.append(score)
                for task in TASKS:
                    for metric in METRICS:
                        table[(str(seed), arm, str(draw), task, metric)] = (
                            float(score["task"][task][metric]), None)
                for view in VIEWS:
                    table[(str(seed), arm, str(draw), view, "auc_envelope")] = (
                        float(score["auc"][view]), None)
            for scope, metric in ((*[(task, metric) for task in TASKS for metric in METRICS],
                                   *[(view, "auc_envelope") for view in VIEWS])):
                values = [score["task"][scope][metric] if metric != "auc_envelope"
                          else score["auc"][scope] for score in scores]
                table[(str(seed), arm, "mean", scope, metric)] = (
                    float(np.mean(values)),
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return table


def _fixed_cell(path: Path, info: dict, truth: dict, arm: str) -> dict:
    """Check the exact NPZ member allowlist before materializing any member."""
    expected = {"row_id", "unit", "prob_1", "prob_2"}
    for view in VIEWS:
        expected.update(f"reader__{view}__{reader}" for reader in
                        _reader_ids(arm, view))
    require(info.get("npz_keys") == sorted(expected),
            "fixed A normalized cell manifest contains an unregistered member")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected,
                "fixed A normalized cell archive contains an unregistered member")
    return _cell(path, info, truth, arm)


def run(args: argparse.Namespace) -> dict:
    campaign = Path(args.campaign).resolve(strict=True)
    lock_path = campaign / "CAMPAIGN_LOCK.json"
    queue_guard(lock_path, args.original_lock_sha256)
    producer = bound_json(Path(args.producer), args.producer_sha256)
    analysis = bound_json(Path(args.analysis_amendment), args.analysis_amendment_sha256)
    verification = bound_json(Path(args.verification_amendment),
                              args.verification_amendment_sha256)
    receipt = bound_json(Path(args.endpoint_receipt), args.endpoint_receipt_sha256)
    completion = bound_json(campaign / "r03/A_FIXED_ATTACK_COMPLETION_RECEIPT.json",
                            args.completion_sha256)
    independent_attacks = {}
    for seed in (0, 1):
        report = bound_json(
            campaign / "r04" / f"A_FIXED_ATTACKS_S{seed}_INDEPENDENT_REPLAY_V4.json",
            getattr(args, f"independent_attack_s{seed}_sha256"))
        require(report.get("status") == "NINE_FIXED_ATTACKS_INDEPENDENT_REPLAY_PASS" and
                report.get("seed") == seed and len(report.get("cells", [])) == 9 and
                report.get("completion_sha256") == args.completion_sha256,
                "prior independent fixed attack replay differs")
        independent_attacks[seed] = report
    family = campaign / "r02/A_FAMILY_ORDER_S1.json"
    require(producer.get("status") == "LOCALLY_SEALED_A_STAGE" and
            producer.get("original_lock_sha256") == args.original_lock_sha256 and
            analysis.get("status") == "LOCALLY_SEALED_A_FIXED_ENDPOINT_RECONSTRUCTION_CORRECTION" and
            analysis.get("authorized_unit_ids") == [ENDPOINT] and
            analysis.get("producer_amendment_sha256") == args.producer_sha256 and
            verification.get("schema") == "r04-a-fixed-endpoint-verification-proposal-v1" and
            verification.get("status") == "LOCALLY_SEALED_R04_A_FIXED_ENDPOINT_REPLAY" and
            verification.get("authorized_unit_ids") == [args.unit_id] and
            verification.get("producer_amendment_sha256") == args.producer_sha256 and
            verification.get("endpoint_analysis_amendment_sha256") ==
                args.analysis_amendment_sha256 and
            verification.get("endpoint_receipt_sha256") == args.endpoint_receipt_sha256 and
            sha_file(family) == producer["interval_family"]["manifest_sha256"],
            "fixed A independent endpoint seals/family differ")
    require(receipt.get("schema") == "pcrl-a-fixed-endpoint-aggregate-v1" and
            receipt.get("status") == "INCOMPLETE_FITTED_ARM_SOURCES" and
            receipt.get("family_complete") is False and
            receipt.get("full_conjunction_status") == "INCOMPLETE_NOT_ESTIMABLE" and
            receipt.get("registered_attack_instances") == 58 and
            receipt.get("fixed_attack_instances") == 18 and
            receipt.get("unavailable_fitted_attack_instances") == 40 and
            receipt.get("clauses") == 6885 and
            receipt.get("available_clause_count") == 1161 and
            receipt.get("missing_clause_count") == 5724 and
            receipt.get("source_manifest_sha256") == sha_file(family),
            "fixed A endpoint public availability differs")
    root = Path(args.endpoint_private_root).resolve(strict=True)
    binding = bound_json(root / "A_SOURCE_BINDING.json", receipt["private_binding_sha256"])
    require(binding.get("schema") == "pcrl-a-k6885-private-fixed-source-binding-v1" and
            binding.get("status") == "INCOMPLETE_FITTED_ARM_SOURCES" and
            binding.get("a_amendment_sha256") == args.producer_sha256 and
            binding.get("analysis_amendment_sha256") == args.analysis_amendment_sha256 and
            binding.get("role_manifest_sha256") == args.role_manifest_sha256 and
            binding.get("registered_attack_instances") == 58 and
            binding.get("fixed_family_manifest_sha256") == sha_file(family),
            "fixed A endpoint normalized source binding differs")
    registered = _registered_cells(producer["unit_plan"])
    expected_fixed = {key for key in registered if key[0] in FIXED}
    require(len(expected_fixed) == 18 and set(NAMES) >= FIXED and
            len(completion["cells"]) == 18,
            "fixed A registered cell/family scope differs")
    mapped = {cell["physical_unit_id"]: cell["scientific_cell_id"]
              for cell in completion["cells"]}
    require(len(mapped) == 18 and
            binding.get("physical_to_original_attack_units") == mapped,
            "fixed A physical/scientific completion map differs")
    for seed, report in independent_attacks.items():
        require({cell["physical_unit_id"]: cell["attack_index_sha256"]
                 for cell in report["cells"]} ==
                {cell["physical_unit_id"]: cell["attack_index_sha256"]
                 for cell in completion["cells"] if cell["scientific_cell_id"].endswith(f"-S{seed}")},
                "prior independent fixed attack custody differs")
    truth = _truth(Path(args.role_manifest).resolve(strict=True),
                   args.role_manifest_sha256)
    require(sha_array(np.column_stack((truth["row_id"], truth["unit"]))) ==
            binding["ordered_inner_row_group_sha256"],
            "fixed A INNER task/SEX people differ")
    records = {seed: {"truth": truth, "releases": {}} for seed in (0, 1)}
    seen = set()
    for entry in binding["fixed_cells"]:
        key = (entry["arm"], entry["seed"], entry["draw"])
        physical = entry["physical_attack_unit_id"]
        require(key in expected_fixed and key not in seen and
                entry["original_attack_unit_id"] == registered[key] and
                mapped.get(physical) == registered[key] and
                entry["draw"] == 0 and
                entry["attack_index_sha256"] == next(
                    cell["attack_index_sha256"] for cell in completion["cells"]
                    if cell["physical_unit_id"] == physical),
                "fixed A normalized attack instance differs")
        seen.add(key)
        path = root / "cells" / f"{key[0]}_s{key[1]}_d0.npz"
        info = entry["normalized_private_cell"]
        require(Path(info["private_file"]).resolve(strict=True) == path,
                "fixed A normalized cell path differs")
        records[key[1]]["releases"].setdefault(key[0], {})[0] = _fixed_cell(
            path, info, truth, key[0])
        attack_root = Path(entry["private_attack_root"]).resolve(strict=True)
        require(sha_file(attack_root / "ATTACK_ATTEMPT_INDEX.json") ==
                entry["attack_index_sha256"] and
                sha_file(attack_root / "common/FROZEN_COMPLETE.json") ==
                entry["reader_freeze_sha256"],
                "fixed A saved attack/freeze changed")
        for view in VIEWS:
            require(sha_file(attack_root / "common" / view / "readers/READERS_RECEIPT.json") ==
                    entry["reader_receipt_sha256"][view] and
                    sha_file(attack_root / "common" / view / "controls/CUSTODY_RECEIPT.json") ==
                    entry["v2_control_receipt_sha256"][view],
                    "fixed A frozen reader/control source changed")
    require(seen == expected_fixed, "fixed A endpoint omitted or invented cell")
    unavailable = set()
    fit_rows = {row["unit_id"]: row for row in binding["fit_status_rows"]}
    require(len(fit_rows) == 16 and
            sum(row["category"] == "FINALIZED_NO_FEASIBLE_POLICY"
                for row in fit_rows.values()) == 14 and
            sum(row["category"] == "FAILED_AFTER_INFEASIBLE_STAGE1"
                for row in fit_rows.values()) == 2,
            "fixed A unavailable FIT reason inventory differs")
    for entry in binding["unavailable_fitted_cells"]:
        key = (entry["arm"], entry["seed"], entry["draw"])
        require(key in registered and key not in expected_fixed and
                key not in unavailable and entry["unit_id"] == registered[key] and
                producer["unit_plan"][entry["unit_id"]]["fit_unit_id"] ==
                entry["fit_unit_id"] and entry["category"] ==
                fit_rows[entry["fit_unit_id"]]["category"],
                "fixed A unavailable cell reason differs")
        unavailable.add(key)
    require(unavailable == set(registered) - expected_fixed and len(unavailable) == 40 and
            binding["deterministic_aliases"] ==
            {arm: {"draw1": "draw0", "draw2": "draw0"} for arm in FIXED},
            "fixed A missing/aliased registry differs")
    public_json = Path(args.endpoint_json).resolve(strict=True)
    public_csv = Path(args.endpoint_csv).resolve(strict=True)
    require(sha_file(public_json) == receipt["interval_json_sha256"] and
            sha_file(public_csv) == receipt["point_csv_sha256"],
            "fixed A public result bytes changed")
    point_difference = _compare_csv(public_csv, _point_table(records))
    own = independent_a_intervals(records)
    published = json.loads(public_json.read_text())
    require(own["clause_count"] == 6885 and not own["family_complete"] and
            own["available_clause_count"] == 1161 and
            own["missing_clause_count"] == 5724 and
            len(published["clauses"]) == 6885 and
            tuple(row["name"] for row in published["clauses"]) == independent_ids() and
            own["critical_family"] == published["critical_family"],
            "fixed A 6,885-clause family/status differs")
    maximum = {key: 0.0 for key in ("point_margin", "se", "lower", "upper")}
    for ours, theirs in zip(own["clauses"], published["clauses"]):
        require(ours["name"] == theirs["name"] and
                ours["status"] == theirs["status"] and
                ours["numerical_se_status"] == theirs["numerical_se_status"],
                "fixed A ordered clause status differs")
        for key in maximum:
            left, right = ours[key], theirs[key]
            if left is None:
                require(right is None, "fixed A unavailable clause has a bound")
            else:
                difference = abs(left - right)
                maximum[key] = max(maximum[key], difference)
                require(difference <= TOLERANCE,
                        "fixed A independently recomputed clause differs")
    critical_difference = abs(own["critical_95"] - published["critical_95"])
    require(critical_difference <= TOLERANCE, "fixed A critical value differs")
    return {"schema": "r04-a-fixed-endpoint-independent-replay-v1",
            "status": "FIXED_1161_OF_6885_ENDPOINT_REPLAY_PASS",
            "scope": "18 fixed INNER saved q/readers; 40 fitted cells unavailable",
            "producer_sha256": args.producer_sha256,
            "analysis_amendment_sha256": args.analysis_amendment_sha256,
            "verification_amendment_sha256": args.verification_amendment_sha256,
            "endpoint_receipt_sha256": args.endpoint_receipt_sha256,
            "independent_fixed_attack_report_sha256_by_seed": {
                "s0": args.independent_attack_s0_sha256,
                "s1": args.independent_attack_s1_sha256},
            "private_binding_sha256": receipt["private_binding_sha256"],
            "fixed_attack_cells": len(seen), "unavailable_fitted_cells": len(unavailable),
            "available_clauses": own["available_clause_count"],
            "not_estimable_clauses": own["missing_clause_count"],
            "family_complete": False,
            "maximum_point_csv_difference": point_difference,
            "maximum_clause_differences": maximum,
            "critical_95_difference": critical_difference}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "original-lock-sha256", "producer", "producer-sha256",
                 "analysis-amendment", "analysis-amendment-sha256",
                 "verification-amendment", "verification-amendment-sha256",
                 "endpoint-private-root", "endpoint-receipt", "endpoint-receipt-sha256",
                 "endpoint-json", "endpoint-csv", "role-manifest",
                 "role-manifest-sha256", "completion-sha256",
                 "independent-attack-s0-sha256", "independent-attack-s1-sha256",
                 "unit-id", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("fixed A endpoint replay output exists")
    result = run(args)
    with output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

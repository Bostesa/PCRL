"""Data-free independent missing-arm identities for the frozen A family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_a_family_s1 import independent_ids
from .independent_a_source import sha_file


EXPECTED_WORKER_SHA = "c2d7ed22a25e8c09405a31f641b209fcb69973c090b7df439984ec11ed488b3d"
EXPECTED_TEST_SHA = "fb8e3464c16d567e4e71b2b6ef30005e8dfd28a18df3b195bdb4b832f1b5f6a3"
SUMMARIES = ("s0_d0", "s0_d1", "s0_d2", "s1_d0", "s1_d1", "s1_d2",
             "s0_mean", "s1_mean", "mean")


def missing_ids(names: tuple[str, ...], nulls: set[tuple[str, int]]) -> set[str]:
    missing = set()
    for name in names:
        parts = name.split(":")
        summary, kind = parts[:2]
        if summary not in SUMMARIES or kind not in ("utility", "arm_pair", "ref_pair"):
            raise ValueError("A frozen clause ID malformed")
        arms = (parts[2], parts[3]) if kind == "arm_pair" else (parts[2],)
        if summary == "mean":
            seeds = (0, 1)
        elif summary.startswith("s0_"):
            seeds = (0,)
        else:
            seeds = (1,)
        if any((arm, seed) in nulls for arm in arms for seed in seeds):
            missing.add(name)
    return missing


def run(campaign: Path) -> dict:
    family_path = campaign / "r02/A_FAMILY_ORDER_S1.json"
    worker_path = campaign / "r02/a_missing_endpoint.py"
    test_path = campaign / "r02/test_a_missing_endpoint.py"
    family = json.loads(family_path.read_text())
    names = independent_ids()
    if (len(names) != 6885 or tuple(family["ordered_clause_ids"]) != names or
            sha_file(worker_path) != EXPECTED_WORKER_SHA or
            sha_file(test_path) != EXPECTED_TEST_SHA):
        raise ValueError("A fixed family or missing-endpoint source changed")
    one = missing_ids(names, {("A_JOINT_STOCH", 0)})
    two = missing_ids(names, {("A_JOINT_STOCH", 0),
                              ("A_CENTRAL_PRODUCT", 0)})
    if (len(one) != 450 or len(two) != 885 or not one <= two or
            len(set(names) - one) != 6435 or len(set(names) - two) != 6000):
        raise ValueError("A fixed-K missing-arm clause count differs")
    by_summary = {summary: sum(name.startswith(summary + ":") for name in one)
                  for summary in SUMMARIES}
    if by_summary != {summary: (90 if summary in
                               ("s0_d0", "s0_d1", "s0_d2", "s0_mean", "mean") else 0)
                      for summary in SUMMARIES}:
        raise ValueError("A missing s0 arm propagated to wrong summaries")
    return {"schema": "pcrl-r04-a-missing-endpoint-source-audit-v1",
            "status": "FIXED_6885_MISSING_ARM_SOURCE_GO",
            "scope": "data-free source and ordered-ID check; final null set follows FIT drain",
            "family_sha256": sha_file(family_path),
            "missing_endpoint_worker_sha256": EXPECTED_WORKER_SHA,
            "missing_endpoint_test_sha256": EXPECTED_TEST_SHA,
            "fixed_family_clause_count": 6885,
            "joint_s0_only": {"not_estimable": len(one), "estimable": 6435,
                              "not_estimable_by_summary": by_summary},
            "joint_and_central_s0": {"not_estimable": len(two), "estimable": 6000},
            "missing_clause_interpretation":
                "INCOMPLETE_MISSING_ARM means not estimable; retain ID and null bounds",
            "complete_common_criterion_claim_available": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.campaign.resolve(strict=True))
    if args.public_receipt.exists():
        raise FileExistsError(args.public_receipt)
    args.public_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

"""Independently enumerate the prospective A seven-reference finite family."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from .independent_a_source import sha_file


ARMS = ("A_JOINT_STOCH", "A_TASK_STOCH", "A_TASK_DET", "A_JOINT_DET",
        "A_CENTRAL_PRODUCT", "A_LOCAL", "A_SEQ12", "A_SEQ21",
        "A_IDENTITY", "A_CLASS_ONLY")
REFERENCES = ("T_STAR", "P_STAR", "U0", "Ucal", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21")
SUMMARIES = ("s0_d0", "s0_d1", "s0_d2", "s1_d0", "s1_d1", "s1_d2",
             "s0_mean", "s1_mean", "mean")
VIEWS = ("r1", "r2", "pair")


def independent_ids() -> tuple[str, ...]:
    base = []
    for arm in ARMS:
        for task in ("income", "occupation"):
            for ref in REFERENCES:
                for metric in ("accuracy", "nll", "brier"):
                    base.append(f"utility:{arm}:{task}:{ref}:{metric}")
    for left, right in combinations(ARMS, 2):
        for view in VIEWS:
            base.append(f"arm_pair:{left}:{right}:{view}:auc")
    for arm in ARMS:
        for ref in REFERENCES:
            for view in VIEWS:
                base.append(f"ref_pair:{arm}:{ref}:{view}:auc")
    if len(base) != 765 or len(set(base)) != len(base):
        raise AssertionError("A S1 base registry count/collision differs")
    return tuple(f"{summary}:{name}" for summary in SUMMARIES for name in base)


def run(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest).resolve(strict=True)
    if sha_file(manifest_path) != args.manifest_sha256:
        raise ValueError("A S1 family manifest bytes changed")
    manifest = json.loads(manifest_path.read_text())
    old_path = Path(args.old_manifest).resolve(strict=True)
    old_sha = sha_file(old_path)
    d0_path = Path(args.d0_replay).resolve(strict=True)
    pstar_path = Path(args.pstar_replay).resolve(strict=True)
    d0_sha, pstar_sha = sha_file(d0_path), sha_file(pstar_path)
    d0 = json.loads(d0_path.read_text())
    pstar = json.loads(pstar_path.read_text())
    if (manifest.get("schema") != "pcrl-a-finite-family-order-s1-v1"
            or manifest.get("status") != "PRE_FIT_PROSPECTIVE_S1"
            or manifest.get("supersedes_manifest_sha256") != old_sha
            or manifest.get("arms") != list(ARMS)
            or manifest.get("references") != list(REFERENCES)
            or manifest.get("summaries") != list(SUMMARIES)
            or manifest.get("clause_count") != 6885
            or manifest.get("base_counts") != {
                "utility_10x7x2x3": 420, "new_arm_pairs_45x3": 135,
                "new_arm_vs_ref_10x7x3": 210, "total": 765}
            or manifest.get("ordered_clause_ids") != list(independent_ids())
            or len(set(manifest["ordered_clause_ids"])) != 6885
            or d0.get("status") != "SIX_D0_INNER_UTILITY_REPLAY_PASS"
            or pstar.get("status") != "PSTAR_ATT_INNER_SOURCE_REPLAY_PASS"):
        raise ValueError("A S1 prospective ordered family or admission differs")
    utility_sha = manifest["d0_prefit_utility_receipt_sha256"]
    for source_name, arm in (("D0_LOCAL", "LOCAL"), ("D0_SEQ12", "SEQ12"),
                             ("D0_SEQ21", "SEQ21")):
        for seed in (0, 1):
            if utility_sha[source_name][f"s{seed}"] != d0["arms"][arm][f"s{seed}"]["aggregate_receipt_sha256"]:
                raise ValueError("A S1 D0 reference receipt differs from independent replay")
        if not any(d0["arms"][arm][f"s{seed}"]["point_eligible"] for seed in (0, 1)):
            raise ValueError("A S1 D0 control did not meet prefit escalation rule")
    interval = manifest["simultaneous_interval"]
    if (interval.get("replicates") != 2048
            or interval.get("confidence") != 0.95
            or "all 6885" not in interval.get("full_family", "")
            or "retain all 6885" not in interval.get("partial_family", "")
            or "fixed-K" not in interval.get("partial_family", "")
            or "unavailable clauses incomplete" not in interval.get("partial_family", "")
            or "same draw indices" not in interval.get("seed_and_grand_means", "")):
        raise ValueError("A S1 fixed-K/missing-reference interval policy differs")
    return {"schema": "pcrl-r04-a-s1-family-independent-audit-v1",
            "status": "PROSPECTIVE_6885_FAMILY_ORDER_PASS",
            "manifest_sha256": args.manifest_sha256,
            "superseded_manifest_sha256": old_sha,
            "d0_independent_replay_sha256": d0_sha,
            "pstar_independent_replay_sha256": pstar_sha,
            "summaries": list(SUMMARIES), "base_counts": manifest["base_counts"],
            "ordered_clause_count": 6885,
            "all_ids_unique_and_exact_order": True,
            "fixed_k_missing_source_rule": True,
            "d0_escalation_by_arm": {
                name: {"s0": d0["arms"][arm]["s0"]["point_eligible"],
                       "s1": d0["arms"][arm]["s1"]["point_eligible"]}
                for name, arm in (("D0_LOCAL", "LOCAL"), ("D0_SEQ12", "SEQ12"),
                                  ("D0_SEQ21", "SEQ21"))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "old-manifest", "d0-replay",
                 "pstar-replay", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("A S1 independent family audit receipt exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

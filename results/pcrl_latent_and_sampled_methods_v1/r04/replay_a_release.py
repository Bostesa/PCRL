"""R03-only independent A sampled code, complete wire and task replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .independent_a_inputs import read_admitted_a_role, verify_policy_source
from .independent_a_law import load_independent_policy
from .independent_a_source import bound_json
from .independent_a_stream import replay_sampled_release
from .independent_replay import task_metrics, task_point_gate
from .replay_a_prefit import queue_guard


def run(args: argparse.Namespace) -> dict:
    lock_path = Path(args.original_lock).resolve(strict=True)
    queue_guard(lock_path, args.original_lock_sha256)
    lock = bound_json(lock_path, args.original_lock_sha256)
    amendment = bound_json(Path(args.a_amendment), args.a_amendment_sha256)
    if (lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or amendment.get("status") != "LOCALLY_SEALED_A_STAGE"
            or amendment.get("original_lock_sha256") != args.original_lock_sha256):
        raise PermissionError("A release replay requires effective A stage seal")
    directory = Path(args.release_dir).resolve(strict=True)
    release = bound_json(directory / "RELEASE_RECEIPT.json", args.release_receipt_sha256)
    policy = load_independent_policy(Path(args.fit_dir), args.policy_receipt_sha256)
    fit = bound_json(Path(args.fit_dir) / "FIT_RECEIPT.json", args.fit_receipt_sha256)
    arm, seed, role, draw = (policy.receipt["arm_id"], policy.receipt["source_seed"],
                             release["role"], release["realization"])
    unit_id = f"A-RELEASE-{arm}-S{seed}-{'ATT' if role == 'ATTACK_FIT_NEW' else 'INNER'}-R{draw}"
    unit = amendment["unit_plan"].get(unit_id, {})
    if (role not in ("ATTACK_FIT_NEW", "INNER_SELECTION")
            or unit.get("kind") != "release" or unit.get("role") != role
            or unit.get("arm_id") != arm or unit.get("seed") != seed
            or unit.get("realization") != draw or unit.get("fit_unit_id") != fit["unit_id"]
            or policy.receipt["fit_receipt_sha256"] != args.fit_receipt_sha256
            or fit["a_amendment_sha256"] != args.a_amendment_sha256):
        raise ValueError("A sampled release not in sealed unit/selected FIT chain")
    source = read_admitted_a_role(
        seed=seed, role=role, portfolio=Path(args.portfolio_package),
        manifest_path=Path(args.role_manifest),
        manifest_sha256=args.role_manifest_sha256,
        source_audit_path=Path(args.source_audit),
        source_audit_sha256=args.source_audit_sha256,
        labels=role == "INNER_SELECTION")
    source_parity = verify_policy_source(policy, source)
    public_root = Path(args.public_stream_manifest)
    if (amendment["realizations"]["stream_root_hash_manifest_sha256"] !=
            args.public_stream_manifest_sha256):
        raise ValueError("A root manifest differs from effective amendment")
    custody = replay_sampled_release(
        policy, directory=directory,
        expected_receipt_sha256=args.release_receipt_sha256,
        root_path=Path(args.private_stream_roots),
        public_stream_manifest_path=public_root,
        expected_stream_manifest_sha256=args.public_stream_manifest_sha256,
        row_id=source.row_id, unit=source.unit,
        score1=source.score1, score2=source.score2,
        cell1=source.cell1, cell2=source.cell2,
        expected_source_identity=source.source_identity)
    result = {"schema": "pcrl-r04-a-sampled-release-independent-replay-v1",
              "status": "SAMPLED_RELEASE_WIRE_REPLAY_PASS",
              "arm_id": arm, "source_seed": seed, "role": role, "realization": draw,
              "original_lock_sha256": args.original_lock_sha256,
              "a_amendment_sha256": args.a_amendment_sha256,
              "fit_receipt_sha256": args.fit_receipt_sha256,
              "policy_receipt_sha256": args.policy_receipt_sha256,
              "release_receipt_sha256": args.release_receipt_sha256,
              "public_stream_manifest_sha256": args.public_stream_manifest_sha256,
              "role_manifest_sha256": args.role_manifest_sha256,
              "source_audit_sha256": args.source_audit_sha256,
              "source_parity": source_parity, "custody": custody}
    if role == "INNER_SELECTION":
        with np.load(directory / "release_codes.npz", allow_pickle=False) as archive:
            codes = {r: np.array(archive[f"code_r{r}"], copy=True) for r in (1, 2)}
        actual, references = {}, {}
        for task, r in (("income", 1), ("occupation", 2)):
            y = source.labels[task]
            p = policy.arrays[f"bank{r}_probability"][codes[r]]
            d = policy.arrays[f"bank{r}_decision"][codes[r]]
            actual[task] = task_metrics(p, y, decision=d)
            references[task] = {name: task_metrics(source.references[f"{name}_{r}"], y)
                                for name in ("U0", "Ucal")}
        result["task"] = {"actual_sampled": actual, "references": references,
                          "point_gate": task_point_gate(actual, references)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "original-lock-sha256", "a-amendment",
                 "a-amendment-sha256", "fit-dir", "fit-receipt-sha256",
                 "policy-receipt-sha256", "release-dir", "release-receipt-sha256",
                 "private-stream-roots", "public-stream-manifest",
                 "public-stream-manifest-sha256", "portfolio-package",
                 "role-manifest", "role-manifest-sha256", "source-audit",
                 "source-audit-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("A sampled release replay receipt already exists")
    output.write_text(json.dumps(result, sort_keys=True, indent=2,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

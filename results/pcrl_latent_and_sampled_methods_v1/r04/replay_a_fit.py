"""R03-only independent selected A FIT expected-law replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .independent_a_fit import compare_fit_metrics
from .independent_a_inputs import read_admitted_a_role, verify_policy_source
from .independent_a_law import load_independent_policy
from .independent_a_source import bound_json, bound_npz, sha_file
from .replay_a_prefit import queue_guard


def run(args: argparse.Namespace) -> dict:
    lock_path = Path(args.original_lock).resolve(strict=True)
    queue_guard(lock_path, args.original_lock_sha256)
    lock = bound_json(lock_path, args.original_lock_sha256)
    amendment_path = Path(args.a_amendment).resolve(strict=True)
    amendment = bound_json(amendment_path, args.a_amendment_sha256)
    if (lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or amendment.get("status") != "LOCALLY_SEALED_A_STAGE"
            or amendment.get("original_lock_sha256") != args.original_lock_sha256):
        raise PermissionError("A FIT replay requires effective A stage seal")
    fit_dir = Path(args.fit_dir).resolve(strict=True)
    fit_path = fit_dir / "FIT_RECEIPT.json"
    fit = bound_json(fit_path, args.fit_receipt_sha256)
    if (fit.get("a_amendment_sha256") != args.a_amendment_sha256
            or fit.get("status") != "SELECTED_FIT_POLICY"):
        raise ValueError("A selected FIT receipt not in effective amendment")
    arm, seed = fit["arm_id"], fit["source_seed"]
    unit_plan = amendment["unit_plan"].get(fit["unit_id"], {})
    if (unit_plan.get("arm_id") != arm or unit_plan.get("seed") != seed
            or unit_plan.get("kind") != "fit"):
        raise ValueError("A selected FIT unit absent from sealed plan")
    policy = load_independent_policy(fit_dir, args.policy_receipt_sha256)
    if (policy.receipt["fit_receipt_sha256"] != args.fit_receipt_sha256
            or policy.receipt["fit_unit_id"] != fit["unit_id"]):
        raise ValueError("A selected policy/fit receipt binding differs")
    source = read_admitted_a_role(
        seed=seed, role="OSF_DEFENSE_FIT", portfolio=Path(args.portfolio_package),
        manifest_path=Path(args.role_manifest),
        manifest_sha256=args.role_manifest_sha256,
        source_audit_path=Path(args.source_audit),
        source_audit_sha256=args.source_audit_sha256, labels=True)
    source_parity = verify_policy_source(policy, source)
    if (amendment["role_bindings"]["source_binding_sha256_by_seed"][f"s{seed}"] !=
            source.source_binding_sha256 or
            amendment["role_bindings"]["fit_row_unit_pair_sha256"] !=
            source.row_unit_pair_sha256):
        raise ValueError("A FIT source identity differs from sealed amendment")
    protected = unit_plan.get("objective_kind") == "protected_bayes"
    if not protected and unit_plan.get("objective_kind") != "task_rate":
        raise ValueError("A FIT objective kind differs")
    sex = None
    if protected:
        manifest_path = Path(args.role_manifest).resolve(strict=True)
        manifest = bound_json(manifest_path, args.role_manifest_sha256)
        key = "OSF_DEFENSE_FIT:sex"
        path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
        record = bound_npz(path, manifest["role_file_sha256"][key],
                           {"row_id", "unit", "sex"})
        if (not np.array_equal(record["row_id"], source.row_id)
                or not np.array_equal(record["unit"], source.unit)):
            raise ValueError("A FIT SEX truth differs from task/source rows")
        sex = record["sex"]
    result = compare_fit_metrics(policy, source, fit, sex)
    result.update({"schema": "pcrl-r04-a-selected-fit-independent-replay-v1",
                   "original_lock_sha256": args.original_lock_sha256,
                   "a_amendment_sha256": args.a_amendment_sha256,
                   "fit_receipt_sha256": args.fit_receipt_sha256,
                   "policy_receipt_sha256": args.policy_receipt_sha256,
                   "role_manifest_sha256": args.role_manifest_sha256,
                   "source_audit_sha256": args.source_audit_sha256,
                   "source_parity": source_parity,
                   "protected_fit_truth_used": protected})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "original-lock-sha256", "a-amendment",
                 "a-amendment-sha256", "fit-dir", "fit-receipt-sha256",
                 "policy-receipt-sha256", "portfolio-package", "role-manifest",
                 "role-manifest-sha256", "source-audit", "source-audit-sha256",
                 "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("A selected FIT replay receipt already exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

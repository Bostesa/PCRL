"""R03-only independent replay of a null-selected A FIT checkpoint search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_diagnostic_q_parity import _actions, _canonical
from .independent_a_fit import _reference_caps, expected_fit_metrics
from .independent_a_inputs import read_admitted_a_role
from .independent_a_law import IndependentAPolicy, _action_hash, _bank_hash
from .independent_a_source import bound_json, bound_npz, sha_file
from .replay_a_prefit import queue_guard


TASK_KEYS = ("r1_nll", "r1_brier", "r2_nll", "r2_brier")
CHECKPOINTS = tuple(range(0, 301, 25))
STARTS = ("identity_biased", "uniform_logit")
TOLERANCE = 2e-8


def _source_policy_arrays(source, fit: dict) -> tuple[dict, tuple, tuple, dict]:
    arrays, action_rows, hashes = {}, {}, {}
    for recipient, bank in ((1, source.bank1), (2, source.bank2)):
        probability, decision, old_to_global = _canonical(
            bank["probability"], bank["token_id"], bank["token_class"])
        actions = _actions(probability, decision, old_to_global,
                           bank["input_class"], bank["fine_to_hcal_token"])
        if (_bank_hash(probability, decision) !=
                fit["bank_sha256_by_recipient"][f"r{recipient}"] or
                _action_hash(actions) !=
                fit["action_sha256_by_recipient"][f"r{recipient}"]):
            raise ValueError("null FIT canonical public bank/action differs")
        arrays[f"bank{recipient}_probability"] = probability
        arrays[f"bank{recipient}_decision"] = decision
        arrays[f"bank{recipient}_old_to_global"] = old_to_global
        action_rows[recipient] = actions
        hashes[f"r{recipient}"] = {"bank": _bank_hash(probability, decision),
                                   "actions": _action_hash(actions)}
    return arrays, action_rows[1], action_rows[2], hashes


def _identity_guard(code: np.ndarray, sex: np.ndarray) -> float:
    if (code.shape != sex.shape or not np.isin(sex, (0, 1)).all()):
        raise ValueError("null FIT identity guard inputs differ")
    values = 0
    for name in np.unique(code):
        mask = code == name
        values += max(int(np.count_nonzero(mask & (sex == 0))),
                      int(np.count_nonzero(mask & (sex == 1))))
    return values / len(sex)


def run(args: argparse.Namespace) -> dict:
    lock_path = Path(args.original_lock).resolve(strict=True)
    queue_guard(lock_path, args.original_lock_sha256)
    amendment = bound_json(Path(args.a_amendment).resolve(strict=True),
                           args.a_amendment_sha256)
    if (amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != args.original_lock_sha256):
        raise PermissionError("null FIT replay requires effective A seal")
    fit_dir = Path(args.fit_dir).resolve(strict=True)
    fit = bound_json(fit_dir / "FIT_RECEIPT.json", args.fit_receipt_sha256)
    arm, seed, fit_unit = fit["arm_id"], fit["source_seed"], fit["unit_id"]
    plan = amendment["unit_plan"].get(fit_unit, {})
    if (fit["status"] != "COMPLETE_NO_FEASIBLE_POLICY" or fit["selected"] is not None or
            fit["a_amendment_sha256"] != args.a_amendment_sha256 or
            plan.get("arm_id") != arm or plan.get("seed") != seed or
            plan.get("objective_kind") != "protected_bayes" or
            arm not in ("A_JOINT_STOCH", "A_TASK_STOCH") or
            (fit_dir / "POLICY_RECEIPT.json").exists() or
            (fit_dir / "policy.npz").exists()):
        raise ValueError("null FIT receipt or selected policy state differs")
    source = read_admitted_a_role(
        seed=seed, role="OSF_DEFENSE_FIT", portfolio=Path(args.portfolio_package),
        manifest_path=Path(args.role_manifest),
        manifest_sha256=args.role_manifest_sha256,
        source_audit_path=Path(args.source_audit),
        source_audit_sha256=args.source_audit_sha256, labels=True)
    if (fit["source_binding_sha256"] != source.source_binding_sha256 or
            fit["row_unit_pair_sha256"] != source.row_unit_pair_sha256 or
            amendment["role_bindings"]["source_binding_sha256_by_seed"][f"s{seed}"] !=
                source.source_binding_sha256 or
            amendment["role_bindings"]["fit_row_unit_pair_sha256"] !=
                source.row_unit_pair_sha256):
        raise ValueError("null FIT admitted role source differs")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = bound_json(manifest_path, args.role_manifest_sha256)
    key = "OSF_DEFENSE_FIT:sex"
    sex_path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
    sex_record = bound_npz(sex_path, manifest["role_file_sha256"][key],
                           {"row_id", "unit", "sex"})
    if (not np.array_equal(sex_record["row_id"], source.row_id) or
            not np.array_equal(sex_record["unit"], source.unit) or
            sex_record["sex"].dtype != np.int64):
        raise ValueError("null FIT protected truth order differs")
    sex = sex_record["sex"]
    caps = _reference_caps(source)
    if set(fit["caps"]) != set(caps) or max(
            abs(fit["caps"][name] - caps[name]) for name in caps) > 1e-10:
        raise ValueError("null FIT U0/Ucal task caps differ")
    arrays, actions1, actions2, source_hashes = _source_policy_arrays(source, fit)
    guards = {}
    for recipient, old in ((1, source.token1), (2, source.token2)):
        code = arrays[f"bank{recipient}_old_to_global"][old]
        guards[f"r{recipient}_bayes"] = _identity_guard(code, sex)
    required_caps = {**caps, **guards}
    if set(fit["starts"]) != set(STARTS):
        raise ValueError("null FIT two-start plan differs")
    checked, claimed_feasible = 0, 0
    max_metric_error = max_residual_error = max_objective_error = 0.0
    minimum_r2_nll_excess = float("inf")
    minimum_max_residual = float("inf")
    for start in STARTS:
        record = fit["starts"][start]
        checkpoints = record["checkpoints"]
        if (record["completed"] is not True or len(checkpoints) != len(CHECKPOINTS) or
                tuple(item["update"] for item in checkpoints) != CHECKPOINTS):
            raise ValueError("null FIT start/checkpoint schedule incomplete")
        for item in checkpoints:
            path = fit_dir / item["filename"]
            if (path.name != item["filename"] or
                    path.name != f"{start}_u{item['update']:03d}.npz" or
                    sha_file(path) != item["file_sha256"]):
                raise ValueError("null FIT checkpoint file custody differs")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"w1", "b1", "w2", "b2"}:
                    raise ValueError("null FIT checkpoint weight members differ")
                weights = {name: np.array(archive[name], copy=True)
                           for name in archive.files}
            if any(value.dtype != np.float64 or not np.isfinite(value).all()
                   for value in weights.values()):
                raise ValueError("null FIT checkpoint weight values invalid")
            policy = IndependentAPolicy(
                {"law_kind": "stochastic_joint_v1"}, {**arrays, **weights},
                actions1, actions2)
            metrics = expected_fit_metrics(policy, source, sex)
            if set(item["metrics"]) != set(metrics):
                raise ValueError("null FIT reported metric slate differs")
            max_metric_error = max(max_metric_error, *(
                abs(metrics[name] - item["metrics"][name]) for name in metrics))
            residual = {name: metrics[name] - required_caps[name]
                        for name in required_caps}
            if set(item["residual"]) != set(residual):
                raise ValueError("null FIT reported residual slate differs")
            max_residual_error = max(max_residual_error, *(
                abs(residual[name] - item["residual"][name]) for name in residual))
            max_objective_error = max(max_objective_error,
                                      abs(metrics["protected_objective"] - item["objective"]))
            feasible = all(value <= 0 for value in residual.values())
            if feasible != item["feasible"] or feasible:
                raise ValueError("null FIT feasibility differs or a checkpoint passed")
            claimed_feasible += int(item["feasible"])
            minimum_r2_nll_excess = min(minimum_r2_nll_excess, residual["r2_nll"])
            minimum_max_residual = min(minimum_max_residual, max(residual.values()))
            checked += 1
    if (checked != 26 or claimed_feasible != 0 or
            max(max_metric_error, max_residual_error, max_objective_error) > TOLERANCE):
        raise ValueError("null FIT independent checkpoint replay differs")
    return {"schema": "pcrl-r04-a-null-fit-checkpoint-replay-v1",
            "status": "NULL_SELECTED_FIT_26_CHECKPOINT_REPLAY_PASS",
            "scope": "saved bounded search only; no global infeasibility claim",
            "arm_id": arm, "source_seed": seed, "fit_unit_id": fit_unit,
            "a_amendment_sha256": args.a_amendment_sha256,
            "fit_receipt_sha256": args.fit_receipt_sha256,
            "source_binding_sha256": source.source_binding_sha256,
            "checkpoint_count": checked, "feasible_checkpoint_count": claimed_feasible,
            "selected_policy_absent": True,
            "canonical_bank_action_hashes": source_hashes,
            "max_metric_error": max_metric_error,
            "max_residual_error": max_residual_error,
            "max_objective_error": max_objective_error,
            "minimum_r2_nll_excess": minimum_r2_nll_excess,
            "minimum_max_residual": minimum_max_residual,
            "fit_caps": caps, "identity_local_bayes_guards": guards,
            "rows": len(source.row_id), "groups": len(np.unique(source.unit))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "original-lock-sha256", "a-amendment",
                 "a-amendment-sha256", "fit-dir", "fit-receipt-sha256",
                 "portfolio-package", "role-manifest", "role-manifest-sha256",
                 "source-audit", "source-audit-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("null FIT independent replay receipt already exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

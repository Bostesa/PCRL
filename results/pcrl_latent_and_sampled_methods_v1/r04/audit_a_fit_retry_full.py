"""Read-only independent prefit audit of the A FIT retry successor seal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SHA = "526f509f8077f2ee1c412f6d290df939966a6b81f1ae5a87e9639d4bd317271f"
PRIOR_DRAFT_SHA = "e0d0f20e15f8dc35d4e5a5a33f7f6eab05a92e0652de49042c9ada8c3e9311f3"
SCIENCE_FIELDS = ("attack_plan", "bank_action_sha256_by_seed", "control_recipe",
                  "frozen_candidate_sha256", "input_scope", "interval_family",
                  "optimizer_recipe", "original_lock_sha256", "owner",
                  "realizations", "role_bindings", "schema", "stage_scope",
                  "status", "utility_caps")
CHANGED_FIELDS = {"amendment_kind", "cpu_budget", "dated_utc",
                  "engineering_correction", "implementation_source_sha256",
                  "independent_prefit_audit_sha256", "previous_a_amendment_sha256",
                  "scope_note", "source_sha256", "unit_plan"}
ADDED_SOURCE = {"r03/A_FIRST_FIT_PREFIT_FAILURE_RECEIPT.json",
                "r03/A_FIT_RENDERER_EXACT_SEAL_CORRECTION.json",
                "r03/A_FIT_RETRY_GRAPH_PREFLIGHT.json",
                "r03/R12_A_PROVENANCE_SYNTHETIC_VERIFICATION.json",
                "r03/R13_A_PROVENANCE_SYNTHETIC_VERIFICATION.json",
                "r03/R17_A_Q_RETRY_GUARD_VERIFICATION.json",
                "r03/test_a_specs.py", "r04/A_FIT_RETRY_PLAN_INDEPENDENT_AUDIT.json",
                "r17/A_Q_RETRY_GUARD_SETUP_RECEIPT.json",
                "sampled_channel_v1/attacks/test_run_q_preprocess.py",
                "sampled_channel_v1/design/A_FIT_RETRY_UNIT_PLAN_PROPOSAL_001.json"}
CHANGED_IMPL = {"r03/a_fit_specs_v2.py", "r03/a_specs.py",
                "sampled_channel_v1/attacks/run_q_preprocess.py",
                "sampled_channel_v1/controls/run_admitted_controls.py",
                "sampled_channel_v1/controls/run_diagnostic_release.py",
                "sampled_channel_v1/impl/admitted_fit.py",
                "sampled_channel_v1/impl/run_reference.py",
                "sampled_channel_v1/impl/run_release.py"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(campaign: Path, role_manifest: Path, queue_state: Path) -> dict:
    old_path = campaign / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_PATH_CORRECTION.json"
    path = campaign / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json"
    retry_path = campaign / "sampled_channel_v1/design/A_FIT_RETRY_UNIT_PLAN_PROPOSAL_001.json"
    retry_audit_path = campaign / "r04/A_FIT_RETRY_PLAN_INDEPENDENT_AUDIT.json"
    family_path = campaign / "r02/A_FAMILY_ORDER_S1.json"
    lock_path = campaign / "CAMPAIGN_LOCK.json"
    old = json.loads(old_path.read_text())
    new_bytes = path.read_bytes()
    new = json.loads(new_bytes)
    retry = json.loads(retry_path.read_text())
    retry_audit = json.loads(retry_audit_path.read_text())
    queue_bytes = queue_state.read_bytes()
    queue = json.loads(queue_bytes)
    if (sha(path) != EXPECTED_SHA or
            new_bytes.count(b'"forecast_residual_seconds": 960,') != 1 or
            hashlib.sha256(new_bytes.replace(
                b'"forecast_residual_seconds": 960,',
                b'"forecast_residual_seconds": 1222,')).hexdigest() != PRIOR_DRAFT_SHA or
            set(old) != set(new) or
            {key for key in old if old[key] != new[key]} != CHANGED_FIELDS or
            any(old[key] != new[key] for key in SCIENCE_FIELDS) or
            new["schema"] != "pcrl-latent-sampled-A-stage-amendment-v1" or
            new["status"] != "LOCALLY_SEALED_A_STAGE" or
            new["input_scope"] != "ADMITTED_DEVELOPMENT_ROLES_ONLY" or
            new["previous_a_amendment_sha256"] != sha(old_path) or
            new["original_lock_sha256"] != sha(lock_path) or
            new["role_bindings"]["repair_manifest_sha256"] != sha(role_manifest) or
            new["interval_family"]["manifest_sha256"] != sha(family_path) or
            new["interval_family"]["ordered_clause_count"] != 6885 or
            new["interval_family"]["scorer_sha256"] !=
                "DEFERRED_TO_DATED_ENDPOINT_AMENDMENT"):
        raise ValueError("A retry full draft science, role, seal or byte identity differs")
    if (new["unit_plan"] != retry["unit_plan"] or
            new["engineering_correction"]["retry_plan_canonical_json_sha256"] !=
                retry["unit_plan_canonical_json_sha256"] or
            retry_audit["status"] != "INERT_RETRY_PLAN_ID_ONLY_PASS" or
            retry_audit["proposal_sha256"] != sha(retry_path) or
            new["source_sha256"].get(
                "r04/A_FIT_RETRY_PLAN_INDEPENDENT_AUDIT.json") != sha(retry_audit_path)):
        raise ValueError("A successor does not bind independently audited retry graph")
    if (set(new["source_sha256"]) - set(old["source_sha256"]) != ADDED_SOURCE or
            set(old["source_sha256"]) - set(new["source_sha256"]) or
            {key for key in old["source_sha256"] if
             old["source_sha256"][key] != new["source_sha256"][key]} !=
                {"r03/test_a_fit_specs_v2.py"} or
            set(new["implementation_source_sha256"]) !=
                set(old["implementation_source_sha256"]) or
            {key for key in old["implementation_source_sha256"] if
             old["implementation_source_sha256"][key] !=
             new["implementation_source_sha256"][key]} != CHANGED_IMPL):
        raise ValueError("A successor source delta differs from pre-role guard fixes")
    for field, count in (("source_sha256", 62),
                         ("implementation_source_sha256", 43)):
        if len(new[field]) != count:
            raise ValueError("A successor source inventory count differs")
        for relative, expected in new[field].items():
            if sha(campaign / relative) != expected:
                raise ValueError(f"A successor source changed: {field}/{relative}")
    plan = new["unit_plan"]
    cpu = sum(unit["cpu_seconds"] for unit in plan.values())
    wall = sum(unit["wall_seconds"] for unit in plan.values())
    if (len(plan) != 220 or cpu != 14240 or wall != 14240 or
            any("ASSESSMENT" in str(unit) for unit in plan.values()) or
            new["cpu_budget"]["all_unit_cpu_and_wall_reserves_including_80_releases"] != cpu):
        raise ValueError("A retry plan count, cap or development role differs")
    budget = new["cpu_budget"]
    old_budget = old["cpu_budget"]
    if (any(budget[key] != value for key, value in old_budget.items()
            if key != "forecast_residual_seconds") or
            set(budget) - set(old_budget) != {
                "fit_retry_forecast_total_with_2300_margin_seconds",
                "fit_retry_prior_charge_ceiling_seconds"} or
            budget["fit_retry_prior_charge_ceiling_seconds"] != 500 or
            budget["fit_retry_forecast_total_with_2300_margin_seconds"] != 17040 or
            budget["forecast_residual_seconds"] != 960 or
            budget["inference_and_failure_margin_breakdown_seconds"] != {
                "R02_K6885_inference_prospective": 1800,
                "failed_attempt_margin": 500}):
        raise ValueError("A successor retry budget changed beyond prior charge ceiling")
    used_a = float(queue["used_by_stage_s"]["a"])
    used_global = float(queue["used_cpu_s"])
    projected_a = used_a + cpu + 2300
    projected_global = used_global + cpu + 2300
    if (queue.get("closed") or queue["seal"]["sha256"] != sha(lock_path) or
            used_a > 500 or projected_a > 18000 or projected_global > 72000 or
            sha(path) != EXPECTED_SHA or
            hashlib.sha256(queue_state.read_bytes()).hexdigest() !=
                hashlib.sha256(queue_bytes).hexdigest()):
        raise ValueError("A successor queue or budget changed during prefit audit")
    return {"schema": "pcrl-r04-a-fit-retry-full-prefit-audit-v1",
            "status": "A_FIT_RETRY_FULL_PREFIT_GO",
            "scope": "read-only pre-registration draft; endpoint scorer still deferred",
            "successor_amendment_sha256": EXPECTED_SHA,
            "prior_draft_sha256": PRIOR_DRAFT_SHA,
            "predecessor_amendment_sha256": sha(old_path),
            "retry_plan_independent_receipt_sha256": sha(retry_audit_path),
            "retry_plan_canonical_sha256": retry["unit_plan_canonical_json_sha256"],
            "queue_state_sha256_at_audit": hashlib.sha256(queue_bytes).hexdigest(),
            "science_fields_unchanged": list(SCIENCE_FIELDS),
            "source_hashes_checked": {"source": 62, "implementation": 43},
            "source_delta": {"added": len(ADDED_SOURCE),
                             "changed_existing": 1,
                             "changed_implementation": len(CHANGED_IMPL)},
            "unit_count": 220, "unit_cpu_and_wall_reserve_seconds": cpu,
            "prior_a_charge_seconds_at_audit": used_a,
            "projected_a_with_inference_and_failure_margin_seconds": projected_a,
            "projected_global_with_margin_seconds": projected_global,
            "a_ceiling_seconds": 18000, "global_ceiling_seconds": 72000,
            "role_boundary": "FIT, ATTACK_FIT_NEW and INNER_SELECTION; no assessment unit"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign.resolve(strict=True),
                   args.role_manifest.resolve(strict=True),
                   args.queue_state.resolve(strict=True))
    if args.public_receipt.exists():
        raise FileExistsError(args.public_receipt)
    args.public_receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()

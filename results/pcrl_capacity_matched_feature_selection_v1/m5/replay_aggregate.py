"""Role-free, sealed aggregate verification of the focused pair replays.

The aggregate reads only private hash-bound derived artifacts. It never opens
an Adult role export and never imports the M2/M3 selector or M4 scorer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit
from results.pcrl_capacity_matched_feature_selection_v1.m0.queue import sha256
from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_controls import (
    replay_strongest_controls,
)
from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_inference import (
    CLAUSE_NAMES, infer_from_saved_values,
)
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_fit_validation import array_sha256


STUDY = Path(__file__).resolve().parents[1]
PHASE_SEEDS = {"SCREEN": (0, 1), "FINALIST": (0, 1, 2)}
RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")


def _pre_role_gate(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.phase not in PHASE_SEEDS:
        raise PermissionError("M5 aggregate phase outside frozen study")
    unit_id = f"M5-REPLAY-AGGREGATE-{args.phase}"
    output = STUDY / ".private/m5/aggregates" / args.phase / "AGGREGATE_REPORT.json"
    queue_root = STUDY / ".private/queue"
    if (args.unit_id != unit_id or Path(args.output_report).resolve() != output.resolve()
            or Path(args.queue_root).resolve() != queue_root.resolve()):
        raise PermissionError("M5 frozen aggregate unit, queue or output differs")
    spec = assert_running_unit(unit_id, args.queue_root,
                               STUDY / "CAMPAIGN_LOCK.json",
                               "independent_aggregate", output)
    if (spec.get("role_inputs") != {} or spec.get("role_access") != []
            or spec.get("inputs") != {}
            or spec.get("data_access") != "NONE_PRIVATE_DERIVED_ONLY"
            or spec.get("allow_child_processes") is not False
            or spec.get("outputs") != [str(output)]
            or spec.get("deps") != [f"M4-ENDPOINTS-{args.phase}",
                                     f"M3-INFERENCE-{args.phase}"]):
        raise PermissionError("M5 aggregate role-free parent contract differs")
    lock_path = STUDY / "CAMPAIGN_LOCK.json"
    lock = json.loads(lock_path.read_text())
    return spec, lock


def _done(unit_id: str, output: Path, *, state: dict, registry: dict,
          outputs: tuple[Path, ...] | None = None) -> str:
    """Verify a completed output against its exact presealed potential spec."""
    row = state.get("units", {}).get(unit_id)
    sealed = registry.get("unit_specs", {}).get(unit_id)
    if (not isinstance(row, dict) or row.get("status") != "DONE"
            or not isinstance(sealed, dict)):
        raise PermissionError("M5 aggregate required child is not sealed and DONE")
    spec_path = Path(row["spec_path"]).resolve(strict=True)
    if (not spec_path.is_relative_to(STUDY / ".private/queue/specs")
            or sha256(spec_path) != row.get("spec_sha256")):
        raise PermissionError("M5 aggregate child queue spec changed")
    expected = {**sealed, "seal_sha256": state["gate"]["seal_sha256"]}
    if json.loads(spec_path.read_text()) != expected:
        raise PermissionError("M5 aggregate child differs from presealed registry")
    path = output.resolve(strict=True)
    private = (STUDY / ".private").resolve(strict=True)
    if not path.is_relative_to(private):
        raise PermissionError("M5 aggregate child escaped private root")
    digest = sha256(path)
    required = outputs or (output,)
    if (set(row.get("output_sha256", {})) != {str(item) for item in required}
            or row["output_sha256"].get(str(output)) != digest):
        raise PermissionError("M5 aggregate child output SHA changed")
    return digest


def _wire_equal(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes(order="C") == b.tobytes(order="C"))


def _near(left: object, right: object, *, tolerance: float = 1e-10) -> bool:
    return (type(left) in (int, float) and type(right) in (int, float)
            and np.isfinite(left) and np.isfinite(right)
            and abs(float(left) - float(right)) <= tolerance)


def _compare_endpoint(replayed: dict, original: dict) -> None:
    """Compare independently replayed numerical values with the frozen M4 row."""
    if original.get("status") != "DEVELOPMENT_POINT_ONLY":
        raise PermissionError("M4 frozen endpoint status differs")
    for task in ("income", "occupation_group"):
        ours, theirs = replayed["task"][task], original["task"][task]
        for key in ("accuracy", "nll", "brier", "constant_accuracy",
                    "accuracy_gain_over_constant"):
            if not _near(ours["candidate"][key], theirs["candidate"][key]):
                raise PermissionError("M5 task endpoint differs from M4")
        for ref in ("U0", "Ucal", "F"):
            for key in ("accuracy", "nll", "brier", "accuracy_difference",
                        "nll_excess", "brier_excess"):
                if not _near(ours["references"][ref][key],
                             theirs["references"][ref][key]):
                    raise PermissionError("M5 U0/Ucal/F endpoint differs")
            if ours["references"][ref]["point_eligible"] != theirs["references"][ref]["point_eligible"]:
                raise PermissionError("M5 utility point eligibility differs")
        if (ours["point_eligible_all_references"] != theirs["point_eligible_all_references"]
                or ours["per_class"] != theirs["per_class"]):
            raise PermissionError("M5 class or utility status differs")
    for view in ("income", "occupation_group", "coalition"):
        ours, theirs = replayed["recovery"][view], original["recovery"][view]
        if (set(ours["readers"]) != set(theirs["readers"])
                or ours["v2_status"] != theirs["v2_status"]
                or not _near(ours["primary_envelope_auc"],
                             theirs["primary_envelope_auc"])):
            raise PermissionError("M5 fixed reader envelope differs")
        for reader_id in ours["readers"]:
            for key in ("positive_alias_auc", "reverse_alias_auc", "fixed_alias_max_auc"):
                if not _near(ours["readers"][reader_id][key],
                             theirs["readers"][reader_id][key]):
                    raise PermissionError("M5 fixed reader AUC differs")


def _compare_intervals(ours: dict, theirs: dict) -> None:
    if (ours.get("schema") != theirs.get("schema")
            or ours.get("status") != theirs.get("status")
            or ours.get("bootstrap_replicates") != theirs.get("bootstrap_replicates")
            or ours.get("private_seed_sha256") != theirs.get("private_seed_sha256")
            or ours.get("registered_clause_count") != theirs.get("registered_clause_count")
            or ours.get("active_clause_count") != theirs.get("active_clause_count")
            or ours.get("invalid_fit_rule_ids") != theirs.get("invalid_fit_rule_ids")
            or len(ours.get("clauses", [])) != 207
            or len(theirs.get("clauses", [])) != 207):
        raise PermissionError("M5 independent interval family differs from M3")
    if ours["status"] != "DEVELOPMENT_SIMULTANEOUS_INTERVALS":
        raise PermissionError("M5 independent interval replay is incomplete")
    if not _near(ours["critical_95"], theirs["critical_95"], tolerance=1e-9):
        raise PermissionError("M5 simultaneous critical value differs")
    for a, b, name in zip(ours["clauses"], theirs["clauses"], CLAUSE_NAMES,
                          strict=True):
        if (a.get("name") != name or b.get("name") != name
                or a.get("status") != b.get("status")):
            raise PermissionError("M5 simultaneous clause identity/status differs")
        if a["status"] == "UNRESOLVED_INVALID_FIT_RULE":
            continue
        for key in ("point_margin", "se", "lower", "upper"):
            if not _near(a[key], b[key], tolerance=1e-9):
                raise PermissionError("M5 simultaneous numerical interval differs")
        if a["numerical_se_status"] != b["numerical_se_status"]:
            raise PermissionError("M5 simultaneous numerical SE status differs")


def _write_once(path: Path, payload: dict) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return sha256(path)


def run(args: argparse.Namespace) -> dict:
    """Verify complete selected-rule membership and independent point controls."""
    spec, lock = _pre_role_gate(args)
    queue_root = STUDY / ".private/queue"
    state = json.loads((queue_root / "QUEUE.json").read_text())
    registry_path = STUDY / ".private/UNIT_SPECS.json"
    if sha256(registry_path) != lock.get("private_unit_registry_sha256"):
        raise PermissionError("M5 aggregate potential unit registry changed")
    registry = json.loads(registry_path.read_text())
    endpoint_path = STUDY / ".private/m4/endpoints" / args.phase / "INFERENCE_INDEX.json"
    inference_path = STUDY / ".private/m3/inference" / f"{args.phase}.json"
    endpoint_sha = _done(f"M4-ENDPOINTS-{args.phase}", endpoint_path,
                         state=state, registry=registry)
    inference_sha = _done(f"M3-INFERENCE-{args.phase}", inference_path,
                          state=state, registry=registry)
    index = json.loads(endpoint_path.read_text())
    if (index.get("schema") != "pcrl-focused-inference-input-index-v1"
            or index.get("status") != "FROZEN" or index.get("phase") != args.phase
            or index.get("lock_sha256") != sha256(STUDY / "CAMPAIGN_LOCK.json")
            or set(index.get("seeds", {})) != set(map(str, PHASE_SEEDS[args.phase]))):
        raise PermissionError("M5 aggregate source endpoint index differs")
    aggregate_path = endpoint_path.parent / index["aggregate_endpoints_file"]
    if (aggregate_path.resolve(strict=True).parent != endpoint_path.parent.resolve()
            or sha256(aggregate_path) != index["aggregate_endpoints_sha256"]):
        raise PermissionError("M5 aggregate point endpoints changed")
    points = json.loads(aggregate_path.read_text())
    plans, endpoints, saved_values = {}, {}, {}
    pair_receipts = {}
    for seed in PHASE_SEEDS[args.phase]:
        entry = index["seeds"][str(seed)]
        plan_path = STUDY / ".private/m2/fit" / f"seed_{seed}" / "CONTROL_PLAN.json"
        if (Path(entry["fit_plan_receipt_path"]).resolve(strict=True) !=
                plan_path.resolve(strict=True)
                or sha256(plan_path) != entry["fit_plan_receipt_sha256"]):
            raise PermissionError("M5 aggregate FIT control plan path or SHA differs")
        plan = json.loads(plan_path.read_text())
        if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
                or set(plan.get("rules", {})) != set(RULES)):
            raise PermissionError("M5 aggregate FIT rule slate differs")
        plans[seed] = plan
        original_endpoints = points["seeds"][str(seed)]
        endpoints[seed], saved_values[seed] = {}, {}
        seed_npz = endpoint_path.parent / entry["npz_file"]
        if (seed_npz.resolve(strict=True).parent != endpoint_path.parent.resolve()
                or sha256(seed_npz) != entry["npz_sha256"]):
            raise PermissionError("M4 frozen INNER vector file changed")
        with np.load(seed_npz, allow_pickle=False) as source:
            if set(source.files) != set(entry["array_sha256"]):
                raise PermissionError("M4 frozen INNER vector keys changed")
            m4_arrays = {key: source[key] for key in source.files}
        if any(array_sha256(value) != entry["array_sha256"][key]
               for key, value in m4_arrays.items()):
            raise PermissionError("M4 frozen INNER vector array changed")
        for pair_id in sorted({row["pair_id"] for row in plan["rules"].values()
                               if row.get("status") == "SELECTED"}):
            uid = f"M5-REPLAY-S{seed}-{pair_id}"
            folder = STUDY / ".private/m5/replays" / f"seed_{seed}" / pair_id
            report_path, values_path = folder / "REPLAY_REPORT.json", folder / "REPLAY_VALUES.npz"
            both = (report_path, values_path)
            report_sha = _done(uid, report_path, state=state, registry=registry,
                               outputs=both)
            values_sha = _done(uid, values_path, state=state, registry=registry,
                               outputs=both)
            report = json.loads(report_path.read_text())
            if (report.get("schema") != "pcrl-focused-m5-pair-replay-v1"
                    or report.get("status") != "REPLAYED_PAIR_PASS"
                    or report.get("seed") != seed or report.get("pair_id") != pair_id
                    or report.get("values_sha256") != values_sha
                    or report.get("lock_sha256") != sha256(STUDY / "CAMPAIGN_LOCK.json")
                    or report.get("manifest_sha256") != index["m1_manifest_sha256"]):
                raise PermissionError("M5 aggregate pair replay is incomplete")
            with np.load(values_path, allow_pickle=False) as source:
                if set(source.files) != set(report.get("values_array_sha256", {})):
                    raise PermissionError("M5 pair values member set changed")
                values = {key: source[key] for key in source.files}
            if any(array_sha256(value) != report["values_array_sha256"][key]
                   for key, value in values.items()):
                raise PermissionError("M5 pair values array hash changed")
            saved_values[seed][pair_id] = values
            pair_receipts[f"{seed}|{pair_id}"] = {
                "unit_id": uid, "report_sha256": report_sha,
                "values_sha256": values_sha,
            }
        for rule in RULES:
            choice = plan["rules"][rule]
            original = original_endpoints[rule]
            if choice.get("status") != "SELECTED":
                if (original != {"status": choice["status"],
                                 "pair_id": choice["pair_id"]}):
                    raise PermissionError("M4 invalid FIT rule gained endpoints")
                continue
            pair = choice["pair_id"]
            replayed = json.loads((STUDY / ".private/m5/replays" / f"seed_{seed}" /
                                   pair / "REPLAY_REPORT.json").read_text())["endpoint"]
            _compare_endpoint(replayed, original)
            for field in ("row_id", "unit", "income", "occupation", "sex"):
                if not _wire_equal(saved_values[seed][pair][f"truth__{field}"],
                                   m4_arrays[f"truth__{field}"]):
                    raise PermissionError("M5 INNER people or task labels differ")
            for task in ("income", "occupation_group"):
                target = "income" if task == "income" else "occupation"
                if not _wire_equal(saved_values[seed][pair][f"candidate__{task}"],
                                   m4_arrays[f"rule__{rule}__prob_{target}"]):
                    raise PermissionError("M5 candidate probabilities differ")
                for ref in ("U0", "Ucal", "F"):
                    key = f"reference__{task}__{ref}"
                    if not _wire_equal(saved_values[seed][pair][key], m4_arrays[key]):
                        raise PermissionError("M5 reference probabilities differ")
            for view in ("income", "occupation_group", "coalition"):
                ids = entry["reader_ids_by_rule_view"][rule][view]
                if set(ids) != set(replayed["recovery"][view]["readers"]):
                    raise PermissionError("M5 fixed reader slate differs")
                for reader in ids:
                    if not _wire_equal(saved_values[seed][pair][f"reader__{view}__{reader}"],
                                       m4_arrays[f"rule__{rule}__reader__{view}__{reader}"]):
                        raise PermissionError("M5 saved reader predictions differ")
            fields = ("pair_id", "income_mask_id", "occupation_mask_id", "union_sha256",
                      "recipient_capacity_signatures", "union_capacity_signature")
            endpoints[seed][rule] = {**replayed, "rule_id": rule,
                                     **{name: choice[name] for name in fields}}
    selected = replay_strongest_controls(plans, endpoints,
                                         expected_seeds=PHASE_SEEDS[args.phase])
    inference = json.loads(inference_path.read_text())
    if (inference.get("schema") != "pcrl-focused-m3-inference-output-v1"
            or inference.get("status") != "DEVELOPMENT_PENDING_M5_INDEPENDENT_REPLAY"
            or inference.get("phase") != args.phase
            or inference.get("lock_sha256") != sha256(STUDY / "CAMPAIGN_LOCK.json")
            or inference.get("m4_inference_index_sha256") != endpoint_sha
            or inference.get("m4_aggregate_endpoint_sha256") !=
            index["aggregate_endpoints_sha256"]):
        raise PermissionError("M3 inference result identity differs")
    m3_selected = inference.get("strongest_controls", {})
    if m3_selected.get("schema") != "pcrl-focused-m3-strongest-controls-v1":
        raise PermissionError("M3 strongest-control registry missing")
    for pool in ("main", "attribution"):
        a, b = selected[pool], m3_selected[pool]
        if (a["status"] != b["status"] or a["rule_id"] != b["rule_id"]
                or a["eligible_rule_ids"] != b["eligible_rule_ids"]):
            raise PermissionError("M5 strongest-control selection differs")
        if a["status"] == "SELECTED" and (
                not _near(a["paired_seed_mean_coalition_auc"],
                          b["paired_seed_mean_coalition_auc"])
                or a["ties_joint_mask_pair"] != b["ties_joint_mask_pair"]):
            raise PermissionError("M5 strongest-control AUC or pair alias differs")
    seed_hex = lock.get("inference", {}).get("bootstrap_seed_hex")
    if (not isinstance(seed_hex, str) or len(seed_hex) != 64
            or any(char not in "0123456789abcdef" for char in seed_hex)):
        raise PermissionError("M5 locked inference seed differs")
    intervals = infer_from_saved_values(
        saved_values, plans, bytes.fromhex(seed_hex),
        expected_seeds=PHASE_SEEDS[args.phase])
    _compare_intervals(intervals, inference["simultaneous_inference"])
    indexed_clauses = {row["name"]: row for row in intervals["clauses"]}
    for pool in ("main", "attribution"):
        picked = selected[pool]
        saved = inference["selected_control_clauses"][pool]
        if picked["status"] != "SELECTED":
            if saved.get("status") != picked["status"]:
                raise PermissionError("M5 incomplete selected interval differs")
            continue
        for kind in ("coalition_benefit", "income_local_cap", "occupation_local_cap"):
            own = indexed_clauses[f"J|vs|{picked['rule_id']}|{kind}"]
            if saved[kind]["name"] != own["name"] or saved[kind]["status"] != own["status"]:
                raise PermissionError("M5 selected interval mapping differs")
    report = {
        "schema": "pcrl-focused-m5-aggregate-replay-v1",
        "status": "INDEPENDENT_207_CLAUSE_REPLAY_PASS",
        "phase": args.phase, "seeds": list(PHASE_SEEDS[args.phase]),
        "lock_sha256": sha256(STUDY / "CAMPAIGN_LOCK.json"),
        "m4_endpoint_index_sha256": endpoint_sha,
        "m3_inference_sha256": inference_sha,
        "pair_replays": pair_receipts,
        "independent_strongest_controls": selected,
        "independent_simultaneous_inference": intervals,
        "claim_scope": "Development intervals conditional on frozen FIT selection, models and exposed INNER rows",
    }
    _write_once(Path(args.output_report), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASE_SEEDS), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "unit_id": args.unit_id,
                      "output_sha256": sha256(Path(args.output_report))}, sort_keys=True))


if __name__ == "__main__":
    main()

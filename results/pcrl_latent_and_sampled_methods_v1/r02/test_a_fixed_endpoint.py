"""Data-free 18 fixed-attack endpoint and FIT-reason census tests."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_driver import (
    expected_attack_cells, sha,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint import (
    FAILED, FITTED_ARMS, FIXED_ARMS, NULL, classify_fixed_scope,
    map_fixed_physical_units, _producer_and_correction,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_inference import a_margin_at
from results.pcrl_latent_and_sampled_methods_v1.r02.test_a_inference import _seed


def _synthetic_census(tmp_path):
    producer = "a" * 64
    correction = "c" * 64
    queue_root = tmp_path / ".private" / "queue"
    plans, units, r03_rows, r04_rows = {}, {}, [], []
    retries = {}
    old_failure_sha = {}
    for arm in FITTED_ARMS:
        for seed in (0, 1):
            fit_id = f"fit-{arm}-s{seed}"
            failed = arm == "A_SEQ21"
            module = ("results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1."
                      "controls.run_admitted_controls")
            root = tmp_path / ".private" / fit_id
            root.mkdir(parents=True)
            path = root / "FIT_RECEIPT.json"
            fit = {"success": False, "complete": not failed}
            if failed:
                fit.update(stage1_complete=True, stage2_starts=[], stage1_starts=[
                    {"start": start, "completed": True, "updates": 150,
                     "checkpoints": [{"feasible": False, "update": 0},
                                     {"feasible": False, "update": 150}]}
                    for start in ("identity_biased", "uniform_logit")])
            path.write_text(json.dumps({
                "schema": "SCV1-private-control-fit-v1",
                "status": "INCOMPLETE_RESOURCE" if failed else "COMPLETE_NO_FEASIBLE_POLICY",
                "unit_id": fit_id, "arm_id": arm, "source_seed": seed,
                "a_amendment_sha256": producer, "fit": fit}))
            digest = sha(path)
            q_receipt = queue_root / "units" / fit_id / "RECEIPT.json"
            q_receipt.parent.mkdir(parents=True)
            q_receipt.write_text(json.dumps({"unit_id": fit_id, "done": not failed}))
            queue_digest = sha(q_receipt)
            plans[fit_id] = {"kind": "fit", "arm_id": arm, "seed": seed}
            units[fit_id] = {"status": "failed" if failed else "done",
                             "returncode": 1 if failed else 0,
                             "output_sha256": {} if failed else {str(path): digest},
                             "spec": {"unit_id": fit_id, "output_id": fit_id,
                                      "stage": "a", "amendment_sha256": producer,
                                      "outputs": [str(path)],
                                      "argv": ["python", "-m", module,
                                               "--unit-id", fit_id,
                                               "--arm-id", arm, "--seed", str(seed),
                                               "--a-amendment-sha256", producer,
                                               "--private-output", str(root)]}}
            status = "INCOMPLETE_RESOURCE" if failed else "COMPLETE_NO_FEASIBLE_POLICY"
            r03_rows.append({"unit_id": fit_id, "arm_id": arm, "seed": seed,
                             "queue_status": units[fit_id]["status"],
                             "returncode": units[fit_id]["returncode"],
                             "scientific_status": status,
                             "private_fit_receipt_sha256": digest,
                             "queue_receipt_sha256": queue_digest,
                             "selected_policy_present": False})
            r04_rows.append({"unit_id": fit_id, "arm_id": arm, "seed": seed,
                             "queue_status": units[fit_id]["status"],
                             "returncode": units[fit_id]["returncode"],
                             "fit_status": status, "fit_receipt_sha256": digest,
                             "queue_receipt_sha256": queue_digest})
    for arm in (*FITTED_ARMS, *FIXED_ARMS):
        for seed in (0, 1):
            for draw in (range(3) if arm in
                         {"A_JOINT_STOCH", "A_TASK_STOCH", "A_CENTRAL_PRODUCT",
                          "A_LOCAL", "A_SEQ12", "A_SEQ21"} else (0,)):
                uid = (f"A-DIAG-ATTACK-{arm}-S{seed}" if arm in FIXED_ARMS[:2] else
                       f"A-REFERENCE-ATTACK-{arm}-S{seed}" if arm in FIXED_ARMS else
                       f"attack-{arm}-s{seed}-d{draw}")
                kind = ("reference_attack" if arm in FIXED_ARMS[2:] else
                        "diagnostic_attack" if arm in FIXED_ARMS else "attack")
                plans[uid] = {"kind": kind, "output_id": uid, "seed": seed,
                              "realization": draw,
                              ("reference_id" if kind == "reference_attack" else "arm_id"): arm}
                if arm in FITTED_ARMS:
                    plans[uid]["fit_unit_id"] = f"fit-{arm}-s{seed}"
                else:
                    if arm in {"U0", "Ucal", "P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"}:
                        plans[uid].update(deps=[], q_unit_id=None,
                                          release_unit_ids={"ATT": "release-att",
                                                            "INNER": "release-inner"})
                        physical = uid + "-R1"
                        retries[physical] = {"original_attack_unit_id": uid,
                                             "physical_unit_id": physical,
                                             "output_id": physical,
                                             "reference_id": arm, "seed": seed,
                                             "deps": [], "q_unit_id": None,
                                             "release_unit_ids": plans[uid]["release_unit_ids"],
                                             "original_attempt_status": (
                                                 "FAILED_TECHNICAL_PRESERVED" if
                                                 (arm, seed) in {("U0", 0), ("P_STAR", 0)}
                                                 else "NOT_ENQUEUED_SOURCE_CORRECTION")}
                        units[physical] = {"status": "done", "returncode": 0,
                                           "spec": {"unit_id": physical,
                                                    "output_id": physical,
                                                    "stage": "a",
                                                    "amendment_sha256": correction,
                                                    "producer_amendment_sha256": producer,
                                                    "deps": [],
                                                    "original_attack_unit_id": uid}}
                        if (arm, seed) in {("U0", 0), ("P_STAR", 0)}:
                            units[uid] = {"status": "failed", "returncode": 1}
                            failure_receipt = queue_root / "units" / uid / "RECEIPT.json"
                            failure_receipt.parent.mkdir(parents=True, exist_ok=True)
                            failure_receipt.write_text(json.dumps({"unit_id": uid,
                                                                   "status": "failed",
                                                                   "returncode": 1}))
                            old_failure_sha[uid] = sha(failure_receipt)
                            retries[physical]["original_failed_queue_receipt_sha256"] = old_failure_sha[uid]
                    else:
                        units[uid] = {"status": "done", "returncode": 0,
                                      "spec": {"unit_id": uid, "stage": "a",
                                               "amendment_sha256": producer}}
    assert len(expected_attack_cells(plans)) == 58
    r03 = {"schema": "pcrl-r03-a-producer-fit-closeout-v1",
           "effective_a_amendment_sha256": producer, "fit_plan_count": 16,
           "fit_done_rc0_count": 14, "fit_failed_rc1_count": 2,
           "selected_policy_count": 0, "rows": r03_rows}
    r04 = {"schema": "pcrl-r04-a-fit-closeout-linkage-v1",
           "producer_amendment_sha256": producer, "planned_fit_count": 16,
           "complete_null_count": 14, "stage1_stop_failed_count": 2,
           "selected_policy_count": 0, "rows": r04_rows}
    retry_plan = {"producer_amendment_sha256": producer,
                  "physical_attempts": retries,
                  "old_attempt_failed_queue_receipt_sha256": old_failure_sha}
    return plans, units, r03, r04, producer, queue_root, retry_plan, correction


def test_fixed_scope_preserves_58_registry_and_14_vs_2_reasons(tmp_path):
    inputs = _synthetic_census(tmp_path)
    cells, fixed, missing, reports = classify_fixed_scope(*inputs)
    assert len(cells) == 58 and len(fixed) == 18 and len(missing) == 40
    assert sum(row["category"] == NULL for row in reports) == 14
    assert sum(row["category"] == FAILED for row in reports) == 2
    assert {row["arm_id"] for row in reports if row["category"] == FAILED} == {"A_SEQ21"}


def test_failed_seq21_is_not_promoted_to_finalized_null(tmp_path):
    plans, units, r03, r04, producer, queue, retry_plan, correction = _synthetic_census(tmp_path)
    path = Path(units["fit-A_SEQ21-s0"]["spec"]["outputs"][0])
    receipt = json.loads(path.read_text())
    receipt["fit"]["stage2_starts"] = [{"selected": True}]
    path.write_text(json.dumps(receipt))
    r03["rows"][-2]["private_fit_receipt_sha256"] = sha(path)
    r04["rows"][-2]["fit_receipt_sha256"] = sha(path)
    with pytest.raises(PermissionError, match="stage1 resource"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue,
                             retry_plan, correction)


def test_stray_policy_or_technical_descendant_blocks_fixed_report(tmp_path):
    plans, units, r03, r04, producer, queue, retry_plan, correction = _synthetic_census(tmp_path)
    (Path(units["fit-A_LOCAL-s0"]["spec"]["outputs"][0]).parent / "policy.npz").write_bytes(b"x")
    with pytest.raises(PermissionError, match="selected or resource artifact"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue,
                             retry_plan, correction)
    (Path(units["fit-A_LOCAL-s0"]["spec"]["outputs"][0]).parent / "policy.npz").unlink()
    plans["q-A_LOCAL-s0"] = {"kind": "q_preprocess", "deps": ["fit-A_LOCAL-s0"]}
    units["q-A_LOCAL-s0"] = {"status": "failed", "returncode": 1}
    with pytest.raises(PermissionError, match="descendant"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue,
                             retry_plan, correction)


def test_only_18_fixed_cells_leave_1161_of_k6885_estimable():
    records = {0: _seed(), 1: _seed()}
    for seed in (0, 1):
        for arm in FITTED_ARMS:
            del records[seed]["releases"][arm]
    names, margins = a_margin_at(records, np.arange(4))
    assert len(names) == 6885
    assert np.isnan(margins).sum() == 5724
    assert np.isfinite(margins).sum() == 1161


def test_exact_six_original_plus_twelve_r1_physical_mapping(tmp_path):
    correction_sha = "c" * 64
    producer_sha = "a" * 64
    fixed = {}
    units = {}
    attempts = {}
    failure_sha = {}
    queue_root = tmp_path / ".private" / "queue"
    corrected = {"U0", "Ucal", "P_STAR", "D0_LOCAL", "D0_SEQ12", "D0_SEQ21"}
    for arm in FIXED_ARMS:
        for seed in (0, 1):
            logical = (f"A-DIAG-ATTACK-{arm}-S{seed}" if arm.startswith("A_") else
                       f"A-REFERENCE-ATTACK-{arm}-S{seed}")
            fixed[(arm, seed, 0)] = logical
            if arm not in corrected:
                units[logical] = {"status": "done", "returncode": 0,
                                  "spec": {"unit_id": logical, "stage": "a",
                                           "amendment_sha256": producer_sha}}
                continue
            physical = logical + "-R1"
            attempts[physical] = {"original_attack_unit_id": logical,
                                  "physical_unit_id": physical,
                                  "output_id": physical, "reference_id": arm,
                                  "seed": seed,
                                  "original_attempt_status": (
                                      "FAILED_TECHNICAL_PRESERVED" if
                                      (arm, seed) in {("U0", 0), ("P_STAR", 0)} else
                                      "NOT_ENQUEUED_SOURCE_CORRECTION")}
            units[physical] = {"status": "done", "returncode": 0,
                               "spec": {"unit_id": physical, "output_id": physical,
                                        "stage": "a", "amendment_sha256": correction_sha,
                                        "original_attack_unit_id": logical}}
            if attempts[physical]["original_attempt_status"] == "FAILED_TECHNICAL_PRESERVED":
                units[logical] = {"status": "failed", "returncode": 1}
                queue_receipt = queue_root / "units" / logical / "RECEIPT.json"
                queue_receipt.parent.mkdir(parents=True)
                queue_receipt.write_text(json.dumps({"unit_id": logical,
                                                     "status": "failed",
                                                     "returncode": 1}))
                failure_sha[logical] = sha(queue_receipt)
                attempts[physical]["original_failed_queue_receipt_sha256"] = failure_sha[logical]
    proposal = {"producer_amendment_sha256": producer_sha,
                "physical_attempts": attempts,
                "old_attempt_failed_queue_receipt_sha256": failure_sha}
    mapped = map_fixed_physical_units(fixed, units, proposal,
                                      producer_sha, correction_sha, queue_root)
    assert len(mapped) == 18
    assert len({uid for uid in mapped.values() if uid.endswith("-R1")}) == 12
    assert mapped[("U0", 0, 0)] == "A-REFERENCE-ATTACK-U0-S0-R1"
    assert mapped[("T_STAR", 0, 0)] == "A-REFERENCE-ATTACK-T_STAR-S0"
    units["A-REFERENCE-ATTACK-U0-S0-R1"]["spec"]["amendment_sha256"] = producer_sha
    with pytest.raises(PermissionError, match="correction"):
        map_fixed_physical_units(fixed, units, proposal, producer_sha, correction_sha,
                                 queue_root)


def test_old_failed_attempt_receipt_cannot_be_replaced_in_r1_mapping(tmp_path):
    # The two old technical failures stay distinct historical queue attempts.
    inputs = _synthetic_census(tmp_path)
    plans, units, r03, r04, producer, queue, retry_plan, correction = inputs
    for logical in ("A-REFERENCE-ATTACK-U0-S0", "A-REFERENCE-ATTACK-P_STAR-S0"):
        receipt = queue / "units" / logical / "RECEIPT.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"unit_id": logical, "status": "failed",
                                       "returncode": 1}))
    retry_plan["old_attempt_failed_queue_receipt_sha256"] = {
        logical: sha(queue / "units" / logical / "RECEIPT.json")
        for logical in ("A-REFERENCE-ATTACK-U0-S0", "A-REFERENCE-ATTACK-P_STAR-S0")}
    cells = expected_attack_cells(plans)
    fixed = {key: uid for key, uid in cells.items() if key[0] in FIXED_ARMS}
    map_fixed_physical_units(fixed, units, retry_plan, producer, correction, queue)
    (queue / "units" / "A-REFERENCE-ATTACK-U0-S0" / "RECEIPT.json").write_text("{}")
    with pytest.raises(PermissionError, match="failed"):
        map_fixed_physical_units(fixed, units, retry_plan, producer, correction,
                                 queue)


def test_r1_parent_list_must_equal_original_scientific_plan(tmp_path):
    plans, units, r03, r04, producer, queue, retry_plan, correction = _synthetic_census(tmp_path)
    retry_plan["physical_attempts"]["A-REFERENCE-ATTACK-U0-S0-R1"]["deps"] = [
        "different-release"]
    with pytest.raises(PermissionError, match="parent"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue,
                             retry_plan, correction)


def test_correction_must_be_latest_a_while_original_producer_stays_registered(tmp_path):
    producer_path = tmp_path / "producer.json"
    correction_path = tmp_path / "correction.json"
    producer_path.write_text(json.dumps({
        "schema": "pcrl-latent-sampled-A-stage-amendment-v1",
        "status": "LOCALLY_SEALED_A_STAGE",
        "interval_family": {"ordered_clause_count": 6885}, "unit_plan": {},
    }))
    producer_sha = sha(producer_path)
    plan_sha = hashlib.sha256(b"{}").hexdigest()
    correction_path.write_text(json.dumps({
        "schema": "pcrl-a-fixed-attack-correction-v1",
        "status": "LOCALLY_SEALED_FIXED_ATTACK_CORRECTION",
        "producer_amendment_sha256": producer_sha,
        "producer_unit_plan_sha256": plan_sha,
        "physical_attempts": {},
    }))
    correction_sha = sha(correction_path)
    state = {"amendments": [
        {"path": str(producer_path), "sha256": producer_sha, "stages": ["a"]},
        {"path": str(correction_path), "sha256": correction_sha, "stages": ["a"]},
    ]}
    producer, correction = _producer_and_correction(
        state, producer_path, producer_sha, correction_path, correction_sha)
    assert producer["unit_plan"] == {}
    assert correction["producer_amendment_sha256"] == producer_sha
    bad = dict(correction, producer_unit_plan_sha256="f" * 64)
    correction_path.write_text(json.dumps(bad))
    bad_sha = sha(correction_path)
    state["amendments"][-1]["sha256"] = bad_sha
    with pytest.raises(PermissionError, match="scientific lineage"):
        _producer_and_correction(state, producer_path, producer_sha,
                                 correction_path, bad_sha)
    correction_path.write_text(json.dumps(correction))
    state["amendments"][-1]["sha256"] = correction_sha
    state["amendments"].append({"path": str(tmp_path / "other.json"),
                                "sha256": "f" * 64, "stages": ["a"]})
    with pytest.raises(PermissionError, match="latest"):
        _producer_and_correction(state, producer_path, producer_sha,
                                 correction_path, correction_sha)


def test_superseded_inert_proposal_retains_original_family_only():
    from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint import CAMPAIGN

    proposal = json.loads((CAMPAIGN / "r02/A_FIXED_ENDPOINT_ANALYSIS_PROPOSAL.json").read_text())
    assert proposal["status"] == "PREFIT_DRAFT_NOT_EFFECTIVE"
    assert proposal["fixed_family_clause_count"] == 6885
    assert len(proposal["fixed_attack_unit_ids"]) == 18
    assert len(proposal["unavailable_attack_unit_ids"]) == 40
    assert sum(row["category"] == NULL for row in proposal["fit_reason_rows"]) == 14
    assert sum(row["category"] == FAILED for row in proposal["fit_reason_rows"]) == 2
    assert "correction_amendment_sha256" not in proposal
    assert proposal["source_sha256"]["r02/a_fixed_endpoint.py"] != sha(
        CAMPAIGN / "r02/a_fixed_endpoint.py")

"""Data-free 18 fixed-attack endpoint and FIT-reason census tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_driver import (
    expected_attack_cells, sha,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint import (
    FAILED, FITTED_ARMS, FIXED_ARMS, NULL, classify_fixed_scope,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_inference import a_margin_at
from results.pcrl_latent_and_sampled_methods_v1.r02.test_a_inference import _seed


def _synthetic_census(tmp_path):
    producer = "a" * 64
    queue_root = tmp_path / ".private" / "queue"
    plans, units, r03_rows, r04_rows = {}, {}, [], []
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
                uid = f"attack-{arm}-s{seed}-d{draw}"
                kind = ("reference_attack" if arm in FIXED_ARMS[2:] else
                        "diagnostic_attack" if arm in FIXED_ARMS else "attack")
                plans[uid] = {"kind": kind, "output_id": uid, "seed": seed,
                              "realization": draw,
                              ("reference_id" if kind == "reference_attack" else "arm_id"): arm}
                if arm in FITTED_ARMS:
                    plans[uid]["fit_unit_id"] = f"fit-{arm}-s{seed}"
                else:
                    units[uid] = {"status": "done", "returncode": 0}
    assert len(expected_attack_cells(plans)) == 58
    r03 = {"schema": "pcrl-r03-a-producer-fit-closeout-v1",
           "effective_a_amendment_sha256": producer, "fit_plan_count": 16,
           "fit_done_rc0_count": 14, "fit_failed_rc1_count": 2,
           "selected_policy_count": 0, "rows": r03_rows}
    r04 = {"schema": "pcrl-r04-a-fit-closeout-linkage-v1",
           "producer_amendment_sha256": producer, "planned_fit_count": 16,
           "complete_null_count": 14, "stage1_stop_failed_count": 2,
           "selected_policy_count": 0, "rows": r04_rows}
    return plans, units, r03, r04, producer, queue_root


def test_fixed_scope_preserves_58_registry_and_14_vs_2_reasons(tmp_path):
    inputs = _synthetic_census(tmp_path)
    cells, fixed, missing, reports = classify_fixed_scope(*inputs)
    assert len(cells) == 58 and len(fixed) == 18 and len(missing) == 40
    assert sum(row["category"] == NULL for row in reports) == 14
    assert sum(row["category"] == FAILED for row in reports) == 2
    assert {row["arm_id"] for row in reports if row["category"] == FAILED} == {"A_SEQ21"}


def test_failed_seq21_is_not_promoted_to_finalized_null(tmp_path):
    plans, units, r03, r04, producer, queue = _synthetic_census(tmp_path)
    path = Path(units["fit-A_SEQ21-s0"]["spec"]["outputs"][0])
    receipt = json.loads(path.read_text())
    receipt["fit"]["stage2_starts"] = [{"selected": True}]
    path.write_text(json.dumps(receipt))
    r03["rows"][-2]["private_fit_receipt_sha256"] = sha(path)
    r04["rows"][-2]["fit_receipt_sha256"] = sha(path)
    with pytest.raises(PermissionError, match="stage1 resource"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue)


def test_stray_policy_or_technical_descendant_blocks_fixed_report(tmp_path):
    plans, units, r03, r04, producer, queue = _synthetic_census(tmp_path)
    (Path(units["fit-A_LOCAL-s0"]["spec"]["outputs"][0]).parent / "policy.npz").write_bytes(b"x")
    with pytest.raises(PermissionError, match="selected or resource artifact"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue)
    (Path(units["fit-A_LOCAL-s0"]["spec"]["outputs"][0]).parent / "policy.npz").unlink()
    plans["q-A_LOCAL-s0"] = {"kind": "q_preprocess", "deps": ["fit-A_LOCAL-s0"]}
    units["q-A_LOCAL-s0"] = {"status": "failed", "returncode": 1}
    with pytest.raises(PermissionError, match="descendant"):
        classify_fixed_scope(plans, units, r03, r04, producer, queue)


def test_only_18_fixed_cells_leave_1161_of_k6885_estimable():
    records = {0: _seed(), 1: _seed()}
    for seed in (0, 1):
        for arm in FITTED_ARMS:
            del records[seed]["releases"][arm]
    names, margins = a_margin_at(records, np.arange(4))
    assert len(names) == 6885
    assert np.isnan(margins).sum() == 5724
    assert np.isfinite(margins).sum() == 1161


def test_inert_proposal_pins_exact_sources_and_fixed_family():
    from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint import (
        CAMPAIGN, SOURCE_NAMES_FIXED, SHARED_NAMES, WORKTREE,
    )

    proposal = json.loads((CAMPAIGN / "r02/A_FIXED_ENDPOINT_ANALYSIS_PROPOSAL.json").read_text())
    assert proposal["status"] == "PREFIT_DRAFT_NOT_EFFECTIVE"
    assert proposal["fixed_family_clause_count"] == 6885
    assert len(proposal["fixed_attack_unit_ids"]) == 18
    assert len(proposal["unavailable_attack_unit_ids"]) == 40
    assert sum(row["category"] == NULL for row in proposal["fit_reason_rows"]) == 14
    assert sum(row["category"] == FAILED for row in proposal["fit_reason_rows"]) == 2
    for name in SOURCE_NAMES_FIXED:
        assert proposal["source_sha256"][name] == sha(CAMPAIGN / name)
    for name in SHARED_NAMES:
        relative = f"results/pcrl_method_portfolio_v1/shared/{name}"
        assert proposal["source_sha256"][relative] == sha(WORKTREE / relative)

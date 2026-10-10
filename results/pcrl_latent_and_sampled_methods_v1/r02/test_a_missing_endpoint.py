"""Data-free checks for the alternate finalized-null A endpoint path."""

import json

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_driver import (
    expected_attack_cells, sha,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_inference import (
    A_ARMS, A_REFS, STOCHASTIC_ARMS, a_margin_at,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_missing_endpoint import (
    classify_null_descendants,
)


def _plan_and_units(tmp_path):
    plan, units = {}, {}
    producer = "a" * 64
    null_id = "fit-A_JOINT_STOCH-s0"
    plan[null_id] = {"kind": "fit", "arm_id": "A_JOINT_STOCH", "seed": 0}
    receipt_path = tmp_path / "FIT_RECEIPT.json"
    receipt_path.write_text(json.dumps({
        "schema": "SCV1-private-stochastic-fit-v1",
        "status": "COMPLETE_NO_FEASIBLE_POLICY", "selected": None,
        "unit_id": null_id, "arm_id": "A_JOINT_STOCH", "source_seed": 0,
        "a_amendment_sha256": producer}))
    receipt = {"unit_id": null_id, "arm_id": "A_JOINT_STOCH", "seed": 0,
               "receipt_sha256": sha(receipt_path)}
    units[null_id] = {"status": "done", "returncode": 0,
                      "spec": {"unit_id": null_id, "stage": "a",
                               "amendment_sha256": producer,
                               "outputs": [str(receipt_path)],
                               "argv": ["python", "-m",
                                        "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.run_admitted",
                                        "--unit-id", null_id, "--arm-id", "A_JOINT_STOCH",
                                        "--seed", "0", "--a-amendment-sha256", producer,
                                        "--private-output", str(tmp_path)]},
                      "output_sha256": {str(receipt_path): receipt["receipt_sha256"]}}
    for arm in (*A_ARMS, *A_REFS):
        for seed in (0, 1):
            for draw in (range(3) if arm in STOCHASTIC_ARMS else (0,)):
                uid = f"attack-{arm}-s{seed}-d{draw}"
                kind = ("reference_attack" if arm in A_REFS else
                        "diagnostic_attack" if arm in ("A_IDENTITY", "A_CLASS_ONLY")
                        else "attack")
                plan[uid] = {"kind": kind, "output_id": uid,
                             "seed": seed, "realization": draw,
                             ("reference_id" if arm in A_REFS else "arm_id"): arm}
                if arm == "A_JOINT_STOCH" and seed == 0:
                    plan[uid]["fit_unit_id"] = null_id
                else:
                    units[uid] = {"status": "done", "returncode": 0}
    assert len(expected_attack_cells(plan)) == 58
    return plan, units, receipt, receipt_path, producer


def test_finalized_null_removes_whole_stochastic_arm_seed_only(tmp_path):
    plan, units, receipt, _, producer = _plan_and_units(tmp_path)
    cells, available, missing, nulls = classify_null_descendants(
        plan, units, [receipt], producer)
    assert len(cells) == 58 and len(available) == 55 and len(missing) == 3
    assert set(missing) == {("A_JOINT_STOCH", 0, d) for d in range(3)}
    assert ("A_JOINT_STOCH", 1, 0) in available
    assert set(nulls) == {receipt["unit_id"]}


def test_raw_failure_partial_draw_and_wrong_null_receipt_do_not_shrink_family(tmp_path):
    plan, units, receipt, path, producer = _plan_and_units(tmp_path)
    units[receipt["unit_id"]]["status"] = "failed"
    with pytest.raises(PermissionError, match="completed exact"):
        classify_null_descendants(plan, units, [receipt], producer)
    units[receipt["unit_id"]]["status"] = "done"
    units[receipt["unit_id"]]["spec"]["outputs"] = [str(tmp_path / "wrong.json")]
    with pytest.raises(PermissionError, match="completed exact"):
        classify_null_descendants(plan, units, [receipt], producer)
    units[receipt["unit_id"]]["spec"]["outputs"] = [str(path)]
    plan["attack-A_JOINT_STOCH-s0-d2"].pop("fit_unit_id")
    with pytest.raises(PermissionError, match="entire arm"):
        classify_null_descendants(plan, units, [receipt], producer)
    plan["attack-A_JOINT_STOCH-s0-d2"]["fit_unit_id"] = receipt["unit_id"]
    path.write_text(json.dumps({"status": "SELECTED_FIT_POLICY", "selected": {"x": 1},
                                "unit_id": receipt["unit_id"],
                                "arm_id": "A_JOINT_STOCH", "source_seed": 0,
                                "a_amendment_sha256": producer}))
    units[receipt["unit_id"]]["output_sha256"][str(path)] = sha(path)
    receipt["receipt_sha256"] = sha(path)
    with pytest.raises(PermissionError, match="no feasible policy"):
        classify_null_descendants(plan, units, [receipt], producer)


def test_null_descendant_technical_failure_is_not_a_scientific_missing_cell(tmp_path):
    plan, units, receipt, _, producer = _plan_and_units(tmp_path)
    q_id = "q-A_JOINT_STOCH-s0"
    plan[q_id] = {"kind": "q_preprocess", "deps": [receipt["unit_id"]]}
    units[q_id] = {"status": "failed", "returncode": 1}
    with pytest.raises(PermissionError, match="technical failure"):
        classify_null_descendants(plan, units, [receipt], producer)


def test_null_fit_with_stray_selected_policy_archive_is_rejected(tmp_path):
    plan, units, receipt, _, producer = _plan_and_units(tmp_path)
    (tmp_path / "policy.npz").write_bytes(b"stray selected policy")
    with pytest.raises(PermissionError, match="no feasible policy"):
        classify_null_descendants(plan, units, [receipt], producer)


def test_missing_seed_keeps_k6885_and_seed1_clauses_available():
    from results.pcrl_latent_and_sampled_methods_v1.r02.test_a_inference import _seed

    records = {0: _seed(), 1: _seed()}
    del records[0]["releases"]["A_JOINT_STOCH"]
    names, margins = a_margin_at(records, np.arange(4))
    assert len(names) == 6885
    assert np.isnan(margins[names.index(
        "s0_d0:utility:A_JOINT_STOCH:income:U0:nll")])
    assert np.isnan(margins[names.index(
        "mean:utility:A_JOINT_STOCH:income:U0:nll")])
    assert np.isfinite(margins[names.index(
        "s1_d0:utility:A_JOINT_STOCH:income:U0:nll")])
    assert np.isnan(margins).sum() == 450
    assert np.isfinite(margins).sum() == 6435

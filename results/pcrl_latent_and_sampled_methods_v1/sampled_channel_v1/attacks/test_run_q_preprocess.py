"""Data-free admission and public-map checks for the queued Q worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import run_q_preprocess as worker


def _bank_record(recipient: int):
    if recipient == 1:
        p = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        fine = (np.arange(64) % 2).astype(np.int64)
    else:
        p = np.asarray([
            [0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.7, 0.1, 0.05, 0.05, 0.05],
        ], dtype=np.float64)
        fine = (np.arange(641) % 2).astype(np.int64)
    return {
        "probability": p,
        "token_id": np.arange(2, dtype=np.int64),
        "token_class": np.argmax(p, axis=1).astype(np.int64),
        "input_class": np.argmax(p[fine], axis=1).astype(np.int64),
        "fine_to_hcal_token": fine,
    }


def test_fixed_reference_map_uses_original_bank_codes_and_public_identity():
    bank1, bank2, law, descriptor = worker.build_fixed_public_policy(
        arm_id="T_STAR", seed=0, source_kind="historical_finite_reference",
        records=(_bank_record(1), _bank_record(2)),
        release_id="T_STAR-s0-v1", source_binding_sha256="a" * 64,
        source_receipt_sha256="b" * 64)
    assert bank1.probability.shape == (2, 2)
    assert bank2.probability.shape == (2, 6)
    assert law(bank1.probability[0], bank2.probability[1], 0, 1) == ((0, 1, 1.0),)
    assert descriptor["fine_to_code_sha256_by_recipient"]["r1"] != descriptor["fine_to_code_sha256_by_recipient"]["r2"]
    assert descriptor["source_kind"] == "historical_finite_reference"
    assert descriptor["mechanism_sha256"] == bank1.public_metadata["mechanism_sha256"]


def test_q_admission_refuses_without_active_slot_before_any_source_read(tmp_path):
    args = argparse.Namespace(unit_id="A-Q-PREPROCESS-A_JOINT_STOCH-S0",
                              original_lock=tmp_path / "LOCK.json",
                              a_amendment=tmp_path / "A.json",
                              a_amendment_sha256="a" * 64,
                              campaign_root=tmp_path,
                              private_output=tmp_path / ".private" / "q")
    with mock.patch.object(worker, "assert_active_from_env",
                           side_effect=RuntimeError("active queue slot environment missing")), \
         mock.patch.object(np, "load", side_effect=AssertionError("source read occurred")):
        with pytest.raises(RuntimeError, match="active queue slot"):
            worker.run(args)


def test_unit_contract_rejects_unregistered_extra_seed_or_dependency(tmp_path):
    output = tmp_path / ".private" / "q"
    output.parent.mkdir(mode=0o700)
    unit_id = "A-Q-PREPROCESS-A_JOINT_STOCH-S0"
    plan = {"kind": "q_preprocess", "arm_id": "A_JOINT_STOCH",
            "seed": 0, "source_kind": "selected_fit", "output_id": unit_id,
            "fit_unit_id": "A-FIT-A_JOINT_STOCH-S0",
            "deps": ["A-FIT-A_JOINT_STOCH-S0"],
            "expected_outputs": ["Q_TABLE_RECEIPT.json", "q_table.npz"],
            "source_descriptor_sha256": "c" * 64,
            "public_centroids_sha256": "d" * 64,
            "cpu_seconds": 4.0, "wall_seconds": 6.0}
    spec = {"stage": "a", "mode": "real_data", "arm_id": plan["arm_id"],
            "seed": 0, "source_kind": "selected_fit", "output_id": unit_id,
            "fit_unit_id": plan["fit_unit_id"], "deps": list(plan["deps"]),
            "source_descriptor_sha256": plan["source_descriptor_sha256"],
            "public_centroids_sha256": plan["public_centroids_sha256"],
            "zero_support_rule": worker.ZERO_SUPPORT_RULE,
            "cpu_reserve_s": 4.0, "max_wall_s": 6.0,
            "outputs": [str(output / name) for name in plan["expected_outputs"]]}
    worker.validate_q_unit_contract(unit_id, plan, spec, output)
    bad = dict(spec, deps=[])
    with pytest.raises(PermissionError, match="unit plan"):
        worker.validate_q_unit_contract(unit_id, plan, bad, output)
    bad = dict(spec, seed=1)
    with pytest.raises(PermissionError, match="unit plan"):
        worker.validate_q_unit_contract(unit_id, plan, bad, output)


def test_unit_contract_accepts_registered_retry_fit_only_for_same_arm_seed(tmp_path):
    output = tmp_path / ".private" / "q"
    unit_id = "A-Q-PREPROCESS-A_JOINT_STOCH-S0"
    fit_id = "A-FIT-A_JOINT_STOCH-S0-R1"
    plan = {"kind": "q_preprocess", "arm_id": "A_JOINT_STOCH", "seed": 0,
            "source_kind": "selected_fit", "output_id": unit_id,
            "fit_unit_id": fit_id, "deps": [fit_id],
            "expected_outputs": ["Q_TABLE_RECEIPT.json", "q_table.npz"],
            "cpu_seconds": 30, "wall_seconds": 30}
    spec = {"stage": "a", "mode": "real_data", "arm_id": "A_JOINT_STOCH",
            "seed": 0, "source_kind": "selected_fit", "output_id": unit_id,
            "fit_unit_id": fit_id, "deps": [fit_id],
            "outputs": [str(output / name) for name in plan["expected_outputs"]],
            "cpu_reserve_s": 30, "max_wall_s": 30,
            "source_descriptor_sha256": "a" * 64,
            "public_centroids_sha256": "b" * 64,
            "zero_support_rule": worker.ZERO_SUPPORT_RULE}
    assert worker.validate_q_unit_contract(unit_id, plan, spec, output) == "A_JOINT_STOCH"
    for wrong_fit_id in ("A-FIT-A_TASK_STOCH-S0-R1",
                         "A-FIT-A_JOINT_STOCH-S1-R1",
                         "A-FIT-A_JOINT_STOCH-S0-R0"):
        wrong_plan = dict(plan, fit_unit_id=wrong_fit_id, deps=[wrong_fit_id])
        wrong_spec = dict(spec, fit_unit_id=wrong_fit_id, deps=[wrong_fit_id])
        with pytest.raises(PermissionError, match="FIT ID"):
            worker.validate_q_unit_contract(unit_id, wrong_plan, wrong_spec, output)
    wrong_plan = dict(plan, source_kind="same_bank_diagnostic")
    wrong_spec = dict(spec, source_kind="same_bank_diagnostic")
    with pytest.raises(PermissionError):
        worker.validate_q_unit_contract(unit_id, wrong_plan, wrong_spec, output)


def test_clean_centroid_loader_rejects_extra_npz_member(tmp_path):
    (tmp_path / "r01").mkdir()
    source = tmp_path / "r01" / "centroids.npz"
    np.savez(source, centroid_r1=np.asarray([[0.9, 0.1]], dtype=np.float64),
             centroid_r2=np.asarray([[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]], dtype=np.float64),
             cell_class_r1=np.asarray([0], dtype=np.int64),
             cell_class_r2=np.asarray([0], dtype=np.int64),
             forbidden_sex=np.asarray([1], dtype=np.int64))
    audit = {"clean_relative_path": f"r01/{source.name}",
             "clean_sha256": worker.sha_file(source),
             "prototype_array_pair_sha256": "e" * 64}
    with pytest.raises(ValueError, match="member set"):
        worker.load_clean_public_centroids(source, audit)


def test_private_output_refuses_symlink_ancestor_and_public_mode(tmp_path):
    private = tmp_path / ".private"
    private.mkdir(mode=0o700)
    assert worker._private_new_output(private / "new") == private / "new"
    public = private / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="owner-only"):
        worker._private_new_output(public / "new")
    link = private / "link"
    link.symlink_to(public, target_is_directory=True)
    with pytest.raises(PermissionError, match="owner-only"):
        worker._private_new_output(link / "new")
    selected = private / "selected"
    selected.mkdir(mode=0o700)
    assert worker._private_existing_dir(selected) == selected.resolve()
    with pytest.raises(PermissionError, match="owner-only"):
        worker._private_existing_dir(link)


def test_public_fixed_map_rejects_release_identity_mismatch(tmp_path):
    campaign = tmp_path
    source = campaign / "r01"
    source.mkdir()
    prefix = source / "A_FIXED_MAP_T_STAR_S0"
    left, right = _bank_record(1), _bank_record(2)
    arrays = {"bank_r1": left["probability"], "bank_r2": right["probability"],
              "fine_to_code_r1": left["fine_to_hcal_token"],
              "fine_to_code_r2": right["fine_to_hcal_token"]}
    np.savez(prefix.with_suffix(".npz"), **arrays)
    record = {"schema": "pcrl-r01-public-fixed-fine-map-v1",
              "status": "LABEL_FREE_FIXED_PUBLIC_SOURCE", "arm_id": "T_STAR",
              "seed": 0, "release_id": "true-T-star-id",
              "npz_sha256": worker.sha_file(prefix.with_suffix(".npz")),
              "original_bank_receipt_sha256": "a" * 64,
              "source_binding_sha256": "b" * 64,
              "source_audit_sha256": "c" * 64,
              "array_sha256": {name: worker.sha_array(value)
                               for name, value in arrays.items()},
              "array_shape": {name: list(value.shape)
                              for name, value in arrays.items()}}
    prefix.with_suffix(".json").write_text(json.dumps(record))
    spec = {"fixed_map_npz_sha256": record["npz_sha256"],
            "source_receipt_sha256": "a" * 64,
            "source_binding_sha256": "b" * 64,
            "source_audit_sha256": "c" * 64,
            "release_id": "wrong-release-id"}
    centroid = (np.tile(np.asarray([[.9, .1]]), (64, 1)),
                np.tile(np.asarray([[.7, .1, .05, .05, .05, .05]]), (641, 1)))
    with pytest.raises(PermissionError, match="receipt/source identity"):
        worker._fixed_source(spec, campaign, "T_STAR", 0, centroid)


def test_selected_policy_must_match_parent_fit_and_checkpoint():
    plan = {"fit_unit_id": "A-FIT-A_JOINT_STOCH-S0"}
    fit = {"schema": "SCV1-private-stochastic-fit-v1",
           "status": "SELECTED_FIT_POLICY", "unit_id": plan["fit_unit_id"],
           "a_amendment_sha256": "a" * 64,
           "selected": {"start": 0, "update": 25}}
    for field, value in {"arm_id": "A_JOINT_STOCH", "source_seed": 0,
                         "source_binding_sha256": "b" * 64,
                         "row_unit_pair_sha256": "c" * 64,
                         "bank_sha256_by_recipient": {"r1": "d" * 64,
                                                      "r2": "e" * 64},
                         "action_sha256_by_recipient": {"r1": "f" * 64,
                                                        "r2": "0" * 64}}.items():
        fit[field] = value
    policy = {key: fit[key] for key in (
        "arm_id", "source_seed", "source_binding_sha256",
        "row_unit_pair_sha256", "bank_sha256_by_recipient",
        "action_sha256_by_recipient")}
    policy.update(status="SELECTED_FIT_POLICY",
                  fit_unit_id=plan["fit_unit_id"],
                  selected_start="identity_biased", selected_update=25)
    worker._selected_parent_identity(fit, policy, plan, "a" * 64)
    with pytest.raises(PermissionError, match="checkpoint selection"):
        worker._selected_parent_identity(fit, dict(policy, selected_update=50),
                                         plan, "a" * 64)
    with pytest.raises(PermissionError, match="completed FIT"):
        worker._selected_parent_identity(fit,
            dict(policy, row_unit_pair_sha256="1" * 64), plan, "a" * 64)


def test_hard_control_tied_terminal_metrics_use_first_start():
    plan = {"fit_unit_id": "A-FIT-A_TASK_DET-S0"}
    fit = {"schema": "SCV1-private-control-fit-v1",
           "status": "SELECTED_FIT_POLICY", "unit_id": plan["fit_unit_id"],
           "a_amendment_sha256": "a" * 64, "arm_id": "A_TASK_DET",
           "fit": {"complete": True, "success": True, "selection": {"loss": 1.0},
                   "starts": [
                       {"start": "identity_biased", "terminal_metrics": {"loss": 1.0},
                        "chosen_checkpoint_update": 125},
                       {"start": "uniform", "terminal_metrics": {"loss": 1.0},
                        "chosen_checkpoint_update": 200}]}}
    for field, value in {"source_seed": 0, "source_binding_sha256": "b" * 64,
                         "row_unit_pair_sha256": "c" * 64,
                         "bank_sha256_by_recipient": {"r1": "d" * 64,
                                                      "r2": "e" * 64},
                         "action_sha256_by_recipient": {"r1": "f" * 64,
                                                        "r2": "0" * 64}}.items():
        fit[field] = value
    policy = {key: fit[key] for key in (
        "arm_id", "source_seed", "source_binding_sha256",
        "row_unit_pair_sha256", "bank_sha256_by_recipient",
        "action_sha256_by_recipient")}
    policy.update(status="SELECTED_FIT_POLICY", fit_unit_id=plan["fit_unit_id"],
                  selected_start="identity_biased", selected_update=125)
    worker._selected_parent_identity(fit, policy, plan, "a" * 64)

"""Data-free custody and queue-order checks for the M3 inference worker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_inference import RULES, _inputs


def _seed_archive(tmp_path):
    from inference import _digest
    from inference_cli import LOCAL_READER_IDS, _sha_file, _sha_json

    data, plans = _inputs()
    private = tmp_path / ".private"
    index_dir = private / "m4" / "endpoints" / "SCREEN"
    index_dir.mkdir(parents=True)
    plan_dir = private / "m2" / "fit" / "seed_0"
    plan_dir.mkdir(parents=True)
    plan = plans[0]
    plan["selection_receipt_sha256"] = _sha_json(plan)
    plan_path = plan_dir / "CONTROL_PLAN.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    arrays = {f"truth__{name}": values for name, values in data[0]["truth"].items()}
    arrays.update({f"reference__{task}__{ref}": values
                   for task, bank in data[0]["references"].items()
                   for ref, values in bank.items()})
    ids_by_rule, indexed_rules = {}, {}
    for rule_id in RULES:
        release = data[0]["rules"][rule_id]
        indexed_rules[rule_id] = {"status": "SELECTED", "pair_id": release["pair_id"]}
        arrays[f"rule__{rule_id}__prob_income"] = release["prob_income"]
        arrays[f"rule__{rule_id}__prob_occupation"] = release["prob_occupation"]
        ids_by_rule[rule_id] = {}
        local = {}
        for view in ("income", "occupation_group"):
            local[view] = {name: release["reader_predictions"][view]["informative"]
                           if j == 5 else release["reader_predictions"][view]["constant"]
                           for j, name in enumerate(sorted(LOCAL_READER_IDS))}
        pair = {name: release["reader_predictions"]["coalition"]["informative"]
                if j == 5 else release["reader_predictions"]["coalition"]["constant"]
                for j, name in enumerate(sorted(LOCAL_READER_IDS))}
        pair.update({f"IGNORE_R2__{name}": value
                     for name, value in local["income"].items()})
        pair.update({f"IGNORE_R1__{name}": value
                     for name, value in local["occupation_group"].items()})
        for view, bank in (("income", local["income"]),
                           ("occupation_group", local["occupation_group"]),
                           ("coalition", pair)):
            ids_by_rule[rule_id][view] = list(bank)
            arrays.update({f"rule__{rule_id}__reader__{view}__{name}": value
                           for name, value in bank.items()})
    path = index_dir / "seed_0.npz"
    np.savez_compressed(path, **arrays)
    entry = {
        "npz_file": path.name, "npz_sha256": _sha_file(path),
        "array_sha256": {name: _digest(value) for name, value in arrays.items()},
        "fit_plan_receipt_path": str(plan_path),
        "fit_plan_receipt_sha256": _sha_file(plan_path),
        "rules": indexed_rules, "reader_ids_by_rule_view": ids_by_rule,
    }
    return entry, index_dir, private


def test_private_seed_archive_restores_exact_arrays_and_rejects_hash_tamper(tmp_path):
    from inference_cli import _decode_seed

    entry, index_dir, private = _seed_archive(tmp_path)
    data, plan = _decode_seed(entry, index_dir, private, 0)
    assert len(data["rules"]) == 10
    assert len(data["rules"]["J"]["reader_predictions"]["coalition"]) == 18
    assert plan["rules"]["J"]["pair_id"] == "J"
    key = "truth__sex"
    entry["array_sha256"][key] = "0" * 64
    try:
        _decode_seed(entry, index_dir, private, 0)
    except ValueError as error:
        assert "per-array SHA" in str(error)
    else:
        raise AssertionError("a tampered per-person label SHA passed custody")


def test_seed_archive_rejects_relocated_plan_and_substituted_reader(tmp_path):
    from inference_cli import _decode_seed, _sha_file

    entry, index_dir, private = _seed_archive(tmp_path)
    original = Path(entry["fit_plan_receipt_path"])
    relocated = private / "alternate" / "CONTROL_PLAN.json"
    relocated.parent.mkdir()
    relocated.write_bytes(original.read_bytes())
    entry["fit_plan_receipt_path"] = str(relocated)
    entry["fit_plan_receipt_sha256"] = _sha_file(relocated)
    try:
        _decode_seed(entry, index_dir, private, 0)
    except ValueError as error:
        assert "seed-specific" in str(error)
    else:
        raise AssertionError("relocated same-byte FIT plan passed seed custody")
    entry["fit_plan_receipt_path"] = str(original)
    entry["fit_plan_receipt_sha256"] = _sha_file(original)
    entry["reader_ids_by_rule_view"]["J"]["income"][0] = "BOGUS_READER"
    try:
        _decode_seed(entry, index_dir, private, 0)
    except ValueError as error:
        assert "fixed reader ID slate" in str(error)
    else:
        raise AssertionError("substituted reader passed the fixed slate")


def test_fit_source_receipt_requires_done_queue_output(tmp_path):
    from inference_cli import _sha_file, _verify_fit_source

    entry, _, private = _seed_archive(tmp_path)
    lock_sha, manifest_sha = "1" * 64, "2" * 64
    fit_dir = private / "m2" / "fit" / "seed_0"
    fit_path = fit_dir / "FIT_RECEIPT.json"
    fit_path.write_text(json.dumps({
        "schema": "pcrl-focused-fit-receipt-v1", "status": "FIT_COMPLETE",
        "unit_id": "M2-FIT-S0", "seed": 0,
        "campaign_lock_sha256": lock_sha, "data_manifest_sha256": manifest_sha,
        "control_plan_sha256": entry["fit_plan_receipt_sha256"],
        "control_plan_relative_path": "m2/fit/seed_0/CONTROL_PLAN.json",
    }))
    queue_root = private / "queue"
    queue_root.mkdir()
    source = {"fit_receipt_sha256_by_seed": {"0": _sha_file(fit_path)}}
    queue = {"units": {"M2-FIT-S0": {
        "status": "DONE", "output_sha256": {str(fit_path): _sha_file(fit_path)},
    }}}
    _verify_fit_source(entry, seed=0, private_root=private, queue=queue,
                       source_receipts=source, lock_sha256=lock_sha,
                       manifest_sha256=manifest_sha)
    queue["units"]["M2-FIT-S0"]["output_sha256"][str(fit_path)] = "0" * 64
    try:
        _verify_fit_source(entry, seed=0, private_root=private, queue=queue,
                           source_receipts=source, lock_sha256=lock_sha,
                           manifest_sha256=manifest_sha)
    except ValueError as error:
        assert "DONE output" in str(error)
    else:
        raise AssertionError("FIT receipt without matching DONE output passed")


def test_cli_calls_live_queue_guard_before_opening_inner_index(tmp_path):
    from inference_cli import main

    study = tmp_path / "study"
    study.mkdir()
    lock = study / "CAMPAIGN_LOCK.json"
    lock.write_text("{}")
    private = study / ".private"
    index = private / "m4" / "endpoints" / "SCREEN" / "INFERENCE_INDEX.json"
    output = private / "m3" / "inference" / "SCREEN.json"
    argv = ["--unit-id", "M3-INFERENCE-SCREEN", "--queue-root", str(private / "compute"),
            "--lock-path", str(lock), "--index-path", str(index),
            "--output", str(output)]
    with patch("results.pcrl_capacity_matched_feature_selection_v1.m0.admission.assert_running_unit",
               side_effect=RuntimeError("QUEUE_DENIED")):
        try:
            main(argv)
        except RuntimeError as error:
            assert str(error) == "QUEUE_DENIED"
        else:
            raise AssertionError("private index was read without live queue admission")


def test_independent_point_discrepancy_blocks_strongest_selection():
    from inference import point_endpoints, prepare
    from inference_cli import compare_point_endpoints

    data, plans = _inputs()
    own = point_endpoints(prepare(data, plans, expected_rows=40, expected_groups=20),
                          plans)
    reported = {"seeds": {str(seed): {rule: dict(values) for rule, values in rows.items()}
                          for seed, rows in own.items()}}
    compare_point_endpoints(own, reported)
    reported["seeds"]["0"]["J"]["recovery"] = {
        **reported["seeds"]["0"]["J"]["recovery"],
        "coalition": {"primary_envelope_auc": 0.5},
    }
    try:
        compare_point_endpoints(own, reported)
    except ValueError as error:
        assert "envelope" in str(error)
    else:
        raise AssertionError("M4/M3 recovery disagreement entered selected inference")

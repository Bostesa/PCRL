"""Synthetic pre-role gate and frozen unit identity checks."""

from __future__ import annotations

import argparse
import hashlib

import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry
from results.pcrl_capacity_matched_feature_selection_v1.m2.bank import build_shared_bank
from results.pcrl_method_portfolio_v1.teams.c.prototype import AdultGroupedSchema
from worker import (_expected_output, _expected_unit_id, _view_geometry, run)


def test_worker_refuses_unregistered_unit_before_any_role_read(monkeypatch):
    import worker

    def forbidden(*args, **kwargs):
        raise AssertionError("role loader must not be reached")

    monkeypatch.setattr(worker, "_store", forbidden)
    args = argparse.Namespace(
        mode="release", unit_id="UNREGISTERED", seed=0,
        pair_id="M_DROP_A__M_DROP_B", view=None, union_sha=None,
        queue_root=str(worker.STUDY / ".private/queue"),
        output_index=str(worker.STUDY / ".private/elsewhere.json"),
    )
    with pytest.raises(PermissionError, match="frozen unit/output"):
        run(args)


def test_done_no_feasible_fit_stops_raw_before_any_role_read(tmp_path, monkeypatch):
    import json
    import worker

    study = tmp_path / "study"
    path = study / ".private/m2/fit/seed_0/FIT_RECEIPT.json"
    path.parent.mkdir(parents=True)
    (study / "CAMPAIGN_LOCK.json").write_text("{}\n")
    (study / "m1").mkdir()
    (study / "m1/DATA_ACCESS_MANIFEST.json").write_text("{}\n")
    receipt = {
        "schema": "pcrl-focused-fit-receipt-v1", "status": "NO_FEASIBLE_J",
        "seed": 0,
        "campaign_lock_sha256": worker._sha(study / "CAMPAIGN_LOCK.json"),
        "data_manifest_sha256": worker._sha(study / "m1/DATA_ACCESS_MANIFEST.json"),
        "prior_data_manifest_sha256": "a" * 64,
    }
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(worker, "STUDY", study)
    monkeypatch.setattr(worker, "_dependency_output", lambda *a: worker._sha(path))
    with pytest.raises(PermissionError, match="M2 FIT receipt"):
        worker._fit_receipt({"deps": ["M2-FIT-S0"]}, 0,
                            {"prior_data_manifest_sha256": "a" * 64}, {},
                            study / ".private/queue")
    monkeypatch.setattr(worker, "_pre_role_gate", lambda args: (
        {"deps": ["M2-FIT-S0"]}, {"prior_data_manifest_sha256": "a" * 64}, {}))
    monkeypatch.setattr(worker, "_store", lambda *a: (_ for _ in ()).throw(
        AssertionError("role file was opened after no-go FIT")))
    args = argparse.Namespace(mode="raw_union", seed=0, union_sha="a" * 64,
                              unit_id="M4-RAW-S0-AAAAAAAAAAAAAAAA", pair_id=None,
                              view=None, queue_root=str(study / ".private/queue"),
                              output_index=str(study / "unused.json"))
    with pytest.raises(PermissionError, match="M2 FIT receipt"):
        run(args)


def test_frozen_pair_and_raw_unit_ids_and_paths():
    import worker

    args = argparse.Namespace(mode="release", seed=1,
                              pair_id="M_DROP_A__M_DROP_B", view=None,
                              union_sha=None)
    assert _expected_unit_id(args) == "M4-REL-S1-M_DROP_A__M_DROP_B"
    assert str(_expected_output(args)).endswith(
        "/m4/releases/seed_1/M_DROP_A__M_DROP_B/RELEASE_RECEIPT.json")
    args.mode = "raw_union"
    args.pair_id = None
    args.union_sha = "a" * 64
    assert _expected_unit_id(args) == "M4-RAW-S1-AAAAAAAAAAAAAAAA"
    assert str(_expected_output(args)).endswith(
        "/m4/raw_union/seed_1/" + "a" * 64 + "/RAW_UNION_INDEX.json")
    args.mode = "attack"
    args.pair_id = "M_DROP_A__M_DROP_B"
    args.union_sha = None
    args.view = "occupation_group"
    assert _expected_unit_id(args) == "M4-ATT-S1-M_DROP_A__M_DROP_B-OCCUPATION"
    assert str(_expected_output(args)).endswith(
        "/m4/attacks/seed_1/M_DROP_A__M_DROP_B/occupation_group/ATTACK_INDEX.json")


def test_local_view_uses_own_raw_union_and_pair_view_uses_pair_union():
    schema = AdultGroupedSchema()
    bank = build_shared_bank(capacity_geometry(schema.feature_names)["masks"])
    pair = next(pair for pair in bank.pairs
                if pair.income_mask_id != pair.occupation_mask_id)
    income_raw = _view_geometry(bank, pair, "income")
    occupation_raw = _view_geometry(bank, pair, "occupation_group")
    coalition_raw = _view_geometry(bank, pair, "coalition")
    assert income_raw.income_mask_id == income_raw.occupation_mask_id
    assert occupation_raw.income_mask_id == occupation_raw.occupation_mask_id
    assert income_raw.union_sha256 != coalition_raw.union_sha256
    assert occupation_raw.union_sha256 != coalition_raw.union_sha256
    assert len(coalition_raw.union_columns) == 83


def test_prior_manifest_is_exact_admitted_sibling_not_arbitrary_path(tmp_path, monkeypatch):
    import worker

    root = tmp_path / "focused"
    root.mkdir()
    expected = tmp_path / "pcrl-method-portfolio-v1" / "results/pcrl_method_portfolio_v1"
    expected.mkdir(parents=True)
    path = expected / "DATA_ACCESS_MANIFEST.json"
    path.write_text("{}\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "REPO", root)
    monkeypatch.setattr(worker, "EXPECTED_PRIOR", expected)
    lock = {"prior_package_relative_to_worktree":
            "../pcrl-method-portfolio-v1/results/pcrl_method_portfolio_v1",
            "prior_data_manifest_sha256": digest}
    manifest = {"prior_package_id": "pcrl_method_portfolio_v1",
                "portfolio_manifest_sha256": digest}
    assert worker._prior_manifest_path(lock, manifest) == path
    other = tmp_path / "other"
    other.mkdir()
    (other / "DATA_ACCESS_MANIFEST.json").write_bytes(path.read_bytes())
    lock["prior_package_relative_to_worktree"] = "../other"
    with pytest.raises(PermissionError, match="admitted sibling"):
        worker._prior_manifest_path(lock, manifest)

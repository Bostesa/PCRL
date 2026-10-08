"""Data-free FIT worker admission and pre-SEX custody checks."""

import json
import hashlib

import pytest

from bank import build_shared_bank
from fit_worker import (_exact_role_keys, _publish_task_only,
                        _validation_receipt_fields,
                        _resolve_prior_package, _manifest_identities)
from test_bank_joint import _records


def _validation_registry(bank):
    def cache_key(purpose, mask_id):
        return hashlib.sha256(f"{purpose}:{mask_id}:validation".encode()).hexdigest()

    return {
        purpose: {
            mask_id: {
                "stage": "validation", "purpose": purpose, "mask_id": mask_id,
                "cache_key": cache_key(purpose, mask_id),
                "model_id": f"{purpose}:{mask_id}:validation",
                "head_sha256": "b" * 64,
                "calibrator_sha256": "c" * 64,
                "package_sha256": "d" * 64,
                "receipt_sha256": "e" * 64,
                "head_pool_receipt_sha256": "e" * 64,
                "relative_package_path":
                    f"{'b' * 64}/head_pool/{cache_key(purpose, mask_id)}/release_package.json",
                "head_pool_receipt_relative_path":
                    f"{'b' * 64}/head_pool/{cache_key(purpose, mask_id)}/HEAD_POOL_RECEIPT.json",
            }
            for mask_id in ("F_FULL83", *(mask.mask_id for mask in bank.masks))
        }
        for purpose in ("income", "occupation_group")
    }


def test_fit_role_capability_is_exact_and_excludes_screen_roles():
    fields = _exact_role_keys(1)
    assert len(fields) == 8
    assert "OSF_DEFENSE_FIT:sex" in fields
    assert "OSF_DEFENSE_FIT:references_s1" in fields
    assert all(not key.startswith(("INNER_SELECTION:", "ATTACK_FIT_NEW:",
                                   "HEAD_VALIDATION:", "OSF_DEVELOPMENT_ASSESSMENT:"))
               for key in fields)


def test_task_only_publication_binds_all_five_masks_and_precedes_sex(tmp_path):
    bank = build_shared_bank(_records())
    rows = lambda purpose: [{"mask_id": mask.mask_id,
                             "mask_sha256": mask.mask_sha256,
                             "head_id": f"{purpose}:{mask.mask_id}:validation"}
                            for mask in bank.masks]
    table = {"task_table_sha256": "a" * 64, "split_receipt": {"fit_rows": 10},
             "income_masks": rows("income"),
             "occupation_masks": rows("occupation_group"),
             "bank_sha256": bank.bank_sha256}
    choice = {"pair_id": bank.pairs[0].pair_id}
    result = _publish_task_only(tmp_path, "M2-FIT-S0", 0, "b" * 64,
                                bank, table, choice, _validation_registry(bank))
    assert (tmp_path / "TASK_TABLE.json").is_file()
    receipt = json.loads((tmp_path / "TASK_ONLY_RECEIPT.json").read_text())
    assert result == receipt["task_table_sha256"]
    assert receipt["bank_masks"] == [
        {"mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256}
        for mask in bank.masks]
    assert sum(map(len, receipt["validation_head_packages"].values())) == 12
    assert len(receipt["validation_head_packages_sha256"]) == 64
    assert _validation_receipt_fields(
        tmp_path, _validation_registry(bank))["validation_head_packages_sha256"] == \
        receipt["validation_head_packages_sha256"]
    with pytest.raises(RuntimeError, match="validation package registry"):
        changed = _validation_registry(bank)
        changed["income"][bank.masks[0].mask_id]["package_sha256"] = "f" * 64
        _validation_receipt_fields(tmp_path, changed)
    assert len(receipt["task_table_sha256"]) == 64
    with pytest.raises(ValueError, match="five mask"):
        _publish_task_only(tmp_path, "M2-FIT-S0", 0, "b" * 64,
                           bank, {**table, "income_masks": rows("income")[:-1]}, choice,
                           _validation_registry(bank))


def test_task_receipt_rejects_missing_or_wrong_validation_package(tmp_path):
    bank = build_shared_bank(_records())
    rows = lambda purpose: [{"mask_id": mask.mask_id,
                             "mask_sha256": mask.mask_sha256,
                             "head_id": f"{purpose}:{mask.mask_id}:validation"}
                            for mask in bank.masks]
    table = {"task_table_sha256": "a" * 64, "split_receipt": {"fit_rows": 10},
             "income_masks": rows("income"),
             "occupation_masks": rows("occupation_group"),
             "bank_sha256": bank.bank_sha256}
    choice = {"pair_id": bank.pairs[0].pair_id}
    missing = _validation_registry(bank)
    del missing["income"][bank.masks[0].mask_id]
    with pytest.raises(ValueError, match="validation package"):
        _publish_task_only(tmp_path, "M2-FIT-S0", 0, "b" * 64,
                           bank, table, choice, missing)
    wrong = _validation_registry(bank)
    wrong["income"][bank.masks[0].mask_id]["stage"] = "final"
    with pytest.raises(ValueError, match="validation package"):
        _publish_task_only(tmp_path, "M2-FIT-S0", 0, "b" * 64,
                           bank, table, choice, wrong)
    duplicate = _validation_registry(bank)
    first, second = bank.masks[:2]
    duplicate["income"][second.mask_id]["cache_key"] = \
        duplicate["income"][first.mask_id]["cache_key"]
    with pytest.raises(ValueError, match="cache key is reused"):
        _publish_task_only(tmp_path, "M2-FIT-S0", 0, "b" * 64,
                           bank, table, choice, duplicate)
    assert not (tmp_path / "TASK_ONLY_RECEIPT.json").exists()


def test_prior_role_package_may_be_the_pinned_sibling_worktree(tmp_path):
    focused = tmp_path / "focused"
    focused.mkdir()
    prior = tmp_path / "prior" / "results" / "pcrl_method_portfolio_v1"
    prior.mkdir(parents=True)
    lock = {"prior_package_relative_to_worktree":
            "../prior/results/pcrl_method_portfolio_v1"}
    assert _resolve_prior_package(focused, lock, {
        "prior_package_id": "pcrl_method_portfolio_v1"}) == prior
    with pytest.raises(PermissionError, match="sibling"):
        _resolve_prior_package(focused, {
            "prior_package_relative_to_worktree":
            "../../unrelated/results/pcrl_method_portfolio_v1"},
            {"prior_package_id": "pcrl_method_portfolio_v1"})


def test_focused_manifest_is_package_identity_and_prior_has_separate_hash(tmp_path):
    study = tmp_path / "results" / "pcrl_capacity_matched_feature_selection_v1"
    m1 = study / "m1"
    m1.mkdir(parents=True)
    (m1 / "DATA_ACCESS_MANIFEST.json").write_text('{"schema":"focused"}')
    prior = tmp_path / "prior" / "DATA_ACCESS_MANIFEST.json"
    prior.parent.mkdir()
    prior.write_text('{"schema":"portfolio"}')
    import hashlib
    focused_sha = hashlib.sha256((m1 / "DATA_ACCESS_MANIFEST.json").read_bytes()).hexdigest()
    prior_sha = hashlib.sha256(prior.read_bytes()).hexdigest()
    identities = _manifest_identities(
        study, prior, {"source_map_sha256": focused_sha,
                       "prior_data_manifest_sha256": prior_sha})
    assert identities == (focused_sha, prior_sha)
    assert focused_sha != prior_sha


def test_task_only_no_go_receipt_still_precedes_sensitive_read(tmp_path):
    bank = build_shared_bank(_records())
    rows = lambda purpose: [{"mask_id": mask.mask_id,
                             "mask_sha256": mask.mask_sha256,
                             "head_id": f"{purpose}:{mask.mask_id}:validation"}
                            for mask in bank.masks]
    table = {"task_table_sha256": "a" * 64, "split_receipt": {"fit_rows": 10},
             "income_masks": rows("income"),
             "occupation_masks": rows("occupation_group"),
             "bank_sha256": bank.bank_sha256}
    choice = {"status": "NO_UTILITY_FEASIBLE_T", "pair_id": None,
              "reason": "no utility-feasible task-only income mask",
              "task_table_sha256": table["task_table_sha256"],
              "bank_sha256": bank.bank_sha256}
    _publish_task_only(tmp_path, "M2-FIT-S0", 0, "b" * 64,
                       bank, table, choice, _validation_registry(bank))
    receipt = json.loads((tmp_path / "TASK_ONLY_RECEIPT.json").read_text())
    assert receipt["selection"] == choice
    assert sum(map(len, receipt["validation_head_packages"].values())) == 12

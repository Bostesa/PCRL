"""Synthetic fixed-bank replay checks; no Adult role or snapshot is opened."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r18.run_restored_a_fixed import (
    _read_npz, _read_reference, array_sha256, rederive_role, wire_digest,
)


def _bank(width: int) -> dict:
    probability = np.eye(width, dtype=np.float64)
    return {
        "probability": probability,
        "token_class": np.arange(width, dtype=np.int64),
        "token_id": np.arange(width, dtype=np.int64),
        "input_class": np.arange(width, dtype=np.int64),
        "fine_to_hcal_token": np.arange(width, dtype=np.int64),
    }


def _fixture() -> tuple[dict, dict, dict, dict, dict]:
    banks = {1: _bank(2), 2: _bank(6)}
    ids = np.array([11, 12], dtype=np.int64)
    units = np.array([21, 22], dtype=np.int64)
    tokens = {"row_id": ids, "unit": units,
              "cell_1": np.array([0, 1], np.int64),
              "cell_2": np.array([3, 5], np.int64),
              "token_1": np.array([0, 1], np.int64),
              "token_2": np.array([3, 5], np.int64)}
    release = {"row_id": ids.copy(), "unit": units.copy()}
    receipt = {"release_id": "invented-fixed-bank", "source_binding_sha256": "a" * 64,
               "bank_sha256_by_recipient": {}, "wire_sha256_by_recipient": {}}
    for r in (1, 2):
        code = tokens[f"token_{r}"]
        probability = banks[r]["probability"][code]
        decision = banks[r]["token_class"][code]
        release[f"code_r{r}"] = code.copy()
        release[f"prob_r{r}"] = probability.copy()
        release[f"decision_r{r}"] = decision.copy()
        bank_sha = array_sha256(banks[r]["probability"])
        receipt["bank_sha256_by_recipient"][f"r{r}"] = bank_sha
        receipt["wire_sha256_by_recipient"][f"r{r}"] = wire_digest(
            probability, decision, code, recipient=r, release_id=receipt["release_id"],
            source_binding=receipt["source_binding_sha256"], bank_hash=bank_sha,
            bank_size=len(banks[r]["token_id"]))
    return banks, tokens, release, receipt, {"row_id": ids, "unit": units}


def test_restored_fixed_pair_derives_saved_code_probability_decision_and_wire() -> None:
    banks, tokens, release, receipt, reference = _fixture()
    result = rederive_role(banks, tokens, release, receipt, reference)
    assert result["rows"] == 2
    assert result["wire_sha256_by_recipient"] == receipt["wire_sha256_by_recipient"]


def test_changed_saved_probability_or_fine_mapping_refuses() -> None:
    banks, tokens, release, receipt, reference = _fixture()
    altered = copy.deepcopy(release)
    altered["prob_r1"][0] = altered["prob_r1"][1]
    with pytest.raises(ValueError, match="saved release"):
        rederive_role(banks, tokens, altered, receipt, reference)
    altered_tokens = copy.deepcopy(tokens)
    altered_tokens["cell_2"][0] = 4
    with pytest.raises(ValueError, match="fine-to-code"):
        rederive_role(banks, altered_tokens, release, receipt, reference)


def test_changed_wire_digest_or_source_order_refuses() -> None:
    banks, tokens, release, receipt, reference = _fixture()
    altered_receipt = copy.deepcopy(receipt)
    altered_receipt["wire_sha256_by_recipient"]["r2"] = "0" * 64
    with pytest.raises(ValueError, match="wire SHA"):
        rederive_role(banks, tokens, release, altered_receipt, reference)
    altered_reference = copy.deepcopy(reference)
    altered_reference["unit"] = altered_reference["unit"][::-1].copy()
    with pytest.raises(ValueError, match="source identity"):
        rederive_role(banks, tokens, release, receipt, altered_reference)


def test_unexpected_archive_member_refuses_before_any_array_load(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "invented.npz"
    np.savez(archive, row_id=np.array([1], dtype=np.int64),
             unit=np.array([2], dtype=np.int64),
             sex=np.array([0], dtype=np.int64))

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("array loaded before member gate")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(PermissionError, match="member set"):
        _read_npz(archive, {"row_id", "unit"})
    with pytest.raises(PermissionError, match="member set"):
        _read_reference(archive)

"""Public metadata-only checks for M5's source-independent bank."""

import json
from pathlib import Path

import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_bank import reconstruct


SPEC = Path(__file__).resolve().parents[1] / "m1/CAPACITY_SPEC.json"


def test_reconstructed_five_by_five_public_bank():
    bank = reconstruct(SPEC)
    assert len(bank.masks) == 5
    assert len(bank.pairs) == 25
    assert bank.bank_sha256 == "7e10e6401c48fb50c62b4d2fb2190aeb602c1b653d0eae6f3d7989478fa3558e"
    assert {len(pair.union_columns) for pair in bank.pairs if
            pair.income_mask_id != pair.occupation_mask_id} == {83}


def test_changed_public_union_is_rejected(tmp_path):
    data = json.loads(SPEC.read_text())
    data["pairs"][0]["union_column_indices"][0] = 82
    candidate = tmp_path / "capacity.json"
    candidate.write_text(json.dumps(data))
    with pytest.raises(PermissionError, match="union"):
        reconstruct(candidate)

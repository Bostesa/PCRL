"""The conditional comparison reuses the exact public focused bank."""

from __future__ import annotations

from pathlib import Path

import pytest

from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
SOURCE_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


def test_frozen_bank_has_unchanged_leads_and_union_geometry():
    bank = load_frozen_bank(SOURCE, expected_sha256=SOURCE_SHA)
    assert len(bank.masks) == 5
    assert len(bank.pairs) == 25
    assert len({pair.union_sha256 for pair in bank.pairs}) == 6
    assert bank.mask("M_DROP_NATIVE_COUNTRY").encoded_columns == 42
    assert bank.mask("M_DROP_MARITAL_STATUS").encoded_columns == 76
    for pair in bank.pairs:
        if pair.income_mask_id != pair.occupation_mask_id:
            assert pair.union_encoded_columns == 83


def test_frozen_bank_rejects_unsealed_public_bank(tmp_path):
    changed = tmp_path / "MASK_BANK.json"
    changed.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(PermissionError, match="public bank SHA"):
        load_frozen_bank(changed, expected_sha256=SOURCE_SHA)


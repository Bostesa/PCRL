"""Data-free output accounting for the two transported RAW_ONLY readers."""

from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2.raw_union_worker import (
    expected_outputs, required_union_shas,
)
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank


BANK_FILE = Path(__file__).resolve().parents[2] / \
    "pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
BANK_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


def test_raw_union_reader_output_inventory_counts_every_model_and_transport():
    output = Path("/tmp/.private/raw_union/seed0/union/RAW_UNION_INDEX.json")
    paths = expected_outputs(output)
    assert len(paths) == 11
    assert str(output) in paths
    assert str(output.parent / "model__RAW_LR_C1.joblib") in paths
    assert str(output.parent / "prediction__RAW_HGB_15x200__inner.npy") in paths
    assert str(output.parent / "transport__RAW_HGB_15x200__att.json") in paths


def test_selected_offdiagonal_requires_both_local_unions_and_coalition():
    bank = load_frozen_bank(BANK_FILE, expected_sha256=BANK_SHA)
    pair = next(pair for pair in bank.pairs
                if pair.income_mask_id != pair.occupation_mask_id)
    plan = {"rules": {"J": {"status": "SELECTED", "pair_id": pair.pair_id}}}
    required = required_union_shas(bank, plan)
    local = {item.union_sha256 for item in bank.pairs
             if (item.income_mask_id == item.occupation_mask_id
                 and item.income_mask_id in
                 (pair.income_mask_id, pair.occupation_mask_id))}
    assert len(required) == 3
    assert required == local | {pair.union_sha256}

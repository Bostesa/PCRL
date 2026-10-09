"""Pure tests for the independent task-only ranking boundary."""

import hashlib
import json

import pytest

from results.pcrl_utility_reference_repair_v1.r3.task_table_selection_audit import (
    content_sha, select_task_mask,
)


def test_task_only_choice_uses_eligible_nll_then_hash_tie_break():
    rows = [
        {"mask_id": "A", "mask_sha256": "b" * 64,
         "utility_feasible": True, "metrics": {"nll": .4}},
        {"mask_id": "B", "mask_sha256": "c" * 64,
         "utility_feasible": False, "metrics": {"nll": .1}},
        {"mask_id": "C", "mask_sha256": "a" * 64,
         "utility_feasible": True, "metrics": {"nll": .4}},
    ]
    assert select_task_mask(rows)["mask_id"] == "C"
    with pytest.raises(ValueError):
        select_task_mask(rows[1:2])


def test_content_sha_uses_canonical_json():
    value = {"b": 2, "a": [1, 3]}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    assert content_sha(value) == hashlib.sha256(raw).hexdigest()

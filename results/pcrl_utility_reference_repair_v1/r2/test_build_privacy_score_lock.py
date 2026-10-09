"""q20 INNER SEX scoring waits for every complete-view ATT predecessor."""

import pytest

from results.pcrl_utility_reference_repair_v1.r2 import build_privacy_score_lock


def test_score_predecessor_requires_all_four_coalitions_done():
    ids = {f"R2_ATT_S{i}_P{j}_COALITION"
           for i in (0, 1) for j in (0, 1)}
    q19 = {"units": {uid: {"status": "DONE", "charged_cpu_s": 2.0}
                     for uid in ids},
           "carry_forward": {"charged_cpu_s": 150.0},
           "used_cpu_s": 158.0}
    carry = build_privacy_score_lock.validate_coalition_predecessor(
        q19, ids, "a" * 64)
    assert carry == {"archived_queue_relative": "queue_g19",
                     "archived_queue_sha256": "a" * 64,
                     "charged_cpu_s": 158.0, "unit_count": 4}
    q19["units"][next(iter(ids))]["status"] = "FAILED"
    with pytest.raises(PermissionError, match="coalition ATT"):
        build_privacy_score_lock.validate_coalition_predecessor(
            q19, ids, "a" * 64)

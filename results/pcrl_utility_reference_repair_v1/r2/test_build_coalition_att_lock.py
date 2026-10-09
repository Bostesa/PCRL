"""q19 coalition readers require all local reader outputs under q18."""

import pytest

from results.pcrl_utility_reference_repair_v1.r2 import build_coalition_att_lock


def test_q19_requires_all_eight_local_units_and_exact_carry():
    local_ids = {f"R2_ATT_S{i}_{j}_{view.upper()}"
                 for i in (0, 1) for j in ("A", "B")
                 for view in ("income", "occupation_group")}
    q18 = {"units": {uid: {"status": "DONE", "charged_cpu_s": 1.0}
                     for uid in local_ids},
           "carry_forward": {"charged_cpu_s": 120.0},
           "used_cpu_s": 128.0}
    carry = build_coalition_att_lock.validate_local_predecessor(
        q18, local_ids, "a" * 64)
    assert carry == {"archived_queue_relative": "queue_g18",
                     "archived_queue_sha256": "a" * 64,
                     "charged_cpu_s": 128.0, "unit_count": 8}
    q18["units"][next(iter(local_ids))]["status"] = "FAILED"
    with pytest.raises(PermissionError, match="local ATT"):
        build_coalition_att_lock.validate_local_predecessor(
            q18, local_ids, "a" * 64)

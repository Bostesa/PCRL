"""Frozen two-lead branch needs both tasks and seeds at unchanged caps."""

from __future__ import annotations

import pytest

from results.pcrl_utility_reference_repair_v1.r2.lead_branch import (
    qualify_historical_leads,
)


LEADS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS")
TASKS = ("income", "occupation_group")


def _rows():
    return {
        str(seed): {
            mask: {
                task: {"references": {
                    ref: {"point_eligible": True}
                    for ref in ("U0", "Ucal", "F")
                }} for task in TASKS
            } for mask in LEADS
        } for seed in (0, 1)
    }


def test_one_lead_qualifies_only_if_both_tasks_and_seeds_pass():
    rows = _rows()
    rows["1"]["M_DROP_MARITAL_STATUS"]["occupation_group"][
        "references"]["Ucal"]["point_eligible"] = False
    assert qualify_historical_leads(rows) == ("M_DROP_NATIVE_COUNTRY",)


def test_branch_fails_closed_on_missing_clause_or_nonboolean():
    rows = _rows()
    del rows["0"]["M_DROP_NATIVE_COUNTRY"]["income"]["references"]["F"]
    with pytest.raises(ValueError, match="reference coverage"):
        qualify_historical_leads(rows)
    rows = _rows()
    rows["0"]["M_DROP_NATIVE_COUNTRY"]["income"]["references"]["F"][
        "point_eligible"] = 1
    with pytest.raises(ValueError, match="boolean"):
        qualify_historical_leads(rows)


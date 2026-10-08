"""Frozen J choice over the same finite pair table used by matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from table import PairRow, PairTable, validate_pair_table


@dataclass(frozen=True)
class JointSelection:
    pair_id: str
    row: PairRow
    t_pair_id: str
    local_auc_cap_excess: float
    evaluated_pair_count: int
    utility_feasible_pair_count: int
    feasible_pair_count: int
    alias_of_t: bool

    def to_payload(self) -> dict:
        return {
            "schema": "pcrl-focused-joint-selection-v1",
            "pair_id": self.pair_id,
            "t_pair_id": self.t_pair_id,
            "local_auc_cap_excess": self.local_auc_cap_excess,
            "evaluated_pair_count": self.evaluated_pair_count,
            "utility_feasible_pair_count": self.utility_feasible_pair_count,
            "feasible_pair_count": self.feasible_pair_count,
            "alias_of_t": self.alias_of_t,
            "selected_row": self.row.to_payload(),
        }


def select_joint(
    table: PairTable, *, t_pair_id: str,
    local_auc_cap_excess: float = 0.010,
) -> JointSelection:
    """Minimize coalition recovery under utility and T-relative local guards.

    All 25 assignments have already been scored on fitting-role folds.  This
    is an exact optimum only within that frozen bank and search-reader slate.
    The utility caps and boundary convention are encoded in `table`; local
    AUC guard equality is admitted.  An exact alias to T remains a valid tie.
    """
    if not isfinite(local_auc_cap_excess) or local_auc_cap_excess < 0:
        raise ValueError("local AUC cap excess must be finite and nonnegative")
    validate_pair_table(table)
    if not table.pairs:
        raise ValueError("joint selector requires a nonempty pair table")
    utility_feasible = tuple(row for row in table.pairs if row.pair_feasible)
    if not utility_feasible:
        raise ValueError("no utility-feasible pair in shared bank")
    t = table.pair(t_pair_id)
    if not t.pair_feasible:
        raise ValueError("T reference must be utility feasible")
    eligible = tuple(row for row in utility_feasible if (
        row.income_local_search_auc <=
        t.income_local_search_auc + local_auc_cap_excess
        and row.occupation_local_search_auc <=
        t.occupation_local_search_auc + local_auc_cap_excess
    ))
    if not eligible:
        raise ValueError("no pair meets T-relative local AUC guards")
    chosen = min(eligible, key=lambda row: (
        row.coalition_search_auc, row.task_loss,
        row.income_mask_sha256, row.occupation_mask_sha256,
        row.pair_id,
    ))
    return JointSelection(
        chosen.pair_id, chosen, t_pair_id, local_auc_cap_excess,
        len(table.pairs), len(utility_feasible), len(eligible),
        chosen.pair_id == t_pair_id,
    )

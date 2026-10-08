"""Common immutable mask/pair score table for all focused selection rules.

Inputs are fitting-role validation and search-reader aggregates.  This module
does no fitting or role access.  M3 consumes exactly this table for T, I, S12,
S21 and union-matched controls; M2 J has no separate candidate search.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from bank import SharedBank


PURPOSES = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
METRICS = ("accuracy", "nll", "brier")
CAPS = {"accuracy": 0.010, "nll": 0.010, "brier": 0.005}


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class TaskMetrics:
    accuracy: float
    nll: float
    brier: float

    def __post_init__(self) -> None:
        if (not all(isfinite(x) for x in vars(self).values())
                or not 0 <= self.accuracy <= 1
                or self.nll < 0 or not 0 <= self.brier <= 2):
            raise ValueError("task metrics must be finite and in range")


def _metrics(raw: object) -> TaskMetrics:
    if isinstance(raw, TaskMetrics):
        return raw
    if not isinstance(raw, Mapping) or set(raw) != set(METRICS):
        raise ValueError("task metrics require accuracy, nll and brier")
    return TaskMetrics(**{key: float(raw[key]) for key in METRICS})


@dataclass(frozen=True)
class MaskScore:
    mask_id: str
    mask_sha256: str
    recipient_capacity_signature: tuple[int, int, int]
    head_id: str
    task_validation_accuracy: float
    task_validation_nll: float
    task_validation_brier: float
    utility_margins_by_ref_metric: tuple[tuple[str, tuple[float, float, float]], ...]
    local_search_auc: float
    utility_feasible: bool

    def margins(self, reference: str) -> dict[str, float]:
        for name, values in self.utility_margins_by_ref_metric:
            if name == reference:
                return dict(zip(METRICS, values))
        raise KeyError(reference)

    def to_payload(self) -> dict:
        return {
            "mask_id": self.mask_id, "mask_sha256": self.mask_sha256,
            "recipient_capacity_signature": {
                "numeric_groups": self.recipient_capacity_signature[0],
                "categorical_groups": self.recipient_capacity_signature[1],
                "encoded_columns_max": self.recipient_capacity_signature[2],
            },
            "head_id": self.head_id,
            "task_validation_accuracy": self.task_validation_accuracy,
            "task_validation_nll": self.task_validation_nll,
            "task_validation_brier": self.task_validation_brier,
            "utility_margins_by_ref_metric": {
                name: dict(zip(METRICS, margins))
                for name, margins in self.utility_margins_by_ref_metric
            },
            "local_search_auc": self.local_search_auc,
            "utility_feasible": self.utility_feasible,
        }


@dataclass(frozen=True)
class PairRow:
    pair_id: str
    income_mask_id: str
    occupation_mask_id: str
    income_mask_sha256: str
    occupation_mask_sha256: str
    union_atoms: tuple[str, ...]
    union_columns: tuple[int, ...]
    union_group_count: int
    union_encoded_columns: int
    union_sha256: str
    coalition_search_auc: float
    income_local_search_auc: float
    occupation_local_search_auc: float
    task_loss: float
    pair_feasible: bool

    @property
    def union_size_signature(self) -> tuple[int, int]:
        return self.union_group_count, self.union_encoded_columns

    def to_payload(self) -> dict:
        payload = dict(vars(self))
        payload["union_size_signature"] = {
            "source_group_count": self.union_group_count,
            "encoded_columns": self.union_encoded_columns,
        }
        return payload


@dataclass(frozen=True)
class PairTable:
    income_masks: tuple[MaskScore, ...]
    occupation_masks: tuple[MaskScore, ...]
    pairs: tuple[PairRow, ...]
    bank_sha256: str
    table_sha256: str
    critic_input_kind: str

    def pair_id(self, income_mask_id: str, occupation_mask_id: str) -> str:
        pair_id = f"{income_mask_id}__{occupation_mask_id}"
        if not any(row.pair_id == pair_id for row in self.pairs):
            raise KeyError(pair_id)
        return pair_id

    def pair(self, pair_id: str) -> PairRow:
        matches = [row for row in self.pairs if row.pair_id == pair_id]
        if len(matches) != 1:
            raise KeyError(pair_id)
        return matches[0]

    def to_payload(self) -> dict:
        return {
            "schema": "pcrl-focused-pair-table-v1",
            "bank_sha256": self.bank_sha256,
            "table_sha256": self.table_sha256,
            "critic_input_kind": self.critic_input_kind,
            "income_masks": [mask.to_payload() for mask in self.income_masks],
            "occupation_masks": [mask.to_payload() for mask in self.occupation_masks],
            "pairs": [row.to_payload() for row in self.pairs],
        }


def _mask_scores(
    bank: SharedBank, evaluations: Mapping[str, object],
    references: Mapping[str, object],
) -> tuple[MaskScore, ...]:
    expected_ids = {mask.mask_id for mask in bank.masks}
    if set(evaluations) != expected_ids or set(references) != set(REFERENCES):
        raise ValueError("mask evaluations or reference identities are incomplete")
    refs = {name: _metrics(references[name]) for name in REFERENCES}
    result = []
    for mask in bank.masks:
        raw = evaluations[mask.mask_id]
        if not isinstance(raw, Mapping) or set(raw) != {
            "head_id", "metrics", "local_search_auc",
        }:
            raise ValueError("mask evaluation keys differ from frozen contract")
        metrics = _metrics(raw["metrics"])
        auc = float(raw["local_search_auc"])
        if not 0.5 <= auc <= 1 or not raw["head_id"]:
            raise ValueError("local search AUC must be oriented and head identified")
        margin_rows = []
        for name in REFERENCES:
            ref = refs[name]
            margins = (
                metrics.accuracy - (ref.accuracy - CAPS["accuracy"]),
                ref.nll + CAPS["nll"] - metrics.nll,
                ref.brier + CAPS["brier"] - metrics.brier,
            )
            margin_rows.append((name, margins))
        feasible = all(value >= 0 for _, margins in margin_rows for value in margins)
        result.append(MaskScore(
            mask.mask_id, mask.mask_sha256, mask.capacity_signature,
            str(raw["head_id"]), metrics.accuracy, metrics.nll,
            metrics.brier, tuple(margin_rows), auc, feasible,
        ))
    return tuple(result)


def build_pair_table(
    bank: SharedBank,
    evaluations_by_purpose: Mapping[str, Mapping[str, object]],
    references_by_purpose: Mapping[str, Mapping[str, object]],
    coalition_search_auc_by_union: Mapping[str, float],
    *, critic_input_kind: str = "RAW_UNION_ONLY",
) -> PairTable:
    """Join one shared bank to cached fitting-role scores, without refitting."""
    if critic_input_kind != "RAW_UNION_ONLY":
        raise ValueError("union-hash critic cache requires RAW_UNION_ONLY inputs")
    if set(evaluations_by_purpose) != set(PURPOSES) or set(references_by_purpose) != set(PURPOSES):
        raise ValueError("both purpose evaluations and references are required")
    income = _mask_scores(bank, evaluations_by_purpose["income"],
                          references_by_purpose["income"])
    occupation = _mask_scores(bank, evaluations_by_purpose["occupation_group"],
                              references_by_purpose["occupation_group"])
    unions = {pair.union_sha256 for pair in bank.pairs}
    if set(coalition_search_auc_by_union) != unions:
        raise ValueError("coalition search scores must cover exactly each unique union")
    income_by_id = {mask.mask_id: mask for mask in income}
    occupation_by_id = {mask.mask_id: mask for mask in occupation}
    rows = []
    for pair in bank.pairs:
        first = income_by_id[pair.income_mask_id]
        second = occupation_by_id[pair.occupation_mask_id]
        coalition_auc = float(coalition_search_auc_by_union[pair.union_sha256])
        if not 0.5 <= coalition_auc <= 1:
            raise ValueError("coalition search AUC must be oriented and in range")
        rows.append(PairRow(
            pair.pair_id, pair.income_mask_id, pair.occupation_mask_id,
            first.mask_sha256, second.mask_sha256, pair.union_atoms,
            pair.union_columns, pair.union_group_count,
            pair.union_encoded_columns, pair.union_sha256, coalition_auc,
            first.local_search_auc, second.local_search_auc,
            first.task_validation_nll + second.task_validation_nll,
            first.utility_feasible and second.utility_feasible,
        ))
    payload = {
        "critic_input_kind": critic_input_kind,
        "bank_sha256": bank.bank_sha256,
        "income_masks": [mask.to_payload() for mask in income],
        "occupation_masks": [mask.to_payload() for mask in occupation],
        "pairs": [row.to_payload() for row in rows],
    }
    table = PairTable(income, occupation, tuple(rows), bank.bank_sha256,
                      _sha(payload), critic_input_kind)
    validate_pair_table(table, bank)
    return table


def validate_pair_table(table: PairTable, bank: SharedBank | None = None) -> None:
    """Fail closed on persisted table mutation before any selector consumes it."""
    if table.critic_input_kind != "RAW_UNION_ONLY":
        raise ValueError("pair table cannot cache a complete-view critic by raw union")
    if bank is not None and table.bank_sha256 != bank.bank_sha256:
        raise ValueError("pair table bank identity changed")
    masks = {}
    for purpose, scores in (("income", table.income_masks),
                            ("occupation", table.occupation_masks)):
        ids = [score.mask_id for score in scores]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{purpose} mask score IDs are missing or duplicate")
        if bank is not None and ids != [mask.mask_id for mask in bank.masks]:
            raise ValueError("mask score order differs from shared bank")
        for score in scores:
            if (not 0.5 <= score.local_search_auc <= 1
                    or not all(isfinite(value) for value in (
                        score.task_validation_accuracy,
                        score.task_validation_nll,
                        score.task_validation_brier,
                    ))
                    or tuple(name for name, _ in score.utility_margins_by_ref_metric)
                    != REFERENCES
                    or any(len(margins) != 3 or not all(isfinite(v) for v in margins)
                           for _, margins in score.utility_margins_by_ref_metric)
                    or score.utility_feasible != all(
                        value >= 0 for _, margins in
                        score.utility_margins_by_ref_metric for value in margins
                    )):
                raise ValueError("mask fitting score or utility status is inconsistent")
            if bank is not None:
                mask = bank.mask(score.mask_id)
                if (score.mask_sha256 != mask.mask_sha256
                        or score.recipient_capacity_signature !=
                        mask.capacity_signature):
                    raise ValueError("mask score identity differs from shared bank")
        masks[purpose] = {score.mask_id: score for score in scores}
    expected_ids = {
        f"{left.mask_id}__{right.mask_id}"
        for left in table.income_masks for right in table.occupation_masks
    }
    if len(table.pairs) != len(expected_ids) or {row.pair_id for row in table.pairs} != expected_ids:
        raise ValueError("pair score table is not the full shared Cartesian bank")
    geometry = {pair.pair_id: pair for pair in bank.pairs} if bank is not None else None
    union_scores = {}
    for row in table.pairs:
        first = masks["income"][row.income_mask_id]
        second = masks["occupation"][row.occupation_mask_id]
        if (row.pair_id != f"{row.income_mask_id}__{row.occupation_mask_id}"
                or row.income_mask_sha256 != first.mask_sha256
                or row.occupation_mask_sha256 != second.mask_sha256
                or row.income_local_search_auc != first.local_search_auc
                or row.occupation_local_search_auc != second.local_search_auc
                or row.task_loss != first.task_validation_nll +
                second.task_validation_nll
                or row.pair_feasible !=
                (first.utility_feasible and second.utility_feasible)
                or not 0.5 <= row.coalition_search_auc <= 1):
            raise ValueError("pair row differs from its cached mask scores")
        if geometry is not None:
            pair = geometry[row.pair_id]
            if (row.union_atoms != pair.union_atoms
                    or row.union_columns != pair.union_columns
                    or row.union_group_count != pair.union_group_count
                    or row.union_encoded_columns != pair.union_encoded_columns
                    or row.union_sha256 != pair.union_sha256):
                raise ValueError("pair row union geometry differs from shared bank")
        prior = union_scores.setdefault(row.union_sha256,
                                        row.coalition_search_auc)
        if prior != row.coalition_search_auc:
            raise ValueError("raw-union critic differs across identical unions")
    payload = {
        "critic_input_kind": table.critic_input_kind,
        "bank_sha256": table.bank_sha256,
        "income_masks": [mask.to_payload() for mask in table.income_masks],
        "occupation_masks": [mask.to_payload() for mask in table.occupation_masks],
        "pairs": [row.to_payload() for row in table.pairs],
    }
    if table.table_sha256 != _sha(payload):
        raise ValueError("pair table SHA differs from scored contents")

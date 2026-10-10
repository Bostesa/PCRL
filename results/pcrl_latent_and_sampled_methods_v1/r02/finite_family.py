"""One frozen simultaneous INNER margin family across screened candidate IDs.

Pure in-memory arithmetic. The registry is supplied by R00 before outcomes;
missing entries are errors, so the max-t dimension cannot silently contract.
All arms use one ordered INNER group vector and identical bootstrap positions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

_SHARED = Path(__file__).resolve().parents[2] / "pcrl_method_portfolio_v1" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
from common_endpoints import calculate_margins, margin_names
from common_eval import assert_same_people
from paired_inference import paired_group_inference


def finite_margin_names(registry: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Enumerate candidate × tasks/references/metrics and matched controls."""
    if not registry or len(set(registry)) != len(registry):
        raise ValueError("finite comparison registry is required")
    out = []
    for candidate, controls in registry.items():
        if not isinstance(candidate, str) or not candidate or ":" in candidate:
            raise ValueError("invalid finite candidate ID")
        out.extend(f"{candidate}:{name}" for name in margin_names(tuple(controls)))
    if len(set(out)) != len(out):
        raise ValueError("duplicate finite margin name")
    return tuple(out)


def registered_margin_at(registry: Mapping[str, tuple[str, ...]],
                         records: Mapping[str, dict[int, dict]], positions) -> tuple[tuple[str, ...], np.ndarray]:
    """Evaluate every registered clause on the same paired resampled people."""
    names = finite_margin_names(registry)
    if set(records) != set(registry):
        raise ValueError("finite comparison registry coverage missing")
    anchor = None
    values = []
    for candidate, controls in registry.items():
        seed_records = records[candidate]
        if not seed_records:
            raise ValueError("finite comparison registry seed missing")
        for seed in sorted(seed_records):
            truth = seed_records[seed]["truth"]
            if anchor is None:
                anchor = truth
            else:
                assert_same_people(anchor, truth)
        item = calculate_margins(seed_records, tuple(controls), positions)
        if tuple(item["names"]) != tuple(name.removeprefix(candidate + ":")
                                    for name in names[len(values):len(values) + len(item["values"])]):
            raise ValueError("finite margin schema differs")
        values.extend(np.asarray(item["values"], dtype=np.float64))
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(names),):
        raise ValueError("finite margin vector width differs")
    return names, vector


def run_finite_family_inference(registry: Mapping[str, tuple[str, ...]],
                                records: Mapping[str, dict[int, dict]],
                                private_seed: bytes) -> dict:
    """Inherited 2048 paired-group max-|t| inference over all fixed clauses."""
    if set(records) != set(registry):
        raise ValueError("finite comparison registry coverage missing")
    first = next(iter(records.values()))
    unit = np.asarray(first[min(first)]["truth"]["unit"], dtype=np.int64)
    names = finite_margin_names(registry)

    def at(positions: np.ndarray, draw: int) -> np.ndarray:
        if draw != 0:
            raise ValueError("deterministic family has only draw zero")
        got, values = registered_margin_at(registry, records, positions)
        if got != names:
            raise ValueError("finite margin schema changed")
        return values

    result = paired_group_inference(unit, names, at, private_seed)
    result["finite_registry"] = {k: list(v) for k, v in registry.items()}
    result["candidate_count"] = len(registry)
    result["clause_count"] = len(names)
    return result

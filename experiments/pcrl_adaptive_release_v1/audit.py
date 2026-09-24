"""Exact 17-token audit for adaptive partitions and legally routed H views.

Private leaf IDs are consumed only to compute a per-person token law. A frozen
predictor receives H plus a token indicator, never a leaf or probability row.
Selection remains confined to the supplied validation rows; callers must lock
the resulting route before any assessment score.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited


N_TOKENS = 17
FLOOR = inherited.FLOOR
ROLES = inherited.ROLES
parse_role = inherited.parse_role
expected_token_loss = inherited.expected_token_loss
score_weightings = inherited.score_weightings
select_validation = inherited.select_validation
assert_household_disjoint = inherited.assert_household_disjoint
model_directory_hash = inherited.model_directory_hash
build_role_route_bank = inherited.build_role_route_bank
score_fixed_binary_decoder = inherited.score_fixed_binary_decoder


def person_token_law(channel: np.ndarray, private_leaf_ids: np.ndarray) -> np.ndarray:
    """Compute exact token laws behind the evaluation boundary only.

    The returned matrix is private scoring data, never a release/API field.
    """
    q = np.asarray(channel, dtype=np.float64)
    leaf = np.asarray(private_leaf_ids)
    if (q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != N_TOKENS or
            not np.isfinite(q).all() or np.min(q) < 0 or np.max(q) > 1 or
            np.max(np.abs(q.sum(axis=1) - 1)) > 1e-10):
        raise ValueError("invalid finite 17-token release channel")
    if (leaf.ndim != 1 or len(leaf) == 0 or
            not np.issubdtype(leaf.dtype, np.integer) or
            np.any(leaf < 0) or np.any(leaf >= len(q))):
        raise ValueError("invalid private leaf ID")
    return q[leaf].copy()


def view_features(rows: dict[str, Any], view: str, *, strict: bool = False) -> np.ndarray:
    """Return A's four H columns, B's two, or their coalition union.

    ``strict`` validates a deployable input dictionary. A private fitting row
    may also carry labels and leaf IDs, but those are never passed to a model.
    """
    if strict and set(rows) - {"ha", "hb"}:
        raise ValueError("deployable feature input contains hidden or undeclared fields")
    return inherited.view_features(rows["ha"], rows.get("hb"), view)


def validate_token_law(token_probs: np.ndarray, *, n: int | None = None,
                       h_only: bool = False) -> np.ndarray:
    p = inherited.validate_token_probs(token_probs, n=n)
    if p.shape[1] != (1 if h_only else N_TOKENS):
        raise ValueError("audit token law does not match declared wire")
    return p


def _registered_rows(rows: dict) -> None:
    # The adaptive-release contract publishes H plus one token. The inherited
    # audit also supports an auxiliary continuous wire for the older study;
    # reject it here before that extra feature can reach a fitted predictor.
    if rows.get("aux") is not None:
        raise ValueError("undeclared auxiliary service coordinate")


def role_arrays(rows: dict, token_probs: np.ndarray, role: str, *,
                require_full_fit_support: bool = False) -> dict:
    _registered_rows(rows)
    _, view, _ = parse_role(role)
    p = inherited.validate_token_probs(token_probs)
    validate_token_law(p, h_only=(view == "B" or p.shape[1] == 1))
    return inherited.role_arrays(rows, p, role,
                                 require_full_fit_support=require_full_fit_support)


def fit_role_slate(fit_rows: dict, fit_token_probs: np.ndarray,
                   validation_rows: dict, validation_token_probs: np.ndarray,
                   role: str, output_dir, seed: int, *, release_id: str,
                   slate: str = "standard") -> dict:
    role_arrays(fit_rows, fit_token_probs, role)
    role_arrays(validation_rows, validation_token_probs, role)
    return inherited.fit_role_slate(
        fit_rows, fit_token_probs, validation_rows, validation_token_probs,
        role, output_dir, seed, release_id=release_id, slate=slate)


def select_frozen_routes(validation_rows: dict, token_probs: np.ndarray,
                         role: str, routes: dict[str, dict], *,
                         release_id: str) -> dict:
    role_arrays(validation_rows, token_probs, role)
    return inherited.select_frozen_routes(
        validation_rows, token_probs, role, routes, release_id=release_id)


def score_frozen_route(rows: dict, token_probs: np.ndarray, lock: dict) -> dict:
    role_arrays(rows, token_probs, lock["role"])
    return inherited.score_frozen_route(rows, token_probs, lock)


def candidate_route_bank(role: str, release_id: str, own: dict, h_only: dict,
                         *, a_same: dict | None = None,
                         b_h_only: dict | None = None,
                         include_fixed_decoder: bool = False) -> dict[str, dict]:
    """Require all legal same-release and H-only ancestors for each role."""
    return inherited.build_role_route_bank(
        role, release_id, own, h_only, a_same=a_same,
        b_h_only=b_h_only, include_fixed_decoder=include_fixed_decoder)

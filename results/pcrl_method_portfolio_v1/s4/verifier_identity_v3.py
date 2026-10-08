"""Pure identity checks for S4 replay across historical fit and V2 locks.

This module never opens role data or imports campaign scoring code. File and
Git blob hashes are checked by the caller before these decoded objects arrive.
"""

from __future__ import annotations

import re


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_VIEWS_B = frozenset(("income", "occupation", "pair"))
_VIEWS_ANCHOR = frozenset(("r1", "r2", "pair"))


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _hash(value: object, reason: str) -> str:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None, reason)
    return value


def _view_hashes(record: dict, names: frozenset[str], field: str) -> None:
    values = record.get(field)
    _require(isinstance(values, dict) and set(values) == names, f"{field} view set differs")
    for name in names:
        _hash(values[name], f"{field}/{name} SHA differs")


def _verifier_unit(verifier: dict, unit_id: str, kind: str,
                   fit_sha: str, control_sha: str) -> dict:
    _require(verifier.get("schema") == "pcrl-s4-verifier-admission-v3" and
             verifier.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE",
             "S4 verifier admission is not effective")
    fit_allowed = verifier.get("accepted_fit_lock_sha256")
    control_allowed = verifier.get("accepted_control_v2_lock_sha256")
    _require(isinstance(fit_allowed, list) and fit_sha in fit_allowed and
             isinstance(control_allowed, list) and control_sha in control_allowed,
             "historical lock is not admitted")
    for value in fit_allowed + control_allowed:
        _hash(value, "accepted lock SHA differs")
    specs = verifier.get("unit_specs")
    _require(isinstance(specs, dict) and isinstance(specs.get(unit_id), dict),
             "S4 unit is not admitted")
    spec = specs[unit_id]
    _require(spec.get("kind") == kind and spec.get("fit_lock_sha256") == fit_sha and
             spec.get("control_v2_lock_sha256") == control_sha,
             "S4 unit historical lock binding differs")
    return spec


def _fit_lock(fit_lock: dict, fit_sha: str) -> None:
    _hash(fit_sha, "fit lock SHA differs")
    _require(fit_lock.get("schema") == "pcrl-method-portfolio-campaign-lock-v1" and
             fit_lock.get("status") == "LOCAL_GIT_SEALED_EFFECTIVE" and
             fit_lock.get("fit_admission") == "B_ONLY",
             "historical B fit lock identity differs")


def admit_b_replay_identity(*, verifier: dict, unit_id: str, arm: str, seed: int,
                            fit_lock: dict, fit_lock_sha256: str,
                            attempt_index: dict, attempt_index_sha256: str,
                            control_v2_index: dict, control_v2_index_sha256: str,
                            control_v2_lock_sha256: str) -> dict:
    """Check a B screen and its separate V2 probe before person-value replay."""
    fit_sha = _hash(fit_lock_sha256, "fit lock SHA differs")
    control_sha = _hash(control_v2_lock_sha256, "V2 lock SHA differs")
    attempt_sha = _hash(attempt_index_sha256, "attempt index SHA differs")
    probe_sha = _hash(control_v2_index_sha256, "V2 index SHA differs")
    spec = _verifier_unit(verifier, unit_id, "B_SCREEN", fit_sha, control_sha)
    _fit_lock(fit_lock, fit_sha)
    _require(isinstance(arm, str) and arm.startswith("B-") and seed in (0, 1) and
             spec.get("arm") == arm and spec.get("seed") == seed and
             arm in fit_lock.get("admitted_arms", {}).get("B", []),
             "B arm or seed is not admitted")
    screen = fit_lock.get("families", {}).get("B", {}).get("screen_unit_ids", {}).get(
        f"{arm}:s{seed}")
    _require(isinstance(screen, str) and spec.get("screen_unit_id") == screen and
             spec.get("attempt_index_sha256") == attempt_sha,
             "B screen or attempt SHA differs")
    _require(attempt_index.get("schema") == "pcrl-team-b-private-attempt-index-v1" and
             attempt_index.get("status") == "FROZEN_FOR_INDEPENDENT_REPLAY" and
             attempt_index.get("arm") == arm and attempt_index.get("seed") == seed and
             attempt_index.get("screen_unit_id") == screen and
             attempt_index.get("campaign_lock_sha256") == fit_sha,
             "B frozen attempt identity differs")
    probe_unit = spec.get("control_v2_unit_id")
    _require(isinstance(probe_unit, str) and
             spec.get("control_v2_index_sha256") == probe_sha and
             control_v2_index.get("schema") == "pcrl-team-b-real-control-index-v2" and
             control_v2_index.get("status") in ("PASS", "FAIL") and
             control_v2_index.get("arm") == arm and
             control_v2_index.get("seed") == seed and
             control_v2_index.get("queue_unit_id") == probe_unit and
             control_v2_index.get("screen_unit_id") == screen and
             control_v2_index.get("fit_lock_sha256") == fit_sha and
             control_v2_index.get("control_v2_lock_sha256") == control_sha and
             control_v2_index.get("attempt_index_sha256") == attempt_sha,
             "B V2 probe parent or lock differs")
    _view_hashes(control_v2_index, _VIEWS_B, "view_report_sha256")
    _view_hashes(control_v2_index, _VIEWS_B, "view_custody_receipt_sha256")
    status = control_v2_index["status"]
    return {"kind": "B_SCREEN", "unit_id": unit_id, "arm": arm, "seed": seed,
            "screen_unit_id": screen, "fit_lock_sha256": fit_sha,
            "attempt_index_sha256": attempt_sha, "control_v2_unit_id": probe_unit,
            "control_v2_lock_sha256": control_sha,
            "control_v2_index_sha256": probe_sha, "v2_probe_status": status,
            "comparison_eligible": status == "PASS"}


def admit_anchor_replay_identity(*, verifier: dict, unit_id: str, seed: int,
                                 fit_lock: dict, fit_lock_sha256: str,
                                 anchor_receipt: dict, anchor_receipt_sha256: str,
                                 control_v2_receipt: dict,
                                 control_v2_receipt_sha256: str,
                                 control_v2_lock_sha256: str) -> dict:
    """Check a feature anchor and its R2 V2 probe before person-value replay."""
    fit_sha = _hash(fit_lock_sha256, "fit lock SHA differs")
    control_sha = _hash(control_v2_lock_sha256, "V2 lock SHA differs")
    anchor_sha = _hash(anchor_receipt_sha256, "anchor receipt SHA differs")
    probe_sha = _hash(control_v2_receipt_sha256, "V2 receipt SHA differs")
    spec = _verifier_unit(verifier, unit_id, "FEATURE_ANCHOR", fit_sha, control_sha)
    _fit_lock(fit_lock, fit_sha)
    shared = f"S2-FEATURE-REF-RAW83-U-S{seed}"
    _require(seed in (0, 1) and spec.get("seed") == seed and
             spec.get("arm") == "FEATURE_REF_RAW83_U" and
             spec.get("shared_unit_id") == shared and
             spec.get("anchor_receipt_sha256") == anchor_sha and
             shared in fit_lock.get("admitted_shared_unit_ids", []),
             "feature anchor is not admitted")
    _require(anchor_receipt.get("schema") ==
             "pcrl-feature-ref-raw83-u-anchor-receipt-v1" and
             anchor_receipt.get("status") == "EXPLORATORY_DEVELOPMENT" and
             anchor_receipt.get("arm") == "FEATURE_REF_RAW83_U" and
             anchor_receipt.get("seed") == seed and
             anchor_receipt.get("campaign_lock_sha256") == fit_sha,
             "feature anchor parent differs")
    probe_unit = spec.get("control_v2_unit_id")
    _require(isinstance(probe_unit, str) and probe_unit == shared + "-V2-PROBE-R2" and
             spec.get("control_v2_receipt_sha256") == probe_sha and
             control_v2_receipt.get("schema") ==
             "pcrl-feature-ref-raw83-u-v2-probe-receipt-v1" and
             control_v2_receipt.get("status") in ("PASS", "FAIL") and
             control_v2_receipt.get("arm") == "FEATURE_REF_RAW83_U" and
             control_v2_receipt.get("seed") == seed and
             control_v2_receipt.get("s3_unit_id") == probe_unit and
             control_v2_receipt.get("parent_anchor_receipt_sha256") == anchor_sha and
             control_v2_receipt.get("parent_campaign_lock_sha256") == fit_sha and
             control_v2_receipt.get("control_lock_sha256") == control_sha,
             "feature anchor V2 probe parent or lock differs")
    _view_hashes(control_v2_receipt, _VIEWS_ANCHOR, "probe_report_sha256")
    _view_hashes(control_v2_receipt, _VIEWS_ANCHOR, "probe_custody_receipt_sha256")
    status = control_v2_receipt["status"]
    return {"kind": "FEATURE_ANCHOR", "unit_id": unit_id,
            "arm": "FEATURE_REF_RAW83_U", "seed": seed,
            "shared_unit_id": shared, "fit_lock_sha256": fit_sha,
            "anchor_receipt_sha256": anchor_sha, "control_v2_unit_id": probe_unit,
            "control_v2_lock_sha256": control_sha,
            "control_v2_receipt_sha256": probe_sha, "v2_probe_status": status,
            "comparison_eligible": status == "PASS"}

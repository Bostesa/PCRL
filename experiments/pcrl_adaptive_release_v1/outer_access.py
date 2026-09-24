"""One locked, provenance-verified path to 2018 outer-development labels.

This is intentionally separate from `roles.pooled_role`, which always refuses
outer labels. The coordinator writes the private unlock receipt only after the
selection lock is committed, pushed and independently remote-byte-verified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.pcrl_task_aligned_cuts_v1 import data
from . import inference, outer_pool, roles

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "results/pcrl_adaptive_release_v1/SELECTION_LOCK.json"
UNLOCK_PATH = ROOT / "results/pcrl_adaptive_release_v1/private/OUTER_UNLOCK.json"
CORRECTION_PATH = ROOT / "results/pcrl_adaptive_release_v1/SCORER_CORRECTION.json"
CORRECTION_UNLOCK_PATH = ROOT / "results/pcrl_adaptive_release_v1/private/SCORER_CORRECTION_UNLOCK.json"
CENSUS_PATH = ROOT / "results/pcrl_adaptive_release_v1/DATA_ROLE_COUNTS.json"
ORIGINAL_RESTORE_ROOT = ROOT / "results/pcrl_adaptive_release_v1/private/original_2018_restore"
BRANCH = "research/pcrl-adaptive-release-v1"
MANDATORY_FAMILIES = ("simple", "task_only", "gradient", "deterministic")
SCORING_CODE = ("outer_audit.py", "outer_access.py", "outer_pool.py",
                "inference_run.py", "inference.py", "external_audit.py",
                "release_specs.py")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hex(value: object, size: int) -> bool:
    return (isinstance(value, str) and len(value) == size and
            all(char in "0123456789abcdef" for char in value))


def scoring_code_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    result = {name: _sha(directory / name) for name in SCORING_CODE}
    result["paired_inference.py"] = _sha(
        ROOT / "experiments/pcrl_task_aligned_cuts_v1/inference.py")
    result["paired_uncertainty.py"] = _sha(
        ROOT / "experiments/pcrl_task_directed_release_v1/uncertainty.py")
    return result


def _scorer_hashes_authorized(lock: dict) -> bool:
    """Allow only the remotely verified, post-lock object-ID scorer repair."""
    frozen = lock.get("scoring_code_sha256")
    current = scoring_code_hashes()
    if frozen == current:
        return True
    if (not isinstance(frozen, dict) or set(frozen) != set(current) or
            not CORRECTION_PATH.is_file() or not CORRECTION_UNLOCK_PATH.is_file() or
            not LOCK_PATH.is_file()):
        return False
    changed = sorted(key for key in current if current[key] != frozen[key])
    if changed != ["outer_access.py", "outer_pool.py"]:
        return False
    lock_sha = _sha(LOCK_PATH)
    correction = json.loads(CORRECTION_PATH.read_text())
    receipt = json.loads(CORRECTION_UNLOCK_PATH.read_text())
    return (
        correction.get("schema") == "pcrl-adaptive-postlock-scorer-correction-v1" and
        correction.get("status") == "TECHNICAL_SCORER_REPAIR" and
        correction.get("lock_sha256") == lock_sha and
        correction.get("original_scoring_code_sha256") == frozen and
        correction.get("corrected_scoring_code_sha256") == current and
        correction.get("changed_modules") == changed and
        correction.get("score_outputs_written_before_repair") is False and
        correction.get("candidate_panel_unchanged") is True and
        receipt.get("schema") == "pcrl-adaptive-scorer-correction-unlock-v1" and
        receipt.get("lock_sha256") == lock_sha and
        receipt.get("correction_sha256") == _sha(CORRECTION_PATH) and
        receipt.get("remote_verified") is True and
        receipt.get("branch") == BRANCH and
        _hex(receipt.get("remote_commit_sha"), 40))


def verify_lock_structure(lock: dict) -> None:
    """Verify every endpoint and mandatory family without reading assessment data."""
    if (lock.get("schema") != "pcrl-adaptive-selection-lock-v1" or
            lock.get("status") != "LOCKED" or
            lock.get("study") != "pcrl_adaptive_release_v1" or
            lock.get("assessment_year") != 2018 or
            lock.get("outer_assessment_authorized") is not True or
            not _hex(lock.get("protocol_sha256"), 64) or
            not _hex(lock.get("input_index_sha256"), 64)):
        raise PermissionError("outer development panel was not frozen")
    slots = lock.get("slots")
    manifest = lock.get("family_manifest")
    anchors = lock.get("anchors")
    if (not isinstance(slots, list) or not 1 <= len(slots) <= 2 or
            not isinstance(manifest, dict) or
            not isinstance(manifest.get("endpoints"), list) or
            manifest.get("n_endpoints") != len(manifest["endpoints"]) or
            len({endpoint.get("id") for endpoint in manifest["endpoints"]}) !=
            len(manifest["endpoints"]) or
            not isinstance(anchors, dict) or set(anchors) != {"0", "1", "2"}):
        raise PermissionError("selection family or three-anchor panel is incomplete")
    try:
        rebuilt = inference.family_manifest(slots, alias_of=lock.get("alias_of", {}))
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("selection endpoint family cannot be reconstructed") from error
    if rebuilt != manifest:
        raise PermissionError("selection endpoint family differs from registered slots")
    if (not _scorer_hashes_authorized(lock) or
            lock.get("outer_population") != {
                "census_sha256": _sha(CENSUS_PATH),
                "pools": list(data.POOLS),
                "role": "outer_assessment"}):
        raise PermissionError("locked scorer or five-pool population differs")
    capability = lock.get("capability_manifest")
    expected_capability = inference.capability_endpoints(
        [slot["id"] for slot in slots])
    if capability != {
            "schema": 1, "endpoints": expected_capability,
            "n_endpoints": len(expected_capability),
            "multiplicity": "separate two-sided Bonferroni H-capability family",
            "scope": "2018 development; not a primary-clause rescue"}:
        raise PermissionError("H-capability uncertainty family was not frozen")
    representatives = lock.get("family_representatives")
    aliases = lock.get("alias_of", {})
    if (not isinstance(representatives, dict) or
            set(representatives) != set(MANDATORY_FAMILIES) or
            not isinstance(aliases, dict)):
        raise PermissionError("mandatory independent comparator families are absent")
    for slot in slots:
        if (not isinstance(slot.get("source_name"), str) or
                not slot["source_name"] or "D17" not in slot["comparators"]):
            raise PermissionError("slot source or D17 comparator is absent")
        for family in MANDATORY_FAMILIES:
            representative = representatives[family].get(slot["arm"])
            if (not isinstance(representative, str) or not representative or
                    representative not in slot["comparators"]):
                raise PermissionError("mandatory family representative is absent")
    needed = {name for endpoint in manifest["endpoints"]
              for name in (endpoint["plus"], endpoint["minus"])}
    for slot in slots:
        needed.add(slot["source_name"])
        needed.update(slot["comparators"])
    for alias, canonical in aliases.items():
        if alias not in needed and canonical not in needed:
            raise PermissionError("unrelated global alias in locked family")
        for anchor in ("0", "1", "2"):
            mapping = anchors[anchor].get("logical_to_canonical", {})
            if mapping.get(alias) != mapping.get(canonical):
                raise PermissionError("global alias is not exact across anchors")
    for record in anchors.values():
        if (not isinstance(record, dict) or
                not _hex(record.get("inner_panel_complete_sha256"), 64) or
                not _hex(record.get("inner_audit_sha256"), 64) or
                not isinstance(record.get("logical_to_canonical"), dict) or
                not isinstance(record.get("releases"), dict) or
                not record["releases"]):
            raise PermissionError("anchor audit objects are not pinned")
        mapping = record["logical_to_canonical"]
        for name in needed:
            canonical = mapping.get(name)
            if (not isinstance(canonical, str) or
                    canonical not in record["releases"]):
                raise PermissionError("locked endpoint does not resolve to scored release")
        for slot in slots:
            if mapping[slot["id"]] != mapping[slot["source_name"]]:
                raise PermissionError("slot differs from frozen selected release")
    external = lock.get("external_context")
    if external is not None:
        j = external.get("J") if isinstance(external, dict) else None
        if not isinstance(j, dict) or set(j) != {"0", "1", "2"}:
            raise PermissionError("contextual J panel is incomplete")
        for record in j.values():
            if (not isinstance(record, dict) or
                    not _hex(record.get("inner_complete_sha256"), 64) or
                    not _hex(record.get("inner_audit_sha256"), 64) or
                    not isinstance(record.get("source"), dict)):
                raise PermissionError("contextual J source is not pinned")
    return None


def verify_outer_unlock(selection_lock_path: str | Path,
                        expected_lock_sha256: str) -> dict:
    """Check the frozen panel and remote-verification receipt before loading rows."""
    path = Path(selection_lock_path).resolve()
    if path != LOCK_PATH.resolve() or not _hex(expected_lock_sha256, 64):
        raise PermissionError("registered selection lock path and SHA-256 required")
    if not path.is_file() or _sha(path) != expected_lock_sha256:
        raise PermissionError("selection lock bytes differ from expected SHA-256")
    lock = json.loads(path.read_text())
    verify_lock_structure(lock)
    if not UNLOCK_PATH.is_file():
        raise PermissionError("remote-verified outer unlock receipt is absent")
    unlock = json.loads(UNLOCK_PATH.read_text())
    if (unlock.get("schema") != "pcrl-adaptive-outer-unlock-v1" or
            unlock.get("lock_sha256") != expected_lock_sha256 or
            unlock.get("remote_verified") is not True or
            unlock.get("branch") != BRANCH or
            not _hex(unlock.get("remote_commit_sha"), 40)):
        raise PermissionError("outer unlock was not remotely verified")
    return lock


def load_locked_outer_role(index_path: str | Path, anchor: int,
                           selection_lock_path: str | Path,
                           expected_lock_sha256: str) -> dict:
    """Load original 2018 outer households once the registered lock is verified."""
    lock = verify_outer_unlock(selection_lock_path, expected_lock_sha256)
    if anchor not in (0, 1, 2):
        raise ValueError("registered anchor required")
    index_file = Path(index_path).resolve()
    if not index_file.is_file() or _sha(index_file) != lock["input_index_sha256"]:
        raise ValueError("2018 pinned input index differs from selection lock")
    index = data.index(index_file)
    if data._sanitized_record(index, anchor) is None:
        raise FileNotFoundError("verified label-stripped prepared receipt required")
    census = json.loads(CENSUS_PATH.read_text())
    if (census.get("schema") != "pcrl-adaptive-role-summary-v1" or
            census.get("outer_labels_accessed") is not False or
            census.get("global_role_overlap") != 0):
        raise ValueError("registered five-pool household census differs")
    return outer_pool.load_verified_outer(
        index, anchor, census["anchors"][str(anchor)]["outer_assessment"],
        ORIGINAL_RESTORE_ROOT)

"""Read-only source and FIT lineage gate for the A downstream sidecar."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "pcrl-a-downstream-source-manifest-v1"
STATUS = "LOCALLY_SEALED_DOWNSTREAM"
KEYS = frozenset({
    "schema", "status", "dated_utc", "producer_amendment_sha256",
    "original_lock_sha256", "unit_plan_sha256", "new_source_sha256",
    "attack_plan", "stage_scope", "scientific_scope_unchanged",
})
STAGE_SCOPE = ["attack", "diagnostic", "q", "reference", "release"]
SHARED = frozenset({"attack_eval.py", "attack_controls_v2.py",
                    "common_eval.py", "safe_data_guard.py"})
REGISTRATION_SCHEMA = "pcrl-a-downstream-source-registration-v1"
REGISTRATION_STATUS = "LOCALLY_REGISTERED_BY_R00"
REGISTRATION_KEYS = frozenset({
    "schema", "status", "registered_utc", "producer_amendment_sha256",
    "manifest_relative_path", "manifest_sha256", "manifest_commit_full_sha",
    "stage_scope", "independent_review_sha256", "source_map_sha256",
})
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_plan_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _source_map(spec: dict) -> dict[str, str]:
    rows = spec.get("sources")
    if (type(rows) is not list or not rows or
            any(type(row) is not dict or set(row) != {"path", "sha256"}
                for row in rows)):
        raise PermissionError("A downstream R03 source ledger is malformed")
    mapped = {}
    for row in rows:
        path, digest = row["path"], row["sha256"]
        if (type(path) is not str or type(digest) is not str or
                HEX64.fullmatch(digest) is None or path in mapped):
            raise PermissionError("A downstream R03 source ledger duplicates a path")
        mapped[path] = digest
    return mapped


def _check_campaign_source(campaign: Path, sources: dict[str, str],
                           relative: str, digest: str) -> None:
    if (type(relative) is not str or type(digest) is not str or
            HEX64.fullmatch(digest) is None or Path(relative).is_absolute() or
            ".." in Path(relative).parts):
        raise PermissionError("A downstream source record is malformed")
    path = (campaign / relative).resolve(strict=True)
    if (not path.is_relative_to(campaign) or not path.is_file() or
            _sha(path) != digest or sources.get(str(path)) != digest):
        raise PermissionError("A downstream source differs from R03 pin")


def verify_downstream_sidecar(*, sidecar_path: Path, sidecar_sha256: str,
                              producer_path: Path, producer_sha256: str,
                              original_lock_path: Path,
                              original_lock_sha256: str,
                              campaign_root: Path, spec: dict,
                              required_new_paths: frozenset[str]) -> dict:
    """Verify unchanged producer files, every new file, and inherited readers."""
    sidecar_path = Path(sidecar_path).resolve(strict=True)
    producer_path = Path(producer_path).resolve(strict=True)
    original_lock_path = Path(original_lock_path).resolve(strict=True)
    campaign = Path(campaign_root).resolve(strict=True)
    if (type(sidecar_sha256) is not str or HEX64.fullmatch(sidecar_sha256) is None or
            type(producer_sha256) is not str or HEX64.fullmatch(producer_sha256) is None or
            type(original_lock_sha256) is not str or
            HEX64.fullmatch(original_lock_sha256) is None or
            _sha(sidecar_path) != sidecar_sha256 or
            _sha(producer_path) != producer_sha256 or
            _sha(original_lock_path) != original_lock_sha256):
        raise PermissionError("A downstream sidecar, producer or lock SHA differs")
    sources = _source_map(spec)
    if (spec.get("amendment_sha256") != producer_sha256 or
            spec.get("downstream_manifest_sha256") != sidecar_sha256 or
            spec.get("downstream_manifest_path") != str(sidecar_path) or
            sources.get(str(sidecar_path)) != sidecar_sha256 or
            sources.get(str(producer_path)) != producer_sha256 or
            sources.get(str(original_lock_path)) != original_lock_sha256):
        raise PermissionError("R03 unit does not bind sidecar and producer bytes")
    producer = json.loads(producer_path.read_text())
    sidecar = json.loads(sidecar_path.read_text())
    if (type(sidecar) is not dict or set(sidecar) != KEYS or
            sidecar.get("schema") != SCHEMA or sidecar.get("status") != STATUS or
            sidecar.get("producer_amendment_sha256") != producer_sha256 or
            sidecar.get("original_lock_sha256") != original_lock_sha256 or
            sidecar.get("unit_plan_sha256") !=
            _unit_plan_sha(producer.get("unit_plan")) or
            sidecar.get("stage_scope") != STAGE_SCOPE or
            sidecar.get("scientific_scope_unchanged") is not True):
        raise PermissionError("A downstream sidecar changes producer scientific scope")
    try:
        dated = dt.datetime.fromisoformat(sidecar["dated_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PermissionError("A downstream sidecar date differs") from exc
    if dated.tzinfo is None:
        raise PermissionError("A downstream sidecar date needs a timezone")
    new = sidecar.get("new_source_sha256")
    attack_plan = sidecar.get("attack_plan")
    inherited = (attack_plan.get("inherited_shared_source_sha256")
                 if type(attack_plan) is dict and
                 set(attack_plan) == {"inherited_shared_source_sha256"} else None)
    baseline = producer.get("implementation_source_sha256")
    if (type(new) is not dict or not required_new_paths <= set(new) or
            type(inherited) is not dict or set(inherited) != SHARED or
            type(baseline) is not dict or set(new) & set(baseline)):
        raise PermissionError("A downstream source allowlist differs")
    for relative, digest in (*baseline.items(), *new.items()):
        _check_campaign_source(campaign, sources, relative, digest)
    shared_root = campaign.parent / "pcrl_method_portfolio_v1/shared"
    for basename, digest in inherited.items():
        if type(digest) is not str or HEX64.fullmatch(digest) is None:
            raise PermissionError("inherited reader SHA is malformed")
        path = (shared_root / basename).resolve(strict=True)
        if _sha(path) != digest or sources.get(str(path)) != digest:
            raise PermissionError("inherited reader differs from sidecar and R03")
    return sidecar


def verify_downstream_registration(*, registration_path: Path,
                                   registration_sha256: str,
                                   sidecar_path: Path, sidecar_sha256: str,
                                   producer_sha256: str, repo_root: Path,
                                   spec: dict) -> dict:
    """Require R00's byte-pinned receipt and the committed sidecar bytes."""
    repo = Path(repo_root).resolve(strict=True)
    registration = Path(registration_path).resolve(strict=True)
    sidecar = Path(sidecar_path).resolve(strict=True)
    if (type(registration_sha256) is not str or
            HEX64.fullmatch(registration_sha256) is None or
            _sha(registration) != registration_sha256 or
            spec.get("downstream_registration_path") != str(registration) or
            spec.get("downstream_registration_sha256") != registration_sha256 or
            _source_map(spec).get(str(registration)) != registration_sha256):
        raise PermissionError("A downstream R00 registration SHA differs")
    if (type(sidecar_sha256) is not str or
            HEX64.fullmatch(sidecar_sha256) is None or
            _sha(sidecar) != sidecar_sha256 or
            type(producer_sha256) is not str or
            HEX64.fullmatch(producer_sha256) is None):
        raise PermissionError("A downstream manifest SHA differs from registration")
    value = json.loads(registration.read_text())
    if (type(value) is not dict or set(value) != REGISTRATION_KEYS or
            value.get("schema") != REGISTRATION_SCHEMA or
            value.get("status") != REGISTRATION_STATUS or
            value.get("producer_amendment_sha256") != producer_sha256 or
            value.get("manifest_sha256") != sidecar_sha256 or
            value.get("stage_scope") != STAGE_SCOPE or
            type(value.get("independent_review_sha256")) is not str or
            HEX64.fullmatch(value["independent_review_sha256"]) is None or
            type(value.get("manifest_commit_full_sha")) is not str or
            HEX40.fullmatch(value["manifest_commit_full_sha"]) is None):
        raise PermissionError("A downstream R00 registration fields differ")
    try:
        dated = dt.datetime.fromisoformat(value["registered_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PermissionError("A downstream registration date differs") from exc
    if dated.tzinfo is None:
        raise PermissionError("A downstream registration date needs a timezone")
    if (not sidecar.is_relative_to(repo) or
            value.get("manifest_relative_path") != sidecar.relative_to(repo).as_posix()):
        raise PermissionError("A downstream registered manifest path differs")
    manifest = json.loads(sidecar.read_text())
    source_map = {
        "new_source_sha256": manifest.get("new_source_sha256"),
        "inherited_shared_source_sha256":
            manifest.get("attack_plan", {}).get("inherited_shared_source_sha256"),
    }
    expected_map_sha = hashlib.sha256(json.dumps(
        source_map, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if value.get("source_map_sha256") != expected_map_sha:
        raise PermissionError("A downstream registered source map differs")
    try:
        committed = subprocess.run(
            ["git", "show", f"{value['manifest_commit_full_sha']}:{value['manifest_relative_path']}"],
            cwd=repo, capture_output=True, check=True, timeout=10).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PermissionError("A downstream manifest commit is unavailable") from exc
    if committed != sidecar.read_bytes():
        raise PermissionError("A downstream manifest commit differs from live sidecar")
    try:
        registration_relative = registration.relative_to(repo).as_posix()
        committed_registration = subprocess.run(
            ["git", "show", f"HEAD:{registration_relative}"],
            cwd=repo, capture_output=True, check=True, timeout=10).stdout
    except (ValueError, OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise PermissionError("A downstream registration lacks Git custody") from exc
    if committed_registration != registration.read_bytes():
        raise PermissionError("A downstream registration differs from HEAD Git custody")
    try:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor",
             value["manifest_commit_full_sha"], "HEAD"],
            cwd=repo, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PermissionError("A downstream manifest commit is outside HEAD ancestry") from exc
    if ancestor.returncode != 0:
        raise PermissionError("A downstream manifest commit is outside HEAD ancestry")
    return value


def verify_predecessor_fit(*, parent: dict, fit_receipt_path: Path,
                           producer_sha256: str, fit_unit_id: str,
                           arm_id: str, seed: int) -> dict:
    """Accept only a selected FIT from the unchanged producer amendment."""
    fit_receipt_path = Path(fit_receipt_path).resolve(strict=True)
    receipt_sha = _sha(fit_receipt_path)
    parent_spec = parent.get("spec", {})
    if (parent.get("status") != "done" or
            parent_spec.get("unit_id") != fit_unit_id or
            parent_spec.get("stage") != "a" or
            parent_spec.get("mode") != "real_data" or
            parent_spec.get("amendment_sha256") != producer_sha256 or
            parent_spec.get("arm_id") != arm_id or
            parent_spec.get("seed") != seed or
            parent_spec.get("outputs") != [str(fit_receipt_path)] or
            parent.get("output_sha256", {}).get(str(fit_receipt_path)) != receipt_sha):
        raise PermissionError("A downstream parent lacks producer FIT custody")
    receipt = json.loads(fit_receipt_path.read_text())
    if (receipt.get("status") != "SELECTED_FIT_POLICY" or
            receipt.get("unit_id") != fit_unit_id or
            receipt.get("arm_id") != arm_id or
            receipt.get("source_seed") != seed or
            receipt.get("a_amendment_sha256") != producer_sha256 or
            (fit_receipt_path.parent / "INCOMPLETE_RESOURCE.json").exists()):
        raise PermissionError("A downstream parent FIT was not selected")
    return receipt


__all__ = ["verify_downstream_sidecar", "verify_downstream_registration",
           "verify_predecessor_fit"]

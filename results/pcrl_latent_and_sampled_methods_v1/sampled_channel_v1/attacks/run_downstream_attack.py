"""Queue-only sidecar gate for attacks on immutable A producer releases.

The original selected/reference workers retain their producer-pinned bytes.
This wrapper checks the separately sealed downstream reader/source contract in
the same queue child, then calls the original worker without a subprocess.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCHEMA = "pcrl-a-downstream-source-manifest-v1"
STATUS = "LOCALLY_SEALED_DOWNSTREAM"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SHARED = ("attack_eval.py", "attack_controls_v2.py", "common_eval.py",
          "safe_data_guard.py")
MANIFEST_KEYS = {"schema", "status", "dated_utc", "producer_amendment_sha256",
                 "original_lock_sha256", "unit_plan_sha256", "new_source_sha256",
                 "attack_plan", "stage_scope", "scientific_scope_unchanged"}
SELECTED_MODULE = "sampled_channel_v1/attacks/runner.py"
REFERENCE_MODULE = "sampled_channel_v1/attacks/run_reference_attack.py"
THIS_MODULE = "sampled_channel_v1/attacks/run_downstream_attack.py"
RENDERER_MODULE = "sampled_channel_v1/attacks/render_attack_specs.py"
MODULE_ID = ("results.pcrl_latent_and_sampled_methods_v1."
             "sampled_channel_v1.attacks.run_downstream_attack")
REGISTRATION_SCHEMA = "pcrl-a-downstream-source-registration-v1"
REGISTRATION_STATUS = "LOCALLY_REGISTERED_BY_R00"
REGISTRATION_KEYS = {"schema", "status", "registered_utc",
                     "producer_amendment_sha256", "manifest_relative_path",
                     "manifest_sha256", "manifest_commit_full_sha",
                     "stage_scope", "independent_review_sha256",
                     "source_map_sha256"}
STAGE_SCOPE = ["attack", "diagnostic", "q", "reference", "release"]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: object) -> str:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("downstream source digest is malformed")
    return value


def _dated(value: object) -> None:
    if type(value) is not str:
        raise PermissionError("downstream registration date is absent")
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermissionError("downstream registration date is malformed") from exc
    if stamp.tzinfo is None:
        raise PermissionError("downstream registration date lacks timezone")


def _campaign_file(campaign: Path, relative: str) -> Path:
    part = Path(relative)
    if (type(relative) is not str or part.is_absolute() or not part.parts or
            ".." in part.parts):
        raise PermissionError("downstream source path escapes campaign")
    supplied = campaign / part
    if any(path.is_symlink() for path in (supplied, *supplied.parents)
           if path != campaign.parent):
        raise PermissionError("downstream source path is redirected")
    path = supplied.resolve(strict=True)
    if not path.is_relative_to(campaign):
        raise PermissionError("downstream source path is redirected")
    return path


def _queue_spec(unit_id: str) -> dict:
    root = os.environ.get("PCRL_QUEUE_ROOT")
    if not root:
        raise PermissionError("R03 queue root is absent")
    state_path = Path(root) / "state.json"
    state = json.loads(state_path.read_text())
    spec = state.get("units", {}).get(unit_id, {}).get("spec")
    if type(spec) is not dict or spec.get("unit_id") != unit_id:
        raise PermissionError("active downstream queue spec is absent")
    return spec


def _sources(spec: dict) -> dict[Path, str]:
    rows = spec.get("sources")
    if type(rows) is not list:
        raise PermissionError("downstream source list is absent")
    result = {}
    for row in rows:
        if type(row) is not dict or row.keys() != {"path", "sha256"}:
            raise PermissionError("downstream queue source record differs")
        path = Path(row["path"]).resolve(strict=True)
        digest = _digest(row["sha256"])
        if path in result:
            raise PermissionError("downstream queue source is duplicated")
        result[path] = digest
    return result


def _pinned(path: Path, digest: str, sources: dict[Path, str]) -> None:
    if path.is_symlink() or sources.get(path.resolve(strict=True)) != digest or \
            _sha(path) != digest:
        raise PermissionError("downstream source differs from sealed queue and manifest")


def _committed_blob(repo: Path, revision: str, path: Path) -> bytes:
    relative = path.relative_to(repo).as_posix()
    result = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=repo,
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise PermissionError("downstream source is not in local Git custody")
    return result.stdout


def validate_registration(*, campaign: Path, producer_sha256: str,
                          manifest_path: Path, manifest_sha256: str,
                          manifest: dict,
                          registration_path: Path,
                          registration_sha256: str) -> dict:
    """Require R00's separately committed local registration before labels."""
    if registration_path.is_symlink():
        raise PermissionError("downstream registration path is a symlink")
    registration_path = registration_path.resolve(strict=True)
    if (registration_path.parent != campaign or
            _sha(registration_path) != _digest(registration_sha256)):
        raise PermissionError("downstream registration location or bytes differ")
    record = json.loads(registration_path.read_text())
    commit = record.get("manifest_commit_full_sha")
    repo = campaign.parents[1]
    relative = manifest_path.relative_to(repo).as_posix()
    source_map = {
        "new_source_sha256": manifest.get("new_source_sha256"),
        "inherited_shared_source_sha256": manifest.get(
            "attack_plan", {}).get("inherited_shared_source_sha256"),
    }
    map_sha = hashlib.sha256(json.dumps(
        source_map, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if (set(record) != REGISTRATION_KEYS or
            record.get("schema") != REGISTRATION_SCHEMA or
            record.get("status") != REGISTRATION_STATUS or
            record.get("producer_amendment_sha256") != producer_sha256 or
            record.get("manifest_relative_path") != relative or
            record.get("manifest_sha256") != manifest_sha256 or
            record.get("source_map_sha256") != map_sha or
            record.get("stage_scope") != STAGE_SCOPE or
            type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None or
            type(record.get("independent_review_sha256")) is not str or
            HEX64.fullmatch(record["independent_review_sha256"]) is None):
        raise PermissionError("R00 downstream registration does not bind manifest")
    _dated(record["registered_utc"])
    if (hashlib.sha256(_committed_blob(repo, commit, manifest_path)).hexdigest()
            != manifest_sha256 or
            hashlib.sha256(_committed_blob(repo, "HEAD", registration_path)).hexdigest()
            != registration_sha256):
        raise PermissionError("downstream manifest/registration lack Git custody")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit,
                               "HEAD"], cwd=repo, capture_output=True,
                              check=False)
    if ancestor.returncode != 0:
        raise PermissionError("downstream manifest commit is outside local history")
    return record


def validate_downstream(*, unit_id: str, producer_amendment_path: Path,
                        producer_amendment_sha256: str,
                        downstream_manifest_path: Path,
                        downstream_manifest_sha256: str,
                        downstream_registration_path: Path,
                        downstream_registration_sha256: str,
                        original_lock_path: Path, original_lock_sha256: str,
                        private_root: Path, argv: list[str],
                        spec: dict | None = None) -> tuple[dict, dict, dict]:
    """Fail before ATT labels for every sidecar, module and wire-source pin."""
    producer_amendment_path = Path(producer_amendment_path)
    downstream_manifest_path = Path(downstream_manifest_path)
    downstream_registration_path = Path(downstream_registration_path)
    original_lock_path = Path(original_lock_path)
    if any(path.is_symlink() for path in (producer_amendment_path,
                                         downstream_manifest_path,
                                         original_lock_path)):
        raise PermissionError("downstream protocol path is a symlink")
    producer_amendment_path = producer_amendment_path.resolve(strict=True)
    downstream_manifest_path = downstream_manifest_path.resolve(strict=True)
    original_lock_path = original_lock_path.resolve(strict=True)
    campaign = producer_amendment_path.parent
    if (downstream_manifest_path.parent != campaign or
            original_lock_path.parent != campaign):
        raise PermissionError("downstream protocol files are outside campaign")
    producer_sha = _digest(producer_amendment_sha256)
    manifest_sha = _digest(downstream_manifest_sha256)
    registration_sha = _digest(downstream_registration_sha256)
    lock_sha = _digest(original_lock_sha256)
    if (_sha(producer_amendment_path) != producer_sha or
            _sha(downstream_manifest_path) != manifest_sha or
            _sha(original_lock_path) != lock_sha):
        raise PermissionError("downstream protocol bytes differ")
    producer = json.loads(producer_amendment_path.read_text())
    manifest = json.loads(downstream_manifest_path.read_text())
    plan_sha = hashlib.sha256(json.dumps(
        producer.get("unit_plan"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
    if (set(manifest) != MANIFEST_KEYS or
            manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS or
            manifest.get("producer_amendment_sha256") != producer_sha or
            manifest.get("original_lock_sha256") != lock_sha or
            manifest.get("unit_plan_sha256") != plan_sha or
            manifest.get("stage_scope") !=
            ["attack", "diagnostic", "q", "reference", "release"] or
            manifest.get("scientific_scope_unchanged") is not True or
            producer.get("original_lock_sha256") != lock_sha or
            producer.get("status") != "LOCALLY_SEALED_A_STAGE"):
        raise PermissionError("downstream source manifest lacks producer lineage")
    _dated(manifest["dated_utc"])
    validate_registration(
        campaign=campaign, producer_sha256=producer_sha,
        manifest_path=downstream_manifest_path, manifest_sha256=manifest_sha,
        manifest=manifest,
        registration_path=downstream_registration_path,
        registration_sha256=registration_sha)
    plan = producer.get("unit_plan", {}).get(unit_id)
    if type(plan) is not dict or plan.get("kind") not in (
            "attack", "reference_attack", "diagnostic_attack"):
        raise PermissionError("downstream unit is absent from original graph")
    if spec is None:
        spec = _queue_spec(unit_id)
    if (spec.get("amendment_sha256") != producer_sha or
            spec.get("downstream_manifest_sha256") != manifest_sha or
            spec.get("downstream_manifest_path") != str(downstream_manifest_path) or
            spec.get("downstream_registration_sha256") != registration_sha or
            spec.get("downstream_registration_path") != str(
                downstream_registration_path.resolve(strict=True)) or
            spec.get("unit_id") != unit_id or
            spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("data_access") !=
            "ATTACK_FIT_NEW_SEX_THEN_INNER_SELECTION_SEX" or
            spec.get("deps") != plan.get("deps") or
            spec.get("arm_id") != plan.get(
                "arm_id", plan.get("reference_id")) or
            spec.get("seed") != plan.get("seed") or
            spec.get("realization") != plan.get("realization") or
            spec.get("output_id") != unit_id or
            spec.get("outputs") !=
            [str(Path(private_root) / "ATTACK_ATTEMPT_INDEX.json")]):
        raise PermissionError("downstream queue unit differs from original graph")
    command = spec.get("argv")
    if (type(command) is not list or len(command) < 4 or
            command[1:3] != ["-m", MODULE_ID] or command[3:] != argv):
        raise PermissionError("downstream worker argv differs from queue spec")
    sources = _sources(spec)
    _pinned(producer_amendment_path, producer_sha, sources)
    _pinned(downstream_manifest_path, manifest_sha, sources)
    _pinned(downstream_registration_path, registration_sha, sources)
    _pinned(original_lock_path, lock_sha, sources)
    new_sources = manifest.get("new_source_sha256")
    if (type(new_sources) is not dict or
            not {THIS_MODULE, RENDERER_MODULE} <= set(new_sources)):
        raise PermissionError("downstream manifest lacks wrapper/renderer sources")
    for relative, digest in new_sources.items():
        path = _campaign_file(campaign, relative)
        _pinned(path, _digest(digest), sources)
    if Path(__file__).resolve(strict=True) != _campaign_file(campaign, THIS_MODULE):
        raise PermissionError("executing downstream wrapper origin differs")
    old_sources = producer.get("implementation_source_sha256")
    if type(old_sources) is not dict or not old_sources:
        raise PermissionError("original A implementation source map is absent")
    for relative, digest in old_sources.items():
        path = _campaign_file(campaign, relative)
        _pinned(path, _digest(digest), sources)
    attack_plan = manifest.get("attack_plan")
    if type(attack_plan) is not dict or set(attack_plan) != {
            "inherited_shared_source_sha256"}:
        raise PermissionError("downstream attack plan contains unsealed fields")
    shared_pins = attack_plan.get("inherited_shared_source_sha256")
    if type(shared_pins) is not dict or set(shared_pins) != set(SHARED):
        raise PermissionError("downstream inherited reader pins are incomplete")
    shared_dir = campaign.parent / "pcrl_method_portfolio_v1" / "shared"
    for name in SHARED:
        _pinned(shared_dir / name, _digest(shared_pins[name]), sources)
    from results.pcrl_latent_and_sampled_methods_v1.r03.a_graph import compile_graph
    compile_graph(producer_amendment_path, producer_sha, original_lock_path,
                  lock_sha, source_root=campaign, expected_units=220)
    from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import (
        assert_active_from_env,
    )
    assert_active_from_env(unit_id=unit_id, stage="a")
    return producer, manifest, spec


def _worker_module(kind: str, campaign: Path, producer: dict,
                   sources: dict[Path, str]):
    """Import only the original worker and prove its actual module origin."""
    if kind == "attack":
        from . import runner as worker
        relative = SELECTED_MODULE
    else:
        from . import run_reference_attack as worker
        relative = REFERENCE_MODULE
    expected = _campaign_file(campaign, relative)
    origin = Path(worker.__file__).resolve(strict=True)
    digest = _digest(producer["implementation_source_sha256"][relative])
    if origin != expected:
        raise PermissionError("imported A attack worker origin differs")
    _pinned(origin, digest, sources)
    shared_dir = campaign.parent / "pcrl_method_portfolio_v1" / "shared"
    for name in SHARED:
        module = sys.modules.get(name.removesuffix(".py"))
        if module is None or Path(module.__file__).resolve(strict=True) != \
                (shared_dir / name).resolve(strict=True):
            raise PermissionError("imported inherited reader origin differs")
    return worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sidecar-gated A ATT/INNER attack")
    for flag in ("unit-id", "kind", "producer-amendment-path",
                 "producer-amendment-sha256", "downstream-manifest-path",
                 "downstream-manifest-sha256", "downstream-registration-path",
                 "downstream-registration-sha256", "original-lock-path",
                 "original-lock-sha256", "role-manifest-path",
                 "role-manifest-sha256", "att-release-dir",
                 "inner-release-dir", "private-root"):
        parser.add_argument("--" + flag, required=True)
    for flag in ("policy-package-dir", "q-table-dir", "fixed-map-receipt",
                 "fixed-map-npz"):
        parser.add_argument("--" + flag)
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw)
    producer, _, spec = validate_downstream(
        unit_id=args.unit_id,
        producer_amendment_path=Path(args.producer_amendment_path),
        producer_amendment_sha256=args.producer_amendment_sha256,
        downstream_manifest_path=Path(args.downstream_manifest_path),
        downstream_manifest_sha256=args.downstream_manifest_sha256,
        downstream_registration_path=Path(args.downstream_registration_path),
        downstream_registration_sha256=args.downstream_registration_sha256,
        original_lock_path=Path(args.original_lock_path),
        original_lock_sha256=args.original_lock_sha256,
        private_root=Path(args.private_root), argv=list(raw))
    kind = producer["unit_plan"][args.unit_id]["kind"]
    if args.kind != kind:
        raise PermissionError("downstream worker kind differs from original unit")
    sources = _sources(spec)
    campaign = Path(args.producer_amendment_path).resolve(strict=True).parent
    worker = _worker_module(kind, campaign, producer, sources)
    # The existing workers enforce full parent receipt, sampled wire, Q table,
    # ATT-before-INNER and private output checks before their own first label read.
    if kind == "attack":
        if (args.policy_package_dir is None or args.q_table_dir is None or
                args.fixed_map_receipt is not None or args.fixed_map_npz is not None):
            raise PermissionError("selected attack package flags differ")
        outcome = worker.run_sampled_attack(
            policy_package_dir=args.policy_package_dir,
            att_release_dir=args.att_release_dir,
            inner_release_dir=args.inner_release_dir,
            q_table_dir=args.q_table_dir,
            role_manifest_path=args.role_manifest_path,
            role_manifest_sha256=args.role_manifest_sha256,
            original_lock_path=args.original_lock_path,
            original_lock_sha256=args.original_lock_sha256,
            a_amendment_path=args.producer_amendment_path,
            a_amendment_sha256=args.producer_amendment_sha256,
            queue_unit_id=args.unit_id, private_root=args.private_root)
    else:
        finite = kind == "diagnostic_attack" or producer["unit_plan"][args.unit_id].get(
            "reference_id") not in ("U0", "Ucal")
        if (args.policy_package_dir is not None or
                (args.q_table_dir is not None) != finite or
                (args.fixed_map_receipt is not None) != finite or
                (args.fixed_map_npz is not None) != finite):
            raise PermissionError("reference attack public-map flags differ")
        outcome = worker.run_reference_attack(
            unit_id=args.unit_id,
            original_lock_path=args.original_lock_path,
            original_lock_sha256=args.original_lock_sha256,
            a_amendment_path=args.producer_amendment_path,
            a_amendment_sha256=args.producer_amendment_sha256,
            role_manifest_path=args.role_manifest_path,
            role_manifest_sha256=args.role_manifest_sha256,
            att_release_dir=args.att_release_dir,
            inner_release_dir=args.inner_release_dir,
            private_root=args.private_root,
            fixed_map_receipt=args.fixed_map_receipt,
            fixed_map_npz=args.fixed_map_npz,
            q_table_dir=args.q_table_dir)
    print(json.dumps({"status": outcome["status"],
                      "index_sha256": outcome["index_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

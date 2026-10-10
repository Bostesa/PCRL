"""R03-only cached A release for one arm, role and precommitted draw.

This worker reads U0 scores and T* fine cells on one admitted role. It never
opens task or SEX labels. R14 attacks consume its write-once private package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import TStarSourceStore
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.control_package import (
    load_selected_policy_package,
)

from .admitted_fit import MANDATORY_IMPLEMENTATION, _sha, validate_a_amendment_envelope
from .materialize import materialize_role
from .release import CachedJointRelease, _bank_hash, _mechanism_hash
from .run_admitted import _binding, _independent_source_gate


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ROLES = frozenset({"ATTACK_FIT_NEW", "INNER_SELECTION"})
RELEASE_IMPLEMENTATION = MANDATORY_IMPLEMENTATION | frozenset({
    "sampled_channel_v1/impl/run_release.py",
    "sampled_channel_v1/impl/materialize.py",
    "sampled_channel_v1/controls/control_package.py",
    "sampled_channel_v1/controls/deterministic_policy.py",
    "sampled_channel_v1/controls/factor_controls.py",
})


def _spec_source(spec: dict, path: Path, expected_sha256: str) -> None:
    listed = {Path(row["path"]).resolve(): row["sha256"]
              for row in spec.get("sources", [])}
    if (type(expected_sha256) is not str or HEX64.fullmatch(expected_sha256) is None or
            listed.get(path.resolve()) != expected_sha256 or _sha(path) != expected_sha256):
        raise PermissionError("A release source bytes differ from R03 unit")


def _runtime_origin_guard(campaign: Path) -> None:
    prefix = "results.pcrl_latent_and_sampled_methods_v1"
    origin = {
        f"{prefix}.sampled_channel_v1.impl.{name}":
            f"sampled_channel_v1/impl/{name}.py"
        for name in ("admitted_fit", "run_admitted", "run_release", "materialize",
                     "bank", "channel", "fit", "source_bridge", "policy_package",
                     "release")
    } | {
        f"{prefix}.sampled_channel_v1.controls.{name}":
            f"sampled_channel_v1/controls/{name}.py"
        for name in ("control_package", "deterministic_policy", "factor_controls")
    } | {
        f"{prefix}.r01.role_store": "r01/role_store.py",
        f"{prefix}.r01.tstar_source": "r01/tstar_source.py",
        f"{prefix}.r03.compute_queue": "r03/compute_queue.py",
    }
    for module_name, relative in origin.items():
        module = sys.modules.get(module_name)
        if module is None and module_name.endswith(".run_release"):
            module = sys.modules.get("__main__")
        if (module is None or Path(getattr(module, "__file__", "")).resolve() !=
                (campaign / relative).resolve(strict=True)):
            raise PermissionError("A release imported source differs from pinned bytes")


def admit_release_unit(*, unit_id: str, lock_path, amendment_path,
                       amendment_sha256: str, output_dir) -> tuple[dict, dict, dict]:
    """R03 and R00 gate, before private roots, policy or any role data read."""
    active = assert_active_from_env(unit_id=unit_id, stage="a")
    lock = Path(lock_path).resolve(strict=True)
    amendment_file = Path(amendment_path).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    if (_sha(lock) != active["seal_sha256"] or
            _sha(amendment_file) != amendment_sha256):
        raise PermissionError("A release original/amended lock SHA differs")
    original = json.loads(lock.read_text())
    amendment = json.loads(amendment_file.read_text())
    validate_a_amendment_envelope(amendment)
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != active["seal_sha256"]):
        raise PermissionError("A release has no effective stage seal")
    pins = original.get("source_files_sha256", {})
    bindings = amendment.get("role_bindings", {})
    implementation = amendment.get("implementation_source_sha256", {})
    if (bindings.get("repair_manifest_sha256") != pins.get("source_role_manifest") or
            bindings.get("source_audit_sha256") != pins.get("r01/SOURCE_AUDIT.json") or
            implementation.get("r03/compute_queue.py") != pins.get("r03/compute_queue.py") or
            not RELEASE_IMPLEMENTATION <= set(implementation)):
        raise PermissionError("A release source inventory differs from original lock")
    queue = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue / "state.json").read_text())
    amendments = [item for item in state.get("amendments", [])
                  if "a" in item.get("stages", [])]
    spec = state.get("units", {}).get(unit_id, {}).get("spec", {})
    plan = amendment.get("unit_plan", {}).get(unit_id)
    if (not amendments or amendments[-1].get("sha256") != amendment_sha256 or
            Path(amendments[-1].get("path", "")).resolve() != amendment_file or
            spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("amendment_sha256") != amendment_sha256 or
            spec.get("outputs") != [str(output / "RELEASE_RECEIPT.json")] or
            output.exists() or
            type(plan) is not dict or plan.get("kind") != "release" or
            plan.get("arm_id") != spec.get("arm_id") or
            plan.get("seed") != spec.get("seed") or
            plan.get("role") != spec.get("role") or
            plan.get("realization") != spec.get("realization") or
            plan.get("output_id") != spec.get("output_id") or
            plan.get("role") not in ROLES or plan.get("seed") not in (0, 1) or
            plan.get("realization") not in (0, 1, 2) or
            type(plan.get("cpu_seconds")) not in (int, float) or
            plan["cpu_seconds"] <= 0 or
            type(plan.get("wall_seconds")) not in (int, float) or
            plan["wall_seconds"] <= 0 or
            spec.get("cpu_reserve_s", 0) < plan["cpu_seconds"] or
            spec.get("max_wall_s", 0) < plan["wall_seconds"]):
        raise PermissionError("A release unit/role/draw/resource plan differs")
    for relative in RELEASE_IMPLEMENTATION:
        expected = implementation[relative]
        path = (lock.parent / relative).resolve(strict=True)
        if not path.is_relative_to(lock.parent):
            raise PermissionError("A release implementation escapes campaign package")
        _spec_source(spec, path, expected)
    _spec_source(spec, lock, active["seal_sha256"])
    _spec_source(spec, amendment_file, amendment_sha256)
    public_stream = (lock.parent /
                     "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json")
    streams = amendment.get("realizations", {})
    _spec_source(spec, public_stream, streams.get("stream_root_hash_manifest_sha256"))
    manifest = json.loads(public_stream.read_text())
    if (streams.get("ids") != [0, 1, 2] or
            streams.get("release_root_sha256_by_realization") !=
            manifest.get("release_realization_root_sha256") or
            streams.get("seq_fit_root_sha256") != manifest.get("seq_fit_root_sha256") or
            streams.get("seq_fit_hmac_domain") != "SCV1-seq-fit-first-v1" or
            len(set(manifest["release_realization_root_sha256"].values()) |
                {manifest["seq_fit_root_sha256"]}) != 4):
        raise PermissionError("A release root commitment differs from fit seal")
    return amendment, spec, state


def load_selected_stream_root(private_path, public_path, *, realization: int,
                              expected_public_sha256: str) -> bytes:
    """Read only the selected 32-byte root after the effective queue gate."""
    supplied_private = Path(private_path)
    if supplied_private.is_symlink():
        raise ValueError("private release root path cannot be a symlink")
    private = supplied_private.resolve(strict=True)
    public = Path(public_path).resolve(strict=True)
    if (realization not in (0, 1, 2) or
            _sha(public) != expected_public_sha256 or
            not private.is_file() or
            stat.S_IMODE(private.parent.stat().st_mode) & 0o077 or
            stat.S_IMODE(private.stat().st_mode) != 0o600):
        raise ValueError("private release stream custody differs")
    manifest = json.loads(public.read_text())
    if (_sha(private) != manifest.get("private_file_sha256") or
            manifest.get("schema") != "scv1-public-stream-root-hashes-v1" or
            manifest.get("private_file_mode") != "0600" or
            manifest.get("seq_fit_hmac_domain") != "SCV1-seq-fit-first-v1" or
            manifest.get("release_hmac_message") !=
            "compact ASCII JSON [SCV1-uniform-v1,source_seed,role,realization,unit]"):
        raise ValueError("private and public release stream commitments differ")
    values = json.loads(private.read_text())
    if (type(values) is not dict or values.keys() != {
            "schema", "release_realization_roots_hex", "seq_fit_root_hex"} or
            values["schema"] != "scv1-private-stream-roots-v1" or
            set(values["release_realization_roots_hex"]) != {"0", "1", "2"}):
        raise ValueError("private release stream root schema differs")
    decoded = {}
    for name, value in (*values["release_realization_roots_hex"].items(),
                        ("seq", values["seq_fit_root_hex"])):
        if (type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise ValueError("private release root is not 32-byte lowercase hex")
        decoded[name] = bytes.fromhex(value)
    if (len(set(decoded.values())) != 4 or
            {key: hashlib.sha256(decoded[key]).hexdigest()
             for key in ("0", "1", "2")} !=
            manifest["release_realization_root_sha256"] or
            hashlib.sha256(decoded["seq"]).hexdigest() !=
            manifest["seq_fit_root_sha256"]):
        raise ValueError("private root bytes differ from public hash commitment")
    return decoded[str(realization)]


def _selected_parent_identity(fit: dict, policy: dict, *, unit_id: str,
                              arm_id: str, seed: int,
                              amendment_sha256: str) -> None:
    """Bind every common selected-policy identity to the completed FIT receipt."""
    shared = ("arm_id", "source_seed", "source_binding_sha256",
              "row_unit_pair_sha256", "bank_sha256_by_recipient",
              "action_sha256_by_recipient")
    if (fit.get("status") != "SELECTED_FIT_POLICY" or
            policy.get("status") != "SELECTED_FIT_POLICY" or
            fit.get("unit_id") != unit_id or
            policy.get("fit_unit_id") != unit_id or
            fit.get("a_amendment_sha256") != amendment_sha256 or
            any(fit.get(key) != policy.get(key) for key in shared) or
            fit.get("arm_id") != arm_id or fit.get("source_seed") != seed):
        raise PermissionError("A release selected policy differs from FIT receipt")
    if fit.get("schema") == "SCV1-private-stochastic-fit-v1":
        selected = fit.get("selected")
        if (not isinstance(selected, dict) or
                policy.get("selected_start") !=
                ("identity_biased" if selected.get("start") == 0 else
                 "uniform_logit" if selected.get("start") == 1 else None) or
                policy.get("selected_update") != selected.get("update")):
            raise PermissionError("A release stochastic checkpoint selection differs")
    elif fit.get("schema") == "SCV1-private-control-fit-v1":
        details = fit.get("fit")
        if (not isinstance(details, dict) or details.get("complete") is not True or
                details.get("success") is not True):
            raise PermissionError("A release control FIT was not complete and selected")
        start, update = _control_selected_provenance(arm_id, details)
        if (policy.get("selected_start") != start or
                policy.get("selected_update") != update):
            raise PermissionError("A release control policy differs from selected checkpoint")
    else:
        raise PermissionError("A release FIT receipt schema differs")


def _control_selected_provenance(arm_id: str, detail: dict) -> tuple[str, int]:
    """Replay R13's frozen selected-control checkpoint identity."""
    if arm_id in ("A_TASK_DET", "A_JOINT_DET"):
        selected = detail.get("selection")
        for start in detail.get("starts", []):
            if start.get("terminal_metrics") == selected:
                update = start.get("chosen_checkpoint_update")
                return start["start"], 300 if update is None else int(update)
        raise PermissionError("A release hard selected gate has no start receipt")
    if arm_id == "A_CENTRAL_PRODUCT":
        rank = detail.get("diagnostic_best")
        if (not isinstance(rank, (list, tuple)) or len(rank) < 3 or
                rank[1] not in (0, 1)):
            raise PermissionError("A release product selection receipt differs")
        return ("identity_biased" if rank[1] == 0 else "uniform_logit",
                int(rank[2]))
    if arm_id in ("A_LOCAL", "A_SEQ12", "A_SEQ21"):
        return "composite", 0
    raise PermissionError("A release control arm is not registered")


def _completed_fit_dependency(spec: dict, parent: dict, *, dependency: str,
                              arm_id: str, seed: int, fit_receipt_path: Path,
                              fit_sha256: str) -> None:
    """Require the selected receipt at the exact completed R03 parent output."""
    if (spec.get("deps") != [dependency] or parent.get("status") != "done" or
            parent.get("spec", {}).get("arm_id") != arm_id or
            parent.get("spec", {}).get("seed") != seed or
            parent.get("spec", {}).get("outputs") != [str(fit_receipt_path)] or
            parent.get("output_sha256", {}).get(str(fit_receipt_path)) !=
            fit_sha256):
        raise PermissionError("A release lacks completed selected fit dependency")


def run(args: argparse.Namespace) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    amendment, spec, state = admit_release_unit(
        unit_id=args.unit_id, lock_path=args.original_lock,
        amendment_path=args.a_amendment,
        amendment_sha256=args.a_amendment_sha256,
        output_dir=args.private_output)
    campaign = Path(args.campaign_root).resolve(strict=True)
    _runtime_origin_guard(campaign)
    unit = amendment["unit_plan"][args.unit_id]
    seed, role, realization = unit["seed"], unit["role"], unit["realization"]
    if (args.seed != seed or args.role != role or args.realization != realization or
            args.arm_id != unit["arm_id"]):
        raise PermissionError("A release CLI differs from sealed unit identity")
    supplied_output = Path(args.private_output)
    output = supplied_output.resolve(strict=False)
    if (supplied_output.is_symlink() or output.exists() or
            not output.parent.is_dir() or output.parent.is_symlink() or
            ".private" not in output.parts or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077):
        raise PermissionError("A release output requires a new owner-only private directory")
    # Every future checkpoint/role path is hash-bound by this immutable unit.
    policy_dir = Path(args.policy_dir).resolve(strict=True)
    policy_receipt_path = policy_dir / "POLICY_RECEIPT.json"
    fit_receipt_path = policy_dir / "FIT_RECEIPT.json"
    policy_npz_path = policy_dir / "policy.npz"
    policy_sha = spec.get("policy_receipt_sha256")
    fit_sha = spec.get("fit_receipt_sha256")
    for path, expected in ((policy_receipt_path, policy_sha),
                           (fit_receipt_path, fit_sha),
                           (policy_npz_path, spec.get("policy_file_sha256"))):
        _spec_source(spec, path, expected)
    dependency = spec.get("fit_unit_id")
    parent = state.get("units", {}).get(dependency, {})
    if dependency is None:
        raise PermissionError("A release lacks completed selected fit dependency")
    _completed_fit_dependency(
        spec, parent, dependency=dependency, arm_id=unit["arm_id"],
        seed=seed, fit_receipt_path=fit_receipt_path, fit_sha256=fit_sha)
    fit_receipt = json.loads(fit_receipt_path.read_text())
    if (fit_receipt.get("status") != "SELECTED_FIT_POLICY" or
            (policy_dir / "INCOMPLETE_RESOURCE.json").exists()):
        raise PermissionError("A release parent fit is not selected and complete")
    policy, policy_receipt = load_selected_policy_package(
        policy_dir, receipt_sha256=policy_sha)
    _selected_parent_identity(
        fit_receipt, policy_receipt, unit_id=dependency,
        arm_id=unit["arm_id"], seed=seed,
        amendment_sha256=args.a_amendment_sha256)
    if (policy_receipt.get("fit_receipt_sha256") != fit_sha or
            policy_receipt.get("policy_file_sha256") !=
            spec["policy_file_sha256"] or
            policy_receipt.get("mechanism_sha256") != _mechanism_hash(policy)):
        raise PermissionError("A selected policy differs from completed FIT source")
    repair, audit, package, bindings = _binding(amendment, spec, args, seed)
    _independent_source_gate(amendment, spec, campaign,
                             bindings["repair_manifest_sha256"],
                             bindings["source_audit_sha256"])
    public_root = campaign / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
    # Keep the caller's path until the custody check has refused a symlink.
    private_root_file = Path(args.private_stream_roots)
    _spec_source(spec, private_root_file,
                 json.loads(public_root.read_text())["private_file_sha256"])
    root = load_selected_stream_root(
        private_root_file, public_root, realization=realization,
        expected_public_sha256=amendment["realizations"]
        ["stream_root_hash_manifest_sha256"])
    source = TStarSourceStore(package, repair, audit, seed)
    if (source.source_binding_sha256 != policy_receipt["source_binding_sha256"] or
            source.source_binding_sha256 !=
            bindings["source_binding_sha256_by_seed"][f"s{seed}"]):
        raise PermissionError("A release source binding differs from FIT policy")
    roles = RoleOnlyStore(repair)
    tokens_file = source.root / f"tokens_{role}.npz"
    ref_key = f"{role}:references_s{seed}"
    reference_file = (roles.package_root /
                      roles.manifest["role_file_relative_path"][ref_key]).resolve(strict=True)
    _spec_source(spec, tokens_file,
                 source.receipt["private_output_sha256"][tokens_file.name])
    _spec_source(spec, reference_file,
                 roles.manifest["role_file_sha256"][ref_key])
    cells = source.read_role_cells(role)
    reference = roles.read(role, f"references_s{seed}")
    assert_same_people(cells, reference)
    row_sha = roles.manifest["roles"][role]["row_unit_pair_sha256"]
    if (policy_receipt["bank_sha256_by_recipient"] != {
            "r1": _bank_hash(policy.bank1), "r2": _bank_hash(policy.bank2)} or
            fit_receipt["bank_sha256_by_recipient"] !=
            policy_receipt["bank_sha256_by_recipient"]):
        raise PermissionError("A released public bank differs from selected FIT")
    if (output.exists() or not output.parent.is_dir() or output.parent.is_symlink() or
            ".private" not in output.parts or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077 or
            time.process_time() - started_cpu >= unit["cpu_seconds"] or
            time.monotonic() - started_wall >= unit["wall_seconds"]):
        raise PermissionError("A release output/resource guard failed before drawing")
    output.mkdir(mode=0o700)
    service = CachedJointRelease(policy, cache_path=output / "cache.sqlite3",
                                 release_id=policy_receipt["release_id"],
                                 source_seed=seed, role=role,
                                 realization=realization,
                                 source_binding_sha256=source.source_binding_sha256,
                                 private_root=root, offline=True)
    source_identity = {
        "tstar_bank_admission_receipt_sha256":
            bindings["tstar_bank_receipt_sha256_by_seed"][f"s{seed}"],
        "tstar_role_tokens_file_sha256": source.receipt["private_output_sha256"]
            [f"tokens_{role}.npz"],
        "reference_file_sha256": roles.manifest["role_file_sha256"]
            [f"{role}:references_s{seed}"],
    }
    result = materialize_role(
        service, row_id=cells["row_id"], unit=cells["unit"],
        score1=reference["U0_1"], score2=reference["U0_2"],
        cell1=cells["cell_1"], cell2=cells["cell_2"],
        output_dir=output, expected_row_unit_sha256=row_sha,
        source_identity=source_identity)
    if (time.process_time() - started_cpu >= unit["cpu_seconds"] or
            time.monotonic() - started_wall >= unit["wall_seconds"]):
        marker = output / "INCOMPLETE_RESOURCE.json"
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema": "SCV1-private-realization-v1",
                       "status": "POST_RELEASE_RESOURCE_OVERRUN"}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        raise TimeoutError("A release unit exceeded whole-unit resource ceiling")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("unit_id", "arm_id", "role", "original_lock", "a_amendment",
                 "a_amendment_sha256", "private_output", "repair_manifest",
                 "source_audit", "portfolio_package", "campaign_root", "policy_dir",
                 "private_stream_roots"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--realization", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

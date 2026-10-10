"""Fail-closed A-stage gate and role-only FIT capability construction.

This module provides the admission and data boundary for a later queue worker.
Importing it reads no role file. R00 must fill and register an effective A-stage
amendment before a real call can pass ``admit_fit_unit``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import RoleOnlyStore
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env

if __package__:
    from .source_bridge import bind_task_inputs, attach_protected
else:
    from source_bridge import bind_task_inputs, attach_protected


SCHEMA = "pcrl-latent-sampled-A-stage-amendment-v1"
REQUIRED = frozenset({"schema", "status", "original_lock_sha256", "source_sha256",
                      "role_bindings", "bank_action_sha256_by_seed",
                      "optimizer_recipe", "control_recipe", "utility_caps",
                      "attack_plan", "realizations", "interval_family",
                      "cpu_budget", "implementation_source_sha256", "unit_plan"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MANDATORY_IMPLEMENTATION = frozenset({
    "sampled_channel_v1/impl/__init__.py",
    "sampled_channel_v1/impl/admitted_fit.py",
    "sampled_channel_v1/impl/run_admitted.py",
    "sampled_channel_v1/impl/bank.py",
    "sampled_channel_v1/impl/channel.py",
    "sampled_channel_v1/impl/fit.py",
    "sampled_channel_v1/impl/source_bridge.py",
    "sampled_channel_v1/impl/policy_package.py",
    "sampled_channel_v1/impl/release.py",
    "r01/role_store.py", "r01/tstar_source.py", "r03/compute_queue.py",
})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def admit_fit_unit(*, unit_id: str, lock_path, amendment_path,
                   amendment_sha256: str, output_dir) -> tuple[dict, dict]:
    """Refuse all role access until R03 and latest R00 A amendment agree."""
    active = assert_active_from_env(unit_id=unit_id, stage="a")
    lock = Path(lock_path).resolve(strict=True)
    amendment_file = Path(amendment_path).resolve(strict=True)
    supplied_output = Path(output_dir)
    output = supplied_output.resolve(strict=False)
    if (supplied_output.is_symlink() or supplied_output.parent.is_symlink() or
            ".private" not in output.parts or output.exists() or
            not output.parent.is_dir() or output.parent.is_symlink() or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077):
        raise PermissionError("A FIT output requires a new owner-only private directory")
    if (type(amendment_sha256) is not str or
            HEX64.fullmatch(amendment_sha256) is None or
            _sha(amendment_file) != amendment_sha256 or
            _sha(lock) != active["seal_sha256"]):
        raise PermissionError("original seal or A amendment byte hash differs")
    original = json.loads(lock.read_text())
    amendment = json.loads(amendment_file.read_text())
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            amendment.keys() != REQUIRED or amendment.get("schema") != SCHEMA or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != active["seal_sha256"]):
        raise PermissionError("A-stage amendment is not effective")
    original_pins = original.get("source_files_sha256", {})
    binding = amendment.get("role_bindings", {})
    if (binding.get("repair_manifest_sha256") !=
            original_pins.get("source_role_manifest") or
            binding.get("source_audit_sha256") !=
            original_pins.get("r01/SOURCE_AUDIT.json") or
            amendment.get("implementation_source_sha256", {}).get("r03/compute_queue.py")
            != original_pins.get("r03/compute_queue.py")):
        raise PermissionError("A source/queue amendment differs from original lock pins")
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((root / "state.json").read_text())
    candidates = [item for item in state.get("amendments", [])
                  if "a" in item.get("stages", [])]
    spec = state.get("units", {}).get(unit_id, {}).get("spec", {})
    if (not candidates or candidates[-1].get("sha256") != amendment_sha256 or
            Path(candidates[-1].get("path", "")).resolve() != amendment_file or
            spec.get("stage") != "a" or
            spec.get("amendment_sha256") != amendment_sha256 or
            spec.get("outputs") != [str(output / "FIT_RECEIPT.json")] or
            output.exists()):
        raise PermissionError("A unit/amendment/output differs from latest queue registration")
    plan = amendment.get("unit_plan", {}).get(unit_id)
    if (type(plan) is not dict or type(plan.get("output_id")) is not str or
            not plan["output_id"] or spec.get("output_id") != plan["output_id"] or
            plan.get("seed") not in (0, 1) or
            plan.get("objective_kind") not in ("task_rate", "protected_bayes") or
            plan.get("arm_id") not in ("A_TASK_STOCH", "A_JOINT_STOCH") or
            (plan["arm_id"] == "A_TASK_STOCH") !=
            (plan["objective_kind"] == "task_rate") or
            spec.get("seed") != plan["seed"] or
            spec.get("arm_id") != plan["arm_id"]):
        raise PermissionError("A unit is absent from frozen arm/seed plan")
    listed = {Path(row["path"]).resolve(): row["sha256"]
              for row in spec.get("sources", [])}
    implementation = amendment.get("implementation_source_sha256", {})
    if (type(implementation) is not dict or
            not MANDATORY_IMPLEMENTATION <= set(implementation)):
        raise PermissionError("A mandatory implementation source pins are absent")
    implementations = {}
    for relative, digest in implementation.items():
        if (type(relative) is not str or Path(relative).is_absolute() or
                ".." in Path(relative).parts):
            raise PermissionError("A implementation source path is not campaign-relative")
        path = (lock.parent / relative).resolve(strict=True)
        if not path.is_relative_to(lock.parent.resolve()):
            raise PermissionError("A implementation source escapes campaign package")
        implementations[path] = digest
    required_paths = (lock, amendment_file, *implementations)
    for path in required_paths:
        expected = (active["seal_sha256"] if path == lock else
                    amendment_sha256 if path == amendment_file else
                    implementations.get(path))
        if (listed.get(path) != expected or _sha(path) != expected):
            raise PermissionError("A implementation source differs from queue/amendment")
    manifest_path = (lock.parent /
                     "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json").resolve(strict=True)
    streams = json.loads(manifest_path.read_text())
    realization = amendment.get("realizations")
    if (type(realization) is not dict or realization.keys() != {
            "ids", "stream_root_hash_manifest_sha256",
            "release_root_sha256_by_realization", "seq_fit_root_sha256",
            "seq_fit_hmac_domain"} or realization["ids"] != [0, 1, 2] or
            streams.get("schema") != "scv1-public-stream-root-hashes-v1" or
            set(streams.get("release_realization_root_sha256", {})) != {"0", "1", "2"} or
            len(set(streams["release_realization_root_sha256"].values()) |
                {streams.get("seq_fit_root_sha256")}) != 4 or
            realization["stream_root_hash_manifest_sha256"] != _sha(manifest_path) or
            realization["release_root_sha256_by_realization"] !=
                streams["release_realization_root_sha256"] or
            realization["seq_fit_root_sha256"] != streams["seq_fit_root_sha256"] or
            realization["seq_fit_hmac_domain"] != "SCV1-seq-fit-first-v1" or
            streams.get("seq_fit_hmac_domain") != "SCV1-seq-fit-first-v1" or
            listed.get(manifest_path) != _sha(manifest_path)):
        raise PermissionError("A three-draw and SEQ FIT stream commitment is absent")
    return amendment, spec


def load_fit_capability(source, roles: RoleOnlyStore, *, seed: int,
                        objective_kind: str, expected):
    """Task-only path never opens SEX; protected path opens it last."""
    if seed not in (0, 1) or objective_kind not in ("task_rate", "protected_bayes"):
        raise ValueError("FIT seed or objective is outside A-stage plan")
    bank1, bank2 = source.read_bank(1), source.read_bank(2)
    cells = source.read_role_cells("OSF_DEFENSE_FIT")
    reference = roles.read("OSF_DEFENSE_FIT", f"references_s{seed}")
    income = roles.read("OSF_DEFENSE_FIT", "income", "task_fit")
    occupation = roles.read("OSF_DEFENSE_FIT", "occupation", "task_fit")
    task, policy, caps, binding = bind_task_inputs(
        bank1, bank2, cells, reference, income, occupation,
        source_binding_sha256=source.source_binding_sha256, expected=expected)
    if objective_kind == "task_rate":
        return task, policy, caps, binding
    sex = roles.read("OSF_DEFENSE_FIT", "sex", "defense_fit")
    protected = attach_protected(task, sex,
                                 expected_row_unit_sha256=expected.row_unit_pair_sha256)
    return protected, policy, caps, binding

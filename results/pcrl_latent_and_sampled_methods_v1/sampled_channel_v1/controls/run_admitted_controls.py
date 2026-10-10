"""R03-only real FIT worker for the six matched sampled-channel controls.

Admission, source hashing and process-slot verification precede all role reads.
The task-only deterministic branch never requests a SEX role file. A selected
policy is write-once private; the sequential FIT first-code cache stays private.
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

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import RoleOnlyStore
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import TStarSourceStore
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from ..impl.admitted_fit import (MANDATORY_IMPLEMENTATION, REQUIRED,
                                 load_fit_capability, _sha)
from ..impl.run_admitted import (_binding, _independent_source_gate, _once_json)
from ..impl.source_bridge import ExpectedIdentity, sha_array
from .control_dispatch import ARMS, fit_control_arm
from .control_package import save_control_policy_package


SCHEMA = "SCV1-private-control-fit-v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TASK_ONLY = frozenset(("A_TASK_DET",))
CONTROL_FILES = frozenset({
    "sampled_channel_v1/controls/control_dispatch.py",
    "sampled_channel_v1/controls/run_admitted_controls.py",
    "sampled_channel_v1/controls/control_package.py",
    "sampled_channel_v1/controls/deterministic_policy.py",
    "sampled_channel_v1/controls/deterministic_trainer.py",
    "sampled_channel_v1/controls/factor_controls.py",
    "sampled_channel_v1/controls/fit_product.py",
    "sampled_channel_v1/controls/fit_local.py",
    "sampled_channel_v1/controls/seq_train.py",
})
IMPLEMENTATION = MANDATORY_IMPLEMENTATION | CONTROL_FILES
MODULE_ORIGINS = {
    f"results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.{name}":
        f"sampled_channel_v1/controls/{name}.py"
    for name in ("run_admitted_controls", "control_dispatch", "control_package",
                 "deterministic_policy", "deterministic_trainer", "factor_controls",
                 "fit_product", "fit_local", "seq_train")
}


def _hex(value):
    return type(value) is str and HEX64.fullmatch(value) is not None


def _private_output(path: Path) -> Path:
    path = path.resolve(strict=False)
    if (".private" not in path.parts or path.exists() or
            not path.parent.is_dir() or path.parent.is_symlink() or
            stat.S_IMODE(path.parent.stat().st_mode) & 0o077):
        raise PermissionError("control FIT output must be fresh in owner-only .private custody")
    path.mkdir(mode=0o700)
    return path


def _stream_commitment(amendment: dict, spec: dict, campaign_root: Path) -> None:
    manifest = campaign_root / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
    root_hash = amendment.get("realizations", {})
    listed = {Path(row["path"]).resolve(): row["sha256"]
              for row in spec.get("sources", [])}
    stream = json.loads(manifest.read_text())
    if (manifest.resolve() not in listed or listed[manifest.resolve()] != _sha(manifest) or
            stream.get("schema") != "scv1-public-stream-root-hashes-v1" or
            root_hash.get("stream_root_hash_manifest_sha256") != _sha(manifest) or
            root_hash.get("seq_fit_root_sha256") != stream.get("seq_fit_root_sha256") or
            root_hash.get("release_root_sha256_by_realization") !=
            stream.get("release_realization_root_sha256") or
            root_hash.get("seq_fit_hmac_domain") != "SCV1-seq-fit-first-v1" or
            root_hash.get("ids") != [0, 1, 2]):
        raise PermissionError("control FIT stream commitment differs")


def admit_control_unit(*, unit_id: str, lock_path, amendment_path,
                       amendment_sha256: str, output_dir) -> tuple[dict, dict]:
    """Fail closed before a role label, bank, or private stream is opened."""
    active = assert_active_from_env(unit_id=unit_id, stage="a")
    lock = Path(lock_path).resolve(strict=True)
    amendment_file = Path(amendment_path).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    if (not _hex(amendment_sha256) or _sha(amendment_file) != amendment_sha256 or
            _sha(lock) != active["seal_sha256"]):
        raise PermissionError("A control original seal or amendment SHA differs")
    original, amendment = json.loads(lock.read_text()), json.loads(amendment_file.read_text())
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            amendment.keys() != REQUIRED or
            amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != active["seal_sha256"]):
        raise PermissionError("A control amendment is not effective")
    original_pins = original.get("source_files_sha256", {})
    binding = amendment.get("role_bindings", {})
    if (binding.get("repair_manifest_sha256") !=
            original_pins.get("source_role_manifest") or
            binding.get("source_audit_sha256") !=
            original_pins.get("r01/SOURCE_AUDIT.json") or
            amendment.get("implementation_source_sha256", {}).get("r03/compute_queue.py")
            != original_pins.get("r03/compute_queue.py")):
        raise PermissionError("A control inherited custody differs")
    queue_root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue_root / "state.json").read_text())
    latest = [item for item in state.get("amendments", []) if "a" in item.get("stages", [])]
    spec = state.get("units", {}).get(unit_id, {}).get("spec", {})
    plan = amendment.get("unit_plan", {}).get(unit_id)
    if (".private" not in output.parts or not latest or
            latest[-1].get("sha256") != amendment_sha256 or
            Path(latest[-1].get("path", "")).resolve() != amendment_file or
            spec.get("stage") != "a" or spec.get("amendment_sha256") != amendment_sha256 or
            spec.get("outputs") != [str(output / "FIT_RECEIPT.json")] or
            output.exists() or
            type(plan) is not dict or plan.get("arm_id") not in ARMS or
            plan.get("seed") not in (0, 1) or
            plan.get("objective_kind") !=
            ("task_rate" if plan["arm_id"] in TASK_ONLY else "protected_bayes") or
            spec.get("seed") != plan["seed"] or spec.get("arm_id") != plan["arm_id"] or
            spec.get("output_id") != plan.get("output_id") or
            type(plan.get("recipe_seed")) is not int or plan["recipe_seed"] < 0 or
            type(plan.get("cpu_seconds")) not in (int, float) or
            type(plan.get("wall_seconds")) not in (int, float) or
            plan["cpu_seconds"] <= 0 or plan["wall_seconds"] <= 0 or
            spec.get("cpu_reserve_s", 0) < plan["cpu_seconds"] or
            spec.get("max_wall_s", 0) < plan["wall_seconds"]):
        raise PermissionError("A control unit and global queue spec differ")
    implementation = amendment.get("implementation_source_sha256", {})
    if (type(implementation) is not dict or not IMPLEMENTATION <= set(implementation)):
        raise PermissionError("A control implementation pins are incomplete")
    listed = {Path(row["path"]).resolve(): row["sha256"]
              for row in spec.get("sources", [])}
    for relative, digest in implementation.items():
        if (type(relative) is not str or Path(relative).is_absolute() or
                ".." in Path(relative).parts or not _hex(digest)):
            raise PermissionError("A control source pin is malformed")
        path = (lock.parent / relative).resolve(strict=True)
        if (not path.is_relative_to(lock.parent.resolve()) or
                listed.get(path) != digest or _sha(path) != digest):
            raise PermissionError("A control imported source differs from queue pin")
    if (listed.get(lock) != active["seal_sha256"] or
            listed.get(amendment_file) != amendment_sha256):
        raise PermissionError("A control lock files differ from queue pin")
    _stream_commitment(amendment, spec, lock.parent)
    return amendment, spec


def _module_origin_guard(campaign_root: Path) -> None:
    for module_name, relative in MODULE_ORIGINS.items():
        module = sys.modules.get(module_name)
        if module is None and module_name.endswith(".run_admitted_controls"):
            module = sys.modules.get("__main__")
        if (module is None or
                Path(getattr(module, "__file__", "")).resolve() !=
                (campaign_root / relative).resolve(strict=True)):
            raise PermissionError("control Python import origin differs from hashed source")


def _seq_root(campaign_root: Path, amendment: dict) -> bytes:
    """Open the fourth committed secret only for an admitted SEQ FIT unit."""
    path = campaign_root / "sampled_channel_v1/design/.private/stream_roots_v1.json"
    manifest = json.loads((campaign_root /
        "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json").read_text())
    if (path.is_symlink() or not path.is_file() or
            stat.S_IMODE(path.stat().st_mode) & 0o077 or
            _sha(path) != manifest.get("private_file_sha256")):
        raise PermissionError("private fourth stream custody differs")
    stream = json.loads(path.read_text())
    if (stream.get("schema") != "scv1-private-stream-roots-v1" or
            type(stream.get("seq_fit_root_hex")) is not str):
        raise PermissionError("private fourth stream schema differs")
    root = bytes.fromhex(stream["seq_fit_root_hex"])
    if (len(root) != 32 or hashlib.sha256(root).hexdigest() !=
            amendment["realizations"]["seq_fit_root_sha256"]):
        raise PermissionError("private fourth stream hash differs")
    return root


def _selected_provenance(arm_id: str, receipt: dict) -> tuple[str, int]:
    if arm_id in ("A_TASK_DET", "A_JOINT_DET"):
        selected = receipt.get("selection")
        for start in receipt["starts"]:
            if start.get("terminal_metrics") == selected:
                update = start.get("chosen_checkpoint_update")
                return start["start"], 300 if update is None else int(update)
        raise ValueError("hard selected gate has no start receipt")
    if arm_id == "A_CENTRAL_PRODUCT":
        rank = receipt["diagnostic_best"]
        return ("identity_biased" if rank[1] == 0 else "uniform_logit", int(rank[2]))
    return "composite", 0


def run(args: argparse.Namespace) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    amendment, spec = admit_control_unit(
        unit_id=args.unit_id, lock_path=args.original_lock,
        amendment_path=args.a_amendment,
        amendment_sha256=args.a_amendment_sha256,
        output_dir=args.private_output)
    root = Path(args.campaign_root).resolve(strict=True)
    if root != Path(args.original_lock).resolve(strict=True).parent:
        raise PermissionError("control campaign root differs from effective lock")
    _module_origin_guard(root)
    unit = amendment["unit_plan"][args.unit_id]
    arm, seed = unit["arm_id"], unit["seed"]
    if (args.arm_id != arm or args.seed != seed or spec.get("mode") != "real_data" or
            amendment.get("utility_caps") != {
                "nll_excess": .010, "brier_excess": .005,
                "reference_rule": "min_U0_Ucal_on_FIT"}):
        raise PermissionError("control arm/resource/utility identity differs")
    repair, audit, package, bindings = _binding(amendment, spec, args, seed)
    _independent_source_gate(amendment, spec, root,
                             bindings["repair_manifest_sha256"],
                             bindings["source_audit_sha256"])
    source = TStarSourceStore(package, repair, audit, seed)
    if source.source_binding_sha256 != bindings["source_binding_sha256_by_seed"][f"s{seed}"]:
        raise PermissionError("control source binding differs")
    identity = amendment["bank_action_sha256_by_seed"][f"s{seed}"]
    if identity.keys() != {"bank_r1", "bank_r2", "action_r1", "action_r2"}:
        raise PermissionError("control bank/action lock differs")
    expected = ExpectedIdentity(
        source_binding_sha256=source.source_binding_sha256,
        row_unit_pair_sha256=bindings["fit_row_unit_pair_sha256"],
        bank_sha256=(identity["bank_r1"], identity["bank_r2"]),
        action_sha256=(identity["action_r1"], identity["action_r2"]))
    roles = RoleOnlyStore(repair)
    capability, base, caps, binding_receipt = load_fit_capability(
        source, roles, seed=seed, objective_kind=unit["objective_kind"],
        expected=expected)
    cells = source.read_role_cells("OSF_DEFENSE_FIT")
    if (sha_array(np.column_stack((cells["row_id"], cells["unit"])).astype(np.int64))
            != expected.row_unit_pair_sha256):
        raise ValueError("control unit IDs differ from bound FIT people")
    elapsed_cpu = time.process_time() - started_cpu
    if (elapsed_cpu >= float(unit["cpu_seconds"]) or
            time.monotonic() - started_wall >= float(unit["wall_seconds"])):
        raise TimeoutError("control source admission exhausted arm ceiling")
    seq_root = _seq_root(root, amendment) if arm in ("A_SEQ12", "A_SEQ21") else None
    output_path = Path(args.private_output).resolve(strict=False)
    if spec.get("outputs") != [str(output_path / "FIT_RECEIPT.json")]:
        raise PermissionError("control output changed after queue admission")
    output = _private_output(output_path)
    remaining_cpu = float(unit["cpu_seconds"]) - (time.process_time() - started_cpu)
    if remaining_cpu <= 0:
        raise TimeoutError("control custody setup exhausted arm ceiling")
    selected, fit = fit_control_arm(
        arm, capability, base, caps, optimizer=amendment["optimizer_recipe"],
        recipe_seed=unit["recipe_seed"],
        cpu_seconds=remaining_cpu,
        unit_ids=tuple(int(x) for x in cells["unit"]), source_seed=seed,
        source_binding_sha256=source.source_binding_sha256,
        seq_private_root=seq_root,
        seq_private_cache_path=(output / "SEQ_FIRST_FIT_DRAW.json"
                                if arm in ("A_SEQ12", "A_SEQ21") else None))
    status = ("SELECTED_FIT_POLICY" if fit["success"] else
              "COMPLETE_NO_FEASIBLE_POLICY" if fit["complete"] else
              "INCOMPLETE_RESOURCE")
    receipt = {
        "schema": SCHEMA, "status": status, "unit_id": args.unit_id,
        "arm_id": arm, "source_seed": seed,
        "source_binding_sha256": source.source_binding_sha256,
        "row_unit_pair_sha256": expected.row_unit_pair_sha256,
        "bank_sha256_by_recipient": {"r1": expected.bank_sha256[0],
                                      "r2": expected.bank_sha256[1]},
        "action_sha256_by_recipient": {"r1": expected.action_sha256[0],
                                        "r2": expected.action_sha256[1]},
        "a_amendment_sha256": args.a_amendment_sha256,
        "binding": binding_receipt, "caps": caps, "fit": fit,
        "elapsed_cpu_seconds": time.process_time() - started_cpu,
        "elapsed_wall_seconds": time.monotonic() - started_wall,
    }
    receipt_sha = _once_json(output / "FIT_RECEIPT.json", receipt)
    if selected is not None:
        start, update = _selected_provenance(arm, fit)
        save_control_policy_package(
            output, selected, arm_id=arm, release_id=f"{arm}-s{seed}-v1",
            source_seed=seed, source_binding_sha256=source.source_binding_sha256,
            row_unit_pair_sha256=expected.row_unit_pair_sha256,
            bank_sha256=expected.bank_sha256,
            action_sha256=expected.action_sha256,
            fit_unit_id=args.unit_id, start=start, update=update,
            fit_receipt_sha256=receipt_sha)
    if (time.process_time() - started_cpu >= float(unit["cpu_seconds"]) or
            time.monotonic() - started_wall >= float(unit["wall_seconds"])):
        _once_json(output / "INCOMPLETE_RESOURCE.json", {
            "schema": SCHEMA, "status": "POST_SAVE_RESOURCE_OVERRUN",
            "fit_receipt_sha256": receipt_sha})
        raise TimeoutError("A control exceeded whole-arm budget after custody")
    if not fit["complete"]:
        raise TimeoutError("A control two-start/stage schedule incomplete")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("unit_id", "arm_id", "original_lock", "a_amendment",
                "a_amendment_sha256", "private_output", "repair_manifest",
                "source_audit", "portfolio_package", "campaign_root"):
        parser.add_argument(f"--{key.replace('_', '-')}", required=True)
    parser.add_argument("--seed", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

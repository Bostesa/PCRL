"""R03-only deterministic IDENTITY and CLASS_ONLY role releases.

These diagnostic arms use the frozen full public bank and admitted fine cells.
They fit nothing, read no task or SEX label, and draw no random code. The
actual code, probability and decision wire is written by R12's common
deterministic reference materializer under owner-only custody.
"""

from __future__ import annotations

import argparse
import json
import math
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
from ..impl.admitted_fit import validate_a_amendment_envelope, _sha
from ..impl.bank import FixedBank, build_actions
from ..impl.reference_materialize import materialize_reference_role
from ..impl.release import _bank_hash
from ..impl.run_admitted import _binding, _independent_source_gate, _once_json
from ..impl.source_bridge import action_map_sha256, sha_array
from .diagnostic_codes import ARMS, assert_group_coherent, select_codes


HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ROLES = frozenset({"ATTACK_FIT_NEW", "INNER_SELECTION"})
SCHEMA = "SCV1-private-reference-release-v1"
REQUIRED_SOURCE = frozenset({
    "r01/role_store.py", "r01/tstar_source.py", "r03/compute_queue.py",
    "sampled_channel_v1/impl/admitted_fit.py",
    "sampled_channel_v1/impl/run_admitted.py",
    "sampled_channel_v1/impl/bank.py",
    "sampled_channel_v1/impl/release.py",
    "sampled_channel_v1/impl/source_bridge.py",
    "sampled_channel_v1/impl/reference_materialize.py",
    "sampled_channel_v1/controls/diagnostic_codes.py",
    "sampled_channel_v1/controls/run_diagnostic_release.py",
})
ORIGINS = {
    f"results.pcrl_latent_and_sampled_methods_v1.{module}": relative
    for module, relative in (
        ("r01.role_store", "r01/role_store.py"),
        ("r01.tstar_source", "r01/tstar_source.py"),
        ("r03.compute_queue", "r03/compute_queue.py"),
        ("sampled_channel_v1.impl.admitted_fit", "sampled_channel_v1/impl/admitted_fit.py"),
        ("sampled_channel_v1.impl.run_admitted", "sampled_channel_v1/impl/run_admitted.py"),
        ("sampled_channel_v1.impl.bank", "sampled_channel_v1/impl/bank.py"),
        ("sampled_channel_v1.impl.release", "sampled_channel_v1/impl/release.py"),
        ("sampled_channel_v1.impl.source_bridge", "sampled_channel_v1/impl/source_bridge.py"),
        ("sampled_channel_v1.impl.reference_materialize",
         "sampled_channel_v1/impl/reference_materialize.py"),
        ("sampled_channel_v1.controls.diagnostic_codes",
         "sampled_channel_v1/controls/diagnostic_codes.py"),
        ("sampled_channel_v1.controls.run_diagnostic_release",
         "sampled_channel_v1/controls/run_diagnostic_release.py"),
    )
}


def _source(spec: dict, path: Path, expected: str) -> None:
    listed = {Path(item["path"]).resolve(): item["sha256"]
              for item in spec.get("sources", [])}
    if (type(expected) is not str or HEX64.fullmatch(expected) is None or
            listed.get(path.resolve(strict=True)) != expected or _sha(path) != expected):
        raise PermissionError("diagnostic source path/SHA differs from R03 unit")


def _private_output(path) -> Path:
    supplied = Path(path).absolute()
    output = supplied.resolve(strict=False)
    if (supplied.is_symlink() or supplied.parent.is_symlink() or
            ".private" not in output.parts or output.exists() or
            not output.parent.is_dir() or
            output.parent.stat().st_uid != os.getuid() or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077):
        raise PermissionError("diagnostic output requires a fresh owner-only .private path")
    return output


def _origin_guard(campaign: Path) -> None:
    for name, relative in ORIGINS.items():
        module = sys.modules.get(name)
        if module is None and name.endswith(".run_diagnostic_release"):
            module = sys.modules.get("__main__")
        if (module is None or Path(getattr(module, "__file__", "")).resolve() !=
                (campaign / relative).resolve(strict=True)):
            raise PermissionError("diagnostic executed source differs from pinned bytes")


def admit_unit(*, unit_id: str, lock_path, amendment_path,
               amendment_sha256: str, output_dir) -> tuple[dict, dict, Path]:
    """Verify the effective seal, queue, source bytes and output before roles."""
    active = assert_active_from_env(unit_id=unit_id, stage="a")
    lock = Path(lock_path).resolve(strict=True)
    amendment_file = Path(amendment_path).resolve(strict=True)
    output = _private_output(output_dir)
    if (_sha(lock) != active["seal_sha256"] or
            _sha(amendment_file) != amendment_sha256):
        raise PermissionError("diagnostic original/amended lock SHA differs")
    original = json.loads(lock.read_text())
    amendment = json.loads(amendment_file.read_text())
    validate_a_amendment_envelope(amendment)
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != active["seal_sha256"]):
        raise PermissionError("diagnostic effective A seal is absent")
    old_pins = original.get("source_files_sha256", {})
    bindings = amendment.get("role_bindings", {})
    if (bindings.get("repair_manifest_sha256") !=
            old_pins.get("source_role_manifest") or
            bindings.get("source_audit_sha256") !=
            old_pins.get("r01/SOURCE_AUDIT.json") or
            amendment.get("implementation_source_sha256", {}).get(
                "r03/compute_queue.py") != old_pins.get("r03/compute_queue.py")):
        raise PermissionError("diagnostic inherited role/source custody differs")
    queue = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue / "state.json").read_text())
    latest = [item for item in state.get("amendments", []) if "a" in item.get("stages", [])]
    spec = state.get("units", {}).get(unit_id, {}).get("spec", {})
    plan = amendment.get("unit_plan", {}).get(unit_id)
    if (not latest or latest[-1].get("sha256") != amendment_sha256 or
            Path(latest[-1].get("path", "")).resolve() != amendment_file or
            type(plan) is not dict or plan.get("kind") != "diagnostic_release" or
            plan.get("arm_id") not in ARMS or plan.get("seed") not in (0, 1) or
            plan.get("role") not in ROLES or plan.get("realization") != 0 or
            spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("amendment_sha256") != amendment_sha256 or
            any(spec.get(key) != plan.get(key) for key in
                ("arm_id", "seed", "role", "realization", "output_id")) or
            spec.get("outputs") != [str(output / "REFERENCE_RELEASE_RECEIPT.json")] or
            type(plan.get("cpu_seconds")) not in (int, float) or
            type(plan.get("wall_seconds")) not in (int, float) or
            not math.isfinite(plan["cpu_seconds"]) or
            not math.isfinite(plan["wall_seconds"]) or
            plan["cpu_seconds"] <= 0 or plan["wall_seconds"] <= 0 or
            spec.get("cpu_reserve_s", 0) < plan["cpu_seconds"] or
            spec.get("max_wall_s", 0) < plan["wall_seconds"]):
        raise PermissionError("diagnostic unit, role, output or resource plan differs")
    implementation = amendment.get("implementation_source_sha256", {})
    if (type(implementation) is not dict or
            not REQUIRED_SOURCE <= set(implementation)):
        raise PermissionError("diagnostic source inventory is incomplete")
    for relative, digest in implementation.items():
        if (type(relative) is not str or Path(relative).is_absolute() or
                ".." in Path(relative).parts):
            raise PermissionError("diagnostic source path escapes campaign")
        source = (lock.parent / relative).resolve(strict=True)
        if not source.is_relative_to(lock.parent):
            raise PermissionError("diagnostic source escapes campaign")
        _source(spec, source, digest)
    _source(spec, lock, active["seal_sha256"])
    _source(spec, amendment_file, amendment_sha256)
    return amendment, spec, output


def _fixed_bank(record: dict, *, expected_bank: str,
                expected_action: str) -> tuple[FixedBank, tuple[tuple[int, ...], ...]]:
    bank = FixedBank.from_source(record["probability"], record["token_id"],
                                 record["token_class"])
    actions = build_actions(bank, input_class=record["input_class"],
                            fine_to_old_token=record["fine_to_hcal_token"])
    if (_bank_hash(bank) != expected_bank or
            action_map_sha256(actions) != expected_action):
        raise PermissionError("diagnostic bank/action differs from sealed A source")
    return bank, actions


def run(args: argparse.Namespace) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    amendment, spec, output = admit_unit(
        unit_id=args.unit_id, lock_path=args.original_lock,
        amendment_path=args.a_amendment,
        amendment_sha256=args.a_amendment_sha256,
        output_dir=args.private_output)
    campaign = Path(args.campaign_root).resolve(strict=True)
    if campaign != Path(args.original_lock).resolve(strict=True).parent:
        raise PermissionError("diagnostic campaign root differs from effective lock")
    _origin_guard(campaign)
    plan = amendment["unit_plan"][args.unit_id]
    seed, role, arm = plan["seed"], plan["role"], plan["arm_id"]
    if (args.seed != seed or args.role != role or args.arm_id != arm):
        raise PermissionError("diagnostic CLI differs from sealed unit")
    repair, audit, package, bindings = _binding(amendment, spec, args, seed)
    _independent_source_gate(amendment, spec, campaign,
                             bindings["repair_manifest_sha256"],
                             bindings["source_audit_sha256"])
    source = TStarSourceStore(package, repair, audit, seed)
    if source.source_binding_sha256 != bindings["source_binding_sha256_by_seed"][f"s{seed}"]:
        raise PermissionError("diagnostic T* source binding differs")
    roles = RoleOnlyStore(repair)
    ref_key = f"{role}:references_s{seed}"
    ref_path = (roles.package_root /
                roles.manifest["role_file_relative_path"][ref_key]).resolve(strict=True)
    token_path = source.root / f"tokens_{role}.npz"
    for path, digest in ((token_path, source.receipt["private_output_sha256"][token_path.name]),
                         (ref_path, roles.manifest["role_file_sha256"][ref_key])):
        _source(spec, path, digest)
    identity = amendment["bank_action_sha256_by_seed"][f"s{seed}"]
    if identity.keys() != {"bank_r1", "bank_r2", "action_r1", "action_r2"}:
        raise PermissionError("diagnostic bank/action lock differs")
    banks = []
    actions = []
    for recipient in (1, 2):
        path = source.root / f"bank_r{recipient}.npz"
        _source(spec, path, source.receipt["private_output_sha256"][path.name])
        record = source.read_bank(recipient)
        bank, action = _fixed_bank(
            record, expected_bank=identity[f"bank_r{recipient}"],
            expected_action=identity[f"action_r{recipient}"])
        banks.append(bank)
        actions.append(action)
    cells = source.read_role_cells(role)
    reference = roles.read(role, f"references_s{seed}")
    assert_same_people(cells, reference)
    codes = [select_codes(bank, action, cells[f"cell_{recipient}"], arm_id=arm)
             for recipient, (bank, action) in enumerate(zip(banks, actions), start=1)]
    assert_group_coherent(cells["unit"], codes[0], codes[1])
    decisions = [bank.decision[code] for bank, code in zip(banks, codes)]
    for recipient, decision in enumerate(decisions, start=1):
        if not np.array_equal(decision, reference[f"d_{recipient}"]):
            raise ValueError("diagnostic release changed the frozen source decision")
    if (time.process_time() - started_cpu >= plan["cpu_seconds"] or
            time.monotonic() - started_wall >= plan["wall_seconds"] or
            _private_output(output) != output):
        raise TimeoutError("diagnostic source admission exhausted unit ceiling")
    source_identity = {
        "bank_receipt_sha256":
            bindings["tstar_bank_receipt_sha256_by_seed"][f"s{seed}"],
        "role_source_file_sha256":
            source.receipt["private_output_sha256"][token_path.name],
        "reference_file_sha256": roles.manifest["role_file_sha256"][ref_key],
        "source_audit_sha256": bindings["source_audit_sha256"],
    }
    receipt = materialize_reference_role(
        row_id=cells["row_id"], unit=cells["unit"],
        prob1=banks[0].probability[codes[0]],
        prob2=banks[1].probability[codes[1]],
        decision1=decisions[0], decision2=decisions[1],
        code1=codes[0], code2=codes[1], arm_id=arm, seed=seed, role=role,
        release_id=f"{arm}-s{seed}-v1",
        source_binding_sha256=source.source_binding_sha256,
        source_identity=source_identity, interface_kind="finite",
        bank_sha256_by_recipient={"r1": sha_array(banks[0].probability),
                                  "r2": sha_array(banks[1].probability)},
        bank1=banks[0].probability, bank2=banks[1].probability,
        output_dir=output,
        expected_row_unit_sha256=roles.manifest["roles"][role]["row_unit_pair_sha256"])
    if (time.process_time() - started_cpu >= plan["cpu_seconds"] or
            time.monotonic() - started_wall >= plan["wall_seconds"]):
        _once_json(output / "INCOMPLETE_RESOURCE.json", {
            "schema": SCHEMA, "status": "POST_RELEASE_RESOURCE_OVERRUN",
            "unit_id": args.unit_id})
        raise TimeoutError("diagnostic release exceeded whole-unit ceiling")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit_id", "arm_id", "role", "original_lock", "a_amendment",
                 "a_amendment_sha256", "private_output", "repair_manifest",
                 "source_audit", "portfolio_package", "campaign_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--seed", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

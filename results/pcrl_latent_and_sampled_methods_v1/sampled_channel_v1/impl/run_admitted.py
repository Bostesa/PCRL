"""R03-only first A stochastic fit, with write-once private checkpoints.

The task-only arm branches before any SEX role read. This worker accepts only
the two sealed central stochastic arms; R13 controls have separate workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import RoleOnlyStore
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import TStarSourceStore

from .admitted_fit import admit_fit_unit, load_fit_capability, _sha
from .fit import FitRecipe, fit_gate
from .policy_package import save_policy_package
from .source_bridge import ExpectedIdentity


SCHEMA = "SCV1-private-stochastic-fit-v1"
ROLE = "OSF_DEFENSE_FIT"
PACKAGE = "results.pcrl_latent_and_sampled_methods_v1"
MODULE_ORIGINS = {
    f"{PACKAGE}.sampled_channel_v1.impl.{name}":
        f"sampled_channel_v1/impl/{name}.py"
    for name in ("admitted_fit", "run_admitted", "bank", "channel", "fit",
                 "source_bridge", "policy_package", "release")
} | {
    f"{PACKAGE}.r01.role_store": "r01/role_store.py",
    f"{PACKAGE}.r01.tstar_source": "r01/tstar_source.py",
    f"{PACKAGE}.r03.compute_queue": "r03/compute_queue.py",
}


def _once_json(path: Path, value: dict) -> str:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _spec_source(spec: dict, path: Path, expected: str) -> None:
    registered = {Path(row["path"]).resolve(): row["sha256"]
                  for row in spec.get("sources", [])}
    if registered.get(path.resolve()) != expected or _sha(path) != expected:
        raise PermissionError("A private source path/SHA differs from R03 unit")


def _module_origin_guard(campaign_root: Path) -> None:
    """The bytes verified by the amendment must be the bytes Python imported."""
    for module_name, relative in MODULE_ORIGINS.items():
        module = sys.modules.get(module_name)
        if module is None and module_name.endswith(".run_admitted"):
            module = sys.modules.get("__main__")
        expected = (campaign_root / relative).resolve(strict=True)
        if module is None or Path(getattr(module, "__file__", "")).resolve() != expected:
            raise PermissionError("A imported source differs from hash-verified package")


def _recipe(amendment: dict, unit: dict, remaining_cpu: float) -> FitRecipe:
    declared = amendment["optimizer_recipe"]
    required = {"updates", "checkpoint_every", "learning_rate", "rho",
                "dual_every", "gradient_norm", "init_scale",
                "identity_logit_bias"}
    if (type(declared) is not dict or declared.keys() != required or
            declared["updates"] != 300 or declared["checkpoint_every"] != 25 or
            declared["dual_every"] != 25 or declared["learning_rate"] != .01 or
            declared["rho"] != 100 or declared["gradient_norm"] != 5 or
            declared["init_scale"] != .01 or
            declared["identity_logit_bias"] != 4 or
            remaining_cpu <= 0 or type(unit.get("recipe_seed")) is not int):
        raise PermissionError("A optimizer recipe differs from prefit design")
    return FitRecipe(cpu_seconds=remaining_cpu, seed=unit["recipe_seed"], **declared)


def _binding(amendment: dict, spec: dict, args, seed: int):
    bindings = amendment["role_bindings"]
    required = {"role_store_id", "repair_manifest_sha256", "source_audit_sha256",
                "fit_row_unit_pair_sha256", "source_binding_sha256_by_seed",
                "tstar_bank_receipt_sha256_by_seed"}
    if type(bindings) is not dict or bindings.keys() != required or \
            bindings["role_store_id"] != "UTILITY_REPAIR_ROLE_EXPORT_V1":
        raise PermissionError("A role/source binding is incomplete")
    repair = Path(args.repair_manifest).resolve(strict=True)
    audit = Path(args.source_audit).resolve(strict=True)
    package = Path(args.portfolio_package).resolve(strict=True)
    receipt = package / f"shared/.private/tstar_score_bank_s{seed}_v1/BANK_ADMISSION_RECEIPT.json"
    for path, digest in ((repair, bindings["repair_manifest_sha256"]),
                         (audit, bindings["source_audit_sha256"]),
                         (receipt, bindings["tstar_bank_receipt_sha256_by_seed"][f"s{seed}"])):
        _spec_source(spec, path, digest)
    if audit.parent.parent != Path(args.campaign_root).resolve(strict=True):
        raise PermissionError("A source audit is outside the campaign package")
    return repair, audit, package, bindings


def _independent_source_gate(amendment: dict, spec: dict, campaign_root: Path,
                             repair_manifest_sha256: str,
                             source_audit_sha256: str) -> None:
    sources = amendment["source_sha256"]
    relative = "r04/A_PREFIT_INDEPENDENT_REPLAY.json"
    if type(sources) is not dict or relative not in sources:
        raise PermissionError("independent A source replay is absent from amendment")
    for name, digest in sources.items():
        if type(name) is not str or Path(name).is_absolute() or ".." in Path(name).parts:
            raise PermissionError("independent A source path is not campaign-relative")
        path = (campaign_root / name).resolve(strict=True)
        if not path.is_relative_to(campaign_root):
            raise PermissionError("independent A source escapes campaign package")
        _spec_source(spec, path, digest)
    report = json.loads((campaign_root / relative).read_text())
    if (report.get("schema") != "pcrl-r04-a-prefit-independent-replay-v1" or
            report.get("status") != "PREFIT_SOURCE_DECODER_IDENTITY_PASS" or
            report.get("role_manifest_sha256") != repair_manifest_sha256 or
            report.get("source_audit_sha256") != source_audit_sha256 or
            set(report.get("seeds", {})) != {"0", "1"}):
        raise PermissionError("R04 independent T* replay does not bind both seeds")
    bindings = amendment["role_bindings"]
    for seed in (0, 1):
        key = f"s{seed}"
        checked = report["seeds"][str(seed)]
        if (checked.get("source_binding_sha256") !=
                bindings["source_binding_sha256_by_seed"].get(key) or
                checked.get("receipt_sha256") !=
                bindings["tstar_bank_receipt_sha256_by_seed"].get(key)):
            raise PermissionError("R04 replayed bank/source differs from A fit binding")


def run(args: argparse.Namespace) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    amendment, spec = admit_fit_unit(unit_id=args.unit_id,
                                     lock_path=args.original_lock,
                                     amendment_path=args.a_amendment,
                                     amendment_sha256=args.a_amendment_sha256,
                                     output_dir=args.private_output)
    _module_origin_guard(Path(args.campaign_root).resolve(strict=True))
    unit = amendment["unit_plan"][args.unit_id]
    seed, objective = unit["seed"], unit["objective_kind"]
    if (args.seed != seed or args.arm_id != unit["arm_id"] or
            spec.get("mode") != "real_data" or
            type(unit.get("cpu_seconds")) not in (int, float) or
            unit["cpu_seconds"] <= 0 or
            type(unit.get("wall_seconds")) not in (int, float) or
            unit["wall_seconds"] <= 0 or
            spec.get("cpu_reserve_s", 0) < unit["cpu_seconds"] or
            spec.get("max_wall_s", 0) < unit["wall_seconds"]):
        raise PermissionError("A fit unit identity/resource ceiling differs")
    repair, audit, package, bindings = _binding(amendment, spec, args, seed)
    _independent_source_gate(amendment, spec,
                             Path(args.campaign_root).resolve(strict=True),
                             bindings["repair_manifest_sha256"],
                             bindings["source_audit_sha256"])
    if (amendment["utility_caps"] != {
            "nll_excess": .010, "brier_excess": .005,
            "reference_rule": "min_U0_Ucal_on_FIT"}):
        raise PermissionError("A FIT utility caps differ from original common criteria")
    source = TStarSourceStore(package, repair, audit, seed)
    if source.source_binding_sha256 != bindings["source_binding_sha256_by_seed"][f"s{seed}"]:
        raise PermissionError("A fine-cell source binding differs")
    identity = amendment["bank_action_sha256_by_seed"][f"s{seed}"]
    if identity.keys() != {"bank_r1", "bank_r2", "action_r1", "action_r2"}:
        raise PermissionError("A canonical bank/action map identity differs")
    expected = ExpectedIdentity(
        source_binding_sha256=source.source_binding_sha256,
        row_unit_pair_sha256=bindings["fit_row_unit_pair_sha256"],
        bank_sha256=(identity["bank_r1"], identity["bank_r2"]),
        action_sha256=(identity["action_r1"], identity["action_r2"]))
    roles = RoleOnlyStore(repair)
    capability, policy, caps, binding_receipt = load_fit_capability(
        source, roles, seed=seed, objective_kind=objective, expected=expected)
    elapsed_cpu = time.process_time() - started_cpu
    elapsed_wall = time.monotonic() - started_wall
    remaining_cpu = float(unit["cpu_seconds"]) - elapsed_cpu
    if remaining_cpu <= 0 or elapsed_wall >= float(unit["wall_seconds"]):
        raise TimeoutError("A arm resource ceiling exhausted during admitted source binding")
    recipe = _recipe(amendment, unit, remaining_cpu)
    output = Path(args.private_output).resolve(strict=False)
    if (Path(args.private_output).is_symlink() or ".private" not in output.parts or
            output.exists() or not output.parent.is_dir() or output.parent.is_symlink() or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077):
        raise PermissionError("A fit output must be a fresh owner-only private directory")
    output.mkdir(mode=0o700)
    checkpoint_hashes = {}

    def persist_checkpoint(start_name, checkpoint):
        name = f"{start_name}_u{checkpoint['update']:03d}.npz"
        target = output / name
        weights = checkpoint["weights"]
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, w1=weights.w1, b1=weights.b1,
                                w2=weights.w2, b2=weights.b2)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint_hashes[name] = _sha(target)

    result = fit_gate(capability, policy, objective_kind=objective,
                      caps=caps, recipe=recipe,
                      checkpoint_sink=persist_checkpoint)
    checkpoint_index = {}
    for start in result["starts"]:
        entries = []
        for checkpoint in start["checkpoints"]:
            name = f"{start['name']}_u{checkpoint['update']:03d}.npz"
            target = output / name
            if checkpoint_hashes.get(name) != _sha(target):
                raise ValueError("persisted A checkpoint differs before FIT receipt")
            entries.append({"update": checkpoint["update"],
                            "file_sha256": checkpoint_hashes[name], "filename": name,
                            "metrics": checkpoint["metrics"],
                            "residual": checkpoint["residual"],
                            "feasible": checkpoint["feasible"],
                            "objective": checkpoint["objective"]})
        checkpoint_index[start["name"]] = {
            "completed": start["completed"],
            "elapsed_cpu_seconds": start["elapsed_cpu_seconds"],
            "checkpoints": entries}
    selected = result["diagnostic_best"] if result["success"] else None
    status = ("SELECTED_FIT_POLICY" if result["success"] else
              "COMPLETE_NO_FEASIBLE_POLICY" if result["complete"] else
              "INCOMPLETE_RESOURCE")
    fit_receipt = {
        "schema": SCHEMA, "status": status, "arm_id": args.arm_id,
        "source_seed": seed, "unit_id": args.unit_id,
        "objective_kind": objective,
        "source_binding_sha256": source.source_binding_sha256,
        "row_unit_pair_sha256": expected.row_unit_pair_sha256,
        "bank_sha256_by_recipient": {"r1": expected.bank_sha256[0],
                                      "r2": expected.bank_sha256[1]},
        "action_sha256_by_recipient": {"r1": expected.action_sha256[0],
                                        "r2": expected.action_sha256[1]},
        "a_amendment_sha256": args.a_amendment_sha256,
        "caps": caps, "binding": binding_receipt,
        "starts": checkpoint_index, "selected": selected,
        "elapsed_cpu_seconds": time.process_time() - started_cpu,
        "elapsed_wall_seconds": time.monotonic() - started_wall,
    }
    fit_sha = _once_json(output / "FIT_RECEIPT.json", fit_receipt)
    if result["success"]:
        save_policy_package(output, result["policy"], arm_id=args.arm_id,
                            release_id=f"{args.arm_id}-s{seed}-v1", source_seed=seed,
                            source_binding_sha256=source.source_binding_sha256,
                            row_unit_pair_sha256=expected.row_unit_pair_sha256,
                            bank_sha256=expected.bank_sha256,
                            action_sha256=expected.action_sha256,
                            fit_unit_id=args.unit_id,
                            start=("identity_biased" if selected["start"] == 0 else
                                   "uniform_logit"), update=selected["update"],
                            fit_receipt_sha256=fit_sha)
    if (time.process_time() - started_cpu >= float(unit["cpu_seconds"]) or
            time.monotonic() - started_wall >= float(unit["wall_seconds"])):
        _once_json(output / "INCOMPLETE_RESOURCE.json", {
            "schema": SCHEMA, "status": "POST_SAVE_RESOURCE_OVERRUN",
            "fit_receipt_sha256": fit_sha})
        raise TimeoutError("A arm total resource ceiling exceeded after checkpoint custody")
    if not result["complete"]:
        raise TimeoutError("A two-start stochastic arm did not complete")
    return fit_receipt


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

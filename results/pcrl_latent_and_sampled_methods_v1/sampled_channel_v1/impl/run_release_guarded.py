"""Sidecar-pinned A release entrypoint over the sealed producer worker.

The producer amendment, its 43 implementation sources, and every FIT receipt
keep their original identity. This wrapper checks the later downstream
sidecar and selected checkpoint, then calls the original worker in this same
process so R03's active PID and the producer's gates remain effective.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import (
    assert_active_from_env,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.run_admitted_controls import (
    _selected_provenance,
)

from . import run_release as sealed_release
from . import downstream_sources, release_specs
from .downstream_sources import (_sha, verify_downstream_sidecar,
                                 verify_predecessor_fit)


NEW_REQUIRED = frozenset({
    "sampled_channel_v1/impl/run_release_guarded.py",
    "sampled_channel_v1/impl/release_specs.py",
    "sampled_channel_v1/impl/downstream_sources.py",
})
ARMS = frozenset({"A_JOINT_STOCH", "A_TASK_STOCH", "A_JOINT_DET",
                  "A_TASK_DET", "A_CENTRAL_PRODUCT", "A_LOCAL",
                  "A_SEQ12", "A_SEQ21"})
MODULE = ("results.pcrl_latent_and_sampled_methods_v1."
          "sampled_channel_v1.impl.run_release_guarded")


def _flag(argv: list[str], name: str) -> str:
    flag = "--" + name.replace("_", "-")
    if argv.count(flag) != 1:
        raise PermissionError(f"guarded release CLI lacks unique {flag}")
    at = argv.index(flag) + 1
    if at >= len(argv):
        raise PermissionError("guarded release CLI flag has no value")
    return argv[at]


def _listed(spec: dict, path: Path) -> str | None:
    found = [row.get("sha256") for row in spec.get("sources", [])
             if type(row) is dict and row.get("path") == str(path)]
    return found[0] if len(found) == 1 else None


def _runtime_origin_guard(campaign: Path) -> None:
    for module, relative in (
            (sealed_release, "sampled_channel_v1/impl/run_release.py"),
            (downstream_sources, "sampled_channel_v1/impl/downstream_sources.py"),
            (release_specs, "sampled_channel_v1/impl/release_specs.py")):
        if Path(module.__file__).resolve(strict=True) != \
                (campaign / relative).resolve(strict=True):
            raise PermissionError("guarded release imported source origin differs")
    current = Path(__file__).resolve(strict=True)
    if current != (campaign / "sampled_channel_v1/impl/run_release_guarded.py").resolve(strict=True):
        raise PermissionError("guarded release entrypoint origin differs")


def _selected_identity(fit: dict, policy: dict, *, fit_id: str,
                       arm_id: str, seed: int, producer_sha256: str) -> None:
    sealed_release._selected_parent_identity(
        fit, policy, unit_id=fit_id, arm_id=arm_id, seed=seed,
        amendment_sha256=producer_sha256)
    if fit.get("schema") == "SCV1-private-stochastic-fit-v1":
        selected = fit.get("selected")
        starts = fit.get("starts")
        if (type(selected) is not dict or type(starts) is not dict or
                set(starts) != {"identity_biased", "uniform_logit"} or
                any(starts[name].get("completed") is not True for name in starts)):
            raise PermissionError("A stochastic release lacks completed starts")
        name = {0: "identity_biased", 1: "uniform_logit"}.get(selected.get("start"))
        if (name is None or not any(
                item.get("update") == selected.get("update") and
                item.get("feasible") is True
                for item in starts[name].get("checkpoints", []))):
            raise PermissionError("A stochastic release lacks feasible selected checkpoint")
    elif fit.get("schema") == "SCV1-private-control-fit-v1":
        start, update = _selected_provenance(arm_id, fit["fit"])
        if (policy.get("selected_start") != start or
                policy.get("selected_update") != update):
            raise PermissionError("A control release policy differs from FIT selection")
    else:
        raise PermissionError("A selected FIT receipt schema differs")


def preflight(args: argparse.Namespace) -> tuple[dict, dict]:
    """Check sidecar and completed producer FIT before any role data read."""
    active = assert_active_from_env(unit_id=args.unit_id, stage="a")
    campaign = Path(args.campaign_root).resolve(strict=True)
    producer_path = Path(args.a_amendment).resolve(strict=True)
    original_path = Path(args.original_lock).resolve(strict=True)
    sidecar_path = Path(args.downstream_manifest).resolve(strict=True)
    if (campaign != producer_path.parent or campaign != original_path.parent or
            _sha(original_path) != active["seal_sha256"]):
        raise PermissionError("A guarded release original producer path differs")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    spec = state.get("units", {}).get(args.unit_id, {}).get("spec", {})
    producer = json.loads(producer_path.read_text())
    unit = producer.get("unit_plan", {}).get(args.unit_id)
    if (type(unit) is not dict or unit.get("kind") != "release" or
            unit.get("arm_id") not in ARMS or
            (args.arm_id, args.seed, args.role, args.realization) !=
            (unit.get("arm_id"), unit.get("seed"), unit.get("role"),
             unit.get("realization")) or
            spec.get("unit_id") != args.unit_id or spec.get("stage") != "a" or
            spec.get("mode") != "real_data" or
            spec.get("arm_id") != args.arm_id or spec.get("seed") != args.seed or
            spec.get("role") != args.role or
            spec.get("realization") != args.realization or
            spec.get("deps") != [unit.get("fit_unit_id")] or
            spec.get("cpu_reserve_s") != 30 or spec.get("max_wall_s") != 30 or
            unit.get("cpu_seconds") != 30 or unit.get("wall_seconds") != 30):
        raise PermissionError("A guarded release differs from original 220-unit plan")
    argv = spec.get("argv", [])
    if (type(argv) is not list or len(argv) < 3 or argv[2] != MODULE or
            _flag(argv, "unit_id") != args.unit_id or
            _flag(argv, "arm_id") != args.arm_id or
            _flag(argv, "seed") != str(args.seed) or
            _flag(argv, "role") != args.role or
            _flag(argv, "realization") != str(args.realization) or
            _flag(argv, "a_amendment_sha256") != args.a_amendment_sha256 or
            _flag(argv, "downstream_manifest_sha256") !=
            args.downstream_manifest_sha256 or
            Path(_flag(argv, "downstream_manifest")).resolve(strict=True) != sidecar_path):
        raise PermissionError("A guarded release argv differs from immutable R03 spec")
    for name in ("original_lock", "a_amendment", "private_output",
                 "repair_manifest", "source_audit", "portfolio_package",
                 "campaign_root", "policy_dir", "private_stream_roots"):
        if Path(_flag(argv, name)).resolve(strict=False) != \
                Path(getattr(args, name)).resolve(strict=False):
            raise PermissionError("A guarded release CLI path differs from R03 spec")
    output = Path(args.private_output).resolve(strict=False)
    if (spec.get("outputs") != [str(output / "RELEASE_RECEIPT.json")] or
            output.exists() or output.is_symlink() or
            ".private" not in output.parts or
            not output.parent.is_dir() or
            stat.S_IMODE(output.parent.stat().st_mode) & 0o077):
        raise PermissionError("A guarded release output custody differs")
    verify_downstream_sidecar(
        sidecar_path=sidecar_path,
        sidecar_sha256=args.downstream_manifest_sha256,
        producer_path=producer_path,
        producer_sha256=args.a_amendment_sha256,
        original_lock_path=original_path,
        original_lock_sha256=active["seal_sha256"],
        campaign_root=campaign, spec=spec,
        required_new_paths=NEW_REQUIRED)
    _runtime_origin_guard(campaign)
    fit_id = unit["fit_unit_id"]
    fit_dir = Path(args.policy_dir).resolve(strict=True)
    fit_path = fit_dir / "FIT_RECEIPT.json"
    policy_path = fit_dir / "POLICY_RECEIPT.json"
    policy_file = fit_dir / "policy.npz"
    parent = state.get("units", {}).get(fit_id, {})
    if (Path(_flag(parent.get("spec", {}).get("argv", []), "private_output"))
            .resolve(strict=True) != fit_dir or
            fit_dir.name != fit_id):
        raise PermissionError("A guarded release FIT output directory differs")
    fit = verify_predecessor_fit(
        parent=parent, fit_receipt_path=fit_path,
        producer_sha256=args.a_amendment_sha256,
        fit_unit_id=fit_id, arm_id=args.arm_id, seed=args.seed)
    for path, field in ((fit_path, "fit_receipt_sha256"),
                        (policy_path, "policy_receipt_sha256"),
                        (policy_file, "policy_file_sha256")):
        digest = _sha(path)
        if spec.get(field) != digest or _listed(spec, path) != digest:
            raise PermissionError("A guarded release selected package SHA differs")
    policy = json.loads(policy_path.read_text())
    _selected_identity(fit, policy, fit_id=fit_id, arm_id=args.arm_id,
                       seed=args.seed,
                       producer_sha256=args.a_amendment_sha256)
    if (policy.get("fit_receipt_sha256") != _sha(fit_path) or
            policy.get("policy_file_sha256") != _sha(policy_file)):
        raise PermissionError("A guarded release policy receipt differs from FIT")
    return spec, fit


def run(args: argparse.Namespace) -> dict:
    preflight(args)
    return sealed_release.run(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit_id", "arm_id", "original_lock", "a_amendment",
                 "a_amendment_sha256", "downstream_manifest",
                 "downstream_manifest_sha256", "private_output",
                 "repair_manifest", "source_audit", "portfolio_package",
                 "campaign_root", "policy_dir", "private_stream_roots", "role"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--realization", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

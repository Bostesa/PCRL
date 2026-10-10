"""Fail-closed, receipt-bound R03 queue specs for the effective A graph.

Only ``fit`` units can be materialized before any A parent receipts exist.
Later kinds are deliberately refused until their completed parent artifacts can
be incorporated into an immutable spec. This module never enqueues or runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .a_graph import compile_graph


PYTHON = Path(sys.executable)
THREAD_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
FIT_BINDINGS = frozenset({
    "role_store_id", "repair_manifest_sha256", "source_audit_sha256",
    "fit_row_unit_pair_sha256", "source_binding_sha256_by_seed",
    "tstar_bank_receipt_sha256_by_seed",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_source(sources: dict[str, str], path: Path, expected: str) -> None:
    path = Path(path).resolve(strict=True)
    if _sha(path) != expected:
        raise ValueError(f"source hash changed: {path.name}")
    old = sources.setdefault(str(path), expected)
    if old != expected:
        raise ValueError("source has conflicting SHA records")


def _owner_private_dir(path: Path) -> None:
    if (not path.is_dir() or path.is_symlink() or
            path.stat().st_uid != os.getuid() or
            stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise PermissionError("A spec/output parent requires owner-only custody")


def _latest_a_amendment(queue_state: Path, amendment: Path, digest: str,
                        seal_path: Path, seal_sha256: str) -> None:
    state = json.loads(queue_state.read_text())
    registered_seal = state.get("seal", {})
    registrations = [entry for entry in state.get("amendments", [])
                     if "a" in entry.get("stages", [])]
    if (not registrations or registrations[-1]["sha256"] != digest or
            Path(registrations[-1]["path"]).resolve(strict=True) != amendment or
            registered_seal.get("sha256") != seal_sha256 or
            Path(registered_seal.get("path", "")).resolve() != seal_path or
            state.get("occupancy") is not None):
        raise PermissionError("effective A amendment is not latest in idle queue")


def fit_spec(*, unit_id: str, amendment_path: Path, amendment_sha256: str,
             seal_path: Path, seal_sha256: str, repair_manifest: Path,
             portfolio_package: Path, queue_state: Path,
             private_parent: Path) -> dict:
    """Render a first or subsequent FIT spec, without opening role arrays."""
    amendment_path = Path(amendment_path).resolve(strict=True)
    seal_path = Path(seal_path).resolve(strict=True)
    campaign = amendment_path.parent
    worktree = campaign.parents[1]
    repair_manifest = Path(repair_manifest).resolve(strict=True)
    portfolio_package = Path(portfolio_package).resolve(strict=True)
    queue_state = Path(queue_state).resolve(strict=True)
    private_parent = Path(private_parent).resolve(strict=True)
    _owner_private_dir(private_parent)
    _latest_a_amendment(queue_state, amendment_path, amendment_sha256,
                        seal_path, seal_sha256)
    graph = compile_graph(amendment_path, amendment_sha256, seal_path,
                          seal_sha256, source_root=campaign,
                          expected_units=220)
    if graph["unit_count"] != 220:
        raise ValueError("A graph incomplete")
    amendment = json.loads(amendment_path.read_text())
    bindings = amendment["role_bindings"]
    if (set(bindings) != FIT_BINDINGS or
            _sha(repair_manifest) != bindings["repair_manifest_sha256"] or
            portfolio_package != (worktree / "results/pcrl_method_portfolio_v1").resolve()):
        raise ValueError("FIT role/source binding differs")
    plan = amendment["unit_plan"].get(unit_id)
    if not isinstance(plan, dict) or plan.get("kind") != "fit" or plan.get("deps", []) != []:
        raise ValueError("unit is not a source FIT")
    seed, arm = plan["seed"], plan["arm_id"]
    if seed not in (0, 1):
        raise ValueError("FIT source seed differs")
    module = ("results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1."
              + ("impl.run_admitted" if arm in {"A_TASK_STOCH", "A_JOINT_STOCH"}
                 else "controls.run_admitted_controls"))
    output = private_parent / unit_id
    if output.exists() or output.is_symlink():
        raise FileExistsError("FIT output already exists")
    source_audit = campaign / "r01/SOURCE_AUDIT.json"
    bank_receipt = (portfolio_package /
                    f"shared/.private/tstar_score_bank_s{seed}_v1/BANK_ADMISSION_RECEIPT.json")
    stream_manifest = campaign / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
    private_stream = campaign / "sampled_channel_v1/design/.private/stream_roots_v1.json"
    sources: dict[str, str] = {}
    _add_source(sources, seal_path, seal_sha256)
    _add_source(sources, amendment_path, amendment_sha256)
    for relative, expected in sorted(amendment["implementation_source_sha256"].items()):
        _add_source(sources, campaign / relative, expected)
    for relative, expected in sorted(amendment["source_sha256"].items()):
        _add_source(sources, campaign / relative, expected)
    _add_source(sources, repair_manifest, bindings["repair_manifest_sha256"])
    _add_source(sources, source_audit, bindings["source_audit_sha256"])
    _add_source(sources, bank_receipt,
                bindings["tstar_bank_receipt_sha256_by_seed"][f"s{seed}"])
    _add_source(sources, stream_manifest,
                amendment["realizations"]["stream_root_hash_manifest_sha256"])
    streams = json.loads(stream_manifest.read_text())
    _add_source(sources, private_stream, streams["private_file_sha256"])
    cli = [str(PYTHON), "-m", module,
           "--unit-id", unit_id, "--arm-id", arm, "--seed", str(seed),
           "--original-lock", str(seal_path), "--a-amendment", str(amendment_path),
           "--a-amendment-sha256", amendment_sha256,
           "--private-output", str(output), "--repair-manifest", str(repair_manifest),
           "--source-audit", str(source_audit),
           "--portfolio-package", str(portfolio_package),
           "--campaign-root", str(campaign)]
    spec = {
        "unit_id": unit_id, "stage": "a", "mode": "real_data",
        "data_access": "ADMITTED_FIT_ROLE_ONLY", "amendment_sha256": amendment_sha256,
        "arm_id": arm, "seed": seed, "output_id": plan["output_id"],
        "argv": cli, "cwd": str(worktree), "env": THREAD_ENV,
        "deps": [], "sources": [{"path": path, "sha256": digest}
                               for path, digest in sorted(sources.items())],
        "outputs": [str(output / "FIT_RECEIPT.json")],
        "cpu_reserve_s": plan["cpu_seconds"],
        "max_wall_s": plan["wall_seconds"],
        "max_rss_gib": 3.0, "projected_output_bytes": 300_000_000,
    }
    return spec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "amendment", "amendment-sha256", "seal",
                 "seal-sha256", "repair-manifest", "portfolio-package",
                 "queue-state", "private-parent", "spec-output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    spec = fit_spec(unit_id=args.unit_id, amendment_path=Path(args.amendment),
                    amendment_sha256=args.amendment_sha256,
                    seal_path=Path(args.seal), seal_sha256=args.seal_sha256,
                    repair_manifest=Path(args.repair_manifest),
                    portfolio_package=Path(args.portfolio_package),
                    queue_state=Path(args.queue_state),
                    private_parent=Path(args.private_parent))
    target = Path(args.spec_output)
    _owner_private_dir(target.parent)
    data = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"unit_id": spec["unit_id"], "source_count": len(spec["sources"]),
                      "source_spec_sha256": hashlib.sha256(data).hexdigest(),
                      "spec_path": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()

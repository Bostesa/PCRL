"""Render role-specific A reference and diagnostic release specs, without role reads.

R03 alone registers and runs returned specs. This module reads sealed JSON
metadata and records declared hashes of private role archives; it never opens
those archives or loads per-person arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r03.a_graph import compile_graph
from results.pcrl_latent_and_sampled_methods_v1.r03.a_specs import (
    PYTHON, THREAD_ENV, _add_source, _owner_private_dir,
)


REFERENCES = ("D0_LOCAL", "D0_SEQ12", "D0_SEQ21", "P_STAR", "T_STAR", "U0", "Ucal")
DIAGNOSTICS = ("A_CLASS_ONLY", "A_IDENTITY")
D0_KEYS = {"D0_LOCAL": "hcal_local_d0", "D0_SEQ12": "hcal_seq12_d0",
           "D0_SEQ21": "lra_seq21"}
ROLE_SUFFIX = {"ATTACK_FIT_NEW": "ATT", "INNER_SELECTION": "INNER"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_source_release_ids(amendment: dict) -> tuple[list[str], list[str]]:
    """Check the complete fixed A source-release inventory before rendering."""
    plan = amendment["unit_plan"]
    references = [f"A-REFERENCE-RELEASE-{name}-S{seed}-{suffix}"
                  for name in REFERENCES for seed in (0, 1)
                  for suffix in ROLE_SUFFIX.values()]
    diagnostics = [f"A-DIAG-RELEASE-{name}-S{seed}-{suffix}"
                   for name in DIAGNOSTICS for seed in (0, 1)
                   for suffix in ROLE_SUFFIX.values()]
    if (len(references) != 28 or len(diagnostics) != 8 or
            {name for name, row in plan.items() if row.get("kind") == "reference_release"}
            != set(references) or
            {name for name, row in plan.items() if row.get("kind") == "diagnostic_release"}
            != set(diagnostics)):
        raise ValueError("sealed A source-release inventory differs")
    return references, diagnostics


def _pin_declared(sources: dict[str, str], path: Path, digest: str, root: Path) -> None:
    """Record a private archive from an audited manifest without opening it."""
    if (type(digest) is not str or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("private source SHA malformed")
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise PermissionError("private archive path escaped admitted root")
    old = sources.setdefault(str(path), digest)
    if old != digest:
        raise ValueError("conflicting declared private archive hashes")


def _bank_sources(*, sources: dict[str, str], reference_id: str, seed: int,
                  role: str, portfolio: Path, cache: Path,
                  source_audit: dict, historical_audit: dict,
                  reference_sha: str) -> None:
    if reference_id in ("U0", "Ucal"):
        return
    if reference_id in ("T_STAR", "A_IDENTITY", "A_CLASS_ONLY"):
        bank_root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
        record = source_audit["tstar_banks"][str(seed)]
        receipt_sha = record["receipt_sha256"]
        role_sha = record["file_sha256"][f"tokens_{role}.npz"]
        bank_sha = {f"bank_r{recipient}.npz": record["file_sha256"][f"bank_r{recipient}.npz"]
                    for recipient in (1, 2)}
    else:
        entry = (historical_audit["pstar"][f"s{seed}"] if reference_id == "P_STAR"
                 else historical_audit["d0"][D0_KEYS[reference_id]][f"s{seed}"])
        bank_root = portfolio / Path(entry["receipt_path"]).parent
        receipt_sha = entry["receipt_sha256"]
        role_record = entry["roles"][role]
        if role_record["source_reference_sha256"] != reference_sha:
            raise ValueError("historical role and repair reference SHA differ")
        role_sha = role_record["sha256"]
        bank_sha = {f"bank_r{recipient}.npz": entry["banks"][f"r{recipient}"]["sha256"]
                    for recipient in (1, 2)}
        for item in entry["source_paths_sha256"].values():
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("historical cache source escapes cache root")
            _pin_declared(sources, cache / relative, item["sha256"], cache)
    bank_root = bank_root.resolve(strict=True)
    _add_source(sources, bank_root / "BANK_ADMISSION_RECEIPT.json", receipt_sha)
    _pin_declared(sources, bank_root / f"tokens_{role}.npz", role_sha, bank_root)
    for name, digest in bank_sha.items():
        _pin_declared(sources, bank_root / name, digest, bank_root)


def source_release_spec(*, unit_id: str, amendment_path: Path,
                        amendment_sha256: str, seal_path: Path,
                        seal_sha256: str, repair_manifest: Path,
                        portfolio_package: Path, historical_cache_root: Path,
                        queue_state: Path, private_parent: Path) -> dict:
    """Return one exact R03 source-release spec; never enqueue or read role arrays."""
    amendment_path = Path(amendment_path).resolve(strict=True)
    seal_path = Path(seal_path).resolve(strict=True)
    repair_manifest = Path(repair_manifest).resolve(strict=True)
    portfolio = Path(portfolio_package).resolve(strict=True)
    cache = Path(historical_cache_root).resolve(strict=True)
    queue_state = Path(queue_state).resolve(strict=True)
    private_parent = Path(private_parent).resolve(strict=True)
    _owner_private_dir(private_parent)
    if ".private" not in private_parent.parts:
        raise PermissionError("source releases require a .private output parent")
    campaign = amendment_path.parent
    worktree = campaign.parents[1]
    compile_graph(amendment_path, amendment_sha256, seal_path, seal_sha256,
                  source_root=campaign, expected_units=220)
    amendment = json.loads(amendment_path.read_text())
    reference_ids, diagnostic_ids = planned_source_release_ids(amendment)
    if unit_id not in set(reference_ids + diagnostic_ids):
        raise ValueError("unit is not a sealed A source release")
    state = json.loads(queue_state.read_text())
    latest = [row for row in state["amendments"] if "a" in row.get("stages", [])]
    if (not latest or latest[-1]["sha256"] != amendment_sha256 or
            Path(latest[-1]["path"]).resolve(strict=True) != amendment_path or
            state.get("seal", {}).get("sha256") != seal_sha256):
        raise PermissionError("source release lacks latest registered A seal")
    plan = amendment["unit_plan"][unit_id]
    kind, seed, role = plan["kind"], plan["seed"], plan["role"]
    suffix = ROLE_SUFFIX[role]
    identity = plan["reference_id"] if kind == "reference_release" else plan["arm_id"]
    expected_id = (f"A-REFERENCE-RELEASE-{identity}-S{seed}-{suffix}"
                   if kind == "reference_release"
                   else f"A-DIAG-RELEASE-{identity}-S{seed}-{suffix}")
    expected_caps = 30 if kind == "reference_release" else 10
    if (unit_id != expected_id or seed not in (0, 1) or
            plan.get("deps", []) != [] or plan.get("output_id") != unit_id or
            plan.get("realization") != 0 or
            plan.get("cpu_seconds") != expected_caps or
            plan.get("wall_seconds") != expected_caps or
            (kind == "reference_release" and plan.get("expected_outputs") !=
             ["reference_release.npz", "REFERENCE_RELEASE_RECEIPT.json"])):
        raise ValueError("source release ID, cap or output contract differs")
    output = private_parent / unit_id
    if output.exists() or output.is_symlink():
        raise FileExistsError("source release output already exists")
    bindings = amendment["role_bindings"]
    if _sha(repair_manifest) != bindings["repair_manifest_sha256"]:
        raise ValueError("repair manifest SHA differs from A seal")
    repair = json.loads(repair_manifest.read_text())
    portfolio_manifest = portfolio / "DATA_ACCESS_MANIFEST.json"
    if _sha(portfolio_manifest) != repair["portfolio_manifest_sha256"]:
        raise ValueError("predecessor portfolio SHA differs from repair manifest")
    reference_key = f"{role}:references_s{seed}"
    reference_relative = Path(repair["role_file_relative_path"][reference_key])
    if reference_relative.is_absolute() or ".." in reference_relative.parts:
        raise PermissionError("repair reference path escapes role package")
    role_root = repair_manifest.parent.parent
    reference_file = role_root / reference_relative
    reference_sha = repair["role_file_sha256"][reference_key]
    source_audit_file = campaign / "r01/SOURCE_AUDIT.json"
    historical_audit_file = campaign / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
    source_audit = json.loads(source_audit_file.read_text())
    historical_audit = json.loads(historical_audit_file.read_text())
    sources: dict[str, str] = {}
    _add_source(sources, seal_path, seal_sha256)
    _add_source(sources, amendment_path, amendment_sha256)
    for relative, digest in sorted(amendment["implementation_source_sha256"].items()):
        _add_source(sources, campaign / relative, digest)
    for relative, digest in sorted(amendment["source_sha256"].items()):
        _add_source(sources, campaign / relative, digest)
    _add_source(sources, Path(__file__), _sha(Path(__file__)))
    _add_source(sources, repair_manifest, bindings["repair_manifest_sha256"])
    _add_source(sources, portfolio_manifest, repair["portfolio_manifest_sha256"])
    _pin_declared(sources, reference_file, reference_sha, role_root)
    # All finite references and diagnostics retain the exact public fixed-map
    # identities. These are already in amendment.source_sha256; insist on them.
    fixed_identity = "T_STAR" if kind == "diagnostic_release" else identity
    if fixed_identity not in ("U0", "Ucal"):
        for extension in ("json", "npz"):
            relative = f"r01/A_FIXED_MAP_{fixed_identity}_S{seed}.{extension}"
            if relative not in amendment["source_sha256"]:
                raise ValueError("fixed public map is absent from sealed source inventory")
    _bank_sources(sources=sources, reference_id=identity, seed=seed, role=role,
                  portfolio=portfolio, cache=cache, source_audit=source_audit,
                  historical_audit=historical_audit, reference_sha=reference_sha)
    common = ["--unit-id", unit_id, "--amendment-sha256", amendment_sha256,
              "--seed", str(seed), "--role", role,
              "--repair-manifest-path", str(repair_manifest),
              "--source-audit-path", str(source_audit_file),
              "--historical-audit-path", str(historical_audit_file),
              "--portfolio-package", str(portfolio),
              "--historical-cache-root", str(cache)]
    if kind == "reference_release":
        module = "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.run_reference"
        argv = [str(PYTHON), "-m", module, *common,
                "--lock-path", str(seal_path),
                "--amendment-path", str(amendment_path),
                "--reference-id", identity, "--output-dir", str(output)]
    else:
        module = "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.run_diagnostic_release"
        argv = [str(PYTHON), "-m", module,
                "--unit-id", unit_id, "--arm-id", identity,
                "--role", role, "--seed", str(seed),
                "--original-lock", str(seal_path),
                "--a-amendment", str(amendment_path),
                "--a-amendment-sha256", amendment_sha256,
                "--private-output", str(output),
                "--repair-manifest", str(repair_manifest),
                "--source-audit", str(source_audit_file),
                "--portfolio-package", str(portfolio),
                "--campaign-root", str(campaign)]
    spec = {
        "unit_id": unit_id, "stage": "a", "mode": "real_data",
        "data_access": "ADMITTED_REFERENCE_ROLE_ONLY",
        "amendment_sha256": amendment_sha256,
        "seed": seed, "role": role, "realization": 0,
        "output_id": plan["output_id"],
        "argv": argv, "cwd": str(worktree), "env": THREAD_ENV,
        "deps": [],
        "sources": [{"path": path, "sha256": digest}
                    for path, digest in sorted(sources.items())],
        "outputs": [str(output / "REFERENCE_RELEASE_RECEIPT.json")],
        "cpu_reserve_s": plan["cpu_seconds"],
        "max_wall_s": plan["wall_seconds"],
        "max_rss_gib": 2.0,
        "projected_output_bytes": 200_000_000,
    }
    if kind == "reference_release":
        spec["reference_id"] = identity
    else:
        spec["arm_id"] = identity
    return spec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "amendment", "amendment-sha256", "seal",
                 "seal-sha256", "repair-manifest", "portfolio-package",
                 "historical-cache-root", "queue-state", "private-parent",
                 "spec-output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    spec = source_release_spec(
        unit_id=args.unit_id, amendment_path=Path(args.amendment),
        amendment_sha256=args.amendment_sha256, seal_path=Path(args.seal),
        seal_sha256=args.seal_sha256, repair_manifest=Path(args.repair_manifest),
        portfolio_package=Path(args.portfolio_package),
        historical_cache_root=Path(args.historical_cache_root),
        queue_state=Path(args.queue_state), private_parent=Path(args.private_parent))
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

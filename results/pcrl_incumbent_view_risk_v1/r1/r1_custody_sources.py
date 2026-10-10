"""Inventory exact deployment sources and opaque campaign-private artifacts.

This command hashes files; it never deserializes an NPZ, model, or prediction.
Private campaign arrays are included for recovery under owner-only custody.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from incumbent_view.r1_deployment_sources import catalog, write_once


ROLES = ("r0", "r1", "r2", "r3", "r4", "r5")
BARRED_PARTS = {"osf_development_assessment", "head_validation", "all_role",
                "all_roles", "old_assessment"}


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_admitted_private_path(relative: Path) -> None:
    lowered = {part.lower() for part in relative.parts}
    if lowered & BARRED_PARTS:
        raise PermissionError("barred role appears in campaign-private custody path")
    if relative.name == "release.npz" and "teacher" in lowered:
        raise PermissionError("historical all-role teacher release is barred")


def build(repo: Path, arms: tuple[str, ...]) -> tuple[list[dict], dict[str, str], dict]:
    base, paths = catalog(repo)
    records = {entry["alias"]: entry for entry in base}
    counts = {role: {"files": 0, "bytes": 0} for role in ROLES}

    def add(alias: str, path: Path) -> None:
        path = path.absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("custody source is missing or symlinked")
        record = {"alias": alias, "sha256": hash_file(path),
                  "size_bytes": path.stat().st_size}
        if alias in records and (records[alias] != record or paths[alias] != str(path)):
            raise ValueError("custody aliases conflict")
        records[alias], paths[alias] = record, str(path)

    root = repo / "results/pcrl_incumbent_view_risk_v1"
    control_lock = root / "r0/STRONGEST_CONTROL_LOCK.json"
    control = json.loads(control_lock.read_text())
    if (control["schema"] != "IVR1-frozen-strongest-control-v1" or
            control["selected"] not in arms or control["nominee"] not in arms):
        raise ValueError("frozen strongest-control lock differs from custody arms")
    add("locks/strongest_control.json", control_lock)
    for relative in ("incumbent_view/r4_views.py",
                     "results/pcrl_method_portfolio_v1/shared/attack_eval.py",
                     "results/pcrl_incumbent_view_risk_v1/__init__.py"):
        add("code/" + relative, repo / relative)
    for path in sorted((repo / "incumbent_view").glob("r[45]*.py")):
        add("code/" + str(path.relative_to(repo)), path)
    for path in sorted((root / "r1").glob("*.py")):
        add("code/" + str(path.relative_to(repo)), path)
    for path in sorted((root / "r5").glob("*.py")):
        add("code/" + str(path.relative_to(repo)), path)

    # R0's completed unit receipts/logs are immutable. Live queue state,
    # occupancy and slot locks are deliberately excluded.
    queue_units = root / "r0/.private/compute/units"
    for path in sorted(queue_units.rglob("*")):
        if path.is_symlink():
            raise PermissionError("queue unit custody source is symlinked")
        if not path.is_file():
            continue
        relative = path.relative_to(root / "r0/.private")
        assert_admitted_private_path(relative)
        alias = "campaign/r0/" + relative.as_posix()
        add(alias, path)
        counts["r0"]["files"] += 1
        counts["r0"]["bytes"] += records[alias]["size_bytes"]

    # Copy the admitted role-scoped trees opaquely, including R3 label-derived
    # vectors and R4 control probes. Exclude the destination custody tree itself.
    for role in ROLES:
        if role == "r0":
            continue
        private = root / role / ".private"
        for path in sorted(private.rglob("*")):
            relative = path.relative_to(private)
            if role == "r1" and relative.parts[0] == "custody":
                continue
            if path.is_symlink():
                raise PermissionError("campaign-private symlink is not an admitted custody source")
            if not path.is_file():
                continue
            assert_admitted_private_path(relative)
            alias = f"campaign/{role}/" + relative.as_posix()
            add(alias, path)
            counts[role]["files"] += 1
            counts[role]["bytes"] += records[alias]["size_bytes"]

    # Public receipts bind representative NEW frozen reader checkpoint trees.
    for arm in arms:
        for seed in (0, 1):
            attack_unit = f"R4_FINAL_{arm}_S{seed}_R0"
            public = root / "r4" / f"{attack_unit}_RECEIPT.json"
            public_record = json.loads(public.read_text())
            if (public_record["arm_id"] != arm or
                    public_record["source_seed"] != seed or
                    public_record["realization"] != 0):
                raise ValueError("representative R4 frozen attack identity differs")
            add(f"public/r4/{attack_unit}_RECEIPT.json", public)
            private = root / "r4/.private/attacks" / attack_unit
            for view in ("r1", "r2", "pair"):
                reader_receipt = private / view / "readers/READERS_RECEIPT.json"
                expected = public_record["attack"]["reader_custody"][view]["private_receipt_sha256"]
                if hash_file(reader_receipt) != expected:
                    raise ValueError("representative R4 public/private reader receipt differs")

    entries = [records[alias] for alias in sorted(records)]
    return entries, paths, counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--nominee", required=True)
    parser.add_argument("--control", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve(strict=True)
    if args.nominee != "V_L":
        raise ValueError("custody nomination differs from sealed V_L")
    allowed = {"P0", "O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21"}
    if args.control not in allowed:
        raise ValueError("unregistered strongest control")
    arms = tuple(sorted({"P0", args.nominee, "V_J", args.control}))
    records, paths, counts = build(repo, arms)
    root = repo / "results/pcrl_incumbent_view_risk_v1/r1"
    private = {"schema": "IVR1-r1-deployment-source-map-v1", "version": args.version,
               "selected_arms": list(arms), "opaque_campaign_private": True,
               "entries": [{**entry, "path": paths[entry["alias"]]} for entry in records]}
    private_path = root / ".private" / f"CUSTODY_SOURCES_{args.version}.json"
    private_sha = write_once(private_path, private, 0o600)
    public = {"schema": "IVR1-r1-custody-source-index-v2", "version": args.version,
              "selected_arms": list(arms), "entries": records,
              "private_map_sha256": private_sha, "opaque_campaign_private": True,
              "campaign_private_counts": counts,
              "excluded_sources": ["historical_all_role_teacher_release",
                                   "HEAD_VALIDATION", "old_assessment"],
              "off_device": "PENDING_EXTERNAL_DRIVE_ABSENT"}
    public_path = root / f"CUSTODY_SOURCE_INDEX_{args.version}.json"
    public_sha = write_once(public_path, public, 0o644)
    print(json.dumps({"entry_count": len(records),
                      "bytes": sum(e["size_bytes"] for e in records),
                      "campaign_private_counts": counts,
                      "public_sha256": public_sha,
                      "private_sha256": private_sha}, sort_keys=True))


if __name__ == "__main__":
    main()

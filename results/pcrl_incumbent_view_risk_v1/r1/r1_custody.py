"""Write-once, source-pinned same-device custody for incumbent raw releases.

`plan` only hashes sources. `snapshot` requires a prospective custody amendment
and independent pre-copy audit. `restore` copies only snapshot bytes. None of
these commands opens person-level arrays or starts model inference.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCHEMA = "IVR1-r1-custody-plan-v1"
ALLOWED_ARMS = {"P0", "O_J", "V_J", "V_P", "V_D", "V_L", "V_S12", "V_S21"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        fcntl.fcntl(stream.fileno(), fcntl.F_NOCACHE, 1)
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: dict, mode: int = 0o600) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return hash_file(path)


def check_alias(alias: str) -> None:
    path = Path(alias)
    if (not alias or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts)
            or path.as_posix() != alias or "\\" in alias):
        raise ValueError("unsafe custody alias")


def validate_entry(entry: dict, *, with_source: bool) -> None:
    alias = entry["alias"]
    check_alias(alias)
    if len(entry["sha256"]) != 64 or entry["size_bytes"] < 0:
        raise ValueError("invalid custody hash/size")
    if with_source:
        source = Path(entry["path"])
        if source.is_symlink() or not source.is_file() or source.stat().st_size != entry["size_bytes"]:
            raise ValueError("custody source missing, symlinked or size-changed")
        if hash_file(source) != entry["sha256"]:
            raise ValueError("custody source bytes changed")


def plan(args) -> None:
    repo = Path(args.repo).resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1/r1"
    source_map = Path(args.source_map).resolve(strict=True)
    if hash_file(source_map) != args.source_map_sha256:
        raise ValueError("frozen source map differs")
    catalog = json.loads(source_map.read_text())
    if catalog["schema"] != "IVR1-r1-deployment-source-map-v1":
        raise ValueError("source map schema differs")
    if args.nominee not in ALLOWED_ARMS - {"P0"} or args.control not in ALLOWED_ARMS:
        raise ValueError("nominee or matched control is not a registered fitted arm")
    if (catalog.get("selected_arms") != sorted({"P0", args.nominee, "V_J", args.control}) or
            catalog.get("opaque_campaign_private") is not True):
        raise ValueError("custody source map selection or opaque archive declaration differs")
    nomination = Path(args.nomination_lock).resolve(strict=True)
    nomination_sha = hash_file(nomination)
    if nomination_sha != args.nomination_lock_sha256:
        raise ValueError("nomination lock differs")
    nominated = json.loads(nomination.read_text())
    if nominated["nominee"] != args.nominee or nominated["joint_nominee"] != "V_J":
        raise ValueError("custody arm selection differs from prospective nomination")
    control_item = next((item for item in catalog["entries"]
                         if item["alias"] == "locks/strongest_control.json"), None)
    if control_item is None:
        raise ValueError("frozen strongest-control lock absent from source map")
    control_path = Path(control_item["path"])
    validate_entry(control_item, with_source=True)
    control_record = json.loads(control_path.read_text())
    if (control_record["schema"] != "IVR1-frozen-strongest-control-v1" or
            control_record["selected"] != args.control or
            control_record["nominee"] != args.nominee or
            control_record["nomination_lock"]["sha256"] != nomination_sha):
        raise ValueError("custody control differs from frozen selection rule")
    arms = sorted({args.nominee, "V_J", args.control})
    selected = []
    for item in catalog["entries"]:
        alias = item["alias"]
        if alias.startswith("policies/") and alias.split("/")[1] not in arms:
            continue
        if alias.startswith("code/") or alias.startswith(("private/", "locks/", "public/", "frozen/",
                                                        "banks/", "role/", "reader/", "policies/",
                                                        "campaign/")):
            selected.append(item)
    if not selected or len({item["alias"] for item in selected}) != len(selected):
        raise ValueError("custody source aliases missing or repeated")
    for item in selected:
        validate_entry(item, with_source=True)
    plan_obj = {"schema": SCHEMA, "version": args.version,
                "source_map_sha256": args.source_map_sha256,
                "nomination_lock_sha256": nomination_sha,
                "strongest_control_lock_sha256": control_item["sha256"],
                "nominee": args.nominee, "joint_diagnostic": "V_J", "strongest_control": args.control,
                "contract_realizations": [0, 1, 2], "entries": selected,
                "forbidden_all_role_teacher_release": "excluded; direct hash-bound model/head loader",
                "opaque_campaign_private": True,
                "opaque_copy_only": "No array or model deserialization by custody tooling",
                "off_device_status": "PENDING_EXTERNAL_DRIVE_ABSENT"}
    private_path = root / ".private/custody/plans" / f"PLAN_{args.version}.json"
    private_sha = write_once(private_path, plan_obj)
    public = {k: v for k, v in plan_obj.items() if k != "entries"}
    public["entries"] = [{k: e[k] for k in ("alias", "sha256", "size_bytes")}
                         for e in selected]
    public["private_plan_sha256"] = private_sha
    public_path = root / f"CUSTODY_PLAN_{args.version}.json"
    public_sha = write_once(public_path, public, 0o644)
    print(json.dumps({"private_plan_sha256": private_sha, "public_plan_sha256": public_sha,
                      "entry_count": len(selected), "bytes": sum(e["size_bytes"] for e in selected)},
                     sort_keys=True))


def source_map_paths(source_map: Path) -> dict[str, str]:
    record = json.loads(source_map.read_text())
    return {entry["alias"]: entry["path"] for entry in record["entries"]}


def copy_pinned(source: Path, target: Path, sha256: str, size: int) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size != size:
        raise ValueError("source changed before custody copy")
    if hash_file(source) != sha256:
        raise ValueError("source hash changed before custody copy")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with source.open("rb") as src, os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                                    0o600), "wb") as dst:
        fcntl.fcntl(src.fileno(), fcntl.F_NOCACHE, 1)
        shutil.copyfileobj(src, dst, 1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    if target.stat().st_size != size or hash_file(target) != sha256:
        raise ValueError("custody copy hash differs")


def runtime_manifest(plan_obj: dict, files_root: Path) -> dict:
    aliases = {entry["alias"]: entry for entry in plan_obj["entries"]}
    seeds = {}
    for seed in (0, 1):
        key = str(seed)
        pins = {}
        for alias, entry in aliases.items():
            prefix = f"frozen/s{seed}/"
            if alias.startswith(prefix):
                part = alias[len(prefix):]
                if part.startswith("teacher/"):
                    filename = Path(part).name
                    name = {"model.pt": "model", "head_0.joblib": "head0",
                            "head_1.joblib": "head1", "record.json": "record",
                            "COMPLETE.json": "complete"}.get(filename)
                else:
                    name = part.split("/", 1)[0]
                if name:
                    pins[name] = {"path": str(files_root / alias), "sha256": entry["sha256"]}
        bank_alias = f"banks/s{seed}/P0_PUBLIC_BANK.json"
        fit_alias = f"role/s{seed}/FIT_P0_RELEASE.npz"
        if not {"model", "head0", "head1", "record", "complete", "decoder", "policy"} <= pins.keys():
            raise ValueError("restored teacher/policy source pin incomplete")
        seeds[key] = {"pins": pins,
                      "public_bank": {"path": str(files_root / bank_alias),
                                      "sha256": aliases[bank_alias]["sha256"]},
                      "roles": {"OSF_DEFENSE_FIT": {"path": str(files_root / fit_alias),
                                                    "sha256": aliases[fit_alias]["sha256"]}}}
    return {"schema": "IVR1-restored-P0-source-manifest-v1", "seeds": seeds,
            "source_map_sha256": plan_obj["source_map_sha256"]}


def snapshot(args) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("uncached F_NOCACHE custody requires the audited macOS runtime")
    plan_path = Path(args.plan).resolve(strict=True)
    source_map = Path(args.source_map).resolve(strict=True)
    for path, expected in ((plan_path, args.plan_sha256), (source_map, args.source_map_sha256),
                           (Path(args.audit_receipt), args.audit_sha256),
                           (Path(args.amendment), args.amendment_sha256),
                           (Path(args.nomination_lock), args.nomination_lock_sha256)):
        if hash_file(path) != expected:
            raise ValueError("prospective custody plan/audit/amendment bytes differ")
    plan_obj = json.loads(plan_path.read_text())
    if (plan_obj["schema"] != SCHEMA or plan_obj["source_map_sha256"] != args.source_map_sha256
            or plan_obj["nomination_lock_sha256"] != args.nomination_lock_sha256):
        raise ValueError("custody plan identity differs")
    source_paths = source_map_paths(source_map)
    repo = Path(args.repo).resolve(strict=True)
    branch = subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo,
                                     text=True).strip()
    if branch != "research/pcrl-incumbent-view-risk-v1":
        raise ValueError("snapshot Git branch differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    code_aliases = [entry for entry in plan_obj["entries"] if entry["alias"].startswith("code/")]
    code_aliases.append({"alias": "code/results/pcrl_incumbent_view_risk_v1/r1/r1_custody.py",
                         "sha256": hash_file(Path(__file__))})
    for entry in code_aliases:
        relative = entry["alias"].removeprefix("code/")
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo,
                                   capture_output=True, check=False, timeout=30)
        if (committed.returncode or
                hashlib.sha256(committed.stdout).hexdigest() != entry["sha256"]):
            raise ValueError("Git bundle tip does not contain exact custody/runtime source")
    target = Path(args.target).absolute()
    if target.exists():
        raise FileExistsError("write-once versioned snapshot exists")
    target.mkdir(parents=True, mode=0o700)
    files_root = target / "files"
    for source, label, sha256 in ((plan_path, "CUSTODY_PLAN.json", args.plan_sha256),
                                  (source_map, "DEPLOYMENT_SOURCE_MAP.json", args.source_map_sha256),
                                  (Path(args.audit_receipt), "PRECOPY_AUDIT.json", args.audit_sha256),
                                  (Path(args.amendment), "CUSTODY_AMENDMENT.json", args.amendment_sha256),
                                  (Path(args.nomination_lock), "NOMINATION_LOCK.json",
                                   args.nomination_lock_sha256)):
        copy_pinned(source, target / "metadata" / label, sha256, source.stat().st_size)
    for entry in plan_obj["entries"]:
        validate_entry(entry, with_source=False)
        copy_pinned(Path(source_paths[entry["alias"]]), files_root / entry["alias"],
                    entry["sha256"], entry["size_bytes"])
    runtime = target / "runtime/P0_SOURCE_PATHS.json"
    runtime_sha = write_once(runtime, runtime_manifest(plan_obj, files_root))
    bundle = target / "research.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), branch], cwd=repo,
                   check=True, timeout=120)
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=repo,
                   check=True, timeout=120, capture_output=True)
    heads = subprocess.check_output(["git", "bundle", "list-heads", str(bundle)], cwd=repo,
                                    text=True, timeout=120)
    if f"{head} refs/heads/{branch}" not in heads.splitlines():
        raise ValueError("Git bundle does not carry exact research branch tip")
    bundle_sha = hash_file(bundle)
    receipt = {"schema": "IVR1-r1-snapshot-receipt-v1", "status": "COPIED_AND_HASHED",
               "version": plan_obj["version"], "plan_sha256": args.plan_sha256,
               "audit_sha256": args.audit_sha256, "amendment_sha256": args.amendment_sha256,
               "entry_count": len(plan_obj["entries"]), "runtime_manifest_sha256": runtime_sha,
               "git_branch": branch, "git_head": head, "bundle_sha256": bundle_sha,
               "hash_method": "SHA-256 F_NOCACHE source and snapshot", "off_device": "PENDING"}
    write_once(target / "SNAPSHOT_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def restore(args) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("uncached F_NOCACHE custody requires the audited macOS runtime")
    snapshot_root = Path(args.snapshot).resolve(strict=True)
    plan_path = snapshot_root / "metadata/CUSTODY_PLAN.json"
    if hash_file(plan_path) != args.plan_sha256:
        raise ValueError("restore plan differs")
    plan_obj = json.loads(plan_path.read_text())
    receipt_path = snapshot_root / "SNAPSHOT_RECEIPT.json"
    if hash_file(receipt_path) != args.snapshot_receipt_sha256:
        raise ValueError("externally pinned snapshot receipt differs")
    snapshot_receipt = json.loads(receipt_path.read_text())
    if (snapshot_receipt["status"] != "COPIED_AND_HASHED" or
            snapshot_receipt["plan_sha256"] != args.plan_sha256 or
            snapshot_receipt["entry_count"] != len(plan_obj["entries"])):
        raise ValueError("snapshot receipt/plan identity differs")
    expected_metadata = {
        "DEPLOYMENT_SOURCE_MAP.json": plan_obj["source_map_sha256"],
        "PRECOPY_AUDIT.json": snapshot_receipt["audit_sha256"],
        "CUSTODY_AMENDMENT.json": snapshot_receipt["amendment_sha256"],
        "NOMINATION_LOCK.json": plan_obj["nomination_lock_sha256"],
    }
    for label, expected in expected_metadata.items():
        if hash_file(snapshot_root / "metadata" / label) != expected:
            raise ValueError("snapshot metadata hash differs from frozen receipt/plan")
    if hash_file(snapshot_root / "research.bundle") != snapshot_receipt["bundle_sha256"]:
        raise ValueError("snapshot Git bundle hash differs from frozen receipt")
    heads = subprocess.check_output(["git", "bundle", "list-heads",
                                     str(snapshot_root / "research.bundle")],
                                    text=True, timeout=120)
    if (f'{snapshot_receipt["git_head"]} refs/heads/{snapshot_receipt["git_branch"]}'
            not in heads.splitlines()):
        raise ValueError("snapshot Git bundle research tip differs from receipt")
    target = Path(args.target).absolute()
    if target.exists():
        raise FileExistsError("write-once versioned restore exists")
    target.mkdir(parents=True, mode=0o700)
    for label in ("CUSTODY_PLAN.json", *expected_metadata):
        source = snapshot_root / "metadata" / label
        copy_pinned(source, target / "metadata" / label, hash_file(source), source.stat().st_size)
    for entry in plan_obj["entries"]:
        validate_entry(entry, with_source=False)
        copy_pinned(snapshot_root / "files" / entry["alias"], target / "files" / entry["alias"],
                    entry["sha256"], entry["size_bytes"])
    original = snapshot_root / "research.bundle"
    copy_pinned(original, target / "research.bundle",
                snapshot_receipt["bundle_sha256"], original.stat().st_size)
    runtime_sha = write_once(target / "runtime/P0_SOURCE_PATHS.json",
                             runtime_manifest(plan_obj, target / "files"))
    receipt = {"schema": "IVR1-r1-restore-receipt-v1", "status": "RESTORED_FROM_SNAPSHOT_ONLY",
               "version": plan_obj["version"], "plan_sha256": args.plan_sha256,
               "entry_count": len(plan_obj["entries"]), "runtime_manifest_sha256": runtime_sha,
               "bundle_sha256": snapshot_receipt["bundle_sha256"],
               "snapshot_receipt_sha256": args.snapshot_receipt_sha256,
               "hash_method": "SHA-256 F_NOCACHE snapshot and restore", "off_device": "PENDING"}
    write_once(target / "RESTORE_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", allow_abbrev=False)
    for name in ("repo", "version", "source-map", "source-map-sha256", "nomination-lock",
                 "nomination-lock-sha256", "nominee", "control"):
        p.add_argument("--" + name, required=True)
    p.set_defaults(func=plan)
    p = commands.add_parser("snapshot", allow_abbrev=False)
    for name in ("repo", "plan", "plan-sha256", "source-map", "source-map-sha256",
                 "audit-receipt", "audit-sha256", "amendment", "amendment-sha256",
                 "nomination-lock", "nomination-lock-sha256", "target"):
        p.add_argument("--" + name, required=True)
    p.set_defaults(func=snapshot)
    p = commands.add_parser("restore", allow_abbrev=False)
    for name in ("snapshot", "snapshot-receipt-sha256", "plan-sha256", "target"):
        p.add_argument("--" + name, required=True)
    p.set_defaults(func=restore)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

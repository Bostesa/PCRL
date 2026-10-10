"""Prepare a role-safe manifest and queued recovery interval replay unit.

Preparation reads hashes and public receipts only. INNER SEX and all private
reader predictions remain unopened until the verification queue runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results/pcrl_bank_anchored_release_v1"
R4, R5 = CAMP / "r4", CAMP / "r5"
UID = "R5_RECOVERY_INTERVALS_K1575_V1"


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("R5 recovery source symlink refused")
    source = path.resolve(strict=True)
    if not source.is_file():
        raise PermissionError("R5 recovery source is not a file")
    return {"path": str(source), "sha256": sha(source)}


def write_once(path: Path, payload: dict, mode: int) -> str:
    data = (json.dumps(payload, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != data:
            raise FileExistsError("R5 recovery artifact already has different bytes")
        return sha(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb") as target:
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    return hashlib.sha256(data).hexdigest()


def prepare() -> dict:
    plan_path = R5 / "ATTACK_REPLAY_PLAN.json"
    closeout_path = R5 / "ATTACK_REPLAY_CLOSEOUT.json"
    plan, closeout = (json.loads(path.read_text()) for path in
                      (plan_path, closeout_path))
    if (plan.get("schema") != "BAR1-r5-effective-attack-replay-plan-v1" or
            closeout.get("schema") != "BAR1-r5-independent-attack-replay-closeout-v1" or
            closeout.get("status") != "PASS" or
            closeout.get("source_plan_sha256") != sha(plan_path) or
            closeout.get("source_units_checked") != 38):
        raise PermissionError("exhaustive R5 attack replay has not passed")
    receipts = {row["source_unit_id"]: row for row in closeout["units"]}
    if set(receipts) != set(plan["source_attack_unit_ids"]):
        raise PermissionError("R5 source unit identity differs")
    sources = []
    pinned = []
    role_manifest_path = None
    role_manifest_sha = None
    for uid in plan["source_attack_unit_ids"]:
        spec_path = R4 / ".private/units" / f"{uid}.json"
        r4_path = R4 / f"{uid}_RECEIPT.json"
        r5_path = R5 / f"R5_REPLAY_{uid}_RECEIPT.json"
        spec = json.loads(spec_path.read_text())
        r4 = json.loads(r4_path.read_text())
        r5 = json.loads(r5_path.read_text())
        link = receipts[uid]
        if (sha(r4_path) != link["r4_receipt_sha256"] or
                sha(r5_path) != link["r5_receipt_sha256"] or
                spec.get("unit_id") != uid or r4.get("unit_id") != uid or
                r5.get("source_unit_id") != uid or r5.get("status") != "PASS" or
                r4.get("unit_spec_sha256") != sha(spec_path) or
                r5.get("r4_receipt_sha256") != sha(r4_path)):
            raise PermissionError("R4/R5 source receipt chain differs")
        manifest = Path(spec["role_manifest_path"])
        if sha(manifest) != spec["role_manifest_sha256"]:
            raise PermissionError("allowed role manifest changed")
        if role_manifest_path is None:
            role_manifest_path, role_manifest_sha = manifest, sha(manifest)
        elif manifest != role_manifest_path or sha(manifest) != role_manifest_sha:
            raise PermissionError("attack units use different role exports")
        private = Path(spec["private_output_path"])
        if ".private" not in private.parts or not private.is_dir():
            raise PermissionError("private frozen attack directory differs")
        pinned.extend(pin(path) for path in (spec_path, r4_path, r5_path))
        for view in ("r1", "r2", "pair"):
            reader = private / view / "readers"
            custody_path = reader / "READERS_RECEIPT.json"
            custody = json.loads(custody_path.read_text())
            if (sha(custody_path) !=
                    r4["attack"]["reader_custody"][view]["private_receipt_sha256"] or
                    sha(reader / "predictions_inner.npz") !=
                    custody["files"]["predictions_inner.npz"] or
                    sha(reader / "view_inner.npz") !=
                    custody["files"]["view_inner.npz"]):
                raise PermissionError("frozen reader or prediction bytes differ")
            pinned.extend(pin(path) for path in
                          (custody_path, reader / "predictions_inner.npz",
                           reader / "view_inner.npz"))
        sources.append({"unit_id": uid, "arm_id": spec["arm_id"],
                        "source_seed": spec["source_seed"],
                        "realization": spec["realization"],
                        "r4_receipt_path": str(r4_path.resolve()),
                        "r4_receipt_sha256": sha(r4_path),
                        "r5_replay_path": str(r5_path.resolve()),
                        "r5_replay_sha256": sha(r5_path),
                        "private_attack_dir": str(private.resolve())})
    if role_manifest_path is None:
        raise PermissionError("no allowed INNER role manifest")
    manifest_data = json.loads(role_manifest_path.read_text())
    sex = role_manifest_path.parent.parent / manifest_data["role_file_relative_path"][
        "INNER_SELECTION:sex"]
    if ("INNER_SELECTION" not in sex.parts or sex.name != "sex.npz" or
            sha(sex) != manifest_data["role_file_sha256"]["INNER_SELECTION:sex"]):
        raise PermissionError("only sealed INNER SEX export may enter recovery inference")
    pinned.extend(pin(path) for path in (plan_path, closeout_path,
                                        role_manifest_path, sex))
    unique = {}
    for item in pinned:
        if item["path"] in unique and unique[item["path"]] != item:
            raise PermissionError("conflicting R5 recovery source bytes")
        unique[item["path"]] = item
    private = R5 / ".private"
    if not private.is_dir() or private.stat().st_mode & 0o077:
        raise PermissionError("R5 recovery manifest parent must be owner-only")
    record = {"schema": "BAR1-r5-recovery-source-manifest-v1",
              "status": "PREFIT_NO_RECOVERY_LABEL_READ", "fixed_family_K": 1575,
              "bootstrap_replicates": 2048, "inner_groups": 2234,
              "source_plan_sha256": sha(plan_path),
              "attack_replay_closeout_sha256": sha(closeout_path),
              "role_manifest_path": str(role_manifest_path.resolve()),
              "role_manifest_sha256": role_manifest_sha,
              "inner_sex_path": str(sex.resolve()),
              "inner_sex_sha256": sha(sex), "sources": sources,
              "source_pins": list(unique.values())}
    manifest_path = private / "R5_RECOVERY_SOURCE_MANIFEST.json"
    digest = write_once(manifest_path, record, 0o600)
    public = {"schema": "BAR1-r5-recovery-prefit-source-index-v1",
              "status": "MANIFEST_READY_NO_RECOVERY_LABEL_READ",
              "manifest_sha256": digest, "source_release_units": len(sources),
              "frozen_views": 3 * len(sources),
              "source_pin_count": len(unique),
              "worker_sha256": sha(ROOT / "bank_anchored/r5_recovery_intervals.py"),
              "builder_sha256": sha(Path(__file__))}
    write_once(R5 / "RECOVERY_PREFIT_SOURCE_INDEX.json", public, 0o644)
    return public


def finalize(amendment_path: Path, python_entry: Path) -> dict:
    source_path = R5 / ".private/R5_RECOVERY_SOURCE_MANIFEST.json"
    index_path = R5 / "RECOVERY_PREFIT_SOURCE_INDEX.json"
    index = json.loads(index_path.read_text())
    if (index.get("schema") != "BAR1-r5-recovery-prefit-source-index-v1" or
            index.get("manifest_sha256") != sha(source_path) or
            index.get("builder_sha256") != sha(Path(__file__)) or
            index.get("worker_sha256") !=
            sha(ROOT / "bank_anchored/r5_recovery_intervals.py")):
        raise PermissionError("prefit R5 recovery source or code changed")
    record = json.loads(source_path.read_text())
    state = json.loads((CAMP / "r0/.private/compute/state.json").read_text())
    amendment = json.loads(amendment_path.read_text())
    amendment_pin = pin(amendment_path)
    latest = [a["sha256"] for a in state["amendments"]
              if "verification" in a["stages"]][-1]
    if (amendment.get("schema") != "pcrl-bank-dated-amendment-v1" or
            "verification" not in amendment.get("affected_stages", ()) or
            latest != amendment_pin["sha256"] or
            not python_entry.is_file() or not os.access(python_entry, os.X_OK)):
        raise PermissionError("recovery queue lacks registered verification amendment")
    for item in record["source_pins"]:
        if pin(Path(item["path"])) != item:
            raise PermissionError("frozen recovery source changed")
    files = [*record["source_pins"], *(pin(path) for path in (
        source_path, index_path, amendment_path, Path(__file__),
        ROOT / "bank_anchored/r5_recovery_intervals.py",
        CAMP / "r0/compute_queue.py", python_entry.resolve(strict=True)))]
    unique = {}
    for item in files:
        if item["path"] in unique and unique[item["path"]] != item:
            raise PermissionError("conflicting final recovery source pin")
        unique[item["path"]] = item
    output = R5 / f"{UID}.json"
    receipt = R5 / f"{UID}_RECEIPT.json"
    spec = {"unit_id": UID, "stage": "verification",
            "amendment_sha256": amendment_pin["sha256"],
            "argv": [str(python_entry), "-m", "bank_anchored.r5_recovery_intervals",
                     "--manifest", str(source_path.resolve()),
                     "--manifest-sha256", sha(source_path),
                     "--output", str(output), "--receipt", str(receipt)],
            "cwd": str(ROOT), "env": {},
            "deps": [f"R5_REPLAY_{row['unit_id']}" for row in record["sources"]],
            "max_wall_s": 1200, "cpu_reserve_s": 1200,
            "max_rss_gib": 2.0, "projected_output_bytes": 2_000_000,
            "outputs": [str(output), str(receipt)], "sources": list(unique.values())}
    digest = write_once(R5 / ".private" / f"{UID}_QUEUE_SPEC.json", spec, 0o600)
    public = {"schema": "BAR1-r5-recovery-queue-index-v1", "unit_id": UID,
              "queue_spec_sha256": digest, "manifest_sha256": sha(source_path),
              "source_pin_count": len(unique),
              "amendment_sha256": amendment_pin["sha256"]}
    write_once(R5 / "RECOVERY_QUEUE_INDEX.json", public, 0o644)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    final = sub.add_parser("finalize")
    final.add_argument("--amendment", type=Path, required=True)
    final.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else finalize(
        args.amendment, args.python)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

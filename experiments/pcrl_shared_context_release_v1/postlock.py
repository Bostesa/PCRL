"""Post-lock host steps, run from a SEPARATE checkout (/opt/pcrl/lockcheck).

The queue runner keeps using /opt/pcrl/work at the launch commit; nothing here
modifies that tree. This module runs with PYTHONPATH=/opt/pcrl/lockcheck (a
checkout of the pushed commit that contains the committed SELECTION_LOCK.json),
so `audit_panel.LOCK_PATH`, `UNLOCK_PATH` and `ORIGINAL_RESTORE_ROOT` resolve
inside the lockcheck tree, while fitted units, inner panels, the pinned index
and the audit source lists are read from /opt/pcrl/work by explicit path.

Commands:

    prepare         copy the three sanitized 2018 anchors + receipt from the work
                    tree into this checkout (label-stripped; SHA-verified through
                    TAC `data._sanitized_record`), so loaders in this checkout work
    restore-outer   refuses unless `audit_panel.verify_outer_gate` passes (lock SHA,
                    remote-verified unlock, commit + lock bytes re-checked on origin);
                    then fetches AR's three ORIGINAL prepared objects (these carry
                    the outer labels) by exact key / version / SHA-256 into
                    results/pcrl_shared_context_release_v1/private/original_2018_restore,
                    verifies size and SHA-256, chmod 600, writes a receipt
    run-outer       gate again, then `audit_panel.score_outer` and `score_outer_j`
                    for anchors 0,1,2 and `assess.assess` (INFERENCE.json,
                    ENDPOINT_TABLE.json, FULL_RESULTS.csv)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Mapping, Sequence

from experiments.pcrl_task_aligned_cuts_v1 import data

from . import audit_panel, laws

STUDY = audit_panel.STUDY
BUCKET = "pcrl-ux-archive-ed9d21fd"
WORK = Path("/opt/pcrl/work")
PRIVATE_WORK = WORK / "results" / STUDY / "private"
UNITS = PRIVATE_WORK / "units"
PLAN = PRIVATE_WORK / "plan"
INDEX = WORK / data.INDEX_RELATIVE
OUT = audit_panel.RESULTS / "private"          # inside the lockcheck checkout
ASSESS_OUT = audit_panel.RESULTS / "assessment"
RESTORE_RECEIPT = audit_panel.RESULTS / "private" / "ORIGINAL_RESTORE.json"
# AR's staged ORIGINAL prepared objects (outer labels). Pins copied from AR's
# private ORIGINAL_PREPARED_S3.json; head-object verified 2026-09-24 (size, SSE,
# version). The SHA-256 equals the index pin for anchor `prepared`.
ORIGINAL_OBJECTS = {
    0: {"key": "pcrl_adaptive_release_v1/inputs/original_prepared_anchor_0.joblib",
        "version_id": "GotQn1gutDW1.JIT5OtUjtNEQwOufY8Y", "bytes": 26542284,
        "sha256": "01d8bf65a6c2576f2e949264f1e09fbe604c8a6059251d1d54bb72e2edf75341"},
    1: {"key": "pcrl_adaptive_release_v1/inputs/original_prepared_anchor_1.joblib",
        "version_id": "FZTabGKmb7BxaQpDrBUNp.SyZcHELLif", "bytes": 26370921,
        "sha256": "cdede00b751f34dc4ed47e4dd6de9b85f3a31eeecf318085e4f464a28b11ad66"},
    2: {"key": "pcrl_adaptive_release_v1/inputs/original_prepared_anchor_2.joblib",
        "version_id": "tbHR5wcc.iWdl.4aAtTfFGY9VHXb3A0q", "bytes": 26366724,
        "sha256": "12ffb406a447e1d7c9029fccd12d179a17240e08c291ae2c40d27c44ba9cf97c"},
}


def _s3_get(key: str, version_id: str, target: Path) -> None:
    subprocess.run(["aws", "--region", "us-east-1", "s3api", "get-object", "--bucket", BUCKET,
                    "--key", key, "--version-id", version_id, str(target)],
                   check=True, capture_output=True)


def prepare(work: Path = WORK) -> dict:
    """Label-stripped sanitized copies into this checkout; verified before use."""
    source = work / data.SANITIZED_RELATIVE
    target = data._sanitized_root()
    if source.resolve() == target.resolve():
        raise ValueError("prepare runs from the separate lockcheck checkout only")
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("SANITIZATION.json", "anchor_0.joblib", "anchor_1.joblib", "anchor_2.joblib"):
        destination = target / name
        if destination.exists():
            if laws.sha256_file(destination) != laws.sha256_file(source / name):
                raise ValueError(f"existing {name} differs from the work tree copy")
            continue
        shutil.copy2(source / name, destination.with_name(name + ".partial"))
        os.replace(destination.with_name(name + ".partial"), destination)
        copied.append(name)
    value = data.index(work / data.INDEX_RELATIVE)
    for anchor in (0, 1, 2):
        if data._sanitized_record(value, anchor) is None:
            raise FileNotFoundError("sanitized receipt missing after copy")
    return {"copied": copied, "verified_anchors": [0, 1, 2], "target": str(target)}


def restore_outer(lock_sha256: str, *, index_path: Path = INDEX, gate: Callable | None = None,
                  fetch: Callable[[str, str, Path], None] = _s3_get,
                  restore_root: Path | None = None) -> dict:
    """Open the outer role's source objects only after the coordinator's gate."""
    (gate or audit_panel.verify_outer_gate)(audit_panel.LOCK_PATH, lock_sha256)
    root = restore_root or audit_panel.ORIGINAL_RESTORE_ROOT
    value = data.index(index_path)
    records = []
    for anchor, pin in ORIGINAL_OBJECTS.items():
        member = data.member_record(value, anchor, "prepared")
        if member["sha256"] != pin["sha256"] or member["archive"]["member_sha256"] != pin["sha256"]:
            raise ValueError(f"anchor {anchor}: S3 pin differs from the pinned index")
        relative = Path(member["archive"]["member_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe archive member path")
        destination = root / relative
        if destination.is_file() and laws.sha256_file(destination) == pin["sha256"]:
            records.append({"anchor": anchor, "action": "already_present_verified", **pin})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        partial = destination.with_name(destination.name + ".partial")
        fetch(pin["key"], pin["version_id"], partial)
        if partial.stat().st_size != pin["bytes"] or laws.sha256_file(partial) != pin["sha256"]:
            partial.unlink()
            raise ValueError(f"anchor {anchor}: restored original differs from its SHA-256 pin")
        os.chmod(partial, 0o600)
        os.replace(partial, destination)
        records.append({"anchor": anchor, "action": "restored_verified", **pin})
    for directory in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        directory.chmod(0o700)
    receipt = {"schema": "pcrl-sc-original-restore-v1", "lock_sha256": lock_sha256,
               "utc": datetime.now(timezone.utc).isoformat(), "restore_root": str(root),
               "records": records, "all_verified": True}
    RESTORE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RESTORE_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.chmod(RESTORE_RECEIPT, 0o600)
    return receipt


def run_outer(lock_sha256: str, *, anchors: Sequence[int] = (0, 1, 2),
              with_j: bool = True, assess_after: bool = True) -> dict:
    """score_outer + score_outer_j per anchor, then assess; every step re-runs the gate."""
    from . import assess

    done = {}
    for anchor in anchors:
        sources = audit_panel.read_sources(PLAN / f"a{anchor}_audit_sources.json")
        outer_dir = OUT / "outer" / f"a{anchor}"
        if not (outer_dir / "COMPLETE.json").is_file():
            audit_panel.score_outer(anchor, sources, INDEX, UNITS / f"a{anchor}_inner_audit",
                                    audit_panel.LOCK_PATH, lock_sha256, outer_dir)
        done[f"a{anchor}"] = str(outer_dir)
        if with_j:
            j_dir = OUT / "outer_j" / f"a{anchor}"
            if not (j_dir / "COMPLETE.json").is_file():
                audit_panel.score_outer_j(anchor, INDEX, UNITS / f"a{anchor}_J_inner",
                                          audit_panel.LOCK_PATH, lock_sha256, j_dir)
            done[f"a{anchor}_J"] = str(j_dir)
    if assess_after:
        assess.assess(audit_panel.LOCK_PATH, lock_sha256,
                      {a: OUT / "outer" / f"a{a}" for a in (0, 1, 2)}, ASSESS_OUT,
                      j_outer_dirs=({a: OUT / "outer_j" / f"a{a}" for a in (0, 1, 2)}
                                    if with_j else None))
        done["assessment"] = str(ASSESS_OUT)
    return done


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    item = sub.add_parser("restore-outer")
    item.add_argument("--lock-sha256", required=True)
    item = sub.add_parser("run-outer")
    item.add_argument("--lock-sha256", required=True)
    item.add_argument("--anchors", type=int, nargs="+", default=[0, 1, 2])
    item.add_argument("--no-j", action="store_true")
    item.add_argument("--no-assess", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare()
    elif args.action == "restore-outer":
        result = restore_outer(args.lock_sha256)
        result = {k: result[k] for k in ("all_verified", "restore_root")}
    else:
        result = run_outer(args.lock_sha256, anchors=args.anchors, with_j=not args.no_j,
                           assess_after=not args.no_assess)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

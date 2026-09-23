"""Locked, single-owner entry points for 2018 task-aligned development units."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from . import data, fit, scheduler, release, inner_panel


STUDY = Path("results/pcrl_task_aligned_cuts_v1")


def source_tree() -> dict:
    directory = Path(__file__).resolve().parent
    members = {}
    for path in sorted(directory.glob("*.py")):
        members[path.name] = data.sha256_file(path)
    digest = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
    return {"source_tree_sha256": digest, "source_files": members}


def _record_and_close(out: Path, command: str, scientific_record: dict) -> None:
    ident = os.environ.get("PCRL_UNIT_ID")
    queue_sha = os.environ.get("PCRL_QUEUE_SHA256")
    if not ident or not queue_sha:
        raise ValueError("Scientific units must run through the registered scheduler")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance = {"schema": 1, "unit_id": ident, "command": command,
                  "git_head_at_fit": head, **source_tree(),
                  "scientific_record": scientific_record,
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
    (out / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n")
    names = [str(path.relative_to(out)) for path in sorted(out.rglob("*")) if path.is_file()
             and path.name != "COMPLETE.json"]
    scheduler.write_completion(out, ident, queue_sha, names)


def _admit(anchor: int, value: dict, out: Path) -> dict:
    source = data.verify_source_files(Path.cwd(), value)
    members = data.verify_local_members(value)
    all_prepared = {k: data.load_prepared(value, k) for k in (0, 1, 2)}
    split = data.global_validation_split(all_prepared)
    prepared = all_prepared[anchor]
    overlaps = data.household_disjointness(prepared)
    if not source["valid"] or not members["valid"] or any(overlaps.values()):
        raise ValueError("hash, source or household admission failed")
    report = {"schema": 1, "anchor": anchor, "source": source,
              "pinned_private_members": members,
              "frozen_encoder_sha256": data.member_record(value, anchor, "encoder")["sha256"],
              "prepared_sha256": data.member_record(value, anchor, "prepared")["sha256"],
              "mechanism_rows": len(prepared["roles"]["mechanism"]),
              "historical_pool_household_overlaps": overlaps,
              "global_inner_split_sha256": split["assignment_sha256"],
              "inner_selection_rows": split["anchors"][anchor]["row_counts"]["inner_selection"],
              "inner_pilot_rows": split["anchors"][anchor]["row_counts"]["inner_pilot"],
              "class_support": split["anchors"][anchor]["class_support"],
              "outer_labels": "stripped_by_default_loader"}
    (out / "ADMISSION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"admitted": True, "anchor": anchor}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("admit", "fit-u1", "fit-bank", "fit-arm", "audit-inner"))
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=("U0P0", "U1P0", "U0P1", "U1P1"))
    parser.add_argument("--budget")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if "private" not in out.parts:
        raise ValueError("Scientific unit output must be private")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and args.command != "fit-bank":
        raise FileExistsError("Scientific unit already has artifacts")
    value = data.index(args.index)
    root = STUDY / "private/run"
    if args.command == "admit":
        result = _admit(args.anchor, value, out)
    elif args.command == "fit-u1":
        result = fit.fit_u1(args.anchor, args.index, out)
    elif args.command == "fit-bank":
        result = fit.fit_reference_bank(args.anchor, args.index, out,
                                        source_channels=fit.SOURCES,
                                        attack_slate="standard", resume=True)
    elif args.command == "fit-arm":
        if args.arm is None or args.budget is None:
            raise ValueError("Registered arm and budget required")
        u1 = root / f"a{args.anchor}_u1_decoder" if args.arm.startswith("U1") else None
        if args.arm.endswith("P1"):
            result = fit.solve_from_bank(args.anchor, args.index,
                root / f"a{args.anchor}_reference_bank", out, arm=args.arm,
                delta=float(args.budget), u1_dir=u1)
        else:
            result = fit.solve_p0_arm(args.anchor, args.index, out, arm=args.arm,
                                      budget=float(args.budget), u1_dir=u1)
    else:
        if args.arm is None or args.budget is None:
            raise ValueError("Registered arm and budget required")
        budget_tag = {"-0.002":"m002", "0":"z000", "+0.002":"p002",
                      "0.005":"b005", "0.01":"b010", "0.02":"b020"}.get(args.budget)
        if budget_tag is None:
            raise ValueError("Unregistered budget spelling")
        fit_dir = root / f"a{args.anchor}_{args.arm.lower()}_{budget_tag}"
        fit_file = fit_dir / ("FIT_P1_ARM.json" if args.arm.endswith("P1") else "FIT_P0_ARM.json")
        fit_record = json.loads(fit_file.read_text())
        channel_relative = fit_record.get("channel_relative")
        if channel_relative is None:
            result = {"status": "NOT_TRIGGERED_NO_VALID_CHANNEL", "fit_unit": fit_dir.name}
            (out / "INNER_PANEL.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            channel_dir = fit_dir / channel_relative
            if data.sha256_file(channel_dir / "Q.npz") != fit_record["channel_sha256"]:
                raise ValueError("Frozen channel hash differs from completed fit")
            channel = release.ChannelArtifact.load(channel_dir)
            prepared = data.load_prepared(value, args.anchor)
            all_prepared = {k: data.load_prepared(value, k) for k in (0, 1, 2)}
            split = data.global_validation_split(all_prepared)
            center_h = root / f"a{args.anchor}_u1p1_z000_audit" / "H"
            shared_h = center_h if fit_dir.name != f"a{args.anchor}_u1p1_z000" and center_h.is_dir() else None
            result = inner_panel.run_inner_panel(prepared, channel.Q, split,
                fit_dir.name, out, slate="standard", shared_h_root=shared_h)
    _record_and_close(out, args.command, result)
    print(json.dumps({"unit": os.environ["PCRL_UNIT_ID"], "status": "complete"}, sort_keys=True))


if __name__ == "__main__":
    main()

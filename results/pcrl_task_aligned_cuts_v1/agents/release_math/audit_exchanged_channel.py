"""Run one registered standard-slate inner-pilot audit of the exchanged Q.

This is descriptive 2018 development validation. It never opens the outer pool
or changes the historical candidate, bank, nomination, or source programme.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.pcrl_task_aligned_cuts_v1 import data, inner_panel, release, scheduler
from experiments.pcrl_task_aligned_cuts_v1.pipeline import source_tree


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--sidecar-sha256", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    sidecar_path = Path(args.sidecar)
    if file_sha(sidecar_path) != args.sidecar_sha256:
        raise ValueError("exchange-audit sidecar bytes changed")
    pin = json.loads(sidecar_path.read_text())
    if (pin.get("schema") != "pcrl-exchange-audit-sidecar-v1" or
            pin.get("status") != "frozen_unrun" or pin.get("anchor") != 0 or
            pin.get("slate") != "standard" or pin.get("outer_pool_opened") is not False):
        raise ValueError("exchange audit scope differs from registration")
    if file_sha(Path(__file__)) != pin["execution_script_sha256"]:
        raise ValueError("exchange-audit script changed")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != pin["source_commit"]:
        raise ValueError("scientific source HEAD changed")
    if source_tree()["source_tree_sha256"] != pin["source_tree_sha256"]:
        raise ValueError("scientific source tree changed")
    for item in pin["input_files"].values():
        if file_sha(Path(item["relative_path"])) != item["sha256"]:
            raise ValueError(f"exchange audit pinned input differs: {item['relative_path']}")
    output = Path(pin["output_relative_dir"])
    if output.exists():
        raise FileExistsError("existing exchange audit output must remain immutable")
    channel = release.ChannelArtifact.load(pin["channel_relative_dir"])
    if file_sha(Path(pin["channel_relative_dir"]) / "Q.npz") != pin["input_files"]["channel_Q"]["sha256"]:
        raise ValueError("channel changed before audit")
    index = data.index(pin["input_index_relative_path"])
    prepared = data.load_prepared(index, 0)
    all_prepared = {anchor: data.load_prepared(index, anchor) for anchor in (0, 1, 2)}
    split = data.global_validation_split(all_prepared)
    if split["assignment_sha256"] != pin["split_assignment_sha256"]:
        raise ValueError("inner household assignment changed")
    report = inner_panel.run_inner_panel(
        prepared, channel.Q, split, pin["release_id"], output,
        slate="standard", shared_h_root=pin["shared_H_relative_dir"])
    if report["channel_sha256"] != pin["channel_array_sha256"]:
        raise ValueError("audited Q differs from registered Q value bytes")
    provenance = {
        "schema": "pcrl-exchange-audit-provenance-v1",
        "unit_id": pin["unit_id"],
        "sidecar_sha256": args.sidecar_sha256,
        "source_commit": pin["source_commit"],
        "source_tree_sha256": pin["source_tree_sha256"],
        "channel_array_sha256": pin["channel_array_sha256"],
        "slate": pin["slate"],
        "outer_pool_opened": False,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    artifacts = [str(path.relative_to(output)) for path in sorted(output.rglob("*"))
                 if path.is_file() and path.name != "COMPLETE.json"]
    scheduler.write_completion(output, pin["unit_id"], args.sidecar_sha256, artifacts)
    print(json.dumps({"unit_id": pin["unit_id"], "status": "complete",
                      "channel_array_sha256": report["channel_sha256"],
                      "roles": sorted(report["roles"]),
                      "outer_pool_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()

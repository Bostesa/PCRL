"""Replay one frozen exchanged attack bank with the matched deterministic MILP.

This results-side diagnostic changes no fitted release, predictor, or protocol.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import controls, scheduler
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
        raise ValueError("exchanged-MILP sidecar byte hash changed")
    pin = json.loads(sidecar_path.read_text())
    if pin.get("schema") != "pcrl-exchanged-milp-sidecar-v1" or pin.get("status") != "frozen_unrun":
        raise ValueError("MILP diagnostic is not registered and frozen")
    if file_sha(Path(__file__)) != pin["execution_script_sha256"]:
        raise ValueError("MILP diagnostic script changed after registration")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != pin["source_commit"]:
        raise ValueError("scientific source commit changed")
    if source_tree()["source_tree_sha256"] != pin["source_tree_sha256"]:
        raise ValueError("scientific source tree changed")
    for item in pin["input_files"].values():
        if file_sha(Path(item["relative_path"])) != item["sha256"]:
            raise ValueError(f"pinned input differs: {item['relative_path']}")
    output = Path(pin["output_relative_dir"])
    output.mkdir(parents=True, exist_ok=False)
    cost, cuts, bank_sha = controls.load_saved_bank(pin["bank_relative_dir"])
    if bank_sha != pin["bank_sha256"] or len(cuts) != pin["cut_count"]:
        raise ValueError("exchanged bank hash or cut count differs")
    solved = controls.solve_deterministic_p1(
        cost, cuts, time_limit_seconds=float(pin["time_limit_seconds"]),
        relative_gap=float(pin["relative_gap_target"]))
    q = solved.pop("Q")
    solved.pop("replay")
    solved.pop("fixed_bank_cut_ids")
    replay = None
    channel_file = None
    if q is not None:
        replay = controls.replay(q, controls.make_bank(cost, cuts), deterministic=True)
        if not replay["feasible"]:
            raise AssertionError("MILP incumbent failed independent bank replay")
        np.savez_compressed(output / "MILP_Q.npz", Q=q)
        channel_file = "MILP_Q.npz"
    report = {
        "schema": "pcrl-exchanged-milp-result-v1",
        "unit_id": pin["unit_id"],
        "sidecar_sha256": args.sidecar_sha256,
        "source_commit": pin["source_commit"],
        "source_tree_sha256": pin["source_tree_sha256"],
        "bank_sha256": bank_sha,
        "cut_count": len(cuts),
        "solver": solved,
        "incumbent_channel_file": channel_file,
        "incumbent_channel_file_sha256": file_sha(output / channel_file) if channel_file else None,
        "independent_replay": {
            "objective_nats": replay["objective"],
            "max_cut_violation_nats": replay["maximum_cut_violation"],
            "simplex_row_error": replay["row_error"],
            "binary_error": replay["binary_error"],
            "feasible": replay["feasible"],
        } if replay else None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "anchor-0 fixed 324-cut bank; no independent audit or global attack guarantee",
    }
    (output / "EXCHANGED_MILP.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    artifacts = ["EXCHANGED_MILP.json"] + ([channel_file] if channel_file else [])
    scheduler.write_completion(output, pin["unit_id"], args.sidecar_sha256, artifacts)
    print(json.dumps({"bank_sha256": bank_sha, "cut_count": len(cuts),
                      "incumbent_valid": solved["incumbent_valid"],
                      "incumbent_objective": solved["incumbent_objective"],
                      "valid_lower_bound": solved["lower_bound"],
                      "solver_status": solved["status_code"]}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Immutable private A-center privacy-first LP unit and T0 audit adapter.

This runner consumes only a completed frozen bank and its completed D17
control. It performs no attack, decoder, encoder, or independent audit fit.
One channel is saved for a later coordinator-controlled common audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import fit_controls, privacy_first


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "COMPLETE.json"}


def _json_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def _private(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("private unit path cannot be a symlink")
    root = raw.resolve()
    if "private" not in root.parts:
        raise ValueError("privacy-first fitted channel requires a private output path")
    return root


def load_release_spec(output_dir: str | Path) -> dict[str, Any]:
    """Return the standard evaluate.token_law_for_release T0 spec after hash replay."""
    root = _private(output_dir)
    complete = json.loads((root/"COMPLETE.json").read_text())
    if (complete.get("schema") != "pcrl-adaptive-privacy-first-fit-v1"
            or complete.get("status") != "COMPLETE"
            or complete.get("artifact_sha256") != _inventory(root)):
        raise ValueError("privacy-first artifact inventory or completion differs")
    record = json.loads((root/"RELEASE_SPEC.json").read_text())
    if (record.get("schema") != "pcrl-adaptive-privacy-first-T0-spec-v1"
            or record.get("router") != "T0"
            or record.get("channel_relative_path") != "channel/Q.npz"):
        raise ValueError("privacy-first release spec differs from T0 contract")
    path = root/record["channel_relative_path"]
    if _sha(path) != record.get("channel_file_sha256"):
        raise ValueError("privacy-first channel file hash differs")
    with np.load(path, allow_pickle=False) as archive:
        if list(archive.files) != ["Q"]:
            raise ValueError("privacy-first channel archive must contain only Q")
        q = archive["Q"].copy()
    if (q.shape != (32, 17) or fit_controls._array_sha(q) != record.get("channel_array_sha256")
            or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))
            or np.max(np.abs(q.sum(axis=1)-1.)) > 1e-10):
        raise ValueError("privacy-first T0 channel content differs")
    return {"Q": q, "router": "T0", "channel_artifact_path": str(path),
            "channel_artifact_sha256": record["channel_file_sha256"]}


def _verified_complete(root: Path, inputs: dict, cost_pair, cuts, d17) -> dict:
    complete = json.loads((root/"COMPLETE.json").read_text())
    if (complete.get("schema") != "pcrl-adaptive-privacy-first-fit-v1"
            or complete.get("status") != "COMPLETE"
            or complete.get("inputs") != inputs
            or complete.get("artifact_sha256") != _inventory(root)):
        raise ValueError("completed privacy-first source, input or artifact inventory differs")
    spec = load_release_spec(root)
    fit = json.loads((root/"FIT.json").read_text())
    replay = privacy_first.replay_privacy_first(
        spec["Q"], float(complete["tau"]), cost_pair, cuts, d17)
    if (fit.get("status") != "OPTIMAL" or not replay["feasible"]
            or abs(float(fit["tau"])-float(complete["tau"])) > 1e-10
            or abs(replay["maximum_cut_violation"]-
                   float(fit["replay"]["maximum_cut_violation"])) > 1e-9
            or abs(replay["minimum_cut_slack"]-
                   float(fit["replay"]["minimum_cut_slack"])) > 1e-9):
        raise ValueError("completed privacy-first channel fails frozen-bank replay")
    return complete


def run_fit_unit(*, branch: str, anchor: int, delta: float,
                 center_dir: str | Path, controls_dir: str | Path,
                 output_dir: str | Path, time_limit_seconds: float = 60.) -> dict:
    """Solve once, retain all weights/provenance, or verify a completed unit."""
    if branch != "A" or anchor not in (0, 1, 2) or delta != .001:
        raise ValueError("registered A-center anchor and .001 allowance required")
    root = _private(output_dir)
    center_root = _private(center_dir)
    controls_root = _private(controls_dir)
    cost_pair, cuts, d17, source = privacy_first.load_frozen_inputs(
        branch, center_root, controls_root, anchor=anchor, delta=delta)
    if d17.shape != (32, 17):
        raise ValueError("privacy-first T0 unit requires 32 states and 17 tokens")
    center_complete = center_root/"COMPLETE.json"
    controls_complete = controls_root/"COMPLETE.json"
    if _sha(center_complete) != source.get("center_sha256"):
        raise ValueError("frozen center completion differs from selected source")
    inputs = {"branch": branch, "anchor": anchor, "delta": delta,
              "time_limit_seconds": float(time_limit_seconds),
              "source": source,
              "center_complete_sha256": _sha(center_complete),
              "controls_complete_sha256": _sha(controls_complete),
              "selected_bank_sha256": source["selected_bank_sha256"],
              "fixed_bank_sha256": fit_controls._bank_sha(
                  .5*(cost_pair["U"]+cost_pair["W"]), cuts),
              "d17_array_sha256": fit_controls._array_sha(d17),
              "cost_pair_sha256": {key: fit_controls._array_sha(cost_pair[key])
                                   for key in ("U", "W")},
              "privacy_first_source_sha256": _sha(Path(privacy_first.__file__)),
              "fit_source_sha256": _sha(Path(__file__))}
    if (not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0
            or inputs["selected_bank_sha256"] is None):
        raise ValueError("positive time limit and frozen bank required")
    if (root/"COMPLETE.json").exists():
        return _verified_complete(root, inputs, cost_pair, cuts, d17)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("partial privacy-first unit retained for technical review")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    _json_new(root/"INPUTS.json", inputs)
    result = privacy_first.solve_privacy_first(
        cost_pair, cuts, d17, time_limit_seconds=time_limit_seconds)
    report = {key: value for key, value in result.items() if key != "Q"}
    if result["status"] != "OPTIMAL" or result["Q"] is None:
        _json_new(root/"FIT_UNRESOLVED.json", report)
        raise RuntimeError("privacy-first LP unresolved; partial unit preserved")
    channel_dir = root/"channel"
    channel_dir.mkdir(mode=0o700)
    channel_path = channel_dir/"Q.npz"
    np.savez_compressed(channel_path, Q=result["Q"])
    channel_path.chmod(0o600)
    _json_new(root/"FIT.json", report)
    release_record = {"schema": "pcrl-adaptive-privacy-first-T0-spec-v1",
                      "router": "T0", "channel_relative_path": "channel/Q.npz",
                      "channel_file_sha256": _sha(channel_path),
                      "channel_array_sha256": fit_controls._array_sha(result["Q"]),
                      "model_status": "EXPERIMENTAL_UNVALIDATED",
                      "wire": "unchanged H_A plus one of 17 tokens",
                      "hidden_state": "stored Linux x86 T0 code is never released"}
    _json_new(root/"RELEASE_SPEC.json", release_record)
    receipt = {"schema": "pcrl-adaptive-privacy-first-fit-v1",
               "status": "COMPLETE", "inputs": inputs,
               "tau": float(result["tau"]),
               "phase_one_minimum_common_violation": result["phase_one"]["minimum_common_violation"],
               "maximum_cut_violation": result["replay"]["maximum_cut_violation"],
               "maximum_task_cap_violation": max(result["replay"]["task_cap_violation"].values()),
               "dual_gap": result["dual_gap"],
               "artifact_sha256": _inventory(root)}
    _json_new(root/"COMPLETE.json", receipt)
    return _verified_complete(root, inputs, cost_pair, cuts, d17)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("A",), required=True)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--delta", type=float, choices=(.001,), required=True)
    parser.add_argument("--center-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=60.)
    args = parser.parse_args(argv)
    result = run_fit_unit(**vars(args))
    print(json.dumps({"status": result["status"], "tau": result["tau"],
                      "complete_sha256": _sha(args.output_dir/"COMPLETE.json"),
                      "fixed_bank_sha256": result["inputs"]["fixed_bank_sha256"]},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

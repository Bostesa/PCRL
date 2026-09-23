"""Run one pre-registered, paired LP/deterministic response-exchange unit.

The immutable sidecar pins every input byte and the executable source tree.
Only 2018 inner fitting/selection and mechanism coefficient rows are read;
outer assessment is never opened here. The private completion receipt retains
all fitted response models and numerical evidence for replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import data, fit, release, scheduler

REGISTRATION = Path("results/pcrl_task_aligned_cuts_v1/agents/release_math/EXCHANGE_SIDECAR.json")


def _array_digest(q):
    q = np.asarray(q, dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()


def _verify_registration(args, registration):
    from .pipeline import source_tree
    if registration.get("schema") != "pcrl-exchange-sidecar-v1":
        raise ValueError("unrecognized exchange sidecar schema")
    if registration.get("status") != "frozen_unrun":
        raise ValueError("exchange sidecar must be source-pinned and frozen before fitting")
    if registration.get("input_channel_array_digest_algorithm") != (
            "SHA256 of contiguous float64 Q value bytes only; shape is pinned separately, with no shape prefix"):
        raise ValueError("unrecognized channel array digest contract")
    expected = registration["argv"]
    current = [expected[0], "-m", "experiments.pcrl_task_aligned_cuts_v1.exchange_cli",
               *sys.argv[1:]]
    if current != expected or not Path(expected[0]).samefile(sys.executable):
        raise ValueError("invocation differs from the immutable sidecar argv")
    if registration["unit_id"] != "a0_u1p1_z000_exchange_r01":
        raise ValueError("unregistered exchange unit")
    if (args.anchor != registration["anchor"] or args.round != registration["round_index"]
            or args.delta != registration["delta_nats"]
            or args.output_dir != registration["output_relative_dir"]):
        raise ValueError("exchange parameters differ from registration")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (head != registration.get("execution_source_commit") or
            source_tree()["source_tree_sha256"] != registration.get("execution_source_tree_sha256")):
        raise ValueError("exchange source commit/tree differs from frozen sidecar")
    for item in registration["input_files"].values():
        path = Path(item["relative_path"])
        if data.sha256_file(path) != item["sha256"]:
            raise ValueError(f"exchange input hash mismatch: {path}")
    env_unit = os.environ.get("PCRL_UNIT_ID")
    if env_unit is not None and env_unit != registration["unit_id"]:
        raise ValueError("sidecar runner supplied a different immutable unit ID")


def _load_q_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        if list(archive.files) != ["Q"]:
            raise ValueError("deterministic control archive must contain only Q")
        return archive["Q"].copy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--index", required=True)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--cost-dir", required=True)
    parser.add_argument("--lp-channel", required=True)
    parser.add_argument("--det-channel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args(argv)
    if argv is not None:
        raise ValueError("exchange CLI must be invoked through its registered module argv")
    registration_bytes = REGISTRATION.read_bytes()
    registration = json.loads(registration_bytes)
    _verify_registration(args, registration)
    for key, value in (("--index", args.index), ("--bank-dir", args.bank_dir),
                       ("--cost-dir", args.cost_dir), ("--lp-channel", args.lp_channel),
                       ("--det-channel", args.det_channel), ("--output-dir", args.output_dir)):
        position = registration["argv"].index(key)
        if registration["argv"][position + 1] != value:
            raise ValueError(f"{key} path differs from sidecar")
    lp = release.ChannelArtifact.load(args.lp_channel).Q
    det = _load_q_npz(args.det_channel)
    for name, q in (("lp_q", lp), ("det_q", det)):
        pinned = registration["input_channel_arrays"][name]
        if list(q.shape) != pinned["shape"] or _array_digest(q) != pinned["sha256"]:
            raise ValueError(f"{name} array differs from sidecar pin")
    incident_retry = os.environ.get("PCRL_RETRY_INCIDENT") == "1"
    record = fit.exchange_round(args.anchor, args.index, args.bank_dir,
                                args.cost_dir, {"LP": lp, "DET": det},
                                args.output_dir, round_index=args.round,
                                delta=args.delta, attack_slate=registration["attack_slate"],
                                resume=incident_retry)
    out = Path(args.output_dir)
    provenance = {
        "schema": 1, "unit_id": registration["unit_id"],
        "sidecar_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "execution_source_commit": registration["execution_source_commit"],
        "execution_source_tree_sha256": registration["execution_source_tree_sha256"],
        "source_bank_sha256": registration["source_bank_sha256"],
        "result_bank_sha256": record["bank_sha256"],
        "result_status": record["status"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "SOURCE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    artifact_names = [str(path.relative_to(out)) for path in sorted(out.rglob("*"))
                      if path.is_file() and path.name != "COMPLETE.json"]
    scheduler.write_completion(out, registration["unit_id"],
                               hashlib.sha256(registration_bytes).hexdigest(), artifact_names)
    print(json.dumps({"unit_id": registration["unit_id"], "status": record["status"],
                      "registered_feasible": record["registered_feasible"],
                      "new_cut_count": record["new_cut_count"],
                      "bank_sha256": record["bank_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

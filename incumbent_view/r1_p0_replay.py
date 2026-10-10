"""Queued raw83 P0 CLI replay, exact original recipient JSONL wires."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r1_bank import load_bank, sha_file, wires_for_codes, wire_hash
from .r1_replay import P0_KEYS, exact_npz


def write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    active = assert_active_from_env(unit_id=args.unit_id, stage="verification")
    if sha_file(args.config) != args.config_sha256:
        raise PermissionError("P0 replay config bytes differ")
    config = json.loads(Path(args.config).read_text())
    if config["schema"] != "IVR1-r1-p0-replay-config-v1" or config["unit_id"] != args.unit_id:
        raise ValueError("P0 replay config identity differs")
    pins = {item["path"]: item["sha256"] for item in config["sources"]}
    if len(pins) != len(config["sources"]):
        raise ValueError("P0 replay source pin paths repeat")
    for field in ("source_manifest", "public_bank", "p0_fit", "raw32"):
        if pins.get(config[field]["path"]) != config[field]["sha256"]:
            raise ValueError("essential P0 replay source not pinned")
    for item in config["sources"]:
        if sha_file(item["path"]) != item["sha256"]:
            raise PermissionError("P0 replay pinned source bytes differ")
    protocol = Path(config["repo"]) / "results/pcrl_incumbent_view_risk_v1/r0"
    if (active["seal_sha256"] != sha_file(protocol / "ADMISSION_LOCK.json") or
            config["science_lock_sha256"] != sha_file(protocol / "CAMPAIGN_LOCK.json")):
        raise ValueError("P0 replay queue and science locks differ")

    source = exact_npz(config["p0_fit"]["path"], P0_KEYS)
    n = len(source["row_id"])
    if n != 15434 or any(source[key].shape != (n,) or source[key].dtype != np.int64
                         for key in ("row_id", "unit", "code_r1", "code_r2")):
        raise ValueError("P0 FIT row/unit/code shape differs")
    bank = load_bank(config["public_bank"]["path"], config["public_bank"]["sha256"])
    sample = np.linspace(0, n-1, 32, dtype=np.int64)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError("write-once P0 replay directory exists")
    output.mkdir(parents=True, mode=0o700)
    wire_hashes = {}
    for recipient in (1, 2):
        path = output / f"recipient{recipient}.jsonl"
        command = [config["python"], "-m", "incumbent_view.r1_deploy",
                   "--source-manifest", config["source_manifest"]["path"],
                   "--source-manifest-sha256", config["source_manifest"]["sha256"],
                   "--seed", str(config["source_seed"]),
                   "--recipient", str(recipient),
                   "--input", config["raw32"]["path"], "--output", str(path)]
        result = subprocess.run(command, cwd=config["repo"], capture_output=True,
                                text=True, timeout=90, check=False)
        if result.returncode:
            write_json(output / f"PRIVATE_CLI_FAILURE_R{recipient}.json", {
                "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError("P0 raw CLI failed; preserved private diagnostics")
        metadata = json.loads(result.stdout)
        if (metadata["seed"] != config["source_seed"] or
                metadata["recipient"] != recipient or metadata["rows"] != 32 or
                metadata["release_sha256"] != sha_file(path)):
            raise ValueError("P0 raw CLI stdout identity differs")
        with path.open() as stream:
            actual = [json.loads(line) for line in stream]
        expected = wires_for_codes(bank, recipient, source[f"code_r{recipient}"][sample])
        if len(actual) != 32 or tuple(actual) != expected:
            raise ValueError("raw83 P0 CLI differs from original FIT code wire")
        wire_hashes[f"r{recipient}"] = {"file_sha256": sha_file(path),
                                        "content_sha256": wire_hash(actual)}
    write_json(config["receipt_output"], {
        "schema": "IVR1-r1-p0-raw32-replay-v1", "status": "PASS",
        "unit_id": args.unit_id, "source_seed": config["source_seed"],
        "p0_fit_sha256": config["p0_fit"]["sha256"],
        "raw32_sha256": config["raw32"]["sha256"],
        "sample_rule": "np.linspace(0,15433,32,dtype=int64)",
        "release_sha256_by_recipient": wire_hashes,
        "parity": "32/32 each recipient against exact frozen P0 FIT wire",
        "label_fields_read": [], "queue_seal_sha256": active["seal_sha256"],
        "science_lock_sha256": config["science_lock_sha256"]})


if __name__ == "__main__":
    main()

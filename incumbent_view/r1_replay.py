"""Queued 32-person raw83 deployment replay against R2's frozen FIT wire.

Verification reads only admitted FIT P0 codes and the selected FIT release.
It never opens task or sensitive labels. R5 must independently replay it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import offline_uniform
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import offline_factor_uniforms
from .r1_bank import load_bank, sha_file, wires_for_codes, wire_hash
from .r1_channel import COIN_WIDTH
from .r2_package import load_policy


P0_KEYS = ("row_id", "unit", "code_r1", "code_r2", "prob_r1", "prob_r2", "decision_r1", "decision_r2")
SELECTED_KEYS = ("row_id", "unit", "code_r1", "code_r2")


def exact_npz(path, keys):
    with ZipFile(path) as stream:
        if (len(stream.infolist()) != len(keys) or
                {member.filename for member in stream.infolist()} != {key + ".npy" for key in keys}):
            raise PermissionError("role-only replay archive has unexpected members")
    with np.load(path, allow_pickle=False) as source:
        return {key: np.array(source[key], copy=True) for key in keys}


def write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def derive_coins(root, *, seed, unit, arm, role="OSF_DEFENSE_FIT", realization=0):
    width = COIN_WIDTH[arm]
    if width == 0:
        return None
    if width == 1:
        return np.asarray([offline_uniform(root, source_seed=seed, role=role,
                                           realization=realization, unit=int(u))
                           for u in unit], dtype=np.float64)
    return np.asarray([offline_factor_uniforms(root, source_seed=seed, role=role,
                                               realization=realization, unit=int(u))
                       for u in unit], dtype=np.float64)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    active = assert_active_from_env(unit_id=args.unit_id, stage="verification")
    if sha_file(args.config) != args.config_sha256:
        raise PermissionError("replay config bytes differ")
    config = json.loads(Path(args.config).read_text())
    if config["unit_id"] != args.unit_id or config["schema"] != "IVR1-r1-selected-replay-config-v1":
        raise ValueError("replay unit/config identity differs")
    source_pins = {item["path"]: item["sha256"] for item in config["sources"]}
    if len(source_pins) != len(config["sources"]):
        raise ValueError("replay source pins repeat a path")
    for field in ("public_bank", "p0_fit", "selected_fit", "selected_receipt",
                  "raw32", "stream_roots", "source_manifest"):
        item = config[field]
        if source_pins.get(item["path"]) != item["sha256"]:
            raise ValueError("essential replay source missing from pinned source list")
    for item in config["sources"]:
        if sha_file(item["path"]) != item["sha256"]:
            raise PermissionError("pinned replay source bytes differ")

    seed = config["source_seed"]
    bank = load_bank(config["public_bank"]["path"], config["public_bank"]["sha256"])
    source = exact_npz(config["p0_fit"]["path"], P0_KEYS)
    chosen = exact_npz(config["selected_fit"]["path"], SELECTED_KEYS)
    if (not np.array_equal(source["row_id"], chosen["row_id"]) or
            not np.array_equal(source["unit"], chosen["unit"])):
        raise ValueError("P0 and selected FIT rows/groups differ")
    n = len(source["row_id"])
    if n != 15434 or any(a.shape != (n,) or a.dtype != np.int64
                         for a in (source["row_id"], source["unit"],
                                   source["code_r1"], source["code_r2"],
                                   chosen["code_r1"], chosen["code_r2"])):
        raise ValueError("selected FIT shape/type differs")
    for recipient in (1, 2):
        p = np.asarray(bank["recipients"][str(recipient)]["probability"], dtype=np.float64)
        c = source[f"code_r{recipient}"]
        if (not np.array_equal(source[f"prob_r{recipient}"], p[c]) or
                not np.array_equal(source[f"decision_r{recipient}"], p[c].argmax(1))):
            raise ValueError("P0 FIT source code/decoder differs")

    receipt_path = Path(config["policy_dir"]) / "POLICY_RECEIPT.json"
    if sha_file(receipt_path) != config["policy_receipt_sha256"]:
        raise PermissionError("policy receipt differs")
    policy_receipt = json.loads(receipt_path.read_text())
    policy = load_policy(config["policy_dir"], expected_receipt_sha256=config["policy_receipt_sha256"])
    if (policy_receipt["source_seed"] != seed or policy_receipt["arm_id"] != config["arm_id"] or
            policy_receipt["source_lock_sha256"] != config["policy_lock_sha256"] or
            policy_receipt["public_bank_sha256"] != config["public_bank"]["sha256"]):
        raise ValueError("selected policy lock, source or bank differs")
    selected_receipt = json.loads(Path(config["selected_receipt"]["path"]).read_text())
    if (selected_receipt["source_seed"] != seed or
            selected_receipt["arm_id"] != config["arm_id"] or
            selected_receipt["role"] != "OSF_DEFENSE_FIT" or
            selected_receipt["realization"] != 0 or
            selected_receipt["policy_receipt_sha256"] != config["policy_receipt_sha256"] or
            selected_receipt["source_binding_sha256"] != bank["source_binding_sha256"] or
            selected_receipt["codes_file_sha256"] != config["selected_fit"]["sha256"]):
        raise ValueError("selected cached FIT release binding differs")

    sample = np.linspace(0, n-1, 32, dtype=np.int64)
    roots = json.loads(Path(config["stream_roots"]["path"]).read_text())
    if roots["schema"] != "scv1-private-stream-roots-v1":
        raise ValueError("private release stream schema differs")
    root = bytes.fromhex(roots["release_realization_roots_hex"]["0"])
    if hashlib.sha256(root).hexdigest() != selected_receipt["private_stream_root_sha256"]:
        raise ValueError("cached FIT release used a different private draw root")
    coins = derive_coins(root, seed=seed, unit=source["unit"][sample], arm=policy.law_kind)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError("write-once raw replay directory already exists")
    output.mkdir(parents=True, mode=0o700)
    coin_path = output / "uniforms.npz"
    if coins is not None:
        with os.fdopen(os.open(coin_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            np.savez_compressed(stream, uniform=coins)
            stream.flush()
            os.fsync(stream.fileno())
    wire_paths = [output / "recipient1.jsonl", output / "recipient2.jsonl"]
    command = [config["python"], "-m", "incumbent_view.r1_selected_deploy",
               "--source-manifest", config["source_manifest"]["path"],
               "--source-manifest-sha256", config["source_manifest"]["sha256"],
               "--seed", str(seed), "--policy-dir", config["policy_dir"],
               "--policy-receipt-sha256", config["policy_receipt_sha256"],
               "--policy-lock-sha256", config["policy_lock_sha256"],
               "--arm-id", config["arm_id"],
               "--input", config["raw32"]["path"],
               "--input-sha256", config["raw32"]["sha256"],
               "--output-r1", str(wire_paths[0]), "--output-r2", str(wire_paths[1])]
    if coins is not None:
        command.extend(["--uniforms", str(coin_path), "--uniforms-sha256", sha_file(coin_path)])
    result = subprocess.run(command, cwd=config["repo"], capture_output=True,
                            text=True, timeout=90, check=False)
    if result.returncode:
        write_json(output / "PRIVATE_CLI_FAILURE.json", {
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        raise RuntimeError("selected raw CLI failed; preserved private diagnostics")
    metadata = json.loads(result.stdout)
    if metadata["rows"] != 32 or metadata["arm_id"] != config["arm_id"]:
        raise ValueError("raw selected CLI stdout identity differs")
    actual = [read_jsonl(path) for path in wire_paths]
    if any(len(wires) != 32 for wires in actual):
        raise ValueError("selected raw CLI row count differs")
    for recipient in (1, 2):
        expected = wires_for_codes(bank, recipient, chosen[f"code_r{recipient}"][sample])
        if actual[recipient-1] != expected:
            raise ValueError("raw83 selected CLI differs from full FIT cached original-ID wire")
    record = {"schema": "IVR1-r1-selected-raw32-replay-v1", "unit_id": args.unit_id,
              "status": "PASS", "source_seed": seed, "arm_id": config["arm_id"],
              "policy_receipt_sha256": config["policy_receipt_sha256"],
              "selected_fit_receipt_sha256": config["selected_receipt"]["sha256"],
              "p0_fit_sha256": config["p0_fit"]["sha256"],
              "raw32_sha256": config["raw32"]["sha256"],
              "stream_roots_sha256": config["stream_roots"]["sha256"],
              "sample_rule": "np.linspace(0,15433,32,dtype=int64)",
              "coin_kind": policy.law_kind, "coin_file_sha256": sha_file(coin_path) if coins is not None else None,
              "release_sha256_by_recipient": {f"r{r}": sha_file(wire_paths[r-1]) for r in (1, 2)},
              "wire_content_sha256_by_recipient": {f"r{r}": wire_hash(actual[r-1]) for r in (1, 2)},
              "parity": "32/32 each recipient against R2 FIT realization 0 codes and P0 original wire",
              "label_fields_read": [], "queue_seal_sha256": active["seal_sha256"]}
    write_json(config["receipt_output"], record)


if __name__ == "__main__":
    main()

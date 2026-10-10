"""Queue-only black-box deployment and reader replay from restored bytes.

Run this module with PYTHONPATH set only to the restored `files/code` tree.
All scientific inputs are resolved below the restored root. The queue control
plane remains external, but no original model, bank, policy or role data is
read. This worker opens no task or sensitive label array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r1_bank import load_bank, sha_file, wires_for_codes, wire_hash
from .r1_channel import COIN_WIDTH
from .r1_replay import P0_KEYS, SELECTED_KEYS, derive_coins, exact_npz, read_jsonl
from .r2_package import load_policy


def write_json(path: Path, value: dict) -> str:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def within(root: Path, path: Path) -> bool:
    return path.resolve(strict=True).is_relative_to(root.resolve(strict=True))


def source_alias(root: Path, entries: dict, alias: str) -> Path:
    item = entries[alias]
    path = root / "files" / alias
    if (path.is_symlink() or not within(root, path) or
            path.stat().st_size != item["size_bytes"] or sha_file(path) != item["sha256"]):
        raise PermissionError("restored source byte or locality differs")
    return path


def cli(command: list[str], *, cwd: Path, output: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=90, check=False)
    if result.returncode:
        write_json(output, {"returncode": result.returncode,
                            "stdout": result.stdout, "stderr": result.stderr})
        raise RuntimeError("restored CLI failed; private diagnostics preserved")
    return json.loads(result.stdout)


def release_replay(config: dict, root: Path, entries: dict, output: Path) -> dict:
    seed, arm, draw = config["seed"], config["arm_id"], config["realization"]
    code_root = root / "files/code"
    python = config["python"]
    manifest = root / "runtime/P0_SOURCE_PATHS.json"
    if not within(root, manifest):
        raise ValueError("restored source manifest path differs")
    bank_path = source_alias(root, entries, f"banks/s{seed}/P0_PUBLIC_BANK.json")
    bank = load_bank(bank_path, entries[f"banks/s{seed}/P0_PUBLIC_BANK.json"]["sha256"])
    raw = source_alias(root, entries, f"role/s{seed}/raw32.npz")
    p0_path = source_alias(root, entries, f"role/s{seed}/FIT_P0_RELEASE.npz")
    p0 = exact_npz(p0_path, P0_KEYS)
    sample = np.linspace(0, len(p0["unit"])-1, 32, dtype=np.int64)
    if len(p0["unit"]) != 15434:
        raise ValueError("restored P0 FIT person count differs")
    outputs = [output / f"recipient{i}.jsonl" for i in (1, 2)]
    if arm == "P0":
        if draw != 0:
            raise ValueError("P0 is a deterministic exact alias; replay only draw 0")
        for recipient in (1, 2):
            command = [python, "-m", "incumbent_view.r1_deploy",
                       "--source-manifest", str(manifest),
                       "--source-manifest-sha256", sha_file(manifest),
                       "--seed", str(seed), "--recipient", str(recipient),
                       "--input", str(raw), "--output", str(outputs[recipient-1])]
            metadata = cli(command, cwd=code_root,
                           output=output / f"PRIVATE_CLI_FAILURE_R{recipient}.json")
            if metadata["release_sha256"] != sha_file(outputs[recipient-1]):
                raise ValueError("restored P0 CLI stdout hash differs")
        expected_codes = (p0["code_r1"], p0["code_r2"])
        coin_sha = None
        policy_sha = None
    else:
        policy_alias = f"policies/{arm}/s{seed}/policy.npz"
        receipt_alias = f"policies/{arm}/s{seed}/POLICY_RECEIPT.json"
        source_alias(root, entries, policy_alias)
        receipt = source_alias(root, entries, receipt_alias)
        policy_dir = receipt.parent
        policy_sha = sha_file(receipt)
        policy = load_policy(policy_dir, expected_receipt_sha256=policy_sha)
        policy_record = json.loads(receipt.read_text())
        if (policy_record["arm_id"] != arm or policy_record["source_seed"] != seed or
                policy_record["source_lock_sha256"] != config["policy_lock_sha256"]):
            raise ValueError("restored policy arm differs")
        cache = source_alias(root, entries,
                             f"policies/{arm}/s{seed}/draw{draw}/release_codes.npz")
        cache_receipt = source_alias(root, entries,
                                     f"policies/{arm}/s{seed}/draw{draw}/RELEASE_RECEIPT.json")
        cached = exact_npz(cache, SELECTED_KEYS)
        rec = json.loads(cache_receipt.read_text())
        if (rec["source_seed"] != seed or rec["arm_id"] != arm or
                rec["realization"] != draw or rec["policy_receipt_sha256"] != policy_sha or
                not np.array_equal(cached["row_id"], p0["row_id"]) or
                not np.array_equal(cached["unit"], p0["unit"])):
            raise ValueError("restored cached FIT release identity differs")
        width = COIN_WIDTH[policy.law_kind]
        command = [python, "-m", "incumbent_view.r1_selected_deploy",
                   "--source-manifest", str(manifest),
                   "--source-manifest-sha256", sha_file(manifest),
                   "--seed", str(seed), "--policy-dir", str(policy_dir),
                   "--policy-receipt-sha256", policy_sha,
                   "--policy-lock-sha256", config["policy_lock_sha256"],
                   "--arm-id", arm, "--input", str(raw), "--input-sha256", sha_file(raw),
                   "--output-r1", str(outputs[0]), "--output-r2", str(outputs[1])]
        coin_sha = None
        if width:
            root_path = source_alias(root, entries, "private/stream_roots.json")
            roots = json.loads(root_path.read_text())
            private_root = bytes.fromhex(roots["release_realization_roots_hex"][str(draw)])
            if hashlib.sha256(private_root).hexdigest() != rec["private_stream_root_sha256"]:
                raise ValueError("restored cached draw root differs")
            coins = derive_coins(private_root, seed=seed, unit=p0["unit"][sample],
                                 arm=policy.law_kind, realization=draw)
            coin_path = output / "uniforms.npz"
            with os.fdopen(os.open(coin_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                np.savez_compressed(stream, uniform=coins)
                stream.flush()
                os.fsync(stream.fileno())
            coin_sha = sha_file(coin_path)
            command.extend(["--uniforms", str(coin_path), "--uniforms-sha256", coin_sha])
        metadata = cli(command, cwd=code_root, output=output / "PRIVATE_CLI_FAILURE.json")
        if (metadata["rows"] != 32 or metadata["arm_id"] != arm or
                metadata["policy_receipt_sha256"] != policy_sha or
                metadata["release_r1_sha256"] != sha_file(outputs[0]) or
                metadata["release_r2_sha256"] != sha_file(outputs[1])):
            raise ValueError("restored selected CLI stdout differs")
        expected_codes = (cached["code_r1"], cached["code_r2"])
    hashes = {}
    for recipient in (1, 2):
        actual = read_jsonl(outputs[recipient-1])
        expected = wires_for_codes(bank, recipient, expected_codes[recipient-1][sample])
        if len(actual) != 32 or tuple(actual) != expected:
            raise ValueError("restored raw83 wire differs from archived FIT cache")
        hashes[f"r{recipient}"] = {"file_sha256": sha_file(outputs[recipient-1]),
                                    "wire_sha256": wire_hash(actual)}
    return {"arm_id": arm, "seed": seed, "realization": draw,
            "policy_receipt_sha256": policy_sha, "coin_sha256": coin_sha,
            "release_sha256_by_recipient": hashes,
            "parity": "32/32 both recipients against archived FIT source and original wire"}


def reader_replay(config: dict, root: Path, entries: dict, output: Path) -> dict:
    seed = config["seed"]
    model_path = source_alias(root, entries,
                              f"reader/s{seed}/model/model__PROB_HGB_15x200.joblib")
    p0_path = source_alias(root, entries,
                           f"reader/s{seed}/p0_inner_release/P0_RELEASE.npz")
    pred_path = source_alias(root, entries,
                             f"reader/s{seed}/predictions_inner/predictions_inner.npz")
    with np.load(p0_path, allow_pickle=False) as p0:
        x = np.concatenate((np.asarray(p0["prob_r1"][:32], dtype=np.float64),
                            np.asarray(p0["prob_r2"][:32], dtype=np.float64)), axis=1)
    with np.load(pred_path, allow_pickle=False) as saved:
        expected = np.asarray(saved["PROB_HGB_15x200"][:32], dtype=np.float64)
    actual = np.asarray(joblib.load(model_path).predict_proba(x)[:, 1], dtype=np.float64)
    if actual.shape != (32,) or not np.array_equal(actual, expected):
        raise ValueError("restored historical P* H2 reader predictions differ")
    prediction_path = output / "reader_predictions.npy"
    with os.fdopen(os.open(prediction_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.save(stream, actual)
        stream.flush()
        os.fsync(stream.fileno())
    return {"arm_id": "P0_H2_FROZEN_READER", "seed": seed,
            "model_sha256": sha_file(model_path), "prediction_sha256": sha_file(prediction_path),
            "max_prediction_gap": 0.0, "prediction_count": 32,
            "label_fields_read": []}


def fresh_reader_replay(config: dict, root: Path, entries: dict, output: Path) -> dict:
    """Recompute a frozen NEW R4 probability reader from restored public codes."""
    seed, arm, draw = config["seed"], config["arm_id"], config["realization"]
    unit = f"R4_FINAL_{arm}_S{seed}_R{draw}"
    public = source_alias(root, entries, f"public/r4/{unit}_RECEIPT.json")
    public_record = json.loads(public.read_text())
    if (public_record["unit_id"] != unit or public_record["arm_id"] != arm or
            public_record["source_seed"] != seed or
            public_record["realization"] != draw):
        raise ValueError("restored fresh-reader public identity differs")
    base = f"campaign/r4/attacks/{unit}/pair/readers/"
    reader_receipt = source_alias(root, entries, base + "READERS_RECEIPT.json")
    reader_record = json.loads(reader_receipt.read_text())
    if (sha_file(reader_receipt) !=
            public_record["attack"]["reader_custody"]["pair"]["private_receipt_sha256"]):
        raise ValueError("restored new reader public/private custody pin differs")
    model_name = "model__PROB_HGB_15x200.bin"
    model_path = source_alias(root, entries, base + model_name)
    prediction_path = source_alias(root, entries, base + "predictions_inner.npz")
    view_path = source_alias(root, entries, base + "view_inner.npz")
    if (sha_file(model_path) != reader_record["files"][model_name] or
            sha_file(prediction_path) != reader_record["files"]["predictions_inner.npz"] or
            sha_file(view_path) != reader_record["files"]["view_inner.npz"]):
        raise ValueError("restored new reader file pin differs")
    bank_path = source_alias(root, entries, f"banks/s{seed}/P0_PUBLIC_BANK.json")
    bank = load_bank(bank_path, entries[f"banks/s{seed}/P0_PUBLIC_BANK.json"]["sha256"])
    if arm == "P0":
        release_alias = (f"campaign/r1/R1_P0_S{seed}_ADMISSION/"
                         "INNER_SELECTION/P0_RELEASE.npz")
        release = exact_npz(source_alias(root, entries, release_alias), P0_KEYS)
    else:
        release_alias = (f"campaign/r2/units/R2_{arm}_S{seed}_FIT/"
                         f"INNER_SELECTION_R{draw}/release_codes.npz")
        release = exact_npz(source_alias(root, entries, release_alias), SELECTED_KEYS)
    p1 = np.asarray(bank["recipients"]["1"]["probability"], dtype=np.float64)
    p2 = np.asarray(bank["recipients"]["2"]["probability"], dtype=np.float64)
    decoded = np.concatenate((p1[release["code_r1"][:32]],
                              p2[release["code_r2"][:32]]), axis=1)
    spans = {name: (int(start), int(stop)) for name, start, stop in
             reader_record["view_schema"]["component_columns"]}
    with np.load(view_path, allow_pickle=False) as view:
        if not np.array_equal(view["row_id"][:32], release["row_id"][:32]):
            raise ValueError("restored fresh reader row alignment differs")
        x = np.concatenate((view["x"][:32, slice(*spans["r1.probability"])],
                            view["x"][:32, slice(*spans["r2.probability"])]), axis=1)
    if x.shape != (32, 8) or not np.array_equal(x, decoded):
        raise ValueError("restored reader's public view differs from decoded P0 bank")
    with np.load(prediction_path, allow_pickle=False) as saved:
        expected = np.asarray(saved["PROB_HGB_15x200"][:32], dtype=np.float64)
    actual = np.asarray(joblib.load(model_path).predict_proba(x)[:, 1], dtype=np.float64)
    if actual.shape != (32,) or not np.array_equal(actual, expected):
        raise ValueError("restored fresh R4 probability reader predictions differ")
    out = output / "fresh_reader_predictions.npy"
    with os.fdopen(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.save(stream, actual)
        stream.flush()
        os.fsync(stream.fileno())
    return {"arm_id": arm, "seed": seed, "realization": draw,
            "reader_unit": unit, "reader_name": "PROB_HGB_15x200",
            "model_sha256": sha_file(model_path),
            "prediction_sha256": sha_file(out), "prediction_count": 32,
            "max_prediction_gap": 0.0, "label_fields_read": []}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    active = assert_active_from_env(unit_id=args.unit_id, stage="verification")
    if sha_file(args.config) != args.config_sha256:
        raise PermissionError("restored replay config differs")
    config = json.loads(Path(args.config).read_text())
    if config["schema"] != "IVR1-r1-restored-replay-config-v1" or config["unit_id"] != args.unit_id:
        raise ValueError("restored replay unit/config identity differs")
    root = Path(config["restore_root"]).resolve(strict=True)
    code_root = root / "files/code"
    if not Path(__file__).resolve(strict=True).is_relative_to(code_root):
        raise PermissionError("restored replay worker is not executing from archived code")
    receipt = json.loads((root / "RESTORE_RECEIPT.json").read_text())
    if sha_file(root / "RESTORE_RECEIPT.json") != config["restore_receipt_sha256"]:
        raise PermissionError("restored custody receipt differs")
    plan = json.loads((root / "metadata/CUSTODY_PLAN.json").read_text())
    if (plan["schema"] != "IVR1-r1-custody-plan-v1" or
            receipt["plan_sha256"] != sha_file(root / "metadata/CUSTODY_PLAN.json") or
            config["policy_lock_sha256"] != sha_file(root / "files/locks/campaign.json")):
        raise ValueError("restored plan or policy lock differs")
    control_lock = root / "files/locks/strongest_control.json"
    if sha_file(control_lock) != plan["strongest_control_lock_sha256"]:
        raise ValueError("restored strongest-control lock bytes differ")
    control_record = json.loads(control_lock.read_text())
    if (control_record["selected"] != plan["strongest_control"] or
            control_record["nominee"] != plan["nominee"]):
        raise ValueError("restored strongest-control decision differs")
    entries = {item["alias"]: item for item in plan["entries"]}
    if len(entries) != len(plan["entries"]):
        raise ValueError("restored plan aliases repeat")
    if config["mode"] in ("release", "fresh_reader"):
        arm = config["arm_id"]
        if arm not in {"P0", plan["nominee"], plan["joint_diagnostic"],
                       plan["strongest_control"]}:
            raise ValueError("unselected restored release arm")
        if config["mode"] == "fresh_reader" and config["realization"] != 0:
            raise ValueError("representative new reader is fixed at draw 0")
    elif config["mode"] not in ("reader", "fresh_reader"):
        raise ValueError("restored replay mode differs")
    if config["seed"] not in (0, 1) or config["realization"] not in (0, 1, 2):
        raise ValueError("restored seed/draw differs")
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError("write-once restored replay output exists")
    output.mkdir(parents=True, mode=0o700)
    if config["mode"] == "release":
        result = release_replay(config, root, entries, output)
    elif config["mode"] == "fresh_reader":
        result = fresh_reader_replay(config, root, entries, output)
    else:
        result = reader_replay(config, root, entries, output)
    write_json(Path(config["receipt_output"]), {
        "schema": "IVR1-r1-restored-numerical-replay-receipt-v1", "status": "PASS",
        "unit_id": args.unit_id, "restore_receipt_sha256": config["restore_receipt_sha256"],
        "queue_seal_sha256": active["seal_sha256"], "label_fields_read": [], **result})


if __name__ == "__main__":
    main()

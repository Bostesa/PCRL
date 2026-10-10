"""V2 queue-only black-box restore of B0/J wires and one HGB reader.

Deployment input, policy, models, draws, code, and reader all come from the
versioned local snapshot restore. Original files are never consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
from zipfile import ZipFile

import joblib
import numpy as np

from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env


WIRE = {"code", "prob", "decision"}
STDOUT = {"schema", "arm_id", "seed", "purpose", "rows",
          "policy_receipt_sha256", "public_table_sha256", "release_sha256"}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _new_file(path: Path, data: dict, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w") as out:
        json.dump(data, out, sort_keys=True, indent=2)
        out.write("\n")
        out.flush(); os.fsync(out.fileno())


def _code_matches_archive(restore: Path) -> int:
    tar = restore / "code/runtime_source.tar"
    count = 0
    with tarfile.open(tar, "r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            actual = restore / "code_root" / member.name
            if (actual.is_symlink() or not actual.is_file() or
                    actual.stat().st_size != member.size):
                raise ValueError("restored source archive member differs")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError("source archive member cannot be opened")
            h = hashlib.sha256(payload.read()).hexdigest()
            if _sha(actual) != h:
                raise ValueError("restored source bytes differ")
            count += 1
    return count


def _wire(path: Path, rows: int, classes: int) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        if {part.filename for part in archive.infolist()} != {
                f"{key}.npy" for key in WIRE}:
            raise ValueError("recipient wire has hidden field")
    with np.load(path, allow_pickle=False) as archive:
        value = {key: np.array(archive[key], copy=True) for key in WIRE}
    if (value["code"].dtype != np.uint8 or
            value["code"].shape != (rows, 10) or
            not np.all((value["code"] == 0) | (value["code"] == 1)) or
            value["prob"].dtype != np.float64 or
            value["prob"].shape != (rows, classes) or
            value["decision"].dtype != np.int64 or
            value["decision"].shape != (rows,)):
        raise ValueError("recipient wire shape or dtype differs")
    return value


def _run_restored_cli(command: list[str], *, code_root: Path, env: dict,
                      output: Path, seed: int, arm: str, purpose: int) -> dict:
    proc = subprocess.run(command, cwd=code_root, env=env,
                          capture_output=True, text=True, check=False,
                          timeout=30)
    if proc.returncode:
        failure = output / "NESTED_CLI_FAILURE.json"
        _new_file(failure, {
            "schema": "pcrl-bank-r1-private-nested-cli-failure-v1",
            "seed": seed, "arm_id": arm, "purpose": purpose,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-65536:],
            "stdout": proc.stdout[-65536:],
        }, 0o600)
        raise RuntimeError(
            f"restored CLI failed for S{seed}_{arm}_R{purpose}; "
            f"private diagnostic SHA-256 {_sha(failure)}")
    return json.loads(proc.stdout)


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    queue = Path(os.environ["PCRL_QUEUE_ROOT"])
    state = json.loads((queue / "state.json").read_text())
    sources = {Path(s["path"]): s["sha256"]
               for s in state["units"][active["unit_id"]]["spec"]["sources"]}
    if (config_path not in sources or _sha(config_path) != sources[config_path] or
            config_path.is_symlink() or
            stat.S_IMODE(config_path.stat().st_mode) & 0o077):
        raise PermissionError("restore replay config is not queue-pinned")
    cfg = json.loads(config_path.read_text())
    if (cfg.get("schema") != "pcrl-bank-r1-restore-replay-config-v2" or
            cfg.get("unit_id") != active["unit_id"]):
        raise ValueError("restore replay identity differs")
    restore = Path(cfg["restore_root"])
    if not restore.is_dir() or stat.S_IMODE(restore.stat().st_mode) & 0o077:
        raise PermissionError("restore root must be owner-only")
    receipt_path = restore / "RESTORE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "HASH_VERIFIED" or
            _sha(receipt_path) != cfg["restore_receipt_sha256"] or
            receipt.get("plan_sha256") != cfg["custody_plan_sha256"] or
            receipt.get("runtime_archive_sha256") != cfg["runtime_archive_sha256"]):
        raise ValueError("restored custody receipt differs")
    plan_path = Path(cfg["snapshot_root"]) / "PRIVATE_PLAN.json"
    if _sha(plan_path) != cfg["custody_plan_sha256"]:
        raise ValueError("snapshot plan differs")
    plan = json.loads(plan_path.read_text())
    if len(plan["entries"]) != 55:
        raise ValueError("restored source inventory differs")
    for item in plan["entries"]:
        target = restore / item["archive_name"]
        if (target.is_symlink() or not target.is_file() or
                _sha(target) != item["sha256"]):
            raise ValueError("restored dependency differs")
    code_count = _code_matches_archive(restore)
    output = Path(cfg["private_output_dir"])
    output.mkdir(mode=0o700)
    if stat.S_IMODE(output.stat().st_mode) != 0o700:
        raise PermissionError("private restore output must be owner-only")
    code_root = restore / "code_root"
    env = {**os.environ, "PYTHONPATH": str(code_root),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    checks: dict[str, dict] = {}
    b0_s0 = {}
    for seed in (0, 1):
        seed_dir = restore / "data" / f"seed{seed}"
        witness = json.loads((seed_dir / "sample/witness_receipt.json").read_text())
        witness_cfg = json.loads((seed_dir / "sample/witness_config.json").read_text())
        table = seed_dir / "public_table" / f"W_BANK_PUBLIC_TABLES_S{seed}.json"
        table_json = json.loads(table.read_text())
        indices = np.asarray(witness_cfg["indices"], dtype=np.int64)
        if (witness["status"] != "PASS" or witness["seed"] != seed or
                indices.shape != (32,) or not np.array_equal(
                    indices, np.linspace(0, 15433, 32, dtype=np.int64))):
            raise ValueError("representative FIT raw sample differs")
        for arm in ("B0", "J"):
            arm_dir = seed_dir / "selected" / arm
            policy = arm_dir / "policy/POLICY_RECEIPT.json"
            policy_json = json.loads(policy.read_text())
            cached = arm_dir / "OSF_DEFENSE_FIT_R0/release_codes.npz"
            with np.load(cached, allow_pickle=False) as archive:
                if set(archive.files) != {"row_id", "unit", "code_r1", "code_r2"}:
                    raise ValueError("cached selected release fields differ")
                cached_ids = {purpose: np.asarray(archive[f"code_r{purpose+1}"][indices],
                                                  dtype=np.int64)
                              for purpose in (0, 1)}
            public_receipt = Path(cfg["public_deployment_receipts"][f"S{seed}_{arm}"])
            if sources.get(public_receipt) != _sha(public_receipt):
                raise ValueError("completed deployment receipt not queue-pinned")
            prior = json.loads(public_receipt.read_text())
            if prior.get("status") != "PASS":
                raise ValueError("original CLI deployment was not complete")
            for purpose in (0, 1):
                key = f"S{seed}_{arm}_R{purpose}"
                wire_path = output / f"{key}.npz"
                command = [cfg["python"], "-m", "bank_anchored.r1_deploy",
                           "--purpose", str(purpose), "--seed", str(seed),
                           "--source-root", str(seed_dir / "source_root"),
                           "--source-sha256", witness["source_model_sha256"],
                           "--qpc-root", str(seed_dir / "qpc_root"),
                           "--public-table", str(table),
                           "--public-table-sha256", _sha(table),
                           "--schema", str(seed_dir / "sample/schema.json"),
                           "--schema-sha256", witness_cfg["schema_sha256"],
                           "--policy-dir", str(arm_dir / "policy"),
                           "--policy-receipt-sha256", _sha(policy),
                           "--source-binding-sha256", policy_json["source_binding_sha256"],
                           "--input", str(seed_dir / "sample/input.npz"),
                           "--output", str(wire_path)]
                if arm == "J":
                    draws = arm_dir / "draws.npz"
                    command += ["--draws", str(draws), "--draws-sha256", _sha(draws)]
                stdout = _run_restored_cli(command, code_root=code_root,
                                           env=env, output=output, seed=seed,
                                           arm=arm, purpose=purpose)
                if (set(stdout) != STDOUT or stdout["schema"] !=
                        "pcrl-bank-anchored-deployment-v1" or
                        stdout["release_sha256"] != _sha(wire_path)):
                    raise ValueError("restored CLI metadata differs")
                expected_sha = prior["wire_by_recipient"][str(purpose)]["release_sha256"]
                if _sha(wire_path) != expected_sha:
                    raise ValueError("restored CLI wire bytes differ from queued original")
                value = _wire(wire_path, 32, (2, 6)[purpose])
                table_np = np.asarray(table_json["families"]["W_TSTAR"][f"table{purpose}"],
                                      dtype=np.float64)
                ids = (value["code"].astype(np.int64) *
                       (1 << np.arange(10, dtype=np.int64))).sum(axis=1)
                if (not np.array_equal(ids, cached_ids[purpose]) or
                        not np.array_equal(value["prob"], table_np[ids]) or
                        not np.array_equal(value["decision"],
                                           table_np[ids].argmax(axis=1))):
                    raise ValueError("restored code, bank, or cached FIT law differs")
                checks[key] = {"wire_sha256": _sha(wire_path), "rows": 32,
                               "fixed_bank_decode": "PASS", "cached_code_parity": True}
                if seed == 0 and arm == "B0":
                    b0_s0[purpose] = value["prob"]
    model_path = restore / "reader/model__PROB_HGB_15x200.bin"
    original_model = restore / "reader/original_model__PROB_HGB_15x200.joblib"
    if _sha(model_path) != _sha(original_model):
        raise ValueError("reader custody duplicate differs")
    reader = joblib.load(model_path)
    features = np.concatenate((b0_s0[0], b0_s0[1]), axis=1)
    prediction = np.asarray(reader.predict_proba(features)[:, 1], dtype=np.float64)
    if (features.shape != (32, 8) or prediction.shape != (32,) or
            not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1))):
        raise ValueError("restored nonlinear reader output differs")
    prediction_path = output / "B0_S0_READER_PREDICTION.npz"
    with os.fdopen(os.open(prediction_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "wb") as stream:
        np.savez_compressed(stream, prediction=prediction)
        stream.flush(); os.fsync(stream.fileno())
    result = {"schema": "pcrl-bank-r1-snapshot-cli-reader-replay-v2",
              "status": "PASS", "unit_id": active["unit_id"],
              "snapshot_receipt_sha256": cfg["snapshot_receipt_sha256"],
              "restore_receipt_sha256": _sha(receipt_path),
              "runtime_code_files_verified": code_count,
              "deployment_wires": checks,
              "reader": {"model_sha256": _sha(model_path),
                         "input_shape": [32, 8], "prediction_shape": [32],
                         "prediction_file_sha256": _sha(prediction_path),
                         "finite_unit_interval": True},
              "scope": "Snapshot-only raw CLI and nonlinear reader on 32 FIT samples; no labels"}
    _new_file(output / "REPLAY_RECEIPT.json", result, 0o600)
    _new_file(Path(cfg["public_receipt"]), result, 0o644)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), sort_keys=True))


if __name__ == "__main__":
    main()

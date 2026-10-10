"""Prepare the eight-arm, two-seed utility interval queue unit without role reads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_file
from .utility_interval_worker import validate_interval_config


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
R1 = STUDY / "r1"
R4 = STUDY / "r4"
UNIT_ID = "R4_UTILITY_INTERVALS_ALL8_A04"


def record(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha_file(resolved)}


def _write_new(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(role_manifest: Path, gate_b10: Path, gate_b16: Path,
            private_dir: Path) -> dict:
    role = record(role_manifest)
    manifest = json.loads(Path(role["path"]).read_text())
    for bits, path in ((10, gate_b10), (16, gate_b16)):
        gate = json.loads(path.read_text())
        if (gate.get("schema") != "pcrl-r2-t0-utility-gate-receipt-v1"
                or gate.get("status") not in ("POINT_FAIL", "POINT_ALL_CAPS_PASS")
                or gate.get("bits") != bits
                or gate.get("role_manifest_sha256") != role["sha256"]):
            raise PermissionError("T0 point gate receipt missing or differs")
    seal = record(STUDY / "CAMPAIGN_LOCK.json")
    randomness = record(STUDY / "INFERENCE_RANDOMNESS_LOCK.json")
    execution = record(STUDY / "UTILITY_INTERVAL_EXECUTION_LOCK.json")
    source_lock = record(STUDY / "UTILITY_INTERVAL_SOURCE_LOCK.json")
    ladder_decision = record(STUDY / "CAPACITY_LADDER_DECISION.json")
    amendment = record(STUDY / "AMENDMENT_04_DEPLOYMENT_DIAGNOSTIC.json")
    worker = record(R4 / "utility_interval_worker.py")
    utility = record(R4 / "utility_intervals.py")
    inference = record(R4 / "inference_rule.py")
    family = record(R4 / "finite_family.py")
    config = {
        "schema": "pcrl-r4-utility-interval-config-v1",
        "unit_id": UNIT_ID, "worker_sha256": worker["sha256"],
        "utility_source_sha256": utility["sha256"],
        "inference_rule_source_sha256": inference["sha256"],
        "finite_family_source_sha256": family["sha256"],
        "seal": seal, "inference_lock": randomness,
        "inference_proposal": record(R4 / "INFERENCE_RULE_PROPOSAL.json"),
        "finite_family_proposal": record(R4 / "D_FINITE_FAMILY_PROPOSAL.json"),
        "role_manifest": role, "candidates": [],
        "output": str((R4 / "UTILITY_INTERVALS_RECEIPT.json").resolve()),
    }
    all_sources = {item["path"]: item for item in (
        seal, randomness, execution, source_lock, ladder_decision, amendment,
        worker, utility, inference,
        family, config["inference_proposal"], config["finite_family_proposal"],
        role, record(gate_b10), record(gate_b16))}

    def add(rec: dict) -> None:
        previous = all_sources.setdefault(rec["path"], rec)
        if previous != rec:
            raise ValueError("source path has inconsistent SHA")

    for field in ("income", "occupation", "references_s0", "references_s1"):
        key = f"INNER_SELECTION:{field}"
        path = Path(role["path"]).parent.parent / manifest["role_file_relative_path"][key]
        item = record(path)
        if item["sha256"] != manifest["role_file_sha256"][key]:
            raise PermissionError("admitted INNER role archive differs")
        add(item)

    for bits in (10, 16):
        for seed in (0, 1):
            arm = f"T0_B{bits}"
            suffix = "_R1" if bits == 16 else ""
            receipt = record(R1 / f"{arm}_S{seed}{suffix}_RECEIPT.json")
            add(receipt)
            doc = json.loads(Path(receipt["path"]).read_text())
            if (doc.get("schema") != "code-utility-witness-worker-v1"
                    or doc.get("status") != "COMPLETE" or doc.get("stage") != "fit"
                    or doc.get("seed") != seed or doc["result"].get("arm_id") != arm):
                raise PermissionError("T0 source receipt differs")
            private = R1 / ".private" / f"{arm}_S{seed}{suffix}"
            release = record(private / doc["result"]["roles"]["INNER_SELECTION"]
                             ["private_release_basename"])
            service = record(private / f"service__{arm}__s{seed}" / "manifest.json")
            weights = record(Path(service["path"]).with_name("weights.npz"))
            for item in (release, service, weights):
                add(item)
            if (release["sha256"] != doc["result"]["roles"]["INNER_SELECTION"]
                    ["private_release_sha256"]
                    or service["sha256"] != doc["result"]["package"]["manifest_sha256"]
                    or weights["sha256"] != doc["result"]["package"]["weights_sha256"]):
                raise PermissionError("T0 private release/service bytes differ")
            config["candidates"].append({
                "arm_id": arm, "kind": "learned", "seed": seed,
                "source_model_sha256": doc["source_model_sha256"],
                "receipt": receipt, "release": release, "service_manifest": service,
            })

    for seed in (0, 1):
        receipt = record(R1 / f"W_BANK_S{seed}_R1_RECEIPT.json")
        table = record(R1 / f"W_BANK_PUBLIC_TABLES_S{seed}.json")
        add(receipt)
        add(table)
        doc = json.loads(Path(receipt["path"]).read_text())
        if (doc.get("schema") != "code-utility-witness-worker-v1"
                or doc.get("status") != "COMPLETE" or doc.get("stage") != "bank"
                or doc.get("seed") != seed):
            raise PermissionError("bank source receipt differs")
        for arm in ("W_Q", "W_TSTAR"):
            item = doc["result"]["roles"]["INNER_SELECTION"]["arms"][arm]
            release = record(R1 / ".private" / f"W_BANK_S{seed}_R1" /
                             item["private_release_basename"])
            add(release)
            if (release["sha256"] != item["private_release_sha256"]
                    or table["sha256"] != item["public_table_sha256"]):
                raise PermissionError("bank release/table bytes differ")
            config["candidates"].append({
                "arm_id": arm, "kind": "bank", "seed": seed,
                "source_model_sha256": doc["source_model_sha256"],
                "receipt": receipt, "release": release, "public_table": table,
                "policy_sha256": item["policy_sha256"],
                "decoder_sha256": item["decoder_sha256"],
            })

    for rel in ("code_witness/model.py", "code_witness/service.py",
                "results/pcrl_latent_and_sampled_methods_v1/r01/role_store.py",
                "results/pcrl_code_utility_witness_v1/r4/attack_worker.py",
                "results/pcrl_code_utility_witness_v1/r4/bank_service.py",
                "results/pcrl_code_utility_witness_v1/r0/compute_queue.py"):
        add(record(ROOT / rel))
    validate_interval_config(
        config, unit_id=UNIT_ID, worker_sha256=worker["sha256"],
        utility_source_sha256=utility["sha256"],
        inference_rule_source_sha256=inference["sha256"],
        finite_family_source_sha256=family["sha256"],
        seal_sha256=seal["sha256"],
        inference_lock_sha256=randomness["sha256"])
    private_dir = private_dir.resolve(strict=False)
    if private_dir.exists() or private_dir.parent != R4 / ".private":
        raise PermissionError("new R4 owner-only unit directory required")
    private_dir.parent.mkdir(mode=0o700, exist_ok=True)
    private_dir.mkdir(mode=0o700)
    config_path = private_dir / "config.json"
    _write_new(config_path, config)
    add(record(config_path))
    spec = {
        "unit_id": UNIT_ID, "stage": "verification",
        "amendment_sha256": amendment["sha256"],
        "argv": [str(ROOT / ".venv/bin/python"), "-m",
                 "results.pcrl_code_utility_witness_v1.r4.utility_interval_worker",
                 "--config", str(config_path)],
        "cwd": str(ROOT), "max_wall_s": 900, "cpu_reserve_s": 900,
        "max_rss_gib": 2, "projected_output_bytes": 10000000,
        "deps": ["R2_T0_B10_GATE", "R2_T0_B16_GATE", "W_BANK_S0_R1", "W_BANK_S1_R1",
                 "T0_B10_S0", "T0_B10_S1", "T0_B16_S0_R1", "T0_B16_S1_R1"],
        "outputs": [config["output"]], "env": {},
        "sources": sorted(all_sources.values(), key=lambda x: x["path"]),
    }
    spec_path = private_dir / "spec.json"
    _write_new(spec_path, spec)
    return {"config_path": str(config_path), "config_sha256": sha_file(config_path),
            "spec_path": str(spec_path), "spec_sha256": sha_file(spec_path),
            "source_count": len(spec["sources"]),
            "candidate_ids": [f"{x['arm_id']}_s{x['seed']}" for x in config["candidates"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--gate-b10", type=Path, required=True)
    parser.add_argument("--gate-b16", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.role_manifest, args.gate_b10, args.gate_b16,
                             args.private_dir), sort_keys=True))


if __name__ == "__main__":
    main()

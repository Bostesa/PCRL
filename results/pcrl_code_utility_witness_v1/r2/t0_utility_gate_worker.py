"""Queued two-seed task-only point gate; opens only admitted INNER role arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from code_witness.service import load_service
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array,
)
from .loss_metrics import task_metrics


HERE = Path(__file__).resolve().parent
WORKER = HERE / "t0_utility_gate_worker.py"
METRICS = HERE / "loss_metrics.py"
MEMBERS = {"row_id", "unit", "code0", "prob0", "decision0",
           "code1", "prob1", "decision1"}
CAPS = {"accuracy_loss": 0.010, "nll_excess": 0.010, "brier_excess": 0.005}
HEADER = ("capacity_bits", "seed", "task", "reference", "metric", "candidate_value",
          "reference_value", "cost", "cap", "point_status")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _registered(config_path: Path, unit_id: str) -> dict[Path, str]:
    state = _read(Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json")
    sources = {Path(source["path"]).resolve(strict=True): source["sha256"]
               for source in state["units"][unit_id]["spec"]["sources"]}
    if sources.get(config_path) != _sha(config_path):
        raise PermissionError("private gate config is not queue registered")
    return sources


def _pin(path: Path, expected: str, sources: dict[Path, str]) -> Path:
    if path.is_symlink():
        raise PermissionError("gate source is a symlink")
    resolved = path.resolve(strict=True)
    if len(expected) != 64 or sources.get(resolved) != expected or _sha(resolved) != expected:
        raise PermissionError("gate source is not byte pinned by queue")
    return resolved


def _record(record: dict, sources: dict[Path, str]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError("gate source record needs exact path/SHA fields")
    return _pin(Path(record["path"]), record["sha256"], sources)


def _private_dir(path: Path) -> Path:
    if path.is_symlink():
        raise PermissionError("gate private directory is a symlink")
    value = path.resolve(strict=True)
    if (".private" not in value.parts or not value.is_dir()
            or stat.S_IMODE(value.stat().st_mode) & 0o077):
        raise PermissionError("gate source directory is not owner-only")
    return value


def _release(path: Path, expected: str, sources: dict[Path, str],
             row: np.ndarray, unit: np.ndarray, bits: int, service) -> dict:
    source = _pin(path, expected, sources)
    if ".private" not in source.parts:
        raise PermissionError("T0 person-level release must stay private")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != MEMBERS:
            raise ValueError("T0 release member inventory differs")
        wire = {name: np.array(archive[name], copy=True) for name in MEMBERS}
    if (not np.array_equal(wire["row_id"], row)
            or not np.array_equal(wire["unit"], unit)):
        raise PermissionError("T0 release is not aligned to admitted INNER people")
    for purpose, width in ((0, 2), (1, 6)):
        code = wire[f"code{purpose}"]
        prob = wire[f"prob{purpose}"]
        decision = wire[f"decision{purpose}"]
        if (code.dtype != np.uint8 or code.shape != (len(row), bits)
                or not np.isin(code, (0, 1)).all()
                or prob.dtype != np.float64 or prob.shape != (len(row), width)
                or decision.dtype != np.int64 or decision.shape != (len(row),)):
            raise ValueError("deployed hard-code wire schema differs")
        reconstructed = service.probabilities_from_code(purpose, code)
        if (not np.array_equal(prob, reconstructed)
                or not np.array_equal(decision, np.argmax(prob, axis=1))):
            raise ValueError("T0 probability/decision is not hard-code-derived")
    return wire


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="witness")
    config_path = config_path.resolve(strict=True)
    if (".private" not in config_path.parts or not config_path.is_file()
            or stat.S_IMODE(config_path.stat().st_mode) & 0o077
            or stat.S_IMODE(config_path.parent.stat().st_mode) & 0o077):
        raise PermissionError("T0 gate config must be owner-only")
    sources = _registered(config_path, active["unit_id"])
    _pin(WORKER, _sha(WORKER), sources)
    _pin(METRICS, _sha(METRICS), sources)
    config = _read(config_path)
    bits = config.get("bits")
    if (config.get("schema") != "pcrl-r2-t0-utility-gate-config-v1"
            or config.get("unit_id") != active["unit_id"]
            or bits not in (10, 16) or set(config.get("seeds", {})) != {"0", "1"}):
        raise PermissionError("T0 gate config/capacity differs")
    seal_path = _record(config["seal"], sources)
    if config["seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("T0 gate seal differs from queue")
    seal = _read(seal_path)
    role_path = _record(config["role_manifest"], sources)
    if seal.get("role_manifest_sha256") != config["role_manifest"]["sha256"]:
        raise PermissionError("T0 gate role manifest differs from seal")
    store = RoleOnlyStore(role_path)
    for field in ("income", "occupation", "references_s0", "references_s1"):
        key = f"INNER_SELECTION:{field}"
        path = store.package_root / store.manifest["role_file_relative_path"][key]
        _pin(path, store.manifest["role_file_sha256"][key], sources)
    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    assert_same_people(income, occupation)
    row, unit = income["row_id"], income["unit"]
    pair_sha = sha_array(np.column_stack((row, unit)))
    if pair_sha != store.manifest["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"]:
        raise PermissionError("T0 gate row/group identity differs")
    rows = []
    absolute = {}
    receipt_hashes = {}
    for seed in (0, 1):
        assert_active_from_env(unit_id=active["unit_id"], stage="witness")
        references = store.read("INNER_SELECTION", f"references_s{seed}")
        assert_same_people(income, references)
        entry = config["seeds"][str(seed)]
        receipt_path = _record(entry["receipt"], sources)
        receipt = _read(receipt_path)
        arm = f"T0_B{bits}"
        result = receipt.get("result", {})
        role_result = result.get("roles", {}).get("INNER_SELECTION", {})
        if (receipt.get("schema") != "code-utility-witness-worker-v1"
                or receipt.get("status") != "COMPLETE"
                or receipt.get("stage") != "fit" or receipt.get("seed") != seed
                or receipt.get("class_order") != {"income": [0, 1],
                                                  "occupation_group": [0, 1, 2, 3, 4, 5]}
                or receipt.get("role_manifest_sha256") != config["role_manifest"]["sha256"]
                or result.get("arm_id") != arm or result.get("bit_count") != bits
                or role_result.get("row_unit_pair_sha256") != pair_sha):
            raise PermissionError("T0 fit receipt/role/capacity differs")
        private = _private_dir(Path(entry["private_root"]))
        package = result["package"]
        manifest_path = private / f"service__{arm}__s{seed}" / "manifest.json"
        _pin(manifest_path, package["manifest_sha256"], sources)
        _pin(manifest_path.with_name("weights.npz"), package["weights_sha256"], sources)
        service = load_service(manifest_path,
                               expected_manifest_sha256=package["manifest_sha256"],
                               expected_source_sha256=receipt["source_model_sha256"],
                               expected_bits=bits)
        if (service.model_sha256 != package["model_sha256"]
                or service.scaler_sha256 != package["scaler_sha256"]):
            raise PermissionError("T0 restored model/scaler identity differs")
        release_path = private / role_result["private_release_basename"]
        wire = _release(release_path, role_result["private_release_sha256"],
                        sources, row, unit, bits, service)
        for key, value in role_result["array_sha256"].items():
            if key not in MEMBERS or sha_array(wire[key]) != value:
                raise PermissionError("T0 release array digest differs from receipt")
        receipt_hashes[str(seed)] = entry["receipt"]["sha256"]
        absolute[str(seed)] = {}
        for purpose, task, label_field in ((0, "income", "income"),
                                           (1, "occupation", "occupation")):
            label = (income if task == "income" else occupation)[label_field]
            candidate = task_metrics(wire[f"prob{purpose}"], label, task)
            absolute[str(seed)][task] = {name: candidate[name] for name in
                                          ("accuracy", "nll", "brier")}
            for ref_name in ("U0", "Ucal"):
                reference = task_metrics(references[f"{ref_name}_{purpose+1}"],
                                         label, task)
                costs = {
                    "accuracy_loss": (reference["accuracy"] - candidate["accuracy"],
                                      candidate["accuracy"], reference["accuracy"]),
                    "nll_excess": (candidate["nll"] - reference["nll"],
                                   candidate["nll"], reference["nll"]),
                    "brier_excess": (candidate["brier"] - reference["brier"],
                                     candidate["brier"], reference["brier"]),
                }
                for metric, (cost, cand, ref) in costs.items():
                    rows.append((bits, seed, task, ref_name, metric, cand, ref, cost,
                                 CAPS[metric], "PASS" if cost <= CAPS[metric] else "FAIL"))
    if len(rows) != 24:
        raise AssertionError("T0 gate requires exactly 24 point clauses")
    csv_path = Path(config["output_csv"]).resolve(strict=False)
    receipt_path = Path(config["output_receipt"]).resolve(strict=False)
    if (csv_path.parent != HERE or receipt_path.parent != HERE
            or csv_path.exists() or receipt_path.exists()):
        raise PermissionError("T0 gate output names must be new in R2 namespace")
    with csv_path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    failures = [dict(seed=seed, task=task, reference=reference, metric=metric,
                     cost=cost, cap=cap)
                for _, seed, task, reference, metric, _, _, cost, cap, status in rows
                if status == "FAIL"]
    result = {
        "schema": "pcrl-r2-t0-utility-gate-receipt-v1",
        "status": "POINT_ALL_CAPS_PASS" if not failures else "POINT_FAIL",
        "bits": bits, "unit_id": active["unit_id"],
        "seal_sha256": active["seal_sha256"],
        "role_manifest_sha256": config["role_manifest"]["sha256"],
        "inner_row_unit_pair_sha256": pair_sha,
        "inner_rows": len(row), "inner_groups": len(np.unique(unit)),
        "source_fit_receipts": receipt_hashes,
        "csv_sha256": _sha(csv_path), "clauses": len(rows),
        "failures": failures, "absolute_candidate": absolute,
        "scope": "reused INNER development point gate; intervals and privacy not assessed",
    }
    with receipt_path.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"status": result["status"], "bits": result["bits"],
                      "clauses": result["clauses"]}, sort_keys=True))


if __name__ == "__main__":
    main()

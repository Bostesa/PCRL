"""Queued, fail-closed pre-refit audit of old F/T/D-K and optional W banks.

The caller supplies a private, SHA-registered JSON config; this worker refuses
direct execution before opening any role or person-level release. Outputs are
aggregates, plus an owner-only hash/row-identity receipt without predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.service import load_service
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.f_service import load_f_service
from .loss_metrics import code_states, distillation_kl, task_metrics


HERE = Path(__file__).resolve().parent
WORKER = HERE / "loss_audit_worker.py"
METRICS = HERE / "loss_metrics.py"
OLD_HARD_MEMBERS = {"row_id", "unit", "code0", "prob0", "decision0",
                    "code1", "prob1", "decision1"}
OLD_F_MEMBERS = {"row_id", "unit", "features0", "prob0", "decision0",
                 "features1", "prob1", "decision1"}
TASKS = ((0, "income", "income", 2), (1, "occupation", "occupation", 6))
HEADER = ("seed", "task", "record_type", "first", "second", "metric", "value",
          "role", "interpretation")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _private(path: Path, *, exists: bool) -> Path:
    if path.is_symlink():
        raise PermissionError("private source/output cannot be a symlink")
    value = path.resolve(strict=exists)
    if ".private" not in value.parts:
        raise PermissionError("private source/output must have owner-only custody")
    target = value if exists else value.parent
    if not target.is_dir() or stat.S_IMODE(target.stat().st_mode) & 0o077:
        raise PermissionError("private custody directory is absent or group-readable")
    if not exists and value.exists():
        raise FileExistsError("audit private output must be new")
    return value


def _registered_sources(config_path: Path, unit_id: str) -> dict[Path, str]:
    queue_root = Path(os.environ["PCRL_QUEUE_ROOT"])
    unit = _read_json(queue_root / "state.json")["units"][unit_id]
    registered = {Path(x["path"]).resolve(strict=True): x["sha256"]
                  for x in unit["spec"]["sources"]}
    if registered.get(config_path) != _sha(config_path):
        raise PermissionError("private config is not byte-pinned by the active unit")
    return registered


def _require(path: Path, expected: str, registered: dict[Path, str]) -> Path:
    if path.is_symlink():
        raise PermissionError("registered source cannot be a symlink")
    value = path.resolve(strict=True)
    if (len(expected) != 64 or
            registered.get(value) != expected or _sha(value) != expected):
        raise PermissionError("registered source byte hash differs")
    return value


def _bound(record: dict, registered: dict[Path, str]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise ValueError("source record needs exact path/SHA fields")
    return _require(Path(record["path"]), record["sha256"], registered)


def _npz(path: Path, expected: str, members: set[str],
         registered: dict[Path, str]) -> dict[str, np.ndarray]:
    path = _require(path, expected, registered)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != members:
            raise ValueError("saved release has missing or extra arrays")
        return {name: np.array(archive[name], copy=True) for name in members}


def _identity(arrays: dict, row: np.ndarray, unit: np.ndarray, pair_sha: str) -> None:
    actual_row, actual_unit = arrays["row_id"], arrays["unit"]
    if (actual_row.dtype != np.int64 or actual_unit.dtype != np.int64
            or not np.array_equal(actual_row, row)
            or not np.array_equal(actual_unit, unit)
            or sha_array(np.column_stack((actual_row, actual_unit))) != pair_sha):
        raise PermissionError("saved release differs from admitted ordered INNER people")


def _hard(arrays: dict, service, labels: dict, refs: dict) -> tuple[dict, dict, dict]:
    out, states, training = {}, {}, {}
    for purpose, task, field, width in TASKS:
        bits = arrays[f"code{purpose}"]
        prob = arrays[f"prob{purpose}"]
        decision = arrays[f"decision{purpose}"]
        if (bits.dtype != np.uint8 or bits.shape != (len(decision), 10)
                or prob.dtype != np.float64 or prob.shape != (len(decision), width)
                or decision.dtype != np.int64 or not np.isin(bits, (0, 1)).all()):
            raise ValueError("old hard-code wire schema differs")
        reconstructed = service.probabilities_from_code(purpose, bits)
        if (not np.array_equal(prob, reconstructed) or
                not np.array_equal(decision, np.argmax(reconstructed, axis=1))):
            raise ValueError("old hard-code output does not reconstruct from public bits")
        out[task] = task_metrics(prob, labels[field], task)
        states[task] = code_states(bits)
        uncalibrated = service.uncalibrated_probabilities_from_code(purpose, bits)
        training[task] = {
            "uncalibrated_true_label_nll": task_metrics(uncalibrated, labels[field], task)["nll"],
            "uncalibrated_teacher_kl_Ucal": distillation_kl(refs[f"Ucal_{purpose+1}"],
                                                              uncalibrated, task),
            "calibrated_true_label_nll": out[task]["nll"],
        }
    return out, states, training


def _continuous(arrays: dict, service, labels: dict, refs: dict) -> tuple[dict, dict, dict]:
    out, training = {}, {}
    for purpose, task, field, width in TASKS:
        features = arrays[f"features{purpose}"]
        prob = arrays[f"prob{purpose}"]
        decision = arrays[f"decision{purpose}"]
        if (features.dtype != np.float32 or features.shape != (len(decision), 16)
                or prob.dtype != np.float64 or prob.shape != (len(decision), width)
                or decision.dtype != np.int64):
            raise ValueError("old continuous F wire schema differs")
        uncalibrated = service.uncalibrated_probabilities_from_features(purpose, features)
        alpha = float(service.calibrators[purpose]["alpha"])
        clipped = np.maximum(uncalibrated, 1e-12)
        clipped /= clipped.sum(axis=1, keepdims=True)
        expected = clipped ** alpha
        expected /= expected.sum(axis=1, keepdims=True)
        if (not np.array_equal(prob, expected) or
                not np.array_equal(decision, np.argmax(expected, axis=1))):
            raise ValueError("old F output does not reconstruct from transmitted features")
        out[task] = task_metrics(prob, labels[field], task)
        training[task] = {
            "uncalibrated_true_label_nll": task_metrics(uncalibrated, labels[field], task)["nll"],
            "uncalibrated_teacher_kl_Ucal": distillation_kl(refs[f"Ucal_{purpose+1}"],
                                                              uncalibrated, task),
            "calibrated_true_label_nll": out[task]["nll"],
        }
    states = {task: {"transmitted_continuous_dimensions": 16,
                     "finite_code_states": None} for _, task, _, _ in TASKS}
    return out, states, training


def _bank(arrays: dict, tables: dict, family: str, labels: dict) -> tuple[dict, dict, dict]:
    bank = tables["families"][family]
    out, states = {}, {}
    for purpose, task, field, width in TASKS:
        code = arrays[f"code{purpose}"]
        prob = arrays[f"prob{purpose}"]
        decision = arrays[f"decision{purpose}"]
        table = np.asarray(bank[f"table{purpose}"], dtype=np.float64)
        token_class = np.asarray(bank[f"token_class{purpose}"], dtype=np.int64)
        if (code.dtype != np.uint8 or code.shape != (len(decision), 10)
                or not np.isin(code, (0, 1)).all() or table.ndim != 2
                or table.shape[1] != width or len(table) > 1024
                or token_class.shape != (len(table),) or
                not np.array_equal(table.argmax(axis=1), token_class)):
            raise ValueError("compiled historical bank or ten-bit wire differs")
        ids = (code.astype(np.int64) * (1 << np.arange(10, dtype=np.int64))).sum(axis=1)
        if (np.any(ids >= len(table)) or prob.dtype != np.float64
                or not np.array_equal(prob, table[ids]) or
                decision.dtype != np.int64 or
                not np.array_equal(decision, np.argmax(table[ids], axis=1))):
            raise ValueError("W bank output is not exactly public-code-derived")
        out[task] = task_metrics(prob, labels[field], task)
        states[task] = {**code_states(code), "global_bank_states": int(len(table)),
                        "minimum_capacity_bits_for_bank": int((len(table)-1).bit_length())}
    return out, states, {}


def _append_rows(rows: list[dict], seed: int, metrics: dict, states: dict,
                 training: dict) -> None:
    for task in ("income", "occupation"):
        for arm, data in metrics[task].items():
            for metric in ("accuracy", "nll", "brier", "constant_accuracy",
                           "accuracy_gain_over_constant"):
                rows.append((seed, task, "absolute", arm, "", metric, data[metric],
                             "INNER_SELECTION", "same ordered people; unweighted"))
            for class_id, count in enumerate(data["class_counts"]):
                rows.append((seed, task, "class_count", arm, "", f"class_{class_id}", count,
                             "INNER_SELECTION", "same people; class count is label based"))
            for class_id, value in enumerate(data["per_class_recall"]):
                rows.append((seed, task, "recall", arm, "", f"class_{class_id}", value,
                             "INNER_SELECTION", "null only when class has no INNER row"))
            for metric, value in states[task].get(arm, {}).items():
                rows.append((seed, task, "state", arm, "", metric, value,
                             "INNER_SELECTION", "global code capacity vs occupied states"))
            for metric, value in training[task].get(arm, {}).items():
                rows.append((seed, task, "training_diagnostic", arm, "", metric,
                             value, "INNER_SELECTION", "Ucal teacher; uncalibrated fit head unless named calibrated"))
        for first, second in (("U0", "F"), ("F", "T"), ("T", "D-K")):
            for metric in ("accuracy", "nll", "brier"):
                rows.append((seed, task, "difference", first, second, metric,
                             metrics[task][second][metric] - metrics[task][first][metric],
                             "INNER_SELECTION", "second minus first; descriptive, not causal"))


def run(config_path: Path) -> dict:
    # Active queue verification precedes even reading a private config.
    active = assert_active_from_env(stage="witness")
    config_path = config_path.resolve(strict=True)
    if (".private" not in config_path.parts or not config_path.is_file()
            or stat.S_IMODE(config_path.parent.stat().st_mode) & 0o077
            or stat.S_IMODE(config_path.stat().st_mode) & 0o077):
        raise PermissionError("pre-refit config requires an owner-only private file")
    registered = _registered_sources(config_path, active["unit_id"])
    config = _read_json(config_path)
    if (config.get("schema") != "pcrl-r2-pre-refit-loss-config-v1"
            or config.get("unit_id") != active["unit_id"]
            or set(config.get("seeds", {})) != {"0", "1"}):
        raise PermissionError("pre-refit audit config identity differs")
    seal_path = _bound(config["seal"], registered)
    if config["seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("R0 seal differs from queue")
    seal = _read_json(seal_path)
    pins = seal.get("source_files_sha256", {})
    if (pins.get("r2/loss_audit_worker.py") != _sha(WORKER)
            or pins.get("r2/loss_metrics.py") != _sha(METRICS)):
        raise PermissionError("R2 endpoint sources are not pinned by R0 seal")
    role_manifest = _bound(config["role_manifest"], registered)
    store = RoleOnlyStore(role_manifest)
    if seal.get("role_manifest_sha256") != config["role_manifest"]["sha256"]:
        raise PermissionError("admitted role manifest not bound by R0 seal")
    meta = store.manifest["roles"]["INNER_SELECTION"]
    for field in ("income", "occupation", "references_s0", "references_s1"):
        key = f"INNER_SELECTION:{field}"
        path = (store.package_root / store.manifest["role_file_relative_path"][key]).resolve(strict=True)
        _require(path, store.manifest["role_file_sha256"][key], registered)
    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    assert_same_people(income, occupation)
    labels = {"income": income["income"], "occupation": occupation["occupation"]}
    row, unit = income["row_id"], income["unit"]
    pair_sha = meta["row_unit_pair_sha256"]
    public_rows: list[tuple] = []
    private_sources = {}
    seed_summary = {}
    for seed in (0, 1):
        active = assert_active_from_env(unit_id=config["unit_id"], stage="witness")
        ref = store.read("INNER_SELECTION", f"references_s{seed}")
        assert_same_people(income, ref)
        plan = config["seeds"][str(seed)]
        first_receipt_path = _bound(plan["first_receipt"], registered)
        follow_receipt_path = _bound(plan["followup_receipt"], registered)
        first, follow = _read_json(first_receipt_path), _read_json(follow_receipt_path)
        if (first.get("schema") != "latent-repair-first-paired-seed-receipt-v1"
                or first.get("status") != "T_DK_RELEASES_READY"
                or follow.get("schema") != "latent-repair-followup-controls-receipt-v1"
                or follow.get("status") != "D_FOLLOWUP_RELEASES_READY"
                or first.get("seed") != seed or follow.get("seed") != seed
                or follow.get("first_receipt_sha256") != plan["first_receipt"]["sha256"]
                or first.get("role_manifest_sha256") != config["role_manifest"]["sha256"]
                or follow.get("role_manifest_sha256") != config["role_manifest"]["sha256"]
                or first.get("source_sha256") != follow.get("source_sha256")):
            raise PermissionError("old first/followup package lineage differs")
        first_root = _private(Path(plan["first_private_root"]), exists=True)
        follow_root = _private(Path(plan["followup_private_root"]), exists=True)
        metrics = {task: {} for _, task, _, _ in TASKS}
        states = {task: {} for _, task, _, _ in TASKS}
        training = {task: {} for _, task, _, _ in TASKS}
        for purpose, task, field, _ in TASKS:
            for anchor in ("U0", "Ucal"):
                metrics[task][anchor] = task_metrics(ref[f"{anchor}_{purpose+1}"], labels[field], task)
        for arm, source, root in (("T", first, first_root), ("D-K", first, first_root),
                                  ("F", follow, follow_root)):
            package = source["packages"][arm]
            manifest_path = root / arm / "manifest.json"
            _require(manifest_path, package["manifest_sha256"], registered)
            _require(manifest_path.with_name("weights.npz"), package["weights_sha256"], registered)
            release_path = root / f"release__{arm}__INNER_SELECTION.npz"
            release_sha = source["roles"]["INNER_SELECTION"][arm]["private_release_sha256"]
            arrays = _npz(release_path, release_sha,
                          OLD_F_MEMBERS if arm == "F" else OLD_HARD_MEMBERS,
                          registered)
            _identity(arrays, row, unit, pair_sha)
            if arm == "F":
                service = load_f_service(manifest_path,
                    expected_manifest_sha256=package["manifest_sha256"],
                    expected_source_sha256=source["source_sha256"])
                task_out, state_out, train_out = _continuous(arrays, service, labels, ref)
            else:
                service = load_service(manifest_path,
                    expected_manifest_sha256=package["manifest_sha256"],
                    expected_source_sha256=source["source_sha256"])
                task_out, state_out, train_out = _hard(arrays, service, labels, ref)
            for task in ("income", "occupation"):
                metrics[task][arm] = task_out[task]
                states[task][arm] = state_out[task]
                training[task][arm] = train_out[task]
            private_sources[f"s{seed}_{arm}_release_sha256"] = release_sha
        bank_specs = plan.get("banks", {})
        if bank_specs:
            tables_path = _bound(plan["public_tables"], registered)
            tables = _read_json(tables_path)
            if (tables.get("seed") != seed or tables.get("bit_count") != 10
                    or set(bank_specs) - set(tables.get("families", {}))):
                raise PermissionError("W bank table seed/family/capacity differs")
            for family, record in sorted(bank_specs.items()):
                release_path = _bound(record["release"], registered)
                arrays = _npz(release_path, record["release"]["sha256"],
                              OLD_HARD_MEMBERS, registered)
                _identity(arrays, row, unit, pair_sha)
                task_out, state_out, train_out = _bank(arrays, tables, family, labels)
                for task in ("income", "occupation"):
                    metrics[task][family] = task_out[task]
                    states[task][family] = state_out[task]
                    training[task][family] = train_out
                private_sources[f"s{seed}_{family}_release_sha256"] = record["release"]["sha256"]
        _append_rows(public_rows, seed, metrics, states, training)
        seed_summary[str(seed)] = {
            task: {arm: {"accuracy": value["accuracy"], "nll": value["nll"],
                         "brier": value["brier"]}
                   for arm, value in metrics[task].items()}
            for task in ("income", "occupation")}
    private_dir = _private(Path(config["output_private_dir"]), exists=False)
    public_csv = Path(config["output_csv"]).resolve(strict=False)
    public_receipt = Path(config["output_receipt"]).resolve(strict=False)
    if (public_csv.exists() or public_receipt.exists()
            or public_csv.parent != HERE or public_receipt.parent != HERE):
        raise PermissionError("R2 outputs must be new public files in owned namespace")
    private_dir.mkdir(mode=0o700)
    with public_csv.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(public_rows)
        stream.flush()
        os.fsync(stream.fileno())
    private_receipt = {
        "schema": "pcrl-r2-pre-refit-private-audit-v1",
        "status": "OLD_RELEASE_CODE_AND_FEATURE_RECONSTRUCTION_PASS",
        "unit_id": config["unit_id"], "seal_sha256": active["seal_sha256"],
        "role_manifest_sha256": config["role_manifest"]["sha256"],
        "inner_row_unit_pair_sha256": pair_sha,
        "inner_rows": int(len(row)), "inner_groups": int(len(np.unique(unit))),
        "sources": private_sources,
        "public_csv_sha256": _sha(public_csv),
        "person_array_payload": False,
    }
    private_path = private_dir / "AUDIT_RECEIPT.json"
    with private_path.open("x") as stream:
        json.dump(private_receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    private_path.chmod(0o600)
    receipt = {
        "schema": "pcrl-r2-pre-refit-public-audit-v1",
        "status": "PASS", "unit_id": config["unit_id"],
        "seal_sha256": active["seal_sha256"],
        "csv_sha256": _sha(public_csv), "csv_rows": len(public_rows),
        "private_receipt_sha256": _sha(private_path),
        "role_manifest_sha256": config["role_manifest"]["sha256"],
        "source_receipts": {str(seed): {
            "first": config["seeds"][str(seed)]["first_receipt"]["sha256"],
            "followup": config["seeds"][str(seed)]["followup_receipt"]["sha256"],
        } for seed in (0, 1)},
        "metrics": seed_summary,
        "scope": "INNER role reused; old F/T/D-K restored; optional W_BANK; no confirmation",
    }
    with public_receipt.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"status": result["status"], "unit_id": result["unit_id"],
                      "csv_rows": result["csv_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()

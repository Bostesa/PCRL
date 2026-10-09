"""Sealed independent replay of one frozen focused mask pair and seed.

The worker restores hash-bound inherited head packages and reads only its
eleven declared FIT-task, ATT and INNER role files. It imports no M2/M3
selector or M4 scoring code. All person arrays stay in private outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
SHARED = REPO / "results/pcrl_method_portfolio_v1/shared"
C_SOURCE = REPO / "results/pcrl_method_portfolio_v1/teams/c"
for directory in (REPO, SHARED, C_SOURCE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common_eval import apply_global_temperature, load_calibrator
from prototype import AdultGroupedSchema
from release_package import restore_release_package
from safe_data_guard import SafeRoleStore

from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit
from results.pcrl_capacity_matched_feature_selection_v1.m0.queue import sha256
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_bank import reconstruct
from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_metrics import (
    binary_auc, task_metrics,
)
from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_selection import replay_fit_selection
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_artifacts import read_frozen_release
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_endpoint import replay_pair_endpoint
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_fit_validation import (
    array_sha256, compare_task_row, task_only_split,
)


VIEWS = ("income", "occupation_group", "coalition")
VIEW_TAGS = {"income": "INCOME", "occupation_group": "OCCUPATION",
             "coalition": "COALITION"}
FIXED_LOCAL_READERS = frozenset((
    "LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
    "RAW_LR_C1", "RAW_HGB_15x200",
))


def role_keys(seed: int) -> set[str]:
    return {
        "OSF_DEFENSE_FIT:features", "OSF_DEFENSE_FIT:income",
        "OSF_DEFENSE_FIT:occupation", f"OSF_DEFENSE_FIT:references_s{seed}",
        "ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
        "INNER_SELECTION:features", "INNER_SELECTION:income",
        "INNER_SELECTION:occupation", "INNER_SELECTION:sex",
        f"INNER_SELECTION:references_s{seed}",
    }


def _digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _private(path: Path) -> Path:
    root = (STUDY / ".private").resolve(strict=True)
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise PermissionError("M5 artifact escaped private custody")
    return resolved


def _same_people(*records: dict) -> None:
    first = records[0]
    row, unit = np.asarray(first["row_id"]), np.asarray(first["unit"])
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or len(np.unique(row)) != len(row)):
        raise PermissionError("M5 role row/group identity malformed")
    for record in records[1:]:
        if (not np.array_equal(np.asarray(record["row_id"]), row)
                or not np.array_equal(np.asarray(record["unit"]), unit)):
            raise PermissionError("M5 role row/group order differs")


def _prior_manifest(lock: dict, manifest: dict) -> Path:
    relative = Path(lock["prior_package_relative_to_worktree"])
    if relative.is_absolute() or not relative.parts:
        raise PermissionError("prior role package is not worktree relative")
    prior = (REPO / relative).resolve(strict=True)
    expected = (REPO.parent / "pcrl-method-portfolio-v1/results/"
                "pcrl_method_portfolio_v1").resolve()
    path = prior / "DATA_ACCESS_MANIFEST.json"
    if (prior != expected or manifest.get("prior_package_id") != prior.name
            or sha256(path) != lock["prior_data_manifest_sha256"]
            or sha256(path) != manifest["portfolio_manifest_sha256"]):
        raise PermissionError("prior admitted role manifest changed")
    return path


def _pre_role_gate(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    pair = args.pair_id
    if (type(args.seed) is not int or args.seed not in (0, 1, 2)
            or re.fullmatch(r"M_DROP_[A-Z_]+__M_DROP_[A-Z_]+", pair or "") is None):
        raise PermissionError("M5 seed or finite pair is outside frozen bank")
    unit = f"M5-REPLAY-S{args.seed}-{pair}"
    folder = STUDY / ".private/m5/replays" / f"seed_{args.seed}" / pair
    report, values = folder / "REPLAY_REPORT.json", folder / "REPLAY_VALUES.npz"
    if (args.unit_id != unit or Path(args.output_report).resolve() != report.resolve()
            or Path(args.output_values).resolve() != values.resolve()
            or Path(args.queue_root).resolve() != (STUDY / ".private/queue").resolve()):
        raise PermissionError("M5 frozen unit, queue or output identity differs")
    spec = assert_running_unit(unit, args.queue_root,
                               STUDY / "CAMPAIGN_LOCK.json",
                               "independent_replay", report)
    if (set(spec.get("role_inputs", {})) != role_keys(args.seed)
            or spec.get("outputs") != [str(report), str(values)]
            or spec.get("allow_child_processes") is not False):
        raise PermissionError("M5 role, output or process contract differs")
    lock_path = STUDY / "CAMPAIGN_LOCK.json"
    manifest_path = STUDY / "m1/DATA_ACCESS_MANIFEST.json"
    lock, manifest = json.loads(lock_path.read_text()), json.loads(
        manifest_path.read_text())
    if (sha256(manifest_path) != lock.get("source_map_sha256")
            or spec.get("source_map_sha256") != sha256(manifest_path)):
        raise PermissionError("focused role manifest differs from seal")
    _prior_manifest(lock, manifest)
    return spec, lock, manifest


def _done_parent(spec: dict, unit_id: str, path: Path,
                 state: dict, registry: dict) -> str:
    if unit_id not in spec.get("deps", []) or unit_id not in registry.get("unit_specs", {}):
        raise PermissionError("M5 required parent is absent from sealed dependencies")
    row = state.get("units", {}).get(unit_id)
    if not isinstance(row, dict) or row.get("status") != "DONE":
        raise PermissionError("M5 required parent has not completed")
    queue_root = STUDY / ".private/queue"
    spec_path = Path(row["spec_path"]).resolve(strict=True)
    if (not spec_path.is_relative_to(queue_root / "specs")
            or sha256(spec_path) != row.get("spec_sha256")):
        raise PermissionError("M5 parent queue spec changed")
    observed = json.loads(spec_path.read_text())
    sealed = {**registry["unit_specs"][unit_id],
              "seal_sha256": state["gate"]["seal_sha256"]}
    if observed != sealed or observed.get("unit_id") != unit_id:
        raise PermissionError("M5 parent differs from presealed unit")
    path = _private(path)
    digest = sha256(path)
    if row.get("output_sha256") != {str(path): digest}:
        raise PermissionError("M5 parent output SHA or output set changed")
    return digest


def _head_package(record: dict, *, purpose: str, mask_id: str,
                  seed: int, lock_sha: str, manifest_sha: str,
                  manifest: dict):
    root = STUDY / ".private"
    relative = Path(record["relative_package_path"])
    expected = (lock_sha, "head_pool", record["cache_key"],
                "release_package.json")
    if (relative.is_absolute() or relative.parts != expected
            or record.get("stage") not in (None, "validation")
            or record.get("purpose", purpose) != purpose
            or record.get("mask_id", mask_id) != mask_id):
        raise PermissionError("M5 head package path or purpose differs")
    path = _private(root / relative)
    receipt_path = _private(path.parent / "HEAD_POOL_RECEIPT.json")
    if (sha256(path) != record["package_sha256"]
            or sha256(receipt_path) != record["receipt_sha256"]):
        raise PermissionError("M5 head package or cache receipt SHA differs")
    receipt = json.loads(receipt_path.read_text())
    identity = receipt["cache_identity"]
    label = "occupation" if purpose == "occupation_group" else "income"
    specs = manifest["role_file_sha256"]
    source_hashes = {
        "fit_features": specs["OSF_DEFENSE_FIT:features"],
        "fit_label": specs[f"OSF_DEFENSE_FIT:{label}"],
        "calibration_features": specs["CALIBRATION_HELDOUT:features"],
        "calibration_label": specs[f"CALIBRATION_HELDOUT:{label}"],
    }
    if (identity.get("cache_key") != record["cache_key"]
            or identity.get("mask_id") != mask_id
            or identity.get("purpose") != purpose or identity.get("seed") != seed
            or identity.get("campaign_lock_sha256") != lock_sha
            or identity.get("data_manifest_sha256") != manifest_sha
            or identity.get("role_file_sha256") != source_hashes
            or receipt.get("package_sha256") != record["package_sha256"]
            or receipt.get("head_sha256") != record["head_sha256"]
            or receipt.get("calibrator_sha256") != record["calibrator_sha256"]):
        raise PermissionError("M5 head cache identity differs from admitted FIT/CAL")
    return restore_release_package(
        path, expected_sha256=record["package_sha256"], private_root=root,
        expected_campaign_lock_sha256=lock_sha,
        expected_data_manifest_sha256=manifest_sha,
        expected_source_role_hashes=source_hashes,
        load_calibrator=load_calibrator,
        apply_global_temperature=apply_global_temperature,
    )


def _write_json_once(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return sha256(path)


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return sha256(path)


def _task_metric_subset(labels: np.ndarray, probabilities: np.ndarray,
                        width: int) -> dict[str, float]:
    scored = task_metrics(labels, probabilities, width=width)
    return {key: float(scored[key]) for key in ("accuracy", "nll", "brier")}


def _fit_validation(seed: int, fit_receipt: dict, fit_roles: dict,
                    manifest: dict, lock_sha: str, manifest_sha: str,
                    bank: object) -> tuple[dict, dict]:
    """Restore all twelve FIT validation heads and recompute T/J arithmetic."""
    features = fit_roles["features"]
    income, occupation = fit_roles["income"], fit_roles["occupation"]
    refs_raw = fit_roles["references"]
    _same_people(features, income, occupation, refs_raw)
    x = np.asarray(features["X"])
    row, unit = np.asarray(features["row_id"]), np.asarray(features["unit"])
    names = tuple(map(str, features["feature_names"]))
    if (x.dtype != np.float32 or x.shape != (len(row), 83)
            or len(names) != 83 or not np.isfinite(x).all()):
        raise PermissionError("M5 FIT feature wire differs from 83-column schema")
    labels = {"income": np.asarray(income["income"]),
              "occupation_group": np.asarray(occupation["occupation"])}
    if any(value.dtype != np.int64 for value in labels.values()):
        raise PermissionError("M5 FIT task labels need exact int64")
    fit_idx, val_idx, split_receipt = task_only_split(
        unit, labels["income"], labels["occupation_group"], seed)
    if (split_receipt != fit_receipt.get("task_split_receipt")
            or array_sha256(np.column_stack((row, unit)).astype(np.int64)) !=
            fit_receipt.get("fit_row_unit_sha256")):
        raise PermissionError("M5 FIT split or row/group receipt differs")
    folder = STUDY / ".private/m2/fit" / f"seed_{seed}"
    table_path, task_receipt_path = folder / "TASK_TABLE.json", folder / "TASK_ONLY_RECEIPT.json"
    pair_path, plan_path = folder / "PAIR_TABLE.json", folder / "CONTROL_PLAN.json"
    for path, field in ((table_path, "task_table_sha256"),
                        (task_receipt_path, "task_only_receipt_sha256"),
                        (pair_path, "pair_table_sha256"),
                        (plan_path, "control_plan_sha256")):
        if sha256(_private(path)) != fit_receipt[field]:
            raise PermissionError(f"M5 FIT {field} file differs")
    table = json.loads(table_path.read_text())
    task_receipt = json.loads(task_receipt_path.read_text())
    pair_table = json.loads(pair_path.read_text())
    plan = json.loads(plan_path.read_text())
    if (table.get("schema") != "pcrl-focused-task-table-v1"
            or table.get("split_receipt") != split_receipt
            or table.get("bank_sha256") != bank.bank_sha256
            or table.get("task_table_sha256") != _digest_json({
                key: value for key, value in table.items() if key != "task_table_sha256"})
            or task_receipt.get("schema") != "pcrl-focused-task-only-receipt-v1"
            or task_receipt.get("seed") != seed
            or task_receipt.get("campaign_lock_sha256") != lock_sha
            or task_receipt.get("task_table_sha256") != sha256(table_path)
            or task_receipt.get("split_receipt") != split_receipt
            or plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
            or plan.get("selection_receipt_sha256") != _digest_json({
                key: value for key, value in plan.items()
                if key != "selection_receipt_sha256"})):
        raise PermissionError("M5 FIT task/plan self-hash or source identity differs")
    registry = task_receipt.get("validation_head_packages")
    if (registry != fit_receipt.get("validation_head_packages")
            or _digest_json(registry) !=
            fit_receipt.get("validation_head_packages_sha256")
            or _digest_json(registry) !=
            task_receipt.get("validation_head_packages_sha256")):
        raise PermissionError("M5 twelve validation package registry changed")
    expected_ids = {"F_FULL83", *(mask.mask_id for mask in bank.masks)}
    if (set(registry) != {"income", "occupation_group"}
            or any(set(registry[purpose]) != expected_ids for purpose in registry)):
        raise PermissionError("M5 FIT validation head coverage incomplete")
    public_schema = AdultGroupedSchema()
    metric_by_purpose, ref_metrics = {}, {}
    for purpose, label, width, suffix, table_key in (
            ("income", "income", 2, "1", "income_masks"),
            ("occupation_group", "occupation_group", 6, "2", "occupation_masks")):
        y_val = labels[purpose][val_idx]
        ref_metrics[purpose] = {}
        for ref in ("U0", "Ucal"):
            probability = np.asarray(refs_raw[f"{ref}_{suffix}"])
            if probability.shape != (len(row), width):
                raise PermissionError("M5 FIT U0/Ucal reference shape differs")
            ref_metrics[purpose][ref] = _task_metric_subset(
                y_val, probability[val_idx], width)
        f_package = _head_package(
            registry[purpose]["F_FULL83"], purpose=purpose,
            mask_id="F_FULL83", seed=seed, lock_sha=lock_sha,
            manifest_sha=manifest_sha, manifest=manifest)
        if tuple(public_schema.column_indices(f_package.selected_atoms)) != tuple(range(83)):
            raise PermissionError("M5 FIT F validation head lacks full 83-column input")
        f_prob = np.asarray(f_package.view(
            np.ascontiguousarray(x[val_idx]), names)["probabilities"])
        ref_metrics[purpose]["F"] = _task_metric_subset(y_val, f_prob, width)
        saved_rows = {row["mask_id"]: row for row in table[table_key]}
        if set(saved_rows) != {mask.mask_id for mask in bank.masks}:
            raise PermissionError("M5 FIT task table omitted a mask")
        metric_by_purpose[purpose] = {}
        for mask in bank.masks:
            record = registry[purpose][mask.mask_id]
            package = _head_package(
                record, purpose=purpose, mask_id=mask.mask_id,
                seed=seed, lock_sha=lock_sha, manifest_sha=manifest_sha,
                manifest=manifest)
            if tuple(package.selected_atoms) != tuple(mask.atoms):
                raise PermissionError("M5 FIT validation package mask differs")
            probability = np.asarray(package.view(
                np.ascontiguousarray(x[val_idx]), names)["probabilities"])
            metric = _task_metric_subset(y_val, probability, width)
            saved = saved_rows[mask.mask_id]
            if (saved.get("mask_sha256") != mask.mask_sha256
                    or saved.get("head_id") != record["model_id"]):
                raise PermissionError("M5 FIT task row mask/head identity differs")
            compare_task_row(saved, metric, ref_metrics[purpose])
            metric_by_purpose[purpose][mask.mask_id] = metric
    if (pair_table.get("schema") != "pcrl-focused-pair-table-v1"
            or pair_table.get("bank_sha256") != bank.bank_sha256
            or pair_table.get("table_sha256") != _digest_json({
                key: value for key, value in pair_table.items()
                if key not in ("table_sha256", "schema")})
            or set(plan.get("rules", {})) !=
            {"J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U"}):
        raise PermissionError("M5 FIT pair table or full control slate differs")
    for purpose, key in (("income", "income_masks"),
                         ("occupation_group", "occupation_masks")):
        for row in pair_table[key]:
            score = metric_by_purpose[purpose][row["mask_id"]]
            if any(abs(float(row[f"task_validation_{metric}"]) - score[metric]) > 1e-12
                   for metric in ("accuracy", "nll", "brier")):
                raise PermissionError("M5 pair table utility metric differs from head replay")
    selected = replay_fit_selection(pair_table, ref_metrics)
    for rule, pair_id in selected["selected_pair_ids"].items():
        saved = plan["rules"][rule]
        if saved.get("pair_id") != pair_id or \
                (saved.get("status") == "SELECTED") != (pair_id is not None):
            raise PermissionError("M5 FIT selected control differs from independent rule")
    return {"validation_package_count": 12,
            "task_table_sha256": sha256(table_path),
            "task_only_receipt_sha256": sha256(task_receipt_path),
            "pair_table_sha256": sha256(pair_path),
            "control_plan_sha256": sha256(plan_path),
            "independent_selected_pair_ids": selected["selected_pair_ids"],
            "fit_row_unit_sha256": fit_receipt["fit_row_unit_sha256"]}, plan


def _attack_tune_indices(unit: np.ndarray) -> tuple[np.ndarray, str]:
    groups = np.asarray(unit)
    if groups.dtype != np.int64 or groups.ndim != 1 or len(np.unique(groups)) != 4061:
        raise PermissionError("M5 ATT source has wrong frozen group geometry")
    salt = "pcrl-method-portfolio-v1|attack-tune|20261008|"
    ranked = sorted((hashlib.sha256((salt + str(int(group))).encode()).hexdigest(),
                     int(group)) for group in np.unique(groups))
    chosen = np.asarray([group for _, group in ranked[:1015]], dtype=np.int64)
    return np.flatnonzero(np.isin(groups, chosen)), array_sha256(np.sort(chosen))


def _reader_bank(path: Path, index_sha: str, *, seed: int, pair_id: str,
                 view: str, lock_sha: str, manifest_sha: str, release_sha: str,
                 raw_sha: str, inner_row: np.ndarray, inner_unit: np.ndarray,
                 sex_att: np.ndarray, tune_indices: np.ndarray,
                 attack_split_sha: str) -> tuple[dict[str, np.ndarray], dict]:
    """Check saved fixed readers and independently replay ATT tune AUCs."""
    if sha256(_private(path)) != index_sha:
        raise PermissionError("M5 ATT reader index changed")
    index = json.loads(path.read_text())
    expected_ids = (FIXED_LOCAL_READERS if view != "coalition" else
                    FIXED_LOCAL_READERS |
                    {f"IGNORE_R2__{name}" for name in FIXED_LOCAL_READERS} |
                    {f"IGNORE_R1__{name}" for name in FIXED_LOCAL_READERS})
    if (index.get("schema") != "pcrl-focused-complete-reader-index-v1"
            or index.get("status") != "FROZEN" or index.get("seed") != seed
            or index.get("pair_id") != pair_id or index.get("view_name") != view
            or index.get("lock_sha256") != lock_sha
            or index.get("manifest_sha256") != manifest_sha
            or index.get("release_index_sha256") != release_sha
            or index.get("raw_index_sha256") != raw_sha
            or index.get("attack_split_sha256") != attack_split_sha
            or index.get("inner_row_unit_sha256") != array_sha256(
                np.column_stack((inner_row, inner_unit)))
            or set(index.get("reader_order", ())) != expected_ids
            or len(index["reader_order"]) != len(expected_ids)
            or set(index.get("validation", {})) != expected_ids):
        raise PermissionError("M5 fixed reader identity, split or source differs")
    core_path = _private(path.parent / "core_readers/READERS_RECEIPT.json")
    if sha256(core_path) != index["core_readers_receipt_sha256"]:
        raise PermissionError("M5 complete reader core receipt changed")
    core = json.loads(core_path.read_text())
    for filename, digest in core.get("files", {}).items():
        child = _private(core_path.parent / filename)
        if sha256(child) != digest:
            raise PermissionError("M5 complete reader model/prediction changed")
    files = (("inner", "ALL_READER_PREDICTIONS.npz",
              "all_reader_predictions_sha256"),
             ("tune", "ALL_READER_TUNE_PREDICTIONS.npz",
              "all_reader_tune_predictions_sha256"))
    values = {}
    for role, name, field in files:
        saved = _private(path.parent / name)
        if sha256(saved) != index[field]:
            raise PermissionError("M5 reader probability bank SHA changed")
        with np.load(saved, allow_pickle=False) as archive:
            if set(archive.files) != expected_ids:
                raise PermissionError("M5 reader probability slate differs")
            values[role] = {key: np.array(archive[key], copy=True)
                            for key in archive.files}
    validation = {}
    tune_y = np.asarray(sex_att)[tune_indices]
    for reader in index["reader_order"]:
        tune = values["tune"][reader]
        inner = values["inner"][reader]
        if (tune.dtype != np.float64 or tune.shape != (len(tune_indices),)
                or inner.dtype != np.float64 or inner.shape != (len(inner_row),)
                or not np.isfinite(tune).all() or not np.isfinite(inner).all()
                or np.any((tune < 0) | (tune > 1))
                or np.any((inner < 0) | (inner > 1))):
            raise PermissionError("M5 reader probability dtype/shape/range differs")
        positive = binary_auc(tune_y, tune)
        expected = {"positive_alias_auc": positive,
                    "reverse_alias_auc": 1.0 - positive,
                    "fixed_alias_max_auc": max(positive, 1.0 - positive)}
        saved = index["validation"][reader]
        if (any(abs(float(saved[key]) - value) > 1e-12
                for key, value in expected.items())
                or saved.get("selected_orientation") !=
                ("positive" if positive >= 1.0 - positive else "reverse")):
            raise PermissionError("M5 ATT tune reader orientation differs")
        validation[reader] = expected
    winner = index["reader_order"][0]
    best = -1.0
    for reader in index["reader_order"]:
        score = validation[reader]["fixed_alias_max_auc"]
        if score > best + 1e-12:
            winner, best = reader, score
    if (index.get("validation_winner") != winner
            or index.get("validation_winner_orientation") !=
            index["validation"][winner]["selected_orientation"]):
        raise PermissionError("M5 ATT frozen winner differs from replayed tune values")
    return values["inner"], {
        "attack_index_sha256": index_sha,
        "core_readers_receipt_sha256": index["core_readers_receipt_sha256"],
        "inner_bank_sha256": index["all_reader_predictions_sha256"],
        "tune_bank_sha256": index["all_reader_tune_predictions_sha256"],
        "reader_ids": index["reader_order"], "validation_winner": winner,
    }


def _v2_pass(path: Path, digest: str, *, seed: int, pair_id: str, view: str,
             lock_sha: str, manifest_sha: str, release_sha: str) -> dict:
    if sha256(_private(path)) != digest:
        raise PermissionError("M5 V2 index SHA changed")
    index = json.loads(path.read_text())
    probe = index.get("probe", {})
    gates = probe.get("gates", {})
    if (index.get("schema") != "pcrl-focused-real-v2-view-index-v1"
            or index.get("status") != "PASS" or probe.get("status") != "PASS"
            or set(gates) != {"null_exact", "shuffled_independent_inner_at_most_0.60",
                              "planted_at_least_0.99"}
            or not all(gates.values()) or index.get("seed") != seed
            or index.get("pair_id") != pair_id or index.get("view_name") != view
            or index.get("lock_sha256") != lock_sha
            or index.get("manifest_sha256") != manifest_sha
            or index.get("release_index_sha256") != release_sha):
        raise PermissionError("M5 real V2 controls failed or changed")
    custody = index["probe_custody"]
    directory = _private(path.parent / "probe_custody")
    receipt = _private(directory / "CUSTODY_RECEIPT.json")
    if sha256(receipt) != custody["private_receipt_sha256"]:
        raise PermissionError("M5 V2 private probe receipt changed")
    for filename, expected in custody["files"].items():
        if Path(filename).name != filename or sha256(_private(directory / filename)) != expected:
            raise PermissionError("M5 V2 planted/shuffled artifact changed")
    return {"v2_index_sha256": digest,
            "probe_receipt_sha256": custody["private_receipt_sha256"],
            "status": "PASS"}


def run(args: argparse.Namespace) -> dict:
    """Replay one exact finite pair under M0's live child and source seal."""
    spec, lock, manifest = _pre_role_gate(args)
    lock_sha = sha256(STUDY / "CAMPAIGN_LOCK.json")
    manifest_sha = sha256(STUDY / "m1/DATA_ACCESS_MANIFEST.json")
    queue_state = json.loads((STUDY / ".private/queue/QUEUE.json").read_text())
    registry_path = STUDY / ".private/UNIT_SPECS.json"
    if sha256(registry_path) != lock.get("private_unit_registry_sha256"):
        raise PermissionError("M5 presealed potential-unit registry changed")
    registry = json.loads(registry_path.read_text())
    bank = reconstruct(STUDY / "m1/CAPACITY_SPEC.json")
    pair = next((item for item in bank.pairs if item.pair_id == args.pair_id), None)
    if pair is None:
        raise PermissionError("M5 pair is absent from five-by-five public bank")
    left, right = pair.income_mask_id, pair.occupation_mask_id
    local_union = {mask.mask_id: next(item.union_sha256 for item in bank.pairs
                  if item.income_mask_id == mask.mask_id and
                  item.occupation_mask_id == mask.mask_id) for mask in bank.masks}
    raw_by_view = {"income": local_union[left],
                   "occupation_group": local_union[right],
                   "coalition": pair.union_sha256}
    raw_ids = {f"M4-RAW-S{args.seed}-{digest[:16].upper()}"
               for digest in raw_by_view.values()}
    attack_ids = {f"M4-ATT-S{args.seed}-{args.pair_id}-{tag}"
                  for tag in VIEW_TAGS.values()}
    v2_ids = {f"M4-V2-S{args.seed}-{args.pair_id}-{tag}"
              for tag in VIEW_TAGS.values()}
    fit_id = f"M2-FIT-S{args.seed}"
    release_id = f"M4-REL-S{args.seed}-{args.pair_id}"
    expected_deps = {fit_id, release_id} | raw_ids | attack_ids | v2_ids
    if set(spec.get("deps", [])) != expected_deps or len(spec["deps"]) != len(expected_deps):
        raise PermissionError("M5 parent DAG differs from frozen pair/union geometry")
    fit_path = STUDY / ".private/m2/fit" / f"seed_{args.seed}" / "FIT_RECEIPT.json"
    fit_sha = _done_parent(spec, fit_id, fit_path, queue_state, registry)
    fit = json.loads(fit_path.read_text())
    if (fit.get("schema") != "pcrl-focused-fit-receipt-v1"
            or fit.get("status") != "FIT_COMPLETE" or fit.get("seed") != args.seed
            or fit.get("unit_id") != fit_id
            or fit.get("campaign_lock_sha256") != lock_sha
            or fit.get("data_manifest_sha256") != manifest_sha
            or fit.get("prior_data_manifest_sha256") != lock["prior_data_manifest_sha256"]
            or fit.get("bank_sha256") != bank.bank_sha256):
        raise PermissionError("M5 FIT task/privacy selection is not complete")
    release_path = (STUDY / ".private/m4/releases" / f"seed_{args.seed}" /
                    args.pair_id / "RELEASE_RECEIPT.json")
    release_sha = _done_parent(spec, release_id, release_path,
                               queue_state, registry)
    release_index = json.loads(release_path.read_text())
    expected_package = {
        "income": fit["head_packages"]["income"][left]["package_sha256"],
        "occupation_group": fit["head_packages"]["occupation_group"][right]["package_sha256"],
    }
    if (release_index.get("package_sha256_by_purpose") != expected_package
            or release_index.get("lock_sha256") != lock_sha
            or release_index.get("manifest_sha256") != manifest_sha):
        raise PermissionError("M5 release package or source identity differs from FIT")
    raw_source = {}
    for union_sha in set(raw_by_view.values()):
        uid = f"M4-RAW-S{args.seed}-{union_sha[:16].upper()}"
        path = (STUDY / ".private/m4/raw_union" / f"seed_{args.seed}" /
                union_sha / "RAW_UNION_INDEX.json")
        raw_source[union_sha] = _done_parent(spec, uid, path, queue_state, registry)
    attack_source, v2_source = {}, {}
    for view, tag in VIEW_TAGS.items():
        attack_id = f"M4-ATT-S{args.seed}-{args.pair_id}-{tag}"
        v2_id = f"M4-V2-S{args.seed}-{args.pair_id}-{tag}"
        attack_path = (STUDY / ".private/m4/attacks" / f"seed_{args.seed}" /
                       args.pair_id / view / "ATTACK_INDEX.json")
        v2_path = (STUDY / ".private/m4/v2" / f"seed_{args.seed}" /
                   args.pair_id / view / "V2_INDEX.json")
        attack_source[view] = (attack_path, _done_parent(
            spec, attack_id, attack_path, queue_state, registry))
        v2_source[view] = (v2_path, _done_parent(
            spec, v2_id, v2_path, queue_state, registry))
    if (Path(args.output_report).exists() or Path(args.output_values).exists()):
        raise PermissionError("M5 private output is not write-once")

    # All selected parent identities above are checked before a role is read.
    store = SafeRoleStore(_prior_manifest(lock, manifest))
    fit_features = store.read("OSF_DEFENSE_FIT", "features")
    fit_income = store.read("OSF_DEFENSE_FIT", "income", "defense_fit")
    fit_occupation = store.read("OSF_DEFENSE_FIT", "occupation", "defense_fit")
    fit_references = store.read("OSF_DEFENSE_FIT", f"references_s{args.seed}")
    att_features = store.read("ATTACK_FIT_NEW", "features")
    att_sex = store.read("ATTACK_FIT_NEW", "sex", "attack")
    inner_features = store.read("INNER_SELECTION", "features")
    inner_income = store.read("INNER_SELECTION", "income", "screen")
    inner_occupation = store.read("INNER_SELECTION", "occupation", "screen")
    inner_sex = store.read("INNER_SELECTION", "sex", "screen")
    inner_references = store.read("INNER_SELECTION", f"references_s{args.seed}")
    _same_people(fit_features, fit_income, fit_occupation, fit_references)
    _same_people(att_features, att_sex)
    _same_people(inner_features, inner_income, inner_occupation,
                 inner_sex, inner_references)
    fit_replay, plan = _fit_validation(
        args.seed, fit,
        {"features": fit_features, "income": fit_income,
         "occupation": fit_occupation, "references": fit_references},
        manifest, lock_sha, manifest_sha, bank)
    selected_columns = {
        "income": tuple(bank.mask(left).column_indices),
        "occupation_group": tuple(bank.mask(right).column_indices),
        "coalition": tuple(pair.union_columns),
    }
    role_feature_sha = manifest["role_file_sha256"]
    for role, source in (("ATTACK_FIT_NEW", att_features),
                         ("INNER_SELECTION", inner_features)):
        read_frozen_release(
            index_path=release_path, index_sha256=release_sha,
            private_root=STUDY / ".private", role=role, seed=args.seed,
            pair_id=args.pair_id, lock_sha256=lock_sha,
            manifest_sha256=manifest_sha,
            role_feature_sha256=role_feature_sha[f"{role}:features"],
            source_x=source["X"], source_row_id=source["row_id"],
            source_unit=source["unit"], selected_columns=selected_columns,
        )
    decoded = read_frozen_release(
        index_path=release_path, index_sha256=release_sha,
        private_root=STUDY / ".private", role="INNER_SELECTION",
        seed=args.seed, pair_id=args.pair_id, lock_sha256=lock_sha,
        manifest_sha256=manifest_sha,
        role_feature_sha256=role_feature_sha["INNER_SELECTION:features"],
        source_x=inner_features["X"], source_row_id=inner_features["row_id"],
        source_unit=inner_features["unit"], selected_columns=selected_columns,
    )
    tune_indices, split_sha = _attack_tune_indices(att_features["unit"])
    readers, reader_custody, controls = {}, {}, {}
    for view in VIEWS:
        path, attack_sha = attack_source[view]
        scores, custody = _reader_bank(
            path, attack_sha, seed=args.seed, pair_id=args.pair_id,
            view=view, lock_sha=lock_sha, manifest_sha=manifest_sha,
            release_sha=release_sha, raw_sha=raw_source[raw_by_view[view]],
            inner_row=inner_features["row_id"], inner_unit=inner_features["unit"],
            sex_att=att_sex["sex"], tune_indices=tune_indices,
            attack_split_sha=split_sha)
        readers[view], reader_custody[view] = scores, custody
        controls[view] = _v2_pass(
            *v2_source[view], seed=args.seed, pair_id=args.pair_id,
            view=view, lock_sha=lock_sha, manifest_sha=manifest_sha,
            release_sha=release_sha)
    if any(not np.array_equal(readers["coalition"][f"IGNORE_R2__{name}"],
                                  readers["income"][name]) or
           not np.array_equal(readers["coalition"][f"IGNORE_R1__{name}"],
                              readers["occupation_group"][name])
           for name in FIXED_LOCAL_READERS):
        raise PermissionError("M5 coalition transported local reader changed")
    truth = {"row_id": inner_features["row_id"], "unit": inner_features["unit"],
             "income": inner_income["income"],
             "occupation": inner_occupation["occupation"],
             "sex": inner_sex["sex"]}
    fit_labels = {"income": fit_income["income"],
                  "occupation": fit_occupation["occupation"]}
    candidate = {purpose: {
        "row_id": truth["row_id"], "unit": truth["unit"],
        "prob": decoded["candidate_probabilities"][purpose],
    } for purpose in ("income", "occupation_group")}
    references = {purpose: {} for purpose in candidate}
    names = tuple(map(str, inner_features["feature_names"]))
    for purpose, suffix in (("income", "1"), ("occupation_group", "2")):
        for ref in ("U0", "Ucal"):
            references[purpose][ref] = {
                "row_id": truth["row_id"], "unit": truth["unit"],
                "prob": np.ascontiguousarray(
                    inner_references[f"{ref}_{suffix}"], dtype=np.float64),
            }
        f_package = _head_package(
            fit["full_f_packages"][purpose], purpose=purpose,
            mask_id="F_FULL83", seed=args.seed, lock_sha=lock_sha,
            manifest_sha=manifest_sha, manifest=manifest)
        if tuple(AdultGroupedSchema().column_indices(f_package.selected_atoms)) != tuple(range(83)):
            raise PermissionError("M5 final F reference is not full83")
        references[purpose]["F"] = {
            "row_id": truth["row_id"], "unit": truth["unit"],
            "prob": np.ascontiguousarray(f_package.view(
                inner_features["X"], names)["probabilities"], dtype=np.float64),
        }
    endpoint = replay_pair_endpoint(
        seed=args.seed, pair_id=args.pair_id, truth=truth,
        fit_labels=fit_labels, candidate=candidate, references=references,
        reader_scores=readers, v2_status_by_view={view: "PASS" for view in VIEWS})
    values = {f"truth__{key}": np.asarray(value).copy()
              for key, value in truth.items()}
    for purpose in candidate:
        values[f"candidate__{purpose}"] = candidate[purpose]["prob"]
        for ref in ("U0", "Ucal", "F"):
            values[f"reference__{purpose}__{ref}"] = references[purpose][ref]["prob"]
    for view in VIEWS:
        for reader_id, score in readers[view].items():
            values[f"reader__{view}__{reader_id}"] = score
    output = Path(args.output_report).parent
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    values_sha = _write_npz_once(Path(args.output_values), values)
    report = {
        "schema": "pcrl-focused-m5-pair-replay-v1",
        "status": "REPLAYED_PAIR_PASS", "seed": args.seed,
        "pair_id": args.pair_id, "unit_id": args.unit_id,
        "lock_sha256": lock_sha, "manifest_sha256": manifest_sha,
        "fit_receipt_sha256": fit_sha, "fit_validation": fit_replay,
        "release_index_sha256": release_sha,
        "raw_index_sha256_by_union": raw_source,
        "reader_custody_by_view": reader_custody,
        "v2_custody_by_view": controls,
        "endpoint": endpoint,
        "values_sha256": values_sha,
        "values_array_sha256": {key: array_sha256(value)
                                for key, value in values.items()},
        "scope": "Independent FIT task-head numeric/selection replay, saved reader tune/INNER AUC and U0/Ucal/F endpoint replay; no FIT SEX read",
    }
    _write_json_once(Path(args.output_report), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-values", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "unit_id": args.unit_id,
                      "report_sha256": sha256(Path(args.output_report)),
                      "values_sha256": result["values_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

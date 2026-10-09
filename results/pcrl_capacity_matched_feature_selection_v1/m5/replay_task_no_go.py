"""Independent FIT-task numerical replay for a frozen pre-sensitive no-go.

The worker reads only the four admitted OSF_DEFENSE_FIT task/reference files.
It restores frozen validation packages but never imports M2/M3/M4 scoring or
selection code and never opens FIT SEX or CALIBRATION_HELDOUT arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_fit_validation import (
    array_sha256, compare_task_row, task_only_split,
)
from results.pcrl_capacity_matched_feature_selection_v1.m5.independent_metrics import task_metrics
from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_bank import reconstruct


REPO = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
ROLE = "OSF_DEFENSE_FIT"
DATA_ACCESS = "INDEPENDENT_TASK_NO_GO_ADMITTED_ROLES_ONLY"


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def audit_no_go_table(bank: object, table: dict, task_receipt: dict,
                      plan: dict, observed: dict, references: dict) -> dict:
    """Replay all ten saved task rows and the finite no-go rule, aggregate only."""
    masks = tuple(bank.masks)
    expected_ids = tuple(mask.mask_id for mask in masks)
    if (len(masks) != 5 or len(set(expected_ids)) != 5
            or table.get("schema") != "pcrl-focused-task-table-v1"
            or table.get("bank_sha256") != bank.bank_sha256
            or table.get("task_table_sha256") != digest_json({
                key: value for key, value in table.items() if key != "task_table_sha256"})):
        raise ValueError("task table or five-mask bank differs")
    if set(observed) != set(TASKS) or set(references) != set(TASKS):
        raise ValueError("both independent task replay sources are required")
    counts = {}
    for purpose, field in (("income", "income_masks"),
                           ("occupation_group", "occupation_masks")):
        rows = table.get(field)
        if (not isinstance(rows, list) or len(rows) != 5
                or tuple(row.get("mask_id") for row in rows) != expected_ids
                or set(observed[purpose]) != set(expected_ids)
                or set(references[purpose]) != set(REFERENCES)):
            raise ValueError("task mask rows or reference coverage differs")
        feasible = 0
        for mask, saved in zip(masks, rows, strict=True):
            if saved.get("mask_sha256") != mask.mask_sha256:
                raise ValueError("task row mask identity differs")
            compare_task_row(saved, observed[purpose][mask.mask_id],
                             references[purpose])
            feasible += int(saved["utility_feasible"])
        counts[purpose] = feasible
    if all(counts.values()):
        raise ValueError("task-only no-go contradicts independently feasible pair")
    selection = task_receipt.get("selection")
    expected_reason = ("no utility-feasible task-only income mask" if
                       counts["income"] == 0 else
                       "no utility-feasible task-only occupation mask")
    if (not isinstance(selection, dict)
            or task_receipt.get("bank_sha256") != bank.bank_sha256
            or selection.get("bank_sha256") != bank.bank_sha256
            or selection.get("task_table_sha256") != table["task_table_sha256"]
            or selection.get("status") != "NO_UTILITY_FEASIBLE_T"
            or selection.get("reason") != expected_reason
            or selection.get("pair_id") is not None):
        raise ValueError("task-only T no-go selection differs")
    if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
            or plan.get("bank_sha256") != bank.bank_sha256
            or plan.get("task_table_sha256") != table["task_table_sha256"]
            or plan.get("selection_receipt_sha256") != digest_json({
                key: value for key, value in plan.items()
                if key != "selection_receipt_sha256"})
            or plan.get("status") != "NO_FEASIBLE_J"
            or plan.get("reason") != "NO_UTILITY_FEASIBLE_TASK_MASK"
            or plan.get("joint_pair_id") is not None
            or plan.get("task_only_pair_id") is not None
            or plan.get("table_sha256") is not None):
        raise ValueError("control no-go plan identity differs")
    if (set(plan.get("rules", {})) != set(RULES)
            or any(row.get("status") != "INVALID_NO_UTILITY_FEASIBLE_PAIR"
                   or row.get("reason") != "NO_UTILITY_FEASIBLE_TASK_MASK"
                   or row.get("pair_id") is not None
                   for row in plan["rules"].values())
            or plan.get("dropped_invalid_rules") != list(RULES)
            or plan.get("main_control_order") != list(RULES[1:])
            or plan.get("attribution_control_order") != list(RULES[6:]) + list(RULES[1:6])
            or plan.get("joint_union_quota") is not None
            or plan.get("release_alias_status") != "NOT_APPLICABLE_NO_SELECTED_RELEASE"):
        raise ValueError("control rules are not all invalid under task no-go")
    return {"status": "INDEPENDENT_TASK_NO_GO_PASS",
            "mask_row_count": 10,
            "feasible_mask_count_by_task": counts,
            "task_only_selection": selection["status"],
            "control_rule_count_invalid": len(RULES)}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _under_private(path: Path, *, must_exist: bool = True) -> Path:
    private = (STUDY / ".private").resolve(strict=True)
    resolved = Path(path).resolve(strict=must_exist)
    if not resolved.is_relative_to(private):
        raise PermissionError("M5 no-go artifact escaped private custody")
    return resolved


def _relative_private(relative: str, expected: Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or (STUDY / ".private" / candidate).resolve() != expected.resolve():
        raise PermissionError("M5 no-go private artifact path differs")
    return _under_private(expected)


def _same_people(*records: dict) -> None:
    first = records[0]
    row, unit = np.asarray(first["row_id"]), np.asarray(first["unit"])
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or len(np.unique(row)) != len(row)):
        raise PermissionError("M5 FIT ordered person/group identity invalid")
    for record in records[1:]:
        if (not np.array_equal(np.asarray(record["row_id"]), row)
                or not np.array_equal(np.asarray(record["unit"]), unit)):
            raise PermissionError("M5 FIT role files differ in ordered people")


def _pre_role_gate(args: argparse.Namespace) -> tuple[dict, dict, dict, str]:
    """Authenticate the verification generation and old FIT output, no role read."""
    from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit

    if type(args.seed) is not int or args.seed not in (0, 1):
        raise PermissionError("M5 no-go seed outside completed screen")
    unit_id = f"M5-TASK-NOGO-REPLAY-S{args.seed}"
    queue_root = STUDY / ".private/queue"
    output = STUDY / ".private/m5/task_no_go" / f"seed_{args.seed}" / "REPORT.json"
    if (args.unit_id != unit_id or Path(args.queue_root).resolve() != queue_root.resolve()
            or Path(args.output_report).resolve() != output.resolve()
            or any(os.environ.get(key) != "1" for key in THREAD_KEYS)):
        raise PermissionError("M5 no-go process, queue, output or thread scope differs")
    spec = assert_running_unit(unit_id, queue_root, STUDY / "CAMPAIGN_LOCK.json",
                               "independent_task_no_go", output)
    keys = {f"{ROLE}:features", f"{ROLE}:income", f"{ROLE}:occupation",
            f"{ROLE}:references_s{args.seed}"}
    if (spec.get("role_access") != [ROLE] or set(spec.get("role_inputs", {})) != keys
            or spec.get("data_access") != DATA_ACCESS or spec.get("deps") != []
            or spec.get("outputs") != [str(output)]
            or spec.get("allow_child_processes") is not False):
        raise PermissionError("M5 no-go sealed role or unit scope differs")
    lock_path = STUDY / "CAMPAIGN_LOCK.json"
    lock = json.loads(lock_path.read_text())
    manifest_path = STUDY / "m1/DATA_ACCESS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if (lock.get("schema") != "pcrl-focused-postfit-verification-lock-v1"
            or lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("verification_only") is not True
            or lock.get("adult_fit_permission") is not False
            or _sha(manifest_path) != lock.get("source_map_sha256")
            or spec.get("source_map_sha256") != lock["source_map_sha256"]):
        raise PermissionError("M5 verification-only lock or manifest differs")
    source = lock.get("historical_fit")
    if not isinstance(source, dict):
        raise PermissionError("M5 historical FIT admission absent")
    prior_root = (REPO / lock["prior_package_relative_to_worktree"]).resolve(strict=True)
    expected_prior = (REPO.parent / "pcrl-method-portfolio-v1/results/"
                      "pcrl_method_portfolio_v1").resolve(strict=True)
    prior_manifest = prior_root / "DATA_ACCESS_MANIFEST.json"
    if (prior_root != expected_prior or _sha(prior_manifest) !=
            lock["prior_data_manifest_sha256"] or
            manifest.get("portfolio_manifest_sha256") != _sha(prior_manifest)):
        raise PermissionError("M5 no-go prior role package differs")
    for key in keys:
        source_path = (prior_root / manifest["role_file_relative_path"][key]).resolve(strict=True)
        declared = spec["role_inputs"][key]
        if (not source_path.is_relative_to(prior_root)
                or declared != {"path": str(source_path),
                                 "sha256": manifest["role_file_sha256"][key]}):
            raise PermissionError("M5 no-go FIT role file differs")
    fit_receipt_sha = _historical_fit_receipt(source, args.seed)
    if output.exists():
        raise PermissionError("M5 no-go report is not write-once")
    return spec, lock, manifest, fit_receipt_sha


def _historical_fit_receipt(source: dict, seed: int) -> str:
    """Verify archived queue, spec, old lock/seal and immutable FIT receipt."""
    archive_rel = Path(source["archived_queue_relative_path"])
    old_lock_rel = Path(source["archived_lock_relative_path"])
    old_seal_rel = Path(source["archived_seal_relative_path"])
    old_review_rel = Path(source["archived_review_relative_path"])
    if any(path.is_absolute() for path in
           (archive_rel, old_lock_rel, old_seal_rel, old_review_rel)):
        raise PermissionError("M5 historical FIT path is absolute")
    archive_path = _under_private(STUDY / archive_rel)
    archive = archive_path.parent
    if archive_path.name != "QUEUE.json" or archive.name == "queue":
        raise PermissionError("M5 historical queue is current or absent")
    old_lock = (STUDY / old_lock_rel).resolve(strict=True)
    old_seal = (STUDY / old_seal_rel).resolve(strict=True)
    old_review = (STUDY / old_review_rel).resolve(strict=True)
    if (not old_lock.is_relative_to(STUDY / "amendments")
            or not old_seal.is_relative_to(STUDY / "amendments")
            or not old_review.is_relative_to(STUDY / "amendments")
            or _sha(old_lock) != source["lock_sha256"]
            or _sha(old_seal) != source["local_seal_sha256"]
            or _sha(archive_path) != source["archived_queue_sha256"]):
        raise PermissionError("M5 historical FIT lock, seal or queue SHA differs")
    archived_seal = json.loads(old_seal.read_text())
    if (_sha(old_review) != source["archived_review_sha256"]
            or archived_seal.get("m5_review_sha256") !=
            source["archived_review_sha256"]
            or json.loads(old_review.read_text()).get("status") != "PASS"):
        raise PermissionError("M5 historical FIT review differs")
    if (archived_seal.get("lock_sha256") != source["lock_sha256"]
            or archived_seal.get("status") != "PASS"):
        raise PermissionError("M5 historical FIT seal is not PASS")
    old = json.loads(archive_path.read_text())
    uid = f"M2-FIT-S{seed}"
    row = old.get("units", {}).get(uid)
    entry = source.get("by_seed", {}).get(str(seed))
    if (not isinstance(row, dict) or not isinstance(entry, dict)
            or row.get("status") != "DONE" or row.get("exit_code") != 0
            or row.get("output_sha256") != {
                str(STUDY / ".private/m2/fit" / f"seed_{seed}" / "FIT_RECEIPT.json"):
                entry.get("fit_receipt_sha256")}
            or row.get("spec_sha256") != entry.get("spec_sha256")):
        raise PermissionError("M5 historical FIT unit did not complete exactly")
    spec_path = archive / "specs" / f"{uid}.json"
    if _sha(spec_path) != entry["spec_sha256"]:
        raise PermissionError("M5 historical FIT unit spec changed")
    spec = json.loads(spec_path.read_text())
    if (spec.get("unit_id") != uid or spec.get("stage") != "fit_bank"
            or spec.get("seal_sha256") != source["local_seal_sha256"]):
        raise PermissionError("M5 historical FIT stage or seal differs")
    fit_path = _under_private(STUDY / ".private/m2/fit" / f"seed_{seed}" / "FIT_RECEIPT.json")
    if _sha(fit_path) != entry["fit_receipt_sha256"]:
        raise PermissionError("M5 historical FIT output changed")
    return entry["fit_receipt_sha256"]


def _head_package(record: dict, *, purpose: str, mask_id: str, seed: int,
                  source_lock_sha: str, manifest_sha: str, manifest: dict):
    """Restore a hash-bound generic validation package without fitting."""
    for directory in (REPO, REPO / "results/pcrl_method_portfolio_v1/shared",
                      REPO / "results/pcrl_method_portfolio_v1/teams/c"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from common_eval import apply_global_temperature, load_calibrator
    from release_package import restore_release_package

    cache_key = record["cache_key"]
    root = STUDY / ".private"
    expected_dir = root / source_lock_sha / "head_pool" / cache_key
    package_path = _relative_private(
        record["relative_package_path"], expected_dir / "release_package.json")
    receipt_path = _relative_private(
        record["head_pool_receipt_relative_path"],
        expected_dir / "HEAD_POOL_RECEIPT.json")
    if (record.get("stage") != "validation" or record.get("purpose") != purpose
            or record.get("mask_id") != mask_id
            or record.get("receipt_sha256") != record.get("head_pool_receipt_sha256")
            or _sha(package_path) != record["package_sha256"]
            or _sha(receipt_path) != record["receipt_sha256"]):
        raise PermissionError("M5 no-go validation package or receipt differs")
    receipt = json.loads(receipt_path.read_text())
    identity = receipt["cache_identity"]
    suffix = "occupation" if purpose == "occupation_group" else "income"
    roles = manifest["role_file_sha256"]
    source_hashes = {
        "fit_features": roles[f"{ROLE}:features"],
        "fit_label": roles[f"{ROLE}:{suffix}"],
        "calibration_features": roles["CALIBRATION_HELDOUT:features"],
        "calibration_label": roles[f"CALIBRATION_HELDOUT:{suffix}"],
    }
    if (identity.get("cache_key") != cache_key
            or identity.get("mask_id") != mask_id
            or identity.get("purpose") != purpose or identity.get("seed") != seed
            or identity.get("model_id") != record["model_id"]
            or identity.get("campaign_lock_sha256") != source_lock_sha
            or identity.get("data_manifest_sha256") != manifest_sha
            or identity.get("role_file_sha256") != source_hashes
            or receipt.get("package_sha256") != record["package_sha256"]
            or receipt.get("head_sha256") != record["head_sha256"]
            or receipt.get("calibrator_sha256") != record["calibrator_sha256"]):
        raise PermissionError("M5 no-go validation cache identity differs")
    return restore_release_package(
        package_path, expected_sha256=record["package_sha256"],
        private_root=root, expected_campaign_lock_sha256=source_lock_sha,
        expected_data_manifest_sha256=manifest_sha,
        expected_source_role_hashes=source_hashes,
        load_calibrator=load_calibrator,
        apply_global_temperature=apply_global_temperature,
    )


def _metric(labels: np.ndarray, probability: np.ndarray, width: int) -> dict:
    scored = task_metrics(labels, probability, width=width)
    return {key: float(scored[key]) for key in ("accuracy", "nll", "brier")}


def _load_no_go_artifacts(seed: int, lock: dict, fit_sha: str,
                          bank: object) -> tuple[dict, dict, dict, dict]:
    folder = STUDY / ".private/m2/fit" / f"seed_{seed}"
    fit_path = _under_private(folder / "FIT_RECEIPT.json")
    task_path = _under_private(folder / "TASK_ONLY_RECEIPT.json")
    table_path = _under_private(folder / "TASK_TABLE.json")
    plan_path = _under_private(folder / "CONTROL_PLAN.json")
    fit = json.loads(fit_path.read_text())
    task = json.loads(task_path.read_text())
    table = json.loads(table_path.read_text())
    plan = json.loads(plan_path.read_text())
    source_lock_sha = lock["historical_fit"]["lock_sha256"]
    manifest_sha = lock["source_map_sha256"]
    registry = task.get("validation_head_packages")
    if (_sha(fit_path) != fit_sha
            or fit.get("schema") != "pcrl-focused-fit-receipt-v1"
            or fit.get("status") != "NO_FEASIBLE_J"
            or fit.get("reason") != "NO_UTILITY_FEASIBLE_TASK_MASK"
            or fit.get("seed") != seed or fit.get("unit_id") != f"M2-FIT-S{seed}"
            or fit.get("campaign_lock_sha256") != source_lock_sha
            or fit.get("data_manifest_sha256") != manifest_sha
            or fit.get("bank_sha256") != bank.bank_sha256
            or fit.get("task_table_relative_path") != f"m2/fit/seed_{seed}/TASK_TABLE.json"
            or fit.get("control_plan_relative_path") != f"m2/fit/seed_{seed}/CONTROL_PLAN.json"
            or fit.get("task_table_sha256") != _sha(table_path)
            or fit.get("task_only_receipt_sha256") != _sha(task_path)
            or fit.get("control_plan_sha256") != _sha(plan_path)
            or fit.get("pair_table_relative_path") is not None
            or fit.get("pair_table_sha256") is not None
            or fit.get("joint_selection_relative_path") is not None
            or fit.get("joint_selection_sha256") is not None
            or fit.get("head_packages") != {}
            or fit.get("full_f_packages") != {}
            or fit.get("critic_spec") is not None
            or fit.get("search_receipt") != {"status": "NOT_RUN_TASK_NO_GO"}
            or task.get("schema") != "pcrl-focused-task-only-receipt-v1"
            or task.get("seed") != seed
            or task.get("unit_id") != f"M2-FIT-S{seed}"
            or task.get("campaign_lock_sha256") != source_lock_sha
            or task.get("task_table_sha256") != _sha(table_path)
            or task.get("bank_sha256") != bank.bank_sha256
            or task.get("split_receipt") != fit.get("task_split_receipt")
            or task.get("validation_head_packages_sha256") != digest_json(registry)
            or fit.get("validation_head_packages_sha256") != digest_json(registry)
            or fit.get("validation_head_packages") != registry):
        raise PermissionError("M5 FIT task no-go source custody differs")
    expected_masks = [{"mask_id": row.mask_id, "mask_sha256": row.mask_sha256}
                      for row in bank.masks]
    expected_ids = {"F_FULL83", *(row.mask_id for row in bank.masks)}
    if (task.get("bank_masks") != expected_masks
            or set(registry) != set(TASKS)
            or any(set(registry[purpose]) != expected_ids for purpose in TASKS)):
        raise PermissionError("M5 FIT 12-package bank coverage differs")
    for purpose, field in (("income", "income_masks"),
                           ("occupation_group", "occupation_masks")):
        for row in table[field]:
            model_id = registry[purpose][row["mask_id"]]["model_id"]
            if row.get("head_id") != model_id:
                raise PermissionError("M5 task row and frozen validation head differ")
    return fit, task, table, plan


def run(args: argparse.Namespace) -> dict:
    _, lock, manifest, fit_sha = _pre_role_gate(args)
    seed = args.seed
    bank_path = STUDY / "m1/CAPACITY_SPEC.json"
    bank = reconstruct(bank_path)
    fit, task, table, plan = _load_no_go_artifacts(seed, lock, fit_sha, bank)
    manifest_sha = lock["source_map_sha256"]
    source_lock_sha = lock["historical_fit"]["lock_sha256"]
    for directory in (REPO, REPO / "results/pcrl_method_portfolio_v1/shared",
                      REPO / "results/pcrl_method_portfolio_v1/teams/c"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from prototype import AdultGroupedSchema
    from safe_data_guard import SafeRoleStore

    prior = (REPO / lock["prior_package_relative_to_worktree"]).resolve(strict=True)
    store = SafeRoleStore(prior / "DATA_ACCESS_MANIFEST.json")
    features = store.read(ROLE, "features")
    income = store.read(ROLE, "income", "defense_fit")
    occupation = store.read(ROLE, "occupation", "defense_fit")
    source_references = store.read(ROLE, f"references_s{seed}")
    _same_people(features, income, occupation, source_references)
    row = np.asarray(features["row_id"])
    unit = np.asarray(features["unit"])
    x = np.asarray(features["X"])
    names = tuple(map(str, features["feature_names"]))
    if (x.dtype != np.float32 or x.shape != (len(row), 83)
            or len(names) != 83 or not np.isfinite(x).all()
            or array_sha256(np.column_stack((row, unit)).astype(np.int64)) !=
            fit["fit_row_unit_sha256"]):
        raise PermissionError("M5 FIT source wire or row/group receipt differs")
    labels = {"income": np.asarray(income["income"]),
              "occupation_group": np.asarray(occupation["occupation"])}
    if any(value.dtype != np.int64 for value in labels.values()):
        raise PermissionError("M5 FIT task labels are not exact int64")
    _, validation_indices, split = task_only_split(
        unit, labels["income"], labels["occupation_group"], seed)
    if split != fit["task_split_receipt"] or split != table.get("split_receipt"):
        raise PermissionError("M5 independent FIT split differs")
    observed, reference_metrics = {}, {}
    registry = task["validation_head_packages"]
    schema = AdultGroupedSchema()
    for purpose, label, width, suffix in (("income", "income", 2, "1"),
                                           ("occupation_group", "occupation", 6, "2")):
        y = labels[purpose][validation_indices]
        reference_metrics[purpose] = {}
        for name in ("U0", "Ucal"):
            probability = np.asarray(source_references[f"{name}_{suffix}"])
            if probability.shape != (len(row), width):
                raise PermissionError("M5 FIT saved reference shape differs")
            reference_metrics[purpose][name] = _metric(
                y, probability[validation_indices], width)
        full = _head_package(registry[purpose]["F_FULL83"], purpose=purpose,
                             mask_id="F_FULL83", seed=seed,
                             source_lock_sha=source_lock_sha,
                             manifest_sha=manifest_sha, manifest=manifest)
        if tuple(schema.column_indices(full.selected_atoms)) != tuple(range(83)):
            raise PermissionError("M5 FIT F package does not have full83 source")
        prediction = np.asarray(full.view(
            np.ascontiguousarray(x[validation_indices]), names)["probabilities"])
        reference_metrics[purpose]["F"] = _metric(y, prediction, width)
        observed[purpose] = {}
        for mask in bank.masks:
            record = registry[purpose][mask.mask_id]
            package = _head_package(record, purpose=purpose, mask_id=mask.mask_id,
                                    seed=seed, source_lock_sha=source_lock_sha,
                                    manifest_sha=manifest_sha, manifest=manifest)
            if tuple(package.selected_atoms) != tuple(mask.atoms):
                raise PermissionError("M5 validation package selected atoms differ")
            probability = np.asarray(package.view(
                np.ascontiguousarray(x[validation_indices]), names)["probabilities"])
            observed[purpose][mask.mask_id] = _metric(y, probability, width)
    decision = audit_no_go_table(bank, table, task, plan, observed,
                                 reference_metrics)
    report = {
        "schema": "pcrl-focused-m5-task-no-go-replay-v1",
        "status": decision["status"],
        "seed": seed, "unit_id": args.unit_id,
        "source_fit_lock_sha256": source_lock_sha,
        "source_fit_receipt_sha256": fit_sha,
        "verification_lock_sha256": _sha(STUDY / "CAMPAIGN_LOCK.json"),
        "data_access_manifest_sha256": manifest_sha,
        "bank_sha256": bank.bank_sha256,
        "task_table_sha256": fit["task_table_sha256"],
        "task_only_receipt_sha256": fit["task_only_receipt_sha256"],
        "control_plan_sha256": fit["control_plan_sha256"],
        "fit_split_receipt": split,
        "head_package_replay_count": 12,
        "independent_reference_metrics_by_task": reference_metrics,
        "independent_mask_metrics_by_task": observed,
        "feasible_mask_count_by_task": decision["feasible_mask_count_by_task"],
        "independent_task_only_status": decision["task_only_selection"],
        "invalid_control_rule_count": decision["control_rule_count_invalid"],
        "fit_sex_read": False,
        "calibration_array_read": False,
        "interpretation": "No utility-feasible pair under frozen FIT-validation caps; no J/T or privacy critic was run. Development-only result.",
    }
    output = STUDY / ".private/m5/task_no_go" / f"seed_{seed}" / "REPORT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o600)
    fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"status": report["status"], "report_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()

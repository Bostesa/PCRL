"""Sealed M2 FIT worker: one shared five-mask bank and full-F reference per seed.

This script opens only admitted OSF_DEFENSE_FIT and CALIBRATION_HELDOUT role
exports, after the M0 live S3 queue gate. It never reads INNER_SELECTION,
ATTACK_FIT_NEW, HEAD_VALIDATION, or old assessment arrays. The task-only T
receipt is durably published before the first protected-label read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from bank import build_shared_bank
from fit_pipeline import _array_sha, run_fit_pipeline
from head_pool import (_file_sha, _sha, _write_once_json, fit_or_restore_task_package,
                       make_head_cache_identity)


FIT_ROLE = "OSF_DEFENSE_FIT"
CAL_ROLE = "CALIBRATION_HELDOUT"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
CRITIC_SPEC = {
    "schema": "pcrl-focused-raw-search-critic-v1",
    "readers": [
        {"name": "logistic", "C": 1.0, "max_iter": 300,
         "solver": "lbfgs", "scaler": "StandardScaler"},
        {"name": "hist_gradient_boosting", "max_iter": 64,
         "max_leaf_nodes": 15, "learning_rate": 0.05,
         "max_depth": None, "random_state": "model_seed"},
    ],
    "orientation": "max(AUC,1-AUC) per reader; then max readers",
    "fit_role": FIT_ROLE,
    "validation_role": "OSF_DEFENSE_FIT/internal_group_disjoint",
    "local_input_kind": "RAW_SELECTED_ONLY",
    "coalition_input_kind": "RAW_UNION_ONLY",
}


def _exact_role_keys(seed: int) -> set[str]:
    return {
        f"{FIT_ROLE}:features", f"{FIT_ROLE}:income",
        f"{FIT_ROLE}:occupation", f"{FIT_ROLE}:references_s{seed}",
        f"{FIT_ROLE}:sex", f"{CAL_ROLE}:features",
        f"{CAL_ROLE}:income", f"{CAL_ROLE}:occupation",
    }


def _check_identical_people(*records: dict) -> None:
    if not records:
        raise ValueError("role identity comparison needs records")
    first = records[0]
    row, unit = np.asarray(first["row_id"]), np.asarray(first["unit"])
    if (row.dtype != np.int64 or unit.dtype != np.int64
            or row.ndim != 1 or unit.shape != row.shape
            or len(np.unique(row)) != len(row)):
        raise ValueError("role row/group identity is malformed")
    for record in records[1:]:
        if (not np.array_equal(record["row_id"], row)
                or not np.array_equal(record["unit"], unit)):
            raise ValueError("role files differ in ordered row/group identity")


def _search_auc_factory(x: np.ndarray, seed: int):
    """Return a fixed two-reader, reversible-AUC raw-feature search proxy."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def score(columns, fit_rows, validation_rows, sex):
        cols = tuple(columns)
        xx_fit = np.ascontiguousarray(x[np.ix_(fit_rows, cols)], dtype=np.float64)
        xx_val = np.ascontiguousarray(x[np.ix_(validation_rows, cols)], dtype=np.float64)
        readers = (
            make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=300, solver="lbfgs", random_state=seed)),
            HistGradientBoostingClassifier(
                max_iter=64, max_leaf_nodes=15, learning_rate=0.05,
                max_depth=None, random_state=seed),
        )
        values = []
        for model in readers:
            model.fit(xx_fit, sex[fit_rows])
            probability = model.predict_proba(xx_val)[:, 1]
            auc = float(roc_auc_score(sex[validation_rows], probability))
            values.append(max(auc, 1.0 - auc))
        return max(values)

    return score


def _source_hashes(study: Path, lock: dict) -> dict[str, str]:
    repo = study.parents[1]
    governed = lock.get("governed_source_sha256_by_path", {})
    paths = {
        "head_pool.py": study / "m2" / "head_pool.py",
        **{name: repo / "results/pcrl_method_portfolio_v1/teams/c" / name
           for name in ("training.py", "prototype.py", "scorecard_builder.py",
                        "release_package.py")},
        **{name: repo / "results/pcrl_method_portfolio_v1/shared" / name
           for name in ("common_eval.py", "safe_data_guard.py")},
    }
    out = {}
    for name, path in paths.items():
        relative = path.relative_to(repo).as_posix()
        actual = _file_sha(path)
        if governed.get(relative) != actual:
            raise PermissionError(f"head source {name} differs from effective lock")
        out[name] = actual
    return out


def _resolve_prior_package(repo: Path, lock: dict,
                           focused_manifest: dict) -> Path:
    """Allow only the lock-pinned package in this or a sibling worktree."""
    relative = Path(lock.get("prior_package_relative_to_worktree", ""))
    if relative.is_absolute() or not relative.parts:
        raise PermissionError("prior package must be worktree relative")
    prior = (repo / relative).resolve()
    if (not prior.is_dir() or not prior.is_relative_to(repo.parent)
            or prior.name != "pcrl_method_portfolio_v1"
            or focused_manifest.get("prior_package_id") != prior.name):
        raise PermissionError("prior package escapes the pinned sibling worktree scope")
    return prior


def _manifest_identities(study: Path, prior_manifest: Path,
                         lock: dict) -> tuple[str, str]:
    """Keep focused admission and inherited loader manifests distinct."""
    focused_path = study / "m1" / "DATA_ACCESS_MANIFEST.json"
    focused_sha = _file_sha(focused_path)
    prior_sha = _file_sha(prior_manifest)
    if (focused_sha != lock.get("source_map_sha256")
            or prior_sha != lock.get("prior_data_manifest_sha256")
            or focused_sha == prior_sha):
        raise PermissionError("focused or inherited role manifest differs from lock")
    return focused_sha, prior_sha


def _role_file_hashes(manifest: dict, purpose: str) -> dict[str, str]:
    label = "occupation" if purpose == "occupation_group" else "income"
    specs = manifest["real_file_specs"]
    return {
        "fit_features": specs[f"{FIT_ROLE}:features"]["sha256"],
        "fit_label": specs[f"{FIT_ROLE}:{label}"]["sha256"],
        "calibration_features": specs[f"{CAL_ROLE}:features"]["sha256"],
        "calibration_label": specs[f"{CAL_ROLE}:{label}"]["sha256"],
    }


def _validate_validation_packages(bank, table: dict, packages: dict,
                                  lock_sha: str) -> None:
    """Require all twelve hash-bound validation heads before first SEX read."""
    ids = {"F_FULL83", *(mask.mask_id for mask in bank.masks)}
    if not isinstance(packages, dict) or set(packages) != {"income", "occupation_group"}:
        raise ValueError("validation package registry must have both purposes")
    seen_cache_keys = set()
    for purpose, table_key in (("income", "income_masks"),
                               ("occupation_group", "occupation_masks")):
        entries = packages[purpose]
        if not isinstance(entries, dict) or set(entries) != ids:
            raise ValueError("validation package registry must have six heads per purpose")
        table_heads = {row.get("mask_id"): row.get("head_id")
                       for row in table[table_key]}
        for mask_id, entry in entries.items():
            if (not isinstance(entry, dict)
                    or entry.get("stage") != "validation"
                    or entry.get("purpose") != purpose
                    or entry.get("mask_id") != mask_id
                    or not isinstance(entry.get("model_id"), str)
                    or not entry["model_id"]
                    or (mask_id != "F_FULL83"
                        and table_heads.get(mask_id) != entry["model_id"])):
                raise ValueError("validation package identity differs from task table")
            for key in ("cache_key", "head_sha256", "calibrator_sha256",
                        "package_sha256", "receipt_sha256",
                        "head_pool_receipt_sha256"):
                value = entry.get(key)
                if (not isinstance(value, str) or len(value) != 64
                        or any(ch not in "0123456789abcdef" for ch in value)):
                    raise ValueError("validation package SHA is malformed")
            if entry["receipt_sha256"] != entry["head_pool_receipt_sha256"]:
                raise ValueError("validation package HEAD_POOL_RECEIPT SHA differs")
            if entry["cache_key"] in seen_cache_keys:
                raise ValueError("validation package cache key is reused")
            seen_cache_keys.add(entry["cache_key"])
            package_path = Path(entry.get("relative_package_path", ""))
            receipt_path = Path(entry.get("head_pool_receipt_relative_path", ""))
            if (package_path.is_absolute() or ".." in package_path.parts
                    or package_path.parts !=
                    (lock_sha, "head_pool", entry["cache_key"], "release_package.json")
                    or receipt_path.is_absolute()
                    or ".." in receipt_path.parts
                    or receipt_path != package_path.parent / "HEAD_POOL_RECEIPT.json"):
                raise ValueError("validation package private path is malformed")


def _validation_receipt_fields(output_dir: Path, packages: dict) -> dict:
    """Recheck the pre-SEX file before binding it into either FIT outcome."""
    saved = json.loads((output_dir / "TASK_ONLY_RECEIPT.json").read_text())
    digest = _sha(packages)
    if (saved.get("validation_head_packages") != packages
            or saved.get("validation_head_packages_sha256") != digest):
        raise RuntimeError("validation package registry differs from pre-SEX receipt")
    return {"validation_head_packages": packages,
            "validation_head_packages_sha256": digest}


def _publish_task_only(output_dir: Path, unit_id: str, seed: int,
                       lock_sha: str, bank, table: dict, choice: dict,
                       validation_packages: dict) -> str:
    """Write and rehash the full pre-sensitive T chain before any SEX read."""
    expected = [{"mask_id": mask.mask_id, "mask_sha256": mask.mask_sha256}
                for mask in bank.masks]
    chosen = choice.get("pair_id") in {pair.pair_id for pair in bank.pairs}
    no_go = (choice.get("status") == "NO_UTILITY_FEASIBLE_T"
             and choice.get("pair_id") is None
             and str(choice.get("reason", "")).startswith("no utility-feasible task-only ")
             and choice.get("task_table_sha256") == table.get("task_table_sha256")
             and choice.get("bank_sha256") == bank.bank_sha256)
    if (table.get("bank_sha256") != bank.bank_sha256
            or not (chosen or no_go)
            or any([{"mask_id": row.get("mask_id"),
                     "mask_sha256": row.get("mask_sha256")}
                    for row in table.get(key, [])] != expected
                   for key in ("income_masks", "occupation_masks"))):
        raise ValueError("task-only table must bind all five mask IDs and SHAs")
    _validate_validation_packages(bank, table, validation_packages, lock_sha)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    table_path = output_dir / "TASK_TABLE.json"
    table_sha = _write_once_json(table_path, table)
    if table_sha != _file_sha(table_path):
        raise RuntimeError("task-only table publication failed")
    receipt = {
        "schema": "pcrl-focused-task-only-receipt-v1",
        "unit_id": unit_id, "seed": seed,
        "campaign_lock_sha256": lock_sha,
        "bank_sha256": bank.bank_sha256,
        "bank_masks": expected,
        "task_table_sha256": table_sha,
        "selection": choice,
        "split_receipt": table["split_receipt"],
        "validation_head_packages": validation_packages,
        "validation_head_packages_sha256": _sha(validation_packages),
    }
    receipt_path = output_dir / "TASK_ONLY_RECEIPT.json"
    receipt_sha = _write_once_json(receipt_path, receipt)
    if receipt_sha != _file_sha(receipt_path):
        raise RuntimeError("task-only choice publication failed")
    return table_sha


def _publish_pair_selection(output_dir: Path, bank, table,
                            task_choice: dict) -> dict:
    """Freeze J and every M3 matched rule from the single immutable table."""
    from joint import select_joint
    from results.pcrl_capacity_matched_feature_selection_v1.m3.control_rules import (
        build_control_plan, select_task_only,
    )

    if (table.bank_sha256 != bank.bank_sha256
            or select_task_only(table).pair_id != task_choice.get("pair_id")):
        raise ValueError("full-table T differs from the pre-sensitive task-only choice")
    joint = select_joint(table, t_pair_id=task_choice["pair_id"])
    plan = build_control_plan(table, bank, joint.pair_id)
    if (plan["task_only_pair_id"] != task_choice["pair_id"]
            or plan["joint_pair_id"] != joint.pair_id
            or len(plan["rules"]) != 10):
        raise ValueError("matched control plan differs from frozen J/T choices")
    records = (
        ("PAIR_TABLE.json", table.to_payload(), "pair_table_sha256"),
        ("JOINT_SELECTION.json", joint.to_payload(), "joint_selection_sha256"),
        ("CONTROL_PLAN.json", plan, "control_plan_sha256"),
    )
    result = {"joint_pair_id": joint.pair_id}
    for name, payload, key in records:
        path = output_dir / name
        expected = _write_once_json(path, payload)
        if _file_sha(path) != expected:
            raise RuntimeError(f"{name} publication failed")
        result[key] = expected
        result[key.replace("_sha256", "_relative_path")] = name
    return result


def _publish_pre_sensitive_no_go(output_dir: Path, bank,
                                 task_table: dict, choice: dict) -> dict:
    """Publish ten invalid rule slots from task evidence, without SEX/critics."""
    from results.pcrl_capacity_matched_feature_selection_v1.m3.task_only import (
        build_pre_sensitive_no_go_control_plan,
    )

    if (choice.get("status") != "NO_UTILITY_FEASIBLE_T"
            or choice.get("pair_id") is not None
            or choice.get("task_table_sha256") != task_table.get("task_table_sha256")
            or choice.get("bank_sha256") != bank.bank_sha256):
        raise ValueError("pre-sensitive no-go choice differs from task table")
    plan = build_pre_sensitive_no_go_control_plan(
        task_table, bank, reason="NO_UTILITY_FEASIBLE_TASK_MASK")
    path = output_dir / "CONTROL_PLAN.json"
    expected = _write_once_json(path, plan)
    if _file_sha(path) != expected:
        raise RuntimeError("pre-sensitive no-go plan publication failed")
    return {"status": "NO_FEASIBLE_J", "reason": plan["reason"],
            "control_plan_sha256": expected,
            "control_plan_relative_path": path.name}


def run_worker(*, unit_id: str, seed: int, lock_path: Path,
               queue_root: Path, private_root: Path) -> str:
    from results.pcrl_capacity_matched_feature_selection_v1.m0.admission import assert_running_unit

    study = Path(__file__).resolve().parents[1]
    expected_private = study / ".private"
    if (unit_id != f"M2-FIT-S{seed}" or seed not in (0, 1, 2)
            or lock_path.resolve() != study / "CAMPAIGN_LOCK.json"
            or private_root.resolve() != expected_private
            or queue_root.resolve() != expected_private / "queue"
            or any(os.environ.get(key) != "1" for key in THREAD_KEYS)
            or os.getpgid(os.getpid()) != os.getpid()):
        raise PermissionError("FIT worker command, numerical threads or process group differs from queue")
    output_dir = expected_private / "m2" / "fit" / f"seed_{seed}"
    output = output_dir / "FIT_RECEIPT.json"
    spec = assert_running_unit(unit_id, queue_root, lock_path,
                               "fit_bank", output)
    if (set(spec.get("role_inputs", {})) != _exact_role_keys(seed)
            or spec.get("role_access") != [FIT_ROLE, CAL_ROLE]
            or spec.get("allow_child_processes") is not False
            or spec.get("outputs") != [str(output)]):
        raise PermissionError("FIT worker role or output declaration differs from exact contract")
    lock_bytes = lock_path.read_bytes()
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    lock = json.loads(lock_bytes)
    repo = study.parents[1]
    focused_manifest = json.loads((study / "m1" / "DATA_ACCESS_MANIFEST.json").read_text())
    prior = _resolve_prior_package(repo, lock, focused_manifest)
    manifest_path = prior / "DATA_ACCESS_MANIFEST.json"
    manifest_sha, prior_manifest_sha = _manifest_identities(
        study, manifest_path, lock)
    source_hashes = _source_hashes(study, lock)

    # SafeRoleStore construction and every role read occur after the live gate.
    from results.pcrl_method_portfolio_v1.shared.safe_data_guard import SafeRoleStore
    from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry
    from results.pcrl_capacity_matched_feature_selection_v1.m3.task_only import select_task_only_pre_sensitive
    c_dir = repo / "results/pcrl_method_portfolio_v1/teams/c"
    if str(c_dir) not in sys.path:
        sys.path.insert(0, str(c_dir))
    from training import HeadSpec

    store = SafeRoleStore(manifest_path)
    f = store.read(FIT_ROLE, "features")
    yi = store.read(FIT_ROLE, "income", "defense_fit")
    yo = store.read(FIT_ROLE, "occupation", "defense_fit")
    refs = store.read(FIT_ROLE, f"references_s{seed}")
    cf = store.read(CAL_ROLE, "features")
    cyi = store.read(CAL_ROLE, "income", "calibration")
    cyo = store.read(CAL_ROLE, "occupation", "calibration")
    _check_identical_people(f, yi, yo, refs)
    _check_identical_people(cf, cyi, cyo)
    names = tuple(str(name) for name in f["feature_names"])
    if names != tuple(str(name) for name in cf["feature_names"]):
        raise ValueError("FIT and CAL predictor schema differ")
    geometry = capacity_geometry(names)
    bank = build_shared_bank(geometry["masks"])
    head_spec = HeadSpec()
    fit_pair_hash = _array_sha(np.column_stack((f["row_id"], f["unit"])))
    cal_pair_hash = _array_sha(np.column_stack((cf["row_id"], cf["unit"])))

    def fit_package(purpose, mask, stage, rows):
        label_name = "occupation" if purpose == "occupation_group" else "income"
        fit_label = yo if label_name == "occupation" else yi
        cal_label = cyo if label_name == "occupation" else cyi
        subset_pair_sha = _array_sha(np.column_stack((f["row_id"][rows], f["unit"][rows])))
        identity = make_head_cache_identity(
            mask=mask, purpose=purpose, seed=seed,
            campaign_lock_sha256=lock_sha,
            data_manifest_sha256=manifest_sha,
            role_file_sha256=_role_file_hashes(store.manifest, purpose),
            fit_row_unit_sha256=subset_pair_sha,
            calibration_row_unit_sha256=cal_pair_hash,
            head_spec=head_spec, source_code_sha256=source_hashes,
        )
        return fit_or_restore_task_package(
            identity, fit_x=f["X"][rows], fit_y=fit_label[label_name][rows],
            fit_row_id=f["row_id"][rows], fit_unit=f["unit"][rows],
            calibration_x=cf["X"], calibration_y=cal_label[label_name],
            calibration_row_id=cf["row_id"], calibration_unit=cf["unit"],
            feature_names=names, private_root=expected_private,
            head_spec=head_spec,
            assert_queue_admitted=lambda: assert_running_unit(
                unit_id, queue_root, lock_path, "fit_bank", output),
        )

    selected_task = {}
    selected_pair = {}

    def task_sink(table, choice, validation_packages):
        _publish_task_only(output_dir, unit_id, seed, lock_sha,
                           bank, table, choice, validation_packages)
        selected_task.update(choice)

    def read_sex():
        task_receipt = output_dir / "TASK_ONLY_RECEIPT.json"
        if not task_receipt.is_file():
            raise PermissionError("immutable task-only selection must precede SEX read")
        protected = store.read(FIT_ROLE, "sex", "defense_fit")
        _check_identical_people(f, protected)
        return protected["sex"]

    def pair_sink(table):
        if not selected_task:
            raise RuntimeError("pre-sensitive T choice is absent")
        selected_pair.update(_publish_pair_selection(
            output_dir, bank, table, selected_task))

    result = run_fit_pipeline(
        bank=bank, seed=seed, fit_x=f["X"], fit_row_id=f["row_id"],
        fit_unit=f["unit"], fit_income=yi["income"],
        fit_occupation=yo["occupation"],
        fit_references={name: refs[name] for name in
                        ("U0_1", "U0_2", "Ucal_1", "Ucal_2")},
        feature_names=names, fit_package=fit_package, read_sex=read_sex,
        fit_raw_search_auc=_search_auc_factory(f["X"], seed),
        select_task_only=lambda table, bank: select_task_only_pre_sensitive(table),
        on_task_complete=task_sink, on_pair_complete=pair_sink,
    )
    if result.pair_table is None:
        no_go = _publish_pre_sensitive_no_go(
            output_dir, bank, result.task_table, result.task_only_choice)
        receipt = {
            "schema": "pcrl-focused-fit-receipt-v1",
            "status": "NO_FEASIBLE_J", "reason": no_go["reason"],
            "unit_id": unit_id, "seed": seed,
            "campaign_lock_sha256": lock_sha,
            "data_manifest_sha256": manifest_sha,
            "prior_data_manifest_sha256": prior_manifest_sha,
            "bank_sha256": bank.bank_sha256,
            "task_only_receipt_sha256": _file_sha(output_dir / "TASK_ONLY_RECEIPT.json"),
            **_validation_receipt_fields(output_dir, result.validation_packages),
            "task_table_sha256": _file_sha(output_dir / "TASK_TABLE.json"),
            "task_table_relative_path": (output_dir / "TASK_TABLE.json").relative_to(expected_private).as_posix(),
            "control_plan_sha256": no_go["control_plan_sha256"],
            "control_plan_relative_path": (output_dir / "CONTROL_PLAN.json").relative_to(expected_private).as_posix(),
            "pair_table_sha256": None, "pair_table_relative_path": None,
            "joint_selection_sha256": None, "joint_selection_relative_path": None,
            "head_packages": {}, "full_f_packages": {},
            "fit_row_unit_sha256": fit_pair_hash,
            "cal_row_unit_sha256": cal_pair_hash,
            "source_hashes": source_hashes,
            "critic_spec": None,
            "search_receipt": result.raw_search_receipt,
            "task_split_receipt": result.split_receipt,
        }
        return _write_once_json(output, receipt)
    table_path = output_dir / "PAIR_TABLE.json"
    if not table_path.is_file() or json.loads(table_path.read_text()) != result.pair_table.to_payload():
        raise RuntimeError("pair table custody differs from computed result")
    if not selected_pair or selected_pair["pair_table_sha256"] != _file_sha(table_path):
        raise RuntimeError("J and matched-control custody is incomplete")
    receipt = {
        "schema": "pcrl-focused-fit-receipt-v1",
        "status": "FIT_COMPLETE",
        "unit_id": unit_id, "seed": seed,
        "campaign_lock_sha256": lock_sha,
        "data_manifest_sha256": manifest_sha,
        "prior_data_manifest_sha256": prior_manifest_sha,
        "bank_sha256": bank.bank_sha256,
        "task_only_receipt_sha256": _file_sha(output_dir / "TASK_ONLY_RECEIPT.json"),
        **_validation_receipt_fields(output_dir, result.validation_packages),
        "task_table_sha256": _file_sha(output_dir / "TASK_TABLE.json"),
        "task_table_relative_path": (output_dir / "TASK_TABLE.json").relative_to(expected_private).as_posix(),
        "pair_table_sha256": selected_pair["pair_table_sha256"],
        "pair_table_relative_path": table_path.relative_to(expected_private).as_posix(),
        "joint_selection_sha256": selected_pair["joint_selection_sha256"],
        "joint_selection_relative_path": (output_dir / "JOINT_SELECTION.json").relative_to(expected_private).as_posix(),
        "control_plan_sha256": selected_pair["control_plan_sha256"],
        "control_plan_relative_path": (output_dir / "CONTROL_PLAN.json").relative_to(expected_private).as_posix(),
        "head_packages": result.final_packages,
        "full_f_packages": result.full_f_packages,
        "fit_row_unit_sha256": fit_pair_hash,
        "cal_row_unit_sha256": cal_pair_hash,
        "source_hashes": source_hashes,
        "critic_spec": CRITIC_SPEC,
        "search_receipt": result.raw_search_receipt,
        "task_split_receipt": result.split_receipt,
    }
    return _write_once_json(output, receipt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--queue-root", required=True, type=Path)
    parser.add_argument("--private-root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(run_worker(unit_id=args.unit_id, seed=args.seed,
                     lock_path=args.lock, queue_root=args.queue_root,
                     private_root=args.private_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

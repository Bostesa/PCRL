"""Future queued S2 producer for a protected Team A FIT capability.

This is a separate preprocessing stage. The current A bank-only seal does not
admit it. Only this producer has SafeRoleStore authority; its output contains
the two task-capsule sides plus FIT SEX, with no ATT/INNER or assessment rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from a_protected_fit_capsule import MEMBERS, SCHEMA, _sha_arr, load_protected_capsule
from a_task_only_capsule import load_task_capsule
from compute_queue import Queue
from safe_data_guard import SafeRoleStore, sha_file


CAMPAIGN = Path(__file__).resolve().parent.parent
SHARED = Path(__file__).resolve().parent
PRIVATE = SHARED / ".private"
DATA_ACCESS = "A_FIT_SEX_AND_TWO_TASK_CAPSULES"
S4_BANK_SCHEMA = "pcrl-s4-a-score-bank-verification-v1"
TSTAR_RELEASE = "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"


def _arg(argv: list[str], name: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        raise PermissionError("queued protected capsule argument missing or duplicated")
    return str(argv[indices[0] + 1])


def _running_unit(seed: int, lock_sha: str, queue_root: Path) -> dict:
    uid = f"S2-A-PROTECTED-FIT-CAPSULE-S{seed}"
    queue = Queue(queue_root)
    deadline = time.monotonic() + 5.0
    while True:
        state = queue.status()
        unit = state.get("units", {}).get(uid)
        if not unit or unit.get("status") != "running":
            raise PermissionError("protected FIT capsule lacks a running S3 unit")
        spec = unit.get("spec", {})
        argv = spec.get("argv", [])
        output = PRIVATE / f"a_protected_fit_capsule_s{seed}_v1" / "RECEIPT.json"
        if (len(argv) < 2 or Path(argv[1]).resolve() != Path(__file__).resolve() or
                spec.get("unit_id") != uid or spec.get("mode") != "real_data" or
                spec.get("family") != "SHARED" or spec.get("team") != "SHARED" or
                spec.get("arm") != "A_PROTECTED_FIT_CAPSULE" or
                spec.get("arm_id") != "A_PROTECTED_FIT_CAPSULE" or
                spec.get("stage") != "a_protected_fit_capsule_preprocess" or
                spec.get("data_access") != DATA_ACCESS or
                spec.get("seed") != seed or
                spec.get("runner_sha256") != sha_file(Path(__file__)) or
                spec.get("outputs") != [str(output)] or
                _arg(argv, "--seed") != str(seed) or
                _arg(argv, "--campaign-lock-sha256") != lock_sha or
                _arg(argv, "--queue-root") != str(queue_root)):
            raise PermissionError("running protected capsule unit identity differs")
        child_pgid = unit.get("child_pgid")
        if child_pgid is not None:
            if child_pgid != os.getpgrp():
                raise PermissionError("protected capsule process group differs from S3")
            return state
        if time.monotonic() >= deadline:
            raise PermissionError("protected capsule S3 child process group unavailable")
        time.sleep(0.02)


def _admit(seed: int, lock_sha: str, state: dict) -> tuple[SafeRoleStore, dict, dict]:
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != lock_sha:
        raise PermissionError("protected capsule campaign lock SHA differs")
    lock = json.loads(lock_path.read_text())
    uid = f"S2-A-PROTECTED-FIT-CAPSULE-S{seed}"
    if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
            lock.get("adult_fit_permission") is not True or
            lock.get("families", {}).get("A", {}).get("fit_admission") != "DENIED" or
            lock.get("families", {}).get("A", {}).get(
                "protected_capsule_preprocessing_admission") != "ALLOWED" or
            uid not in lock.get("admitted_a_protected_capsule_unit_ids", []) or
            lock.get("a_protected_capsule_runner_sha256") != sha_file(Path(__file__)) or
            lock.get("a_protected_capsule_role_fields") !=
            {"OSF_DEFENSE_FIT": ["sex"]} or
            not Queue._local_seal_ok(state.get("gate", {}))):
        raise PermissionError("protected FIT preprocessing is not separately sealed")
    manifest_path = CAMPAIGN / "DATA_ACCESS_MANIFEST.json"
    if lock.get("data_admission", {}).get("manifest_sha256") != sha_file(manifest_path):
        raise PermissionError("protected FIT role manifest differs")
    report_path = CAMPAIGN / f"S4_A_SCORE_BANK_VERIFICATION_s{seed}.json"
    report_sha = sha_file(report_path)
    if report_sha != lock.get("a_score_bank_verification_sha256_by_seed", {}).get(f"s{seed}"):
        raise PermissionError("T* source bank lacks S4 numeric parity admission")
    report = json.loads(report_path.read_text())
    bank_path = PRIVATE / f"tstar_score_bank_s{seed}_v1" / "BANK_ADMISSION_RECEIPT.json"
    bank_sha = sha_file(bank_path)
    bank = json.loads(bank_path.read_text())
    if (report.get("schema") != S4_BANK_SCHEMA or report.get("status") != "PASS" or
            report.get("role_cell_token_bank_parity") != "PASS" or
            report.get("bank_admission_receipt_sha256") != bank_sha or
            report.get("source_binding_sha256") != bank.get("source_binding_sha256") or
            bank.get("status") != "ROLE_ONLY_BANK_ADMITTED" or
            bank.get("seed") != seed or bank.get("release_id") != TSTAR_RELEASE or
            bank.get("data_access_manifest_sha256") != sha_file(manifest_path)):
        raise PermissionError("protected FIT T* bank/source identity differs")
    store = SafeRoleStore(manifest_path)
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("protected FIT role files are not S4 admitted")
    return store, bank, lock


def _task_capsules(seed: int, lock: dict, bank: dict) -> tuple[dict, dict, dict, dict]:
    family = lock.get("families", {}).get("A", {})
    receipt_map = family.get("task_capsule_receipt_sha256_by_seed_recipient", {})
    report_map = family.get("task_capsule_s4_sha256_by_seed_recipient", {})
    expected = {f"r{r}": receipt_map.get(f"s{seed}:r{r}") for r in (1, 2)}
    verified = {f"r{r}": report_map.get(f"s{seed}:r{r}") for r in (1, 2)}
    if (any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in list(expected.values()) + list(verified.values()))):
        raise PermissionError("two recipient task capsules are not lock-pinned")
    tasks = []
    report_sha = sha_file(CAMPAIGN / f"S4_A_SCORE_BANK_VERIFICATION_s{seed}.json")
    for recipient in (1, 2):
        root = PRIVATE / f"a_task_only_capsule_s{seed}_r{recipient}_v1"
        verification_path = CAMPAIGN / f"S4_A_TASK_CAPSULE_VERIFICATION_s{seed}_r{recipient}.json"
        if not verification_path.is_file() or \
                sha_file(verification_path) != verified[f"r{recipient}"]:
            raise PermissionError("S4 recipient task capsule report SHA differs")
        verification = json.loads(verification_path.read_text())
        if (verification.get("schema") != "pcrl-s4-a-task-capsule-verification-v1" or
                verification.get("status") != "PASS" or
                verification.get("seed") != seed or
                verification.get("recipient") != recipient or
                verification.get("capsule_sha256") != sha_file(root / "task_only.npz") or
                verification.get("capsule_receipt_sha256") != expected[f"r{recipient}"] or
                verification.get("bank_admission_receipt_sha256") !=
                sha_file(PRIVATE / f"tstar_score_bank_s{seed}_v1" /
                         "BANK_ADMISSION_RECEIPT.json") or
                verification.get("source_binding_sha256") != bank["source_binding_sha256"] or
                verification.get("fit_row_unit_pair_sha256") !=
                bank["role_receipts"]["OSF_DEFENSE_FIT"]["row_unit_pair_sha256"] or
                any(verification.get(field) != "PASS" for field in (
                    "own_task_label_parity", "no_sex_or_partner_members",
                    "source_bank_parity")) or
                re.fullmatch(r"[0-9a-f]{64}",
                             str(verification.get("reviewer_source_sha256", ""))) is None):
            raise PermissionError("S4 recipient task capsule parity differs")
        task = load_task_capsule(
            root / "task_only.npz", root / "RECEIPT.json",
            expected_receipt_sha256=expected[f"r{recipient}"],
            recipient=recipient, model_seed=seed)
        record = task["receipt"]
        binding = bank["source_binding"]["recipients"][f"r{recipient}"]
        if (record.get("bank_admission_receipt_sha256") !=
                sha_file(PRIVATE / f"tstar_score_bank_s{seed}_v1" /
                         "BANK_ADMISSION_RECEIPT.json") or
                record.get("s4_bank_report_sha256") != report_sha or
                record.get("row_unit_pair_sha256") !=
                verification["fit_row_unit_pair_sha256"] or
                record.get("source_binding_sha256") != bank["source_binding_sha256"] or
                _sha_arr(task["bank"]) != binding["bank_probability_sha256"] or
                _sha_arr(task["input_class"]) != binding["input_class_sha256"] or
                _sha_arr(task["fine_to_hcal_token"]) !=
                binding["fine_to_hcal_token_sha256"] or
                _sha_arr(task["token_class"]) != binding["token_class_sha256"] or
                _sha_arr(task["cell"]) != bank["role_receipts"]["OSF_DEFENSE_FIT"][
                    f"cell_{recipient}_sha256"]):
            raise PermissionError("protected FIT task capsule differs from admitted T* bank")
        tasks.append(task)
    if (not np.array_equal(tasks[0]["row_id"], tasks[1]["row_id"]) or
            not np.array_equal(tasks[0]["unit"], tasks[1]["unit"])):
        raise PermissionError("protected FIT task capsule pair is unaligned")
    return tasks[0], tasks[1], expected, verified


def _assemble_arrays(task1: dict, task2: dict, sex_record: dict) -> dict:
    for item in (task2, sex_record):
        if (not np.array_equal(task1["row_id"], item["row_id"]) or
                not np.array_equal(task1["unit"], item["unit"])):
            raise PermissionError("protected FIT person/group order differs")
    if "sex" not in sex_record or set(sex_record) - {"row_id", "unit", "sex"}:
        raise PermissionError("protected FIT SEX record exposes extra fields")
    arrays = {"row_id": task1["row_id"], "unit": task1["unit"],
              "cell_1": task1["cell"], "cell_2": task2["cell"],
              "income": task1["task_label"],
              "occupation": task2["task_label"], "sex": sex_record["sex"],
              "U0_1": task1["U0"], "U0_2": task2["U0"],
              "Ucal_1": task1["Ucal"], "Ucal_2": task2["Ucal"],
              "bank_1": task1["bank"], "bank_2": task2["bank"],
              "input_class_1": task1["input_class"],
              "input_class_2": task2["input_class"],
              "fine_to_hcal_token_1": task1["fine_to_hcal_token"],
              "fine_to_hcal_token_2": task2["fine_to_hcal_token"],
              "token_class_1": task1["token_class"],
              "token_class_2": task2["token_class"],
              "source_binding_sha256": task1["source_binding_sha256"]}
    if set(arrays) != MEMBERS or not np.array_equal(
            task1["source_binding_sha256"], task2["source_binding_sha256"]):
        raise PermissionError("protected FIT source/members differ")
    return {name: np.asarray(value) for name, value in arrays.items()}


def run(seed: int, lock_sha: str, queue_root: Path) -> dict:
    if seed not in (0, 1, 2):
        raise ValueError("protected FIT source seed is outside the locked set")
    state = _running_unit(seed, lock_sha, queue_root)
    store, bank, lock = _admit(seed, lock_sha, state)
    task1, task2, task_receipts, task_reports = _task_capsules(seed, lock, bank)
    role = "OSF_DEFENSE_FIT"
    sex_record = store.read(role, "sex", "defense_fit")
    arrays = _assemble_arrays(task1, task2, sex_record)
    output = PRIVATE / f"a_protected_fit_capsule_s{seed}_v1"
    if output.exists():
        raise FileExistsError("protected FIT capsule output is write-once")
    output.mkdir(parents=True, mode=0o700)
    npz = output / "protected_fit.npz"
    fd = os.open(npz, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    pair_sha = _sha_arr(np.column_stack((arrays["row_id"], arrays["unit"])).astype(np.int64))
    fit_sex_sha = store.manifest["real_file_specs"][f"{role}:sex"]["sha256"]
    bank_sha = sha_file(PRIVATE / f"tstar_score_bank_s{seed}_v1" /
                        "BANK_ADMISSION_RECEIPT.json")
    report_sha = sha_file(CAMPAIGN / f"S4_A_SCORE_BANK_VERIFICATION_s{seed}.json")
    receipt = {"schema": SCHEMA, "status": "FROZEN", "model_seed": seed,
               "role": role, "rows": len(arrays["row_id"]),
               "groups": int(np.unique(arrays["unit"]).size),
               "members": sorted(MEMBERS), "npz_sha256": sha_file(npz),
               "array_sha256": {name: _sha_arr(value) for name, value in arrays.items()},
               "row_unit_pair_sha256": pair_sha,
               "source_binding_sha256": bank["source_binding_sha256"],
               "bank_admission_receipt_sha256": bank_sha,
               "s4_bank_report_sha256": report_sha,
               "task_capsule_receipt_sha256_by_recipient": task_receipts,
               "s4_task_capsule_report_sha256_by_recipient": task_reports,
               "fit_sex_role_file_sha256": fit_sex_sha,
               "data_access_manifest_sha256": sha_file(store.manifest_path),
               "campaign_lock_sha256": lock_sha,
               "producer_sha256": sha_file(Path(__file__)),
               "scope": "FIT task capsules and FIT SEX only; no ATT, INNER or assessment"}
    rp = output / "RECEIPT.json"
    fd = os.open(rp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt_sha = sha_file(rp)
    load_protected_capsule(npz, rp, expected_receipt_sha256=receipt_sha,
                           model_seed=seed)
    return {"model_seed": seed, "private_receipt_sha256": receipt_sha}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--campaign-lock-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.seed, args.campaign_lock_sha256, args.queue_root)
    except BaseException as exc:
        print("A_PROTECTED_FIT_CAPSULE_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))

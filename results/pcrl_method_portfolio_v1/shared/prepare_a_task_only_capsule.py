"""Queued S2 custodian split of one admitted T* FIT bank into one task capability.

Only this producer imports SafeRoleStore. A task-only fit worker receives one
recipient-only capsule through a separate pure loader and cannot access the
custodian's combined reference or broad label-loading API by that interface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np

from a_task_only_capsule import MEMBERS, SCHEMA, _sha_arr, load_task_capsule
from compute_queue import Queue
from safe_data_guard import SafeRoleStore, sha_file


CAMPAIGN = Path(__file__).resolve().parent.parent
PRIVATE = Path(__file__).resolve().parent / ".private"
BANK_RELEASE_ID = "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"
S4_SCHEMA = "pcrl-s4-a-score-bank-verification-v1"
DATA_ACCESS = "A_FIT_OWN_TASK_AND_REFERENCES_CAPSULE"


def _arg(argv: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise PermissionError("queued A task capsule argument missing or duplicated")
    return str(argv[positions[0] + 1])


def _running_unit(seed: int, recipient: int, lock_sha: str,
                  queue_root: Path) -> dict:
    unit_id = f"S2-A-TASK-CAPSULE-S{seed}-R{recipient}"
    queue = Queue(queue_root)
    deadline = time.monotonic() + 5.0
    while True:
        state = queue.status()
        unit = state.get("units", {}).get(unit_id)
        if not unit or unit.get("status") != "running":
            raise PermissionError("A task capsule lacks running S3 unit")
        spec = unit.get("spec", {})
        argv = spec.get("argv", [])
        output = PRIVATE / f"a_task_only_capsule_s{seed}_r{recipient}_v1" / "RECEIPT.json"
        if (len(argv) < 2 or Path(argv[1]).resolve() != Path(__file__).resolve() or
                spec.get("unit_id") != unit_id or spec.get("mode") != "real_data" or
                spec.get("family") != "SHARED" or spec.get("team") != "SHARED" or
                spec.get("stage") != "a_task_capsule_preprocess" or
                spec.get("data_access") != DATA_ACCESS or
                spec.get("seed") != seed or spec.get("recipient") != recipient or
                spec.get("runner_sha256") != sha_file(Path(__file__)) or
                spec.get("outputs") != [str(output)] or
                _arg(argv, "--seed") != str(seed) or
                _arg(argv, "--recipient") != str(recipient) or
                _arg(argv, "--campaign-lock-sha256") != lock_sha or
                _arg(argv, "--queue-root") != str(queue_root)):
            raise PermissionError("running A task capsule unit identity differs")
        child_pgid = unit.get("child_pgid")
        if child_pgid is not None:
            if child_pgid != os.getpgrp():
                raise PermissionError("A task capsule process group differs from S3")
            return state
        if time.monotonic() >= deadline:
            raise PermissionError("S3 task capsule child process group unavailable")
        time.sleep(0.02)


def _admit(seed: int, recipient: int, lock_sha: str,
           queue_state: dict) -> tuple[SafeRoleStore, Path, dict]:
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != lock_sha:
        raise PermissionError("A task capsule science lock SHA differs")
    lock = json.loads(lock_path.read_text())
    unit_id = f"S2-A-TASK-CAPSULE-S{seed}-R{recipient}"
    if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
            lock.get("adult_fit_permission") is not True or
            lock.get("families", {}).get("A", {}).get("preprocessing_admission") != "ALLOWED" or
            lock.get("families", {}).get("A", {}).get("fit_admission") != "DENIED" or
            unit_id not in lock.get("admitted_a_task_capsule_unit_ids", []) or
            lock.get("a_task_capsule_runner_sha256") != sha_file(Path(__file__))):
        raise PermissionError("A task capsule preprocessing is not admitted")
    if not Queue._local_seal_ok(queue_state.get("gate", {})):
        raise PermissionError("A task capsule committed local science seal absent")
    manifest_path = CAMPAIGN / "DATA_ACCESS_MANIFEST.json"
    if lock.get("data_admission", {}).get("manifest_sha256") != sha_file(manifest_path):
        raise PermissionError("A task capsule role manifest differs from science lock")
    bank_root = PRIVATE / f"tstar_score_bank_s{seed}_v1"
    receipt_path = bank_root / "BANK_ADMISSION_RECEIPT.json"
    report_path = CAMPAIGN / f"S4_A_SCORE_BANK_VERIFICATION_s{seed}.json"
    if sha_file(report_path) != lock.get("a_score_bank_verification_sha256_by_seed", {}).get(f"s{seed}"):
        raise PermissionError("S4 A score-bank verification is not frozen")
    report = json.loads(report_path.read_text())
    bank_receipt_sha = sha_file(receipt_path)
    bank_receipt = json.loads(receipt_path.read_text())
    if (report.get("schema") != S4_SCHEMA or report.get("status") != "PASS" or
            report.get("role_cell_token_bank_parity") != "PASS" or
            report.get("bank_admission_receipt_sha256") != bank_receipt_sha or
            report.get("source_binding_sha256") != bank_receipt.get("source_binding_sha256") or
            bank_receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1" or
            bank_receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED" or
            bank_receipt.get("seed") != seed or bank_receipt.get("release_id") != BANK_RELEASE_ID or
            bank_receipt.get("data_access_manifest_sha256") != sha_file(manifest_path)):
        raise PermissionError("A source bank S4 admission/source identity differs")
    store = SafeRoleStore(manifest_path)
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("A task capsule role files are not S4 admitted")
    return store, bank_root, bank_receipt


def _private_npz(path: Path, expected_members: set[str], expected_sha: str) -> dict:
    if sha_file(path) != expected_sha:
        raise PermissionError("A source bank private file SHA differs")
    with ZipFile(path) as archive:
        names = {entry.filename for entry in archive.infolist()}
    if names != {name + ".npy" for name in expected_members}:
        raise PermissionError("A source bank private member list differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in expected_members}


def run(seed: int, recipient: int, lock_sha: str, queue_root: Path) -> dict:
    if recipient not in (1, 2):
        raise ValueError("A task capsule recipient must be one or two")
    state = _running_unit(seed, recipient, lock_sha, queue_root)
    store, bank_root, bank_receipt = _admit(seed, recipient, lock_sha, state)
    role = "OSF_DEFENSE_FIT"
    task = "income" if recipient == 1 else "occupation"
    token_path = bank_root / f"tokens_{role}.npz"
    bank_path = bank_root / f"bank_r{recipient}.npz"
    token = _private_npz(token_path,
                         {"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"},
                         bank_receipt["private_output_sha256"][token_path.name])
    bank = _private_npz(bank_path,
                        {"probability", "token_class", "token_id", "input_class",
                         "fine_to_hcal_token"},
                        bank_receipt["private_output_sha256"][bank_path.name])
    # These two S2 custodian reads occur only after the S4 bank and S3 role
    # gates. The task-only fit process never imports SafeRoleStore or receives
    # this combined source-reference object.
    label = store.read(role, task, "defense_fit")
    reference = store.read(role, f"references_s{seed}")
    for item in (label, reference):
        if not np.array_equal(token["row_id"], item["row_id"]) or \
                not np.array_equal(token["unit"], item["unit"]):
            raise PermissionError("A task capsule FIT row/group order differs")
    pair_sha = _sha_arr(np.column_stack((token["row_id"], token["unit"])).astype(np.int64))
    if (pair_sha != bank_receipt["role_receipts"][role]["row_unit_pair_sha256"] or
            bank_receipt["role_receipts"][role]["source_reference_sha256"] !=
            store.manifest["real_file_specs"][f"{role}:references_s{seed}"]["sha256"]):
        raise PermissionError("A task capsule source FIT identity differs")
    arrays = {"row_id": np.asarray(token["row_id"], dtype=np.int64),
              "unit": np.asarray(token["unit"], dtype=np.int64),
              "cell": np.asarray(token[f"cell_{recipient}"], dtype=np.int64),
              "task_label": np.asarray(label[task], dtype=np.int64),
              "U0": np.asarray(reference[f"U0_{recipient}"], dtype=np.float64),
              "Ucal": np.asarray(reference[f"Ucal_{recipient}"], dtype=np.float64),
              "bank": np.asarray(bank["probability"], dtype=np.float64),
              "input_class": np.asarray(bank["input_class"], dtype=np.int64),
              "fine_to_hcal_token": np.asarray(bank["fine_to_hcal_token"], dtype=np.int64),
              "token_class": np.asarray(bank["token_class"], dtype=np.int64),
              "source_binding_sha256": np.array(bank_receipt["source_binding_sha256"])}
    if set(arrays) != MEMBERS:
        raise RuntimeError("A task capsule member contract differs")
    output = PRIVATE / f"a_task_only_capsule_s{seed}_r{recipient}_v1"
    if output.exists():
        raise FileExistsError("A task capsule output is write-once")
    output.mkdir(parents=True, mode=0o700)
    path = output / "task_only.npz"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": SCHEMA, "status": "FROZEN", "model_seed": seed,
               "recipient": recipient, "task": task, "role": role,
               "rows": len(token["row_id"]),
               "groups": int(np.unique(token["unit"]).size),
               "members": sorted(MEMBERS), "npz_sha256": sha_file(path),
               "array_sha256": {name: _sha_arr(value) for name, value in arrays.items()},
               "row_unit_pair_sha256": pair_sha,
               "source_binding_sha256": bank_receipt["source_binding_sha256"],
               "bank_admission_receipt_sha256": sha_file(bank_root / "BANK_ADMISSION_RECEIPT.json"),
               "s4_bank_report_sha256": sha_file(CAMPAIGN / f"S4_A_SCORE_BANK_VERIFICATION_s{seed}.json"),
               "source_role_file_sha256": {
                   f"{role}:{task}": store.manifest["real_file_specs"][f"{role}:{task}"]["sha256"],
                   f"{role}:references_s{seed}":
                       store.manifest["real_file_specs"][f"{role}:references_s{seed}"]["sha256"]},
               "campaign_lock_sha256": lock_sha,
               "data_access_manifest_sha256": sha_file(store.manifest_path),
               "producer_sha256": sha_file(Path(__file__)),
               "scope": "one recipient task labels and scores only; no SEX/partner members"}
    receipt_path = output / "RECEIPT.json"
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt_sha = sha_file(receipt_path)
    load_task_capsule(path, receipt_path, expected_receipt_sha256=receipt_sha,
                      recipient=recipient, model_seed=seed)
    return {"seed": seed, "recipient": recipient,
            "private_receipt_sha256": receipt_sha}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--recipient", type=int, choices=(1, 2), required=True)
    parser.add_argument("--campaign-lock-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.seed, args.recipient, args.campaign_lock_sha256, args.queue_root)
    except BaseException as exc:
        print("A_TASK_CAPSULE_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))

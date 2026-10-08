"""Pure sealed input boundary for a Team A protected paired-law fit.

This module has no role store, source-study loader, or filesystem discovery.
The S2 producer is a separate queued process and remains unadmitted until a
future A fit seal. Task-only workers must never receive this capsule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np


SCHEMA = "pcrl-portfolio-a-protected-fit-capsule-v1"
MEMBERS = frozenset({
    "row_id", "unit", "cell_1", "cell_2", "income", "occupation", "sex",
    "U0_1", "U0_2", "Ucal_1", "Ucal_2", "bank_1", "bank_2",
    "input_class_1", "input_class_2", "fine_to_hcal_token_1",
    "fine_to_hcal_token_2", "token_class_1", "token_class_2",
    "source_binding_sha256",
})
HEX64 = re.compile(r"[0-9a-f]{64}")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_arr(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_protected_capsule(npz_path: str | Path, receipt_path: str | Path,
                           *, expected_receipt_sha256: str,
                           model_seed: int) -> dict:
    """Load exactly one private FIT capsule after its receipt is pinned."""
    path, receipt_file = Path(npz_path), Path(receipt_path)
    if not isinstance(expected_receipt_sha256, str) or \
            HEX64.fullmatch(expected_receipt_sha256) is None:
        raise ValueError("protected FIT receipt SHA is malformed")
    if _sha_file(receipt_file) != expected_receipt_sha256:
        raise PermissionError("protected FIT receipt SHA differs")
    receipt = json.loads(receipt_file.read_text())
    n = receipt.get("rows")
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "FROZEN" or
            receipt.get("model_seed") != model_seed or
            receipt.get("role") != "OSF_DEFENSE_FIT" or
            not isinstance(n, int) or n <= 0 or
            receipt.get("npz_sha256") != _sha_file(path) or
            receipt.get("members") != sorted(MEMBERS) or
            set(receipt.get("array_sha256", {})) != MEMBERS or
            set(receipt.get("task_capsule_receipt_sha256_by_recipient", {})) !=
            {"r1", "r2"} or
            set(receipt.get("s4_task_capsule_report_sha256_by_recipient", {})) !=
            {"r1", "r2"}):
        raise PermissionError("protected FIT identity or member contract differs")
    for key in ("bank_admission_receipt_sha256", "s4_bank_report_sha256",
                "fit_sex_role_file_sha256", "source_binding_sha256",
                "row_unit_pair_sha256", "data_access_manifest_sha256",
                "campaign_lock_sha256", "producer_sha256"):
        if HEX64.fullmatch(str(receipt.get(key, ""))) is None:
            raise PermissionError("protected FIT source receipt SHA is malformed")
    for sha in receipt["task_capsule_receipt_sha256_by_recipient"].values():
        if HEX64.fullmatch(str(sha)) is None:
            raise PermissionError("protected FIT task capsule SHA is malformed")
    for sha in receipt["s4_task_capsule_report_sha256_by_recipient"].values():
        if HEX64.fullmatch(str(sha)) is None:
            raise PermissionError("protected FIT S4 task report SHA is malformed")
    with ZipFile(path) as archive:
        actual = {entry.filename for entry in archive.infolist()}
        if len(archive.infolist()) != len(MEMBERS) or \
                actual != {name + ".npy" for name in MEMBERS}:
            raise PermissionError("protected FIT archive has missing or forbidden members")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in MEMBERS}
    if values["row_id"].dtype != np.int64 or values["unit"].dtype != np.int64 or \
            values["row_id"].shape != (n,) or values["unit"].shape != (n,):
        raise ValueError("protected FIT ordered person/group schema differs")
    if (n > 1 and not np.all(np.diff(values["row_id"]) > 0)) or \
            int(np.unique(values["unit"]).size) != receipt.get("groups"):
        raise ValueError("protected FIT ordered people/groups differ")
    pair = np.column_stack((values["row_id"], values["unit"])).astype(np.int64)
    if _sha_arr(pair) != receipt["row_unit_pair_sha256"]:
        raise PermissionError("protected FIT row/group binding differs")
    for name, classes in (("income", 2), ("occupation", 6), ("sex", 2)):
        value = values[name]
        if value.dtype != np.int64 or value.shape != (n,) or \
                np.any(value < 0) or np.any(value >= classes):
            raise ValueError("protected FIT label schema or range differs")
    for recipient, classes in ((1, 2), (2, 6)):
        cell = values[f"cell_{recipient}"]
        input_class = values[f"input_class_{recipient}"]
        fine_to_token = values[f"fine_to_hcal_token_{recipient}"]
        token_class = values[f"token_class_{recipient}"]
        bank = values[f"bank_{recipient}"]
        u0 = values[f"U0_{recipient}"]
        ucal = values[f"Ucal_{recipient}"]
        for item in (cell, input_class, fine_to_token, token_class):
            if item.dtype != np.int64 or item.ndim != 1:
                raise ValueError("protected FIT cell/bank ID schema differs")
        if (cell.shape != (n,) or input_class.size == 0 or
                fine_to_token.shape != input_class.shape or
                bank.dtype != np.float64 or bank.ndim != 2 or
                bank.shape[1] != classes or token_class.shape != (bank.shape[0],) or
                u0.dtype != np.float64 or u0.shape != (n, classes) or
                ucal.dtype != np.float64 or ucal.shape != (n, classes)):
            raise ValueError("protected FIT probability/bank schema differs")
        for probability in (bank, u0, ucal):
            if (not np.all(np.isfinite(probability)) or np.any(probability < 0) or
                    not np.allclose(probability.sum(axis=1), 1, atol=1e-8, rtol=0)):
                raise ValueError("protected FIT probability simplex differs")
        if (np.any(cell < 0) or np.any(cell >= len(input_class)) or
                np.any(fine_to_token < 0) or np.any(fine_to_token >= len(token_class)) or
                np.any(input_class < 0) or np.any(input_class >= classes) or
                np.any(token_class < 0) or np.any(token_class >= classes)):
            raise ValueError("protected FIT code/class range differs")
        if (not np.array_equal(np.argmax(bank, axis=1), token_class) or
                not np.array_equal(token_class[fine_to_token], input_class) or
                not np.array_equal(input_class[cell], np.argmax(u0, axis=1)) or
                not np.array_equal(np.argmax(u0, axis=1), np.argmax(ucal, axis=1))):
            raise ValueError("protected FIT code/bank/reference decisions differ")
    binding = values["source_binding_sha256"]
    if binding.shape != () or str(binding.item()) != receipt["source_binding_sha256"] or \
            HEX64.fullmatch(str(binding.item())) is None:
        raise PermissionError("protected FIT source binding differs")
    for name in MEMBERS:
        if _sha_arr(values[name]) != receipt["array_sha256"][name]:
            raise PermissionError("protected FIT array SHA differs")
    values["receipt"] = receipt
    return values

"""Pure recipient-only input boundary for Team A task-only channel controls.

This module has no data-loader or inherited-study import. A task-only worker
receives exactly one verified capsule and cannot obtain SEX or partner fields
through this API. The custodian producer is a separate queued process.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np


SCHEMA = "pcrl-portfolio-a-task-only-capsule-v1"
MEMBERS = frozenset({"row_id", "unit", "cell", "task_label", "U0", "Ucal",
                     "bank", "input_class", "fine_to_hcal_token", "token_class",
                     "source_binding_sha256"})


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


def load_task_capsule(npz_path: str | Path, receipt_path: str | Path,
                      *, expected_receipt_sha256: str, recipient: int,
                      model_seed: int) -> dict:
    """Admit one immutable task-only capsule; reject every extra member."""
    path = Path(npz_path)
    receipt_file = Path(receipt_path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256):
        raise ValueError("task capsule receipt SHA is malformed")
    if _sha_file(receipt_file) != expected_receipt_sha256:
        raise PermissionError("task capsule receipt SHA differs")
    receipt = json.loads(receipt_file.read_text())
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "FROZEN" or
            receipt.get("recipient") != recipient or receipt.get("model_seed") != model_seed or
            receipt.get("task") != ("income" if recipient == 1 else "occupation") or
            receipt.get("role") != "OSF_DEFENSE_FIT" or
            receipt.get("npz_sha256") != _sha_file(path) or
            receipt.get("members") != sorted(MEMBERS)):
        raise PermissionError("task capsule identity or member contract differs")
    with ZipFile(path) as archive:
        actual = {entry.filename for entry in archive.infolist()}
    if actual != {name + ".npy" for name in MEMBERS}:
        raise PermissionError("task capsule contains missing or forbidden members")
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in MEMBERS}
    n = receipt.get("rows")
    k = 2 if recipient == 1 else 6 if recipient == 2 else None
    if k is None or not isinstance(n, int) or n <= 0:
        raise ValueError("task capsule recipient or row count differs")
    for name in ("row_id", "unit", "cell", "task_label", "input_class",
                 "fine_to_hcal_token", "token_class"):
        if values[name].dtype != np.int64 or values[name].ndim != 1:
            raise ValueError("task capsule integer-array schema differs")
    for name in ("row_id", "unit", "cell", "task_label"):
        if values[name].shape != (n,):
            raise ValueError("task capsule ordered row shape differs")
    if (n > 1 and not np.all(np.diff(values["row_id"]) > 0)) or \
            np.unique(values["unit"]).size != receipt.get("groups"):
        raise ValueError("task capsule ordered people/groups differ")
    pair = np.column_stack((values["row_id"], values["unit"])).astype(np.int64)
    if _sha_arr(pair) != receipt.get("row_unit_pair_sha256"):
        raise PermissionError("task capsule row/group binding differs")
    binding = values["source_binding_sha256"]
    if binding.shape != () or str(binding.item()) != receipt.get("source_binding_sha256") or \
            not re.fullmatch(r"[0-9a-f]{64}", str(binding.item())):
        raise PermissionError("task capsule source binding differs")
    bank = values["bank"]
    input_class = values["input_class"]
    fine_to_token = values["fine_to_hcal_token"]
    token_class = values["token_class"]
    if (bank.dtype != np.float64 or bank.ndim != 2 or bank.shape[1] != k or
            values["U0"].dtype != np.float64 or values["U0"].shape != (n, k) or
            values["Ucal"].dtype != np.float64 or values["Ucal"].shape != (n, k) or
            input_class.size == 0 or fine_to_token.shape != input_class.shape or
            token_class.shape != (bank.shape[0],)):
        raise ValueError("task capsule probability/bank schema differs")
    for probability in (bank, values["U0"], values["Ucal"]):
        if (not np.all(np.isfinite(probability)) or np.any(probability < 0) or
                not np.allclose(probability.sum(axis=1), 1, atol=1e-8, rtol=0)):
            raise ValueError("task capsule probability simplex differs")
    if (np.any(values["cell"] < 0) or np.any(values["cell"] >= input_class.size) or
            np.any(fine_to_token < 0) or np.any(fine_to_token >= len(token_class)) or
            np.any(values["task_label"] < 0) or np.any(values["task_label"] >= k) or
            np.any(input_class < 0) or np.any(input_class >= k) or
            np.any(token_class < 0) or np.any(token_class >= k)):
        raise ValueError("task capsule class/index range differs")
    if (not np.array_equal(np.argmax(bank, axis=1), token_class) or
            not np.array_equal(token_class[fine_to_token], input_class) or
            not np.array_equal(input_class[values["cell"]], np.argmax(values["U0"], axis=1)) or
            not np.array_equal(np.argmax(values["U0"], axis=1),
                               np.argmax(values["Ucal"], axis=1))):
        raise ValueError("task capsule fine-cell, bank, and reference decisions differ")
    for name, expected in receipt.get("array_sha256", {}).items():
        if name not in MEMBERS or _sha_arr(values[name]) != expected:
            raise PermissionError("task capsule array SHA differs")
    if set(receipt.get("array_sha256", {})) != MEMBERS:
        raise PermissionError("task capsule array hash list differs")
    values["receipt"] = receipt
    return values

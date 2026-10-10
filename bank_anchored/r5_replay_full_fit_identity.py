"""Independent saved-wire replay of the queued R2 full-FIT identity receipts.

Uses the predecessor's immutable public table and saved B0 wire. It imports no
R2 channel law, source loader, runner endpoint, or selection implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .r5_independent import public_id_from_bits, audit_table


def _sha_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def replay(seed: int, config: Path, receipt_path: Path, table_path: Path) -> dict:
    config = Path(config)
    receipt_path = Path(receipt_path)
    table_path = Path(table_path)
    cfg = json.loads(config.read_text())
    rec = json.loads(receipt_path.read_text())
    table_doc = json.loads(table_path.read_text())
    if (seed not in (0, 1) or cfg["seed"] != seed or
            rec["source_seed"] != seed or rec["status"] != "PASS" or
            rec["role"] != "OSF_DEFENSE_FIT" or rec["rows"] != 15434 or
            cfg["unit_id"] != rec["unit_id"] or
            table_doc["seed"] != seed):
        raise ValueError("seed, role, queued unit or status differs")
    saved_path = Path(cfg["pins"]["saved_b0"]["path"])
    if (_sha_file(saved_path) != cfg["pins"]["saved_b0"]["sha256"] or
            rec["saved_b0_wire_sha256"] != _sha_file(saved_path)):
        raise ValueError("saved B0 FIT wire differs")
    with np.load(saved_path, allow_pickle=False) as archive:
        if set(archive.files) != {"row_id", "unit", "code0", "code1", "prob0",
                                  "prob1", "decision0", "decision1"}:
            raise ValueError("saved B0 FIT wire field set differs")
        saved = {key: np.array(archive[key], copy=True) for key in archive.files}
    if (len(saved["row_id"]) != 15434 or
            len(np.unique(saved["unit"])) != 15428 or
            rec["groups"] != 15428 or
            rec["row_unit_sha256"] != _sha_array(
                np.column_stack((saved["row_id"], saved["unit"])))):
        raise ValueError("row/group identity differs")
    family = table_doc["families"]["W_TSTAR"]
    purpose_receipts = {}
    for purpose, label in ((0, "r1"), (1, "r2")):
        table = np.asarray(family[f"table{purpose}"], dtype=np.float64)
        classes = np.asarray(family[f"token_class{purpose}"], dtype=np.int64)
        audit_table(table, classes)
        code = public_id_from_bits(saved[f"code{purpose}"])
        if (np.any(code >= len(table)) or
                not np.array_equal(table[code], saved[f"prob{purpose}"]) or
                not np.array_equal(classes[code], saved[f"decision{purpose}"]) or
                rec[f"{label}_code_sha256"] != _sha_array(code) or
                rec[f"{label}_table_sha256"] != _sha_array(table)):
            raise ValueError(f"{label} original ID, code, probability or decision differs")
        purpose_receipts[label] = {"states": len(table),
                                   "code_sha256": _sha_array(code),
                                   "table_sha256": _sha_array(table)}
    return {"seed": seed, "status": "PASS", "rows": 15434,
            "groups": 15428, "config_sha256": _sha_file(config),
            "r2_receipt_sha256": _sha_file(receipt_path),
            "public_table_sha256": _sha_file(table_path),
            "purpose": purpose_receipts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.seed, args.config, args.receipt, args.table),
                     sort_keys=True))


if __name__ == "__main__":
    main()

"""Independent queued check of the restored bank release and reader.

This verifier reads the emitted wire, fixed public bank, cached FIT codes, and
the restored reader. It does not import the release or endpoint runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np

from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact_npz(path: Path, names: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        if {part.filename for part in archive.infolist()} != {name + ".npy" for name in names}:
            raise ValueError("unexpected NPZ member")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: np.asarray(arrays[name]).copy() for name in names}


def decode(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.uint8 or bits.ndim != 2 or bits.shape[1] != 10 or np.any(bits > 1):
        raise ValueError("public code must be ten little-endian bits")
    return (bits.astype(np.int64) * (1 << np.arange(10, dtype=np.int64))).sum(axis=1)


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    pinned = {Path(item["path"]): item["sha256"]
              for item in state["units"][active["unit_id"]]["spec"]["sources"]}
    if pinned.get(config_path) != sha(config_path):
        raise ValueError("configuration is not pinned")
    cfg = json.loads(config_path.read_text())
    if cfg["schema"] != "pcrl-bank-r5-independent-restore-check-v1" or cfg["unit_id"] != active["unit_id"]:
        raise ValueError("verification identity mismatch")
    for item in cfg["inputs"]:
        path = Path(item["path"])
        if pinned.get(path) != item["sha256"] or sha(path) != item["sha256"]:
            raise ValueError("verification input is not queue pinned")
    by_name = {item["name"]: Path(item["path"]) for item in cfg["inputs"]}
    original_receipt = json.loads(by_name["restored_receipt"].read_text())
    if original_receipt["status"] != "PASS" or original_receipt["unit_id"] != "R1_RESTORE_B0_J_READER_V3":
        raise ValueError("restored run did not pass")
    if (sha(by_name["snapshot_receipt"]) != original_receipt["snapshot_receipt_sha256"] or
            sha(by_name["restore_receipt"]) != original_receipt["restore_receipt_sha256"]):
        raise ValueError("custody receipts differ")
    checks = {}
    b0_prob = {}
    for seed in (0, 1):
        table = json.loads(by_name[f"table_s{seed}"].read_text())["families"]["W_TSTAR"]
        for arm in ("B0", "J"):
            prior = json.loads(by_name[f"original_s{seed}_{arm}"].read_text())
            cached = exact_npz(by_name[f"cached_s{seed}_{arm}"],
                               {"row_id", "unit", "code_r1", "code_r2"})
            if prior["status"] != "PASS" or prior["arm_id"] != arm or prior["seed"] != seed:
                raise ValueError("original deployment receipt differs")
            indices = np.linspace(0, 15433, 32, dtype=np.int64)
            for purpose in (0, 1):
                key = f"S{seed}_{arm}_R{purpose}"
                path = by_name[f"wire_s{seed}_{arm}_r{purpose}"]
                expected = original_receipt["deployment_wires"][key]["wire_sha256"]
                if sha(path) != expected or expected != prior["wire_by_recipient"][str(purpose)]["release_sha256"]:
                    raise ValueError("restored wire byte mismatch")
                wire = exact_npz(path, {"code", "prob", "decision"})
                ids = decode(wire["code"])
                bank = np.asarray(table[f"table{purpose}"], dtype=np.float64)
                if (wire["prob"].dtype != np.float64 or
                        wire["prob"].shape != (32, (2, 6)[purpose]) or
                        wire["decision"].dtype != np.int64 or
                        wire["decision"].shape != (32,) or
                        not np.array_equal(ids, cached[f"code_r{purpose+1}"][indices]) or
                        not np.array_equal(wire["prob"], bank[ids]) or
                        not np.array_equal(wire["decision"], bank[ids].argmax(axis=1))):
                    raise ValueError("restored wire numerical mismatch")
                checks[key] = {"wire_sha256": expected, "rows": 32,
                               "cached_code_parity": True, "fixed_bank_decode": "PASS"}
                if seed == 0 and arm == "B0":
                    b0_prob[purpose] = wire["prob"]
    model_path = by_name["reader_model"]
    if sha(model_path) != original_receipt["reader"]["model_sha256"]:
        raise ValueError("reader model hash differs")
    reader = joblib.load(model_path)
    x = np.concatenate([b0_prob[0], b0_prob[1]], axis=1)
    prediction = np.asarray(reader.predict_proba(x)[:, 1], dtype=np.float64)
    prior_prediction = exact_npz(by_name["stored_prediction"], {"prediction"})["prediction"]
    if (x.shape != (32, 8) or prediction.shape != (32,) or
            not np.isfinite(prediction).all() or
            np.any((prediction < 0) | (prediction > 1)) or
            not np.array_equal(prediction, prior_prediction)):
        raise ValueError("independent reader prediction mismatch")
    result = {"schema": "pcrl-bank-r5-independent-restored-numerical-v1",
              "status": "PASS", "unit_id": active["unit_id"],
              "r1_receipt_sha256": sha(by_name["restored_receipt"]),
              "wire_count": len(checks), "wire_checks": checks,
              "reader_input_shape": [32, 8],
              "reader_prediction_count": 32,
              "reader_prediction_max_abs_gap": float(np.max(np.abs(prediction - prior_prediction))),
              "reader_prediction_sha256": sha(by_name["stored_prediction"]),
              "no_labels_accessed": True}
    destination = Path(cfg["public_receipt"])
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))


if __name__ == "__main__":
    main()

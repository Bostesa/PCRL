"""Queue-only, independent raw bank deployment parity check.

The production CLI is exercised as a black box. Its score router and decoder
are never imported for the independent code/table checks below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env


ROOT = Path(__file__).resolve().parent
WIRE = {"row_id", "unit", "code0", "prob0", "decision0", "code1", "prob1", "decision1"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(record: dict, registered: dict[Path, str]) -> Path:
    if set(record) != {"path", "sha256"}:
        raise PermissionError("source pin schema differs")
    raw = Path(record["path"])
    if raw.is_symlink():
        raise PermissionError("source symlink refused")
    path = raw.resolve(strict=True)
    if registered.get(path) != record["sha256"] or digest(path) != record["sha256"]:
        raise PermissionError("source not queue pinned")
    return path


def decode_public(code: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (code.dtype != np.uint8 or code.ndim != 2 or code.shape[1] != 10
            or not np.isin(code, (0, 1)).all()):
        raise ValueError("bank code is not 10 hard bits")
    token = (code.astype(np.int64) * (1 << np.arange(10, dtype=np.int64))).sum(axis=1)
    if np.any(token >= len(table)):
        raise ValueError("bank token outside public table")
    prob = table[token]
    return prob, prob.argmax(axis=1).astype(np.int64)


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    config_path = config_path.resolve(strict=True)
    if ".private" not in config_path.parts or stat.S_IMODE(config_path.stat().st_mode) & 0o077:
        raise PermissionError("owner-only config required")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    registered = {Path(r["path"]).resolve(strict=True): r["sha256"]
                  for r in state["units"][active["unit_id"]]["spec"]["sources"]}
    if registered.get(config_path) != digest(config_path):
        raise PermissionError("config not queued")
    c = json.loads(config_path.read_text())
    if (c.get("schema") != "pcrl-r5-bank-deployment-config-v1"
            or c.get("unit_id") != active["unit_id"] or c.get("seed") != 1):
        raise PermissionError("bank deployment unit identity differs")
    paths = {key: pin(value, registered) for key, value in c["pins"].items()}
    lock = json.loads(paths["campaign_lock"].read_text())
    if c["pins"]["campaign_lock"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("campaign seal differs")
    if c["pins"]["role_manifest"]["sha256"] != lock["role_manifest_sha256"]:
        raise PermissionError("role manifest differs")
    sample_receipt = json.loads(paths["sample_receipt"].read_text())
    if (sample_receipt.get("status") != "PASS" or sample_receipt.get("rows") != 32
            or sample_receipt.get("raw_samples_sha256") != c["pins"]["raw_sample"]["sha256"]
            or sample_receipt.get("source_model_sha256")
               != lock["source_model_files_sha256_by_seed"]["1"]["model_pt_sha256"]):
        raise PermissionError("sample lineage differs")
    with np.load(paths["raw_sample"], allow_pickle=False) as data:
        if set(data.files) != {"X", "feature_names"}:
            raise ValueError("raw sample inventory differs")
        names = tuple(str(x) for x in data["feature_names"])
        if data["X"].shape != (32, 83) or len(names) != 83:
            raise ValueError("raw sample schema differs")
    bank_receipt = json.loads(paths["bank_receipt"].read_text())
    if (bank_receipt.get("status") != "COMPLETE" or bank_receipt.get("seed") != 1
            or bank_receipt.get("role_manifest_sha256")
               != c["pins"]["role_manifest"]["sha256"]
            or bank_receipt.get("source_model_sha256")
               != sample_receipt["source_model_sha256"]):
        raise PermissionError("bank producer differs")
    public = json.loads(paths["public_table"].read_text())
    if (public.get("seed") != 1 or public.get("bit_count") != 10
            or public.get("source_model_sha256") != sample_receipt["source_model_sha256"]
            or public.get("feature_schema_sha256") != sample_receipt["source_schema_sha256"]):
        raise PermissionError("bank public table lineage differs")
    indices = np.linspace(0, 2234, num=32, dtype=np.int64)
    expected = {}
    for bank in ("W_Q", "W_TSTAR"):
        record = bank_receipt["result"]["roles"]["INNER_SELECTION"]["arms"][bank]
        source_key = f"saved_{bank}"
        if (record["private_release_sha256"] != c["pins"][source_key]["sha256"]
                or record["public_table_sha256"] != c["pins"]["public_table"]["sha256"]):
            raise PermissionError("bank release/table differs from producer")
        with np.load(paths[source_key], allow_pickle=False) as archive:
            if set(archive.files) != WIRE or len(archive["row_id"]) != 2235:
                raise ValueError("saved bank wire inventory differs")
            expected[bank] = {key: np.array(archive[key][indices], copy=True)
                              for key in WIRE - {"row_id", "unit"}}
    max_gap = 0.0
    with TemporaryDirectory(dir=ROOT / ".private") as temp:
        temp = Path(temp)
        schema = temp / "schema.json"
        schema.write_text(json.dumps(list(names)))
        for bank in ("W_Q", "W_TSTAR"):
            for purpose in (0, 1):
                assert_active_from_env(unit_id=active["unit_id"], stage="verification")
                output = temp / f"{bank}_{purpose}.npz"
                command = [c["python"], "-m", "code_witness.bank_deploy",
                           "--bank", bank, "--purpose", str(purpose), "--seed", "1",
                           "--source-root", c["source_root"],
                           "--source-sha256", sample_receipt["source_model_sha256"],
                           "--qpc-root", c["qpc_root"],
                           "--public-table", str(paths["public_table"]),
                           "--public-table-sha256", c["pins"]["public_table"]["sha256"],
                           "--schema", str(schema),
                           "--schema-sha256", sample_receipt["source_schema_sha256"],
                           "--input", str(paths["raw_sample"]), "--output", str(output)]
                subprocess.run(command, cwd=c["cwd"], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                with np.load(output, allow_pickle=False) as archive:
                    if set(archive.files) != {"code", "prob", "decision"}:
                        raise ValueError("bank CLI output inventory differs")
                    code, prob, decision = (np.asarray(archive[k])
                                            for k in ("code", "prob", "decision"))
                saved = expected[bank]
                table = np.asarray(public["families"][bank][f"table{purpose}"],
                                   dtype=np.float64)
                decoded_prob, decoded_decision = decode_public(code, table)
                if (not np.array_equal(code, saved[f"code{purpose}"])
                        or not np.array_equal(prob, saved[f"prob{purpose}"])
                        or not np.array_equal(decision, saved[f"decision{purpose}"])
                        or not np.array_equal(prob, decoded_prob)
                        or not np.array_equal(decision, decoded_decision)):
                    raise ValueError("raw bank CLI differs from saved/public decoded wire")
                max_gap = max(max_gap, float(np.max(np.abs(prob - saved[f"prob{purpose}"]))))
    private = Path(c["private_output"]).resolve(strict=False)
    if private.exists() or private.parent != ROOT / ".private":
        raise PermissionError("new private bank wire output required")
    fd = os.open(private, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **{f"{bank}_{key}": value
                                        for bank, wire in expected.items()
                                        for key, value in wire.items()})
        stream.flush()
        os.fsync(stream.fileno())
    result = {"schema": "pcrl-r5-bank-raw-deployment-receipt-v1", "status": "PASS",
              "unit_id": active["unit_id"], "seed": 1, "rows": 32,
              "banks": ["W_Q", "W_TSTAR"], "purposes": [0, 1],
              "bit_count": 10, "max_probability_gap": max_gap,
              "exact_saved_and_public_table_parity": True,
              "private_wires_sha256": digest(private),
              "raw_samples_sha256": c["pins"]["raw_sample"]["sha256"],
              "public_table_sha256": c["pins"]["public_table"]["sha256"],
              "scope": "technical raw deployment parity; no task or SEX labels read"}
    receipt = Path(c["public_output"]).resolve(strict=False)
    if receipt.exists() or receipt.parent != ROOT:
        raise PermissionError("new public bank receipt required")
    receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return {"status": "PASS", "receipt_sha256": digest(receipt)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))

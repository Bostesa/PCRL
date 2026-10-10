"""Queue-only fixed-family utility intervals from private INNER cost vectors.

The worker reads no raw scores, task labels, SEX labels or attack predictions.
R5 independently replays its reported numbers from the pinned vector inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from .r3_family import ARMS
from .r3_intervals import (BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FIELDS,
                           K_FULL, REAL_GROUPS, expected_utility_intervals)


CONFIG_FIELDS = {"schema", "unit_id", "stage", "seal", "entries", "outputs"}
ENTRY_FIELDS = {"arm", "seed", "vector", "receipt"}
PIN_FIELDS = {"path", "sha256"}
OUTPUT_FIELDS = {"results", "receipt"}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin_valid(value: object) -> bool:
    return (isinstance(value, dict) and set(value) == PIN_FIELDS
            and isinstance(value["path"], str)
            and isinstance(value["sha256"], str)
            and len(value["sha256"]) == 64)


def validate_config(config: dict, active: dict) -> None:
    """Fix membership, order, scope and custody before any vector is opened."""
    if (set(config) != CONFIG_FIELDS
            or config.get("schema") != "pcrl-bank-r3-interval-job-v1"
            or config.get("unit_id") != active.get("unit_id")
            or config.get("stage") != active.get("stage")
            or config.get("stage") not in ("protection", "verification")
            or not _pin_valid(config.get("seal"))
            or set(config.get("outputs", {})) != OUTPUT_FIELDS
            or not isinstance(config.get("entries"), list)):
        raise PermissionError("R3 interval configuration differs")
    keys = []
    for entry in config["entries"]:
        if (not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS
                or entry.get("arm") not in ARMS or entry.get("seed") not in (0, 1)
                or not _pin_valid(entry.get("vector"))
                or not _pin_valid(entry.get("receipt"))):
            raise PermissionError("R3 interval entry differs")
        keys.append((entry["arm"], entry["seed"]))
    order = {(arm, seed): 2 * index + seed
             for index, arm in enumerate(ARMS) for seed in (0, 1)}
    if (len(keys) != len(set(keys))
            or keys != sorted(keys, key=order.__getitem__)
            or not {("B0", 0), ("B0", 1)}.issubset(keys)):
        raise PermissionError("paired B0 or fixed arm/seed order differs")
    paths = [Path(value) for value in config["outputs"].values()]
    if (len(set(paths)) != len(paths) or any(".private" in path.parts for path in paths)):
        raise PermissionError("interval aggregate output paths differ")


def validate_entry_receipt(entry: dict, receipt: dict, seal_sha256: str) -> None:
    if (receipt.get("schema") != "pcrl-bank-r3-utility-receipt-v1"
            or receipt.get("role") != "INNER_SELECTION"
            or receipt.get("stage") not in ("witness", "protection")
            or receipt.get("arm") != entry["arm"]
            or receipt.get("seed") != entry["seed"]
            or receipt.get("source_seal_sha256") != seal_sha256
            or receipt.get("private_vector_sha256") != entry["vector"]["sha256"]
            or receipt.get("primary_clause_count") != 12):
        raise PermissionError("INNER vector receipt identity differs")


def _pinned(record: dict, sources: dict[Path, str]) -> Path:
    path = Path(record["path"])
    if path.is_symlink():
        raise PermissionError("symlink source refused")
    path = path.resolve(strict=True)
    if sources.get(path) != record["sha256"] or _sha(path) != record["sha256"]:
        raise PermissionError("source differs from active queue pin")
    return path


def _new_file(path: Path):
    if path.is_symlink() or not path.parent.is_dir():
        raise PermissionError("public interval output path differs")
    return os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), "wb")


def run(config_path: Path) -> dict:
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env

    active = assert_active_from_env()
    if config_path.is_symlink():
        raise PermissionError("R3 interval config symlink refused")
    config_path = config_path.resolve(strict=True)
    if (".private" not in config_path.parts
            or stat.S_IMODE(config_path.stat().st_mode) & 0o077):
        raise PermissionError("R3 interval config must be owner-only")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {Path(rec["path"]).resolve(strict=True): rec["sha256"]
               for rec in state["units"][active["unit_id"]]["spec"]["sources"]}
    if sources.get(config_path) != _sha(config_path):
        raise PermissionError("R3 interval config is not queued by hash")
    config = json.loads(config_path.read_text())
    validate_config(config, active)
    if config["seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("campaign seal differs")
    _pinned(config["seal"], sources)
    for source in (Path(__file__), Path(__file__).with_name("r3_intervals.py"),
                   Path(__file__).with_name("r3_family.py")):
        source = source.resolve(strict=True)
        if sources.get(source) != _sha(source):
            raise PermissionError("R3 interval implementation not queued by hash")
    records = {}
    provenance = []
    for entry in config["entries"]:
        receipt_path = _pinned(entry["receipt"], sources)
        receipt = json.loads(receipt_path.read_text())
        validate_entry_receipt(entry, receipt, active["seal_sha256"])
        vector_path = _pinned(entry["vector"], sources)
        if ".private" not in vector_path.parts:
            raise PermissionError("person vector must stay private")
        with np.load(vector_path, allow_pickle=False) as archive:
            if set(archive.files) != FIELDS:
                raise PermissionError("private cost vector inventory differs")
            records[(entry["arm"], entry["seed"])] = {
                key: np.array(archive[key], copy=True) for key in FIELDS}
        provenance.append({"arm": entry["arm"], "seed": entry["seed"],
                           "vector_sha256": entry["vector"]["sha256"],
                           "receipt_sha256": entry["receipt"]["sha256"]})
    intervals = expected_utility_intervals(records)
    if len(intervals) != 360:
        raise AssertionError("fixed utility family count differs")
    counts = {}
    for cell in intervals.values():
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    result = {
        "schema": "pcrl-bank-r3-utility-intervals-v1",
        "source_seal_sha256": active["seal_sha256"],
        "fixed_family_K": K_FULL, "utility_slots": 360,
        "group_count": REAL_GROUPS, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_generator": "PCG64DXSM", "bootstrap_seed": BOOTSTRAP_SEED,
        "scope": "conditional development intervals over reused INNER people; no post-selection confirmation",
        "entries": provenance, "status_counts": counts, "intervals": intervals,
    }
    output = Path(config["outputs"]["results"])
    with _new_file(output) as stream:
        stream.write((json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    receipt_out = {"schema": "pcrl-bank-r3-utility-interval-receipt-v1",
                   "unit_id": active["unit_id"], "source_seal_sha256": active["seal_sha256"],
                   "result_sha256": _sha(output), "entry_count": len(provenance),
                   "estimable_slots": 360 - counts.get("NOT_ESTIMABLE_DEPENDENCY", 0),
                   "status_counts": counts}
    with _new_file(Path(config["outputs"]["receipt"])) as stream:
        stream.write((json.dumps(receipt_out, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return receipt_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), sort_keys=True))


if __name__ == "__main__":
    main()

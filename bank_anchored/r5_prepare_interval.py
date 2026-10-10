"""Receipt-only queue builder for independent fixed-family interval replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMP = ROOT / "results/pcrl_bank_anchored_release_v1"
R3 = CAMP / "r3"
R5 = CAMP / "r5"
UID = "R5_REPLAY_R3_UTILITY_INTERVAL_G1_V1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("interval replay source symlink refused")
    source = path.resolve(strict=True)
    if not source.is_file():
        raise PermissionError("interval replay source is not a file")
    return {"path": str(source), "sha256": sha(source)}


def write_once(path: Path, value: dict, mode: int) -> str:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != raw:
            raise FileExistsError("interval replay spec differs from existing bytes")
        return sha(path)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def build(amendment_path: Path, python_entry: Path) -> dict:
    state = json.loads((CAMP / "r0/.private/compute/state.json").read_text())
    amendment_pin = pin(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    latest = [item["sha256"] for item in state["amendments"]
              if "verification" in item["stages"]][-1]
    if (amendment.get("schema") != "pcrl-bank-dated-amendment-v1" or
            "verification" not in amendment.get("affected_stages", ()) or
            latest != amendment_pin["sha256"] or
            state["units"].get("R3_UTILITY_INTERVAL_G1_V1", {}).get("status") != "done"):
        raise PermissionError("R5 interval replay lacks dated verification admission")
    if not python_entry.is_file() or not os.access(python_entry, os.X_OK):
        raise PermissionError("scientific Python entrypoint missing")
    config = R3 / ".private/R3_UTILITY_INTERVAL_G1_V1_config.json"
    result = R3 / "R3_UTILITY_INTERVAL_G1_V1.json"
    receipt = R3 / "R3_UTILITY_INTERVAL_G1_V1_RECEIPT.json"
    source_spec = R3 / ".private/R3_UTILITY_INTERVAL_G1_V1_queue_spec.json"
    r3_queue = json.loads(source_spec.read_text())
    r3_result = json.loads(result.read_text())
    r3_receipt = json.loads(receipt.read_text())
    if (r3_queue.get("unit_id") != "R3_UTILITY_INTERVAL_G1_V1" or
            r3_receipt.get("unit_id") != r3_queue["unit_id"] or
            r3_receipt.get("result_sha256") != sha(result) or
            r3_result.get("fixed_family_K") != 1575 or
            r3_result.get("utility_slots") != 360):
        raise PermissionError("completed R3 interval result identity differs")
    for item in r3_queue["sources"]:
        if pin(Path(item["path"])) != item:
            raise PermissionError("R3 interval source changed before R5 replay")
    files = [*r3_queue["sources"], *(pin(path) for path in (
        config, result, receipt, source_spec, amendment_path,
        Path(__file__), ROOT / "bank_anchored/r5_replay_intervals.py",
        CAMP / "r0/compute_queue.py", python_entry.resolve(strict=True)))]
    sources: dict[str, dict[str, str]] = {}
    for item in files:
        if item["path"] in sources and sources[item["path"]] != item:
            raise PermissionError("conflicting R5 interval source pin")
        sources[item["path"]] = item
    private = R5 / ".private"
    if private.is_symlink() or private.stat().st_mode & 0o077:
        raise PermissionError("R5 interval spec parent is not owner-only")
    output = R5 / f"{UID}_RECEIPT.json"
    spec = {"unit_id": UID, "stage": "verification",
            "amendment_sha256": amendment_pin["sha256"],
            "argv": [str(python_entry), "-m", "bank_anchored.r5_replay_intervals",
                     "--config", str(config.resolve()), "--config-sha256", sha(config),
                     "--results", str(result.resolve()), "--results-sha256", sha(result),
                     "--r3-receipt", str(receipt.resolve()),
                     "--r3-receipt-sha256", sha(receipt), "--output", str(output)],
            "cwd": str(ROOT), "env": {}, "deps": ["R3_UTILITY_INTERVAL_G1_V1"],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2.0,
            "projected_output_bytes": 100000, "outputs": [str(output)],
            "sources": list(sources.values())}
    spec_sha = write_once(private / f"{UID}_QUEUE_SPEC.json", spec, 0o600)
    index = {"schema": "BAR1-r5-interval-replay-unit-index-v1", "unit_id": UID,
             "r3_result_sha256": sha(result), "r3_receipt_sha256": sha(receipt),
             "r3_queue_spec_sha256": sha(source_spec),
             "verification_amendment_sha256": amendment_pin["sha256"],
             "source_count": len(sources), "queue_spec_sha256": spec_sha}
    write_once(R5 / f"{UID}_SPEC_INDEX.json", index, 0o644)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.amendment, args.python), sort_keys=True))


if __name__ == "__main__":
    main()

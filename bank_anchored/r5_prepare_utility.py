"""Prepare hash-pinned queue replay for one completed R3 protected utility unit.

This builder reads receipts and source bytes only. It never opens a role archive,
law member or protected label. R0 owns queue registration and stage admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .r5_utility_links import validate_links


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/pcrl_bank_anchored_release_v1"
R3 = CAMPAIGN / "r3"
R5 = CAMPAIGN / "r5"
UNIT_PATTERN = re.compile(r"R3_(T|D|P|L|J|S12|S21)_S[01]_(FIT|INNER)(?:[-_][A-Za-z0-9]+)*")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("verification source symlink refused")
    absolute = path.resolve(strict=True)
    if not absolute.is_file():
        raise PermissionError("verification source is not a regular file")
    return {"path": str(absolute), "sha256": sha(absolute)}


def _unique_source(spec: dict, expected_sha: str) -> Path:
    matches = [Path(item["path"]) for item in spec["sources"]
               if item["sha256"] == expected_sha]
    if len(matches) != 1:
        raise PermissionError("selected source hash is absent or ambiguous")
    return matches[0]


def verify_provenance(index: dict, config: dict, spec: dict, r3: dict,
                      release: dict, policy: dict, unit: dict, *,
                      config_sha: str, spec_sha: str, csv_sha: str) -> dict:
    if (index.get("schema") != "pcrl-bank-r3-protection-unit-index-v1" or
            index.get("unit_id") != config.get("unit_id") or
            index.get("unit_id") != spec.get("unit_id") or
            index.get("arm") != config.get("arm") or
            index.get("seed") != config.get("seed") or
            index.get("role") != config.get("role") or
            index.get("config_sha256") != config_sha or
            index.get("queue_spec_sha256") != spec_sha or
            index.get("r2_law_sha256") != config.get("law", {}).get("sha256") or
            index.get("r2_selected_unit_id") != unit.get("unit_id") or
            unit.get("unit_id") not in spec.get("deps", ()) or
            index.get("source_binding_sha256") != unit.get("source_binding_sha256") or
            config.get("seal", {}).get("sha256") != unit.get("source_lock_sha256") or
            spec.get("stage") != "protection"):
        raise PermissionError("R3 indexed selected-source provenance differs")
    for name, digest in (("r2_law_sha256", config["law"]["sha256"]),
                         ("r2_release_receipt_sha256", index["r2_release_receipt_sha256"]),
                         ("r2_policy_receipt_sha256", index["r2_policy_receipt_sha256"]),
                         ("r2_selected_unit_receipt_sha256", index["r2_selected_unit_receipt_sha256"])):
        if index[name] != digest or sum(item.get("sha256") == digest
                                        for item in spec.get("sources", ())) != 1:
            raise PermissionError("selected source is not uniquely pinned in R3 spec")
    if sum(item.get("sha256") == config_sha
           for item in spec.get("sources", ())) != 1:
        raise PermissionError("R3 config is not uniquely pinned in R3 spec")
    return validate_links(config, r3, release, policy, unit,
                          csv_sha=csv_sha,
                          release_sha=index["r2_release_receipt_sha256"],
                          policy_sha=index["r2_policy_receipt_sha256"])


def _write_once(path: Path, payload: dict, mode: int) -> str:
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.read_bytes() != raw:
            raise FileExistsError("immutable independent replay spec differs")
        return sha(path)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def build(index_path: Path, amendment_path: Path, python_entry: Path) -> dict:
    index_path = index_path.resolve(strict=True)
    if index_path.parent != R3 or index_path.is_symlink():
        raise PermissionError("R3 index custody differs")
    index_pin = pin(index_path)
    index = json.loads(index_path.read_text())
    unit_id = index.get("unit_id", "")
    if not UNIT_PATTERN.fullmatch(unit_id):
        raise PermissionError("R3 protected unit ID differs")
    config_path = R3 / ".private" / f"{unit_id}_config.json"
    spec_path = R3 / ".private" / f"{unit_id}_queue_spec.json"
    receipt_path = R3 / f"{unit_id}_RECEIPT.json"
    csv_path = R3 / f"{unit_id}.csv"
    config_pin, spec_pin, receipt_pin, csv_pin = map(
        pin, (config_path, spec_path, receipt_path, csv_path))
    if (config_pin["sha256"] != index.get("config_sha256") or
            spec_pin["sha256"] != index.get("queue_spec_sha256")):
        raise PermissionError("R3 index does not bind completed config/spec bytes")
    config = json.loads(config_path.read_text())
    spec = json.loads(spec_path.read_text())
    if (not isinstance(spec.get("sources"), list) or
            not spec["sources"] or
            any(pin(Path(item["path"])) != item for item in spec["sources"])):
        raise PermissionError("R3 source pins changed before independent replay")
    release_path = _unique_source(spec, index["r2_release_receipt_sha256"])
    policy_path = _unique_source(spec, index["r2_policy_receipt_sha256"])
    selected_unit_path = _unique_source(spec, index["r2_selected_unit_receipt_sha256"])
    result = verify_provenance(
        index, config, spec, json.loads(receipt_path.read_text()),
        json.loads(release_path.read_text()), json.loads(policy_path.read_text()),
        json.loads(selected_unit_path.read_text()),
        config_sha=config_pin["sha256"], spec_sha=spec_pin["sha256"],
        csv_sha=csv_pin["sha256"])
    amendment_pin = pin(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    if (amendment.get("schema") != "pcrl-bank-dated-amendment-v1" or
            "verification" not in amendment.get("affected_stages", ()) or
            amendment.get("source_lock_sha256") != config["seal"]["sha256"]):
        raise PermissionError("verification stage amendment differs")
    entry = python_entry.absolute()
    if not entry.is_file() or not os.access(entry, os.X_OK):
        raise PermissionError("verification interpreter is not executable")
    smoke = subprocess.run(
        [str(entry), "-c", "import numpy; import bank_anchored.r5_replay_utility"],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False)
    if smoke.returncode:
        raise RuntimeError("exact verification interpreter cannot import worker")
    private = R5 / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    if private.stat().st_mode & 0o077:
        raise PermissionError("R5 private spec parent is not owner-only")
    replay_id = f"R5_VERIFY_{unit_id}"
    output = R5 / f"{replay_id}_RECEIPT.json"
    sources = [*spec["sources"], spec_pin, index_pin, receipt_pin, csv_pin,
               amendment_pin]
    for source in (Path(__file__), ROOT / "bank_anchored/r5_replay_utility.py",
                   ROOT / "bank_anchored/r5_utility_links.py",
                   ROOT / "bank_anchored/r5_independent.py",
                   CAMPAIGN / "r0/compute_queue.py"):
        sources.append(pin(source))
    by_path = {}
    for item in sources:
        if item["path"] in by_path and by_path[item["path"]] != item:
            raise PermissionError("same source path has conflicting bytes")
        by_path[item["path"]] = item
    queue_spec = {
        "unit_id": replay_id, "stage": "verification",
        "argv": [str(entry), "-m", "bank_anchored.r5_replay_utility",
                 "--config", str(config_path), "--config-sha256", config_pin["sha256"],
                 "--r3-receipt", str(receipt_path),
                 "--r3-receipt-sha256", receipt_pin["sha256"],
                 "--output", str(output)],
        "cwd": str(ROOT), "env": {}, "max_wall_s": 120,
        "cpu_reserve_s": 120, "max_rss_gib": 1.0,
        "projected_output_bytes": 100000,
        "deps": [unit_id], "outputs": [str(output)],
        "sources": list(by_path.values()),
        "amendment_sha256": amendment_pin["sha256"],
    }
    spec_out = private / f"{replay_id}_QUEUE_SPEC.json"
    spec_sha = _write_once(spec_out, queue_spec, 0o600)
    public = {"schema": "BAR1-r5-utility-replay-unit-index-v1",
              "unit_id": replay_id, "source_r3_unit_id": unit_id,
              "source_r2_unit_id": result["r2_unit_id"],
              "arm_id": result["arm_id"], "source_seed": result["source_seed"],
              "role": result["role"], "law_sha256": result["law_sha256"],
              "r3_index_sha256": index_pin["sha256"],
              "r3_receipt_sha256": receipt_pin["sha256"],
              "r3_csv_sha256": csv_pin["sha256"],
              "queue_spec_sha256": spec_sha,
              "verification_amendment_sha256": amendment_pin["sha256"],
              "source_count": len(by_path)}
    _write_once(R5 / f"{replay_id}_SPEC_INDEX.json", public, 0o644)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--r3-index", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.r3_index, args.amendment, args.python), sort_keys=True))


if __name__ == "__main__":
    main()

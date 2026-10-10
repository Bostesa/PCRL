"""Bind an immutable R1 replay spec to a later verification amendment.

The original statically audited config and spec remain untouched. R0 registers
the new spec only after the amendment is sealed and R5 has audited admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _new_json(path: Path, value: dict, *, private: bool) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush(); os.fsync(out.fileno())


def version(root: Path, *, arm_id: str, seed: int, amendment: Path,
            amendment_sha256: str) -> tuple[Path, dict]:
    root = root.resolve(strict=True)
    pkg = root / "results/pcrl_bank_anchored_release_v1"
    r1 = pkg / "r1"
    private = r1 / ".private"
    if (arm_id not in ("B0", "J") or seed not in (0, 1) or
            stat.S_IMODE(private.stat().st_mode) & 0o077 or
            re.fullmatch(r"[0-9a-f]{64}", amendment_sha256) is None):
        raise ValueError("versioned R1 unit or private root differs")
    amendment = amendment.resolve(strict=True)
    if (not amendment.is_file() or not amendment.is_relative_to(pkg / "r0") or
            _sha(amendment) != amendment_sha256):
        raise ValueError("sealed verification amendment source differs")
    unit = f"R1_DEPLOY_S{seed}_{arm_id}"
    original_path = private / f"{unit}_QUEUE_SPEC.json"
    original = json.loads(original_path.read_text())
    original_index = json.loads((r1 / f"{unit}_SPEC_INDEX.json").read_text())
    config = private / f"{unit}_CONFIG.json"
    if (original.get("unit_id") != unit or
            original.get("stage") != "verification" or
            original.get("amendment_sha256") is not None or
            original_index.get("unit_id") != unit or
            original_index.get("private_queue_spec_sha256") != _sha(original_path) or
            original_index.get("private_config_sha256") != _sha(config) or
            original.get("argv", [])[-1] != str(config) or
            any(item["path"] == str(amendment) for item in original["sources"])):
        raise ValueError("original static-approved R1 spec differs")
    successor = {**original, "amendment_sha256": amendment_sha256,
                 "sources": [*original["sources"],
                             {"path": str(amendment), "sha256": amendment_sha256}]}
    spec_path = private / f"{unit}_QUEUE_SPEC_V2.json"
    _new_json(spec_path, successor, private=True)
    index = {"schema": "pcrl-bank-r1-deployment-replay-amended-spec-index-v1",
             "unit_id": unit, "arm_id": arm_id, "seed": seed,
             "original_spec_sha256": _sha(original_path),
             "private_config_sha256": _sha(config),
             "verification_amendment_sha256": amendment_sha256,
             "private_queue_spec_sha256": _sha(spec_path),
             "source_count": len(successor["sources"]),
             "stage": "verification", "status": "AWAITING_R5_ADMISSION_AUDIT"}
    _new_json(r1 / f"{unit}_SPEC_INDEX_V2.json", index, private=False)
    return spec_path, index


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm-id", choices=("B0", "J"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    args = parser.parse_args()
    path, index = version(args.root, arm_id=args.arm_id, seed=args.seed,
                          amendment=args.amendment,
                          amendment_sha256=args.amendment_sha256)
    print(json.dumps({"queue_spec": str(path), "index": index}, sort_keys=True))


if __name__ == "__main__":
    main()

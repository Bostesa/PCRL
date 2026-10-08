"""Independent byte comparison of two versioned role-only exports.

Reads only private export bytes, never mixed-role source members or values.
Emits aggregate counts and hashes, with no person-level content or private path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files(root: Path) -> dict[str, Path]:
    return {str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        first, second = files(args.first), files(args.second)
        if set(first) != set(second) or len(first) != 26:
            raise ValueError("export_file_set_mismatch")
        expected = {"EXPORT_RECEIPT.json"}
        for role, labels in (("OSF_DEFENSE_FIT", ("income", "occupation", "sex")),
                             ("CALIBRATION_HELDOUT", ("income", "occupation")),
                             ("ATTACK_FIT_NEW", ("sex",)),
                             ("INNER_SELECTION", ("income", "occupation", "sex"))):
            expected.update({f"{role}/features.npz", *(f"{role}/{field}.npz" for field in labels),
                             *(f"{role}/references_s{k}.npz" for k in (0, 1, 2))})
        if set(first) != expected:
            raise ValueError("unexpected_export_file")
        hashes = {name: digest(first[name]) for name in sorted(first)}
        if any(hashes[name] != digest(second[name]) for name in hashes):
            raise ValueError("export_byte_mismatch")
        tree_hash = hashlib.sha256("\n".join(f"{name} {hashes[name]}" for name in sorted(hashes)).encode()).hexdigest()
        result = {"schema": "pcrl-s4-independent-export-replay-v1", "status": "PASS",
                  "file_count": 26, "role_npz_count": 25, "first_second_byte_identical": True,
                  "receipt_sha256": hashes["EXPORT_RECEIPT.json"], "canonical_tree_sha256": tree_hash,
                  "scope": "two queued executions of the same reviewed exporter on the same pinned mixed source; S4 compared only exported bytes"}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except BaseException as error:
        print("S4_EXPORT_REPLAY_FAILED:" + type(error).__name__, file=sys.stderr)
        return 1
    print("S4_EXPORT_REPLAY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

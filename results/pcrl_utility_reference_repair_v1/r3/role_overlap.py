"""Independent identity-only overlap check for admitted role-only exports.

Only ZIP members row_id.npy and unit.npy are decompressed. No X or label
member is opened, and no person ID is written to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROLES = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "INNER_SELECTION")
MANIFEST_SHA = "c228dfdbef8a8e0a46281716e89c986e888fb4f64553fc35b3573020c0c83a11"


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def identity_only(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with ZipFile(path) as archive:
        if set(archive.namelist()) != {"X.npy", "feature_names.npy", "row_id.npy", "unit.npy"}:
            raise ValueError("role feature archive member schema changed")
        row = np.load(io.BytesIO(archive.read("row_id.npy")), allow_pickle=False)
        unit = np.load(io.BytesIO(archive.read("unit.npy")), allow_pickle=False)
    if (row.dtype != np.int64 or unit.dtype != np.int64 or row.ndim != 1
            or unit.shape != row.shape or len(np.unique(row)) != len(row)):
        raise ValueError("role identity shape, type or uniqueness changed")
    return row, unit


def audit(portfolio_package: Path) -> dict:
    package = portfolio_package.resolve(strict=True)
    manifest_path = package / "DATA_ACCESS_MANIFEST.json"
    if file_sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("admitted role manifest bytes differ")
    manifest = json.loads(manifest_path.read_text())
    identities = {}
    receipts = {}
    for role in ROLES:
        spec = manifest["real_file_specs"][f"{role}:features"]
        rel = Path(spec["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("role archive path escapes package")
        path = (package / rel).resolve(strict=True)
        if not path.is_relative_to(package) or path.name != "features.npz":
            raise ValueError("unexpected role archive")
        actual_sha = file_sha(path)
        if actual_sha != spec["sha256"]:
            raise ValueError("role archive byte identity differs")
        row, unit = identity_only(path)
        expected = manifest["roles"][role]
        if (len(row) != expected["rows"] or len(np.unique(unit)) != expected["groups"]
                or array_sha(np.sort(row)) != expected["row_id_sha256"]
                or array_sha(np.unique(unit)) != expected["group_id_sha256"]
                or array_sha(np.column_stack((row, unit)).astype(np.int64)) !=
                spec["row_unit_pair_sha256"]):
            raise ValueError("role membership differs from admitted receipt")
        identities[role] = (row, unit)
        receipts[role] = {"rows": len(row), "groups": len(np.unique(unit)),
                          "feature_archive_sha256": actual_sha,
                          "row_id_sha256": expected["row_id_sha256"],
                          "group_id_sha256": expected["group_id_sha256"],
                          "row_unit_pair_sha256": spec["row_unit_pair_sha256"]}
    overlaps = {}
    for i, left in enumerate(ROLES):
        for right in ROLES[i + 1:]:
            lr, lu = identities[left]
            rr, ru = identities[right]
            overlaps[f"{left}__{right}"] = {
                "rows": int(len(np.intersect1d(lr, rr))),
                "groups": int(len(np.intersect1d(lu, ru))),
            }
    return {"schema": "pcrl-repair-r3-role-overlap-v1",
            "scope": "role-only row_id/unit members; X and label members not decompressed",
            "portfolio_manifest_sha256": MANIFEST_SHA,
            "role_identity": receipts, "overlap": overlaps,
            "direct_common_role_disjoint": all(not v["rows"] and not v["groups"]
                                               for v in overlaps.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("portfolio_package", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = audit(args.portfolio_package)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

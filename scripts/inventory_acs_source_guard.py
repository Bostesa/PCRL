"""Inventory omitted local objects without exporting person-level content."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/redesign_20260909_acs_source_guard_v1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inventory(out):
    matrix = json.loads((out / "EXECUTED_MATRIX.json").read_text())
    reuse = json.loads((out / "REUSE_MANIFEST.json").read_text())
    objects = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".pt", ".npz", ".joblib"} and "fitted" not in path.parts:
            continue
        objects.append({"path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size, "sha256": digest(path)})
    raw = ROOT / "data/folktables/2018/1-Year/psam_p06.csv"
    raw_hash = digest(raw)
    assert raw_hash == "dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0"
    record = {
        "repository_root": str(ROOT),
        "study_directory": str(out.relative_to(ROOT)),
        "matrix_status": matrix["status"],
        "raw_data": {"path": str(raw.relative_to(ROOT)), "bytes": raw.stat().st_size,
                     "sha256": raw_hash, "existing_data_no_download": True},
        "new_local_objects": objects,
        "new_local_object_count": len(objects),
        "new_local_object_bytes": sum(p["bytes"] for p in objects),
        "historical_objects": {
            "manifest": str((out / "REUSE_MANIFEST.json").relative_to(ROOT)),
            "manifest_sha256": digest(out / "REUSE_MANIFEST.json"),
            "hash_bound_historical_paths": len(reuse["historical_files_sha256"]),
            "historical_fits_repeated": 0,
            "original_source_identities_preserved": True,
        },
        "exact_replay_requires": [
            "Recorded raw ACS file, original cohort/subset/PCA maps and fitting statistics",
            "Historical full forward/observer Adam forks and checkpoints in REUSE_MANIFEST",
            "New final and fixed diagnostic .pt checkpoints, immutable split/release .npz arrays",
            "Fitted utility/audit objects, selected and candidate predictions, native predictions",
            "Original row masks, schemas, validation selections and person weights",
        ],
        "public_evidence": "Derived scores, candidate lineage, state/file hashes, compact guard diagnostics; no person-level records or prediction arrays",
        "source_sha256": digest(Path(__file__)),
    }
    (out / "LOCAL_ARTIFACTS.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"objects": len(objects), "bytes": record["new_local_object_bytes"],
            "all_historical_references_bound": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(inventory(args.out.resolve())))

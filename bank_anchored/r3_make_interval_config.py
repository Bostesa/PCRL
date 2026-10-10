"""Hash-only builder for one queued fixed-K R3 utility interval report.

The private manifest names completed INNER vector and receipt paths. This
builder hashes vector bytes opaquely; only the queued worker opens members.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .r3_family import ARMS
from .r3_interval_worker import validate_config, validate_entry_receipt
from .r3_make_witness_configs import ROOT, OUT, record, write_new


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def derive_documents(*, unit_id: str, stage: str, seal_record: dict,
                     entries: list[dict], config_path: str, outputs: dict,
                     sources: list[dict], amendment_record: dict,
                     deps: list[str]) -> tuple[dict, dict]:
    if (not unit_id or stage not in ("protection", "verification")
            or not deps or len(deps) != len(set(deps))
            or len(sources) != len({item["path"] for item in sources})
            or not all(isinstance(item.get("sha256"), str) and
                       len(item["sha256"]) == 64 for item in sources)):
        raise PermissionError("interval queue dependencies or sources differ")
    cfg = {"schema": "pcrl-bank-r3-interval-job-v1", "unit_id": unit_id,
           "stage": stage, "seal": seal_record, "entries": entries,
           "outputs": outputs}
    validate_config(cfg, {"unit_id": unit_id, "stage": stage})
    config_rec = {"path": config_path,
                  "sha256": hashlib.sha256(_bytes(cfg)).hexdigest()}
    all_sources = [config_rec, seal_record, *sources]
    if amendment_record not in all_sources:
        all_sources.append(amendment_record)
    spec = {"unit_id": unit_id, "stage": stage,
            "argv": [sys.executable, "-m", "bank_anchored.r3_interval_worker",
                     "--config", config_path],
            "cwd": str(ROOT), "max_wall_s": 120,
            "cpu_reserve_s": 120, "max_rss_gib": 1.0,
            "projected_output_bytes": 1_000_000,
            "deps": deps, "outputs": list(outputs.values()),
            "env": {}, "sources": all_sources,
            "amendment_sha256": amendment_record["sha256"]}
    return cfg, spec


def build(*, manifest: Path, seal: Path, amendment: Path,
          unit_id: str, stage: str) -> dict:
    manifest_record = record(manifest)
    if ".private" not in Path(manifest_record["path"]).parts:
        raise PermissionError("interval source manifest must be owner-only")
    document = json.loads(Path(manifest_record["path"]).read_text())
    if (set(document) != {"schema", "entries"}
            or document["schema"] != "pcrl-bank-r3-interval-sources-v1"
            or not isinstance(document["entries"], list)):
        raise PermissionError("interval source manifest inventory differs")
    seal_record = record(seal)
    amendment_record = record(amendment)
    source_records = [record(ROOT / f"bank_anchored/{name}") for name in
                      ("r3_interval_worker.py", "r3_intervals.py", "r3_family.py")]
    entries, deps = [], []
    for item in document["entries"]:
        if set(item) != {"arm", "seed", "receipt_path", "vector_path"}:
            raise PermissionError("interval source entry differs")
        arm, seed = item["arm"], item["seed"]
        if arm not in ARMS or seed not in (0, 1):
            raise PermissionError("interval arm/seed differs")
        receipt_record = record(Path(item["receipt_path"]))
        vector_record = record(Path(item["vector_path"]))
        if ".private" not in Path(vector_record["path"]).parts:
            raise PermissionError("person vector source must be private")
        receipt = json.loads(Path(receipt_record["path"]).read_text())
        entry = {"arm": arm, "seed": seed,
                 "receipt": receipt_record, "vector": vector_record}
        validate_entry_receipt(entry, receipt, seal_record["sha256"])
        entries.append(entry)
        deps.append(receipt["unit_id"])
        source_records.extend((receipt_record, vector_record))
    private = OUT / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)
    config_path = private / f"{unit_id}_config.json"
    results_path = OUT / f"{unit_id}.json"
    receipt_path = OUT / f"{unit_id}_RECEIPT.json"
    outputs = {"results": str(results_path), "receipt": str(receipt_path)}
    # Deduplicate source files while preserving the registration order.
    unique = {item["path"]: item for item in (manifest_record, *source_records)}
    cfg, spec = derive_documents(unit_id=unit_id, stage=stage,
                                 seal_record=seal_record, entries=entries,
                                 config_path=str(config_path), outputs=outputs,
                                 sources=list(unique.values()),
                                 amendment_record=amendment_record,
                                 deps=deps)
    cfg_sha = write_new(config_path, _bytes(cfg), mode=0o600)
    if cfg_sha != spec["sources"][0]["sha256"]:
        raise AssertionError("interval config byte digest differs")
    spec_path = private / f"{unit_id}_queue_spec.json"
    spec_sha = write_new(spec_path, _bytes(spec), mode=0o600)
    index = {"schema": "pcrl-bank-r3-interval-unit-index-v1",
             "unit_id": unit_id, "stage": stage, "entry_count": len(entries),
             "entry_ids": [{"arm": x["arm"], "seed": x["seed"]} for x in entries],
             "config_sha256": cfg_sha, "queue_spec_sha256": spec_sha,
             "source_manifest_sha256": manifest_record["sha256"],
             "source_seal_sha256": seal_record["sha256"],
             "amendment_sha256": amendment_record["sha256"],
             "dependencies": deps}
    index_path = OUT / f"{unit_id}_UNIT_INDEX.json"
    index_sha = write_new(index_path, _bytes(index), mode=0o644)
    return {**index, "index_sha256": index_sha}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--stage", choices=("protection", "verification"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(manifest=args.manifest, seal=args.seal,
                           amendment=args.amendment, unit_id=args.unit_id,
                           stage=args.stage), sort_keys=True))


if __name__ == "__main__":
    main()

"""Create four owner-only, byte-pinned B0 utility witness queue units.

This builder reads file hashes and public manifests only. It does not open a
role label or per-person source archive. Run only after R0 writes the first
campaign lock; the generated configs are then immutable queue inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pcrl_bank_anchored_release_v1/r3"
ROLE_SHA = "605bcdc9111f8d228a8526a649aa5cdec34c3f932d798d65e0cdf9672d86c45c"
R5_SHA = "d3998384ce96b5fa20a01beecde5d1e513eb5a8c7d53cd8d698b2cb90366a4ec"
TABLE_SHAS = (
    "23f8406880a983f6cc16cf1fbab2f38236757dfd3e05aca5846b30216fa1e362",
    "19fe88eca92c050375976f3cdcf421b3128766e6337d0e34c2082b0390170b4d",
)
RECEIPT_SHAS = (
    "a63dbf8628aec486dd97c6a79f81b71fe64f6f0348b316992a558efea7c1f5c7",
    "7927d02380dd780cd15030fcef3ed81d31183ca48919dbfcc69b837807fd9616",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path, expected: str | None = None) -> dict:
    path = path.resolve(strict=True)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise PermissionError("witness source digest differs: " + path.name)
    return {"path": str(path), "sha256": digest}


def write_new(path: Path, content: bytes, *, mode: int) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def make(*, seal: Path, repair_root: Path, witness_root: Path) -> dict:
    seal_rec = record(seal)
    if Path(seal_rec["path"]).name != "CAMPAIGN_LOCK.json":
        raise PermissionError("campaign lock filename differs")
    role_rec = record(repair_root / "m1/DATA_ACCESS_MANIFEST.json", ROLE_SHA)
    role_manifest = json.loads(Path(role_rec["path"]).read_text())
    if role_manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise PermissionError("role-only source manifest schema differs")
    r5_rec = record(witness_root / "r5/BANK_OLD_POINT_AUDIT.json", R5_SHA)
    worker_rec = record(ROOT / "bank_anchored/r3_worker.py")
    scorer_rec = record(ROOT / "bank_anchored/r3_utility.py")
    private = OUT / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)
    unit_index = []
    for seed in (0, 1):
        table_rec = record(witness_root / f"r1/W_BANK_PUBLIC_TABLES_S{seed}.json",
                           TABLE_SHAS[seed])
        receipt_rec = record(witness_root / f"r1/W_BANK_S{seed}_R1_RECEIPT.json",
                             RECEIPT_SHAS[seed])
        receipt = json.loads(Path(receipt_rec["path"]).read_text())
        for role, suffix in (("OSF_DEFENSE_FIT", "FIT"),
                             ("INNER_SELECTION", "INNER")):
            unit = f"R3_B0_S{seed}_{suffix}"
            release = receipt["result"]["roles"][role]["arms"]["W_TSTAR"]
            wire_rec = record(witness_root / "r1/.private" /
                              f"W_BANK_S{seed}_R1" / release["private_release_basename"],
                              release["private_release_sha256"])
            archive_recs = []
            for field in ("income", "occupation", f"references_s{seed}"):
                key = f"{role}:{field}"
                archive_recs.append(record(repair_root /
                                           role_manifest["role_file_relative_path"][key],
                                           role_manifest["role_file_sha256"][key]))
            csv_path = OUT / f"{unit}.csv"
            vector_path = private / f"{unit}_vectors.npz"
            receipt_path = OUT / f"{unit}_RECEIPT.json"
            config = {
                "schema": "pcrl-bank-r3-utility-job-v1", "unit_id": unit,
                "stage": "witness", "role": role, "arm": "B0", "seed": seed,
                "seal": seal_rec, "role_manifest": role_rec,
                "public_table": table_rec, "bank_receipt": receipt_rec,
                "bank_wire": wire_rec, "r5_source_replay": r5_rec,
                "law": None,
                "outputs": {"csv": str(csv_path),
                            "private_vectors": str(vector_path),
                            "receipt": str(receipt_path)},
            }
            config_path = private / f"{unit}_config.json"
            config_sha = write_new(config_path,
                                   (json.dumps(config, sort_keys=True, indent=2) + "\n").encode(),
                                   mode=0o600)
            config_rec = {"path": str(config_path), "sha256": config_sha}
            sources = [config_rec, seal_rec, role_rec, table_rec, receipt_rec,
                       wire_rec, r5_rec, worker_rec, scorer_rec, *archive_recs]
            spec = {
                "unit_id": unit, "stage": "witness",
                "argv": [sys.executable, "-m", "bank_anchored.r3_worker",
                         "--config", str(config_path)],
                "cwd": str(ROOT), "max_wall_s": 60, "cpu_reserve_s": 60,
                "max_rss_gib": 1.0, "projected_output_bytes": 8_000_000,
                "deps": [],
                "outputs": [str(csv_path), str(vector_path), str(receipt_path)],
                "env": {}, "sources": sources,
            }
            spec_path = private / f"{unit}_queue_spec.json"
            spec_sha = write_new(spec_path,
                                 (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode(),
                                 mode=0o600)
            unit_index.append({"unit_id": unit, "role": role, "seed": seed,
                               "stage": "witness", "config_sha256": config_sha,
                               "queue_spec_sha256": spec_sha,
                               "source_sha256": {
                                   "campaign_lock": seal_rec["sha256"],
                                   "role_manifest": role_rec["sha256"],
                                   "public_table": table_rec["sha256"],
                                   "bank_receipt": receipt_rec["sha256"],
                                   "bank_wire": wire_rec["sha256"],
                                   "r5_source_replay": r5_rec["sha256"],
                                   "worker": worker_rec["sha256"],
                                   "scorer": scorer_rec["sha256"],
                                   "role_archives": [r["sha256"] for r in archive_recs]},
                               "queue_proposal": {"max_wall_s": 60,
                                                  "cpu_reserve_s": 60,
                                                  "max_rss_gib": 1.0,
                                                  "projected_output_bytes": 8_000_000}})
    index = {"schema": "pcrl-bank-r3-witness-unit-index-v1",
             "scope": "hashes and logical units only; private source paths in owner-only configs",
             "units": unit_index}
    write_new(OUT / "WITNESS_UNIT_INDEX.json",
              (json.dumps(index, sort_keys=True, indent=2) + "\n").encode(), mode=0o644)
    return index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--witness-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = make(seal=args.seal, repair_root=args.repair_root,
                  witness_root=args.witness_root)
    print(json.dumps({"unit_ids": [x["unit_id"] for x in result["units"]],
                      "index_sha256": sha(OUT / "WITNESS_UNIT_INDEX.json")}, sort_keys=True))


if __name__ == "__main__":
    main()

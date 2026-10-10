"""Export label-free fixed historical fine-cell maps for queued public Q work.

Only fixed bank archives are opened. No role tokens, person rows, FIT counts,
labels, assessment arrays, or predecessor all-role loaders are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .d0_source import D0SourceStore, RELEASES
from .pstar_source import PStarSourceStore
from .role_store import sha_array, sha_file
from .tstar_source import TStarSourceStore


ARMS = {
    "T_STAR": None,
    "P_STAR": None,
    "D0_LOCAL": "hcal_local_d0",
    "D0_SEQ12": "hcal_seq12_d0",
    "D0_SEQ21": "lra_seq21",
}


def export(*, portfolio_package: Path, repair_manifest: Path,
           source_audit: Path, historical_audit: Path, cache_root: Path,
           output_root: Path) -> dict:
    output_root = output_root.resolve(strict=True)
    if output_root != Path(__file__).resolve().parent:
        raise ValueError("public fixed maps must stay in the R01 namespace")
    historical = json.loads(historical_audit.read_text())
    tstar_audit = json.loads(source_audit.read_text())
    records = {}
    for arm, release_key in ARMS.items():
        for seed in (0, 1):
            if arm == "T_STAR":
                source = TStarSourceStore(portfolio_package, repair_manifest,
                                          source_audit, seed)
                audit_sha = sha_file(source_audit)
                original = tstar_audit["tstar_banks"][str(seed)]
                receipt_sha = original["receipt_sha256"]
                release_id = "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"
            elif arm == "P_STAR":
                source = PStarSourceStore(portfolio_package, repair_manifest,
                                          historical_audit, cache_root, seed)
                audit_sha = sha_file(historical_audit)
                original = historical["pstar"][f"s{seed}"]
                receipt_sha = original["receipt_sha256"]
                release_id = "U|JOINT|i8o64|l0.1|H-CLASS-TEMP"
            else:
                source = D0SourceStore(portfolio_package, repair_manifest,
                                       historical_audit, cache_root,
                                       release_key, seed)
                audit_sha = sha_file(historical_audit)
                original = historical["d0"][release_key][f"s{seed}"]
                receipt_sha = original["receipt_sha256"]
                release_id = RELEASES[release_key]
            if sha_file(source.root / "BANK_ADMISSION_RECEIPT.json") != receipt_sha:
                raise ValueError("original source bank receipt differs")
            first, second = source.read_bank(1), source.read_bank(2)
            values = {
                "bank_r1": first["probability"],
                "bank_r2": second["probability"],
                "fine_to_code_r1": first["fine_to_hcal_token"],
                "fine_to_code_r2": second["fine_to_hcal_token"],
            }
            if (values["bank_r1"].dtype != np.float64 or
                    values["bank_r2"].dtype != np.float64 or
                    values["fine_to_code_r1"].dtype != np.int64 or
                    values["fine_to_code_r2"].dtype != np.int64 or
                    values["bank_r1"].shape[1] != 2 or
                    values["bank_r2"].shape[1] != 6 or
                    values["fine_to_code_r1"].shape != (64,) or
                    values["fine_to_code_r2"].shape != (641,)):
                raise ValueError("historical bank or fine grid shape differs")
            for recipient in (1, 2):
                bank = values[f"bank_r{recipient}"]
                mapping = values[f"fine_to_code_r{recipient}"]
                if np.any(mapping < 0) or np.any(mapping >= len(bank)):
                    raise ValueError("fine map code escapes fixed bank")
            stem = f"A_FIXED_MAP_{arm}_S{seed}"
            archive = output_root / f"{stem}.npz"
            receipt_path = output_root / f"{stem}.json"
            if archive.exists() or receipt_path.exists():
                raise FileExistsError("fixed public map export is write-once")
            np.savez_compressed(archive, **values)
            receipt = {
                "schema": "pcrl-r01-public-fixed-fine-map-v1",
                "status": "LABEL_FREE_FIXED_PUBLIC_SOURCE",
                "arm_id": arm,
                "seed": seed,
                "release_id": release_id,
                "source_kind": "historical_finite_reference",
                "source_binding_sha256": source.source_binding_sha256,
                "source_audit_sha256": audit_sha,
                "original_bank_receipt_sha256": receipt_sha,
                "original_bank_file_sha256_by_recipient": {
                    f"r{i}": sha_file(source.root / f"bank_r{i}.npz")
                    for i in (1, 2)
                },
                "array_sha256": {name: sha_array(value)
                                 for name, value in sorted(values.items())},
                "array_shape": {name: list(value.shape)
                                for name, value in sorted(values.items())},
                "npz_sha256": sha_file(archive),
            }
            receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
            records[f"{arm}:s{seed}"] = {
                "npz_path": str(archive), "npz_sha256": receipt["npz_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha_file(receipt_path),
            }
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("portfolio-package", "repair-manifest", "source-audit",
                 "historical-audit", "cache-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(export(**{key.replace("-", "_"): value
                               for key, value in args.items()}),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

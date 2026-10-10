"""R03-only independent P* ATT/INNER cell, token and decoder replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .independent_a_source import (BANK_FIELDS, REFERENCE_FIELDS, TOKEN_FIELDS,
                                   bound_json, bound_npz, role_reference_path,
                                   routed_cells, sha_array, sha_file)
from .independent_pstar import check_pstar_bank
from .replay_a_prefit import _role_npz, queue_guard


ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
RELEASE_ID = "U|JOINT|i8o64|l0.1|H-CLASS-TEMP"


def run(args: argparse.Namespace) -> dict:
    seal_path = Path(args.seal).resolve(strict=True)
    queue_guard(seal_path, args.seal_sha256)
    lock = bound_json(seal_path, args.seal_sha256)
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("P* source verification requires effective local seal")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = bound_json(manifest_path, args.role_manifest_sha256)
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise ValueError("R01 role manifest schema differs")
    original_audit = bound_json(Path(args.source_audit), args.source_audit_sha256)
    if (original_audit.get("status") != "READ_ONLY_HASH_AND_MEMBERSHIP_PASS"
            or args.source_audit_sha256 != lock["source_files_sha256"]["r01/SOURCE_AUDIT.json"]
            or original_audit["source_schema_sha256"] != manifest["source_schema_sha256"]):
        raise ValueError("P* source is not bound to original effective audit")
    audit = bound_json(Path(args.historical_audit), args.historical_audit_sha256)
    if (audit.get("schema") != "pcrl-r01-a-historical-score-source-audit-v1"
            or audit.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED"
            or audit.get("pstar_release_id") != RELEASE_ID):
        raise ValueError("R01 historical P* source audit differs")
    portfolio = Path(args.portfolio_package).resolve(strict=True)
    cache_root = Path(args.cache_root).resolve(strict=True)
    if sha_file(portfolio / "DATA_ACCESS_MANIFEST.json") != manifest["portfolio_manifest_sha256"]:
        raise ValueError("P* predecessor portfolio admission differs")
    report = {"schema": "pcrl-r04-pstar-source-independent-replay-v1",
              "status": "PSTAR_ATT_INNER_SOURCE_REPLAY_PASS",
              "scope": "ATTACK_FIT_NEW and INNER_SELECTION only; source/bank/wire, no fitted A release",
              "effective_seal_sha256": args.seal_sha256,
              "role_manifest_sha256": args.role_manifest_sha256,
              "r01_source_audit_sha256": args.source_audit_sha256,
              "r01_historical_audit_sha256": args.historical_audit_sha256,
              "seeds": {}}
    for seed in (0, 1):
        entry = audit["pstar"][f"s{seed}"]
        receipt_path = (portfolio / entry["receipt_path"]).resolve(strict=True)
        receipt = bound_json(receipt_path, entry["receipt_sha256"])
        if (receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
                or receipt.get("preprocessing_only") is not True
                or receipt.get("release_id") != RELEASE_ID
                or receipt.get("seed") != seed
                or receipt.get("data_access_manifest_sha256") !=
                   sha_file(portfolio / "DATA_ACCESS_MANIFEST.json")
                or receipt.get("source_binding_sha256") != entry["source_binding_sha256"]):
            raise ValueError("P* admitted score bank receipt differs")
        binding_bytes = json.dumps(receipt["source_binding"], sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=True,
                                   allow_nan=False).encode()
        if hashlib.sha256(binding_bytes).hexdigest() != entry["source_binding_sha256"]:
            raise ValueError("P* source binding digest differs")
        source = {}
        source_names = {"model.pt": "teacher_model_sha256", "fine.json": "fine_sha256",
                        "policy.json": "policy_sha256", "decoder.json": "decoder_file_sha256"}
        for name, receipt_name in source_names.items():
            record = entry["source_paths_sha256"][name]
            path = (cache_root / record["path"]).resolve(strict=True)
            if (sha_file(path) != record["sha256"]
                    or record["sha256"] != receipt["source_json_sha256"][receipt_name]):
                raise ValueError(f"P* original {name} bytes differ")
            if name != "model.pt":
                source[name] = json.loads(path.read_text())
        if (source["decoder.json"]["decoder_sha256"] != receipt["source_json_sha256"]["decoder_sha256"]
                or source["decoder.json"]["policy_pair_fingerprint"] != source["policy.json"]["fingerprint"]
                or receipt["policy_fingerprint"] != source["policy.json"]["fingerprint"]):
            raise ValueError("P* policy and decoder fingerprint differs")
        seed_report = {"source_binding_sha256": entry["source_binding_sha256"],
                       "receipt_sha256": entry["receipt_sha256"], "banks": {}, "roles": {}}
        banks = {}
        for recipient in (1, 2):
            name = f"bank_r{recipient}.npz"
            bank_record = entry["banks"][f"r{recipient}"]
            if receipt["private_output_sha256"][name] != bank_record["sha256"]:
                raise ValueError("P* bank byte hash not admitted")
            bank = bound_npz(portfolio / bank_record["path"], bank_record["sha256"], BANK_FIELDS)
            banks[recipient] = bank
            detail = check_pstar_bank(bank, source["policy.json"][f"p{recipient}"],
                                      source["fine.json"][f"fine{recipient}"],
                                      source["decoder.json"][f"r{recipient}"],
                                      receipt["source_binding"]["recipients"][f"r{recipient}"])
            if (detail["public_actions"] != bank_record["tokens"]
                    or detail["input_fine_cells"] != bank_record["input_fine_cells"]):
                raise ValueError("P* historical bank dimensions differ")
            seed_report["banks"][f"r{recipient}"] = detail
        for role in ROLES:
            role_record = entry["roles"][role]
            token_name = f"tokens_{role}.npz"
            if receipt["private_output_sha256"][token_name] != role_record["sha256"]:
                raise ValueError("P* role tokens not admitted")
            tokens = bound_npz(portfolio / role_record["path"], role_record["sha256"], TOKEN_FIELDS)
            references = _role_npz(manifest, manifest_path, role,
                                   f"references_s{seed}", REFERENCE_FIELDS)
            reference_path = role_reference_path(manifest_path, manifest, role, seed)
            if (sha_file(reference_path) != role_record["source_reference_sha256"]
                    or not np.array_equal(tokens["row_id"], references["row_id"])
                    or not np.array_equal(tokens["unit"], references["unit"])
                    or len(tokens["row_id"]) != role_record["rows"]
                    or sha_array(np.column_stack((tokens["row_id"], tokens["unit"]))) !=
                       role_record["row_unit_pair_sha256"]):
                raise ValueError("P* role source/person order differs")
            role_report = {"rows": int(len(tokens["row_id"])), "recipients": {}}
            for recipient in (1, 2):
                bank = banks[recipient]
                if not np.array_equal(np.argmax(references[f"U0_{recipient}"], axis=1),
                                      references[f"d_{recipient}"]):
                    raise ValueError("P* original source decision differs")
                cells = routed_cells(references[f"U0_{recipient}"],
                                     source["fine.json"][f"fine{recipient}"])
                codes = bank["fine_to_hcal_token"][cells]
                bound = receipt["role_receipts"][role]
                if (not np.array_equal(cells, tokens[f"cell_{recipient}"])
                        or not np.array_equal(codes, tokens[f"token_{recipient}"])
                        or sha_array(cells) != bound[f"cell_{recipient}_sha256"]
                        or sha_array(codes) != bound[f"token_{recipient}_sha256"]
                        or not np.array_equal(bank["token_class"][codes],
                                              references[f"d_{recipient}"])):
                    raise ValueError("P* independent fine routing or public token differs")
                role_report["recipients"][f"r{recipient}"] = {
                    "cell_sha256": sha_array(cells), "public_code_sha256": sha_array(codes),
                    "public_probability_sha256": sha_array(bank["probability"][codes]),
                    "public_decision_sha256": sha_array(bank["token_class"][codes])}
            seed_report["roles"][role] = role_report
        report["seeds"][f"s{seed}"] = seed_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "role-manifest", "role-manifest-sha256",
                 "source-audit", "source-audit-sha256", "historical-audit",
                 "historical-audit-sha256", "portfolio-package", "cache-root",
                 "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    report = run(args)
    path = Path(args.public_receipt)
    if path.exists():
        raise FileExistsError("P* independent replay receipt already exists")
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(report["status"])


if __name__ == "__main__":
    main()

"""Independent T* ATT/INNER release replay; run only in an R03 verification unit.

This module reads admitted role-only token/bank exports directly. It imports no
candidate release writer, source loader, attack scorer, or endpoint code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .independent_a_source import bound_npz, sha_array, sha_file


RELEASE_FIELDS = {"row_id", "unit", "prob_r1", "prob_r2", "decision_r1",
                  "decision_r2", "code_r1", "code_r2"}
TOKEN_FIELDS = {"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"}
BANK_FIELDS = {"probability", "token_class", "token_id", "input_class",
               "fine_to_hcal_token"}
ROLES = {"ATT": "ATTACK_FIT_NEW", "INNER": "INNER_SELECTION"}


def require(ok: bool, label: str) -> None:
    if not ok:
        raise ValueError(label)


def same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def wire_digest(q: np.ndarray, d: np.ndarray, code: np.ndarray, *, recipient: int,
                release_id: str, source_binding: str, bank_hash: str,
                bank_size: int) -> str:
    digest = hashlib.sha256()
    metadata = {"model_id": release_id, "source_binding_sha256": source_binding,
                "bank_sha256": bank_hash,
                "code_length_bits": (bank_size - 1).bit_length()}
    for i in range(len(code)):
        wire = {"schema": "SCV1-reference-score-release-v1", "recipient": recipient,
                "probability": q[i].tolist(), "decision": int(d[i]),
                "release_id": release_id, "public_metadata": metadata,
                "code_id": int(code[i])}
        digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--portfolio", type=Path, required=True)
    ap.add_argument("--repair-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--amendment-sha256", required=True)
    args = ap.parse_args()
    camp = args.campaign.resolve(strict=True)
    portfolio = args.portfolio.resolve(strict=True)
    manifest_path = args.repair_manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    source_audit_path = camp / "r01/SOURCE_AUDIT.json"
    source_audit = json.loads(source_audit_path.read_text())
    bank_audit = source_audit["tstar_banks"]["0"]
    bank_root = portfolio / "shared/.private/tstar_score_bank_s0_v1"
    bank_receipt_path = bank_root / "BANK_ADMISSION_RECEIPT.json"
    require(sha_file(bank_receipt_path) == bank_audit["receipt_sha256"], "bank receipt SHA")
    bank_receipt = json.loads(bank_receipt_path.read_text())
    require(bank_receipt["status"] == "ROLE_ONLY_BANK_ADMITTED" and bank_receipt["seed"] == 0,
            "bank receipt status/seed")
    require(bank_receipt["source_binding_sha256"] == bank_audit["source_binding_sha256"],
            "bank source binding")
    banks = {}
    for r in (1, 2):
        name = f"bank_r{r}.npz"
        bank = bound_npz(bank_root / name, bank_audit["file_sha256"][name], BANK_FIELDS)
        require(np.array_equal(bank["token_id"], np.arange(len(bank["token_id"]))),
                f"r{r} token IDs")
        require(np.array_equal(np.argmax(bank["probability"], axis=1), bank["token_class"]),
                f"r{r} bank decisions")
        require(np.array_equal(bank["input_class"],
                               bank["token_class"][bank["fine_to_hcal_token"]]),
                f"r{r} fine class mapping")
        require(sha_array(bank["probability"]) ==
                bank_receipt["source_binding"]["recipients"][f"r{r}"]["bank_probability_sha256"],
                f"r{r} bank array hash")
        banks[r] = bank
    out = {"schema": "r04-independent-fixed-tstar-pair-replay-v1",
           "status": "TSTAR_S0_ATT_INNER_RELEASE_REPLAY_PASS",
           "scope": "role-only bank/token to public finite wire, no labels or reader fits",
           "seed": 0, "amendment_sha256": args.amendment_sha256,
           "source_audit_sha256": sha_file(source_audit_path),
           "repair_manifest_sha256": sha_file(manifest_path),
           "bank_receipt_sha256": sha_file(bank_receipt_path),
           "source_binding_sha256": bank_audit["source_binding_sha256"],
           "roles": {}}
    for short, role in ROLES.items():
        root = camp / "r03/.private/a_fixed_units" / f"A-REFERENCE-RELEASE-T_STAR-S0-{short}"
        receipt_path = root / "REFERENCE_RELEASE_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        queue_path = camp / "r03/.private/compute/units" / f"A-REFERENCE-RELEASE-T_STAR-S0-{short}" / "RECEIPT.json"
        queue = json.loads(queue_path.read_text())
        require(queue["status"] == "done" and queue["returncode"] == 0 and
                queue["unit_id"] == f"A-REFERENCE-RELEASE-T_STAR-S0-{short}" and
                queue["output_sha256"][str(receipt_path.resolve())] == sha_file(receipt_path),
                f"{role} R03 completed output binding")
        package = bound_npz(root / "reference_release.npz", receipt["codes_file_sha256"],
                            RELEASE_FIELDS)
        token_name = f"tokens_{role}.npz"
        token = bound_npz(bank_root / token_name,
                          bank_audit["file_sha256"][token_name], TOKEN_FIELDS)
        key = f"{role}:references_s0"
        reference_path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
        require(sha_file(reference_path) == manifest["role_file_sha256"][key],
                f"{role} reference source SHA")
        with np.load(reference_path, allow_pickle=False) as ref:
            require(same_bytes(token["row_id"], ref["row_id"]) and
                    same_bytes(token["unit"], ref["unit"]), f"{role} source identity")
        require(receipt["schema"] == "SCV1-private-reference-release-v1" and
                receipt["status"] == "FROZEN_DETERMINISTIC" and
                receipt["arm_id"] == "T_STAR" and receipt["source_seed"] == 0 and
                receipt["role"] == role and receipt["interface_kind"] == "finite",
                f"{role} receipt contract")
        require(receipt["release_id"] == bank_receipt["release_id"] and
                receipt["source_binding_sha256"] == bank_audit["source_binding_sha256"] and
                receipt["source_identity"]["bank_receipt_sha256"] == sha_file(bank_receipt_path) and
                receipt["source_identity"]["role_source_file_sha256"] ==
                    bank_audit["file_sha256"][token_name] and
                receipt["source_identity"]["reference_file_sha256"] ==
                    manifest["role_file_sha256"][key] and
                receipt["source_identity"]["source_audit_sha256"] == sha_file(source_audit_path),
                f"{role} receipt provenance")
        pair = np.column_stack((token["row_id"], token["unit"]))
        require(receipt["row_unit_pair_sha256"] == sha_array(pair) ==
                manifest["roles"][role]["row_unit_pair_sha256"], f"{role} row/group SHA")
        require(receipt["rows"] == len(pair) and
                receipt["groups"] == len(np.unique(token["unit"])), f"{role} counts")
        require(same_bytes(package["row_id"], token["row_id"]) and
                same_bytes(package["unit"], token["unit"]), f"{role} release identity")
        summary = {"rows": len(pair), "groups": receipt["groups"],
                   "producer_queue_receipt_sha256": sha_file(queue_path),
                   "source_token_sha256": sha_file(bank_root / token_name),
                   "release_npz_sha256": sha_file(root / "reference_release.npz"),
                   "release_receipt_sha256": sha_file(receipt_path),
                   "row_unit_pair_sha256": sha_array(pair), "recipients": {}}
        for r in (1, 2):
            bank = banks[r]
            code = token[f"token_{r}"]
            require(np.array_equal(code, bank["fine_to_hcal_token"][token[f"cell_{r}"]]),
                    f"{role} r{r} fine-to-code")
            q = bank["probability"][code]
            d = bank["token_class"][code]
            require(same_bytes(package[f"code_r{r}"], code) and
                    same_bytes(package[f"prob_r{r}"], q) and
                    same_bytes(package[f"decision_r{r}"], d), f"{role} r{r} release output")
            bank_hash = sha_array(bank["probability"])
            require(receipt["bank_sha256_by_recipient"][f"r{r}"] == bank_hash,
                    f"{role} r{r} bank hash")
            wire_hash = wire_digest(q, d, code, recipient=r,
                                    release_id=receipt["release_id"],
                                    source_binding=receipt["source_binding_sha256"],
                                    bank_hash=bank_hash, bank_size=len(bank["token_id"]))
            require(receipt["wire_sha256_by_recipient"][f"r{r}"] == wire_hash,
                    f"{role} r{r} public wire SHA")
            by_group = {}
            for i, group in enumerate(token["unit"]):
                symbol = (int(code[i]), int(d[i]), q[i].tobytes())
                require(group not in by_group or by_group[group] == symbol,
                        f"{role} r{r} duplicate group consistency")
                by_group[group] = symbol
            summary["recipients"][f"r{r}"] = {
                "bank_actions": len(bank["token_id"]),
                "observed_code_ids": len(np.unique(code)),
                "wire_sha256": wire_hash, "bank_probability_sha256": bank_hash}
        pair_id = token["token_1"].astype(np.int64) * len(banks[2]["token_id"]) + token["token_2"]
        summary["observed_complete_view_pairs"] = len(np.unique(pair_id))
        out["roles"][role] = summary
    require(not args.output.exists(), "public receipt exists")
    args.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()

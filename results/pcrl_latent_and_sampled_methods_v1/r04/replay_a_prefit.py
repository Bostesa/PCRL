"""Queue-only independent A T* source, decoder and identity utility replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .independent_a_source import (BANK_FIELDS, REFERENCE_FIELDS, ROLES,
                                   TOKEN_FIELDS, bound_json, bound_npz, check_bank,
                                   role_reference_path, routed_cells, sha_array,
                                   sha_file)
from .independent_replay import task_metrics, task_point_gate


def queue_guard(seal: Path, seal_sha: str) -> None:
    if (os.environ.get("PCRL_QUEUE_STAGE") != "verification"
            or os.environ.get("PCRL_QUEUE_SEAL_SHA256") != seal_sha
            or sha_file(seal) != seal_sha):
        raise PermissionError("R04 A prefit replay requires effective R03 verification slot")
    root = Path(os.environ["PCRL_QUEUE_ROOT"])
    unit = os.environ["PCRL_QUEUE_UNIT_ID"]
    state = json.loads((root / "state.json").read_text())
    item = state.get("units", {}).get(unit, {})
    if (state.get("closed") or state.get("seal", {}).get("sha256") != seal_sha
            or item.get("status") != "running" or item.get("child_pid") != os.getpid()
            or item.get("spec", {}).get("stage") != "verification"):
        raise PermissionError("R04 is not active R03 verification child")


def _role_npz(manifest: dict, manifest_path: Path, role: str, kind: str,
              fields: set[str] | None = None) -> dict[str, np.ndarray]:
    key = f"{role}:{kind}"
    path = manifest_path.parent.parent / manifest["role_file_relative_path"][key]
    expected = manifest["role_file_sha256"][key]
    if sha_file(path) != expected:
        raise ValueError(f"role source hash changed: {key}")
    with np.load(path, allow_pickle=False) as archive:
        if fields is not None and set(archive.files) != fields:
            raise ValueError(f"role source fields changed: {key}")
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _source_paths(source_root: Path, seed: int) -> dict[str, Path]:
    return {
        "fine": source_root / f"qpc_v1/run/units/fine__s{seed}/fine.json",
        "policy": source_root / f"qpc_v1/run/units/pol__s{seed}__U_FINE-TASK_i8o64/policy.json",
        "decoder": source_root / f"hcal_v1/package/decoder__s{seed}__U_FINE-TASK_i8o64_H-GLOBAL-TEMP.json",
        "teacher": source_root / f"hcal_v1/admitted/rel__s{seed}__U/model.pt",
    }


def _align(name: str, left: dict, right: dict) -> None:
    if (not np.array_equal(left["row_id"], right["row_id"])
            or not np.array_equal(left["unit"], right["unit"])):
        raise ValueError(f"ordered role row/group identity differs: {name}")


def _oracle_bank_lower_bounds(bank: dict, input_class: np.ndarray,
                              labels: np.ndarray) -> dict[str, float]:
    """An optimistic label-aware bound over allowed class-preserving outputs."""
    p = bank["probability"]
    tc = bank["token_class"]
    y = np.asarray(labels, dtype=np.int64)
    nll_total = 0.0
    brier_total = 0.0
    for cls in range(p.shape[1]):
        rows = np.flatnonzero(input_class == cls)
        if not len(rows):
            continue
        allowed = p[tc == cls]
        if not len(allowed):
            raise ValueError("source class has no public bank action")
        loss_nll = -np.log(np.maximum(allowed[:, y[rows]], 1e-12))
        squared = np.sum(allowed * allowed, axis=1)[:, None] + 1 - 2 * allowed[:, y[rows]]
        nll_total += float(np.min(loss_nll, axis=0).sum())
        brier_total += float(np.min(squared, axis=0).sum())
    return {"nll": nll_total / len(y), "brier": brier_total / len(y),
            "meaning": "separate optimistic label-aware minima; necessary, not sufficient, for one channel"}


def run(args: argparse.Namespace) -> dict:
    seal_path = Path(args.seal).resolve(strict=True)
    queue_guard(seal_path, args.seal_sha256)
    lock = bound_json(seal_path, args.seal_sha256)
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("effective local seal status differs")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = bound_json(manifest_path, args.role_manifest_sha256)
    if manifest.get("schema") != "pcrl-utility-repair-role-admission-v1":
        raise ValueError("source role manifest schema differs")
    audit = bound_json(Path(args.source_audit), args.source_audit_sha256)
    if (audit.get("status") != "READ_ONLY_HASH_AND_MEMBERSHIP_PASS"
            or audit["source_schema_sha256"] != manifest["source_schema_sha256"]
            or args.source_audit_sha256 != lock["source_files_sha256"]["r01/SOURCE_AUDIT.json"]):
        raise ValueError("A source audit not bound to effective campaign seal")
    portfolio = Path(args.portfolio_package).resolve(strict=True)
    if sha_file(portfolio / "DATA_ACCESS_MANIFEST.json") != manifest["portfolio_manifest_sha256"]:
        raise ValueError("predecessor portfolio admission differs")
    source_root = Path(args.source_root).resolve(strict=True)
    result = {"schema": "pcrl-r04-a-prefit-independent-replay-v1",
              "status": "PREFIT_SOURCE_DECODER_IDENTITY_PASS", "scope": "seeds 0,1; admitted FIT/ATT/INNER only; FIT/INNER task truth only; no channel fit",
              "seal_sha256": args.seal_sha256, "source_audit_sha256": args.source_audit_sha256,
              "role_manifest_sha256": args.role_manifest_sha256, "seeds": {}}
    for seed in (0, 1):
        bank_root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
        receipt_path = bank_root / "BANK_ADMISSION_RECEIPT.json"
        expected_receipt = audit["tstar_banks"][str(seed)]["receipt_sha256"]
        receipt = bound_json(receipt_path, expected_receipt)
        if (receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
                or receipt.get("preprocessing_only") is not True
                or receipt.get("seed") != seed
                or receipt.get("data_access_manifest_sha256") !=
                    sha_file(portfolio / "DATA_ACCESS_MANIFEST.json")):
            raise ValueError("T* source admission receipt differs")
        binding_bytes = json.dumps(receipt["source_binding"], sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=True,
                                   allow_nan=False).encode()
        if (hashlib.sha256(binding_bytes).hexdigest() != receipt["source_binding_sha256"]
                or receipt["source_binding_sha256"] !=
                    audit["tstar_banks"][str(seed)]["source_binding_sha256"]):
            raise ValueError("T* source binding differs")
        paths = _source_paths(source_root, seed)
        hashes = receipt["source_json_sha256"]
        fine = bound_json(paths["fine"], hashes["fine_sha256"])
        policy = bound_json(paths["policy"], hashes["policy_sha256"])
        decoder = bound_json(paths["decoder"], hashes["decoder_file_sha256"])
        if (sha_file(paths["teacher"]) != hashes["teacher_model_sha256"]
                or hashes["teacher_model_sha256"] !=
                    audit["u0_models"][str(seed)]["components_sha256"]["model.pt"]
                or decoder["decoder_sha256"] != hashes["decoder_sha256"]
                or decoder["policy_pair_fingerprint"] != policy["fingerprint"]
                or receipt["policy_fingerprint"] != policy["fingerprint"]):
            raise ValueError("teacher/policy/decoder binding differs")
        banks = {}
        bank_reports = {}
        for recipient in (1, 2):
            name = f"bank_r{recipient}.npz"
            expected = receipt["private_output_sha256"][name]
            if expected != audit["tstar_banks"][str(seed)]["file_sha256"][name]:
                raise ValueError("A bank hash not in R01 audit")
            banks[recipient] = bound_npz(bank_root / name, expected, BANK_FIELDS)
            bank_reports[f"r{recipient}"] = check_bank(
                banks[recipient], policy[f"p{recipient}"], fine[f"fine{recipient}"],
                decoder[f"r{recipient}"], receipt["source_binding"]["recipients"][f"r{recipient}"])
        seed_result = {"receipt_sha256": expected_receipt,
                       "source_binding_sha256": receipt["source_binding_sha256"],
                       "banks": bank_reports, "roles": {}}
        for role in ROLES:
            token_name = f"tokens_{role}.npz"
            expected = receipt["private_output_sha256"][token_name]
            if expected != audit["tstar_banks"][str(seed)]["file_sha256"][token_name]:
                raise ValueError("A role token hash not in R01 audit")
            tokens = bound_npz(bank_root / token_name, expected, TOKEN_FIELDS)
            references = _role_npz(manifest, manifest_path, role,
                                   f"references_s{seed}", REFERENCE_FIELDS)
            if (sha_file(role_reference_path(manifest_path, manifest, role, seed)) !=
                    receipt["role_receipts"][role]["source_reference_sha256"]):
                raise ValueError("A source reference differs from bank receipt")
            _align(f"{seed}:{role}", tokens, references)
            if (len(tokens["row_id"]) != receipt["role_receipts"][role]["rows"]
                    or sha_array(np.column_stack((tokens["row_id"], tokens["unit"])))
                       != receipt["role_receipts"][role]["row_unit_pair_sha256"]):
                raise ValueError("ordered T* role identity differs")
            role_result = {"rows": int(len(tokens["row_id"])),
                           "groups": int(len(np.unique(tokens["unit"]))),
                           "source_cell_exact": True, "identity_token_exact": True}
            identity_prob = {}
            identity_decision = {}
            for recipient in (1, 2):
                source = references[f"U0_{recipient}"]
                if not np.array_equal(np.argmax(source, axis=1), references[f"d_{recipient}"]):
                    raise ValueError("U0 source decision not its argmax")
                cell = routed_cells(source, fine[f"fine{recipient}"])
                token = banks[recipient]["fine_to_hcal_token"][cell]
                if (not np.array_equal(cell, tokens[f"cell_{recipient}"])
                        or not np.array_equal(token, tokens[f"token_{recipient}"])
                        or sha_array(cell) != receipt["role_receipts"][role][f"cell_{recipient}_sha256"]
                        or sha_array(token) != receipt["role_receipts"][role][f"token_{recipient}_sha256"]):
                    raise ValueError("independent fine routing or fixed token differs")
                identity_prob[recipient] = banks[recipient]["probability"][token]
                identity_decision[recipient] = banks[recipient]["token_class"][token]
                if not np.array_equal(identity_decision[recipient], references[f"d_{recipient}"]):
                    raise ValueError("source decision lost through identity bank")
            if role in ("OSF_DEFENSE_FIT", "INNER_SELECTION"):
                label = {}
                majority = {}
                for task in ("income", "occupation"):
                    record = _role_npz(manifest, manifest_path, role, task)
                    _align(f"{role}:{task}", tokens, record)
                    label[task] = record[task]
                    majority[task] = int(np.argmax(np.bincount(record[task],
                                                                minlength=2 if task == "income" else 6)))
                current = {}
                baseline = {}
                lower = {}
                for task, recipient in (("income", 1), ("occupation", 2)):
                    y = label[task]
                    current[task] = task_metrics(identity_prob[recipient], y,
                                                 decision=identity_decision[recipient],
                                                 majority_class=majority[task])
                    baseline[task] = {
                        ref: task_metrics(references[f"{ref}_{recipient}"], y,
                                          majority_class=majority[task])
                        for ref in ("U0", "Ucal")}
                    lower[task] = _oracle_bank_lower_bounds(
                        banks[recipient], banks[recipient]["input_class"][tokens[f"cell_{recipient}"]], y)
                role_result["identity_task"] = current
                role_result["references"] = baseline
                role_result["identity_point_gate"] = task_point_gate(current, baseline)
                role_result["bank_label_oracle_lower_bound"] = lower
            seed_result["roles"][role] = role_result
        result["seeds"][str(seed)] = seed_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "role-manifest", "role-manifest-sha256",
                 "source-audit", "source-audit-sha256", "portfolio-package",
                 "source-root", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = run(args)
    path = Path(args.public_receipt)
    if path.exists():
        raise FileExistsError("A prefit verification receipt already exists")
    path.write_text(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(f"R04 A prefit {output['status']}")


if __name__ == "__main__":
    main()

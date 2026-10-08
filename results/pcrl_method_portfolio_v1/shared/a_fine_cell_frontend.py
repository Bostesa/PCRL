"""Trusted one-recipient T* fine-cell frontend for Team A deployment.

This is a provider-side adapter, not a recipient release. It accepts one
preprocessed 83-column input and emits only one recipient's fine-cell ID and
the independently admitted source binding. No score, label or other-party
cell is printed. Run under the pinned .venv after S4 admits the role-only bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dpc import deploy as DD
from qpc import deploy as QD
from qpc import kmeans as KM
from qpc import partition as QP

from admit_a_score_bank import CAMPAIGN, RELEASES, _source_files
from compute_queue import ComputeQueue
from safe_data_guard import sha_arr, sha_file


def verify_independent_admission(seed: int, receipt_path: Path, receipt: dict,
                                 source_binding_sha256: str) -> None:
    """Require S4 output replay and S1's committed deployment registry."""
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or \
            lock.get("adult_fit_permission") is not True:
        raise ValueError("effective campaign science lock absent")
    seal_path = CAMPAIGN / "LOCAL_SEAL_RECEIPT.json"
    if not ComputeQueue._local_seal_ok({"local_seal_path": str(seal_path),
                                        "local_seal_sha256": sha_file(seal_path)}):
        raise ValueError("effective committed local science seal absent")
    registry_path = CAMPAIGN / "A_BANK_DEPLOYMENT_REGISTRY.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("schema") != "pcrl-a-bank-deployment-registry-v1" or \
            registry.get("status") != "SEALED" or \
            registry.get("campaign_lock_sha256") != sha_file(lock_path) or \
            registry.get("frontend_code_sha256") != sha_file(Path(__file__)) or \
            registry.get("source_adapter_sha256") != sha_file(
                Path(__file__).with_name("admit_a_score_bank.py")) or \
            receipt.get("science_lock_sha256") != sha_file(lock_path) or \
            receipt.get("data_access_manifest_sha256") != sha_file(
                CAMPAIGN / "DATA_ACCESS_MANIFEST.json"):
        raise ValueError("trusted A deployment registry identity mismatch")
    repo = CAMPAIGN.parent.parent
    relative = registry_path.relative_to(repo)
    committed = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relative}"],
                               capture_output=True, check=False, timeout=10)
    if committed.returncode or committed.stdout != registry_path.read_bytes():
        raise ValueError("A deployment registry is not committed at HEAD")
    entry = registry["seeds"][str(seed)]
    report_path = CAMPAIGN / f"S4_A_SCORE_BANK_VERIFICATION_s{seed}.json"
    report = json.loads(report_path.read_text())
    receipt_sha = sha_file(receipt_path)
    if (entry.get("s4_report_sha256") != sha_file(report_path) or
            entry.get("bank_admission_receipt_sha256") != receipt_sha or
            entry.get("source_binding_sha256") != source_binding_sha256 or
            report.get("schema") != "pcrl-s4-a-score-bank-verification-v1" or
            report.get("status") != "PASS" or
            report.get("role_cell_token_bank_parity") != "PASS" or
            report.get("source_binding_sha256") != source_binding_sha256 or
            report.get("bank_admission_receipt_sha256") != receipt_sha or
            report.get("source_files_sha256") != receipt["source_json_sha256"] or
            report.get("verified_role_output_sha256") != receipt["private_output_sha256"] or
            report.get("role_export_receipt_sha256") != sha_file(
                CAMPAIGN / "shared" / ".private" / "role_export_v1" /
                "EXPORT_RECEIPT.json") or
            report.get("reviewed_code_sha256") != registry.get("reviewed_code_sha256")):
        raise ValueError("S4 A bank verification differs from trusted registry")


def one_recipient_cell(cache_root: Path, seed: int, recipient: int,
                       input_npz: Path, receipt_path: Path) -> dict:
    if recipient not in (1, 2):
        raise ValueError("recipient must be 1 or 2")
    if receipt_path.name != "BANK_ADMISSION_RECEIPT.json" or \
            receipt_path.parent.name != f"tstar_score_bank_s{seed}_v1":
        raise ValueError("unrecognized admitted T* bank receipt")
    rec = json.loads(receipt_path.read_text())
    binding = rec["source_binding"]
    serialized = json.dumps(binding, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True, allow_nan=False).encode("utf-8")
    binding_sha = hashlib.sha256(serialized).hexdigest()
    if rec.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1" or \
            rec.get("status") != "ROLE_ONLY_BANK_ADMITTED" or \
            rec.get("seed") != seed or \
            rec.get("release_id") != RELEASES["tstar"]["release"] or \
            rec.get("source_binding_sha256") != binding_sha:
        raise ValueError("T* source bank receipt identity mismatch")
    verify_independent_admission(seed, receipt_path, rec, binding_sha)
    model_manifest = json.loads((CAMPAIGN.parent / "pcrl_heldout_calibration_v1" /
                                 "MODEL_MANIFEST.json").read_text())
    source_admission = json.loads((CAMPAIGN.parent / "pcrl_heldout_calibration_v1" /
                                   "SOURCE_ADMISSION.json").read_text())
    paths = _source_files(cache_root, seed, model_manifest, source_admission,
                          RELEASES["tstar"])
    fine = QP.load_fine(json.loads(paths["fine"].read_text()))
    pair = QD.load_pair(paths["policy"])
    schema_hash = binding["feature_names_sha256_newline_join"]
    QD.check_bindings(pair, paths["teacher_model_sha256"], schema_hash)
    if pair.fingerprint() != binding["policy_fingerprint"]:
        raise ValueError("T* policy fingerprint differs from source binding")
    if (pair.p1.fine.fingerprint() != fine[0].fingerprint() or
            pair.p2.fine.fingerprint() != fine[1].fingerprint()):
        raise ValueError("T* fine partition differs from source binding")
    source_files = (("fine_json_sha256", paths["fine_sha256"]),
                    ("policy_json_sha256", paths["policy_sha256"]),
                    ("decoder_json_sha256", paths["decoder_file_sha256"]),
                    ("decoder_internal_sha256", paths["decoder_sha256"]),
                    ("teacher_model_sha256", paths["teacher_model_sha256"]))
    if any(binding[key] != value for key, value in source_files):
        raise ValueError("T* source files differ from admitted binding")
    bank_path = receipt_path.parent / f"bank_r{recipient}.npz"
    if sha_file(bank_path) != rec["private_output_sha256"][bank_path.name]:
        raise ValueError("T* bank file differs from admission receipt")
    with np.load(bank_path, allow_pickle=False) as bank_file:
        if set(bank_file.files) != {"probability", "token_class", "token_id",
                                    "input_class", "fine_to_hcal_token"}:
            raise ValueError("T* bank schema differs from admission")
        bank = {key: bank_file[key] for key in bank_file.files}
    rb = binding["recipients"][f"r{recipient}"]
    if (fine[recipient - 1].fingerprint() != rb["fine_fingerprint"] or
            rb["class_order"] != list(range(len(bank["probability"][0]))) or
            sha_arr(bank["probability"]) != rb["bank_probability_sha256"] or
            sha_arr(bank["token_class"]) != rb["token_class_sha256"] or
            sha_arr(bank["input_class"]) != rb["input_class_sha256"] or
            sha_arr(bank["fine_to_hcal_token"]) != rb["fine_to_hcal_token_sha256"]):
        raise ValueError("T* bank array differs from admitted source binding")
    unit_dir = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U"
    model, heads, model_sha = QD.load_teacher(unit_dir, seed)
    if model_sha != paths["teacher_model_sha256"]:
        raise ValueError("T* teacher model differs from source binding")
    names = DD.schema_names(cache_root / "hcal_v1" / "admitted" / "inputs" /
                            "schema.json")
    if DD.schema_sha256(names) != schema_hash:
        raise ValueError("T* teacher schema differs from source binding")
    X = DD.load_permitted_input(input_npz, names)
    if X.shape[0] != 1:
        raise ValueError("T* frontend accepts exactly one input")
    probability = QD.teacher_probs(model, heads, X)[recipient - 1]
    decision = np.argmax(probability, axis=1).astype(np.int64)
    cell = KM.assign_fine(probability, decision, fine[recipient - 1])
    if bank["input_class"][cell[0]] != decision[0]:
        raise ValueError("T* fine cell class differs from teacher decision")
    return {"schema": "A-token-request-v1", "recipient": recipient,
            "input_cell": int(cell[0]), "source_binding_sha256": binding_sha}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--recipient", type=int, choices=(1, 2), required=True)
    parser.add_argument("--X", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = one_recipient_cell(args.cache_root, args.seed, args.recipient,
                                    args.X, args.bank_receipt)
    except BaseException as exc:
        print("A_FINE_CELL_FRONTEND_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))

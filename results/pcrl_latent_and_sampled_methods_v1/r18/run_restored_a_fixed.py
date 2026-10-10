"""R03-only numerical replay of T* ATT and INNER from restored private bytes.

The worker reads no original role archive. It derives fixed public codes,
probabilities and decisions from the 18-member R18 restore, compares both
saved releases exactly, and writes only a hash/count report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from zipfile import ZipFile

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r18.local_snapshot import (
    FIXED_A_NAMES, uncached_sha256,
)


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
QUEUE_ROOT = CAMPAIGN / "r03/.private/compute"
CUSTODY_ROOT = HERE / ".private/a_fixed_custody"
VERSION = "20261010T032100Z_a_tstar_s0_v1"
UNIT_ID = "R18-A-FIXED-TSTAR-S0-RESTORE-REPLAY-001"
OUTPUT_ROOT = HERE / ".private/a_fixed_replay"
PUBLIC_CUSTODY = HERE / "A_FIXED_REFERENCE_LOCAL_SNAPSHOT_RESTORE_RECEIPT.json"
DIRECT_R04 = CAMPAIGN / "r04/A_TSTAR_S0_FIXED_PAIR_INDEPENDENT_REPLAY.json"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ROLES = {"ATT": "ATTACK_FIT_NEW", "INNER": "INNER_SELECTION"}
BANK_FIELDS = {"probability", "token_class", "token_id", "input_class",
               "fine_to_hcal_token"}
TOKEN_FIELDS = {"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"}
REFERENCE_FIELDS = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1", "Ucal_2",
                    "d_1", "d_2"}
RELEASE_FIELDS = {"row_id", "unit", "prob_r1", "prob_r2", "decision_r1",
                  "decision_r2", "code_r1", "code_r2"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def wire_digest(probability: np.ndarray, decision: np.ndarray, code: np.ndarray,
                *, recipient: int, release_id: str, source_binding: str,
                bank_hash: str, bank_size: int) -> str:
    digest = hashlib.sha256()
    metadata = {"model_id": release_id, "source_binding_sha256": source_binding,
                "bank_sha256": bank_hash,
                "code_length_bits": (bank_size - 1).bit_length()}
    for index in range(len(code)):
        wire = {"schema": "SCV1-reference-score-release-v1", "recipient": recipient,
                "probability": probability[index].tolist(),
                "decision": int(decision[index]), "release_id": release_id,
                "public_metadata": metadata, "code_id": int(code[index])}
        digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def _validate_bank(bank: dict, recipient: int) -> None:
    q = bank["probability"]
    cls = bank["token_class"]
    ids = bank["token_id"]
    fine = bank["fine_to_hcal_token"]
    input_class = bank["input_class"]
    width = 2 if recipient == 1 else 6
    if (q.dtype != np.float64 or q.ndim != 2 or q.shape[1] != width or
            not len(q) or cls.dtype != np.int64 or cls.shape != (len(q),) or
            ids.dtype != np.int64 or ids.shape != cls.shape or
            fine.dtype != np.int64 or fine.ndim != 1 or
            input_class.dtype != np.int64 or input_class.shape != fine.shape or
            not np.array_equal(ids, np.arange(len(ids))) or
            not np.isfinite(q).all() or np.any(q < 0) or
            np.max(np.abs(q.sum(axis=1) - 1)) > 1e-8 or
            not np.array_equal(np.argmax(q, axis=1), cls) or
            np.any((fine < 0) | (fine >= len(ids))) or
            not np.array_equal(input_class, cls[fine])):
        raise ValueError("restored fixed bank differs")


def rederive_role(banks: dict[int, dict], tokens: dict, release: dict,
                  receipt: dict, reference: dict) -> dict:
    """Pure fixed-map replay with byte-exact saved release and wire comparison."""
    ids = tokens["row_id"]
    groups = tokens["unit"]
    if (ids.dtype != np.int64 or groups.dtype != np.int64 or
            ids.ndim != 1 or groups.shape != ids.shape or not len(ids) or
            np.any(np.diff(ids) <= 0) or
            not same_bytes(ids, release["row_id"]) or
            not same_bytes(groups, release["unit"]) or
            not same_bytes(ids, reference["row_id"]) or
            not same_bytes(groups, reference["unit"])):
        raise ValueError("restored fixed source identity differs")
    wire_hashes = {}
    bank_hashes = {}
    for recipient in (1, 2):
        bank = banks[recipient]
        _validate_bank(bank, recipient)
        cell = tokens[f"cell_{recipient}"]
        code = tokens[f"token_{recipient}"]
        if (cell.dtype != np.int64 or code.dtype != np.int64 or
                cell.shape != ids.shape or code.shape != ids.shape or
                np.any((cell < 0) | (cell >= len(bank["fine_to_hcal_token"]))) or
                not np.array_equal(bank["fine_to_hcal_token"][cell], code)):
            raise ValueError("restored fine-to-code mapping differs")
        probability = bank["probability"][code]
        decision = bank["token_class"][code]
        if (not same_bytes(release[f"code_r{recipient}"], code) or
                not same_bytes(release[f"prob_r{recipient}"], probability) or
                not same_bytes(release[f"decision_r{recipient}"], decision)):
            raise ValueError("restored saved release differs")
        bank_sha = array_sha256(bank["probability"])
        key = f"r{recipient}"
        if receipt["bank_sha256_by_recipient"].get(key) != bank_sha:
            raise ValueError("restored fixed bank SHA differs")
        wire_sha = wire_digest(probability, decision, code, recipient=recipient,
                               release_id=receipt["release_id"],
                               source_binding=receipt["source_binding_sha256"],
                               bank_hash=bank_sha, bank_size=len(bank["token_id"]))
        if receipt["wire_sha256_by_recipient"].get(key) != wire_sha:
            raise ValueError("restored fixed wire SHA differs")
        first_by_group = {}
        for index, group in enumerate(groups):
            symbol = (int(code[index]), int(decision[index]),
                      probability[index].tobytes())
            if group in first_by_group and first_by_group[group] != symbol:
                raise ValueError("restored duplicate group release differs")
            first_by_group[group] = symbol
        wire_hashes[key] = wire_sha
        bank_hashes[key] = bank_sha
    return {"rows": len(ids), "groups": len(np.unique(groups)),
            "row_unit_pair_sha256": array_sha256(np.column_stack((ids, groups))),
            "wire_sha256_by_recipient": wire_hashes,
            "bank_probability_sha256_by_recipient": bank_hashes}


def _read_npz(path: Path, fields: set[str]) -> dict:
    with ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        if (len(names) != len(fields) or set(names) != {
                f"{name}.npy" for name in fields}):
            raise PermissionError("restored NPZ member set differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in fields}


def _read_reference(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        if (len(names) != len(REFERENCE_FIELDS) or set(names) != {
                f"{name}.npy" for name in REFERENCE_FIELDS}):
            raise PermissionError("restored reference member set differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True)
                for name in ("row_id", "unit")}


def _owner_dir(path: Path) -> None:
    candidate = path.absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise PermissionError("restored path has a symlink ancestor")
    info = path.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) & 0o077):
        raise PermissionError("restored directory is not owner-only")


def _registered_sha(spec: dict, path: Path, expected: str | None = None) -> str:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PermissionError("restored source has a symlink ancestor")
    listed = [row["sha256"] for row in spec.get("sources", [])
              if Path(row["path"]).absolute() == path]
    actual = uncached_sha256(path)[0] if ".private" in path.parts else sha256_file(path)
    if len(listed) != 1 or listed[0] != actual or (expected is not None and actual != expected):
        raise PermissionError("restored source SHA differs from queued pin")
    return actual


def _active_spec(unit_id: str) -> tuple[dict, dict, str]:
    active = assert_active_from_env(unit_id=unit_id, stage="verification")
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    if root != QUEUE_ROOT.resolve(strict=True):
        raise PermissionError("restored A queue root differs")
    state = json.loads((root / "state.json").read_text())
    unit = state["units"][unit_id]
    spec = unit["spec"]
    if (unit["status"] != "running" or unit["spec_sha256"] != active["spec_sha256"] or
            spec["unit_id"] != unit_id or spec["stage"] != "verification" or
            Path(spec["cwd"]).resolve() != Path.cwd().resolve() or
            Path(spec["argv"][0]).resolve() != Path(sys.executable).resolve() or
            Path(spec["argv"][1]).resolve() != Path(__file__).resolve() or
            spec["argv"][1:] != sys.argv):
        raise PermissionError("restored A invocation differs from active queue unit")
    return state, spec, active["seal_sha256"]


def _write_once(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def run(args: argparse.Namespace) -> dict:
    started_cpu = time.process_time()
    state, spec, original_seal_sha = _active_spec(args.unit_id)
    version = Path(args.snapshot_dir).absolute()
    custody_path = Path(args.custody_receipt).absolute()
    amendment_path = Path(args.verification_amendment).absolute()
    report_path = Path(args.private_output).absolute()
    if (args.unit_id != UNIT_ID or version != CUSTODY_ROOT / VERSION or
            custody_path != PUBLIC_CUSTODY or
            amendment_path.parent != CAMPAIGN or
            report_path != OUTPUT_ROOT / UNIT_ID / "REPORT.json" or
            spec.get("deps") != [f"A-REFERENCE-RELEASE-T_STAR-S0-{short}"
                                 for short in ROLES] or
            spec.get("data_access") != "RESTORED_PRIVATE_A_TSTAR_S0_ROLE_ONLY" or
            spec.get("outputs") != [str(report_path)] or
            spec.get("cpu_reserve_s") != 120 or spec.get("max_wall_s") != 120 or
            spec.get("max_rss_gib") != 2 or
            spec.get("amendment_sha256") != args.verification_amendment_sha256):
        raise PermissionError("restored A queue scope differs")
    for directory in (HERE / ".private", CUSTODY_ROOT, version,
                      version / "snapshot", version / "restore", OUTPUT_ROOT):
        _owner_dir(directory)
    if report_path.parent.exists() or report_path.exists():
        raise FileExistsError("restored A replay output must be fresh")
    source_paths = [Path(__file__), HERE / "local_snapshot.py",
                    HERE / "test_restored_a_fixed.py",
                    CAMPAIGN / "r03/compute_queue.py", custody_path,
                    version / "snapshot/MANIFEST.json",
                    version / "RESTORE_PROOF.json", DIRECT_R04, amendment_path]
    for path in source_paths:
        _registered_sha(spec, path)
    if _registered_sha(spec, amendment_path) != args.verification_amendment_sha256:
        raise PermissionError("restored A verification amendment differs")
    latest = [row for row in state.get("amendments", [])
              if "verification" in row.get("stages", [])]
    if (not latest or latest[-1].get("sha256") != args.verification_amendment_sha256 or
            Path(latest[-1].get("path", "")).resolve() != amendment_path.resolve()):
        raise PermissionError("restored A verification amendment is not latest")
    amendment = json.loads(amendment_path.read_text())
    custody = json.loads(custody_path.read_text())
    manifest = json.loads((version / "snapshot/MANIFEST.json").read_text())
    proof = json.loads((version / "RESTORE_PROOF.json").read_text())
    expected = custody.get("member_sha256_by_logical_id", {})
    if (amendment.get("schema") != "pcrl-a-fixed-restored-replay-verification-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_VERIFICATION" or
            amendment.get("unit_id") != UNIT_ID or
            amendment.get("snapshot_public_receipt_sha256") != sha256_file(custody_path) or
            amendment.get("snapshot_private_manifest_sha256") != sha256_file(
                version / "snapshot/MANIFEST.json") or
            amendment.get("snapshot_restore_proof_sha256") != sha256_file(
                version / "RESTORE_PROOF.json") or
            amendment.get("r04_direct_source_replay_sha256") != sha256_file(DIRECT_R04) or
            amendment.get("original_campaign_lock_sha256") != original_seal_sha or
            amendment.get("producer_amendment_sha256") != expected.get(
                "a_producer_amendment.json") or
            amendment.get("snapshot_restored_member_sha256_by_logical_id") != expected or
            amendment.get("original_queue_receipt_sha256_by_unit") !=
                custody.get("parent_queue_receipt_sha256_by_unit") or
            custody.get("schema") != "pcrl-r18-a-fixed-reference-local-custody-public-v1" or
            custody.get("status") != "LOCAL_UNCACHED_18_MEMBER_SNAPSHOT_RESTORE_PASS" or
            custody.get("version") != VERSION or
            set(expected) != FIXED_A_NAMES or
            manifest.get("schema") != "pcrl-r18-private-a-fixed-snapshot-v1" or
            proof.get("schema") != "pcrl-r18-private-a-fixed-restore-proof-v1" or
            proof.get("snapshot_manifest_sha256") != sha256_file(
                version / "snapshot/MANIFEST.json") or
            {name: info["sha256"] for name, info in proof["members"].items()} != expected):
        raise PermissionError("restored A snapshot or sealed amendment differs")
    restore = version / "restore"
    for name in sorted(FIXED_A_NAMES):
        path = restore / name
        if (_registered_sha(spec, path, expected[name]) != expected[name] or
                stat.S_IMODE(path.stat().st_mode) != 0o600 or
                path.stat().st_size != proof["members"][name]["bytes"]):
            raise PermissionError("restored A member differs")
    original_lock = json.loads((restore / "original_lock.json").read_text())
    producer = json.loads((restore / "a_producer_amendment.json").read_text())
    repair = json.loads((restore / "repair_manifest.json").read_text())
    audit = json.loads((restore / "r01_source_audit.json").read_text())
    bank_receipt = json.loads((restore / "bank_admission_receipt.json").read_text())
    r04 = json.loads(DIRECT_R04.read_text())
    if (original_lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original_lock.get("status") != "LOCALLY_SEALED_D_STAGE" or
            original_lock["source_files_sha256"]["source_role_manifest"] !=
                expected["repair_manifest.json"] or
            original_lock["source_files_sha256"]["r01/SOURCE_AUDIT.json"] !=
                expected["r01_source_audit.json"] or
            producer.get("status") != "LOCALLY_SEALED_A_STAGE" or
            producer.get("original_lock_sha256") != original_seal_sha or
            repair.get("portfolio_manifest_sha256") != expected["portfolio_manifest.json"] or
            audit["tstar_banks"]["0"]["receipt_sha256"] !=
                expected["bank_admission_receipt.json"] or
            bank_receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED" or
            bank_receipt.get("seed") != 0 or
            bank_receipt.get("source_binding_sha256") !=
                audit["tstar_banks"]["0"]["source_binding_sha256"] or
            r04.get("bank_receipt_sha256") != expected["bank_admission_receipt.json"] or
            r04.get("repair_manifest_sha256") != expected["repair_manifest.json"] or
            r04.get("source_audit_sha256") != expected["r01_source_audit.json"] or
            r04.get("status") != "TSTAR_S0_ATT_INNER_RELEASE_REPLAY_PASS"):
        raise PermissionError("restored A historical source chain differs")
    banks = {}
    for recipient in (1, 2):
        name = f"bank_r{recipient}.npz"
        if bank_receipt["private_output_sha256"][name] != expected[name]:
            raise PermissionError("restored A bank file SHA differs")
        bank = _read_npz(restore / name, BANK_FIELDS)
        _validate_bank(bank, recipient)
        if len(bank["token_id"]) != (16 if recipient == 1 else 321):
            raise PermissionError("restored A fixed bank capacity differs")
        expected_probability_sha = bank_receipt["source_binding"]["recipients"][
            f"r{recipient}"]["bank_probability_sha256"]
        if array_sha256(bank["probability"]) != expected_probability_sha:
            raise PermissionError("restored A public bank array differs")
        banks[recipient] = bank
    summary = {}
    for short, role in ROLES.items():
        parent_id = f"A-REFERENCE-RELEASE-T_STAR-S0-{short}"
        parent = state["units"].get(parent_id, {})
        queue_name = f"{short.lower()}_queue_receipt.json"
        release_name = f"{short.lower()}_release_receipt.json"
        saved_name = f"{short.lower()}_release.npz"
        token_name = f"tokens_{short.lower()}.npz"
        ref_name = f"references_{short.lower()}.npz"
        copied_queue = json.loads((restore / queue_name).read_text())
        original_output = (CAMPAIGN / "r03/.private/a_fixed_units" / parent_id /
                           "REFERENCE_RELEASE_RECEIPT.json")
        if (parent.get("status") != "done" or parent.get("returncode") != 0 or
                parent.get("spec", {}).get("outputs") != [str(original_output)] or
                parent.get("output_sha256") !=
                    {str(original_output): expected[release_name]} or
                parent.get("spec", {}).get("amendment_sha256") !=
                    expected["a_producer_amendment.json"] or
                sha256_file(QUEUE_ROOT / "units" / parent_id / "RECEIPT.json") !=
                    expected[queue_name] or
                copied_queue.get("status") != "done" or
                copied_queue.get("returncode") != 0 or
                copied_queue.get("output_sha256", {}).get(str(original_output)) !=
                    expected[release_name]):
            raise PermissionError("restored A completed parent differs")
        receipt = json.loads((restore / release_name).read_text())
        if (receipt.get("schema") != "SCV1-private-reference-release-v1" or
                receipt.get("status") != "FROZEN_DETERMINISTIC" or
                receipt.get("arm_id") != "T_STAR" or
                receipt.get("source_seed") != 0 or
                receipt.get("role") != role or
                receipt.get("interface_kind") != "finite" or
                receipt.get("release_id") != bank_receipt["release_id"] or
                receipt.get("source_binding_sha256") !=
                    bank_receipt["source_binding_sha256"] or
                receipt.get("codes_file_sha256") != expected[saved_name] or
                receipt["source_identity"]["bank_receipt_sha256"] !=
                    expected["bank_admission_receipt.json"] or
                receipt["source_identity"]["role_source_file_sha256"] !=
                    expected[token_name] or
                receipt["source_identity"]["reference_file_sha256"] !=
                    expected[ref_name] or
                receipt["source_identity"]["source_audit_sha256"] !=
                    expected["r01_source_audit.json"] or
                bank_receipt["private_output_sha256"][f"tokens_{role}.npz"] !=
                    expected[token_name] or
                repair["role_file_sha256"][f"{role}:references_s0"] !=
                    expected[ref_name]):
            raise PermissionError("restored A saved receipt/source differs")
        token = _read_npz(restore / token_name, TOKEN_FIELDS)
        release = _read_npz(restore / saved_name, RELEASE_FIELDS)
        reference = _read_reference(restore / ref_name)
        replayed = rederive_role(banks, token, release, receipt, reference)
        if (replayed["rows"] != receipt["rows"] or
                replayed["groups"] != receipt["groups"] or
                replayed["row_unit_pair_sha256"] !=
                    receipt["row_unit_pair_sha256"] !=
                    repair["roles"][role]["row_unit_pair_sha256"] or
                r04["roles"][role]["release_receipt_sha256"] !=
                    expected[release_name] or
                r04["roles"][role]["release_npz_sha256"] != expected[saved_name] or
                any(r04["roles"][role]["recipients"][f"r{recipient}"]["wire_sha256"] !=
                    replayed["wire_sha256_by_recipient"][f"r{recipient}"]
                    for recipient in (1, 2))):
            raise ValueError("restored A role identity/count differs")
        summary[role] = replayed
    if time.process_time() - started_cpu >= 120:
        raise TimeoutError("restored A replay CPU ceiling reached")
    report = {"schema": "pcrl-r18-restored-a-fixed-tstar-pair-replay-v1",
              "status": "RESTORED_TSTAR_S0_ATT_INNER_EXACT_PASS",
              "unit_id": UNIT_ID, "snapshot_version": VERSION,
              "snapshot_receipt_sha256": sha256_file(custody_path),
              "snapshot_restore_proof_sha256": sha256_file(
                  version / "RESTORE_PROOF.json"),
              "producer_amendment_sha256": expected["a_producer_amendment.json"],
              "r04_direct_source_replay_sha256": sha256_file(DIRECT_R04),
              "roles": summary,
              "checks": {"restored_18_member_hashes_exact": True,
                         "fixed_source_token_mapping_exact": True,
                         "both_saved_role_arrays_exact": True,
                         "four_public_wire_sha256_exact": True,
                         "no_original_role_archive_read": True,
                         "no_fit_or_selected_policy": True}}
    report_path.parent.mkdir(mode=0o700)
    _write_once(report_path, report)
    return {"status": report["status"], "unit_id": UNIT_ID,
            "output_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit_id", "snapshot_dir", "custody_receipt",
                 "verification_amendment", "verification_amendment_sha256",
                 "private_output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

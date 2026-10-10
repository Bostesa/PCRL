"""Queued numerical replay of the one-row D-K release from restored snapshot bytes.

The snapshot already contains the sole admitted row. This worker opens no role
export, labels, SEX column, or original fitted package. It writes one fresh
private scratch package and a hash-only replay report under R03's slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.service import load_service
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r18.acceptance_contract import (
    FEATURE_NAMES_SHA256, check_d_public_release, check_raw_schema,
    check_source_checkpoint, sha256_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r18.independent_d_head import IndependentDHead


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parents[2]
QUEUE_ROOT = HERE.parent / "r03" / ".private" / "compute"
CUSTODY_ROOT = HERE / ".private" / "custody"
SCRATCH_ROOT = HERE / ".private" / "restore_replay"
ARCHIVE_TO_SCRATCH = {
    "d_first_public_receipt.json": "d_first_public_receipt.json",
    "d_one_row_result.json": "d_one_row_result.json",
    "d_service_manifest.json": "service/manifest.json",
    "d_service_weights.npz": "service/weights.npz",
    "role_admission_manifest.json": "role_admission_manifest.json",
    "u0_source_model.pt": "rel__s0__U/model.pt",
}


def _owner_private_directory(path: Path) -> None:
    _reject_symlink_ancestors(path)
    if (not path.is_dir() or path.stat().st_uid != os.getuid()
            or stat.S_IMODE(path.stat().st_mode) != 0o700):
        raise PermissionError("snapshot/scratch directory is not owner-private")


def _owner_private_file(path: Path) -> None:
    _reject_symlink_ancestors(path)
    if (not path.is_file() or path.stat().st_uid != os.getuid()
            or stat.S_IMODE(path.stat().st_mode) != 0o600):
        raise PermissionError("snapshot member is not an owner-private plain file")


def _reject_symlink_ancestors(path: Path) -> None:
    current = Path(path).absolute()
    if any(part.is_symlink() for part in (current, *current.parents)):
        raise PermissionError("snapshot path has a symlink ancestor")


def materialize_verified_restore(restore_dir: Path, scratch_dir: Path,
                                 expected_sha256: dict[str, str]) -> dict[str, Path]:
    """Verify all six archive members before copying them to canonical names."""
    source = Path(restore_dir)
    target = Path(scratch_dir)
    _owner_private_directory(source)
    if (set(expected_sha256) != set(ARCHIVE_TO_SCRATCH)
            or set(path.name for path in source.iterdir()) != set(ARCHIVE_TO_SCRATCH)
            or target.exists() or target.is_symlink()):
        raise PermissionError("snapshot inventory or fresh scratch identity differs")
    for name in ARCHIVE_TO_SCRATCH:
        original = source / name
        _owner_private_file(original)
        if sha256_file(original) != expected_sha256[name]:
            raise ValueError("snapshot member hash differs")
    _owner_private_directory(target.parent)
    target.mkdir(mode=0o700)
    (target / "service").mkdir(mode=0o700)
    (target / "rel__s0__U").mkdir(mode=0o700)
    copies: dict[str, Path] = {}
    for name, relative in ARCHIVE_TO_SCRATCH.items():
        original = source / name
        copied = target / relative
        fd = os.open(copied, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with original.open("rb") as input_stream, os.fdopen(fd, "wb") as output_stream:
            for block in iter(lambda: input_stream.read(1 << 20), b""):
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if sha256_file(copied) != expected_sha256[name]:
            raise ValueError("scratch snapshot hash differs after copy")
        copies[name] = copied
    return copies


def _registered_sha(spec: dict, path: Path) -> str:
    target = Path(path).resolve(strict=True)
    _reject_symlink_ancestors(path)
    if not target.is_file():
        raise PermissionError("R18 replay source is not a plain file")
    matches = [record["sha256"] for record in spec["sources"]
               if Path(record["path"]).resolve(strict=True) == target]
    if len(matches) != 1 or sha256_file(target) != matches[0]:
        raise PermissionError("R18 replay source differs from the queued pin")
    return matches[0]


def validate_completed_parents(state: dict, spec: dict, snapshot_hashes: dict[str, str],
                               *, first_receipt_path: Path, one_row_path: Path) -> None:
    """Bind both restored records to their exact completed R03 output paths."""
    fit_id = "D_FIRST_T_DK_S0_R1"
    original_id = "R18-D-DEPLOY-REAL-S0"
    fit = state["units"].get(fit_id, {})
    original = state["units"].get(original_id, {})
    first_path = str(first_receipt_path)
    result_path = str(one_row_path)
    if (spec.get("deps") != [fit_id, original_id]
            or fit.get("status") != "done" or fit.get("returncode") != 0
            or original.get("status") != "done" or original.get("returncode") != 0
            or first_path not in fit.get("spec", {}).get("outputs", [])
            or fit.get("output_sha256", {}).get(first_path)
            != snapshot_hashes["d_first_public_receipt.json"]
            or original.get("spec", {}).get("outputs") != [result_path]
            or original.get("output_sha256")
            != {result_path: snapshot_hashes["d_one_row_result.json"]}):
        raise PermissionError("R18 snapshot parent receipt or output differs")


def _active_spec(unit_id: str) -> tuple[dict, dict, str]:
    active = assert_active_from_env(unit_id=unit_id, stage="verification")
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    if root != QUEUE_ROOT.resolve(strict=True):
        raise PermissionError("R18 replay queue root differs")
    state = json.loads((root / "state.json").read_text())
    unit = state["units"][unit_id]
    spec = unit["spec"]
    if (unit["status"] != "running" or unit["spec_sha256"] != active["spec_sha256"]
            or spec["unit_id"] != unit_id or spec["stage"] != "verification"
            or Path(spec["cwd"]).resolve() != Path.cwd().resolve()
            or Path(spec["argv"][0]).resolve() != Path(sys.executable).resolve()
            or len(spec["argv"]) < 2
            or Path(spec["argv"][1]).resolve() != Path(__file__).resolve()
            or spec["argv"][1:] != sys.argv):
        raise PermissionError("R18 replay invocation differs from the active queue spec")
    return state, spec, active["seal_sha256"]


def _plain_release(value: dict) -> dict:
    return {"code": np.asarray(value["code"], dtype=np.uint8).tolist(),
            "probabilities": np.asarray(value["probabilities"], dtype=np.float64).tolist(),
            "decision": np.asarray(value["decision"], dtype=np.int64).tolist(),
            "metadata": dict(value["metadata"])}


def _write_once(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def run(args: argparse.Namespace) -> dict:
    state, spec, seal_sha = _active_spec(args.unit_id)
    version = Path(args.snapshot_dir).absolute()
    receipt_path = Path(args.custody_receipt).absolute()
    private_manifest_path = Path(args.private_manifest).absolute()
    scratch = Path(args.scratch_dir).absolute()
    report_path = Path(args.private_output).absolute()
    if (version.parent != CUSTODY_ROOT
            or version.name != "20261010T014004Z_d_one_row_v1"
            or receipt_path != HERE / "D_LOCAL_SNAPSHOT_RESTORE_RECEIPT.json"
            or private_manifest_path != version / "PRIVATE_MANIFEST.json"
            or scratch.parent != SCRATCH_ROOT
            or scratch.name != args.unit_id or report_path != scratch / "REPORT.json"):
        raise PermissionError("R18 replay snapshot or output path differs")
    _owner_private_directory(HERE / ".private")
    _owner_private_directory(CUSTODY_ROOT)
    _owner_private_directory(version)
    _owner_private_directory(version / "restore")
    _owner_private_directory(SCRATCH_ROOT)
    _owner_private_file(private_manifest_path)
    if scratch.exists() or report_path.exists():
        raise PermissionError("R18 replay output must be fresh")
    expected_outputs = {str(scratch / name) for name in ARCHIVE_TO_SCRATCH.values()}
    expected_outputs.add(str(report_path))
    if set(spec["outputs"]) != expected_outputs or len(spec["outputs"]) != len(expected_outputs):
        raise PermissionError("R18 replay output list differs from queued spec")
    for path in (Path(__file__), HERE / "acceptance_contract.py", HERE / "independent_d_head.py",
                 HERE.parent / "r01" / "frozen_u.py", HERE.parent / "r03" / "compute_queue.py",
                 HERE.parent / "latent_repair_v1" / "impl" / "model.py",
                 HERE.parent / "latent_repair_v1" / "impl" / "service.py",
                 CAMPAIGN_ROOT / "jcv" / "train.py", receipt_path, private_manifest_path):
        _registered_sha(spec, path)
    custody = json.loads(receipt_path.read_text())
    manifest = json.loads(private_manifest_path.read_text())
    records = {record["logical_id"]: record for record in custody["artifacts"]}
    if (custody.get("schema") != "pcrl-r18-d-local-snapshot-restore-public-v1"
            or custody.get("status") != "LOCAL_BYTE_RESTORE_PASS_RUNTIME_REPLAY_PENDING"
            or custody.get("version") != version.name
            or custody.get("artifact_count") != 6
            or set(records) != set(ARCHIVE_TO_SCRATCH)
            or manifest.get("schema") != "pcrl-r18-private-d-artifact-snapshot-v1"
            or manifest.get("version") != version.name
            or manifest.get("byte_restore_status") != "PASS"
            or set(manifest.get("records", {})) != set(records)
            or sha256_file(private_manifest_path) != custody.get("private_snapshot_manifest_sha256")):
        raise PermissionError("R18 byte-restored snapshot custody differs")
    expected: dict[str, str] = {}
    for name, record in records.items():
        sha = record["sha256"]
        restored_path = version / "restore" / name
        if (sha != manifest["records"][name]["sha256"]
                or record["bytes"] != manifest["records"][name]["bytes"]
                or _registered_sha(spec, restored_path) != sha
                or restored_path.stat().st_size != record["bytes"]):
            raise PermissionError("R18 restored member differs from custody receipt")
        expected[name] = sha
    validate_completed_parents(
        state, spec, expected,
        first_receipt_path=CAMPAIGN_ROOT / "results" / "pcrl_latent_and_sampled_methods_v1"
        / "D_FIRST_S0_R1_RECEIPT.json",
        one_row_path=HERE / ".private" / "deploy" / "R18-D-DEPLOY-REAL-S0.json",
    )
    seal = state["seal"]
    lock_path = Path(seal["path"])
    if _registered_sha(spec, lock_path) != seal_sha:
        raise PermissionError("R18 original D seal differs")
    lock = json.loads(lock_path.read_text())
    pinned_implementation = lock["implementation_source_sha256"]
    locked_sources = {
        "results/pcrl_latent_and_sampled_methods_v1/r01/frozen_u.py":
            HERE.parent / "r01" / "frozen_u.py",
        "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1/impl/model.py":
            HERE.parent / "latent_repair_v1" / "impl" / "model.py",
        "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1/impl/service.py":
            HERE.parent / "latent_repair_v1" / "impl" / "service.py",
        "jcv/train.py": CAMPAIGN_ROOT / "jcv" / "train.py",
    }
    if (lock.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or any(_registered_sha(spec, path) != pinned_implementation.get(relative)
                   for relative, path in locked_sources.items())
            or lock["source_files_sha256"].get("source_role_manifest")
            != expected["role_admission_manifest.json"]
            or lock["role_identity_and_file_hashes"]["source_model_sha256"].get("0")
            != expected["u0_source_model.pt"]):
        raise PermissionError("R18 restored implementation differs from original D seal")
    copies = materialize_verified_restore(version / "restore", scratch, expected)
    saved = json.loads(copies["d_one_row_result.json"].read_text())
    fit = json.loads(copies["d_first_public_receipt.json"].read_text())
    role = json.loads(copies["role_admission_manifest.json"].read_text())
    service_manifest = json.loads(copies["d_service_manifest.json"].read_text())
    d_package = fit.get("packages", {}).get("D-K", {})
    if (saved.get("schema") != "pcrl-r18-real-d-deployment-acceptance-v1"
            or saved.get("status") != "REAL_D_K_ONE_ROW_PASS"
            or saved.get("unit_id") != "R18-D-DEPLOY-REAL-S0"
            or saved.get("role") != "INNER_SELECTION" or saved.get("row_index") != 0
            or saved.get("seal_sha256") != seal_sha
            or saved.get("fit_receipt_sha256") != expected["d_first_public_receipt.json"]
            or saved.get("service_manifest_sha256") != expected["d_service_manifest.json"]
            or saved.get("source_checkpoint_sha256") != expected["u0_source_model.pt"]
            or saved.get("role_manifest_sha256") != expected["role_admission_manifest.json"]
            or fit.get("seal_sha256") != seal_sha or fit.get("seed") != 0
            or fit.get("status") != "T_DK_RELEASES_READY"
            or fit.get("source_sha256") != expected["u0_source_model.pt"]
            or fit.get("role_manifest_sha256") != expected["role_admission_manifest.json"]
            or d_package.get("manifest_sha256") != expected["d_service_manifest.json"]
            or d_package.get("weights_sha256") != expected["d_service_weights.npz"]
            or role.get("source_schema_sha256") != FEATURE_NAMES_SHA256
            or service_manifest.get("arm_id") != "D-K"
            or service_manifest.get("source_seed") != 0
            or service_manifest.get("source_sha256") != expected["u0_source_model.pt"]
            or service_manifest.get("model_sha256") != d_package.get("model_sha256")
            or service_manifest.get("weights_sha256") != expected["d_service_weights.npz"]):
        raise PermissionError("R18 restored D fit, row, and service chain differs")
    row_bytes = bytes.fromhex(saved["input_row_float32_le_hex"])
    if len(row_bytes) != 83 * 4 or hashlib.sha256(row_bytes).hexdigest() != saved["input_row_sha256"]:
        raise PermissionError("R18 restored one-row bytes differ")
    row = np.frombuffer(row_bytes, dtype="<f4").reshape(1, 83).copy()
    columns = saved["input_columns"]
    check_raw_schema(row, columns, columns)
    check_source_checkpoint(copies["u0_source_model.pt"], expected["u0_source_model.pt"])
    encoder = FrozenUEncoder.load(copies["u0_source_model.pt"],
                                  expected["u0_source_model.pt"], columns,
                                  FEATURE_NAMES_SHA256, 0)
    service = load_service(copies["d_service_manifest.json"],
                           expected_manifest_sha256=expected["d_service_manifest.json"],
                           expected_source_sha256=expected["u0_source_model.pt"])
    independent = IndependentDHead(copies["d_service_manifest.json"],
                                   manifest_sha256=expected["d_service_manifest.json"],
                                   weights_sha256=expected["d_service_weights.npz"],
                                   source_sha256=expected["u0_source_model.pt"],
                                   model_sha256=d_package["model_sha256"])
    output_shas = {}
    for purpose in (0, 1):
        released = service.recipient_from_raw(purpose, row, columns, encoder)
        repeated = service.recipient_from_raw(purpose, row, columns, encoder)
        expected_metadata = independent.expected_metadata(purpose)
        check_d_public_release(released, purpose, expected_metadata, independent.probabilities)
        check_d_public_release(repeated, purpose, expected_metadata, independent.probabilities)
        actual = _plain_release(released)
        if actual != _plain_release(repeated) or actual != saved["releases"][str(purpose)]:
            raise ValueError("restored D service differs from the original one-row release")
        output_shas[str(purpose)] = hashlib.sha256(json.dumps(
            actual, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    report = {
        "schema": "pcrl-r18-restored-d-one-row-replay-v1",
        "status": "RESTORED_D_K_ONE_ROW_EXACT_PASS",
        "unit_id": args.unit_id,
        "snapshot_version": version.name,
        "snapshot_receipt_sha256": sha256_file(receipt_path),
        "original_one_row_sha256": expected["d_one_row_result.json"],
        "source_checkpoint_sha256": expected["u0_source_model.pt"],
        "service_manifest_sha256": expected["d_service_manifest.json"],
        "input_row_sha256": saved["input_row_sha256"],
        "public_release_sha256_by_purpose": output_shas,
        "scratch_sha256_by_archive_name": {name: sha256_file(path) for name, path in copies.items()},
        "checks": {"snapshot_bytes_exact": True, "no_role_export_read": True,
                   "frozen_source_restored": True, "calibrated_code_head_independent": True,
                   "one_row_both_public_releases_exact": True, "repeat_exact": True},
    }
    _write_once(report_path, report)
    return {"status": report["status"], "unit_id": args.unit_id,
            "output_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--custody-receipt", required=True)
    parser.add_argument("--private-manifest", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--private-output", required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()

"""Queued, private Team D pooled-arm fit/calibrate/release stages.

This runner refuses an ineffective S1 science seal before opening role-only
arrays. S3 must launch each numerical stage as a registered queue unit. The
runner itself opens no mixed-role archive, old assessment, or HEAD_VALIDATION.
It writes row-level releases only under an explicit owner-only private root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from attack_views import AlignedDViews
from cache_store import load_private_cache, save_private_cache
from checkpoint import (_private_target, _sha256, _write_json,
                        load_private_checkpoint, save_private_checkpoint)
from controls_fit import ControlFitConfig, prepare_control_cache
from deployment import (DCalibratedDeployment, calibrator_record_hash,
                        normalize_head_probabilities)
from encoder_adapter import FrozenUEncoder, hash_file
from trainer import DTrainConfig, DTrainingData, train_d


SOURCE_TIP = "1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5"
SOURCE_NPZ_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
FEATURE_SHA = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"
U_MODEL_SHA = {
    0: "5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7",
    1: "f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c",
    2: "60928c8caefbe8d09c5dc5a1390f9f73a76336a18617a825652a6ac2923d1c15",
}
ARMS = {
    "D_VQ_JOINT": (True, 0.25),
    "D_VQ_TASK_B0": (False, 0.0),
    "D_VQ_TASK": (False, 0.25),
    "D_VQ_TASK_B1": (False, 1.0),
}
SOURCE_FILES = ("prototype.py", "trainer.py", "checkpoint.py", "controls_fit.py",
                "cache_store.py", "encoder_adapter.py", "deployment.py",
                "attack_views.py", "private_runner.py")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _full_hex(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_seal(lock_path: Path, lock_sha256: str, seal_path: Path,
                data_manifest_path: Path, shared_root: Path) -> tuple[dict, dict, str]:
    """Verify the effective, committed S1 lock and independent role admission."""
    if (lock_path.name != "CAMPAIGN_LOCK.json" or not _full_hex(lock_sha256)
            or hash_file(lock_path) != lock_sha256):
        raise PermissionError("campaign lock hash/name is absent or changed")
    lock = json.loads(lock_path.read_text())
    if (lock.get("schema") != "pcrl-method-portfolio-campaign-lock-v1"
            or lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
            or lock.get("adult_fit_permission") is not True
            or lock.get("source_tip") != SOURCE_TIP):
        raise PermissionError("S1 campaign lock is not effective for Adult D stages")
    manifest_sha = hash_file(data_manifest_path)
    if lock.get("data_admission", {}).get("manifest_sha256") != manifest_sha:
        raise PermissionError("lock does not bind admitted role-only data manifest")
    audit_sha = lock.get("data_admission", {}).get("s4_audit_receipt_sha256")
    audit_path = lock_path.with_name("S4_ROLE_EXPORT_VERIFICATION.json")
    if not _full_hex(audit_sha) or hash_file(audit_path) != audit_sha:
        raise PermissionError("independent S4 role audit hash mismatch")
    audit = json.loads(audit_path.read_text())
    if (audit.get("schema") != "pcrl-s4-independent-role-export-audit-v1"
            or audit.get("status") != "PASS"
            or audit.get("source_sha256") != SOURCE_NPZ_SHA
            or audit.get("barred_id_overlap") != 0
            or audit.get("export_group_overlap") != 0
            or audit.get("export_npz_count") != 25):
        raise PermissionError("independent S4 role admission did not pass")
    manifest = json.loads(data_manifest_path.read_text())
    if (manifest.get("schema") != "pcrl-method-portfolio-data-access-v1"
            or manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED"
            or manifest.get("source", {}).get("source_npz_sha256") != SOURCE_NPZ_SHA
            or manifest.get("source", {}).get("feature_names_sha256_newline_join") != FEATURE_SHA
            or manifest.get("custodian_export", {}).get("s4_report_sha256") != audit_sha):
        raise PermissionError("S2 role manifest is not the S4-admitted source/schema")
    family = lock.get("families", {}).get("D", {})
    if family.get("method_spec_sha256") != hash_file(Path(__file__).with_name("METHOD_SPEC.md")):
        raise PermissionError("S1 lock does not bind Team D method spec")
    source_map = family.get("source_code_sha256")
    if not isinstance(source_map, dict) or set(source_map) != set(SOURCE_FILES):
        raise PermissionError("S1 lock lacks the exact D source-code map")
    for filename in SOURCE_FILES:
        if source_map[filename] != hash_file(Path(__file__).with_name(filename)):
            raise PermissionError(f"Team D source differs from S1 lock: {filename}")
    shared_map = lock.get("shared_source_code_sha256")
    if (not isinstance(shared_map, dict)
            or not {"safe_data_guard.py", "common_eval.py"} <= set(shared_map)):
        raise PermissionError("S1 lock lacks S2 role-store/calibration source hashes")
    for filename in ("safe_data_guard.py", "common_eval.py"):
        if shared_map[filename] != hash_file(shared_root / filename):
            raise PermissionError(f"S2 shared source differs from S1 lock: {filename}")
    locked_arms = lock.get("arms", {})
    for arm_id, (protected, weight) in ARMS.items():
        arm = locked_arms.get(arm_id, {})
        if (arm.get("family") != "D" or arm.get("track") != "feature"
                or arm.get("protected") is not protected
                or arm.get("reconstruction_weight") != weight):
            raise PermissionError(f"S1 lock does not bind D arm {arm_id}")
    if (seal_path.name != "LOCAL_SEAL_RECEIPT.json"
            or seal_path.resolve().parent != lock_path.resolve().parent):
        raise PermissionError("local Git seal receipt must be adjacent to lock")
    seal = json.loads(seal_path.read_text())
    commit = seal.get("commit_sha", seal.get("lock_commit_sha", ""))
    if (not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or seal.get("campaign_lock_sha256", seal.get("lock_sha256")) != lock_sha256):
        raise PermissionError("local Git seal receipt does not bind this lock")
    repo = lock_path.resolve().parents[2]
    relative = lock_path.resolve().relative_to(repo)
    committed = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                               capture_output=True, check=False)
    if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != lock_sha256:
        raise PermissionError("campaign lock bytes are not in the sealed Git commit")
    return lock, manifest, manifest_sha


def _load_shared(shared_root: Path) -> None:
    """Import only the S1-hash-checked S2 modules, before any role-only read."""
    root = shared_root.resolve(strict=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    guard = importlib.import_module("safe_data_guard")
    evaluation = importlib.import_module("common_eval")
    for module, filename in ((guard, "safe_data_guard.py"),
                             (evaluation, "common_eval.py")):
        if Path(module.__file__).resolve() != root / filename:
            raise PermissionError("S2 shared module resolved outside the sealed source root")
    globals().update({
        "SafeRoleStore": guard.SafeRoleStore,
        "assert_same_people": evaluation.assert_same_people,
        "fit_global_temperature": evaluation.fit_global_temperature,
        "apply_global_temperature": evaluation.apply_global_temperature,
        "load_calibrator": evaluation.load_calibrator,
        "save_calibrator": evaluation.save_calibrator,
    })


def _arm_config(lock: dict, arm_id: str) -> tuple[bool, float]:
    if arm_id not in ARMS or arm_id not in lock["arms"]:
        raise PermissionError("unregistered D pooled arm")
    return ARMS[arm_id]


def _private_root(path: Path) -> Path:
    root = path.resolve(strict=True)
    _private_target(root / "child", root)
    return root


def _private_name(value: str) -> str:
    if ID_PATTERN.fullmatch(value) is None:
        raise ValueError("private cache/run ID must be a simple path component")
    return value


def _model_identity(model_manifest_path: Path, model_path: Path,
                    seed: int) -> str:
    if seed not in U_MODEL_SHA:
        raise PermissionError("source model seed is not registered")
    model_manifest = json.loads(model_manifest_path.read_text())
    matches = [p for p in model_manifest.get("packages", []) if p.get("release") == "SRC|U"]
    expected = U_MODEL_SHA[seed]
    if (len(matches) != 1
            or matches[0].get("model_pt_sha256", {}).get(str(seed)) != expected
            or model_path.name != "model.pt"
            or model_path.parent.name != f"rel__s{seed}__U"
            or hash_file(model_path) != expected):
        raise PermissionError("U source checkpoint differs from hcal's admitted pin")
    return expected


def _cache_binding(lock_sha: str, data_sha: str) -> dict:
    return {"campaign_lock_sha256": lock_sha,
            "data_manifest_sha256": data_sha,
            "controls_fit_sha256": hash_file(Path(__file__).with_name("controls_fit.py")),
            "encoder_adapter_sha256": hash_file(Path(__file__).with_name("encoder_adapter.py"))}


def _fit_binding(manifest: dict, lock_sha: str, data_sha: str,
                 model_sha: str, seed: int, protected: bool) -> dict:
    fields = ["features", "income", "occupation", f"references_s{seed}"]
    if protected:
        fields.append("sex")
    specs = manifest["real_file_specs"]
    role_files = {field: specs[f"OSF_DEFENSE_FIT:{field}"]["sha256"] for field in fields}
    role_files["data_manifest_sha256"] = data_sha
    role_files["fit_row_unit_pair_sha256"] = specs["OSF_DEFENSE_FIT:features"]["row_unit_pair_sha256"]
    return {"campaign_lock_sha256": lock_sha,
            "role_export_sha256": _canonical_hash(role_files),
            "source_model_sha256": model_sha,
            "trainer_sha256": hash_file(Path(__file__).with_name("trainer.py"))}


def _fit_data(store: SafeRoleStore, encoder: FrozenUEncoder, seed: int,
              protected: bool) -> tuple[DTrainingData, dict]:
    role = "OSF_DEFENSE_FIT"
    feature = store.read(role, "features")
    income = store.read(role, "income", "defense_fit")
    occupation = store.read(role, "occupation", "defense_fit")
    references = store.read(role, f"references_s{seed}")
    records = [feature, income, occupation, references]
    sex = store.read(role, "sex", "defense_fit") if protected else None
    if sex is not None:
        records.append(sex)
    assert_same_people(*records)
    names = tuple(str(name) for name in feature["feature_names"])
    h0, h1, source_receipt = encoder.encode_with_receipt(feature["X"], names, role)
    rid = feature["row_id"]
    data = DTrainingData(h0, h1, income["income"], occupation["occupation"],
                         references["Ucal_1"], references["Ucal_2"],
                         None if sex is None else sex["sex"], rid, rid, role)
    return data, source_receipt


def _store_and_encoder(args, manifest: dict, model_sha: str) -> tuple[SafeRoleStore, FrozenUEncoder]:
    store = SafeRoleStore(args.data_manifest)
    if store.manifest != manifest:
        raise PermissionError("SafeRoleStore did not load the sealed data manifest")
    first_role = {"init": "OSF_DEFENSE_FIT", "fit": "OSF_DEFENSE_FIT",
                  "calibrate": "CALIBRATION_HELDOUT",
                  "release": "ATTACK_FIT_NEW"}[args.stage]
    # SafeRoleStore verifies the source-pinned name hash before returning this
    # stage's own role-only feature file. No FIT array is opened in CAL/release.
    feature = store.read(first_role, "features")
    names = tuple(str(value) for value in feature["feature_names"])
    encoder = FrozenUEncoder.load(args.model_pt, model_sha, tuple(names),
                                  FEATURE_SHA, seed=args.seed)
    return store, encoder


def _context(args) -> tuple[dict, dict, str, str, SafeRoleStore, FrozenUEncoder, Path]:
    lock, manifest, data_sha = _check_seal(args.campaign_lock, args.lock_sha256,
                                           args.local_seal_receipt, args.data_manifest,
                                           args.shared_root)
    if args.seed not in (0, 1, 2):
        raise PermissionError("unregistered paired model seed")
    if args.stage != "init":
        _arm_config(lock, args.arm_id)
    _load_shared(args.shared_root)
    model_sha = _model_identity(args.model_manifest, args.model_pt, args.seed)
    private_root = _private_root(args.private_root)
    store, encoder = _store_and_encoder(args, manifest, model_sha)
    return lock, manifest, data_sha, model_sha, store, encoder, private_root


def run_initializer(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root = _context(args)
    cache_id = _private_name(args.cache_id)
    target = _private_target(private_root / cache_id, private_root)
    if target.exists():
        raise FileExistsError("initializer cache already exists; use its immutable identity")
    data, source_receipt = _fit_data(store, encoder, args.seed, protected=False)
    pair_sha = manifest["real_file_specs"]["OSF_DEFENSE_FIT:features"]["row_unit_pair_sha256"]
    config = ControlFitConfig()
    cache = prepare_control_cache(data, args.seed, config,
                                  source_model_sha256=(model_sha, model_sha),
                                  row_order_sha256=pair_sha)
    cache.validate(data, args.seed, config, (model_sha, model_sha), pair_sha)
    saved = save_private_cache(cache, target, _cache_binding(args.lock_sha256, data_sha),
                               private_root=private_root,
                               source_encoder_receipt=source_receipt)
    return {"stage": "INIT", "seed": args.seed, "cache_id": cache_id,
            "initializer_cache_id": cache.receipt["cache_id"],
            "cache_complete_sha256": _sha256(target / "COMPLETE.json"),
            "cache_arrays_sha256": saved["arrays_sha256"],
            "physical_kmeans_fits": 3,
            "source_encoder_forward_calls": 1,
            "purpose_encoder_forwards": 2,
            "sex_labels_read": False}


def _fit_complete(run_dir: Path, expected_arm: str, expected_seed: int,
                  expected_lock_sha: str, expected_model_sha: str) -> dict:
    record = json.loads((run_dir / "FIT_COMPLETE.json").read_text())
    if (record.get("schema") != "team-d-private-fit-stage-v1"
            or record.get("arm_id") != expected_arm
            or record.get("seed") != expected_seed
            or record.get("campaign_lock_sha256") != expected_lock_sha
            or record.get("source_model_sha256") != expected_model_sha
            or record.get("checkpoint_complete_sha256") !=
            _sha256(run_dir / "checkpoint" / "COMPLETE.json")):
        raise ValueError("D private fit stage identity or checkpoint hash mismatch")
    return record


def run_fit(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root = _context(args)
    protected, weight = _arm_config(lock, args.arm_id)
    cache_id, run_id = _private_name(args.cache_id), _private_name(args.run_id)
    if cache_id == run_id:
        raise ValueError("fit and cache IDs must differ")
    run_dir = _private_target(private_root / run_id, private_root)
    if run_dir.exists():
        raise FileExistsError("private D fit ID already exists; preserve it and use a new ID")
    data, source_receipt = _fit_data(store, encoder, args.seed, protected)
    pair_sha = manifest["real_file_specs"]["OSF_DEFENSE_FIT:features"]["row_unit_pair_sha256"]
    cache_dir = _private_target(private_root / cache_id, private_root)
    cache = load_private_cache(cache_dir, _cache_binding(args.lock_sha256, data_sha),
                               private_root=private_root)
    cache.validate(data, args.seed, ControlFitConfig(),
                   (model_sha, model_sha), pair_sha)
    config = DTrainConfig(reconstruction_weight=weight)
    run_dir.mkdir(mode=0o700)
    fit = train_d(data, args.seed, protected, config,
                  initializer=(cache.normalizer, cache.pooled_centroids))
    binding = _fit_binding(manifest, args.lock_sha256, data_sha,
                           model_sha, args.seed, protected)
    checkpoint_manifest = save_private_checkpoint(fit, run_dir / "checkpoint",
                                                  binding, private_root=private_root)
    record = {
        "schema": "team-d-private-fit-stage-v1", "arm_id": args.arm_id,
        "seed": args.seed, "protected": protected,
        "reconstruction_weight": weight,
        "campaign_lock_sha256": args.lock_sha256,
        "data_manifest_sha256": data_sha,
        "source_model_sha256": model_sha,
        "fit_row_unit_pair_sha256": pair_sha,
        "source_encoder_parity": source_receipt,
        "cache_id": cache_id,
        "initializer_cache_id": cache.receipt["cache_id"],
        "initializer_cache_arrays_sha256": _sha256(cache_dir / "arrays.npz"),
        "fit_binding": binding,
        "checkpoint_complete_sha256": _sha256(run_dir / "checkpoint" / "COMPLETE.json"),
        "checkpoint_weights_sha256": checkpoint_manifest["weights_sha256"],
        "codebook_sha256": hashlib.sha256(
            np.ascontiguousarray(fit.release.codebook.detach().cpu().numpy()).tobytes()).hexdigest(),
        "fit_rows": fit.fit_rows,
        "fit_code_change": fit.fit_code_change,
        "source_encoder_forward_calls": 1,
        "purpose_encoder_forwards": 2,
        "head_probability_conversion": "torch float32 softmax -> float64 cast -> row normalization",
    }
    _write_json(run_dir / "FIT_COMPLETE.json", record)
    return {"stage": "FIT", "arm_id": args.arm_id, "seed": args.seed,
            "run_id": run_id, "fit_rows": fit.fit_rows,
            "fit_complete_sha256": _sha256(run_dir / "FIT_COMPLETE.json"),
            "checkpoint_weights_sha256": checkpoint_manifest["weights_sha256"],
            "initializer_cache_id": cache.receipt["cache_id"],
            "source_encoder_forward_calls": 1,
            "purpose_encoder_forwards": 2,
            "sex_labels_read": protected}


def _restored_fit(args, manifest: dict, data_sha: str, model_sha: str,
                  private_root: Path, protected: bool) -> tuple[Path, object, dict]:
    run_dir = _private_target(private_root / _private_name(args.run_id), private_root)
    record = _fit_complete(run_dir, args.arm_id, args.seed,
                           args.lock_sha256, model_sha)
    expected = _fit_binding(manifest, args.lock_sha256, data_sha,
                            model_sha, args.seed, protected)
    if record.get("fit_binding") != expected or record.get("protected") is not protected:
        raise ValueError("private D fit has a different data/arm/source binding")
    fit = load_private_checkpoint(run_dir / "checkpoint", expected,
                                  private_root=private_root)
    if (fit.protected is not protected or fit.seed != args.seed
            or fit.fit_rows != manifest["roles"]["OSF_DEFENSE_FIT"]["rows"]
            or fit.config.reconstruction_weight != ARMS[args.arm_id][1]):
        raise ValueError("restored D fit differs from registered arm/role")
    codebook_hash = hashlib.sha256(np.ascontiguousarray(
        fit.release.codebook.detach().cpu().numpy()).tobytes()).hexdigest()
    if record.get("codebook_sha256") != codebook_hash:
        raise ValueError("restored D public codebook differs from fit receipt")
    return run_dir, fit, record


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256(path)


def _existing_npz_equals(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (set(archive.files) == set(arrays)
                    and all(np.array_equal(archive[key], value)
                            for key, value in arrays.items()))
    except (OSError, ValueError):
        return False


def _save_role_package(run_dir: Path, private_root: Path, store: SafeRoleStore,
                       manifest: dict, encoder: FrozenUEncoder,
                       deployed: DCalibratedDeployment, role: str,
                       fit_record: dict) -> dict:
    if role not in ("CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise PermissionError("D release construction requires a permitted development role")
    feature = store.read(role, "features")
    names = tuple(str(name) for name in feature["feature_names"])
    h0, h1, parity = encoder.encode_with_receipt(feature["X"], names, role)
    wires = tuple(deployed.recipient(purpose, feature["X"], names, role=role)
                  for purpose in (0, 1))
    # The full role-specific deployment path must reproduce the exact hard IDs
    # and decoded code vectors of the TorchDRelease fed the source latents.
    for purpose, latent in enumerate((h0, h1)):
        raw = deployed.release.recipient(purpose, latent)
        if (not np.array_equal(wires[purpose]["code_ids"], raw["code_ids"])
                or not np.array_equal(wires[purpose]["decoded"], raw["decoded"])):
            raise ValueError("D runtime release differs from source-latent code path")
    pair_hash = manifest["real_file_specs"][f"{role}:features"]["row_unit_pair_sha256"]
    aligned = AlignedDViews(feature["row_id"], feature["unit"],
                            feature["row_id"], feature["row_id"],
                            wires[0], wires[1], pair_hash)
    target = _private_target(run_dir / role, private_root)
    if not target.exists():
        target.mkdir(mode=0o700)
    elif not target.is_dir() or (target.stat().st_mode & 0o077):
        raise ValueError("partial role release package has invalid private custody")
    arrays = {"row_id": aligned.row_id, "unit": aligned.unit_id}
    for purpose, wire in enumerate((aligned.recipient0, aligned.recipient1)):
        for field in ("code_ids", "decoded", "probabilities", "decision"):
            arrays[f"r{purpose}_{field}"] = wire[field]
    wire_path = target / "wire.npz"
    if wire_path.exists():
        if not _existing_npz_equals(wire_path, arrays):
            raise ValueError("partial role wire differs from deterministic replay")
        npz_hash = _sha256(wire_path)
    else:
        npz_hash = _write_npz_once(wire_path, arrays)
    package = {
        "schema": "team-d-private-calibrated-wire-v1",
        "role": role, "rows": int(len(aligned.row_id)),
        "arm_id": deployed.arm_id, "seed": deployed.seed,
        "row_unit_pair_sha256": pair_hash,
        "role_features_sha256": manifest["real_file_specs"][f"{role}:features"]["sha256"],
        "source_encoder_parity": parity,
        "source_model_sha256": encoder.model_sha256,
        "checkpoint_weights_sha256": fit_record["checkpoint_weights_sha256"],
        "codebook_sha256": fit_record["codebook_sha256"],
        "calibrator_sha256": list(deployed.calibrator_sha256),
        "wire_metadata": [wire["metadata"] for wire in (aligned.recipient0, aligned.recipient1)],
        "wire_members": sorted(arrays),
        "wire_npz_sha256": npz_hash,
        "head_probability_conversion": fit_record["head_probability_conversion"],
        "source_encoder_forward_calls": 3,
        "purpose_encoder_forwards": 4,
        "head_forward_calls": 4,
        "forward_accounting": "one pair parity plus two local deployment encoder calls; two deployment and two replay head calls",
        "payload_scope": "private aligned released wire only; no labels or raw X",
    }
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != package:
            raise ValueError("partial role manifest differs from deterministic replay")
    else:
        _write_json(manifest_path, package)
    complete = {
        "schema": package["schema"],
        "manifest_sha256": _sha256(manifest_path),
        "wire_npz_sha256": npz_hash,
    }
    complete_path = target / "COMPLETE.json"
    if complete_path.exists():
        if json.loads(complete_path.read_text()) != complete:
            raise ValueError("completed role package hash differs from deterministic replay")
    else:
        _write_json(complete_path, complete)
    return {"role": role, "rows": package["rows"],
            "row_unit_pair_sha256": pair_hash,
            "wire_npz_sha256": npz_hash,
            "package_complete_sha256": _sha256(complete_path),
            "source_parity": parity["bitwise_source_parity"],
            "source_encoder_forward_calls": 3,
            "purpose_encoder_forwards": 4,
            "head_forward_calls": 4}


def run_calibrate(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root = _context(args)
    protected, _ = _arm_config(lock, args.arm_id)
    run_dir, fit, fit_record = _restored_fit(args, manifest, data_sha, model_sha,
                                             private_root, protected)
    if (run_dir / "CAL_COMPLETE.json").exists():
        raise FileExistsError("D calibration is already complete and immutable")
    role = "CALIBRATION_HELDOUT"
    feature = store.read(role, "features")
    income = store.read(role, "income", "calibration")
    occupation = store.read(role, "occupation", "calibration")
    assert_same_people(feature, income, occupation)
    names = tuple(str(value) for value in feature["feature_names"])
    h0, h1, _ = encoder.encode_with_receipt(feature["X"], names, role)
    raw = (fit.release.recipient(0, h0), fit.release.recipient(1, h1))
    tasks = ("income", "occupation")
    labels = (income["income"], occupation["occupation"])
    records = tuple(fit_global_temperature(
        normalize_head_probabilities(raw[purpose]["probabilities"]), labels[purpose],
        task=tasks[purpose], arm_id=args.arm_id, model_seed=args.seed,
        row_id=feature["row_id"], unit=feature["unit"])
        for purpose in (0, 1))
    cal_dir = _private_target(run_dir / "calibrators", private_root)
    if not cal_dir.exists():
        cal_dir.mkdir(mode=0o700)
    elif not cal_dir.is_dir() or (cal_dir.stat().st_mode & 0o077):
        raise ValueError("partial private calibrator directory has invalid custody")
    hashes = tuple(calibrator_record_hash(record) for record in records)
    for purpose, record in enumerate(records):
        path = cal_dir / f"purpose{purpose}.json"
        if path.exists():
            if load_calibrator(path, hashes[purpose]) != record:
                raise ValueError("partial D calibrator differs from deterministic refit")
        elif save_calibrator(record, path) != hashes[purpose]:
            raise ValueError("S2 saved calibrator hash differs from bound record")
    checked = tuple(load_calibrator(cal_dir / f"purpose{purpose}.json", hashes[purpose])
                    for purpose in (0, 1))
    deployed = DCalibratedDeployment(encoder, fit.release, checked, hashes,
                                     args.arm_id, args.seed,
                                     apply_global_temperature)
    package = _save_role_package(run_dir, private_root, store, manifest,
                                 encoder, deployed, role, fit_record)
    result = {
        "schema": "team-d-private-calibration-stage-v1",
        "arm_id": args.arm_id, "seed": args.seed,
        "campaign_lock_sha256": args.lock_sha256,
        "fit_complete_sha256": _sha256(run_dir / "FIT_COMPLETE.json"),
        "calibrator_sha256": list(hashes),
        "calibration_role_release": package,
        "calibration_family": "H-GLOBAL-TEMP",
        "head_probability_conversion": fit_record["head_probability_conversion"],
        "additional_source_encoder_forward_calls_before_package": 1,
        "additional_purpose_encoder_forwards_before_package": 2,
        "additional_head_forward_calls_before_package": 2,
    }
    _write_json(run_dir / "CAL_COMPLETE.json", result)
    return {"stage": "CALIBRATE", "arm_id": args.arm_id, "seed": args.seed,
            "run_id": args.run_id,
            "cal_complete_sha256": _sha256(run_dir / "CAL_COMPLETE.json"),
            "calibrator_sha256": list(hashes),
            "calibration_rows": package["rows"], "sex_labels_read": False}


def run_release(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root = _context(args)
    protected, _ = _arm_config(lock, args.arm_id)
    run_dir, fit, fit_record = _restored_fit(args, manifest, data_sha, model_sha,
                                             private_root, protected)
    cal_record = json.loads((run_dir / "CAL_COMPLETE.json").read_text())
    if (cal_record.get("schema") != "team-d-private-calibration-stage-v1"
            or cal_record.get("campaign_lock_sha256") != args.lock_sha256
            or cal_record.get("fit_complete_sha256") != _sha256(run_dir / "FIT_COMPLETE.json")
            or cal_record.get("arm_id") != args.arm_id
            or cal_record.get("seed") != args.seed):
        raise ValueError("D private calibration stage has wrong fit/lock identity")
    if (run_dir / "RELEASE_COMPLETE.json").exists():
        raise FileExistsError("D ATT/INNER release stage is already complete")
    hashes = tuple(cal_record["calibrator_sha256"])
    cal_dir = run_dir / "calibrators"
    records = tuple(load_calibrator(cal_dir / f"purpose{purpose}.json", hashes[purpose])
                    for purpose in (0, 1))
    deployed = DCalibratedDeployment(encoder, fit.release, records, hashes,
                                     args.arm_id, args.seed,
                                     apply_global_temperature)
    packages = [_save_role_package(run_dir, private_root, store, manifest,
                                   encoder, deployed, role, fit_record)
                for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")]
    result = {
        "schema": "team-d-private-screen-wire-stage-v1",
        "arm_id": args.arm_id, "seed": args.seed,
        "campaign_lock_sha256": args.lock_sha256,
        "fit_complete_sha256": _sha256(run_dir / "FIT_COMPLETE.json"),
        "cal_complete_sha256": _sha256(run_dir / "CAL_COMPLETE.json"),
        "calibrator_sha256": list(hashes),
        "packages": packages,
        "sex_or_task_labels_read": False,
    }
    _write_json(run_dir / "RELEASE_COMPLETE.json", result)
    return {"stage": "RELEASE", "arm_id": args.arm_id, "seed": args.seed,
            "run_id": args.run_id,
            "release_complete_sha256": _sha256(run_dir / "RELEASE_COMPLETE.json"),
            "packages": packages, "sex_or_task_labels_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "fit", "calibrate", "release"))
    parser.add_argument("--campaign-lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--local-seal-receipt", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-pt", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache-id")
    parser.add_argument("--run-id")
    parser.add_argument("--arm-id")
    args = parser.parse_args()
    if args.stage == "init":
        if args.cache_id is None or args.run_id is not None or args.arm_id is not None:
            parser.error("init requires cache ID and no arm/run ID")
        receipt = run_initializer(args)
    else:
        if args.run_id is None or args.arm_id is None:
            parser.error("fit/calibrate/release require run and arm IDs")
        if args.stage == "fit":
            if args.cache_id is None:
                parser.error("fit requires the registered initializer cache ID")
            receipt = run_fit(args)
        elif args.stage == "calibrate":
            receipt = run_calibrate(args)
        else:
            receipt = run_release(args)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

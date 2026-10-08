"""S3-queued private FIT/CAL/RELEASE stages for Team D's U-source controls.

This module cannot open Adult role files until the effective S1 lock, S4 role
admission and S3 queue gate are verified by the caller and `private_runner`.
Every stage saves only private checkpoints, calibrated wire or public hashes.
RAW-J requires a separately admitted historical encoder and is not silently
substituted by any U-source control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import private_runner as base
from attack_views import AlignedDViews
from cache_store import load_private_cache
from controls_fit import (ControlFitConfig, fit_continuous,
                          fit_local_or_sequential)
from deployment import DCalibratedDeployment, calibrator_record_hash, normalize_head_probabilities


CONTROL_ARMS = {
    "D_U_FROZEN": ("UNCHANGED", False),
    "D_VQ_LOCAL": ("LOCAL", True),
    "D_VQ_SEQ12": ("SEQ12", True),
    "D_VQ_SEQ21": ("SEQ21", True),
    "D_LEACE": ("LEACE", True),
}
CONTROL_SOURCE_FILES = ("control_runner.py", "control_checkpoint.py",
                        "controls_fit.py", "deployment.py", "attack_views.py")


def _check_control_lock(lock: dict, arm_id: str) -> tuple[str, bool]:
    if arm_id == "D_RAW_J":
        raise PermissionError("RAW-J needs separate S2-admitted historical source identity")
    if arm_id not in CONTROL_ARMS:
        raise PermissionError("unknown or unregistered D control arm")
    mode, protected = CONTROL_ARMS[arm_id]
    family = lock.get("families", {}).get("D", {})
    expected = family.get("control_source_code_sha256")
    if not isinstance(expected, dict) or set(expected) != set(CONTROL_SOURCE_FILES):
        raise PermissionError("S1 lock lacks the exact D control-runner source map")
    for name in CONTROL_SOURCE_FILES:
        if expected[name] != base.hash_file(Path(__file__).with_name(name)):
            raise PermissionError(f"D control source differs from S1 lock: {name}")
    arm = lock.get("arms", {}).get(arm_id, {})
    if (arm.get("family") != "D" or arm.get("track") != "feature"
            or arm.get("mode") != mode or arm.get("protected") is not protected):
        raise PermissionError("S1 lock does not bind this D control construction")
    return mode, protected


def _context(args):
    # Refuse a missing control arm/source identity before base._context opens
    # even an allowed role-only feature archive.
    prelock, _, _ = base._check_seal(args.campaign_lock, args.lock_sha256,
                                     args.local_seal_receipt, args.data_manifest,
                                     args.shared_root)
    mode, protected = _check_control_lock(prelock, args.arm_id)
    lock, manifest, data_sha, model_sha, store, encoder, private_root = base._context(args)
    if lock != prelock:
        raise PermissionError("D control campaign lock changed during admission")
    return lock, manifest, data_sha, model_sha, store, encoder, private_root, mode, protected


def _binding(manifest: dict, lock_sha: str, data_sha: str,
             model_sha: str, seed: int, protected: bool) -> dict:
    fields = ["features", "income", "occupation", f"references_s{seed}"]
    if protected:
        fields.append("sex")
    specs = manifest["real_file_specs"]
    files = {field: specs[f"OSF_DEFENSE_FIT:{field}"]["sha256"] for field in fields}
    files["data_manifest_sha256"] = data_sha
    files["fit_row_unit_pair_sha256"] = specs["OSF_DEFENSE_FIT:features"]["row_unit_pair_sha256"]
    return {"campaign_lock_sha256": lock_sha,
            "role_export_sha256": base._canonical_hash(files),
            "source_model_sha256": model_sha,
            "trainer_sha256": base.hash_file(Path(__file__).with_name("controls_fit.py"))}


def _checkpoint_api():
    from control_checkpoint import (load_private_control_checkpoint,
                                    save_private_control_checkpoint)
    return load_private_control_checkpoint, save_private_control_checkpoint


def run_fit(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root, mode, protected = _context(args)
    cache_id, run_id = base._private_name(args.cache_id), base._private_name(args.run_id)
    if cache_id == run_id:
        raise ValueError("control cache and run identities must differ")
    run_dir = base._private_target(private_root / run_id, private_root)
    if run_dir.exists():
        raise FileExistsError("D control fit identity is already present")
    data, source_receipt = base._fit_data(store, encoder, args.seed, protected)
    pair_sha = manifest["real_file_specs"]["OSF_DEFENSE_FIT:features"]["row_unit_pair_sha256"]
    config = ControlFitConfig()
    cache_dir = base._private_target(private_root / cache_id, private_root)
    cache = load_private_cache(cache_dir, base._cache_binding(args.lock_sha256, data_sha),
                               private_root=private_root)
    cache.validate(data, args.seed, config, (model_sha, model_sha), pair_sha)
    if mode in ("LOCAL", "SEQ12", "SEQ21"):
        fit = fit_local_or_sequential(data, args.seed, mode, config,
                                      normalizer=cache.normalizer,
                                      initializers=cache.purpose_centroids)
    else:
        fit = fit_continuous(data, args.seed, mode, config, normalizer=cache.normalizer)
    run_dir.mkdir(mode=0o700)
    _, save_checkpoint = _checkpoint_api()
    binding = _binding(manifest, args.lock_sha256, data_sha, model_sha,
                       args.seed, protected)
    checkpoint = save_checkpoint(fit, run_dir / "checkpoint", binding,
                                 private_root=private_root)
    internal_fits = ({"transform_training_fits": 2, "separate_head_fits": 0,
                      "leace_map_fits": 0} if mode in ("LOCAL", "SEQ12", "SEQ21") else
                     {"transform_training_fits": 0, "separate_head_fits": 2,
                      "leace_map_fits": 2 if mode == "LEACE" else 0})
    record = {
        "schema": "team-d-private-control-fit-v1", "arm_id": args.arm_id,
        "mode": mode, "seed": args.seed, "protected": protected,
        "campaign_lock_sha256": args.lock_sha256,
        "data_manifest_sha256": data_sha, "source_model_sha256": model_sha,
        "fit_row_unit_pair_sha256": pair_sha, "source_encoder_parity": source_receipt,
        "cache_id": cache_id, "initializer_cache_id": cache.receipt["cache_id"],
        "initializer_cache_arrays_sha256": base._sha256(cache_dir / "arrays.npz"),
        "fit_binding": binding,
        "checkpoint_complete_sha256": base._sha256(run_dir / "checkpoint" / "COMPLETE.json"),
        "checkpoint_manifest": checkpoint,
        "fit_rows": fit.fit_rows, "fit_history_epochs": len(fit.history),
        "internal_fit_units": internal_fits,
        "map_metadata": fit.map_metadata,
        "source_encoder_forward_calls": 1, "purpose_encoder_forwards": 2,
        "head_probability_conversion": "torch float32 softmax -> float64 cast -> row normalization",
    }
    base._write_json(run_dir / "FIT_COMPLETE.json", record)
    return {"stage": "FIT", "arm_id": args.arm_id, "seed": args.seed,
            "run_id": run_id, "fit_complete_sha256": base._sha256(run_dir / "FIT_COMPLETE.json"),
            "initializer_cache_id": cache.receipt["cache_id"],
            "internal_fit_units": internal_fits,
            "sex_labels_read": protected}


def _restore_fit(args, manifest: dict, data_sha: str, model_sha: str,
                 private_root: Path, mode: str, protected: bool):
    run_dir = base._private_target(private_root / base._private_name(args.run_id), private_root)
    record = json.loads((run_dir / "FIT_COMPLETE.json").read_text())
    binding = _binding(manifest, args.lock_sha256, data_sha, model_sha,
                       args.seed, protected)
    if (record.get("schema") != "team-d-private-control-fit-v1"
            or record.get("arm_id") != args.arm_id or record.get("mode") != mode
            or record.get("seed") != args.seed or record.get("protected") is not protected
            or record.get("fit_binding") != binding
            or record.get("checkpoint_complete_sha256") !=
            base._sha256(run_dir / "checkpoint" / "COMPLETE.json")):
        raise ValueError("control checkpoint identity or FIT receipt differs")
    load_checkpoint, _ = _checkpoint_api()
    fit = load_checkpoint(run_dir / "checkpoint", binding,
                          private_root=private_root, expected_mode=mode)
    if fit.fit_rows != manifest["roles"]["OSF_DEFENSE_FIT"]["rows"]:
        raise ValueError("restored control fitting row count differs")
    return run_dir, fit, record


def _save_wire_package(run_dir: Path, private_root: Path, store, manifest: dict,
                       encoder, deployed: DCalibratedDeployment, role: str,
                       fit_record: dict) -> dict:
    if role not in ("CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise PermissionError("D control wire role is not admitted")
    feature = store.read(role, "features")
    names = tuple(str(name) for name in feature["feature_names"])
    h0, h1, parity = encoder.encode_with_receipt(feature["X"], names, role)
    wires = tuple(deployed.recipient(purpose, feature["X"], names, role=role)
                  for purpose in (0, 1))
    for purpose, latent in enumerate((h0, h1)):
        raw = deployed.release.recipient(purpose, latent)
        if (not np.array_equal(wires[purpose]["code_ids"], raw["code_ids"])
                or not np.array_equal(wires[purpose]["decoded"], raw["decoded"])):
            raise ValueError("D control deployment differs from source-latent replay")
    pair_sha = manifest["real_file_specs"][f"{role}:features"]["row_unit_pair_sha256"]
    aligned = AlignedDViews(feature["row_id"], feature["unit"],
                            feature["row_id"], feature["row_id"],
                            wires[0], wires[1], pair_sha)
    target = base._private_target(run_dir / role, private_root)
    if not target.exists():
        target.mkdir(mode=0o700)
    elif not target.is_dir() or (target.stat().st_mode & 0o077):
        raise ValueError("partial control wire package has invalid private custody")
    arrays = {"row_id": aligned.row_id, "unit": aligned.unit_id}
    for purpose, wire in enumerate((aligned.recipient0, aligned.recipient1)):
        for field in ("code_ids", "decoded", "probabilities", "decision"):
            arrays[f"r{purpose}_{field}"] = wire[field]
    wire_path = target / "wire.npz"
    if wire_path.exists():
        if not base._existing_npz_equals(wire_path, arrays):
            raise ValueError("partial control wire differs from deterministic replay")
        wire_sha = base._sha256(wire_path)
    else:
        wire_sha = base._write_npz_once(wire_path, arrays)
    package = {
        "schema": "team-d-private-control-wire-v1", "role": role,
        "arm_id": deployed.arm_id, "seed": deployed.seed,
        "rows": int(len(aligned.row_id)), "row_unit_pair_sha256": pair_sha,
        "role_features_sha256": manifest["real_file_specs"][f"{role}:features"]["sha256"],
        "source_encoder_parity": parity,
        "source_model_sha256": encoder.model_sha256,
        "checkpoint_complete_sha256": fit_record["checkpoint_complete_sha256"],
        "calibrator_sha256": list(deployed.calibrator_sha256),
        "wire_metadata": [wire["metadata"] for wire in (aligned.recipient0, aligned.recipient1)],
        "wire_members": sorted(arrays), "wire_npz_sha256": wire_sha,
        "source_encoder_forward_calls": 3,
        "purpose_encoder_forwards": 4,
        "head_forward_calls": 4,
        "forward_accounting": "one pair parity plus two local deployment encoder calls; two deployment and two replay head calls",
        "payload_scope": "private aligned released wire only; no labels, raw X or frozen latents",
    }
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != package:
            raise ValueError("partial control wire manifest differs from deterministic replay")
    else:
        base._write_json(manifest_path, package)
    complete = {"schema": package["schema"],
                "manifest_sha256": base._sha256(manifest_path),
                "wire_npz_sha256": wire_sha}
    complete_path = target / "COMPLETE.json"
    if complete_path.exists():
        if json.loads(complete_path.read_text()) != complete:
            raise ValueError("completed control wire hash differs from deterministic replay")
    else:
        base._write_json(complete_path, complete)
    return {"role": role, "rows": package["rows"], "row_unit_pair_sha256": pair_sha,
            "wire_npz_sha256": wire_sha,
            "package_complete_sha256": base._sha256(complete_path),
            "source_encoder_forward_calls": 3,
            "purpose_encoder_forwards": 4,
            "head_forward_calls": 4}


def run_calibrate(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root, mode, protected = _context(args)
    run_dir, fit, fit_record = _restore_fit(args, manifest, data_sha, model_sha,
                                             private_root, mode, protected)
    if (run_dir / "CAL_COMPLETE.json").exists():
        raise FileExistsError("control calibration already completed")
    role = "CALIBRATION_HELDOUT"
    feature = store.read(role, "features")
    income = store.read(role, "income", "calibration")
    occupation = store.read(role, "occupation", "calibration")
    base.assert_same_people(feature, income, occupation)
    names = tuple(str(name) for name in feature["feature_names"])
    h0, h1, _ = encoder.encode_with_receipt(feature["X"], names, role)
    raw = (fit.release.recipient(0, h0), fit.release.recipient(1, h1))
    records = tuple(base.fit_global_temperature(
        normalize_head_probabilities(raw[purpose]["probabilities"]),
        labels, task=task, arm_id=args.arm_id, model_seed=args.seed,
        row_id=feature["row_id"], unit=feature["unit"])
        for purpose, (task, labels) in enumerate((("income", income["income"]),
                                                   ("occupation", occupation["occupation"]))))
    cal_dir = base._private_target(run_dir / "calibrators", private_root)
    cal_dir.mkdir(mode=0o700, exist_ok=True)
    hashes = tuple(calibrator_record_hash(record) for record in records)
    for purpose, record in enumerate(records):
        path = cal_dir / f"purpose{purpose}.json"
        if path.exists():
            if base.load_calibrator(path, hashes[purpose]) != record:
                raise ValueError("partial control calibrator differs")
        elif base.save_calibrator(record, path) != hashes[purpose]:
            raise ValueError("saved control calibrator differs")
    checked = tuple(base.load_calibrator(cal_dir / f"purpose{purpose}.json", hashes[purpose])
                    for purpose in (0, 1))
    deployed = DCalibratedDeployment(encoder, fit.release, checked, hashes,
                                     args.arm_id, args.seed, base.apply_global_temperature)
    package = _save_wire_package(run_dir, private_root, store, manifest, encoder,
                                 deployed, role, fit_record)
    receipt = {"schema": "team-d-private-control-cal-v1", "arm_id": args.arm_id,
               "seed": args.seed, "campaign_lock_sha256": args.lock_sha256,
               "fit_complete_sha256": base._sha256(run_dir / "FIT_COMPLETE.json"),
               "calibrator_sha256": list(hashes), "calibration_role_release": package,
               "calibration_family": "H-GLOBAL-TEMP",
               "additional_source_encoder_forward_calls_before_package": 1,
               "additional_purpose_encoder_forwards_before_package": 2,
               "additional_head_forward_calls_before_package": 2}
    base._write_json(run_dir / "CAL_COMPLETE.json", receipt)
    return {"stage": "CALIBRATE", "arm_id": args.arm_id, "seed": args.seed,
            "run_id": args.run_id, "cal_complete_sha256": base._sha256(run_dir / "CAL_COMPLETE.json"),
            "calibrator_sha256": list(hashes), "sex_labels_read": False}


def run_release(args) -> dict:
    lock, manifest, data_sha, model_sha, store, encoder, private_root, mode, protected = _context(args)
    run_dir, fit, fit_record = _restore_fit(args, manifest, data_sha, model_sha,
                                             private_root, mode, protected)
    cal = json.loads((run_dir / "CAL_COMPLETE.json").read_text())
    if (cal.get("schema") != "team-d-private-control-cal-v1"
            or cal.get("arm_id") != args.arm_id or cal.get("seed") != args.seed
            or cal.get("campaign_lock_sha256") != args.lock_sha256
            or cal.get("fit_complete_sha256") != base._sha256(run_dir / "FIT_COMPLETE.json")):
        raise ValueError("control calibration does not bind this fit")
    if (run_dir / "RELEASE_COMPLETE.json").exists():
        raise FileExistsError("control development release already completed")
    hashes = tuple(cal["calibrator_sha256"])
    records = tuple(base.load_calibrator(run_dir / "calibrators" / f"purpose{purpose}.json",
                                         hashes[purpose]) for purpose in (0, 1))
    deployed = DCalibratedDeployment(encoder, fit.release, records, hashes,
                                     args.arm_id, args.seed, base.apply_global_temperature)
    packages = [_save_wire_package(run_dir, private_root, store, manifest, encoder,
                                   deployed, role, fit_record)
                for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")]
    receipt = {"schema": "team-d-private-control-release-v1", "arm_id": args.arm_id,
               "seed": args.seed, "campaign_lock_sha256": args.lock_sha256,
               "fit_complete_sha256": base._sha256(run_dir / "FIT_COMPLETE.json"),
               "cal_complete_sha256": base._sha256(run_dir / "CAL_COMPLETE.json"),
               "calibrator_sha256": list(hashes), "packages": packages,
               "sex_or_task_labels_read": False}
    base._write_json(run_dir / "RELEASE_COMPLETE.json", receipt)
    return {"stage": "RELEASE", "arm_id": args.arm_id, "seed": args.seed,
            "run_id": args.run_id,
            "release_complete_sha256": base._sha256(run_dir / "RELEASE_COMPLETE.json"),
            "packages": packages, "sex_or_task_labels_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "calibrate", "release"))
    for name in ("campaign_lock", "local_seal_receipt", "data_manifest", "shared_root",
                 "model_manifest", "model_pt", "private_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--cache-id")
    args = parser.parse_args()
    if args.stage == "fit":
        if args.cache_id is None:
            parser.error("control FIT requires registered cache ID")
        receipt = run_fit(args)
    elif args.stage == "calibrate":
        receipt = run_calibrate(args)
    else:
        receipt = run_release(args)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

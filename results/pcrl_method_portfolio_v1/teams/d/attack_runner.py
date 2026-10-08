"""S3-queued private Team D fresh-reader fit on complete released wires.

This unit reads only admitted ATTACK_FIT_NEW and INNER_SELECTION SEX labels.
The former fits and tunes readers; the latter is opened after all reader
models, orientations and ignore-recipient aliases are frozen. A failed unit
keeps its per-reader private receipts and must receive a new S1/S3 identity
before any rerun. It never loads the historical all-role archive.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np

import private_runner as base
from attack_views import AlignedDViews
from d_cell_slate import fit_d_cell_readers
from s2_attack_adapter import as_s2_attack_view


CODE_ARMS = frozenset(("D_VQ_TASK_B0", "D_VQ_TASK", "D_VQ_TASK_B1",
                       "D_VQ_JOINT", "D_VQ_LOCAL", "D_VQ_SEQ12", "D_VQ_SEQ21"))
CONTINUOUS_ARMS = frozenset(("D_U_FROZEN", "D_LEACE", "D_RAW_J"))
ATTACK_SOURCE_FILES = ("attack_runner.py", "attack_views.py",
                       "s2_attack_adapter.py", "d_cell_slate.py")
REQUIRED_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")
VIEWS = ("recipient0", "recipient1", "pair")


def _admit(args):
    lock, manifest, data_sha = base._check_seal(
        args.campaign_lock, args.lock_sha256, args.local_seal_receipt,
        args.data_manifest, args.shared_root)
    if args.arm_id not in CODE_ARMS | CONTINUOUS_ARMS or args.seed not in (0, 1, 2) \
            or args.attacker_seed not in (0, 1, 2):
        raise PermissionError("D arm or paired seed is not registered")
    if args.arm_id == "D_RAW_J":
        raise PermissionError("RAW-J attack waits for separate S2/S4 source parity admission")
    arm = lock.get("arms", {}).get(args.arm_id, {})
    if arm.get("family") != "D" or arm.get("track") != "feature":
        raise PermissionError("D attack arm has no sealed feature-track identity")
    registered = lock.get("attack_units", {}).get(args.attack_run_id)
    if (not isinstance(registered, dict)
            or registered.get("family") != "D"
            or registered.get("arm_id") != args.arm_id
            or registered.get("source_seed") != args.seed
            or registered.get("attacker_seed") != args.attacker_seed
            or registered.get("source_run_id") != args.source_run_id
            or registered.get("source_model_sha256") != base.U_MODEL_SHA[args.seed]
            or registered.get("stage") not in ("SCREEN", "FINALIST")):
        raise PermissionError("D source/attacker unit pairing is not S1-registered")
    family = lock.get("families", {}).get("D", {})
    expected = family.get("attack_source_code_sha256")
    if not isinstance(expected, dict) or set(expected) != set(ATTACK_SOURCE_FILES):
        raise PermissionError("S1 lock lacks the exact D attack-runner source map")
    for name in ATTACK_SOURCE_FILES:
        if expected[name] != base.hash_file(Path(__file__).with_name(name)):
            raise PermissionError(f"D attack source differs from effective lock: {name}")
    shared_source = lock.get("shared_source_code_sha256", {})
    shared_attack = Path(args.shared_root).resolve(strict=True) / "attack_eval.py"
    if shared_source.get("attack_eval.py") != base.hash_file(shared_attack):
        raise PermissionError("S2 fresh-reader source differs from effective lock")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise PermissionError(f"{variable}=1 is required by the S3 attack unit")
    base._load_shared(args.shared_root)
    attack = importlib.import_module("attack_eval")
    if Path(attack.__file__).resolve() != shared_attack:
        raise PermissionError("S2 attack module resolved outside the sealed source root")
    if args.source_run_id == args.attack_run_id:
        raise ValueError("source and attack run identities must differ")
    root = base._private_root(args.private_root)
    if (root / base._private_name(args.source_run_id)).is_symlink():
        raise PermissionError("D source run directory may not be a symlink")
    source = base._private_target(root / base._private_name(args.source_run_id), root)
    target = base._private_target(root / base._private_name(args.attack_run_id), root)
    if not source.is_dir() or (source.stat().st_mode & 0o077) or target.exists():
        raise PermissionError("source custody missing or attack run identity already used")
    store = base.SafeRoleStore(args.data_manifest)
    if store.manifest != manifest:
        raise PermissionError("S2 role store does not match sealed manifest")
    return lock, manifest, data_sha, attack, store, source, target, root


def _source_release_record(source: Path, args) -> dict:
    fit_path, cal_path, release_path = (source / name for name in
                                        ("FIT_COMPLETE.json", "CAL_COMPLETE.json",
                                         "RELEASE_COMPLETE.json"))
    if any(path.is_symlink() or (path.stat().st_mode & 0o077)
           for path in (fit_path, cal_path, release_path)):
        raise PermissionError("D source stage file custody is invalid")
    fit, cal, release = (json.loads(path.read_text()) for path in
                         (fit_path, cal_path, release_path))
    pooled = args.arm_id in {"D_VQ_TASK_B0", "D_VQ_TASK", "D_VQ_TASK_B1",
                             "D_VQ_JOINT"}
    expected_schemas = (
        ("team-d-private-fit-stage-v1", "team-d-private-calibration-stage-v1",
         "team-d-private-screen-wire-stage-v1") if pooled else
        ("team-d-private-control-fit-v1", "team-d-private-control-cal-v1",
         "team-d-private-control-release-v1"))
    if (fit.get("arm_id") != args.arm_id or cal.get("arm_id") != args.arm_id
            or release.get("arm_id") != args.arm_id
            or tuple(record.get("schema") for record in (fit, cal, release)) != expected_schemas
            or any(record.get("seed") != args.seed for record in (fit, cal, release))
            or any(record.get("campaign_lock_sha256") != args.lock_sha256
                   for record in (fit, cal, release))
            or fit.get("source_model_sha256") != base.U_MODEL_SHA[args.seed]
            or cal.get("fit_complete_sha256") != base._sha256(fit_path)
            or release.get("fit_complete_sha256") != base._sha256(fit_path)
            or release.get("cal_complete_sha256") != base._sha256(cal_path)
            or len(release.get("packages", [])) != 2):
        raise ValueError("D source wire stage identity or fit/calibration chain differs")
    if {item.get("role") for item in release["packages"]} != {
            "ATTACK_FIT_NEW", "INNER_SELECTION"}:
        raise ValueError("D source release has an incomplete ATT/INNER role set")
    return {**release, "fit_source_model_sha256": fit["source_model_sha256"]}


def _load_wire(source: Path, role: str, release_record: dict,
               manifest: dict, args) -> AlignedDViews:
    if role not in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise PermissionError("D attack runner may read only ATT/INNER released wire")
    package_dir = source / role
    if (package_dir.is_symlink() or not package_dir.is_dir()
            or (package_dir.stat().st_mode & 0o077)
            or any((package_dir / name).is_symlink() for name in
                   ("manifest.json", "wire.npz", "COMPLETE.json"))):
        raise PermissionError("D source wire package custody is invalid")
    package_path = package_dir / "manifest.json"
    wire_path = package_dir / "wire.npz"
    complete_path = package_dir / "COMPLETE.json"
    package = json.loads(package_path.read_text())
    complete = json.loads(complete_path.read_text())
    stage = next(item for item in release_record["packages"] if item["role"] == role)
    spec = manifest["real_file_specs"][f"{role}:features"]
    if (package.get("schema") not in ("team-d-private-calibrated-wire-v1",
                                     "team-d-private-control-wire-v1")
            or package.get("role") != role or package.get("arm_id") != args.arm_id
            or package.get("seed") != args.seed
            or package.get("rows") != manifest["roles"][role]["rows"]
            or package.get("row_unit_pair_sha256") != spec["row_unit_pair_sha256"]
            or package.get("role_features_sha256") != spec["sha256"]
            or package.get("calibrator_sha256") != release_record["calibrator_sha256"]
            or package.get("source_model_sha256") != release_record["fit_source_model_sha256"]
            or complete.get("schema") != package["schema"]
            or complete.get("manifest_sha256") != base._sha256(package_path)
            or complete.get("wire_npz_sha256") != base._sha256(wire_path)
            or package.get("wire_npz_sha256") != complete["wire_npz_sha256"]
            or stage.get("package_complete_sha256") != base._sha256(complete_path)
            or stage.get("wire_npz_sha256") != complete["wire_npz_sha256"]):
        raise ValueError("released D role wire differs from sealed role or source stage")
    expected_members = {"row_id", "unit"} | {
        f"r{purpose}_{field}" for purpose in (0, 1)
        for field in ("code_ids", "decoded", "probabilities", "decision")}
    if set(package.get("wire_members", [])) != expected_members:
        raise ValueError("D released wire member declaration differs")
    with np.load(wire_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_members:
            raise ValueError("D released wire contains omitted or extra arrays")
        arrays = {field: np.array(archive[field], copy=True) for field in expected_members}
    metadata = package.get("wire_metadata")
    if not isinstance(metadata, list) or len(metadata) != 2:
        raise ValueError("D public metadata declaration is incomplete")
    if any(not isinstance(meta, dict) or meta.get("calibrator_sha256") !=
           release_record["calibrator_sha256"][purpose]
           for purpose, meta in enumerate(metadata)):
        raise ValueError("D transmitted calibrator identity differs from source stage")
    receivers = tuple({field: arrays[f"r{purpose}_{field}"] for field in
                       ("code_ids", "decoded", "probabilities", "decision")} |
                      {"metadata": metadata[purpose]} for purpose in (0, 1))
    return AlignedDViews(arrays["row_id"], arrays["unit"],
                         arrays["row_id"], arrays["row_id"],
                         receivers[0], receivers[1], spec["row_unit_pair_sha256"])


def _check_sex_alignment(views: AlignedDViews, sex: dict, role: str) -> None:
    if not np.array_equal(views.row_id, sex["row_id"]) or \
            not np.array_equal(views.unit_id, sex["unit"]):
        raise ValueError(f"{role} released wire and admitted SEX role order differ")


def run_attack(args) -> dict:
    _, manifest, data_sha, attack, store, source, target, root = _admit(args)
    release_record = _source_release_record(source, args)
    att = _load_wire(source, "ATTACK_FIT_NEW", release_record, manifest, args)
    inner = _load_wire(source, "INNER_SELECTION", release_record, manifest, args)
    if any((view.recipient0["code_ids"].shape[1] == 1) != (args.arm_id in CODE_ARMS)
           or (view.recipient1["code_ids"].shape[1] == 1) != (args.arm_id in CODE_ARMS)
           for view in (att, inner)):
        raise ValueError("D code/continuous arm identity differs from released interface")
    sex_att = store.read("ATTACK_FIT_NEW", "sex", "attack")
    _check_sex_alignment(att, sex_att, "ATTACK_FIT_NEW")
    target.mkdir(mode=0o700)
    fitted_by_view, s2_views = {}, {}
    for name in VIEWS:
        view_dir = target / name
        view_dir.mkdir(mode=0o700)
        att_s2 = as_s2_attack_view(att, name, attack.AttackView)
        inner_s2 = as_s2_attack_view(inner, name, attack.AttackView)
        fitted = attack.fit_slate(att_s2, sex_att, inner_s2,
                                  required_readers=REQUIRED_READERS,
                                  attacker_seed=args.attacker_seed,
                                  progress_dir=view_dir / "learned_progress")
        if args.arm_id in CODE_ARMS:
            specialist = fit_d_cell_readers(
                att, inner, name, att_s2, inner_s2, sex_att, attack,
                model_dir=view_dir / "specialist_progress", private_root=root)
            fitted = attack.add_external_fixed_readers(
                fitted, att_s2, sex_att, inner_s2, specialist)
        fitted_by_view[name] = fitted
        s2_views[name] = (att_s2, inner_s2)
        base._write_json(view_dir / "FIT_COMPLETE.json", {
            "schema": "team-d-private-reader-view-fit-v1", "view": name,
            "arm_id": args.arm_id, "seed": args.seed,
            "attacker_seed": args.attacker_seed,
            "required_readers": list(REQUIRED_READERS),
            "reader_order": fitted["reader_order"],
            "reader_progress": fitted.get("reader_progress", []),
            "external_readers": fitted.get("external_readers", []),
            "att_view_content_sha256": att_s2.content_hashes(),
            "inner_view_content_sha256": inner_s2.content_hashes(),
        })
    fitted_by_view["pair"] = attack.add_ignore_recipient_aliases(
        fitted_by_view["pair"], fitted_by_view["recipient0"],
        fitted_by_view["recipient1"])
    # Immutable models, ATTACK_TUNE orientations and all aliases are in
    # custody before any INNER label is opened. A failed compound unit retains
    # partial receipts but requires a new S1/S3 run identity to refit.
    custody_by_view = {}
    for name in VIEWS:
        att_s2, inner_s2 = s2_views[name]
        fitted = fitted_by_view[name]
        custody_by_view[name] = attack.save_fitted_slate(
            fitted, target / name / "frozen", att_view=att_s2,
            inner_view=inner_s2)
    base._write_json(target / "FROZEN_COMPLETE.json", {
        "schema": "team-d-private-attack-freeze-v1",
        "arm_id": args.arm_id, "seed": args.seed,
        "attacker_seed": args.attacker_seed,
        "private_reader_custody": custody_by_view,
    })
    sex_inner = store.read("INNER_SELECTION", "sex", "screen")
    _check_sex_alignment(inner, sex_inner, "INNER_SELECTION")
    scores = {}
    for name in VIEWS:
        fitted = fitted_by_view[name]
        scores[name] = attack.score_slate(fitted, sex_inner)
        base._write_json(target / name / "SCORE_COMPLETE.json", {
            "schema": "team-d-private-reader-view-score-v1",
            "view": name, "arm_id": args.arm_id, "seed": args.seed,
            "attacker_seed": args.attacker_seed,
            "source_release_complete_sha256": base._sha256(source / "RELEASE_COMPLETE.json"),
            "frozen_complete_sha256": base._sha256(target / "FROZEN_COMPLETE.json"),
            "private_reader_custody": custody_by_view[name], "score": scores[name],
        })
    aggregate = {
        "schema": "team-d-private-attack-unit-v1", "arm_id": args.arm_id,
        "seed": args.seed, "attacker_seed": args.attacker_seed,
        "campaign_lock_sha256": args.lock_sha256,
        "data_manifest_sha256": data_sha,
        "source_release_complete_sha256": base._sha256(source / "RELEASE_COMPLETE.json"),
        "reader_fit_units": 21 if args.arm_id in CODE_ARMS else 12,
        "ignore_other_recipient_aliases_new_fits": 0,
        "score_complete_sha256": {name: base._sha256(target / name / "SCORE_COMPLETE.json")
                                  for name in VIEWS},
        "primary_envelope_auc": {name: scores[name]["primary_envelope_auc"]
                                 for name in VIEWS},
        "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
    }
    base._write_json(target / "COMPLETE.json", aggregate)
    return {"arm_id": args.arm_id, "seed": args.seed,
            "attacker_seed": args.attacker_seed,
            "reader_fit_units": aggregate["reader_fit_units"],
            "complete_sha256": base._sha256(target / "COMPLETE.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign_lock", "local_seal_receipt", "data_manifest",
                 "shared_root", "private_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attacker-seed", type=int, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--attack-run-id", required=True)
    print(json.dumps(run_attack(parser.parse_args()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

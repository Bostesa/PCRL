"""Queue-only independent saved-view and reader replay for five D controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .independent_d_view import build_d_view, compare_saved_view
from .independent_f_view import build_f_view
from .independent_reader_custody import (verify_control_custody,
                                         verify_ignore_aliases,
                                         verify_reader_custody,
                                         verify_tune_selection)
from .independent_replay import reader_recovery, weighted_binary_auc
from .replay_first_pair import (_npz, _queue_guard, _read_json,
                                _role_npz, _wire, sha256)


ARMS = ("L", "S12", "S21", "C", "F")
VIEWS = ("r1", "r2", "pair")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
D_FIELDS = {"row_id", "unit", "code0", "code1", "prob0", "prob1", "decision0", "decision1"}
F_FIELDS = {"row_id", "unit", "features0", "features1", "prob0", "prob1", "decision0", "decision1"}


def _f_wire(release: dict, manifest: dict, purpose: int) -> dict:
    calibrator = manifest["calibrators"][purpose]
    payload = (json.dumps(calibrator, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    metadata = {"schema": "latent-unchanged-v1", "purpose": purpose,
                "latent_dimensions": manifest["latent_dim"],
                "version": manifest["version"],
                "source_sha256": manifest["source_sha256"],
                "calibrator_sha256": hashlib.sha256(payload).hexdigest(),
                "calibration_family": "H-GLOBAL-TEMP"}
    return {"features": release[f"features{purpose}"],
            "probabilities": release[f"prob{purpose}"],
            "decision": release[f"decision{purpose}"], "metadata": metadata}


def run(args: argparse.Namespace) -> dict:
    seal = Path(args.seal).resolve(strict=True)
    _queue_guard(seal, args.seal_sha256)
    lock = _read_json(seal, args.seal_sha256)
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("effective D seal differs")
    amendment = _read_json(Path(args.attack_amendment), args.attack_amendment_sha256)
    if (amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != args.seal_sha256):
        raise PermissionError("D followup attack amendment differs")
    role_path = Path(args.role_manifest).resolve(strict=True)
    role = _read_json(role_path, args.role_manifest_sha256)
    if args.role_manifest_sha256 != lock["source_files_sha256"]["source_role_manifest"]:
        raise ValueError("followup attack role source differs")
    source_receipt = _read_json(Path(args.source_receipt), args.source_receipt_sha256)
    source_verify = _read_json(Path(args.source_verification), args.source_verification_sha256)
    first_verify = _read_json(Path(args.first_verification), args.first_verification_sha256)
    if (source_receipt.get("status") != "D_FOLLOWUP_RELEASES_READY"
            or source_receipt.get("seed") != args.seed
            or source_receipt.get("seal_sha256") != args.seal_sha256
            or source_verify.get("status") != "FOLLOWUP_RELEASE_REPLAY_PASS"
            or source_verify.get("followup_receipt_sha256") != args.source_receipt_sha256
            or source_verify.get("seed") != args.seed
            or first_verify.get("status") != "TECHNICAL_REPLAY_PASS"
            or first_verify.get("seed", 0) != args.seed
            or amendment.get("control_receipt_sha256_by_seed", {}).get(str(args.seed))
               != args.source_receipt_sha256):
        raise ValueError("paired first/followup independent source verification differs")
    attack_map = _read_json(Path(args.attack_map), args.attack_map_sha256)
    if set(attack_map) != set(ARMS):
        raise ValueError("all five fixed attack arms required")
    sex = {role_name: _role_npz(role, role_path, role_name, "sex") for role_name in ROLES}
    y = np.asarray(sex["INNER_SELECTION"]["sex"], dtype=np.int64)
    public = {"schema": "pcrl-r04-d-followup-attack-replay-v1",
              "status": "FOLLOWUP_COMPLETE_VIEW_REPLAY_PASS",
              "seed": args.seed, "seal_sha256": args.seal_sha256,
              "attack_amendment_sha256": args.attack_amendment_sha256,
              "source_receipt_sha256": args.source_receipt_sha256,
              "source_verification_sha256": args.source_verification_sha256,
              "first_verification_sha256": args.first_verification_sha256,
              "arms": {}, "all_seven_point": {}}
    source_root = Path(args.source_private).resolve(strict=True)
    if ".private" not in source_root.parts:
        raise PermissionError("five-arm source must remain in private custody")
    for arm in ARMS:
        entry = attack_map[arm]
        if set(entry) != {"receipt", "receipt_sha256", "private_root"}:
            raise ValueError("attack map needs exact receipt/hash/private fields")
        attack_receipt = _read_json(Path(entry["receipt"]), entry["receipt_sha256"])
        private = Path(entry["private_root"]).resolve(strict=True)
        if ".private" not in private.parts:
            raise PermissionError("attack reader custody must remain private")
        if (attack_receipt.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY"
                or attack_receipt.get("arm_id") != arm
                or attack_receipt.get("source_seed") != args.seed
                or attack_receipt.get("source_receipt_sha256") != args.source_receipt_sha256
                or attack_receipt.get("effective_lock_sha256") != args.seal_sha256
                or attack_receipt.get("attack_amendment_sha256") != args.attack_amendment_sha256):
            raise ValueError("control attack receipt differs from source/lock")
        manifest_path = source_root / arm / "manifest.json"
        manifest_sha = source_receipt["packages"][arm]["manifest_sha256"]
        manifest = _read_json(manifest_path, manifest_sha)
        if attack_receipt["service_manifest_sha256"] != manifest_sha:
            raise ValueError("attack used different public service package")
        release = {}
        for role_name in ROLES:
            release_sha = source_receipt["roles"][role_name][arm]["private_release_sha256"]
            if attack_receipt["source_release_sha256"][role_name] != release_sha:
                raise ValueError("attack used different private source release")
            release[role_name] = _npz(source_root / f"release__{arm}__{role_name}.npz",
                                      release_sha, F_FIELDS if arm == "F" else D_FIELDS)
            person = sex[role_name]
            if (not np.array_equal(release[role_name]["row_id"], person["row_id"])
                    or not np.array_equal(release[role_name]["unit"], person["unit"])):
                raise ValueError("followup attack release/SEX role identity differs")
        if sha256(private / "FROZEN_COMPLETE.json") != attack_receipt["attack"]["freeze_receipt_sha256"]:
            raise ValueError("pre-INNER reader freeze receipt differs")
        att, inner = release["ATTACK_FIT_NEW"], release["INNER_SELECTION"]
        if arm == "F":
            wire = lambda z, purpose: _f_wire(z, manifest, purpose)
            builder = build_f_view
        else:
            wire = lambda z, purpose: _wire(z, manifest, purpose)
            builder = build_d_view
        views = {}
        readers = {}
        controls = {}
        recovery = {}
        for view in VIEWS:
            att_built, inner_built = builder(wire(att, 0), wire(att, 1),
                                             wire(inner, 0), wire(inner, 1), view)
            reader_dir = private / view / "readers"
            views[view] = {
                "att": compare_saved_view(str(reader_dir / "view_att.npz"),
                                           att_built, att["row_id"], att["unit"]),
                "inner": compare_saved_view(str(reader_dir / "view_inner.npz"),
                                             inner_built, inner["row_id"], inner["unit"])}
            readers[view] = verify_reader_custody(
                reader_dir, expected_row_id=inner["row_id"],
                expected_unit=inner["unit"])
            readers[view]["tune"] = verify_tune_selection(
                reader_dir, att["row_id"], att["unit"], sex["ATTACK_FIT_NEW"]["sex"])
            if (readers[view]["receipt_sha256"] !=
                    attack_receipt["attack"]["reader_custody"][view]["private_receipt_sha256"]):
                raise ValueError("reader custody receipt differs from public attack")
            with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as archive:
                predictions = {name: np.array(archive[name], copy=True) for name in archive.files}
            recovery[view] = reader_recovery(predictions, y)
            published = attack_receipt["attack"]["scores"][view]
            if (abs(recovery[view]["auc"] - published["primary_envelope_auc"]) > 1e-10
                    or set(recovery[view]["reader_auc"]) != set(published["readers"])
                    or readers[view]["tune"]["winner"] != published["validation_winner"]
                    or readers[view]["tune"]["orientation"] !=
                       published["validation_winner_orientation"]):
                raise ValueError("public fixed-slate AUC/selection differs")
            for name, auc in recovery[view]["reader_auc"].items():
                if abs(auc - published["readers"][name]["fixed_alias_max_auc"]) > 1e-10:
                    raise ValueError("published individual reader AUC differs")
            selected_name = readers[view]["tune"]["winner"]
            selected_auc = weighted_binary_auc(predictions[selected_name], y)
            if readers[view]["tune"]["orientation"] == "reverse":
                selected_auc = 1 - selected_auc
            if abs(selected_auc - published["validation_winner_inner_auc"]) > 1e-10:
                raise ValueError("ATT-selected reader INNER AUC differs")
            controls[view] = verify_control_custody(
                private / view / "controls", inner["row_id"], inner["unit"])
            if (controls[view]["receipt_sha256"] !=
                    attack_receipt["attack"]["controls"][view]["custody"]["private_receipt_sha256"]):
                raise ValueError("engineering control receipt differs")
            for key in ("null_auc", "shuffled_att_independent_inner_auc", "planted_auc"):
                replay_key = ("shuffled_independent_inner_auc" if key ==
                              "shuffled_att_independent_inner_auc" else key)
                if abs(controls[view][replay_key] - attack_receipt["attack"]["controls"][view]["scores"][key]) > 1e-10:
                    raise ValueError("engineering control AUC differs")
        readers["aliases"] = verify_ignore_aliases(
            private / "pair" / "readers",
            private / "r1" / "readers", private / "r2" / "readers")
        public["arms"][arm] = {"attack_receipt_sha256": entry["receipt_sha256"],
                               "manifest_sha256": manifest_sha,
                               "views": views, "readers": readers,
                               "controls": controls, "sex_recovery": recovery,
                               "task": source_verify["arms"][arm]["task"],
                               "point_gate": source_verify["arms"][arm]["point_gate"]}
    public["all_seven_point"] = {
        **{arm: {"task": first_verify["task"][arm],
                 "point_gate": first_verify["point_gates"][arm],
                 "sex_recovery": first_verify["sex_recovery"][arm]}
           for arm in ("T", "D-K")},
        **{arm: {"task": public["arms"][arm]["task"],
                 "point_gate": public["arms"][arm]["point_gate"],
                 "sex_recovery": public["arms"][arm]["sex_recovery"]}
           for arm in ARMS}}
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("followup attack replay receipt already exists")
    output.write_text(json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"status": public["status"], "public_receipt_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "attack-amendment", "attack-amendment-sha256",
                 "role-manifest", "role-manifest-sha256", "source-private",
                 "source-receipt", "source-receipt-sha256",
                 "source-verification", "source-verification-sha256",
                 "first-verification", "first-verification-sha256",
                 "attack-map", "attack-map-sha256", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()

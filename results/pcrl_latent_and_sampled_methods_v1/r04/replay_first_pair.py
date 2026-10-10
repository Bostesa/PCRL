"""Queue-only independent replay of the first real D-T versus D-K comparison.

All per-person inputs and intermediate arrays remain in inherited private
custody. This module imports only the R04 implementations, not candidate or
attack runner scoring. Its public receipt contains aggregates and file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .independent_d_restore import verify_d_raw_to_wire, verify_fit_role_scaler
from .independent_d_view import build_d_view, compare_saved_view
from .independent_reader_custody import (
    verify_control_custody, verify_ignore_aliases, verify_reader_custody,
    verify_tune_selection,
)
from .independent_replay import (comparison_margins, reader_recovery, task_metrics,
                                 task_point_gate, weighted_binary_auc)


ARMS = ("T", "D-K")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
VIEWS = ("r1", "r2", "pair")
RELEASE_KEYS = {"row_id", "unit", "code0", "code1", "prob0", "prob1", "decision0", "decision1"}
ENDPOINT_KEYS = {"row_id", "unit", "income", "occupation", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path, expected: str | None = None) -> dict:
    if expected is not None and sha256(path) != expected:
        raise ValueError(f"JSON byte identity changed: {path.name}")
    return json.loads(path.read_text())


def _npz(path: Path, expected: str, members: set[str]) -> dict[str, np.ndarray]:
    if sha256(path) != expected:
        raise ValueError(f"private array byte identity changed: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != members:
            raise ValueError(f"private array member inventory changed: {path.name}")
        return {key: np.array(archive[key], copy=True) for key in members}


def _queue_guard(seal: Path, seal_sha: str) -> None:
    if (os.environ.get("PCRL_QUEUE_STAGE") != "verification"
            or os.environ.get("PCRL_QUEUE_SEAL_SHA256") != seal_sha
            or sha256(seal) != seal_sha):
        raise PermissionError("R04 replay requires the effective verification queue slot")
    root = Path(os.environ["PCRL_QUEUE_ROOT"])
    unit_id = os.environ["PCRL_QUEUE_UNIT_ID"]
    state = json.loads((root / "state.json").read_text())
    item = state.get("units", {}).get(unit_id, {})
    if (state.get("closed") or state.get("seal", {}).get("sha256") != seal_sha
            or item.get("status") != "running" or item.get("child_pid") != os.getpid()
            or item.get("spec", {}).get("stage") != "verification"):
        raise PermissionError("R04 replay is not the active R03 verification child")


def _role_npz(manifest: dict, manifest_path: Path, role: str, kind: str) -> dict:
    key = f"{role}:{kind}"
    relative = manifest["role_file_relative_path"][key]
    path = manifest_path.parent.parent / relative
    if path.is_symlink() or sha256(path) != manifest["role_file_sha256"][key]:
        raise ValueError(f"admitted role file changed: {key}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _metadata(manifest: dict, purpose: int) -> dict:
    calibrator = manifest["calibrators"][purpose]
    canonical = (json.dumps(calibrator, sort_keys=True, separators=(",", ":"),
                            allow_nan=False) + "\n").encode()
    return {"schema": "latent-repair-hardbits-v1", "purpose": purpose,
            "bits": 10, "bit_order": "c0_least_significant", "capacity_states": 1024,
            "version": manifest["version"], "source_sha256": manifest["source_sha256"],
            "model_sha256": manifest["model_sha256"],
            "head_sha256": manifest["head_sha256"][purpose],
            "calibration_family": "H-GLOBAL-TEMP",
            "calibrator_sha256": hashlib.sha256(canonical).hexdigest()}


def _wire(z: dict, manifest: dict, purpose: int) -> dict:
    return {"code": z[f"code{purpose}"], "probabilities": z[f"prob{purpose}"],
            "decision": z[f"decision{purpose}"], "metadata": _metadata(manifest, purpose)}


def run(args: argparse.Namespace) -> dict:
    seal = Path(args.seal).resolve(strict=True)
    _queue_guard(seal, args.seal_sha256)
    lock = _read_json(seal, args.seal_sha256)
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise PermissionError("effective D stage seal absent")
    role_path = Path(args.role_manifest).resolve(strict=True)
    if sha256(role_path) != lock["source_files_sha256"]["source_role_manifest"]:
        raise ValueError("inherited role manifest is not the sealed source")
    roles = _read_json(role_path)
    locked_roles = lock["role_identity_and_file_hashes"]
    if (roles.get("role_file_sha256") != locked_roles.get("role_file_sha256")
            or roles.get("roles") != locked_roles.get("role_membership")
            or roles.get("source_schema_sha256") != locked_roles.get("ordered_predictors_sha256")):
        raise ValueError("role archive/identity contract differs from effective seal")
    source_receipt = _read_json(Path(args.source_receipt), args.source_receipt_sha256)
    if (source_receipt.get("status") != "T_DK_RELEASES_READY"
            or source_receipt.get("seal_sha256") != args.seal_sha256
            or source_receipt.get("seed") != args.seed
            or source_receipt.get("role_manifest_sha256") != sha256(role_path)
            or source_receipt.get("source_sha256") !=
            locked_roles["source_model_sha256"][str(args.seed)]
            or set(source_receipt.get("packages", {})) != set(ARMS)):
        raise ValueError("first paired-seed source receipt identity differs")
    first_private = Path(args.source_private).resolve(strict=True)
    if ".private" not in first_private.parts:
        raise PermissionError("first pair source must be in private custody")
    checkpoint = Path(args.source_checkpoint).resolve(strict=True)
    if sha256(checkpoint) != source_receipt["source_sha256"]:
        raise ValueError("frozen source checkpoint changed")
    fit_feature = _role_npz(roles, role_path, "OSF_DEFENSE_FIT", "features")
    fit_income = _role_npz(roles, role_path, "OSF_DEFENSE_FIT", "income")
    fit_occupation = _role_npz(roles, role_path, "OSF_DEFENSE_FIT", "occupation")
    att_feature = _role_npz(roles, role_path, "ATTACK_FIT_NEW", "features")
    inner_feature = _role_npz(roles, role_path, "INNER_SELECTION", "features")
    features = {"OSF_DEFENSE_FIT": fit_feature,
                "ATTACK_FIT_NEW": att_feature, "INNER_SELECTION": inner_feature}
    for role, feature in features.items():
        expected = roles["roles"][role]
        if (len(feature["row_id"]) != expected["rows"]
                or len(np.unique(feature["unit"])) != expected["groups"]):
            raise ValueError("role row/group identity changed")
    endpoint_sha = source_receipt["roles"]["INNER_SELECTION"]["endpoint_labels_references_sha256"]
    endpoint = _npz(first_private / "endpoint__INNER_SELECTION.npz", endpoint_sha, ENDPOINT_KEYS)
    if (not np.array_equal(endpoint["row_id"], inner_feature["row_id"])
            or not np.array_equal(endpoint["unit"], inner_feature["unit"])):
        raise ValueError("task endpoint people differ from admitted INNER")
    sex = _role_npz(roles, role_path, "INNER_SELECTION", "sex")
    if (not np.array_equal(sex["row_id"], endpoint["row_id"])
            or not np.array_equal(sex["unit"], endpoint["unit"])):
        raise ValueError("SEX endpoint people differ from admitted INNER")
    sex_y = np.asarray(sex["sex"], dtype=np.int64)
    att_sex = _role_npz(roles, role_path, "ATTACK_FIT_NEW", "sex")
    if (not np.array_equal(att_sex["row_id"], att_feature["row_id"])
            or not np.array_equal(att_sex["unit"], att_feature["unit"])):
        raise ValueError("ATT SEX people differ from admitted attacker-fit role")
    feature_names = tuple(str(x) for x in fit_feature["feature_names"])
    feature_hash = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    for label in (fit_income, fit_occupation):
        if (not np.array_equal(label["row_id"], fit_feature["row_id"])
                or not np.array_equal(label["unit"], fit_feature["unit"])):
            raise ValueError("FIT constant-reference labels are not role aligned")
    majority = {
        "income": int(np.argmax(np.bincount(fit_income["income"], minlength=2))),
        "occupation": int(np.argmax(np.bincount(fit_occupation["occupation"], minlength=6))),
    }
    scaler = {}
    restore = {}
    view_replay = {}
    custody = {}
    task = {}
    recovery = {}
    for arm in ARMS:
        manifest_path = first_private / arm / "manifest.json"
        manifest_sha = source_receipt["packages"][arm]["manifest_sha256"]
        model_manifest = _read_json(manifest_path, manifest_sha)
        scaler[arm] = verify_fit_role_scaler(
            manifest_path=manifest_path, manifest_sha256=manifest_sha,
            source_checkpoint=checkpoint, source_sha256=source_receipt["source_sha256"],
            fit_x83=fit_feature["X"])
        release = {}
        restore[arm] = {}
        for role in ROLES:
            release_sha = source_receipt["roles"][role][arm]["private_release_sha256"]
            z = _npz(first_private / f"release__{arm}__{role}.npz", release_sha, RELEASE_KEYS)
            feature = features[role]
            if (not np.array_equal(z["row_id"], feature["row_id"])
                    or not np.array_equal(z["unit"], feature["unit"])):
                raise ValueError("release people differ from admitted role")
            release[role] = z
            restore[arm][role] = []
            for purpose in (0, 1):
                restore[arm][role].append(verify_d_raw_to_wire(
                    manifest_path=manifest_path, manifest_sha256=manifest_sha,
                    source_checkpoint=checkpoint, source_sha256=source_receipt["source_sha256"],
                    x83=feature["X"], feature_names=feature_names,
                    feature_names_sha256=feature_hash, purpose=purpose,
                    actual_wire=_wire(z, model_manifest, purpose)))
        attack_private = Path(args.attack_t_private if arm == "T" else args.attack_k_private).resolve(strict=True)
        attack_receipt_path = Path(args.attack_t_receipt if arm == "T" else args.attack_k_receipt)
        attack_sha = args.attack_t_receipt_sha256 if arm == "T" else args.attack_k_receipt_sha256
        attack_receipt = _read_json(attack_receipt_path, attack_sha)
        if (attack_receipt.get("arm_id") != arm or attack_receipt.get("source_seed") != args.seed
                or attack_receipt.get("source_receipt_sha256") != args.source_receipt_sha256
                or attack_receipt.get("effective_lock_sha256") != args.seal_sha256):
            raise ValueError("attack/source identity differs")
        if sha256(attack_private / "FROZEN_COMPLETE.json") != attack_receipt["attack"]["freeze_receipt_sha256"]:
            raise ValueError("pre-INNER frozen attack receipt differs")
        for role in ROLES:
            expected_sha = source_receipt["roles"][role][arm]["private_release_sha256"]
            if attack_receipt["source_release_sha256"][role] != expected_sha:
                raise ValueError("attack worker used different role release")
        custody[arm] = {}
        view_replay[arm] = {}
        recovery[arm] = {}
        a, i = release["ATTACK_FIT_NEW"], release["INNER_SELECTION"]
        for view in VIEWS:
            att_built, inner_built = build_d_view(_wire(a, model_manifest, 0),
                                                   _wire(a, model_manifest, 1),
                                                   _wire(i, model_manifest, 0),
                                                   _wire(i, model_manifest, 1), view)
            reader_dir = attack_private / view / "readers"
            view_replay[arm][view] = {
                "att": compare_saved_view(str(reader_dir / "view_att.npz"), att_built,
                                           a["row_id"], a["unit"]),
                "inner": compare_saved_view(str(reader_dir / "view_inner.npz"), inner_built,
                                             i["row_id"], i["unit"])}
            custody[arm][view] = verify_reader_custody(
                reader_dir, expected_row_id=i["row_id"], expected_unit=i["unit"])
            custody[arm][view]["tune"] = verify_tune_selection(
                reader_dir, a["row_id"], a["unit"], att_sex["sex"])
            if (custody[arm][view]["receipt_sha256"] !=
                    attack_receipt["attack"]["reader_custody"][view]["private_receipt_sha256"]):
                raise ValueError("published reader custody SHA differs")
            with np.load(reader_dir / "predictions_inner.npz", allow_pickle=False) as pred:
                reader_predictions = {name: np.array(pred[name], copy=True) for name in pred.files}
            recovery[arm][view] = reader_recovery(reader_predictions, sex_y)
            published = attack_receipt["attack"]["scores"][view]
            if (abs(recovery[arm][view]["auc"] - published["primary_envelope_auc"]) > 1e-10
                    or set(recovery[arm][view]["reader_auc"]) != set(published["readers"])
                    or custody[arm][view]["tune"]["winner"] != published["validation_winner"]
                    or custody[arm][view]["tune"]["orientation"] !=
                    published["validation_winner_orientation"]):
                raise ValueError("public primary recovery differs from independently replayed bank")
            for name, auc in recovery[arm][view]["reader_auc"].items():
                if abs(auc - published["readers"][name]["fixed_alias_max_auc"]) > 1e-10:
                    raise ValueError("public fixed-reader AUC differs")
            selected_name = custody[arm][view]["tune"]["winner"]
            selected_auc = weighted_binary_auc(reader_predictions[selected_name], sex_y)
            if custody[arm][view]["tune"]["orientation"] == "reverse":
                selected_auc = 1 - selected_auc
            if abs(selected_auc - published["validation_winner_inner_auc"]) > 1e-10:
                raise ValueError("published ATT-selected reader INNER AUC differs")
            custody[arm][view]["controls"] = verify_control_custody(
                attack_private / view / "controls", i["row_id"], i["unit"])
            if (custody[arm][view]["controls"]["receipt_sha256"] !=
                    attack_receipt["attack"]["controls"][view]["custody"]["private_receipt_sha256"]):
                raise ValueError("published control custody SHA differs")
            published_controls = attack_receipt["attack"]["controls"][view]["scores"]
            for key in ("null_auc", "shuffled_att_independent_inner_auc", "planted_auc"):
                replay_key = ("shuffled_independent_inner_auc" if key ==
                              "shuffled_att_independent_inner_auc" else key)
                if abs(custody[arm][view]["controls"][replay_key] - published_controls[key]) > 1e-10:
                    raise ValueError("published engineering control differs from private replay")
        custody[arm]["aliases"] = verify_ignore_aliases(
            attack_private / "pair" / "readers",
            attack_private / "r1" / "readers", attack_private / "r2" / "readers")
        task[arm] = {
            "income": task_metrics(i["prob0"], endpoint["income"], decision=i["decision0"],
                                   majority_class=majority["income"]),
            "occupation": task_metrics(i["prob1"], endpoint["occupation"], decision=i["decision1"],
                                       majority_class=majority["occupation"])}
    references = {
        "income": {ref: task_metrics(endpoint[f"{ref}_1"], endpoint["income"])
                   for ref in ("U0", "Ucal")},
        "occupation": {ref: task_metrics(endpoint[f"{ref}_2"], endpoint["occupation"])
                       for ref in ("U0", "Ucal")},
    }
    gates = {arm: task_point_gate(task[arm], references) for arm in ARMS}
    auc = {arm: {view: recovery[arm][view]["auc"] for view in VIEWS} for arm in ARMS}
    comparison = comparison_margins(auc["D-K"], auc["T"])
    receipt = {"schema": "r04-first-real-d-pair-replay-v1",
               "status": "TECHNICAL_REPLAY_PASS",
               "seed": args.seed,
               "effective_lock_sha256": args.seal_sha256,
               "source_receipt_sha256": args.source_receipt_sha256,
               "attack_receipt_sha256": {"T": args.attack_t_receipt_sha256,
                                         "D-K": args.attack_k_receipt_sha256},
               "role_manifest_sha256": sha256(role_path),
               "source_checkpoint_sha256": sha256(checkpoint),
               "endpoint_sha256": endpoint_sha,
               "fit_scaler": scaler, "raw_to_wire": restore,
               "complete_view": view_replay, "reader_custody": custody,
               "task": task, "references": references, "point_gates": gates,
               "sex_recovery": recovery, "D_K_vs_T_margins": comparison,
               "inference_scope": "one paired seed point comparison only; no two-seed simultaneous PASS"}
    output = Path(args.public_receipt)
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("R04 public replay receipt must be new")
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with output.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"status": receipt["status"], "public_receipt_sha256": sha256(output),
            "D_K_vs_T_margins": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "role-manifest", "source-checkpoint",
                 "source-private", "source-receipt", "source-receipt-sha256",
                 "attack-t-private", "attack-t-receipt", "attack-t-receipt-sha256",
                 "attack-k-private", "attack-k-receipt", "attack-k-receipt-sha256",
                 "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()

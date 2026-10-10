"""Queue-only independent D followup release and task replay for five arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .independent_d_restore import verify_d_raw_to_wire, verify_fit_role_scaler
from .independent_f_restore import verify_f_fit_scaler, verify_f_raw_to_wire
from .independent_replay import task_metrics, task_point_gate
from .replay_first_pair import (_npz, _queue_guard, _read_json,
                                _role_npz, _wire, sha256)


ARMS = ("L", "S12", "S21", "C", "F")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
D_FIELDS = {"row_id", "unit", "code0", "code1", "prob0", "prob1", "decision0", "decision1"}
F_FIELDS = {"row_id", "unit", "features0", "features1", "prob0", "prob1", "decision0", "decision1"}
ENDPOINT_FIELDS = {"row_id", "unit", "income", "occupation", "U0_1", "U0_2", "Ucal_1", "Ucal_2"}


def run(args: argparse.Namespace) -> dict:
    seal = Path(args.seal).resolve(strict=True)
    _queue_guard(seal, args.seal_sha256)
    lock = _read_json(seal, args.seal_sha256)
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE" or args.seed not in (0, 1):
        raise PermissionError("effective D seed/lock differs")
    amendment = _read_json(Path(args.amendment), args.amendment_sha256)
    if (amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != args.seal_sha256):
        raise PermissionError("followup amendment differs")
    role_path = Path(args.role_manifest).resolve(strict=True)
    roles = _read_json(role_path, args.role_manifest_sha256)
    if (args.role_manifest_sha256 != lock["source_files_sha256"]["source_role_manifest"]
            or roles.get("role_file_sha256") !=
                lock["role_identity_and_file_hashes"]["role_file_sha256"]):
        raise ValueError("admitted role source differs")
    source_receipt = _read_json(Path(args.followup_receipt), args.followup_receipt_sha256)
    first_receipt = _read_json(Path(args.first_receipt), args.first_receipt_sha256)
    if (source_receipt.get("status") != "D_FOLLOWUP_RELEASES_READY"
            or source_receipt.get("seed") != args.seed
            or source_receipt.get("seal_sha256") != args.seal_sha256
            or source_receipt.get("first_receipt_sha256") != args.first_receipt_sha256
            or source_receipt.get("source_sha256") != first_receipt.get("source_sha256")
            or source_receipt.get("role_manifest_sha256") != args.role_manifest_sha256
            or set(source_receipt.get("packages", {})) != set(ARMS)):
        raise ValueError("followup source and first pair differ")
    if (amendment.get("first_receipt_sha256_by_seed", {}).get(str(args.seed))
            != args.first_receipt_sha256
            or amendment.get("source_seed_sha256_by_seed", {}).get(str(args.seed))
            != source_receipt["source_sha256"]):
        raise ValueError("followup amendment does not bind paired seed source")
    checkpoint = Path(args.source_checkpoint).resolve(strict=True)
    if sha256(checkpoint) != source_receipt["source_sha256"]:
        raise ValueError("frozen U checkpoint changed")
    first_root = Path(args.first_private).resolve(strict=True)
    followup_root = Path(args.followup_private).resolve(strict=True)
    if ".private" not in first_root.parts or ".private" not in followup_root.parts:
        raise PermissionError("D private release custody absent")
    endpoint_hash = first_receipt["roles"]["INNER_SELECTION"]["endpoint_labels_references_sha256"]
    endpoint = _npz(first_root / "endpoint__INNER_SELECTION.npz", endpoint_hash, ENDPOINT_FIELDS)
    features = {role: _role_npz(roles, role_path, role, "features") for role in ROLES}
    fit_features = _role_npz(roles, role_path, "OSF_DEFENSE_FIT", "features")
    for role, feature in features.items():
        if (len(feature["row_id"]) != roles["roles"][role]["rows"]
                or len(np.unique(feature["unit"])) != roles["roles"][role]["groups"]):
            raise ValueError("followup role row/group count differs")
    if (not np.array_equal(endpoint["row_id"], features["INNER_SELECTION"]["row_id"])
            or not np.array_equal(endpoint["unit"], features["INNER_SELECTION"]["unit"])):
        raise ValueError("followup endpoint role identity differs")
    fit_labels = {task: _role_npz(roles, role_path, "OSF_DEFENSE_FIT", task)[task]
                  for task in ("income", "occupation")}
    majority = {task: int(np.argmax(np.bincount(labels,
                           minlength=2 if task == "income" else 6)))
                for task, labels in fit_labels.items()}
    feature_names = tuple(str(x) for x in fit_features["feature_names"])
    feature_hash = hashlib.sha256("\n".join(feature_names).encode()).hexdigest()
    references = {"income": {}, "occupation": {}}
    for task, purpose in (("income", 1), ("occupation", 2)):
        references[task] = {ref: task_metrics(endpoint[f"{ref}_{purpose}"], endpoint[task])
                            for ref in ("U0", "Ucal")}
    report = {"schema": "pcrl-r04-d-followup-source-replay-v1",
              "status": "FOLLOWUP_RELEASE_REPLAY_PASS", "seed": args.seed,
              "seal_sha256": args.seal_sha256,
              "amendment_sha256": args.amendment_sha256,
              "first_receipt_sha256": args.first_receipt_sha256,
              "followup_receipt_sha256": args.followup_receipt_sha256,
              "endpoint_sha256": endpoint_hash,
              "references": references, "arms": {}}
    for arm in ARMS:
        manifest_path = followup_root / arm / "manifest.json"
        manifest_hash = source_receipt["packages"][arm]["manifest_sha256"]
        manifest = _read_json(manifest_path, manifest_hash)
        scaler_checker = verify_f_fit_scaler if arm == "F" else verify_fit_role_scaler
        scaler = scaler_checker(manifest_path=manifest_path,
                                manifest_sha256=manifest_hash,
                                source_checkpoint=checkpoint,
                                source_sha256=source_receipt["source_sha256"],
                                fit_x83=fit_features["X"])
        by_role = {}
        for role in ROLES:
            release_hash = source_receipt["roles"][role][arm]["private_release_sha256"]
            release = _npz(followup_root / f"release__{arm}__{role}.npz",
                           release_hash, F_FIELDS if arm == "F" else D_FIELDS)
            feature = features[role]
            if (not np.array_equal(release["row_id"], feature["row_id"])
                    or not np.array_equal(release["unit"], feature["unit"])):
                raise ValueError("followup wire not aligned to role")
            purpose_report = []
            for purpose in (0, 1):
                if arm == "F":
                    item = verify_f_raw_to_wire(
                        manifest_path=manifest_path, manifest_sha256=manifest_hash,
                        source_checkpoint=checkpoint,
                        source_sha256=source_receipt["source_sha256"],
                        x83=feature["X"], purpose=purpose,
                        actual_features=release[f"features{purpose}"],
                        actual_probabilities=release[f"prob{purpose}"],
                        actual_decision=release[f"decision{purpose}"])
                else:
                    item = verify_d_raw_to_wire(
                        manifest_path=manifest_path, manifest_sha256=manifest_hash,
                        source_checkpoint=checkpoint,
                        source_sha256=source_receipt["source_sha256"],
                        x83=feature["X"], feature_names=feature_names,
                        feature_names_sha256=feature_hash, purpose=purpose,
                        actual_wire=_wire(release, manifest, purpose))
                purpose_report.append(item)
            by_role[role] = {"release_sha256": release_hash,
                             "raw_to_wire": purpose_report}
            if role == "INNER_SELECTION":
                tasks = {
                    "income": task_metrics(release["prob0"], endpoint["income"],
                                           decision=release["decision0"],
                                           majority_class=majority["income"]),
                    "occupation": task_metrics(release["prob1"], endpoint["occupation"],
                                               decision=release["decision1"],
                                               majority_class=majority["occupation"])}
        report["arms"][arm] = {"manifest_sha256": manifest_hash,
                                "fit_scaler": scaler, "roles": by_role,
                                "task": tasks,
                                "point_gate": task_point_gate(tasks, references)}
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("followup replay receipt must be new")
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"status": report["status"], "public_receipt_sha256": sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "seal-sha256", "amendment", "amendment-sha256",
                 "role-manifest", "role-manifest-sha256", "source-checkpoint",
                 "first-private", "first-receipt", "first-receipt-sha256",
                 "followup-private", "followup-receipt", "followup-receipt-sha256",
                 "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()

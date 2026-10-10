"""Queue-only first paired-seed T versus D-K fit and private release package.

This command refuses all role reads before checking the effective protocol
bytes and its own active R03 queue child PID. It never imports a mixed-role
loader, writes row arrays only below its owner-only private output directory,
and publishes only aggregate counts and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.reference_replay import replay_reference_role
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import outer_split
from .fit import (FitData, TrainConfig, calibrate_privacy_weight, fit_arm,
                  make_batch_plan)
from .model import FixedScaler, HardCodeRelease
from .service import save_service, sha256_file


SCHEMA = "latent-repair-first-paired-seed-receipt-v1"


def _json_once(path: Path, value: dict) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _npz_once(path: Path, arrays: dict[str, np.ndarray]) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _queue_guard(queue_state: Path, unit_id: str, seal: Path, seal_sha256: str) -> None:
    if sha_file(seal) != seal_sha256:
        raise PermissionError("effective seal byte hash mismatch")
    for _ in range(20):
        state = json.loads(queue_state.read_text())
        unit = state.get("units", {}).get(unit_id, {})
        active = (state.get("schema") == "pcrl-latent-sampled-queue-v1"
                  and not state.get("closed")
                  and state.get("seal", {}).get("sha256") == seal_sha256
                  and Path(state["seal"]["path"]).resolve() == seal.resolve()
                  and unit.get("status") == "running"
                  and unit.get("spec", {}).get("stage") == "d_first"
                  and unit.get("child_pid") == os.getpid())
        if active:
            return
        time.sleep(0.1)
    raise PermissionError("first D fit must be this active R03 d_first queue child")


def _common_eval(shared_dir: Path, expected_sha256: str):
    if sha_file(shared_dir / "common_eval.py") != expected_sha256:
        raise PermissionError("inherited calibration source byte hash mismatch")
    if str(shared_dir) not in sys.path:
        sys.path.insert(0, str(shared_dir))
    import common_eval
    if Path(common_eval.__file__).resolve() != (shared_dir / "common_eval.py").resolve():
        raise PermissionError("calibration module resolved outside inherited source")
    return common_eval


def _fit_data(store: RoleOnlyStore, encoder: FrozenUEncoder, seed: int
              ) -> tuple[FitData, FitData, np.ndarray, tuple[str, ...]]:
    role = "OSF_DEFENSE_FIT"
    feature = store.read(role, "features")
    income = store.read(role, "income", "task_fit")
    occupation = store.read(role, "occupation", "task_fit")
    reference = store.read(role, f"references_s{seed}")
    sex = store.read(role, "sex", "defense_fit")
    assert_same_people(feature, income, occupation, reference, sex)
    names = tuple(str(x) for x in feature["feature_names"])
    h0, h1 = encoder.encode_pair(feature["X"], names, role)
    args = (h0, h1, income["income"], occupation["occupation"],
            reference["Ucal_1"], reference["Ucal_2"])
    task = FitData(*args, None, feature["row_id"], role)
    protected = FitData(*args, sex["sex"], feature["row_id"], role)
    return task, protected, feature["unit"], names


def _occupied(code: np.ndarray) -> dict:
    ids = np.sum(code.astype(np.uint16) * (1 << np.arange(10)), axis=1)
    counts = np.bincount(ids, minlength=1024)
    occupied = counts[counts > 0]
    p = occupied / occupied.sum()
    return {"occupied_states": int(len(occupied)),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "rare_state_row_fraction": float(occupied[occupied < 5].sum() / len(code)),
            "capacity_states": 1024}


def run(args: argparse.Namespace) -> dict:
    seal = Path(args.seal).resolve(strict=True)
    queue_state = Path(args.queue_state).resolve(strict=True)
    _queue_guard(queue_state, args.unit_id, seal, args.seal_sha256)
    protocol = json.loads(seal.read_text())
    if (protocol.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1"
            or protocol.get("status") != "LOCALLY_SEALED_D_STAGE"
            or protocol.get("d_model", {}).get("states_per_recipient_max") != 1024
            or "H-GLOBAL-TEMP" not in protocol.get("d_model", {}).get("calibration", "")):
        raise PermissionError("R00 D protocol seal is not effective")
    sources = protocol.get("source_files_sha256", {})
    if (not isinstance(sources, dict) or
            "latent_repair_v1/design/PROPOSED_D_LOCK.md" not in sources):
        raise PermissionError("D method specification is not bound by effective seal")
    if (sources.get("source_role_manifest") != args.role_manifest_sha256
            or sources.get("inherited_calibration_code") != args.common_eval_sha256):
        raise PermissionError("role manifest or inherited CAL code not bound by effective seal")
    for relative, expected in sources.items():
        if relative in ("source_role_manifest", "inherited_calibration_code"):
            continue
        if sha_file(seal.parent / relative) != expected:
            raise PermissionError(f"effective seal source changed: {relative}")
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    if sha_file(manifest_path) != args.role_manifest_sha256:
        raise PermissionError("role-only repair manifest SHA mismatch")
    admitted = json.loads(manifest_path.read_text())
    source_role = protocol.get("role_identity_and_file_hashes", {})
    if (admitted.get("role_file_sha256") != source_role.get("role_file_sha256")
            or admitted.get("roles") != source_role.get("role_membership")
            or admitted.get("source_schema_sha256") != source_role.get("ordered_predictors_sha256")
            or source_role.get("source_model_sha256", {}).get(str(args.seed)) != args.source_sha256):
        raise PermissionError("effective seal does not bind role files, membership and U source")
    common = _common_eval(Path(args.shared_dir).resolve(strict=True), args.common_eval_sha256)
    model_pt = Path(args.model_pt).resolve(strict=True)
    if args.seed not in (0, 1) or sha_file(model_pt) != args.source_sha256:
        raise PermissionError("paired source checkpoint pin mismatch")
    out = Path(args.private_output).absolute()
    if (out.exists() or not out.parent.is_dir() or out.parent.name != ".private"
            or out.parent.parent.resolve() != seal.parent
            or out.name != f"d_first_s{args.seed}"):
        raise FileExistsError("first-seed private output must be a new directory")
    out.mkdir(mode=0o700)
    try:
        store = RoleOnlyStore(manifest_path)
        fit_feature = store.read("OSF_DEFENSE_FIT", "features")
        names = tuple(str(x) for x in fit_feature["feature_names"])
        encoder = FrozenUEncoder.load(model_pt, args.source_sha256, names,
                                      store.manifest["source_schema_sha256"], args.seed)
        _, reference_fit_receipt = replay_reference_role(
            store, encoder, model_pt.parents[2], "OSF_DEFENSE_FIT")
        task, protected, unit, names = _fit_data(store, encoder, args.seed)
        train, val, fold_receipt = outer_split(unit, task.y0, task.y1, args.seed)
        scaler = FixedScaler.fit(task.h0, task.h1)

        # All stages use identical aligned-person batches. The plan is built
        # once and passed into the no-SEX T trainer without its group labels.
        split_plan = make_batch_plan(train, seed=args.seed, epochs=20,
                                     batch_size=256, sex=protected.sex)
        warm_split = fit_arm(task, TrainConfig(), args.seed, mode="T", scaler=scaler,
                             train_idx=train, val_idx=val, batch_plan=split_plan)
        e0 = warm_split.selected_epoch
        t_split = fit_arm(task, TrainConfig(), args.seed, mode="T",
                          init_state=warm_split.state_dict, scaler=scaler,
                          train_idx=train, val_idx=val, batch_plan=split_plan)
        ec = t_split.selected_epoch
        lambda_receipt = calibrate_privacy_weight(protected, warm_split.state_dict,
                                                   scaler, args.seed, train, val)
        full_plan = make_batch_plan(np.arange(task.rows), seed=args.seed,
                                    epochs=max(e0, ec), batch_size=256,
                                    sex=protected.sex)
        warm_full = fit_arm(task, TrainConfig(epochs=e0), args.seed, mode="T",
                            scaler=scaler, full_fit=True, batch_plan=full_plan[:e0])
        t_full = fit_arm(task, TrainConfig(epochs=ec), args.seed, mode="T",
                         init_state=warm_full.state_dict, scaler=scaler,
                         full_fit=True, batch_plan=full_plan[:ec])
        k_full = fit_arm(protected, TrainConfig(epochs=ec,
                         privacy_weight=lambda_receipt["privacy_weight"]), args.seed,
                         mode="D-K", init_state=warm_full.state_dict,
                         scaler=scaler, full_fit=True, batch_plan=full_plan[:ec])
        fits = {"T": t_full, "D-K": k_full}
        cal_feature = store.read("CALIBRATION_HELDOUT", "features")
        cal_income = store.read("CALIBRATION_HELDOUT", "income", "calibration")
        cal_occupation = store.read("CALIBRATION_HELDOUT", "occupation", "calibration")
        assert_same_people(cal_feature, cal_income, cal_occupation)
        cal_latent = encoder.encode_pair(cal_feature["X"], names, "CALIBRATION_HELDOUT")
        releases: dict[str, HardCodeRelease] = {}
        packages = {}
        fit_receipt = {"source_seed": args.seed, "fold": fold_receipt,
                       "warm_selected_epoch": e0, "continuation_epoch": ec,
                       "warm_split_updates": warm_split.optimizer_updates,
                       "t_split_updates": t_split.optimizer_updates,
                       "warm_full_updates": warm_full.optimizer_updates,
                       "continuation_updates": t_full.optimizer_updates,
                       "gradient_ratio": lambda_receipt}
        for arm_id, fit in fits.items():
            raw = HardCodeRelease(fit.model, scaler, f"D-{arm_id}-s{args.seed}-v1",
                                  args.source_sha256)
            records = []
            for purpose, label, task_name in ((0, cal_income["income"], "income"),
                                              (1, cal_occupation["occupation"], "occupation")):
                code = raw.recipient(purpose, cal_latent[purpose])["code"]
                p = raw.uncalibrated_probabilities_from_code(purpose, code)
                records.append(common.fit_global_temperature(
                    p, label, task=task_name, arm_id=arm_id, model_seed=args.seed,
                    row_id=cal_feature["row_id"], unit=cal_feature["unit"]))
            release = HardCodeRelease(fit.model, scaler, raw.version,
                                      args.source_sha256, tuple(records))
            releases[arm_id] = release
            packages[arm_id] = save_service(out / arm_id, release,
                                            source_seed=args.seed, arm_id=arm_id,
                                            role_manifest_sha256=args.role_manifest_sha256,
                                            fit_receipt=fit_receipt)

        role_receipts = {}
        _, reference_inner_receipt = replay_reference_role(
            store, encoder, model_pt.parents[2], "INNER_SELECTION")
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            feature = store.read(role, "features")
            if tuple(str(x) for x in feature["feature_names"]) != names:
                raise ValueError("service role feature schema/order changed")
            h0, h1 = encoder.encode_pair(feature["X"], names, role)
            role_receipts[role] = {}
            if role == "INNER_SELECTION":
                income = store.read(role, "income", "screen")
                occupation = store.read(role, "occupation", "screen")
                reference = store.read(role, f"references_s{args.seed}")
                assert_same_people(feature, income, occupation, reference)
                endpoint = {
                    "row_id": feature["row_id"], "unit": feature["unit"],
                    "income": income["income"], "occupation": occupation["occupation"],
                    "U0_1": reference["U0_1"], "U0_2": reference["U0_2"],
                    "Ucal_1": reference["Ucal_1"], "Ucal_2": reference["Ucal_2"],
                }
                role_receipts[role]["endpoint_labels_references_sha256"] = _npz_once(
                    out / "endpoint__INNER_SELECTION.npz", endpoint)
            for arm_id, release in releases.items():
                r0 = release.recipient(0, h0)
                r1 = release.recipient(1, h1)
                arrays = {"row_id": feature["row_id"], "unit": feature["unit"],
                          "code0": r0["code"], "prob0": r0["probabilities"],
                          "decision0": r0["decision"],
                          "code1": r1["code"], "prob1": r1["probabilities"],
                          "decision1": r1["decision"]}
                filename = f"release__{arm_id}__{role}.npz"
                digest = _npz_once(out / filename, arrays)
                role_receipts[role][arm_id] = {
                    "rows": len(feature["row_id"]), "private_release_sha256": digest,
                    "recipient0": _occupied(r0["code"]),
                    "recipient1": _occupied(r1["code"]),
                }
        receipt = {"schema": SCHEMA, "status": "T_DK_RELEASES_READY",
                   "seed": args.seed, "unit_id": args.unit_id,
                   "seal_sha256": args.seal_sha256,
                   "role_manifest_sha256": args.role_manifest_sha256,
                   "source_sha256": args.source_sha256,
                   "common_eval_sha256": args.common_eval_sha256,
                   "reference_replay": {"OSF_DEFENSE_FIT": reference_fit_receipt,
                                        "INNER_SELECTION": reference_inner_receipt},
                   "fit": fit_receipt,
                   "packages": {arm: {k: v for k, v in package.items() if k != "manifest_path"}
                                for arm, package in packages.items()},
                   "roles": role_receipts,
                   "private_payload_scope": "two model packages and role-only row releases; no public row arrays"}
        _json_once(out / "private_index.json", {
            "schema": SCHEMA, "unit_id": args.unit_id,
            "package_paths": {arm: package["manifest_path"] for arm, package in packages.items()},
            "release_files": [str(out / f"release__{arm}__{role}.npz")
                              for role in role_receipts for arm in releases],
            "endpoint_file": str(out / "endpoint__INNER_SELECTION.npz"),
        })
        _json_once(Path(args.public_receipt), receipt)
        return receipt
    except BaseException:
        # Preserve incomplete private attempt for queue/custody audit.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("seal", "seal-sha256", "queue-state", "unit-id", "role-manifest",
                 "role-manifest-sha256", "shared-dir", "common-eval-sha256",
                 "model-pt", "source-sha256", "private-output", "public-receipt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "unit_id": result["unit_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Queued W_BANK compilation or one predeclared T0 hard-code fit.

The entry point fails closed before any private path is read unless its
inherited queue slot identifies the requested witness unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder
from results.pcrl_latent_and_sampled_methods_v1.r01.reference_replay import replay_reference_role
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array, sha_file,
)
from results.pcrl_utility_reference_repair_v1.r1.fit_splits import outer_split

from .bank import admit_policy_pair, route_scores
from .learner import FitData, TrainConfig, fit_task_only
from .model import HardCodeRelease
from .service import save_service


SCHEMA = "code-utility-witness-worker-v1"
ROLES_BANK = ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION")
ROLES_FIT = ("ATTACK_FIT_NEW", "INNER_SELECTION")
CLASS_ORDER = {"income": [0, 1], "occupation_group": [0, 1, 2, 3, 4, 5]}


def _write_once(path: Path, writer) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def _json_once(path: Path, obj: dict) -> str:
    payload = (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    return _write_once(path, lambda stream: stream.write(payload))


def _npz_once(path: Path, arrays: dict[str, np.ndarray]) -> str:
    return _write_once(path, lambda stream: np.savez_compressed(stream, **arrays))


def release_payload(row_id: np.ndarray, unit: np.ndarray,
                    income: dict, occupation: dict, *, bits: int) -> dict[str, np.ndarray]:
    ids, groups = np.asarray(row_id), np.asarray(unit)
    if (ids.dtype != np.int64 or groups.dtype != np.int64
            or ids.ndim != 1 or groups.shape != ids.shape
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("aligned row/group IDs required")
    out = {"row_id": ids.copy(), "unit": groups.copy()}
    for purpose, record in enumerate((income, occupation)):
        code = np.asarray(record["code"])
        probability = np.asarray(record["probabilities"])
        decision = np.asarray(record["decision"])
        classes = (2, 6)[purpose]
        if (code.dtype != np.uint8 or code.shape != (len(ids), bits)
                or not np.isin(code, (0, 1)).all()
                or probability.dtype != np.float64
                or probability.shape != (len(ids), classes)
                or not np.isfinite(probability).all() or np.any(probability < 0)
                or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-10)
                or decision.dtype != np.int64 or decision.shape != ids.shape
                or not np.array_equal(decision, probability.argmax(axis=1))):
            raise ValueError("complete hard-code recipient wire invalid")
        out[f"code{purpose}"] = code.copy()
        out[f"prob{purpose}"] = probability.copy()
        out[f"decision{purpose}"] = decision.copy()
    return out


def _array_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: sha_array(value) for name, value in arrays.items()}


def _occupied(code: np.ndarray) -> dict:
    _, count = np.unique(code, axis=0, return_counts=True)
    p = count / count.sum()
    return {"occupied_states": int(len(count)),
            "entropy_nats": float(-(p * np.log(p)).sum()),
            "singletons": int(np.count_nonzero(count == 1)),
            "singleton_person_fraction": float(count[count == 1].sum() / count.sum())}


def _source(store: RoleOnlyStore, source_root: Path, seed: int,
            source_sha256: str, fit_features: dict) -> tuple[FrozenUEncoder, tuple]:
    source = source_root / "admitted" / f"rel__s{seed}__U"
    complete = json.loads((source / "COMPLETE.json").read_text())
    if complete.get("id") != f"rel__s{seed}__U":
        raise ValueError("U source unit identity differs")
    for name in ("model.pt", "head_0.joblib", "head_1.joblib", "record.json"):
        if sha_file(source / name) != complete.get("files", {}).get(name):
            raise ValueError("U source component hash differs")
    if complete["files"]["model.pt"] != source_sha256:
        raise ValueError("U source checkpoint differs from sealed model")
    names = tuple(str(item) for item in fit_features["feature_names"])
    encoder = FrozenUEncoder.load(source / "model.pt", source_sha256,
                                  names, store.manifest["source_schema_sha256"], seed)
    heads = tuple(joblib.load(source / f"head_{i}.joblib") for i in (0, 1))
    if any(not np.array_equal(heads[i].classes_, np.arange((2, 6)[i]))
           for i in (0, 1)):
        raise ValueError("historical task class order differs")
    return encoder, heads


def _bank_tables(public_path: Path, expected_sha256: str, seed: int,
                 source_sha256: str, schema_sha256: str,
                 qpc_root: Path, hcal_root: Path) -> dict:
    from hcal.deploy import load_decoder

    if sha_file(public_path) != expected_sha256:
        raise ValueError("public table byte pin differs")
    records = json.loads(public_path.read_text())
    if (records.get("schema") != "code-utility-public-bank-tables-v1"
            or records.get("seed") != seed or records.get("bit_count") != 10
            or records.get("source_model_sha256") != source_sha256
            or records.get("feature_schema_sha256") != schema_sha256):
        raise ValueError("public bank table/source identity differs")
    out = {}
    for name, source_name in (("W_Q", "DIRECT-TASK"), ("W_TSTAR", "FINE-TASK")):
        item = records["families"][name]
        policy_path = qpc_root / "run" / "units" / f"pol__s{seed}__U_{source_name}_i8o64" / "policy.json"
        pair, admission = admit_policy_pair(policy_path,
                                             expected_sha256=item["policy_sha256"],
                                             source_sha256=source_sha256,
                                             schema_sha256=schema_sha256)
        if admission["policy_fingerprint"] != item["policy_fingerprint"]:
            raise ValueError("public bank policy fingerprint differs")
        if name == "W_TSTAR":
            decoder_path = (hcal_root / "package" /
                            f"decoder__s{seed}__U_FINE-TASK_i8o64_H-GLOBAL-TEMP.json")
            family, tables, embedded, _ = load_decoder(
                decoder_path, pair, item["decoder_sha256"])
            if family != "H-GLOBAL-TEMP" or embedded != item["decoder_sha256"]:
                raise ValueError("T* historical decoder differs")
        else:
            if item["decoder_sha256"] is not None:
                raise ValueError("Q bank unexpectedly adds decoder")
            tables = (pair.p1.token_proto, pair.p2.token_proto)
        for purpose in (0, 1):
            if (not np.array_equal(tables[purpose],
                                   np.asarray(item[f"table{purpose}"], dtype=np.float64))
                    or not np.array_equal((pair.p1, pair.p2)[purpose].token_class,
                                          np.asarray(item[f"token_class{purpose}"],
                                                     dtype=np.int64))):
                raise ValueError("public bank table/class differs from source")
        out[name] = (pair, tables, item)
    return out


def _bank(args, store: RoleOnlyStore, private: Path, queue: dict) -> dict:
    fit_feature = store.read("OSF_DEFENSE_FIT", "features")
    encoder, heads = _source(store, Path(args.source_root), args.seed,
                             args.source_sha256, fit_feature)
    banks = _bank_tables(Path(args.public_table), args.public_table_sha256,
                         args.seed, args.source_sha256,
                         store.manifest["source_schema_sha256"],
                         Path(args.qpc_root), Path(args.source_root))
    roles = {}
    for role in ROLES_BANK:
        assert_active_from_env(args.unit_id, "witness")
        feature = fit_feature if role == "OSF_DEFENSE_FIT" else store.read(role, "features")
        names = tuple(str(item) for item in feature["feature_names"])
        _, reference_parity = replay_reference_role(store, encoder, args.source_root, role)
        latent = encoder.encode_pair(feature["X"], names, role)
        scores = tuple(np.asarray(heads[i].predict_proba(latent[i].astype(np.float64)),
                                  dtype=np.float64) for i in (0, 1))
        role_result = {"rows": int(len(feature["row_id"])),
                       "groups": int(len(np.unique(feature["unit"]))),
                       "row_unit_pair_sha256": sha_array(np.column_stack(
                           (feature["row_id"], feature["unit"]))),
                       "source_reference_parity": reference_parity["status"],
                       "arms": {}}
        for arm_id, (pair, tables, item) in banks.items():
            recipient = [route_scores(pair, i, scores[i], 10,
                                      tables[i] if arm_id == "W_TSTAR" else None)
                         for i in (0, 1)]
            arrays = release_payload(feature["row_id"], feature["unit"],
                                     recipient[0], recipient[1], bits=10)
            filename = f"release__{arm_id}__s{args.seed}__{role}.npz"
            digest = _npz_once(private / filename, arrays)
            role_result["arms"][arm_id] = {
                "private_release_basename": filename,
                "private_release_sha256": digest,
                "array_sha256": _array_hashes(arrays),
                "recipient0": _occupied(arrays["code0"]),
                "recipient1": _occupied(arrays["code1"]),
                "public_table_sha256": args.public_table_sha256,
                "policy_sha256": item["policy_sha256"],
                "decoder_sha256": item["decoder_sha256"],
                "version": f"{arm_id}_s{args.seed}_compiled10_v1",
            }
        roles[role] = role_result
    return {"arm_ids": ["W_Q", "W_TSTAR"], "bit_count": 10,
            "fit_status": "HISTORICAL_FROZEN_BANK_NO_NEW_FIT",
            "current_role_utility_status": "RELEASES_READY_FOR_R2",
            "roles": roles}


def _fit(args, store: RoleOnlyStore, private: Path, queue: dict) -> dict:
    if args.bits not in (10, 16):
        raise ValueError("only predeclared capacity ladder allowed")
    shared_dir = (Path(__file__).resolve().parents[1] /
                  "results/pcrl_method_portfolio_v1/shared")
    if sha_file(shared_dir / "common_eval.py") != args.common_eval_sha256:
        raise ValueError("inherited calibration implementation differs")
    if str(shared_dir) not in sys.path:
        sys.path.insert(0, str(shared_dir))
    import common_eval
    if Path(common_eval.__file__).resolve() != (shared_dir / "common_eval.py").resolve():
        raise ValueError("calibration module resolved outside pinned source")
    feature = store.read("OSF_DEFENSE_FIT", "features")
    income = store.read("OSF_DEFENSE_FIT", "income", "task_fit")
    occupation = store.read("OSF_DEFENSE_FIT", "occupation", "task_fit")
    reference = store.read("OSF_DEFENSE_FIT", f"references_s{args.seed}")
    assert_same_people(feature, income, occupation, reference)
    encoder, _ = _source(store, Path(args.source_root), args.seed,
                         args.source_sha256, feature)
    _, fit_reference_parity = replay_reference_role(
        store, encoder, args.source_root, "OSF_DEFENSE_FIT")
    names = tuple(str(item) for item in feature["feature_names"])
    h0, h1 = encoder.encode_pair(feature["X"], names, "OSF_DEFENSE_FIT")
    data = FitData(h0, h1, income["income"], occupation["occupation"],
                   reference["Ucal_1"].astype(np.float32),
                   reference["Ucal_2"].astype(np.float32),
                   None, feature["row_id"], "OSF_DEFENSE_FIT")
    train_idx, val_idx, fold_receipt = outer_split(feature["unit"],
                                                    income["income"],
                                                    occupation["occupation"],
                                                    args.seed)
    assert_active_from_env(args.unit_id, "witness")
    config = TrainConfig()
    result = fit_task_only(data, config, seed=args.seed, bits=args.bits,
                           train_idx=train_idx, val_idx=val_idx)
    arm_id = f"T0_B{args.bits}"
    cal_feature = store.read("CALIBRATION_HELDOUT", "features")
    cal_income = store.read("CALIBRATION_HELDOUT", "income", "calibration")
    cal_occupation = store.read("CALIBRATION_HELDOUT", "occupation", "calibration")
    assert_same_people(cal_feature, cal_income, cal_occupation)
    cal_names = tuple(str(item) for item in cal_feature["feature_names"])
    cal_latent = encoder.encode_pair(cal_feature["X"], cal_names, "CALIBRATION_HELDOUT")
    raw = HardCodeRelease(result.model, result.scaler,
                          f"{arm_id}_s{args.seed}_v1", args.source_sha256)
    calibrators = []
    for purpose, task in enumerate(("income", "occupation")):
        code = raw.recipient(purpose, cal_latent[purpose])["code"]
        probability = raw.uncalibrated_probabilities_from_code(purpose, code)
        record = common_eval.fit_global_temperature(
            probability, (cal_income["income"], cal_occupation["occupation"])[purpose],
            task=task, arm_id=arm_id, model_seed=args.seed,
            row_id=cal_feature["row_id"], unit=cal_feature["unit"])
        calibrators.append(record)
    release = HardCodeRelease(result.model, result.scaler,
                              f"{arm_id}_s{args.seed}_v1", args.source_sha256,
                              tuple(calibrators))
    fit_receipt = {
        "fold": fold_receipt, "config": vars(config),
        "selected_epoch": result.selected_epoch,
        "fold_updates": result.fold_updates,
        "full_updates": result.full_updates,
        "fold_history": result.fold_history,
        "full_history": result.full_history,
        "source_reference_parity": fit_reference_parity["status"],
        "source_encoder_internal_validation_exposure":
            "frozen U encoder historically trained on all FIT; new head internal fold is not source-unseen",
    }
    package = save_service(private / f"service__{arm_id}__s{args.seed}", release,
                           source_seed=args.seed, arm_id=arm_id,
                           role_manifest_sha256=args.role_manifest_sha256,
                           source_schema_sha256=store.manifest["source_schema_sha256"],
                           fit_receipt=fit_receipt)
    role_result = {}
    for role in ROLES_FIT:
        assert_active_from_env(args.unit_id, "witness")
        role_feature = store.read(role, "features")
        role_names = tuple(str(item) for item in role_feature["feature_names"])
        latent = encoder.encode_pair(role_feature["X"], role_names, role)
        one = release.recipient(0, latent[0])
        two = release.recipient(1, latent[1])
        arrays = release_payload(role_feature["row_id"], role_feature["unit"],
                                 one, two, bits=args.bits)
        filename = f"release__{arm_id}__s{args.seed}__{role}.npz"
        digest = _npz_once(private / filename, arrays)
        role_result[role] = {
            "rows": int(len(role_feature["row_id"])),
            "groups": int(len(np.unique(role_feature["unit"]))),
            "row_unit_pair_sha256": sha_array(np.column_stack(
                (role_feature["row_id"], role_feature["unit"]))),
            "private_release_basename": filename,
            "private_release_sha256": digest,
            "array_sha256": _array_hashes(arrays),
            "recipient0": _occupied(arrays["code0"]),
            "recipient1": _occupied(arrays["code1"]),
            "version": one["metadata"]["version"],
            "metadata": [one["metadata"], two["metadata"]],
        }
    return {"arm_id": arm_id, "bit_count": args.bits,
            "fit": fit_receipt,
            "calibration": [{"task": item["task"], "alpha": item["alpha"],
                             "boundary": item["boundary"],
                             "fit_rows": item["fit_rows"],
                             "fit_row_unit_pair_sha256": item["fit_row_unit_pair_sha256"],
                             "input_probability_sha256": item["input_probability_sha256"]}
                            for item in calibrators],
            "package": {key: value for key, value in package.items()
                        if key != "manifest_path"},
            "roles": role_result,
            "current_role_utility_status": "RELEASES_READY_FOR_R2"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m code_witness.worker",
                                     allow_abbrev=False)
    parser.add_argument("--stage", choices=("bank", "fit"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--bits", type=int, choices=(10, 16))
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--role-manifest-sha256", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--qpc-root")
    parser.add_argument("--public-table")
    parser.add_argument("--public-table-sha256")
    parser.add_argument("--common-eval-sha256")
    parser.add_argument("--private-out", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    queue = assert_active_from_env(args.unit_id, "witness")
    if args.stage == "bank" and (args.qpc_root is None or args.public_table is None
                                  or args.public_table_sha256 is None):
        raise ValueError("bank unit requires historical policy root and table pin")
    if args.stage == "fit" and (args.bits is None or args.common_eval_sha256 is None):
        raise ValueError("fit unit requires bit capacity and calibration code pin")
    if sha_file(Path(args.role_manifest)) != args.role_manifest_sha256:
        raise ValueError("role admission manifest byte pin differs")
    store = RoleOnlyStore(args.role_manifest)
    private = Path(args.private_out)
    private.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(mode=0o700)
    result = (_bank(args, store, private, queue) if args.stage == "bank"
              else _fit(args, store, private, queue))
    private_index = {"schema": SCHEMA, "stage": args.stage,
                     "unit_id": args.unit_id, "seed": args.seed,
                     "private_payloads": sorted(path.name for path in private.iterdir())}
    _json_once(private / "private_index.json", private_index)
    receipt = {
        "schema": SCHEMA, "status": "COMPLETE", "stage": args.stage,
        "seed": args.seed, "unit_id": args.unit_id,
        "queue_seal_sha256": queue["seal_sha256"],
        "queue_spec_sha256": queue["spec_sha256"],
        "role_manifest_sha256": args.role_manifest_sha256,
        "source_model_sha256": args.source_sha256,
        "source_feature_schema_sha256": store.manifest["source_schema_sha256"],
        "class_order": CLASS_ORDER,
        "private_index_sha256": sha_file(private / "private_index.json"),
        "worker_sha256": sha_file(Path(__file__)),
        "result": result,
        "privacy_scope": "aggregate metadata and hashes only; private releases and model not published",
    }
    _json_once(Path(args.receipt), receipt)
    print(json.dumps({"unit_id": args.unit_id, "status": "COMPLETE",
                      "receipt_sha256": sha_file(Path(args.receipt))}))


if __name__ == "__main__":
    main()

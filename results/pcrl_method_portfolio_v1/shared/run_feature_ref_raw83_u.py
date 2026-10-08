"""Queued fixed unprotected feature-track anchor on admitted Adult roles.

Fits only fresh SEX readers on ATTACK_FIT_NEW. It makes no defense fit or
candidate selection and never opens the old assessment. Every person-level
release, reader model and endpoint input remains under private write-once
custody for independent S4 replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from attack_eval import add_ignore_recipient_aliases, fit_slate, save_fitted_slate
from attack_controls import run_control_probes
from common_eval import assert_same_people
from compute_queue import ComputeQueue
from reference_views import ARM_ID, feature_ref_raw83_u
from safe_data_guard import SafeRoleStore, sha_file


CAMPAIGN = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent / ".private"
READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
           "RAW_LR_C1", "RAW_HGB_15x200")


def _write_json_once(path: Path, obj: dict) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def _write_npz_once(path: Path, arrays: dict) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def _seal_and_source(seed: int, store: SafeRoleStore, lock_sha: str,
                     cache_root: Path) -> str:
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != lock_sha:
        raise PermissionError("campaign lock SHA mismatch")
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or \
            lock.get("adult_fit_permission") is not True:
        raise PermissionError("effective Adult science seal absent")
    unit_id = f"S2-FEATURE-REF-RAW83-U-S{seed}"
    if unit_id not in lock.get("admitted_shared_unit_ids", []):
        raise PermissionError("feature anchor unit is not admitted by science lock")
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED" or \
            lock.get("data_admission", {}).get("manifest_sha256") != sha_file(store.manifest_path):
        raise PermissionError("admitted role manifest differs from science lock")
    attack_sha = sha_file(Path(__file__).with_name("attack_eval.py"))
    controls_sha = sha_file(Path(__file__).with_name("attack_controls.py"))
    if lock.get("attack_protocol", {}).get("source_module_sha256") != attack_sha or \
            lock.get("shared_attack_controls_sha256") != controls_sha:
        raise PermissionError("shared attacker or real-control code differs from science lock")
    seal_path = CAMPAIGN / "LOCAL_SEAL_RECEIPT.json"
    if not ComputeQueue._local_seal_ok({"local_seal_path": str(seal_path),
                                        "local_seal_sha256": sha_file(seal_path)}):
        raise PermissionError("committed local science seal absent")
    manifest_path = CAMPAIGN.parent / "pcrl_heldout_calibration_v1" / "MODEL_MANIFEST.json"
    source_lock = lock.get("source_model_admission", {})
    if source_lock.get("hcal_model_manifest_sha256") != sha_file(manifest_path):
        raise PermissionError("hcal source model manifest is not bound by science lock")
    manifest = json.loads(manifest_path.read_text())
    u0 = next(p for p in manifest["packages"] if p.get("release") == "SRC|U")
    source_sha = u0["model_pt_sha256"][str(seed)]
    if source_lock.get("U_model_pt_sha256_by_seed", {}).get(f"s{seed}") != source_sha:
        raise PermissionError("U source checkpoint identity is not bound by science lock")
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise PermissionError("U source checkpoint SHA unrecognized")
    model_path = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U" / "model.pt"
    if not model_path.is_file() or sha_file(model_path) != source_sha:
        raise PermissionError("U source checkpoint bytes differ from pinned identity")
    return source_sha


def run(seed: int, lock_sha: str, cache_root: Path) -> dict:
    store = SafeRoleStore(CAMPAIGN / "DATA_ACCESS_MANIFEST.json")
    source_sha = _seal_and_source(seed, store, lock_sha, cache_root)
    output = ROOT / f"feature_ref_raw83_u_s{seed}_v1"
    if output.exists():
        raise FileExistsError("feature reference anchor unit is write-once")
    output.mkdir(parents=True, mode=0o700)
    views = {}
    role_inputs = {}
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        features = store.read(role, "features")
        references = store.read(role, f"references_s{seed}")
        assert_same_people(features, references)
        views[role] = feature_ref_raw83_u(features, references,
                                           source_model_sha256=source_sha)
        role_inputs[role] = (features, references)
    sex_att = store.read("ATTACK_FIT_NEW", "sex", "attack")
    assert_same_people(role_inputs["ATTACK_FIT_NEW"][0], sex_att)
    sex_inner = store.read("INNER_SELECTION", "sex", "screen")
    assert_same_people(role_inputs["INNER_SELECTION"][0], sex_inner)
    probe_receipts = {}
    for view in ("r1", "r2", "pair"):
        probe = run_control_probes(views["ATTACK_FIT_NEW"][view], sex_att,
                                   views["INNER_SELECTION"][view], sex_inner)
        probe_receipts[view] = _write_json_once(output / f"PROBE_{view}.json", probe)
        if probe["status"] != "PASS":
            raise RuntimeError("feature anchor real engineering probe failed")
    fitted = {}
    receipts = {}
    for view in ("r1", "r2", "pair"):
        fitted[view] = fit_slate(
            views["ATTACK_FIT_NEW"][view], sex_att,
            views["INNER_SELECTION"][view], required_readers=READERS,
            attacker_seed=seed, progress_dir=output / f"progress_{view}")
    fitted["pair"] = add_ignore_recipient_aliases(fitted["pair"],
                                                   fitted["r1"], fitted["r2"])
    for view in ("r1", "r2", "pair"):
        receipts[view] = save_fitted_slate(
            fitted[view], output / f"readers_{view}",
            att_view=views["ATTACK_FIT_NEW"][view],
            inner_view=views["INNER_SELECTION"][view])
    features, refs = role_inputs["INNER_SELECTION"]
    labels = {task: store.read("INNER_SELECTION", task, "screen")
              for task in ("income", "occupation")}
    assert_same_people(features, refs, sex_inner, *labels.values())
    arrays = {"row_id": np.asarray(features["row_id"], dtype=np.int64),
              "unit": np.asarray(features["unit"], dtype=np.int64),
              "income": np.asarray(labels["income"]["income"], dtype=np.int64),
              "occupation": np.asarray(labels["occupation"]["occupation"], dtype=np.int64),
              "sex": np.asarray(sex_inner["sex"], dtype=np.int64)}
    for recipient in (1, 2):
        arrays[f"prob_{recipient}"] = np.asarray(refs[f"Ucal_{recipient}"], dtype=np.float64)
        for ref in ("U0", "Ucal"):
            arrays[f"{ref}_{recipient}"] = np.asarray(refs[f"{ref}_{recipient}"], dtype=np.float64)
    reader_keys = {}
    for view in ("r1", "r2", "pair"):
        reader_keys[view] = {}
        for index, (reader, pred) in enumerate(fitted[view]["predictions_inner"].items()):
            key = f"reader__{view}__{index}"
            arrays[key] = np.asarray(pred, dtype=np.float64)
            reader_keys[view][reader] = key
    npz_sha = _write_npz_once(output / "ENDPOINT_INPUTS.npz", arrays)
    endpoint_sha = _write_json_once(output / "ENDPOINT_INPUTS.json", {
        "schema": "pcrl-feature-ref-raw83-u-endpoint-input-v1",
        "arm": ARM_ID, "seed": seed, "role": "INNER_SELECTION",
        "npz_sha256": npz_sha, "reader_keys": reader_keys,
        "contains_private_person_arrays": True})
    receipt = {"schema": "pcrl-feature-ref-raw83-u-anchor-receipt-v1",
               "status": "EXPLORATORY_DEVELOPMENT", "arm": ARM_ID, "seed": seed,
               "source_model_sha256": source_sha,
               "campaign_lock_sha256": lock_sha,
               "data_access_manifest_sha256": sha_file(store.manifest_path),
               "builder_sha256": sha_file(Path(__file__).with_name("reference_views.py")),
               "attack_eval_sha256": sha_file(Path(__file__).with_name("attack_eval.py")),
               "attack_controls_sha256": sha_file(Path(__file__).with_name("attack_controls.py")),
               "runner_sha256": sha_file(Path(__file__)),
               "role_file_sha256": {f"{role}:{field}": store.manifest["real_file_specs"][f"{role}:{field}"]["sha256"]
                                    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION")
                                    for field in (("features", f"references_s{seed}", "sex")
                                                  if role == "ATTACK_FIT_NEW" else
                                                  ("features", f"references_s{seed}", "sex", "income", "occupation"))},
               "reader_receipt_sha256": {view: receipts[view]["private_receipt_sha256"]
                                         for view in ("r1", "r2", "pair")},
               "real_probe_receipt_sha256": probe_receipts,
               "endpoint_input_receipt_sha256": endpoint_sha,
               "reader_fits": 18, "real_probe_fits": 6,
               "null_probe_scorings": 3, "ignore_recipient_alias_fits": 0}
    receipt_sha = _write_json_once(output / "ANCHOR_RECEIPT.json", receipt)
    return {"arm": ARM_ID, "seed": seed, "reader_fits": 18,
            "real_probe_fits": 6,
            "private_receipt_sha256": receipt_sha}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--campaign-lock-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.seed, args.campaign_lock_sha256, args.cache_root)
    except BaseException as exc:
        print("FEATURE_REF_ANCHOR_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))

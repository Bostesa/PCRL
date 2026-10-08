"""Role-only admission of the inherited hcal FINE-TASK score bank for Team A.

Run with the pinned .venv interpreter only after S1's effective Adult science
seal and under S3's queue. This reads S4-admitted role-specific U0 references
and public/registered fine, policy and decoder JSON files; it never opens
all-role assign.npz, teacher.npz, release.npz or assessment person arrays.
It fits no model. Output tokens and banks are private, hash-bound inputs to
Team A's pure system-Python channel fitter and one-recipient deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from hcal import deploy as HD
from qpc import deploy as QD
from qpc import kmeans as KM
from qpc import partition as QP
from qpc import release as QR

from safe_data_guard import SafeRoleStore, sha_arr, sha_file
from compute_queue import Queue


CAMPAIGN = Path(__file__).resolve().parent.parent
SHARED = Path(__file__).resolve().parent
ROLES = ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION")
SCHEMA = "pcrl-portfolio-a-score-bank-admission-v1"
DATA_ACCESS = "A_ROLE_ONLY_REFERENCES_PREPROCESS"
RELEASES = {
    "tstar": {
        "release": "U|FINE-TASK|i8o64|H-GLOBAL-TEMP",
        "policy_stem": "U_FINE-TASK_i8o64",
        "decoder_stem": "U_FINE-TASK_i8o64_H-GLOBAL-TEMP",
        "family": "H-GLOBAL-TEMP",
        "decoder_file_sha256": {
            0: "83b88643ba3b81207558bd2936b15a0ea9a59ea3346dcf34b756a325cfcb6d9d",
            1: "0d534c671ff3a62e2e08dd5f04eb6704627fb784c9212e58bdaec864f9f224ae",
            2: "0994eb1adcde11c122e6c719abcdc41f35685b8a0bec55b848c399a411285ab5",
        },
    },
    "pstar": {
        "release": "U|JOINT|i8o64|l0.1|H-CLASS-TEMP",
        "policy_stem": "U_JOINT_i8o64_l0.1",
        "decoder_stem": "U_JOINT_i8o64_l0.1_H-CLASS-TEMP",
        "family": "H-CLASS-TEMP",
        "decoder_file_sha256": {
            0: "a1a5f7fb45a9f61f89f0809d86ef072461cf40efde5c5660a1c6a4b642de2841",
            1: "89d2c1e1df8bb0f4c52cb32e336792ecd9fbaa2574d2add7b8123c117f5cc4b8",
            2: "96788c5665ba0673fc9529d5891400b2eca43d4e20c7e3145611bc0944d2b2f0",
        },
    },
    "lra_seq21": {
        "release": "U|SEQ-21|i8o64|l0.1",
        "policy_stem": "U_SEQ-21_i8o64_l0.1",
        "family": "D0",
    },
    "hcal_local_d0": {
        "release": "U|LOCAL|i8o64|l0.1",
        "policy_stem": "U_LOCAL_i8o64_l0.1",
        "family": "D0",
    },
    "hcal_seq12_d0": {
        "release": "U|SEQ-12|i8o64|l0.1",
        "policy_stem": "U_SEQ-12_i8o64_l0.1",
        "family": "D0",
    },
}


def _arg(argv: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise PermissionError("queued A bank argument missing or duplicated")
    return str(argv[positions[0] + 1])


def _running_preprocess_unit(cache_root: Path, seed: int, release_key: str,
                             lock_sha: str, queue_root: Path) -> dict:
    unit_id = f"S2-A-SCORE-BANK-{release_key.upper().replace('_', '-')}-S{seed}"
    output = SHARED / ".private" / f"{release_key}_score_bank_s{seed}_v1" / "BANK_ADMISSION_RECEIPT.json"
    queue = Queue(queue_root)
    deadline = time.monotonic() + 5.0
    while True:
        state = queue.status()
        unit = state.get("units", {}).get(unit_id)
        if not unit or unit.get("status") != "running":
            raise PermissionError("A score-bank preprocessing lacks running S3 unit")
        spec = unit.get("spec", {})
        argv = spec.get("argv", [])
        if (len(argv) < 2 or Path(argv[1]).resolve() != Path(__file__).resolve() or
                spec.get("unit_id") != unit_id or spec.get("mode") != "real_data" or
                spec.get("family") != "SHARED" or spec.get("team") != "SHARED" or
                spec.get("stage") != "a_score_bank_preprocess" or
                spec.get("data_access") != DATA_ACCESS or
                spec.get("seed") != seed or spec.get("release_key") != release_key or
                spec.get("runner_sha256") != sha_file(Path(__file__)) or
                spec.get("outputs") != [str(output)] or
                _arg(argv, "--seed") != str(seed) or
                _arg(argv, "--release") != release_key or
                _arg(argv, "--cache-root") != str(cache_root) or
                _arg(argv, "--campaign-lock-sha256") != lock_sha or
                _arg(argv, "--queue-root") != str(queue_root)):
            raise PermissionError("running A score-bank preprocessing identity differs")
        child_pgid = unit.get("child_pgid")
        if child_pgid is not None:
            if child_pgid != os.getpgrp():
                raise PermissionError("A score-bank process group differs from S3")
            return state
        if time.monotonic() >= deadline:
            raise PermissionError("S3 score-bank child process group unavailable")
        time.sleep(0.02)


def _source_files(cache_root: Path, seed: int, model_manifest: dict,
                  source_admission: dict, config: dict) -> dict:
    admitted = cache_root / "hcal_v1" / "admitted" / "units"
    fine = admitted / f"fine__s{seed}" / "fine.json"
    policy = admitted / f"pol__s{seed}__{config['policy_stem']}" / "policy.json"
    decoder = cache_root / "hcal_v1" / "package" / f"decoder__s{seed}__{config['decoder_stem']}.json"
    teacher = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U" / "model.pt"
    pair = next(x for x in model_manifest["packages"] if x.get("release") == config["release"])
    psha = pair["seeds"][str(seed)]["policy_json_sha256"]
    dsha = pair["seeds"][str(seed)]["decoder_sha256"]
    fsha = source_admission["units"][f"fine__s{seed}"]["files_sha256"]["fine.json"]
    for path, expected in ((fine, fsha), (policy, psha),
                           (decoder, config["decoder_file_sha256"][seed])):
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError("registered A source JSON absent or hash mismatch")
    if not teacher.is_file() or sha_file(teacher) != pair["teacher_model_sha256"][str(seed)]:
        raise ValueError("registered A teacher checkpoint absent or hash mismatch")
    return {"fine": fine, "policy": policy, "decoder": decoder,
            "fine_sha256": fsha, "policy_sha256": psha, "decoder_sha256": dsha,
            "decoder_file_sha256": config["decoder_file_sha256"][seed],
            "teacher_model_sha256": pair["teacher_model_sha256"][str(seed)]}


def _lra_source_files(cache_root: Path, seed: int, hcal_manifest: dict,
                      source_admission: dict, config: dict) -> dict:
    lra_manifest = json.loads((CAMPAIGN.parent /
        "pcrl_adult_learned_decoder_release_v1" / "MODEL_MANIFEST.json").read_text())
    lra_pair = next(x for x in lra_manifest["packages"]
                    if x.get("config") == config["release"])
    fine = (cache_root / "hcal_v1" / "admitted" / "units" /
            f"fine__s{seed}" / "fine.json")
    policy = (cache_root / "lra_v1" / "run" / "units" /
              f"pol__s{seed}__{config['policy_stem']}" / "policy.json")
    teacher = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U" / "model.pt"
    fsha = source_admission["units"][f"fine__s{seed}"]["files_sha256"]["fine.json"]
    psha = lra_pair["seeds"][str(seed)]["policy_json_sha256"]
    u0 = next(x for x in hcal_manifest["packages"] if x.get("release") == "SRC|U")
    msha = u0["model_pt_sha256"][str(seed)]
    for path, expected in ((fine, fsha), (policy, psha), (teacher, msha)):
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError("registered lra D0 source absent or hash mismatch")
    if lra_pair["seeds"][str(seed)]["decoder_json_sha256"] is not None:
        raise ValueError("lra D0 historical control unexpectedly has a decoder")
    return {"fine": fine, "policy": policy, "fine_sha256": fsha,
            "policy_sha256": psha, "decoder_sha256": None,
            "decoder_file_sha256": None, "teacher_model_sha256": msha}


def _hcal_d0_source_files(cache_root: Path, seed: int, hcal_manifest: dict,
                          source_admission: dict, config: dict) -> dict:
    unit_id = f"pol__s{seed}__{config['policy_stem']}"
    fine = (cache_root / "hcal_v1" / "admitted" / "units" /
            f"fine__s{seed}" / "fine.json")
    policy = (cache_root / "hcal_v1" / "admitted" / "units" /
              unit_id / "policy.json")
    teacher = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U" / "model.pt"
    fsha = source_admission["units"][f"fine__s{seed}"]["files_sha256"]["fine.json"]
    psha = source_admission["units"][unit_id]["files_sha256"]["policy.json"]
    u0 = next(x for x in hcal_manifest["packages"] if x.get("release") == "SRC|U")
    msha = u0["model_pt_sha256"][str(seed)]
    for path, expected in ((fine, fsha), (policy, psha), (teacher, msha)):
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError("registered hcal D0 source absent or hash mismatch")
    return {"fine": fine, "policy": policy, "fine_sha256": fsha,
            "policy_sha256": psha, "decoder_sha256": None,
            "decoder_file_sha256": None, "teacher_model_sha256": msha}


def admit_one_seed(cache_root: Path, seed: int, release_key: str,
                   lock_sha: str, queue_root: Path) -> dict:
    config = RELEASES[release_key]
    queue_state = _running_preprocess_unit(cache_root, seed, release_key,
                                           lock_sha, queue_root)
    lock_path = CAMPAIGN / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != lock_sha:
        raise RuntimeError("A score-bank science lock SHA differs")
    lock = json.loads((CAMPAIGN / "CAMPAIGN_LOCK.json").read_text())
    if lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or lock.get("adult_fit_permission") is not True:
        raise RuntimeError("S1 effective Adult science seal absent")
    unit_id = f"S2-A-SCORE-BANK-{release_key.upper().replace('_', '-')}-S{seed}"
    if (lock.get("families", {}).get("A", {}).get("preprocessing_admission") != "ALLOWED" or
            lock.get("families", {}).get("A", {}).get("fit_admission") != "DENIED" or
            unit_id not in lock.get("admitted_a_score_bank_unit_ids", []) or
            lock.get("a_score_bank_runner_sha256") != sha_file(Path(__file__))):
        raise RuntimeError("Team A preprocessing-only source unit is not admitted")
    if not Queue._local_seal_ok(queue_state.get("gate", {})):
        raise RuntimeError("A preprocessing committed local science seal absent")
    store = SafeRoleStore(CAMPAIGN / "DATA_ACCESS_MANIFEST.json")
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise RuntimeError("S4 role-only admission absent")
    if lock.get("data_admission", {}).get("manifest_sha256") != sha_file(store.manifest_path):
        raise RuntimeError("role manifest differs from A preprocessing seal")
    output_root = SHARED / ".private" / f"{release_key}_score_bank_s{seed}_v1"
    if output_root.exists():
        raise FileExistsError("A score-bank admission is write-once")
    model_manifest_path = CAMPAIGN.parent / "pcrl_heldout_calibration_v1" / "MODEL_MANIFEST.json"
    source_admission_path = CAMPAIGN.parent / "pcrl_heldout_calibration_v1" / "SOURCE_ADMISSION.json"
    lra_manifest_path = CAMPAIGN.parent / "pcrl_adult_learned_decoder_release_v1" / "MODEL_MANIFEST.json"
    source_lock = lock.get("source_model_admission", {})
    if source_lock.get("hcal_model_manifest_sha256") != sha_file(model_manifest_path):
        raise RuntimeError("hcal source model manifest not bound by science lock")
    if (source_lock.get("hcal_source_admission_sha256") != sha_file(source_admission_path) or
            source_lock.get("lra_model_manifest_sha256") != sha_file(lra_manifest_path)):
        raise RuntimeError("A fine/policy source manifests not bound by science lock")
    model_manifest = json.loads(model_manifest_path.read_text())
    u0 = next(x for x in model_manifest["packages"] if x.get("release") == "SRC|U")
    if source_lock.get("U_model_pt_sha256_by_seed", {}).get(f"s{seed}") != \
            u0["model_pt_sha256"][str(seed)]:
        raise RuntimeError("U checkpoint not bound by science lock")
    source_admission = json.loads(source_admission_path.read_text())
    if release_key == "lra_seq21":
        paths = _lra_source_files(cache_root, seed, model_manifest, source_admission, config)
    elif config["family"] == "D0":
        paths = _hcal_d0_source_files(cache_root, seed, model_manifest, source_admission, config)
    else:
        paths = _source_files(cache_root, seed, model_manifest, source_admission, config)
    fine1, fine2 = QP.load_fine(json.loads(paths["fine"].read_text()))
    pair = QD.load_pair(paths["policy"])
    if pair.p1.fine.fingerprint() != fine1.fingerprint() or pair.p2.fine.fingerprint() != fine2.fingerprint():
        raise ValueError("policy embedded fine partition differs from registered fine.json")
    QD.check_bindings(pair, paths["teacher_model_sha256"], store.manifest["source"]["feature_names_sha256_newline_join"])
    if config["family"] == "D0":
        family, tables, decoder_sha = "D0", (pair.p1.token_proto, pair.p2.token_proto), None
    else:
        family, tables, decoder_sha, _ = HD.load_decoder(
            paths["decoder"], pair, paths["decoder_sha256"])
    if family != config["family"] or decoder_sha != paths["decoder_sha256"]:
        raise ValueError("A inherited decoder identity mismatch")
    banks = {}
    for r, pol, table in ((1, pair.p1, tables[0]), (2, pair.p2, tables[1])):
        bank = np.asarray(table, dtype=np.float64)
        classes = np.asarray(pol.token_class, dtype=np.int64)
        if bank.shape != (pol.T, pol.K) or classes.shape != (pol.T,) or \
                not np.array_equal(np.argmax(bank, axis=1), classes):
            raise ValueError("A output-bank schema/decision mismatch")
        for c in range(pol.K):
            if not np.any(classes == c) or not np.all(bank[classes == c, c] >
                                                       np.max(np.where(np.arange(pol.K) == c, -np.inf,
                                                                       bank[classes == c]), axis=1)):
                raise ValueError("A bank lacks strict predicted-class coverage")
        banks[r] = (bank, classes, np.asarray(pol.fine.cell_class, dtype=np.int64),
                    np.asarray(pol.cell_token, dtype=np.int64))
    source_binding = {
        "schema": "pcrl-portfolio-a-source-binding-v1",
        "release_id": config["release"],
        "model_seed": seed,
        "teacher_model_sha256": paths["teacher_model_sha256"],
        "feature_names_sha256_newline_join":
            store.manifest["source"]["feature_names_sha256_newline_join"],
        "fine_json_sha256": paths["fine_sha256"],
        "policy_json_sha256": paths["policy_sha256"],
        "decoder_json_sha256": paths["decoder_file_sha256"],
        "decoder_internal_sha256": paths["decoder_sha256"],
        "policy_fingerprint": pair.fingerprint(),
        "recipients": {},
    }
    for r, pol, fine in ((1, pair.p1, fine1), (2, pair.p2, fine2)):
        bank, classes, input_classes, fine_to_token = banks[r]
        source_binding["recipients"][f"r{r}"] = {
            "class_order": list(range(pol.K)),
            "fine_fingerprint": fine.fingerprint(),
            "input_class_sha256": sha_arr(input_classes),
            "fine_to_hcal_token_sha256": sha_arr(fine_to_token),
            "bank_probability_sha256": sha_arr(bank),
            "token_class_sha256": sha_arr(classes),
        }
    source_binding_sha256 = hashlib.sha256(json.dumps(
        source_binding, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    role_tokens = {}
    role_receipts = {}
    for role in ROLES:
        rec = store.read(role, f"references_s{seed}")
        out = {"row_id": rec["row_id"], "unit": rec["unit"]}
        for r, pol, fine in ((1, pair.p1, fine1), (2, pair.p2, fine2)):
            p = rec[f"U0_{r}"]
            d = rec[f"d_{r}"]
            cell = KM.assign_fine(p, d, fine)
            if not np.array_equal(np.asarray(fine.cell_class, dtype=np.int64)[cell], d):
                raise ValueError("A fine input-cell class mismatch")
            token = np.asarray(pol.cell_token, dtype=np.int64)[cell]
            token2, _, decision2 = QR.encode(pol, p)
            if not np.array_equal(token, token2) or not np.array_equal(decision2, d):
                raise ValueError("A role-only fine/token assignment differs from policy release")
            bank, classes, input_classes, fine_to_token = banks[r]
            if not np.array_equal(classes[token], d) or \
                    not np.array_equal(np.argmax(bank[token], axis=1), d) or \
                    not np.array_equal(input_classes[cell], d) or \
                    not np.array_equal(fine_to_token[cell], token):
                raise ValueError("A role-only token/class/decision mismatch")
            # A channel input is the fine cell. The inherited hcal coarse
            # token is a separate TASK-DET start/control, never substituted
            # for the fine input discretization under the same design ID.
            out[f"cell_{r}"] = cell
            out[f"token_{r}"] = token
        role_tokens[role] = out
        role_receipts[role] = {"rows": len(rec["row_id"]),
                               "row_unit_pair_sha256": sha_arr(np.column_stack((rec["row_id"], rec["unit"]))),
                               "source_reference_sha256": store.manifest["real_file_specs"][f"{role}:references_s{seed}"]["sha256"],
                               "cell_1_sha256": sha_arr(out["cell_1"]),
                               "cell_2_sha256": sha_arr(out["cell_2"]),
                               "token_1_sha256": sha_arr(out["token_1"]),
                               "token_2_sha256": sha_arr(out["token_2"])}
    output_root.mkdir(parents=True, mode=0o700)
    outputs = {}
    for r, (bank, classes, input_classes, fine_to_token) in banks.items():
        path = output_root / f"bank_r{r}.npz"
        np.savez_compressed(path, probability=bank, token_class=classes,
                            token_id=np.arange(len(classes), dtype=np.int64),
                            input_class=input_classes, fine_to_hcal_token=fine_to_token)
        path.chmod(0o600)
        outputs[path.name] = sha_file(path)
    for role, arrays in role_tokens.items():
        path = output_root / f"tokens_{role}.npz"
        np.savez_compressed(path, **arrays)
        path.chmod(0o600)
        outputs[path.name] = sha_file(path)
    receipt = {"schema": SCHEMA, "status": "ROLE_ONLY_BANK_ADMITTED", "seed": seed,
               "release_id": config["release"], "source_binding": source_binding,
               "source_binding_sha256": source_binding_sha256,
               "source_json_sha256": {k: paths[k] for k in
                   ("fine_sha256", "policy_sha256", "decoder_sha256", "decoder_file_sha256",
                    "teacher_model_sha256")},
               "role_receipts": role_receipts, "private_output_sha256": outputs,
               "policy_fingerprint": pair.fingerprint(), "decoder_sha256": decoder_sha,
               "data_access_manifest_sha256": sha_file(CAMPAIGN / "DATA_ACCESS_MANIFEST.json"),
               "science_lock_sha256": sha_file(CAMPAIGN / "CAMPAIGN_LOCK.json"),
               "s3_preprocessing_unit_id": unit_id,
               "preprocessing_only": True,
               "numpy_version": np.__version__, "python_version": sys.version.split()[0],
               "scope": "admitted roles only; no all-role assignment archive opened"}
    path = output_root / "BANK_ADMISSION_RECEIPT.json"
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    path.chmod(0o600)
    return {"seed": seed, "roles": len(ROLES), "banks": 2, "receipt_sha256": sha_file(path)}


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--release", choices=tuple(RELEASES), required=True)
    parser.add_argument("--campaign-lock-sha256", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = admit_one_seed(args.cache_root, args.seed, args.release,
                                args.campaign_lock_sha256, args.queue_root)
    except BaseException as exc:
        print("A_BANK_ADMISSION_FAILED:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True))

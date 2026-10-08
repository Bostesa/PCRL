"""Queue-only S4 numerical replay of the fixed raw83+Ucal feature anchor.

No candidate, S2 anchor, attacker, endpoint, or selector implementation is
imported. The report contains aggregates and custody hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from replay_b_attempt import _role, _same_people, _task_result
from replay_b_readers import auc
from replay_b_release import sha_file
from replay_feature_anchor import (ANCHOR_ID, build_views, replay_readers,
                                   source_forward)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _role_fields(seed: int) -> list[str]:
    return [f"ATTACK_FIT_NEW:{field}" for field in
            ("features", f"references_s{seed}", "sex")] + \
        [f"INNER_SELECTION:{field}" for field in
         ("features", f"references_s{seed}", "sex", "income", "occupation")]


def verify_anchor(campaign: Path, anchor_root: Path, cache_root: Path,
                  *, seed: int, fit_lock_sha: str, verifier: dict) -> dict:
    fit_lock_path = campaign / "CAMPAIGN_LOCK.json"
    if sha_file(fit_lock_path) != fit_lock_sha:
        raise ValueError("anchor fit lock bytes differ")
    lock = _json(fit_lock_path)
    shared_unit = f"S2-FEATURE-REF-RAW83-U-S{seed}"
    if (lock.get("fit_admission") != "B_ONLY"
            or shared_unit not in lock.get("admitted_shared_unit_ids", [])):
        raise PermissionError("raw83+Ucal anchor was not admitted")
    manifest_path = campaign / "DATA_ACCESS_MANIFEST.json"
    if sha_file(manifest_path) != lock["data_admission"]["manifest_sha256"]:
        raise ValueError("anchor role manifest differs from fit lock")
    manifest = _json(manifest_path)
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("anchor role export lacks S4 admission")
    receipt_path = anchor_root / "ANCHOR_RECEIPT.json"
    receipt_sha = sha_file(receipt_path)
    receipt = _json(receipt_path)
    admitted = verifier["unit_specs"][verifier["active_unit_id"]]
    if (verifier.get("schema") != "pcrl-s4-anchor-verifier-admission-v1"
            or verifier.get("status") != "SEALED"
            or verifier.get("accepted_fit_lock_sha256") != fit_lock_sha
            or verifier.get("allowed_role_fields") != _role_fields(seed)
            or admitted.get("seed") != seed
            or admitted.get("shared_unit_id") != shared_unit
            or admitted.get("anchor_receipt_sha256") != receipt_sha
            or receipt.get("status") != "EXPLORATORY_DEVELOPMENT"
            or receipt.get("arm") != ANCHOR_ID or receipt.get("seed") != seed
            or receipt.get("campaign_lock_sha256") != fit_lock_sha
            or receipt.get("data_access_manifest_sha256") != sha_file(manifest_path)):
        raise PermissionError("S4 anchor verifier identity differs")
    source_sha = lock["source_model_admission"]["U_model_pt_sha256_by_seed"][f"s{seed}"]
    if receipt.get("source_model_sha256") != source_sha:
        raise ValueError("anchor source U model identity differs")
    hcal_admission = campaign.parent / "pcrl_heldout_calibration_v1" / "SOURCE_ADMISSION.json"
    if sha_file(hcal_admission) != verifier["hcal_source_admission_sha256"]:
        raise ValueError("hcal source admission bytes differ")
    admitted_source = _json(hcal_admission)["admitted_teacher_dirs"][f"rel__s{seed}__U"]
    if admitted_source["model.pt"] != source_sha:
        raise ValueError("source admission does not bind U checkpoint")
    source_dir = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U"
    model_path = source_dir / "model.pt"
    heads = (source_dir / "head_0.joblib", source_dir / "head_1.joblib")

    data = {}
    for key in _role_fields(seed):
        role, field = key.split(":", 1)
        data[key] = _role(campaign, manifest, role, field)
        if receipt["role_file_sha256"][key] != manifest["real_file_specs"][key]["sha256"]:
            raise ValueError("anchor runner used a different admitted role file")
    att_features = data["ATTACK_FIT_NEW:features"]
    att_ref = data[f"ATTACK_FIT_NEW:references_s{seed}"]
    att_sex = data["ATTACK_FIT_NEW:sex"]
    inner_features = data["INNER_SELECTION:features"]
    inner_ref = data[f"INNER_SELECTION:references_s{seed}"]
    inner_sex = data["INNER_SELECTION:sex"]
    att_ids, att_groups = _same_people(att_features, att_ref, att_sex)
    inner_ids, inner_groups = _same_people(
        inner_features, inner_ref, inner_sex,
        data["INNER_SELECTION:income"], data["INNER_SELECTION:occupation"])
    if (len(att_ids), np.unique(att_groups).size, len(inner_ids),
            np.unique(inner_groups).size) != (4064, 4061, 2235, 2234):
        raise ValueError("anchor ATT/INNER role geometry differs")
    if np.intersect1d(att_ids, inner_ids).size:
        raise ValueError("anchor ATT and INNER share a person")
    source_residual = {}
    for role, features, refs in (("ATTACK_FIT_NEW", att_features, att_ref),
                                 ("INNER_SELECTION", inner_features, inner_ref)):
        source_residual[role] = source_forward(model_path, heads,
                                               admitted_source, features["X"], refs)
    att_views = build_views(att_features, att_ref, source_sha)
    inner_views = build_views(inner_features, inner_ref, source_sha)
    original_probe = {}
    for name in ("r1", "r2", "pair"):
        path = anchor_root / f"PROBE_{name}.json"
        if sha_file(path) != receipt["real_probe_receipt_sha256"][name]:
            raise ValueError("anchor original engineering probe bytes differ")
        probe = _json(path)
        original_probe[name] = probe["status"]
    local = {}
    for name in ("r1", "r2", "pair"):
        result = replay_readers(anchor_root, name, att_views[name], inner_views[name],
                                np.asarray(att_sex["sex"], dtype=np.int64), local)
        if result["receipt_sha256"] != receipt["reader_receipt_sha256"][name]:
            raise ValueError("anchor fitted reader custody differs")
        local[name] = result
    endpoint_json = anchor_root / "ENDPOINT_INPUTS.json"
    if sha_file(endpoint_json) != receipt["endpoint_input_receipt_sha256"]:
        raise ValueError("anchor endpoint sidecar bytes differ")
    endpoint_map = _json(endpoint_json)
    endpoint_npz = anchor_root / "ENDPOINT_INPUTS.npz"
    if sha_file(endpoint_npz) != endpoint_map["npz_sha256"]:
        raise ValueError("anchor endpoint arrays bytes differ")
    with np.load(endpoint_npz, allow_pickle=False) as endpoint:
        for key, values in (("row_id", inner_ids), ("unit", inner_groups),
                            ("sex", inner_sex["sex"]),
                            ("income", data["INNER_SELECTION:income"]["income"]),
                            ("occupation", data["INNER_SELECTION:occupation"]["occupation"])):
            if not np.array_equal(endpoint[key], values):
                raise ValueError("anchor endpoint person/task/SEX array differs")
        for r in (1, 2):
            for key in (f"U0_{r}", f"Ucal_{r}"):
                if not np.array_equal(endpoint[key], inner_ref[key]):
                    raise ValueError("anchor endpoint source reference differs")
            if not np.array_equal(endpoint[f"prob_{r}"], inner_ref[f"Ucal_{r}"]):
                raise ValueError("anchor endpoint release differs from Ucal score")
        for name in ("r1", "r2", "pair"):
            for reader, key in endpoint_map["reader_keys"][name].items():
                if not np.array_equal(endpoint[key],
                                      local[name]["predictions"]["inner"][reader]):
                    raise ValueError("anchor endpoint reader vector differs")
    tasks = {}
    for task, r in (("income", 1), ("occupation", 2)):
        label = np.asarray(data[f"INNER_SELECTION:{task}"][task], dtype=np.int64)
        q = np.asarray(inner_ref[f"Ucal_{r}"], dtype=np.float64)
        tasks[task] = _task_result(q, label, {name: np.asarray(inner_ref[f"{name}_{r}"],
                                                             dtype=np.float64)
                                               for name in ("U0", "Ucal")})
    y = np.asarray(inner_sex["sex"], dtype=np.int64)
    recovery = {}
    for name in ("r1", "r2", "pair"):
        by_reader = {reader: max(auc(y, score), 1.0 - auc(y, score))
                     for reader, score in local[name]["predictions"]["inner"].items()}
        recovery[name] = {"primary_envelope_auc": max(by_reader.values()),
                          "reader_fixed_alias_auc": by_reader,
                          "validation_winner": local[name]["validation_winner"]}
    return {"schema": "pcrl-s4-independent-feature-anchor-replay-v1",
            "status": "REPLAYED_ORIGINAL_CONTROLS_" +
            ("PASS" if all(value == "PASS" for value in original_probe.values()) else "FAIL"),
            "arm": ANCHOR_ID, "seed": seed, "shared_unit_id": shared_unit,
            "accepted_fit_lock_sha256": fit_lock_sha,
            "anchor_receipt_sha256": receipt_sha,
            "data_manifest_sha256": sha_file(manifest_path),
            "source_model_head_forward_max_abs_error": source_residual,
            "all_six_readers_and_pair_aliases": "PASS",
            "original_v1_probe_status": original_probe,
            "utility": tasks, "recovery": recovery,
            "scope": "reused INNER_SELECTION anchor; V2 control and paired comparison require separate admission"}


def _queue_gate(args, verifier: dict) -> None:
    if (verifier.get("schema") != "pcrl-s4-anchor-verifier-admission-v1"
            or verifier.get("status") != "SEALED"
            or verifier.get("active_unit_id") != args.unit_id
            or verifier.get("accepted_fit_lock_sha256") != args.lock_sha256
            or verifier.get("allowed_role_fields") != _role_fields(args.seed)
            or verifier.get("source_sha256_by_file", {}).get("replay_anchor_attempt.py") !=
            sha_file(Path(__file__))
            or verifier.get("source_sha256_by_file", {}).get("replay_feature_anchor.py") !=
            sha_file(Path(__file__).with_name("replay_feature_anchor.py"))
            or any(verifier.get("source_sha256_by_file", {}).get(name) !=
                   sha_file(Path(__file__).with_name(name)) for name in
                   ("replay_b_attempt.py", "replay_b_readers.py",
                    "replay_b_release.py", "independent_endpoints.py"))):
        raise PermissionError("S4 anchor numerical verifier not separately admitted")
    state = _json(Path(args.queue_root) / "QUEUE.json")
    unit = state.get("units", {}).get(args.unit_id, {})
    spec = unit.get("spec", {})
    argv = spec.get("argv", [])
    option = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)
              if isinstance(argv[i], str) and argv[i].startswith("--")}
    expected = {"--campaign": str(Path(args.campaign).resolve()),
                "--anchor-root": str(Path(args.anchor_root).resolve()),
                "--cache-root": str(Path(args.cache_root).resolve()),
                "--queue-root": str(Path(args.queue_root).resolve()),
                "--unit-id": args.unit_id, "--lock-sha256": args.lock_sha256,
                "--private-report": str(Path(args.private_report).resolve()),
                "--seed": str(args.seed),
                "--verifier-lock": str(Path(args.verifier_lock).resolve()),
                "--verifier-lock-sha256": args.verifier_lock_sha256}
    if (unit.get("status") != "running" or unit.get("child_pgid") != os.getpgrp()
            or spec.get("mode") != "s4_adult_replay" or spec.get("family") != "S4"
            or spec.get("team") != "S4" or spec.get("stage") != "independent_feature_anchor_replay"
            or spec.get("unit_id") != args.unit_id or spec.get("seed") != args.seed
            or spec.get("arm") != ANCHOR_ID
            or len(argv) < 2 or Path(argv[1]).resolve() != Path(__file__).resolve()
            or option != expected or state.get("gate", {}).get("safe_inputs") is not True
            or state.get("gate", {}).get("occupancy_confirmed") is not True
            or state.get("gate", {}).get("manually_closed") is True):
        raise PermissionError("S4 anchor replay lacks exact running S3 unit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "anchor-root", "cache-root", "queue-root", "unit-id",
                 "lock-sha256", "private-report", "verifier-lock", "verifier-lock-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    verifier_path = Path(args.verifier_lock).resolve()
    if (sha_file(verifier_path) != args.verifier_lock_sha256
            or sha_file(Path(args.campaign).resolve() / "CAMPAIGN_LOCK.json") != args.lock_sha256):
        raise PermissionError("S4 anchor or fit lock hash differs before role access")
    verifier = _json(verifier_path)
    _queue_gate(args, verifier)
    campaign = Path(args.campaign).resolve()
    expected_root = campaign / "shared" / ".private" / f"feature_ref_raw83_u_s{args.seed}_v1"
    if Path(args.anchor_root).resolve() != expected_root.resolve():
        raise PermissionError("S4 anchor root differs from admitted private source")
    output = Path(args.private_report).resolve()
    if ".private" not in output.parts or output.exists():
        raise ValueError("S4 anchor report must use a fresh private path")
    report = verify_anchor(campaign, expected_root, Path(args.cache_root).resolve(),
                           seed=args.seed, fit_lock_sha=args.lock_sha256,
                           verifier=verifier)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": report["schema"], "status": report["status"],
                      "seed": args.seed,
                      "private_report_sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()

"""Independent numerical replay for a raw83+Ucal feature anchor and V2 probes.

The caller first verifies the separate S4 admission and running S3 unit. This
module uses only S4 implementations and hash-admitted role/source artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from replay_anchor_attempt import _role, _same_people, _task_result
from replay_b_readers import auc
from replay_b_release import sha_file
from replay_feature_anchor import ANCHOR_ID, build_views, replay_readers, source_forward
from replay_v2_controls import replay_probe
from verifier_identity_v3 import admit_anchor_replay_identity


def _json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("anchor JSON must be an object")
    return result


def verify_anchor_values(campaign: Path, anchor_root: Path, probe_root: Path,
                         cache_root: Path, *, seed: int, verifier: dict,
                         unit_id: str, fit_lock: dict, fit_lock_sha256: str,
                         control_v2_lock_sha256: str) -> dict:
    """Replay source U, complete wire, readers, endpoints and V2 gates."""
    unit = verifier["unit_specs"][unit_id]
    anchor_path = anchor_root / "ANCHOR_RECEIPT.json"
    probe_path = probe_root / "V2_ANCHOR_PROBE_RECEIPT.json"
    anchor_sha, probe_sha = sha_file(anchor_path), sha_file(probe_path)
    anchor, probe_index = _json(anchor_path), _json(probe_path)
    admitted = admit_anchor_replay_identity(
        verifier=verifier, unit_id=unit_id, seed=seed,
        fit_lock=fit_lock, fit_lock_sha256=fit_lock_sha256,
        anchor_receipt=anchor, anchor_receipt_sha256=anchor_sha,
        control_v2_receipt=probe_index,
        control_v2_receipt_sha256=probe_sha,
        control_v2_lock_sha256=control_v2_lock_sha256)
    if (unit.get("parent_unit_id") != admitted["shared_unit_id"] or
            unit.get("control_v2_unit_id") != admitted["control_v2_unit_id"]):
        raise PermissionError("S4 anchor parent/V2 unit map differs")
    manifest_path = campaign / "DATA_ACCESS_MANIFEST.json"
    manifest_sha = sha_file(manifest_path)
    if (manifest_sha != fit_lock["data_admission"]["manifest_sha256"] or
            manifest_sha != anchor["data_access_manifest_sha256"]):
        raise ValueError("anchor admitted role manifest differs")
    manifest = _json(manifest_path)
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("anchor role export lacks S4 admission")
    source_sha = fit_lock["source_model_admission"]["U_model_pt_sha256_by_seed"][f"s{seed}"]
    if anchor.get("source_model_sha256") != source_sha:
        raise ValueError("anchor source U checkpoint differs")
    hcal_admission_path = campaign.parent / "pcrl_heldout_calibration_v1" / "SOURCE_ADMISSION.json"
    if (sha_file(hcal_admission_path) !=
            verifier["hcal_source_admission_sha256"]):
        raise ValueError("anchor hcal source admission file differs")
    hcal_admission = _json(hcal_admission_path)
    source = hcal_admission["admitted_teacher_dirs"][f"rel__s{seed}__U"]
    if source["model.pt"] != source_sha:
        raise ValueError("anchor hcal source admission U hash differs")
    source_dir = cache_root / "hcal_v1" / "admitted" / f"rel__s{seed}__U"
    model = source_dir / "model.pt"
    heads = (source_dir / "head_0.joblib", source_dir / "head_1.joblib")
    if sha_file(model) != source_sha:
        raise ValueError("anchor source U model bytes differ")

    role_fields = [("OSF_DEFENSE_FIT", "income"),
                   ("OSF_DEFENSE_FIT", "occupation")] + \
                  [("ATTACK_FIT_NEW", field) for field in
                   ("features", f"references_s{seed}", "sex")] + \
                  [("INNER_SELECTION", field) for field in
                   ("features", f"references_s{seed}", "sex", "income", "occupation")]
    if unit.get("allowed_role_fields") != [f"{role}:{field}" for role, field in role_fields]:
        raise PermissionError("anchor independent role allowlist differs")
    data = {}
    for role, field in role_fields:
        key = f"{role}:{field}"
        data[key] = _role(campaign, manifest, role, field)
        if (role != "OSF_DEFENSE_FIT" and
                anchor["role_file_sha256"][key] !=
                manifest["real_file_specs"][key]["sha256"]):
            raise ValueError("anchor runner used a different admitted role file")
    fit_id, fit_group = _same_people(
        data["OSF_DEFENSE_FIT:income"], data["OSF_DEFENSE_FIT:occupation"])
    if (len(fit_id), np.unique(fit_group).size) != (15434, 15428):
        raise ValueError("anchor FIT task-reference row/group geometry differs")
    att_features = data["ATTACK_FIT_NEW:features"]
    att_ref = data[f"ATTACK_FIT_NEW:references_s{seed}"]
    att_sex = data["ATTACK_FIT_NEW:sex"]
    inner_features = data["INNER_SELECTION:features"]
    inner_ref = data[f"INNER_SELECTION:references_s{seed}"]
    inner_sex = data["INNER_SELECTION:sex"]
    att_id, att_group = _same_people(att_features, att_ref, att_sex)
    inner_id, inner_group = _same_people(
        inner_features, inner_ref, inner_sex,
        data["INNER_SELECTION:income"], data["INNER_SELECTION:occupation"])
    if (len(att_id), np.unique(att_group).size, len(inner_id),
            np.unique(inner_group).size) != (4064, 4061, 2235, 2234):
        raise ValueError("anchor ATT/INNER row/group geometry differs")
    if np.intersect1d(att_id, inner_id).size:
        raise ValueError("anchor ATT/INNER people overlap")
    source_residual = {}
    for role, features, references in (("ATTACK_FIT_NEW", att_features, att_ref),
                                       ("INNER_SELECTION", inner_features, inner_ref)):
        source_residual[role] = source_forward(model, heads, source,
                                               features["X"], references)
    att_views = build_views(att_features, att_ref, source_sha)
    inner_views = build_views(inner_features, inner_ref, source_sha)
    readers, probes = {}, {}
    for view in ("r1", "r2", "pair"):
        readers[view] = replay_readers(
            anchor_root, view, att_views[view], inner_views[view],
            np.asarray(att_sex["sex"], dtype=np.int64), readers)
        if readers[view]["receipt_sha256"] != anchor["reader_receipt_sha256"][view]:
            raise ValueError("anchor frozen reader receipt differs")
        report_path = probe_root / f"PROBE_{view}.json"
        custody = probe_root / f"custody_{view}"
        if (sha_file(report_path) != unit["view_report_sha256"][view] or
                sha_file(report_path) != probe_index["probe_report_sha256"][view] or
                sha_file(custody / "CUSTODY_RECEIPT.json") !=
                unit["view_custody_receipt_sha256"][view] or
                sha_file(custody / "CUSTODY_RECEIPT.json") !=
                probe_index["probe_custody_receipt_sha256"][view]):
            raise ValueError("anchor V2 report/custody hash differs")
        probes[view] = replay_probe(
            custody, _json(report_path), att_views[view], inner_views[view],
            att_sex["sex"], inner_sex["sex"])
    expected_probe_status = "PASS" if all(
        probes[view]["status"] == "PASS" for view in probes) else "FAIL"
    if expected_probe_status != probe_index["status"]:
        raise ValueError("anchor V2 receipt status differs from independent replays")

    endpoint_json = anchor_root / "ENDPOINT_INPUTS.json"
    endpoint_npz = anchor_root / "ENDPOINT_INPUTS.npz"
    if sha_file(endpoint_json) != anchor["endpoint_input_receipt_sha256"]:
        raise ValueError("anchor endpoint sidecar bytes differ")
    sidecar = _json(endpoint_json)
    if sha_file(endpoint_npz) != sidecar["npz_sha256"]:
        raise ValueError("anchor endpoint array bytes differ")
    with np.load(endpoint_npz, allow_pickle=False) as endpoint:
        for key, values in (("row_id", inner_id), ("unit", inner_group),
                            ("sex", inner_sex["sex"]),
                            ("income", data["INNER_SELECTION:income"]["income"]),
                            ("occupation", data["INNER_SELECTION:occupation"]["occupation"])):
            if not np.array_equal(endpoint[key], values):
                raise ValueError("anchor endpoint person/task/SEX differs")
        for recipient in (1, 2):
            for name in (f"U0_{recipient}", f"Ucal_{recipient}"):
                if not np.array_equal(endpoint[name], inner_ref[name]):
                    raise ValueError("anchor endpoint U reference differs")
            if not np.array_equal(endpoint[f"prob_{recipient}"],
                                  inner_ref[f"Ucal_{recipient}"]):
                raise ValueError("anchor endpoint release probability differs")
        for view in ("r1", "r2", "pair"):
            for reader, key in sidecar["reader_keys"][view].items():
                if not np.array_equal(endpoint[key],
                                      readers[view]["predictions"]["inner"][reader]):
                    raise ValueError("anchor endpoint reader prediction differs")
    utility = {}
    for task, recipient in (("income", 1), ("occupation", 2)):
        label = np.asarray(data[f"INNER_SELECTION:{task}"][task], dtype=np.int64)
        q = np.asarray(inner_ref[f"Ucal_{recipient}"], dtype=np.float64)
        utility[task] = _task_result(
            q, label, {name: np.asarray(inner_ref[f"{name}_{recipient}"], dtype=np.float64)
                       for name in ("U0", "Ucal")})
        fit_label = np.asarray(data[f"OSF_DEFENSE_FIT:{task}"][task], dtype=np.int64)
        if fit_label.ndim != 1 or np.any((fit_label < 0) | (fit_label >= q.shape[1])):
            raise ValueError("anchor FIT task-reference class differs")
        majority = int(np.argmax(np.bincount(fit_label, minlength=q.shape[1])))
        utility[task]["fit_majority_constant_baseline"] = {
            "status": "FIT_MAJORITY_REPLAYED", "predicted_class": majority,
            "accuracy": float(np.mean(label == majority))}
    labels = np.asarray(inner_sex["sex"], dtype=np.int64)
    recovery = {}
    for view in ("r1", "r2", "pair"):
        by_reader = {reader: max(auc(labels, scores), 1.0 - auc(labels, scores))
                     for reader, scores in readers[view]["predictions"]["inner"].items()}
        recovery[view] = {"primary_envelope_auc": max(by_reader.values()),
                          "reader_fixed_alias_auc": by_reader,
                          "validation_winner": readers[view]["validation_winner"]}
    return {"schema": "pcrl-s4-independent-feature-anchor-replay-v3",
            "status": "REPLAYED_V2_PASS" if expected_probe_status == "PASS" else
                      "REPLAYED_V2_FAIL",
            "comparison_eligible_v2_control": expected_probe_status == "PASS",
            "arm": ANCHOR_ID, "seed": seed,
            "shared_unit_id": admitted["shared_unit_id"],
            "fit_lock_sha256": fit_lock_sha256,
            "anchor_receipt_sha256": anchor_sha,
            "control_v2_unit_id": admitted["control_v2_unit_id"],
            "control_v2_lock_sha256": control_v2_lock_sha256,
            "control_v2_receipt_sha256": probe_sha,
            "data_manifest_sha256": manifest_sha,
            "source_model_head_forward_max_abs_error": source_residual,
            "all_six_readers_and_pair_aliases": "PASS",
            "v2_control_model_label_prediction_replay": "PASS",
            "v2_control": {view: {"status": result["status"], "auc": result["auc"],
                                   "gates": result["gates"],
                                   "custody_receipt_sha256": result["custody_receipt_sha256"]}
                           for view, result in probes.items()},
            "utility": utility, "recovery": recovery,
            "scope": "reused INNER_SELECTION feature anchor; local AUC caps and competitive intervals pending"}

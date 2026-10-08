"""Independent numerical replay for one frozen Team B arm and V2 controls.

This module is called only after the separate S4 verifier/queue identity gate.
It reads admitted ATT/INNER role files and frozen private B outputs; it never
imports a Team B, S2 evaluator, or campaign selection implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from replay_b_attempt import (_check_aggregate_task, _match_number, _role,
                              _same_people, _task_result)
from replay_b_readers import MAIN_READERS, auc, replay_view
from replay_b_release import (head_state, load_final_state, load_saved_wire,
                              model_sha, rebuild_view, replay_encoder, sha_file)
from replay_v2_controls import replay_probe
from verifier_identity_v3 import admit_b_replay_identity


def _json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("frozen B JSON must be an object")
    return result


def verify_b_values(campaign: Path, attempt: Path, probe_root: Path, *,
                    arm: str, seed: int, verifier: dict, unit_id: str,
                    fit_lock: dict, fit_lock_sha256: str,
                    control_v2_lock_sha256: str) -> dict:
    """Replay wire, readers, references, endpoints and V2 probes in memory."""
    unit = verifier["unit_specs"][unit_id]
    index_path = attempt / "ATTEMPT_INDEX.json"
    probe_index_path = probe_root / "REAL_CONTROL_V2_INDEX.json"
    index_sha = sha_file(index_path)
    probe_index_sha = sha_file(probe_index_path)
    index, probe_index = _json(index_path), _json(probe_index_path)
    admitted = admit_b_replay_identity(
        verifier=verifier, unit_id=unit_id, arm=arm, seed=seed,
        fit_lock=fit_lock, fit_lock_sha256=fit_lock_sha256,
        attempt_index=index, attempt_index_sha256=index_sha,
        control_v2_index=probe_index,
        control_v2_index_sha256=probe_index_sha,
        control_v2_lock_sha256=control_v2_lock_sha256)
    if unit.get("parent_unit_id") != admitted["screen_unit_id"] or \
            unit.get("control_v2_unit_id") != admitted["control_v2_unit_id"]:
        raise PermissionError("S4 parent/probe unit map differs")
    manifest_path = campaign / "DATA_ACCESS_MANIFEST.json"
    manifest_sha = sha_file(manifest_path)
    if (manifest_sha != fit_lock["data_access_manifest_sha256"] or
            manifest_sha != index["data_manifest_sha256"]):
        raise ValueError("B admitted role-only manifest differs from historical fit lock")
    manifest = _json(manifest_path)
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("B role-only files lack independent admission")

    role_fields = (("OSF_DEFENSE_FIT", "income"),
                   ("OSF_DEFENSE_FIT", "occupation"),
                   ("ATTACK_FIT_NEW", "features"), ("ATTACK_FIT_NEW", "sex"),
                   ("INNER_SELECTION", "features"), ("INNER_SELECTION", "sex"),
                   ("INNER_SELECTION", "income"),
                   ("INNER_SELECTION", "occupation"),
                   ("INNER_SELECTION", f"references_s{seed}"))
    if unit.get("allowed_role_fields") != [f"{role}:{field}" for role, field in role_fields]:
        raise PermissionError("B independent role allowlist differs")
    data = {f"{role}:{field}": _role(campaign, manifest, role, field)
            for role, field in role_fields}
    fit_id, fit_group = _same_people(
        data["OSF_DEFENSE_FIT:income"], data["OSF_DEFENSE_FIT:occupation"])
    if (len(fit_id), len(np.unique(fit_group))) != (15434, 15428):
        raise ValueError("B FIT task-reference row/group geometry differs")
    att = data["ATTACK_FIT_NEW:features"]
    inner = data["INNER_SELECTION:features"]
    ref = data[f"INNER_SELECTION:references_s{seed}"]
    att_id, att_group = _same_people(att, data["ATTACK_FIT_NEW:sex"])
    inner_id, inner_group = _same_people(
        inner, data["INNER_SELECTION:sex"], data["INNER_SELECTION:income"],
        data["INNER_SELECTION:occupation"], ref)
    if (len(att_id), len(np.unique(att_group)), len(inner_id),
            len(np.unique(inner_group))) != (4064, 4061, 2235, 2234):
        raise ValueError("B ATT/INNER role row/group geometry differs")
    if np.intersect1d(att_id, inner_id).size:
        raise ValueError("B ATT and INNER people overlap")
    feature_names = tuple(str(x) for x in att["feature_names"])
    if (len(feature_names) != 83 or feature_names !=
            tuple(str(x) for x in inner["feature_names"])):
        raise ValueError("B permitted 83-column schema differs")

    state = load_final_state(attempt, index)
    if model_sha(state) != index["fit_model_sha256"]:
        raise ValueError("B frozen model fingerprint differs")
    heads = head_state(state)
    calibrators = {}
    for purpose in ("income", "occupation"):
        path = attempt / "calibration" / f"{purpose}.json"
        if sha_file(path) != index["calibration_record_sha256"][purpose]:
            raise ValueError("B calibration record file hash differs")
        calibrators[purpose] = _json(path)
        if (calibrators[purpose].get("task") != purpose or
                calibrators[purpose].get("fit_role") != "CALIBRATION_HELDOUT"):
            raise ValueError("B held-out calibration role/task differs")
    secret = (attempt / "sampler_secret.bin").read_bytes()
    if len(secret) != 32:
        raise ValueError("B one-shot sampler secret length differs")
    wires = {}
    for role, features, ids in (("ATTACK_FIT_NEW", att, att_id),
                                ("INNER_SELECTION", inner, inner_id)):
        wires[role] = load_saved_wire(
            attempt / f"release__{role}.json", ids, features["X"],
            index["saved_release_sha256"][role], index["source_sha256"],
            index["fit_model_sha256"], index["calibration_record_sha256"])
        replay_encoder(wires[role], features["X"], ids, state, secret,
                       index["source_sha256"], feature_names)

    sex_att = np.asarray(data["ATTACK_FIT_NEW:sex"]["sex"], dtype=np.int64)
    sex_inner = np.asarray(data["INNER_SELECTION:sex"]["sex"], dtype=np.int64)
    readers, probes = {}, {}
    for view in ("income", "occupation", "pair"):
        att_view = rebuild_view(wires["ATTACK_FIT_NEW"], view, heads, calibrators)
        inner_view = rebuild_view(wires["INNER_SELECTION"], view, heads, calibrators)
        aliases = None
        if view == "pair":
            aliases = {}
            for recipient, short in (("income", "r1"), ("occupation", "r2")):
                prior = readers[recipient]
                aliases[short] = (
                    [prior["predictions_tune"][name] for name in MAIN_READERS],
                    [prior["predictions_inner"][name] for name in MAIN_READERS])
        readers[view] = replay_view(
            attempt, view, att_view, inner_view,
            att_id, att_group, inner_id, inner_group, sex_att,
            pair_aliases=aliases)
        if readers[view]["reader_receipt_sha256"] != index["reader_receipt_sha256"][view]:
            raise ValueError("B reader receipt differs from frozen attempt index")
        report_path = probe_root / f"REAL_CONTROL_V2__{view}.json"
        custody = probe_root / f"custody__{view}"
        if (sha_file(report_path) != unit["view_report_sha256"][view] or
                sha_file(report_path) != probe_index["view_report_sha256"][view] or
                sha_file(custody / "CUSTODY_RECEIPT.json") !=
                unit["view_custody_receipt_sha256"][view] or
                sha_file(custody / "CUSTODY_RECEIPT.json") !=
                probe_index["view_custody_receipt_sha256"][view]):
            raise ValueError("B V2 report/custody hash differs")
        probes[view] = replay_probe(
            custody, _json(report_path),
            {"row_id": att_id, "unit": att_group, "x": att_view["x"]},
            {"row_id": inner_id, "unit": inner_group, "x": inner_view["x"]},
            sex_att, sex_inner)
    expected_probe_status = "PASS" if all(
        probes[view]["status"] == "PASS" for view in probes) else "FAIL"
    if expected_probe_status != probe_index["status"]:
        raise ValueError("B V2 index status differs from independent view replays")

    sidecar_path = attempt / "ENDPOINT_INPUTS.json"
    endpoint_path = attempt / "ENDPOINT_INPUTS.npz"
    if sha_file(sidecar_path) != index["endpoint_input_receipt_sha256"]:
        raise ValueError("B endpoint sidecar hash differs")
    sidecar = _json(sidecar_path)
    if (sha_file(endpoint_path) != index["endpoint_arrays_sha256"] or
            sidecar["npz_sha256"] != index["endpoint_arrays_sha256"]):
        raise ValueError("B endpoint array file hash differs")
    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        for key, values in (("row_id", inner_id), ("unit", inner_group),
                            ("sex", sex_inner),
                            ("income", data["INNER_SELECTION:income"]["income"]),
                            ("occupation", data["INNER_SELECTION:occupation"]["occupation"])):
            if not np.array_equal(endpoint[key], values):
                raise ValueError("B endpoint person/label array differs")
        for purpose, suffix in (("income", 1), ("occupation", 2)):
            q = np.asarray([person[purpose]["probabilities"]
                            for person in wires["INNER_SELECTION"]], dtype=np.float64)
            if not np.array_equal(endpoint[f"prob_{suffix}"], q):
                raise ValueError("B endpoint probability differs from actual release")
            for reference in ("U0", "Ucal"):
                if not np.array_equal(endpoint[f"{reference}_{suffix}"],
                                      ref[f"{reference}_{suffix}"]):
                    raise ValueError("B endpoint frozen reference differs")
        for view, short in (("income", "r1"), ("occupation", "r2"), ("pair", "pair")):
            for reader, key in sidecar["reader_keys"][short].items():
                if (reader not in readers[view]["predictions_inner"] or
                        not np.array_equal(endpoint[key],
                                           readers[view]["predictions_inner"][reader])):
                    raise ValueError("B endpoint reader vector differs")

    aggregate_path = attempt / "AGGREGATE.json"
    if sha_file(aggregate_path) != index["aggregate_sha256"]:
        raise ValueError("B aggregate report hash differs")
    aggregate = _json(aggregate_path)
    utility = {}
    for purpose, suffix in (("income", 1), ("occupation", 2)):
        label = np.asarray(data[f"INNER_SELECTION:{purpose}"][purpose], dtype=np.int64)
        q = np.asarray([person[purpose]["probabilities"]
                        for person in wires["INNER_SELECTION"]], dtype=np.float64)
        utility[purpose] = _task_result(
            q, label, {reference: np.asarray(ref[f"{reference}_{suffix}"], dtype=np.float64)
                       for reference in ("U0", "Ucal")})
        fit_label = np.asarray(data[f"OSF_DEFENSE_FIT:{purpose}"][purpose],
                               dtype=np.int64)
        if fit_label.ndim != 1 or np.any((fit_label < 0) | (fit_label >= q.shape[1])):
            raise ValueError("B FIT task-reference class differs")
        majority = int(np.argmax(np.bincount(fit_label, minlength=q.shape[1])))
        utility[purpose]["fit_majority_constant_baseline"] = {
            "status": "FIT_MAJORITY_REPLAYED", "predicted_class": majority,
            "accuracy": float(np.mean(label == majority))}
        _check_aggregate_task(utility[purpose], aggregate["task"], purpose)
        _match_number(utility[purpose]["fit_majority_constant_baseline"]["accuracy"],
                      aggregate["task"][purpose]["candidate"]["constant_accuracy"],
                      f"{purpose}/FIT-majority constant accuracy")
        _match_number(utility[purpose]["accuracy"] -
                      utility[purpose]["fit_majority_constant_baseline"]["accuracy"],
                      aggregate["task"][purpose]["candidate"]["accuracy_gain_over_constant"],
                      f"{purpose}/FIT-majority constant gain")
    recovery = {}
    for view in ("income", "occupation", "pair"):
        by_reader = {name: max(auc(sex_inner, prediction),
                               1.0 - auc(sex_inner, prediction))
                     for name, prediction in readers[view]["predictions_inner"].items()}
        envelope = max(by_reader.values())
        _match_number(envelope, aggregate["recovery"][view]["primary_envelope_auc"],
                      f"{view}/coalition-AUC")
        for name, value in by_reader.items():
            _match_number(value,
                          aggregate["recovery"][view]["readers"][name]["fixed_alias_max_auc"],
                          f"{view}/{name}/AUC")
        recovery[view] = {"primary_envelope_auc": envelope,
                          "reader_auc_fixed_alias": by_reader,
                          "validation_winner": readers[view]["validation_winner"]}
    return {"schema": "pcrl-s4-independent-b-attempt-replay-v3",
            "status": "REPLAYED_V2_PASS" if expected_probe_status == "PASS" else
                      "REPLAYED_V2_FAIL",
            "comparison_eligible_v2_control": expected_probe_status == "PASS",
            "arm": arm, "seed": seed, "screen_unit_id": admitted["screen_unit_id"],
            "fit_lock_sha256": fit_lock_sha256,
            "attempt_index_sha256": index_sha,
            "control_v2_unit_id": admitted["control_v2_unit_id"],
            "control_v2_lock_sha256": control_v2_lock_sha256,
            "control_v2_index_sha256": probe_index_sha,
            "data_manifest_sha256": manifest_sha,
            "admitted_role_file_sha256": {
                key: manifest["real_file_specs"][key]["sha256"] for key in data},
            "att_group_split": {"fit_rows": readers["income"]["att_fit_rows"],
                                "tune_rows": readers["income"]["att_tune_rows"],
                                "fit_sex_counts": readers["income"]["att_fit_sex_counts"],
                                "tune_sex_counts": readers["income"]["att_tune_sex_counts"]},
            "complete_wire_public_head_encoder_sampler_calibration": "PASS",
            "all_frozen_reader_predictions_and_ignore_aliases": "PASS",
            "v2_control_model_label_prediction_replay": "PASS",
            "v2_control": {view: {"status": result["status"], "auc": result["auc"],
                                   "gates": result["gates"],
                                   "custody_receipt_sha256": result["custody_receipt_sha256"]}
                           for view, result in probes.items()},
            "utility": utility, "recovery": recovery,
            "scope": "single arm/seed on reused INNER_SELECTION; competitive selection and confidence intervals pending"}

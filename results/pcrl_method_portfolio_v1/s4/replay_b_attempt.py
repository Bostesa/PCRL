"""Independently replay one completed Team B Adult screen from private custody.

The CLI is gated for a separately locked S3 S4-replay unit. It reads only the
admitted ATTACK_FIT_NEW and INNER_SELECTION role-specific files, the saved
Team B attempt and its real-control reports. It never imports Team B or S2
method, attack, endpoint, or selection implementations. Public output is an
aggregate verification report without person-level arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from independent_endpoints import TaskRows, _task_metric
from replay_b_readers import MAIN_READERS, auc, replay_view
from replay_b_release import (head_state, load_final_state, load_saved_wire,
                              model_sha, rebuild_view, replay_encoder,
                              schema_sha, sha_array, sha_file)


ROLE_FIELDS = (("ATTACK_FIT_NEW", "features"), ("ATTACK_FIT_NEW", "sex"),
               ("INNER_SELECTION", "features"), ("INNER_SELECTION", "income"),
               ("INNER_SELECTION", "occupation"), ("INNER_SELECTION", "sex"))


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _role(campaign: Path, manifest: dict, role: str, field: str) -> dict[str, np.ndarray]:
    spec = manifest["real_file_specs"][f"{role}:{field}"]
    path = (campaign / spec["path"]).resolve()
    if ".private" not in path.parts or not path.is_relative_to(campaign.resolve()) or \
            sha_file(path) != spec["sha256"]:
        raise ValueError("admitted role file path/hash differs")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(spec["members"]):
            raise ValueError("role file members differ from admitted allowlist")
        result = {name: z[name].copy() for name in z.files}
    row = np.asarray(result["row_id"], dtype=np.int64)
    unit = np.asarray(result["unit"], dtype=np.int64)
    if row.ndim != 1 or unit.shape != row.shape or \
            sha_array(np.column_stack((row, unit)).astype(np.int64)) != \
            spec["row_unit_pair_sha256"]:
        raise ValueError("role row/group identity differs")
    return result


def _same_people(*records: dict) -> tuple[np.ndarray, np.ndarray]:
    first = records[0]
    ids = np.asarray(first["row_id"], dtype=np.int64)
    groups = np.asarray(first["unit"], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate admitted row")
    if any(not np.array_equal(np.asarray(record["row_id"], dtype=np.int64), ids)
           or not np.array_equal(np.asarray(record["unit"], dtype=np.int64), groups)
           for record in records[1:]):
        raise ValueError("role-specific files have different ordered people/groups")
    return ids, groups


def _task_result(q: np.ndarray, label: np.ndarray, reference: dict[str, np.ndarray]) -> dict:
    n, width = q.shape
    task = TaskRows(q, q.argmax(axis=1)).checked(n, width)
    weights = np.ones(n, dtype=np.float64)
    acc, nll, brier = _task_metric(task, label, weights)
    recall = [float(np.mean(task.decision[label == c] == c)) if np.any(label == c) else None
              for c in range(width)]
    result = {"rows": n, "accuracy": acc, "nll": nll, "brier": brier,
              "per_class_recall": recall,
              "class_counts": np.bincount(label, minlength=width).astype(int).tolist(),
              "references": {}}
    for name, ref in reference.items():
        ref_task = TaskRows(ref, ref.argmax(axis=1)).checked(n, width)
        ra, rn, rb = _task_metric(ref_task, label, weights)
        result["references"][name] = {
            "accuracy": ra, "nll": rn, "brier": rb,
            "accuracy_difference": acc - ra,
            "nll_excess": nll - rn, "brier_excess": brier - rb,
            "point_eligible": acc - ra >= -0.010 and nll - rn <= 0.010 and brier - rb <= 0.005}
    return result


def _match_number(actual: float, reported: float, name: str, tol: float = 1e-10) -> None:
    if not np.isfinite(actual) or not np.isfinite(reported) or abs(actual - reported) > tol:
        raise ValueError(f"independent {name} differs from reported aggregate")


def _check_aggregate_task(actual: dict, reported: dict, purpose: str) -> None:
    row = reported[purpose]
    for metric in ("accuracy", "nll", "brier"):
        _match_number(actual[metric], row["candidate"][metric], f"{purpose}/{metric}")
    if row["candidate"]["class_counts"] != actual["class_counts"]:
        raise ValueError("reported per-class counts differ")
    for a, b in zip(actual["per_class_recall"], row["candidate"]["per_class_recall"]):
        if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-10):
            raise ValueError("reported per-class recall differs")
    for ref in ("U0", "Ucal"):
        for metric in ("accuracy", "nll", "brier"):
            _match_number(actual["references"][ref][metric], row[ref][metric],
                          f"{purpose}/{ref}/{metric}")
        if actual["references"][ref]["point_eligible"] != \
                row["comparison"][ref]["point_eligible"]:
            raise ValueError("reported utility point gate differs")


def verify_attempt(campaign: Path, attempt: Path, probe_dir: Path,
                   *, arm: str, seed: int, lock_sha: str, verifier: dict) -> dict:
    """Run all person-level work in memory; return aggregate-only verification."""
    lock_path = campaign / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != lock_sha:
        raise ValueError("effective campaign lock hash differs")
    lock = _json(lock_path)
    if (verifier.get("schema") != "pcrl-s4-b-verifier-admission-v1"
            or verifier.get("status") != "SEALED"
            or verifier.get("accepted_fit_lock_sha256") != lock_sha):
        raise PermissionError("separate S4 verifier seal does not bind fit lock")
    if (lock.get("fit_admission") != "B_ONLY"
            or arm not in lock["admitted_arms"]["B"] or seed not in (0, 1)):
        raise PermissionError("B-only arm/seed not admitted")
    manifest_path = campaign / "DATA_ACCESS_MANIFEST.json"
    if sha_file(manifest_path) != lock["data_access_manifest_sha256"]:
        raise ValueError("role-only manifest differs from lock")
    manifest = _json(manifest_path)
    if manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("role-only files not S4 admitted")
    index_path = attempt / "ATTEMPT_INDEX.json"
    index_sha = sha_file(index_path)
    index = _json(index_path)
    expected_screen_unit = lock["families"]["B"]["screen_unit_ids"][f"{arm}:s{seed}"]
    if (index.get("status") != "FROZEN_FOR_INDEPENDENT_REPLAY"
            or index.get("arm") != arm or index.get("seed") != seed
            or index.get("screen_unit_id") != expected_screen_unit
            or index.get("campaign_lock_sha256") != verifier["accepted_fit_lock_sha256"]
            or index.get("data_manifest_sha256") != sha_file(manifest_path)):
        raise ValueError("fitted attempt identity differs from lock")
    probe_index_path = probe_dir / "REAL_CONTROL_INDEX.json"
    probe_index = _json(probe_index_path)
    expected_probe_unit = lock["families"]["B"]["real_control_unit_ids"][f"{arm}:s{seed}"]
    if (probe_index.get("status") != "PASS" or probe_index.get("arm") != arm
            or probe_index.get("seed") != seed or probe_index.get("screen_unit_id") != expected_screen_unit
            or probe_index.get("queue_unit_id") != expected_probe_unit
            or probe_index.get("attempt_index_sha256") != index_sha):
        raise ValueError("real postrelease controls are absent or failed")
    admitted = verifier.get("unit_specs", {}).get(verifier.get("active_unit_id"), {})
    if (admitted.get("arm") != arm or admitted.get("seed") != seed
            or admitted.get("screen_unit_id") != expected_screen_unit
            or admitted.get("attempt_index_sha256") != index_sha
            or admitted.get("real_control_index_sha256") != sha_file(probe_index_path)):
        raise PermissionError("S4 verifier seal does not bind this completed attempt/probe")
    for view in ("income", "occupation", "pair"):
        p = probe_dir / f"REAL_CONTROL__{view}.json"
        if sha_file(p) != probe_index["view_report_sha256"][view]:
            raise ValueError("real control report hash differs")
        report = _json(p)
        gates = report.get("probe", {}).get("gates", {})
        if report.get("status") != "PASS" or report.get("attempt_index_sha256") != index_sha or \
                report.get("view") != view or report.get("probe", {}).get("status") != "PASS" or \
                set(gates) != {"null_exact", "shuffled_at_most_0.60", "planted_at_least_0.99"} or \
                not all(gates.values()):
            raise ValueError("real null/shuffle/planted control did not pass")

    data = {f"{role}:{field}": _role(campaign, manifest, role, field)
            for role, field in ROLE_FIELDS}
    ref_key = f"INNER_SELECTION:references_s{seed}"
    data[ref_key] = _role(campaign, manifest, "INNER_SELECTION", f"references_s{seed}")
    att = data["ATTACK_FIT_NEW:features"]
    inner = data["INNER_SELECTION:features"]
    att_ids, att_groups = _same_people(att, data["ATTACK_FIT_NEW:sex"])
    inner_ids, inner_groups = _same_people(
        inner, data["INNER_SELECTION:income"], data["INNER_SELECTION:occupation"],
        data["INNER_SELECTION:sex"], data[ref_key])
    if (len(att_ids), len(np.unique(att_groups)), len(inner_ids),
            len(np.unique(inner_groups))) != (4064, 4061, 2235, 2234):
        raise ValueError("ATT/INNER role sizes differ from lock")
    if np.intersect1d(att_ids, inner_ids).size:
        raise ValueError("ATT/INNER people overlap")
    feature_names = tuple(str(x) for x in att["feature_names"])
    if (len(feature_names) != 83 or feature_names !=
            tuple(str(x) for x in inner["feature_names"])):
        raise ValueError("83-column permitted feature schema differs")

    state = load_final_state(attempt, index)
    if model_sha(state) != index["fit_model_sha256"]:
        raise ValueError("public model fingerprint differs")
    heads = head_state(state)
    calibrators = {}
    for purpose in ("income", "occupation"):
        p = attempt / "calibration" / f"{purpose}.json"
        if sha_file(p) != index["calibration_record_sha256"][purpose]:
            raise ValueError("held-out calibrator hash differs")
        calibrators[purpose] = _json(p)
        if (calibrators[purpose].get("task") != purpose
                or calibrators[purpose].get("fit_role") != "CALIBRATION_HELDOUT"):
            raise ValueError("calibrator role/task identity differs")
    secret_path = attempt / "sampler_secret.bin"
    secret = secret_path.read_bytes()
    if len(secret) != 32:
        raise ValueError("one-shot sampler secret length differs")
    wires = {}
    for role, features, ids in (("ATTACK_FIT_NEW", att, att_ids),
                                ("INNER_SELECTION", inner, inner_ids)):
        wires[role] = load_saved_wire(
            attempt / f"release__{role}.json", ids, features["X"],
            index["saved_release_sha256"][role], index["source_sha256"],
            index["fit_model_sha256"], index["calibration_record_sha256"])
        replay_encoder(wires[role], features["X"], ids, state, secret,
                       index["source_sha256"], feature_names)

    outputs = {}
    for view in ("income", "occupation", "pair"):
        att_view = rebuild_view(wires["ATTACK_FIT_NEW"], view, heads, calibrators)
        inner_view = rebuild_view(wires["INNER_SELECTION"], view, heads, calibrators)
        pair_aliases = None
        if view == "pair":
            pair_aliases = {}
            for recipient, short in (("income", "r1"), ("occupation", "r2")):
                previous = outputs[recipient]
                pair_aliases[short] = (
                    [previous["predictions_tune"][name] for name in MAIN_READERS],
                    [previous["predictions_inner"][name] for name in MAIN_READERS])
        outputs[view] = replay_view(
            attempt, view, att_view, inner_view,
            att_ids, att_groups, inner_ids, inner_groups,
            data["ATTACK_FIT_NEW:sex"]["sex"], pair_aliases=pair_aliases)
        if outputs[view]["reader_receipt_sha256"] != index["reader_receipt_sha256"][view]:
            raise ValueError("fitted reader receipt differs from attempt index")

    endpoint_sidecar = attempt / "ENDPOINT_INPUTS.json"
    if sha_file(endpoint_sidecar) != index["endpoint_input_receipt_sha256"]:
        raise ValueError("endpoint sidecar hash differs")
    endpoint_map = _json(endpoint_sidecar)
    endpoint_file = attempt / "ENDPOINT_INPUTS.npz"
    if (sha_file(endpoint_file) != index["endpoint_arrays_sha256"]
            or endpoint_map["npz_sha256"] != index["endpoint_arrays_sha256"]):
        raise ValueError("private endpoint arrays hash differs")
    with np.load(endpoint_file, allow_pickle=False) as endpoint:
        for key, values in (("row_id", inner_ids), ("unit", inner_groups),
                            ("income", data["INNER_SELECTION:income"]["income"]),
                            ("occupation", data["INNER_SELECTION:occupation"]["occupation"]),
                            ("sex", data["INNER_SELECTION:sex"]["sex"])):
            if not np.array_equal(endpoint[key], values):
                raise ValueError("endpoint role labels/people differ")
        for purpose, suffix in (("income", 1), ("occupation", 2)):
            q = np.asarray([v[purpose]["probabilities"] for v in wires["INNER_SELECTION"]],
                           dtype=np.float64)
            if not np.array_equal(endpoint[f"prob_{suffix}"], q):
                raise ValueError("endpoint probability differs from actual release")
            for ref in ("U0", "Ucal"):
                if not np.array_equal(endpoint[f"{ref}_{suffix}"], data[ref_key][f"{ref}_{suffix}"]):
                    raise ValueError("endpoint reference differs from admitted role file")
        for view, short in (("income", "r1"), ("occupation", "r2"), ("pair", "pair")):
            for reader, key in endpoint_map["reader_keys"][short].items():
                if reader not in outputs[view]["predictions_inner"] or \
                        not np.array_equal(endpoint[key], outputs[view]["predictions_inner"][reader]):
                    raise ValueError("endpoint attack vector differs from replayed reader")

    aggregate_path = attempt / "AGGREGATE.json"
    if sha_file(aggregate_path) != index["aggregate_sha256"]:
        raise ValueError("reported aggregate hash differs")
    aggregate = _json(aggregate_path)
    tasks = {}
    for purpose, suffix in (("income", 1), ("occupation", 2)):
        label = np.asarray(data[f"INNER_SELECTION:{purpose}"][purpose], dtype=np.int64)
        q = np.asarray([v[purpose]["probabilities"] for v in wires["INNER_SELECTION"]],
                       dtype=np.float64)
        tasks[purpose] = _task_result(
            q, label, {ref: np.asarray(data[ref_key][f"{ref}_{suffix}"], dtype=np.float64)
                       for ref in ("U0", "Ucal")})
        _check_aggregate_task(tasks[purpose], aggregate["task"], purpose)
    recovery = {}
    y = np.asarray(data["INNER_SELECTION:sex"]["sex"], dtype=np.int64)
    for view in ("income", "occupation", "pair"):
        reader_auc = {name: max(auc(y, score), 1.0 - auc(y, score))
                      for name, score in outputs[view]["predictions_inner"].items()}
        envelope = max(reader_auc.values())
        _match_number(envelope, aggregate["recovery"][view]["primary_envelope_auc"],
                      f"{view}/AUC")
        for name, value in reader_auc.items():
            _match_number(value,
                          aggregate["recovery"][view]["readers"][name]["fixed_alias_max_auc"],
                          f"{view}/{name}/AUC")
        recovery[view] = {"primary_envelope_auc": envelope,
                          "reader_auc_fixed_alias": reader_auc,
                          "validation_winner": outputs[view]["validation_winner"]}
    return {"schema": "pcrl-s4-independent-b-attempt-replay-v1",
            "status": "PASS", "arm": arm, "seed": seed,
            "screen_unit_id": expected_screen_unit,
            "real_probe_unit_id": expected_probe_unit,
            "campaign_lock_sha256": lock_sha,
            "attempt_index_sha256": index_sha,
            "real_control_index_sha256": sha_file(probe_index_path),
            "data_manifest_sha256": sha_file(manifest_path),
            "admitted_role_file_hashes": {
                key: manifest["real_file_specs"][key]["sha256"] for key in data},
            "att_group_split": {"fit_rows": outputs["income"]["att_fit_rows"],
                                "tune_rows": outputs["income"]["att_tune_rows"],
                                "fit_sex_counts": outputs["income"]["att_fit_sex_counts"],
                                "tune_sex_counts": outputs["income"]["att_tune_sex_counts"]},
            "complete_wire_public_head_encoder_sampler_calibration": "PASS",
            "all_frozen_reader_predictions_and_ignore_aliases": "PASS",
            "real_null_shuffle_planted_controls": "PASS",
            "utility": tasks, "recovery": recovery,
            "scope": "single arm/seed on reused INNER_SELECTION; no competitive/control selection or confidence interval"}


def _queue_gate(args, verifier: dict) -> None:
    """Fail closed unless S3 runs the exact independently admitted verifier."""
    if (verifier.get("schema") != "pcrl-s4-b-verifier-admission-v1"
            or verifier.get("status") != "SEALED"
            or verifier.get("active_unit_id") != args.unit_id
            or verifier.get("accepted_fit_lock_sha256") != args.lock_sha256
            or verifier.get("source_sha256_by_file", {}).get("replay_b_attempt.py") !=
            sha_file(Path(__file__))
            or verifier.get("source_sha256_by_file", {}).get("replay_b_release.py") !=
            sha_file(Path(__file__).with_name("replay_b_release.py"))
            or verifier.get("source_sha256_by_file", {}).get("replay_b_readers.py") !=
            sha_file(Path(__file__).with_name("replay_b_readers.py"))
            or verifier.get("source_sha256_by_file", {}).get("independent_endpoints.py") !=
            sha_file(Path(__file__).with_name("independent_endpoints.py"))
            or verifier.get("allowed_role_fields") !=
            [f"{role}:{field}" for role, field in ROLE_FIELDS] +
            [f"INNER_SELECTION:references_s{args.seed}"]):
        raise PermissionError("S4 numerical verifier not admitted by amended lock")
    state = _json(Path(args.queue_root) / "QUEUE.json")
    unit = state.get("units", {}).get(args.unit_id, {})
    spec = unit.get("spec", {})
    argv = spec.get("argv", [])
    option = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)
              if isinstance(argv[i], str) and argv[i].startswith("--")}
    admitted = verifier["unit_specs"].get(args.unit_id, {})
    expected = {"--campaign": str(Path(args.campaign).resolve()),
                "--attempt": str(Path(args.attempt).resolve()),
                "--probe-dir": str(Path(args.probe_dir).resolve()),
                "--queue-root": str(Path(args.queue_root).resolve()),
                "--unit-id": args.unit_id, "--lock-sha256": args.lock_sha256,
                "--arm": args.arm, "--private-report": str(Path(args.private_report).resolve()),
                "--seed": str(args.seed),
                "--verifier-lock": str(Path(args.verifier_lock).resolve()),
                "--verifier-lock-sha256": args.verifier_lock_sha256}
    if (unit.get("status") != "running" or unit.get("child_pgid") != os.getpgrp()
            or spec.get("mode") != "s4_adult_replay"
            or spec.get("family") != "S4" or spec.get("team") != "S4"
            or spec.get("stage") != "independent_b_attempt_replay"
            or spec.get("unit_id") != args.unit_id
            or spec.get("arm") != admitted.get("arm")
            or spec.get("seed") != admitted.get("seed")
            or len(argv) < 2 or Path(argv[1]).resolve() != Path(__file__).resolve()
            or option != expected
            or state.get("gate", {}).get("safe_inputs") is not True
            or state.get("gate", {}).get("occupancy_confirmed") is not True
            or state.get("gate", {}).get("manually_closed") is True):
        raise PermissionError("S4 replay lacks matching running S3 unit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("campaign", "attempt", "probe-dir", "queue-root", "unit-id",
                "lock-sha256", "arm", "private-report", "verifier-lock",
                "verifier-lock-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    campaign = Path(args.campaign).resolve()
    lock_path = campaign / "CAMPAIGN_LOCK.json"
    if sha_file(lock_path) != args.lock_sha256:
        raise PermissionError("S4 lock hash differs before role access")
    verifier_path = Path(args.verifier_lock).resolve()
    if sha_file(verifier_path) != args.verifier_lock_sha256:
        raise PermissionError("separate S4 verifier lock hash differs before role access")
    verifier = _json(verifier_path)
    _queue_gate(args, verifier)
    output = Path(args.private_report).resolve()
    if ".private" not in output.parts or output.exists():
        raise ValueError("S4 private report must be a new write-once path")
    report = verify_attempt(campaign, Path(args.attempt).resolve(),
                            Path(args.probe_dir).resolve(),
                            arm=args.arm, seed=args.seed, lock_sha=args.lock_sha256,
                            verifier=verifier)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": report["schema"], "status": report["status"],
                      "arm": args.arm, "seed": args.seed,
                      "private_report_sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()

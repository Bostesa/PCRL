"""Queued independent 756-slot recovery and A/B/C claim replay.

The source reader predictions and per-release AUC vectors must already have
passed R5's separate frozen-reader replay. This worker imports no R3/R4
scoring, interval, selection, or claim implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_attack_math import (ARMS, BOOTSTRAPS, K, STOCHASTIC, VIEWS,
                             fixed_recovery_family, paired_plan,
                             summarize_release_vectors)
from .r5_claims_protocol import all_claims
from .r5_selection_math import strongest_control


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def exact_npz(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        names = [x.filename for x in archive.infolist()]
        if len(names) != len(expected) or set(names) != {x + ".npy" for x in expected}:
            raise PermissionError("private replay vector member inventory differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in expected}


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    spec = state["units"][active["unit_id"]]["spec"]
    source_index = {str(Path(x["path"]).resolve(strict=True)): x["sha256"] for x in spec["sources"]}

    def pinned(item: dict) -> Path:
        raw = Path(item["path"])
        if raw.is_symlink():
            raise PermissionError("symlinked inference source refused")
        path = raw.resolve(strict=True)
        if source_index.get(str(path)) != item["sha256"] or sha(path) != item["sha256"]:
            raise PermissionError("inference source differs from queue pin")
        return path

    config_path = Path(config_path).resolve(strict=True)
    if source_index.get(str(config_path)) != sha(config_path):
        raise PermissionError("R5 recovery config unpinned")
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema") != "IVR1-r5-recovery-claim-replay-config-v1" or cfg.get("unit_id") != active["unit_id"]:
        raise PermissionError("R5 recovery unit identity differs")
    admission = pinned(cfg["admission_lock"])
    science = pinned(cfg["campaign_seal"])
    amendment = pinned(cfg["verification_amendment"])
    if (sha(admission) != active["seal_sha256"] or
            sha(admission) != state["seal"]["sha256"] or
            sha(science) not in {x["sha256"] for x in state["amendments"] if "protection" in x["stages"]}):
        raise PermissionError("R5 recovery dual-lock binding differs")
    latest = [x for x in state["amendments"] if "verification" in x["stages"]]
    if (not latest or latest[-1]["sha256"] != sha(amendment) or
            spec.get("amendment_sha256") != sha(amendment) or
            json.loads(amendment.read_text()).get("campaign_lock_sha256") != sha(science)):
        raise PermissionError("latest inference amendment differs")

    reported = json.loads(pinned(cfg["r3_recovery"]).read_text())
    utility = json.loads(pinned(cfg["r3_utility"]).read_text())
    claim_report = json.loads(pinned(cfg["r3_claims"]).read_text())
    nomination = json.loads(pinned(cfg["nomination"]).read_text())
    aggregate = json.loads(pinned(cfg["r4_aggregate"]).read_text())
    if (reported.get("schema") != "pcrl-incumbent-r3-recovery-inference-v1" or
            reported.get("K") != K or reported.get("recovery_slots") != 756 or
            utility.get("K") != K or utility.get("utility_slots") != 192 or
            aggregate.get("fixed_slots") != 40 or len(aggregate.get("units", [])) != 40 or
            claim_report.get("K") != K or claim_report.get("nominee") != nomination.get("nominee")):
        raise PermissionError("reported fixed inference family differs")
    if set(cfg["gates"]) != set(ARMS):
        raise PermissionError("fixed eight-arm gate family differs")
    gates = {}
    for arm in ARMS:
        gate = json.loads(pinned(cfg["gates"][arm]).read_text())
        if (gate.get("arm_id") != arm or
                gate.get("original_inner", {}).get("required") != 24):
            raise PermissionError("original point gate differs")
        gates[arm] = gate
    slot_map = {(x["arm_id"], x["source_seed"], x["realization"]): x
                for x in aggregate["units"]}
    expected_slots = {(arm, seed, draw)
                      for arm in ARMS for seed in (0, 1)
                      for draw in (range(3) if arm in STOCHASTIC else (0,))}
    if set(slot_map) != expected_slots:
        raise PermissionError("fixed 40 attack slots differ")
    eligible = set()
    for arm in ARMS:
        members = [slot_map[key] for key in expected_slots if key[0] == arm]
        if (gates[arm].get("all_two_seed_task_reference_point_caps_pass") is True and
                all(x.get("effective_status") == "COMPLETE" and
                    x.get("all_control_gates_pass") is True for x in members)):
            eligible.add(arm)
    if eligible != set(reported["eligible_arms"]):
        raise PermissionError("reported eligible arm set differs from full gates/attacks")
    for arm in ARMS:
        expected_status = ("COMPLETE_VALID_ATTACKS" if arm in eligible else
                           "POINT_INELIGIBLE" if gates[arm].get(
                               "all_two_seed_task_reference_point_caps_pass") is not True else
                           "INCOMPLETE_ATTACK" if any(
                               x.get("effective_status") != "COMPLETE" for
                               key, x in slot_map.items() if key[0] == arm) else
                           "CONTROL_FAILED")
        if reported["arm_status"][arm]["status"] != expected_status:
            raise PermissionError("reported fixed-arm inference status differs")
    expected_replays = {(arm, seed, draw) for arm in eligible for seed in (0, 1)
                        for draw in (range(3) if arm in STOCHASTIC else (0,))}
    entries = {}
    for item in cfg["r5_replays"]:
        key = (item["arm_id"], item["source_seed"], item["realization"])
        if key in entries or key not in expected_replays:
            raise PermissionError("R5 per-release replay inventory differs")
        receipt_path = pinned(item["receipt"])
        vector_path = pinned(item["vectors"])
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != "IVR1-r5-frozen-auc-replay-v1" or
                receipt.get("status") != "PASS" or
                (receipt.get("arm_id"), receipt.get("source_seed"), receipt.get("realization")) != key or
                receipt.get("private_auc_vectors_sha256") != sha(vector_path) or
                receipt.get("frozen_r4_receipt_sha256") != slot_map[key].get("receipt_sha256") or
                reported["R4_source_receipt_sha256"].get(f"{key[0]}:s{key[1]}:d{key[2]}") !=
                receipt.get("frozen_r4_receipt_sha256")):
            raise PermissionError("per-release frozen reader replay differs")
        entries[key] = (receipt, exact_npz(vector_path, set(VIEWS)))
    if set(entries) != expected_replays:
        raise PermissionError("all eligible release replays required")
    expected_auc_keys = {f"{arm}:s{seed}:d{draw}:{view}"
                         for arm, seed, draw in expected_replays for view in VIEWS}
    if set(reported["point_envelope_aucs"]) != expected_auc_keys:
        raise PermissionError("reported per-release point AUC inventory differs")
    # R1's source P0 wire contains the aligned INNER group ID but no SEX.
    inner_path = pinned(cfg["p0_inner_wire"])
    inner_receipt = json.loads(pinned(cfg["p0_inner_receipt"]).read_text())
    if (inner_receipt.get("rows") != 2235 or inner_receipt.get("groups") != 2234 or
            inner_receipt.get("release_sha256") != sha(inner_path) or
            inner_receipt.get("parity") != "ALL_ROWS_EXACT_HISTORICAL_CODE_PROB_DECISION_AND_METADATA_WIRE"):
        raise PermissionError("P0 aligned INNER source receipt differs")
    with ZipFile(inner_path) as archive:
        names = {x.filename for x in archive.infolist()}
        if names != {x + ".npy" for x in ("row_id", "unit", "code_r1", "prob_r1",
                                             "decision_r1", "code_r2", "prob_r2", "decision_r2")}:
            raise PermissionError("P0 INNER public wire members differ")
    with np.load(inner_path, allow_pickle=False) as archive:
        row = np.array(archive["row_id"], copy=True)
        unit = np.array(archive["unit"], copy=True)
    if row.dtype != np.int64 or row.shape != (2235,) or np.unique(row).size != 2235 or (
            unit.dtype != np.int64 or unit.shape != (2235,)):
        raise PermissionError("INNER group ID schema differs")
    pair = np.ascontiguousarray(np.column_stack((row, unit)))
    digest = hashlib.sha256()
    digest.update(str(pair.dtype).encode())
    digest.update(str(tuple(pair.shape)).encode())
    digest.update(pair.tobytes())
    if digest.hexdigest() != inner_receipt["row_unit_pair_sha256"]:
        raise PermissionError("P0 INNER row/group order differs from source receipt")
    _, _, draws = paired_plan(unit)
    values = {}
    for (arm, seed, draw), (receipt, vectors) in entries.items():
        for view in VIEWS:
            vector = vectors[view]
            if (vector.dtype != np.float64 or vector.shape != (BOOTSTRAPS + 1,) or
                    not np.isfinite(vector).all() or
                    abs(float(vector[0]) - receipt["point_envelope_auc"][view]) > 1e-11 or
                    abs(float(vector[0]) - reported["point_envelope_aucs"][
                        f"{arm}:s{seed}:d{draw}:{view}"]) > 1e-11):
                raise ArithmeticError("frozen reader point/vector differs")
            values[(arm, seed, draw, view)] = vector
    summaries = summarize_release_vectors(values, eligible, draws)
    independent = fixed_recovery_family(summaries, eligible)
    if set(independent) != set(reported["slots"]):
        raise ArithmeticError("fixed K948 recovery slot IDs differ")
    max_gap = 0.0
    disagreements = []
    for key, ours in independent.items():
        theirs = reported["slots"][key]
        if ours["status"] != theirs["status"]:
            disagreements.append(key)
        if ours["status"] == "BOUND_ONLY" and (
                ours["df"] != theirs.get("df") or ours["K"] != theirs.get("K")):
            disagreements.append(f"{key}:family")
        for field in ("point", "se", "lower", "upper"):
            a, b = ours[field], theirs[field]
            if (a is None) != (b is None):
                disagreements.append(f"{key}:{field}:null")
            elif a is not None:
                max_gap = max(max_gap, abs(float(a) - float(b)))
    if disagreements or max_gap > 1e-11:
        raise ArithmeticError(f"independent recovery intervals differ: {disagreements[:4]}, {max_gap}")
    nominee = nomination["nominee"]
    slate = [x for x in nomination["slate"] if x["arm_id"] == nominee]
    alias = bool(slate[0]["both_seed_exact_p0_alias"]) if slate else False
    claims = all_claims(nominee, gates, utility["slots"], independent,
                        nominee_exact_p0_alias=alias)
    claim_gaps = {letter: (claims[letter]["status"], claim_report["claims"][letter]["status"])
                  for letter in ("A", "B", "C") if
                  claims[letter]["status"] != claim_report["claims"][letter]["status"]}
    pair_means = {arm: float(summaries[("mean", arm, "pair")][0]) for arm in eligible}
    control = (None if nominee is None else strongest_control(nominee, gates, eligible, pair_means))
    if (claim_gaps or (control is not None and
                       control["selected"] != claim_report["strongest_control"]["selected"])):
        raise ArithmeticError("independent fixed claim/control accounting differs")
    output = Path(cfg["output"]).resolve()
    owner = Path(__file__).resolve().parents[1] / "results/pcrl_incumbent_view_risk_v1/r5"
    if output.parent != owner or output.exists() or output.is_symlink():
        raise PermissionError("R5 write-once recovery output differs")
    result = {"schema": "IVR1-r5-recovery-and-claim-replay-v1", "unit_id": active["unit_id"],
              "status": "PASS", "K": K, "recovery_slots_checked": len(independent),
              "eligible_arms": sorted(eligible, key=ARMS.index),
              "max_interval_numeric_gap": max_gap, "slot_status_disagreements": disagreements,
              "claim_statuses": {letter: claims[letter]["status"] for letter in ("A", "B", "C")},
              "strongest_control": None if control is None else control["selected"],
              "source_r3_recovery_sha256": sha(pinned(cfg["r3_recovery"])),
              "source_r3_claims_sha256": sha(pinned(cfg["r3_claims"])),
              "queue": active, "SEX_labels_read": False,
              "scope": "replay from 40 frozen-reader AUC vectors; no new fits", "not_confirmation": True}
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    result = run(Path(parser.parse_args().config))
    print(json.dumps({"status": result["status"], "K": result["K"]}, sort_keys=True))


if __name__ == "__main__":
    main()

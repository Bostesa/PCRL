"""Publish aggregate privacy readers and frozen matched controls after R3 replay.

This program opens only aggregate JSON. It has no role-export or per-person
array access and never chooses a comparator before the independent q21 replay.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from results.pcrl_capacity_matched_feature_selection_v1.m3.control_rules import (
    strongest_controls,
)


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
PRIVATE = STUDY / ".private"
RULE_SOURCE = (REPO / "results/pcrl_capacity_matched_feature_selection_v1"
               / "m3/control_rules.py")
READER_OUTPUT = STUDY / "PRIVACY_READERS.csv"
SELECTION_OUTPUT = STUDY / "MATCHED_PRIVACY_CONTROL_SELECTION.json"
VIEWS = ("income", "occupation_group", "coalition")
FIELDS = (
    "seed", "pair_id", "rule_ids", "view", "reader_id", "positive_auc",
    "reverse_auc", "fixed_alias_max_auc", "primary_envelope_auc",
    "validation_winner", "validation_winner_inner_auc", "controls_status",
    "score_sha256", "score_receipt_sha256", "r3_replay_sha256",
    "development_status",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    if path.suffix != ".json":
        raise PermissionError("aggregate builder reads JSON only")
    return json.loads(path.read_text())


def replay_admitted(replay: dict, queue: dict, replay_path: Path,
                    unit_id: str) -> None:
    if unit_id not in ("R3_PRIVACY_REPLAY", "R3_PRIVACY_REPLAY_RETRY"):
        raise PermissionError("independent replay unit identity differs")
    unit = queue.get("units", {}).get(unit_id, {})
    if (replay.get("schema") !=
            "pcrl-utility-r3-independent-privacy-replay-v1"
            or replay.get("status") != "REPLAYED"
            or replay.get("all_control_gates_pass") is not True
            or replay.get("max_abs_reported_metric_difference", 1) > 2e-7
            or unit.get("status") != "DONE"
            or unit.get("output_sha256", {}).get(str(replay_path)) !=
            sha(replay_path)):
        raise PermissionError("independent q21 privacy replay did not pass")


def _view_rows(seed: int, pair_id: str, view: str, rule_ids: list[str],
               score: dict, replay: dict, score_sha: str,
               receipt_sha: str, replay_sha: str) -> list[dict]:
    if (score.get("schema") != "pcrl-utility-complete-privacy-score-v1"
            or score.get("seed") != seed or score.get("pair_id") != pair_id
            or score.get("view_name") != view
            or score.get("controls", {}).get("status") != "PASS"
            or replay.get("controls_status") != "PASS"
            or len(score.get("reader_order", [])) !=
            len(score.get("readers", {}))
            or set(score.get("reader_order", [])) !=
            set(score.get("readers", {}))):
        raise PermissionError("privacy view aggregate inventory differs")
    if (abs(score["primary_envelope_auc"] -
            replay["primary_envelope_auc"]) > 2e-7 or
            score["validation_winner"] != replay["validation_winner"] or
            abs(score["validation_winner_inner_auc"] -
                replay["validation_winner_inner_auc"]) > 2e-7):
        raise PermissionError("independent privacy endpoint differs")
    if abs(max(item["fixed_alias_max_auc"] for item in
               score["readers"].values()) -
           score["primary_envelope_auc"]) > 1e-12:
        raise PermissionError("privacy reader envelope differs")
    rows = []
    for reader in score["reader_order"]:
        auc = score["readers"][reader]
        rows.append({
            "seed": seed, "pair_id": pair_id, "rule_ids": ",".join(rule_ids),
            "view": view, "reader_id": reader,
            "positive_auc": auc["positive_alias_auc"],
            "reverse_auc": auc["reverse_alias_auc"],
            "fixed_alias_max_auc": auc["fixed_alias_max_auc"],
            "primary_envelope_auc": score["primary_envelope_auc"],
            "validation_winner": score["validation_winner"],
            "validation_winner_inner_auc":
            score["validation_winner_inner_auc"],
            "controls_status": "PASS", "score_sha256": score_sha,
            "score_receipt_sha256": receipt_sha,
            "r3_replay_sha256": replay_sha,
            "development_status": "exploratory_reused_Adult_people",
        })
    return rows


def build(study: Path, replay_queue_path: Path, replay_path: Path,
          verification_path: Path, replay_lock_path: Path,
          expected_replay_sha256: str,
          unit_id: str) -> tuple[list[dict], dict]:
    """Use sealed JSON aggregates only; refuse incomplete independent replay."""
    study = study.resolve(strict=True)
    private = study / ".private"
    replay_path = replay_path.resolve(strict=True)
    queue_path = replay_queue_path.resolve(strict=True)
    verification_path = verification_path.resolve(strict=True)
    replay_lock_path = replay_lock_path.resolve(strict=True)
    if (not queue_path.is_relative_to(private.resolve(strict=True))
            or queue_path.name != "QUEUE.json"
            or not replay_path.is_relative_to(
                (private / "r3").resolve(strict=True))
            or replay_path.name not in ("PRIVACY_REPLAY.json",
                                        "PRIVACY_REPLAY_RETRY.json")
            or verification_path !=
            (study / "r3/PRIVACY_NUMERICAL_VERIFICATION.json").resolve()
            or not replay_lock_path.is_relative_to(
                (private / "stage_seals").resolve(strict=True))
            or replay_lock_path.name != "CAMPAIGN_LOCK.json"
            or len(expected_replay_sha256) != 64):
        raise PermissionError("replay queue or expected SHA identity differs")
    replay = read_json(replay_path)
    replay_admitted(replay, read_json(queue_path), replay_path, unit_id)
    replay_sha = sha(replay_path)
    if replay_sha != expected_replay_sha256:
        raise PermissionError("independent replay expected SHA differs")
    verification = read_json(verification_path)
    if (verification.get("schema") !=
            "pcrl-utility-r3-privacy-numerical-verification-v1"
            or verification.get("status") !=
            "PASS_INDEPENDENT_NUMERICAL_REPLAY"
            or verification.get("private_replay_sha256") !=
            replay_sha
            or verification.get("replay_queue_state_sha256") !=
            sha(queue_path)
            or verification.get("replay_source_lock_sha256") !=
            sha(replay_lock_path)
            or verification.get("all_control_gates_pass") is not True
            or verification.get("pair_view_count") != 12
            or verification.get("max_abs_reported_metric_difference", 1) >
            2e-7
            or verification.get("max_abs_core_model_reforward_difference", 1)
            > 2e-7):
        raise PermissionError("public R3 replay verification differs")
    release_verification_path = study / "r3/RELEASE_NUMERICAL_VERIFICATION.json"
    release_verification = read_json(release_verification_path)
    if (release_verification.get("status") !=
            "PASS_INDEPENDENT_COMPLETE_VIEW_REPLAY" or
            replay.get("independent_release_verification_sha256") !=
            sha(release_verification_path)):
        raise PermissionError("independent release view verification differs")
    plans = {}
    endpoints = {}
    rows = []
    sources = {"r3_replay": replay_sha, "r3_queue": sha(queue_path),
               "r3_replay_lock": sha(replay_lock_path),
               "r3_privacy_verification": sha(verification_path),
               "r3_release_verification": sha(release_verification_path),
               "frozen_control_rule_source": sha(RULE_SOURCE)}
    selected_pairs = {}
    for seed in (0, 1):
        plan_path = private / "pair_select" / f"seed{seed}" / "CONTROL_PLAN.json"
        score_dir = private / "privacy_score" / f"seed{seed}"
        score_path = score_dir / "PRIVACY_SCORES.json"
        receipt_path = score_dir / "PRIVACY_SCORE_RECEIPT.json"
        plan = read_json(plan_path)
        score = read_json(score_path)
        receipt = read_json(receipt_path)
        score_sha, receipt_sha = sha(score_path), sha(receipt_path)
        sources[f"seed{seed}_plan"] = sha(plan_path)
        sources[f"seed{seed}_score"] = score_sha
        sources[f"seed{seed}_score_receipt"] = receipt_sha
        if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
                or score.get("schema") != "pcrl-utility-seed-privacy-scores-v1"
                or score.get("seed") != seed
                or receipt.get("schema") !=
                "pcrl-utility-seed-privacy-score-receipt-v1"
                or receipt.get("status") != "SCORED"
                or receipt.get("seed") != seed
                or receipt.get("score_file_sha256") != score_sha):
            raise PermissionError("frozen seed privacy aggregate differs")
        pairs = sorted({rule["pair_id"] for rule in plan["rules"].values()
                        if rule["status"] == "SELECTED"})
        selected_pairs[str(seed)] = pairs
        if (sorted(score["pairs"]) != pairs or
                replay["selected_pairs_by_seed"][str(seed)] != pairs or
                sorted(replay["by_seed_pair_view"][str(seed)]) != pairs or
                score["selected_rule_pair_ids"] != {
                    key: rule["pair_id"] if rule["status"] == "SELECTED"
                    else None for key, rule in plan["rules"].items()}):
            raise PermissionError("selected pair list differs")
        plans[seed] = plan
        endpoints[seed] = {}
        gate_cache = {}
        for pair_id in pairs:
            rules = sorted(key for key, rule in plan["rules"].items()
                           if rule["status"] == "SELECTED" and
                           rule["pair_id"] == pair_id)
            if set(score["pairs"][pair_id]) != set(VIEWS):
                raise PermissionError("recipient and coalition views incomplete")
            for view in VIEWS:
                rows.extend(_view_rows(
                    seed, pair_id, view, rules,
                    score["pairs"][pair_id][view],
                    replay["by_seed_pair_view"][str(seed)][pair_id][view],
                    score_sha, receipt_sha, replay_sha))
        for rule_id, rule in plan["rules"].items():
            if rule["status"] != "SELECTED":
                continue
            task = {}
            for purpose, mask_id in (("income", rule["income_mask_id"]),
                                     ("occupation_group",
                                      rule["occupation_mask_id"])):
                if mask_id not in gate_cache:
                    gate_path = (private / "mask_gate" / mask_id /
                                 f"seed{seed}/MASK_GATE_RECEIPT.json")
                    gate_cache[mask_id] = read_json(gate_path)
                    sources[f"seed{seed}_mask_{mask_id}"] = sha(gate_path)
                gate = gate_cache[mask_id]
                if (gate.get("seed") != seed or gate.get("mask_id") != mask_id
                        or gate.get("schema") !=
                        "pcrl-utility-masked-common-task-gate-v1"):
                    raise PermissionError("mask utility gate identity differs")
                task[purpose] = {"references":
                    gate["point_eligibility"][purpose]["references"]}
            endpoints[seed][rule_id] = {
                "seed": seed, "rule_id": rule_id,
                "pair_id": rule["pair_id"],
                "income_mask_id": rule["income_mask_id"],
                "occupation_mask_id": rule["occupation_mask_id"],
                "union_sha256": rule["union_sha256"],
                "recipient_capacity_signatures":
                    rule["recipient_capacity_signatures"],
                "union_capacity_signature":
                    rule["union_capacity_signature"],
                "task": task,
                "recovery": {"coalition": {"primary_envelope_auc":
                    score["pairs"][rule["pair_id"]]["coalition"]
                    ["primary_envelope_auc"]}},
            }
    if verification.get("selected_pairs_by_seed") != selected_pairs:
        raise PermissionError("public independent selected pairs differ")
    result = strongest_controls(plans, endpoints)
    receipt = {
        "schema": "pcrl-utility-matched-privacy-control-selection-v1",
        "status": "INDEPENDENT_REPLAYED",
        "development_only": True,
        "replay_status": replay["status"],
        "independent_complete_release_verification_status":
            release_verification["status"],
        "selected_pairs_by_seed": selected_pairs,
        "reader_row_count": len(rows),
        "frozen_control_rule": "m3.control_rules.strongest_controls",
        "selection": result,
        "input_sha256": sources,
        "interpretation_limit": "selected development result on reused Adult people; not independent confirmation",
    }
    return rows, receipt


def _atomic_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-queue", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--verification-receipt", type=Path, required=True)
    parser.add_argument("--replay-lock", type=Path, required=True)
    parser.add_argument("--replay-unit-id", required=True)
    parser.add_argument("--expected-replay-sha256", required=True)
    args = parser.parse_args()
    rows, selection = build(STUDY, args.replay_queue, args.replay_output,
                            args.verification_receipt, args.replay_lock,
                            args.expected_replay_sha256,
                            args.replay_unit_id)
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = buffer.getvalue().encode()
    selection["reader_csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    json_bytes = (json.dumps(selection, sort_keys=True, indent=2,
                             allow_nan=False) + "\n").encode()
    if READER_OUTPUT.exists() or SELECTION_OUTPUT.exists():
        raise FileExistsError("aggregate privacy output already exists")
    _atomic_new_bytes(READER_OUTPUT, csv_bytes)
    _atomic_new_bytes(SELECTION_OUTPUT, json_bytes)
    print(json.dumps({"rows": len(rows), "reader_sha256":
                      sha(READER_OUTPUT), "selection_sha256":
                      sha(SELECTION_OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()

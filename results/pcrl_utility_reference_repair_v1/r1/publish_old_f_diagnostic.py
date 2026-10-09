"""Publish only aggregate old/new F task results after independent same-role replay.

This publisher reads queue states and JSON receipts. It never opens role archives,
person-level predictions, or any old assessment material.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
FIELDS = ("seed", "task", "reference", "old_fit_rows", "new_fit_rows",
          "old_accuracy", "new_accuracy", "reference_accuracy",
          "old_nll", "new_nll", "reference_nll",
          "old_brier", "new_brier", "reference_brier",
          "old_minus_new_accuracy", "old_minus_new_nll", "old_minus_new_brier",
          "old_accuracy_drop_vs_reference", "new_accuracy_drop_vs_reference",
          "old_nll_excess_vs_reference", "new_nll_excess_vs_reference",
          "old_brier_excess_vs_reference", "new_brier_excess_vs_reference")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check_queue(study: Path, generation: int,
                expected_outputs: dict[str, str]) -> tuple[dict, str]:
    path = study / f".private/queue_g{generation}/QUEUE.json"
    state = read_json(path)
    if set(state.get("units", {})) != set(expected_outputs):
        raise PermissionError(f"q{generation} units differ")
    for unit_id, output in expected_outputs.items():
        unit = state["units"][unit_id]
        if (unit.get("status") != "DONE"
                or unit.get("output_sha256") != {output: sha(Path(output))}):
            raise PermissionError(f"q{generation} output not frozen")
    return state, sha(path)


def check_archive(study: Path, key: str, queue: dict) -> dict:
    archive = study / ".private/stage_seals" / key
    lock = archive / "CAMPAIGN_LOCK.json"
    review = archive / "PREFIT_REVIEW.json"
    seal = archive / "LOCAL_SEAL_RECEIPT.json"
    registry = archive / "UNIT_SPECS.json"
    lock_json, review_json, seal_json = map(read_json, (lock, review, seal))
    if (review_json.get("status") != "PASS"
            or review_json.get("lock_sha256") != sha(lock)
            or seal_json.get("status") != "PASS"
            or seal_json.get("lock_sha256") != sha(lock)
            or seal_json.get("r3_review_sha256") != sha(review)
            or lock_json.get("private_unit_registry_sha256") != sha(registry)
            or queue.get("gate", {}).get("seal_sha256") != sha(seal)):
        raise PermissionError(f"{key} archive not independently sealed")
    return {"lock_sha256": sha(lock), "review_sha256": sha(review),
            "seal_sha256": sha(seal), "registry_sha256": sha(registry)}


def _close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def build_rows(old_receipts: dict[int, dict], new_receipts: dict[int, dict],
               replay: dict) -> list[dict]:
    if (replay.get("schema") != "pcrl-utility-r3-old-f-inner-replay-v1"
            or replay.get("status") != "OBSERVATIONAL_DIAGNOSTIC_REPLAYED"
            or set(replay.get("by_seed", {})) != {"0", "1"}):
        raise PermissionError("independent old-F replay absent")
    rows = []
    for seed in (0, 1):
        old = old_receipts[seed]
        new = new_receipts[seed]
        independent = replay["by_seed"][str(seed)]
        if (old.get("seed") != seed or new.get("seed") != seed
                or old.get("causal_attribution") != "NOT_SUPPORTED"
                or old.get("old_fit_rows") != {
                    task: (11574 if seed == 0 else 11575) for task in TASKS}
                or new.get("selected_recipe") != "A"
                or independent.get("max_abs_old_receipt_metric_difference", 1) > 1e-12):
            raise PermissionError("old/new F diagnostic identity differs")
        for task in TASKS:
            old_m = old["metrics"]["old_validation_F"][task]
            new_m = new["metrics"]["F"][task]
            check_old = independent["metrics"]["old_validation_F"][task]
            check_new = independent["metrics"]["new_all_FIT_F"][task]
            for metric in METRICS:
                if (not _close(old_m[metric], check_old[metric])
                        or not _close(new_m[metric], check_new[metric])
                        or not _close(old_m[metric] - new_m[metric],
                                      independent["old_minus_new"][task][metric])):
                    raise PermissionError("old/new F endpoint differs from replay")
            for reference in REFERENCES:
                ref_m = new["metrics"][reference][task]
                check_ref = independent["reference_metrics"][reference][task]
                for metric in METRICS:
                    if (not _close(ref_m[metric], check_ref[metric])
                            or not _close(old["metrics"][reference][task][metric],
                                          check_ref[metric])):
                        raise PermissionError("U0/Ucal reference differs from replay")
                row = {"seed": seed, "task": task, "reference": reference,
                       "old_fit_rows": old["old_fit_rows"][task],
                       "new_fit_rows": 15434}
                for metric in METRICS:
                    row[f"old_{metric}"] = old_m[metric]
                    row[f"new_{metric}"] = new_m[metric]
                    row[f"reference_{metric}"] = ref_m[metric]
                    row[f"old_minus_new_{metric}"] = old_m[metric] - new_m[metric]
                row["old_accuracy_drop_vs_reference"] = ref_m["accuracy"] - old_m["accuracy"]
                row["new_accuracy_drop_vs_reference"] = ref_m["accuracy"] - new_m["accuracy"]
                for metric in ("nll", "brier"):
                    row[f"old_{metric}_excess_vs_reference"] = old_m[metric] - ref_m[metric]
                    row[f"new_{metric}_excess_vs_reference"] = new_m[metric] - ref_m[metric]
                if set(row) != set(FIELDS):
                    raise AssertionError("public diagnostic columns differ")
                rows.append(row)
    return rows


def publish(study: Path) -> dict:
    study = study.resolve(strict=True)
    private = study / ".private"
    old_paths = {seed: private / f"r1/old_f_inner/seed_{seed}/DIAGNOSTIC_RECEIPT.json"
                 for seed in (0, 1)}
    new_paths = {seed: private / f"r1/gate/seed_{seed}/GATE_RECEIPT.json"
                 for seed in (0, 1)}
    replay_path = private / "r3/OLD_F_INNER_REPLAY.json"
    q23_outputs = {}
    for seed in (0, 1):
        old_receipt = read_json(old_paths[seed])
        prediction_path = private / f"r1/old_f_inner/seed_{seed}/OLD_F_INNER_PREDICTIONS.npz"
        if (old_receipt.get("predictions_relative_path") !=
                str(prediction_path.relative_to(private))):
            raise PermissionError("old prediction path differs")
        q23_outputs[f"R1_OLD_F_DIAGNOSTIC_S{seed}"] = {
            str(old_paths[seed]): sha(old_paths[seed]),
            str(prediction_path): old_receipt["predictions_sha256"]}
    # Person-level prediction bytes were checked by the sealed queues and R3.
    q23_path = private / "queue_g23/QUEUE.json"
    q23 = read_json(q23_path)
    if set(q23.get("units", {})) != set(q23_outputs):
        raise PermissionError("old-F queue units differ")
    for unit_id, output_hashes in q23_outputs.items():
        unit = q23["units"][unit_id]
        if unit.get("status") != "DONE" or unit.get("output_sha256") != output_hashes:
            raise PermissionError("old-F output not frozen")
    q23_sha = sha(q23_path)
    q24, q24_sha = check_queue(study, 24, {
        "R3_OLD_F_INNER_REPLAY": str(replay_path)})
    if q24.get("carry_forward") != {
            "archived_queue_relative": "queue_g23",
            "archived_queue_sha256": q23_sha,
            "charged_cpu_s": q23["used_cpu_s"], "unit_count": 2}:
        raise PermissionError("independent replay queue lineage differs")
    q23_archive = check_archive(study, "old_f_diagnostic", q23)
    q24_archive = check_archive(study, "old_f_replay", q24)
    old = {seed: read_json(path) for seed, path in old_paths.items()}
    new = {seed: read_json(path) for seed, path in new_paths.items()}
    replay = read_json(replay_path)
    public_replay_path = study / "r3/OLD_F_INNER_NUMERICAL_VERIFICATION.json"
    public_replay = read_json(public_replay_path)
    gate_path = study / "FULL_INPUT_GATE.json"
    if replay.get("primary_F_gate_sha256") != sha(gate_path):
        raise PermissionError("full-input F gate differs")
    if (public_replay.get("schema") !=
            "pcrl-utility-r3-old-f-inner-numerical-verification-v1"
            or public_replay.get("status") != "PASS_OBSERVATIONAL_NUMERICAL_REPLAY"
            or public_replay.get("input_sha256", {}) != {
                "q23_diagnostic_lock": q23_archive["lock_sha256"],
                "q23_diagnostic_queue": q23_sha,
                "q24_private_replay": sha(replay_path),
                "q24_replay_lock": q24_archive["lock_sha256"],
                "q24_replay_queue": q24_sha,
                "q24_replay_seal": q24_archive["seal_sha256"],
            }):
        raise PermissionError("public independent replay verification differs")
    for seed in (0, 1):
        if old[seed].get("primary_F_gate_sha256") != sha(gate_path):
            raise PermissionError("old F not linked to primary gate")
    rows = build_rows(old, new, replay)
    for row in rows:
        reported = public_replay["old_minus_new_F_task_metric_by_seed"][str(row["seed"])][row["task"]]
        for metric in METRICS:
            if not _close(row[f"old_minus_new_{metric}"], reported[metric]):
                raise PermissionError("public independent task contrast differs")
    csv_path = study / "OLD_F_SAME_INNER_DIAGNOSTIC.csv"
    summary_path = study / "r1/OLD_F_SAME_INNER_VERIFICATION.json"
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError("public diagnostic is write-once")
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "pcrl-utility-old-f-same-inner-public-verification-v1",
        "status": "INDEPENDENT_NUMERICAL_REPLAY_PASSED",
        "comparison_type": "old_validation_F_vs_new_all_FIT_F_same_INNER",
        "causal_attribution": "NOT_SUPPORTED_UNEQUAL_FIT_EXPOSURE",
        "scope": "observational task utility diagnostic on reused Adult INNER_SELECTION people",
        "old_fit_rows_by_seed": {str(seed): old[seed]["old_fit_rows"] for seed in (0, 1)},
        "new_fit_rows": 15434, "rows_per_seed": 2235,
        "q23_queue_sha256": q23_sha, "q24_queue_sha256": q24_sha,
        "q23_archive": q23_archive, "q24_archive": q24_archive,
        "primary_F_gate_sha256": sha(gate_path),
        "old_receipt_sha256_by_seed": {str(seed): sha(old_paths[seed]) for seed in (0, 1)},
        "new_receipt_sha256_by_seed": {str(seed): sha(new_paths[seed]) for seed in (0, 1)},
        "independent_replay_sha256": sha(replay_path),
        "public_independent_verification_sha256": sha(public_replay_path),
        "csv_sha256": sha(csv_path), "row_count": len(rows),
        "interpretation_limit": "matched INNER people; historical and new F differ in task-label fitting exposure and other pipeline details; no causal repair or privacy claim",
    }
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return {"csv_sha256": sha(csv_path), "summary_sha256": sha(summary_path),
            "rows": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.study), sort_keys=True))

"""Publish aggregate, replay-checked qualification of the two fixed leads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


LEADS = ("M_DROP_NATIVE_COUNTRY", "M_DROP_MARITAL_STATUS")
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def publish(study: Path) -> dict:
    private = study / ".private"
    output = study / "LEAD_BRANCH_RECEIPT.json"
    if output.exists():
        raise FileExistsError(output)
    queue_path = private / "queue_g8/QUEUE.json"
    state = json.loads(queue_path.read_text())
    unit = state.get("units", {}).get("R3_LEAD_MASK_TASK_REPLAY")
    if set(state.get("units", {})) != {"R3_LEAD_MASK_TASK_REPLAY"} or unit["status"] != "DONE":
        raise PermissionError("independent lead replay incomplete")
    replay_path = private / "r3/LEAD_MASK_TASK_REPLAY.json"
    if unit["output_sha256"].get(str(replay_path.resolve())) != sha(replay_path):
        raise PermissionError("independent replay output differs from queue")
    replay = json.loads(replay_path.read_text())
    gate_path = study / "FULL_INPUT_GATE.json"
    bank_path = study.parents[1] / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
    if (replay.get("schema") != "pcrl-utility-r3-masked-task-replay-v1"
            or replay.get("status") != "REPLAYED"
            or replay.get("full_input_gate_sha256") != sha(gate_path)
            or replay.get("mask_bank_sha256") != sha(bank_path)
            or set(replay.get("masks", [])) != set(LEADS)
            or replay.get("paired_seeds") != [0, 1]
            or replay.get("max_abs_restored_probability_difference", 1) > 2e-7
            or replay.get("max_abs_metric_difference", 1) > 1e-12):
        raise PermissionError("independent lead replay does not validate packages")
    rows = {}
    receipt_hashes = {}
    for seed in (0, 1):
        rows[str(seed)] = {}
        receipt_hashes[str(seed)] = {}
        for mask_id in LEADS:
            path = private / "mask_gate" / mask_id / f"seed{seed}" / "MASK_GATE_RECEIPT.json"
            receipt = json.loads(path.read_text())
            own = replay["by_seed_mask"][str(seed)][mask_id]
            if (receipt.get("mask_id") != mask_id or receipt.get("seed") != seed
                    or receipt.get("full_input_gate_sha256") != sha(gate_path)
                    or type(receipt.get("all_task_reference_points_eligible")) is not bool
                    or type(own.get("point_eligible")) is not bool
                    or receipt["all_task_reference_points_eligible"] is not
                    own["point_eligible"]):
                raise PermissionError("R2/R3 lead point identities differ")
            rows[str(seed)][mask_id] = {}
            receipt_hashes[str(seed)][mask_id] = sha(path)
            for task in TASKS:
                refs = receipt["point_eligibility"][task]["references"]
                if set(refs) != set(REFERENCES):
                    raise PermissionError("lead task reference coverage differs")
                rows[str(seed)][mask_id][task] = {"references": {}}
                for reference in REFERENCES:
                    item = refs[reference]
                    independent = own["task"][task]["point"][reference]
                    if (type(item.get("point_eligible")) is not bool
                            or item["point_eligible"] is not independent["point_eligible"]
                            or any(abs(item[key] - independent[key]) > 1e-12 for key in
                                   ("accuracy_loss", "nll_excess", "brier_excess"))):
                        raise PermissionError("R2/R3 lead task endpoint differs")
                    rows[str(seed)][mask_id][task]["references"][reference] = item
    qualified = [mask_id for mask_id in LEADS if all(
        rows[str(seed)][mask_id][task]["references"][reference]["point_eligible"]
        for seed in (0, 1) for task in TASKS for reference in REFERENCES)]
    result = {
        "schema": "pcrl-utility-lead-branch-v1",
        "status": "LEAD_QUALIFIED" if qualified else "NO_QUALIFIED_LEAD",
        "development_only": True,
        "selected_recipe": json.loads(gate_path.read_text())["selected_recipe"],
        "full_input_gate_sha256": sha(gate_path),
        "mask_bank_sha256": sha(bank_path),
        "independent_replay_sha256": sha(replay_path),
        "independent_replay_queue_state_sha256": sha(queue_path),
        "lead_gate_lock_sha256": sha(private / "stage_seals/lead_gate/CAMPAIGN_LOCK.json"),
        "lead_fit_lock_sha256": sha(private / "stage_seals/lead_fit/CAMPAIGN_LOCK.json"),
        "gate_receipt_sha256_by_seed_mask": receipt_hashes,
        "qualified_leads": qualified,
        "point_eligibility": rows,
        "uncertainty_status": "PENDING_SIMULTANEOUS_DEVELOPMENT_INTERVALS",
        "privacy_status": "NOT_RUN",
        "next_branch": "fixed remaining three-mask bank; no SEX until task-only table",
    }
    payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    study = Path(__file__).resolve().parents[1]
    result = publish(study)
    print(json.dumps({"status": result["status"],
                      "qualified_leads": result["qualified_leads"],
                      "output_sha256": sha(study / "LEAD_BRANCH_RECEIPT.json")},
                     sort_keys=True))

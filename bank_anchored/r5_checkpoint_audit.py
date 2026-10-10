"""Independent read-only audit of retained FIT checkpoint custody and rank.

Only public FIT receipts and the referenced private model-weight archive are
opened. No FIT task/SEX archive, protected runner, or selection function is
imported. This is a custody/ranking check, not endpoint verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "results/pcrl_bank_anchored_release_v1/r2"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def refs(value):
    if isinstance(value, dict):
        if set(value) == {"private_array_member", "array_sha256"}:
            yield value
        else:
            for item in value.values():
                yield from refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from refs(item)


def _start_index(name: str) -> int:
    return ("identity_biased", "uniform_logit").index(name)


def _rank(rows: list[dict], eta_keys: tuple[str, ...]) -> dict:
    if not rows:
        raise ValueError("projected checkpoint inventory is empty")
    return min(rows, key=lambda row: (float(row["objective"]),
                                     _start_index(row["start"]),
                                     int(row["update"]),
                                     *(float(row[key]) for key in eta_keys)))


def rank_selection(arm: str, summary: dict) -> dict:
    selected = summary.get("selection")
    if summary.get("status") == "INCOMPLETE_SEARCH":
        return {"status": "INCOMPLETE_SEARCH", "selected": False}
    if not isinstance(selected, dict):
        raise ValueError("completed arm has no selection")
    if arm in ("P", "S12", "S21"):
        inventory = summary["projected_checkpoints"]
        eta = ("eta1", "eta2") if arm == "P" else ("eta_second",)
        chosen = _rank(inventory, eta)
        for key in ("start", "update", "objective", *eta):
            if selected[key] != chosen[key]:
                raise ValueError("selected projected checkpoint is not frozen minimum")
        if arm in ("S12", "S21"):
            source = summary["source"]
            if (not source.get("stage1_complete") or
                    source["first_eta"] != selected["eta_first"] or
                    not isinstance(source.get("first_checkpoint_sha256"), str) or
                    not isinstance(source.get("draw_cache_sha256"), str)):
                raise ValueError("sequential first stage was not frozen")
        return {"status": "PASS", "projected_checkpoints": len(inventory),
                "selected_start": chosen["start"], "selected_update": chosen["update"]}
    if arm == "L":
        names = []
        counts = []
        updates = []
        for recipient in (1, 2):
            rows = summary["projected_checkpoints"][f"r{recipient}"]
            chosen = _rank(rows, ("eta",))
            chosen_selected = selected[f"recipient{recipient}"]
            for key in ("objective", "eta"):
                if chosen[key] != chosen_selected[key]:
                    raise ValueError("local selected checkpoint is not frozen minimum")
            names.append(chosen["start"])
            counts.append(len(rows))
            updates.append(chosen["update"])
        if (selected["start"] != "+".join(names) or
                selected["update"] != max(updates) or
                selected["eta1"] != selected["recipient1"]["eta"] or
                selected["eta2"] != selected["recipient2"]["eta"]):
            raise ValueError("local selected factor identity differs")
        return {"status": "PASS", "projected_checkpoints": counts,
                "selected_start": selected["start"],
                "selected_update": selected["update"]}
    if arm == "D":
        starts = summary["source"]["starts"]
        matching = [start for start in starts
                    if start["start"] == selected["start"] and
                    start["chosen_checkpoint_update"] == selected["update"] and
                    start["terminal_metrics"] == selected["terminal_metrics"]]
        if len(matching) != 1:
            raise ValueError("deterministic selected checkpoint absent from starts")
        checkpoints = summary["_checkpoint_models"]
        if not any(item[0] == starts.index(matching[0]) and
                   item[1] == selected["update"]
                   for item in checkpoints):
            raise ValueError("deterministic selected checkpoint weights absent")
        return {"status": "PASS", "retained_checkpoints": len(checkpoints),
                "selected_start": selected["start"],
                "selected_update": selected["update"]}
    if arm in ("J", "T"):
        matches = [cp for start in summary["source"]["starts"]
                   if start["name"] == selected["start"]
                   for cp in start["checkpoints"]
                   if cp["update"] == selected["update"]]
        if len(matches) != 1 or not list(refs(matches[0].get("weights", {}))):
            raise ValueError("central selected checkpoint weights absent")
        return {"status": "PASS", "retained_checkpoints": sum(
            len(start["checkpoints"]) for start in summary["source"]["starts"]),
            "selected_start": selected["start"],
            "selected_update": selected["update"]}
    if arm == "B0":
        if selected.get("identity_alias") is not True:
            raise ValueError("B0 exact identity selection differs")
        return {"status": "PASS", "retained_checkpoints": 0,
                "selected_start": "exact_identity", "selected_update": 0}
    raise ValueError("unregistered FIT arm")


def audit(unit_id: str) -> dict:
    if not unit_id.startswith("R2_") or not unit_id.endswith(("_FIT_G1", "_FIT_F1")):
        raise ValueError("explicit admitted F1/G1 unit ID required")
    receipt_path = R2 / f"{unit_id}_FIT_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("unit_id") != unit_id or receipt.get("schema") != "BAR1-r2-fit-receipt-v1":
        raise ValueError("R2 FIT receipt identity differs")
    archive_path = R2 / ".private" / "units" / unit_id / "checkpoints" / "CHECKPOINTS.npz"
    inventory = list(refs(receipt["fit_summary"]))
    if (sha(archive_path) != receipt["checkpoint"]["checkpoint_archive_sha256"] or
            len(inventory) != receipt["checkpoint"]["checkpoint_array_count"] or
            len({item["private_array_member"] for item in inventory}) != len(inventory)):
        raise ValueError("checkpoint archive count/byte custody differs")
    with ZipFile(archive_path) as source:
        if {item.filename for item in source.infolist()} != {
                item["private_array_member"] + ".npy" for item in inventory}:
            raise ValueError("checkpoint archive member inventory differs")
    with np.load(archive_path, allow_pickle=False) as source:
        for item in inventory:
            value = np.ascontiguousarray(source[item["private_array_member"]])
            digest = hashlib.sha256((str(value.dtype) + str(value.shape)).encode() +
                                    value.tobytes()).hexdigest()
            if digest != item["array_sha256"]:
                raise ValueError("checkpoint array hash differs")
    ranking = rank_selection(receipt["arm_id"], receipt["fit_summary"])
    return {"unit_id": unit_id, "arm_id": receipt["arm_id"],
            "seed": receipt["source_seed"],
            "fit_receipt_sha256": sha(receipt_path),
            "checkpoint_archive_sha256": sha(archive_path),
            "arrays": len(inventory), "selection": ranking}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("unit_id", nargs="+")
    args = parser.parse_args()
    records = [audit(unit_id) for unit_id in args.unit_id]
    print(json.dumps({"schema": "BAR1-r5-checkpoint-static-audit-v1",
                      "status": "PASS", "units": records},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

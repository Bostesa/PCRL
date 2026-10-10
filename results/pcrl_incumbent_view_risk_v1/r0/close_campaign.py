"""Close the owned queue and export aggregate resource/custody metadata."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from .compute_queue import Queue, START, atomic


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    q = Queue()
    state = q.read()
    terminal = {"done", "failed", "incomplete", "superseded_not_run"}
    assert all(u["status"] in terminal for u in state["units"].values())
    required = ["R5_INDEPENDENT_SNAPSHOT_REPLAY_V1", "R1_CUSTODY_RESTORE_V3_Q",
                "R1_CUSTODY_SNAPSHOT_V3_Q"]
    required += [u["unit_id"] for u in json.loads(
        (root / "r1/RESTORED_REPLAY_SPEC_INDEX_V1.json").read_text())["units"]]
    assert all(state["units"][uid]["status"] == "done" for uid in required)
    manifest = root / "INDEPENDENT_VERIFICATION.json"
    assert manifest.read_bytes() == (root / "r5/INDEPENDENT_VERIFICATION.json").read_bytes()
    for slot in (0, 1):
        with (q.root / f"slot{slot}.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
    q.mark("verification_complete", manifest)
    q.close("Registered incumbent objective/support comparison complete; no post-result sweep")
    state = q.read()
    events = [json.loads(line) for line in q.events.read_text().splitlines()]
    running: set[str] = set()
    peak = 0
    for event in events:
        if event["event"] == "start":
            running.add(event["unit_id"])
            peak = max(peak, len(running))
        elif event["event"] == "finish":
            running.discard(event["unit_id"])
    assert not running
    finishes = [event for event in events if event["event"] == "finish"]
    private = root / "r0/.private/closeout/FINAL_QUEUE_STATE_20261010"
    private.mkdir(parents=True, exist_ok=False)
    private.chmod(0o700)
    files = {}
    for source in (q.state, q.events, q.root / "host_occupancy.json"):
        target = private / source.name
        shutil.copyfile(source, target)
        target.chmod(0o600)
        assert digest(source) == digest(target)
        files[target.name] = {"sha256": digest(target), "bytes": target.stat().st_size}
    conservative = sum(float(x["cpu_s"]) for x in state["charges"].values())
    receipt = {
        "schema": "IVR1-owned-queue-final-closeout-v1",
        "status": "CLOSED_NO_OWNED_HEAVY_CHILD",
        "closed_utc": state["closed_utc"],
        "epoch_start_utc": state["epoch_start_utc"],
        "deadline_utc": state["deadline_utc"],
        "elapsed_s": (datetime.now(timezone.utc) - START).total_seconds(),
        "elapsed_limit_s": 36000,
        "aggregate_cpu_budget_s": state["cpu_limit_s"],
        "aggregate_charge_s": state["used_cpu_s"],
        "conservative_unmetered_engineering_charge_s": conservative,
        "queued_resource_charge_s": state["used_cpu_s"] - conservative,
        "charge_semantics": "Queued charge=max(wall,direct child CPU,sampled descendant CPU); engineering allowance is conservative, not measured CPU",
        "charge_by_stage_s": state["used_by_stage_s"],
        "registered_units": len(state["units"]),
        "status_counts": dict(Counter(u["status"] for u in state["units"].values())),
        "peak_campaign_queue_concurrency": peak,
        "host_heavy_ceiling": 2,
        "campaign_reserved_heavy_slots": 1,
        "external_reserved_heavy_slots": 1,
        "resource_limited_attempts": [
            {"unit_id": e["unit_id"], "resource_limit": e["resource_limit"],
             "status": e["status"], "max_observed_cpu_rate": e["max_observed_cpu_rate"]}
            for e in finishes if e.get("resource_limit")],
        "maximum_sampled_tree_rss_bytes": max(e["sampled_tree_rss_peak_bytes"] for e in finishes),
        "numerical_threads": 1,
        "git_packing_threads_after_Q": 1,
        "both_owned_slot_locks_free": True,
        "active_queue_children": [],
        "owned_watchdogs_remaining": 0,
        "other_session_processes_stopped": False,
        "cloud_spend": 0,
        "disk_free_bytes_at_closeout": shutil.disk_usage(root).free,
        "private_state_copy_label": str(private.relative_to(root)),
        "private_state_copy_files": files,
        "independent_manifest_sha256": digest(manifest),
    }
    atomic(root / "r0/RESOURCE_FINAL_CLOSEOUT.json", receipt)
    print(json.dumps({key: receipt[key] for key in (
        "status", "registered_units", "status_counts", "aggregate_charge_s",
        "elapsed_s", "peak_campaign_queue_concurrency")}))


if __name__ == "__main__":
    main()

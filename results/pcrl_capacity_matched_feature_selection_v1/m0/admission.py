"""Fail-closed worker-side validation before any admitted role is opened."""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

from .queue import FocusedQueue, sha256


def assert_running_unit(unit_id: str, queue_root: str | Path,
                        lock_path: str | Path, expected_stage: str,
                        expected_output: str | Path) -> dict:
    """Return sealed unit spec only for this live worker process and exact output."""
    root = Path(queue_root).resolve(strict=True)
    lock = Path(lock_path).resolve(strict=True)
    output = Path(expected_output).resolve()
    state_path = root / "QUEUE.json"
    # Popen starts the worker before its parent can publish child_pid. No role
    # read occurs during this bounded wait; every other mismatch fails closed.
    for attempt in range(30):
        state = json.loads(state_path.read_text())
        rec = state.get("units", {}).get(unit_id)
        if isinstance(rec, dict) and rec.get("child_pid") == os.getpid():
            break
        if not isinstance(rec, dict) or rec.get("status") != "RUNNING":
            raise RuntimeError("unit is not running")
        if rec.get("child_pid") is not None:
            raise RuntimeError("worker PID does not match queue child")
        if attempt < 29:
            time.sleep(0.1)
    if not isinstance(rec, dict) or rec.get("status") != "RUNNING":
        raise RuntimeError("unit is not running")
    if rec.get("child_pid") != os.getpid():
        raise RuntimeError("worker PID does not match queue child")
    if (rec.get("child_pgid") != os.getpgid(0) or
            resource.getrlimit(resource.RLIMIT_NPROC) != (0, 0)):
        raise RuntimeError("worker process containment invalid")
    spec_path = Path(rec["spec_path"]).resolve(strict=True)
    if (not spec_path.is_relative_to(root / "specs") or
            sha256(spec_path) != rec["spec_sha256"]):
        raise RuntimeError("unit spec identity changed")
    spec = json.loads(spec_path.read_text())
    if (spec.get("unit_id") != unit_id or spec.get("mode") != "real" or
            spec.get("stage") != expected_stage or
            str(output) not in spec.get("outputs", [])):
        raise RuntimeError("unit stage/output identity invalid")
    seal_path = (root.parent.parent / "LOCAL_SEAL_RECEIPT.json").resolve(strict=True)
    gate = state.get("gate", {})
    if (not gate.get("science") or Path(gate.get("seal_path", "")).resolve() != seal_path or
            spec.get("seal_sha256") != gate.get("seal_sha256") or
            not FocusedQueue._seal_ok(seal_path, gate["seal_sha256"])):
        raise RuntimeError("local science seal invalid")
    if (lock != root.parent.parent / "CAMPAIGN_LOCK.json" or
            sha256(lock) != json.loads(seal_path.read_text()).get("lock_sha256")):
        raise RuntimeError("lock identity invalid")
    queue = FocusedQueue(root, state["global_slot"],
                         epoch_start_utc=state["epoch_start_utc"])
    queue._validate_spec(spec, state)
    if not queue._inputs_ok(spec, state):
        raise RuntimeError("role/dependency input changed")
    return spec

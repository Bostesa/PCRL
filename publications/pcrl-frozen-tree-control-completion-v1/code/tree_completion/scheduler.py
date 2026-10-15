"""Single durable heavy-job queue for the frozen-tree control-completion campaign.

All participating roles must invoke this same absolute file. No science unit
starts until R00 supplies a local effective protocol seal. Runtime state is
private and shared across roles, not copied into individual worktrees.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # Resource usage still works without optional process sampling.
    psutil = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / ".private" / "tree_completion" / "compute"
QUEUE_SCHEMA = "pcrl-frozen-tree-control-completion-queue-v1"


def _start() -> dt.datetime:
    """Recorded queue start; written once by `init` and immutable afterwards."""
    path = ROOT / "START_UTC.json"
    if not path.is_file():
        return dt.datetime.max.replace(tzinfo=dt.timezone.utc) - dt.timedelta(days=2)
    return dt.datetime.fromisoformat(json.loads(path.read_text())["start_utc"])


START = _start()
DEADLINE = START + dt.timedelta(hours=8)
CPU_LIMIT = 12 * 3600
WITNESS_RESERVE = 0 * 3600
PROTECTION_RESERVE = 2 * 3600
VERIFY_RESERVE = 2 * 3600
CHANNEL_LIMIT = 7 * 3600
DISK_FLOOR = 15 * 1024**3
STAGES = {"admission", "witness", "protection", "attacks", "verification",
          "channel", "custody"}
PRIORITY = {"witness": 0, "verification": 1, "protection": 2,
            "attacks": 3, "admission": 4, "channel": 5, "custody": 6}
THREAD_ENV = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "TORCH_NUM_THREADS",
)}


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp() -> str:
    return now().isoformat(timespec="seconds")


def digest(path: Path) -> str:
    from tree_completion.admission import inner_truth_path, admitted_path
    if inner_truth_path(path):
        from tree_completion.admission import REPORTING_LOCK, reporting_admitted_path
        if not REPORTING_LOCK.is_file():
            raise PermissionError('INNER truth hashing barred before committed reporting lock')
        path = reporting_admitted_path(path)
    else:
        path = admitted_path(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(obj, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def group_pids(pgid: int) -> tuple[set[int], bool]:
    """Enumerate only our private process group when sandbox blocks ps/psutil.pids."""
    try:
        result = subprocess.run(["lsof", "-nP", "-a", f"-g{pgid}", "-d", "cwd", "-F", "pg"],
                                capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {pgid}, False
    if result.returncode not in (0, 1):
        return {pgid}, False
    pids = {int(line[1:]) for line in result.stdout.splitlines()
            if line.startswith("p") and line[1:].isdigit()}
    return pids | {pgid}, True


def assert_active_from_env(unit_id: str | None = None, stage: str | None = None) -> dict:
    """Fail closed when a protected worker is not running under its queue slot.

    Call in the worker process before protected data access and each fit. The
    wrapper supplies the environment and inherited locked FD; direct launches
    do not satisfy this check. A short wait covers the child-start state write.
    """
    required = ("PCRL_QUEUE_ROOT", "PCRL_QUEUE_UNIT_ID", "PCRL_QUEUE_STAGE",
                "PCRL_QUEUE_SLOT", "PCRL_QUEUE_SLOT_FD", "PCRL_QUEUE_SEAL_SHA256",
                "PCRL_QUEUE_SPEC_SHA256")
    if any(not os.environ.get(key) for key in required):
        raise RuntimeError("active queue slot environment missing")
    root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve()
    uid = os.environ["PCRL_QUEUE_UNIT_ID"]
    actual_stage = os.environ["PCRL_QUEUE_STAGE"]
    slot = int(os.environ["PCRL_QUEUE_SLOT"])
    fd = int(os.environ["PCRL_QUEUE_SLOT_FD"])
    if unit_id is not None and uid != unit_id:
        raise RuntimeError("active queue unit ID mismatch")
    if stage is not None and actual_stage != stage:
        raise RuntimeError("active queue stage mismatch")
    held = os.fstat(fd)
    lock_path = root / f"slot{slot}.lock"
    path_stat = lock_path.stat()
    if (held.st_dev, held.st_ino) != (path_stat.st_dev, path_stat.st_ino):
        raise RuntimeError("inherited slot FD does not identify queue slot")
    probe = lock_path.open("a+")
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeError("queue slot lock is not held")
    finally:
        probe.close()
    deadline = time.monotonic() + 3
    while True:
        state = json.loads((root / "state.json").read_text())
        unit = state.get("units", {}).get(uid)
        seal = state.get("seal")
        if (unit and unit.get("child_pid") == os.getpgid(0)
                and unit.get("status") == "running"):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("unit is not recorded running for owned process group")
        time.sleep(0.05)
    if (unit["slot"] != slot or unit["spec"]["stage"] != actual_stage
            or unit["spec_sha256"] != os.environ["PCRL_QUEUE_SPEC_SHA256"]):
        raise RuntimeError("active queue state does not match worker environment")
    if state.get("schema") != QUEUE_SCHEMA:
        raise RuntimeError("wrong campaign queue capability")
    if (state["closed"] or not seal or seal["sha256"] != os.environ["PCRL_QUEUE_SEAL_SHA256"]
            or digest(Path(seal["path"])) != seal["sha256"]):
        raise RuntimeError("campaign closed or effective seal changed")
    return {"unit_id": uid, "stage": actual_stage, "slot": slot,
            "child_pgid": unit["child_pid"], "seal_sha256": seal["sha256"],
            "spec_sha256": unit["spec_sha256"]}


class Queue:
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / "state.lock"
        self.state = self.root / "state.json"
        self.events = self.root / "events.jsonl"
        with self.mutex():
            if not self.state.exists():
                atomic(self.state, {
                    "schema": QUEUE_SCHEMA,
                    "epoch_start_utc": START.isoformat(),
                    "deadline_utc": DEADLINE.isoformat(),
                    "cpu_limit_s": CPU_LIMIT,
                    "used_cpu_s": 0.0,
                    "used_by_stage_s": {stage: 0.0 for stage in STAGES},
                    "seal": None,
                    "amendments": [],
                    "occupancy": None,
                    "milestones": {},
                    "closed": False,
                    "charges": {},
                    "units": {},
                })

    def mutex(self):
        class Held:
            def __init__(self, path): self.path = path
            def __enter__(self):
                self.handle = self.path.open("a+")
                fcntl.flock(self.handle, fcntl.LOCK_EX)
                return self.handle
            def __exit__(self, *_):
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
        return Held(self.lock)

    def read(self) -> dict:
        return json.loads(self.state.read_text())

    def event(self, kind: str, **fields) -> None:
        with self.events.open("a") as stream:
            stream.write(canonical({"at_utc": stamp(), "event": kind, **fields}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def open(self, seal: Path, sha256: str) -> dict:
        seal = seal.resolve()
        if not seal.is_file() or digest(seal) != sha256:
            raise ValueError("effective protocol seal missing or SHA-256 mismatch")
        with self.mutex():
            state = self.read()
            value = {"path": str(seal), "sha256": sha256, "opened_utc": stamp()}
            if state["seal"] and (state["seal"]["path"], state["seal"]["sha256"]) != (value["path"], value["sha256"]):
                raise ValueError("seal is immutable; record a dated amendment separately")
            if state["seal"] is None:
                state["seal"] = value
                atomic(self.state, state)
                self.event("open", seal=value)
            return state["seal"]

    def occupancy(self, evidence: Path, external_heavy: int) -> dict:
        if external_heavy not in (0, 1, 2):
            raise ValueError("external_heavy must be 0, 1, or 2")
        evidence = evidence.resolve()
        if not evidence.is_file():
            raise ValueError("occupancy evidence missing")
        with self.mutex():
            state = self.read()
            value = {"path": str(evidence), "sha256": digest(evidence),
                     "checked_utc": stamp(), "external_heavy": external_heavy}
            state["occupancy"] = value
            atomic(self.state, state)
            self.event("occupancy", **value)
            return value

    def amend(self, path: Path, sha256: str, dated_utc: str, stages_csv: str) -> dict:
        """Append a dated amendment without replacing the original protocol seal."""
        path = path.resolve()
        if not path.is_file() or digest(path) != sha256:
            raise ValueError("amendment file missing or SHA-256 mismatch")
        try:
            dated = dt.datetime.fromisoformat(dated_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("dated_utc must be ISO 8601 with timezone") from exc
        if dated.tzinfo is None or dated > now():
            raise ValueError("amendment date must be timezone-aware and not future")
        stages = sorted(set(stages_csv.split(",")))
        if not stages or not set(stages) <= STAGES:
            raise ValueError("affected stages must be valid queue stages")
        with self.mutex():
            state = self.read()
            if state["seal"] is None or digest(Path(state["seal"]["path"])) != state["seal"]["sha256"]:
                raise ValueError("original seal missing or changed")
            if any(u["status"] == "running" for u in state["units"].values()):
                raise RuntimeError("register amendment between jobs, before affected fit")
            amendments = state.setdefault("amendments", [])
            for prior in amendments:
                if prior["sha256"] == sha256:
                    if prior["path"] != str(path) or prior["stages"] != stages:
                        raise ValueError("amendment SHA already registered differently")
                    return prior
            value = {"path": str(path), "sha256": sha256, "dated_utc": dated_utc,
                     "stages": stages, "registered_utc": stamp(), "sequence": len(amendments) + 1}
            amendments.append(value)
            atomic(self.state, state)
            self.event("amend", **value)
            return value

    def mark(self, milestone: str, receipt: Path) -> dict:
        if milestone not in ("witness_pass", "witness_stop", "protection_complete",
                             "verification_complete", "channel_identity_pass"):
            raise ValueError("unknown milestone")
        receipt = receipt.resolve()
        if not receipt.is_file():
            raise ValueError("milestone receipt missing")
        if milestone in ("witness_pass", "channel_identity_pass"):
            payload = json.loads(receipt.read_text())
            if milestone == "witness_pass" and not (
                payload.get("schema") == "pcrl-incumbent-admission-gate-v1"
                and payload.get("status") == "P0_IDENTITY_AND_POINT_GATE_PASS"
                and payload.get("selected_bank") == "P0"
                and payload.get("all_two_seed_task_reference_point_caps_pass") is True
                and payload.get("source_lock_sha256") == self.read()["seal"]["sha256"]
            ):
                raise ValueError("selected bank identity receipt does not establish gate")
            if milestone == "channel_identity_pass" and not (
                payload.get("schema") == "pcrl-incumbent-channel-identity-gate-v1"
                and payload.get("exact_identity") == "PASS"
                and payload.get("bank_relative_proxy") == "PASS"
                and payload.get("original_inner_point_caps") == "PASS"
                and payload.get("constraint_semantics") == "ESTABLISHED"
                and payload.get("source_lock_sha256") == self.read()["seal"]["sha256"]
            ):
                raise ValueError("bank-relative channel identity receipt does not establish gate")
        with self.mutex():
            state = self.read()
            value = {"path": str(receipt), "sha256": digest(receipt), "at_utc": stamp()}
            prior = state["milestones"].get(milestone)
            if prior and prior["sha256"] != value["sha256"]:
                raise ValueError("milestone identity is immutable")
            if not prior:
                state["milestones"][milestone] = value
                atomic(self.state, state)
                self.event("milestone", name=milestone, **value)
            return state["milestones"][milestone]

    def enqueue(self, path: Path) -> dict:
        path = path.resolve()
        spec = json.loads(path.read_text())
        required = ("unit_id", "stage", "argv", "cwd", "max_wall_s", "cpu_reserve_s", "max_rss_gib", "projected_output_bytes")
        if any(key not in spec for key in required):
            raise ValueError(f"required unit fields: {required}")
        uid = spec["unit_id"]
        if not isinstance(uid, str) or not uid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in uid):
            raise ValueError("unit_id must use letters, digits, underscore or hyphen")
        if spec["stage"] not in STAGES:
            raise ValueError("invalid stage")
        if not isinstance(spec["argv"], list) or not spec["argv"] or not all(isinstance(x, str) for x in spec["argv"]):
            raise ValueError("argv must be a nonempty string array; no shell")
        if not Path(spec["cwd"]).is_dir():
            raise ValueError("cwd missing")
        if not (0 < float(spec["max_wall_s"]) <= 36000 and float(spec["cpu_reserve_s"]) >= float(spec["max_wall_s"])):
            raise ValueError("cpu reserve must cover positive wall cap within campaign")
        if not 0 < float(spec["max_rss_gib"]) <= 4:
            raise ValueError("RSS cap must be in (0,4] GiB")
        if not isinstance(spec["projected_output_bytes"], int) or spec["projected_output_bytes"] < 0:
            raise ValueError("projected_output_bytes must be a nonnegative integer")
        if not isinstance(spec.get("deps", []), list) or not isinstance(spec.get("outputs", []), list):
            raise ValueError("deps and outputs must be arrays")
        if any(not isinstance(item, str) for item in spec.get("deps", []) + spec.get("outputs", [])):
            raise ValueError("deps and outputs must contain only strings")
        spec.setdefault("deps", [])
        spec.setdefault("outputs", [])
        spec.setdefault("env", {})
        spec.setdefault("sources", [])
        if not isinstance(spec["env"], dict) or any(k in THREAD_ENV and str(v) != "1" for k, v in spec["env"].items()):
            raise ValueError("custom environment cannot override one-thread limits")
        if not isinstance(spec["sources"], list) or not spec["sources"] or any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in spec["sources"]):
            raise ValueError("sources must contain at least one path/SHA-256 record")
        if any(not isinstance(item["path"], str) or not isinstance(item["sha256"], str)
               or len(item["sha256"]) != 64 for item in spec["sources"]):
            raise ValueError("source path or SHA-256 malformed")
        with self.mutex():
            state = self.read()
            existing = state["units"].get(uid)
            if existing:
                if existing["spec"] != spec:
                    raise ValueError("unit ID already has an immutable different spec")
                return existing
            state["units"][uid] = {"status": "queued", "spec": spec,
                                   "spec_sha256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
                                   "submitted_utc": stamp(), "source_spec_sha256": digest(path)}
            atomic(self.state, state)
            self.event("enqueue", unit_id=uid, stage=spec["stage"], spec_sha256=state["units"][uid]["spec_sha256"])
            return state["units"][uid]

    def _capacity(self, state: dict) -> int:
        receipt = state["occupancy"]
        if receipt is None:
            return 0  # No host capacity claimed without a fresh evidence receipt.
        try:
            checked = dt.datetime.fromisoformat(receipt["checked_utc"])
            if (now() - checked).total_seconds() > 120 or digest(Path(receipt["path"])) != receipt["sha256"]:
                return 0
        except (OSError, ValueError, KeyError):
            return 0
        return min(1, max(0, 2 - receipt["external_heavy"]))

    def _admit_reason(self, state: dict, spec: dict) -> str | None:
        if state["closed"]:
            return "closed"
        if state["seal"] is None:
            return "protocol_seal_missing"
        try:
            seal_matches = digest(Path(state["seal"]["path"])) == state["seal"]["sha256"]
        except OSError:
            seal_matches = False
        if not seal_matches:
            return "protocol_seal_changed"
        latest_amendment = None
        for amendment in state.get("amendments", []):
            try:
                if digest(Path(amendment["path"])) != amendment["sha256"]:
                    return "amendment_changed"
            except OSError:
                return "amendment_missing"
            if spec["stage"] in amendment["stages"]:
                latest_amendment = amendment["sha256"]
        if latest_amendment and spec.get("amendment_sha256") != latest_amendment:
            return "affected_stage_requires_latest_amendment_sha256"
        if now() >= DEADLINE or now() + dt.timedelta(seconds=float(spec["max_wall_s"])) > DEADLINE:
            return "wall_deadline"
        projected = int(spec["projected_output_bytes"])
        running_outputs = sum(int(u["spec"]["projected_output_bytes"]) for u in state["units"].values() if u["status"] == "running")
        if shutil.disk_usage(self.root).free - projected - running_outputs < DISK_FLOOR:
            return "disk_floor"
        for source in spec["sources"]:
            try:
                if digest(Path(source["path"])) != source["sha256"]:
                    return "source_hash_changed"
            except OSError:
                return "source_hash_missing"
        stage = spec["stage"]
        if stage in ("witness", "protection", "channel", "attacks"):
            needed = "attacks" if stage == "attacks" else "science"
            if not any(needed in Path(a["path"]).name.lower() for a in state.get("amendments", [])):
                return needed + "_stage_lock_missing"
        if "witness_pass" in state["milestones"] and "witness_stop" in state["milestones"]:
            return "contradictory_witness_milestones"
        for milestone in state["milestones"].values():
            try:
                if digest(Path(milestone["path"])) != milestone["sha256"]:
                    return "milestone_receipt_changed"
            except OSError:
                return "milestone_receipt_missing"
        units = state["units"]
        if any(units.get(dep, {}).get("status") != "done" for dep in spec["deps"]):
            return "dependencies"
        running = [u for u in units.values() if u["status"] == "running"]
        reserve = float(spec["cpu_reserve_s"])
        active_reserve = sum(float(u["spec"]["cpu_reserve_s"]) for u in running)
        used = float(state["used_cpu_s"])
        def remaining_reserve(stage_names: tuple[str, ...], floor: float, released: bool) -> float:
            if released:
                return 0.0
            stage_running = sum(float(u["spec"]["cpu_reserve_s"]) for u in running
                                if u["spec"]["stage"] in stage_names)
            used_stage = sum(float(state["used_by_stage_s"][name]) for name in stage_names)
            return max(0.0, floor - used_stage - stage_running)
        future_witness = remaining_reserve(("witness",), WITNESS_RESERVE,
                                           "witness_pass" in state["milestones"] or
                                           "witness_stop" in state["milestones"])
        future_protection = remaining_reserve(("protection", "attacks"), PROTECTION_RESERVE,
                                              "witness_stop" in state["milestones"] or
                                              "protection_complete" in state["milestones"])
        future_verify = remaining_reserve(("verification",), VERIFY_RESERVE,
                                          "verification_complete" in state["milestones"])
        protected = ((future_witness if stage != "witness" else 0.0) +
                     (future_protection if stage not in ("protection", "attacks") else 0.0) +
                     (future_verify if stage != "verification" else 0.0))
        if used + active_reserve + reserve + protected > CPU_LIMIT:
            return "cpu_budget_or_protected_reserve"
        if stage == "channel":
            channel_used = float(state["used_by_stage_s"]["channel"]) + sum(
                float(u["spec"]["cpu_reserve_s"]) for u in running
                if u["spec"]["stage"] == "channel")
            if channel_used + reserve > CHANNEL_LIMIT:
                return "channel_cpu_cap"
        rss = sum(float(u["spec"]["max_rss_gib"]) for u in running)
        if rss + float(spec["max_rss_gib"]) > 4:
            return "memory_reservation"
        if psutil is not None and psutil.virtual_memory().available - float(spec["max_rss_gib"]) * 1024**3 < 1024**3:
            return "host_memory_headroom"
        return None

    def status(self) -> dict:
        with self.mutex():
            state = self.read()
            state["capacity_now"] = self._capacity(state)
            state["free_disk_bytes"] = shutil.disk_usage(self.root).free
            state["projected_output_bytes_queued"] = sum(
                int(u["spec"]["projected_output_bytes"]) for u in state["units"].values()
                if u["status"] == "queued")
            state["projected_output_bytes_running"] = sum(
                int(u["spec"]["projected_output_bytes"]) for u in state["units"].values()
                if u["status"] == "running")
            state["reserved_cpu_s_running"] = sum(
                float(u["spec"]["cpu_reserve_s"]) for u in state["units"].values()
                if u["status"] == "running")
            state["uncommitted_cpu_remaining_s"] = max(
                0, CPU_LIMIT - state["used_cpu_s"] - state["reserved_cpu_s_running"])
            if psutil is not None:
                state["host_memory_available_bytes"] = psutil.virtual_memory().available
            state["wall_remaining_s"] = max(0, (DEADLINE - now()).total_seconds())
            state["cpu_remaining_s"] = max(0, CPU_LIMIT - state["used_cpu_s"])
            return state

    def close(self, reason: str) -> dict:
        with self.mutex():
            state = self.read()
            state["closed"] = True
            state["close_reason"] = reason
            state["closed_utc"] = stamp()
            atomic(self.state, state)
            self.event("close", reason=reason)
            return {"closed": True, "reason": reason}

    def charge(self, charge_id: str, stage: str, cpu_s: float, receipt: Path) -> dict:
        """Account measured or conservative CPU outside a heavy queue unit."""
        if not charge_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in charge_id):
            raise ValueError("invalid charge_id")
        if stage not in STAGES or not 0 < cpu_s <= CPU_LIMIT:
            raise ValueError("invalid stage or cpu_s")
        receipt = receipt.resolve()
        if not receipt.is_file():
            raise ValueError("charge receipt missing")
        value = {"charge_id": charge_id, "stage": stage, "cpu_s": cpu_s,
                 "receipt_path": str(receipt), "receipt_sha256": digest(receipt), "at_utc": stamp()}
        with self.mutex():
            state = self.read()
            state.setdefault("charges", {})
            if charge_id in state["charges"]:
                prior = state["charges"][charge_id]
                if any(prior[k] != value[k] for k in ("stage", "cpu_s", "receipt_path", "receipt_sha256")):
                    raise ValueError("charge ID already has different immutable values")
                return prior
            state["charges"][charge_id] = value
            state["used_cpu_s"] += cpu_s
            state["used_by_stage_s"][stage] += cpu_s
            if state["used_cpu_s"] >= CPU_LIMIT:
                state["closed"] = True
                state["close_reason"] = "cpu_budget_ceiling"
            atomic(self.state, state)
            self.event("charge", **value)
            return value

    def cache_put(self, source: Path) -> dict:
        """Publish a private artifact by content hash; existing objects never change."""
        source = source.resolve()
        if not source.is_file():
            raise ValueError("cache source missing")
        sha = digest(source)
        target = self.root / "cache" / "objects" / sha[:2] / sha
        with self.mutex():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != sha:
                    raise RuntimeError("immutable cache object is corrupt")
            else:
                if shutil.disk_usage(self.root).free - source.stat().st_size < DISK_FLOOR:
                    raise RuntimeError("cache publication would cross disk floor")
                temp = target.with_name(target.name + f".{os.getpid()}.tmp")
                try:
                    with source.open("rb") as inp, temp.open("xb") as out:
                        shutil.copyfileobj(inp, out, length=1024 * 1024)
                        out.flush()
                        os.fsync(out.fileno())
                    if digest(temp) != sha:
                        raise RuntimeError("cache copy SHA-256 mismatch")
                    os.chmod(temp, 0o444)
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
                self.event("cache_put", sha256=sha, size_bytes=target.stat().st_size)
            return {"sha256": sha, "size_bytes": target.stat().st_size, "path": str(target)}

    def reconcile(self, uid: str) -> dict:
        """Charge an orphan's full reserve only after process and lock checks."""
        with self.mutex():
            state = self.read()
            unit = state["units"].get(uid)
            if not unit or unit["status"] != "running":
                raise ValueError("unit is not recorded running")
            slot = unit["slot"]
            handle = (self.root / f"slot{slot}.lock").open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise RuntimeError("slot still held; owner or descendant may still run")
            try:
                for pid in (unit.get("wrapper_pid"), unit.get("child_pid")):
                    if pid:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError(f"PID {pid} still exists or cannot be cleared")
                pgid = unit.get("child_pid")
                if pgid:
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError("child process group still exists")
                charge = float(unit["spec"]["cpu_reserve_s"])
                state["used_cpu_s"] += charge
                state["used_by_stage_s"][unit["spec"]["stage"]] += charge
                unit.update(status="incomplete", charged_cpu_s=charge, recovered_utc=stamp(),
                            recovery_reason="wrapper_missing; full reservation charged")
                atomic(self.state, state)
                self.event("reconcile", unit_id=uid, charged_cpu_s=charge)
                return unit
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()

    def run_once(self) -> dict:
        with self.mutex():
            state = self.read()
            capacity = self._capacity(state)
            if not capacity:
                return {"status": "external_capacity_full"}
        slot = None
        handle = None
        for index in range(capacity):
            candidate = (self.root / f"slot{index}.lock").open("a+")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                slot, handle = index, candidate
                break
            except BlockingIOError:
                candidate.close()
        if handle is None:
            return {"status": "capacity_full"}
        try:
            with self.mutex():
                state = self.read()
                if slot >= self._capacity(state):
                    return {"status": "external_capacity_full"}
                for other in state["units"].values():
                    if other["status"] != "running":
                        continue
                    if other["slot"] == slot:
                        return {"status": "orphan_unresolved", "unit_id": other["spec"]["unit_id"]}
                    probe = (self.root / f"slot{other['slot']}.lock").open("a+")
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        fcntl.flock(probe, fcntl.LOCK_UN)
                        return {"status": "orphan_unresolved", "unit_id": other["spec"]["unit_id"]}
                    finally:
                        probe.close()
                queued = sorted(((uid, u) for uid, u in state["units"].items() if u["status"] == "queued"),
                                key=lambda pair: (PRIORITY[pair[1]["spec"]["stage"]], pair[1]["submitted_utc"], pair[0]))
                choices = [(uid, u) for uid, u in queued if self._admit_reason(state, u["spec"]) is None]
                if not choices:
                    reason = self._admit_reason(state, queued[0][1]["spec"]) if queued else "empty"
                    return {"status": "not_admitted", "reason": reason}
                uid, unit = choices[0]
                unit.update(status="running", slot=slot, wrapper_pid=os.getpid(), started_utc=stamp())
                atomic(self.state, state)
                self.event("start", unit_id=uid, stage=unit["spec"]["stage"], slot=slot,
                           wrapper_pid=os.getpid(), spec_sha256=unit["spec_sha256"])
            return self._execute(uid, unit["spec"], slot, handle)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _execute(self, uid: str, spec: dict, slot: int, slot_handle) -> dict:
        outdir = self.root / "units" / uid
        outdir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in spec["env"].items()})
        env.update(THREAD_ENV)
        with self.mutex():
            seal_sha256 = self.read()["seal"]["sha256"]
        env.update({
            "PCRL_QUEUE_ROOT": str(self.root),
            "PCRL_QUEUE_UNIT_ID": uid,
            "PCRL_QUEUE_STAGE": spec["stage"],
            "PCRL_QUEUE_SLOT": str(slot),
            "PCRL_QUEUE_SLOT_FD": str(slot_handle.fileno()),
            "PCRL_QUEUE_SEAL_SHA256": seal_sha256,
            "PCRL_QUEUE_SPEC_SHA256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
        })
        t0 = time.monotonic()
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc = None
        returncode = None
        interrupted = False
        timed_out = False
        resource_limit = None
        cpu_tree_peak_s = 0.0
        rss_peak_bytes = 0
        seen_pids = set()
        descendants_visible = psutil is not None
        cpu_by_pid = {}
        pid_info = {}
        last_lsof = 0.0
        last_cpu_check = None
        max_observed_cpu_rate = 0.0
        old_handlers = {}

        def stop_group(sig=signal.SIGTERM):
            if proc is not None:
                try: os.killpg(proc.pid, sig)
                except ProcessLookupError: pass

        def on_signal(signum, frame):
            nonlocal interrupted
            interrupted = True
            stop_group()

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, on_signal)
        try:
            with (outdir / "stdout.log").open("ab") as stdout, (outdir / "stderr.log").open("ab") as stderr:
                proc = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                        start_new_session=True, pass_fds=(slot_handle.fileno(),))
                with self.mutex():
                    state = self.read()
                    state["units"][uid]["child_pid"] = proc.pid
                    atomic(self.state, state)
                    self.event("child", unit_id=uid, child_pid=proc.pid, pgid=proc.pid)
                while True:
                    elapsed = time.monotonic() - t0
                    if psutil is not None:
                        try:
                            leader = psutil.Process(proc.pid)
                            try:
                                tree = [leader] + leader.children(recursive=True)
                            except (PermissionError, psutil.AccessDenied):
                                if elapsed - last_lsof >= 0.8 or not pid_info:
                                    discovered, descendants_visible = group_pids(proc.pid)
                                    last_lsof = elapsed
                                else:
                                    discovered = set(pid_info) | {proc.pid}
                                tree = []
                                for pid in discovered:
                                    try: tree.append(psutil.Process(pid))
                                    except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                            rss = 0
                            previous_cpu_by_pid = dict(cpu_by_pid)
                            for child in tree:
                                try:
                                    seen_pids.add(child.pid)
                                    rss += child.memory_info().rss
                                    times = child.cpu_times()
                                    cpu_by_pid[child.pid] = max(cpu_by_pid.get(child.pid, 0.0), times.user + times.system)
                                    pid_info[child.pid] = {"pid": child.pid, "ppid": child.ppid(),
                                                           "started_epoch_s": child.create_time()}
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            rss_peak_bytes = max(rss_peak_bytes, rss)
                            cpu_tree_peak_s = sum(cpu_by_pid.values())
                            if last_cpu_check is not None:
                                prev_elapsed = last_cpu_check
                                if elapsed > prev_elapsed + 0.1:
                                    observed_increment = sum(max(0.0, cpu_by_pid[pid] - value)
                                                             for pid, value in previous_cpu_by_pid.items())
                                    rate = observed_increment / (elapsed - prev_elapsed)
                                    max_observed_cpu_rate = max(max_observed_cpu_rate, rate)
                                    if rate > 1.5:
                                        resource_limit = "parallel_heavy_cpu"
                            last_cpu_check = elapsed
                            if rss > float(spec["max_rss_gib"]) * 1024**3:
                                resource_limit = "unit_memory"
                            if cpu_tree_peak_s + self.read()["used_cpu_s"] >= CPU_LIMIT:
                                resource_limit = "campaign_cpu"
                        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                            pass
                    returncode = proc.poll()
                    if returncode is not None:
                        break
                    if self.read()["closed"]:
                        resource_limit = "campaign_closed"
                    if elapsed >= float(spec["max_wall_s"]) or now() >= DEADLINE or interrupted or resource_limit:
                        timed_out = elapsed >= float(spec["max_wall_s"]) or now() >= DEADLINE
                        stop_group()
                        try: returncode = proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            stop_group(signal.SIGKILL)
                            returncode = proc.wait()
                        break
                    time.sleep(0.25)
                # Descendants may outlive a supervisor. The slot remains held until
                # the entire owned group exits; kill only this campaign's group.
                try: os.killpg(proc.pid, 0)
                except ProcessLookupError: pass
                else:
                    stop_group()
                    time.sleep(0.2)
                    try: os.killpg(proc.pid, 0)
                    except ProcessLookupError: pass
                    else:
                        stop_group(signal.SIGKILL)
                        interrupted = True
        except Exception as exc:
            interrupted = True
            stop_group()
            if proc is not None:
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    stop_group(signal.SIGKILL)
                    proc.wait()
            with self.mutex():
                self.event("exception", unit_id=uid, exception_kind=type(exc).__name__, message=str(exc))
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_direct = max(0.0, r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime)
            wall = time.monotonic() - t0
            # One-thread worker contract: elapsed time covers short-lived descendants
            # that cannot be sampled after reaping. Child CPU is also recorded.
            charge = max(cpu_direct, cpu_tree_peak_s, wall)
            output_hashes = {}
            outputs_ok = returncode == 0 and not interrupted and not timed_out and not resource_limit
            if outputs_ok:
                for name in spec["outputs"]:
                    target = Path(name)
                    if not target.is_absolute(): target = Path(spec["cwd"]) / target
                    if not target.is_file():
                        outputs_ok = False
                        break
                    output_hashes[name] = digest(target)
            status = "done" if outputs_ok else ("incomplete" if interrupted or timed_out or resource_limit else "failed")
            receipt = {"unit_id": uid, "status": status, "stage": spec["stage"],
                       "spec_sha256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
                       "slot": slot, "wrapper_pid": os.getpid(), "child_pid": proc.pid if proc else None,
                       "observed_tree_pids": sorted(seen_pids), "returncode": returncode,
                       "descendants_visible": descendants_visible,
                       "observed_processes": sorted(pid_info.values(), key=lambda p: p["pid"]),
                       "max_observed_cpu_rate": round(max_observed_cpu_rate, 3),
                       "wall_s": round(wall, 3), "direct_child_cpu_s": round(cpu_direct, 3),
                       "sampled_tree_cpu_peak_s": round(cpu_tree_peak_s, 3),
                       "charged_cpu_s": round(charge, 3), "sampled_tree_rss_peak_bytes": rss_peak_bytes,
                       "child_maxrss_raw": r1.ru_maxrss, "maxrss_platform": platform.system(),
                       "timed_out": timed_out, "interrupted": interrupted,
                       "resource_limit": resource_limit,
                       "output_sha256": output_hashes, "finished_utc": stamp()}
            with self.mutex():
                state = self.read()
                state["used_cpu_s"] += charge
                state["used_by_stage_s"][spec["stage"]] += charge
                state["units"][uid].update(receipt)
                if state["used_cpu_s"] >= CPU_LIMIT or now() >= DEADLINE:
                    state["closed"] = True
                    state["close_reason"] = "budget_ceiling"
                atomic(self.state, state)
                self.event("finish", **receipt)
            atomic(outdir / "RECEIPT.json", receipt)
        return {"status": status, "unit_id": uid, "returncode": returncode,
                "receipt": str(outdir / "RECEIPT.json")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("init")
    opening = sub.add_parser("open")
    opening.add_argument("seal", type=Path)
    opening.add_argument("sha256")
    occ = sub.add_parser("occupancy")
    occ.add_argument("evidence", type=Path)
    occ.add_argument("external_heavy", type=int)
    amendment = sub.add_parser("amend")
    amendment.add_argument("path", type=Path)
    amendment.add_argument("sha256")
    amendment.add_argument("dated_utc")
    amendment.add_argument("stages_csv", help="comma-separated affected stages")
    mark = sub.add_parser("mark")
    mark.add_argument("milestone", choices=("witness_pass", "witness_stop",
                                            "protection_complete", "verification_complete",
                                            "channel_identity_pass"))
    mark.add_argument("receipt", type=Path)
    enq = sub.add_parser("enqueue")
    enq.add_argument("spec", type=Path)
    sub.add_parser("run-once")
    close = sub.add_parser("close")
    close.add_argument("reason")
    rec = sub.add_parser("reconcile")
    rec.add_argument("unit_id")
    cache = sub.add_parser("cache-put")
    cache.add_argument("source", type=Path)
    charge = sub.add_parser("charge")
    charge.add_argument("charge_id")
    charge.add_argument("stage", choices=sorted(STAGES))
    charge.add_argument("cpu_s", type=float)
    charge.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "init":
        path = ROOT / "START_UTC.json"
        ROOT.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(path.read_text()); return 0
        with path.open("x") as stream:
            json.dump({"start_utc": stamp(), "deadline_hours": 8, "cpu_limit_s": CPU_LIMIT}, stream)
        print(path.read_text()); return 0
    queue = Queue(args.root)
    if args.command == "status": result = queue.status()
    elif args.command == "open": result = queue.open(args.seal, args.sha256)
    elif args.command == "occupancy": result = queue.occupancy(args.evidence, args.external_heavy)
    elif args.command == "amend": result = queue.amend(args.path, args.sha256, args.dated_utc, args.stages_csv)
    elif args.command == "mark": result = queue.mark(args.milestone, args.receipt)
    elif args.command == "enqueue": result = queue.enqueue(args.spec)
    elif args.command == "run-once": result = queue.run_once()
    elif args.command == "close": result = queue.close(args.reason)
    elif args.command == "cache-put": result = queue.cache_put(args.source)
    elif args.command == "charge": result = queue.charge(args.charge_id, args.stage, args.cpu_s, args.receipt)
    else: result = queue.reconcile(args.unit_id)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("status") not in ("failed", "incomplete") else 1


if __name__ == "__main__":
    sys.exit(main())

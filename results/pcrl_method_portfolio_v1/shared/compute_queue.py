"""Durable, local, two-slot compute queue for the PCRL method portfolio.

The queue is private runtime state; point ``--root`` at one shared writable
directory outside Git. Every numerical job, including smoke timing and replay,
must enter this queue. The default admission gate is closed. This semaphore
only coordinates cooperating portfolio workers; external jobs are reserved
through ``external_heavy`` after a fresh machine inventory.

The one-process/one-thread contract makes elapsed wall time a conservative CPU
charge when descendant CPU accounting is incomplete. Jobs may not daemonize.
No shell is used to launch a unit. A stopped unit is retained as incomplete.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
SLOTS = 2
CPU_BUDGET_S = 20 * 3600
WALL_BUDGET_S = 10 * 3600
MEMORY_BUDGET_GIB = 6.0
DISK_FLOOR_GIB = 5.0
OCCUPANCY_TTL_S = 120
PRELOCK_MAX_WALL_S = 120
PRELOCK_MAX_RSS_GIB = 1.5
UNIT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class Queue:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "QUEUE.json"
        self.mutex_path = self.root / "QUEUE.lock"
        self.receipts_path = self.root / "RECEIPTS.jsonl"
        with self._mutex():
            if not self.state_path.exists():
                self._save({
                    "schema": "pcrl-portfolio-compute-queue-v1",
                    "created_utc": utc_now(),
                    "campaign_start_utc": utc_now(),
                    "budget_cpu_s": CPU_BUDGET_S,
                    "budget_wall_s": WALL_BUDGET_S,
                    "memory_budget_gib": MEMORY_BUDGET_GIB,
                    "disk_floor_gib": DISK_FLOOR_GIB,
                    "used_cpu_s": 0.0,
                    "gate": {
                        "safe_inputs": False,
                        "lock_pushed": False,
                        "occupancy_confirmed": False,
                        "external_heavy": 2,
                        "occupancy_checked_utc": None,
                        "evidence": {},
                        "manually_closed": False,
                        "phase1_lock_path": None,
                        "phase1_lock_sha256": None,
                    },
                    "units": {},
                    "deviations": {},
                })

    @contextlib.contextmanager
    def _mutex(self):
        with self.mutex_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self) -> dict:
        return json.loads(self.state_path.read_text())

    def _save(self, state: dict) -> None:
        atomic_json(self.state_path, state)

    def _receipt(self, event: str, **fields) -> None:
        record = {"at_utc": utc_now(), "event": event, **fields}
        with self.receipts_path.open("a") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def status(self) -> dict:
        with self._mutex():
            state = self._read()
            state["elapsed_wall_s"] = max(0.0, (dt.datetime.now(dt.timezone.utc) - parse_utc(state["campaign_start_utc"])).total_seconds())
            state["free_disk_gib"] = self._free_disk_gib()
            return state

    def enqueue(self, spec: dict) -> dict:
        needed = ("unit_id", "argv", "cwd", "deps", "max_wall_s", "cpu_reserve_s", "max_rss_gib", "stage")
        missing = [k for k in needed if k not in spec]
        if missing:
            raise ValueError(f"missing unit fields: {missing}")
        if not UNIT_ID.fullmatch(spec["unit_id"]):
            raise ValueError("invalid unit_id")
        if not isinstance(spec["argv"], list) or not spec["argv"] or not all(isinstance(x, str) for x in spec["argv"]):
            raise ValueError("argv must be a nonempty list of strings, with no shell")
        if not isinstance(spec["deps"], list) or not all(isinstance(x, str) for x in spec["deps"]):
            raise ValueError("deps must be a list of unit IDs")
        if float(spec["max_wall_s"]) <= 0 or float(spec["cpu_reserve_s"]) < float(spec["max_wall_s"]):
            raise ValueError("cpu_reserve_s must cover max_wall_s under conservative accounting")
        if not 0 < float(spec["max_rss_gib"]) <= MEMORY_BUDGET_GIB:
            raise ValueError("max_rss_gib is outside the predeclared memory budget")
        if not Path(spec["cwd"]).is_dir():
            raise ValueError("cwd does not exist")
        spec = dict(spec)
        spec.setdefault("outputs", [])
        spec.setdefault("env", {})
        spec.setdefault("mode", "real_data")
        if not isinstance(spec["outputs"], list) or not isinstance(spec["env"], dict):
            raise ValueError("outputs or env malformed")
        if spec["mode"] == "prelock_synthetic":
            if spec.get("data_access") != "NONE" or not all(spec.get(k) for k in ("team", "test_file", "test_sha256")):
                raise ValueError("prelock synthetic unit needs NONE data access, team, and frozen test identity")
            if float(spec["max_wall_s"]) > PRELOCK_MAX_WALL_S or float(spec["max_rss_gib"]) > PRELOCK_MAX_RSS_GIB:
                raise ValueError("prelock synthetic unit exceeds its short wall/RSS limits")
        elif spec["mode"] != "real_data":
            raise ValueError("unknown unit mode")
        with self._mutex():
            state = self._read()
            old = state["units"].get(spec["unit_id"])
            if old:
                if canonical(old["spec"]) != canonical(spec):
                    raise ValueError("unit ID is immutable; use a new ID for a changed command or scope")
                return old
            state["units"][spec["unit_id"]] = {"spec": spec, "status": "planned"}
            self._save(state)
            self._receipt("enqueue", unit_id=spec["unit_id"], stage=spec["stage"], spec_sha256=hashlib.sha256(canonical(spec).encode()).hexdigest())
            return state["units"][spec["unit_id"]]

    def record_deviation(self, deviation_id: str, charged_cpu_s: float, note: str) -> dict:
        if not UNIT_ID.fullmatch(deviation_id) or charged_cpu_s <= 0:
            raise ValueError("invalid deviation ID or charge")
        record = {"charged_cpu_s": float(charged_cpu_s), "note": note}
        with self._mutex():
            state = self._read()
            state.setdefault("deviations", {})
            old = state["deviations"].get(deviation_id)
            if old:
                if old != record:
                    raise ValueError("deviation ID is immutable")
                return old
            state["deviations"][deviation_id] = record
            state["used_cpu_s"] += charged_cpu_s
            self._save(state)
            self._receipt("deviation_charge", deviation_id=deviation_id, **record)
            return record

    def update_gate(self, *, safe_inputs: bool, lock_pushed: bool, occupancy_confirmed: bool,
                    external_heavy: int, evidence: dict | None = None, manually_closed: bool = False,
                    phase1_lock_path: str | None = None, phase1_lock_sha256: str | None = None) -> dict:
        if external_heavy not in (0, 1, 2):
            raise ValueError("external_heavy must be 0, 1, or 2")
        if phase1_lock_path is not None:
            if not phase1_lock_sha256 or sha256_file(Path(phase1_lock_path)) != phase1_lock_sha256:
                raise ValueError("PHASE1_LOCK file/hash mismatch")
            lock = json.loads(Path(phase1_lock_path).read_text())
            laws = lock.get("shared_laws", {})
            law_path = Path(phase1_lock_path).parent / laws.get("path", "")
            if not law_path.is_file() or sha256_file(law_path) != laws.get("sha256"):
                raise ValueError("PHASE1_LOCK law file/hash mismatch")
        with self._mutex():
            state = self._read()
            if any(u["status"] == "running" for u in state["units"].values()):
                raise RuntimeError("do not change occupancy while portfolio units run")
            state["gate"] = {
                "safe_inputs": bool(safe_inputs),
                "lock_pushed": bool(lock_pushed),
                "occupancy_confirmed": bool(occupancy_confirmed),
                "external_heavy": external_heavy,
                "occupancy_checked_utc": utc_now() if occupancy_confirmed else None,
                "evidence": evidence or {},
                "manually_closed": bool(manually_closed),
                "phase1_lock_path": phase1_lock_path or state["gate"].get("phase1_lock_path"),
                "phase1_lock_sha256": phase1_lock_sha256 or state["gate"].get("phase1_lock_sha256"),
            }
            self._save(state)
            self._receipt("gate", gate=state["gate"])
            return state["gate"]

    def _free_disk_gib(self) -> float:
        fs = os.statvfs(self.root)
        return fs.f_bavail * fs.f_frsize / 1024**3

    def _gate_status(self, state: dict, spec: dict) -> str | None:
        gate = state["gate"]
        if gate["manually_closed"] or not gate["occupancy_confirmed"]:
            return "gate_closed"
        checked = gate["occupancy_checked_utc"]
        if not checked or (dt.datetime.now(dt.timezone.utc) - parse_utc(checked)).total_seconds() > OCCUPANCY_TTL_S:
            return "occupancy_stale"
        if gate["external_heavy"] >= SLOTS:
            return "external_capacity_full"
        if (dt.datetime.now(dt.timezone.utc) - parse_utc(state["campaign_start_utc"])).total_seconds() >= state["budget_wall_s"]:
            return "wall_budget_exhausted"
        if state["used_cpu_s"] >= state["budget_cpu_s"]:
            return "cpu_budget_exhausted"
        if self._free_disk_gib() < state["disk_floor_gib"]:
            return "disk_floor_reached"
        if spec.get("mode", "real_data") == "prelock_synthetic":
            lock_path = gate.get("phase1_lock_path")
            lock_hash = gate.get("phase1_lock_sha256")
            if not lock_path or not lock_hash or not Path(lock_path).is_file() or sha256_file(Path(lock_path)) != lock_hash:
                return "phase1_lock_unverified"
            lock = json.loads(Path(lock_path).read_text())
            laws = lock.get("shared_laws", {})
            law_path = Path(lock_path).parent / laws.get("path", "")
            if not law_path.is_file() or sha256_file(law_path) != laws.get("sha256"):
                return "phase1_law_changed"
            registered = lock.get("pre_algorithm_tests", {}).get(spec.get("team"), [])
            if isinstance(registered, dict):
                registered = [registered] if "path" in registered else list(registered.values())
            matched = False
            for test in registered:
                registered_path = Path(test.get("path", ""))
                test_path = Path(spec["test_file"])
                suffix_match = (len(registered_path.parts) > 0
                                and tuple(test_path.parts[-len(registered_path.parts):]) == registered_path.parts)
                direct_match = (test_path.resolve() == (Path(lock_path).parent / registered_path).resolve())
                if (test_path.is_file() and (suffix_match or direct_match)
                        and test.get("sha256") == spec.get("test_sha256")
                        and sha256_file(test_path) == test.get("sha256")):
                    matched = True
                    break
            if not matched:
                return "phase1_test_unregistered"
        elif not gate["safe_inputs"] or not gate["lock_pushed"]:
            return "gate_closed"
        return None

    def run_once(self) -> dict:
        # A slot is held by flock until the launched process group has exited.
        with self._mutex():
            state = self._read()
            planned = [u["spec"] for u in state["units"].values() if u["status"] == "planned"]
            if not planned:
                return {"status": "empty"}
            reasons = [self._gate_status(state, spec) for spec in planned]
            if all(reason is not None for reason in reasons):
                return {"status": reasons[0]}
            available_slots = SLOTS - state["gate"]["external_heavy"]
        slot = None
        slot_handle = None
        for number in range(available_slots):
            handle = (self.root / f"slot{number}.lock").open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            slot, slot_handle = number, handle
            break
        if slot_handle is None:
            return {"status": "portfolio_capacity_full"}
        try:
            with self._mutex():
                state = self._read()
                if slot >= SLOTS - state["gate"]["external_heavy"]:
                    return {"status": "external_capacity_full"}
                running_rss = sum(float(u["spec"]["max_rss_gib"]) for u in state["units"].values() if u["status"] == "running")
                eligible = []
                for uid, unit in state["units"].items():
                    if unit["status"] != "planned":
                        continue
                    spec = unit["spec"]
                    if self._gate_status(state, spec) is not None:
                        continue
                    if not all(state["units"].get(dep, {}).get("status") == "done" for dep in spec["deps"]):
                        continue
                    elapsed = (dt.datetime.now(dt.timezone.utc) - parse_utc(state["campaign_start_utc"])).total_seconds()
                    if state["used_cpu_s"] + sum(float(u["spec"]["cpu_reserve_s"]) for u in state["units"].values() if u["status"] == "running") + float(spec["cpu_reserve_s"]) > state["budget_cpu_s"]:
                        continue
                    if elapsed + float(spec["max_wall_s"]) > state["budget_wall_s"]:
                        continue
                    if running_rss + float(spec["max_rss_gib"]) > state["memory_budget_gib"]:
                        continue
                    eligible.append(uid)
                if not eligible:
                    if any(u["status"] == "planned" for u in state["units"].values()):
                        return {"status": "no_admissible_unit"}
                    return {"status": "empty"}
                uid = eligible[0]
                unit = state["units"][uid]
                unit.update(status="running", slot=slot, worker_pid=os.getpid(), started_utc=utc_now())
                self._save(state)
                self._receipt("start", unit_id=uid, slot=slot, worker_pid=os.getpid())
            return self._run_unit(uid, unit["spec"], slot)
        finally:
            fcntl.flock(slot_handle, fcntl.LOCK_UN)
            slot_handle.close()

    def _run_unit(self, uid: str, spec: dict, slot: int) -> dict:
        unit_dir = self.root / "units" / uid
        unit_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in spec["env"].items()})
        env.update(THREAD_ENV)
        start_wall = time.monotonic()
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        timed_out = False
        interrupted = False
        returncode = None
        process = None

        def stop_own_group(sig: int = signal.SIGTERM) -> None:
            if process is not None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass

        old_handlers = {}
        def on_signal(signum, frame):
            nonlocal interrupted
            interrupted = True
            stop_own_group()

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, on_signal)
        try:
            with (unit_dir / "stdout.log").open("ab") as out, (unit_dir / "stderr.log").open("ab") as err:
                process = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                                           stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                           start_new_session=True)
                deadline = start_wall + float(spec["max_wall_s"])
                while True:
                    try:
                        returncode = process.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline or interrupted:
                            timed_out = time.monotonic() >= deadline
                            stop_own_group()
                            try:
                                returncode = process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                stop_own_group(signal.SIGKILL)
                                returncode = process.wait()
                            break
                # A job is not allowed to leave descendants running after its leader exits.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    stop_own_group(signal.SIGTERM)
                    time.sleep(0.1)
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        stop_own_group(signal.SIGKILL)
                        interrupted = True
        except Exception as exc:
            interrupted = True
            stop_own_group(signal.SIGTERM)
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    stop_own_group(signal.SIGKILL)
                    process.wait()
            with self._mutex():
                self._receipt("worker_exception", unit_id=uid, kind=type(exc).__name__, message=str(exc))
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

            wall_s = max(0.0, time.monotonic() - start_wall)
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu_s = max(0.0, (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))
            charge_s = max(child_cpu_s, wall_s)
            output_hashes = {}
            outputs_ok = True
            if returncode == 0 and not interrupted and not timed_out:
                for name in spec["outputs"]:
                    target = Path(name)
                    if not target.is_absolute():
                        target = Path(spec["cwd"]) / target
                    if not target.is_file():
                        outputs_ok = False
                        break
                    output_hashes[str(name)] = sha256_file(target)
            final_status = "done" if returncode == 0 and not timed_out and not interrupted and outputs_ok else ("incomplete" if timed_out or interrupted else "failed")
            receipt = {
                "unit_id": uid,
                "status": final_status,
                "slot": slot,
                "returncode": returncode,
                "wall_s": round(wall_s, 3),
                "child_cpu_s": round(child_cpu_s, 3),
                "charged_cpu_s": round(charge_s, 3),
                "maxrss_raw": r1.ru_maxrss,
                "maxrss_platform": platform.system(),
                "output_sha256": output_hashes,
                "timed_out": timed_out,
                "interrupted": interrupted,
                "finished_utc": utc_now(),
            }
            with self._mutex():
                state = self._read()
                state["used_cpu_s"] += charge_s
                state["units"][uid].update(receipt)
                self._save(state)
                self._receipt("finish", **receipt)
            atomic_json(unit_dir / "RECEIPT.json", receipt)
            if final_status == "done":
                atomic_json(unit_dir / "COMPLETE.json", receipt)
        return {"status": final_status, "unit_id": uid, "returncode": returncode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="shared private queue directory")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    enq = sub.add_parser("enqueue")
    enq.add_argument("spec", help="JSON file with a single exact unit specification")
    gate = sub.add_parser("gate")
    gate.add_argument("--safe-inputs", action="store_true")
    gate.add_argument("--lock-pushed", action="store_true")
    gate.add_argument("--occupancy-confirmed", action="store_true")
    gate.add_argument("--external-heavy", type=int, default=2)
    gate.add_argument("--evidence", help="JSON file with gate receipt identities")
    gate.add_argument("--close", action="store_true")
    gate.add_argument("--phase1-lock-path")
    gate.add_argument("--phase1-lock-sha256")
    sub.add_parser("run-once")
    charge = sub.add_parser("charge")
    charge.add_argument("deviation_id")
    charge.add_argument("charged_cpu_s", type=float)
    charge.add_argument("note")
    args = parser.parse_args(argv)
    queue = Queue(args.root)
    if args.action == "status":
        result = queue.status()
    elif args.action == "enqueue":
        result = queue.enqueue(json.loads(Path(args.spec).read_text()))
    elif args.action == "gate":
        evidence = json.loads(Path(args.evidence).read_text()) if args.evidence else {}
        result = queue.update_gate(safe_inputs=args.safe_inputs, lock_pushed=args.lock_pushed,
                                   occupancy_confirmed=args.occupancy_confirmed,
                                   external_heavy=args.external_heavy, evidence=evidence,
                                   manually_closed=args.close,
                                   phase1_lock_path=args.phase1_lock_path,
                                   phase1_lock_sha256=args.phase1_lock_sha256)
    elif args.action == "charge":
        result = queue.record_deviation(args.deviation_id, args.charged_cpu_s, args.note)
    else:
        result = queue.run_once()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

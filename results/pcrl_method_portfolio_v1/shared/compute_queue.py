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
PRESEAL_SMOKE_MAX_WALL_S = 1800
PRESEAL_SMOKE_MAX_RSS_GIB = 2.0
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
                        "export_authorized": False,
                        "export_reviewed_sha256": None,
                        "audit_authorized": False,
                        "audit_reviewed_sha256": None,
                        "replay_authorized": False,
                        "replay_reviewed_sha256": None,
                        "export_replay_authorized": False,
                        "export_replay_reviewed_sha256": None,
                        "compare_authorized": False,
                        "compare_reviewed_sha256": None,
                        "local_seal_path": None,
                        "local_seal_sha256": None,
                        "s4_local_seal_path": None,
                        "s4_local_seal_sha256": None,
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
            if "budget_epoch" in state:
                state["budget_epoch"]["elapsed_wall_s"] = max(0.0, (dt.datetime.now(dt.timezone.utc) - parse_utc(state["budget_epoch"]["start_utc"])).total_seconds())
            state["free_disk_gib"] = self._free_disk_gib()
            return state

    def begin_continuation(self, start_utc: str) -> dict:
        """Start one new authorized budget epoch without erasing campaign history."""
        parsed = parse_utc(start_utc)
        if parsed.tzinfo is None:
            raise ValueError("continuation start must include UTC offset")
        with self._mutex():
            state = self._read()
            if any(u["status"] == "running" for u in state["units"].values()):
                raise RuntimeError("cannot change budget epoch while a unit runs")
            old = state.get("budget_epoch")
            if old:
                if old["start_utc"] != start_utc:
                    raise ValueError("continuation epoch is immutable")
                return old
            state["budget_epoch"] = {
                "start_utc": start_utc,
                "budget_cpu_s": CPU_BUDGET_S,
                "budget_wall_s": WALL_BUDGET_S,
                "used_cpu_s": 0.0,
                "prior_campaign_used_cpu_s": state["used_cpu_s"],
            }
            self._save(state)
            self._receipt("continuation_epoch", epoch=state["budget_epoch"])
            return state["budget_epoch"]

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
        if spec["mode"] in ("prelock_synthetic", "preseal_smoke"):
            if spec.get("data_access") != "NONE" or not all(spec.get(k) for k in ("team", "test_file", "test_sha256")):
                raise ValueError("synthetic unit needs NONE data access, team, and frozen test identity")
            wall_cap = PRELOCK_MAX_WALL_S if spec["mode"] == "prelock_synthetic" else PRESEAL_SMOKE_MAX_WALL_S
            rss_cap = PRELOCK_MAX_RSS_GIB if spec["mode"] == "prelock_synthetic" else PRESEAL_SMOKE_MAX_RSS_GIB
            if float(spec["max_wall_s"]) > wall_cap or float(spec["max_rss_gib"]) > rss_cap:
                raise ValueError("synthetic unit exceeds its predeclared wall/RSS limits")
        elif spec["mode"] == "verifier_synthetic":
            reason = self._verifier_synthetic_reason(spec)
            if reason:
                raise ValueError(reason)
        elif spec["mode"] == "s4_adult_replay":
            if (spec.get("family") != "S4" or spec.get("team") != "S4" or
                spec.get("data_access") not in (
                    "S4_B_FIT_TASK_ATT_INNER_ROLE_ONLY_REPLAY",
                    "S4_ANCHOR_FIT_TASK_ATT_INNER_ROLE_ONLY_REPLAY")):
                raise ValueError("s4_adult_scope_invalid")
        elif spec["mode"] == "role_export":
            if spec.get("data_access") != "S4_REVIEWED_SINGLE_PASS_SOURCE_ZIP":
                raise ValueError("role export needs exact S4-reviewed data access")
            reviewed = Path(spec.get("reviewed_script", ""))
            if not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256"):
                raise ValueError("role export script/hash mismatch")
            argv_paths = [(Path(spec["cwd"]) / arg).resolve() for arg in spec["argv"]]
            if reviewed.resolve() not in argv_paths:
                raise ValueError("reviewed export script must appear in argv")
            if float(spec["max_wall_s"]) > 120 or float(spec["max_rss_gib"]) > 1.0:
                raise ValueError("role export exceeds S4-reviewed ceiling")
        elif spec["mode"] == "role_audit":
            if spec.get("data_access") != "IDENTITY_ONLY_SOURCE_AND_ALLOWED_ROLE_EXPORT":
                raise ValueError("role audit needs exact identity-only access declaration")
            reviewed = Path(spec.get("reviewed_script", ""))
            if not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256"):
                raise ValueError("role audit script/hash mismatch")
            argv_paths = [(Path(spec["cwd"]) / arg).resolve() for arg in spec["argv"]]
            if reviewed.resolve() not in argv_paths:
                raise ValueError("reviewed audit script must appear in argv")
            if float(spec["max_wall_s"]) > 120 or float(spec["max_rss_gib"]) > 1.0:
                raise ValueError("role audit exceeds bounded ceiling")
        elif spec["mode"] == "role_replay":
            if spec.get("data_access") != "ADMITTED_ROLE_EXPORT_ONLY":
                raise ValueError("role replay must read only admitted role exports")
            reviewed = Path(spec.get("reviewed_script", ""))
            if not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256"):
                raise ValueError("role replay script/hash mismatch")
            argv_paths = [(Path(spec["cwd"]) / arg).resolve() for arg in spec["argv"]]
            if reviewed.resolve() not in argv_paths:
                raise ValueError("reviewed replay script must appear in argv")
            if float(spec["max_wall_s"]) > 60 or float(spec["max_rss_gib"]) > 0.5:
                raise ValueError("role replay exceeds bounded ceiling")
        elif spec["mode"] == "role_export_replay":
            if spec.get("data_access") != "S4_REVIEWED_REPEAT_FOR_BYTE_DETERMINISM":
                raise ValueError("repeat export needs explicit S4-reviewed source access")
            reviewed = Path(spec.get("reviewed_script", ""))
            if not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256"):
                raise ValueError("repeat exporter script/hash mismatch")
            argv_paths = [(Path(spec["cwd"]) / arg).resolve() for arg in spec["argv"]]
            if reviewed.resolve() not in argv_paths:
                raise ValueError("reviewed exporter must appear in argv")
            if not spec.get("preserved_first_path") or not spec.get("replay_output_path"):
                raise ValueError("repeat export needs first and second paths")
            if float(spec["max_wall_s"]) > 120 or float(spec["max_rss_gib"]) > 1.0:
                raise ValueError("repeat export exceeds bounded ceiling")
        elif spec["mode"] == "role_compare":
            if spec.get("data_access") != "TWO_ALLOWED_EXPORT_TREES_ONLY":
                raise ValueError("comparison may read only two allowed export trees")
            reviewed = Path(spec.get("reviewed_script", ""))
            if not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256"):
                raise ValueError("comparison script/hash mismatch")
            argv_paths = [(Path(spec["cwd"]) / arg).resolve() for arg in spec["argv"]]
            if reviewed.resolve() not in argv_paths:
                raise ValueError("reviewed comparison script must appear in argv")
            if float(spec["max_wall_s"]) > 120 or float(spec["max_rss_gib"]) > 0.5:
                raise ValueError("comparison exceeds bounded ceiling")
        elif spec["mode"] == "role_probe":
            if spec.get("data_access") != "ADMITTED_ATTACK_FIT_SEX_ONLY":
                raise ValueError("role probe may read only admitted ATTACK_FIT_NEW SEX")
            reviewed = Path(spec.get("reviewed_script", ""))
            if not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256"):
                raise ValueError("role probe script/hash mismatch")
            argv_paths = [(Path(spec["cwd"]) / arg).resolve() for arg in spec["argv"]]
            if reviewed.resolve() not in argv_paths:
                raise ValueError("reviewed role probe script must appear in argv")
            if float(spec["max_wall_s"]) > 120 or float(spec["max_rss_gib"]) > 1.0:
                raise ValueError("role probe exceeds bounded ceiling")
        elif spec["mode"] != "real_data":
            raise ValueError("unknown unit mode")
        if spec["mode"] == "real_data":
            if spec.get("family") not in ("A", "B", "C", "D", "SHARED") or not isinstance(spec.get("arm_id"), str):
                raise ValueError("real_data unit requires explicit family and arm_id")
        with self._mutex():
            state = self._read()
            if spec["mode"] == "real_data":
                reason = self._real_admission_reason(state["gate"], spec)
                if reason is None:
                    reason = self._v2_parent_reason(state, spec)
                if reason:
                    raise ValueError(reason)
            elif spec["mode"] == "s4_adult_replay":
                reason = self._s4_admission_reason(state["gate"], spec)
                if reason is None:
                    reason = self._s4_parent_reason(state, spec)
                if reason:
                    raise ValueError(reason)
            old = state["units"].get(spec["unit_id"])
            if old:
                if canonical(old["spec"]) != canonical(spec):
                    raise ValueError("unit ID is immutable; use a new ID for a changed command or scope")
                return old
            if spec["mode"] in ("role_export", "role_audit", "role_replay", "role_export_replay", "role_compare") and any(u["spec"].get("mode") == spec["mode"] for u in state["units"].values()):
                raise ValueError(f"single {spec['mode']} already registered")
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
            if "budget_epoch" in state:
                state["budget_epoch"]["used_cpu_s"] += charged_cpu_s
            self._save(state)
            self._receipt("deviation_charge", deviation_id=deviation_id, **record)
            return record

    def update_gate(self, *, safe_inputs: bool, lock_pushed: bool, occupancy_confirmed: bool,
                    external_heavy: int, evidence: dict | None = None, manually_closed: bool = False,
                    phase1_lock_path: str | None = None, phase1_lock_sha256: str | None = None,
                    export_authorized: bool = False, export_reviewed_sha256: str | None = None,
                    audit_authorized: bool = False, audit_reviewed_sha256: str | None = None,
                    replay_authorized: bool = False, replay_reviewed_sha256: str | None = None,
                    export_replay_authorized: bool = False, export_replay_reviewed_sha256: str | None = None,
                    compare_authorized: bool = False, compare_reviewed_sha256: str | None = None,
                    local_seal_path: str | None = None, local_seal_sha256: str | None = None,
                    v2_local_seal_path: str | None = None,
                    v2_local_seal_sha256: str | None = None,
                    s4_local_seal_path: str | None = None,
                    s4_local_seal_sha256: str | None = None) -> dict:
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
        if local_seal_path is not None:
            if not local_seal_sha256 or sha256_file(Path(local_seal_path)) != local_seal_sha256:
                raise ValueError("local seal receipt file/hash mismatch")
        if v2_local_seal_path is not None:
            if not v2_local_seal_sha256 or sha256_file(Path(v2_local_seal_path)) != v2_local_seal_sha256:
                raise ValueError("V2 local seal receipt file/hash mismatch")
        if s4_local_seal_path is not None:
            if not s4_local_seal_sha256 or sha256_file(Path(s4_local_seal_path)) != s4_local_seal_sha256:
                raise ValueError("S4 local seal receipt file/hash mismatch")
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
                "export_authorized": bool(export_authorized),
                "export_reviewed_sha256": export_reviewed_sha256,
                "audit_authorized": bool(audit_authorized),
                "audit_reviewed_sha256": audit_reviewed_sha256,
                "replay_authorized": bool(replay_authorized),
                "replay_reviewed_sha256": replay_reviewed_sha256,
                "export_replay_authorized": bool(export_replay_authorized),
                "export_replay_reviewed_sha256": export_replay_reviewed_sha256,
                "compare_authorized": bool(compare_authorized),
                "compare_reviewed_sha256": compare_reviewed_sha256,
                "local_seal_path": local_seal_path,
                "local_seal_sha256": local_seal_sha256,
                "v2_local_seal_path": v2_local_seal_path,
                "v2_local_seal_sha256": v2_local_seal_sha256,
                "s4_local_seal_path": s4_local_seal_path,
                "s4_local_seal_sha256": s4_local_seal_sha256,
            }
            self._save(state)
            self._receipt("gate", gate=state["gate"])
            return state["gate"]

    def _free_disk_gib(self) -> float:
        fs = os.statvfs(self.root)
        return fs.f_bavail * fs.f_frsize / 1024**3

    @staticmethod
    def _verifier_synthetic_reason(spec: dict) -> str | None:
        """Bind a no-data S4 test to its own committed, immutable index."""
        registered_indexes = {
            "S4_VERIFIER_QUEUE_INDEX_V3.json": "928c6d00b859263e79a5cb8a8e65ccf573e29e89ac2cb1ea8128acd655c10ad6",
            "S4_VERIFIER_QUEUE_INDEX_V5.json": "ac3103d2514cc939d9520aa633852d66dfa4de4c9cb38996276ee1d510e89fb1",
            "S4_VERIFIER_QUEUE_INDEX_V6.json": "e74504d76b34c169d2261f88a9a1845709e0ee394fbd6d73675a0e94787df3f2",
            "S4_VERIFIER_QUEUE_INDEX_V7.json": "c39cb299b6218f28dae372babb4c44eec8117d23b67667a333f2173b0f9a3cc7",
            "S4_VERIFIER_QUEUE_INDEX_V8.json": "840b95c254a66ebe902310e87cebc2af3437b412e1c168ab92bb7dc3b5b5decb",
            "S4_VERIFIER_QUEUE_INDEX_V9.json": "1a8219dab7bb0ce5de1cee4d387640618520a9a9a83598037e43b081a53fe109",
        }
        if (spec.get("mode") != "verifier_synthetic" or
            spec.get("family") != "S4" or spec.get("team") != "S4" or
            spec.get("data_access") != "NONE" or
            spec.get("deps") != [] or spec.get("outputs") != []):
            return "verifier_scope_invalid"
        try:
            cwd = Path(spec["cwd"]).resolve(strict=True)
            index_path = Path(spec["verifier_index_path"]).resolve(strict=True)
            expected_index = cwd / "results/pcrl_method_portfolio_v1/s4" / index_path.name
            index_sha = registered_indexes.get(index_path.name)
            if (index_path != expected_index or index_sha is None or
                spec.get("verifier_index_sha256") != index_sha or
                sha256_file(index_path) != index_sha):
                return "verifier_index_unregistered"
            committed = subprocess.run(
                ["git", "-C", str(cwd), "show", f"HEAD:results/pcrl_method_portfolio_v1/s4/{index_path.name}"],
                capture_output=True, check=False, timeout=10)
            if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != index_sha:
                return "verifier_index_uncommitted"
            index = json.loads(index_path.read_text())
            units = index.get("units", {})
            registered = units.get(spec.get("unit_id"))
            if (index.get("schema") != "pcrl-s4-verifier-synthetic-index-v2" or
                index.get("status") != "REGISTERED_SYNTHETIC_ONLY" or
                index.get("adult_role_access") is not False or
                index.get("cwd") != "coordinator_worktree" or
                index.get("data_access") != "NONE" or
                index.get("interpreter") != "PCRL_PROJECT_VENV_PYTHON" or
                not isinstance(registered, dict) or
                registered.get("mode") != "verifier_synthetic" or
                registered.get("family") != "S4" or registered.get("team") != "S4" or
                registered.get("data_access") != "NONE" or
                spec.get("stage") != registered.get("stage")):
                return "verifier_unit_unregistered"
            test_path = (cwd / registered["test_path"]).resolve(strict=True)
            if (Path(spec.get("test_file", "")).resolve() != test_path or
                spec.get("test_sha256") != registered.get("test_sha256") or
                sha256_file(test_path) != registered.get("test_sha256")):
                return "verifier_test_changed"
            source_map = registered.get("source_sha256_by_file")
            if not isinstance(source_map, dict) or spec.get("source_sha256_by_file", {}) != source_map:
                return "verifier_sources_unregistered"
            for relative, expected_sha in source_map.items():
                source = (cwd / relative).resolve(strict=True)
                if (not source.is_relative_to(cwd) or
                    not isinstance(expected_sha, str) or
                    sha256_file(source) != expected_sha):
                    return "verifier_source_changed"
            venv = cwd.parent.parent / ".venv/bin/python"
            argv = list(registered.get("argv_template", ()))
            if not argv or argv[0] != "<PCRL_PROJECT_VENV_PYTHON>":
                return "verifier_argv_unregistered"
            argv[0] = str(venv)
            if spec.get("argv") != argv or not venv.is_file():
                return "verifier_argv_mismatch"
            expected_env = dict(index.get("env", {}))
            expected_env["PYTHONPATH"] = str(cwd / expected_env["PYTHONPATH"])
            actual_env = spec.get("env", {})
            if (not isinstance(actual_env, dict) or
                any(actual_env.get(key) != value for key, value in expected_env.items()) or
                any(key not in expected_env and (key != "PYTHONDONTWRITEBYTECODE" or value != "1")
                    for key, value in actual_env.items())):
                return "verifier_env_mismatch"
            if (float(spec["max_wall_s"]) > registered["max_wall_seconds"] or
                float(spec["cpu_reserve_s"]) > registered["max_wall_seconds"] or
                float(spec["max_rss_gib"]) * 1024**3 > registered["max_rss_bytes"] or
                registered.get("threads") != 1):
                return "verifier_resource_limit_mismatch"
            return None
        except (OSError, ValueError, TypeError, KeyError, IndexError, subprocess.TimeoutExpired):
            return "verifier_index_or_spec_invalid"

    @staticmethod
    def _local_seal_ok(gate: dict) -> bool:
        """Verify current lock, S2 admission, S4 review, and committed Git blob."""
        receipt_path = Path(gate.get("local_seal_path") or "")
        receipt_sha = gate.get("local_seal_sha256")
        if not receipt_sha or not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha:
            return False
        try:
            receipt = json.loads(receipt_path.read_text())
            commit = receipt.get("commit_sha")
            lock_sha = receipt.get("campaign_lock_sha256")
            if (receipt.get("status") != "PASS" or
                not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or
                not isinstance(lock_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lock_sha)):
                return False
            if receipt.get("lock_commit_sha") != commit:
                return False
            if receipt.get("lock_sha256") != lock_sha:
                return False
            review_name = receipt.get("review_path") or receipt.get("s4_review_path")
            review_sha = receipt.get("review_sha256") or receipt.get("s4_review_sha256")
            if review_name != "S4_PREFIT_SEAL_REVIEW.json" or not isinstance(review_sha, str):
                return False
            review_path = receipt_path.parent / review_name
            if not review_path.is_file() or sha256_file(review_path) != review_sha:
                return False
            review = json.loads(review_path.read_text())
            if (review.get("status") != "PASS" or
                (review.get("lock_commit_sha") or review.get("commit_sha")) != commit or
                (review.get("lock_sha256") or review.get("campaign_lock_sha256")) != lock_sha):
                return False
            lock_path = receipt_path.parent / "CAMPAIGN_LOCK.json"
            if not lock_path.is_file() or sha256_file(lock_path) != lock_sha:
                return False
            lock = json.loads(lock_path.read_text())
            if lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or lock.get("adult_fit_permission") is not True:
                return False
            scope = lock.get("fit_admission")
            if scope not in ("B_ONLY", "ALL_FAMILIES") or review.get("fit_admission") != scope or receipt.get("fit_admission") != scope:
                return False
            families = lock.get("families", {})
            if set(families) != set("ABCD"):
                return False
            if scope == "B_ONLY" and (families["B"].get("fit_admission") != "ALLOWED" or
                                      any(families[k].get("fit_admission") != "DENIED" for k in "ACD")):
                return False
            admitted = lock.get("admitted_arms", {})
            if not isinstance(admitted, dict) or not all(isinstance(v, list) and len(v) == len(set(v))
                                                         for v in admitted.values()):
                return False
            if scope == "B_ONLY" and (not admitted.get("B") or any(admitted.get(k) for k in "ACD")):
                return False
            shared_ids = lock.get("admitted_shared_unit_ids", [])
            if not isinstance(shared_ids, list) or len(shared_ids) != len(set(shared_ids)):
                return False
            admission = lock.get("data_admission", {})
            manifest = receipt_path.parent / "DATA_ACCESS_MANIFEST.json"
            audit = receipt_path.parent / "S4_ROLE_EXPORT_VERIFICATION.json"
            if (not manifest.is_file() or not audit.is_file() or
                sha256_file(manifest) != admission.get("manifest_sha256") or
                sha256_file(audit) != admission.get("s4_audit_receipt_sha256") or
                json.loads(manifest.read_text()).get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED" or
                json.loads(audit.read_text()).get("status") != "PASS"):
                return False
            repo = receipt_path.parent.parent.parent
            relative = lock_path.relative_to(repo)
            committed = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                                       capture_output=True, check=False, timeout=10)
            if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != lock_sha:
                return False
            ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                                      capture_output=True, check=False, timeout=10)
            return ancestor.returncode == 0
        except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _local_v2_seal_ok(gate: dict) -> bool:
        """Verify the separate committed S1/S4 V2 control lock and receipt."""
        receipt_path = Path(gate.get("v2_local_seal_path") or "")
        receipt_sha = gate.get("v2_local_seal_sha256")
        if (receipt_path.name != "CONTROL_V2_LOCAL_SEAL_RECEIPT.json" or
            not isinstance(receipt_sha, str) or
            not re.fullmatch(r"[0-9a-f]{64}", receipt_sha) or
            not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha):
            return False
        try:
            receipt = json.loads(receipt_path.read_text())
            commit = receipt.get("commit_sha")
            lock_sha = receipt.get("control_lock_sha256")
            if (receipt.get("schema") != "pcrl-control-v2-local-seal-receipt-v1" or
                receipt.get("status") != "PASS" or receipt.get("scope") != "B_CONTROL_V2" or
                not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or
                not isinstance(lock_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lock_sha) or
                receipt.get("lock_commit_sha") != commit or receipt.get("lock_sha256") != lock_sha or
                receipt.get("review_path") != "S4_CONTROL_V2_PREFIT_REVIEW.json"):
                return False
            review_sha = receipt.get("review_sha256")
            review_path = receipt_path.parent / receipt["review_path"]
            if (not isinstance(review_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", review_sha) or
                not review_path.is_file() or sha256_file(review_path) != review_sha):
                return False
            review = json.loads(review_path.read_text())
            if (review.get("status") != "PASS" or review.get("scope") != "B_CONTROL_V2" or
                review.get("lock_commit_sha") != commit or review.get("lock_sha256") != lock_sha or
                review.get("control_lock_sha256") != lock_sha):
                return False
            lock_path = receipt_path.parent / "CONTROL_V2_LOCK.json"
            if not lock_path.is_file() or sha256_file(lock_path) != lock_sha:
                return False
            lock = json.loads(lock_path.read_text())
            if (lock.get("schema") != "pcrl-control-v2-lock-v1" or
                lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
                lock.get("scope") != "B_CONTROL_V2" or
                lock.get("control_version") != 2 or
                lock.get("adult_probe_permission") is not True or
                not isinstance(lock.get("probe_unit_ids"), dict) or
                not isinstance(lock.get("accepted_fit_lock_sha256"), list) or
                not lock["accepted_fit_lock_sha256"] or
                any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
                    for x in lock["accepted_fit_lock_sha256"])):
                return False
            repo = receipt_path.parent.parent.parent
            relative = lock_path.relative_to(repo)
            committed = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                                       capture_output=True, check=False, timeout=10)
            if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != lock_sha:
                return False
            ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                                      capture_output=True, check=False, timeout=10)
            return ancestor.returncode == 0
        except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _s4_local_seal_ok(gate: dict) -> bool:
        """Validate the separate, committed S4 replay admission chain."""
        seal_path = Path(gate.get("s4_local_seal_path") or "")
        seal_sha = gate.get("s4_local_seal_sha256")
        if (seal_path.name != "S4_VERIFIER_LOCAL_SEAL_RECEIPT.json" or
            not isinstance(seal_sha, str) or not seal_path.is_file() or
            sha256_file(seal_path) != seal_sha):
            return False
        try:
            campaign = seal_path.parent.resolve(strict=True)
            repo = campaign.parents[1]
            seal = json.loads(seal_path.read_text())
            admission_path = campaign / "S4_VERIFIER_ADMISSION.json"
            review_path = campaign / "S4_VERIFIER_PREFIT_REVIEW.json"
            admission_sha = sha256_file(admission_path)
            review_sha = sha256_file(review_path)
            review = json.loads(review_path.read_text())
            admission = json.loads(admission_path.read_text())
            commit = seal.get("commit_sha")
            if (seal.get("schema") != "pcrl-s4-verifier-local-seal-v3" or
                seal.get("status") != "PASS" or seal.get("scope") != "S4_ADULT_REPLAY" or
                seal.get("admission_sha256") != admission_sha or
                seal.get("review_path") != review_path.name or
                seal.get("review_sha256") != review_sha or
                review.get("schema") != "pcrl-s4-verifier-prefit-review-v3" or
                review.get("status") != "PASS" or review.get("scope") != "S4_ADULT_REPLAY" or
                review.get("admission_sha256") != admission_sha or
                review.get("commit_sha") != commit or
                not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or
                admission.get("schema") != "pcrl-s4-verifier-admission-v3" or
                admission.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"):
                return False
            relative = admission_path.relative_to(repo)
            blob = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                                  capture_output=True, check=False, timeout=10)
            ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                                       commit, "HEAD"], capture_output=True, check=False, timeout=10)
            if (blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != admission_sha or
                ancestor.returncode):
                return False
            map_path = campaign / "s4/S4_ADULT_REPLAY_PARENT_MAP_V2.json"
            schema_path = campaign / "s4/S4_ADULT_REPLAY_UNIT_SCHEMA_V2.json"
            map_sha = "f6100b8fb3ca9f7b2b72d89f486375ed7f4c0b79218db87ec05694018149bce4"
            schema_sha = "6aed1ed024153211a15f6cd5be000169a77db393314fe71fa18781f744d896fc"
            if (sha256_file(map_path) != map_sha or sha256_file(schema_path) != schema_sha or
                admission.get("parent_map_sha256") != map_sha or
                admission.get("unit_schema_sha256") != schema_sha):
                return False
            parent_map = json.loads(map_path.read_text())
            schema = json.loads(schema_path.read_text())
            unit_specs = parent_map.get("unit_specs")
            if (parent_map.get("schema") != "pcrl-s4-adult-replay-parent-map-v2" or
                schema.get("schema") != "pcrl-s4-adult-replay-unit-schema-v2" or
                schema.get("parent_map_sha256") != map_sha or
                not isinstance(unit_specs, dict) or len(unit_specs) != 18 or
                admission.get("unit_specs") != unit_specs or
                admission.get("accepted_fit_lock_sha256") !=
                    parent_map.get("accepted_fit_lock_sha256") or
                admission.get("accepted_control_v2_lock_sha256") !=
                    parent_map.get("accepted_control_v2_lock_sha256")):
                return False
            fit_shas = {u["fit_lock_sha256"] for u in unit_specs.values()}
            control_shas = {u["control_v2_lock_sha256"] for u in unit_specs.values()}
            historical = admission.get("historical_seal_by_lock_sha256")
            if (set(admission["accepted_fit_lock_sha256"]) != fit_shas or
                set(admission["accepted_control_v2_lock_sha256"]) != control_shas or
                not isinstance(historical, dict) or set(historical) != fit_shas | control_shas or
                any(historical[sha].get("kind") != ("FIT" if sha in fit_shas else "CONTROL_V2") or
                    historical[sha].get("lock_sha256") != sha for sha in historical)):
                return False
            sources = admission.get("source_sha256_by_file")
            names = ("run_adult_replay_v3", "replay_b_values_v3", "replay_anchor_values_v3",
                     "replay_b_attempt", "replay_anchor_attempt", "replay_b_readers",
                     "replay_b_release", "replay_feature_anchor", "replay_v2_controls",
                     "verifier_identity_v3", "independent_endpoints")
            expected_source_names = {f"results/pcrl_method_portfolio_v1/s4/{name}.py" for name in names}
            if not isinstance(sources, dict) or set(sources) != expected_source_names:
                return False
            for relative, expected in sources.items():
                source = (repo / relative).resolve(strict=True)
                if (not source.is_relative_to(repo) or not re.fullmatch(r"[0-9a-f]{64}", expected) or
                    sha256_file(source) != expected):
                    return False
            if (admission.get("runner_sha256") != sources[
                    "results/pcrl_method_portfolio_v1/s4/run_adult_replay_v3.py"] or
                admission.get("scheduler_source_sha256") != sha256_file(Path(__file__))):
                return False
            for key in ("queue_root_relative_to_coordinator",
                        "private_output_root_relative_to_coordinator"):
                value = admission.get(key)
                if (not isinstance(value, str) or Path(value).is_absolute() or
                    not (repo / value).resolve().is_relative_to(repo)):
                    return False
            cache_rel = admission.get("hcal_cache_root_relative_to_coordinator")
            if (cache_rel != "../../../PCRL_eval_cache_private" or
                not (repo / cache_rel).resolve().is_dir()):
                return False
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError,
                subprocess.TimeoutExpired):
            return False

    @classmethod
    def _s4_admission_reason(cls, gate: dict, spec: dict) -> str | None:
        """Bind an S4 worker to one exact historical replay and fresh report."""
        if not gate.get("safe_inputs") or not cls._s4_local_seal_ok(gate):
            return "s4_local_seal_invalid"
        try:
            seal_path = Path(gate["s4_local_seal_path"]).resolve(strict=True)
            campaign = seal_path.parent
            repo = campaign.parents[1]
            admission_path = campaign / "S4_VERIFIER_ADMISSION.json"
            admission_sha = sha256_file(admission_path)
            admission = json.loads(admission_path.read_text())
            unit = admission["unit_specs"].get(spec.get("unit_id"))
            if not isinstance(unit, dict):
                return "s4_unit_not_admitted"
            uid = spec["unit_id"]
            kind = unit.get("kind")
            stage = ("independent_b_v2_replay" if kind == "B_SCREEN" else
                     "independent_anchor_v2_replay" if kind == "FEATURE_ANCHOR" else None)
            if (stage is None or spec.get("mode") != "s4_adult_replay" or
                spec.get("family") != "S4" or spec.get("team") != "S4" or
                spec.get("stage") != stage or spec.get("kind") != kind or
                spec.get("arm") != unit.get("arm") or
                spec.get("arm_id") != unit.get("arm") or
                type(spec.get("seed")) is not int or spec["seed"] != unit.get("seed") or
                spec.get("data_access") != unit.get("data_access") or
                spec.get("parent_map_unit_sha256") !=
                    hashlib.sha256(canonical(unit).encode()).hexdigest() or
                spec.get("parent_unit_id") != unit.get("parent_unit_id") or
                spec.get("control_v2_unit_id") != unit.get("control_v2_unit_id") or
                spec.get("deps") != [unit.get("parent_unit_id"),
                                     unit.get("control_v2_unit_id")]):
                return "s4_unit_scope_mismatch"
            expected_uid = (f"S4-{unit['arm']}-S{unit['seed']}" if kind == "B_SCREEN"
                            else f"S4-FEATURE-ANCHOR-S{unit['seed']}")
            if uid != expected_uid:
                return "s4_unit_id_mismatch"
            for name in ("parent_queue_receipt_sha256", "control_v2_queue_receipt_sha256",
                         "fit_lock_sha256", "control_v2_lock_sha256"):
                if spec.get(name) != unit.get(name):
                    return "s4_parent_binding_mismatch"
            if (float(spec["max_wall_s"]) != 600 or
                float(spec["cpu_reserve_s"]) != 600 or
                float(spec["max_rss_gib"]) != 2.0):
                return "s4_resource_limit_mismatch"
            cwd = Path(spec["cwd"]).resolve(strict=True)
            if cwd != repo:
                return "s4_cwd_mismatch"
            runner = campaign / "s4/run_adult_replay_v3.py"
            runner_sha = admission["runner_sha256"]
            if spec.get("runner_sha256") != runner_sha or sha256_file(runner) != runner_sha:
                return "s4_runner_mismatch"
            queue_root = (repo / admission["queue_root_relative_to_coordinator"]).resolve()
            output_root = (repo / admission["private_output_root_relative_to_coordinator"]).resolve()
            cache_root = (repo / admission["hcal_cache_root_relative_to_coordinator"]).resolve()
            output = output_root / uid / "REPORT.json"
            if (queue_root != campaign / "shared/.private/compute" or
                not output_root.is_relative_to(campaign) or
                ".private" not in output_root.parts or
                spec.get("outputs") != [str(output)] or output.exists()):
                return "s4_output_or_queue_root_mismatch"
            venv = repo.parent.parent / ".venv/bin/python"
            argv = [str(venv), str(runner), "--unit-id", uid,
                    "--admission", str(admission_path),
                    "--admission-sha256", admission_sha,
                    "--local-seal", str(seal_path),
                    "--local-seal-sha256", gate["s4_local_seal_sha256"],
                    "--queue-root", str(queue_root),
                    "--private-report", str(output),
                    "--cache-root", str(cache_root)]
            if spec.get("argv") != argv or not venv.is_file():
                return "s4_argv_mismatch"
            env = spec.get("env", {})
            thread_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS")
            if (not isinstance(env, dict) or
                any(env.get(name) != "1" for name in thread_names) or
                env.get("PYTHONDONTWRITEBYTECODE") != "1" or
                env.get("PYTHONPATH") != str(campaign / "s4")):
                return "s4_thread_or_source_path_mismatch"
            return None
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return "s4_admission_or_spec_invalid"

    def _s4_parent_reason(self, state: dict, spec: dict) -> str | None:
        """Recheck both immutable S3 parent outputs before an S4 role read."""
        if spec.get("mode") != "s4_adult_replay":
            return None
        try:
            campaign = Path(state["gate"]["s4_local_seal_path"]).parent
            repo = campaign.parents[1]
            admission = json.loads((campaign / "S4_VERIFIER_ADMISSION.json").read_text())
            unit = admission["unit_specs"][spec["unit_id"]]
            if spec.get("deps") != [unit["parent_unit_id"], unit["control_v2_unit_id"]]:
                return "s4_parent_dependencies_changed"
            for role, parent_id, queue_sha, index_sha, basename in (
                ("fit", unit["parent_unit_id"], unit["parent_queue_receipt_sha256"],
                 unit["attempt_index_sha256"] if unit["kind"] == "B_SCREEN" else
                 unit["anchor_receipt_sha256"],
                 "ATTEMPT_INDEX.json" if unit["kind"] == "B_SCREEN" else "ANCHOR_RECEIPT.json"),
                ("v2", unit["control_v2_unit_id"], unit["control_v2_queue_receipt_sha256"],
                 unit["control_v2_index_sha256"] if unit["kind"] == "B_SCREEN" else
                 unit["control_v2_receipt_sha256"],
                 "REAL_CONTROL_V2_INDEX.json" if unit["kind"] == "B_SCREEN" else
                 "V2_ANCHOR_PROBE_RECEIPT.json"),
            ):
                parent = state["units"].get(parent_id)
                receipt_path = self.root / "units" / parent_id / "RECEIPT.json"
                if (not parent or parent.get("status") != "done" or
                    parent.get("returncode") != 0 or
                    not receipt_path.is_file() or sha256_file(receipt_path) != queue_sha):
                    return f"s4_{role}_parent_not_complete"
                receipt = json.loads(receipt_path.read_text())
                outputs = parent["spec"].get("outputs", [])
                if (receipt.get("unit_id") != parent_id or receipt.get("status") != "done" or
                    receipt.get("returncode") != 0 or len(outputs) != 1 or
                    receipt.get("output_sha256") != parent.get("output_sha256") or
                    Path(outputs[0]).name != basename or not Path(outputs[0]).is_absolute() or
                    ".private" not in Path(outputs[0]).parts or
                    not Path(outputs[0]).is_file() or
                    sha256_file(Path(outputs[0])) != index_sha or
                    parent.get("output_sha256", {}).get(outputs[0]) != index_sha):
                    return f"s4_{role}_parent_output_changed"
                parent_spec = parent["spec"]
                expected_stage = ("screen" if unit["kind"] == "B_SCREEN" else
                                  "feature_anchor") if role == "fit" else "postrelease_control_v2"
                expected_family = ("B" if unit["kind"] == "B_SCREEN" else "SHARED")
                if (parent_spec.get("unit_id") != parent_id or
                    parent_spec.get("family") != expected_family or
                    parent_spec.get("stage") != expected_stage or
                    parent_spec.get("seed") != unit["seed"] or
                    parent_spec.get("arm_id") != unit["arm"]):
                    return f"s4_{role}_parent_identity_changed"
                lock_sha = (unit["fit_lock_sha256"] if role == "fit" else
                            unit["control_v2_lock_sha256"])
                lock_commit = (unit["fit_lock_commit_sha"] if role == "fit" else
                               unit["control_v2_lock_commit_sha"])
                lock_name = "CAMPAIGN_LOCK.json" if role == "fit" else "CONTROL_V2_LOCK.json"
                blob = subprocess.run(["git", "-C", str(repo), "show",
                                       f"{lock_commit}:results/pcrl_method_portfolio_v1/{lock_name}"],
                                      capture_output=True, check=False, timeout=10)
                ancestor = subprocess.run(["git", "-C", str(repo), "merge-base",
                                          "--is-ancestor", lock_commit, "HEAD"],
                                         capture_output=True, check=False, timeout=10)
                if (blob.returncode or ancestor.returncode or
                    hashlib.sha256(blob.stdout).hexdigest() != lock_sha):
                    return "s4_historical_lock_blob_changed"
                historical = json.loads(blob.stdout)
                expected_script_sha = (
                    historical["families"]["B"]["screen_driver_sha256"] if role == "fit" and
                    unit["kind"] == "B_SCREEN" else
                    historical["shared_anchor_runner_sha256"] if role == "fit" else
                    historical["probe_runner_sha256"] if unit["kind"] == "B_SCREEN" else
                    historical["anchor_probe_runner_sha256"])
                argv = parent_spec.get("argv", [])
                if (len(argv) < 2 or not Path(argv[1]).is_absolute() or
                    not Path(argv[1]).is_file() or sha256_file(Path(argv[1])) != expected_script_sha):
                    return f"s4_{role}_parent_script_changed"
            if any(Path(path).exists() for path in spec["outputs"]):
                return "s4_report_already_exists"
            return None
        except (OSError, ValueError, TypeError, KeyError, AttributeError,
                subprocess.TimeoutExpired):
            return "s4_parent_identity_invalid"

    @classmethod
    def _real_admission_reason(cls, gate: dict, spec: dict) -> str | None:
        if spec.get("stage") == "postrelease_control_v2":
            return cls._v2_admission_reason(gate, spec)
        if not gate.get("safe_inputs") or not cls._local_seal_ok(gate):
            return "local_seal_invalid"
        lock = json.loads((Path(gate["local_seal_path"]).parent / "CAMPAIGN_LOCK.json").read_text())
        family = spec.get("family")
        arm_id = spec.get("arm_id")
        seed = spec.get("seed")
        if type(seed) is not int or seed not in (0, 1):
            return "seed_not_admitted"

        def exact_script(relative_path: str, expected_sha: object) -> bool:
            script = Path(spec["cwd"]) / relative_path
            try:
                script = script.resolve(strict=True)
            except OSError:
                return False
            if (not isinstance(expected_sha, str) or
                not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or
                sha256_file(script) != expected_sha):
                return False
            argv = spec.get("argv", ())
            return (len(argv) >= 2 and
                    re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(argv[0]).name) is not None and
                    (Path(spec["cwd"]) / argv[1]).resolve() == script)

        def flag_value(flag: str) -> str | None:
            argv = spec.get("argv", ())
            values = []
            for i, arg in enumerate(argv[2:], start=2):
                if arg == flag and i + 1 < len(argv):
                    values.append(argv[i + 1])
                elif arg.startswith(flag + "="):
                    values.append(arg[len(flag) + 1:])
            return values[0] if len(values) == 1 else None

        if spec.get("stage") == "a_score_bank_preprocess":
            releases = ("tstar", "pstar", "lra_seq21", "hcal_local_d0", "hcal_seq12_d0")
            release = spec.get("release_key")
            expected_ids = [f"S2-A-SCORE-BANK-{key.upper().replace('_', '-')}-S{s}"
                            for key in releases for s in (0, 1)]
            uid = (f"S2-A-SCORE-BANK-{release.upper().replace('_', '-')}-S{seed}"
                   if isinstance(release, str) else None)
            family_a = lock.get("families", {}).get("A", {})
            if (lock.get("fit_admission") != "B_ONLY" or
                family_a.get("preprocessing_admission") != "ALLOWED" or
                family_a.get("fit_admission") != "DENIED" or
                lock.get("admitted_a_score_bank_unit_ids") != expected_ids or
                release not in releases or spec.get("unit_id") != uid or
                spec.get("family") != "SHARED" or spec.get("team") != "SHARED" or
                spec.get("arm_id") != "A_SCORE_BANK_PREPROCESS" or
                spec.get("arm") != "A_SCORE_BANK_PREPROCESS" or
                spec.get("mode") != "real_data" or
                spec.get("data_access") != "A_ROLE_ONLY_REFERENCES_PREPROCESS" or
                spec.get("deps") != []):
                return "a_score_bank_scope_invalid"
            caps = lock.get("a_score_bank_unit_caps")
            fixed_cap = {"max_wall_s": 300, "cpu_reserve_s": 300, "max_rss_gib": 1.5}
            if (not isinstance(caps, dict) or set(caps) != set(expected_ids) or
                any(caps.get(unit_id) != fixed_cap for unit_id in expected_ids)):
                return "a_score_bank_caps_not_locked"
            if (float(spec["max_wall_s"]) > caps[uid]["max_wall_s"] or
                float(spec["cpu_reserve_s"]) > caps[uid]["cpu_reserve_s"] or
                float(spec["max_rss_gib"]) > caps[uid]["max_rss_gib"]):
                return "a_score_bank_resource_limit_exceeded"
            script_sha = lock.get("a_score_bank_runner_sha256")
            if (spec.get("runner_sha256") != script_sha or
                not exact_script("results/pcrl_method_portfolio_v1/shared/admit_a_score_bank.py", script_sha)):
                return "a_score_bank_script_invalid"
            campaign = Path(gate["local_seal_path"]).parent
            lock_path = campaign / "CAMPAIGN_LOCK.json"
            manifest = campaign / "DATA_ACCESS_MANIFEST.json"
            output = (campaign / "shared/.private" / f"{release}_score_bank_s{seed}_v1" /
                      "BANK_ADMISSION_RECEIPT.json")
            cache_rel = lock.get("a_score_bank_cache_root")
            if not isinstance(cache_rel, str) or Path(cache_rel).is_absolute():
                return "a_score_bank_cache_invalid"
            try:
                cache = (Path(spec["cwd"]) / cache_rel).resolve(strict=True)
                argv_cache = Path(flag_value("--cache-root") or "").resolve(strict=True)
            except (OSError, TypeError):
                return "a_score_bank_cache_invalid"
            if (not cache.is_dir() or argv_cache != cache or
                Path(spec.get("cache_root", "")).resolve() != cache or
                spec.get("outputs") != [str(output)] or output.exists() or
                not manifest.is_file() or
                sha256_file(manifest) != lock.get("data_admission", {}).get("manifest_sha256") or
                flag_value("--seed") != str(seed) or flag_value("--release") != release or
                flag_value("--campaign-lock-sha256") != sha256_file(lock_path) or
                flag_value("--queue-root") != str(campaign / "shared/.private/compute")):
                return "a_score_bank_input_or_output_invalid"
            return None

        if family == "SHARED":
            expected_id = f"S2-FEATURE-REF-RAW83-U-S{seed}"
            if (arm_id != "FEATURE_REF_RAW83_U" or spec.get("arm") != arm_id or
                spec.get("team") != "SHARED" or spec.get("stage") != "feature_anchor" or
                spec.get("unit_id") != expected_id or
                expected_id not in lock.get("admitted_shared_unit_ids", [])):
                return "shared_unit_not_admitted"
            if not exact_script("results/pcrl_method_portfolio_v1/shared/run_feature_ref_raw83_u.py",
                                lock.get("shared_anchor_runner_sha256")):
                return "shared_script_not_admitted"
            if flag_value("--seed") != str(seed):
                return "shared_seed_argv_mismatch"
            return None
        if family not in ("A", "B", "C", "D") or lock["families"][family].get("fit_admission") != "ALLOWED":
            return "family_not_admitted"
        if arm_id not in lock.get("admitted_arms", {}).get(family, []):
            return "arm_not_admitted"
        if spec.get("team") != family or spec.get("arm") != arm_id:
            return "arm_alias_mismatch"
        if family == "B" and lock.get("fit_admission") == "B_ONLY":
            b = lock["families"]["B"]
            stage = spec.get("stage")
            if stage == "screen":
                unit_map = b.get("screen_unit_ids", {})
                script_name = "screen_driver.py"
                script_sha = b.get("screen_driver_sha256")
            elif stage == "postrelease_control":
                unit_map = b.get("real_control_unit_ids", {})
                script_name = "real_control_probe.py"
                script_sha = b.get("real_control_probe_sha256")
            else:
                return "stage_not_admitted"
            if (not isinstance(unit_map, dict) or
                unit_map.get(f"{arm_id}:s{seed}") != spec.get("unit_id")):
                return "unit_id_not_admitted"
            if not exact_script(f"results/pcrl_method_portfolio_v1/teams/b/{script_name}", script_sha):
                return "script_not_admitted"
            if (flag_value("--arm") != arm_id or flag_value("--seed") != str(seed) or
                flag_value("--unit-id") != spec.get("unit_id")):
                return "unit_argv_mismatch"
        return None

    @classmethod
    def _v2_admission_reason(cls, gate: dict, spec: dict) -> str | None:
        """Admit only one exact, sealed B or shared-anchor V2 probe unit."""
        if not gate.get("safe_inputs") or not cls._local_v2_seal_ok(gate):
            return "v2_local_seal_invalid"
        lock_path = Path(gate["v2_local_seal_path"]).parent / "CONTROL_V2_LOCK.json"
        lock = json.loads(lock_path.read_text())
        family = spec.get("family")
        arm = spec.get("arm_id")
        seed = spec.get("seed")
        uid = spec.get("unit_id")
        if (family not in ("B", "SHARED") or spec.get("team") != family or
            not isinstance(arm, str) or spec.get("arm") != arm or
            type(seed) is not int or seed not in (0, 1) or
            spec.get("stage") != "postrelease_control_v2" or
            spec.get("mode") != "real_data" or
            spec.get("data_access") != "ATTACK_INNER_FEATURES_AND_SEX_V2_CONTROL"):
            return "v2_scope_invalid"
        key = f"{arm}:s{seed}"
        target = lock.get("unit_specs", {}).get(uid)
        if (lock.get("probe_unit_ids", {}).get(key) != uid or not isinstance(target, dict)):
            return "v2_unit_not_admitted"

        argv = spec.get("argv", ())
        def flag(name: str) -> str | None:
            values = []
            for i, arg in enumerate(argv[2:], start=2):
                if arg == name and i + 1 < len(argv):
                    values.append(argv[i + 1])
                elif arg.startswith(name + "="):
                    values.append(arg[len(name) + 1:])
            return values[0] if len(values) == 1 else None

        def exact_script(relative: str, expected_sha: object) -> bool:
            try:
                path = (Path(spec["cwd"]) / relative).resolve(strict=True)
                actual = (Path(spec["cwd"]) / argv[1]).resolve(strict=True)
                return (len(argv) >= 2 and actual == path and
                        re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(argv[0]).name) is not None and
                        isinstance(expected_sha, str) and sha256_file(path) == expected_sha)
            except (OSError, IndexError, TypeError):
                return False

        receipt_sha = gate["v2_local_seal_sha256"]
        control_sha = sha256_file(lock_path)
        if family == "B":
            if (not arm.startswith("B-") or
                not exact_script("results/pcrl_method_portfolio_v1/teams/b/real_control_probe_v2.py",
                                 lock.get("probe_runner_sha256"))):
                return "v2_script_not_admitted"
            parent_id = target.get("screen_unit_id")
            if (target.get("arm") != arm or target.get("seed") != seed or
                target.get("stage") != "postrelease_control_v2" or
                target.get("v1_probe_index_sha256", object()) != spec.get("v1_probe_index_sha256", object()) or
                spec.get("parent_unit_id") != parent_id or
                flag("--arm") != arm or flag("--seed") != str(seed) or
                flag("--unit-id") != uid or flag("--screen-unit-id") != parent_id or
                flag("--attempt-index-sha256") != spec.get("parent_attempt_index_sha256") or
                flag("--parent-fit-lock-sha256") != spec.get("parent_fit_lock_sha256") or
                flag("--control-v2-lock-sha256") != control_sha or
                flag("--v2-local-seal-sha256") != receipt_sha or
                flag("--control-v2-lock") != str(lock_path) or
                flag("--v2-local-seal-receipt") != gate["v2_local_seal_path"] or
                flag("--config-sha256") != lock.get("screen_config_sha256") or
                flag("--manifest-sha256") != lock.get("data_access_manifest_sha256")):
                return "v2_argv_or_target_mismatch"
            expected_output = f"real_probe_v2__{arm}__s{seed}__{parent_id}"
            outputs = spec.get("outputs", [])
            private_root = Path(spec["cwd"]) / "results/pcrl_method_portfolio_v1/teams/b/.private"
            if (len(outputs) != 1 or Path(outputs[0]).name != "REAL_CONTROL_V2_INDEX.json" or
                Path(outputs[0]).parent.name != expected_output or
                Path(outputs[0]).parent.parent.resolve() != private_root.resolve() or
                ".private" not in Path(outputs[0]).parts or
                flag("--private-output") != str(Path(outputs[0]).parent) or
                flag("--v1-probe-index-sha256") != target.get("v1_probe_index_sha256")):
                return "v2_output_or_v1_mismatch"
        else:
            if (arm != "FEATURE_REF_RAW83_U" or
                not exact_script("results/pcrl_method_portfolio_v1/shared/run_feature_ref_raw83_u_v2_probe_r2.py",
                                 lock.get("anchor_probe_runner_sha256"))):
                return "v2_anchor_script_not_admitted"
            parent_id = f"S2-FEATURE-REF-RAW83-U-S{seed}"
            expected = {"family": "SHARED", "arm": arm, "seed": seed,
                        "stage": "postrelease_control_v2", "parent_unit_id": parent_id}
            output = lock_path.parent / "shared/.private" / f"feature_ref_raw83_u_s{seed}_v2_probe_r2" / "V2_ANCHOR_PROBE_RECEIPT.json"
            if (target != expected or spec.get("parent_unit_id") != parent_id or
                spec.get("outputs") != [str(output)] or
                spec.get("runner_sha256") != lock.get("anchor_probe_runner_sha256") or
                flag("--seed") != str(seed) or
                flag("--unit-id") != uid or
                flag("--control-lock-sha256") != control_sha or
                flag("--control-seal-sha256") != receipt_sha or
                flag("--data-manifest-sha256") != lock.get("data_access_manifest_sha256") or
                flag("--parent-anchor-receipt-sha256") != spec.get("parent_anchor_receipt_sha256")):
                return "v2_anchor_target_mismatch"
        if spec.get("deps") != [parent_id]:
            return "v2_parent_dependency_mismatch"
        return None

    def _v2_parent_reason(self, state: dict, spec: dict) -> str | None:
        if spec.get("stage") != "postrelease_control_v2":
            return None
        parent_id = spec.get("parent_unit_id")
        parent = state["units"].get(parent_id)
        if (not parent or parent.get("status") != "done" or parent.get("returncode") != 0 or
            parent.get("spec", {}).get("seed") != spec.get("seed") or
            spec.get("deps") != [parent_id]):
            return "v2_parent_not_complete"
        receipt = self.root / "units" / parent_id / "RECEIPT.json"
        if (not receipt.is_file() or
            sha256_file(receipt) != spec.get("parent_receipt_sha256")):
            return "v2_parent_receipt_mismatch"
        lock_path = Path(state["gate"]["v2_local_seal_path"]).parent / "CONTROL_V2_LOCK.json"
        lock = json.loads(lock_path.read_text())
        if lock.get("parent_queue_receipt_sha256_by_unit", {}).get(parent_id) != spec.get("parent_receipt_sha256"):
            return "v2_parent_receipt_not_locked"
        outputs = parent["spec"].get("outputs", [])
        if len(outputs) != 1 or not Path(outputs[0]).is_file():
            return "v2_parent_output_missing"
        output_sha = sha256_file(Path(outputs[0]))
        if parent.get("output_sha256", {}).get(outputs[0]) != output_sha:
            return "v2_parent_output_changed"
        if spec.get("family") == "B":
            key = f"{spec['arm_id']}:s{spec['seed']}"
            if (parent["spec"].get("team") != "B" or parent["spec"].get("stage") != "screen" or
                parent["spec"].get("arm") != spec["arm_id"] or
                lock.get("parent_attempt_index_sha256", {}).get(key) != output_sha or
                spec.get("parent_attempt_index_sha256") != output_sha):
                return "v2_parent_attempt_mismatch"
            child_argv = spec.get("argv", [])
            attempt_args = [child_argv[i + 1] for i, arg in enumerate(child_argv[:-1]) if arg == "--attempt"]
            if attempt_args != [str(Path(outputs[0]).parent)]:
                return "v2_parent_attempt_path_mismatch"
            parent_flag = "--lock-sha256"
        else:
            if (parent["spec"].get("team") != "SHARED" or parent["spec"].get("stage") != "feature_anchor" or
                lock.get("parent_anchor_receipt_sha256_by_unit", {}).get(parent_id) != output_sha or
                spec.get("parent_anchor_receipt_sha256") != output_sha):
                return "v2_parent_anchor_mismatch"
            try:
                parent_reader_shas = json.loads(Path(outputs[0]).read_text()).get("reader_receipt_sha256")
            except (OSError, ValueError, TypeError, AttributeError):
                return "v2_parent_reader_receipts_mismatch"
            child_reader_shas = spec.get("reader_receipt_sha256")
            if (not isinstance(parent_reader_shas, dict) or
                set(parent_reader_shas) != {"r1", "r2", "pair"} or
                any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in parent_reader_shas.values()) or
                child_reader_shas != parent_reader_shas):
                return "v2_parent_reader_receipts_mismatch"
            parent_flag = "--campaign-lock-sha256"
        argv = parent["spec"].get("argv", [])
        matches = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == parent_flag]
        if (len(matches) != 1 or matches[0] not in lock.get("accepted_fit_lock_sha256", []) or
            spec.get("parent_fit_lock_sha256") != matches[0]):
            return "v2_parent_fit_lock_mismatch"
        child_argv = spec.get("argv", [])
        roots = [child_argv[i + 1] for i, arg in enumerate(child_argv[:-1]) if arg == "--queue-root"]
        if roots != [str(self.root)]:
            return "v2_queue_root_mismatch"
        if any(Path(path).exists() for path in spec.get("outputs", [])):
            return "v2_output_already_exists"
        return None

    def _gate_status(self, state: dict, spec: dict) -> str | None:
        gate = state["gate"]
        if gate["manually_closed"] or not gate["occupancy_confirmed"]:
            return "gate_closed"
        checked = gate["occupancy_checked_utc"]
        if not checked or (dt.datetime.now(dt.timezone.utc) - parse_utc(checked)).total_seconds() > OCCUPANCY_TTL_S:
            return "occupancy_stale"
        if gate["external_heavy"] >= SLOTS:
            return "external_capacity_full"
        epoch = state.get("budget_epoch")
        epoch_start = epoch["start_utc"] if epoch else state["campaign_start_utc"]
        epoch_wall_budget = epoch["budget_wall_s"] if epoch else state["budget_wall_s"]
        epoch_cpu_budget = epoch["budget_cpu_s"] if epoch else state["budget_cpu_s"]
        epoch_cpu_used = epoch["used_cpu_s"] if epoch else state["used_cpu_s"]
        if (dt.datetime.now(dt.timezone.utc) - parse_utc(epoch_start)).total_seconds() >= epoch_wall_budget:
            return "wall_budget_exhausted"
        if epoch_cpu_used >= epoch_cpu_budget:
            return "cpu_budget_exhausted"
        if self._free_disk_gib() < state["disk_floor_gib"]:
            return "disk_floor_reached"
        if spec.get("mode", "real_data") == "role_export":
            reviewed = Path(spec["reviewed_script"])
            if (not gate.get("export_authorized") or
                gate.get("export_reviewed_sha256") != spec.get("reviewed_sha256") or
                not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256")):
                return "export_review_missing"
        elif spec.get("mode", "real_data") == "role_audit":
            reviewed = Path(spec["reviewed_script"])
            if (not gate.get("audit_authorized") or
                gate.get("audit_reviewed_sha256") != spec.get("reviewed_sha256") or
                not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256")):
                return "audit_review_missing"
            if state["units"].get("S2-SAFE-ROLE-EXPORT-V1", {}).get("status") != "done":
                return "export_not_complete"
        elif spec.get("mode", "real_data") == "role_replay":
            reviewed = Path(spec["reviewed_script"])
            if (not gate.get("replay_authorized") or
                gate.get("replay_reviewed_sha256") != spec.get("reviewed_sha256") or
                not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256")):
                return "replay_review_missing"
            if state["units"].get("S4-INDEPENDENT-ROLE-AUDIT-V1", {}).get("status") != "done":
                return "role_audit_not_complete"
        elif spec.get("mode", "real_data") == "role_export_replay":
            reviewed = Path(spec["reviewed_script"])
            first = state["units"].get("S2-SAFE-ROLE-EXPORT-V1", {})
            if (not gate.get("export_replay_authorized") or
                gate.get("export_replay_reviewed_sha256") != spec.get("reviewed_sha256") or
                first.get("status") != "done" or
                first.get("spec", {}).get("reviewed_sha256") != spec.get("reviewed_sha256") or
                not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256")):
                return "export_replay_review_missing"
            if (not Path(spec["preserved_first_path"]).is_dir() or
                Path(spec["replay_output_path"]).exists()):
                return "first_export_not_preserved"
        elif spec.get("mode", "real_data") == "role_compare":
            reviewed = Path(spec["reviewed_script"])
            if (not gate.get("compare_authorized") or
                gate.get("compare_reviewed_sha256") != spec.get("reviewed_sha256") or
                not reviewed.is_file() or sha256_file(reviewed) != spec.get("reviewed_sha256")):
                return "compare_review_missing"
            if state["units"].get("S2-SAFE-ROLE-EXPORT-REPLAY-V1", {}).get("status") != "done":
                return "second_export_not_complete"
        elif spec.get("mode", "real_data") == "role_probe":
            if state["units"].get("S4-INDEPENDENT-ROLE-AUDIT-V1", {}).get("status") != "done":
                return "role_audit_not_complete"
            script = Path(spec["reviewed_script"])
            if not script.is_file() or sha256_file(script) != spec.get("reviewed_sha256"):
                return "role_probe_script_changed"
            manifest = Path(spec.get("admission_manifest_path", ""))
            audit = Path(spec.get("s4_audit_path", ""))
            if (not manifest.is_file() or not audit.is_file() or
                manifest.name != "DATA_ACCESS_MANIFEST.json" or audit.name != "S4_ROLE_EXPORT_VERIFICATION.json" or
                manifest.parent != audit.parent or
                sha256_file(manifest) != spec.get("admission_manifest_sha256") or
                sha256_file(audit) != spec.get("s4_audit_sha256")):
                return "role_probe_admission_changed"
            data = json.loads(manifest.read_text())
            review = json.loads(audit.read_text())
            sex = data.get("real_file_specs", {}).get("ATTACK_FIT_NEW:sex", {})
            sex_path = manifest.parent / sex.get("path", "")
            if (data.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED" or review.get("status") != "PASS" or
                sex.get("role") != "ATTACK_FIT_NEW" or sex.get("field") != "sex" or
                set(sex.get("members", [])) != {"row_id", "unit", "sex"} or
                not sex_path.is_file() or sha256_file(sex_path) != sex.get("sha256") or
                data.get("roles", {}).get("ATTACK_FIT_NEW", {}).get("rows") != review.get("roles", {}).get("ATTACK_FIT_NEW", {}).get("rows")):
                return "role_probe_admission_invalid"
            lock_path = gate.get("phase1_lock_path")
            lock_sha = gate.get("phase1_lock_sha256")
            if not lock_path or not Path(lock_path).is_file() or sha256_file(Path(lock_path)) != lock_sha:
                return "phase1_lock_unverified"
            index = json.loads(Path(lock_path).read_text())
            sources = index.get("source_code_hashes", {}).get(spec.get("team"), [])
            if not any((script.resolve() == (Path(spec["cwd"]) / "results/pcrl_method_portfolio_v1" / entry.get("path", "")).resolve()
                        and entry.get("sha256") == spec.get("reviewed_sha256")) for entry in sources):
                return "role_probe_unregistered"
            for source in sources:
                path = Path(spec["cwd"]) / "results/pcrl_method_portfolio_v1" / source.get("path", "")
                if not path.is_file() or sha256_file(path) != source.get("sha256"):
                    return "phase1_source_changed"
        elif spec.get("mode", "real_data") in ("prelock_synthetic", "preseal_smoke"):
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
            for source in lock.get("source_code_hashes", {}).get(spec.get("team"), []):
                source_path = Path(spec["cwd"]) / "results" / "pcrl_method_portfolio_v1" / source.get("path", "")
                if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
                    return "phase1_source_changed"
        elif spec.get("mode", "real_data") == "verifier_synthetic":
            return self._verifier_synthetic_reason(spec)
        elif spec.get("mode") == "s4_adult_replay":
            reason = self._s4_admission_reason(gate, spec)
            return reason or self._s4_parent_reason(state, spec)
        elif spec.get("mode", "real_data") == "real_data":
            reason = self._real_admission_reason(gate, spec)
            return reason or self._v2_parent_reason(state, spec)
        return None

    def _orphan_unresolved(self, state: dict) -> bool:
        """Fail closed if a recorded running unit has lost its slot lock."""
        for unit in state["units"].values():
            if unit["status"] != "running":
                continue
            slot = unit.get("slot")
            if slot not in range(SLOTS):
                return True
            with (self.root / f"slot{slot}.lock").open("a+") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                else:
                    fcntl.flock(probe, fcntl.LOCK_UN)
                    return True
        return False

    def recover_orphan(self, unit_id: str, note: str) -> dict:
        """Close a dead own-worker claim after proving slot and process group are gone.

        Recovery never reruns the original unit and charges its full CPU reserve.
        The caller must separately register a new unit for any retry.
        """
        if not note.strip():
            raise ValueError("orphan recovery needs an evidence note")

        def alive(pid: int | None, *, group: bool = False) -> bool:
            if not isinstance(pid, int) or pid <= 0:
                return False
            try:
                if group:
                    os.killpg(pid, 0)
                else:
                    os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True

        with self._mutex():
            state = self._read()
            unit = state["units"].get(unit_id)
            if not unit or unit["status"] != "running":
                raise ValueError("unit is not an unresolved running claim")
            slot = unit.get("slot")
            if slot not in range(SLOTS):
                raise RuntimeError("invalid slot claim needs separate investigation")
            with (self.root / f"slot{slot}.lock").open("a+") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("slot or child still holds lock") from exc
                finally:
                    # The context closes the probe; no external process is signalled.
                    pass
                fcntl.flock(probe, fcntl.LOCK_UN)
            if alive(unit.get("worker_pid")) or alive(unit.get("child_pgid"), group=True):
                raise RuntimeError("owned worker or child process group still exists")
            charge = float(unit["spec"]["cpu_reserve_s"])
            receipt = {"unit_id": unit_id, "status": "incomplete", "reason": "orphan_recovered",
                       "evidence_note": note, "charged_cpu_s": charge,
                       "recovered_utc": utc_now(), "slot": slot,
                       "worker_pid": unit.get("worker_pid"),
                       "child_pgid": unit.get("child_pgid")}
            unit.update(receipt)
            state["used_cpu_s"] += charge
            if "budget_epoch" in state:
                state["budget_epoch"]["used_cpu_s"] += charge
            self._save(state)
            self._receipt("orphan_recovery", **receipt)
            atomic_json(self.root / "units" / unit_id / "RECOVERY.json", receipt)
            return receipt

    def run_once(self) -> dict:
        # A slot is held by flock until the launched process group has exited.
        with self._mutex():
            state = self._read()
            if self._orphan_unresolved(state):
                return {"status": "orphan_unresolved"}
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
                if self._orphan_unresolved(state):
                    return {"status": "orphan_unresolved"}
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
                    epoch = state.get("budget_epoch")
                    epoch_start = epoch["start_utc"] if epoch else state["campaign_start_utc"]
                    epoch_cpu_used = epoch["used_cpu_s"] if epoch else state["used_cpu_s"]
                    epoch_cpu_budget = epoch["budget_cpu_s"] if epoch else state["budget_cpu_s"]
                    epoch_wall_budget = epoch["budget_wall_s"] if epoch else state["budget_wall_s"]
                    elapsed = (dt.datetime.now(dt.timezone.utc) - parse_utc(epoch_start)).total_seconds()
                    if epoch_cpu_used + sum(float(u["spec"]["cpu_reserve_s"]) for u in state["units"].values() if u["status"] == "running") + float(spec["cpu_reserve_s"]) > epoch_cpu_budget:
                        continue
                    if elapsed + float(spec["max_wall_s"]) > epoch_wall_budget:
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
            return self._run_unit(uid, unit["spec"], slot, slot_handle)
        finally:
            fcntl.flock(slot_handle, fcntl.LOCK_UN)
            slot_handle.close()

    def _run_unit(self, uid: str, spec: dict, slot: int, slot_handle) -> dict:
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
                                           start_new_session=True, pass_fds=(slot_handle.fileno(),))
                with self._mutex():
                    state = self._read()
                    state["units"][uid]["child_pgid"] = process.pid
                    self._save(state)
                    self._receipt("child_started", unit_id=uid, child_pgid=process.pid)
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
                "maxrss_scope": "wrapper_child_maximum; cumulative upper bound when run_once reused in one wrapper process",
                "output_sha256": output_hashes,
                "timed_out": timed_out,
                "interrupted": interrupted,
                "finished_utc": utc_now(),
            }
            with self._mutex():
                state = self._read()
                state["used_cpu_s"] += charge_s
                if "budget_epoch" in state:
                    state["budget_epoch"]["used_cpu_s"] += charge_s
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
    gate.add_argument("--export-authorized", action="store_true")
    gate.add_argument("--export-reviewed-sha256")
    gate.add_argument("--audit-authorized", action="store_true")
    gate.add_argument("--audit-reviewed-sha256")
    gate.add_argument("--replay-authorized", action="store_true")
    gate.add_argument("--replay-reviewed-sha256")
    gate.add_argument("--export-replay-authorized", action="store_true")
    gate.add_argument("--export-replay-reviewed-sha256")
    gate.add_argument("--compare-authorized", action="store_true")
    gate.add_argument("--compare-reviewed-sha256")
    gate.add_argument("--local-seal-path")
    gate.add_argument("--local-seal-sha256")
    gate.add_argument("--v2-local-seal-path")
    gate.add_argument("--v2-local-seal-sha256")
    gate.add_argument("--s4-local-seal-path")
    gate.add_argument("--s4-local-seal-sha256")
    sub.add_parser("run-once")
    charge = sub.add_parser("charge")
    charge.add_argument("deviation_id")
    charge.add_argument("charged_cpu_s", type=float)
    charge.add_argument("note")
    recovery = sub.add_parser("recover-orphan")
    recovery.add_argument("unit_id")
    recovery.add_argument("note")
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
                                   phase1_lock_sha256=args.phase1_lock_sha256,
                                   export_authorized=args.export_authorized,
                                   export_reviewed_sha256=args.export_reviewed_sha256,
                                   audit_authorized=args.audit_authorized,
                                   audit_reviewed_sha256=args.audit_reviewed_sha256,
                                   replay_authorized=args.replay_authorized,
                                   replay_reviewed_sha256=args.replay_reviewed_sha256,
                                   export_replay_authorized=args.export_replay_authorized,
                                   export_replay_reviewed_sha256=args.export_replay_reviewed_sha256,
                                   compare_authorized=args.compare_authorized,
                                   compare_reviewed_sha256=args.compare_reviewed_sha256,
                                   local_seal_path=args.local_seal_path,
                                   local_seal_sha256=args.local_seal_sha256,
                                   v2_local_seal_path=args.v2_local_seal_path,
                                   v2_local_seal_sha256=args.v2_local_seal_sha256,
                                   s4_local_seal_path=args.s4_local_seal_path,
                                   s4_local_seal_sha256=args.s4_local_seal_sha256)
    elif args.action == "charge":
        result = queue.record_deviation(args.deviation_id, args.charged_cpu_s, args.note)
    elif args.action == "recover-orphan":
        result = queue.recover_orphan(args.unit_id, args.note)
    else:
        result = queue.run_once()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

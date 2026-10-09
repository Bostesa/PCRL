"""Durable focused-study queue using the portfolio's global slot-0 flock.

One observed external heavy process reserves the other host slot. This is a
software gate and does not provide operating-system filesystem isolation.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import psutil

CPU_BUDGET_S = 72000
WALL_BUDGET_S = 36000
THREAD_ENV = {k: "1" for k in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
ROLE_FREE_STAGES = frozenset(("simultaneous_inference", "independent_aggregate"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, sort_keys=True, separators=(",", ":"))
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _deny_child_processes() -> None:
    # Darwin enforces this per-worker limit; threads remain available. The
    # wrapper itself keeps its normal limit, so unrelated sessions are intact.
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


class FocusedQueue:
    def __init__(self, root: str | Path, shared_slot: str | Path, *, epoch_start_utc: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.slot = Path(shared_slot).resolve(strict=True)
        self.state_path = self.root / "QUEUE.json"
        self.mutex_path = self.root / "QUEUE.lock"
        self.receipts_path = self.root / "RECEIPTS.jsonl"
        self.spec_dir = self.root / "specs"
        self.spec_dir.mkdir(exist_ok=True)
        start = dt.datetime.fromisoformat(epoch_start_utc.replace("Z", "+00:00"))
        if start.tzinfo is None:
            raise ValueError("epoch start requires UTC offset")
        self.start = start.astimezone(dt.timezone.utc)
        with self._mutex():
            if self.state_path.exists():
                state = self._read()
                if (state.get("epoch_start_utc") != self.start.isoformat() or
                        state.get("global_slot") != str(self.slot)):
                    raise ValueError("existing queue epoch or slot mismatch")
            else:
                _atomic_json(self.state_path, {
                    "schema": "pcrl-focused-queue-v1", "created_utc": _utc(),
                    "epoch_start_utc": self.start.isoformat(),
                    "budget_cpu_s": CPU_BUDGET_S, "budget_wall_s": WALL_BUDGET_S,
                    "used_cpu_s": 0.0, "global_slot": str(self.slot),
                    "gate": {"synthetic": False, "science": False,
                             "external_heavy": 2, "checked_utc": None,
                             "evidence": None, "seal_path": None,
                             "seal_sha256": None}, "units": {}})

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

    def _event(self, event: str, **fields) -> None:
        record = {"at_utc": _utc(), "event": event, **fields}
        with self.receipts_path.open("a") as out:
            out.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            out.flush()
            os.fsync(out.fileno())

    def status(self) -> dict:
        with self._mutex():
            return self._read()

    def carry_forward_archived_budget(self, archived_root: str | Path) -> dict:
        """Charge a fresh sealed queue for every unit in an immutable prior queue.

        The archive remains on disk. A fresh queue can use the same logical
        worker IDs only because its lock, seal and queue generation differ.
        """
        archived = Path(archived_root).resolve(strict=True)
        if archived == self.root or archived.parent != self.root.parent:
            raise ValueError("archive must be a sibling queue generation")
        old_path = archived / "QUEUE.json"
        old_sha = sha256(old_path)
        old = json.loads(old_path.read_text())
        if (old.get("epoch_start_utc") != self.start.isoformat() or
                old.get("global_slot") != str(self.slot) or
                any(row.get("status") in ("RUNNING", "QUEUED")
                    for row in old.get("units", {}).values())):
            raise ValueError("archived queue is not complete in this epoch/slot")
        inherited = old.get("carry_forward", {}).get("charged_cpu_s", 0.0)
        charged = inherited + sum(row["charged_cpu_s"] for row in old["units"].values())
        if abs(charged - old["used_cpu_s"]) > 1e-6:
            raise ValueError("archived CPU ledger does not reconcile")
        with self._mutex():
            state = self._read()
            if state["units"] or state["used_cpu_s"] != 0 or state.get("carry_forward"):
                raise ValueError("budget may only be carried into a fresh queue")
            state["used_cpu_s"] = old["used_cpu_s"]
            state["carry_forward"] = {
                "archived_queue_relative": archived.name,
                "archived_queue_sha256": old_sha,
                "charged_cpu_s": old["used_cpu_s"],
                "unit_count": len(old["units"]),
            }
            _atomic_json(self.state_path, state)
            self._event("carry_forward", **state["carry_forward"])
        return state["carry_forward"]

    def admit_synthetic(self, *, external_heavy: int, evidence: str) -> None:
        if external_heavy != 1 or not evidence:
            raise ValueError("one external heavy reservation and evidence required")
        with self._mutex():
            state = self._read()
            if any(x["status"] == "RUNNING" for x in state["units"].values()):
                raise RuntimeError("cannot change gate while a worker runs")
            state["gate"].update(synthetic=True, external_heavy=1,
                                 checked_utc=_utc(), evidence=evidence)
            _atomic_json(self.state_path, state)
            self._event("synthetic_gate", evidence=evidence)

    @staticmethod
    def _seal_ok(path: Path, expected_sha: str, *, contained_worker: bool = False) -> bool:
        """Verify local hashes; only the parent may spawn Git for blob replay."""
        try:
            if sha256(path) != expected_sha:
                return False
            seal = json.loads(path.read_text())
            study = path.parent
            lock_path = study / "CAMPAIGN_LOCK.json"
            review_path = study / "m5" / "PREFIT_REVIEW.json"
            if (seal.get("status") != "PASS" or
                    seal.get("lock_sha256") != sha256(lock_path) or
                    seal.get("m5_review_sha256") != sha256(review_path)):
                return False
            lock = json.loads(lock_path.read_text())
            review = json.loads(review_path.read_text())
            commit = seal["lock_commit_sha"]
            if (lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE" or
                    lock.get("adult_fit_permission") is not True or
                    review.get("status") != "PASS" or
                    review.get("lock_sha256") != seal["lock_sha256"] or
                    review.get("lock_commit_sha") != commit):
                return False
            repo = study.parents[1]
            unit_registry = study / ".private" / "UNIT_SPECS.json"
            if (sha256(unit_registry) != lock.get("private_unit_registry_sha256") or
                    lock.get("source_map_sha256") != sha256(study / "m1" / "DATA_ACCESS_MANIFEST.json")):
                return False
            for relative, expected in lock.get("governed_source_sha256_by_path", {}).items():
                source = (repo / relative).resolve(strict=True)
                if not source.is_relative_to(repo) or sha256(source) != expected:
                    return False
            for relative, expected in lock.get(
                    "governed_protocol_sha256_by_relative_path", {}).items():
                document = (study / relative).resolve(strict=True)
                if not document.is_relative_to(study) or sha256(document) != expected:
                    return False
            if contained_worker:
                # The queue parent checks the committed blob at admission and
                # again immediately before launch. This contained child may
                # not spawn Git under its intentional RLIMIT_NPROC=(0, 0).
                return resource.getrlimit(resource.RLIMIT_NPROC) == (0, 0)
            relative = lock_path.relative_to(repo)
            shown = subprocess.run(["git", "-C", str(repo), "show",
                                    f"{commit}:{relative}"],
                                   capture_output=True, check=False, timeout=10)
            return (shown.returncode == 0 and
                    hashlib.sha256(shown.stdout).hexdigest() == seal["lock_sha256"])
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
            return False

    def admit_science(self, *, seal_path: str | Path, seal_sha256: str,
                      external_heavy: int, evidence: str) -> None:
        path = Path(seal_path).resolve(strict=True)
        expected = (self.root.parent.parent / "LOCAL_SEAL_RECEIPT.json").resolve()
        if (path != expected or external_heavy != 1 or not evidence or
                not self._seal_ok(path, seal_sha256)):
            raise ValueError("science seal or occupancy invalid")
        with self._mutex():
            state = self._read()
            if any(x["status"] == "RUNNING" for x in state["units"].values()):
                raise RuntimeError("cannot change gate while a worker runs")
            state["gate"].update(science=True, external_heavy=1,
                                 checked_utc=_utc(), evidence=evidence,
                                 seal_path=str(path), seal_sha256=seal_sha256)
            _atomic_json(self.state_path, state)
            self._event("science_gate", seal_sha256=seal_sha256, evidence=evidence)

    def _gate_fresh(self, state: dict) -> bool:
        try:
            checked = dt.datetime.fromisoformat(state["gate"]["checked_utc"])
            age = (dt.datetime.now(dt.timezone.utc) - checked).total_seconds()
            return state["gate"]["external_heavy"] == 1 and 0 <= age <= 120
        except (KeyError, TypeError, ValueError):
            return False

    def _science_ok(self, state: dict, spec: dict) -> bool:
        gate = state["gate"]
        return (gate["science"] and self._gate_fresh(state) and
                spec.get("seal_sha256") == gate["seal_sha256"] and
                self._seal_ok(Path(gate["seal_path"]), gate["seal_sha256"]))

    def enqueue(self, spec: dict) -> None:
        uid = spec.get("unit_id")
        if not isinstance(uid, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{2,119}", uid):
            raise ValueError("invalid unit_id")
        mode = spec.get("mode")
        if mode not in ("synthetic", "real"):
            raise ValueError("invalid mode")
        with self._mutex():
            state = self._read()
            if mode == "real" and not self._science_ok(state, spec):
                raise ValueError("science gate closed at enqueue")
            if mode == "synthetic" and (not state["gate"]["synthetic"] or
                                        not self._gate_fresh(state)):
                raise ValueError("synthetic gate closed at enqueue")
            if uid in state["units"] or (self.spec_dir / (uid + ".json")).exists():
                raise ValueError("unit identity already registered")
            self._validate_spec(spec, state)
            path = self.spec_dir / (uid + ".json")
            _atomic_json(path, spec)
            state["units"][uid] = {"status": "QUEUED", "spec_path": str(path),
                                   "spec_sha256": sha256(path)}
            _atomic_json(self.state_path, state)
            self._event("enqueue", unit_id=uid, spec_sha256=sha256(path))

    def _validate_spec(self, spec: dict, state: dict) -> None:
        argv = spec.get("argv")
        if not isinstance(argv, list) or len(argv) < 2 or any(not isinstance(x, str) for x in argv):
            raise ValueError("argv invalid")
        script = Path(spec["script"]).resolve(strict=True)
        cwd = Path(spec["cwd"]).resolve(strict=True)
        if (not any(Path(arg).resolve() == script for arg in argv) or
                not script.is_file() or not cwd.is_dir() or
                sha256(script) != spec.get("script_sha256")):
            raise ValueError("script/cwd identity invalid")
        for key, upper in (("cpu_reserve_s", CPU_BUDGET_S),
                           ("wall_limit_s", WALL_BUDGET_S), ("max_rss_gib", 6)):
            value = spec.get(key)
            if not isinstance(value, (int, float)) or not 0 < value <= upper:
                raise ValueError("resource cap invalid: " + key)
        if not isinstance(spec.get("inputs"), dict) or not isinstance(spec.get("outputs"), list):
            raise ValueError("inputs/outputs invalid")
        if not isinstance(spec.get("deps"), list) or any(d not in state["units"] for d in spec["deps"]):
            raise ValueError("dependencies invalid")
        for path, expected in spec["inputs"].items():
            if sha256(Path(path).resolve(strict=True)) != expected:
                raise ValueError("input identity invalid")
        for path in spec["outputs"]:
            output = Path(path)
            if not output.is_absolute():
                raise ValueError("output must be absolute")
            if output.exists():
                raise ValueError("output already exists")
            if any(path in json.loads(Path(rec["spec_path"]).read_text())["outputs"]
                   for rec in state["units"].values()):
                raise ValueError("output collides with registered unit")
            if spec["mode"] == "real" and not output.resolve().is_relative_to(self.root.parent):
                raise ValueError("real output outside study private root")
        if spec["mode"] == "synthetic" and (spec.get("data_access") != "NONE" or spec["inputs"]):
            raise ValueError("synthetic inputs must be empty and data access NONE")
        if spec["mode"] == "real":
            if not spec.get("stage") or not spec.get("source_map_sha256") or \
                    not isinstance(spec.get("role_access"), list):
                raise ValueError("real unit missing stage/role/source map")
            if spec.get("allow_child_processes") is not False:
                raise ValueError("real worker may not spawn child processes")
            study = Path(state["gate"]["seal_path"]).parent
            lock = json.loads((study / "CAMPAIGN_LOCK.json").read_text())
            private_registry = json.loads((study / ".private" / "UNIT_SPECS.json").read_text())
            expected_spec = private_registry.get("unit_specs", {}).get(spec["unit_id"])
            if not isinstance(expected_spec, dict) or expected_spec.get("seal_sha256") != "$LOCAL_SEAL_SHA256":
                raise ValueError("real unit lacks sealed placeholder")
            expected_spec = {**expected_spec, "seal_sha256": state["gate"]["seal_sha256"]}
            if expected_spec != spec:
                raise ValueError("real unit differs from sealed private registry")
            role_inputs = spec.get("role_inputs")
            role_free = spec["stage"] in ROLE_FREE_STAGES
            if not isinstance(role_inputs, dict) or (not role_free and not role_inputs):
                raise ValueError("role input declaration invalid")
            if role_free and (role_inputs or spec["role_access"] or spec["inputs"] or
                              spec.get("data_access") != "NONE_PRIVATE_DERIVED_ONLY"):
                raise ValueError("derived-only stage may not declare role inputs")
            manifest = json.loads((study / "m1" / "DATA_ACCESS_MANIFEST.json").read_text())
            repo = study.parents[1]
            if cwd != repo:
                raise ValueError("real worker cwd must be study worktree root")
            source_root = (repo / lock["prior_package_relative_to_worktree"]).resolve(strict=True)
            for key, actual in role_inputs.items():
                role = key.split(":", 1)[0]
                expected_path = (source_root / manifest["role_file_relative_path"][key]).resolve(strict=True)
                expected_sha = manifest["role_file_sha256"][key]
                role_ok = role in spec["role_access"]
                declaration_ok = actual == {"path": str(expected_path), "sha256": expected_sha}
                input_ok = spec["inputs"].get(str(expected_path)) == expected_sha
                if not (role_ok and declaration_ok and input_ok):
                    raise ValueError(f"role file identity invalid: {role_ok}/{declaration_ok}/{input_ok}")
            if not set(spec["role_access"]).issubset({key.split(":", 1)[0] for key in role_inputs}):
                raise ValueError("role access lacks declared files")
            script_relative = str(script.relative_to(repo)) if script.is_relative_to(repo) else None
            if (spec["source_map_sha256"] != lock.get("source_map_sha256") or
                    not set(spec["role_access"]).issubset(set(lock.get("allowed_roles", []))) or
                    lock.get("governed_source_sha256_by_path", {}).get(script_relative) != spec["script_sha256"]):
                raise ValueError("real source map or role scope invalid")

    @staticmethod
    def _inputs_ok(spec: dict, state: dict) -> bool:
        try:
            if sha256(Path(spec["script"])) != spec["script_sha256"]:
                return False
            for path, expected in spec["inputs"].items():
                if sha256(Path(path)) != expected:
                    return False
            for dep in spec["deps"]:
                old = state["units"].get(dep)
                if not old or old["status"] != "DONE":
                    return False
                for path, expected in old["output_sha256"].items():
                    if sha256(Path(path)) != expected:
                        return False
            return True
        except (OSError, KeyError, TypeError):
            return False

    def run_once(self, *, expected_unit_id: str | None = None) -> dict:
        with self._mutex():
            state = self._read()
            pending = [(uid, rec) for uid, rec in state["units"].items() if rec["status"] == "QUEUED"]
            if not pending:
                return {"status": "IDLE"}
            uid, rec = pending[0]
            if expected_unit_id is not None and uid != expected_unit_id:
                return {"status": "UNEXPECTED_PENDING", "unit_id": uid,
                        "expected_unit_id": expected_unit_id}
            path = Path(rec["spec_path"])
            if sha256(path) != rec["spec_sha256"]:
                return {"status": "SPEC_IDENTITY_CHANGED", "unit_id": uid}
            spec = json.loads(path.read_text())
            if spec["mode"] == "real" and not self._science_ok(state, spec):
                return {"status": "SCIENCE_GATE_CLOSED", "unit_id": uid}
            if spec["mode"] == "synthetic" and (not state["gate"]["synthetic"] or
                                                not self._gate_fresh(state)):
                return {"status": "SYNTHETIC_GATE_CLOSED", "unit_id": uid}
            if not self._inputs_ok(spec, state):
                return {"status": "INPUT_IDENTITY_CHANGED", "unit_id": uid}
            elapsed = (dt.datetime.now(dt.timezone.utc) - self.start).total_seconds()
            if (elapsed + spec["wall_limit_s"] > WALL_BUDGET_S or
                    state["used_cpu_s"] + spec["cpu_reserve_s"] > CPU_BUDGET_S):
                return {"status": "BUDGET_CEILING", "unit_id": uid}
        with self.slot.open("a+") as slot:
            try:
                fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "SLOT_BUSY", "unit_id": uid}
            with self._mutex():
                state = self._read()
                if state["units"][uid]["status"] != "QUEUED":
                    return {"status": "RACE_RETRY", "unit_id": uid}
                if (sha256(path) != rec["spec_sha256"] or
                        not self._inputs_ok(spec, state) or
                        any(Path(p).exists() for p in spec["outputs"])):
                    return {"status": "INPUT_IDENTITY_CHANGED", "unit_id": uid}
                if (spec["mode"] == "real" and not self._science_ok(state, spec)) or (
                        spec["mode"] == "synthetic" and not self._gate_fresh(state)):
                    return {"status": "ADMISSION_CHANGED", "unit_id": uid}
                elapsed = (dt.datetime.now(dt.timezone.utc) - self.start).total_seconds()
                if (elapsed + spec["wall_limit_s"] > WALL_BUDGET_S or
                        state["used_cpu_s"] + spec["cpu_reserve_s"] > CPU_BUDGET_S):
                    return {"status": "BUDGET_CEILING", "unit_id": uid}
                state["units"][uid].update(status="RUNNING", started_utc=_utc(),
                                           wrapper_pid=os.getpid())
                _atomic_json(self.state_path, state)
                self._event("start", unit_id=uid)
            stdout = self.root / (uid + ".stdout")
            stderr = self.root / (uid + ".stderr")
            env = dict(os.environ)
            env.update(THREAD_ENV)
            env["PYTHONPATH"] = str(Path(spec["cwd"])) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            before = time.monotonic()
            usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            timed_out = False
            memory_exceeded = False
            orphan_child = False
            start_error = None
            probe_error = None
            code = None
            observed_cpu_s = 0.0
            observed_peak_rss = 0
            with stdout.open("wb") as out, stderr.open("wb") as err:
                try:
                    child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                                             stdout=out, stderr=err, start_new_session=True,
                                             pass_fds=(slot.fileno(),),
                                             preexec_fn=(_deny_child_processes
                                                         if spec.get("allow_child_processes") is False
                                                         else None))
                except OSError as exc:
                    start_error = f"{type(exc).__name__}: {exc}"
                    err.write((start_error + "\n").encode())
                    err.flush()
                if start_error is None:
                    with self._mutex():
                        state = self._read()
                        state["units"][uid].update(child_pid=child.pid, child_pgid=child.pid)
                        _atomic_json(self.state_path, state)
                    try:
                        leader = psutil.Process(child.pid)
                        while True:
                            try:
                                rss = 0 if leader.status() == psutil.STATUS_ZOMBIE else leader.memory_info().rss
                                times = leader.cpu_times()
                                cpu_now = times.user + times.system
                            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                                rss = 0
                                cpu_now = 0.0
                            observed_peak_rss = max(observed_peak_rss, rss)
                            observed_cpu_s = max(observed_cpu_s, cpu_now)
                            if rss > spec["max_rss_gib"] * 1024**3:
                                memory_exceeded = True
                                break
                            if time.monotonic() - before >= spec["wall_limit_s"]:
                                timed_out = True
                                break
                            if child.poll() is not None:
                                break
                            time.sleep(0.05)
                    except (psutil.Error, PermissionError, OSError) as exc:
                        probe_error = f"{type(exc).__name__}: {exc}"
                    if memory_exceeded or timed_out or probe_error is not None:
                        self._stop_own_group(child)
                    code = child.wait()
                    try:
                        os.killpg(child.pid, 0)
                        orphan_child = True
                    except ProcessLookupError:
                        pass
                    if orphan_child:
                        self._stop_own_group(child)
            wall = time.monotonic() - before
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = max(observed_cpu_s, (usage_after.ru_utime + usage_after.ru_stime) -
                      (usage_before.ru_utime + usage_before.ru_stime))
            charged = max(wall, cpu)
            output_hashes = {p: sha256(Path(p)) for p in spec["outputs"] if Path(p).is_file()}
            if start_error is not None:
                outcome = "FAILED_START"
            elif probe_error is not None:
                outcome = "FAILED_PROBE"
            elif memory_exceeded:
                outcome = "FAILED_MEMORY_LIMIT"
            elif orphan_child:
                outcome = "FAILED_ORPHAN_CHILD"
            else:
                outcome = "DONE" if code == 0 and not timed_out and len(output_hashes) == len(spec["outputs"]) else "FAILED"
            with self._mutex():
                state = self._read()
                state["used_cpu_s"] += charged
                state["units"][uid].update(status=outcome, ended_utc=_utc(),
                                           exit_code=code, timed_out=timed_out,
                                           memory_exceeded=memory_exceeded,
                                           orphan_child=orphan_child, start_error=start_error,
                                           probe_error=probe_error,
                                           wall_s=wall, child_cpu_s=cpu, charged_cpu_s=charged,
                                           observed_peak_rss_bytes=observed_peak_rss,
                                           maxrss_raw=usage_after.ru_maxrss,
                                           output_sha256=output_hashes,
                                           stdout_sha256=sha256(stdout), stderr_sha256=sha256(stderr))
                _atomic_json(self.state_path, state)
                self._event("finish", unit_id=uid, status=outcome,
                            charged_cpu_s=charged, output_sha256=output_hashes)
            return {"status": outcome, "unit_id": uid, "charged_cpu_s": charged,
                    "exit_code": code}

    @staticmethod
    def _stop_own_group(child: subprocess.Popen) -> None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        # A departed leader may leave descendants in its dedicated group.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

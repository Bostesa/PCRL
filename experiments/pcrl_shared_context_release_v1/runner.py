"""Write-once queue runner for shared-context units on one host.

A queue is JSON: {"schema": "pcrl-sc-queue-v1", "units": [unit, ...]} where

    unit = {"id": "a0_NM1_U",                 # safe label, also the directory name
            "argv": ["-m", "experiments...fit_nm", "--out", "{out}", ...],
            "inputs": ["path", ...],          # files or directories hashed into the receipt
            "depends_on": ["a0_bank", ...],   # their runner receipts enter inputs_sha256
            "data_roles": ["nuisance_train", ...],   # declared roles (outer refused)
            "timeout_seconds": 7200,
            "release": {"kind": "nested", "release_id": "NM1_U", "anchor": 0}}  # optional

Placeholders in argv: {out} (the unit directory), {root} (repository root),
{python} (interpreter), {unit:ID} (a completed dependency's directory) and
{json:ID/FILE:KEY} (a scalar read at launch time from a completed dependency's
JSON output, e.g. the cross-anchor K decision). `runner capture` wraps a
command that only prints JSON (fit_nm decide-k) into a unit output file.

Each unit runs as one single-threaded subprocess (OMP/MKL/OPENBLAS/... = 1)
writing directly into `<units>/<id>/`; that directory belongs to the unit's
module (fit_nm writes its own COMPLETE.json there and checks its own
inventory), so the runner never adds files to it except `release.json` when
declared and absent. After exit 0 the runner atomically publishes its receipt
`<units>/_receipts/<id>.json` with the inputs hash, code commit, code tree
hash, data-role hash and every output hash of the unit directory; attempt logs
stay in `<units>/.attempts/<id>/attempt-k/`. A unit with a receipt is never
rewritten: resume re-verifies every recorded hash before reuse and blocks on
any mismatch. Technical failures (nonzero exit other than
REFUSED_EXIT, signal, timeout, interrupted runner) get at most one identical
retry; each failed attempt's directory and logs are preserved under
`<units>/.attempts/<id>/`. Status is mirrored to a JSON status file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from . import laws

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
QUEUE_SCHEMA = "pcrl-sc-queue-v1"
RECEIPT_SCHEMA = "pcrl-sc-unit-complete-v1"
STATUS_SCHEMA = "pcrl-sc-runner-status-v1"
MAX_ATTEMPTS = 2  # first attempt + at most one identical technical retry
REFUSED_EXIT = 3  # deliberate validation refusal: never retried
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
INNER_ROLES = ("nuisance_train", "audit_fit", "coefficient_split",
               "inner_selection", "inner_check")
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
PLACEHOLDER = re.compile(r"\{unit:([A-Za-z0-9_.-]+)\}")
JSON_PLACEHOLDER = re.compile(r"\{json:([A-Za-z0-9_.-]+)/([A-Za-z0-9_./-]+):([A-Za-z0-9_]+)\}")
RECEIPTS = "_receipts"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json_atomic(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    out = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in unit output: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in exclude and "__pycache__" not in path.parts:
            out[relative] = laws.sha256_file(path)
    return out


def path_digest(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return laws.sha256_file(target)
    if target.is_dir():
        return _sha_bytes(_canonical(inventory(target, exclude=())))
    raise FileNotFoundError(f"declared unit input missing: {path}")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def code_commit(root: Path = ROOT) -> str:
    marker = root / "SOURCE_COMMIT.txt"
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        if marker.is_file():
            return marker.read_text().strip()
        raise RuntimeError("code commit unavailable: no git checkout or SOURCE_COMMIT.txt")


def code_tree_sha256(package_dir: Path = PACKAGE_DIR) -> str:
    """Hash of the study package sources actually executed (catches local edits)."""
    files = {path.name: laws.sha256_file(path) for path in sorted(package_dir.glob("*.py"))}
    return _sha_bytes(_canonical(files))


def data_role_hash(data_roles: Sequence[str], index_sha256: str | None) -> str:
    from experiments.pcrl_adaptive_release_v1 import roles
    record = {"salt": roles.SALT, "names": list(roles.ROLE_NAMES),
              "upper": list(roles.ROLE_UPPER), "roles_py": laws.sha256_file(roles.__file__),
              "declared": sorted(data_roles), "index_sha256": index_sha256}
    return _sha_bytes(_canonical(record))


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------

def validate_queue(queue: Mapping) -> list[dict]:
    if queue.get("schema") != QUEUE_SCHEMA or not isinstance(queue.get("units"), list):
        raise ValueError("unknown queue schema")
    units = [dict(unit) for unit in queue["units"]]
    ids = [unit.get("id") for unit in units]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and SAFE_ID.fullmatch(i)
                                            and i[0] not in "._" for i in ids):
        raise ValueError("unit ids must be unique safe labels")
    known = set(ids)
    for unit in units:
        argv = unit.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError(f"{unit['id']}: argv must be a nonempty list of strings")
        deps = unit.setdefault("depends_on", [])
        for arg in argv:
            refs = PLACEHOLDER.findall(arg) + [m[0] for m in JSON_PLACEHOLDER.findall(arg)]
            for ref in refs:
                if ref not in deps:
                    raise ValueError(f"{unit['id']}: {{unit:{ref}}} used without depends_on")
        if not set(deps) <= known or unit["id"] in deps:
            raise ValueError(f"{unit['id']}: unknown or self dependency")
        roles_declared = unit.setdefault("data_roles", [])
        if not set(roles_declared) <= set(INNER_ROLES):
            raise PermissionError(f"{unit['id']}: runner schedules inner roles only; outer is coordinator-gated")
        unit.setdefault("inputs", [])
        unit.setdefault("timeout_seconds", 6 * 3600)
        if "release" in unit:
            release = unit["release"]
            if release.get("kind") not in laws.UNIT_KINDS:
                raise ValueError(f"{unit['id']}: undeclared release kind")
    order, seen = [], set()

    def visit(uid: str, stack: tuple = ()) -> None:
        if uid in seen:
            return
        if uid in stack:
            raise ValueError(f"dependency cycle through {uid}")
        unit = next(u for u in units if u["id"] == uid)
        for dep in unit["depends_on"]:
            visit(dep, stack + (uid,))
        seen.add(uid)
        order.append(unit)
    for uid in ids:
        visit(uid)
    return order


class Runner:
    def __init__(self, queue: Mapping, units_root: str | Path, *, workers: int = 1,
                 status_path: str | Path | None = None, python: str = sys.executable,
                 index_sha256: str | None = None, root: Path = ROOT,
                 poll_seconds: float = 0.2, only: Sequence[str] | None = None,
                 reuse_commits: Sequence[str] = ()):
        if workers < 1:
            raise ValueError("at least one worker")
        self.units = validate_queue(queue)
        self.only = None if only is None else set(only)
        # Explicit coordinator allowance: receipts made at these earlier commits
        # may be reused only when the study-package tree hash is byte-identical.
        self.reuse_commits = set(reuse_commits)
        if self.only is not None and not self.only <= {u["id"] for u in self.units}:
            raise ValueError("--only names an undeclared unit")
        self.by_id = {unit["id"]: unit for unit in self.units}
        self.units_root = Path(units_root).resolve()
        if "private" not in self.units_root.parts:
            raise ValueError("unit outputs (models, person rows) must live under a private path")
        self.attempts_root = self.units_root / ".attempts"
        self.status_path = Path(status_path) if status_path else self.units_root / "STATUS.json"
        self.workers, self.python, self.root = workers, python, Path(root)
        self.index_sha256, self.poll = index_sha256, poll_seconds
        self.commit, self.tree = code_commit(self.root), code_tree_sha256()
        self._lock = threading.Lock()
        self.status = {"schema": STATUS_SCHEMA, "started_utc": utc(), "workers": workers,
                       "code_commit": self.commit, "code_tree_sha256": self.tree,
                       "host": {"system": platform.system(), "machine": platform.machine(),
                                "python": platform.python_version()},
                       "units": {u["id"]: {"state": "PENDING", "attempts": []} for u in self.units}}

    # -- status ------------------------------------------------------------
    def _set(self, uid: str, **fields: Any) -> None:
        with self._lock:
            self.status["units"][uid].update(fields)
            self.status["updated_utc"] = utc()
            counts: dict[str, int] = {}
            for record in self.status["units"].values():
                counts[record["state"]] = counts.get(record["state"], 0) + 1
            self.status["counts"] = counts
            _write_json_atomic(self.status_path, self.status)

    # -- identity ------------------------------------------------------------
    def unit_dir(self, uid: str) -> Path:
        return self.units_root / uid

    def receipt_path(self, uid: str) -> Path:
        return self.units_root / RECEIPTS / f"{uid}.json"

    def expected_identity(self, unit: Mapping) -> dict:
        deps = {dep: laws.sha256_file(self.receipt_path(dep))
                for dep in unit["depends_on"]}
        inputs = {str(path): path_digest(self._resolve(path)) for path in unit["inputs"]}
        spec = {"id": unit["id"], "argv": unit["argv"], "release": unit.get("release"),
                "data_roles": sorted(unit["data_roles"])}
        return {"inputs_sha256": _sha_bytes(_canonical({"spec": spec, "inputs": inputs,
                                                        "dependencies": deps})),
                "input_files_sha256": inputs, "dependency_receipts_sha256": deps,
                "code_commit": self.commit, "code_tree_sha256": self.tree,
                "data_role_hash": data_role_hash(unit["data_roles"], self.index_sha256)}

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def verify_receipt(self, unit: Mapping) -> dict:
        """Reuse a completed unit only if every recorded hash still holds."""
        directory = self.unit_dir(unit["id"])
        receipt = json.loads(self.receipt_path(unit["id"]).read_text())
        expected = self.expected_identity(unit)
        problems = [key for key in ("inputs_sha256", "code_commit", "code_tree_sha256",
                                    "data_role_hash") if receipt.get(key) != expected[key]]
        if (problems == ["code_commit"] and receipt.get("code_commit") in self.reuse_commits):
            problems = []  # tree hash already matched; recorded as cross-commit reuse
            self._set(unit["id"], reused_from_commit=receipt["code_commit"])
        if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("unit_id") != unit["id"]:
            problems.append("schema/unit_id")
        if not directory.is_dir() or receipt.get("outputs_sha256") != inventory(directory):
            problems.append("outputs_sha256")
        if problems:
            raise RuntimeError(f"{unit['id']}: completed unit differs ({', '.join(problems)}); "
                               "refusing reuse or overwrite")
        return receipt

    # -- execution -----------------------------------------------------------
    def _argv(self, unit: Mapping) -> list[str]:
        out = str(self.unit_dir(unit["id"]))

        def value(match: re.Match) -> str:
            uid, relative, key = match.groups()
            receipt = json.loads(self.receipt_path(uid).read_text())
            if relative not in receipt["outputs_sha256"]:
                raise ValueError(f"{relative} is not a recorded output of {uid}")
            source = self.unit_dir(uid) / relative
            if laws.sha256_file(source) != receipt["outputs_sha256"][relative]:
                raise ValueError(f"{uid}/{relative} differs from its receipt")
            found = json.loads(source.read_text())[key]
            if isinstance(found, bool) or not isinstance(found, (int, float, str)) or \
                    not re.fullmatch(r"[A-Za-z0-9_.-]+", str(found)):
                raise ValueError(f"{uid}/{relative}:{key} is not a safe scalar")
            return str(found)

        def fill(arg: str) -> str:
            arg = JSON_PLACEHOLDER.sub(value, arg)
            arg = PLACEHOLDER.sub(lambda m: str(self.unit_dir(m.group(1))), arg)
            return (arg.replace("{out}", out).replace("{root}", str(self.root))
                    .replace("{python}", self.python))
        return [self.python, *(fill(arg) for arg in unit["argv"])]

    def _prior_attempts(self, uid: str) -> list[Path]:
        base = self.attempts_root / uid
        return sorted(base.glob("attempt-*")) if base.is_dir() else []

    def _retire(self, uid: str, attempt_dir: Path, reason: str) -> None:
        """Move a failed/interrupted unit directory beside its attempt logs."""
        directory = self.unit_dir(uid)
        if directory.exists():
            if self.receipt_path(uid).exists():
                raise RuntimeError(f"{uid}: refusing to retire a completed unit")
            shutil.move(str(directory), str(attempt_dir / "unit_output"))
        _write_json_atomic(attempt_dir / "FAILED.json", {"unit_id": uid, "reason": reason,
                                                         "retired_utc": utc()})

    def reuse_only(self, unit: Mapping) -> str:
        """Outside --only: verify a completed receipt, never execute."""
        if not self.receipt_path(unit["id"]).exists():
            self._set(unit["id"], state="SKIPPED", reason="outside --only and not complete")
            return "SKIPPED"
        self.verify_receipt(unit)
        self._set(unit["id"], state="REUSED", verified_utc=utc())
        return "REUSED"

    def run_unit(self, unit: Mapping) -> str:
        uid = unit["id"]
        directory = self.unit_dir(uid)
        if self.receipt_path(uid).exists():
            self.verify_receipt(unit)
            self._set(uid, state="REUSED", verified_utc=utc())
            return "REUSED"
        attempts = self._prior_attempts(uid)
        if directory.exists():  # runner died mid-unit: preserve as an interrupted attempt
            stale = self.attempts_root / uid / f"attempt-{len(attempts) + 1}"
            stale.mkdir(parents=True, exist_ok=False)
            self._retire(uid, stale, "interrupted: unit directory without runner receipt")
            attempts = self._prior_attempts(uid)
        for record in attempts:
            failed = record / "FAILED.json"
            if failed.is_file() and json.loads(failed.read_text()).get("reason", "").startswith("refused"):
                self._set(uid, state="REFUSED", attempts=[p.name for p in attempts])
                return "REFUSED"
        identity = self.expected_identity(unit)
        while len(attempts) < MAX_ATTEMPTS:
            number = len(attempts) + 1
            attempt_dir = self.attempts_root / uid / f"attempt-{number}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            try:
                argv = self._argv(unit)
            except (OSError, ValueError, KeyError) as error:
                shutil.rmtree(attempt_dir)
                self._set(uid, state="BLOCKED", reason=f"argument resolution: {error}")
                return "BLOCKED"
            env = {**os.environ, **THREAD_ENV, "PYTHONPATH": str(self.root),
                   "PYTHONHASHSEED": "0", "PCRL_UNIT_ID": uid, "PCRL_UNIT_OUT": str(directory),
                   "PCRL_UNIT_ATTEMPT": str(number)}
            started = utc()
            self._set(uid, state="RUNNING", attempt=number, started_utc=started)
            _write_json_atomic(attempt_dir / "ATTEMPT.json", {
                "unit_id": uid, "attempt": number, "argv": argv, "started_utc": started,
                "thread_env": THREAD_ENV, **identity})
            clock = time.monotonic()
            with (attempt_dir / "stdout.log").open("wb") as out, \
                    (attempt_dir / "stderr.log").open("wb") as err:
                # Own session/process group so a timeout kills every descendant (M5.6).
                process = subprocess.Popen(argv, cwd=self.root, env=env, stdout=out, stderr=err,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=unit["timeout_seconds"])
                    reason = None if code == 0 else (
                        f"refused: exit {code}" if code == REFUSED_EXIT else f"technical: exit {code}")
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    code, reason = None, f"technical: timeout after {unit['timeout_seconds']} s"
            wall = time.monotonic() - clock
            if reason is None and not directory.is_dir():
                reason = "technical: exit 0 without unit directory"
            if reason is None:
                try:
                    self._publish(unit, attempt_dir, identity, number, started, wall)
                except Exception as error:  # noqa: BLE001 - recorded, then retried once
                    reason = f"technical: publish failed: {error}"
            if reason is None:
                self._set(uid, state="COMPLETE", finished_utc=utc(), wall_seconds=wall,
                          attempts=[p.name for p in self._prior_attempts(uid)])
                return "COMPLETE"
            self._retire(uid, attempt_dir, reason)
            attempts = self._prior_attempts(uid)
            if reason.startswith("refused"):
                self._set(uid, state="REFUSED", reason=reason,
                          attempts=[p.name for p in attempts])
                return "REFUSED"
            self._set(uid, state="RETRYING" if len(attempts) < MAX_ATTEMPTS else "FAILED",
                      reason=reason, attempts=[p.name for p in attempts])
        self._set(uid, state="FAILED")
        return "FAILED"

    def _publish(self, unit: Mapping, attempt_dir: Path, identity: Mapping, number: int,
                 started: str, wall: float) -> None:
        directory = self.unit_dir(unit["id"])
        if "release" in unit and not (directory / laws.DESCRIPTOR).exists():
            release = unit["release"]
            laws.write_descriptor(directory, release["kind"], release_id=release["release_id"],
                                  anchor=release["anchor"])
        receipt = {"schema": RECEIPT_SCHEMA, "unit_id": unit["id"], "attempt": number,
                   "argv": unit["argv"], "started_utc": started, "completed_utc": utc(),
                   "attempt_logs": str(attempt_dir.relative_to(self.units_root)),
                   "attempt_log_sha256": {name: laws.sha256_file(attempt_dir / name)
                                          for name in ("stdout.log", "stderr.log", "ATTEMPT.json")},
                   "wall_seconds": wall, "thread_env": THREAD_ENV,
                   "host": self.status["host"], **identity,
                   "outputs_sha256": inventory(directory)}
        target = self.receipt_path(unit["id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, target)

    # -- scheduling ----------------------------------------------------------
    def run(self) -> dict:
        self.units_root.mkdir(parents=True, exist_ok=True)
        pending = [unit["id"] for unit in self.units]
        final: dict[str, str] = {}
        running: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while pending or running:
                for uid in list(pending):
                    deps = self.by_id[uid]["depends_on"]
                    if any(final.get(d) in ("FAILED", "REFUSED", "BLOCKED", "SKIPPED") for d in deps):
                        final[uid] = "BLOCKED"
                        self._set(uid, state="BLOCKED", reason="dependency did not complete")
                        pending.remove(uid)
                    elif all(final.get(d) in ("COMPLETE", "REUSED") for d in deps) and \
                            len(running) < self.workers:
                        job = (self.run_unit if self.only is None or uid in self.only
                               else self.reuse_only)
                        running[pool.submit(job, self.by_id[uid])] = uid
                        pending.remove(uid)
                if not running:
                    continue
                done, _ = wait(list(running), timeout=self.poll, return_when=FIRST_COMPLETED)
                for future in done:
                    uid = running.pop(future)
                    try:
                        final[uid] = future.result()
                    except Exception as error:  # noqa: BLE001 - e.g. stale receipt
                        final[uid] = "BLOCKED"
                        self._set(uid, state="BLOCKED", reason=str(error))
        with self._lock:
            self.status["finished_utc"] = utc()
            _write_json_atomic(self.status_path, self.status)
        return final


# ---------------------------------------------------------------------------
# panel plan (templates are data; override with --templates JSON)
# ---------------------------------------------------------------------------

M = "experiments.pcrl_shared_context_release_v1"
FIT_ROLES = list(INNER_ROLES)  # rd/fit_nm load_roles admit the five inner roles
AUDIT_ROLES = ["audit_fit", "inner_selection", "inner_check"]
J_INDEX_SHA256 = "461b06f0bba10416e4a6bec6bc0c69a3821ae2ef9733bc45030ffd0dfea4ad8a"
J_INDEX_HOST = ("/opt/pcrl/work/results/pcrl_shared_context_release_v1/private/j_inputs/"
                "EXTERNAL_RELEASE_INPUTS.private.json")
# CLIs confirmed final by the method and baseline owners (2026-09-24, amendment M3).
DEFAULT_TEMPLATES: dict[str, dict] = {
    "bank": {"argv": ["-m", f"{M}.fit_nm", "build-bank", "--anchor", "{anchor}",
                      "--index", "{index}", "--out", "{out}"], "data_roles": FIT_ROLES},
    "decide_k": {"argv": ["-m", f"{M}.runner", "capture", "--out", "{out}", "--file",
                          "DECISION.json", "--", "-m", f"{M}.fit_nm", "decide-k", "--banks",
                          "{unit:a0_bank}", "{unit:a1_bank}", "{unit:a2_bank}"],
                 "data_roles": []},
    "NM": {"argv": ["-m", f"{M}.fit_nm", "fit", "--anchor", "{anchor}", "--variant", "{variant}",
                    "--bank", "{bank}", "--index", "{index}", "--rounds", "6", "--out", "{out}"],
           "data_roles": FIT_ROLES, "kind": "nested"},
    "NM4_extra": {"argv": ["--nm4-k", "{json:decide_k/DECISION.json:nm4_K}"]},
    "DET": {"argv": ["-m", f"{M}.fit_nm", "fit", "--anchor", "{anchor}", "--variant", "{variant}",
                     "--from", "{source}", "--index", "{index}", "--out", "{out}"],
            "data_roles": FIT_ROLES, "kind": "nested"},
    "RD": {"argv": ["-m", f"{M}.rd", "--anchor", "{anchor}", "--variant", "{variant}",
                    "--bank", "{bank}", "--out", "{out}"],
           "data_roles": FIT_ROLES, "kind": "deterministic_policy"},
    "ADV": {"argv": ["-m", f"{M}.adv", "--anchor", "{anchor}", "--beta", "{beta}",
                     "--bank", "{bank}", "--out", "{out}", "--select", "{select}"],
            "data_roles": FIT_ROLES, "kind": "adv_mlp"},
    "POS": {"argv": ["-m", f"{M}.poscontrol", "--anchor", "{anchor}", "--role", "{role}",
                     "--out", "{out}"], "data_roles": FIT_ROLES},
    "AUDIT": {"argv": ["-m", f"{M}.audit_panel", "inner", "--anchor", "{anchor}",
                       "--index", "{index}", "--sources", "{sources}", "--slate", "standard",
                       "--out", "{out}"], "data_roles": AUDIT_ROLES},
    # J continuity reference: AR's continuous-wire panel, unchanged (contextual, not matched).
    "J": {"argv": ["-m", "experiments.pcrl_adaptive_release_v1.external_audit", "--anchor",
                   "{anchor}", "--methods", "J", "--index", "{index}", "--external-index",
                   J_INDEX_HOST, "--external-index-sha256", J_INDEX_SHA256,
                   "--external-root", "{root}", "--slate", "standard", "--out", "{out}"],
          "data_roles": AUDIT_ROLES},
}
NM_VARIANTS = ("NM1_U", "NM1_P", "NM4_U", "NM4_P", "T32_U", "T32_P")
DET_VARIANTS = {"DET_SEL1": "NM1_U", "DET_SEL4": "NM4_U"}
RD_VARIANTS = ("RD_TASK", "RD_PRIV")
# Amendment M3: task-selected ADV_B* supply the U-route representative,
# privacy-selected ADV_B*_P the P-route representative.
ADV_UNITS = {"ADV_B1": ("0.5", "task"), "ADV_B2": ("2.0", "task"),
             "ADV_B1_P": ("0.5", "privacy"), "ADV_B2_P": ("2.0", "privacy")}
POS_ROLES = ("attack:AB/SEX", "attack:AB/RAC1P")
ALL_PARTS = ("NM", "DET", "RD", "ADV", "POS", "AUDIT", "J")


def _fill(template: Mapping, **values: str) -> dict:
    """Plain token replacement; runner placeholders ({out}, {unit:..}, {json:..}) survive."""
    argv = []
    for arg in template["argv"]:
        for key, value in values.items():
            arg = arg.replace("{" + key + "}", value)
        argv.append(arg)
    return {"argv": argv, "data_roles": list(template.get("data_roles", []))}


def plan_panel(anchors: Sequence[int], index_path: str, units_root: str | Path,
               plan_dir: str | Path, *, templates: Mapping[str, dict] | None = None,
               include: Sequence[str] = ALL_PARTS, nm4_k: int | None = None,
               timeout_seconds: int = 4 * 3600) -> dict:
    """Dependency order (per anchor unless noted):

    bank -> decide_k (cross-anchor barrier over a0/a1/a2 banks) -> NM/T32 variants
    NM1_U -> DET_SEL1, NM4_U -> DET_SEL4;  bank -> RD_TASK, RD_PRIV, ADV_B1, ADV_B2
    every release -> inner_audit (with D17, Q_HIST); POS and J need only staged inputs.
    `nm4_k` pins K without the barrier (only for a partial-anchor plan).
    """
    t = {**DEFAULT_TEMPLATES, **(templates or {})}
    unknown = set(include) - set(ALL_PARTS)
    if unknown:
        raise ValueError(f"unknown plan parts {sorted(unknown)}")
    units_root, plan_dir = Path(units_root), Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    units: list[dict] = []

    def add(uid: str, part: str, *, deps: Sequence[str] = (), inputs: Sequence[str] = (),
            release: dict | None = None, **values: str) -> None:
        unit = {"id": uid, **_fill(t[part], **values), "depends_on": list(deps),
                "inputs": list(inputs), "timeout_seconds": timeout_seconds}
        if release:
            unit["release"] = release
        units.append(unit)

    anchors = [int(a) for a in anchors]
    need_nm = "NM" in include or "DET" in include
    barrier = need_nm and nm4_k is None
    if barrier and sorted(anchors) != [0, 1, 2]:
        raise ValueError("the NM4 K decision is a three-anchor barrier; plan all anchors or pass nm4_k")
    for anchor in anchors:
        add(f"a{anchor}_bank", "bank", inputs=[index_path], anchor=str(anchor), index=index_path)
    if barrier:
        add("decide_k", "decide_k", deps=[f"a{a}_bank" for a in (0, 1, 2)])
    for anchor in anchors:
        a = str(anchor)
        bank_id = f"a{a}_bank"
        bank = "{unit:" + bank_id + "}"
        released: dict[str, str] = {}
        if need_nm:
            for variant in NM_VARIANTS:
                uid = f"a{a}_{variant}"
                deps = [bank_id] + (["decide_k"] if barrier else [])
                add(uid, "NM", deps=deps, anchor=a, variant=variant, bank=bank, index=index_path,
                    release={"kind": t["NM"]["kind"], "release_id": f"a{a}_{variant}",
                             "anchor": anchor})
                if variant.startswith("NM4"):
                    extra = (list(t["NM4_extra"]["argv"]) if barrier
                             else ["--nm4-k", str(int(nm4_k))])
                    units[-1]["argv"] += extra
                if "NM" in include:
                    released[variant] = uid
        if "DET" in include:
            for variant, source in DET_VARIANTS.items():
                uid = f"a{a}_{variant}"
                src = f"a{a}_{source}"
                add(uid, "DET", deps=[src], anchor=a, variant=variant,
                    source="{unit:" + src + "}", index=index_path,
                    release={"kind": t["DET"]["kind"], "release_id": f"a{a}_{variant}",
                             "anchor": anchor})
                released[variant] = uid
        if "RD" in include:
            for variant in RD_VARIANTS:
                uid = f"a{a}_{variant}"
                add(uid, "RD", deps=[bank_id], anchor=a, variant=variant, bank=bank,
                    release={"kind": t["RD"]["kind"], "release_id": f"a{a}_{variant}",
                             "anchor": anchor})
                released[variant] = uid
        if "ADV" in include:
            for name, (beta, select) in ADV_UNITS.items():
                uid = f"a{a}_{name}"
                add(uid, "ADV", deps=[bank_id], anchor=a, beta=beta, bank=bank, select=select,
                    release={"kind": t["ADV"]["kind"], "release_id": f"a{a}_{name}",
                             "anchor": anchor})
                released[name] = uid
        if "POS" in include:
            for role in POS_ROLES:
                add(f"a{a}_POS_{role.split(':')[1].replace('/', '_')}", "POS", inputs=[index_path],
                    anchor=a, role=role)
        if "J" in include:
            add(f"a{a}_J_inner", "J", inputs=[index_path, J_INDEX_HOST], anchor=a,
                index=index_path)
        if "AUDIT" in include:
            sources: dict[str, Any] = {
                name: {"kind": "historical_map", "map_name": m, "anchor": anchor,
                       "index_path": index_path}
                for name, m in (("D17", "D17"), ("Q_HIST", "Q"))}
            for name, uid in released.items():
                sources[name] = str(units_root.resolve() / uid)
            sources_path = plan_dir / f"a{a}_audit_sources.json"
            _write_json_atomic(sources_path, sources)
            add(f"a{a}_inner_audit", "AUDIT", deps=list(released.values()),
                inputs=[index_path, str(sources_path)], anchor=a, index=index_path,
                sources=str(sources_path))
    queue = {"schema": QUEUE_SCHEMA, "units": units}
    validate_queue(queue)
    return queue


def capture(out: str | Path, file_name: str, command: Sequence[str]) -> dict:
    """Run a JSON-printing module command and store its last stdout line as a unit file."""
    if not command or command[0] != "-m" or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", file_name):
        raise ValueError("capture runs `-m module ...` into a safe .json file name")
    target = Path(out)
    if target.exists():
        raise FileExistsError("capture output directory must be new")
    process = subprocess.run([sys.executable, *command], check=True, capture_output=True, text=True)
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError:
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError("captured command must print one JSON object last")
    target.mkdir(parents=True)
    _write_json_atomic(target / file_name, value)
    (target / "command.json").write_text(json.dumps({"argv": list(command)}, sort_keys=True) + "\n")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--anchors", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--index", required=True)
    plan.add_argument("--units-root", required=True)
    plan.add_argument("--plan-dir", required=True)
    plan.add_argument("--templates", help="JSON overriding DEFAULT_TEMPLATES entries")
    plan.add_argument("--include", nargs="+", default=list(ALL_PARTS), choices=ALL_PARTS)
    plan.add_argument("--nm4-k", type=int, choices=(2, 4), help="skip the decide_k barrier")
    plan.add_argument("--timeout-seconds", type=int, default=4 * 3600)
    plan.add_argument("--queue-out", required=True)
    run = sub.add_parser("run")
    run.add_argument("--queue", required=True)
    run.add_argument("--units-root", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--status")
    run.add_argument("--index-sha256")
    run.add_argument("--only", nargs="+", help="execute only these ids; others are verify-only")
    run.add_argument("--reuse-commit", nargs="*", default=[],
                     help="earlier commits whose receipts are reusable if the package tree hash matches")
    cap = sub.add_parser("capture")
    cap.add_argument("--out", required=True)
    cap.add_argument("--file", required=True)
    cap.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "capture":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        print(json.dumps(capture(args.out, args.file, command), sort_keys=True))
        return
    if args.action == "plan":
        templates = json.loads(Path(args.templates).read_text()) if args.templates else None
        queue = plan_panel(args.anchors, args.index, args.units_root, args.plan_dir,
                           templates=templates, include=args.include, nm4_k=args.nm4_k,
                           timeout_seconds=args.timeout_seconds)
        _write_json_atomic(Path(args.queue_out), queue)
        print(json.dumps({"units": len(queue["units"]), "queue": args.queue_out}))
        return
    queue = json.loads(Path(args.queue).read_text())
    runner = Runner(queue, args.units_root, workers=args.workers, status_path=args.status,
                    index_sha256=args.index_sha256, only=args.only,
                    reuse_commits=args.reuse_commit)
    final = runner.run()
    print(json.dumps(final, sort_keys=True))
    selected = set(args.only or final)
    raise SystemExit(0 if all(v in ("COMPLETE", "REUSED") for k, v in final.items()
                              if k in selected) else 1)


if __name__ == "__main__":
    main()

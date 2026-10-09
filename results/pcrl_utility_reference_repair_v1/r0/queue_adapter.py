"""Use the focused study's durable queue with this study's independent seal.

The inherited queue owns flock, resource limits, attempted-unit receipts and
write-once outputs. This adapter changes only the research lock/reviewer paths.
"""

from __future__ import annotations

import hashlib
import json
import resource
import re
import subprocess
from pathlib import Path

from results.pcrl_capacity_matched_feature_selection_v1.m0.queue import (
    CPU_BUDGET_S, WALL_BUDGET_S, FocusedQueue, sha256,
)


class RepairQueue(FocusedQueue):
    def _validate_spec(self, spec: dict, state: dict) -> None:
        """Admit the task table as a receipt-only unit with zero role files.

        The inherited queue requires at least one role file for every real
        stage. This one reads only already sealed task aggregates. Its exact
        non-role input map is itself committed in the current study lock.
        """
        if spec.get("stage") != "r2_task_table":
            return super()._validate_spec(spec, state)
        if (spec.get("mode") != "real" or spec.get("role_access") != []
                or spec.get("role_inputs") != {}
                or spec.get("data_access") != "PRIVATE_DERIVED_TASK_AGGREGATES_ONLY"
                or spec.get("allow_child_processes") is not False):
            raise ValueError("task table must be real, role-free and contained")
        uid = spec.get("unit_id")
        if not isinstance(uid, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{2,119}", uid):
            raise ValueError("task table unit ID invalid")
        argv = spec.get("argv")
        if (not isinstance(argv, list) or len(argv) < 2
                or any(not isinstance(arg, str) for arg in argv)):
            raise ValueError("task table argv invalid")
        script = Path(spec["script"]).resolve(strict=True)
        cwd = Path(spec["cwd"]).resolve(strict=True)
        if (not any(Path(arg).resolve() == script for arg in argv)
                or sha256(script) != spec.get("script_sha256")):
            raise ValueError("task table script identity invalid")
        for key, upper in (("cpu_reserve_s", CPU_BUDGET_S),
                           ("wall_limit_s", WALL_BUDGET_S),
                           ("max_rss_gib", 6)):
            value = spec.get(key)
            if not isinstance(value, (int, float)) or not 0 < value <= upper:
                raise ValueError("task table resource cap invalid: " + key)
        if (not isinstance(spec.get("inputs"), dict)
                or not spec["inputs"]
                or not isinstance(spec.get("outputs"), list)
                or len(spec["outputs"]) != 3
                or not isinstance(spec.get("deps"), list)
                or any(dep not in state["units"] for dep in spec["deps"])):
            raise ValueError("task table input/output/dependency inventory invalid")
        study = Path(state["gate"]["seal_path"]).parent.resolve(strict=True)
        repo = study.parents[1]
        private = study / ".private"
        if cwd != repo or not script.is_relative_to(repo):
            raise ValueError("task table worker outside research worktree")
        lock = json.loads((study / "CAMPAIGN_LOCK.json").read_text())
        if (lock.get("verification_only") is True
                or lock.get("source_map_sha256") != spec.get("source_map_sha256")
                or lock.get("allowed_roles") != []
                or lock.get("governed_source_sha256_by_path", {}).get(
                    str(script.relative_to(repo))) != spec["script_sha256"]):
            raise ValueError("task table lock/source scope invalid")
        registry = json.loads((private / "UNIT_SPECS.json").read_text())
        expected = registry.get("unit_specs", {}).get(uid)
        if (not isinstance(expected, dict)
                or expected.get("seal_sha256") != "$LOCAL_SEAL_SHA256"
                or {**expected, "seal_sha256": state["gate"]["seal_sha256"]} != spec):
            raise ValueError("task table differs from sealed registry")
        allowed = lock.get("derived_input_sha256_by_path")
        if (not isinstance(allowed, dict)
                or not set(spec["inputs"]).issubset(allowed)):
            raise ValueError("task table input outside derived allowlist")
        bank = repo / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
        for name, expected_sha in spec["inputs"].items():
            path = Path(name).resolve(strict=True)
            if (str(path) != name or allowed[name] != expected_sha
                    or sha256(path) != expected_sha
                    or not (path.is_relative_to(study) or path == bank)
                    or path.is_relative_to(private / "role_export_v1")):
                raise ValueError("task table input identity or custody invalid")
        for name in spec["outputs"]:
            path = Path(name)
            if (not path.is_absolute() or path.exists()
                    or not path.resolve().is_relative_to(private)
                    or any(name in json.loads(Path(rec["spec_path"]).read_text())["outputs"]
                           for other_id, rec in state["units"].items()
                           if other_id != uid)):
                raise ValueError("task table output path invalid")

    @staticmethod
    def _seal_ok(path: Path, expected_sha: str, *, contained_worker: bool = False) -> bool:
        try:
            path = Path(path).resolve(strict=True)
            if sha256(path) != expected_sha:
                return False
            seal = json.loads(path.read_text())
            study = path.parent
            repo = study.parents[1]
            lock_path = study / "CAMPAIGN_LOCK.json"
            review_path = study / "r3" / "PREFIT_REVIEW.json"
            manifest_path = study / "m1" / "DATA_ACCESS_MANIFEST.json"
            registry_path = study / ".private" / "UNIT_SPECS.json"
            if (seal.get("schema") != "pcrl-utility-local-seal-v1"
                    or seal.get("status") != "PASS"
                    or seal.get("lock_sha256") != sha256(lock_path)
                    or seal.get("r3_review_sha256") != sha256(review_path)):
                return False
            lock = json.loads(lock_path.read_text())
            review = json.loads(review_path.read_text())
            commit = seal["lock_commit_sha"]
            if (lock.get("schema") != "pcrl-utility-campaign-lock-v1"
                    or lock.get("status") != "LOCAL_GIT_SEALED_EFFECTIVE"
                    or lock.get("adult_fit_permission") is not True
                    or lock.get("verification_only") is True
                    or review.get("status") != "PASS"
                    or review.get("lock_sha256") != seal["lock_sha256"]
                    or review.get("lock_commit_sha") != commit
                    or lock.get("private_unit_registry_sha256") != sha256(registry_path)
                    or lock.get("source_map_sha256") != sha256(manifest_path)):
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
                return resource.getrlimit(resource.RLIMIT_NPROC) == (0, 0)
            relative = lock_path.relative_to(repo)
            shown = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                capture_output=True, check=False, timeout=10,
            )
            return (shown.returncode == 0
                    and hashlib.sha256(shown.stdout).hexdigest() == seal["lock_sha256"])
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
            return False

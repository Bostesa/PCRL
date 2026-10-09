"""Use the focused study's durable queue with this study's independent seal.

The inherited queue owns flock, resource limits, attempted-unit receipts and
write-once outputs. This adapter changes only the research lock/reviewer paths.
"""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
from pathlib import Path

from results.pcrl_capacity_matched_feature_selection_v1.m0.queue import (
    FocusedQueue, sha256,
)


class RepairQueue(FocusedQueue):
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

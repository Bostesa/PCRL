"""Deterministic, read-only preflight of the sealed sampled-channel unit graph.

This module never enqueues or launches a unit. It validates the graph and
source snapshot before later queue-spec materialization can use the same plan.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from pathlib import Path


KNOWN_KINDS = frozenset({
    "fit", "release", "attack", "q_preprocess", "reference_release",
    "reference_attack", "diagnostic_release", "diagnostic_attack",
})
UNIT_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_hash(path: Path, expected: str, label: str) -> None:
    if not SHA256.fullmatch(expected):
        raise ValueError(f"{label} SHA-256 malformed")
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{label} hash differs")


def _positive_cap(node: dict, key: str, unit_id: str) -> float:
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{unit_id} {key} missing or nonnumeric")
    cap = float(value)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError(f"{unit_id} {key} must be positive and finite")
    return cap


def _topological_order(plan: dict[str, dict]) -> list[str]:
    following = {unit_id: [] for unit_id in plan}
    remaining = {}
    for unit_id, node in plan.items():
        deps = node.get("deps", [])
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            raise ValueError(f"{unit_id} deps must be a list of unit IDs")
        if len(deps) != len(set(deps)):
            raise ValueError(f"{unit_id} duplicate dependency")
        for dep in deps:
            if dep not in plan:
                raise ValueError(f"{unit_id} dependency {dep} absent from plan")
            following[dep].append(unit_id)
        remaining[unit_id] = len(deps)
    ready = [unit_id for unit_id, count in remaining.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        unit_id = heapq.heappop(ready)
        order.append(unit_id)
        for child in following[unit_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(plan):
        raise ValueError("unit graph contains a cycle")
    return order


def compile_graph(amendment_path: Path, amendment_sha256: str,
                  seal_path: Path, seal_sha256: str, *, source_root: Path,
                  expected_units: int) -> dict:
    """Return a deterministic hash-only graph index after strict preflight."""
    amendment_path = Path(amendment_path)
    seal_path = Path(seal_path)
    source_root = Path(source_root).resolve(strict=True)
    _check_hash(amendment_path, amendment_sha256, "A amendment")
    _check_hash(seal_path, seal_sha256, "original seal")
    amendment = json.loads(amendment_path.read_text())
    if not str(amendment.get("status", "")).startswith("LOCALLY_SEALED"):
        raise ValueError("A amendment is not locally sealed")
    if amendment.get("original_lock_sha256") != seal_sha256:
        raise ValueError("A amendment original seal differs")
    plan = amendment.get("unit_plan")
    if not isinstance(plan, dict) or len(plan) != expected_units:
        raise ValueError(f"unit_plan count differs from expected {expected_units}")
    source_map = amendment.get("implementation_source_sha256")
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("implementation source snapshot missing")
    for relative, digest in sorted(source_map.items()):
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("implementation source record malformed")
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("implementation source escapes source root")
        _check_hash(source, digest, f"implementation source {relative}")
    for unit_id, node in plan.items():
        if not isinstance(unit_id, str) or not UNIT_ID.fullmatch(unit_id):
            raise ValueError("invalid unit ID")
        if not isinstance(node, dict) or node.get("kind") not in KNOWN_KINDS:
            raise ValueError(f"{unit_id} kind unrecognized")
        if node.get("output_id") != unit_id:
            raise ValueError(f"{unit_id} output_id differs")
        cpu = _positive_cap(node, "cpu_seconds", unit_id)
        wall = _positive_cap(node, "wall_seconds", unit_id)
        if cpu < wall:
            raise ValueError(f"{unit_id} cpu_seconds below wall_seconds")
        if any(value is None for value in node.values()):
            raise ValueError(f"{unit_id} has a pending null field")
    order = _topological_order(plan)
    node_hashes = {
        unit_id: hashlib.sha256(json.dumps(plan[unit_id], sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
        for unit_id in sorted(plan)
    }
    return {
        "schema": "r03-a-graph-preflight-v1",
        "original_seal_sha256": seal_sha256,
        "amendment_sha256": amendment_sha256,
        "unit_count": len(plan),
        "kind_counts": dict(sorted(Counter(node["kind"] for node in plan.values()).items())),
        "implementation_source_count": len(source_map),
        "node_sha256": node_hashes,
        "topological_order": order,
    }

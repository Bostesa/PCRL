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


def _check_parent_bindings(plan: dict[str, dict]) -> None:
    for unit_id, node in plan.items():
        kind = node["kind"]
        deps = node.get("deps", [])
        if kind in ("fit", "diagnostic_release", "reference_release"):
            if deps:
                raise ValueError(f"{unit_id} source unit has unexpected deps")
        elif kind == "release":
            if deps != [node.get("fit_unit_id")]:
                raise ValueError(f"{unit_id} release deps differ from selected FIT")
        elif kind == "q_preprocess":
            fit = node.get("fit_unit_id")
            if deps != ([fit] if fit else []):
                raise ValueError(f"{unit_id} Q deps differ from selected FIT")
            if fit:
                parent = plan[fit]
                if (parent["kind"] != "fit" or
                        parent.get("arm_id") != node.get("arm_id") or
                        parent.get("seed") != node.get("seed")):
                    raise ValueError(f"{unit_id} Q selected FIT identity differs")
        elif kind == "attack":
            releases = node.get("release_unit_ids")
            if not isinstance(releases, dict) or set(releases) != {"ATT", "INNER"}:
                raise ValueError(f"{unit_id} attack release binding missing")
            expected = [node.get("fit_unit_id"), releases["ATT"],
                        releases["INNER"], node.get("q_unit_id")]
            if deps != expected:
                raise ValueError(f"{unit_id} attack deps differ from FIT/releases/Q")
            for role, parent_id in (("ATT", releases["ATT"]),
                                    ("INNER", releases["INNER"])):
                parent = plan[parent_id]
                if (parent["kind"] != "release" or
                        parent.get("arm_id") != node.get("arm_id") or
                        parent.get("seed") != node.get("seed") or
                        parent.get("realization") != node.get("realization") or
                        parent.get("role") != ("ATTACK_FIT_NEW" if role == "ATT" else "INNER_SELECTION")):
                    raise ValueError(f"{unit_id} attack {role} release identity differs")
            fit = plan[expected[0]]
            q = plan[expected[3]]
            if (fit["kind"] != "fit" or q["kind"] != "q_preprocess" or
                    fit.get("arm_id") != node.get("arm_id") or
                    q.get("arm_id") != node.get("arm_id") or
                    fit.get("seed") != node.get("seed") or
                    q.get("seed") != node.get("seed")):
                raise ValueError(f"{unit_id} attack FIT/Q identity differs")
        elif kind in ("reference_attack", "diagnostic_attack"):
            releases = node.get("release_unit_ids")
            if kind == "diagnostic_attack" and releases is None and len(deps) == 3:
                releases = {"ATT": deps[0], "INNER": deps[1]}
            if not isinstance(releases, dict) or set(releases) != {"ATT", "INNER"}:
                raise ValueError(f"{unit_id} release binding missing")
            q_id = node.get("q_unit_id")
            expected = [releases["ATT"], releases["INNER"]]
            if q_id is not None:
                expected.append(q_id)
            if deps != expected:
                raise ValueError(f"{unit_id} attack deps differ from releases/Q")
            release_kind = ("reference_release" if kind == "reference_attack"
                            else "diagnostic_release")
            identity = ("reference_id" if kind == "reference_attack" else "arm_id")
            for role, parent_id in (("ATT", releases["ATT"]),
                                    ("INNER", releases["INNER"])):
                parent = plan[parent_id]
                if (parent["kind"] != release_kind or
                        parent.get(identity) != node.get(identity) or
                        parent.get("seed") != node.get("seed") or
                        parent.get("role") != ("ATTACK_FIT_NEW" if role == "ATT"
                                               else "INNER_SELECTION")):
                    raise ValueError(f"{unit_id} {kind.replace('_', ' ')} {role} release identity differs")
            if q_id is None:
                if (kind != "reference_attack" or
                        node.get("reference_id") not in {"U0", "Ucal"} or
                        node.get("trained_reader_and_probe_fits") != 24):
                    raise ValueError(f"{unit_id} Q absent outside continuous reference")
            else:
                q = plan[q_id]
                if (q["kind"] != "q_preprocess" or
                        q.get(identity) != node.get(identity) or
                        q.get("seed") != node.get("seed") or
                        node.get("trained_reader_and_probe_fits") != 33):
                    raise ValueError(f"{unit_id} Q identity differs")


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
    admitted_sources = amendment.get("source_sha256", {})
    if not isinstance(admitted_sources, dict):
        raise ValueError("A source snapshot malformed")
    for relative, digest in sorted(admitted_sources.items()):
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("A source record malformed")
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("A source escapes source root")
        _check_hash(source, digest, f"A source {relative}")
    cap_sum = 0.0
    for unit_id, node in plan.items():
        if not isinstance(unit_id, str) or not UNIT_ID.fullmatch(unit_id):
            raise ValueError("invalid unit ID")
        if not isinstance(node, dict) or node.get("kind") not in KNOWN_KINDS:
            raise ValueError(f"{unit_id} kind unrecognized")
        if node.get("output_id") != unit_id:
            raise ValueError(f"{unit_id} output_id differs")
        cpu = _positive_cap(node, "cpu_seconds", unit_id)
        wall = _positive_cap(node, "wall_seconds", unit_id)
        if cpu != wall:
            raise ValueError(f"{unit_id} cpu_seconds differs from wall_seconds")
        cap_sum += cpu
        allowed_null = ({"q_unit_id"} if node["kind"] == "reference_attack" and
                        node.get("reference_id") in {"U0", "Ucal"} else set())
        if any(value is None for key, value in node.items()
               if key not in allowed_null):
            raise ValueError(f"{unit_id} has a pending null field")
    order = _topological_order(plan)
    _check_parent_bindings(plan)
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
        "admitted_source_count": len(admitted_sources),
        "producer_cpu_cap_sum_s": cap_sum,
        "node_sha256": node_hashes,
        "topological_order": order,
    }

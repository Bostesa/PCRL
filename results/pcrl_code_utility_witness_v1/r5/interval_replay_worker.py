"""Independent queued replay of the 144 estimable utility interval clauses.

Uses R5's separate decoder/row scoring and an independently written paired
bootstrap. R4 endpoint, candidate and interval modules are never imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people,
)
from .independent_checks import task_rows
from .independent_release import replay_bank, replay_hard
from .replay_worker import (_pin, _registered, _role_path, _r1_lineage,
                            _release, _package, _sha)


HERE = Path(__file__).resolve().parent
ARMS = ("T0_B10", "T0_B16", "W_Q", "W_TSTAR")
TASKS = (("income", 0, 2), ("occupation", 1, 6))
REFERENCES = ("U0", "Ucal")
METRICS = ("accuracy", "nll", "brier")
SEED = 24681357
DRAWS = 2048
K_FULL = 2376


def _cost_rows(candidate: np.ndarray, reference: np.ndarray,
               labels: np.ndarray, classes: int) -> dict[str, np.ndarray]:
    """Signed per-person differences, computed apart from R4's scoring."""
    c = task_rows(labels, candidate, classes)
    r = task_rows(labels, reference, classes)
    return {metric: c[metric] - r[metric] for metric in METRICS}


def _group_draws(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) != 2234 or len(groups) != 2235:
        raise ValueError("paired INNER group geometry differs")
    rng = np.random.Generator(np.random.PCG64(SEED))
    sample = rng.integers(0, len(unique), size=(DRAWS, len(unique)),
                          dtype=np.int64)
    counts = np.empty((DRAWS, len(unique)), dtype=np.int64)
    for i in range(DRAWS):
        counts[i] = np.bincount(sample[i], minlength=len(unique))
    return counts, inverse


def _interval(values: np.ndarray, counts: np.ndarray,
              inverse: np.ndarray, critical: float) -> tuple[float, float, float, float, np.ndarray]:
    point = float(np.mean(values))
    group_size = np.bincount(inverse).astype(np.float64)
    group_sum = np.bincount(inverse, weights=values, minlength=len(group_size))
    replicate = (counts @ group_sum) / (counts @ group_size)
    se = float(np.std(replicate, ddof=1))
    return point, se, point - critical * se, point + critical * se, replicate


def _status(metric: str, lo: float, hi: float) -> str:
    if metric == "accuracy":
        return "PASS" if lo >= -.010 else "VIOLATION" if hi < -.010 else "UNRESOLVED"
    cap = .010 if metric == "nll" else .005
    return "PASS" if hi <= cap else "VIOLATION" if lo > cap else "UNRESOLVED"


def _wire(case: dict, registered: dict[Path, str], seal: dict,
          row_id: np.ndarray, unit: np.ndarray,
          labels: tuple[np.ndarray, np.ndarray]) -> tuple[dict, float]:
    arm, seed = case["arm"], case["seed"]
    source_sha = seal["source_model_files_sha256_by_seed"][str(seed)]["model_pt_sha256"]
    _r1_lineage(case, registered, seal["role_manifest_sha256"], source_sha)
    wire = _release(case["release"], registered, case["kind"])
    if case["kind"] == "bank":
        public = json.loads(_pin(case["public_table"], registered).read_text())
        if (public.get("schema") != "code-utility-public-bank-tables-v1"
                or public.get("seed") != seed
                or public.get("source_model_sha256") != source_sha
                or public.get("feature_schema_sha256") != seal["source_schema_sha256"]):
            raise PermissionError("independent bank table source differs")
        family = public["families"][arm]
        checked = replay_bank(wire, row_id, unit, labels,
                              (np.asarray(family["table0"], np.float64),
                               np.asarray(family["table1"], np.float64)))
    else:
        manifest, state = _package(case, registered,
                                   seal["role_manifest_sha256"], source_sha,
                                   seal["source_schema_sha256"])
        alpha = tuple(float(x["alpha"]) for x in manifest["calibrators"])
        checked = replay_hard(wire, row_id, unit, labels, state, case["bits"], alpha)
    return wire, checked["maximum_decoder_probability_gap"]


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    path = config_path.resolve(strict=True)
    registered = _registered(active["unit_id"], path)
    config = json.loads(path.read_text())
    if (config.get("schema") != "pcrl-r5-independent-interval-replay-config-v1"
            or config.get("unit_id") != active["unit_id"]
            or len(config.get("cases", [])) != 8):
        raise PermissionError("independent interval config differs")
    expected = {(arm, seed) for arm in ARMS for seed in (0, 1)}
    if {(case["arm"], case["seed"]) for case in config["cases"]} != expected:
        raise PermissionError("independent eight-case family differs")
    for name, expected_hash in config["verifier_sources_sha256"].items():
        _pin({"path": str(HERE / name), "sha256": expected_hash}, registered)
    seal = json.loads(_pin(config["seal"], registered).read_text())
    if (config["seal"]["sha256"] != active["seal_sha256"]
            or config["role_manifest"]["sha256"] != seal["role_manifest_sha256"]):
        raise PermissionError("independent interval seal or role differs")
    lock = json.loads(_pin(config["inference_lock"], registered).read_text())
    if (lock.get("campaign_lock_sha256") != active["seal_sha256"]
            or lock.get("randomness", {}).get("seed_integer") != SEED
            or lock.get("paired_draws") != DRAWS
            or lock.get("fixed_family_K") != K_FULL):
        raise PermissionError("independent interval randomness lock differs")
    store = RoleOnlyStore(_pin(config["role_manifest"], registered))
    if store.manifest["source_schema_sha256"] != seal["source_schema_sha256"]:
        raise PermissionError("independent interval predictor schema differs")
    for field in ("income", "occupation", "references_s0", "references_s1"):
        _role_path(store, "INNER_SELECTION", field, registered)
    income = store.read("INNER_SELECTION", "income", "screen")
    occupation = store.read("INNER_SELECTION", "occupation", "screen")
    assert_same_people(income, occupation)
    row_id, unit = income["row_id"], income["unit"]
    labels = (income["income"], occupation["occupation"])
    reference = {}
    for seed in (0, 1):
        record = store.read("INNER_SELECTION", f"references_s{seed}")
        assert_same_people(income, record)
        reference[seed] = record
    reported_path = _pin(config["r4_report"], registered)
    reported = json.loads(reported_path.read_text())
    if (reported.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_AGGREGATES"
            or reported.get("family_k") != K_FULL
            or reported.get("bootstrap_seed") != SEED
            or reported.get("bootstrap_draws") != DRAWS
            or reported.get("ordered_ids_sha256") != lock["ordered_ids_sha256"]
            or len(reported.get("ordered_ids", [])) != K_FULL
            or len(reported.get("clauses", {})) != K_FULL):
        raise PermissionError("R4 aggregate family or fixed randomness differs")
    if hashlib.sha256(json.dumps(reported["ordered_ids"], ensure_ascii=False,
                                 separators=(",", ":")).encode()).hexdigest() != lock["ordered_ids_sha256"]:
        raise PermissionError("R4 ordered family digest differs")
    if set(reported["ordered_ids"]) != set(reported["clauses"]):
        raise PermissionError("R4 aggregate clause inventory differs")
    critical = float(t.ppf(1 - .05 / (2 * K_FULL), len(np.unique(unit)) - 1))
    if abs(critical - float(lock["critical_t_df2233"])) > 1e-12:
        raise PermissionError("independently recomputed full-K critical differs")
    counts, inverse = _group_draws(unit)
    own: dict[str, dict] = {}
    paired: dict[str, dict[int, tuple[float, np.ndarray]]] = {}
    wire_gap = 0.0
    for case in config["cases"]:
        assert_active_from_env(unit_id=active["unit_id"], stage="verification")
        wire, gap = _wire(case, registered, seal, row_id, unit, labels)
        wire_gap = max(wire_gap, gap)
        arm, seed = case["arm"], case["seed"]
        for task, purpose, classes in TASKS:
            candidate = wire[f"prob{purpose}"]
            for ref in REFERENCES:
                baseline = reference[seed][f"{ref}_{purpose+1}"]
                for metric, values in _cost_rows(candidate, baseline,
                                                 labels[purpose], classes).items():
                    base = f"utility:{arm}:{task}:{ref}:{metric}"
                    point, se, lo, hi, replicate = _interval(values, counts, inverse, critical)
                    own[f"s{seed}:{base}"] = {"point": point, "se": se,
                                             "lower": lo, "upper": hi,
                                             "status": _status(metric, lo, hi)}
                    paired.setdefault(base, {})[seed] = (point, replicate)
    for base, seeds in paired.items():
        if set(seeds) != {0, 1}:
            raise ValueError("seed mean lacks paired model seeds")
        metric = base.rsplit(":", 1)[-1]
        point = (seeds[0][0] + seeds[1][0]) / 2
        replicate = (seeds[0][1] + seeds[1][1]) / 2
        se = float(np.std(replicate, ddof=1))
        lo, hi = point - critical * se, point + critical * se
        own[f"mean:{base}"] = {"point": point, "se": se,
                              "lower": lo, "upper": hi,
                              "status": _status(metric, lo, hi)}
    if len(own) != 144:
        raise ValueError("independent estimable utility family size differs")
    numeric_gap = 0.0
    statuses = {"PASS": 0, "UNRESOLVED": 0, "VIOLATION": 0}
    for name, clause in reported["clauses"].items():
        if name not in own:
            if clause != {"point": None, "se": None, "lower": None,
                          "upper": None, "status": "NOT_ESTIMABLE_DEPENDENCY"}:
                raise ValueError("unavailable clause was reported as estimable")
            continue
        expected_clause = own[name]
        for key in ("point", "se", "lower", "upper"):
            numeric_gap = max(numeric_gap, abs(expected_clause[key] - clause[key]))
        if expected_clause["status"] != clause["status"]:
            raise ValueError(f"independent interval status differs for {name}")
        statuses[expected_clause["status"]] += 1
    if numeric_gap > 1e-10:
        raise ValueError(f"independent interval numeric gap {numeric_gap} exceeds 1e-10")
    output = Path(config["output"]).resolve(strict=False)
    if output.exists() or output.parent != HERE:
        raise PermissionError("new R5 public aggregate output required")
    result = {"schema": "pcrl-r5-independent-utility-interval-replay-v1",
              "status": "PASS", "unit_id": active["unit_id"],
              "scope": "conditional paired-group reused INNER development intervals; no privacy attacks",
              "campaign_seal_sha256": active["seal_sha256"],
              "inference_lock_sha256": config["inference_lock"]["sha256"],
              "r4_report_sha256": config["r4_report"]["sha256"],
              "rows": len(row_id), "groups": int(len(np.unique(unit))),
              "full_family_k": K_FULL, "estimable": len(own),
              "dependency_missing": K_FULL - len(own),
              "bootstrap_draws": DRAWS, "bootstrap_seed": SEED,
              "critical_95": critical, "interval_status_counts": statuses,
              "max_abs_numeric_gap": numeric_gap,
              "max_public_code_decoder_gap": wire_gap}
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return {"status": "PASS", "unit_id": active["unit_id"],
            "receipt_sha256": _sha(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))

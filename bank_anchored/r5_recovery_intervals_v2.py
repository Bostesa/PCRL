"""Versioned independent recovery intervals with explicit S21 null slots.

The input is the frozen complete-view reader predictions from all 38 effective
ATT-fitted attacks. This module never imports R4 attack or R3 endpoint code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.stats import t as student_t


ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21", "W_Q", "P_STAR")
ESTIMABLE_ARMS = tuple(arm for arm in ARMS if arm != "S21")
VIEWS = ("r1", "r2", "pair")
SUMMARIES = ("s0_d0", "s0_d1", "s0_d2", "s1_d0", "s1_d1", "s1_d2",
             "s0_mean", "s1_mean", "mean")
STOCHASTIC = frozenset(("T", "P", "L", "J", "S12", "S21"))
BOOTSTRAPS = 2048
GROUPS = 2234
K = 1575
FLOOR = 1e-12


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as source:
        members = [entry.filename for entry in source.infolist()]
        if len(members) != len(keys) or set(members) != {name + ".npy" for name in keys}:
            raise PermissionError("reader prediction archive member inventory differs")
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in keys}


def weighted_auc_batch(y: np.ndarray, score: np.ndarray, group_index: np.ndarray,
                       weights: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC under many duplicate-group weight vectors at once."""
    truth = np.asarray(y)
    values = np.asarray(score)
    group = np.asarray(group_index)
    w = np.asarray(weights)
    if (truth.dtype != np.int64 or truth.ndim != 1 or
            values.dtype != np.float64 or values.shape != truth.shape or
            group.dtype != np.int64 or group.shape != truth.shape or
            w.ndim != 2 or w.shape[1] != GROUPS or
            not np.isin(truth, (0, 1)).all() or
            not np.isfinite(values).all() or np.any(w < 0) or
            np.any((group < 0) | (group >= GROUPS))):
        raise ValueError("weighted AUC input or group schema differs")
    order = np.argsort(values, kind="mergesort")
    scored = values[order]
    starts = np.r_[0, np.flatnonzero(scored[1:] != scored[:-1]) + 1]
    observed = w[:, group[order]].astype(np.float64, copy=False)
    positive = np.add.reduceat(observed * truth[order][None, :], starts, axis=1)
    negative = np.add.reduceat(observed * (1 - truth[order])[None, :], starts, axis=1)
    lower_negative = np.cumsum(negative, axis=1) - negative
    numer = np.sum(positive * (lower_negative + .5 * negative), axis=1)
    den = positive.sum(axis=1) * negative.sum(axis=1)
    if np.any(den <= 0):
        raise ValueError("duplicate-group replicate lacks one SEX class")
    return numer / den


def source_auc_vectors(manifest: dict, weights: np.ndarray,
                       truth: dict[str, np.ndarray]) -> dict[tuple, np.ndarray]:
    values: dict[tuple, np.ndarray] = {}
    unit = truth["unit"]
    groups, inverse = np.unique(unit, return_inverse=True)
    if len(groups) != GROUPS or not np.array_equal(weights[0], np.ones(GROUPS)):
        raise ValueError("INNER group identity or point weights differ")
    inverse = inverse.astype(np.int64)
    for entry in manifest["sources"]:
        key = (entry["arm_id"], entry["source_seed"], entry["realization"])
        receipt = Path(entry["r4_receipt_path"])
        replay = Path(entry["r5_replay_path"])
        if (sha(receipt) != entry["r4_receipt_sha256"] or
                sha(replay) != entry["r5_replay_sha256"]):
            raise PermissionError("frozen R4/R5 source receipt changed")
        r4 = json.loads(receipt.read_text())
        r5 = json.loads(replay.read_text())
        if (r4.get("unit_id") != entry["unit_id"] or
                r5.get("source_unit_id") != entry["unit_id"] or
                r5.get("status") != "PASS" or
                r5.get("r4_receipt_sha256") != entry["r4_receipt_sha256"] or
                (r4["arm_id"], r4["source_seed"], r4["realization"]) != key):
            raise PermissionError("frozen release provenance differs")
        private = Path(entry["private_attack_dir"])
        for view in VIEWS:
            reader = private / view / "readers"
            custody_path = reader / "READERS_RECEIPT.json"
            custody_sha = r4["attack"]["reader_custody"][view]["private_receipt_sha256"]
            if sha(custody_path) != custody_sha:
                raise PermissionError("frozen reader custody changed")
            custody = json.loads(custody_path.read_text())
            pred_path = reader / "predictions_inner.npz"
            role_path = reader / "view_inner.npz"
            if (sha(pred_path) != custody["files"]["predictions_inner.npz"] or
                    sha(role_path) != custody["files"]["view_inner.npz"]):
                raise PermissionError("frozen INNER predictions or view changed")
            role = archive(role_path, {"row_id", "unit", "x", "x_defense_aware",
                                       "x_code_bits", "cell_key"} if
                           entry["arm_id"] != "P_STAR" else
                           {"row_id", "unit", "x", "x_defense_aware",
                            "x_metadata", "x_code_bits", "cell_key"})
            if (not np.array_equal(role["row_id"], truth["row_id"]) or
                    not np.array_equal(role["unit"], truth["unit"])):
                raise PermissionError("frozen reader and SEX people differ")
            order = tuple(custody["reader_order"])
            predictions = archive(pred_path, set(order))
            envelope = np.full(len(weights), .5, dtype=np.float64)
            for name in order:
                score = np.asarray(predictions[name])
                auc = weighted_auc_batch(truth["sex"], score, inverse, weights)
                envelope = np.maximum(envelope, np.maximum(auc, 1 - auc))
            if abs(envelope[0] - float(r5["checked_views"][view]["envelope_auc"])) > 1e-11:
                raise ValueError("point frozen-reader envelope differs")
            values[key + (view,)] = envelope
    return values


def intervals(values: dict[tuple, np.ndarray], draw_indices: np.ndarray) -> dict[str, dict]:
    if draw_indices.shape != (BOOTSTRAPS, 2, 3) or np.any((draw_indices < 0) |
                                                            (draw_indices > 2)):
        raise ValueError("coupled complete-release draw resample differs")
    summarized: dict[tuple, np.ndarray] = {}
    for arm in ESTIMABLE_ARMS:
        for seed in (0, 1):
            for view in VIEWS:
                draws = [values[(arm, seed, d if arm in STOCHASTIC else 0, view)]
                         for d in range(3)]
                for d in range(3):
                    summarized[(f"s{seed}_d{d}", arm, view)] = draws[d]
                matrix = np.stack(draws, axis=1)
                sampled = matrix[1:][np.arange(BOOTSTRAPS)[:, None],
                                     draw_indices[:, seed, :]].mean(axis=1)
                summarized[(f"s{seed}_mean", arm, view)] = np.r_[
                    matrix[0].mean(), sampled]
        for view in VIEWS:
            summarized[("mean", arm, view)] = (
                summarized[("s0_mean", arm, view)] +
                summarized[("s1_mean", arm, view)]) / 2
    fixed_crit = float(student_t.ppf(1 - .05 / (2 * K), GROUPS - 1))
    mean_crit = float(student_t.ppf(1 - .05 / (2 * K), 2))
    result: dict[str, dict] = {}
    for summary in SUMMARIES:
        crit = fixed_crit if "_d" in summary else mean_crit
        for first, second in combinations(ARMS, 2):
            for view in VIEWS:
                name = f"{summary}:recovery:{first}:{second}:{view}:auc"
                if "S21" in (first, second):
                    result[name] = {"point": None, "se": None,
                                    "lower": None, "upper": None,
                                    "status": "CONDITIONAL_NULL",
                                    "reason": "S21_INNER_POINT_GATE_FAILED_NO_ATTACK_FIT"}
                    continue
                difference = (summarized[(summary, first, view)] -
                              summarized[(summary, second, view)])
                point = float(difference[0])
                replicate = difference[1:]
                se = (0.0 if np.max(np.abs(replicate - point)) <= FLOOR else
                      float(np.std(replicate, ddof=1)))
                width = crit * se
                result[name] = {"point": point, "se": se,
                                "lower": point - width, "upper": point + width,
                                "status": "BOUND_ONLY"}
    if len(result) != 1215:
        raise AssertionError("fixed recovery interval family changed")
    if (sum(item["status"] == "CONDITIONAL_NULL" for item in result.values()) != 243 or
            sum(item["status"] == "BOUND_ONLY" for item in result.values()) != 972):
        raise AssertionError("S21 conditional null accounting changed")
    return result


def directed_criterion(clauses: dict, first: str, second: str, view: str,
                       *, direction: int, cap: float, kind: str) -> dict:
    pair = (first, second) if ARMS.index(first) < ARMS.index(second) else (second, first)
    source = clauses[f"mean:recovery:{pair[0]}:{pair[1]}:{view}:auc"]
    if source["status"] == "CONDITIONAL_NULL":
        raise ValueError("conditional-null arm cannot enter a directed criterion")
    sign = direction if pair == (first, second) else -direction
    point = sign * source["point"]
    lower, upper = ((source["lower"], source["upper"]) if sign == 1 else
                    (-source["upper"], -source["lower"]))
    if kind == "benefit":
        status = "PASS" if lower >= cap else "VIOLATION" if upper < cap else "UNRESOLVED"
    elif kind == "cost":
        status = "PASS" if upper <= cap else "VIOLATION" if lower > cap else "UNRESOLVED"
    else:
        raise ValueError("criterion direction differs")
    return {"point": point, "lower": lower, "upper": upper, "status": status}


def run(manifest_path: Path, manifest_sha: str, output: Path) -> dict:
    if sha(manifest_path) != manifest_sha:
        raise PermissionError("R5 recovery source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "BAR1-r5-recovery-source-manifest-v1" or
            len(manifest.get("sources", ())) != 38 or
            manifest.get("bootstrap_replicates") != BOOTSTRAPS or
            manifest.get("fixed_family_K") != K):
        raise PermissionError("R5 recovery predeclared family differs")
    if ({row["arm_id"] for row in manifest["sources"]} != set(ESTIMABLE_ARMS) or
            any(row["arm_id"] == "S21" for row in manifest["sources"])):
        raise PermissionError("S21 absence or eligible attack family differs")
    sex_path = Path(manifest["inner_sex_path"])
    if sha(sex_path) != manifest["inner_sex_sha256"]:
        raise PermissionError("allowed INNER SEX export changed")
    truth = archive(sex_path, {"row_id", "unit", "sex"})
    unit = truth["unit"]
    if (len(unit) != 2235 or len(np.unique(unit)) != GROUPS or
            truth["sex"].dtype != np.int64 or
            not np.isin(truth["sex"], (0, 1)).all()):
        raise PermissionError("INNER label/group schema differs")
    rng = np.random.Generator(np.random.PCG64DXSM(20261010))
    sampled = rng.integers(0, GROUPS, size=(BOOTSTRAPS, GROUPS), dtype=np.int64)
    weights = np.empty((BOOTSTRAPS + 1, GROUPS), dtype=np.int16)
    weights[0] = 1
    for b, draw in enumerate(sampled, 1):
        weights[b] = np.bincount(draw, minlength=GROUPS)
    release_rng = np.random.Generator(np.random.PCG64DXSM(20261011))
    draw_indices = release_rng.integers(0, 3, size=(BOOTSTRAPS, 2, 3))
    auc = source_auc_vectors(manifest, weights, truth)
    if len(auc) != 114:
        raise ValueError("38-by-three-view frozen reader source family differs")
    clauses = intervals(auc, draw_indices)
    selected = {
        "coalition_benefit_pstar_minus_j": directed_criterion(
            clauses, "P_STAR", "J", "pair", direction=1, cap=.020, kind="benefit"),
        "j_minus_pstar_r1": directed_criterion(
            clauses, "J", "P_STAR", "r1", direction=1, cap=.010, kind="cost"),
        "j_minus_pstar_r2": directed_criterion(
            clauses, "J", "P_STAR", "r2", direction=1, cap=.010, kind="cost"),
        "j_minus_b0_r1": directed_criterion(
            clauses, "J", "B0", "r1", direction=1, cap=.010, kind="cost"),
        "j_minus_b0_r2": directed_criterion(
            clauses, "J", "B0", "r2", direction=1, cap=.010, kind="cost"),
    }
    result = {"schema": "BAR1-r5-independent-recovery-intervals-v2",
              "status": "COMPLETE_DEVELOPMENT_ONLY", "source_manifest_sha256": manifest_sha,
              "fixed_family_K": K, "recovery_slots": len(clauses),
              "estimable_recovery_slots": 972, "conditional_null_recovery_slots": 243,
              "conditional_null_arm": "S21",
              "bootstrap_replicates": BOOTSTRAPS, "inner_groups": GROUPS,
              "group_seed": 20261010, "release_seed": 20261011,
              "same_group_draw_across_all_arms": True,
              "same_three_complete_draw_ids_per_seed_across_all_arms": True,
              "fixed_draw_df": GROUPS - 1, "mean_draw_df": 2,
              "critical_family": "two-sided-fixed-K-Bonferroni-t-approximation",
              "clauses": clauses, "selected_criteria": selected,
              "scope": "conditional on fitted policies/readers; reused INNER people and three cached complete releases"}
    target = output.resolve(strict=False)
    expected = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r5"
    if target.parent != expected or target.exists():
        raise PermissionError("R5 recovery output path differs")
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": "BAR1-r5-recovery-interval-receipt-v2", "status": "PASS",
            "result_sha256": sha(target), "source_manifest_sha256": manifest_sha,
            "recovery_slots": len(clauses), "estimable_recovery_slots": 972,
            "conditional_null_recovery_slots": 243,
            "selected_criteria": selected}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for arg in ("manifest", "manifest-sha256", "output", "receipt"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    active = assert_active_from_env(stage="verification")
    receipt = run(Path(args.manifest), args.manifest_sha256, Path(args.output))
    receipt["unit_id"] = active["unit_id"]
    path = Path(args.receipt).resolve(strict=False)
    expected = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r5"
    if path.parent != expected or path.exists():
        raise PermissionError("R5 recovery receipt path differs")
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

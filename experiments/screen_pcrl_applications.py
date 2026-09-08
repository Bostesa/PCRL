"""Development-only application screening, without representation-method runs.

Reads HAR official train and Diabetes processed train only. The first oracle
checks were exploratory; their reproduction is explicitly retrospective. The
one learned-output reference has fixed settings and disjoint HAR participants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, mutual_info_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CONFIG = {
    "oracle_seed": 9137,
    "oracle_fit_fraction": 0.7,
    "oracle_order": ["har_official_train", "diabetes_processed_train"],
    "oracle_additive_smoothing": 1.0,
    "learned_reference": {
        "dataset": "HAR official training participants only",
        "subject_split_seed": 9138,
        "subject_group_sizes": [13, 4, 4],
        "task": "is_active = activity source ID <= 3",
        "task_model": "StandardScaler + LogisticRegression",
        "C": 1.0,
        "max_iter": 2000,
        "tol": 1e-4,
        "solver": "lbfgs",
        "attack": "add-one-smoothed categorical conditional frequency",
        "probability_bins": 20,
        "bin_edges": "fixed equal-width [0,1], no data fitting",
        "attribute": "six-way activity",
        "selection": "none; one fixed task model, no search",
    },
    "threads": 1,
}

DATASET_SOURCES = {
    "har_official_train": {
        "version": "UCI HAR Dataset Version 1.0; local README November 2013",
        "url": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        "doi": "10.24432/C54S4K",
    },
    "diabetes_processed_train": {
        "version": "Diabetes 130-US hospitals 1999-2008; repository processed split seed42",
        "url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "doi": "10.24432/C5230J",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def coverage(values: np.ndarray, size: int) -> list[int]:
    return np.bincount(values, minlength=size).tolist()


def conditional_attack(
    code_fit: np.ndarray, code_eval: np.ndarray,
    z_fit: np.ndarray, z_eval: np.ndarray, *, n_codes: int, n_classes: int,
) -> dict:
    counts = np.ones((n_codes, n_classes), dtype=np.float64)
    np.add.at(counts, (code_fit, z_fit), 1)
    p = (counts / counts.sum(1, keepdims=True))[code_eval]
    prior = (np.bincount(z_fit, minlength=n_classes) + 1)/(len(z_fit) + n_classes)
    base = np.tile(prior, (len(z_eval), 1))
    fit_support, eval_support = coverage(z_fit, n_classes), coverage(z_eval, n_classes)
    if min(fit_support) <= 0 or min(eval_support) <= 0:
        raise ValueError("Undefined attribute coverage; this cannot count as a privacy pass")
    base_loss = log_loss(z_eval, base, labels=np.arange(n_classes))
    attack_loss = log_loss(z_eval, p, labels=np.arange(n_classes))
    return {
        "prior_accuracy": float(accuracy_score(z_eval, base.argmax(1))),
        "conditional_accuracy": float(accuracy_score(z_eval, p.argmax(1))),
        "conditional_balanced_accuracy": float(balanced_accuracy_score(z_eval, p.argmax(1))),
        "prior_log_loss_nats": float(base_loss),
        "conditional_log_loss_nats": float(attack_loss),
        "log_loss_gain_nats": float(base_loss - attack_loss),
        "attribute_fit_support": fit_support,
        "attribute_development_support": eval_support,
        "code_fit_support": coverage(code_fit, n_codes),
        "code_development_support": coverage(code_eval, n_codes),
        "all_attribute_classes_supported": True,
        "privacy_pass": None,
    }


def oracle_screen(name: str, arrays: dict, tasks: list[str], attrs: list[str], rng, *, joint=False):
    n = len(arrays[tasks[0]])
    order = rng.permutation(n)
    cut = int(CONFIG["oracle_fit_fraction"]*n)
    fit, dev = order[:cut], order[cut:]
    outputs = {k: arrays[k] for k in tasks}
    if joint:
        outputs["joint_three_label_bank"] = np.ravel_multi_index(
            np.stack([arrays[k] for k in tasks]), (9, 2, 2))
    rows = []
    for task, y in outputs.items():
        for attr in attrs:
            z = arrays[attr]
            metric = conditional_attack(y[fit], y[dev], z[fit], z[dev],
                                        n_codes=int(y.max())+1, n_classes=int(z.max())+1)
            metric.update(task=task, attribute=attr,
                          empirical_training_pool_MI_nats=float(mutual_info_score(y, z)))
            rows.append(metric)
    return {
        "dataset": name, "kind": "oracle exact-label release; retrospective exploratory check",
        "rows": n, "fit_rows": len(fit), "development_rows": len(dev),
        "split_indices_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "training_pool_support": {k: coverage(arrays[k], int(arrays[k].max())+1)
                                  for k in tasks + attrs},
        "metrics": rows,
    }


def learned_har_reference(x: np.ndarray, activity: np.ndarray, subject: np.ndarray, out: Path):
    cfg = CONFIG["learned_reference"]
    people = np.random.default_rng(cfg["subject_split_seed"]).permutation(np.unique(subject))
    assert len(people) == sum(cfg["subject_group_sizes"]) == 21
    train_people, attacker_people, dev_people = people[:13], people[13:17], people[17:]
    group_people = {"task_fit": train_people, "attacker_fit": attacker_people,
                    "development": dev_people}
    indices = {name: np.flatnonzero(np.isin(subject, ids)) for name, ids in group_people.items()}
    assert not set(train_people) & set(attacker_people)
    assert not set(train_people) & set(dev_people)
    assert not set(attacker_people) & set(dev_people)
    active = (activity < 3).astype(np.int64)
    scaler = StandardScaler().fit(x[indices["task_fit"]])
    task = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], tol=cfg["tol"],
                              solver=cfg["solver"], random_state=cfg["subject_split_seed"])
    task.fit(scaler.transform(x[indices["task_fit"]]), active[indices["task_fit"]])
    if int(task.n_iter_.max()) >= cfg["max_iter"]:
        raise RuntimeError("Fixed task model did not converge; retain run as invalid")
    probabilities = {name: task.predict_proba(scaler.transform(x[idx]))[:, 1]
                     for name, idx in indices.items()}
    utility = {}
    for name, idx in indices.items():
        p = probabilities[name]
        utility[name] = {
            "accuracy": float(accuracy_score(active[idx], p >= .5)),
            "balanced_accuracy": float(balanced_accuracy_score(active[idx], p >= .5)),
            "log_loss_nats": float(log_loss(active[idx], np.stack([1-p, p], 1), labels=[0, 1])),
            "task_support": coverage(active[idx], 2),
            "activity_support": coverage(activity[idx], 6),
        }
    fit, dev = indices["attacker_fit"], indices["development"]
    n_bins = cfg["probability_bins"]
    predictions = {
        "oracle_active_bit": ((active[fit]), (active[dev]), 2),
        "learned_hard_active_output": ((probabilities["attacker_fit"] >= .5).astype(int),
                                       (probabilities["development"] >= .5).astype(int), 2),
        "learned_active_probability_20_bins": (
            np.minimum((probabilities["attacker_fit"]*n_bins).astype(int), n_bins-1),
            np.minimum((probabilities["development"]*n_bins).astype(int), n_bins-1), n_bins),
    }
    leakage = {name: conditional_attack(a, b, activity[fit], activity[dev],
                                        n_codes=k, n_classes=6)
               for name, (a, b, k) in predictions.items()}
    np.savez_compressed(out/"learned_task_reference.npz", coef=task.coef_, intercept=task.intercept_,
                        classes=task.classes_, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                        **{f"indices_{k}": v for k, v in indices.items()},
                        **{f"probability_{k}": v for k, v in probabilities.items()})
    return {
        "kind": "single fixed learned-output reference, no representation methods",
        "split": {name: {"subject_ids": group_people[name].tolist(), "rows": len(idx)}
                  for name, idx in indices.items()},
        "task_iterations": task.n_iter_.tolist(), "task_utility": utility,
        "activity_leakage": leakage,
        "unknown_person_identity_attack": "not measured; fixed identity classes do not transfer across disjoint people",
        "privacy_pass": None,
        "checkpoint_sha256": sha256(out/"learned_task_reference.npz"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    protocol = """# Application-screen protocol and chronology

This directory reproduces earlier exploratory oracle-label checks. Their protocol
description is RETROSPECTIVE, not a predeclared method comparison. Earlier values
were already viewed before this file was written. They are retained in metrics.json
and checked against the previous observations rather than silently replaced.

The additional learned-output reference is fixed before execution: only official
HAR training subjects, seed9138 permuted into13 task-fit/4 attacker-fit/4 development
subjects; no official test files. Fit one StandardScaler+binary LogisticRegression
(C1,lbfgs,tol1e-4,max_iter2000) on task-fit subjects to predict active/sedentary.
Freeze it; attack activity from hard output or20 fixed equal-width probability
bins using add-one conditional counts fitted on the attacker subjects. Evaluate
on the last4 subjects. No model, threshold, bin-count, or task selection on these
outcomes. These attacks are bounded diagnostics, not universal guarantees.

Oracle checks use only HAR official train labels and Diabetes processed train
labels, RNG9137 sequentially,70/30 fitting/development row partitions, add-one
conditional-count attackers. These random HAR windows do not establish session
generalization. The learned reference uses different, disjoint participant groups.

The application decision concerns missing evidence for reusable-release necessity;
it does not prove rich releases are unnecessary. No PCRL or other representation
method is run. No real-data application pilot is selected. No final-test results
are used. Inventory counts/names include existing test files; only explicitly
listed training file contents are read or hashed. Configuration, source hash,
data hashes, coverage, exact scores, and compute time are saved. Historical files
are never modified. No downloads or dependencies are installed.
"""
    (out/"PROTOCOL.md").write_text(protocol)
    print("Fixed screening configuration:", json.dumps(CONFIG, sort_keys=True), flush=True)
    inventory = []
    for directory in sorted(Path("data").iterdir()):
        if directory.is_dir() and directory.name != "__MACOSX":
            files = [p for p in directory.rglob("*") if p.is_file()]
            inventory.append({"directory": str(directory), "files": len(files),
                              "bytes": sum(p.stat().st_size for p in files),
                              "content_access": "inventory only except files explicitly hashed below"})
    har_base = Path("data/UCI HAR Dataset/train")
    input_paths = [har_base/"y_train.txt", har_base/"subject_train.txt", har_base/"X_train.txt",
                   Path("data/diabetes_processed/train.npz")]
    hashes = {str(p): sha256(p) for p in input_paths}
    save_json(out/"local_data_inventory.json", {"datasets": inventory,
              "used_training_files_sha256": hashes, "sources": DATASET_SOURCES,
              "BIOS_present": Path("data/bios").exists()})
    activity = np.loadtxt(har_base/"y_train.txt", dtype=np.int64)-1
    subject = np.loadtxt(har_base/"subject_train.txt", dtype=np.int64)
    _, subject_codes = np.unique(subject, return_inverse=True)
    rng = np.random.default_rng(CONFIG["oracle_seed"])
    har = oracle_screen("har_official_train", {"activity": activity,
                        "is_active": (activity < 3).astype(int), "subject": subject_codes},
                        ["activity", "is_active"], ["subject", "activity"], rng)
    keys = ["primary_diagnosis_category", "readmission_outcome", "medication_change_outcome",
            "race", "gender", "age_bucket"]
    with np.load("data/diabetes_processed/train.npz") as source:
        diabetes_data = {k: source[k] for k in keys}
    diabetes = oracle_screen("diabetes_processed_train", diabetes_data, keys[:3], keys[3:], rng, joint=True)
    observed_har = next(row for row in har["metrics"] if row["task"] == "is_active" and row["attribute"] == "activity")
    observed_diabetes = next(row for row in diabetes["metrics"] if row["task"] == "primary_diagnosis_category" and row["attribute"] == "gender")
    assert abs(observed_har["conditional_accuracy"]-0.357207615593835) < 1e-12
    assert abs(observed_har["log_loss_gain_nats"]-0.687252540516186) < 1e-12
    assert abs(observed_diabetes["conditional_accuracy"]-0.5436201385189131) < 1e-12
    print("Earlier oracle development observations reproduced exactly.", flush=True)
    x = np.loadtxt(har_base/"X_train.txt", dtype=np.float64)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=CONFIG["threads"]):
        learned = learned_har_reference(x, activity, subject, out)
    learned_seconds = time.perf_counter()-fit_start
    result = {
        "config": CONFIG, "sources": DATASET_SOURCES,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": sha256(Path(__file__)), "used_training_files_sha256": hashes,
        "oracle_checks_retrospective": [har, diabetes],
        "earlier_observation_checks_passed": True, "learned_output_reference": learned,
        "selection": "no real-data application pilot; insufficient reusable-need evidence",
        "final_test_access": False,
        "runtime": {"total_seconds": time.perf_counter()-start,
                    "learned_task_and_attacks_seconds": learned_seconds,
                    "platform": platform.platform(), "machine": platform.machine(),
                    "logical_cpus": os.cpu_count(), "numerical_threads": CONFIG["threads"]},
    }
    save_json(out/"metrics.json", result)
    print(json.dumps({"learned_output_reference": learned, "runtime": result["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

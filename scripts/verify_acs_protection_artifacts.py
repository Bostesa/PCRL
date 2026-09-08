#!/usr/bin/env python3
"""Inference-only verification of ACS protection artifacts and fitting records.

Requires the local original ACS CSV, parent artifacts, and completed protection
artifacts. It neither fits models nor rewrites historical reports. The original
test pool is development evaluation in this stage, not untouched confirmation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concept_erasure import LeaceEraser
from experiments.acs_protection_maps import FrozenAffineMap, NUMERICAL_POLICY, LEACE_OPTIONS
from experiments.acs_transfer_heads import HEAD_BUDGET
from experiments.acs_transfer_models import state_digest

DEFAULT_RELATIVE = Path("results/redesign_20260908_acs_protection_v1")
TASKS = ("same_residence", "commute_over20", "income_binary", "civilian_at_work", "public_coverage")
SCHEMA = {"SEX": 2, "RAC1P": 9}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def recorded_path(value, out):
    path = Path(value)
    if path.name == "leace.py" and "concept_erasure" in path.parts:
        return Path(inspect.getfile(LeaceEraser))
    try:
        return out / path.relative_to(DEFAULT_RELATIVE)
    except ValueError:
        return ROOT / path


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def coded(value, k):
    value = np.asarray(value, float)
    return np.where(np.isin(value, np.arange(1, k + 1)), value - 1, -1).astype(np.int64)


def labels(frame):
    income = frame.PINCP.to_numpy(float)
    income_valid = np.isfinite(income) & (income >= -19998) & (income <= 4209995)
    esr, pubcov, mig = coded(frame.ESR, 6), coded(frame.PUBCOV, 2), coded(frame.MIG, 3)
    commute = frame.JWMNP.to_numpy(float)
    commute_valid = (np.isfinite(commute) & (commute >= 1) & (commute <= 200)
                     & (commute == np.floor(commute)))
    return {
        "SEX": coded(frame.SEX, 2), "RAC1P": coded(frame.RAC1P, 9),
        "income_binary": np.where(income_valid, income > 50000, -1).astype(np.int64),
        "civilian_at_work": np.where(esr < 0, -1, esr == 0).astype(np.int64),
        "public_coverage": np.where(pubcov < 0, -1, pubcov == 0).astype(np.int64),
        "same_residence": np.where(mig < 0, -1, mig == 0).astype(np.int64),
        "commute_over20": np.where(commute_valid, commute > 20, -1).astype(np.int64),
    }


def support(y, k):
    return {"valid": int((y >= 0).sum()), "missing_or_inapplicable": int((y < 0).sum()),
            "class_counts": np.bincount(y[y >= 0], minlength=k).tolist()}


def centered(value):
    value = np.asarray(value, dtype=np.float64)
    delta = value - value[0]
    mean_delta = delta.mean(0)
    return value[0] + mean_delta, delta - mean_delta


def rank(value):
    _, z = centered(value)
    eigenvalues = np.linalg.eigvalsh(z.T @ z / max(1, len(z) - 1)).clip(0)
    keep = eigenvalues > 1e-10 * eigenvalues.max() if eigenvalues.max() > 0 else np.zeros(len(eigenvalues), bool)
    return int(keep.sum())


def verify(out):
    tick = time.perf_counter()
    cfg = read(out / "config.json")
    freeze = read(out / "protocol_freeze.json")
    parent = ROOT / cfg["parent_results"]
    parent_cfg, oldsupport = read(parent / "config.json"), read(parent / "schema_support.json")
    for path, expected in freeze["sha256"].items():
        require(sha(recorded_path(path, out)) == expected, "Frozen execution/config identity: " + path)
    for path, expected in freeze["parent_record_hashes"].items():
        require(sha(ROOT / path) == expected, "Parent record changed: " + path)
    raw_path = ROOT / parent_cfg["raw_path"]
    require(sha(raw_path) == oldsupport["raw_sha256"], "Raw CSV identity")
    raw = pd.read_csv(raw_path, usecols=["SERIALNO", "SPORDER", "PINCP", "ESR", "PUBCOV", "MIG",
                                      "JWMNP", "SEX", "RAC1P"], dtype={"SERIALNO": str})
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "verification_source_sha256": sha(Path(__file__)), "command": [sys.executable, *sys.argv],
              "evaluation_status": cfg["evaluation_status"], "no_model_fits": True,
              "raw_data_sha256": oldsupport["raw_sha256"], "frozen_source_hashes_verified": True,
              "seeds": {}, "limitations": [
                  "Inference/metadata consistency replay; no independent retraining or reconstructed optimizer execution.",
                  "Source immutability across execution uses recorded before/after flags plus independently verified saved parent file hashes.",
                  "Recorded evaluation households were already used to motivate this stage; these are development results.",
                  "Covariance statistics are empirical and do not certify classification accuracy or nonlinear privacy."]}
    require(tuple(cfg["utility_tasks"]) == TASKS, "Task schema")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in cfg["seeds"]:
            directory, olddir = out / f"seed_{seed}", parent / f"seed_{seed}"
            metrics = read(directory / "metrics.json")
            release_freeze = read(directory / "release_freeze.json")
            selection = read(directory / "selection_before_test.json")
            provenance = read(directory / "parent_provenance.json")
            oldselection, oldmetrics = read(olddir / "source_selection.json"), read(olddir / "metrics.json")
            require(provenance["original_source_selection"] == oldselection, "Original source selection")
            parent_manifest = {record["path"]: record["sha256"] for record in read(olddir / "local_artifacts.json")}
            for path, expected in provenance["used_local_files_sha256"].items():
                require(expected == parent_manifest[path] == sha(ROOT / path), "Used parent artifact: " + path)
            selected_checkpoint = olddir / "source" / f"encoder_{oldselection['selected_lr']}" / "selected.pt"
            state = torch.load(selected_checkpoint, map_location="cpu", weights_only=True)["state_dict"]
            require(state_digest(state) == oldselection["selected_encoder_state_sha256"]
                    == provenance["initial_identity"]["encoder"], "Selected source tensors")
            rows, original_rows = arrays(directory / "split_rows.npz"), arrays(olddir / "split_rows.npz")
            require(rows.keys() == original_rows.keys(), "Split pool schema")
            groups = {}
            pool_labels = {}
            for pool, indices in rows.items():
                require(np.array_equal(indices, original_rows[pool]), "Original ordered split rows: " + pool)
                require(array_hash(indices) == oldsupport["seeds"][str(seed)]["pools"][pool]["raw_row_hash"], "Original split hash")
                groups[pool] = set(raw.iloc[indices].SERIALNO)
                pool_labels[pool] = labels(raw.iloc[indices])
                declared = metrics["support_by_pool"][pool]
                require(declared["raw_row_sha256"] == array_hash(indices) and declared["persons"] == len(indices)
                        and declared["households"] == len(groups[pool]), "Current pool support/identity")
                for target, y in pool_labels[pool].items():
                    kind = "attributes" if target in SCHEMA else "tasks"
                    require(support(y, SCHEMA.get(target, 2)) == declared[kind][target], "Reconstructed label masks: " + target)
            require(len(np.unique(np.concatenate(list(rows.values())))) == sum(map(len, rows.values())), "Person overlap")
            for first, names in enumerate(groups):
                for second in list(groups)[first + 1:]:
                    require(groups[names].isdisjoint(groups[second]), "Household overlap")
            release_names = [name for parent_name in cfg["parents"] for name in (parent_name, parent_name + "_leace")]
            release_names.append(cfg["reference"])
            caches = {name: arrays(directory / f"release_{name}.npz") for name in release_names}
            require(set(release_names) == set(release_freeze["release_metadata"]) == set(metrics["release_metadata"]), "Release schema")
            calibration_labels = pool_labels["representation_fit"]
            known = np.stack([calibration_labels[target] >= 0 for target in SCHEMA])
            complete = known.all(0)
            require(release_freeze["calibration_pool"] == "representation_fit"
                    and release_freeze["calibration_raw_row_sha256"] == array_hash(rows["representation_fit"]), "Map fitting row identity")
            require(release_freeze["reserved_task_labels_entered_release_fitting"] is False, "Map task-access declaration")
            map_records = {}
            for name in cfg["parents"]:
                eraser = FrozenAffineMap.load(directory / f"map_{name}.npz")
                metadata = eraser.metadata
                require(metadata == release_freeze["release_metadata"][name + "_leace"]["eraser"], "Saved map metadata")
                require(eraser.fingerprint() == release_freeze["map_hashes"][name], "Map fingerprint")
                require(metadata["numerical_policy"] == NUMERICAL_POLICY and metadata["leace_options"] == LEACE_OPTIONS, "Fixed numerical map policy")
                require(metadata["fit_split"] == "representation_fit" and metadata["class_schema"] == SCHEMA, "Map fitting schema")
                require(metadata["input_array_sha256"] == array_hash(caches[name]["representation_fit"]), "Map input hash")
                require(np.array_equal(eraser.fit_mask, complete) and np.array_equal(eraser.known_masks, known), "Saved missing masks")
                fitting = caches[name]["representation_fit"][complete].astype(np.float64)
                expected_mean, _ = centered(fitting)
                require(np.array_equal(expected_mean, eraser.mean), "Map complete-case fitting mean")
                require(metadata["fitting_float64_sha256"] == array_hash(fitting), "Actual complete-case fitting features")
                for target in SCHEMA:
                    require(metadata["label_sha256"][target] == array_hash(calibration_labels[target]), "Protected input labels")
                before_fingerprint = eraser.fingerprint()
                for pool in rows:
                    require(np.array_equal(eraser.apply(caches[name][pool]), caches[name + "_leace"][pool]), "Bitwise map replay: " + name + "/" + pool)
                require(before_fingerprint == eraser.fingerprint(), "Replay mutated map")
                z = np.concatenate([np.eye(k)[calibration_labels[target][complete]] for target, k in SCHEMA.items()], axis=1)
                _, zc = centered(z)
                covariance_summary = {}
                for field, data in (("before", fitting), ("after_float64", eraser.apply(fitting, dtype=np.float64)),
                                    ("after_float32", caches[name + "_leace"]["representation_fit"][complete])):
                    _, xc = centered(data)
                    cross = xc.T @ zc / (len(z) - 1)
                    require(np.array_equal(cross, np.asarray(metadata[field]["cross_covariance"])), "Fitting covariance replay: " + name + "/" + field)
                    valid_columns = (z.sum(0) > 0) & (z.sum(0) < len(z))
                    require(metadata[field]["valid_column_mask"] == valid_columns.tolist(), "Covariance class mask")
                    if field != "before":
                        threshold = (NUMERICAL_POLICY["covariance_atol"] + NUMERICAL_POLICY["covariance_rtol"]
                                     * metadata["before"]["cross_covariance_max_abs"])
                        raw_inequality = bool(np.max(np.abs(cross)) <= threshold)
                        require(metadata[field]["covariance_threshold"] == threshold, "Fixed covariance diagnostic threshold")
                        require(metadata[field]["empirical_covariance_within_tolerance"]
                                == (bool(valid_columns.all()) and raw_inequality), "Covariance inequality and coverage conjunction")
                        covariance_summary[field] = {"max_absolute_covariance": float(np.max(np.abs(cross))),
                            "threshold": threshold, "raw_numerical_inequality": raw_inequality,
                            "schema_complete": bool(valid_columns.all()),
                            "coverage_aware_numerical_flag": metadata[field]["empirical_covariance_within_tolerance"]}
                if metadata["exact_constant_output"]:
                    require(not np.any(eraser.matrix), "Full-collapse matrix not exact zero")
                    for pool, value in caches[name + "_leace"].items():
                        require(np.array_equal(value, np.broadcast_to(eraser.mean.astype(np.float32), value.shape)), "Full-collapse output not exact constant")
                map_records[name] = {"fingerprint": eraser.fingerprint(), "pools_replayed_bitwise": list(rows),
                                     "fit_rows": metadata["fit_rows"], "retained_input_rank": metadata["input_retained_rank"],
                                     "retained_projection_rank": metadata["projection_retained_rank"],
                                     "exact_constant_output": metadata["exact_constant_output"],
                                     "attribute_coverage": metadata["coverage"], "covariance_checks": covariance_summary}
            for name, cache in caches.items():
                require(set(cache) == set(rows), "Cached split schema")
                declared = metrics["release_metadata"][name]
                require(declared == release_freeze["release_metadata"][name], "Release metadata identity")
                require(rank(cache["representation_fit"]) == declared["centered_covariance_rank"], "Centered feature rank")
                require(cache["representation_fit"].shape[1] == declared["dimension"], "Stored dimension")
                for pool, value in cache.items():
                    require(value.dtype == np.float32, "Fixed output dtype")
                    expected = (metrics["integrity"]["evaluation_output_hashes"][name] if pool == "test"
                                else release_freeze["output_hashes"][name][pool])
                    require(array_hash(value) == expected, "Frozen cached array hash")
                    if not name.endswith("_leace"):
                        record = oldmetrics["release_metadata"][name]
                        previous = record["test_output_sha256"] if pool == "test" else record["development_hashes"][pool]
                        require(expected == previous, "Unchanged parent release: " + name + "/" + pool)
            head_report = verify_heads(directory, cfg, seed, selection, caches, pool_labels, rows)
            integrity = metrics["integrity"]
            require(selection["release_freeze_sha256"] == sha(directory / "release_freeze.json"), "Release freeze identity")
            require(selection["protocol_freeze_sha256"] == sha(out / "protocol_freeze.json"), "Protocol freeze identity")
            require(integrity["selection_sha256"] == sha(directory / "selection_before_test.json"), "Selection identity")
            require(release_freeze["created_utc"] < selection["created_utc"] < integrity["evaluation_started_utc"], "Release/selection/evaluation order")
            require(integrity["selection_created_utc"] == selection["created_utc"], "Selection timestamp")
            for field in ("source_unchanged", "parent_files_unchanged", "erasers_unchanged", "development_releases_unchanged",
                          "selection_unchanged", "evaluation_parent_outputs_match_historical"):
                require(integrity[field] is True, "Recorded integrity flag: " + field)
            manifest = read(directory / "local_artifacts.json")
            for record in manifest:
                path = recorded_path(record["path"], out)
                require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "New local artifact identity: " + record["path"])
            result["seeds"][str(seed)] = {
                "passed": True, "map_replays": map_records, "new_local_artifact_hashes_verified": len(manifest),
                "parent_artifact_hashes_verified": len(provenance["used_local_files_sha256"]),
                "split_household_isolation_and_masks_verified": True, "source_selected_state_verified": True,
                "release_and_selection_gates_verified": True, "head_checks": head_report,
                "all_parent_cache_hashes_unchanged": True,
                "release_created_utc": release_freeze["created_utc"], "selection_created_utc": selection["created_utc"],
                "evaluation_started_utc": integrity["evaluation_started_utc"]}
    result["all_seeds_passed"] = True
    result["runtime_seconds"] = time.perf_counter() - tick
    return result


def verify_heads(directory, cfg, seed, selection, caches, pool_labels, rows):
    schedules = {}
    checked = 0
    fallbacks = {}
    for key, record in selection["fitting_records"].items():
        role, release, target = key.split("/")
        order = TASKS if role == "transfer" else tuple(SCHEMA)
        j = order.index(target)
        fit_pool, val_pool = (("downstream_fit", "downstream_validation") if role == "transfer"
                              else ("attacker_fit", "attacker_validation"))
        y, yv = pool_labels[fit_pool][target], pool_labels[val_pool][target]
        subset_seed = (1230000 if role == "transfer" else 1240000) + 100 * seed + j
        budget = cfg["head_budget"] if role == "transfer" else cfg["attacker_budget"]
        idx = np.random.default_rng(subset_seed).permutation(np.flatnonzero(y >= 0))[:budget]
        vi = np.flatnonzero(yv >= 0)
        saved_idx = selection["task_fit_indices" if role == "transfer" else "audit_fit_indices"][target]
        require(saved_idx == {"n": len(idx), "pool_indices_sha256": array_hash(idx), "raw_rows_sha256": array_hash(rows[fit_pool][idx])}, "Identical eligible fitting rows")
        nclasses = SCHEMA.get(target, 2)
        if release == "prior":
            meta = record["candidates"]["prior"]
            require(meta["fit_label_hash"] == array_hash(y[idx]), "Prior fitting labels")
            require(selection["head_selections"][key] == "prior", "Prior selection")
            continue
        xf = (np.eye(nclasses)[y[idx]] if release == "exposed" else caches[release][fit_pool][idx]).astype(np.float64)
        xv = (np.eye(nclasses)[yv[vi]] if release == "exposed" else caches[release][val_pool][vi]).astype(np.float64)
        require(record["fit_hashes"] == {"x": array_hash(xf), "y": array_hash(y[idx])}, "Fitting feature/label access: " + key)
        require(record["validation_hashes"] == {"x": array_hash(xv), "y": array_hash(yv[vi])}, "Validation feature/label access: " + key)
        candidate_records = record["candidates"]
        require(selection["head_selections"][key] == min(candidate_records, key=lambda cid: (candidate_records[cid]["validation_scores"]["log_loss"], cid)), "Validation candidate selection")
        families = {m["family"] for m in candidate_records.values()}
        expected_family = {family: min((cid for cid, m in candidate_records.items() if m["family"] == family),
                                      key=lambda cid: (candidate_records[cid]["validation_scores"]["log_loss"], cid)) for family in families}
        require(selection["family_selections"][key] == expected_family, "Within-family validation selection")
        if role == "audit":
            defined = {cid: m["validation_scores"]["auroc"] for cid, m in candidate_records.items()
                       if m["validation_scores"]["auroc"] is not None}
            expected_auc = min(defined, key=lambda cid: (-defined[cid], cid)) if defined else None
            require(selection["auroc_selections"][key] == expected_auc, "Validation AUROC candidate selection")
        for cid, meta in candidate_records.items():
            saved = directory / "fitted" / key / cid
            require(read(saved / "metadata.json") == meta, "Saved candidate metadata: " + key + "/" + cid)
            checked += 1
            if "fallback_reason" in meta:
                fallbacks[key + "/" + cid] = meta["fallback_reason"]
                continue
            expected_mean, std = xf.mean(0), xf.std(0)
            expected_scale = np.where(std > HEAD_BUDGET["preprocessing"]["std_floor"], std, 1.)
            prep = arrays(saved / "preprocessing.npz")
            require(np.array_equal(prep["mean"], expected_mean) and np.array_equal(prep["scale"], expected_scale), "Fitting-only head preprocessing")
            if meta["family"] == "mlp":
                epochs = cfg["head_mlp_epochs"] if role == "transfer" else cfg["audit_mlp_epochs"]
                batch_size = HEAD_BUDGET["mlp"]["batch_size"]
                steps = epochs * math.ceil(len(idx) / batch_size)
                require(meta["optimizer_steps"] == steps and meta["training_row_exposures"] == epochs * len(idx), "MLP update/exposure budget")
                require(meta["row_exposure_min"] == meta["row_exposure_max"] == epochs, "Per-row MLP exposure")
                expected_seed = (1250000 if role == "transfer" else 1260000) + 100 * seed + j
                if cid == "mlp_1":
                    expected_seed += cfg["audit_restart_offset"]
                require(meta["initialization_seed"] == expected_seed and meta["schedule_seed"] == expected_seed + 700000, "Matched initialization and schedule seeds")
                rng, digest = np.random.default_rng(meta["schedule_seed"]), hashlib.sha256()
                for _ in range(epochs):
                    digest.update(rng.permutation(len(idx)).tobytes())
                require(digest.hexdigest() == meta["schedule_hash"], "Reconstructed MLP schedule")
                best = min(meta["validation_curve"], key=lambda value: (value["validation_log_loss"], value["epoch"]))
                require(best["epoch"] == meta["selected_epoch"] and best["optimizer_steps"] == meta["selected_optimizer_steps"], "Validation MLP checkpoint")
                checkpoint = torch.load(saved / "model.pt", map_location="cpu", weights_only=True)
                model_hash = hashlib.sha256()
                for name, value in sorted(checkpoint["state"].items()):
                    model_hash.update(name.encode())
                    model_hash.update(array_hash(value.numpy()).encode())
                require(model_hash.hexdigest() == meta["selected_state_hash"], "Saved MLP selected tensors")
                group = role + "/" + target + "/" + cid
                schedules.setdefault(group, []).append((meta["schedule_hash"], steps, epochs * len(idx)))
            elif meta["family"] == "histgb":
                require(meta["boosting_iterations"] == 150 and meta["tree_fit_row_participations"] == 150 * len(idx), "Fixed audit boosting exposure")
                require(meta["parameters"]["min_samples_leaf"] == (5 if cid == "hist_gb_5" else 20), "Fixed tree leaf variant")
                require(meta["parameters"]["l2_regularization"] == 1. and meta["parameters"]["early_stopping"] is False, "Tree numerical/selection configuration")
    for name, values in schedules.items():
        require(len(set(values)) == 1, "Matching cross-release schedule: " + name)
    return {"candidate_metadata_and_input_checks": checked, "numerical_fallbacks": fallbacks,
            "matching_mlp_schedules": {k: {"interfaces": len(v), "schedule_hash": v[0][0],
                                          "optimizer_steps": v[0][1], "row_exposures": v[0][2]} for k, v in schedules.items()},
            "fit_preprocessing_and_selection_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_RELATIVE)
    parser.add_argument("--report", type=Path, help="Fresh JSON report file; default prints JSON to stdout.")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("Refusing to overwrite an existing report; choose a fresh path")
    report = verify(args.out.expanduser().resolve())
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report is None:
        print(payload, end="")
    else:
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent teacher, affine-geometry, model and score replay; no fitting.

Raw-label/scalar/sklearn metrics and literal saved-network inference reuse the
reviewed replay mechanism. Teacher algebra and decoder SVD calculations inspect
the already frozen artifacts without changing any scientific model or choice.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_acs_bottleneck_scores as check
from scripts.verify_acs_preservation_extended import checkpoint_evidence


def unit_path(out, unit):
    name = "static" if unit[0] == "static" else unit[0]+"_rho"+("0p1" if unit[1] == .1 else "0")
    return out/name/f"seed_{unit[-1]}"


def map_probability_free(record, raw):
    return ((np.asarray(raw, np.float64)-record["mean"]) @ record["matrix"].T+record["mean"]).astype(np.float32)


def load_map(path):
    with np.load(path, allow_pickle=False) as arrays:
        result = {key: arrays[key].copy() for key in ("matrix", "mean", "fit_mask", "known_masks")}
        metadata_json = str(arrays["metadata_json"])
        digest = hashlib.sha256(metadata_json.encode())
        for key in ("matrix", "mean", "fit_mask", "known_masks"):
            digest.update(key.encode())
            digest.update(check.array_hash(result[key]).encode())
        assert digest.hexdigest() == str(arrays["map_sha256"])
    result["metadata"] = {**json.loads(metadata_json), "map_sha256": digest.hexdigest()}
    return result


def teacher_components(raw, maps, mean, scale):
    raw = np.asarray(raw, np.float32)
    r = raw.astype(np.float64)
    e, s = (map_probability_free(maps[name], raw).astype(np.float64) for name in ("E", "S"))
    return {"raw": (r-mean)/scale, "E": (e-mean)/scale, "qE": (r-e)/scale,
            "S": (s-mean)/scale, "qS": (r-s)/scale}


def verify_teachers(directory, pca, labels, raw_rows, seed):
    metadata = check.read(directory/"teachers.json")
    prefit = check.read(directory/"teacher_prefit_freeze.json")
    assert metadata["prefit_freeze_sha256"] == check.sha(directory/"teacher_prefit_freeze.json")
    assert prefit["created_utc"] <= metadata["created_utc"]
    assert prefit["any_teacher_fit_started"] is False
    assert prefit["permutation_file_sha256"] == check.sha(directory/"paired_permutation.npz")
    assert prefit["fit_raw_rows_sha256"] == check.array_hash(raw_rows)
    assert prefit["source_sha256"]["teacher_module"] == check.sha(ROOT/"experiments/acs_selective_teachers.py")
    assert prefit["source_sha256"]["existing_leace_module"] == check.sha(ROOT/"experiments/acs_protection_maps.py")
    for name, digest in metadata["files_sha256"].items():
        assert check.sha(directory/name) == digest
    raw = np.asarray(pca["representation_fit"])[:, :16]
    mean, std = raw.astype(np.float64).mean(0), raw.astype(np.float64).std(0)
    scale = np.where(std > 1e-12, std, 1.)
    np.testing.assert_array_equal(prefit["original_mean"], mean)
    np.testing.assert_array_equal(prefit["original_scale"], scale)
    assert prefit["raw_fit_sha256"] == check.array_hash(raw)
    known = np.stack([labels[name] >= 0 for name in ("SEX", "RAC1P")])
    complete = known.all(0)
    ix = np.flatnonzero(complete)
    order = np.random.default_rng(20260908+seed).permutation(len(ix))
    full = np.arange(len(raw), dtype=np.int64)
    full[ix] = ix[order]
    with np.load(directory/"paired_permutation.npz") as arrays:
        for key, expected in (("complete_rows", ix), ("permutation", order), ("full_row_permutation", full),
                              ("complete_mask", complete), ("known_masks", known)):
            np.testing.assert_array_equal(arrays[key], expected)
    sham = {name: values[full] for name, values in labels.items()}
    joint = np.bincount(labels["SEX"][ix]*9+labels["RAC1P"][ix], minlength=18).reshape(2, 9)
    np.testing.assert_array_equal(joint, metadata["permutation"]["complete_case_joint_counts_SEX_by_RAC1P"])
    np.testing.assert_array_equal(joint, np.bincount(sham["SEX"][ix]*9+sham["RAC1P"][ix], minlength=18).reshape(2, 9))
    for name in labels:
        assert metadata["permutation"]["original_label_sha256"][name] == check.array_hash(labels[name])
        assert metadata["permutation"]["permuted_label_sha256"][name] == check.array_hash(sham[name])
    maps = {name: load_map(directory/f"map_{name}.npz") for name in ("E", "S")}
    max_covariance_error = 0.
    with np.load(directory/"fit_targets.npz") as targets:
        np.testing.assert_array_equal(targets["R"], raw)
        for name, values in (("E", labels), ("S", sham)):
            fitted = maps[name]
            assert fitted["metadata"] == metadata["maps"][name]
            np.testing.assert_array_equal(fitted["fit_mask"], complete)
            np.testing.assert_array_equal(fitted["known_masks"], known)
            expected_mean = raw[ix][0].astype(np.float64)+(raw[ix].astype(np.float64)-raw[ix][0]).mean(0)
            np.testing.assert_array_equal(fitted["mean"], expected_mean)
            centered = raw[ix].astype(np.float64)-expected_mean
            covariance = centered.T @ centered/(len(ix)-1)
            eigenvalues = np.linalg.eigvalsh((covariance+covariance.T)/2)
            np.testing.assert_allclose(eigenvalues, fitted["metadata"]["input_covariance_eigenvalues"], atol=1e-10, rtol=1e-12)
            assert fitted["metadata"]["input_retained_rank"] == int((eigenvalues > max(0., eigenvalues[-1])*1e-10).sum())
            projection_singular = np.linalg.svd(fitted["matrix"], compute_uv=False)
            assert fitted["metadata"]["projection_retained_rank"] == int((projection_singular > max(1., projection_singular[0])*1e-10).sum())
            assert fitted["metadata"]["label_sha256"] == {key: check.array_hash(value) for key, value in values.items()}
            assert fitted["metadata"]["input_array_sha256"] == check.array_hash(raw)
            np.testing.assert_array_equal(targets[name], map_probability_free(fitted, raw))
            assert metadata["fit_target_sha256"][name] == check.array_hash(targets[name])
            concept = np.column_stack([np.eye(k)[values[key][ix]] for key, k in (("SEX", 2), ("RAC1P", 9))])
            output64 = (raw[ix].astype(np.float64)-fitted["mean"]) @ fitted["matrix"].T+fitted["mean"]
            for field, output in (("before", raw[ix].astype(np.float64)), ("after_float64", output64),
                                  ("after_float32", output64.astype(np.float32).astype(np.float64))):
                centered_x = (output-output[0])-(output-output[0]).mean(0)
                centered_z = (concept-concept[0])-(concept-concept[0]).mean(0)
                covariance = centered_x.T @ centered_z/(len(ix)-1)
                stored = np.asarray(fitted["metadata"][field]["cross_covariance"])
                error = float(np.max(np.abs(covariance-stored)))
                assert error < 1e-12
                max_covariance_error = max(max_covariance_error, error)
            valid_columns = (concept.sum(0) > 0) & (concept.sum(0) < len(ix))
            np.testing.assert_array_equal(fitted["metadata"]["after_float64"]["valid_column_mask"], valid_columns)
            assert fitted["metadata"]["after_float64"]["schema_complete"] == bool(valid_columns.all())
        for name in ("R", "E", "S"):
            difference = (targets[name].astype(np.float64)-raw.astype(np.float64))/scale
            expected = np.mean(difference**2, axis=0)
            check.compare(expected.tolist(), metadata["diagnostics"][name]["movement_from_raw_per_coordinate_mse"], f"teacher{seed}/{name}/distortion")
            check.compare(float(expected.mean()), metadata["diagnostics"][name]["movement_from_raw_mean_mse"], f"teacher{seed}/{name}/mean_distortion")
    return maps, {"teacher_manifest_sha256": check.sha(directory/"teachers.json"), "permutation_sha256": check.array_hash(order),
                  "maps": {name: fitted["metadata"]["map_sha256"] for name, fitted in maps.items()},
                  "complete_case_rows": len(ix), "joint_counts_and_missing_masks_preserved": True,
                  "frozen_before_fitting": True, "literal_raw_coordinate_teacher_replay": True,
                  "max_covariance_replay_error": max_covariance_error}


def geometry_score(coefficient, prior, fit_variance, x, y):
    x = np.asarray(x, np.float64)
    predicted = np.column_stack((x, np.ones(len(x)))) @ coefficient
    mse, prior_mse = np.mean((predicted-y)**2, axis=0), np.mean((y-prior)**2, axis=0)
    variance = np.var(y, axis=0)
    defined = (fit_variance > 1e-12) & (variance > 1e-12) & (prior_mse > 1e-12)
    mean_defined = fit_variance.mean() > 1e-12 and variance.mean() > 1e-12 and prior_mse.mean() > 1e-12
    return {"rows": len(x), "mean_mse": float(mse.mean()), "per_coordinate_mse": mse.tolist(),
            "prior_mean_mse": float(prior_mse.mean()), "prior_per_coordinate_mse": prior_mse.tolist(),
            "target_variance_mean": float(variance.mean()), "target_variance_per_coordinate": variance.tolist(),
            "mse_over_prior": float(mse.mean()/prior_mse.mean()) if mean_defined else None,
            "per_coordinate_mse_over_prior": [float(a/b) if keep else None for a, b, keep in zip(mse, prior_mse, defined)],
            "ratio_defined": bool(mean_defined), "per_coordinate_ratio_defined": defined.tolist()}


def verify_geometry(directory, releases, pca, preprocessing, maps):
    report = check.read(directory/"geometry.json")
    frozen = check.read(directory/"affine_freeze.json")
    mean, scale = np.asarray(preprocessing["mean"])[:16], np.asarray(preprocessing["scale"])[:16]
    targets = {pool: teacher_components(raw[:, :16], maps, mean, scale) for pool, raw in pca.items()}
    errors, prediction_errors, count = [], [], 0
    for name, snapshot in report["snapshots"].items():
        x = releases[name]["representation_fit"].astype(np.float64)
        design = np.column_stack((x, np.ones(len(x))))
        left, singular, right = np.linalg.svd(design, full_matrices=False)
        kept = singular > singular[0]*1e-12
        for target, record in snapshot["targets"].items():
            path = directory/record["artifact"]
            assert check.sha(path) == record["artifact_sha256"] == frozen["files_sha256"][path.name]
            with np.load(path) as arrays:
                coefficient, prior, variance = (arrays[k].copy() for k in ("coefficient", "prior", "fit_variance"))
            y = targets["representation_fit"][target]
            assert record["fit_release_sha256"] == check.array_hash(x)
            assert record["fit_target_sha256"] == check.array_hash(y)
            assert record["coefficient_sha256"] == check.array_hash(coefficient)
            np.testing.assert_array_equal(prior, y.mean(0))
            np.testing.assert_array_equal(variance, np.mean((y-prior)**2, axis=0))
            alternative = (right[kept].T/singular[kept]) @ (left[:, kept].T @ y)
            np.testing.assert_allclose(coefficient, alternative, atol=1e-5, rtol=1e-8)
            error = float(np.max(np.abs(coefficient-alternative)))
            prediction_error = float(np.max(np.abs(design @ coefficient-design @ alternative)))
            assert prediction_error < 1e-5
            errors.append(error)
            prediction_errors.append(prediction_error)
            assert record["rank"] == int(kept.sum())
            np.testing.assert_allclose(record["singular_values"], singular, atol=1e-10, rtol=1e-12)
            for field, values in (("target_centered_rank", y-prior), ("target_uncentered_rank", y),
                                   ("release_centered_rank", x-x.mean(0))):
                spectrum = np.linalg.svd(values, compute_uv=False)
                rank = int((spectrum > spectrum[0]*1e-12).sum())
                assert record[field]["rank"] == rank
                np.testing.assert_allclose(record[field]["singular_values"], spectrum, atol=1e-10, rtol=1e-12)
            for pool, key in (("representation_fit", "fit"), ("source_validation", "source_validation"), ("test", "development_evaluation")):
                expected = geometry_score(coefficient, prior, variance, releases[name][pool], targets[pool][target])
                for field, value in expected.items():
                    check.compare(value, record[key][field], f"{directory.name}/{name}/{target}/{key}/{field}")
            count += 1
        for pool, key in (("representation_fit", "fit"), ("source_validation", "source_validation"), ("test", "development_evaluation")):
            raw = pca[pool][:, :16]
            direct_targets = {"R": raw, **{key: map_probability_free(maps[key], raw) for key in ("E", "S")}}
            for target, teacher in direct_targets.items():
                error = np.mean(((releases[name][pool].astype(np.float64)-teacher.astype(np.float64))/scale)**2, axis=0)
                stored = snapshot["direct_teacher_error"][key][target]
                check.compare(float(error.mean()), stored["mean_mse"], f"{name}/{key}/{target}/direct_mean")
                check.compare(error.tolist(), stored["per_coordinate_mse"], f"{name}/{key}/{target}/direct_coordinates")
    return {"affine_decoders": count, "independent_SVD_coefficient_max_error": max(errors),
            "independent_SVD_fitting_prediction_max_error": max(prediction_errors),
            "all_original_scale_absolute_prior_variance_and_ratio_scores_replayed": True}


def verify(out):
    started = time.perf_counter()
    check.errors.clear()
    check.max_error, check.max_error_path, check.numeric_comparisons = 0., None, 0
    cfg = check.read(out/"config.json")
    freeze = check.read(out/"protocol_freeze.json")
    for field in ("sha256", "reference_record_hashes"):
        for name, expected in freeze[field].items():
            assert check.sha(ROOT/name) == expected, (field, name)
    parent = ROOT/cfg["parent_results"]
    pcfg = check.read(parent/"config.json")
    raw_path = ROOT/pcfg["raw_path"]
    assert check.sha(raw_path) == check.read(parent/"schema_support.json")["raw_sha256"]
    raw = pd.read_csv(raw_path, usecols=["MIG", "JWMNP", "PINCP", "ESR", "PUBCOV", "SEX", "RAC1P", "PWGTP"], low_memory=False)
    units = [unit for unit in cfg["execution_order"] if (unit_path(out, unit)/"metrics.json").exists()]
    assert units
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Independent raw-label metrics, saved inference, teacher algebra and affine SVD replay; no scientific fitting or new data",
        "evaluation_status": cfg["evaluation_status"], "script_sha256": check.sha(__file__),
        "independent_metric_helper_sha256": check.sha(check.__file__), "absolute_metric_tolerance": check.TOL,
        "complete_matrix": len(units) == len(cfg["execution_order"]), "completed_units": units,
        "candidate_records": 0, "score_dictionaries": 0, "prediction_sets": 0, "selection_records": 0,
        "audit_terminal_checkpoints": 0, "nested_prefix_checks": 0, "mlp_schedule_checks": 0,
        "saved_direct_coordinate_fidelity": 0, "teacher_seeds": {}, "units": {},
        "limitations": ["Recorded optimizer actions are inspected through state/counters/schedules, not rerun.",
                       "Residual components are not pure sensitive information; high affine error does not preclude nonlinear recovery."]}
    maps_by_seed, pca_by_seed, frames_by_seed, pre_by_seed, rows_by_seed = {}, {}, {}, {}, {}
    for seed in cfg["seeds"]:
        if not (out/"static"/f"seed_{seed}"/"metrics.json").exists():
            continue
        directory = out/"static"/f"seed_{seed}"
        with np.load(directory/"split_rows.npz") as arrays:
            rows = {key: arrays[key].copy() for key in arrays.files}
        with np.load(parent/f"seed_{seed}"/"split_rows.npz") as original:
            assert set(rows) == set(original.files)
            assert all(np.array_equal(value, original[key]) for key, value in rows.items())
        frames = {pool: raw.iloc[ix] for pool, ix in rows.items()}
        with np.load(ROOT/cfg["reference_results"]/f"seed_{seed}"/"release_E_pca.npz") as arrays:
            pca = {key: arrays[key].copy() for key in arrays.files}
        labels = {}
        for target in ("SEX", "RAC1P"):
            y, valid = check.labels(frames["representation_fit"], target)
            labels[target] = np.where(valid, y, -1).astype(np.int64)
        maps_by_seed[seed], report["teacher_seeds"][str(seed)] = verify_teachers(directory/"teachers", pca, labels, rows["representation_fit"], seed)
        pre_by_seed[seed] = check.read(ROOT/cfg["init_reference_results"]/f"seed_{seed}"/"training/training.json")["preprocessing"]
        pca_by_seed[seed], frames_by_seed[seed], rows_by_seed[seed] = pca, frames, rows
    for unit in units:
        directory, seed = unit_path(out, unit), unit[-1]
        measured, selection = check.read(directory/"metrics.json"), check.read(directory/"selection_before_test.json")
        frozen = check.read(directory/"release_freeze.json")
        assert selection["release_freeze_sha256"] == check.sha(directory/"release_freeze.json")
        assert selection["affine_freeze_sha256"] == check.sha(directory/"affine_freeze.json")
        assert selection["protocol_freeze_sha256"] == check.sha(out/"protocol_freeze.json")
        assert selection["created_utc"] <= measured["integrity"]["evaluation_started_utc"]
        assert all(value for key, value in measured["integrity"].items() if key.endswith("_unchanged"))
        for artifact in check.read(directory/"local_artifacts.json"):
            assert check.sha(ROOT/artifact["path"]) == artifact["sha256"]
        completion = check.read(directory/"completion.json")
        assert all(check.sha(directory/name) == expected for name, expected in completion["sha256"].items())
        with np.load(directory/"split_rows.npz") as rows:
            assert all(np.array_equal(rows[pool], ix) for pool, ix in rows_by_seed[seed].items())
        releases = {}
        for name in frozen["output_hashes"]:
            with np.load(directory/f"release_{name}.npz") as arrays:
                releases[name] = {pool: arrays[pool].copy() for pool in arrays.files}
            for pool, values in releases[name].items():
                expected = measured["integrity"]["evaluation_output_hashes"][name] if pool == "test" else frozen["output_hashes"][name][pool]
                assert check.array_hash(values) == expected
                if unit[0] == "static":
                    assert np.array_equal(values, map_probability_free(maps_by_seed[seed][name[0]], pca_by_seed[seed][pool][:, :16]))
        geometry = verify_geometry(directory, releases, pca_by_seed[seed], pre_by_seed[seed], maps_by_seed[seed])
        frames, checkpoints = frames_by_seed[seed], {}
        training_adversaries = {} if unit[0] == "static" else {
            name: torch.load(directory/"training"/name/"final.pt", map_location="cpu", weights_only=True)["adversary_state"]
            for name in ("C", "D")}
        expected_rows = 60 if unit[0] == "static" else 86
        assert len(measured["raw_metrics"]) == expected_rows
        with np.load(directory/"predictions.npz") as probabilities:
            assert len(probabilities.files) == expected_rows*2
            for row in measured["raw_metrics"]:
                role, release, target, cid = (row[field] for field in ("role", "release", "target", "candidate_id"))
                budget = row.get("audit_budget")
                record = selection if budget is None else selection["budgets"][str(budget)]
                prefix, key = ("" if budget is None else f"budget{budget}/"), f"{role}/{release}/{target}"
                metadata = record["fitting_records"][key]["candidates"][cid]
                classes = 9 if target == "RAC1P" else 2
                for split, pool in (("validation", "downstream_validation" if role == "transfer" else "attacker_validation"), ("test", "test")):
                    y, mask = check.labels(frames[pool], target)
                    probability = probabilities[f"{prefix}{split}/{key}/{cid}"]
                    for suffix, weights in (("", None), ("_person_weighted", frames[pool].PWGTP.to_numpy(float)[mask])):
                        check.compare(check.independent_scores(y[mask], probability, classes, weights), row[split+suffix],
                                      f"{unit}/{prefix}{key}/{cid}/{split}{suffix}")
                        report["score_dictionaries"] += 1
                check.compare(metadata["validation_scores"], row["validation"], f"{unit}/{key}/{cid}/fit_validation")
                assert row["selected"] == (record["head_selections"][key] == cid)
                assert row["independent_selected"] == (record["independent_selections"][key] == cid)
                assert row["selected_within_family"] == (record["family_selections"][key][row["family"]] == cid)
                assert row["auc_selected"] == (record["auroc_selections"][key] == cid)
                report["candidate_records"] += 1
            for budget, record in ((None, selection), (120, selection["budgets"]["120"]), (360, selection["budgets"]["360"])):
                for key, fitted in record["fitting_records"].items():
                    role, name, target = key.split("/")
                    candidates = fitted["candidates"]
                    eligible = [cid for cid in candidates if cid != "saved_adversary"]
                    independent = [cid for cid in eligible if cid != "catchup"]
                    assert record["head_selections"][key] == check.primary(candidates, eligible)
                    assert record["independent_selections"][key] == check.primary(candidates, independent)
                    assert record["auroc_selections"][key] == check.primary(candidates, eligible, auc=True)
                    family = lambda cid: cid if cid in ("catchup", "saved_adversary") else candidates[cid]["family"]
                    for fam, cid in record["family_selections"][key].items():
                        assert cid == check.primary(candidates, [c for c in candidates if family(c) == fam])
                    report["selection_records"] += 1
                    fitpool, valpool = ("downstream_fit", "downstream_validation") if role == "transfer" else ("attacker_fit", "attacker_validation")
                    yf, mf = check.labels(frames[fitpool], target)
                    yv, mv = check.labels(frames[valpool], target)
                    j = (cfg["utility_tasks"] if role == "transfer" else ["SEX", "RAC1P"]).index(target)
                    fitseed, limit = ((1230000, 2048) if role == "transfer" else (1240000, 4096))
                    ix = np.random.default_rng(fitseed+100*seed+j).permutation(np.flatnonzero(mf))[:limit]
                    indexrecord = selection["task_fit_indices" if role == "transfer" else "audit_fit_indices"][target]
                    assert indexrecord["pool_indices_sha256"] == check.array_hash(ix)
                    assert indexrecord["raw_rows_sha256"] == check.array_hash(rows_by_seed[seed][fitpool][ix])
                    xf, xv = releases[name][fitpool][ix].astype(np.float64), releases[name][valpool][mv].astype(np.float64)
                    for cid, metadata in candidates.items():
                        cpath = directory/"fitted"/key
                        if role == "transfer":
                            cpath /= cid
                        elif cid == "saved_adversary":
                            cpath = cpath/"saved_start"/"saved"
                        else:
                            cpath = cpath/("saved_start" if cid == "catchup" else "fresh")/f"nested{budget}"/cid
                        assert check.read(cpath/"metadata.json") == metadata
                        predict, mean, scale, state = check.load_inference(cpath, metadata)
                        assert metadata["validation_hashes"] == {"x": check.array_hash(xv), "y": check.array_hash(yv[mv].astype(np.int64))}
                        if cid != "saved_adversary":
                            assert metadata["fit_hashes"] == {"x": check.array_hash(xf), "y": check.array_hash(yf[ix].astype(np.int64))}
                        if cid in ("catchup", "saved_adversary"):
                            np.testing.assert_array_equal(mean, np.zeros(16))
                            np.testing.assert_array_equal(scale, np.ones(16))
                            saved = candidates["saved_adversary"]
                            final_training_state = {k.removeprefix(target+"."): value for k, value in training_adversaries[name].items() if k.startswith(target+".")}
                            assert saved["selected_state_hash"] == check.state_hash(final_training_state)
                            inherited = metadata["inherited_exposure"]
                            ty, tm = check.labels(frames["representation_fit"], target)
                            ty = np.where(tm, ty, -1).astype(np.int64)
                            assert inherited["training_label_sha256"] == check.array_hash(ty)
                            assert inherited["fit_raw_row_sha256"] == check.array_hash(rows_by_seed[seed]["representation_fit"])
                            assert inherited["total_row_exposures"] == len(ty)*260
                            assert inherited["total_optimizer_steps"] == int(np.ceil(len(ty)/256))*260
                            assert metadata["source_state_hash"] == saved["selected_state_hash"]
                            if cid == "saved_adversary":
                                assert np.array_equal(predict(xv), check.network_probability(state, xv))
                                report["saved_direct_coordinate_fidelity"] += 1
                            else:
                                assert metadata["initial_state_hash"] == saved["selected_state_hash"]
                                assert metadata["optimizer"]["initial_state_entries"] == 0
                                assert metadata["optimizer"]["restored_state"] is False
                        else:
                            np.testing.assert_array_equal(mean, xf.mean(0))
                            std = xf.std(0)
                            np.testing.assert_array_equal(scale, np.where(std > 1e-12, std, 1.))
                        prefix = "" if budget is None else f"budget{budget}/"
                        for split, pool in (("validation", valpool), ("test", "test")):
                            _, valid = check.labels(frames[pool], target)
                            assert np.array_equal(predict(releases[name][pool][valid]), probabilities[f"{prefix}{split}/{key}/{cid}"])
                            report["prediction_sets"] += 1
                        if "validation_curve" not in metadata:
                            continue
                        epochs = 40 if budget is None else budget
                        curve = metadata["validation_curve"]
                        assert metadata["parameters"]["epochs"] == epochs
                        assert [v["epoch"] for v in curve] == list(range(0, epochs+1, 5))
                        steps = int(np.ceil(len(ix)/256))
                        assert all(v["optimizer_steps"] == steps*v["epoch"] for v in curve)
                        best = min(curve, key=lambda v: (v["validation_log_loss"], v["epoch"]))
                        assert metadata["selected_epoch"] == best["epoch"]
                        assert metadata["validation_scores"]["log_loss"] == best["validation_log_loss"]
                        assert metadata["optimizer_steps"] == steps*epochs
                        assert metadata["training_row_exposures"] == len(ix)*epochs
                        check.replay_schedule(metadata, len(ix))
                        report["mlp_schedule_checks"] += 1
                        if budget is None:
                            continue
                        terminal = directory/"fitted"/key/("saved_start" if cid == "catchup" else "fresh")/f"last_training_{cid}_epoch{budget}.pt"
                        checkpoints[f"{budget}/{key}/{cid}"] = checkpoint_evidence(terminal, metadata, len(ix))
                        report["audit_terminal_checkpoints"] += 1
                        if budget == 360:
                            lower = selection["budgets"]["120"]["fitting_records"][key]["candidates"][cid]
                            assert curve[:len(lower["validation_curve"])] == lower["validation_curve"]
                            assert metadata["validation_scores"]["log_loss"] <= lower["validation_scores"]["log_loss"]
                            report["nested_prefix_checks"] += 1
        report["units"][str(unit)] = {"passed": True, "metrics_sha256": check.sha(directory/"metrics.json"),
            "selection_sha256": check.sha(directory/"selection_before_test.json"), "predictions_sha256": check.sha(directory/"predictions.npz"),
            "geometry": geometry, "actual_training_checkpoints": checkpoints}
    report.update(passed=not check.errors, errors=check.errors, numeric_values_compared=check.numeric_comparisons,
                  max_absolute_metric_error=check.max_error, max_error_path=check.max_error_path,
                  runtime_seconds=time.perf_counter()-started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/redesign_20260908_acs_selective_preservation_v1")
    parser.add_argument("--report", type=Path, help="Fresh report path; no overwrites; default stdout")
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError("Preserve existing verification evidence; use a fresh destination")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = verify(args.out.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.report:
        with args.report.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

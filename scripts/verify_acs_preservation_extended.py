"""Independent inference/score replay of nested preservation audit budgets.

Reuse the established raw-label, scalar/sklearn metric and literal-network
inference checks. No scientific fitters/scorers are imported. Last-training
checkpoint, Adam and RNG checks inspect saved evidence without optimizer steps.
"""
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


def sources(out, cfg, seed):
    result = {}
    for beta in ("beta_0p1", "beta_1"):
        for arm in cfg["continuation_arms"]:
            result[beta+"_"+arm] = (out/beta/f"seed_{seed}", arm)
    for cfgkey, names in (("init_reference_results", ("C_init", "D_init")),
                          ("pca16_reference_results", ("PCA16",)),
                          ("reference_results", ("B_rich_bank", "C_tree_bank", "prior", "exposed"))):
        for name in names:
            result[name] = (ROOT/cfg[cfgkey]/f"seed_{seed}", name)
    assert len(result) == 15
    return result


def candidate_path(directory, source, key, cid, budget):
    if "/prior/" in key:
        return source/"fitted"/key/cid
    path = directory/"fitted"/key
    if cid == "saved_adversary":
        return path/"saved_start"/"saved"
    if cid == "catchup":
        return path/"saved_start"/f"nested{budget}"/cid
    return path/"fresh"/f"nested{budget}"/cid


def checkpoint_evidence(path, metadata, n):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    epochs = metadata["parameters"]["epochs"]
    steps = int(np.ceil(n/256))*epochs
    assert checkpoint["epoch"] == epochs
    assert checkpoint["optimizer_steps"] == steps == metadata["optimizer_steps"]
    assert check.state_hash(checkpoint["model_state"]) == checkpoint["state_hash"] == metadata["final_state_hash"]
    assert checkpoint["selected_state_hash"] == metadata["selected_state_hash"]
    state = checkpoint["optimizer_state"]
    assert len(state["state"]) == 6
    assert all(int(v["step"]) == steps for v in state["state"].values())
    assert len(state["param_groups"]) == 1
    group = state["param_groups"][0]
    assert group["lr"] == .001 and group["betas"] == (.9, .999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0.
    rng, digest = np.random.default_rng(metadata["schedule_seed"]), hashlib.sha256()
    for _ in range(epochs):
        digest.update(rng.permutation(n).tobytes())
    assert rng.bit_generator.state == checkpoint["schedule_rng_state"]
    assert digest.hexdigest() == checkpoint["schedule_hash"] == metadata["schedule_hash"]
    return {"sha256": check.sha(path), "actual_last_state_hash": checkpoint["state_hash"],
            "selected_state_hash": metadata["selected_state_hash"], "epochs": epochs,
            "optimizer_steps": steps, "actual_state_differs_from_selected": checkpoint["state_hash"] != metadata["selected_state_hash"]}


def verify(out, seeds=None):
    started = time.perf_counter()
    cfg = check.read(out/"config.json")
    check.errors.clear()
    check.max_error, check.max_error_path, check.numeric_comparisons = 0., None, 0
    freeze = check.read(out/"protocol_freeze.json")
    expected_source = freeze["sha256"].copy()
    amendments = {}
    for path in sorted(out.glob("EXECUTION_AMENDMENT_*.json")):
        amendment = check.read(path)
        assert amendment["original_protocol_freeze_sha256"] == check.sha(out/"protocol_freeze.json")
        for source, change in amendment["source_changes"].items():
            assert source.startswith("experiments/")
            assert expected_source[source] == change["before_sha256"]
            expected_source[source] = change["after_sha256"]
        amendments[path.name] = check.sha(path)
    for path, expected in expected_source.items():
        assert check.sha(ROOT/path) == expected, ("amended execution source", path)
    for path, expected in freeze["reference_record_hashes"].items():
        assert check.sha(ROOT/path) == expected, ("original reference record", path)
    parent = ROOT/cfg["parent_results"]
    pcfg = check.read(parent/"config.json")
    raw_path = ROOT/pcfg["raw_path"]
    assert check.sha(raw_path) == check.read(parent/"schema_support.json")["raw_sha256"]
    raw = pd.read_csv(raw_path, usecols=["SEX", "RAC1P", "PWGTP"], low_memory=False)
    available = [seed for seed in cfg["seeds"] if (out/"extended"/f"seed_{seed}"/"metrics.json").exists()]
    seeds = available if seeds is None else seeds
    assert seeds and len(set(seeds)) == len(seeds) and not set(seeds)-set(available)
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Independent saved-model, probability, raw-label metric, nested-prefix and optimizer-evidence replay; no fitting or new data",
        "evaluation_status": cfg["evaluation_status"], "script_sha256": check.sha(__file__),
        "reused_score_helper_sha256": check.sha(check.__file__), "absolute_tolerance": check.TOL,
        "raw_sha256": check.sha(raw_path), "complete_three_seed_extension": len(available) == 3,
        "original_protocol_freeze_sha256": check.sha(out/"protocol_freeze.json"),
        "execution_amendments_sha256": amendments,
        "candidate_records": 0, "independent_score_dictionaries": 0,
        "literal_saved_inference_prediction_sets": 0, "validation_selections": 0,
        "nested_prefixes_equal_original": 0, "actual_training_checkpoints": 0,
        "saved_coordinate_fidelity": 0, "inherited_exposure_checks": 0,
        "fitting_standardizers": 0, "mlp_curves_schedules": 0, "seeds": {},
        "limitations": ["Saved state/counters/RNG are inspected without rerunning optimizer steps.",
                       "Finite predictive audit; absent race fitting categories remain unassessable."]}
    for seed in seeds:
        directory = out/"extended"/f"seed_{seed}"
        measured = check.read(directory/"metrics.json")
        selected = check.read(directory/"selection_before_test.json")
        frozen = check.read(directory/"release_freeze.json")
        assert measured["all_13_releases_and_exposed_controls_complete"]
        assert measured["completed_budgets"] == [120, 360]
        assert check.sha(directory/"selection_before_test.json") == measured["integrity"]["selection_sha256"]
        assert check.sha(directory/"release_freeze.json") == selected["release_freeze_sha256"]
        assert check.sha(out/"protocol_freeze.json") == selected["protocol_freeze_sha256"]
        assert selected["execution_amendments_sha256"] == frozen["execution_amendments_sha256"] == amendments
        assert selected["created_utc"] <= measured["integrity"]["evaluation_started_utc"]
        assert all(v for key, v in measured["integrity"].items() if key.endswith("_unchanged"))
        for path, expected in selected["source_files_sha256"].items():
            assert check.sha(ROOT/path) == expected
        local = check.read(directory/"local_artifacts.json")
        for row in local:
            assert check.sha(ROOT/row["path"]) == row["sha256"]
        with np.load(directory/"split_rows.npz") as rows_npz:
            rows = {key: rows_npz[key].copy() for key in rows_npz.files}
        with np.load(parent/f"seed_{seed}"/"split_rows.npz") as original:
            assert set(rows) == set(original.files)
            assert all(np.array_equal(value, original[key]) for key, value in rows.items())
        frames = {pool: raw.iloc[ix] for pool, ix in rows.items()}
        source = sources(out, cfg, seed)
        source_selections = {name: check.read(path/"selection_before_test.json") for name, (path, _) in source.items()}
        releases = {}
        for name, (path, oldname) in source.items():
            if name in ("exposed", "prior"):
                continue
            with np.load(path/f"release_{oldname}.npz") as arrays:
                releases[name] = {pool: arrays[pool].copy() for pool in arrays.files}
            for pool, values in releases[name].items():
                expected = measured["integrity"]["evaluation_output_hashes"][name] if pool == "test" else frozen["output_hashes"][name][pool]
                assert check.array_hash(values) == expected
        checkpoint_records, controls, race_flags = {}, {}, {}
        with np.load(directory/"predictions.npz") as probabilities:
            assert len(measured["raw_metrics"]) == 364
            assert len(probabilities.files) == 728
            for row in measured["raw_metrics"]:
                budget, name, target, cid = (row[k] for k in ("audit_budget", "release", "target", "candidate_id"))
                key = f"audit/{name}/{target}"
                selection = selected["budgets"][str(budget)]
                metadata = selection["fitting_records"][key]["candidates"][cid]
                classes = 9 if target == "RAC1P" else 2
                for split, pool in (("validation", "attacker_validation"), ("test", "test")):
                    y, valid = check.labels(frames[pool], target)
                    probability = probabilities[f"budget{budget}/{split}/{key}/{cid}"]
                    weights = frames[pool].PWGTP.to_numpy(float)[valid]
                    for suffix, weight in (("", None), ("_person_weighted", weights)):
                        check.compare(check.independent_scores(y[valid], probability, classes, weight), row[split+suffix],
                            f"seed{seed}/budget{budget}/{key}/{cid}/{split}{suffix}")
                        report["independent_score_dictionaries"] += 1
                check.compare(metadata["validation_scores"], row["validation"], f"{key}/{cid}/fitting_validation")
                assert row["selected"] == (selection["head_selections"][key] == cid)
                assert row["independent_selected"] == (selection["independent_selections"][key] == cid)
                assert row["selected_within_family"] == (selection["family_selections"][key][row["family"]] == cid)
                assert row["auc_selected"] == (selection["auroc_selections"][key] == cid)
                if name in ("exposed", "prior"):
                    controls[f"{budget}/{target}/{name}/{cid}"] = {"selected": row["selected"], "log_loss": row["test"]["log_loss"],
                        "accuracy": row["test"]["accuracy"], "recalls": [v["recall"] for v in row["test"]["per_class"]]}
                if target == "RAC1P":
                    race_flags = {split: {k: row[split][k] for k in ("support", "coverage_complete")} for split in ("validation", "test")}
                report["candidate_records"] += 1
            for budget in (120, 360):
                selection = selected["budgets"][str(budget)]
                assert len(selection["fitting_records"]) == 30
                for key, record in selection["fitting_records"].items():
                    _, name, target = key.split("/")
                    classes, j = (9, 1) if target == "RAC1P" else (2, 0)
                    candidates = record["candidates"]
                    eligible = [cid for cid in candidates if cid != "saved_adversary"]
                    independent = [cid for cid in eligible if cid != "catchup"]
                    assert selection["head_selections"][key] == check.primary(candidates, eligible)
                    assert selection["independent_selections"][key] == check.primary(candidates, independent)
                    assert selection["auroc_selections"][key] == check.primary(candidates, eligible, auc=True)
                    family = lambda cid: cid if cid in ("saved_adversary", "catchup") else candidates[cid]["family"]
                    for fam, cid in selection["family_selections"][key].items():
                        assert cid == check.primary(candidates, [c for c in candidates if family(c) == fam])
                    report["validation_selections"] += 1
                    yfit, fitmask = check.labels(frames["attacker_fit"], target)
                    yval, valmask = check.labels(frames["attacker_validation"], target)
                    ix = np.random.default_rng(1240000+100*seed+j).permutation(np.flatnonzero(fitmask))[:4096]
                    assert selected["audit_fit_indices"][target] == source_selections[name]["audit_fit_indices"][target]
                    assert selected["audit_fit_indices"][target]["pool_indices_sha256"] == check.array_hash(ix)
                    if name in ("exposed", "prior"):
                        xf = np.eye(classes)[yfit[ix]] if name == "exposed" else np.zeros((len(ix), 1))
                        xv = np.eye(classes)[yval[valmask]] if name == "exposed" else np.zeros((sum(valmask), 1))
                    else:
                        xf, xv = releases[name]["attacker_fit"][ix], releases[name]["attacker_validation"][valmask]
                    oldkey = f"audit/{source[name][1]}/{target}"
                    originals = source_selections[name]["fitting_records"][oldkey]["candidates"]
                    for cid, metadata in candidates.items():
                        path = candidate_path(directory, source[name][0], key, cid, budget)
                        assert check.read(path/"metadata.json") == metadata
                        if name == "prior":
                            with np.load(path/"prior.npz") as prior:
                                probability = prior["probabilities"].copy()
                            counts = np.bincount(yfit[ix], minlength=classes)
                            assert np.array_equal(probability, (counts+1)/(counts.sum()+classes))
                            assert metadata["fit_label_hash"] == check.array_hash(yfit[ix].astype(np.int64))
                            predict = lambda x, p=probability: np.broadcast_to(p, (len(x), len(p))).copy()
                        else:
                            predict, mean, scale, state = check.load_inference(path, metadata)
                            assert metadata["validation_hashes"] == {"x": check.array_hash(xv.astype(np.float64)), "y": check.array_hash(yval[valmask].astype(np.int64))}
                            if cid != "saved_adversary":
                                assert metadata["fit_hashes"] == {"x": check.array_hash(xf.astype(np.float64)), "y": check.array_hash(yfit[ix].astype(np.int64))}
                                assert metadata["fit_support"] == np.bincount(yfit[ix], minlength=classes).tolist()
                            if cid in ("catchup", "saved_adversary"):
                                np.testing.assert_array_equal(mean, np.zeros(16))
                                np.testing.assert_array_equal(scale, np.ones(16))
                                assert metadata["preprocessing_fit_rows"] == 0
                                saved = originals["saved_adversary"]
                                assert metadata["source_state_hash"] == saved["selected_state_hash"]
                                inherited = metadata["inherited_exposure"]
                                assert inherited == saved["inherited_exposure"]
                                ty, mask = check.labels(frames["representation_fit"], target)
                                ty = np.where(mask, ty, -1).astype(np.int64)
                                assert inherited["training_label_sha256"] == check.array_hash(ty)
                                assert inherited["total_row_exposures"] == len(ty)*260
                                assert inherited["total_optimizer_steps"] == int(np.ceil(len(ty)/256))*260
                                report["inherited_exposure_checks"] += 1
                                if cid == "saved_adversary":
                                    assert check.state_hash(state) == metadata["source_state_hash"]
                                    assert metadata["optimizer_steps"] == 0
                                    assert np.array_equal(predict(xv), check.network_probability(state, xv))
                                    report["saved_coordinate_fidelity"] += 1
                                else:
                                    assert metadata["initial_state_hash"] == saved["selected_state_hash"]
                                    assert metadata["optimizer"]["initial_state_entries"] == 0
                                    assert metadata["optimizer"]["restored_state"] is False
                            else:
                                np.testing.assert_array_equal(mean, xf.astype(np.float64).mean(0))
                                std = xf.astype(np.float64).std(0)
                                np.testing.assert_array_equal(scale, np.where(std > 1e-12, std, 1.))
                                report["fitting_standardizers"] += 1
                        for split, pool in (("validation", "attacker_validation"), ("test", "test")):
                            y, valid = check.labels(frames[pool], target)
                            x = np.eye(classes)[y[valid]] if name == "exposed" else np.zeros((sum(valid), 1)) if name == "prior" else releases[name][pool][valid]
                            assert np.array_equal(predict(x), probabilities[f"budget{budget}/{split}/{key}/{cid}"]), (seed, budget, key, cid, split)
                            report["literal_saved_inference_prediction_sets"] += 1
                        if cid not in ("mlp_0", "mlp_1", "catchup"):
                            continue
                        curve = metadata["validation_curve"]
                        assert metadata["parameters"]["epochs"] == budget
                        assert [v["epoch"] for v in curve] == list(range(0, budget+1, 5))
                        step_count = int(np.ceil(len(ix)/256))
                        assert all(v["optimizer_steps"] == v["epoch"]*step_count for v in curve)
                        best = min(curve, key=lambda v: (v["validation_log_loss"], v["epoch"]))
                        assert metadata["selected_epoch"] == best["epoch"]
                        assert metadata["validation_scores"]["log_loss"] == best["validation_log_loss"]
                        assert metadata["training_row_exposures"] == len(ix)*budget
                        check.replay_schedule(metadata, len(ix))
                        report["mlp_curves_schedules"] += 1
                        prefix = selected["budgets"]["120"]["fitting_records"][key]["candidates"][cid]
                        original = originals[cid]
                        if budget == 120:
                            for field in ("initial_state_hash", "final_state_hash", "selected_state_hash", "schedule_hash", "optimizer_steps",
                                          "training_row_exposures", "selected_epoch", "validation_curve", "fit_hashes", "validation_hashes", "validation_scores"):
                                assert metadata[field] == original[field], (seed, key, cid, field)
                            oldpath = source[name][0]/"fitted"/oldkey/cid
                            old_predict, _, _, _ = check.load_inference(oldpath, original)
                            assert np.array_equal(old_predict(xv), predict(xv))
                            report["nested_prefixes_equal_original"] += 1
                        else:
                            assert curve[:len(prefix["validation_curve"])] == prefix["validation_curve"]
                            assert metadata["validation_scores"]["log_loss"] <= prefix["validation_scores"]["log_loss"]
                        checkpoint = directory/"fitted"/key/("saved_start" if cid == "catchup" else "fresh")/f"last_training_{cid}_epoch{budget}.pt"
                        checkpoint_records[f"{budget}/{key}/{cid}"] = checkpoint_evidence(checkpoint, metadata, len(ix))
                        report["actual_training_checkpoints"] += 1
        report["seeds"][str(seed)] = {"passed": True, "metrics_sha256": check.sha(directory/"metrics.json"),
            "prediction_sha256": check.sha(directory/"predictions.npz"), "selection_sha256": check.sha(directory/"selection_before_test.json"),
            "checked_local_artifacts": len(local), "actual_last_checkpoint_evidence": checkpoint_records,
            "race_support_flags": race_flags, "controls": controls}
    report.update(passed=not check.errors, errors=check.errors, numeric_values_compared=check.numeric_comparisons,
                  max_absolute_metric_error=check.max_error, max_error_path=check.max_error_path,
                  runtime_seconds=time.perf_counter()-started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/redesign_20260908_acs_preservation_v1")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--report", type=Path, help="Fresh JSON report; otherwise stdout; never overwrites")
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError("Preserve existing evidence; choose a fresh report")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = verify(args.out.resolve(), args.seeds)
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

#!/usr/bin/env python3
"""Replay saved bottleneck state/schedule/release checks without model fitting.

Requires local parent arrays/checkpoints and completed ACS bottleneck results.
Prints JSON by default or exclusively creates a fresh --report destination.
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
from experiments.acs_bottleneck_training import BottleneckModel, TRAIN_CONFIG, SOURCE_SCHEMA, ATTRIBUTE_SCHEMA, tree_digest
from experiments.acs_transfer_models import state_digest
from scripts.verify_acs_protection_artifacts import arrays, read, sha, array_hash, labels, require

DEFAULT_RELATIVE = Path("results/redesign_20260908_acs_bottleneck_v1")
ARMS = ("C_bottleneck", "D_protected")


def recorded_path(value, out):
    path = Path(value)
    if path.name == "leace.py" and "concept_erasure" in path.parts:
        return Path(inspect.getfile(LeaceEraser))
    try:
        return out / path.relative_to(DEFAULT_RELATIVE)
    except ValueError:
        return ROOT / path


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def adam_steps(state, expected):
    require(bool(state["state"]), "Expected populated Adam state")
    require(all(int(v["step"]) == expected for v in state["state"].values()), "Actual Adam tensor step counters")
    for group in state["param_groups"]:
        require(group["lr"] == .001 and group["betas"] == (.9, .999)
                and group["eps"] == 1e-8 and group["weight_decay"] == 0, "Fixed Adam configuration")


def verify(out):
    started = time.perf_counter()
    cfg, freeze = read(out / "config.json"), read(out / "protocol_freeze.json")
    for p, expected in freeze["sha256"].items():
        require(sha(recorded_path(p, out)) == expected, "Frozen execution source/config: " + p)
    for p, expected in freeze["reference_record_hashes"].items():
        require(sha(ROOT / p) == expected, "Historical reference record: " + p)
    parent, reference = ROOT / cfg["parent_results"], ROOT / cfg["reference_results"]
    parent_cfg, parent_support = read(parent / "config.json"), read(parent / "schema_support.json")
    require(sha(ROOT / parent_cfg["raw_path"]) == parent_support["raw_sha256"], "Raw CSV hash")
    raw = pd.read_csv(ROOT / parent_cfg["raw_path"], usecols=["SERIALNO", "SEX", "RAC1P", "PINCP", "ESR", "PUBCOV", "MIG", "JWMNP"], dtype={"SERIALNO": str})
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "verification_source_sha256": sha(Path(__file__)),
              "verification_helper_sha256": sha(ROOT / "scripts/verify_acs_protection_artifacts.py"),
              "command": [sys.executable, *sys.argv], "evaluation_status": cfg["evaluation_status"],
              "no_model_fits": True, "seeds": {}, "limitations": [
                  "Checks saved optimizer counters, exact forks, fitting identities and inference; does not retrain or reconstruct every historical optimizer action.",
                  "Uses recorded within-run integrity checks plus current file hashes and independent release replay.",
                  "The historical test population is development evaluation, not untouched confirmation."]}
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for seed in cfg["seeds"]:
            directory, refdir = out / f"seed_{seed}", reference / f"seed_{seed}"
            train_dir = directory / "training"
            meta, metric = read(train_dir / "training.json"), read(directory / "metrics.json")
            rf, selection = read(directory / "release_freeze.json"), read(directory / "selection_before_test.json")
            require(meta["config"] == TRAIN_CONFIG and not meta["miniature"], "Fixed scientific training configuration")
            require(meta["module_source_sha256"] == sha(ROOT / "experiments/acs_bottleneck_training.py"), "Actual training source")
            require(meta["fit_pool"] == "representation_fit" and meta["reserved_labels_received"] is False
                    and meta["final_evaluation_received"] is False, "Training access declaration")
            rows, oldrows = arrays(directory / "split_rows.npz"), arrays(refdir / "split_rows.npz")
            require(rows.keys() == oldrows.keys(), "Split schema")
            groups = {}
            for pool, idx in rows.items():
                require(np.array_equal(idx, oldrows[pool]), "Original ordered split: " + pool)
                require(array_hash(idx) == parent_support["seeds"][str(seed)]["pools"][pool]["raw_row_hash"], "Original split hash")
                groups[pool] = set(raw.iloc[idx].SERIALNO)
            for j, pool in enumerate(groups):
                for other in list(groups)[j + 1:]:
                    require(groups[pool].isdisjoint(groups[other]), "Household pool overlap")
            pca = arrays(refdir / "release_E_pca.npz")
            fit = pca["representation_fit"]
            n = len(fit)
            source = labels(raw.iloc[rows["representation_fit"]])
            validation = labels(raw.iloc[rows["source_validation"]])
            require(meta["source_label_keys"] == list(SOURCE_SCHEMA) and meta["attribute_schema"] == ATTRIBUTE_SCHEMA, "Training label whitelist")
            for target in SOURCE_SCHEMA:
                require(meta["source_label_hashes"][target] == array_hash(source[target]), "Actual source-fitting labels")
                require(meta["source_validation_label_hashes"][target] == array_hash(validation[target]), "Source-only diagnostic labels")
            for target in ATTRIBUTE_SCHEMA:
                y = source[target]
                require(meta["attribute_label_hashes"][target] == array_hash(y), "Actual protected fitting labels")
                counts = np.bincount(y[y >= 0], minlength=ATTRIBUTE_SCHEMA[target])
                probabilities = counts / counts.sum()
                entropy = float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())
                require(meta["prior_entropies"][target]["entropy"] == entropy
                        and meta["prior_entropies"][target]["support"] == counts.tolist(), "Empirical masked fitting prior")
            mean, std = fit.astype(np.float64).mean(0), fit.astype(np.float64).std(0)
            scale = np.where(std > TRAIN_CONFIG["standardizer_std_floor"], std, 1.)
            require(np.array_equal(meta["preprocessing"]["mean"], mean)
                    and np.array_equal(meta["preprocessing"]["scale"], scale), "Representation-fitting-only input standardizer")
            require(meta["fit_input_sha256"] == array_hash(fit)
                    and meta["source_validation_input_sha256"] == array_hash(pca["source_validation"]), "Training feature identity")
            standardized = ((fit.astype(np.float64) - mean) / scale).astype(np.float32)
            require(meta["standardized_fit_sha256"] == array_hash(standardized), "Actual standardized fitting input")
            batch = TRAIN_CONFIG["batch_size"]
            per_epoch = math.ceil(n / batch)
            base_steps, warm_steps = 60 * per_epoch, 20 * per_epoch
            continuation, adv_continuation = 80 * per_epoch, 240 * per_epoch
            for phase, epochs, offset in (("warm_base", 60, 0), ("warm_adversary", 20, 100), ("continuation", 80, 200)):
                schedule = meta["schedules"][phase]
                require(schedule["seed"] == 1280000 + 100 * seed + offset, "Predeclared phase seed")
                generator, digest = np.random.default_rng(schedule["seed"]), hashlib.sha256()
                for _ in range(epochs):
                    digest.update(generator.permutation(n).tobytes())
                require(digest.hexdigest() == schedule["sha256"], "Reconstructed minibatch schedule")
            initial, base, warm = (load(train_dir / name) for name in ("initialization.pt", "warm_base.pt", "warm_adversary.pt"))
            require(initial["counters"] == {"mapper_optimizer_steps": 0, "adversary_optimizer_steps": 0, "epoch": 0}, "True initialization counters")
            require(initial["adversary_state"] == {} and initial["adversary_optimizer_state"] is None, "No adversary before common base training")
            require(not initial["mapper_optimizer_state"]["state"], "Empty initial Adam state")
            require(state_digest(initial["model_state"]) == meta["initialization_hashes"]["model"], "Initial model hash")
            require(state_digest(base["model_state"]) == state_digest(warm["model_state"]), "Adversary warmup leaves mapper/heads/decoder unchanged")
            require(tree_digest(base["mapper_optimizer_state"]) == tree_digest(warm["mapper_optimizer_state"]), "Adversary warmup leaves mapper Adam unchanged")
            adam_steps(base["mapper_optimizer_state"], base_steps)
            adam_steps(warm["adversary_optimizer_state"], warm_steps)
            forks = {arm: load(train_dir / arm / "fork.pt") for arm in ARMS}
            require(tree_digest(forks[ARMS[0]]) == tree_digest(forks[ARMS[1]]), "Exact full C/D fork clone")
            require(meta["common_optimizer_counts"] == {"mapper_optimizer_steps": base_steps, "adversary_optimizer_steps": warm_steps}, "Common counters")
            require(meta["common_base_row_exposures"] == n * 60 and meta["common_adversary_row_exposures"] == n * 20, "Common row exposures")
            arm_reports = {}
            for arm_name in ARMS:
                arm_meta, fork = meta["arms"][arm_name], forks[arm_name]
                final = load(train_dir / arm_name / "final.pt")
                protected = arm_name == "D_protected"
                require(arm_meta["fork_hashes"] == meta["shared_fork_hashes"], "Recorded fork identity")
                for ck, mk in (("model_state", "model"), ("adversary_state", "adversaries"),
                               ("mapper_optimizer_state", "mapper_optimizer"), ("adversary_optimizer_state", "adversary_optimizer")):
                    digest = state_digest(fork[ck]) if ck.endswith("state") and "optimizer" not in ck else tree_digest(fork[ck])
                    require(digest == meta["shared_fork_hashes"][mk], "Actual fork tensor/optimizer hash")
                require(arm_meta["schedule_hash"] == meta["schedules"]["continuation"]["sha256"], "Matched C/D minibatches")
                require(arm_meta["continuation_mapper_optimizer_steps"] == continuation
                        and arm_meta["continuation_adversary_optimizer_steps"] == adv_continuation, "Continuation update counts")
                require(arm_meta["mapper_row_exposures"] == n * 80 and arm_meta["adversary_row_exposures"] == n * 240, "Continuation row exposures")
                expected_counts = {"mapper_optimizer_steps": base_steps + continuation,
                                   "adversary_optimizer_steps": warm_steps + adv_continuation}
                require(arm_meta["optimizer_counts_including_common"] == expected_counts, "Total update accounting")
                require(final["counters"] == {**expected_counts, "continuation_epoch": 80}, "Fixed-final checkpoint counters")
                require(arm_meta["selected_epoch"] == 80, "Fixed epoch without validation selection")
                adam_steps(final["mapper_optimizer_state"], base_steps + continuation)
                adam_steps(final["adversary_optimizer_state"], warm_steps + adv_continuation)
                for schema, prefix, common_epochs, arm_epochs in ((SOURCE_SCHEMA, "source", 60, 80), (ATTRIBUTE_SCHEMA, "attribute", 20, 240)):
                    for target in schema:
                        known = int((source[target] >= 0).sum())
                        require(meta[f"common_{prefix}_valid_label_exposures"][target] == known * common_epochs
                                and arm_meta[f"{prefix}_valid_label_exposures"][target] == known * arm_epochs, "Actual known-label exposure accounting")
                for curve_row in arm_meta["curve"]:
                    require(curve_row["mapper_optimizer_steps"] == base_steps + curve_row["epoch"] * per_epoch
                            and curve_row["adversary_optimizer_steps"] == warm_steps + 3 * curve_row["epoch"] * per_epoch, "Logged epoch/action correspondence")
                for field, state in (("final_model_hash", final["model_state"]), ("final_adversary_hash", final["adversary_state"])):
                    require(arm_meta[field] == state_digest(state), "Final checkpoint state identity")
                require(arm_meta["final_mapper_optimizer_hash"] == tree_digest(final["mapper_optimizer_state"])
                        and arm_meta["final_adversary_optimizer_hash"] == tree_digest(final["adversary_optimizer_state"]), "Final Adam identity")
                diagnostic = arm_meta["first_batch_gradient_diagnostics_at_shared_fork"]
                require(diagnostic["base_first_weight_l2"] > 0 and diagnostic["protection_first_weight_l2"] > 0,
                        "Recorded nonzero shared-fork gradient paths")
                require(diagnostic["protection_source_decoder_gradients_all_absent"] and diagnostic["protection_source_decoder_gradient_l2"] == 0,
                        "Recorded protection-only gradient isolation")
                require(arm_meta["mapper_loss_uses_protection_gradient"] == protected
                        and (arm_meta["applied_protection_mapper_l2_at_shared_fork"] > 0) == protected, "C/D protection-gradient distinction")
                with torch.random.fork_rng(devices=[]):
                    model = BottleneckModel(mean, scale)
                model.load_state_dict(final["model_state"])
                model.freeze()
                cache = arrays(directory / f"release_{arm_name}.npz")
                require(set(cache) == set(rows), "All release pools saved")
                before_hash = state_digest(model.state_dict())
                require(before_hash == rf["learned_state"][arm_name]["model"], "Release uses fixed final model")
                require(state_digest(final["adversary_state"]) == rf["learned_state"][arm_name]["adversaries"], "Frozen final adversary identity")
                for pool, value in cache.items():
                    require(np.array_equal(model.release(pca[pool]), value), "Bitwise frozen release replay: " + arm_name + "/" + pool)
                    expected = (metric["integrity"]["evaluation_output_hashes"][arm_name] if pool == "test"
                                else rf["output_hashes"][arm_name][pool])
                    require(array_hash(value) == expected, "Frozen release output hash")
                require(before_hash == state_digest(model.state_dict()), "Inference modified frozen model")
                arm_reports[arm_name] = {"optimizer_counts": expected_counts, "continuation_mapper_steps": continuation,
                    "continuation_adversary_steps": adv_continuation, "all_saved_adam_step_tensors_verified": True,
                    "identical_shared_model_and_optimizer_fork": True, "schedule_hash": arm_meta["schedule_hash"],
                    "bitwise_release_replays": list(cache), "source_valid_label_exposures": arm_meta["source_valid_label_exposures"],
                    "attribute_valid_label_exposures": arm_meta["attribute_valid_label_exposures"],
                    "fixed_final_epoch": 80, "frozen_model_sha256": before_hash}
            provenance = read(directory / "parent_provenance.json")
            for p, expected in {**provenance["used_reference_files_sha256"], **provenance["used_original_files_sha256"]}.items():
                require(sha(ROOT / p) == expected, "Immutable reference file: " + p)
            require(selection["release_freeze_sha256"] == sha(directory / "release_freeze.json")
                    and selection["protocol_freeze_sha256"] == sha(out / "protocol_freeze.json"), "Release/protocol selection linkage")
            require(metric["integrity"]["selection_sha256"] == sha(directory / "selection_before_test.json"), "Selection hash")
            require(rf["created_utc"] < selection["created_utc"] < metric["integrity"]["evaluation_started_utc"], "Within-run freeze/selection/evaluation order")
            for name, value in metric["integrity"].items():
                if name.endswith("_unchanged"):
                    require(value is True, "Recorded within-run integrity: " + name)
            manifest = read(directory / "local_artifacts.json")
            for item in manifest:
                path = recorded_path(item["path"], out)
                require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Saved artifact file identity")
            result["seeds"][str(seed)] = {"passed": True, "fit_rows": n, "batch_size": batch,
                "common_base_optimizer_steps": base_steps, "common_adversary_optimizer_steps": warm_steps,
                "arms": arm_reports, "source_labels_masks_and_preprocessing_verified": True,
                "parent_split_household_identity_verified": True, "saved_file_hashes_verified": len(manifest),
                "within_run_freeze_and_selection_gates_verified": True}
    result["all_seeds_passed"] = True
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_RELATIVE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error("Refusing to overwrite an existing report")
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

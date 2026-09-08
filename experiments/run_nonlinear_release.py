"""Frozen nonlinear release pilot: simple releases versus matched adversaries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments import run_nonlinear_conflict as old
from experiments.nonlinear_conflict_probes import PROBE_CONFIG, predictive_scores

torch.set_num_threads(1)
PRIOR = ROOT / "results/redesign_20260907_nonlinear_upstream_v1"
THRESHOLDS = dict(old.THRESHOLDS)
CONFIG = {
    "seeds": [0, 1, 2], "prior_results": str(PRIOR.relative_to(ROOT)),
    "fresh_test_rng_seeds": {str(s): 900004 + 100*s for s in (0, 1, 2)},
    "split_sizes": old.CONFIG["split_sizes"], "generator": old.CONFIG["generator"],
    "utility_floor_each_purpose": .99, "thresholds": THRESHOLDS,
    "fixed_weights": [.01, .1], "dual_initial_weights": [.01, .1],
    "leace": old.CONFIG["leace"], "probes": PROBE_CONFIG,
    "selection": "Within each new method: require both validation MLP task R2 >= .99; then minimize maximum of all 10 leakage/threshold ratios; tie by higher minimum purpose utility then key. If no utility-feasible candidate, diagnostic only: highest minimum purpose utility then lowest ratio then key.",
    "device": "cpu", "threads": 1,
}
dump = old.dump


def make_data(seed, include_test=False):
    """Reuse fitting streams exactly; never generate the old final-test stream."""
    data, metadata = old.make_data(seed, include_test=False)
    if include_test:
        rng_seed = CONFIG["fresh_test_rng_seeds"].get(str(seed), 900004 + 100*seed)
        latent = np.random.default_rng(rng_seed).normal(size=(CONFIG["split_sizes"]["test"], 8))
        q1, q2, offset = old.generator_parameters()
        r = latent @ q1
        x = (r + .2*r**3) @ q2 + offset
        x = (x - np.array(metadata["preprocessing_mean"])) / np.array(metadata["preprocessing_std"])
        ids = np.arange(len(x), dtype=np.int64) + np.int64(rng_seed)*100000
        data["test"] = {"x": x, "y": latent[:, :3], "ids": ids}
        metadata["split_seeds"]["test"] = rng_seed
        metadata["splits"]["test"] = {
            "n": len(x), "x_sha256": old.digest_array(x), "target_sha256": old.digest_array(latent[:, :3]),
            "id_sha256": old.digest_array(ids), "id_first": int(ids[0]), "id_last": int(ids[-1]),
            "target_variance": latent[:, :3].var(0).tolist(), "class_support": "not applicable: continuous targets",
        }
    return data, metadata


def _finite(value):
    return value is not None and math.isfinite(value)


def assess(evaluation):
    utility = {k: evaluation["task"]["mlp"][k]["r2"] for k in ("p1_U", "p2_V")}
    utility_defined = all(_finite(v) for v in utility.values())
    utility_pass = utility_defined and all(v >= .99 for v in utility.values())
    leakage = {}
    for key, threshold in THRESHOLDS.items():
        values = {kind: evaluation["attack"][kind][key]["r2"] for kind in ("linear", "mlp")}
        defined = all(_finite(v) for v in values.values())
        worst = max(values.values()) if defined else None
        leakage[key] = {**values, "worst": worst, "threshold": threshold,
                        "ratio": worst / threshold if defined else None}
    defined = utility_defined and all(v["ratio"] is not None for v in leakage.values())
    ratio = max(v["ratio"] for v in leakage.values()) if defined else None
    protection_pass = defined and ratio <= 1.
    return {"utility": utility, "utility_pass": utility_pass, "leakage": leakage,
            "worst_ratio": ratio, "protection_pass": protection_pass,
            "feasible": utility_pass and protection_pass, "defined": defined}


def select_configuration(rows):
    """Only validation evaluations enter; task floors are never averaged."""
    if not rows:
        raise ValueError("Empty configuration grid")
    assessed = {k: assess(v) for k, v in rows.items()}
    def minimum_task(k):
        values = list(assessed[k]["utility"].values())
        return min(values) if all(_finite(v) for v in values) else -float("inf")
    def ratio(k):
        value = assessed[k]["worst_ratio"]
        return value if value is not None else float("inf")
    feasible = [k for k, v in assessed.items() if v["utility_pass"]]
    key = min(feasible, key=lambda k: (ratio(k), -minimum_task(k), k)) if feasible else min(
        assessed, key=lambda k: (-minimum_task(k), ratio(k), k))
    return {"key": key, "utility_feasible": bool(feasible),
            "protection_pass": assessed[key]["protection_pass"],
            "diagnostic_only": not bool(feasible), "all_candidates": assessed,
            "selection_split": "validation"}


@torch.no_grad()
def prediction_views(model, data):
    model["encoder"].eval()
    model["heads"].eval()
    result = {}
    for split, row in data.items():
        h = [old.encode(model["encoder"], row["x"], p) for p in range(2)]
        result[split] = [model["heads"][p](torch.from_numpy(h[p]).float()).double().numpy() for p in range(2)]
    return old.apply_erasers(result, [None, None])


def direct_scores(views, y):
    return predictive_scores(y[:, :2], np.concatenate(views[:2], 1), ["p1_U", "p2_V"])


def diagnostic_views(data, exposed=False):
    raw = {name: [row["y"] if exposed else row["y"][:, p:p+1] for p in range(2)]
           for name, row in data.items()}
    return old.apply_erasers(raw, [None, None])


def save_erasers(out, erasers):
    arrays = {}
    for p, eraser in enumerate(erasers):
        if eraser is not None:
            arrays[f"p{p+1}_matrix"] = eraser.P.numpy()
            arrays[f"p{p+1}_center"] = eraser.bias.numpy()
    np.savez_compressed(out/"erasers.npz", **arrays)


@torch.no_grad()
def training_adversary_scores(model, views, y):
    """Transfer diagnostic only; the independent audit determines selection."""
    model["adversaries"].eval()
    release = model["training_release"]
    h = [release.normalize(torch.from_numpy(views[p]).float(), p) for p in range(2)]
    predictions = model["adversaries"](h + [torch.cat(h, 1)])
    means, stds = (np.array(model["metadata"][k]) for k in ("target_mean", "target_std"))
    result = {}
    for p, targets in enumerate(((1, 2), (0, 2), (2,))):
        names = [f"p{p+1}_{old.SIGNALS[j]}" if p < 2 else "combined_S" for j in targets]
        prediction = predictions[p].double().numpy()*stds[list(targets)]+means[list(targets)]
        result.update(predictive_scores(y[:, targets], prediction, names))
    return result


def run_seed(seed, out, cfg, prior=PRIOR):
    from experiments.nonlinear_release_training import load_pretrained, load_reference, train_adversarial
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(exist_ok=False)
    data, manifest = make_data(seed)
    dump(out/"split_manifest_fit.json", manifest)
    np.savez_compressed(out/"split_ids_fit.npz", **{k: v["ids"] for k, v in data.items()})
    prior_seed = Path(prior)/f"seed_{seed}"
    prior_manifest_path = prior_seed/"split_manifest_fit.json"
    if prior_manifest_path.exists():
        prior_manifest = json.loads(prior_manifest_path.read_text())
        if manifest != prior_manifest:
            raise ValueError("Fitting data or preprocessing differs from saved-checkpoint provenance")
    pretrained = load_pretrained(prior_seed)
    saved = load_reference(prior_seed, "C_task_only")
    raw = {"seed": seed, "configuration": cfg, "methods": {}, "diagnostics": {}}
    prepared = {}

    def audit(key, views, record, *, kind, model=None, erasers=None, diagnostic=False):
        tick = time.perf_counter()
        method_out = out/key
        method_out.mkdir(exist_ok=True)
        print(f"seed={seed}: independent audit {key}", flush=True)
        probes = old.fit_method_probes(views, data, seed, method_out, cfg["probes"])
        evaluation = old.evaluate_probes(probes, views["validation"], data["validation"]["y"])
        record.update(validation=evaluation, assessment_validation=assess(evaluation),
                      release_dimensions=[views["validation"][p].shape[1] for p in range(3)],
                      audit_fit_seconds=time.perf_counter()-tick)
        if kind in ("oracle", "prediction"):
            record["direct_prediction"] = {"validation": direct_scores(views["validation"], data["validation"]["y"])}
        if erasers is not None:
            save_erasers(method_out, erasers)
            record["calibration_covariance"] = {
                label: old.empirical_statistics(h, data["calibration"]["y"])
                for label, h in zip(("p1", "p2", "combined"), views["calibration"])}
        if model is not None and "adversaries" in model:
            record["training_adversary_validation_final_release"] = training_adversary_scores(model, views["validation"], data["validation"]["y"])
        container = raw["diagnostics"] if diagnostic else raw["methods"]
        container[key] = record
        prepared[key] = {"kind": kind, "model": model, "erasers": erasers, "probes": probes,
                         "diagnostic": diagnostic}
        dump(method_out/"validation.json", record)
        a = record["assessment_validation"]
        print(f"  validation {key}: U={a['utility']['p1_U']:.5f} V={a['utility']['p2_V']:.5f} max leak/threshold={a['worst_ratio']:.3f}", flush=True)

    # Revealing controls and attack competence are evaluated before new training.
    audit("A_oracle", diagnostic_views(data), {"provenance": "inaccessible true U/V; population independence diagnostic"}, kind="oracle")
    audit("B_prediction_only", prediction_views(saved, data), {"provenance": saved["metadata"]}, kind="prediction", model=saved)
    audit("exposed_target", diagnostic_views(data, exposed=True),
          {"provenance": "both views deliberately expose U,V,S; competence diagnostic"}, kind="exposed", diagnostic=True)
    for key, model in [("C_saved_task_only", saved)] + [
            (f"R_prior_{w:g}", load_reference(prior_seed, f"D_protection_{w:g}")) for w in (.1, 1.)]:
        h = old.frozen_views(model["encoder"], data, True)
        erasers, views = old.calibrate_and_apply(h, data["calibration"]["y"], True)
        audit(key, views, {"provenance": model["metadata"], "training": None}, kind="representation", model=model, erasers=erasers)

    for key, method, weight in [("C_matched_task_only", "task", 0.)] + [
        (f"D_fixed_{w:g}", "fixed", w) for w in cfg["fixed_weights"]] + [
        (f"E_dual_{w:g}", "dual", w) for w in cfg["dual_initial_weights"]]:
        print(f"seed={seed}: training {key}", flush=True)
        training_started = time.perf_counter()
        model = train_adversarial(pretrained, data["representation_train"]["x"], data["representation_train"]["y"],
                                  seed=seed, method=method, weight=weight, out_dir=out/key/"training", config=cfg["training"])
        training_seconds = time.perf_counter()-training_started
        h = old.frozen_views(model["encoder"], data, True)
        erasers, views = old.calibrate_and_apply(h, data["calibration"]["y"], True)
        audit(key, views, {"training": model["metadata"], "training_seconds": training_seconds,
                          "provenance": "identical saved selected task-pretrained checkpoint"},
              kind="representation", model=model, erasers=erasers)

    raw["selection"] = {label: select_configuration({k: v["validation"] for k, v in raw["methods"].items() if k.startswith(prefix)})
                        for label, prefix in (("D", "D_fixed_"), ("E", "E_dual_"))}
    matched = [v["training"] for v in raw["methods"].values() if v.get("training") is not None]
    match_keys = ("initial_state_hash", "pretrained_state_hash", "schedule_hash", "adversary_initial_state_hash",
                  "adversary_schedule_hash", "optimizer_steps", "adversary_optimizer_steps", "trainable_parameter_count")
    raw["matching"] = {k: len({str(v[k]) for v in matched}) == 1 for k in match_keys}
    if not all(raw["matching"].values()):
        raise AssertionError(f"Unmatched training: {raw['matching']}")
    dump(out/"selection_before_test.json", {"selection": raw["selection"], "test_generated": False,
         "fresh_test_rng_seed": CONFIG["fresh_test_rng_seeds"].get(str(seed), 900004 + 100*seed), "elapsed_seconds": time.perf_counter()-started})
    # Sealed boundary: every model and attacker is frozen, all selection is saved.
    final_started = time.perf_counter()
    final, final_manifest = make_data(seed, include_test=True)
    final = {"test": final["test"]}
    dump(out/"split_manifest_all.json", final_manifest)
    np.savez_compressed(out/"split_ids_test.npz", test=final["test"]["ids"])
    for key, item in prepared.items():
        if item["kind"] in ("oracle", "exposed"):
            views = diagnostic_views(final, exposed=item["kind"] == "exposed")
        elif item["kind"] == "prediction":
            views = prediction_views(item["model"], final)
        else:
            h = old.frozen_views(item["model"]["encoder"], final, True)
            views = old.apply_erasers(h, item["erasers"])
        record = (raw["diagnostics"] if item["diagnostic"] else raw["methods"])[key]
        record["test"] = old.evaluate_probes(item["probes"], views["test"], final["test"]["y"])
        record["assessment_test"] = assess(record["test"])
        if "direct_prediction" in record:
            record["direct_prediction"]["test"] = direct_scores(views["test"], final["test"]["y"])
        dump(out/key/"test.json", {"evaluation": record["test"], "assessment": record["assessment_test"],
                                    "direct_prediction": record.get("direct_prediction", {}).get("test")})
    raw["runtime"] = {"total_seconds": time.perf_counter()-started, "final_test_seconds": time.perf_counter()-final_started}
    dump(out/"metrics.json", raw)
    print(f"seed={seed}: complete in {raw['runtime']['total_seconds']:.2f}s", flush=True)
    return raw


def main():
    from experiments.nonlinear_release_training import TRAIN_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not (out/"PROTOCOL.md").exists():
        raise FileNotFoundError("Write PROTOCOL.md before fitting")
    cfg = json.loads(json.dumps({**CONFIG, "training": TRAIN_CONFIG}))
    sources = ["experiments/run_nonlinear_release.py", "experiments/nonlinear_release_training.py",
               "experiments/run_nonlinear_conflict.py", "experiments/nonlinear_conflict_training.py",
               "experiments/nonlinear_conflict_probes.py", "experiments/run_redesign_conflict.py",
               "pcrl/models/lora.py", "pcrl/training/proxy_lagrangian.py"]
    hashes = {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    for filename, value in (("config.json", cfg), ("frozen_source_hashes.json", hashes)):
        path = out/filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen {filename} changed; preserve evidence and use a fresh directory")
        if not path.exists():
            dump(path, value)
    inputs = [PRIOR/"generator.json", PRIOR/"PROTOCOL.md"]
    for seed in CONFIG["seeds"]:
        inputs.append(PRIOR/f"seed_{seed}"/"pretraining/selected.pt")
        inputs.append(PRIOR/f"seed_{seed}"/"pretraining/metadata.json")
        inputs.append(PRIOR/f"seed_{seed}"/"split_manifest_fit.json")
        for arm in ("C_task_only", "D_protection_0.1", "D_protection_1"):
            inputs.append(PRIOR/f"seed_{seed}"/arm/"adaptation/final.pt")
            inputs.append(PRIOR/f"seed_{seed}"/arm/"adaptation/metadata.json")
    prior_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    frozen = out/"prior_checkpoint_hashes.json"
    if frozen.exists() and json.loads(frozen.read_text()) != prior_hashes:
        raise ValueError("Prior input checkpoint changed")
    if not frozen.exists():
        dump(frozen, prior_hashes)
    protocol_hash = hashlib.sha256((out/"PROTOCOL.md").read_bytes()).hexdigest()
    freeze = out/"protocol_sha256.json"
    if freeze.exists() and json.loads(freeze.read_text()) != protocol_hash:
        raise ValueError("Protocol changed after freeze")
    if not freeze.exists():
        dump(freeze, protocol_hash)
    for name in sources:
        dest = out/"source_snapshot"/name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
    if not (out/"generator.json").exists():
        (out/"generator.json").write_bytes((PRIOR/"generator.json").read_bytes())
    invocation = out/("invocation_prepare.json" if args.prepare_only else f"invocation_{'_'.join(map(str,args.seeds))}.json")
    if invocation.exists():
        raise FileExistsError("Do not overwrite recorded invocations")
    dump(invocation, {"command": sys.argv, "platform": platform.platform(), "python": sys.version,
         "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "concept-erasure")},
         "source_hashes": hashes, "protocol_hash": protocol_hash})
    if args.prepare_only:
        print("Protocol, source, configurations and prior checkpoints frozen; no test generated.")
        return
    for seed in args.seeds:
        if seed not in CONFIG["seeds"]:
            raise ValueError("Undeclared seed")
        run_seed(seed, out/f"seed_{seed}", cfg)


if __name__ == "__main__":
    main()

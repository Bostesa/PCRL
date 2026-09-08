"""Fixed nonlinear PCRL pilot: protection-aware upstream training before LEACE.

L=(U,V,S,E1,...,E5) ~ N(0,I8); X=(LQ1 + .2*(LQ1)^3)Q2+b.
The two fixed orthogonal matrices and the monotone componentwise cubic make
the observation map invertible. No generator parameter depends on outcomes.
"""
from __future__ import annotations

import argparse
import copy
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
from concept_erasure import LeaceEraser
from experiments.run_redesign_conflict import empirical_statistics, score

torch.set_num_threads(1)
SIGNALS = ("U", "V", "S")
PROHIBITED = ((1, 2), (0, 2))
THRESHOLDS = {"p1_V": .05, "p1_S": .05, "p2_U": .05, "p2_S": .05, "combined_S": .10}
CONFIG = {
    "seeds": [0, 1, 2],
    "split_sizes": {"representation_train": 4096, "calibration": 2048,
                    "attacker_fit": 2048, "validation": 2048, "test": 4096},
    "generator": {"mixing_seeds": [20260908, 20260909], "cubic_coefficient": .2,
                  "latent_dimension": 8, "description": __doc__},
    "protection_strengths": [0., .1, 1.], "thresholds": THRESHOLDS,
    "leace": {"shrinkage": False, "constrain_cov_trace": False, "svd_tol": 1e-10},
    "primary_utility": "mean independently fitted MLP task probe R2 for U and V",
    "feasibility": "All five prohibited recipient/target scores satisfy their threshold for BOTH linear and MLP validation attackers",
    "selection": "Among C and two D strengths, highest validation utility if feasible; otherwise minimum sum normalized excess then highest utility, then smaller strength. Also select among positive strengths by the same rule.",
    "device": "cpu", "threads": 1,
}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def generator_parameters():
    qs = []
    for seed in CONFIG["generator"]["mixing_seeds"]:
        q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(8, 8)))
        qs.append(q * np.sign(np.diag(r)))
    return qs[0], qs[1], np.linspace(-.4, .4, 8)


def generate_split(seed, split):
    idx = list(CONFIG["split_sizes"]).index(split)
    rng_seed = 500000 + 100 * seed + idx
    latent = np.random.default_rng(rng_seed).normal(size=(CONFIG["split_sizes"][split], 8))
    q1, q2, offset = generator_parameters()
    rotated = latent @ q1
    x = (rotated + CONFIG["generator"]["cubic_coefficient"] * rotated**3) @ q2 + offset
    ids = np.arange(len(x), dtype=np.int64) + np.int64(rng_seed) * 100000
    return {"x": x, "y": latent[:, :3], "ids": ids}, rng_seed


def make_data(seed, include_test=False):
    """Create distinct split streams; test is not even generated during fitting."""
    names = [name for name in CONFIG["split_sizes"] if name != "test" or include_test]
    raw, split_seeds = {}, {}
    for name in names:
        raw[name], split_seeds[name] = generate_split(seed, name)
    mean = raw["representation_train"]["x"].mean(0)
    std = raw["representation_train"]["x"].std(0)
    data = {name: {**row, "x": (row["x"] - mean) / std} for name, row in raw.items()}
    meta = {"split_seeds": split_seeds, "preprocessing_mean": mean.tolist(),
            "preprocessing_std": std.tolist(), "splits": {name: {
                "n": len(row["x"]), "x_sha256": digest_array(row["x"]),
                "target_sha256": digest_array(row["y"]), "id_sha256": digest_array(row["ids"]),
                "id_first": int(row["ids"][0]), "id_last": int(row["ids"][-1]),
                "target_variance": row["y"].var(0).tolist(),
                "class_support": "not applicable: continuous targets",
            } for name, row in data.items()}}
    return data, meta


def select_configuration(rows):
    """Select only from caller-supplied validation scores, never test outcomes."""
    if not rows:
        raise ValueError("No configurations to select")
    assessed = {}
    for key, row in rows.items():
        leak = row["leakage"]
        defined = set(leak) == set(THRESHOLDS) and all(
            v is not None and math.isfinite(v) for v in leak.values()
        ) and row["utility"] is not None and math.isfinite(row["utility"])
        violations = {name: max(0., leak[name] - threshold) for name, threshold in THRESHOLDS.items()} if defined else {}
        total = sum(violations[k] / THRESHOLDS[k] for k in THRESHOLDS) if defined else float("inf")
        assessed[key] = {"feasible": defined and total == 0, "violations": violations,
                         "total_normalized_violation": total if defined else None,
                         "utility": row["utility"] if row["utility"] is not None and math.isfinite(row["utility"]) else None,
                         "strength": row.get("strength", 0)}
    feasible = [k for k, r in assessed.items() if r["feasible"]]
    if feasible:
        selected = min(feasible, key=lambda k: (-assessed[k]["utility"], assessed[k]["strength"]))
    else:
        selected = min(assessed, key=lambda k: (
            assessed[k]["total_normalized_violation"] if assessed[k]["total_normalized_violation"] is not None else float("inf"),
            -assessed[k]["utility"] if assessed[k]["utility"] is not None else float("inf"), assessed[k]["strength"]))
    return {"key": selected, **assessed[selected], "all_candidates": assessed,
            "thresholds": THRESHOLDS, "selection_split": "validation"}


@torch.no_grad()
def encode(encoder, x, purpose=None):
    encoder.eval()
    parts = []
    for start in range(0, len(x), 512):
        tensor = torch.as_tensor(x[start:start+512], dtype=torch.float32)
        h = encoder(tensor) if purpose is None else encoder(tensor, purpose)
        parts.append(h.detach().double().numpy())
    return np.concatenate(parts)


def frozen_views(encoder, data, adapted):
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return {name: [encode(encoder, row["x"], p if adapted else None) for p in range(2)]
            for name, row in data.items()}


def calibrate_and_apply(raw_views, labels, erased):
    """Fit final LEACE only on the calibration split after encoder freezing."""
    erasers = []
    for p in range(2):
        eraser = LeaceEraser.fit(torch.from_numpy(raw_views["calibration"][p]),
                                torch.from_numpy(labels[:, PROHIBITED[p]]), **CONFIG["leace"]) if erased else None
        erasers.append(eraser)
    views = apply_erasers(raw_views, erasers)
    return erasers, views


def apply_erasers(raw_views, erasers):
    result = {}
    for split, representations in raw_views.items():
        h = [e(torch.from_numpy(x)).numpy() if e is not None else x for e, x in zip(erasers, representations)]
        result[split] = h + [np.concatenate(h, axis=1)]
    return result


def fit_method_probes(views, data, seed, out, probe_config):
    from experiments.nonlinear_conflict_probes import fit_probe
    probes = {}
    for role in ("task", "attack"):
        for p in range(2 if role == "task" else 3):
            label = f"p{p+1}" if p < 2 else "combined"
            targets = [p] if role == "task" else [0, 1, 2]
            target_names = [f"{label}_{SIGNALS[j]}" for j in targets]
            fitting = "representation_train" if role == "task" else "attacker_fit"
            for kind in ("linear", "mlp"):
                key = f"{role}_{label}_{kind}"
                print(f"  fitting {out.name}/{key}", flush=True)
                probe = fit_probe(kind, views[fitting][p], data[fitting]["y"][:, targets],
                                  views["validation"][p], data["validation"]["y"][:, targets],
                                  seed=700000 + 100*seed + 10*p + (0 if role == "task" else 50),
                                  artifact_dir=out/"probes"/key, target_names=target_names, config=probe_config)
                probes[key] = {"probe": probe, "role": role, "kind": kind, "view": p, "targets": targets}
    return probes


def evaluate_probes(probes, views, y):
    results = {role: {kind: {} for kind in ("linear", "mlp")} for role in ("task", "attack")}
    for item in probes.values():
        scores = item["probe"].score(views[item["view"]], y[:, item["targets"]])
        results[item["role"]][item["kind"]].update(scores)
    results["covariance"] = {name: empirical_statistics(h, y) for name, h in zip(("p1", "p2", "combined"), views)}
    return results


def selection_row(evaluation, strength):
    leak = {}
    for key in THRESHOLDS:
        values = [evaluation["attack"][kind][key]["r2"] for kind in ("linear", "mlp")]
        leak[key] = max(values) if all(v is not None and math.isfinite(v) for v in values) else None
    tasks = [evaluation["task"]["mlp"][key]["r2"] for key in ("p1_U", "p2_V")]
    utility = float(np.mean(tasks)) if all(v is not None and math.isfinite(v) for v in tasks) else None
    return {"utility": utility, "leakage": leak, "strength": strength}


def run_seed(seed, out, cfg):
    from experiments.nonlinear_conflict_training import pretrain, adapt
    started = time.perf_counter()
    out.mkdir(exist_ok=False)
    data, metadata = make_data(seed)
    dump(out/"split_manifest_fit.json", metadata)
    np.savez_compressed(out/"split_ids_fit.npz", **{name: row["ids"] for name, row in data.items()})
    print(f"seed={seed}: task pretraining", flush=True)
    pre = pretrain(data["representation_train"]["x"], data["representation_train"]["y"][:, :2],
                   data["validation"]["x"], data["validation"]["y"][:, :2],
                   seed=seed, out_dir=out/"pretraining", config=cfg["training"]["pretrain"])
    pretraining_done = time.perf_counter()
    raw = {"seed": seed, "pretraining": pre["metadata"], "methods": {}, "configuration": cfg}
    methods = [("A_frozen_no_erasure", None, False), ("B_frozen_leace", None, True),
               ("C_task_only", 0., True), ("D_protection_0.1", .1, True), ("D_protection_1", 1., True)]
    prepared = {}
    selections = {}
    for method, strength, erased in methods:
        stage_start = time.perf_counter()
        method_out = out/method
        method_out.mkdir()
        adapted = strength is not None
        print(f"seed={seed}: {method}", flush=True)
        trained = adapt(pre, data["representation_train"]["x"], data["representation_train"]["y"],
                        seed=seed, strength=strength, out_dir=method_out/"adaptation",
                        config=cfg["training"]["adaptation"]) if adapted else None
        encoder = trained["encoder"] if adapted else copy.deepcopy(pre["encoder"])
        views_raw = frozen_views(encoder, data, adapted)
        erasers, views = calibrate_and_apply(views_raw, data["calibration"]["y"], erased)
        eraser_arrays = {}
        for p, eraser in enumerate(erasers):
            if eraser is not None:
                eraser_arrays[f"p{p+1}_matrix"] = eraser.P.numpy()
                eraser_arrays[f"p{p+1}_center"] = eraser.bias.numpy()
        np.savez_compressed(method_out/"erasers.npz", **eraser_arrays)
        probes = fit_method_probes(views, data, seed, method_out, cfg["probes"])
        val = evaluate_probes(probes, views["validation"], data["validation"]["y"])
        calibration_stats = {name: empirical_statistics(h, data["calibration"]["y"]) for name, h in zip(("p1", "p2", "combined"), views["calibration"])}
        raw["methods"][method] = {
            "strength": strength, "adaptation": trained["metadata"] if adapted else None,
            "erased": erased, "validation": val, "calibration_covariance": calibration_stats,
            "fit_runtime_seconds": time.perf_counter() - stage_start,
            "probe_metadata": {key: item["probe"].metadata for key, item in probes.items()},
        }
        if adapted:
            selections[method] = selection_row(val, strength)
        prepared[method] = {"encoder": encoder, "adapted": adapted, "erasers": erasers, "probes": probes}
        dump(method_out/"validation.json", raw["methods"][method])
    raw["selection"] = select_configuration(selections)
    raw["positive_selection"] = select_configuration({k: r for k, r in selections.items() if r["strength"] > 0})
    matched = [row["adaptation"] for row in raw["methods"].values() if row["adaptation"] is not None]
    for key in ("initial_state_hash", "pretrained_state_hash", "schedule_hash", "optimizer_steps"):
        if len({row[key] for row in matched}) != 1:
            raise AssertionError(f"C/D matching failed for {key}")
    if matched[0]["optimizer_steps"] != cfg["training"]["adaptation"]["steps"]:
        raise AssertionError("Adaptation optimizer budget does not match frozen configuration")
    selection_record = {"grid": raw["selection"], "positive_only": raw["positive_selection"],
                        "test_generated": False, "elapsed_seconds": time.perf_counter()-started}
    dump(out/"selection_before_test.json", selection_record)
    # Final test is generated only after every model and configuration selection
    # is written to disk. No fitting or parameter changes below this boundary.
    final_started = time.perf_counter()
    test_data, test_manifest = make_data(seed, include_test=True)
    test_data = {"test": test_data["test"]}
    dump(out/"split_manifest_all.json", test_manifest)
    np.savez_compressed(out/"split_ids_test.npz", test=test_data["test"]["ids"])
    for method, record in prepared.items():
        h_raw = frozen_views(record["encoder"], test_data, record["adapted"])
        h = apply_erasers(h_raw, record["erasers"])["test"]
        evaluation = evaluate_probes(record["probes"], h, test_data["test"]["y"])
        raw["methods"][method]["test"] = evaluation
        dump(out/method/"test.json", evaluation)
    raw["pretraining_test"] = {}
    for stage, encoder_key, head_key in (("before", "initial_encoder", "initial_task_head"), ("after", "encoder", "task_head")):
        if encoder_key in pre:
            h = encode(pre[encoder_key], test_data["test"]["x"])
            with torch.no_grad():
                prediction = pre[head_key](torch.from_numpy(h).float()).double().numpy()
            raw["pretraining_test"][stage] = {name: score(test_data["test"]["y"][:, j], prediction[:, j]) for j, name in enumerate(("U", "V"))}
    raw["runtime"] = {"pretraining_seconds": pretraining_done-started,
                      "final_test_seconds": time.perf_counter()-final_started,
                      "total_seconds": time.perf_counter()-started}
    dump(out/"metrics.json", raw)
    print(f"seed={seed} complete: {raw['runtime']['total_seconds']:.2f}s; validation-selected {raw['selection']['key']} feasible={raw['selection']['feasible']}", flush=True)
    return raw


def table_values(method, split="test"):
    row = method[split]
    tasks = [row["task"][kind][key]["r2"] for kind in ("linear", "mlp") for key in ("p1_U", "p2_V")]
    attacks = [max(row["attack"][kind][key]["r2"] for key in THRESHOLDS if key != "combined_S") for kind in ("linear", "mlp")]
    combined = [row["attack"][kind]["combined_S"]["r2"] for kind in ("linear", "mlp")]
    return tasks + attacks + combined


def summarize(out):
    records = [json.loads(p.read_text()) for p in sorted(out.glob("seed_*/metrics.json"))]
    methods = list(records[0]["methods"])
    columns = ["Linear U", "Linear V", "MLP U", "MLP V", "Worst linear leak", "Worst MLP leak", "Combined S linear", "Combined S MLP"]
    rows = []
    for method in methods:
        values = np.array([table_values(seed["methods"][method]) for seed in records])
        rows.append({"method": method, "mean": values.mean(0).tolist(),
                     "std": values.std(0, ddof=1).tolist() if len(values)>1 else [0.]*len(columns),
                     "per_seed": {str(seed["seed"]): values[i].tolist() for i, seed in enumerate(records)}})
    dump(out/"summary.json", {"seeds": [r["seed"] for r in records], "columns": columns, "rows": rows,
                              "runtime_seconds": sum(r["runtime"]["total_seconds"] for r in records)})
    text = ["# Fixed nonlinear upstream-adaptation pilot", "", "Final-test predictive R²; mean ± sample SD. Negative scores are retained and do not denote negative information.", "",
            "| Method | " + " | ".join(columns) + " |", "|---|" + "---:|"*len(columns)]
    for row in rows:
        text.append("| " + row["method"] + " | " + " | ".join(f"{a:.4f} ± {b:.4f}" for a,b in zip(row["mean"], row["std"])) + " |")
    text += ["", "Worst leak = maximum over P1→V,S and P2→U,S within each seed. Combined recipient may access U,V; only S is prohibited.",
             "", "## Per-seed test results", "", "| Seed | Method | " + " | ".join(columns) + " |", "|---|---|" + "---:|"*len(columns)]
    for seed in records:
        for method in methods:
            text.append(f"| {seed['seed']} | {method} | " + " | ".join(f"{x:.5f}" for x in table_values(seed["methods"][method])) + " |")
    text += ["", "## Every forbidden target", "", "| Seed | Method | Attacker | P1→V | P1→S | P2→U | P2→S | Combined→S |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for seed in records:
        for method in methods:
            for kind in ("linear", "mlp"):
                attacks = seed["methods"][method]["test"]["attack"][kind]
                text.append(f"| {seed['seed']} | {method} | {kind} | " + " | ".join(f"{attacks[key]['r2']:.5f}" for key in THRESHOLDS) + " |")
    text += ["", "## Validation selection", ""]
    for seed in records:
        sel, positive = seed["selection"], seed["positive_selection"]
        text.append(f"- Seed {seed['seed']}: full grid **{sel['key']}**, feasible={sel['feasible']}; positive strengths **{positive['key']}**, feasible={positive['feasible']}. Selection recorded before test generation.")
    text += ["", "## Pretraining", "", "Direct task-head test R² before → after task pretraining (selection uses validation only):", ""]
    for seed in records:
        values = seed["pretraining_test"]
        text.append(f"- Seed {seed['seed']}: " + "; ".join(f"{key}: {values['before'][key]['r2']:.5f} → {values['after'][key]['r2']:.5f}" for key in ("U", "V")))
    text += ["", f"Total measured complete-seed computation: {sum(r['runtime']['total_seconds'] for r in records):.2f} seconds.",
             "", "This small pilot compares the recorded attacks and utility only. Near-zero linear R² is not nonlinear privacy, and a failed attack is not a universal guarantee. See PROTOCOL.md for objective, splits, budgets, thresholds, and method limitations."]
    (out/"TABLE.md").write_text("\n".join(text)+"\n")


def main():
    from experiments.nonlinear_conflict_training import TRAIN_CONFIG
    from experiments.nonlinear_conflict_probes import PROBE_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    # Normalize tuples to JSON arrays before equality checks on later invocations.
    cfg = json.loads(json.dumps({**CONFIG, "training": TRAIN_CONFIG, "probes": PROBE_CONFIG}))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"config.json").exists():
        if json.loads((args.out/"config.json").read_text()) != cfg:
            raise ValueError("Protocol configuration changed; use a fresh directory")
    else:
        dump(args.out/"config.json", cfg)
        q1,q2,b = generator_parameters()
        dump(args.out/"generator.json", {"Q1": q1.tolist(), "Q2": q2.tolist(), "offset": b.tolist(), **CONFIG["generator"]})
    sources = ["experiments/run_nonlinear_conflict.py", "experiments/nonlinear_conflict_training.py",
               "experiments/nonlinear_conflict_probes.py", "experiments/run_redesign_conflict.py",
               "pcrl/models/lora.py", "pcrl/training/proxy_lagrangian.py"]
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    if (args.out/"frozen_source_hashes.json").exists():
        if hashes != json.loads((args.out/"frozen_source_hashes.json").read_text()):
            raise ValueError("Executed source changed after protocol freeze; retain artifacts and use a new directory")
    else:
        dump(args.out/"frozen_source_hashes.json", hashes)
        for name in sources:
            path = args.out/"source_snapshot"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes())
    if not (args.out/"PROTOCOL.md").exists():
        raise FileNotFoundError("Write and review PROTOCOL.md before starting any fit")
    provenance = {"starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                  "command": sys.argv, "platform": platform.platform(), "python": sys.version,
                  "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip() if sys.platform == "darwin" else platform.processor(),
                  "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "concept-erasure")},
                  "source_hashes": hashes}
    invocation = "prepare" if args.prepare_only else "_".join(map(str,args.seeds))
    provenance_path = args.out/f"invocation_{invocation}.json"
    if provenance_path.exists():
        raise FileExistsError("Invocation already recorded; do not overwrite existing evidence")
    dump(provenance_path, provenance)
    if args.prepare_only:
        print("Configuration and source frozen; no models fit and no final test generated.")
        return
    for seed in args.seeds:
        if seed not in CONFIG["seeds"]:
            raise ValueError("Seed outside the declared pilot")
        run_seed(seed, args.out/f"seed_{seed}", cfg)
        summarize(args.out)


if __name__ == "__main__":
    main()

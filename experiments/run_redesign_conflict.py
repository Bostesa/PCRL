"""Small, fixed Gaussian purpose-conflict benchmark. No downloads or sweep.

Population: L=(U,V,S,E1,...,E5) ~ N(0,I8), independently per observation.
X=L Q diag(linspace(.7,1.3,8))+b, where Q is QR(seed=20260907) and
b=linspace(-.4,.4,8). This invertible fixed map mixes task/prohibited signals
and nuisance variation. Targets are U for P1, V for P2 (squared error).
P1 prohibits V,S; P2 prohibits U,S; the combined recipient prohibits only S.

LEACE continuous concept matrix interface: https://github.com/EleutherAI/concept-erasure
and https://arxiv.org/abs/2306.03819 section 4.3. Labels are float64 matrices,
not discretized or one-hot encoded. All results are empirical, not certificates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Set before importing numerical libraries; keep this laptop pilot small.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from concept_erasure import LeaceEraser
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ("U", "V", "S")
PROHIBITED = ((1, 2), (0, 2))
RCOND = 1e-10  # Drop numerical residuals of exact rank-deficient erasure.
CONFIG = {
    "seeds": [0, 1, 2], "split_sizes": {"representation_fit": 2048, "attacker_fit": 2048,
                                      "validation": 2048, "test": 4096},
    "dimension": 8, "mixing_seed": 20260907, "dtype": "float64", "device": "cpu",
    "threads": 1, "ols_relative_singular_cutoff": RCOND,
    "leace": {"shrinkage": False, "constrain_cov_trace": False, "svd_tol": 1e-10},
    "adaptation": {"steps": 20, "lr": 1e-3, "rank": 2, "alpha": 2,
                   "privacy_threshold": .01, "dual_lr": .02, "lambda_init": 1.0,
                   "ridge": 1e-6, "weight_decay": 0.0},
    "selection": "One fixed configuration and final iterate per method; no validation or test tuning.",
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mixing():
    q, r = np.linalg.qr(np.random.default_rng(CONFIG["mixing_seed"]).normal(size=(8, 8)))
    q = q * np.sign(np.diag(r))
    return q @ np.diag(np.linspace(.7, 1.3, 8)), np.linspace(-.4, .4, 8)


def make_data(seed):
    matrix, offset = mixing()
    data, split_seeds = {}, {}
    for idx, (name, size) in enumerate(CONFIG["split_sizes"].items()):
        split_seed = 100000 + 100 * seed + idx
        latent = np.random.default_rng(split_seed).normal(size=(size, 8))
        data[name] = (latent @ matrix + offset, latent[:, :3])
        split_seeds[name] = split_seed
    # The only learned preprocessing: center on representation-fitting data.
    center = data["representation_fit"][0].mean(0)
    data = {name: (x - center, y) for name, (x, y) in data.items()}
    return data, split_seeds, center


def fit_affine(x, y):
    """Fit centering, slopes and intercept on this fitting data only."""
    xm, ym = x.mean(0), y.mean(0)
    w = np.linalg.lstsq(x - xm, y - ym, rcond=RCOND)[0]
    return w, ym - xm @ w


def score(y, prediction):
    """Out-of-sample 1-SSE/SST, SST about evaluation target mean. Never clip."""
    sst = float(np.square(y - y.mean()).sum())
    return {"r2": None if sst <= 0 else 1 - float(np.square(y - prediction).sum()) / sst,
            "mse": float(np.square(y - prediction).mean()), "n": len(y),
            "target_variance": float(y.var()), "defined": sst > 0}


def empirical_statistics(x, y):
    """Evaluation-sample covariance/OLS geometry, distinct from held-out prediction."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    cov = xc.T @ yc / len(x)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    keep = s > (s[0] * RCOND) if len(s) and s[0] > 0 else np.zeros(len(s), bool)
    fitted = u[:, keep] @ (u[:, keep].T @ yc)
    denom = np.square(yc).sum(0)
    return {"rank": int(keep.sum()), "n": len(x),
            "cross_covariance": cov.tolist(),
            "max_abs_covariance_by_signal": np.abs(cov).max(0).tolist(),
            "in_sample_ols_r2_by_signal": (np.square(fitted).sum(0) / denom).tolist(),
            "target_variance_by_signal": y.var(0).tolist(),
            "class_support": "not applicable: continuous Gaussian targets"}


def continuous_r2(h, z):
    hc, zc = h - h.mean(0), z - z.mean(0)
    w = torch.linalg.solve(hc.T @ hc + CONFIG["adaptation"]["ridge"] * torch.eye(h.shape[1], dtype=h.dtype), hc.T @ zc)
    return (1 - (zc - hc @ w).square().sum() / zc.square().sum()).clamp_min(0)


def adapt(erased, labels, seed, out):
    """Use existing PCRL post-erasure LoRA and proxy-dual primitives.

    A deliberately restricted continuous-task adapter experiment, not a claim
    that the categorical V2 trainer has been validated for regression. Gradients
    and dual updates use representation-fit only. No VICReg/vCLUB or search.
    Initial ordinary affine heads are OLS, and final heads are refit by exactly
    the same OLS rule used by the baselines. A linear invertible post-map cannot
    improve optimal affine prediction, so a tie is an expected control.
    """
    torch.manual_seed(200000 + seed)
    backbone = torch.nn.Linear(8, 8, bias=False, dtype=torch.float64)
    with torch.no_grad():
        backbone.weight.copy_(torch.eye(8, dtype=torch.float64))
    encoder = PerPurposeLoRAEncoder(backbone, 2, rank=2, alpha=2).double()
    heads = torch.nn.ModuleList([torch.nn.Linear(8, 1, dtype=torch.float64) for _ in range(2)])
    for p in range(2):
        w, b = fit_affine(erased[p], labels[:, p])
        with torch.no_grad():
            heads[p].weight.copy_(torch.from_numpy(w[None]))
            heads[p].bias.fill_(float(b))
    xs, ys = [torch.from_numpy(x) for x in erased], torch.from_numpy(labels)
    cfg = CONFIG["adaptation"]
    optimizer = torch.optim.Adam(list(encoder.trainable_parameters()) + list(heads.parameters()), lr=cfg["lr"])
    constraints = [Constraint(f"p{p}_{a}", cfg["privacy_threshold"], eta_lambda=cfg["dual_lr"])
                   for p in range(2) for a in PROHIBITED[p]]
    constraints.append(Constraint("combined_S", cfg["privacy_threshold"], eta_lambda=cfg["dual_lr"]))
    proxy = ProxyLagrangianOptimizer(optimizer, constraints)
    initial = copy.deepcopy(encoder.state_dict())
    history = []
    for step in range(cfg["steps"] + 1):
        hs = [encoder(xs[p], p) for p in range(2)]
        task = sum((heads[p](hs[p]).squeeze(1) - ys[:, p]).square().mean() for p in range(2)) / 2
        values = {f"p{p}_{a}": continuous_r2(hs[p], ys[:, a]) for p in range(2) for a in PROHIBITED[p]}
        values["combined_S"] = continuous_r2(torch.cat(hs, dim=1), ys[:, 2])
        history.append({"optimizer_steps": step, "fit_task_mse": float(task.detach()),
                        "fit_constraint_r2": {k: float(v.detach()) for k, v in values.items()}})
        if step in (0, cfg["steps"]):
            torch.save({"optimizer_steps": step, "encoder": encoder.state_dict(), "heads": heads.state_dict(),
                        "optimizer": optimizer.state_dict(), "initialization": "fixed identity post-LEACE; no pretraining",
                        "duals": proxy.diagnostics()}, out / f"adapter_step_{step}.pt")
        if step == cfg["steps"]:
            break
        optimizer.zero_grad()
        proxy.lagrangian_loss(task, values).backward()
        proxy.primal_step()
        proxy.dual_step({k: float(v.detach()) for k, v in values.items()})
    encoder.eval()
    change = sum(float((encoder.state_dict()[k] - v).square().sum()) for k, v in initial.items()) ** .5
    dump(out / "adaptation.json", {"history": history, "parameter_change_l2": change,
                                  "limitation": adapt.__doc__, "duals": proxy.diagnostics()})
    return encoder


def composition_stress():
    """Analytical population covariance and independent fit/test predictions."""
    rng = np.random.default_rng(30907)
    fit, test = rng.normal(size=(8192, 2)), rng.normal(size=(32768, 2))
    rows = []
    for delta in (0., .01, .1, 1.):
        c = np.array([delta, -delta])
        sigma = np.array([[1 + delta**2, 1 - delta**2], [1 - delta**2, 1 + delta**2]])
        expected = delta**2 / (1 + delta**2)
        joint = float(c @ np.linalg.pinv(sigma, rcond=1e-12) @ c)
        views = lambda a: np.column_stack((a[:, 0] + delta*a[:, 1], a[:, 0] - delta*a[:, 1]))
        hf, ht = views(fit), views(test)
        measured = []
        for columns in ([0], [1], [0, 1]):
            w, b = fit_affine(hf[:, columns], fit[:, 1])
            measured.append(score(test[:, 1], ht[:, columns] @ w + b))
        assert np.isclose(joint, float(delta != 0), atol=1e-9)
        if delta:
            assert np.allclose((ht[:, 0] - ht[:, 1]) / (2*delta), test[:, 1])
        rows.append({"delta": delta, "population_single_r2": expected,
                     "population_combined_r2": joint, "predictive_scores_views_then_concat": measured})
    return {"label": "Approximate leakage amplification; not a failure of exact zero-covariance composition",
            "distribution": "N,S independent centered unit-variance Gaussian; h1=N+delta*S, h2=N-delta*S",
            "recovery": "S=(h1-h2)/(2*delta) for delta != 0", "seed": 30907, "rows": rows}


def run_seed(seed, directory):
    started = time.perf_counter()
    out = directory / f"seed_{seed}"
    out.mkdir(exist_ok=False)
    data, seeds, center = make_data(seed)
    xf, yf = data["representation_fit"]
    union = LeaceEraser.fit(torch.from_numpy(xf), torch.from_numpy(yf), **CONFIG["leace"])
    independent = [LeaceEraser.fit(torch.from_numpy(xf), torch.from_numpy(yf[:, cols]), **CONFIG["leace"])
                   for cols in PROHIBITED]
    encoder = adapt([e(torch.from_numpy(xf)).numpy() for e in independent], yf, seed, out)
    erasers = {"shared_union": [union, union], "per_purpose_leace": independent,
               "pcrl_post_erase_lora": independent}
    raw = {"seed": seed, "split_seeds": seeds, "split_sizes": CONFIG["split_sizes"],
           "preprocessing_center": center.tolist(), "methods": {}}
    saved = {"center": center}
    with torch.no_grad():
        for method in ("no_erasure", *erasers):
            representations = {}
            for split, (x, _) in data.items():
                views = [x.copy(), x.copy()] if method == "no_erasure" else [e(torch.from_numpy(x)).numpy() for e in erasers[method]]
                if method == "pcrl_post_erase_lora":
                    views = [encoder(torch.from_numpy(v), p).numpy() for p, v in enumerate(views)]
                representations[split] = views + [np.concatenate(views, axis=1)]
            results = {"output_dimensions": [8, 8, 16], "validation": {}, "test": {}, "fit_covariance": {}}
            for p in range(3):
                fit_h = representations["representation_fit"][p]
                attack_h = representations["attacker_fit"][p]
                w_a, b_a = fit_affine(attack_h, data["attacker_fit"][1])
                saved[f"{method}_view{p}_attacker_w"] = w_a
                saved[f"{method}_view{p}_attacker_b"] = b_a
                if p < 2:
                    w_t, b_t = fit_affine(fit_h, yf[:, p])
                    saved[f"{method}_view{p}_task_w"] = w_t
                    saved[f"{method}_view{p}_task_b"] = b_t
                results["fit_covariance"][str(p)] = empirical_statistics(fit_h, yf)
                for split in ("validation", "test"):
                    h, y = representations[split][p], data[split][1]
                    preds = h @ w_a + b_a
                    row = {"attacker": {a: score(y[:, j], preds[:, j]) for j, a in enumerate(SIGNALS)},
                           "empirical_covariance": empirical_statistics(h, y),
                           "prohibited": [SIGNALS[a] for a in PROHIBITED[p]] if p < 2 else ["S"]}
                    if p < 2:
                        row["task"] = score(y[:, p], h @ w_t + b_t)
                    results[split][str(p)] = row
            raw["methods"][method] = results
    for name, es in erasers.items():
        for p, e in enumerate(es):
            saved[f"{name}_p{p}_projection"] = e.P.numpy()
            saved[f"{name}_p{p}_center"] = e.bias.numpy()
    np.savez_compressed(out / "fitted_arrays.npz", **saved)
    raw["runtime_seconds"] = time.perf_counter() - started
    dump(out / "metrics.json", raw)
    return raw


def summarize(directory):
    seeds = [json.loads(p.read_text()) for p in sorted(directory.glob("seed_*/metrics.json"))]
    rows = []
    for method in seeds[0]["methods"]:
        vals = []
        for seed in seeds:
            m = seed["methods"][method]["test"]
            vals.append([m["0"]["task"]["r2"], m["1"]["task"]["r2"],
                         max(m[str(p)]["attacker"][SIGNALS[a]]["r2"] for p in range(2) for a in PROHIBITED[p]),
                         m["2"]["attacker"]["S"]["r2"]])
        arr = np.array(vals)
        rows.append({"method": method, "mean": arr.mean(0).tolist(),
                     "std": arr.std(0, ddof=1).tolist() if len(arr)>1 else [0.]*4})
    dump(directory / "summary.json", {"seeds": [s["seed"] for s in seeds],
         "columns": ["P1_U_task_r2", "P2_V_task_r2", "worst_individual_prohibited_r2", "combined_S_r2"],
         "rows": rows, "total_seed_runtime_seconds": sum(s["runtime_seconds"] for s in seeds)})
    text = ["# Gaussian conflict pilot", "", "Final-test predictive R², mean ± sample SD across seeds. Negative scores are retained.", "",
            "| Method | P1 task U | P2 task V | Worst individual prohibited | Combined S |",
            "|---|---:|---:|---:|---:|"]
    for row in rows:
        text.append("| " + row["method"] + " | " + " | ".join(f"{a:.6f} ± {b:.6f}" for a,b in zip(row["mean"],row["std"])) + " |")
    text.extend(["", "Worst individual is the maximum of P1→V,S and P2→U,S within each seed. Combined U,V are authorized and recorded separately in metrics.json.",
                 "Empirical fitting/evaluation covariance matrices, ranks, and in-sample OLS geometry are recorded separately from these attacker-fit → test predictive scores.",
                 "", "PCRL here reuses the existing post-erasure LoRA and proxy-dual primitives with continuous squared-error tasks; it does not validate the full categorical V2 trainer. Linear post-erasure adaptation plus optimal affine heads is expected to tie ordinary per-purpose LEACE."])
    (directory / "TABLE.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "config.json").exists():
        dump(args.out / "config.json", CONFIG)
        matrix, offset = mixing()
        dump(args.out / "mixing.json", {"matrix": matrix.tolist(), "offset": offset.tolist(), "description": __doc__})
        dump(args.out / "composition_stress.json", composition_stress())
    elif json.loads((args.out / "config.json").read_text()) != CONFIG:
        raise ValueError("Existing run has different configuration; use a fresh directory")
    files = [Path(__file__), ROOT/"pcrl/models/lora.py", ROOT/"pcrl/training/proxy_lagrangian.py"]
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                  "platform": platform.platform(), "cpu": "Apple M4 Pro; 14 cores; 24 GiB RAM",
                  "python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "concept-erasure")},
                  "command": sys.argv, "code_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    stamp = "_".join(map(str,args.seeds))
    dump(args.out / f"provenance_seeds_{stamp}.json", provenance)
    for seed in args.seeds:
        result = run_seed(seed, args.out)
        print(f"seed={seed} runtime={result['runtime_seconds']:.3f}s", flush=True)
        summarize(args.out)


if __name__ == "__main__":
    main()

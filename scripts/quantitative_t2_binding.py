#!/usr/bin/env python3
"""Numerical verification of the d_TV pair-witness corollary on PCRL data.

Computes, for each (dataset, seed) in {Adult-R5, HMDA-R5, Diabetes-R7} × {0,1,2}:
  γ_i := max_{a,a'} min{p_a, p_{a'}} · d_TV(P(Y_i|A=a), P(Y_i|A=a'))   [from raw labels]
  δ_i := max_{a,a'} d_TV(P(Ŷ_i|A=a), P(Ŷ_i|A=a'))                    [from PCRL preds]
  Err_i := 1 − test-accuracy of purpose i's primary task                [from PCRL preds]
  bound := Σ γ_i − Σ δ_i  (corollary form, see corollary.tex)
  observed := Σ Err_i

Output:
  results/quantitative_t2_binding/bound_numerical_results.json

Design:
- A := race, the only sensitive attribute disallowed in all 3 purposes
  for all 3 datasets. Multi-class (5 categories) — the witness pair is
  the (a,a*) pair achieving the max.
- Predictions: argmax of task-head output (Ŷ_i is the class label, not
  a probability vector). Matches Err_i = P(Ŷ_i ≠ Y_i) and gives a
  well-defined d_TV between empirical conditional distributions.
- Test loader, batch_size 512, num_workers 0 (deterministic).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402

warnings.filterwarnings("ignore")

LORA_BY_DATASET: dict[str, tuple[int, float]] = {
    "adult": (8, 16.0),
    "hmda": (8, 16.0),
    "diabetes": (24, 48.0),
}

DATASETS = [
    ("adult", "ROUND5"),
    ("hmda", "ROUND5"),
    ("diabetes", "ROUND7"),
]
SEEDS = [0, 1, 2]
ATTR = "race"

# Binarize race per FFB ICLR 2024 / PCRL convention. The "majority" class is
# 4 in the AdultDataset / HMDADataset / DiabetesDataset encodings (= White,
# Non-Hispanic). Mapping all other classes to 0 increases min{p_a, p_{a'}}
# and gives the bound a fair shot on imbalanced multi-class race.
RACE_MAJORITY_CLASS = 4
USE_BINARY_RACE = False

OUT_DIR = ROOT / "results" / "quantitative_t2_binding"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── dataset construction ───────────────────────────────────────────────
def build(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(
            purposes=purposes, root=str(ROOT / "data"), split="train", download=False,
        )
        test_ds = AdultDataset(
            purposes=purposes, root=str(ROOT / "data"), split="test", download=False,
            norm_stats=train_ds.norm_stats,
        )
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="train")
        test_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="test")
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    else:
        raise ValueError(name)
    return purposes, train_ds, test_ds


def build_model(input_dim: int, n_purposes: int, dataset: str):
    repr_dim = 64
    rank, alpha = LORA_BY_DATASET[dataset]
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=repr_dim, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=n_purposes, rank=rank, alpha=alpha, dropout=0.0,
    )
    return encoder, repr_dim


def load_checkpoint(encoder, task_heads_dict, ckpt_path, n_purposes, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(n_purposes):
        P_key = f"leace_P_p{p_idx}"
        mu_key = f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
    th_mod = nn.ModuleDict(task_heads_dict)
    th_mod.load_state_dict(ckpt["task_heads"])
    return {n: th_mod[n] for n in task_heads_dict}


# ── inference ─────────────────────────────────────────────────────────
@torch.no_grad()
def inference(purposes, encoder, task_heads, loader, device="cpu"):
    encoder.eval()
    for h in task_heads.values():
        h.eval()
    out = {p.name: {"y_hat": [], "y_true": [], "attr": []} for p in purposes}
    for batch in loader:
        feats = batch["features"].to(device)
        for idx, p in enumerate(purposes):
            z = encoder(feats, idx)
            preds = task_heads[p.name](z)
            if isinstance(preds, dict):
                preds = preds[p.allowed_tasks[0]]
            y_hat = preds.argmax(dim=-1).cpu().numpy()
            y_true = batch["task_labels"][p.allowed_tasks[0]].numpy()
            attr = batch["sensitive_attrs"][ATTR].numpy()
            if USE_BINARY_RACE:
                attr = (attr == RACE_MAJORITY_CLASS).astype(np.int64)
            out[p.name]["y_hat"].append(y_hat)
            out[p.name]["y_true"].append(y_true)
            out[p.name]["attr"].append(attr)
    for p in purposes:
        for k in ("y_hat", "y_true", "attr"):
            out[p.name][k] = np.concatenate(out[p.name][k]).astype(np.int64)
    return out


# ── d_TV pair-witness ─────────────────────────────────────────────────
def empirical_dist(values: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(values, minlength=n_classes).astype(float)
    s = counts.sum()
    return counts / s if s > 0 else counts


def dtv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def _per_class_dists(values: np.ndarray, attr: np.ndarray, n_y: int, n_a: int):
    p_a = np.bincount(attr, minlength=n_a).astype(float) / max(len(attr), 1)
    dists = {}
    for a in range(n_a):
        m = attr == a
        if m.sum() == 0:
            continue
        dists[a] = empirical_dist(values[m], n_y)
    return p_a, dists


def loose_gamma(y_true, attr, n_y, n_a):
    """γ_i^loose = max_{a,a'} min(p_a, p_{a'}) · d_TV(P(Y|a), P(Y|a'))."""
    p_a, dists = _per_class_dists(y_true, attr, n_y, n_a)
    best, witness = 0.0, None
    for a in dists:
        for ap in dists:
            if ap <= a:
                continue
            tv = dtv(dists[a], dists[ap])
            w = min(p_a[a], p_a[ap]) * tv
            if w > best:
                best = w
                witness = {
                    "a": int(a), "a_prime": int(ap),
                    "p_a": float(p_a[a]), "p_a_prime": float(p_a[ap]),
                    "min_p": float(min(p_a[a], p_a[ap])),
                    "dtv_Y": float(tv),
                    "weighted": float(w),
                }
    return float(best), witness


def loose_delta(y_hat, attr, n_y, n_a):
    """δ_i^loose = max_{a,a'} d_TV(P(Ŷ|a), P(Ŷ|a')). NOT pair-locked."""
    _, dists = _per_class_dists(y_hat, attr, n_y, n_a)
    best, witness = 0.0, None
    for a in dists:
        for ap in dists:
            if ap <= a:
                continue
            tv = dtv(dists[a], dists[ap])
            if tv > best:
                best = tv
                witness = {"a": int(a), "a_prime": int(ap), "dtv_Yhat": float(tv)}
    return float(best), witness


def tight_pair_bound(y_true, y_hat, attr, n_y, n_a):
    """Tight per-purpose bound:
        Err_i ≥ max_{a,a'} min{p_a, p_{a'}} · ( d_TV(Y|a,a') - d_TV(Ŷ|a,a') )
    Both terms evaluated at the SAME witness pair. Honest version of the
    Madras-style coupling argument: the proof produces this max-of-difference
    form, and only relaxing to "max γ - max δ" gives the looser version.
    """
    p_a, dY = _per_class_dists(y_true, attr, n_y, n_a)
    _, dYh = _per_class_dists(y_hat, attr, n_y, n_a)
    best, witness = -np.inf, None
    for a in dY:
        for ap in dY:
            if ap <= a:
                continue
            mp = min(p_a[a], p_a[ap])
            dY_v = dtv(dY[a], dY[ap])
            dYh_v = dtv(dYh[a], dYh[ap])
            value = mp * (dY_v - dYh_v)
            if value > best:
                best = value
                witness = {
                    "a": int(a), "a_prime": int(ap),
                    "p_a": float(p_a[a]), "p_a_prime": float(p_a[ap]),
                    "min_p": float(mp),
                    "dtv_Y": float(dY_v),
                    "dtv_Yhat": float(dYh_v),
                    "contribution": float(value),
                }
    return float(best), witness


def err(y_hat, y_true) -> float:
    return float((y_hat != y_true).mean())


# ── main ───────────────────────────────────────────────────────────────
def main():
    device = "cpu"
    all_results = {"per_dataset_seed": [], "summary": []}
    for dataset, round_tag in DATASETS:
        print(f"\n=== {dataset} {round_tag} ===")
        purposes, train_ds, test_ds = build(dataset)
        test_loader = DataLoader(
            test_ds, batch_size=512, shuffle=False,
            collate_fn=collate_pcrl_batch, num_workers=0,
        )
        n_a = 2 if USE_BINARY_RACE else test_ds.info.sensitive_attrs[ATTR]
        input_dim = train_ds.info.num_features
        n_purposes = len(purposes)
        n_y_per = {p.name: p.allowed_task_dims[p.allowed_tasks[0]] for p in purposes}

        # γ_i is a property of the test data + attribute (no model).
        # Compute once per dataset.
        # We pull a single inference pass anyway (need attr arrays); use
        # those y_true arrays for gamma. They are identical across seeds.

        for seed in SEEDS:
            ckpt_path = ROOT / "checkpoints" / f"v2_{dataset}_{round_tag}_s{seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"  seed={seed} MISSING {ckpt_path}")
                continue
            encoder, repr_dim = build_model(input_dim, n_purposes, dataset)
            task_heads = {}
            for p in purposes:
                t_name = p.allowed_tasks[0]
                task_heads[p.name] = TaskHead(
                    repr_dim=repr_dim, output_dim=p.allowed_task_dims[t_name],
                )
            task_heads = load_checkpoint(encoder, task_heads, ckpt_path, n_purposes, device=device)
            encoder.eval()
            preds = inference(purposes, encoder, task_heads, test_loader, device=device)

            per_purpose = []
            sum_gamma_loose = 0.0
            sum_delta_loose = 0.0
            sum_tight = 0.0
            sum_err = 0.0
            for p in purposes:
                pdata = preds[p.name]
                n_y = n_y_per[p.name]
                gamma_i, gw = loose_gamma(
                    pdata["y_true"], pdata["attr"], n_y, n_a,
                )
                delta_i, dw = loose_delta(
                    pdata["y_hat"], pdata["attr"], n_y, n_a,
                )
                tight_i, tw = tight_pair_bound(
                    pdata["y_true"], pdata["y_hat"],
                    pdata["attr"], n_y, n_a,
                )
                err_i = err(pdata["y_hat"], pdata["y_true"])
                per_purpose.append({
                    "purpose": p.name,
                    "task": p.allowed_tasks[0],
                    "n_classes_y": n_y,
                    "n_classes_a": n_a,
                    "gamma_loose": gamma_i,
                    "gamma_loose_witness": gw,
                    "delta_loose": delta_i,
                    "delta_loose_witness": dw,
                    "tight_pair_bound": tight_i,
                    "tight_pair_witness": tw,
                    "err": err_i,
                })
                sum_gamma_loose += gamma_i
                sum_delta_loose += delta_i
                sum_tight += tight_i
                sum_err += err_i

            bound_loose = sum_gamma_loose - sum_delta_loose
            ratio_loose = sum_err / bound_loose if bound_loose > 0 else float("nan")
            ratio_tight = sum_err / sum_tight if sum_tight > 0 else float("nan")
            row = {
                "dataset": dataset,
                "round": round_tag,
                "seed": seed,
                "attr": ATTR,
                "n_test": int(len(preds[purposes[0].name]["y_true"])),
                "per_purpose": per_purpose,
                "sum_gamma_loose": sum_gamma_loose,
                "sum_delta_loose": sum_delta_loose,
                "bound_loose": bound_loose,
                "ratio_loose": ratio_loose,
                "sum_tight_bound": sum_tight,
                "ratio_tight": ratio_tight,
                "sum_err": sum_err,
                "kgamma_max": n_purposes * max(p["gamma_loose"] for p in per_purpose),
                "bound_kgamma_max": (
                    n_purposes * max(p["gamma_loose"] for p in per_purpose) - sum_delta_loose
                ),
            }
            all_results["per_dataset_seed"].append(row)
            print(
                f"  seed={seed}  "
                f"Σγ={sum_gamma_loose:.4f}  Σδ={sum_delta_loose:.4f}  "
                f"loose={bound_loose:+.4f}  tight={sum_tight:+.4f}  "
                f"ΣErr={sum_err:.4f}  "
                f"r_tight={ratio_tight:.2f}"
            )

    # Aggregate: per-(dataset, round) mean across seeds
    by_ds: dict[str, list] = {}
    for r in all_results["per_dataset_seed"]:
        by_ds.setdefault(r["dataset"], []).append(r)
    for dataset, rows in by_ds.items():
        if not rows:
            continue
        gammas = [r["sum_gamma_loose"] for r in rows]
        deltas = [r["sum_delta_loose"] for r in rows]
        bounds_loose = [r["bound_loose"] for r in rows]
        bounds_tight = [r["sum_tight_bound"] for r in rows]
        errs = [r["sum_err"] for r in rows]
        ratios_t = [r["ratio_tight"] for r in rows
                    if r["sum_tight_bound"] > 0
                    and not np.isnan(r["ratio_tight"])]
        all_results["summary"].append({
            "dataset": dataset,
            "round": rows[0]["round"],
            "n_seeds": len(rows),
            "sum_gamma_loose_mean": float(np.mean(gammas)),
            "sum_delta_loose_mean": float(np.mean(deltas)),
            "bound_loose_mean": float(np.mean(bounds_loose)),
            "sum_tight_bound_mean": float(np.mean(bounds_tight)),
            "sum_tight_bound_std": float(np.std(bounds_tight)),
            "sum_err_mean": float(np.mean(errs)),
            "sum_err_std": float(np.std(errs)),
            "ratio_tight_mean": float(np.mean(ratios_t)) if ratios_t else float("nan"),
            "ratio_tight_std": float(np.std(ratios_t)) if ratios_t else float("nan"),
        })

    out_path = OUT_DIR / "bound_numerical_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

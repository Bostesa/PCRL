#!/usr/bin/env python3
"""Fixture C - is the 'cleanly compliant' health bar invariant to harmless rescaling of h? (synthetic, deterministic)

Health definitions copied (re-implemented, not imported) from run_v2_dataset.py@b158abc54:150-175:
  per_dim_std_mean = mean_d std(h[:, d])            (population std, ddof=0)
  effective_rank   = exp(entropy(s^2 / sum s^2)),   s = singular values of column-centred h
Leakage measures:
  R2_onehot   = in-sample OLS R2 of one-hot A on [h, 1] (variance-weighted over columns, as PCRL's certificate)
  AUC_linear  = held-out AUC of an unregularised least-squares discriminant (fit on train, scored on test)
  AUC_knn     = held-out AUC of a 25-NN classifier (Euclidean; nonlinear)
Transforms of a fixed synthetic representation h (n=4000, d=16, binary A with nonlinear + linear signal):
  isotropic scale c * h for c in {0.25, 0.5, 1, 1.25, 1.5, 2, 10};  anisotropic diag scale;  invertible rotation.
Expectation (algebra): R2 and both AUCs unchanged under every isotropic scale; std scales by c; eff_rank
unchanged under isotropic scale but NOT under anisotropic scaling. Also reported: the scale factor that lifts
the stored erase-pilot per-purpose stds above 0.5 (read from the recount output if present).
Run: python3 C_scale_invariance.py > outputs/C_scale_invariance.json   (~2 s)
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(20261001)
n, d = 4000, 16
A = rng.integers(0, 2, n)
Z = rng.normal(size=(n, d)) * np.linspace(0.05, 0.45, d)          # small-scale, anisotropic like the pilot (std ~0.2-0.4)
Z[:, 0] += 0.04 * (A - 0.5)                                        # weak linear signal
Z[:, 1] += 0.25 * np.where(A == 1, np.abs(rng.normal(size=n)), -np.abs(rng.normal(size=n))) * np.sign(Z[:, 2])  # nonlinear (XOR-like) signal
tr, te = np.arange(n) < 3000, np.arange(n) >= 3000


def per_dim_std_mean(h):
    return float(h.std(axis=0).mean())


def eff_rank(h):
    s = np.linalg.svd(h - h.mean(0, keepdims=True), compute_uv=False)
    p = s ** 2 / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def r2_onehot(h, a):
    Y = np.eye(2)[a]
    X = np.c_[h, np.ones(len(h))]
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ B
    return float(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def auc_linear(h, a):
    X = np.c_[h, np.ones(len(h))]
    w, *_ = np.linalg.lstsq(X[tr], a[tr].astype(float), rcond=None)
    return float(roc_auc_score(a[te], X[te] @ w))


def auc_knn(h, a):
    m = KNeighborsClassifier(n_neighbors=25).fit(h[tr], a[tr])
    return float(roc_auc_score(a[te], m.predict_proba(h[te])[:, 1]))


def row(name, h):
    return {"transform": name, "per_dim_std_mean": round(per_dim_std_mean(h), 6),
            "passes_std_0.5": per_dim_std_mean(h) >= 0.5, "eff_rank": round(eff_rank(h), 6),
            "R2_onehot": round(r2_onehot(h, A), 10), "AUC_linear_heldout": round(auc_linear(h, A), 10),
            "AUC_knn_heldout": round(auc_knn(h, A), 10)}


rows = [row(f"isotropic x{c}", c * Z) for c in [0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 10.0]]
D = np.diag(np.linspace(0.5, 3.0, d))
rows.append(row("anisotropic diag(0.5..3.0)", Z @ D))
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
rows.append(row("orthogonal rotation", Z @ Q))
iso = [r for r in rows if r["transform"].startswith("isotropic")]
spread = {k: max(r[k] for r in iso) - min(r[k] for r in iso) for k in ["R2_onehot", "AUC_linear_heldout", "AUC_knn_heldout", "eff_rank"]}
out = {"rows": rows,
       "max_spread_across_isotropic_scales": spread,
       "std_ratio_x2_over_x1": rows[5]["per_dim_std_mean"] / rows[2]["per_dim_std_mean"],
       "verdict": {
           "R2_and_AUCs_invariant_to_isotropic_scale": all(v < 1e-6 for k, v in spread.items() if k != "eff_rank"),
           "eff_rank_invariant_to_isotropic_scale": spread["eff_rank"] < 1e-6,
           "eff_rank_changes_under_anisotropic_scale": abs(rows[7]["eff_rank"] - rows[2]["eff_rank"]) > 1e-3,
           "std_threshold_flips_under_isotropic_scale": len({r["passes_std_0.5"] for r in iso}) == 2}}
p = os.path.join(os.path.dirname(__file__), "..", "notes", "verification", "recount_outputs", "C_erase_vicreg.json")
if os.path.exists(p):
    C = json.load(open(p))
    out["stored_runs_uniform_scale_needed_for_every_purpose_std_ge_0.5"] = {
        run: {ds: e["std_scale_needed_for_all_purposes_to_reach_0.5"] for ds, e in R["per_dataset"].items()}
        for run, R in C["runs"].items()}
    out["implication"] = ("Multiplying every stored erase-pilot / VICReg5 / rank-8 h_p by the listed factor (1.12-1.42) "
                          "would make all strict-pass cells 'cleanly compliant' (eff_rank already >= 10) with R2, "
                          "auditor accuracy and task accuracy (linear heads absorb the scale) unchanged.")
print(json.dumps(out, indent=1))

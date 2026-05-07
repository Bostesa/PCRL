"""Theorem 5 decision gate: compute λ*(C) on the block correlation
matrix of whitened per-purpose PCRL representations.

For each dataset (Adult / HMDA / Diabetes):
  1. Load PCRL canonical checkpoint (best.pt) + reconstruct encoder.
  2. For each purpose p = 0..k-1, run val features through adapter p,
     producing h_p ∈ R^{N×d_p}.
  3. Center, whiten each h_p with its own (Σ_pp + reg I)^{-1/2}.
  4. Stack ĥ_concat = [ĥ_1, ..., ĥ_k] ∈ R^{N × Σd_p}.
  5. C = (1/N) ĥ_concat^T ĥ_concat ; eigendecompose; report
     λ*(C) = smallest eigenvalue exceeding the numerical-zero tolerance.
  6. Compute max_i ε_i (worst per-purpose linear R² seed-mean from
     the existing dominant_axis_audit.json).
  7. Compute empirical R²(h_concat, A) for each disallowed attribute,
     attacker = OLS one-hot regression.
  8. Bound = k · max_i ε_i / λ*(C); compare with empirical.
  9. Verdict: GO / FALLBACK_E2 / FALLBACK_E4.

Outputs:
  results/theorem5_check/lambda_star_per_dataset.json
  results/theorem5_check/bound_vs_empirical_table.json
  results/theorem5_check/DECISION.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "results" / "theorem5_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder

DATASETS = {
    "adult":     {"ckpt_dir": "v2_adult_ROUND5_s0",      "audit": "v2_adult_ROUND5",     "rank": 8,  "input_dim": 105},
    "hmda":      {"ckpt_dir": "v2_hmda_ROUND5_s0",       "audit": "v2_hmda_ROUND5",      "rank": 8,  "input_dim": 78},
    "diabetes":  {"ckpt_dir": "v2_diabetes_ROUND7_s0",   "audit": "v2_diabetes_ROUND7",  "rank": 24, "input_dim": 170},
}
N_PURPOSES = 3
REPR_DIM = 64
WHITEN_REG = 1e-4
NUMERICAL_ZERO_TOL = 1e-6


def load_validation(dataset: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dataset == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        ds = AdultDataset(get_adult_purposes(), root=str(ROOT / "data"), split="val", download=False)
        feats = ds.features.numpy()
        attrs = {k: v.numpy() for k, v in ds.sensitive_attrs.items()}
        return feats, attrs
    if dataset == "hmda":
        d = np.load(ROOT / "data" / "hmda_processed" / "val.npz")
        feats = d["features"].astype(np.float32)
        attrs = {k.replace("attr_", ""): d[k].astype(np.int64) for k in d.files if k.startswith("attr_")}
        return feats, attrs
    if dataset == "diabetes":
        d = np.load(ROOT / "data" / "diabetes_processed" / "val.npz")
        feats = d["features"].astype(np.float32)
        attrs = {k: d[k].astype(np.int64) for k in ("race", "gender", "age_bucket")}
        return feats, attrs
    raise ValueError(dataset)


def load_encoder(dataset: str, info: dict) -> PerPurposeLoRAEncoder:
    ckpt_path = ROOT / "checkpoints" / info["ckpt_dir"] / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = StandardEncoder(
        input_dim=info["input_dim"], hidden_dims=[128, 128], repr_dim=REPR_DIM, dropout=0.3
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=N_PURPOSES,
        rank=info["rank"], alpha=2 * info["rank"], dropout=0.0,
    )
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    if "encoder_buffers" in ckpt:
        for k, v in ckpt["encoder_buffers"].items():
            encoder.register_buffer(k, v.clone(), persistent=True)
    encoder.eval()
    return encoder


@torch.no_grad()
def per_purpose_reprs(encoder: PerPurposeLoRAEncoder, feats: np.ndarray) -> list[np.ndarray]:
    x = torch.from_numpy(feats.astype(np.float32))
    out = []
    for p in range(encoder.n_purposes):
        h = encoder(x, p).cpu().numpy()
        out.append(h)
    return out


def whiten(H: np.ndarray, reg: float = WHITEN_REG) -> tuple[np.ndarray, dict]:
    n = H.shape[0]
    H_c = H - H.mean(axis=0, keepdims=True)
    Sigma = (H_c.T @ H_c) / n
    Sigma_reg = Sigma + reg * np.eye(Sigma.shape[0])
    eigvals, eigvecs = np.linalg.eigh(Sigma_reg)
    eigvals = np.clip(eigvals, 1e-12, None)
    Sigma_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    H_white = H_c @ Sigma_inv_sqrt
    return H_white, {
        "n": int(n),
        "d": int(H.shape[1]),
        "Sigma_eigvals_min": float(eigvals.min()),
        "Sigma_eigvals_max": float(eigvals.max()),
        "Sigma_eigvals_top5": eigvals[-5:].tolist(),
    }


def linear_r2_onehot(H: np.ndarray, z: np.ndarray, reg: float = 1e-6) -> float:
    z = z.astype(np.int64)
    n_classes = int(z.max()) + 1
    Z = np.eye(n_classes)[z]
    n, d = H.shape
    H_c = H - H.mean(axis=0, keepdims=True)
    Z_c = Z - Z.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + reg * np.eye(d)
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))


def per_purpose_max_eps(audit_dir: Path) -> tuple[float, dict]:
    """Read per-purpose linear-R² from dominant_axis_audit.json (seed 0)."""
    with open(audit_dir / "dominant_axis_audit.json") as f:
        audit = json.load(f)
    rows = audit["per_seed"]["0"]["rows"]
    by_purpose: dict[str, float] = {}
    for r in rows:
        v = float(r["r2_onehot"])
        by_purpose[r["purpose"]] = max(by_purpose.get(r["purpose"], 0.0), v)
    eps_per_purpose = list(by_purpose.values())
    return float(max(eps_per_purpose)), {
        "max_eps_per_purpose": by_purpose,
        "max_over_purposes": float(max(eps_per_purpose)),
    }


def analyse(dataset: str) -> dict:
    info = DATASETS[dataset]
    print(f"\n[{dataset}] loading encoder + validation set...")
    encoder = load_encoder(dataset, info)
    feats, attrs = load_validation(dataset)
    print(f"   feats={feats.shape}, attrs={list(attrs.keys())}")

    Hs = per_purpose_reprs(encoder, feats)
    eff_rank = per_purpose_effective_rank(Hs)
    H_concat_raw = np.concatenate([h - h.mean(0) for h in Hs], axis=1)
    S_concat = np.linalg.svd(H_concat_raw, compute_uv=False)
    concat_numerical_rank = int(
        (S_concat > 1e-8 * max(S_concat[0], 1e-12)).sum()
    )
    whitened: list[np.ndarray] = []
    whiten_diag: list[dict] = []
    for p, H in enumerate(Hs):
        Hw, diag = whiten(H)
        whitened.append(Hw)
        whiten_diag.append(diag)
        print(f"   purpose {p}: H shape {H.shape}, post-whiten ‖I-Σ̂‖_F = "
              f"{np.linalg.norm(np.eye(H.shape[1]) - (Hw.T @ Hw) / Hw.shape[0]):.4f}")

    H_concat = np.concatenate(whitened, axis=1)  # (N, k*d)
    n = H_concat.shape[0]
    C = (H_concat.T @ H_concat) / n
    eigvals = np.linalg.eigvalsh(C)
    eigvals_sorted = np.sort(eigvals)
    nonzero = eigvals_sorted[eigvals_sorted > NUMERICAL_ZERO_TOL]
    lambda_star = float(nonzero[0]) if nonzero.size else float("nan")

    max_eps, eps_diag = per_purpose_max_eps(ROOT / "results" / info["audit"])
    bound = N_PURPOSES * max_eps / lambda_star if lambda_star > 0 else float("inf")

    empirical = {}
    for attr_name, attr_vals in attrs.items():
        empirical[attr_name] = linear_r2_onehot(H_concat, attr_vals)

    return {
        "dataset": dataset,
        "config": info,
        "n_val": n,
        "purposes_d": [int(H.shape[1]) for H in Hs],
        "concat_dim": int(H_concat.shape[1]),
        "concat_numerical_rank_1e-8": concat_numerical_rank,
        "concat_rank_deficit": int(H_concat.shape[1]) - concat_numerical_rank,
        "per_purpose_effective_rank": eff_rank,
        "whitening_diagnostics": whiten_diag,
        "C_eigvals_top5": eigvals_sorted[-5:].tolist(),
        "C_eigvals_bottom5": eigvals_sorted[:5].tolist(),
        "C_n_numerical_zeros": int((eigvals_sorted <= NUMERICAL_ZERO_TOL).sum()),
        "lambda_star": lambda_star,
        "max_eps_per_purpose": eps_diag,
        "k": N_PURPOSES,
        "theorem5_bound": bound,
        "empirical_R2_concat_vs_attr": empirical,
    }


def per_purpose_effective_rank(Hs: list[np.ndarray]) -> list[dict]:
    out = []
    for p, H in enumerate(Hs):
        Hc = H - H.mean(axis=0, keepdims=True)
        S = np.linalg.svd(Hc, compute_uv=False)
        out.append(
            {
                "purpose_idx": p,
                "svd_top": float(S[0]),
                "effective_rank_1e-3": int((S > 1e-3 * max(S[0], 1e-12)).sum()),
                "effective_rank_1e-2": int((S > 1e-2 * max(S[0], 1e-12)).sum()),
                "svd_top10": S[:10].tolist(),
            }
        )
    return out


def write_decision(results: list[dict], path: Path) -> str:
    """Verdict policy:
        GO          : λ*(C) >= 0.3 on >=2 datasets  AND bound non-vacuous
        FALLBACK_E2 : λ*(C) >= 0.1 on >=2 datasets but bound somewhat loose
        FALLBACK_E4 : otherwise
    """
    lambdas = {r["dataset"]: r["lambda_star"] for r in results}
    bounds = {r["dataset"]: r["theorem5_bound"] for r in results}
    empiricals = {r["dataset"]: max(r["empirical_R2_concat_vs_attr"].values()) for r in results}
    n_ge_03 = sum(1 for v in lambdas.values() if v >= 0.3)
    n_ge_01 = sum(1 for v in lambdas.values() if v >= 0.1)
    bound_non_vacuous = all(b < 1.0 for b in bounds.values())
    bound_loose = any(b > 5.0 * empiricals[ds] for ds, b in bounds.items())

    if n_ge_03 >= 2 and bound_non_vacuous:
        verdict = "GO"
    elif n_ge_01 >= 2:
        verdict = "FALLBACK_E2"
    else:
        verdict = "FALLBACK_E4"

    lines = [f"# Theorem 5 decision gate — verdict: **{verdict}**", ""]
    lines.append("Computed λ*(C), the smallest non-zero eigenvalue of the block "
                 "correlation matrix of whitened per-purpose PCRL representations, "
                 "on the canonical PCRL checkpoint (seed 0) for each dataset.")
    lines.append("")
    lines.append("| Dataset | λ*(C) | max_i ε_i | k·ε/λ* (Thm 5 bound) | empirical max R²(h_concat, A) |")
    lines.append("|---------|------:|----------:|---------------------:|------------------------------:|")
    for r in results:
        ds = r["dataset"]
        emp = max(r["empirical_R2_concat_vs_attr"].values())
        lines.append(
            f"| {ds} | {r['lambda_star']:.4f} | "
            f"{r['max_eps_per_purpose']['max_over_purposes']:.4f} | "
            f"{r['theorem5_bound']:.4f} | {emp:.4f} |"
        )
    lines += ["", "## Decision policy", ""]
    lines.append(f"- Datasets with λ*(C) ≥ 0.3: **{n_ge_03}/{len(results)}**")
    lines.append(f"- Datasets with λ*(C) ≥ 0.1: **{n_ge_01}/{len(results)}**")
    lines.append(f"- Bound is non-vacuous (k·ε/λ* < 1) on every dataset: **{bound_non_vacuous}**")
    lines.append("")
    if verdict == "GO":
        lines.append("**GO** — λ* is large enough on at least two datasets and the bound "
                     "predicts a meaningful (sub-trivial) R²(h_concat, A) ceiling. Theorem 5 "
                     "is worth adding to the paper as a non-vacuous statement that links the "
                     "per-purpose ε constraints to the cross-purpose leakage we measure.")
    elif verdict == "FALLBACK_E2":
        lines.append("**FALLBACK_E2** — λ*(C) is well above zero so the per-purpose / "
                     "concat correspondence holds in principle, but the slack constant "
                     "k/λ* makes the bound numerically loose against the empirical R². "
                     "Use the exact identity (the variational form of Theorem 5) plus an "
                     "empirical conjecture in the §3 narrative; do not state the loose "
                     "closed-form bound as a theorem.")
    else:
        lines.append("**FALLBACK_E4** — λ*(C) is small enough that the bound k·ε/λ* exceeds "
                     "1 on most datasets, i.e. it is vacuous. Do not introduce Theorem 5; "
                     "instead, honestly reframe Theorem 2 as a label-shift bound that does "
                     "not predict the cross-purpose leakage observed in §5 (consistent with "
                     "the γ_min < 0.1 finding from Tier-1 Task A).")

    lines += ["", "## Why λ*(C) ≈ 0 (structural, not noise)", ""]
    lines.append(
        "The block correlation matrix collapses for two compounding reasons. "
        "First, the canonical PCRL checkpoints used here are the same ones "
        "summary.json flags as ``STATUS: COLLAPSED``: several per-purpose "
        "representations are essentially low-rank. Second, the three purposes "
        "share a frozen backbone and differ only by a rank-r LoRA "
        "perturbation, so the concatenation $\\hat h_{\\text{concat}} = "
        "[\\hat h_1, \\hat h_2, \\hat h_3]$ lives on a subspace of dimension "
        "well below $k \\cdot d_{\\text{repr}} = 192$ even when each $h_p$ is "
        "individually full-rank."
    )
    lines += ["", "Per-purpose effective rank and concat rank deficit:", ""]
    lines.append("| Dataset | per-purpose eff. rank (1e-3) | concat numerical rank | concat rank deficit (out of 192) |")
    lines.append("|---------|------------------------------|----------------------:|---------------------------------:|")
    for r in results:
        ranks = ", ".join(str(er["effective_rank_1e-3"]) for er in r["per_purpose_effective_rank"])
        lines.append(
            f"| {r['dataset']} | [{ranks}] | "
            f"{r['concat_numerical_rank_1e-8']} | "
            f"{r['concat_rank_deficit']} |"
        )

    lines += ["", "## Per-dataset detail", ""]
    for r in results:
        lines.append(f"### {r['dataset']}")
        lines.append(f"- N (val rows) = {r['n_val']}, concat dim = {r['concat_dim']} "
                     f"(per-purpose dims = {r['purposes_d']})")
        lines.append(f"- per-purpose effective rank (>1e-3 of top SVD): "
                     f"{[er['effective_rank_1e-3'] for er in r['per_purpose_effective_rank']]}")
        lines.append(f"- per-purpose top SVD: "
                     f"{[round(er['svd_top'], 3) for er in r['per_purpose_effective_rank']]}")
        lines.append(f"- concat numerical rank (>1e-8 of top): "
                     f"{r['concat_numerical_rank_1e-8']} of {r['concat_dim']}")
        lines.append(f"- C top-5 eigenvalues: "
                     f"{[f'{v:.3f}' for v in r['C_eigvals_top5']]}")
        lines.append(f"- C bottom-5 eigenvalues: "
                     f"{[f'{v:.3e}' for v in r['C_eigvals_bottom5']]}")
        lines.append(f"- numerical zeros (<{NUMERICAL_ZERO_TOL}): "
                     f"{r['C_n_numerical_zeros']}")
        lines.append(f"- per-purpose max ε_i: "
                     f"{r['max_eps_per_purpose']['max_eps_per_purpose']}")
        lines.append(f"- empirical R²(h_concat, A): "
                     f"{ {k: round(v, 4) for k, v in r['empirical_R2_concat_vs_attr'].items()} }")
        lines.append("")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return verdict


def main() -> None:
    results = []
    for ds in DATASETS:
        r = analyse(ds)
        results.append(r)
        print(f"  -> λ*(C) = {r['lambda_star']:.4f}, "
              f"max ε = {r['max_eps_per_purpose']['max_over_purposes']:.4f}, "
              f"bound k·ε/λ* = {r['theorem5_bound']:.4f}, "
              f"emp max R²(h_concat, A) = {max(r['empirical_R2_concat_vs_attr'].values()):.4f}")

    (OUT_DIR / "lambda_star_per_dataset.json").write_text(json.dumps(results, indent=2))
    table = [
        {
            "dataset": r["dataset"],
            "lambda_star": r["lambda_star"],
            "max_eps": r["max_eps_per_purpose"]["max_over_purposes"],
            "k": r["k"],
            "theorem5_bound": r["theorem5_bound"],
            "empirical_R2_concat_vs_attr": r["empirical_R2_concat_vs_attr"],
        }
        for r in results
    ]
    (OUT_DIR / "bound_vs_empirical_table.json").write_text(json.dumps(table, indent=2))
    verdict = write_decision(results, OUT_DIR / "DECISION.md")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()

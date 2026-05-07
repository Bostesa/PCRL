"""Numerical verification of Proposition 6 (Tier 2) — cross-purpose
linear-leakage bound. For each (seed, attribute) cell on Adult, compute:

  - per-purpose R²(h_p; A)
  - λ_min(R) where R is the kd × kd block whitened correlation matrix
  - predicted bound: (Σ_p R²(h_p;A)) / λ_min(R)
  - observed R²(H_concat; A)

All quantities computed from the SAME (test) split of empirical moments,
matching the proposition's distributional setup exactly.
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# Resolve repo root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402

CKPT_BASE = ROOT / "checkpoints"
SEEDS = [0, 1, 2]
CROSS_ATTRS = ["race", "sex", "age_group"]
EPS_PER_PURPOSE = 0.05  # the per-purpose constraint threshold in PCRL R5
REG = 1e-5


def load_encoder(ckpt_path: Path, input_dim: int, n_purposes: int) -> torch.nn.Module:
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone, n_purposes=n_purposes, rank=8, alpha=16.0, dropout=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(n_purposes):
        P_key = f"leace_P_p{p_idx}"
        mu_key = f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
    encoder.eval()
    return encoder


def extract_reps(encoder, loader, purpose_idx):
    chunks = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"]
            h = encoder(x, purpose_idx)
            chunks.append(h.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def extract_attr(loader, attr_name):
    out = []
    for batch in loader:
        out.append(batch["sensitive_attrs"][attr_name].numpy())
    return np.concatenate(out, axis=0).astype(np.int64)


def compute_r2(H, A_oh, reg=REG):
    """trace(Σ_HA^T Σ_H^{-1} Σ_HA) / trace(Σ_A) — closed-form OLS R²."""
    Hc = H - H.mean(0, keepdims=True)
    Ac = A_oh - A_oh.mean(0, keepdims=True)
    n, d = Hc.shape
    SHH = Hc.T @ Hc / n + reg * np.eye(d)
    SHA = Hc.T @ Ac / n
    SAA_trace = float((Ac * Ac).sum() / n)
    val = float(np.trace(SHA.T @ np.linalg.solve(SHH, SHA)))
    return val / max(SAA_trace, 1e-12)


def matrix_inv_sqrt(S, reg=REG):
    """Symmetric inverse square root of PSD S."""
    Sreg = S + reg * np.eye(S.shape[0])
    w, V = np.linalg.eigh(Sreg)
    w = np.maximum(w, reg)  # floor for numerical safety
    return V @ np.diag(w ** -0.5) @ V.T


def block_whitened_corr(H_per, reg=REG):
    """Build R = block whitened correlation matrix.
    H_per: list of n×d arrays (one per purpose).
    Returns R (kd × kd), λ_min(R), λ_+(R) (smallest positive), λ_max(R).
    """
    n = H_per[0].shape[0]
    Hc_per = [H - H.mean(0, keepdims=True) for H in H_per]
    cov_blocks = {}
    inv_sqrt = {}
    for p, Hp in enumerate(Hc_per):
        cov_blocks[(p, p)] = Hp.T @ Hp / n
        inv_sqrt[p] = matrix_inv_sqrt(cov_blocks[(p, p)], reg=reg)
    k = len(Hc_per)
    d = Hc_per[0].shape[1]
    R = np.zeros((k * d, k * d))
    for p in range(k):
        for q in range(k):
            if p == q:
                # whitened diagonal: should be ≈ I
                R_pq = inv_sqrt[p] @ cov_blocks[(p, p)] @ inv_sqrt[q]
            else:
                Spq = Hc_per[p].T @ Hc_per[q] / n
                R_pq = inv_sqrt[p] @ Spq @ inv_sqrt[q]
            R[p * d:(p + 1) * d, q * d:(q + 1) * d] = R_pq
    R = (R + R.T) / 2  # symmetrize against numerical drift
    eigs = np.linalg.eigvalsh(R)
    eigs = np.sort(eigs)
    lam_min = float(eigs[0])
    lam_pos = float(eigs[eigs > 1e-8][0]) if (eigs > 1e-8).any() else float("nan")
    lam_max = float(eigs[-1])
    return R, lam_min, lam_pos, lam_max


def main():
    purposes = get_adult_purposes()
    train_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                            split="train", download=False)
    test_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                           split="test", download=False,
                           norm_stats=train_ds.norm_stats)

    BS = 512
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)
    purpose_names = [p.name for p in purposes]
    k = len(purposes)
    print(f"N_test={len(test_ds)} D={train_ds.info.num_features} k={k}")
    print(f"purposes: {purpose_names}")

    test_attrs = {a: extract_attr(test_loader, a) for a in CROSS_ATTRS}

    rows = []
    # Try both checkpoint sources: Round 5 baseline + CROSSPURP run
    sources = [
        ("R5_baseline", "v2_pcrl_variance_constrained_adult_s{seed}", None),
        ("CROSSPURP", "v2_adult_CROSSPURP_s{seed}", None),
    ]
    # The Round 5 checkpoints actually live in results/.../checkpoints/final.pt
    R5_CKPT_DIR = ROOT / "results/v2_pcrl_variance_constrained"
    for seed in SEEDS:
        candidates = [
            ("R5_baseline", R5_CKPT_DIR / f"adult_s{seed}/checkpoints/final.pt"),
            ("CROSSPURP", CKPT_BASE / f"v2_adult_CROSSPURP_s{seed}/canonical_iterate.pt"),
            ("CROSSPURP", CKPT_BASE / f"v2_adult_CROSSPURP_s{seed}/best.pt"),
            ("CROSSPURP", CKPT_BASE / f"v2_adult_CROSSPURP_s{seed}/final.pt"),
        ]
        for src_label, ckpt_path in candidates:
            if not ckpt_path.exists():
                continue
            tag = f"{src_label}_s{seed}_{ckpt_path.name}"
            print(f"\n=== {tag} ({ckpt_path}) ===")
            try:
                encoder = load_encoder(ckpt_path, train_ds.info.num_features, k)
            except Exception as e:
                print(f"  load failed: {e}")
                continue
            H_te_per = [extract_reps(encoder, test_loader, idx) for idx in range(k)]
            H_te_concat = np.concatenate(H_te_per, axis=1)
            R, lam_min, lam_pos, lam_max = block_whitened_corr(H_te_per)
            print(f"  λ_min(R)={lam_min:.4f}  λ_+(R)={lam_pos:.4f}  λ_max(R)={lam_max:.4f}")
            for attr in CROSS_ATTRS:
                A = test_attrs[attr]
                K = int(A.max() + 1)
                A_oh = np.eye(K, dtype=np.float64)[A]
                r2_per = [compute_r2(H_te_per[p], A_oh) for p in range(k)]
                r2_concat = compute_r2(H_te_concat, A_oh)
                sum_per = sum(r2_per)
                bound_sharp = sum_per / lam_pos
                bound_uniform = k * EPS_PER_PURPOSE / lam_pos
                holds_sharp = r2_concat <= bound_sharp + 1e-6
                row = {
                    "source": src_label, "seed": seed, "ckpt": ckpt_path.name,
                    "attr": attr, "K": K,
                    "r2_per_purpose": r2_per,
                    "max_r2_per_purpose": max(r2_per),
                    "sum_r2_per_purpose": sum_per,
                    "lam_min_R": lam_min, "lam_pos_R": lam_pos,
                    "r2_concat_observed": r2_concat,
                    "bound_sharp_sum_over_lampos": bound_sharp,
                    "bound_uniform_keps_over_lampos": bound_uniform,
                    "bound_sharp_holds": bool(holds_sharp),
                    "bound_uniform_holds_assuming_eps": (max(r2_per) <= EPS_PER_PURPOSE)
                                                       and r2_concat <= bound_uniform + 1e-6,
                    "ratio_observed_over_bound": r2_concat / max(bound_sharp, 1e-12),
                }
                rows.append(row)
                print(f"  [{attr}] r2_per={[f'{x:.4f}' for x in r2_per]} sum={sum_per:.4f} "
                      f"r2_concat={r2_concat:.4f} bound_sharp={bound_sharp:.4f} "
                      f"holds={holds_sharp}")

    out_path = Path("/tmp/tier2_prop6/numerical_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(rows, fh, indent=2, default=lambda o: float(o)
                  if hasattr(o, "item") else str(o))
    print(f"\nWrote {out_path}  ({len(rows)} cells)")
    holds_sharp_all = sum(1 for r in rows if r["bound_sharp_holds"])
    print(f"Sharp bound holds on {holds_sharp_all}/{len(rows)} cells")


if __name__ == "__main__":
    main()

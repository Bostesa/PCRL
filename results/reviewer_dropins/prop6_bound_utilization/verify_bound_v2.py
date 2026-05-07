"""Numerical verification of Proposition 6 — v2 with thin-SVD whitening
(Moore-Penrose pseudoinverse) for the singular case II of the proposition.

This is the mathematically correct formulation when Σ_p has effectively-zero
eigenvalues (which happens with frozen-backbone + low-rank LoRA, where the
purpose representations share a common subspace).

For each purpose, we project onto eigendirections of Σ_p with eigenvalue
> tol * λ_max(Σ_p) (tol = 1e-3), whiten there, and compute R on the
resulting effective-rank representations. λ_+ then captures the smallest
non-degenerate joint mode.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402

CKPT_BASE = ROOT / "checkpoints"
SEEDS = [0, 1, 2]
CROSS_ATTRS = ["race", "sex", "age_group"]
EPS = 0.05
RCOND = 1e-3  # eigenvalue threshold relative to λ_max for thin SVD


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
        P_key = f"leace_P_p{p_idx}"; mu_key = f"leace_mu_p{p_idx}"
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


def thin_whiten(H, rcond=RCOND):
    """Thin-SVD whitening of centered H (n × d) to V (n × r) with Var(V)=I_r."""
    Hc = H - H.mean(0, keepdims=True)
    n = Hc.shape[0]
    S = Hc.T @ Hc / n  # d × d
    w, U = np.linalg.eigh(S)
    # Keep eigenvalues above rcond * λ_max
    if w.max() <= 0:
        return np.zeros((n, 0)), U[:, :0], np.array([])
    keep = w > rcond * w.max()
    U_keep = U[:, keep]
    w_keep = w[keep]
    # V_p = (eigvals)^{-1/2} U_p^T h_p
    V = Hc @ U_keep @ np.diag(w_keep ** -0.5)
    # Sanity: V.T @ V / n ≈ I
    return V, U_keep, w_keep


def compute_r2_via_whiten(H, A_oh, rcond=RCOND):
    """OLS R² via thin-SVD whitening; equivalent to Moore-Penrose OLS."""
    V, _, _ = thin_whiten(H, rcond=rcond)
    if V.shape[1] == 0:
        return 0.0
    Ac = A_oh - A_oh.mean(0, keepdims=True)
    n = V.shape[0]
    M = V.T @ Ac / n  # r × c, with V already whitened
    sa_trace = float((Ac * Ac).sum() / n)
    return float(np.trace(M.T @ M) / max(sa_trace, 1e-12))


def block_R(H_per, rcond=RCOND):
    """Build R = block correlation of thin-whitened reps. Returns R, eigenvalues,
    and per-purpose effective rank."""
    V_per = []
    ranks = []
    for H in H_per:
        V, _, _ = thin_whiten(H, rcond=rcond)
        V_per.append(V)
        ranks.append(V.shape[1])
    V_cat = np.concatenate(V_per, axis=1)  # n × Σ r_p
    n = V_cat.shape[0]
    Vc = V_cat - V_cat.mean(0, keepdims=True)
    R = Vc.T @ Vc / n
    R = (R + R.T) / 2
    eigs = np.sort(np.linalg.eigvalsh(R))
    return R, eigs, ranks


def compute_per_purpose_r2(H, A_oh, rcond=RCOND):
    return compute_r2_via_whiten(H, A_oh, rcond=rcond)


def compute_concat_r2(H_per, A_oh, rcond=RCOND):
    """Concat R² via stacking each purpose's whitened representation, then OLS.
    This is equivalent (up to whitening) to computing R²(concat(h); A) on the
    raw concatenation, projected onto each purpose's effective subspace."""
    V_per = []
    for H in H_per:
        V, _, _ = thin_whiten(H, rcond=rcond)
        V_per.append(V)
    V_cat = np.concatenate(V_per, axis=1)
    Ac = A_oh - A_oh.mean(0, keepdims=True)
    n = V_cat.shape[0]
    Vc = V_cat - V_cat.mean(0, keepdims=True)
    SVV = Vc.T @ Vc / n
    SVA = Vc.T @ Ac / n
    # Use solve with small jitter for numerical robustness on rank-deficient R
    r = SVV.shape[0]
    SVV_reg = SVV + 1e-10 * np.eye(r)
    val = float(np.trace(SVA.T @ np.linalg.solve(SVV_reg, SVA)))
    sa_trace = float((Ac * Ac).sum() / n)
    return val / max(sa_trace, 1e-12)


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
    print(f"N_test={len(test_ds)} D={train_ds.info.num_features} k={k}\n")

    test_attrs = {a: extract_attr(test_loader, a) for a in CROSS_ATTRS}

    rows = []
    R5_DIR = ROOT / "results/v2_pcrl_variance_constrained"
    for seed in SEEDS:
        candidates = [
            ("R5_baseline", R5_DIR / f"adult_s{seed}/checkpoints/final.pt"),
            ("CROSSPURP_canon", CKPT_BASE / f"v2_adult_CROSSPURP_s{seed}/canonical_iterate.pt"),
        ]
        for src_label, ckpt_path in candidates:
            if not ckpt_path.exists():
                continue
            print(f"=== {src_label} s{seed} ({ckpt_path.name}) ===")
            try:
                encoder = load_encoder(ckpt_path, train_ds.info.num_features, k)
            except Exception as e:
                print(f"  load failed: {e}"); continue
            H_te_per = [extract_reps(encoder, test_loader, idx) for idx in range(k)]
            R, eigs, ranks = block_R(H_te_per)
            lam_min = float(eigs[0])
            # smallest positive eigenvalue (above numerical threshold)
            pos = eigs[eigs > 1e-8]
            lam_pos = float(pos[0]) if len(pos) else float("nan")
            lam_max = float(eigs[-1])
            print(f"  per-purpose effective ranks: {ranks}  R shape: {R.shape}")
            print(f"  λ_min(R)={lam_min:.4f}  λ_+(R)={lam_pos:.4f}  λ_max(R)={lam_max:.4f}")
            for attr in CROSS_ATTRS:
                A = test_attrs[attr]
                K = int(A.max() + 1)
                A_oh = np.eye(K, dtype=np.float64)[A]
                r2_per = [compute_per_purpose_r2(H_te_per[p], A_oh) for p in range(k)]
                r2_concat = compute_concat_r2(H_te_per, A_oh)
                sum_per = sum(r2_per)
                bound_sharp = sum_per / lam_pos
                bound_uniform = k * EPS / lam_pos
                holds = r2_concat <= bound_sharp * (1 + 1e-6) + 1e-6
                row = {
                    "source": src_label, "seed": seed,
                    "ckpt": ckpt_path.name, "attr": attr, "K": K,
                    "r2_per_purpose": r2_per,
                    "max_r2_per_purpose": max(r2_per),
                    "sum_r2_per_purpose": sum_per,
                    "lam_min_R": lam_min, "lam_pos_R": lam_pos,
                    "r2_concat_observed": r2_concat,
                    "bound_sharp": bound_sharp,
                    "bound_uniform_keps": bound_uniform,
                    "bound_holds": bool(holds),
                    "ratio_obs_over_bound_sharp": r2_concat / max(bound_sharp, 1e-12),
                    "purpose_ranks": ranks,
                }
                rows.append(row)
                marker = "✓" if holds else "✗ VIOLATION"
                print(f"  [{attr}] r2_per={[f'{x:.4f}' for x in r2_per]} "
                      f"sum={sum_per:.4f} concat={r2_concat:.4f} "
                      f"bound={bound_sharp:.4f} {marker}")
            print()

    out_path = Path("/tmp/tier2_prop6/numerical_verification_v2.json")
    with out_path.open("w") as fh:
        json.dump(rows, fh, indent=2, default=lambda o: float(o)
                  if hasattr(o, "item") else str(o))
    holds_all = sum(1 for r in rows if r["bound_holds"])
    print(f"Bound holds on {holds_all}/{len(rows)} cells")
    informative = sum(1 for r in rows if r["bound_sharp"] < 1.0)
    print(f"Bound is informative (< 1) on {informative}/{len(rows)} cells")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

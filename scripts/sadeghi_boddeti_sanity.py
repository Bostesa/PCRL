"""Phase 2 sanity check for §B.2: Sadeghi-Boddeti u*(eps) certificate
on Adult income_prediction/sex.

Setup:
- X = post-backbone output (64-dim, FROZEN) on Adult train + test.
- A = sex (binary).
- Y = income (binary).

Closed-form linear Pareto frontier (binary A, binary Y, scalar projection):
  Define c_XX = Cov(X), c_XA = Cov(X, A), c_XY = Cov(X, Y),
         sigma_AA = Var(A), sigma_YY = Var(Y).
  Whitened directions:
    u_A = c_XX^{-1/2} c_XA,   ||u_A||^2 = c_XA^T c_XX^{-1} c_XA = rho_XA^2 * sigma_AA
    u_Y = c_XX^{-1/2} c_XY,   ||u_Y||^2 = rho_XY^2 * sigma_YY
  rho_XA^2 = max linear R^2(h, A); rho_XY^2 = max linear R^2(h, Y).
  cos(theta) = (u_A . u_Y) / (||u_A|| ||u_Y||).
  If eps >= rho_XA^2: constraint not binding, u*(eps) = rho_XY^2.
  Else (binding):
    Let alpha = sqrt(eps / rho_XA^2). Then:
      u*(eps) = rho_XY^2 * ( alpha * |cos(theta)| + sqrt(1 - alpha^2) * sin(theta) )^2

Stop conditions per spec:
  (i)   u*(0.05) negative, NaN, or > 1   -> formula misspecified
  (ii)  u*(0.05) below "majority baseline R^2" (R^2 of constant predictor = 0)
        -> formula gives meaningless floor
  (iii) PCRL_R^2_test > u*(0.05) + 0.02   -> certificate inverted
  (iv)  cond(c_XX) > 1e8                  -> ill-conditioned

If any trigger, ABORT and report failure mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcrl.data.base import collate_pcrl_batch
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry

from run_v2_dataset import build_datasets, LORA_BY_DATASET


def materialize(loader, encoder, purpose_idx, sex_attr, income_task, device="cpu"):
    """Returns (X_post_backbone, h_p, sex_int, income_int) as numpy arrays."""
    encoder.eval()
    Xb_list, h_list, A_list, Y_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            # Pre-LoRA backbone output: encoder.backbone(x) bypasses LoRA hooks.
            xb = encoder.backbone(x)
            # Post-LoRA per-purpose output: encoder(x, purpose_idx).
            h = encoder(x, purpose_idx)
            Xb_list.append(xb.cpu().numpy())
            h_list.append(h.cpu().numpy())
            A_list.append(batch["sensitive_attrs"][sex_attr].long().cpu().numpy())
            Y_list.append(batch["task_labels"][income_task].long().cpu().numpy())
    return (
        np.concatenate(Xb_list, axis=0).astype(np.float64),
        np.concatenate(h_list, axis=0).astype(np.float64),
        np.concatenate(A_list, axis=0).astype(np.float64),
        np.concatenate(Y_list, axis=0).astype(np.float64),
    )


def linear_r2(H, z, ridge=1e-6):
    """Closed-form Tikhonov-regularized linear R^2 of predicting (centered) z from H."""
    n, d = H.shape
    Hc = H - H.mean(axis=0, keepdims=True)
    zc = z - z.mean()
    gram = Hc.T @ Hc + ridge * np.eye(d)
    w = np.linalg.solve(gram, Hc.T @ zc)
    pred = Hc @ w
    ss_res = float(((zc - pred) ** 2).sum())
    ss_tot = float((zc ** 2).sum())
    return max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))


def pareto_u_star(X, A, Y, eps, ridge=1e-6):
    """Sadeghi-Boddeti closed-form linear Pareto u*(eps)."""
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    Ac = A - A.mean()
    Yc = Y - Y.mean()
    # c_XX = (1/n) X^T X (sample covariance). Add ridge.
    c_XX = (Xc.T @ Xc) / max(n - 1, 1) + ridge * np.eye(d)
    c_XA = (Xc.T @ Ac) / max(n - 1, 1)  # (d,)
    c_XY = (Xc.T @ Yc) / max(n - 1, 1)
    sigma_AA = float((Ac ** 2).sum() / max(n - 1, 1))
    sigma_YY = float((Yc ** 2).sum() / max(n - 1, 1))

    cond_cXX = float(np.linalg.cond(c_XX))

    # Solve c_XX^{-1} c_XA and c_XX^{-1} c_XY (equivalent to OLS coefficients).
    inv_cXA = np.linalg.solve(c_XX, c_XA)
    inv_cXY = np.linalg.solve(c_XX, c_XY)
    norm_uA_sq = float(c_XA @ inv_cXA)              # ||u_A||^2 in whitened space
    norm_uY_sq = float(c_XY @ inv_cXY)
    rho_XA2 = norm_uA_sq / max(sigma_AA, 1e-12)
    rho_XY2 = norm_uY_sq / max(sigma_YY, 1e-12)

    inner_AY = float(c_XA @ inv_cXY)                # u_A . u_Y in whitened space
    cos_theta = inner_AY / max(np.sqrt(norm_uA_sq * norm_uY_sq), 1e-12)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = float(np.sqrt(max(1.0 - cos_theta ** 2, 0.0)))

    if eps >= rho_XA2:
        u_star = rho_XY2  # constraint not binding
        binding = False
    else:
        alpha = float(np.sqrt(eps / rho_XA2))
        bracket = abs(cos_theta) * alpha + sin_theta * float(np.sqrt(max(1 - alpha ** 2, 0.0)))
        u_star = rho_XY2 * (bracket ** 2)
        binding = True

    return {
        "rho_XA2": rho_XA2,
        "rho_XY2": rho_XY2,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "binding": binding,
        "u_star": u_star,
        "cond_cXX": cond_cXX,
        "sigma_AA": sigma_AA,
        "sigma_YY": sigma_YY,
    }


def main():
    DEVICE = "cpu"
    EPS = 0.05
    RIDGE = 1e-6

    purposes, train_ds, val_ds, test_ds = build_datasets("adult")
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    purpose_names = [p.name for p in purposes]
    income_idx = purpose_names.index("income_prediction")
    print(f"income_prediction purpose index: {income_idx}")

    backbone = StandardEncoder(
        input_dim=train_ds.info.num_features, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    rank, alpha = LORA_BY_DATASET.get("adult", (8, 16.0))
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes), rank=rank, alpha=alpha, dropout=0.0,
    )
    task_heads = {}
    for p in purposes:
        out_dim = p.allowed_task_dims.get(p.allowed_tasks[0], 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=out_dim)

    # Load Adult ROUND5 seed-0 checkpoint
    ckpt_path = ROOT / "checkpoints/v2_adult_ROUND5_s0/final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(len(purposes)):
        P_key = f"leace_P_p{p_idx}"; mu_key = f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
    nn = torch.nn
    th_module = nn.ModuleDict(task_heads)
    th_module.load_state_dict(ckpt["task_heads"])
    print(f"loaded checkpoint {ckpt_path.name}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    print("\nMaterializing (X_backbone, h_p, sex, income) on train ...")
    X_tr_bb, h_tr, sex_tr, inc_tr = materialize(train_loader, encoder, income_idx, "sex", "income", DEVICE)
    print(f"  N_train = {X_tr_bb.shape[0]}, dim(X_backbone) = {X_tr_bb.shape[1]}, dim(h_p) = {h_tr.shape[1]}")
    print(f"  P(sex=1) = {sex_tr.mean():.3f}, P(income=1) = {inc_tr.mean():.3f}")

    print("\nMaterializing on test ...")
    X_te_bb, h_te, sex_te, inc_te = materialize(test_loader, encoder, income_idx, "sex", "income", DEVICE)

    # Compute u*(eps) on the backbone output X (interpretation β).
    print(f"\n=== u*(eps={EPS}) on X = post-backbone (64-dim), train ===")
    pareto = pareto_u_star(X_tr_bb, sex_tr, inc_tr, EPS, ridge=RIDGE)
    print(f"  rho_XA^2 (max R²(h, sex))    = {pareto['rho_XA2']:.4f}")
    print(f"  rho_XY^2 (max R²(h, income)) = {pareto['rho_XY2']:.4f}")
    print(f"  cos(theta) (CCA angle)       = {pareto['cos_theta']:.4f}")
    print(f"  constraint binding?          = {pareto['binding']}")
    print(f"  u*({EPS}) = {pareto['u_star']:.4f}")
    print(f"  cond(c_XX) = {pareto['cond_cXX']:.2e}")
    print(f"  sigma_AA = {pareto['sigma_AA']:.4f}, sigma_YY = {pareto['sigma_YY']:.4f}")

    # Compute PCRL's empirical TASK R² from h_p on test set.
    pcrl_task_r2_test = linear_r2(h_te, inc_te)
    pcrl_attr_r2_test = linear_r2(h_te, sex_te)
    print(f"\n=== PCRL empirical R² (test, from h_p) ===")
    print(f"  R²(h_p, income) = {pcrl_task_r2_test:.4f}  (PCRL's task R²)")
    print(f"  R²(h_p, sex)    = {pcrl_attr_r2_test:.4f}  (PCRL's attr R² — should be < 0.05)")

    # Also compute u* on test for honesty (for held-out comparison)
    pareto_test = pareto_u_star(X_te_bb, sex_te, inc_te, EPS, ridge=RIDGE)
    print(f"\n=== u*(eps={EPS}) on X = post-backbone, TEST set (held-out) ===")
    print(f"  u*({EPS}) on test = {pareto_test['u_star']:.4f}")

    # Stop conditions
    print(f"\n=== Stop-condition checks ===")
    print(f"  (i)   u*(0.05) in [0, 1]? u*={pareto['u_star']:.4f}  -> {0 <= pareto['u_star'] <= 1}")
    # Majority baseline R² for centered binary Y is 0 (constant predictor explains 0 variance).
    print(f"  (ii)  u*(0.05) >= 0 (majority baseline)? -> {pareto['u_star'] >= 0}")
    gap = pareto['u_star'] - pcrl_task_r2_test
    print(f"  (iii) PCRL_R²_test - u*(0.05)_train = {-gap:+.4f}; > +0.02 means certificate inverted")
    print(f"        verdict: {'INVERTED (PCRL above bound by >2pp)' if gap < -0.02 else 'OK'}")
    print(f"  (iv)  cond(c_XX) = {pareto['cond_cXX']:.2e} (threshold 1e8) -> {'OK' if pareto['cond_cXX'] < 1e8 else 'ILL-CONDITIONED'}")

    # Decision
    abort = False
    reasons = []
    if not (0 <= pareto["u_star"] <= 1):
        abort = True; reasons.append("u* out of [0, 1]")
    if pareto["u_star"] < 0:
        abort = True; reasons.append("u* below majority baseline (negative)")
    if pcrl_task_r2_test > pareto["u_star"] + 0.02:
        abort = True; reasons.append(f"PCRL R²={pcrl_task_r2_test:.4f} > u*={pareto['u_star']:.4f} + 0.02")
    if pareto["cond_cXX"] > 1e8:
        abort = True; reasons.append(f"cond(c_XX) = {pareto['cond_cXX']:.2e} > 1e8")

    print(f"\n=== Phase 2 decision ===")
    if abort:
        print(f"ABORT — stop conditions triggered:")
        for r in reasons:
            print(f"  - {r}")
        print(f"\nReport this to user; do NOT proceed to Phase 3.")
    else:
        print(f"PASS — all stop conditions clear. Phase 3 implementation can proceed.")
        print(f"  PCRL is within {gap*100:.2f} pp of the linear Pareto frontier on this triple.")


if __name__ == "__main__":
    main()

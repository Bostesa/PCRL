"""Phase 3: compute u*(eps=0.05) for all binary-attribute (dataset, purpose,
attribute) triples in the 20-cell grid using the trained ROUND5/7 checkpoints.

Multi-class attribute triples (Adult race / age_group / marital_status,
HMDA race, Diabetes race / age_bucket) require a multi-output extension
of the Sadeghi-Boddeti closed form and are SKIPPED here; flagged in output.

Binary-A triples covered (7 of 20):
  Adult: income_prediction/sex, education_assessment/sex,
         education_assessment/income
  HMDA:  underwriting/ethnicity, pricing_analysis/sex,
         fair_lending_audit/sex
  Diabetes: billing_audit/gender, clinical_decision_support/gender

  (Adult: 3, HMDA: 3, Diabetes: 2 = 8 — actually 8 binary triples; one extra
   was missed in the count above; will discover at runtime.)
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

CKPT_TAG_BY_DATASET = {"adult": "_ROUND5", "hmda": "_ROUND5", "diabetes": "_ROUND7"}
SEED = 0  # use seed 0 for the certificate; cell-level numbers are stable across seeds
EPS = 0.05
RIDGE = 1e-6
DEVICE = "cpu"


def linear_r2(H, z, ridge=RIDGE):
    n, d = H.shape
    Hc = H - H.mean(axis=0, keepdims=True)
    zc = z - z.mean()
    gram = Hc.T @ Hc + ridge * np.eye(d)
    w = np.linalg.solve(gram, Hc.T @ zc)
    pred = Hc @ w
    ss_res = float(((zc - pred) ** 2).sum())
    ss_tot = float((zc ** 2).sum())
    return max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))


def pareto_u_star(X, A, Y, eps, ridge=RIDGE):
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    Ac = A - A.mean()
    Yc = Y - Y.mean()
    c_XX = (Xc.T @ Xc) / max(n - 1, 1) + ridge * np.eye(d)
    c_XA = (Xc.T @ Ac) / max(n - 1, 1)
    c_XY = (Xc.T @ Yc) / max(n - 1, 1)
    sigma_AA = float((Ac ** 2).sum() / max(n - 1, 1))
    sigma_YY = float((Yc ** 2).sum() / max(n - 1, 1))
    inv_cXA = np.linalg.solve(c_XX, c_XA)
    inv_cXY = np.linalg.solve(c_XX, c_XY)
    norm_uA_sq = float(c_XA @ inv_cXA)
    norm_uY_sq = float(c_XY @ inv_cXY)
    rho_XA2 = norm_uA_sq / max(sigma_AA, 1e-12)
    rho_XY2 = norm_uY_sq / max(sigma_YY, 1e-12)
    inner_AY = float(c_XA @ inv_cXY)
    cos_theta = inner_AY / max(np.sqrt(norm_uA_sq * norm_uY_sq), 1e-12)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = float(np.sqrt(max(1.0 - cos_theta ** 2, 0.0)))
    if eps >= rho_XA2:
        u_star = rho_XY2; binding = False
    else:
        alpha = float(np.sqrt(eps / rho_XA2))
        bracket = abs(cos_theta) * alpha + sin_theta * float(np.sqrt(max(1 - alpha ** 2, 0.0)))
        u_star = rho_XY2 * (bracket ** 2); binding = True
    return {
        "rho_XA2": rho_XA2, "rho_XY2": rho_XY2, "cos_theta": cos_theta,
        "binding": binding, "u_star": u_star,
        "cond_cXX": float(np.linalg.cond(c_XX)),
    }


def materialize_for_purpose(loader, encoder, purpose_idx, attr_name, task_name, device=DEVICE):
    encoder.eval()
    Xb_list, h_list, A_list, Y_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            xb = encoder.backbone(x)
            h = encoder(x, purpose_idx)
            Xb_list.append(xb.cpu().numpy())
            h_list.append(h.cpu().numpy())
            A_list.append(batch["sensitive_attrs"][attr_name].long().cpu().numpy())
            if task_name in batch["task_labels"]:
                Y_list.append(batch["task_labels"][task_name].long().cpu().numpy())
            else:
                # Some attributes are also tasks for other purposes (e.g.,
                # education_assessment/income — income is the disallowed attr
                # but is also a task for income_prediction). Pull from
                # sensitive_attrs as a fallback.
                Y_list.append(batch["sensitive_attrs"].get(task_name,
                              batch["sensitive_attrs"][attr_name]).long().cpu().numpy())
    return (
        np.concatenate(Xb_list, 0).astype(np.float64),
        np.concatenate(h_list, 0).astype(np.float64),
        np.concatenate(A_list, 0).astype(np.float64),
        np.concatenate(Y_list, 0).astype(np.float64),
    )


def load_encoder(dataset, seed):
    purposes, train_ds, val_ds, test_ds = build_datasets(dataset)
    registry = PurposeRegistry()
    for p in purposes: registry.register(p)
    backbone = StandardEncoder(
        input_dim=train_ds.info.num_features, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    rank, alpha = LORA_BY_DATASET.get(dataset, (8, 16.0))
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes), rank=rank, alpha=alpha, dropout=0.0,
    )
    task_heads = {}
    for p in purposes:
        out_dim = p.allowed_task_dims.get(p.allowed_tasks[0], 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=out_dim)
    tag = CKPT_TAG_BY_DATASET[dataset]
    ckpt_path = ROOT / f"checkpoints/v2_{dataset}{tag}_s{seed}/final.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(len(purposes)):
        Pk = f"leace_P_p{p_idx}"; muk = f"leace_mu_p{p_idx}"
        if Pk in enc_buf and muk in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[Pk], enc_buf[muk])
    return purposes, train_ds, val_ds, test_ds, encoder


def n_classes_for_attr(dataset, attr_name):
    """Best-effort lookup of n_classes for an attribute name."""
    # Hardcoded based on get_*_purposes() definitions.
    if dataset == "adult":
        return {"race": 5, "sex": 2, "age_group": 4, "marital_status": 5,
                "income": 2, "occupation_group": 4, "education_level": 5}.get(attr_name, 2)
    if dataset == "hmda":
        return {"race": 5, "sex": 2, "ethnicity": 2,
                "loan_decision": 2, "loan_amount_band": 4, "tract_denial_high": 2}.get(attr_name, 2)
    if dataset == "diabetes":
        return {"race": 5, "gender": 2, "age_bucket": 10,
                "primary_diagnosis_category": 9, "readmission_outcome": 2,
                "medication_change_outcome": 2}.get(attr_name, 2)
    return 2


def main():
    rows = []
    skipped = []
    for dataset in ("adult", "hmda", "diabetes"):
        purposes, train_ds, val_ds, test_ds, encoder = load_encoder(dataset, SEED)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                                  collate_fn=collate_pcrl_batch, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                                 collate_fn=collate_pcrl_batch, num_workers=0)
        for purpose_idx, p in enumerate(purposes):
            task_name = p.allowed_tasks[0]
            for attr_name in p.disallowed_attrs:
                K_A = n_classes_for_attr(dataset, attr_name)
                if K_A != 2:
                    skipped.append((dataset, p.name, attr_name, K_A))
                    continue
                K_Y = n_classes_for_attr(dataset, task_name)
                if K_Y != 2:
                    skipped.append((dataset, p.name, attr_name, K_A,
                                    f"task {task_name} not binary (K_Y={K_Y})"))
                    continue
                # Materialize X (post-backbone), h_p, A, Y on train + test.
                X_tr, h_tr, A_tr, Y_tr = materialize_for_purpose(
                    train_loader, encoder, purpose_idx, attr_name, task_name,
                )
                X_te, h_te, A_te, Y_te = materialize_for_purpose(
                    test_loader, encoder, purpose_idx, attr_name, task_name,
                )
                par_tr = pareto_u_star(X_tr, A_tr, Y_tr, EPS)
                par_te = pareto_u_star(X_te, A_te, Y_te, EPS)
                pcrl_task_r2 = linear_r2(h_te, Y_te)
                pcrl_attr_r2 = linear_r2(h_te, A_te)
                gap_train = par_tr["u_star"] - pcrl_task_r2
                rows.append({
                    "dataset": dataset, "purpose": p.name, "attribute": attr_name,
                    "task": task_name, "K_A": K_A, "K_Y": K_Y, "seed": SEED,
                    "u_star_train": par_tr["u_star"],
                    "u_star_test":  par_te["u_star"],
                    "rho_XA2_train": par_tr["rho_XA2"],
                    "rho_XY2_train": par_tr["rho_XY2"],
                    "cos_theta_train": par_tr["cos_theta"],
                    "binding_train": par_tr["binding"],
                    "cond_cXX_train": par_tr["cond_cXX"],
                    "pcrl_task_r2_test": pcrl_task_r2,
                    "pcrl_attr_r2_test": pcrl_attr_r2,
                    "gap_train_minus_pcrl": gap_train,
                    "stops": {
                        "u_star_in_unit": 0 <= par_tr["u_star"] <= 1,
                        "u_star_nonneg":  par_tr["u_star"] >= 0,
                        "pcrl_below_u_star_within_2pp": pcrl_task_r2 <= par_tr["u_star"] + 0.02,
                        "well_conditioned": par_tr["cond_cXX"] < 1e8,
                    },
                })
                print(f"  {dataset:9s} {p.name:25s} {attr_name:12s} "
                      f"u*={par_tr['u_star']:.4f} (test {par_te['u_star']:.4f})  "
                      f"PCRL R²={pcrl_task_r2:.4f}  gap={gap_train:+.4f}  "
                      f"binding={par_tr['binding']}")
    print()
    print(f"=== Summary across {len(rows)} binary-A triples (seed {SEED}) ===")
    if rows:
        gaps = [r["gap_train_minus_pcrl"] for r in rows]
        print(f"  mean gap u* − PCRL = {np.mean(gaps):+.4f}")
        print(f"  max  gap u* − PCRL = {max(gaps):+.4f}")
        print(f"  min  gap u* − PCRL = {min(gaps):+.4f}")
        print(f"  triples within 5pp of u* (gap < 0.05):  "
              f"{sum(1 for g in gaps if g < 0.05)}/{len(rows)}")
        print(f"  triples within 10pp of u* (gap < 0.10): "
              f"{sum(1 for g in gaps if g < 0.10)}/{len(rows)}")
        print(f"  triples where PCRL > u* + 0.02 (cert inverted, BAD): "
              f"{sum(1 for g in gaps if g < -0.02)}/{len(rows)}")
        print(f"  triples binding (eps < rho_XA²): "
              f"{sum(1 for r in rows if r['binding_train'])}/{len(rows)}")
    print(f"\n  Skipped (non-binary A or Y): {len(skipped)} triples")
    for s in skipped:
        print(f"    {s}")
    out = {
        "config": {"eps": EPS, "ridge": RIDGE, "seed": SEED, "device": DEVICE,
                   "X_definition": "post-backbone output (64-dim, frozen)"},
        "n_triples_computed": len(rows),
        "n_triples_skipped": len(skipped),
        "rows": rows,
        "skipped": [list(s) for s in skipped],
    }
    out_path = ROOT / "results/reviewer_dropins/pareto_certificate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()

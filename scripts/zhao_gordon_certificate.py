"""Zhao-Gordon TV-Barycenter near-optimality certificate for PCRL.

Theorem 8 + Section 5 multi-class extension from
"Inherent Tradeoffs in Learning Fair Representations" (arXiv:1906.08386).

For each (purpose p, attribute A) cell with A having n classes and Y_p
having m classes, the lower bound on the sum of per-subgroup errors
of any classifier on the representation is:

    sum_a Err_a >= B - (n-1) * Delta_DP

where:
  p_a   := P(Y_p | A=a)  as histogram over m classes (data-only)
  B     := min_q (1/2) sum_a ||p_a - q||_1     (TV-barycenter, simplex LP)
  Delta_DP := max_{a,b} (1/2) sum_k |P(yhat=k|A=a) - P(yhat=k|A=b)|

The bound HOLDS FOR ANY ENCODER (linear or nonlinear), addressing the
structural failure of the Sadeghi-Boddeti linear Pareto bound that
caused §B.2 to be skipped.

Output: results/reviewer_dropins/zhao_gordon_table.csv with columns
  dataset, purpose, attribute, n, m, B, Delta_DP, F, sum_Err, slack
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch
from scipy.optimize import linprog
from torch.utils.data import DataLoader

from pcrl.data.base import collate_pcrl_batch
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry

from run_v2_dataset import build_datasets, LORA_BY_DATASET

CKPT_TAG_BY_DATASET = {"adult": "_ROUND5", "hmda": "_ROUND5", "diabetes": "_ROUND7"}
SEED = 0
DEVICE = "cpu"

# Lemma 1 constant: for ε = R²-cert threshold = 0.05, binary A π=0.5, binary
# ŷ with Var(ŷ) ≤ 1/4: ΔDP_max ≤ √(ε/[4 π(1-π)]) = √0.05 ≈ 0.2236.
# Used as a conservative upper bound on ΔDP for the theoretical floor F_theory.
import math
EPS_R2 = 0.05
DELTA_DP_THEORY_MAX = math.sqrt(EPS_R2 / (4 * 0.5 * (1 - 0.5)))  # = 0.2236


def tv_barycenter(p_groups: np.ndarray) -> tuple[float, np.ndarray]:
    """L1 / TV barycenter of n probability vectors over the m-simplex.

    Solves: min_q (1/2) sum_a ||p_a - q||_1   s.t. q in simplex.

    LP variables: q (m-dim), t_{a,k} (n*m-dim).
    Minimize (1/2) sum t_{a,k}
    Subject to: t_{a,k} >= p_a[k] - q[k]
                t_{a,k} >= q[k] - p_a[k]
                sum_k q[k] = 1
                q >= 0, t >= 0

    Returns (B, q*).
    """
    n, m = p_groups.shape  # n groups, m classes
    n_vars = m + n * m  # q (m) + t (n*m)

    # Objective: 0 on q, 0.5 on t
    c = np.concatenate([np.zeros(m), 0.5 * np.ones(n * m)])

    # Inequality constraints A_ub @ x <= b_ub
    # For each (a, k): -t_{a,k} - q[k] <= -p_a[k]  (i.e., t_{a,k} + q[k] >= p_a[k])
    # For each (a, k): -t_{a,k} + q[k] <= p_a[k]   (i.e., t_{a,k} - q[k] >= -p_a[k])
    A_ub = np.zeros((2 * n * m, n_vars))
    b_ub = np.zeros(2 * n * m)
    row = 0
    for a in range(n):
        for k in range(m):
            t_idx = m + a * m + k
            # t_{a,k} >= p_a[k] - q[k]  =>  -t -(- q[k]) <= -p_a[k]
            # i.e., -t_{a,k} - q[k] <= -p_a[k]
            A_ub[row, t_idx] = -1.0
            A_ub[row, k] = -1.0
            b_ub[row] = -p_groups[a, k]
            row += 1
            # t_{a,k} >= q[k] - p_a[k]
            # -t_{a,k} + q[k] <= p_a[k]
            A_ub[row, t_idx] = -1.0
            A_ub[row, k] = +1.0
            b_ub[row] = p_groups[a, k]
            row += 1

    # Equality: sum_k q[k] = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: q >= 0, t >= 0
    bounds = [(0, None)] * n_vars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs")
    if not res.success:
        raise RuntimeError(f"TV-barycenter LP failed: {res.message}")
    q_star = res.x[:m]
    B = float(res.fun)
    return B, q_star


def delta_dp(yhat_a_dist: np.ndarray) -> float:
    """Max pairwise TV distance between group-conditional ŷ distributions.

    yhat_a_dist: (n_groups, m_classes) histogram of P(ŷ | A=a).
    Returns max_{a, b} (1/2) sum_k |yhat_a_dist[a, k] - yhat_a_dist[b, k]|.
    """
    n = yhat_a_dist.shape[0]
    max_tv = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            tv = 0.5 * float(np.abs(yhat_a_dist[a] - yhat_a_dist[b]).sum())
            if tv > max_tv:
                max_tv = tv
    return max_tv


def materialize_predictions(loader, encoder, task_heads, purpose_idx,
                             attr_name, task_name, device=DEVICE):
    """Returns (yhat_int, y_int, A_int) for the given purpose × attribute × task."""
    encoder.eval()
    yhat_list, y_list, A_list = [], [], []
    th = task_heads[list(task_heads.keys())[purpose_idx]]
    th.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            logits = th(h)
            if isinstance(logits, dict):
                logits = logits[task_name]
            yhat = logits.argmax(dim=1).cpu().numpy()
            yhat_list.append(yhat)
            # Get the task label
            if task_name in batch["task_labels"]:
                y = batch["task_labels"][task_name].long().cpu().numpy()
            else:
                y = batch["sensitive_attrs"][task_name].long().cpu().numpy()
            y_list.append(y)
            A_list.append(batch["sensitive_attrs"][attr_name].long().cpu().numpy())
    return (np.concatenate(yhat_list), np.concatenate(y_list),
            np.concatenate(A_list))


def histogram(labels, n_classes):
    """Empirical histogram of a categorical variable."""
    h = np.zeros(n_classes, dtype=np.float64)
    if len(labels) == 0:
        return h
    for k in range(n_classes):
        h[k] = float((labels == k).sum())
    return h / max(h.sum(), 1)


def per_group_stats(yhat, y, A, n_A, m_Y):
    """Returns:
      p_groups (n_A, m_Y): P(Y | A=a)
      yhat_groups (n_A, m_Y): P(yhat | A=a)
      err_per_group (n_A,): empirical error rate within each group
    """
    p_groups = np.zeros((n_A, m_Y))
    yhat_groups = np.zeros((n_A, m_Y))
    err_per_group = np.zeros(n_A)
    for a in range(n_A):
        mask = (A == a)
        if not mask.any():
            continue
        p_groups[a] = histogram(y[mask], m_Y)
        yhat_groups[a] = histogram(yhat[mask], m_Y)
        err_per_group[a] = float((yhat[mask] != y[mask]).mean())
    return p_groups, yhat_groups, err_per_group


def n_classes_for_attr(dataset, attr_name):
    if dataset == "adult":
        return {"race": 5, "sex": 2, "age_group": 4, "marital_status": 5,
                "income": 2, "occupation_group": 4, "education_level": 5}.get(attr_name, 2)
    if dataset == "hmda":
        return {"race": 5, "sex": 2, "ethnicity": 2,
                "loan_decision": 2, "loan_amount_band": 4,
                "tract_denial_high": 2}.get(attr_name, 2)
    if dataset == "diabetes":
        return {"race": 5, "gender": 2, "age_bucket": 10,
                "primary_diagnosis_category": 9, "readmission_outcome": 2,
                "medication_change_outcome": 2}.get(attr_name, 2)
    return 2


def load_encoder(dataset, seed):
    purposes, train_ds, val_ds, test_ds = build_datasets(dataset)
    backbone = StandardEncoder(input_dim=train_ds.info.num_features,
                               hidden_dims=[128, 128], repr_dim=64, dropout=0.3)
    rank, alpha = LORA_BY_DATASET.get(dataset, (8, 16.0))
    encoder = PerPurposeLoRAEncoder(backbone=backbone, n_purposes=len(purposes),
                                     rank=rank, alpha=alpha, dropout=0.0)
    task_heads = {}
    for p in purposes:
        out_dim = p.allowed_task_dims.get(p.allowed_tasks[0], 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=out_dim)
    th_module = torch.nn.ModuleDict(task_heads)
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
    th_module.load_state_dict(ckpt["task_heads"])
    return purposes, train_ds, val_ds, test_ds, encoder, task_heads


def main():
    rows = []
    print(f"{'cell':<55s} {'n':>3s} {'m':>3s} {'B':>7s} {'ΔDP':>7s} "
          f"{'F':>8s} {'ΣErr':>7s} {'slack':>8s}")
    for dataset in ("adult", "hmda", "diabetes"):
        purposes, train_ds, val_ds, test_ds, encoder, task_heads = load_encoder(dataset, SEED)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                                 collate_fn=collate_pcrl_batch, num_workers=0)
        for purpose_idx, p in enumerate(purposes):
            task_name = p.allowed_tasks[0]
            for attr_name in p.disallowed_attrs:
                n_A = n_classes_for_attr(dataset, attr_name)
                m_Y = n_classes_for_attr(dataset, task_name)
                yhat, y, A = materialize_predictions(
                    test_loader, encoder, task_heads, purpose_idx,
                    attr_name, task_name, DEVICE,
                )
                p_groups, yhat_groups, err_per_group = per_group_stats(
                    yhat, y, A, n_A, m_Y,
                )
                # Skip degenerate cells (a class with zero samples)
                groups_present = [a for a in range(n_A) if (A == a).any()]
                if len(groups_present) < 2:
                    continue
                p_groups_present = p_groups[groups_present]
                yhat_groups_present = yhat_groups[groups_present]
                err_per_group_present = err_per_group[groups_present]
                n_present = len(groups_present)
                B, _ = tv_barycenter(p_groups_present)
                d_dp = delta_dp(yhat_groups_present)
                F = B - (n_present - 1) * d_dp
                sum_err = float(err_per_group_present.sum())
                slack = sum_err - F
                # Fix 2: normalized (avg per-group) slack, in [0, 1-1/m].
                normalized_slack = slack / n_present
                # Fix 3: theoretical floor using Lemma 1's worst-case ΔDP bound.
                # Lemma 1 is derived for binary classification (Var(ŷ_binary) ≤ 1/4).
                # For multi-class tasks (m ≥ 3), the constant 0.2236 is not a
                # valid upper bound on ΔDP for an argmax classifier; we mark
                # F_theory as None on those cells.
                if m_Y == 2:
                    F_theory = B - (n_present - 1) * DELTA_DP_THEORY_MAX
                    theory_slack = sum_err - F_theory
                    theory_applicable = True
                else:
                    F_theory = None
                    theory_slack = None
                    theory_applicable = False
                cell = f"{dataset}/{p.name}/{attr_name}"
                row = {
                    "dataset": dataset, "purpose": p.name, "attribute": attr_name,
                    "task": task_name,
                    "n_groups_total": n_A, "n_groups_present": n_present,
                    "m_classes": m_Y,
                    "B_tv_barycenter": B,
                    "delta_DP": d_dp,
                    "F_floor": F,
                    "sum_Err": sum_err,
                    "slack": slack,
                    "normalized_slack": normalized_slack,
                    "F_theory": F_theory,
                    "theory_slack": theory_slack,
                    "theory_applicable": theory_applicable,
                    "delta_dp_theory_max": DELTA_DP_THEORY_MAX,
                    "per_group_err": err_per_group_present.tolist(),
                    "p_groups": p_groups_present.tolist(),
                    "yhat_groups": yhat_groups_present.tolist(),
                }
                rows.append(row)
                Ft_str = f"{F_theory:>+8.4f}" if F_theory is not None else "    N/A "
                print(f"{cell:<55s} {n_present:>3d} {m_Y:>3d} {B:>7.4f} "
                      f"{d_dp:>7.4f} {F:>+8.4f} {sum_err:>7.4f} {slack:>+8.4f} "
                      f"{normalized_slack:>+7.4f} {Ft_str}")
    # Save CSV
    out_dir = ROOT / "results/reviewer_dropins"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "zhao_gordon_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "purpose", "attribute", "task",
                    "n_groups_total", "n_groups_present", "m_classes",
                    "B", "Delta_DP", "F", "sum_Err", "slack",
                    "normalized_slack", "F_theory", "theory_slack"])
        for r in rows:
            Ft = f"{r['F_theory']:.6f}" if r['F_theory'] is not None else "N/A"
            ts = f"{r['theory_slack']:.6f}" if r['theory_slack'] is not None else "N/A"
            w.writerow([r["dataset"], r["purpose"], r["attribute"], r["task"],
                        r["n_groups_total"], r["n_groups_present"], r["m_classes"],
                        f"{r['B_tv_barycenter']:.6f}",
                        f"{r['delta_DP']:.6f}",
                        f"{r['F_floor']:.6f}",
                        f"{r['sum_Err']:.6f}",
                        f"{r['slack']:.6f}",
                        f"{r['normalized_slack']:.6f}",
                        Ft, ts])
    print(f"\nSaved CSV → {csv_path}")
    json_path = out_dir / "zhao_gordon_table.json"
    with open(json_path, "w") as f:
        json.dump({"config": {"seed": SEED, "device": DEVICE,
                              "ckpt_tag": CKPT_TAG_BY_DATASET},
                   "n_cells": len(rows), "rows": rows}, f, indent=2)
    print(f"Saved JSON → {json_path}")
    # Summary
    slacks = [r["slack"] for r in rows]
    floors = [r["F_floor"] for r in rows]
    bs = [r["B_tv_barycenter"] for r in rows]
    print(f"\n=== Summary across {len(rows)} cells (seed {SEED}) ===")
    print(f"  mean slack = {np.mean(slacks):+.4f}")
    print(f"  max  slack = {max(slacks):+.4f}")
    print(f"  min  slack = {min(slacks):+.4f}")
    print(f"  cells with slack < 0 (BOUND VIOLATED): "
          f"{sum(1 for s in slacks if s < 0)}/{len(rows)}")
    print(f"  cells with slack < 0.05 (very tight): "
          f"{sum(1 for s in slacks if 0 <= s < 0.05)}/{len(rows)}")
    print(f"  cells with F > 0 (informative bound): "
          f"{sum(1 for f in floors if f > 0)}/{len(rows)}")
    print(f"  cells with F > 0.05 (non-trivial floor): "
          f"{sum(1 for f in floors if f > 0.05)}/{len(rows)}")
    print(f"  TV-barycenter B distribution: min={min(bs):.4f}, "
          f"median={np.median(bs):.4f}, max={max(bs):.4f}")
    # Fix 2 + 3 summaries
    norms = [r["normalized_slack"] for r in rows]
    F_th = [r["F_theory"] for r in rows if r["theory_applicable"]]
    th_slacks = [r["theory_slack"] for r in rows if r["theory_applicable"]]
    n_binary_y = sum(1 for r in rows if r["theory_applicable"])
    print(f"\n  normalized slack (slack/n): mean={np.mean(norms):+.4f}, "
          f"max={max(norms):+.4f}, min={min(norms):+.4f}")
    print(f"  Lemma 1 / theoretical floor restricted to binary-Y cells "
          f"(m=2): {n_binary_y}/{len(rows)} cells")
    print(f"  binary-Y cells with theory floor F_theory > 0: "
          f"{sum(1 for f in F_th if f > 0)}/{n_binary_y}")
    print(f"  binary-Y cells with theory_slack < 0 (theory bound violated): "
          f"{sum(1 for s in th_slacks if s < 0)}/{n_binary_y}")
    print(f"  ΔDP_theory_max constant = {DELTA_DP_THEORY_MAX:.4f} "
          f"(from Lemma 1 with ε=0.05, π=0.5, m=2)")


if __name__ == "__main__":
    main()

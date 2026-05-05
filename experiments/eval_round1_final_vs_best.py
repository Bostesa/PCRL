"""Evaluate Round 1 v2 Adult checkpoints (final.pt vs best.pt) per seed.

Loads both checkpoints for seeds 0/1/2 and runs the standard auditor
(linear R² + post-hoc MLP delta) on each (purpose, attr) pair. Writes
results/v2_adult_ROUND1/final_vs_best.md and prints a verdict on
whether the 0/8 pass count is a selection-criterion bug or a real
optimizer/architecture limitation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import generate_report
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry

SEEDS = [0, 1, 2]
DEVICE = "cpu"
THRESHOLD = 0.05
DELTA_THRESHOLD = 0.02


def build(purposes, train_ds):
    backbone = StandardEncoder(
        input_dim=train_ds.info.num_features, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes),
        rank=8, alpha=16.0, dropout=0.0,
    )
    task_heads = {}
    for p in purposes:
        out_dim = p.allowed_task_dims.get(p.allowed_tasks[0], 2)
        task_heads[p.name] = TaskHead(repr_dim=64, output_dim=out_dim)
    return encoder, task_heads


def load_ckpt(encoder, task_heads, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    th_module = torch.nn.ModuleDict(task_heads)
    th_module.load_state_dict(ckpt["task_heads"])
    state = ckpt.get("state", {})
    history = ckpt.get("history", {})
    return state, history


def eval_pairs(encoder, task_heads, train_loader, test_loader, registry):
    encoder.eval()
    reports = generate_report(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=DEVICE,
    )
    rows = []
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        rows.append({
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "linear_r2": float(r.linear_r2),
            "empirical_best_acc": float(r.empirical_best_acc),
            "majority_baseline": float(r.majority_proportion),
            "delta": float(delta),
            "adj_pass": bool(delta < DELTA_THRESHOLD and r.linear_r2 < THRESHOLD),
        })
    return rows


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    purposes = get_adult_purposes()
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False,
                           norm_stats=train_ds.norm_stats)

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    out: dict = {"per_seed": {}}

    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        ckpt_dir = ROOT / "checkpoints" / f"v2_adult_s{seed}"
        torch.manual_seed(seed)
        np.random.seed(seed)
        encoder, task_heads = build(purposes, train_ds)

        seed_out = {}
        for tag in ["best", "final"]:
            path = ckpt_dir / f"{tag}.pt"
            print(f"  loading {tag}.pt")
            state, history = load_ckpt(encoder, task_heads, path)
            rows = eval_pairs(encoder, task_heads, train_loader, test_loader, registry)
            pass_count = sum(int(r["adj_pass"]) for r in rows)
            mean_r2 = float(np.mean([r["linear_r2"] for r in rows]))
            print(f"    pass {pass_count}/8, mean linear R²={mean_r2:.4f}")
            for r in rows:
                tick = "✓" if r["adj_pass"] else " "
                print(f"      [{tick}] {r['purpose']:<22s} {r['attribute']:<14s} R²={r['linear_r2']:.4f}  Δ={r['delta']:.4f}")
            seed_out[tag] = {
                "pass_count": pass_count,
                "mean_linear_r2": mean_r2,
                "rows": rows,
                "epoch": int(state.get("epoch", -1)),
                "best_epoch": int(state.get("best_epoch", -1)),
                "best_val_loss": float(state.get("best_val_loss", float("nan"))),
            }
            # Brief history slice
            comp = history.get("val_composite", [])
            viol = history.get("val_violation_sum", [])
            tloss = history.get("val_task_loss", [])
            seed_out[tag]["history_summary"] = {
                "n_epochs": len(comp),
                "composite_first5": comp[:5],
                "composite_last5": comp[-5:],
                "violation_first5": viol[:5],
                "violation_last5": viol[-5:],
                "task_loss_first5": tloss[:5],
                "task_loss_last5": tloss[-5:],
            }
        out["per_seed"][seed] = seed_out

    out_dir = ROOT / "results" / "v2_adult_ROUND1"
    json_path = out_dir / "final_vs_best.json"
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()

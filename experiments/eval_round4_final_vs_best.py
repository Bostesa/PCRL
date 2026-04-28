"""Evaluate Round 4 v2 Adult checkpoints (final.pt vs best.pt) per seed.

Loads both checkpoints for seeds 0/1/2 and runs the standard auditor
(linear R² + post-hoc MLP delta) on each (purpose, attr) pair. Writes
results/v2_adult_ROUND4/final_vs_best.md (+ .json) and prints the
verdict per the band logic:

  GREEN: final R² < 0.05 on 5+/8 pairs reproducibly
         → Cotter selection is the bug; optimizer worked.
  YELLOW: final R² in 0.05–0.10
         → Real progress past epoch 12 but didn't reach strict threshold;
           honest SOTA at realigned threshold.
  RED:    final R² also ~0.17
         → 200 epochs didn't improve over 12; pivot needed.
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
    return state


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


def verdict_band(final_r2_per_pair: list[float]) -> tuple[str, str]:
    """Apply the 3-band rule on per-pair final.pt R² across all (seed, pair)."""
    arr = np.array(final_r2_per_pair)
    mean_r2 = float(arr.mean())
    n_below_05 = int((arr < 0.05).sum())
    n_total = len(arr)
    # GREEN: 5+ /8 pairs reproducibly below 0.05 (so >= 15/24 across seeds)
    green_threshold_per_seed = 5
    n_seeds = 3
    n_pairs_per_seed = n_total // n_seeds
    if n_below_05 >= green_threshold_per_seed * n_seeds:
        return "GREEN", (
            f"final.pt R² < 0.05 on {n_below_05}/{n_total} pair-seeds "
            f"(threshold: {green_threshold_per_seed * n_seeds}). "
            "Cotter selection is the bug; the optimizer worked. "
            "Recommendation: change selection rule and re-evaluate from "
            "existing checkpoints."
        )
    if mean_r2 < 0.10:
        return "YELLOW", (
            f"mean final.pt R² = {mean_r2:.3f} ∈ [0.05, 0.10). Real progress "
            "past epoch 12 but didn't reach strict R²<0.05. "
            "Recommendation: honest SOTA at realigned threshold (R²<0.10 or "
            "R²<0.15) per the literature deep research."
        )
    if mean_r2 < 0.15:
        return "YELLOW-WIDE", (
            f"mean final.pt R² = {mean_r2:.3f} ∈ [0.10, 0.15). Marginal "
            "improvement; threshold realignment to R²<0.15 may be defensible "
            "but result is borderline."
        )
    return "RED", (
        f"mean final.pt R² = {mean_r2:.3f} ≥ 0.15. 200 epochs didn't "
        "meaningfully improve over the early Cotter-selected iterate. "
        "Pivot needed: probably need a different optimization regime "
        "(e.g. AL with adaptive ρ, longer schedule, or warm-restart "
        "from LEACE init with stronger constraint pressure)."
    )


def md_table(per_seed_results: dict, key: str) -> str:
    """Produce the (Pair × seed × {best,final}) markdown table."""
    rows = per_seed_results[0][key]["rows"]
    pair_keys = [f"{r['purpose']}/{r['attribute']}" for r in rows]
    header = "| Pair | s0 best | s0 final | s1 best | s1 final | s2 best | s2 final |"
    sep    = "|------|---------|----------|---------|----------|---------|----------|"
    lines = [header, sep]
    for i, pair in enumerate(pair_keys):
        cells = [pair]
        for s in SEEDS:
            for tag in ["best", "final"]:
                r = per_seed_results[s][tag]["rows"][i]["linear_r2"]
                p = per_seed_results[s][tag]["rows"][i]["adj_pass"]
                tick = "✓" if p else " "
                cells.append(f"{tick}{r:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
            state = load_ckpt(encoder, task_heads, path)
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
            }
        out["per_seed"][seed] = seed_out

    # Verdict on final.pt R² across all (seed, pair).
    final_r2_all = []
    for s in SEEDS:
        for r in out["per_seed"][s]["final"]["rows"]:
            final_r2_all.append(r["linear_r2"])
    band, recommendation = verdict_band(final_r2_all)
    final_mean = float(np.mean(final_r2_all))
    final_n_below_05 = int(sum(1 for x in final_r2_all if x < 0.05))
    final_n_below_10 = int(sum(1 for x in final_r2_all if x < 0.10))
    out["verdict"] = {
        "band": band,
        "recommendation": recommendation,
        "final_mean_r2": final_mean,
        "final_n_below_05": final_n_below_05,
        "final_n_below_10": final_n_below_10,
        "final_n_total": len(final_r2_all),
    }

    # Per-seed pass counts at final.pt
    final_pass_counts = [out["per_seed"][s]["final"]["pass_count"] for s in SEEDS]
    best_pass_counts = [out["per_seed"][s]["best"]["pass_count"] for s in SEEDS]
    out["pass_counts"] = {"best": best_pass_counts, "final": final_pass_counts}

    out_dir = ROOT / "results" / "v2_adult_ROUND4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_vs_best.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # Markdown report
    md = []
    md.append("# Round 4 — final.pt vs best.pt re-evaluation")
    md.append("")
    md.append(f"**Verdict: {band}**  (mean final R² = {final_mean:.3f}; "
              f"{final_n_below_05}/{len(final_r2_all)} pair-seeds < 0.05; "
              f"{final_n_below_10}/{len(final_r2_all)} < 0.10)")
    md.append("")
    md.append(f"{recommendation}")
    md.append("")
    md.append("## Per-pair linear R² (auditor)")
    md.append("")
    md.append("Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).")
    md.append("")
    md.append(md_table(out["per_seed"], "best"))
    md.append("")
    md.append("## Per-seed pass counts and mean R²")
    md.append("")
    md.append("| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |")
    md.append("|------|--------------|------------------|----------------|-------------------|")
    for s in SEEDS:
        bb = out["per_seed"][s]["best"]
        ff = out["per_seed"][s]["final"]
        md.append(f"| {s} | {bb['pass_count']}/8 | {bb['mean_linear_r2']:.4f} | "
                  f"{ff['pass_count']}/8 | {ff['mean_linear_r2']:.4f} |")
    md.append("")
    md.append("## Aggregate")
    md.append("")
    md.append(f"- best.pt mean R² across 24 pair-seeds: "
              f"{np.mean([r['linear_r2'] for s in SEEDS for r in out['per_seed'][s]['best']['rows']]):.4f}")
    md.append(f"- final.pt mean R² across 24 pair-seeds: {final_mean:.4f}")
    md.append(f"- final.pt R² < 0.05: {final_n_below_05}/{len(final_r2_all)} pair-seeds")
    md.append(f"- final.pt R² < 0.10: {final_n_below_10}/{len(final_r2_all)} pair-seeds")
    md.append("")
    (out_dir / "final_vs_best.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_dir / 'final_vs_best.md'}")
    print(f"\nVERDICT: {band}")
    print(f"  mean final.pt R² = {final_mean:.4f}")
    print(f"  {final_n_below_05}/{len(final_r2_all)} pair-seeds < 0.05")
    print(f"  {final_n_below_10}/{len(final_r2_all)} pair-seeds < 0.10")
    print(f"  recommendation: {recommendation}")


if __name__ == "__main__":
    main()

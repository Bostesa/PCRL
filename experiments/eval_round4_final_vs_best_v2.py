"""Generic Round 4 final.pt vs best.pt re-evaluation.

Adapted from eval_round4_final_vs_best.py to take --dataset {adult,diabetes,hmda}.
Loads both checkpoints for each seed, runs the auditor, applies the verdict
band rule on per-pair linear R² across all (seed, pair) combinations.

Verdict bands (per-pair R² across all (seed,pair) combinations):
  GREEN:        >= ceil(0.625 * pairs_per_seed) pair-seeds with R² < 0.05
                (matches Adult's "5+/8" rule for any pairs_per_seed)
  YELLOW:       mean R² ∈ [0.05, 0.10)
  YELLOW-WIDE:  mean R² ∈ [0.10, 0.15)
  RED:          mean R² >= 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def load_ds(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
        test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False,
                               norm_stats=train_ds.norm_stats)
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
        test_ds = HMDADataset(purposes=purposes, root="data", split="test")
    else:
        raise ValueError(name)
    return purposes, train_ds, test_ds


def build(purposes, train_ds, rank: int = 8, alpha: float = 16.0):
    backbone = StandardEncoder(
        input_dim=train_ds.info.num_features, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes),
        rank=rank, alpha=alpha, dropout=0.0,
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
    return ckpt.get("state", {})


def eval_pairs(encoder, train_loader, test_loader, registry):
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


def verdict_band(final_r2_per_pair: list[float], pairs_per_seed: int, n_seeds: int) -> tuple[str, str]:
    arr = np.array(final_r2_per_pair)
    mean_r2 = float(arr.mean())
    n_below_05 = int((arr < 0.05).sum())
    n_total = len(arr)
    green_threshold_per_seed = math.ceil(0.625 * pairs_per_seed)
    green_threshold_total = green_threshold_per_seed * n_seeds
    if n_below_05 >= green_threshold_total:
        return "GREEN", (
            f"final.pt R² < 0.05 on {n_below_05}/{n_total} pair-seeds "
            f"(threshold {green_threshold_total}={green_threshold_per_seed}/{pairs_per_seed} per seed × {n_seeds}). "
            "Cotter selection is the bug; the optimizer worked. "
            "Recommendation: claim from final.pt as canonical."
        )
    if mean_r2 < 0.10:
        return "YELLOW", (
            f"mean final.pt R² = {mean_r2:.3f} ∈ [0.05, 0.10). "
            f"{n_below_05}/{n_total} pair-seeds < 0.05. "
            "Recommendation: honest SOTA at realigned R²<0.10 threshold."
        )
    if mean_r2 < 0.15:
        return "YELLOW-WIDE", (
            f"mean final.pt R² = {mean_r2:.3f} ∈ [0.10, 0.15). "
            "Marginal improvement; threshold realignment to R²<0.15 may be defensible."
        )
    return "RED", (
        f"mean final.pt R² = {mean_r2:.3f} ≥ 0.15. "
        "200 epochs didn't meaningfully improve on best.pt. Pivot needed."
    )


def md_table(per_seed_results: dict, pair_keys: list[str]) -> str:
    seeds = sorted(per_seed_results.keys())
    cells_header = "| Pair |"
    sep = "|------|"
    for s in seeds:
        cells_header += f" s{s} best | s{s} final |"
        sep += "---------|----------|"
    lines = [cells_header, sep]
    for i, pair in enumerate(pair_keys):
        cells = [pair]
        for s in seeds:
            for tag in ["best", "final"]:
                row = per_seed_results[s][tag]["rows"][i]
                tick = "✓" if row["adj_pass"] else " "
                cells.append(f"{tick}{row['linear_r2']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["adult", "diabetes", "hmda"])
    ap.add_argument("--out-tag", default="_ROUND4")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    purposes, train_ds, test_ds = load_ds(args.dataset)
    pairs_per_seed = sum(len(p.disallowed_attrs) for p in purposes)

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    out: dict = {"per_seed": {}, "dataset": args.dataset, "pairs_per_seed": pairs_per_seed}

    # If checkpoints landed under the run-tagged directory (Round 5 onward,
    # after run_v2_dataset.py started threading out_tag into ckpt_dir), use
    # those. Otherwise fall back to the legacy path (Round 4 and earlier).
    tagged_ck = ROOT / "checkpoints" / f"v2_{args.dataset}{args.out_tag}_s0"
    legacy_ck = ROOT / "checkpoints" / f"v2_{args.dataset}_s0"
    if tagged_ck.exists():
        ckpt_pattern = f"v2_{args.dataset}{args.out_tag}_s{{seed}}"
    elif legacy_ck.exists():
        ckpt_pattern = f"v2_{args.dataset}_s{{seed}}"
    else:
        raise SystemExit(
            f"no checkpoints found at {tagged_ck} or {legacy_ck}"
        )
    print(f"  using ckpt pattern: {ckpt_pattern}")

    # Diabetes Round 5 uses LoRA rank 16 alpha 32 (per-dataset override in
    # run_v2_dataset.py LORA_BY_DATASET); Adult/HMDA use rank 8 alpha 16.
    # Match the trainer's per-dataset shape so state_dict load doesn't
    # raise size-mismatch on Diabetes.
    DEFAULT_RANK_BY_DATASET = {"adult": (8, 16.0), "hmda": (8, 16.0), "diabetes": (16, 32.0)}
    rank, alpha = DEFAULT_RANK_BY_DATASET.get(args.dataset, (8, 16.0))
    # Round 4 Diabetes was rank=8; if loading that legacy checkpoint, fall
    # back. We detect by peeking at the checkpoint A.weight shape.
    if args.out_tag == "_ROUND4" and args.dataset == "diabetes":
        rank, alpha = 8, 16.0

    for seed in SEEDS:
        print(f"\n=== {args.dataset} seed={seed} ===")
        ckpt_dir = ROOT / "checkpoints" / ckpt_pattern.format(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        encoder, task_heads = build(purposes, train_ds, rank=rank, alpha=alpha)

        seed_out = {}
        for tag in ["best", "final"]:
            path = ckpt_dir / f"{tag}.pt"
            print(f"  loading {tag}.pt")
            state = load_ckpt(encoder, task_heads, path)
            rows = eval_pairs(encoder, train_loader, test_loader, registry)
            pass_count = sum(int(r["adj_pass"]) for r in rows)
            mean_r2 = float(np.mean([r["linear_r2"] for r in rows]))
            print(f"    pass {pass_count}/{pairs_per_seed}, mean R²={mean_r2:.4f}")
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

    final_r2_all: list[float] = []
    for s in SEEDS:
        for r in out["per_seed"][s]["final"]["rows"]:
            final_r2_all.append(r["linear_r2"])
    band, recommendation = verdict_band(final_r2_all, pairs_per_seed, len(SEEDS))
    final_mean = float(np.mean(final_r2_all))
    final_n_below_05 = int(sum(1 for x in final_r2_all if x < 0.05))
    final_n_below_10 = int(sum(1 for x in final_r2_all if x < 0.10))
    out["verdict"] = {
        "band": band, "recommendation": recommendation,
        "final_mean_r2": final_mean,
        "final_n_below_05": final_n_below_05,
        "final_n_below_10": final_n_below_10,
        "final_n_total": len(final_r2_all),
    }
    final_pass_counts = [out["per_seed"][s]["final"]["pass_count"] for s in SEEDS]
    best_pass_counts = [out["per_seed"][s]["best"]["pass_count"] for s in SEEDS]
    out["pass_counts"] = {"best": best_pass_counts, "final": final_pass_counts}

    out_dir = ROOT / "results" / f"v2_{args.dataset}{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_vs_best.json", "w") as fh:
        json.dump(out, fh, indent=2)

    pair_keys = [f"{r['purpose']}/{r['attribute']}" for r in out["per_seed"][SEEDS[0]]["best"]["rows"]]
    md = []
    md.append(f"# {args.dataset.upper()} Round 4 — final.pt vs best.pt re-evaluation")
    md.append("")
    md.append(f"**Verdict: {band}**  (mean final R² = {final_mean:.3f}; "
              f"{final_n_below_05}/{len(final_r2_all)} pair-seeds < 0.05; "
              f"{final_n_below_10}/{len(final_r2_all)} < 0.10)")
    md.append("")
    md.append(recommendation)
    md.append("")
    md.append("## Per-pair linear R² (auditor)")
    md.append("")
    md.append("Cell: `<adj_pass tick><R²>`. ✓ = passes paper criterion (R² < 0.05 AND Δ < 0.02 on post-hoc MLP).")
    md.append("")
    md.append(md_table(out["per_seed"], pair_keys))
    md.append("")
    md.append("## Per-seed pass counts and mean R²")
    md.append("")
    md.append(f"| Seed | best.pt pass | best.pt mean R² | final.pt pass | final.pt mean R² |")
    md.append(f"|------|--------------|------------------|----------------|-------------------|")
    for s in SEEDS:
        bb = out["per_seed"][s]["best"]
        ff = out["per_seed"][s]["final"]
        md.append(f"| {s} | {bb['pass_count']}/{pairs_per_seed} | {bb['mean_linear_r2']:.4f} | "
                  f"{ff['pass_count']}/{pairs_per_seed} | {ff['mean_linear_r2']:.4f} |")
    md.append("")
    md.append("## Aggregate")
    md.append("")
    n_total = len(final_r2_all)
    md.append(f"- best.pt mean R² across {n_total} pair-seeds: "
              f"{np.mean([r['linear_r2'] for s in SEEDS for r in out['per_seed'][s]['best']['rows']]):.4f}")
    md.append(f"- final.pt mean R² across {n_total} pair-seeds: {final_mean:.4f}")
    md.append(f"- final.pt R² < 0.05: {final_n_below_05}/{n_total}")
    md.append(f"- final.pt R² < 0.10: {final_n_below_10}/{n_total}")
    md.append("")
    (out_dir / "final_vs_best.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_dir / 'final_vs_best.md'}")
    print(f"VERDICT: {band}")
    print(f"  mean final.pt R² = {final_mean:.4f}")
    print(f"  {final_n_below_05}/{n_total} pair-seeds < 0.05")
    print(f"  recommendation: {recommendation}")


if __name__ == "__main__":
    main()

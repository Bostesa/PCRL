#!/usr/bin/env python3
"""Framework D — Dominant-Axis Auditing.

Re-evaluates Round 4 ``final.pt`` checkpoints across Adult / Diabetes / HMDA
without retraining and computes:

  • R²_onehot — standard linear regression R² on the one-hot encoded
    multi-class sensitive attribute (matches LinearComplianceCertificate).
  • R²_DA = max_k R²_OvR_k — dominant-axis R².
  • per-class OvR R² and empirical priors π_k.
  • MLP-DA delta — best test accuracy minus binary majority baseline over
    the K one-vs-rest binary partitions.

Convex-Combination Identity:
    R²_onehot = Σ_k w_k · R²_OvR_k,   w_k = π_k(1-π_k) / Σ_j π_j(1-π_j)

Outputs:
  results/v2_<dataset>_ROUND4/dominant_axis_audit.json   per (seed, pair)
  results/V2_DOMINANT_AXIS_SUMMARY.md                    cross-dataset table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.certificates import (  # noqa: E402
    _extract_representations_and_labels,
    compute_dominant_axis_r2,
    compute_mlp_ovr_delta,
)
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import PerPurposeLoRAEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry  # noqa: E402
from pcrl.purposes.verification import LinearComplianceCertificate  # noqa: E402

SEEDS = [0, 1, 2]
DATASETS = ["adult", "diabetes", "hmda"]
DEVICE = "cpu"


def load_ds(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                                split="train", download=False)
        test_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                               split="test", download=False,
                               norm_stats=train_ds.norm_stats)
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="train")
        test_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="test")
    else:
        raise ValueError(name)
    return purposes, train_ds, test_ds


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
    return ckpt.get("state", {})


def convex_combo_predicted_r2(per_class_r2: list[float], priors: list[float]) -> float:
    """Predict R²_onehot from OvR R²s via the convex-combination identity."""
    arr = np.asarray(per_class_r2)
    p = np.asarray(priors)
    w = p * (1.0 - p)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float(np.dot(w, arr))


def eval_one_seed(dataset: str, seed: int, mlp_epochs: int, batch_size: int) -> dict:
    purposes, train_ds, test_ds = load_ds(dataset)
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder, task_heads = build(purposes, train_ds)
    ckpt_path = ROOT / "checkpoints" / f"v2_{dataset}_s{seed}" / "final.pt"
    state = load_ckpt(encoder, task_heads, ckpt_path)
    encoder.eval()

    # Pre-compute purpose -> attrs
    purpose_attrs: dict[int, list[str]] = {}
    for idx, purpose in enumerate(purposes):
        if purpose.disallowed_attrs:
            purpose_attrs[idx] = list(purpose.disallowed_attrs)

    train_cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    test_cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for purpose_idx in purpose_attrs:
        attrs = purpose_attrs[purpose_idx]
        train_cache[purpose_idx] = _extract_representations_and_labels(
            encoder, train_loader, purpose_idx, attrs, DEVICE,
        )
        test_cache[purpose_idx] = _extract_representations_and_labels(
            encoder, test_loader, purpose_idx, attrs, DEVICE,
        )

    cert = LinearComplianceCertificate(epsilon=0.05, regularization=1e-6)
    rows = []
    for purpose_idx, purpose in enumerate(purposes):
        if not purpose.disallowed_attrs:
            continue
        for attr_name in purpose.disallowed_attrs:
            train_reprs = train_cache[purpose_idx][0]
            test_reprs = test_cache[purpose_idx][0]
            train_labels = train_cache[purpose_idx][1][attr_name]
            test_labels = test_cache[purpose_idx][1][attr_name]

            num_classes = int(max(train_labels.max(), test_labels.max())) + 1

            t0 = time.time()
            r2_onehot = cert.check(test_reprs, test_labels).r_squared
            da = compute_dominant_axis_r2(test_reprs, test_labels)
            t_linear = time.time() - t0

            mlp_da: dict | None = None
            t_mlp = 0.0
            if num_classes > 2:
                t0 = time.time()
                mlp_da = compute_mlp_ovr_delta(
                    train_reprs, train_labels, test_reprs, test_labels,
                    hidden=256, epochs=mlp_epochs, lr=1e-3, dropout=0.3,
                    batch_size=batch_size, device=DEVICE, random_state=seed,
                )
                t_mlp = time.time() - t0

            predicted_r2 = convex_combo_predicted_r2(da["per_class_r2"], da["priors"])

            row = {
                "purpose": purpose.name,
                "attribute": attr_name,
                "num_classes": num_classes,
                "r2_onehot": float(r2_onehot),
                "r2_da": float(da["r2_da"]),
                "r2_da_argmax_class": int(da["argmax_class"]),
                "per_class_r2": [float(x) for x in da["per_class_r2"]],
                "priors": [float(x) for x in da["priors"]],
                "predicted_r2_onehot_from_convex_combo": predicted_r2,
                "convex_combo_residual": float(predicted_r2 - r2_onehot),
                "is_multiclass": num_classes > 2,
                "mlp_da_delta": (
                    float(mlp_da["mlp_da_delta"]) if mlp_da is not None else None
                ),
                "mlp_da_argmax_class": (
                    int(mlp_da["argmax_class"]) if mlp_da is not None else -1
                ),
                "mlp_per_class_delta": (
                    [float(x) for x in mlp_da["per_class_delta"]]
                    if mlp_da is not None else []
                ),
                "mlp_per_class_acc": (
                    [float(x) for x in mlp_da["per_class_acc"]]
                    if mlp_da is not None else []
                ),
                "wall_seconds": {"linear": t_linear, "mlp": t_mlp},
            }
            rows.append(row)
            tag = f"K={num_classes}" + (" *MC*" if num_classes > 2 else "")
            mlp_str = (
                f"  MLP-DA={mlp_da['mlp_da_delta']:+.4f} (argmax={mlp_da['argmax_class']})"
                if mlp_da is not None else "  MLP-DA=N/A (binary)"
            )
            print(f"    [{purpose.name:<22s} | {attr_name:<14s} | {tag:<8s}] "
                  f"R²_onehot={r2_onehot:.4f}  R²_DA={da['r2_da']:.4f}  "
                  f"argmax={da['argmax_class']}  Δ={float(da['r2_da'])-float(r2_onehot):+.4f}"
                  f"{mlp_str}")

    return {"epoch": int(state.get("epoch", -1)), "rows": rows}


def write_dataset_json(dataset: str, per_seed: dict, mlp_epochs: int) -> Path:
    out_dir = ROOT / "results" / f"v2_{dataset}_ROUND4"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dominant_axis_audit.json"
    with open(path, "w") as fh:
        json.dump({
            "dataset": dataset,
            "seeds": list(per_seed.keys()),
            "mlp_epochs": mlp_epochs,
            "per_seed": per_seed,
        }, fh, indent=2)
    return path


def aggregate_dataset(per_seed: dict) -> dict:
    """Per-dataset aggregates across all (seed, pair) rows."""
    all_rows = []
    for s in per_seed:
        all_rows.extend(per_seed[s]["rows"])

    multi = [r for r in all_rows if r["is_multiclass"]]

    def _arr(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    out: dict = {
        "n_pair_seeds": len(all_rows),
        "n_multiclass_pair_seeds": len(multi),
        "mean_r2_onehot": float(_arr(all_rows, "r2_onehot").mean()) if all_rows else 0.0,
        "mean_r2_da": float(_arr(all_rows, "r2_da").mean()) if all_rows else 0.0,
        "mean_gap_da_minus_onehot": (
            float((_arr(all_rows, "r2_da") - _arr(all_rows, "r2_onehot")).mean())
            if all_rows else 0.0
        ),
        # Cases where the new metric exposes leakage that the old hides:
        "n_da_above_05_onehot_below_05": int(sum(
            1 for r in all_rows if r["r2_da"] > 0.05 and r["r2_onehot"] <= 0.05
        )),
        # Identity validation (multi-class only — binary is trivial):
        "convex_combo_match_within_001_count": int(sum(
            1 for r in multi if abs(r["convex_combo_residual"]) < 0.01
        )),
        "convex_combo_residual_max": (
            float(np.max(np.abs(_arr(multi, "convex_combo_residual"))))
            if multi else 0.0
        ),
    }

    if multi:
        out["multiclass_only"] = {
            "mean_r2_onehot": float(_arr(multi, "r2_onehot").mean()),
            "mean_r2_da": float(_arr(multi, "r2_da").mean()),
            "mean_gap": float((_arr(multi, "r2_da") - _arr(multi, "r2_onehot")).mean()),
            "n_da_above_05_onehot_below_05": int(sum(
                1 for r in multi if r["r2_da"] > 0.05 and r["r2_onehot"] <= 0.05
            )),
        }
        # MLP-DA aggregates (multi-class only)
        mlp = [r for r in multi if r["mlp_da_delta"] is not None]
        if mlp:
            out["multiclass_only"]["mean_mlp_da_delta"] = float(
                np.mean([r["mlp_da_delta"] for r in mlp])
            )

    return out


def write_summary_md(dataset_results: dict, path: Path) -> None:
    """Cross-dataset summary in markdown."""
    lines: list[str] = []
    lines.append("# Framework D — Dominant-Axis Auditing — Cross-Dataset Summary")
    lines.append("")
    lines.append("Re-evaluation of Round 4 `final.pt` checkpoints with R²_DA = max_k R²_OvR_k.")
    lines.append("")
    lines.append("## Per-dataset gap (mean across seed × pair)")
    lines.append("")
    lines.append("| Dataset | Pair-seeds | Mean R²_onehot | Mean R²_DA | Gap (R²_DA − R²_onehot) | # hidden by R²_onehot ≤ 0.05 (R²_DA > 0.05) |")
    lines.append("|---------|------------|----------------|------------|--------------------------|---------------------------------------------|")
    total_pair_seeds = 0
    total_hidden = 0
    for ds in dataset_results:
        agg = dataset_results[ds]["aggregate"]
        lines.append(
            f"| {ds} | {agg['n_pair_seeds']} | "
            f"{agg['mean_r2_onehot']:.4f} | {agg['mean_r2_da']:.4f} | "
            f"{agg['mean_gap_da_minus_onehot']:+.4f} | "
            f"{agg['n_da_above_05_onehot_below_05']} |"
        )
        total_pair_seeds += agg["n_pair_seeds"]
        total_hidden += agg["n_da_above_05_onehot_below_05"]
    lines.append(f"| **Total** | {total_pair_seeds} | — | — | — | **{total_hidden}** |")
    lines.append("")

    lines.append("## Multi-class subset only (where R²_DA can disagree with R²_onehot)")
    lines.append("")
    lines.append("| Dataset | MC pair-seeds | Mean R²_onehot | Mean R²_DA | Gap | Mean MLP-DA Δ |")
    lines.append("|---------|---------------|----------------|------------|-----|----------------|")
    for ds in dataset_results:
        agg = dataset_results[ds]["aggregate"]
        mc = agg.get("multiclass_only")
        if not mc:
            lines.append(f"| {ds} | 0 | — | — | — | — |")
            continue
        mlp_str = f"{mc['mean_mlp_da_delta']:+.4f}" if "mean_mlp_da_delta" in mc else "—"
        lines.append(
            f"| {ds} | {agg['n_multiclass_pair_seeds']} | "
            f"{mc['mean_r2_onehot']:.4f} | {mc['mean_r2_da']:.4f} | "
            f"{mc['mean_gap']:+.4f} | {mlp_str} |"
        )
    lines.append("")

    lines.append("## Convex-combination identity validation")
    lines.append("")
    lines.append(
        "The Convex-Combination Identity predicts "
        "`R²_onehot = Σ_k w_k · R²_OvR_k` with `w_k = π_k(1-π_k)/Σ_j π_j(1-π_j)`. "
        "Per multi-class pair-seed we compare the predicted vs observed one-hot R²:"
    )
    lines.append("")
    lines.append("| Dataset | MC pair-seeds | Match (residual < 0.01) | Max abs residual |")
    lines.append("|---------|---------------|--------------------------|-------------------|")
    total_mc = 0
    total_match = 0
    for ds in dataset_results:
        agg = dataset_results[ds]["aggregate"]
        n_mc = agg["n_multiclass_pair_seeds"]
        n_match = agg["convex_combo_match_within_001_count"]
        max_res = agg["convex_combo_residual_max"]
        total_mc += n_mc
        total_match += n_match
        lines.append(
            f"| {ds} | {n_mc} | {n_match}/{n_mc} | {max_res:.4f} |"
        )
    pct = (100.0 * total_match / total_mc) if total_mc else 0.0
    lines.append(
        f"| **Total** | {total_mc} | **{total_match}/{total_mc} ({pct:.1f}%)** | — |"
    )
    lines.append("")

    # Worked example for Adult income/race (s0)
    adult = dataset_results.get("adult")
    if adult is not None:
        per_seed = adult["per_seed"]
        # Find row for income/race in seed 0 if present
        worked = None
        if 0 in per_seed:
            for r in per_seed[0]["rows"]:
                if r["attribute"] == "race" and (
                    "income" in r["purpose"].lower() or r["purpose"] == "tax_admin"
                ):
                    worked = r
                    break
            if worked is None:
                # Fallback: any race row
                for r in per_seed[0]["rows"]:
                    if r["attribute"] == "race":
                        worked = r
                        break
        if worked is not None:
            lines.append("## Worked example — Adult `<income-purpose>` × race (seed 0)")
            lines.append("")
            lines.append(f"- Purpose: `{worked['purpose']}`")
            lines.append(f"- Empirical priors π = {[round(x, 4) for x in worked['priors']]}")
            lines.append(f"- Per-class R²_OvR = {[round(x, 4) for x in worked['per_class_r2']]}")
            lines.append(f"- Predicted R²_onehot from convex combo = `{worked['predicted_r2_onehot_from_convex_combo']:.4f}`")
            lines.append(f"- Observed R²_onehot = `{worked['r2_onehot']:.4f}`")
            denom = max(abs(worked["r2_onehot"]), 1e-12)
            ratio = worked["predicted_r2_onehot_from_convex_combo"] / denom
            lines.append(f"- Predicted / observed ratio = `{ratio:.4f}` (1.00 = exact identity)")
            lines.append(f"- R²_DA = `{worked['r2_da']:.4f}` at argmax class = `{worked['r2_da_argmax_class']}`")
            lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--mlp-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    dataset_results: dict[str, dict] = {}
    overall_t0 = time.time()
    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        per_seed: dict = {}
        for seed in args.seeds:
            print(f"  -- seed {seed} --")
            t0 = time.time()
            per_seed[seed] = eval_one_seed(dataset, seed, args.mlp_epochs, args.batch_size)
            print(f"  seed {seed} done in {time.time()-t0:.1f}s")
        path = write_dataset_json(dataset, per_seed, args.mlp_epochs)
        agg = aggregate_dataset(per_seed)
        dataset_results[dataset] = {"per_seed": per_seed, "aggregate": agg}
        print(f"  wrote {path}")
        print(f"  aggregate: mean R²_onehot={agg['mean_r2_onehot']:.4f}  "
              f"mean R²_DA={agg['mean_r2_da']:.4f}  gap={agg['mean_gap_da_minus_onehot']:+.4f}  "
              f"hidden_by_onehot={agg['n_da_above_05_onehot_below_05']}/{agg['n_pair_seeds']}")

    out_md = ROOT / "results" / "V2_DOMINANT_AXIS_SUMMARY.md"
    write_summary_md(dataset_results, out_md)
    print(f"\nWrote {out_md}")
    print(f"Total wall time: {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    main()

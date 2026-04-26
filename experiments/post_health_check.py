#!/usr/bin/env python3
"""Post-rerun health check: load PCRL seed-0 best.pt, recompute representation
diagnostics + per-task accuracy on test, write health_check.json into the
results dir, and prepend a STATUS: HEALTHY|COLLAPSED row to summary.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.models.auditor import MultiAttributeAuditor  # noqa: E402
from pcrl.models.encoder import PurposeConditionedEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry  # noqa: E402

REPR_DIM = 64
HIDDEN_DIMS = [128, 128]
PURPOSE_EMB_DIM_DEFAULTS = {"diabetes": 16, "hmda": 32}
CKPT_DIRS = {
    "diabetes": ROOT / "checkpoints" / "diab_pcrl_s0",
    "hmda": ROOT / "checkpoints" / "hmda_pcrl_0",
}


def effective_rank(reprs: np.ndarray) -> float:
    s = np.linalg.svd(reprs - reprs.mean(axis=0, keepdims=True), compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def load_data(dataset: str):
    if dataset == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        return purposes, DiabetesDataset(purposes=purposes, split="test")
    from pcrl.data.hmda import HMDADataset, get_hmda_purposes
    purposes = get_hmda_purposes()
    return purposes, HMDADataset(purposes=purposes, split="test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["diabetes", "hmda"], required=True)
    ap.add_argument("--results-dir", required=True,
                    help="dir containing summary.csv; health_check.json written here")
    ap.add_argument("--ckpt", default=None,
                    help="explicit best.pt path (default: standard PCRL seed-0 dir)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    purposes, test_ds = load_data(args.dataset)
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    feat_dim = test_ds.features.shape[1]

    encoder = PurposeConditionedEncoder(
        input_dim=feat_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM_DEFAULTS[args.dataset],
        conditioning="film", dropout=0.3,
    ).to(device)
    task_heads = nn.ModuleDict({
        p.name: TaskHead(
            repr_dim=REPR_DIM,
            output_dim=p.allowed_task_dims[p.allowed_tasks[0]],
        )
        for p in purposes
    }).to(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else CKPT_DIRS[args.dataset] / "best.pt"
    if not ckpt_path.exists():
        print(f"FATAL: no best.pt at {ckpt_path}", file=sys.stderr)
        return 2
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ck["encoder"])
    task_heads.load_state_dict(ck["task_heads"])
    best_epoch = ck["state"]["epoch"]

    encoder.eval()
    task_heads.eval()
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True)

    per_purpose: dict[str, dict] = {}
    task_acc: dict[str, dict] = {}
    majority_baselines: dict[str, float] = {}

    with torch.no_grad():
        for idx, p in enumerate(purposes):
            reprs_chunks = []
            preds_chunks: dict[str, list] = {t: [] for t in p.allowed_tasks}
            target_chunks: dict[str, list] = {t: [] for t in p.allowed_tasks}
            for batch in test_loader:
                x = batch["features"].to(device)
                h = encoder(x, idx)
                reprs_chunks.append(h.cpu().numpy())
                logits = task_heads[p.name](h)
                pri = p.allowed_tasks[0]
                preds_chunks[pri].append(logits.argmax(dim=-1).cpu().numpy())
                target_chunks[pri].append(batch["task_labels"][pri].numpy())
            reprs = np.concatenate(reprs_chunks)
            per_dim_std = reprs.std(axis=0)
            l2 = np.linalg.norm(reprs, axis=1)
            per_purpose[p.name] = {
                "shape": list(reprs.shape),
                "per_dim_std_mean": float(per_dim_std.mean()),
                "per_dim_std_max": float(per_dim_std.max()),
                "per_dim_std_min": float(per_dim_std.min()),
                "l2_norm_mean": float(l2.mean()),
                "l2_norm_std": float(l2.std()),
                "effective_rank": effective_rank(reprs),
            }
            for tname in p.allowed_tasks:
                if not preds_chunks[tname]:
                    continue
                preds = np.concatenate(preds_chunks[tname])
                targets = np.concatenate(target_chunks[tname])
                acc = float((preds == targets).mean())
                _, cnts = np.unique(targets, return_counts=True)
                maj = float(cnts.max() / cnts.sum())
                task_acc[f"{p.name}/{tname}"] = {
                    "acc": acc, "majority": maj, "delta": acc - maj,
                }
                majority_baselines[tname] = maj

    mean_per_dim_std = float(np.mean(
        [per_purpose[p.name]["per_dim_std_mean"] for p in purposes]
    ))
    above_majority = [
        v for v in task_acc.values() if v["delta"] > 0.05
    ]
    healthy = (
        mean_per_dim_std > 0.5
        and len(above_majority) >= 2
    )
    if args.dataset == "hmda":
        amount = task_acc.get(
            "pricing_analysis/loan_amount_band", {}
        ).get("acc", 0.0)
        if amount <= 0.35:
            healthy = False

    status = "HEALTHY" if healthy else "COLLAPSED"

    health_path = Path(args.results_dir) / "health_check.json"
    health_path.parent.mkdir(parents=True, exist_ok=True)
    with open(health_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "best_epoch": best_epoch,
            "status": status,
            "mean_per_dim_std": mean_per_dim_std,
            "n_tasks_above_majority_5pp": len(above_majority),
            "majority_baselines": majority_baselines,
            "per_purpose": per_purpose,
            "task_acc": task_acc,
        }, f, indent=2)
    print(f"wrote {health_path} (status={status})")

    summary_path = Path(args.results_dir) / "summary.csv"
    if summary_path.exists():
        existing = summary_path.read_text()
        if not existing.startswith("STATUS:"):
            with open(summary_path, "w") as f:
                f.write(f"STATUS: {status}\n")
                f.write(existing)
            print(f"prepended STATUS: {status} to {summary_path}")
    else:
        with open(summary_path, "w") as f:
            f.write(f"STATUS: {status}\n")
        print(f"created {summary_path} with STATUS: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

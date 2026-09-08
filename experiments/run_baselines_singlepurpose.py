#!/usr/bin/env python3
"""Single-purpose LAFTR / INLP baselines for Appendix N.

Trains a fairness baseline on the *first purpose* of each dataset and reports
the same Framework D metrics PCRL uses (R²_onehot, R²_DA, MLP-DA Δ, task
accuracy), so we can place baselines next to PCRL on identical axes.

  • Adult / income_prediction  (income, race+sex)
  • HMDA / underwriting        (loan_decision, race+ethnicity)
  • Diabetes / billing_audit   (primary_diagnosis_category, race+gender)

Random-init backbone (Option A: same architecture, same data splits, same
seeds as PCRL — but no warm-start from PCRL checkpoints).

Usage::

    python experiments/run_baselines_singlepurpose.py \
        --datasets adult --methods laftr --seeds 0 \
        --epochs 200 --output-dir results/baselines_singlepurpose

To run the canonical 18-run grid::

    python experiments/run_baselines_singlepurpose.py \
        --datasets adult hmda diabetes --methods laftr inlp --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.baselines.inlp import INLPEncoder, run_inlp  # noqa: E402
from pcrl.baselines.laftr import LAFTRDiscriminator, train_laftr  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.certificates import (  # noqa: E402
    _extract_representations_and_labels,
    compute_dominant_axis_r2,
    compute_mlp_ovr_delta,
)
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.verification import LinearComplianceCertificate  # noqa: E402

DEVICE = "cpu"


# ─────────────────────────── dataset adapters ──────────────────────────────


def build_datasets(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                                split="train", download=False)
        val_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                              split="val", download=False,
                              norm_stats=train_ds.norm_stats)
        test_ds = AdultDataset(purposes=purposes, root=str(ROOT / "data"),
                               split="test", download=False,
                               norm_stats=train_ds.norm_stats)
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        val_ds = DiabetesDataset(purposes=purposes, split="val")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"),
                               split="train")
        val_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"),
                             split="val")
        test_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"),
                              split="test")
    else:
        raise ValueError(f"unknown dataset {name}")
    return purposes, train_ds, val_ds, test_ds


def first_purpose_spec(purposes) -> dict:
    """Return spec for purpose index 0: task name+dim, disallowed attrs."""
    p = purposes[0]
    task_name = p.allowed_tasks[0]
    return {
        "name": p.name,
        "task_name": task_name,
        "task_dim": p.allowed_task_dims[task_name],
        "disallowed_attrs": list(p.disallowed_attrs),
        "disallowed_attr_dims": dict(p.disallowed_attr_dims),
    }


# ─────────────────────────── per-method training ───────────────────────────


def train_laftr_one(
    dataset: str,
    seed: int,
    args,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    purpose_spec: dict,
    sensitive_dims: dict[str, int],
    input_dim: int,
    log_fn,
) -> tuple[StandardEncoder, TaskHead, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    ).to(DEVICE)
    task_head = TaskHead(
        repr_dim=64, output_dim=purpose_spec["task_dim"],
    ).to(DEVICE)
    discriminators: dict[str, LAFTRDiscriminator] = {
        a: LAFTRDiscriminator(repr_dim=64, num_classes=sensitive_dims[a]).to(DEVICE)
        for a in purpose_spec["disallowed_attrs"]
    }

    history = train_laftr(
        encoder=encoder, task_head=task_head, discriminators=discriminators,
        train_loader=train_loader, val_loader=val_loader,
        purpose_idx=0, task_name=purpose_spec["task_name"],
        disallowed_attrs=purpose_spec["disallowed_attrs"],
        lambda_adv=args.lambda_adv, epochs=args.epochs, lr=args.lr,
        disc_steps=args.disc_steps, device=DEVICE,
        log_every=args.log_every, patience=args.patience,
        log_fn=log_fn,
    )
    diag = {
        "best_epoch": history.best_epoch,
        "best_val_task": history.best_val_task,
        "final_disc_acc_val": history.val_disc_acc[-1] if history.val_disc_acc else {},
    }
    return encoder, task_head, diag, {
        "train_task": history.train_task,
        "train_adv": history.train_adv,
        "train_disc": history.train_disc,
        "val_task": history.val_task,
        "val_disc_acc": history.val_disc_acc,
        "best_epoch": history.best_epoch,
    }


def train_inlp_one(
    dataset: str,
    seed: int,
    args,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    purpose_spec: dict,
    sensitive_dims: dict[str, int],
    input_dim: int,
    log_fn,
) -> tuple[INLPEncoder, TaskHead, dict, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    ).to(DEVICE)
    task_head = TaskHead(
        repr_dim=64, output_dim=purpose_spec["task_dim"],
    ).to(DEVICE)

    inlp_encoder, post_head, diagnostics, history = run_inlp(
        encoder=backbone, task_head=task_head,
        train_loader=train_loader, val_loader=val_loader,
        purpose_idx=0, task_name=purpose_spec["task_name"],
        num_task_classes=purpose_spec["task_dim"],
        disallowed_attrs=purpose_spec["disallowed_attrs"],
        attr_cardinalities={
            a: sensitive_dims[a] for a in purpose_spec["disallowed_attrs"]
        },
        repr_dim=64,
        pretrain_epochs=args.epochs, pretrain_lr=args.lr,
        inlp_max_iters=args.inlp_max_iters, inlp_tol_pp=args.inlp_tol_pp,
        head_epochs=args.epochs, head_lr=args.lr,
        seed=seed, device=DEVICE, log_fn=log_fn,
    )

    diag = {
        "per_attr_iterations": dict(diagnostics.per_attr_iterations),
        "per_attr_majority": diagnostics.per_attr_majority,
        "final_lr_acc": diagnostics.final_lr_acc,
        "pretrain_best_epoch": history["pretrain"].get("best_epoch", -1),
        "post_proj_head_best_epoch": history["post_proj_head"].get("best_epoch", -1),
    }
    full_history = {
        "pretrain": history["pretrain"],
        "post_proj_head": history["post_proj_head"],
        "per_attr_history": diagnostics.per_attr_history,
    }
    return inlp_encoder, post_head, diag, full_history


# ─────────────────────────── evaluation ────────────────────────────────────


def evaluate_run(
    encoder: nn.Module,
    task_head: TaskHead,
    *,
    train_loader: DataLoader,
    test_loader: DataLoader,
    purpose_spec: dict,
    seed: int,
    log_fn,
) -> dict:
    """Same metric stack as scripts/eval_round4_dominant_axis.py for purpose=0.

    Returns a per-attribute dict of {r2_onehot, r2_da, mlp_da_delta, ...}
    plus task accuracy on the test split.
    """
    cert = LinearComplianceCertificate(epsilon=0.05, regularization=1e-6)
    train_reprs, train_labels = _extract_representations_and_labels(
        encoder, train_loader, 0, purpose_spec["disallowed_attrs"], DEVICE,
    )
    test_reprs, test_labels = _extract_representations_and_labels(
        encoder, test_loader, 0, purpose_spec["disallowed_attrs"], DEVICE,
    )

    per_attr: dict[str, dict] = {}
    for attr in purpose_spec["disallowed_attrs"]:
        y_tr = train_labels[attr]
        y_te = test_labels[attr]
        K = purpose_spec["disallowed_attr_dims"].get(attr, int(y_tr.max()) + 1)

        r2_onehot = float(cert.check(test_reprs, y_te, num_classes=K).r_squared)
        da = compute_dominant_axis_r2(test_reprs, y_te, num_classes=K)

        mlp_delta = None
        mlp_argmax = -1
        mlp_coverage = {}
        if K > 2:
            mlp = compute_mlp_ovr_delta(
                train_reprs, y_tr, test_reprs, y_te, num_classes=K,
                hidden=256, epochs=50, lr=1e-3, dropout=0.3,
                batch_size=256, device=DEVICE, random_state=seed,
            )
            mlp_delta = float(mlp["mlp_da_delta"])
            mlp_argmax = int(mlp["argmax_class"])
            mlp_coverage = {key: mlp[key] for key in (
                "train_class_support", "test_class_support", "valid_mask", "coverage_complete",
            )}

        per_attr[attr] = {
            "num_classes": K,
            "r2_onehot": r2_onehot,
            "r2_da": float(da["r2_da"]),
            "r2_da_argmax_class": int(da["argmax_class"]),
            "per_class_r2": [float(x) for x in da["per_class_r2"]],
            "priors": [float(x) for x in da["priors"]],
            "mlp_da_delta": mlp_delta,
            "mlp_da_argmax_class": mlp_argmax,
            "mlp_da_coverage": mlp_coverage,
        }
        log_fn(
            f"    [{purpose_spec['name']}/{attr}/K={K}] "
            f"R²_onehot={r2_onehot:.4f}  R²_DA={float(da['r2_da']):.4f}  "
            f"argmax={int(da['argmax_class'])}"
            + (f"  MLP-DAΔ={mlp_delta:+.4f}" if mlp_delta is not None else "  MLP-DA=N/A")
        )

    # Test-set task accuracy
    task_head = task_head.to(DEVICE)
    task_head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(DEVICE)
            y = batch["task_labels"][purpose_spec["task_name"]].to(DEVICE)
            h = encoder(x, 0)
            preds = task_head(h).argmax(-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    task_acc = correct / max(total, 1)
    log_fn(f"    test task_acc({purpose_spec['task_name']}) = {task_acc:.4f}")
    return {"per_attr": per_attr, "task_acc": float(task_acc)}


# ─────────────────────────── checkpoint I/O ────────────────────────────────


def save_checkpoint(
    out_path: Path,
    *,
    method: str,
    dataset: str,
    seed: int,
    encoder: nn.Module,
    task_head: TaskHead,
    diag: dict,
    config: dict,
) -> None:
    payload: dict = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "config": config,
        "diagnostics": diag,
        "task_head": task_head.state_dict(),
    }
    if method == "laftr":
        payload["backbone"] = encoder.state_dict()
    elif method == "inlp":
        # encoder here is INLPEncoder(backbone, projection)
        payload["backbone"] = encoder.backbone.state_dict()
        payload["projection"] = encoder.projection.detach().cpu().clone()
    else:
        raise ValueError(f"unknown method {method}")
    torch.save(payload, out_path)


# ─────────────────────────── orchestration ─────────────────────────────────


def run_one(
    dataset: str,
    method: str,
    seed: int,
    args,
    out_dir: Path,
) -> dict:
    log_lines: list[str] = []

    def log_fn(msg: str) -> None:
        line = msg if isinstance(msg, str) else str(msg)
        log_lines.append(line)
        print(line, flush=True)

    log_fn(f"\n=== {dataset.upper()} / {method.upper()} / seed={seed} ===")
    t0 = time.time()
    purposes, train_ds, val_ds, test_ds = build_datasets(dataset)
    purpose_spec = first_purpose_spec(purposes)
    sensitive_dims = dict(train_ds.info.sensitive_attrs)
    log_fn(
        f"  N_train={len(train_ds)} N_val={len(val_ds)} N_test={len(test_ds)} "
        f"D={train_ds.info.num_features}"
    )
    log_fn(
        f"  purpose={purpose_spec['name']} task={purpose_spec['task_name']} "
        f"task_dim={purpose_spec['task_dim']} disallowed={purpose_spec['disallowed_attrs']}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )

    if method == "laftr":
        encoder, task_head, diag, history = train_laftr_one(
            dataset, seed, args,
            train_loader=train_loader, val_loader=val_loader,
            purpose_spec=purpose_spec, sensitive_dims=sensitive_dims,
            input_dim=train_ds.info.num_features, log_fn=log_fn,
        )
    elif method == "inlp":
        encoder, task_head, diag, history = train_inlp_one(
            dataset, seed, args,
            train_loader=train_loader, val_loader=val_loader,
            purpose_spec=purpose_spec, sensitive_dims=sensitive_dims,
            input_dim=train_ds.info.num_features, log_fn=log_fn,
        )
    else:
        raise ValueError(f"unknown method {method}")
    train_time = time.time() - t0

    log_fn("  evaluating on test split…")
    metrics = evaluate_run(
        encoder, task_head,
        train_loader=train_loader, test_loader=test_loader,
        purpose_spec=purpose_spec, seed=seed, log_fn=log_fn,
    )

    config = {
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "lambda_adv": args.lambda_adv, "disc_steps": args.disc_steps,
        "inlp_max_iters": args.inlp_max_iters, "inlp_tol_pp": args.inlp_tol_pp,
        "patience": args.patience,
    }

    ckpt_path = out_dir / f"{dataset}_{method}_s{seed}.pt"
    save_checkpoint(
        ckpt_path, method=method, dataset=dataset, seed=seed,
        encoder=encoder, task_head=task_head, diag=diag, config=config,
    )

    record = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "purpose": purpose_spec["name"],
        "task_name": purpose_spec["task_name"],
        "task_dim": purpose_spec["task_dim"],
        "disallowed_attrs": purpose_spec["disallowed_attrs"],
        "config": config,
        "metrics": metrics,
        "diagnostics": diag,
        "train_seconds": float(train_time),
    }
    json_path = out_dir / f"{dataset}_{method}_s{seed}.json"
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    log_path = out_dir / f"{dataset}_{method}_s{seed}.log"
    log_path.write_text("\n".join(log_lines) + "\n")

    history_path = out_dir / f"{dataset}_{method}_s{seed}_history.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    log_fn(f"  wrote {ckpt_path}")
    log_fn(f"  wrote {json_path}")
    log_fn(f"  total wall: {train_time:.1f}s")
    return record


def aggregate(out_dir: Path, records: list[dict]) -> None:
    """Aggregate per-(dataset, method) means and write summary.json."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_key.setdefault((r["dataset"], r["method"]), []).append(r)

    summary: dict[str, list[dict]] = {"runs": []}
    for (ds, method), runs in sorted(by_key.items()):
        attr_set = sorted({a for r in runs for a in r["disallowed_attrs"]})
        agg: dict = {"dataset": ds, "method": method, "n_seeds": len(runs)}
        for attr in attr_set:
            r2_onehot = [r["metrics"]["per_attr"][attr]["r2_onehot"] for r in runs]
            r2_da = [r["metrics"]["per_attr"][attr]["r2_da"] for r in runs]
            agg[f"{attr}_r2_onehot_mean"] = float(np.mean(r2_onehot))
            agg[f"{attr}_r2_onehot_std"] = float(np.std(r2_onehot))
            agg[f"{attr}_r2_da_mean"] = float(np.mean(r2_da))
            agg[f"{attr}_r2_da_std"] = float(np.std(r2_da))
            mlp = [r["metrics"]["per_attr"][attr]["mlp_da_delta"] for r in runs
                   if r["metrics"]["per_attr"][attr]["mlp_da_delta"] is not None]
            if mlp:
                agg[f"{attr}_mlp_da_delta_mean"] = float(np.mean(mlp))
                agg[f"{attr}_mlp_da_delta_std"] = float(np.std(mlp))
        task_acc = [r["metrics"]["task_acc"] for r in runs]
        agg["task_acc_mean"] = float(np.mean(task_acc))
        agg["task_acc_std"] = float(np.std(task_acc))
        summary["runs"].append(agg)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {summary_path}")
    for row in summary["runs"]:
        print(f"  {row['dataset']}/{row['method']} (n={row['n_seeds']}): "
              f"task_acc={row['task_acc_mean']:.4f} ± {row['task_acc_std']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["adult", "hmda", "diabetes"])
    ap.add_argument("--methods", nargs="+", default=["laftr", "inlp"],
                    choices=["laftr", "inlp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--disc-steps", type=int, default=1)
    ap.add_argument("--inlp-max-iters", type=int, default=32)
    ap.add_argument("--inlp-tol-pp", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--output-dir", type=str,
                    default="results/baselines_singlepurpose")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                rec = run_one(dataset, method, seed, args, out_dir)
                records.append(rec)

    aggregate(out_dir, records)


if __name__ == "__main__":
    main()

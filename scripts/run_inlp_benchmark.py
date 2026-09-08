#!/usr/bin/env python3
"""Per-purpose INLP benchmark — one (dataset, purpose, seed) per invocation.

Mirrors the StandardEncoder backbone PCRL uses (MLP[128,128]→64) and the
on-disk artifact format produced by ``scripts/run_laftr_benchmark.py``, so
the existing aggregator and three-way comparison code can consume INLP
metrics without modification.

Pipeline (re-uses ``pcrl.baselines.inlp.run_inlp``):
  1. Pretrain a vanilla StandardEncoder + TaskHead on task loss only.
  2. Extract train/val representations.
  3. For each disallowed attribute (largest-cardinality first), iterate
     INLP: train multinomial LR on (h, attr); project h onto the LR
     null-space; repeat until LR test accuracy is within tol_pp of the
     test-split majority baseline or max_iters is reached.
  4. Train a fresh TaskHead on the projected reprs.
  5. Save encoder + projection matrix + eval reps + test labels + metrics.

Multi-attribute disallowed sets (e.g., {race, sex}) are handled by
SEQUENTIAL per-attribute INLP, applied in order of decreasing cardinality
(an honest deviation from the joint formulation; documented in the paper).

Output layout
-------------
results/inlp_benchmark/{dataset}/{purpose}/seed_{s}/
    encoder.pt          # state_dict + post_head + projection (P) + config
    eval_reps.npz       # PROJECTED test-set reps [N, 64]
    test_labels.npz     # task labels + ALL sensitive attrs
    metrics.json        # R²_onehot, R²_DA, MLP-DA Δ per disallowed attr;
                        # task_acc; per-attr INLP iterations + final LR acc.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.baselines.inlp import run_inlp  # noqa: E402
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
PURPOSE_NAMES = {
    "adult": ["income_prediction", "employment_analysis", "education_assessment"],
    "hmda": ["underwriting", "pricing_analysis", "fair_lending_audit"],
    "diabetes": ["billing_audit", "quality_research", "clinical_decision_support"],
}


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
        train_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="train")
        val_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="val")
        test_ds = HMDADataset(purposes=purposes, root=str(ROOT / "data"), split="test")
    else:
        raise ValueError(f"unknown dataset {name}")
    return purposes, train_ds, val_ds, test_ds


def evaluate(
    *,
    encoder,
    task_head,
    train_loader,
    test_loader,
    purpose,
    seed: int,
    log_fn,
):
    """Same eval contract as run_laftr_benchmark.evaluate.

    The encoder here is an ``INLPEncoder`` (backbone + projection); the
    task_head is the post-projection head trained on projected reprs.
    Forward already applies the projection so existing eval helpers
    consume (B, repr_dim) reprs identically to LAFTR/PCRL.
    """
    cert = LinearComplianceCertificate(epsilon=0.05, regularization=1e-6)

    train_reprs, train_labels_disallowed = _extract_representations_and_labels(
        encoder, train_loader, 0, list(purpose.disallowed_attrs), DEVICE,
    )
    test_ds = test_loader.dataset
    while isinstance(test_ds, Subset):
        test_ds = test_ds.dataset
    all_sensitive_attrs = list(test_ds.info.sensitive_attrs.keys())
    test_reprs, test_labels_all = _extract_representations_and_labels(
        encoder, test_loader, 0, all_sensitive_attrs, DEVICE,
    )

    per_attr: dict[str, dict] = {}
    for attr in purpose.disallowed_attrs:
        y_tr = train_labels_disallowed[attr]
        y_te = test_labels_all[attr]
        K = purpose.disallowed_attr_dims.get(attr, int(y_tr.max()) + 1)

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
            f"    [{purpose.name}/{attr}/K={K}] R²_onehot={r2_onehot:.4f} "
            f"R²_DA={float(da['r2_da']):.4f} argmax={int(da['argmax_class'])}"
            + (f"  MLP-DAΔ={mlp_delta:+.4f}" if mlp_delta is not None else "  MLP-DA=N/A")
        )

    task_name = purpose.allowed_tasks[0]
    task_head.eval()
    encoder.eval()
    correct = 0
    total = 0
    test_task_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(DEVICE)
            y = batch["task_labels"][task_name].to(DEVICE)
            test_task_labels.append(y.cpu().numpy())
            h = encoder(x, 0)
            preds = task_head(h).argmax(-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    task_acc = correct / max(total, 1)
    log_fn(f"    test task_acc({task_name}) = {task_acc:.4f}")

    test_task_labels_dict: dict[str, np.ndarray] = {
        task_name: np.concatenate(test_task_labels),
    }
    metrics = {"per_attr": per_attr, "task_acc": float(task_acc)}
    return metrics, test_reprs, test_labels_all, test_task_labels_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "hmda", "diabetes"], required=True)
    ap.add_argument("--purpose-idx", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--pretrain-epochs", type=int, default=200)
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--inlp-max-iters", type=int, default=50,
                    help="Max INLP iterations per attribute (cap; user spec)")
    ap.add_argument("--inlp-tol-pp", type=float, default=1.0,
                    help="Stop when LR test-acc within tol_pp of majority (percentage points)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke mode: cap pretrain/head epochs at 30, max_iters at 8")
    args = ap.parse_args()

    if args.quick:
        args.pretrain_epochs = min(args.pretrain_epochs, 30)
        args.head_epochs = min(args.head_epochs, 30)
        args.inlp_max_iters = min(args.inlp_max_iters, 8)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_lines: list[str] = []

    def log_fn(msg) -> None:
        line = msg if isinstance(msg, str) else str(msg)
        log_lines.append(line)
        print(line, flush=True)

    log_fn(f"=== INLP {args.dataset.upper()} / purpose_idx={args.purpose_idx} / seed={args.seed} ===")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    purposes, train_ds, val_ds, test_ds = build_datasets(args.dataset)
    purpose = purposes[args.purpose_idx]
    expected_name = PURPOSE_NAMES[args.dataset][args.purpose_idx]
    if purpose.name != expected_name:
        raise RuntimeError(f"purpose name drift: got {purpose.name}, expected {expected_name}")
    sensitive_dims = dict(train_ds.info.sensitive_attrs)
    log_fn(
        f"  N_train={len(train_ds)} N_val={len(val_ds)} N_test={len(test_ds)} "
        f"D={train_ds.info.num_features}"
    )
    log_fn(
        f"  purpose={purpose.name} task={purpose.allowed_tasks[0]} "
        f"task_dim={purpose.allowed_task_dims[purpose.allowed_tasks[0]]} "
        f"disallowed={list(purpose.disallowed_attrs)}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    input_dim = train_ds.info.num_features

    encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=64, dropout=args.dropout,
    ).to(DEVICE)
    task_head = TaskHead(
        repr_dim=64,
        output_dim=purpose.allowed_task_dims[purpose.allowed_tasks[0]],
    ).to(DEVICE)

    n_params_encoder = sum(p.numel() for p in encoder.parameters())
    n_params_task = sum(p.numel() for p in task_head.parameters())
    log_fn(
        f"  params: encoder={n_params_encoder:,} task_head={n_params_task:,}"
    )
    log_fn(
        f"  inlp config: max_iters={args.inlp_max_iters} tol_pp={args.inlp_tol_pp} "
        f"pretrain_epochs={args.pretrain_epochs} head_epochs={args.head_epochs}"
    )

    attr_cardinalities = {a: int(sensitive_dims[a]) for a in purpose.disallowed_attrs}

    inlp_encoder, post_head, diagnostics, history = run_inlp(
        encoder=encoder, task_head=task_head,
        train_loader=train_loader, val_loader=val_loader,
        purpose_idx=0, task_name=purpose.allowed_tasks[0],
        num_task_classes=purpose.allowed_task_dims[purpose.allowed_tasks[0]],
        disallowed_attrs=list(purpose.disallowed_attrs),
        attr_cardinalities=attr_cardinalities,
        repr_dim=64,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.lr,
        inlp_max_iters=args.inlp_max_iters,
        inlp_tol_pp=args.inlp_tol_pp,
        head_epochs=args.head_epochs,
        head_lr=args.lr,
        seed=args.seed,
        device=DEVICE,
        log_fn=log_fn,
    )
    train_time = time.time() - t0
    log_fn(f"  INLP wall: {train_time:.1f}s")

    metrics, test_reprs, test_labels_all, test_task_labels = evaluate(
        encoder=inlp_encoder, task_head=post_head,
        train_loader=train_loader, test_loader=test_loader,
        purpose=purpose, seed=args.seed, log_fn=log_fn,
    )

    # ---- Persist artifacts (mirror LAFTR contract) ----
    proj_np = inlp_encoder.projection.detach().cpu().numpy().astype(np.float32)
    encoder_payload = {
        "method": "inlp",
        "dataset": args.dataset,
        "purpose": purpose.name,
        "purpose_idx": args.purpose_idx,
        "seed": args.seed,
        "config": vars(args),
        "input_dim": input_dim,
        "hidden_dims": [128, 128],
        "repr_dim": 64,
        "dropout": args.dropout,
        "encoder": inlp_encoder.backbone.state_dict(),
        "task_head": post_head.state_dict(),
        "projection": torch.from_numpy(proj_np),
        "task_name": purpose.allowed_tasks[0],
        "task_dim": purpose.allowed_task_dims[purpose.allowed_tasks[0]],
        "disallowed_attrs": list(purpose.disallowed_attrs),
        "disallowed_attr_dims": dict(purpose.disallowed_attr_dims),
    }
    torch.save(encoder_payload, out_dir / "encoder.pt")
    np.savez(out_dir / "eval_reps.npz", reps=test_reprs.astype(np.float32))
    label_arrs = {f"sensitive_{a}": v.astype(np.int64) for a, v in test_labels_all.items()}
    label_arrs.update({f"task_{t}": v.astype(np.int64) for t, v in test_task_labels.items()})
    np.savez(out_dir / "test_labels.npz", **label_arrs)

    inlp_diag = {
        "per_attr_iterations": dict(diagnostics.per_attr_iterations),
        "per_attr_majority": {a: float(v) for a, v in diagnostics.per_attr_majority.items()},
        "final_lr_acc": {a: float(v) for a, v in diagnostics.final_lr_acc.items()},
        "per_attr_history": diagnostics.per_attr_history,
    }

    metrics_full = {
        "dataset": args.dataset,
        "purpose": purpose.name,
        "purpose_idx": args.purpose_idx,
        "seed": args.seed,
        "method": "inlp",
        "task_name": purpose.allowed_tasks[0],
        "disallowed_attrs": list(purpose.disallowed_attrs),
        "metrics": metrics,
        "inlp_diagnostics": inlp_diag,
        "history": history,
        "train_seconds": float(train_time),
        "config": vars(args),
        "n_params": {
            "encoder": int(n_params_encoder),
            "task_head": int(n_params_task),
        },
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics_full, fh, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))

    log_path.write_text("\n".join(log_lines) + "\n")
    log_fn(f"  wrote {out_dir / 'encoder.pt'}")
    log_fn(f"  wrote {out_dir / 'eval_reps.npz'}")
    log_fn(f"  wrote {out_dir / 'test_labels.npz'}")
    log_fn(f"  wrote {out_dir / 'metrics.json'}")
    log_path.write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()

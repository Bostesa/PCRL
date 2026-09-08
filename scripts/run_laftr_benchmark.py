#!/usr/bin/env python3
"""Per-purpose LAFTR benchmark — one (dataset, purpose, seed) per invocation.

Mirrors the StandardEncoder backbone PCRL uses (MLP[128,128]→64) and trains
a LAFTR adversarial setup with one MLP discriminator per disallowed
attribute (loss = task_CE − λ · Σ_a CE(disc_a(h), y_a)). Re-uses
``pcrl.baselines.laftr.train_laftr`` which already supports multi-attribute
adversaries.

Output layout
-------------
results/laftr_benchmark/{dataset}/{purpose}/seed_{s}/
    encoder.pt          # full state_dict + task_head + config
    eval_reps.npz       # test-set reps [N, 64]
    test_labels.npz     # task labels + ALL sensitive attrs (for cross-purpose attack)
    metrics.json        # R²_onehot, R²_DA, MLP-DA Δ per disallowed attr; task_acc; history

CLI
---
    python scripts/run_laftr_benchmark.py \\
        --dataset {adult|hmda|diabetes} \\
        --purpose-idx {0|1|2} \\
        --seed {0|1|2} \\
        --output-dir results/laftr_benchmark/{dataset}/{purpose}/seed_{s}/

Optional flags: --epochs (200), --batch-size (256), --lr (1e-3),
--lambda-adv (annealed 0.1→1.0 over training), --disc-steps (1),
--patience (20), --quick (smoke test: subset to 2000 train, 50 epochs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.baselines.laftr import LAFTRDiscriminator, _disc_accuracy  # noqa: E402
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


def train_laftr_annealed(
    *,
    encoder: nn.Module,
    task_head: nn.Module,
    discriminators: dict[str, LAFTRDiscriminator],
    train_loader: DataLoader,
    val_loader: DataLoader,
    purpose_idx: int,
    task_name: str,
    disallowed_attrs: list[str],
    lambda_max: float,
    lambda_warmup_epochs: int,
    epochs: int,
    lr: float,
    disc_steps: int,
    patience: int,
    log_every: int,
    log_fn,
) -> dict:
    """Same alternating LAFTR loop as ``train_laftr`` but with linear λ
    warmup from 0.1·λ_max → λ_max across the first ``lambda_warmup_epochs``."""
    device = torch.device(DEVICE)
    encoder = encoder.to(device)
    task_head = task_head.to(device)
    for d in discriminators.values():
        d.to(device)

    enc_params = list(encoder.parameters()) + list(task_head.parameters())
    enc_opt = torch.optim.Adam(enc_params, lr=lr)
    disc_params: list[nn.Parameter] = []
    for d in discriminators.values():
        disc_params.extend(d.parameters())
    disc_opt = torch.optim.Adam(disc_params, lr=lr)

    history: dict = {
        "train_task": [], "train_adv": [], "train_disc": [],
        "val_task": [], "val_disc_acc": [], "lambda": [],
        "best_epoch": -1, "best_val_task": float("inf"),
    }
    best_state: dict | None = None
    bad = 0

    for epoch in range(epochs):
        # linear warmup 0.1·λ_max → λ_max
        if lambda_warmup_epochs > 0 and epoch < lambda_warmup_epochs:
            t = (epoch + 1) / max(lambda_warmup_epochs, 1)
            lam = (0.1 + 0.9 * t) * lambda_max
        else:
            lam = lambda_max
        history["lambda"].append(lam)

        encoder.train()
        task_head.train()
        for d in discriminators.values():
            d.train()

        running_task = running_adv = running_disc = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["features"].to(device)
            y_task = batch["task_labels"][task_name].to(device)
            y_attrs = {a: batch["sensitive_attrs"][a].to(device) for a in disallowed_attrs}

            for _ in range(disc_steps):
                disc_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    h_det = encoder(x, purpose_idx)
                d_loss = x.new_zeros(())
                for attr, disc in discriminators.items():
                    d_loss = d_loss + F.cross_entropy(disc(h_det), y_attrs[attr])
                d_loss.backward()
                disc_opt.step()

            enc_opt.zero_grad(set_to_none=True)
            h = encoder(x, purpose_idx)
            t_logits = task_head(h)
            task_loss = F.cross_entropy(t_logits, y_task)

            for disc in discriminators.values():
                for p in disc.parameters():
                    p.requires_grad_(False)
            adv_loss = x.new_zeros(())
            for attr, disc in discriminators.items():
                adv_loss = adv_loss + F.cross_entropy(disc(h), y_attrs[attr])
            for disc in discriminators.values():
                for p in disc.parameters():
                    p.requires_grad_(True)

            enc_loss = task_loss - lam * adv_loss
            enc_loss.backward()
            enc_opt.step()

            running_task += float(task_loss.item())
            running_adv += float(adv_loss.item())
            running_disc += float(d_loss.item())
            n_batches += 1

        history["train_task"].append(running_task / max(n_batches, 1))
        history["train_adv"].append(running_adv / max(n_batches, 1))
        history["train_disc"].append(running_disc / max(n_batches, 1))

        encoder.eval()
        task_head.eval()
        val_task = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y_task = batch["task_labels"][task_name].to(device)
                h = encoder(x, purpose_idx)
                val_task += float(F.cross_entropy(task_head(h), y_task).item())
                n_v += 1
        avg_val = val_task / max(n_v, 1)
        history["val_task"].append(avg_val)

        disc_acc = _disc_accuracy(discriminators, encoder, val_loader, purpose_idx, device)
        history["val_disc_acc"].append(disc_acc)

        if avg_val < history["best_val_task"]:
            history["best_val_task"] = avg_val
            history["best_epoch"] = epoch
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "task_head": {k: v.detach().cpu().clone() for k, v in task_head.state_dict().items()},
                "discriminators": {
                    a: {k: v.detach().cpu().clone() for k, v in d.state_dict().items()}
                    for a, d in discriminators.items()
                },
            }
            bad = 0
        else:
            bad += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            disc_str = " ".join(f"{a}={v:.3f}" for a, v in disc_acc.items())
            log_fn(
                f"  [ep {epoch + 1:3d}/{epochs}] λ={lam:.3f} "
                f"task={history['train_task'][-1]:.4f} "
                f"adv={history['train_adv'][-1]:.4f} "
                f"disc={history['train_disc'][-1]:.4f} "
                f"val_task={avg_val:.4f} disc_acc {disc_str}"
            )

        if bad >= patience:
            log_fn(f"  [laftr] early stop at epoch {epoch + 1} (no val improvement for {patience})")
            break

    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
        task_head.load_state_dict({k: v.to(device) for k, v in best_state["task_head"].items()})
        for a, d in discriminators.items():
            d.load_state_dict({k: v.to(device) for k, v in best_state["discriminators"][a].items()})
    return history


def evaluate(
    *,
    encoder: nn.Module,
    task_head: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    purpose,
    seed: int,
    log_fn,
) -> tuple[dict, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns (metrics_dict, test_reps, test_labels_all_sensitive, test_task_labels)."""
    cert = LinearComplianceCertificate(epsilon=0.05, regularization=1e-6)

    # Reps + ALL sensitive attrs (so cross-purpose attack has labels for any attr)
    train_reprs, train_labels_disallowed = _extract_representations_and_labels(
        encoder, train_loader, 0, list(purpose.disallowed_attrs), DEVICE,
    )
    # On test split we want ALL sensitive attrs from the dataset, not just disallowed.
    test_ds = test_loader.dataset
    # Unwrap torch.utils.data.Subset (used in --quick mode).
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

    # Test-set task accuracy
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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-adv", type=float, default=1.0,
                    help="λ_max — annealed 0.1·λ_max → λ_max over warmup")
    ap.add_argument("--lambda-warmup-epochs", type=int, default=-1,
                    help="defaults to half of --epochs if -1")
    ap.add_argument("--disc-steps", type=int, default=1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke mode: subset 2000 train / 500 val/test, 50 epochs")
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    if args.quick:
        # Use full dataset but cap epochs at 100 — Adult's MLP+BN needs ~50+
        # epochs to escape the uniform-prediction plateau (200 in production).
        args.epochs = min(args.epochs, 100)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_lines: list[str] = []

    def log_fn(msg: str) -> None:
        line = msg if isinstance(msg, str) else str(msg)
        log_lines.append(line)
        print(line, flush=True)

    log_fn(f"=== {args.dataset.upper()} / purpose_idx={args.purpose_idx} / seed={args.seed} ===")
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

    # Note: --quick now caps epochs at 30 but uses full dataset (small
    # subsets give degenerate BN statistics + insufficient training signal).

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
    discriminators: dict[str, LAFTRDiscriminator] = {
        a: LAFTRDiscriminator(repr_dim=64, num_classes=sensitive_dims[a]).to(DEVICE)
        for a in purpose.disallowed_attrs
    }

    n_params_encoder = sum(p.numel() for p in encoder.parameters())
    n_params_task = sum(p.numel() for p in task_head.parameters())
    n_params_disc = sum(p.numel() for d in discriminators.values() for p in d.parameters())
    log_fn(
        f"  params: encoder={n_params_encoder:,} task_head={n_params_task:,} "
        f"discriminators={n_params_disc:,}"
    )

    warmup = args.lambda_warmup_epochs if args.lambda_warmup_epochs >= 0 else max(args.epochs // 2, 1)

    history = train_laftr_annealed(
        encoder=encoder, task_head=task_head, discriminators=discriminators,
        train_loader=train_loader, val_loader=val_loader,
        purpose_idx=0, task_name=purpose.allowed_tasks[0],
        disallowed_attrs=list(purpose.disallowed_attrs),
        lambda_max=args.lambda_adv, lambda_warmup_epochs=warmup,
        epochs=args.epochs, lr=args.lr, disc_steps=args.disc_steps,
        patience=args.patience, log_every=args.log_every, log_fn=log_fn,
    )
    train_time = time.time() - t0
    log_fn(f"  train wall: {train_time:.1f}s; best_epoch={history['best_epoch']}")

    # ---- Re-evaluate on full splits even in --quick mode? No: quick is smoke only.
    metrics, test_reprs, test_labels_all, test_task_labels = evaluate(
        encoder=encoder, task_head=task_head,
        train_loader=train_loader, test_loader=test_loader,
        purpose=purpose, seed=args.seed, log_fn=log_fn,
    )

    # ---- Persist artifacts per the load-bearing format spec ----
    encoder_payload = {
        "method": "laftr",
        "dataset": args.dataset,
        "purpose": purpose.name,
        "purpose_idx": args.purpose_idx,
        "seed": args.seed,
        "config": vars(args),
        "input_dim": input_dim,
        "hidden_dims": [128, 128],
        "repr_dim": 64,
        "dropout": args.dropout,
        "encoder": encoder.state_dict(),
        "task_head": task_head.state_dict(),
        "task_name": purpose.allowed_tasks[0],
        "task_dim": purpose.allowed_task_dims[purpose.allowed_tasks[0]],
        "disallowed_attrs": list(purpose.disallowed_attrs),
        "disallowed_attr_dims": dict(purpose.disallowed_attr_dims),
        "discriminators": {a: d.state_dict() for a, d in discriminators.items()},
    }
    torch.save(encoder_payload, out_dir / "encoder.pt")
    np.savez(out_dir / "eval_reps.npz", reps=test_reprs.astype(np.float32))
    # test_labels.npz: stash all sensitive attrs + task label
    label_arrs = {f"sensitive_{a}": v.astype(np.int64) for a, v in test_labels_all.items()}
    label_arrs.update({f"task_{t}": v.astype(np.int64) for t, v in test_task_labels.items()})
    np.savez(out_dir / "test_labels.npz", **label_arrs)

    metrics_full = {
        "dataset": args.dataset,
        "purpose": purpose.name,
        "purpose_idx": args.purpose_idx,
        "seed": args.seed,
        "task_name": purpose.allowed_tasks[0],
        "disallowed_attrs": list(purpose.disallowed_attrs),
        "metrics": metrics,
        "history": history,
        "train_seconds": float(train_time),
        "config": vars(args),
        "n_params": {
            "encoder": int(n_params_encoder),
            "task_head": int(n_params_task),
            "discriminators": int(n_params_disc),
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

#!/usr/bin/env python3
"""Train a Standard (no-privacy) encoder on {diabetes, hmda} for one seed and
report per-task test accuracy + delta vs majority. Establishes the ceiling
that PCRL is bounded by; informs whether sanity criteria were too strict."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402

REPR_DIM = 64
HIDDEN_DIMS = [128, 128]
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256


def load_data(dataset: str):
    if dataset == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        return (
            purposes,
            DiabetesDataset(purposes=purposes, split="train"),
            DiabetesDataset(purposes=purposes, split="val"),
            DiabetesDataset(purposes=purposes, split="test"),
        )
    from pcrl.data.hmda import HMDADataset, get_hmda_purposes
    purposes = get_hmda_purposes()
    return (
        purposes,
        HMDADataset(purposes=purposes, split="train"),
        HMDADataset(purposes=purposes, split="val"),
        HMDADataset(purposes=purposes, split="test"),
    )


def evaluate_per_task(encoder, heads, loader, purposes, device):
    encoder.eval()
    for h in heads.values():
        h.eval()
    all_preds = {p.name: {t: [] for t in p.allowed_tasks} for p in purposes}
    all_targets = {p.name: {t: [] for t in p.allowed_tasks} for p in purposes}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x)
            for p in purposes:
                logits = heads[p.name](h)
                for tname in p.allowed_tasks:
                    if tname not in batch["task_labels"]:
                        continue
                    pred = logits.argmax(dim=-1).cpu().numpy() if not isinstance(logits, dict) else None
                    if isinstance(logits, dict):
                        if tname not in logits:
                            continue
                        pred = logits[tname].argmax(dim=-1).cpu().numpy()
                    all_preds[p.name][tname].append(pred)
                    all_targets[p.name][tname].append(batch["task_labels"][tname].numpy())
    out = {}
    for p in purposes:
        for tname in p.allowed_tasks:
            if not all_preds[p.name][tname]:
                continue
            preds = np.concatenate(all_preds[p.name][tname])
            targets = np.concatenate(all_targets[p.name][tname])
            acc = float((preds == targets).mean())
            _, cnts = np.unique(targets, return_counts=True)
            maj = float(cnts.max() / cnts.sum())
            out[f"{p.name}/{tname}"] = {
                "acc": acc, "majority": maj, "delta": acc - maj,
            }
    return out


def train(dataset: str, seed: int, epochs: int, patience: int, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{dataset}] device={device}, seed={seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    purposes, train_ds, val_ds, test_ds = load_data(dataset)
    feat_dim = train_ds.features.shape[1]
    print(f"[{dataset}] feat_dim={feat_dim}, train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}, purposes={len(purposes)}")

    encoder = StandardEncoder(
        input_dim=feat_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=0.3,
    ).to(device)
    heads = nn.ModuleDict({
        p.name: TaskHead(repr_dim=REPR_DIM,
                         output_dim=p.allowed_task_dims[p.allowed_tasks[0]])
        for p in purposes
    }).to(device)

    params = list(encoder.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    t0 = time.time()
    for epoch in range(epochs):
        encoder.train()
        heads.train()
        n_batches = 0
        train_loss_sum = 0.0
        for batch in train_loader:
            x = batch["features"].to(device)
            h = encoder(x)
            loss = torch.tensor(0.0, device=device)
            for p in purposes:
                primary = p.allowed_tasks[0]
                if primary not in batch["task_labels"]:
                    continue
                logits = heads[p.name](h)
                if isinstance(logits, dict):
                    logits = logits[primary]
                target = batch["task_labels"][primary].to(device)
                loss = loss + ce(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # Validation
        encoder.eval()
        heads.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                h = encoder(x)
                vloss = torch.tensor(0.0, device=device)
                for p in purposes:
                    primary = p.allowed_tasks[0]
                    if primary not in batch["task_labels"]:
                        continue
                    logits = heads[p.name](h)
                    if isinstance(logits, dict):
                        logits = logits[primary]
                    target = batch["task_labels"][primary].to(device)
                    vloss = vloss + ce(logits, target)
                val_loss_sum += vloss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{dataset}] early stopping at epoch {epoch} (best={best_epoch})")
                break

    train_time_s = time.time() - t0
    encoder.load_state_dict(best_state["encoder"])
    heads.load_state_dict(best_state["heads"])

    test_acc = evaluate_per_task(encoder, heads, test_loader, purposes, device)
    print(f"[{dataset}] best_epoch={best_epoch}  train_time_s={train_time_s:.1f}")
    for k, v in test_acc.items():
        print(f"  {k}: acc={v['acc']:.4f}  maj={v['majority']:.4f}  Δ={v['delta']:+.4f}")

    out = {
        "dataset": dataset,
        "method": "Standard (no privacy)",
        "seed": seed,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"] if history else 0,
        "train_time_s": train_time_s,
        "config": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH,
            "epochs": epochs, "patience": patience,
            "hidden_dims": HIDDEN_DIMS, "repr_dim": REPR_DIM, "dropout": 0.3,
        },
        "test_per_task": test_acc,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{dataset}] wrote {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["diabetes", "hmda"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return train(args.dataset, args.seed, args.epochs, args.patience, args.out)


if __name__ == "__main__":
    sys.exit(main())

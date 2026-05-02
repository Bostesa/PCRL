#!/usr/bin/env python3
"""BIOS-medium drift diagnostic.

Question: does the LEACE post-projection hold under task-loss training, or
does fine-tuning the LoRA shift [CLS] off the distribution P was fit on
enough that dev R²([CLS], gender) climbs above 0.05?

Decision rule (set by user 2026-05-02):
  post_dev_r2 <= 0.05 → Option A: skip proxy-Lagrangian for marginal gender
                        in Phase 1, train pure task loss; the structural
                        post-projection enforces the constraint.
  post_dev_r2 >  0.05 → Option D: proxy-Lagrangian on R² over a held-out
                        4096-sample batch, refreshed every K=10 mini-batches
                        (d/N=0.19 keeps the noise floor below the 0.05
                        threshold so the dual gets a real signal).

Procedure:
  1. Build BertWithLoRA, fit LEACE warm-start on the 50K train subsample
     (identical to Phase 1 production setup).
  2. Pre-training dev R²([CLS], gender) on full dev (31K) — sanity check
     against the existing diagnostic's 0.0438.
  3. Subset the 50K train down to 5K, train LoRA + task head for 1 epoch
     on pure 10-class CrossEntropy loss. NO proxy-Lagrangian, NO gender
     constraint. The post-projection is the only thing keeping gender
     info out of the representation during this run.
  4. Post-training dev R² + dev top-10 task accuracy on full dev.
  5. Save report and verdict to results/v2_bios_DRIFT/drift.json.

Reuses (import-only, never modified):
  pcrl.language: BertWithLoRA, build_bios_loaders, leace_warm_start_bert
  pcrl.models.task_head.TaskHead
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcrl.language import (  # noqa: E402
    BertWithLoRA,
    build_bios_loaders,
    leace_warm_start_bert,
)
from pcrl.models.task_head import TaskHead  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(arg: str) -> torch.device:
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but no CUDA device.")
        return torch.device("cuda")
    if arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS unavailable.")
        return torch.device("mps")
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


@torch.no_grad()
def _eval_dev(
    model, task_head, dev_loader, device, *, decimal_places: int = 4,
) -> dict:
    """Single forward pass over dev: closed-form ridge R²([CLS], gender) +
    optional top-10 task accuracy.

    R² formula matches ``construction_r2`` and ``leace_warmstart._linear_r2_train``
    (one-hot regression, ridge=1e-6, no max_samples cap) so values are
    directly comparable to the construction-time R² from diagnostic #4.
    """
    model.eval()
    if task_head is not None:
        task_head.eval()
    z_blocks: list[torch.Tensor] = []
    g_blocks: list[torch.Tensor] = []
    correct, total = 0, 0
    for batch in dev_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        gen = batch["gender"].long()
        z = model(ids, mask)
        if task_head is not None:
            occ = batch["occupation"].long().to(device)
            logits = task_head(z)
            pred = logits.argmax(-1)
            correct += int((pred == occ).sum().item())
            total += int(occ.numel())
        z_blocks.append(z.detach().cpu())
        g_blocks.append(gen)

    Z = torch.cat(z_blocks, dim=0).float().numpy().astype(np.float64)
    g = torch.cat(g_blocks, dim=0).numpy().astype(np.int64)
    Z_oh = np.eye(2)[g].astype(np.float64)
    H_c = Z - Z.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + 1e-6 * np.eye(H_c.shape[1])
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    r2 = float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    return {
        "r2": r2,
        "r2_rounded": round(r2, decimal_places),
        "n_dev": int(Z.shape[0]),
        "d": int(Z.shape[1]),
        "ols_overfit_bias_estimate": float(Z.shape[1] / max(Z.shape[0], 1)),
        "top10_acc": (
            correct / max(total, 1) if task_head is not None else None
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=50_000,
                   help="LEACE-fit subsample size (matches Phase 1).")
    p.add_argument("--n-drift-train", type=int, default=5_000,
                   help="Task-loss training subset size for the drift test.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--output-dir", type=str, default="results/v2_bios_DRIFT")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    device = _pick_device(args.device)
    output_dir = (
        Path(args.output_dir) if Path(args.output_dir).is_absolute()
        else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BIOS-medium DRIFT diagnostic")
    print(f"device={device}  seed={args.seed}  output_dir={output_dir}")
    print(f"LEACE fit on n={args.n_train}  task-loss train on n={args.n_drift_train}  "
          f"epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print("=" * 78)

    _seed_everything(args.seed)

    print("\n[1/5] Loading dataset, building DataLoaders...")
    t0 = time.time()
    train_loader, dev_loader, info, train_ds, dev_ds = build_bios_loaders(
        n_train=args.n_train,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    print(f"  n_train={info['n_train']}  n_dev={info['n_dev']}  "
          f"({time.time()-t0:.1f}s)")

    print("\n[2/5] Building BertWithLoRA + PEFT injection...")
    t1 = time.time()
    model = BertWithLoRA(
        rank=args.rank, alpha=args.alpha, dropout=args.dropout,
    ).to(device)
    print(f"  ({time.time()-t1:.1f}s)")

    print("\n[3/5] LEACE warm-start (post-projection registered)...")
    t2 = time.time()
    leace_diag = leace_warm_start_bert(model, train_loader, device=device)
    leace_dt = time.time() - t2
    print(f"  pre_r2_train={leace_diag['pre_r2_train']:.4f}  "
          f"post_r2_train_eraser_only={leace_diag['post_r2_train_eraser_only']:.4f}  "
          f"({leace_dt:.1f}s)")

    print("\n[4/5] Pre-training dev eval (full dev, no task head)...")
    t3 = time.time()
    pre_eval = _eval_dev(model, None, dev_loader, device)
    pre_dt = time.time() - t3
    print(f"  pre_dev_r2={pre_eval['r2_rounded']:.4f}  n_dev={pre_eval['n_dev']}  "
          f"d/N={pre_eval['ols_overfit_bias_estimate']:.4f}  ({pre_dt:.1f}s)")

    if pre_eval["r2"] > 0.05:
        raise RuntimeError(
            f"Pre-training dev R² = {pre_eval['r2_rounded']:.4f} > 0.05. "
            f"The earlier diagnostic with the same config reported 0.0438; "
            f"this run produced something different — investigate before "
            f"running the drift test."
        )

    print(f"\n[5/5] 1-epoch task-loss training on {args.n_drift_train}-sample subset...")
    rng = np.random.default_rng(args.seed)
    drift_idx = rng.choice(
        len(train_ds), size=args.n_drift_train, replace=False,
    )
    drift_ds = Subset(train_ds, drift_idx.tolist())
    drift_loader = DataLoader(
        drift_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    task_head = TaskHead(repr_dim=768, output_dim=10, hidden_dim=64).to(device)
    primal_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(task_head.parameters())
    )
    optim = torch.optim.AdamW(primal_params, lr=args.lr)
    n_batches = (len(drift_ds) + args.batch_size - 1) // args.batch_size

    t4 = time.time()
    train_history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        task_head.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for step, batch in enumerate(drift_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            occ = batch["occupation"].long().to(device)
            z = model(ids, mask)
            logits = task_head(z)
            loss = F.cross_entropy(logits, occ)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(primal_params, args.grad_clip)
            optim.step()
            running_loss += float(loss.item())
            pred = logits.argmax(-1)
            running_correct += int((pred == occ).sum().item())
            running_total += int(occ.numel())
            if step % args.log_every == 0 or step == n_batches - 1:
                avg_loss = running_loss / max(step + 1, 1)
                avg_acc = running_correct / max(running_total, 1)
                print(f"  ep {epoch} step {step:4d}/{n_batches}  "
                      f"loss={avg_loss:.4f}  train_acc={avg_acc:.4f}  "
                      f"({time.time()-t4:.1f}s)")
        train_history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(n_batches, 1),
            "train_acc": running_correct / max(running_total, 1),
        })
    train_dt = time.time() - t4
    print(f"  training complete: {train_dt:.1f}s  "
          f"final_train_loss={train_history[-1]['train_loss']:.4f}  "
          f"final_train_acc={train_history[-1]['train_acc']:.4f}")

    print("\n[6/6] Post-training dev eval (full dev, with trained task head)...")
    t5 = time.time()
    post_eval = _eval_dev(model, task_head, dev_loader, device)
    post_dt = time.time() - t5
    print(f"  post_dev_r2={post_eval['r2_rounded']:.4f}  "
          f"post_top10_acc={post_eval['top10_acc']:.4f}  "
          f"n_dev={post_eval['n_dev']}  ({post_dt:.1f}s)")

    delta_r2 = post_eval["r2"] - pre_eval["r2"]
    decision = "OPTION_A" if post_eval["r2"] <= 0.05 else "OPTION_D"
    print("\n" + "=" * 78)
    print("DRIFT VERDICT")
    print(f"  pre_dev_r2  = {pre_eval['r2_rounded']:.4f}")
    print(f"  post_dev_r2 = {post_eval['r2_rounded']:.4f}  "
          f"(Δ={delta_r2:+.4f})")
    print(f"  post_top10_acc = {post_eval['top10_acc']:.4f}")
    print(f"  decision: {decision}")
    if decision == "OPTION_A":
        print("  → post-projection holds. Phase 1 can train pure task loss; the")
        print("    structural projection enforces R² <= 0.05 throughout training.")
    else:
        print("  → post-projection drifts. Phase 1 needs proxy-Lagrangian on a")
        print("    held-out 4096-sample R² eval, refreshed every K=10 mini-batches.")
    print("=" * 78)

    out = {
        "info": info,
        "leace_warmstart": leace_diag,
        "pre_training_dev": pre_eval,
        "post_training_dev": post_eval,
        "delta_r2": delta_r2,
        "n_drift_train": args.n_drift_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "decision": decision,
        "device": str(device),
        "seed": args.seed,
        "train_history": train_history,
        "timings_s": {
            "leace_warmstart": float(leace_dt),
            "pre_eval": float(pre_dt),
            "training": float(train_dt),
            "post_eval": float(post_dt),
        },
    }
    out_path = output_dir / "drift.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved drift verdict to {out_path}")


if __name__ == "__main__":
    main()

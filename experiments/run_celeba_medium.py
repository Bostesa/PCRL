#!/usr/bin/env python3
"""CelebA-medium PCRL: predict Smiling, hide Male AND Young.

Single-purpose attribute-hiding pipeline isolated from the tabular v2 trainer.
Uses ResNet18 + frozen BN + PEFT LoRA on layer4.{0,1}.conv2, LEACE warm-start
with rank-8 SVD into layer4.1.conv2, and the existing
``pcrl.training.proxy_lagrangian.ProxyLagrangianOptimizer`` + VICReg aux.

Usage:
    python experiments/run_celeba_medium.py --preflight-only
        # Construction-only diagnostics (no training).
    python experiments/run_celeba_medium.py --train
        # Full training run (single seed, 25 epochs, batch 256).

Output directory: results/v2_celeba_ROUND1/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcrl.vision.backbone import (  # noqa: E402
    bn_state_summary,
    build_resnet18_erase_task,
    build_resnet18_lora,
    trainable_param_summary,
)
from pcrl.vision.dataset import CelebAMedium, collate  # noqa: E402
from pcrl.vision.leace_warmstart import (  # noqa: E402
    fit_and_set_erase_layer,
    warm_start,
)
from pcrl.vision.train import TrainConfig, train  # noqa: E402

LOG_FMT = "%(asctime)s [%(name)s/%(levelname)s] %(message)s"


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("cpu")  # MPS not used; CPU on Mac for preflight.
    return torch.device("cpu")


def _format_preflight_block(
    trainable: list[tuple[str, tuple[int, ...]]],
    bn_state: list[tuple[str, str, bool]],
    forward_trace: list[tuple[str, tuple[int, ...]]],
    diags,
    holdout_r2: dict[str, float] | None = None,
) -> str:
    lines: list[str] = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("PREFLIGHT REPORT — CelebA-medium PCRL")
    lines.append(sep)

    lines.append("\n[1] Trainable parameters after PEFT injection")
    lines.append("-" * 78)
    if not trainable:
        lines.append("  <none>")
    for name, shape in trainable:
        lines.append(f"  {name}  shape={shape}")

    lines.append("\n[2] BN-like submodule training-flag state (after .train())")
    lines.append("-" * 78)
    for name, cls, is_training in bn_state:
        lines.append(f"  {name:50s}  {cls:24s}  training={is_training}")

    lines.append("\n[3] Forward shape trace: input -> backbone -> penultimate")
    lines.append("-" * 78)
    for name, shape in forward_trace:
        lines.append(f"  {name:30s}  {shape}")

    lines.append("\n[4] Construction-time linear-R^2 (post-LEACE LoRA init, train set)")
    lines.append("-" * 78)
    lines.append(f"  R^2(h_512, Male)   = {diags.construction_r2['male']:.6f}")
    lines.append(f"  R^2(h_512, Young)  = {diags.construction_r2['young']:.6f}")
    if holdout_r2 is not None:
        lines.append("\n  Held-out generalization R^2 (informational; disjoint subsample)")
        lines.append(f"  R^2_holdout(Male)  = {holdout_r2['male']:.5f}")
        lines.append(f"  R^2_holdout(Young) = {holdout_r2['young']:.5f}")

    lines.append("\n[A] LEACE rank diagnostic (M = Q - I, expected rank <= 2)")
    lines.append("-" * 78)
    lines.append(f"  proj_left shape  = {diags.leace_proj_left_shape}")
    lines.append(f"  proj_right shape = {diags.leace_proj_right_shape}")
    lines.append(f"  numerical rank(M) = {diags.rank_M}")
    top_s = diags.singular_values_full[: min(8, len(diags.singular_values_full))]
    lines.append("  top-8 singular values:")
    for i, s in enumerate(top_s):
        lines.append(f"    sigma_{i} = {float(s):.6e}")
    lines.append(
        f"  rank-8 truncation residual ||M - M_8||_F / ||M||_F = {diags.truncation_residual:.6e}"
    )

    lines.append("\n[B] Pre/post LEACE linear-R^2 (per attribute, separate)")
    lines.append("-" * 78)
    lines.append(
        f"  pre  : R^2(Male)={diags.pre_r2['male']:.5f}   "
        f"R^2(Young)={diags.pre_r2['young']:.5f}"
    )
    lines.append(
        f"  closed (eraser): R^2(Male)={diags.post_r2_closed['male']:.5f}   "
        f"R^2(Young)={diags.post_r2_closed['young']:.5f}"
    )
    lines.append(
        f"  forward (LoRA-init): R^2(Male)={diags.construction_r2['male']:.5f}   "
        f"R^2(Young)={diags.construction_r2['young']:.5f}"
    )

    lines.append(f"\n  n_samples (LEACE fit subset): {diags.n_samples}")
    gate_pass = (
        diags.construction_r2["male"] <= 0.05
        and diags.construction_r2["young"] <= 0.05
    )
    verdict = "PASS" if gate_pass else "FAIL"
    lines.append(f"\n  CONSTRUCTION-TIME R^2 GATE (<= 0.05 each): {verdict}")
    lines.append(sep)
    return "\n".join(lines)


def _capture_forward_trace(
    peft_model: nn.Module, head: nn.Module, x: torch.Tensor
) -> list[tuple[str, tuple[int, ...]]]:
    """Trace input -> backbone (post-avgpool) -> penultimate_proj -> head."""
    from pcrl.vision.backbone import get_backbone_only
    trace: list[tuple[str, tuple[int, ...]]] = [("input", tuple(x.shape))]
    backbone = get_backbone_only(peft_model)
    with torch.no_grad():
        h_pre = backbone(x)
        trace.append(("backbone(x) [post-avgpool, pre-proj]", tuple(h_pre.shape)))
        h_proj = peft_model(x)
        trace.append(("penultimate_proj(backbone(x))", tuple(h_proj.shape)))
        y = head(h_proj)
        trace.append(("head(penultimate_proj(...))", tuple(y.shape)))
    return trace


def run_preflight(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = _device()
    logging.info(f"device = {device}")

    # Preflight uses one loader for fit-AND-probe (train-set R^2 must be ~0
    # to confirm wiring). A separate held-out loader gives a generalization
    # readout, informational only.
    n_fit = 60_000 if args.preflight_full else min(args.fit_n, 60_000)
    n_probe = 60_000 if args.preflight_full else min(args.probe_n, 60_000)

    fit_set = CelebAMedium(
        celeba_dir=args.celeba_dir, n=n_fit, train=False,
        image_size=args.image_size, seed=args.seed,
    )
    holdout_set = CelebAMedium(
        celeba_dir=args.celeba_dir, n=n_probe, train=False,
        image_size=args.image_size, seed=args.seed + 1,
    )
    fit_loader = DataLoader(
        fit_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=False,
    )
    holdout_loader = DataLoader(
        holdout_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=False,
    )

    if args.arch == "erase_task":
        peft_model, _ = build_resnet18_erase_task(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        )
    else:
        peft_model, _ = build_resnet18_lora(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        )
    peft_model.to(device)
    head = nn.Linear(512, 2).to(device)

    # [1] Trainable params after PEFT injection (head is separate).
    trainable = trainable_param_summary(peft_model)
    trainable += [
        (f"head.{n}", tuple(p.shape))
        for n, p in head.named_parameters() if p.requires_grad
    ]

    # [2] BN-like training-flag state after .train().
    peft_model.train()
    bn_state = bn_state_summary(peft_model)
    peft_model.eval()

    # [3] Forward shape trace.
    sample = next(iter(fit_loader))
    x = sample["image"][:2].to(device)
    forward_trace = _capture_forward_trace(peft_model, head, x)

    # [4] LEACE warm-start + construction-time R^2 + rank diag.
    # Probe == fit set (train-set R^2, the compliance metric).
    if args.arch == "erase_task":
        diags = fit_and_set_erase_layer(
            peft_model=peft_model,
            fit_loader=fit_loader,
            probe_loader=fit_loader,
            device=device,
        )
    else:
        diags = warm_start(
            peft_model=peft_model,
            fit_loader=fit_loader,
            probe_loader=fit_loader,
            device=device,
            rank=args.rank,
            alpha=args.alpha,
        )

    # Informational: held-out generalization R^2 on a disjoint stratified
    # subsample. Not the compliance gate, but a useful sanity check.
    from pcrl.vision.leace_warmstart import collect_penultimate_post  # noqa: E402
    from pcrl.vision.r2_helper import linear_r2  # noqa: E402
    H_h, m_h, y_h = collect_penultimate_post(peft_model, holdout_loader, device)
    holdout_r2 = {
        "male": linear_r2(H_h.numpy(), m_h.numpy()),
        "young": linear_r2(H_h.numpy(), y_h.numpy()),
    }

    block = _format_preflight_block(
        trainable, bn_state, forward_trace, diags, holdout_r2=holdout_r2,
    )
    print(block)
    log_path = out_dir / "preflight.log"
    log_path.write_text(block + "\n")
    json_path = out_dir / "preflight.json"
    json_path.write_text(
        json.dumps(
            {
                "trainable_parameters": [
                    {"name": n, "shape": list(s)} for n, s in trainable
                ],
                "bn_state": [
                    {"name": n, "class": c, "training": t} for n, c, t in bn_state
                ],
                "forward_trace": [
                    {"name": n, "shape": list(s)} for n, s in forward_trace
                ],
                "leace": {
                    "n_samples": diags.n_samples,
                    "rank_M": diags.rank_M,
                    "top_8_singular_values": [
                        float(s) for s in diags.singular_values_full[:8]
                    ],
                    "truncation_residual": diags.truncation_residual,
                    "leace_proj_left_shape": list(diags.leace_proj_left_shape),
                    "leace_proj_right_shape": list(diags.leace_proj_right_shape),
                    "pre_r2": diags.pre_r2,
                    "post_r2_closed": diags.post_r2_closed,
                    "construction_r2": diags.construction_r2,
                    "holdout_r2": holdout_r2,
                },
                "config": {
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "dropout": args.dropout,
                    "image_size": args.image_size,
                    "fit_n": n_fit,
                    "probe_n": n_probe,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                },
            },
            indent=2,
        )
    )
    print(f"\nWrote {log_path}")
    print(f"Wrote {json_path}")
    gate_pass = (
        diags.construction_r2["male"] <= 0.05
        and diags.construction_r2["young"] <= 0.05
    )
    return 0 if gate_pass else 2


def run_train(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = _device()
    logging.info(f"device = {device}")

    train_set = CelebAMedium(
        celeba_dir=args.celeba_dir, n=args.train_n, train=True,
        image_size=args.image_size, seed=args.seed,
    )
    val_set = CelebAMedium(
        celeba_dir=args.celeba_dir, n=args.val_n, train=False,
        image_size=args.image_size, partition=1, seed=args.seed,
    )
    holdout_set = CelebAMedium(
        celeba_dir=args.celeba_dir, n=args.holdout_n, train=False,
        image_size=args.image_size, partition=1, seed=args.seed + 7,
    )
    fit_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )
    holdout_loader = DataLoader(
        holdout_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    ) if args.holdout_every > 0 else None
    refit_set = CelebAMedium(
        celeba_dir=args.celeba_dir, n=args.refit_n, train=True,
        image_size=args.image_size, seed=args.seed + 13,
    ) if args.refit_every > 0 else None
    refit_loader = DataLoader(
        refit_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    ) if refit_set is not None else None

    if args.arch == "erase_task":
        peft_model, _ = build_resnet18_erase_task(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        )
    else:
        peft_model, _ = build_resnet18_lora(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
        )
    peft_model.to(device)
    head = nn.Linear(512, 2).to(device)

    if args.arch == "erase_task":
        diags = fit_and_set_erase_layer(
            peft_model=peft_model,
            fit_loader=fit_loader,
            probe_loader=val_loader,
            device=device,
        )
    else:
        diags = warm_start(
            peft_model=peft_model,
            fit_loader=fit_loader,
            probe_loader=val_loader,
            device=device,
            rank=args.rank,
            alpha=args.alpha,
        )
    skip_warmup = (
        diags.construction_r2["male"] <= 0.05
        and diags.construction_r2["young"] <= 0.05
    )
    logging.info(
        f"LEACE diag: rank(M)={diags.rank_M} "
        f"R2_male={diags.construction_r2['male']:.4f} "
        f"R2_young={diags.construction_r2['young']:.4f} "
        f"skip_warmup={skip_warmup}"
    )

    cfg = TrainConfig(
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_warmup_if_feasible=skip_warmup,
        holdout_every=args.holdout_every,
        lambda_init=args.lambda_init,
        eta_lambda=args.eta_lambda,
        r2_threshold=args.r2_threshold,
        refit_every=args.refit_every,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    history = train(
        wrapper=peft_model, head=head,
        train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, device=device, skip_warmup=skip_warmup,
        holdout_loader=holdout_loader,
        refit_loader=refit_loader,
    )
    print("Training complete.")
    print(json.dumps(
        {
            "selector": history.selector,
            "final_epoch": asdict(history.epochs[-1]) if history.epochs else None,
        },
        indent=2,
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "v2_celeba_ROUND1"))
    p.add_argument("--celeba-dir", default=str(REPO_ROOT / "data" / "celeba"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--fit-n", type=int, default=4096,
                   help="Preflight-only: subsample size for LEACE fit (default 4096).")
    p.add_argument("--probe-n", type=int, default=4096,
                   help="Preflight-only: subsample size for R^2 probe (default 4096).")
    p.add_argument("--val-n", type=int, default=4096,
                   help="Train: validation subset size from CelebA val partition.")
    p.add_argument("--train-n", type=int, default=60_000,
                   help="Train set size (default 60000; use 5000 for smoke).")
    p.add_argument("--holdout-n", type=int, default=4096,
                   help="Fixed held-out subset for per-step R^2 logging.")
    p.add_argument("--holdout-every", type=int, default=10,
                   help="Run held-out R^2 every K primal steps (0 disables).")
    p.add_argument("--lambda-init", type=float, default=5.0,
                   help="C2 default 5 (sum-form surrogate already large).")
    p.add_argument("--eta-lambda", type=float, default=0.02,
                   help="Standard proxy-Lagrangian dual step size.")
    p.add_argument("--r2-threshold", type=float, default=2.5,
                   help="C2 empirical: surrogate noise floor ~2.0 at LEACE init.")
    p.add_argument("--refit-every", type=int, default=0,
                   help="C3: re-fit LEACE and overwrite LoRA every K steps (0=off).")
    p.add_argument("--refit-n", type=int, default=4096,
                   help="C3: fixed buffer size for LEACE re-fit pass.")
    p.add_argument("--arch", choices=["legacy", "erase_task"], default="erase_task",
                   help="legacy = LEACE-init LoRA on penultimate_proj. "
                        "erase_task = frozen LEACE Linear + LoRA on downstream task_proj.")
    p.add_argument("--preflight-full", action="store_true",
                   help="Use full 60k subsample in preflight (slow on CPU).")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    args = build_parser().parse_args()
    if args.preflight_only:
        return run_preflight(args)
    if args.train:
        return run_train(args)
    print(
        "Specify --preflight-only or --train. See `--help` for options.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

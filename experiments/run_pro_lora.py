"""PRO-LoRA training entry point — Components 1+2+3+4 wired together.

PRO-LoRA = Pre-projected residual + LoRA encoder. Components:
  C1 — EMA-streaming LEACE fitter (pcrl/language/ema_leace.py)
  C2 — pre-projection hooks at layers {0, 6, 11} (pcrl/language/pro_lora.py)
  C3 — counterfactual data augmentation + pair-invariance loss
       (pcrl/language/cda_invariance.py + bios_dataset.include_scrubbed)
  C4 — OGDA dual update (pcrl/language/dual_controllers.OGDADualUpdate)

Driver flags select Checkpoint mode:
  --checkpoint 1  → C1+C2 only, task loss + HSIC primal w/ OGDA. Eval linear-
                    probe R² ≤ 0.05 on dev.
  --checkpoint 2  → C1+C2+C3. Adds CDA pair-invariance. Eval nonlinear MLP-
                    probe R² ≤ 0.35 on dev.
  --full          → All four wired. Same training as ckpt 2 + bail conditions
                    + final TPR-RMS evaluation.

Smoke flag ``--smoke`` reduces n_train, epochs, max_steps for fast Mac
integration sanity checks — NOT for evaluating the pass criterion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `python experiments/run_pro_lora.py ...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcrl.language.bios_dataset import build_bios_loaders, BIOS_TOP10
from pcrl.language.cda_invariance import pair_invariance_loss
from pcrl.language.dual_controllers import OGDADualUpdate
from pcrl.language.hsic import gender_one_hot, nhsic_linear
from pcrl.language.pro_lora import ProLoRAEncoder
from pcrl.language.tpr_gap import (
    tpr_gap_summary, tpr_gaps_per_occupation, theil_adjusted_r2,
)


# ---------------------------------------------------------------------------
# Linear / MLP probe utilities
# ---------------------------------------------------------------------------

def population_linear_r2(X: torch.Tensor, Z: torch.Tensor) -> float:
    """Σ_xz^T · pinv(Σ_xx) · Σ_xz / Var(z), pinv with rcond=1e-10 to drop
    rank-deficient directions cleanly post-erasure.
    """
    X = X.to(torch.float64); Z = Z.to(torch.float64).view(-1, 1)
    Xc = X - X.mean(0); Zc = Z - Z.mean(0)
    n = X.shape[0]
    Sxx = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    var_z = Zc.var(unbiased=False)
    Sxx_inv = torch.linalg.pinv(Sxx, rcond=1e-10)
    return float((Sxz.T @ Sxx_inv @ Sxz / max(var_z.item(), 1e-12)).item())


def mlp_probe_r2(
    X_train: torch.Tensor, Z_train: torch.Tensor,
    X_test: torch.Tensor, Z_test: torch.Tensor,
    *, hidden: int = 64, epochs: int = 200, lr: float = 1e-2,
    weight_decay: float = 1e-3, device: str = "cpu",
) -> float:
    """Two-layer MLP probe with held-out evaluation."""
    Xtr = X_train.to(device).to(torch.float32)
    Ztr = Z_train.to(device).to(torch.float32).view(-1, 1)
    Xte = X_test.to(device).to(torch.float32)
    Zte = Z_test.to(device).to(torch.float32).view(-1, 1)
    d = Xtr.shape[1]
    net = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1)
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(Xtr)
        loss = F.mse_loss(pred, Ztr)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte)
        ss_res = ((Zte - pred) ** 2).sum().item()
        ss_tot = ((Zte - Zte.mean()) ** 2).sum().item()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


# ---------------------------------------------------------------------------
# Forward / loss helpers
# ---------------------------------------------------------------------------

class TaskHead(nn.Module):
    def __init__(self, d_in: int = 768, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.fc(x)


def collect_dev_cls(encoder, head, dev_loader, device, max_batches: int | None = None):
    """Single forward pass over dev (no eraser observe). Returns CLS, gender,
    occupation tensors."""
    encoder.eval()
    cls_buf, gen_buf, occ_buf, logit_buf = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            if max_batches is not None and i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cls = encoder(ids, mask)
            logit = head(cls)
            cls_buf.append(cls.cpu()); logit_buf.append(logit.cpu())
            gen_buf.append(batch["gender"]); occ_buf.append(batch["occupation"])
    encoder.train()
    return (
        torch.cat(cls_buf), torch.cat(gen_buf), torch.cat(occ_buf),
        torch.cat(logit_buf),
    )


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pro_lora] device={device} checkpoint={args.checkpoint} out={out_dir}")
    torch.manual_seed(args.seed)

    use_cda = args.checkpoint >= 2 or args.full

    # ---- Data --------------------------------------------------------------
    n_train = args.n_train if not args.smoke else min(args.n_train, 1024)
    print(f"[pro_lora] loading BIOS-medium n_train={n_train} include_scrubbed={use_cda}")
    train_loader, dev_loader, info, train_ds, dev_ds = build_bios_loaders(
        n_train=n_train, batch_size=args.batch_size, seed=args.seed,
        max_length=args.max_length, num_workers=0, include_scrubbed=use_cda,
    )
    print(f"[pro_lora] n_train={info['n_train']} n_dev={info['n_dev']}")

    # ---- Model -------------------------------------------------------------
    enc = ProLoRAEncoder(
        rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        hook_layers=tuple(args.hook_layers),
        ema_alpha=args.ema_alpha, refit_every=args.refit_every, z_dim=1,
    ).to(device)
    head = TaskHead(d_in=768, n_classes=10).to(device)

    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad) \
              + sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[pro_lora] trainable params: {trainable:,}")

    opt = torch.optim.AdamW(
        [p for p in enc.parameters() if p.requires_grad]
        + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---- Dual --------------------------------------------------------------
    use_ogda = args.full or args.checkpoint >= 1
    dual = OGDADualUpdate(
        eta=args.ogda_eta, lambda_min=0.0, lambda_max=args.lambda_max,
        lambda_init=args.lambda_init,
    ) if use_ogda else None

    # ---- Run loop ----------------------------------------------------------
    start = time.time()
    global_step = 0
    monitor: list[dict] = []
    epochs = args.epochs if not args.smoke else 1
    max_steps = args.max_steps  # 0 = unlimited
    bailed = False
    bail_reason = None

    for epoch in range(epochs):
        for batch in train_loader:
            if max_steps and global_step >= max_steps:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            occ = batch["occupation"].to(device)
            gen = batch["gender"].to(device)

            # Forward x — primary task signal + capture residuals.
            cls = enc(ids, mask)
            logit = head(cls)
            loss_task = F.cross_entropy(logit, occ)

            # Optional CDA branch.
            loss_inv = torch.tensor(0.0, device=device)
            if use_cda:
                ids_cf = batch["input_ids_scrubbed"].to(device)
                mask_cf = batch["attention_mask_scrubbed"].to(device)
                cls_cf = enc(ids_cf, mask_cf)
                logit_cf = head(cls_cf)
                loss_task = loss_task + F.cross_entropy(logit_cf, occ)
                loss_inv = pair_invariance_loss(cls, cls_cf)

            # HSIC primal constraint on final [CLS].
            primal = nhsic_linear(cls, gender_one_hot(gen).to(device))
            lam = float(dual.lam) if dual is not None else 0.0
            loss = loss_task + args.lambda_inv * loss_inv + lam * primal

            opt.zero_grad(); loss.backward(); opt.step()

            # PRO-LoRA: feed pre-projection [CLS] into per-layer EMA fitters,
            # then refit erasers at cadence.
            with torch.no_grad():
                # Captures from THIS step's primary forward are still in
                # encoder._hook_handles. observe_residuals() consumes them.
                enc.observe_residuals(gen)
                refits = enc.refit_erasers(global_step)

            # Dual step (OGDA).
            if dual is not None:
                # Use the same nHSIC value as the constraint signal —
                # mathematically the gradient of the Lagrangian wrt λ.
                dual.step(g_t=float(primal.detach().item()))

            if global_step % args.log_every == 0:
                snap = {
                    "step": global_step,
                    "epoch": epoch,
                    "elapsed_s": time.time() - start,
                    "loss_task": float(loss_task.item()),
                    "loss_inv": float(loss_inv.item()) if use_cda else 0.0,
                    "primal_nhsic": float(primal.item()),
                    "lambda": lam,
                    "ema_steps": {li: enc._hook_handles[li].fitter.t for li in enc.hook_layers},
                    "refit_count": {li: enc._hook_handles[li].refit_count for li in enc.hook_layers},
                }
                monitor.append(snap)
                print(
                    f"  step {global_step:5d} ep{epoch} t={snap['elapsed_s']:6.1f}s  "
                    f"L_task={snap['loss_task']:.3f}  L_inv={snap['loss_inv']:.4f}  "
                    f"nHSIC={snap['primal_nhsic']:.4f}  λ={lam:.3f}  "
                    f"refits={snap['refit_count']}"
                )
            global_step += 1

            # 30-min hard cap for smoke timing.
            if args.hard_cap_s and (time.time() - start) >= args.hard_cap_s:
                bailed = True
                bail_reason = f"hard_cap_s={args.hard_cap_s}"
                break
        if bailed or (max_steps and global_step >= max_steps):
            break

    # ---- Eval --------------------------------------------------------------
    eval_batches = args.eval_batches if args.eval_batches > 0 else None
    print(f"[pro_lora] eval on dev (max_batches={eval_batches}) ...")
    cls_dev, gen_dev, occ_dev, logit_dev = collect_dev_cls(
        enc, head, dev_loader, device, max_batches=eval_batches,
    )
    n_eval = cls_dev.shape[0]
    pred = logit_dev.argmax(-1)
    top10 = float((pred == occ_dev).float().mean().item())
    tpr_gaps = tpr_gaps_per_occupation(
        pred.numpy(), occ_dev.numpy(), gen_dev.numpy(), n_occupations=10,
    )
    tpr_summary = tpr_gap_summary(tpr_gaps)
    lin_r2 = population_linear_r2(cls_dev, gen_dev.float())

    # MLP probe (only used at Checkpoint 2 / full).
    mlp_r2 = None
    if args.checkpoint >= 2 or args.full:
        idx = torch.randperm(n_eval)
        half = n_eval // 2
        mlp_r2 = mlp_probe_r2(
            cls_dev[idx[:half]], gen_dev[idx[:half]].float(),
            cls_dev[idx[half:]], gen_dev[idx[half:]].float(),
            device=str(device),
        )

    summary = {
        "checkpoint": args.checkpoint,
        "full": args.full,
        "smoke": args.smoke,
        "elapsed_s": time.time() - start,
        "global_step": global_step,
        "n_train": info["n_train"],
        "n_dev": info["n_dev"],
        "top10_acc": top10,
        "tpr_rms": float(tpr_summary["rms_gap"]),
        "tpr_max": float(tpr_summary["max_abs_gap"]),
        "tpr_per_occupation": tpr_summary["per_occupation"],
        "linear_probe_r2": lin_r2,
        "mlp_probe_r2": mlp_r2,
        "bailed": bailed,
        "bail_reason": bail_reason,
        "args": vars(args),
    }
    print("[pro_lora] summary:", json.dumps(summary, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "monitor.jsonl").write_text(
        "\n".join(json.dumps(m) for m in monitor)
    )

    # Pass criteria.
    if args.checkpoint == 1:
        passed = lin_r2 <= 0.05
        print(f"\n[ckpt1 gate] linear-probe R² = {lin_r2:.4f} (target ≤ 0.05) → {'PASS' if passed else 'FAIL'}")
    elif args.checkpoint == 2:
        passed = mlp_r2 is not None and mlp_r2 <= 0.35
        print(f"\n[ckpt2 gate] MLP-probe R² = {mlp_r2:.4f} (target ≤ 0.35) → {'PASS' if passed else 'FAIL'}")
    else:
        passed = (summary["tpr_rms"] <= 0.10) or (lin_r2 <= 0.10)
        print(f"\n[full gate] TPR-RMS={summary['tpr_rms']:.4f} or lin-R²={lin_r2:.4f} → "
              f"{'PASS' if passed else 'FAIL'}")
    summary["gate_passed"] = bool(passed)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not passed and args.exit_nonzero_on_fail:
        sys.exit(2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=int, choices=[1, 2], default=1)
    p.add_argument("--full", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-train", dest="n_train", type=int, default=50_000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--max-length", dest="max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=0)
    p.add_argument("--hard-cap-s", dest="hard_cap_s", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--lora-rank", dest="lora_rank", type=int, default=32)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=64)
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.05)
    p.add_argument("--hook-layers", dest="hook_layers", type=int, nargs="+", default=[0, 6, 11])
    p.add_argument("--ema-alpha", dest="ema_alpha", type=float, default=0.02)
    p.add_argument("--refit-every", dest="refit_every", type=int, default=5)
    p.add_argument("--lambda-inv", dest="lambda_inv", type=float, default=1.0)
    p.add_argument("--lambda-init", dest="lambda_init", type=float, default=1.0)
    p.add_argument("--lambda-max", dest="lambda_max", type=float, default=100.0)
    p.add_argument("--ogda-eta", dest="ogda_eta", type=float, default=10.0)
    p.add_argument("--log-every", dest="log_every", type=int, default=20)
    p.add_argument("--eval-batches", dest="eval_batches", type=int, default=0,
                   help="Max dev batches at eval time (0 = full dev set).")
    p.add_argument("--output-dir", dest="output_dir", default="results/v2_bios_PRO_LORA")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

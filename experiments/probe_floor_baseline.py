"""Step 2 of the deep-research audit: empirical floor for R²([CLS], gender)
on a vanilla BERT+LoRA trained with task loss only on BIOS top-10.

Hypothesis: post-training [CLS] R² is bounded below by the analytic floor
implied by the gender↔occupation coupling in the training labels. If we
observe R² ≈ 0.25-0.35 here, then PRO-LoRA's 0.24 sits AT the floor and
the ≤0.05 gate was unreachable in principle.

Output: JSON with linear-probe R², MLP-probe R², top-10 accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.language.bert_encoder import BertWithLoRA, CLS_DIM  # noqa: E402
from pcrl.language.bios_dataset import build_bios_loaders  # noqa: E402


def population_linear_r2(X: torch.Tensor, Z: torch.Tensor) -> float:
    X = X.to(torch.float64); Z = Z.to(torch.float64).view(-1, 1)
    Xc = X - X.mean(0); Zc = Z - Z.mean(0)
    n = X.shape[0]
    Sxx = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    var_z = Zc.var(unbiased=False)
    Sxx_inv = torch.linalg.pinv(Sxx, rcond=1e-10)
    return float((Sxz.T @ Sxx_inv @ Sxz / max(var_z.item(), 1e-12)).item())


def mlp_probe_r2(Xtr, Ztr, Xte, Zte, *, hidden=64, epochs=200, lr=1e-2, wd=1e-3):
    Xtr = Xtr.float(); Ztr = Ztr.float().view(-1, 1)
    Xte = Xte.float(); Zte = Zte.float().view(-1, 1)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(net(Xtr), Ztr)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte)
        ss_res = ((Zte - pred) ** 2).sum().item()
        ss_tot = ((Zte - Zte.mean()) ** 2).sum().item()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def analytic_floor_r2(occ: torch.Tensor, gen: torch.Tensor, n_classes: int = 10) -> float:
    """Var(E[G|Y]) / Var(G): the R² of the optimal task-correlated linear
    predictor of gender given occupation labels. Lower bound on R²([CLS], G)
    for any task-accurate representation by data-processing inequality.
    """
    g = gen.float()
    floor = torch.zeros_like(g)
    for c in range(n_classes):
        m = (occ == c)
        if m.any():
            floor[m] = g[m].mean()
    var_floor = floor.var(unbiased=False).item()
    var_g = g.var(unbiased=False).item()
    return float(var_floor / max(var_g, 1e-12))


@torch.no_grad()
def encode_split(encoder, loader, head, device, max_batches: int | None = None):
    encoder.eval()
    cls_buf, gen_buf, occ_buf, logit_buf = [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cls = encoder(ids, mask)
        logit = head(cls)
        cls_buf.append(cls.cpu()); logit_buf.append(logit.cpu())
        gen_buf.append(batch["gender"]); occ_buf.append(batch["occupation"])
    encoder.train()
    return (torch.cat(cls_buf), torch.cat(gen_buf), torch.cat(occ_buf), torch.cat(logit_buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-eval-batches", type=int, default=80,
                    help="Cap dev encoding to keep wall-clock short")
    ap.add_argument("--output", default="results/floor_baseline.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[floor-probe] device={device} n_train={args.n_train} epochs={args.epochs}")
    t0 = time.time()

    train_loader, dev_loader, info, train_ds, dev_ds = build_bios_loaders(
        n_train=args.n_train, seed=args.seed,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"[floor-probe] BIOS loaded n_train={info['n_train']} n_dev={info['n_dev']} "
          f"@ {time.time()-t0:.1f}s")

    # Analytic floor from training labels (the core deliverable for §5.5)
    floor_train = analytic_floor_r2(train_ds.occupation, train_ds.gender.float())
    floor_dev = analytic_floor_r2(dev_ds.occupation, dev_ds.gender.float())
    print(f"[floor-probe] analytic floor R²(train labels) = {floor_train:.4f}")
    print(f"[floor-probe] analytic floor R²(dev labels)   = {floor_dev:.4f}")

    encoder = BertWithLoRA(rank=32, alpha=64, dropout=0.05).to(device)
    head = nn.Linear(CLS_DIM, 10).to(device)
    params = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    encoder.train()
    step = 0
    for ep in range(args.epochs):
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            occ = batch["occupation"].to(device)
            cls = encoder(ids, mask)
            logits = head(cls)
            loss = F.cross_entropy(logits, occ)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 25 == 0:
                acc = (logits.argmax(-1) == occ).float().mean().item()
                print(f"  step={step} loss={loss.item():.4f} acc={acc:.3f} "
                      f"@ {time.time()-t0:.1f}s")

    print(f"[floor-probe] training done @ {time.time()-t0:.1f}s — encoding dev")
    cls_dev, gen_dev, occ_dev, logit_dev = encode_split(
        encoder, dev_loader, head, device, max_batches=args.max_eval_batches,
    )
    n = cls_dev.shape[0]
    print(f"[floor-probe] dev encoded n={n} @ {time.time()-t0:.1f}s")

    # Population linear-probe R² (matches CKPT1 metric exactly)
    lin_r2 = population_linear_r2(cls_dev, gen_dev.float())

    # Held-out MLP probe (split dev 50/50)
    perm = torch.randperm(n)
    half = n // 2
    Xtr, Xte = cls_dev[perm[:half]], cls_dev[perm[half:]]
    Ztr, Zte = gen_dev[perm[:half]].float(), gen_dev[perm[half:]].float()
    mlp_r2 = mlp_probe_r2(Xtr, Ztr, Xte, Zte)

    top10_acc = (logit_dev.argmax(-1) == occ_dev).float().mean().item()

    # Per-occupation gender share — gives the direction/strength of leakage
    per_occ = []
    for c in range(10):
        m = (occ_dev == c)
        if m.any():
            per_occ.append({"occ": c, "n": int(m.sum()),
                            "p_female": float(gen_dev[m].float().mean())})

    summary = {
        "n_train": int(info["n_train"]), "n_dev_eval": int(n),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "lr": args.lr,
        "elapsed_s": time.time() - t0,
        "analytic_floor_r2_train_labels": floor_train,
        "analytic_floor_r2_dev_labels": floor_dev,
        "linear_probe_r2_dev": lin_r2,
        "mlp_probe_r2_dev_holdout": mlp_r2,
        "top10_acc_dev": float(top10_acc),
        "per_occupation_gender": per_occ,
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== FLOOR PROBE SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_occupation_gender"}, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()

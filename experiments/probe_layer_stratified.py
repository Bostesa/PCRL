"""Step 3: layer-stratified probe — where does gender live in [CLS]?

Trains two encoders on the same n=4000 BIOS top-10 subset:
  (A) vanilla BertWithLoRA + task loss only
  (B) ProLoRAEncoder with C1+C2 active, hooks at layers {0, 6, 11}, task loss
      + λ·nHSIC dual.

For each, encodes the same dev sample and computes:
  R²([CLS]_layer-L, gender) for L in {embedding, 1..12}

For (B), additionally captures the *pre-projection* [CLS] at the hook layers
(via the EMA observation pathway) so we can see the projection's local effect.

Output: a side-by-side table of R² across layers — diagnoses (i) where
gender naturally accumulates, (ii) whether the eraser sticks at hooked
layers, (iii) whether re-injection happens between hooks.
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
from pcrl.language.pro_lora import ProLoRAEncoder  # noqa: E402
from pcrl.language.dual_controllers import OGDADualUpdate  # noqa: E402
from pcrl.language.hsic import nhsic_linear  # noqa: E402


def population_linear_r2(X, Z):
    X = X.to(torch.float64); Z = Z.to(torch.float64).view(-1, 1)
    Xc = X - X.mean(0); Zc = Z - Z.mean(0)
    n = X.shape[0]
    Sxx = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    var_z = Zc.var(unbiased=False)
    Sxx_inv = torch.linalg.pinv(Sxx, rcond=1e-10)
    return float((Sxz.T @ Sxx_inv @ Sxz / max(var_z.item(), 1e-12)).item())


def train_vanilla(encoder, head, loader, opt, device, *, epochs=1, log_every=25):
    encoder.train(); head.train()
    step = 0
    t0 = time.time()
    for _ in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            occ = batch["occupation"].to(device)
            cls = encoder(ids, mask)
            logits = head(cls)
            loss = F.cross_entropy(logits, occ)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % log_every == 0:
                acc = (logits.argmax(-1) == occ).float().mean().item()
                print(f"   [vanilla] step={step} loss={loss.item():.3f} "
                      f"acc={acc:.3f} @ {time.time()-t0:.1f}s")


def train_prolora(encoder, head, loader, opt, dual, device, *,
                  epochs=1, log_every=25, refit_every=5, lambda_max=100.0):
    encoder.train(); head.train()
    step = 0
    t0 = time.time()
    for _ in range(epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            occ = batch["occupation"].to(device)
            gen = batch["gender"].to(device).float()

            cls = encoder(ids, mask)
            logits = head(cls)
            task_loss = F.cross_entropy(logits, occ)
            hsic = nhsic_linear(cls, gen.view(-1, 1))
            lam = float(dual.lam)
            loss = task_loss + lam * hsic
            opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                encoder.observe_residuals(gen)
                encoder.refit_erasers(step)
                dual.step(float(hsic.detach()))

            step += 1
            if step % log_every == 0:
                acc = (logits.argmax(-1) == occ).float().mean().item()
                print(f"   [pro_lora] step={step} loss={loss.item():.3f} "
                      f"acc={acc:.3f} hsic={float(hsic):.4f} lam={lam:.2f} "
                      f"@ {time.time()-t0:.1f}s")


@torch.no_grad()
def per_layer_cls_capture(model: nn.Module, loader, device, *, max_batches: int):
    """Returns (hidden_per_layer, gender_all). hidden_per_layer is a list of
    13 tensors (embedding + 12 layers), each of shape (N, 768) — the [CLS]
    token at each layer.
    """
    model.eval()
    bert = model.peft_model.base_model.model  # underlying BertModel
    bufs: list[list[torch.Tensor]] = [[] for _ in range(13)]
    g_buf = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = bert(input_ids=ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        for L, h in enumerate(out.hidden_states):
            bufs[L].append(h[:, 0, :].detach().cpu())
        g_buf.append(batch["gender"])
    model.train()
    layers = [torch.cat(b) for b in bufs]
    gender = torch.cat(g_buf).float()
    return layers, gender


def per_layer_r2(layers: list[torch.Tensor], gender: torch.Tensor) -> list[float]:
    return [population_linear_r2(L, gender) for L in layers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-eval-batches", type=int, default=80)
    ap.add_argument("--refit-every", type=int, default=5)
    ap.add_argument("--ema-alpha", type=float, default=0.02)
    ap.add_argument("--ogda-eta", type=float, default=10.0)
    ap.add_argument("--lambda-init", type=float, default=1.0)
    ap.add_argument("--lambda-max", type=float, default=100.0)
    ap.add_argument("--output", default="results/layer_stratified_probe.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[layer-probe] device={device} n_train={args.n_train}")
    t0 = time.time()

    train_loader, dev_loader, info, train_ds, dev_ds = build_bios_loaders(
        n_train=args.n_train, seed=args.seed,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    print(f"[layer-probe] BIOS loaded n_train={info['n_train']} n_dev={info['n_dev']} "
          f"@ {time.time()-t0:.1f}s")

    # ===== (A) Vanilla =====
    print("\n[layer-probe] training VANILLA BERT+LoRA (task-only)...")
    enc_v = BertWithLoRA(rank=32, alpha=64, dropout=0.05).to(device)
    head_v = nn.Linear(CLS_DIM, 10).to(device)
    opt_v = torch.optim.AdamW(
        [p for p in enc_v.parameters() if p.requires_grad] + list(head_v.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    train_vanilla(enc_v, head_v, train_loader, opt_v, device, epochs=args.epochs)

    print(f"[layer-probe] vanilla trained @ {time.time()-t0:.1f}s — capturing per-layer CLS")
    layers_v, gen_v = per_layer_cls_capture(enc_v, dev_loader, device, max_batches=args.max_eval_batches)
    r2_vanilla = per_layer_r2(layers_v, gen_v)
    print(f"[layer-probe] vanilla per-layer R² (n={layers_v[0].shape[0]}):")
    for L, r in enumerate(r2_vanilla):
        tag = "embedding" if L == 0 else f"layer_{L:02d}"
        print(f"   {tag}  R² = {r:.4f}")

    del enc_v, head_v, opt_v
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ===== (B) PRO-LoRA =====
    print("\n[layer-probe] training PRO-LoRA (C1+C2+OGDA, hooks at {0,6,11})...")
    enc_p = ProLoRAEncoder(
        rank=32, alpha=64, dropout=0.05,
        hook_layers=(0, 6, 11), ema_alpha=args.ema_alpha,
        refit_every=args.refit_every, z_dim=1,
    ).to(device)
    head_p = nn.Linear(CLS_DIM, 10).to(device)
    opt_p = torch.optim.AdamW(
        [p for p in enc_p.parameters() if p.requires_grad] + list(head_p.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    dual = OGDADualUpdate(
        eta=args.ogda_eta, lambda_min=0.0,
        lambda_max=args.lambda_max, lambda_init=args.lambda_init,
    )
    train_prolora(enc_p, head_p, train_loader, opt_p, dual, device,
                  epochs=args.epochs, refit_every=args.refit_every,
                  lambda_max=args.lambda_max)

    print(f"[layer-probe] pro_lora trained @ {time.time()-t0:.1f}s — capturing per-layer CLS")
    layers_p, gen_p = per_layer_cls_capture(enc_p, dev_loader, device, max_batches=args.max_eval_batches)
    r2_prolora = per_layer_r2(layers_p, gen_p)
    print(f"[layer-probe] pro_lora per-layer R² (n={layers_p[0].shape[0]}):")
    for L, r in enumerate(r2_prolora):
        tag = "embedding" if L == 0 else f"layer_{L:02d}"
        print(f"   {tag}  R² = {r:.4f}")

    enc_p.remove_hooks()

    # ===== Side-by-side =====
    print("\n=== LAYER-STRATIFIED R²([CLS]_layer, gender) ===")
    print(f"{'layer':>11} | {'vanilla':>9} | {'pro_lora':>9} | {'Δ':>9}")
    print("-" * 46)
    delta = []
    for L in range(13):
        tag = "embedding" if L == 0 else f"layer_{L:02d}"
        d = r2_prolora[L] - r2_vanilla[L]
        delta.append(d)
        marker = "  <hook" if L - 1 in (0, 6, 11) else ""  # hook output appears at hidden_states[L+1]
        print(f"{tag:>11} | {r2_vanilla[L]:>9.4f} | {r2_prolora[L]:>9.4f} | {d:>+9.4f}{marker}")

    summary = {
        "n_train": int(info["n_train"]),
        "n_dev_eval": int(layers_v[0].shape[0]),
        "epochs": args.epochs,
        "elapsed_s": time.time() - t0,
        "hook_layers": [0, 6, 11],
        "vanilla": {f"layer_{L}": r2_vanilla[L] for L in range(13)},
        "pro_lora": {f"layer_{L}": r2_prolora[L] for L in range(13)},
        "delta": {f"layer_{L}": delta[L] for L in range(13)},
        "args": vars(args),
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()

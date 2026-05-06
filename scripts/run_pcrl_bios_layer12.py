"""Single-cell PCRL driver on cached BERT layer-12 [CLS] features (BIOS).

One ``(purpose, seed)`` cell per invocation. Reads cached embeddings produced
by ``scripts/cache_bert_embeddings.py``, fits LEACE on (train_features, gender)
to derive a warm-start, and trains:

* identity host Linear (frozen, init = I_768)
* LoRA adapter (rank=16, alpha=16, dropout=0.0) — the only repr-side trainable
* linear task head (768 -> num_classes_purpose)

Loss / optimiser:
* task: cross-entropy on the purpose-specific labels.
* compliance: linear-R²(gender) on the batch representations, enforced via a
  proxy-Lagrangian dual variable (warmup λ→λ_max over first 30%, lr_λ=5e-3,
  λ_max=10, λ ∈ [0, 50]).
* VICReg variance: λ_var=5.0 · mean(ReLU(γ - per_dim_std)) with γ=1.0.
* VICReg covariance: λ_cov=0.5 · off-diagonal cov^2 / d.
* AdamW, no weight decay on LoRA (β=(0.9, 0.999)). Asymmetric LoRA+ LRs:
  lr_A=1e-4, lr_B=3e-4. Task head and LoRA bias use 2e-4.

The warm-start makes the very first forward equal to LEACE-erased features:
identity(x) + adapter(x) ≡ eraser(x). We verify R²(gender) on those step-0
features and abort if it exceeds 0.01 (warm-start broken).

Outputs (per ``--output-dir``):
    encoder_{purpose}_{seed}.pt    — projection-head + task-head state_dict
    eval_reps_{purpose}_{seed}.npz — train/dev representations after training
    metrics_{purpose}_{seed}.json  — training curves, final stats

Usage:
    python scripts/run_pcrl_bios_layer12.py \\
        --purpose P1 --seed 0 \\
        --cache-dir results/bios_pcrl_layer12/cache \\
        --output-dir results/bios_pcrl_layer12 \\
        --epochs 10 --batch-size 256 --device cuda
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
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.models.lora import LoRAAdapter  # noqa: E402
from pcrl.training.proxy_lagrangian import Constraint  # noqa: E402


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Frozen identity Linear + trainable LoRA adapter on cached [CLS] features.

    Out = identity(x) + LoRA(x). At warm-start the LoRA realises the LEACE
    map, so the very first forward produces eraser-erased features.
    """

    def __init__(
        self, dim: int = 768, rank: int = 16, alpha: int = 16, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        host = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            host.weight.copy_(torch.eye(dim))
            host.bias.zero_()
        for p in host.parameters():
            p.requires_grad_(False)
        self.host = host
        self.adapter = LoRAAdapter(
            in_features=dim, out_features=dim, rank=rank, alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.host(x) + self.adapter(x)
        if hasattr(self, "leace_P"):
            P = self.leace_P
            mu = self.leace_mu
            h = mu + (h - mu) @ P.T
        return h

    @torch.no_grad()
    def set_leace_projection(
        self, P: torch.Tensor, mu: torch.Tensor,
    ) -> None:
        """Register a frozen LEACE post-projection (mirrors
        ``pcrl/language/bert_encoder.py``).

        After this, every ``forward`` returns ``mu + (h - mu) @ P.T`` where
        ``h`` is the identity-+-LoRA output. The buffer is non-trainable but
        gradients still flow *through* it during backprop, so the LoRA learns
        inside the null space defined by ``P``. Idempotent w.r.t. the LoRA-
        side LEACE warm-start (P^2 = P for the LEACE projection).
        """
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = P.device
        self.register_buffer(
            "leace_P", P.detach().to(target_device).clone(), persistent=True,
        )
        self.register_buffer(
            "leace_mu", mu.detach().to(target_device).clone(), persistent=True,
        )

    @torch.no_grad()
    def warm_start_from_eraser(
        self, eraser_P: torch.Tensor, eraser_mu: torch.Tensor,
    ) -> None:
        """Initialise the LoRA so identity(x) + adapter(x) ≡ eraser(x).

        concept_erasure's eraser maps row-vector x via
            eraser(x) = (x - mu) @ P.T + mu = x @ P.T + mu @ (I - P.T)
        With identity host (W=I, b=0) the adapter must contribute
            adapter(x) = x @ (P - I).T + (mu - mu @ P.T)
        We SVD-truncate ``P - I`` to rank r. For binary Z (gender) the
        cross-cov has rank 1, so any r ≥ 1 is exact in float32.
        """
        d = eraser_P.shape[0]
        if eraser_P.shape != (d, d):
            raise ValueError(f"P must be (d,d); got {tuple(eraser_P.shape)}")
        if eraser_mu.shape != (d,):
            raise ValueError(f"mu must be (d,); got {tuple(eraser_mu.shape)}")
        target = eraser_P.to(torch.float32) - torch.eye(
            d, dtype=torch.float32, device=eraser_P.device,
        )
        U, S, Vh = torch.linalg.svd(target, full_matrices=False)
        r = self.adapter.rank
        A_w = Vh[:r, :]                     # (r, in)
        B_w = U[:, :r] * S[:r].unsqueeze(0)  # (out, r)
        scaling = self.adapter.scaling
        self.adapter.A.weight.copy_(A_w.to(self.adapter.A.weight.dtype))
        self.adapter.B.weight.copy_((B_w / scaling).to(self.adapter.B.weight.dtype))
        bias_target = eraser_mu.to(torch.float32) - (
            eraser_mu.to(torch.float32) @ eraser_P.to(torch.float32).T
        )
        self.adapter.bias.copy_(bias_target.to(self.adapter.bias.dtype))


class TaskHead(nn.Module):
    """Linear readout on the projection-head output."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ---------------------------------------------------------------------------
# Differentiable population OLS R² (gender, 2 classes) on a batch
# ---------------------------------------------------------------------------


def linear_r2(H: torch.Tensor, Z: torch.Tensor, *, reg: float = 1e-4) -> torch.Tensor:
    """Batch-population multi-output linear R² of one-hot Z from H.

    Identical to ``pcrl.training.losses.VerificationRegularizer.forward`` but
    inlined here so the dependency surface is small. Differentiable in H.
    Solve runs on CPU because ``torch.linalg.solve`` has known MPS bugs.
    """
    n_classes = int(Z.max().item()) + 1
    if n_classes < 2:
        return torch.tensor(0.0, device=H.device)
    Z_oh = F.one_hot(Z.long(), n_classes).float()
    H_c = H - H.mean(dim=0, keepdim=True)
    Z_c = Z_oh - Z_oh.mean(dim=0, keepdim=True)
    d = H.shape[1]
    gram = H_c.T @ H_c + reg * torch.eye(d, device=H.device)
    rhs = H_c.T @ Z_c
    W = torch.linalg.solve(gram.cpu(), rhs.cpu()).to(H.device)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)
    return r2.clamp(min=0.0)


# ---------------------------------------------------------------------------
# VICReg variance / covariance penalty on a batch
# ---------------------------------------------------------------------------


def vicreg_variance(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """λ_var · mean(ReLU(γ - std(z_j))) summed over dims, normalised by 1/d.

    The mean (rather than sum) keeps the magnitude stable across d so the
    VICReg term doesn't dominate as d grows.
    """
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    return F.relu(gamma - std).mean()


def vicreg_covariance(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal squared covariance, normalised by 1/d."""
    z_c = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0] - 1, 1)
    cov = (z_c.T @ z_c) / n  # (d, d)
    d = cov.shape[0]
    off = cov - torch.diag(torch.diagonal(cov))
    return (off ** 2).sum() / d


def per_dim_std(z: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)


def participation_ratio(z: torch.Tensor) -> float:
    """Effective rank ≈ (Σσ)² / Σσ²; integer in [1, d]. Audit only."""
    z_c = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0] - 1, 1)
    cov = (z_c.T @ z_c) / n
    cov = 0.5 * (cov + cov.T)
    eigs = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    sigma = torch.sqrt(eigs)
    return float((sigma.sum() ** 2 / (sigma ** 2).sum()).item())


# ---------------------------------------------------------------------------
# Linear gender probe (sklearn LR, balanced) on numpy reps
# ---------------------------------------------------------------------------


def linear_gender_probe_acc(
    train_z: np.ndarray, train_g: np.ndarray,
    dev_z: np.ndarray, dev_g: np.ndarray,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        class_weight="balanced", max_iter=2000, solver="lbfgs",
    )
    clf.fit(train_z, train_g)
    return {
        "train_acc": float(clf.score(train_z, train_g)),
        "dev_acc": float(clf.score(dev_z, dev_g)),
        "base_rate_train": float(np.bincount(train_g).max() / len(train_g)),
        "base_rate_dev": float(np.bincount(dev_g).max() / len(dev_g)),
    }


def linear_task_acc(
    train_z: np.ndarray, train_y: np.ndarray,
    dev_z: np.ndarray, dev_y: np.ndarray, *, num_classes: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", C=1.0,
    )
    clf.fit(train_z, train_y)
    return {
        "train_acc": float(clf.score(train_z, train_y)),
        "dev_acc": float(clf.score(dev_z, dev_y)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _purpose_label_keys(purpose: str) -> tuple[str, str, int]:
    if purpose == "P1":
        return "occupation_train", "occupation_dev", 28
    if purpose == "P2":
        return "supercategory_train", "supercategory_dev", 5
    if purpose == "P3":
        return "medical_train", "medical_dev", 2
    raise ValueError(f"Unknown purpose '{purpose}', expected P1/P2/P3")


def _compute_eval_reps(
    head: ProjectionHead, X: torch.Tensor, *, batch_size: int, device: torch.device,
) -> np.ndarray:
    head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            chunk = X[start:start + batch_size].to(device)
            out.append(head(chunk).detach().cpu().numpy())
    head.train()
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", choices=["P1", "P2", "P3"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Directory containing cls_layer12_embeddings.npz / labels.npz")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    # NOTE on batch size: the differentiable in-batch R²(gender) requires
    # ``batch_size`` >= ~2·d to avoid the rank-deficient pathology where the
    # in-batch ridge predictor saturates at R²=1 regardless of z, killing the
    # gradient signal (this is the same failure mode as BIOS Round 1/2). With
    # cached 768-d features, 4096 fits trivially on GPU (12MB/batch) and
    # gives a well-conditioned ridge solve. The deep-research spec listed 256
    # which is mathematically broken for d=768; we default to 4096 instead
    # and document the deviation in the run output.
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", type=str, default="auto")

    # LoRA / VICReg / Lagrangian (defaults match the spec).
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr-a", type=float, default=1e-4)
    ap.add_argument("--lr-b", type=float, default=3e-4)
    ap.add_argument("--lr-task-head", type=float, default=2e-4)
    ap.add_argument("--lr-adapter-bias", type=float, default=2e-4)
    ap.add_argument("--lambda-var", type=float, default=5.0)
    ap.add_argument("--lambda-cov", type=float, default=0.5)
    ap.add_argument("--vicreg-gamma", type=float, default=1.0)
    ap.add_argument("--lambda-max", type=float, default=10.0)
    ap.add_argument("--lambda-init", type=float, default=0.0)
    ap.add_argument("--lambda-warmup-frac", type=float, default=0.30)
    ap.add_argument("--lambda-clip-max", type=float, default=50.0)
    ap.add_argument("--dual-lr", type=float, default=5e-3)
    ap.add_argument("--r2-target", type=float, default=0.05)

    # Sanity / smoke knobs.
    ap.add_argument("--max-warm-start-r2", type=float, default=0.01,
                    help="Abort if step-0 R²(gender) > this after warm-start.")
    ap.add_argument("--smoke-epochs", type=int, default=-1,
                    help="If >0, override --epochs (used by smoke launcher).")
    ap.add_argument("--log-every", type=int, default=20)
    # Optional safety net: register a frozen LEACE post-projection on the
    # projection head, mirroring ``BertWithLoRA.set_leace_projection`` in
    # ``pcrl/language/bert_encoder.py``. The LoRA-side warm-start still runs
    # (so step-0 forward is unchanged in float-precision terms), but gradients
    # to the LoRA can only move the representation inside the post-projection's
    # null space. Useful if the proxy-Lagrangian alone fails to keep the
    # representation inside the constraint set during training (the LoRA-only
    # warm-start is a saddle point that CE gradients escape rapidly).
    ap.add_argument("--use-post-projection", action="store_true")
    # Online LEACE refit (BIOS Round 2 verified pattern, see
    # ``pcrl/language/online_leace.py`` and ``experiments/run_bios.py``).
    # The static post-projection fits LEACE on the *original* cached-features
    # distribution; under task-loss pressure the LoRA shifts that distribution
    # and the LEACE zero-cross-cov guarantee no longer holds, so gender info
    # re-enters (smoke #2 + #3 on layer-12 confirmed this empirically — R²
    # stayed at 0.95 even with --use-post-projection). Online refit observes
    # post-step (head(x), gender) into a sliding buffer and refits LEACE
    # every K steps, swapping the post-projection buffer in place. K=50 is
    # the BIOS Round 2 default. Buffer size 4096 (>= d=768 with margin) for
    # well-conditioned cov estimation under shrinkage.
    ap.add_argument("--online-leace", action="store_true",
                    help="Refit the LEACE post-projection every K primal steps "
                    "on a sliding buffer of post-step (head(x), gender). "
                    "Implies --use-post-projection (the buffer is the post-projection "
                    "target).")
    ap.add_argument("--online-leace-refit-every", type=int, default=50)
    ap.add_argument("--online-leace-buffer", type=int, default=4096)
    args = ap.parse_args()
    if args.online_leace:
        args.use_post_projection = True

    if args.smoke_epochs > 0:
        args.epochs = args.smoke_epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device.
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[pcrl_l12] purpose={args.purpose} seed={args.seed} device={device}",
          flush=True)

    # Reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load cached features and labels.
    emb = np.load(args.cache_dir / "cls_layer12_embeddings.npz")
    lbl = np.load(args.cache_dir / "labels.npz")
    X_train = torch.from_numpy(emb["train_standardized"]).float()
    X_dev = torch.from_numpy(emb["dev_standardized"]).float()
    g_train = torch.from_numpy(lbl["gender_train"]).long()
    g_dev = torch.from_numpy(lbl["gender_dev"]).long()
    y_key_train, y_key_dev, num_classes = _purpose_label_keys(args.purpose)
    y_train = torch.from_numpy(lbl[y_key_train]).long()
    y_dev = torch.from_numpy(lbl[y_key_dev]).long()
    d = X_train.shape[1]
    print(f"[pcrl_l12] N_train={X_train.shape[0]} N_dev={X_dev.shape[0]} d={d}"
          f"  num_classes={num_classes}", flush=True)
    if args.batch_size < 2 * d:
        print(
            f"[pcrl_l12] WARN: batch_size={args.batch_size} < 2·d={2*d}; the "
            f"differentiable in-batch R²(gender) will be near the rank-deficient "
            f"saturation ceiling and the proxy-Lagrangian gradient will be "
            f"degenerate. This is a smoke-only configuration — for the AWS "
            f"full run keep batch_size >= 4096 (default).",
            flush=True,
        )

    # ----- LEACE warm-start -----
    from concept_erasure import LeaceEraser

    G_oh_train = F.one_hot(g_train, 2).float()
    eraser = LeaceEraser.fit(X_train, G_oh_train)
    P_eraser = eraser.P.detach().to(torch.float32)  # (d, d)
    mu_eraser = (
        eraser.bias.detach().to(torch.float32)
        if eraser.bias is not None else torch.zeros(d, dtype=torch.float32)
    )
    X_train_erased_full = eraser(X_train)
    pre_r2 = float(linear_r2(X_train, g_train).item())
    post_r2_eraser = float(linear_r2(X_train_erased_full, g_train).item())
    print(f"[pcrl_l12] R²_gender pre={pre_r2:.4f}  post-eraser={post_r2_eraser:.4f}",
          flush=True)

    # ----- Build model and warm-start the LoRA -----
    head = ProjectionHead(
        dim=d, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0,
    ).to(device)
    head.warm_start_from_eraser(P_eraser.to(device), mu_eraser.to(device))
    if args.use_post_projection:
        head.set_leace_projection(P_eraser.to(device), mu_eraser.to(device))
        print("[pcrl_l12] post-projection ENABLED — LoRA learns inside null space.",
              flush=True)
    task_head = TaskHead(dim=d, num_classes=num_classes).to(device)

    # Verify warm-start: identity(x) + adapter(x) ≈ eraser(x) on the FULL train.
    # We use the full set because LEACE only guarantees R²=0 on its fit data —
    # on a sub-sample the empirical (mean, cov) differs and the constraint is
    # not exact. The numerical-equality check below (max|head - eraser|) is the
    # definitive correctness test; the R² check is the spec's stated criterion.
    head.eval()
    with torch.no_grad():
        out_chunks = []
        for start in range(0, X_train.shape[0], 2048):
            ch = X_train[start:start + 2048].to(device)
            out_chunks.append(head(ch).detach().cpu())
        z_warm = torch.cat(out_chunks, dim=0)
    head.train()
    warm_r2 = float(linear_r2(z_warm, g_train).item())
    warm_max_diff = float(
        (z_warm - X_train_erased_full).abs().max().item()
    )
    print(f"[pcrl_l12] step-0 R²_gender(full train)={warm_r2:.4f}  "
          f"max|head(x) - eraser(x)|={warm_max_diff:.2e}", flush=True)
    if warm_r2 > args.max_warm_start_r2:
        msg = (
            f"FATAL: warm-start R²(gender)={warm_r2:.4f} > "
            f"{args.max_warm_start_r2:.4f}; the LoRA is not realising the LEACE "
            f"projection. Check rank/alpha and the SVD init.\n"
            f"NOTE: max|head(x) - eraser(x)|={warm_max_diff:.2e} — if that is "
            f"<1e-4 the head IS the eraser numerically and any R² gap is a "
            f"sampling artefact (try increasing the LEACE fit subset)."
        )
        print(msg, flush=True)
        return 2

    # ----- Online LEACE refit (optional) -----
    online_leace = None
    if args.online_leace:
        from pcrl.language.online_leace import OnlineLeaceRefit
        online_leace = OnlineLeaceRefit(
            d_x=d, d_z=1,
            buffer_size=args.online_leace_buffer,
            refit_every=args.online_leace_refit_every,
            device=device,
            shrinkage=True, constrain_cov_trace=True,
        )
        print(
            f"[pcrl_l12] online LEACE refit ENABLED  "
            f"buffer={args.online_leace_buffer}  "
            f"refit_every={args.online_leace_refit_every}  "
            f"shrinkage=True  constrain_cov_trace=True", flush=True,
        )

    # ----- Optimiser -----
    pg_a = {"params": [head.adapter.A.weight], "lr": args.lr_a, "weight_decay": 0.0}
    pg_b = {"params": [head.adapter.B.weight], "lr": args.lr_b, "weight_decay": 0.0}
    pg_bias = {"params": [head.adapter.bias],  "lr": args.lr_adapter_bias,
               "weight_decay": 0.0}
    pg_task = {"params": list(task_head.parameters()), "lr": args.lr_task_head,
               "weight_decay": 1e-4}
    opt = torch.optim.AdamW(
        [pg_a, pg_b, pg_bias, pg_task],
        betas=(0.9, 0.999),
    )

    # Proxy-Lagrangian dual variable for R²(gender).
    constraint = Constraint(
        name="r2_gender",
        threshold=args.r2_target,
        direction="<=",
        eta_lambda=args.dual_lr,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_clip_max,
        lambda_min=0.0,
    )

    # ----- Data loader -----
    train_ds = TensorDataset(X_train, g_train, y_train)
    g_loader = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=g_loader, num_workers=0,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.lambda_warmup_frac * total_steps))
    print(f"[pcrl_l12] steps_per_epoch={steps_per_epoch}  total={total_steps}"
          f"  warmup={warmup_steps}", flush=True)

    # ----- Training -----
    history: list[dict] = []
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        head.train(); task_head.train()
        for x_b, g_b, y_b in loader:
            x_b = x_b.to(device); g_b = g_b.to(device); y_b = y_b.to(device)
            z = head(x_b)
            logits = task_head(z)
            l_task = F.cross_entropy(logits, y_b)
            r2_b = linear_r2(z, g_b)  # differentiable in z
            l_var = vicreg_variance(z, gamma=args.vicreg_gamma)
            l_cov = vicreg_covariance(z)

            # Lambda warmup ramp on top of the dual ascent value.
            warm_factor = min(1.0, (step + 1) / warmup_steps)
            lam_eff = min(constraint.lambda_value, args.lambda_max) * warm_factor

            l_total = (
                l_task
                + lam_eff * (r2_b - args.r2_target)
                + args.lambda_var * l_var
                + args.lambda_cov * l_cov
            )

            opt.zero_grad(set_to_none=True)
            l_total.backward()
            opt.step()

            # Online LEACE refit. Observe the PRE-projection head output (host +
            # adapter, no leace_P applied) so the new LEACE is fit on the
            # distribution it will be applied to after ``set_leace_projection``
            # replaces the buffer. Observing the post-projection output (smoke #4
            # config, mirroring experiments/run_bios.py:648) caused the refit to
            # produce a P_new for a different distribution than the next
            # forward operates on, leaving R²(g) flat at 0.95 — see
            # HEADLINE_ABORT_2 in S3 and the smoke #5 fix authorization.
            refit_this_step = False
            if online_leace is not None:
                with torch.no_grad():
                    z_pre_proj = (head.host(x_b) + head.adapter(x_b)).detach()
                online_leace.observe(z_pre_proj, g_b)
                if online_leace.should_refit(step):
                    eraser = online_leace.refit(step)
                    Q = eraser.P.detach().to(torch.float32)
                    mu_eraser = (
                        eraser.bias.detach().to(torch.float32)
                        if eraser.bias is not None
                        else torch.zeros(Q.shape[0], dtype=torch.float32)
                    )
                    head.set_leace_projection(Q.to(device), mu_eraser.to(device))
                    refit_this_step = True

            # Dual ascent on the actual constraint value; no warmup multiplier
            # on the dual update itself (warmup is applied to the primal
            # weighting only).
            constraint.update_lambda(float(r2_b.detach().item()))
            # Cap λ at λ_max while still allowing it to wander above (clipped to
            # lambda_clip_max=50.0 by the Constraint class).
            if constraint.lambda_value > args.lambda_clip_max:
                constraint.lambda_value = args.lambda_clip_max

            if step % args.log_every == 0:
                with torch.no_grad():
                    pds = per_dim_std(z).detach()
                history.append({
                    "step": step,
                    "epoch": epoch,
                    "r2_gender_batch": float(r2_b.detach().item()),
                    "task_loss": float(l_task.detach().item()),
                    "vicreg_var": float(l_var.detach().item()),
                    "vicreg_cov": float(l_cov.detach().item()),
                    "lambda_dual": float(constraint.lambda_value),
                    "lambda_eff_primal": float(lam_eff),
                    "warm_factor": float(warm_factor),
                    "per_dim_std_median": float(pds.median().item()),
                    "per_dim_std_min": float(pds.min().item()),
                    "leace_refits": (
                        int(online_leace.refit_count) if online_leace is not None
                        else 0
                    ),
                    "leace_refit_this_step": bool(refit_this_step) if online_leace is not None else False,
                })
            step += 1

        # End-of-epoch eval (dev R²; cheaper than full epoch dev fwd is fine
        # since we already held all features in memory).
        head.eval(); task_head.eval()
        with torch.no_grad():
            # Sample 4096 dev examples for cheap R² estimate.
            n_dev = X_dev.shape[0]
            idx = torch.randperm(n_dev, generator=torch.Generator().manual_seed(epoch))[:4096]
            z_d = head(X_dev[idx].to(device)).detach()
            r2_dev = float(linear_r2(z_d.cpu(), g_dev[idx]).item())
        elapsed = time.time() - t0
        refits_str = (
            f"refits={online_leace.refit_count}  "
            if online_leace is not None else ""
        )
        print(
            f"[pcrl_l12] epoch {epoch+1}/{args.epochs}  step={step}  "
            f"task={history[-1]['task_loss']:.3f}  "
            f"R²(g, batch)={history[-1]['r2_gender_batch']:.4f}  "
            f"R²(g, dev4k)={r2_dev:.4f}  λ={constraint.lambda_value:.2f}  "
            f"per_dim_std_med={history[-1]['per_dim_std_median']:.3f}  "
            f"{refits_str}elapsed={elapsed:.1f}s",
            flush=True,
        )

    # ----- Final eval reps -----
    head.eval()
    train_reps = _compute_eval_reps(head, X_train, batch_size=2048, device=device)
    dev_reps = _compute_eval_reps(head, X_dev, batch_size=2048, device=device)

    # Compliance on full splits (numpy R² via the same closed form).
    def _np_linear_r2(H: np.ndarray, Z: np.ndarray, *, reg: float = 1e-4) -> float:
        n_classes = int(Z.max()) + 1
        if n_classes < 2:
            return 0.0
        Z_oh = np.eye(n_classes)[Z].astype(np.float64)
        H = H.astype(np.float64)
        H_c = H - H.mean(axis=0, keepdims=True)
        Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
        d_ = H.shape[1]
        gram = H_c.T @ H_c + reg * np.eye(d_)
        W = np.linalg.solve(gram, H_c.T @ Z_c)
        Z_pred = H_c @ W
        ss_res = ((Z_c - Z_pred) ** 2).sum()
        ss_tot = (Z_c ** 2).sum()
        return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))

    r2_train = _np_linear_r2(train_reps, g_train.numpy())
    r2_dev = _np_linear_r2(dev_reps, g_dev.numpy())

    # Variance / eff-rank diagnostics on dev reps.
    dev_t = torch.from_numpy(dev_reps)
    pds_dev = per_dim_std(dev_t).numpy()
    eff_rank_dev = participation_ratio(dev_t)

    # Linear utility (occupation/super/medical) and gender probe acc.
    print("[pcrl_l12] training auxiliary linear classifiers (sklearn) ...",
          flush=True)
    task_lin = linear_task_acc(
        train_reps, y_train.numpy(), dev_reps, y_dev.numpy(),
        num_classes=num_classes,
    )
    gender_probe = linear_gender_probe_acc(
        train_reps, g_train.numpy(), dev_reps, g_dev.numpy(),
    )

    # ----- Save -----
    enc_path = args.output_dir / f"encoder_{args.purpose}_{args.seed}.pt"
    torch.save(
        {
            "head_state": head.state_dict(),
            "task_head_state": task_head.state_dict(),
            "args": vars(args),
            "leace": {
                "P_eraser": P_eraser.cpu().numpy(),
                "mu_eraser": mu_eraser.cpu().numpy(),
            },
        },
        enc_path,
    )

    reps_path = args.output_dir / f"eval_reps_{args.purpose}_{args.seed}.npz"
    np.savez_compressed(
        reps_path,
        train_reps=train_reps, dev_reps=dev_reps,
        gender_train=g_train.numpy(), gender_dev=g_dev.numpy(),
        task_train=y_train.numpy(), task_dev=y_dev.numpy(),
    )

    metrics = {
        "purpose": args.purpose,
        "seed": int(args.seed),
        "num_classes": num_classes,
        "n_train": int(X_train.shape[0]),
        "n_dev": int(X_dev.shape[0]),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "warm_start": {
            "pre_r2_train": pre_r2,
            "post_r2_eraser_train": post_r2_eraser,
            "step0_r2_train4k": warm_r2,
            "step0_max_diff_vs_eraser": warm_max_diff,
        },
        "compliance": {
            "r2_gender_train": r2_train,
            "r2_gender_dev": r2_dev,
            "target": args.r2_target,
        },
        "utility_linear_readout": task_lin,
        "linear_gender_probe": gender_probe,
        "repr_health": {
            "per_dim_std_median": float(np.median(pds_dev)),
            "per_dim_std_min": float(pds_dev.min()),
            "per_dim_std_p10": float(np.percentile(pds_dev, 10)),
            "eff_rank_dev_pr": eff_rank_dev,
        },
        "history": history,
        "lambda_history": {
            "values": constraint.value_history,
            "lambdas": constraint.lambda_history,
            "violations": constraint.violation_history,
        },
        "online_leace": (
            {
                "enabled": True,
                "buffer_size": int(args.online_leace_buffer),
                "refit_every": int(args.online_leace_refit_every),
                "total_refits": int(online_leace.refit_count),
            }
            if online_leace is not None
            else {"enabled": False}
        ),
        "elapsed_s": time.time() - t0,
    }
    metrics_path = args.output_dir / f"metrics_{args.purpose}_{args.seed}.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[pcrl_l12] DONE  R²_g(dev)={r2_dev:.4f}  "
          f"task_dev_acc={task_lin['dev_acc']:.4f}  "
          f"gender_probe_dev={gender_probe['dev_acc']:.4f}  "
          f"per_dim_std_med={float(np.median(pds_dev)):.3f}  "
          f"eff_rank={eff_rank_dev:.1f}", flush=True)
    print(f"[pcrl_l12] wrote {enc_path}", flush=True)
    print(f"[pcrl_l12] wrote {reps_path}", flush=True)
    print(f"[pcrl_l12] wrote {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

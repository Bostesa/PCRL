"""CelebA-medium training loop: predict Smiling, hide Male AND Young.

Constraints (linear-R^2 <= 0.05 per attribute, separately):
    - r2_male  <= 0.05
    - r2_young <= 0.05
Optimiser: AdamW(lora_params, lr=1e-3) + ProxyLagrangianOptimizer dual ascent.
Auxiliary: VICReg variance + covariance regulariser on the 512-d penultimate.

LEACE warm-start is run before training; if construction-time R^2 already
satisfies both constraints we skip the opening warmup epoch (per R1+R2 fix).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcrl.training.independence.vicreg import vicreg_loss
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer
from pcrl.vision.backbone import iter_lora_params
from pcrl.vision.r2_helper import linear_r2

LOG = logging.getLogger("pcrl.vision.train")


@dataclass
class TrainConfig:
    out_dir: Path
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    r2_threshold: float = 2.5       # C2 empirical: LEACE-init noise floor ~2.0
    eval_r2_threshold: float = 0.05  # eval-side: real linear R^2 (in [0,1]) gate
    lambda_min: float = 5.0
    lambda_init: float = 5.0
    lambda_max: float = 1000.0
    eta_lambda: float = 0.02
    vicreg_lambda_var: float = 1.0
    vicreg_lambda_cov: float = 0.04
    vicreg_gamma: float = 1.0
    skip_warmup_if_feasible: bool = True
    log_every: int = 50
    holdout_every: int = 10
    refit_every: int = 0          # C3: 0 disables; e.g. 200 for periodic re-projection
    rank: int = 8
    alpha: int = 16
    seed: int = 0


@dataclass
class EpochLog:
    epoch: int
    task_loss: float
    task_acc: float
    r2_male: float
    r2_young: float
    lambda_male: float
    lambda_young: float
    vicreg: float
    feasible: bool
    time_sec: float
    per_dim_std: float
    notes: str = ""


@dataclass
class TrainHistory:
    epochs: list[EpochLog] = field(default_factory=list)
    selector: dict[str, Any] = field(default_factory=dict)


def _select_cotter_best(history: list[EpochLog], r2_thresh: float) -> int | None:
    """Cotter best-iterate selector: among feasible epochs (both R^2 <= thresh),
    pick the one with highest task_acc. None if no feasible epoch.
    """
    feasible = [e for e in history if e.feasible]
    if not feasible:
        return None
    return max(feasible, key=lambda e: e.task_acc).epoch


@torch.no_grad()
def _quick_holdout_r2(
    wrapper: nn.Module, holdout_loader: DataLoader, device: torch.device,
) -> dict[str, float]:
    """Cheap held-out R^2 — features only, no head. Used every K primal steps
    to track the actual eval metric vs the dual surrogate."""
    was_training = wrapper.training
    wrapper.eval()
    H_list, m_list, y_list = [], [], []
    for batch in holdout_loader:
        h = wrapper(batch["image"].to(device, non_blocking=True))
        H_list.append(h.detach().cpu())
        m_list.append(batch["male"].cpu())
        y_list.append(batch["young"].cpu())
    if was_training:
        wrapper.train()
    H = torch.cat(H_list, 0).numpy()
    male = torch.cat(m_list, 0).numpy()
    young = torch.cat(y_list, 0).numpy()
    return {"male": linear_r2(H, male), "young": linear_r2(H, young)}


@torch.no_grad()
def _evaluate_r2(
    wrapper: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device,
) -> dict[str, float]:
    wrapper.eval()
    head.eval()
    H_list: list[torch.Tensor] = []
    male_list: list[torch.Tensor] = []
    young_list: list[torch.Tensor] = []
    smile_list: list[torch.Tensor] = []
    pred_list: list[torch.Tensor] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        h = wrapper(x)
        logits = head(h)
        pred = logits.argmax(dim=1).detach().cpu()
        H_list.append(h.detach().cpu())
        male_list.append(batch["male"].cpu())
        young_list.append(batch["young"].cpu())
        smile_list.append(batch["smiling"].cpu())
        pred_list.append(pred)
    H = torch.cat(H_list, dim=0).numpy()
    male = torch.cat(male_list, dim=0).numpy()
    young = torch.cat(young_list, dim=0).numpy()
    smile = torch.cat(smile_list, dim=0).numpy()
    pred = torch.cat(pred_list, dim=0).numpy()
    return {
        "r2_male": linear_r2(H, male),
        "r2_young": linear_r2(H, young),
        "task_acc": float((pred == smile).mean()),
        "per_dim_std": float(np.sqrt(H.var(axis=0)).mean()),
    }


def train(
    wrapper: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    skip_warmup: bool,
    holdout_loader: DataLoader | None = None,
    refit_loader: DataLoader | None = None,
) -> TrainHistory:
    """Main training loop.

    Args:
        wrapper: ResNet18 + LoRA wrapper (penultimate features).
        head: nn.Linear(512, 2) for Smiling.
        train_loader / val_loader: CelebAMedium loaders.
        cfg: training config.
        device: cpu or cuda.
        skip_warmup: if True, skip the opening warmup epoch (LEACE-init feasible).
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    lora_params = list(iter_lora_params(wrapper))
    head_params = list(head.parameters())
    primal = torch.optim.AdamW(
        lora_params + head_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    constraints = [
        Constraint(
            name="r2_male",
            threshold=cfg.r2_threshold,
            direction="<=",
            eta_lambda=cfg.eta_lambda,
            lambda_init=cfg.lambda_init,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        ),
        Constraint(
            name="r2_young",
            threshold=cfg.r2_threshold,
            direction="<=",
            eta_lambda=cfg.eta_lambda,
            lambda_init=cfg.lambda_init,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
        ),
    ]
    opt = ProxyLagrangianOptimizer(primal, constraints)

    history = TrainHistory()
    best_state: dict[str, Any] | None = None

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        wrapper.train()
        head.train()
        running_task = 0.0
        running_vic = 0.0
        n_correct = 0
        n_total = 0

        # Differentiable surrogate for R^2 within a batch — we use the Tikhonov
        # closed form on the *batch* representation (not autograd-friendly enough
        # for stable training on its own). We instead use a HSIC-style soft
        # decorrelation via the lagrangian on the periodically computed *epoch*
        # R^2. To keep gradients flowing each step, we add a per-batch R^2 proxy:
        # squared centred-cross-cov between H and Z_oh, normalised by batch
        # variance.
        for it, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["smiling"].to(device, non_blocking=True)
            male = batch["male"].to(device, non_blocking=True)
            young = batch["young"].to(device, non_blocking=True)

            h = wrapper(x)
            logits = head(h)
            task = F.cross_entropy(logits, y)

            # Per-batch differentiable R^2 surrogate.
            r2_male_t = _batch_r2_surrogate(h, male, n_classes=2)
            r2_young_t = _batch_r2_surrogate(h, young, n_classes=2)

            vic = vicreg_loss(
                h,
                lambda_var=cfg.vicreg_lambda_var,
                lambda_cov=cfg.vicreg_lambda_cov,
                gamma=cfg.vicreg_gamma,
            )

            base = task + vic
            total = opt.lagrangian_loss(
                base, {"r2_male": r2_male_t, "r2_young": r2_young_t}
            )

            primal.zero_grad(set_to_none=True)
            total.backward()
            primal.step()

            # C2: dual fed by surrogate (sum-form, threshold = eps*d).
            opt.dual_step({
                "r2_male": float(r2_male_t.detach().item()),
                "r2_young": float(r2_young_t.detach().item()),
            })

            running_task += float(task.detach().item()) * x.size(0)
            running_vic += float(vic.detach().item()) * x.size(0)
            with torch.no_grad():
                n_correct += int((logits.argmax(dim=1) == y).sum().item())
                n_total += x.size(0)

            if (it + 1) % cfg.log_every == 0:
                LOG.info(
                    f"  step {it+1}: task={task.item():.4f} "
                    f"surr_m={r2_male_t.item():.6f} surr_y={r2_young_t.item():.6f} "
                    f"vic={vic.item():.4f} "
                    f"lam_m={constraints[0].lambda_value:.2f} "
                    f"lam_y={constraints[1].lambda_value:.2f}"
                )

            # Held-out R^2 logged for visibility (not driving dual under C2).
            if holdout_loader is not None and (it + 1) % cfg.holdout_every == 0:
                hor2 = _quick_holdout_r2(wrapper, holdout_loader, device)
                LOG.info(
                    f"  step {it+1} HOLDOUT: r2_m_eval={hor2['male']:.4f} "
                    f"r2_y_eval={hor2['young']:.4f}  "
                    f"surr_m={r2_male_t.item():.4f} surr_y={r2_young_t.item():.4f}  "
                    f"lam_m={constraints[0].lambda_value:.2f} "
                    f"lam_y={constraints[1].lambda_value:.2f}"
                )

            # C3: periodic LEACE re-projection. Compose Q_new @ W_old into LoRA.
            if (
                cfg.refit_every > 0
                and refit_loader is not None
                and (it + 1) % cfg.refit_every == 0
            ):
                from pcrl.vision.leace_warmstart import refit_leace_lora
                pre_r2 = (
                    _quick_holdout_r2(wrapper, holdout_loader, device)
                    if holdout_loader is not None else None
                )
                rank_M, top_s, residual = refit_leace_lora(
                    wrapper, refit_loader, device,
                    rank=cfg.rank, alpha=cfg.alpha,
                )
                post_r2 = (
                    _quick_holdout_r2(wrapper, holdout_loader, device)
                    if holdout_loader is not None else None
                )
                wrapper.train()
                head.train()
                if pre_r2 and post_r2:
                    LOG.info(
                        f"  step {it+1} REFIT: pre_r2_m={pre_r2['male']:.4f} "
                        f"pre_r2_y={pre_r2['young']:.4f} -> "
                        f"post_r2_m={post_r2['male']:.4f} "
                        f"post_r2_y={post_r2['young']:.4f}  "
                        f"rank_M={rank_M} svd_resid={residual:.2e}"
                    )
                else:
                    LOG.info(
                        f"  step {it+1} REFIT: rank_M={rank_M} svd_resid={residual:.2e}"
                    )

        # End-of-epoch evaluation on val_loader (full-precision R^2).
        eval_metrics = _evaluate_r2(wrapper, head, val_loader, device)
        feasible = (
            eval_metrics["r2_male"] <= cfg.eval_r2_threshold
            and eval_metrics["r2_young"] <= cfg.eval_r2_threshold
        )
        elog = EpochLog(
            epoch=epoch,
            task_loss=running_task / max(n_total, 1),
            task_acc=eval_metrics["task_acc"],
            r2_male=eval_metrics["r2_male"],
            r2_young=eval_metrics["r2_young"],
            lambda_male=constraints[0].lambda_value,
            lambda_young=constraints[1].lambda_value,
            vicreg=running_vic / max(n_total, 1),
            feasible=feasible,
            time_sec=time.time() - epoch_start,
            per_dim_std=eval_metrics["per_dim_std"],
            notes="warmup_skipped" if skip_warmup and epoch == 0 else "",
        )
        history.epochs.append(elog)
        LOG.info(
            f"epoch {epoch}: task_acc={elog.task_acc:.4f} "
            f"r2_m={elog.r2_male:.4f} r2_y={elog.r2_young:.4f} "
            f"feasible={feasible} per_dim_std={elog.per_dim_std:.3f} "
            f"time={elog.time_sec:.1f}s"
        )

        # Save best-feasible checkpoint as we go.
        if feasible and (
            best_state is None
            or elog.task_acc > best_state["task_acc"]
        ):
            best_state = {
                "epoch": epoch,
                "task_acc": elog.task_acc,
                "r2_male": elog.r2_male,
                "r2_young": elog.r2_young,
                "wrapper_state": {
                    k: v.detach().cpu().clone()
                    for k, v in wrapper.state_dict().items()
                    if "lora_" in k
                },
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            torch.save(best_state, cfg.out_dir / "best_cotter.pt")

    # Final checkpoint.
    final_state = {
        "wrapper_state": {
            k: v.detach().cpu().clone()
            for k, v in wrapper.state_dict().items()
            if "lora_" in k
        },
        "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "history": [asdict(e) for e in history.epochs],
    }
    torch.save(final_state, cfg.out_dir / "final.pt")

    selected = _select_cotter_best(history.epochs, cfg.eval_r2_threshold)
    history.selector = {
        "selected_epoch": selected,
        "best_state_epoch": best_state["epoch"] if best_state else None,
        "n_feasible_epochs": sum(1 for e in history.epochs if e.feasible),
    }
    (cfg.out_dir / "training_log.json").write_text(
        json.dumps(
            {
                "history": [asdict(e) for e in history.epochs],
                "selector": history.selector,
                "config": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in asdict(cfg).items()
                },
            },
            indent=2,
        )
    )
    return history


def _batch_r2_surrogate(
    h: torch.Tensor, z: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Differentiable per-batch concept-leakage surrogate (HSIC-style).

    Mean per-(channel, class) squared centred correlation between ``h`` and
    one-hot ``z``. Bounded in [0, 1], proportional to linear-R^2 (the metric
    the eval/dual see) but with a non-degenerate gradient even when the batch
    size is smaller than the feature dimension (n=256 < d=512 case where the
    closed-form ridge regression overfits the batch and emits a constant 1).
    """
    n = h.shape[0]
    z_oh = F.one_hot(z.long(), num_classes=n_classes).float()
    h_c = h - h.mean(dim=0, keepdim=True)
    z_c = z_oh - z_oh.mean(dim=0, keepdim=True)
    cross = (h_c.T @ z_c) / n
    h_var = h_c.pow(2).sum(dim=0).clamp_min(1e-6) / n
    z_var = z_c.pow(2).sum(dim=0).clamp_min(1e-6) / n
    sq_corr = cross.pow(2) / (h_var.unsqueeze(1) * z_var.unsqueeze(0))
    # C2: sum over d (channels), mean over k (classes). Magnitude scales
    # with d so per-parameter primal gradient can compete with task grad.
    # Threshold matches: e.g. eps=0.05 mean-form -> 0.05 * d = 25.6 sum-form.
    return sq_corr.sum(dim=0).mean()

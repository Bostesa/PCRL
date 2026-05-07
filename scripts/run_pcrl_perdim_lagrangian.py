"""Per-dimension Lagrangian PCRL retraining driver.

Replaces the mean-of-clamped-slack scalar variance dual from
run_pcrl_variance_constrained.py with K=64 INDEPENDENT Lagrangian dual
variables per purpose, one per representation dimension. Each per-dim
dual fires only when its own sigma_d < gamma=0.5, eliminating the
"global relaxation" failure mode where one dim crossing the floor
relaxed pressure on all others.

Config matches the variance retraining run except:
  - 64 var duals per purpose instead of 1 (3 purposes * 64 = 192 var duals)
  - Default epochs reduced 150 -> 75 for the time-budgeted experiment
  - lambda_max lowered 100 -> 100 (kept; per-dim duals get less reinforcement
    each since they're scoped to one dim's slack, so saturation is naturally
    rarer; the launcher script can override via --var-lambda-max)

Usage:
    python scripts/run_pcrl_perdim_lagrangian.py \\
        --dataset hmda --seed 0 --epochs 75
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import generate_report
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.independence.vclub import VCLUB
from pcrl.training.proxy_lagrangian import Constraint
from pcrl.training.v2_trainer import V2Trainer, V2TrainerConfig

sys.path.insert(0, str(ROOT / "experiments"))
from run_v2_dataset import (  # noqa: E402
    LORA_BY_DATASET,
    NUM_WORKERS_BY_DATASET,
    build_datasets,
    repr_health,
    reps_for_purpose,
)


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
)
log = logging.getLogger("pcrl_perdim_lagr")

REPR_DIM = 64
VAR_GAMMA_DEFAULT = 0.5
EFF_RANK_THRESHOLD_DEFAULT = 2.0
EFF_RANK_PENALTY_WEIGHT_DEFAULT = 1.0
EFF_RANK_SIGMOID_ALPHA = 2.0
VAR_LAMBDA_INIT_DEFAULT = 1.0
VAR_LAMBDA_MAX_DEFAULT = 100.0
VAR_LAMBDA_MIN_DEFAULT = 0.0  # per-dim duals can fully relax when their dim is healthy
VAR_LR_LAMBDA_DEFAULT = 0.02


def _per_dim_constraint_name(purpose_name: str, k: int) -> str:
    return f"var_dim__{purpose_name}__{k:02d}"


def participation_ratio(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    z_c = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0] - 1, 1)
    cov = (z_c.T @ z_c) / n
    cov = 0.5 * (cov + cov.T)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=eps)
    sigma = torch.sqrt(eigvals)
    return (sigma.sum() ** 2) / (sigma.pow(2).sum() + eps)


def per_dim_std_vector(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Returns sigma_d for each dim d. Differentiable."""
    return torch.sqrt(z.var(dim=0) + eps)


@dataclass
class PerDimLagrangianConfig(V2TrainerConfig):
    var_gamma: float = VAR_GAMMA_DEFAULT
    var_lambda_init: float = VAR_LAMBDA_INIT_DEFAULT
    var_lambda_max: float = VAR_LAMBDA_MAX_DEFAULT
    var_lambda_min: float = VAR_LAMBDA_MIN_DEFAULT
    var_lr_lambda: float = VAR_LR_LAMBDA_DEFAULT
    eff_rank_threshold: float = EFF_RANK_THRESHOLD_DEFAULT
    eff_rank_penalty_weight: float = EFF_RANK_PENALTY_WEIGHT_DEFAULT
    eff_rank_sigmoid_alpha: float = EFF_RANK_SIGMOID_ALPHA
    repr_dim: int = REPR_DIM


class PerDimLagrangianTrainer(V2Trainer):
    """V2Trainer + per-dimension Lagrangian variance constraint.

    For each purpose × each of the K=repr_dim representation dimensions,
    a separate Constraint with `slack_d = relu(gamma - sigma_d)` and
    direction `<=` against threshold 0.0. The Lagrangian contribution
    becomes `Σ_p Σ_d λ_{p,d} · slack_{p,d}` instead of the original's
    `Σ_p λ_p · mean_d(slack_{p,d})`.
    """

    config: PerDimLagrangianConfig  # for type checkers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        added = 0
        for purpose_name in self.purpose_names:
            for k in range(cfg.repr_dim):
                c = Constraint(
                    name=_per_dim_constraint_name(purpose_name, k),
                    threshold=0.0,
                    direction="<=",
                    eta_lambda=cfg.var_lr_lambda,
                    lambda_init=cfg.var_lambda_init,
                    lambda_max=cfg.var_lambda_max,
                    lambda_min=cfg.var_lambda_min,
                )
                self.proxy.constraints[c.name] = c
                added += 1
        log.info(
            "[PERDIM/INIT] added %d per-dim variance Lagrangian duals (%d purposes "
            "× %d dims), gamma=%.2f, lam_init=%.2f, lam_max=%.2f, lr_lambda=%.4f",
            added, len(self.purpose_names), cfg.repr_dim, cfg.var_gamma,
            cfg.var_lambda_init, cfg.var_lambda_max, cfg.var_lr_lambda,
        )

    def _primal_and_dual_step(self, batch, apply_constraints: bool = True):
        from pcrl.training.losses import task_loss
        from pcrl.training.independence.hsic import hsic
        from pcrl.training.independence.vicreg import vicreg_loss
        from pcrl.training.v2_trainer import _pair_key

        self.encoder.train()
        self._freeze_backbone_bn()
        self.task_heads.train()

        reprs: dict[str, torch.Tensor] = {}
        for purpose_name in self.purpose_names:
            reprs[purpose_name] = self.encoder(
                batch["features"], self._purpose_idx(purpose_name),
            )

        L_task = torch.tensor(0.0, device=self.device)
        for purpose_name, z in reprs.items():
            task_name = self._primary_task(purpose_name)
            if task_name not in batch["task_labels"]:
                continue
            head = self.task_heads[purpose_name]
            preds = head(z)
            if isinstance(preds, dict):
                preds = preds[task_name]
            targets = batch["task_labels"][task_name]
            L_task = L_task + task_loss(preds, targets, self._task_type(purpose_name))

        L_vicreg = torch.tensor(0.0, device=self.device)
        for z in reprs.values():
            L_vicreg = L_vicreg + vicreg_loss(
                z,
                lambda_var=self.config.vicreg_lambda_var,
                lambda_cov=self.config.vicreg_lambda_cov,
                gamma=self.config.vicreg_gamma,
            )

        L_vclub = torch.tensor(0.0, device=self.device)
        L_hsic_aux = torch.tensor(0.0, device=self.device)
        L_verify = torch.tensor(0.0, device=self.device)
        constraint_values: dict[str, torch.Tensor] = {}
        constraint_scalars: dict[str, float] = {}
        hsic_scalars: dict[str, float] = {}
        pair_r2_log: dict[str, float] = {}
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pair = (purpose_name, attr_name)
            pair_key_str = _pair_key(purpose_name, attr_name)
            L_vclub = L_vclub + self.vclubs[pair_key_str].mi_upper_bound(z, attr)
            hsic_val = hsic(z, attr)
            L_hsic_aux = L_hsic_aux + hsic_val
            hsic_scalars[pair_key_str] = float(hsic_val.detach().item())

            if int(attr.max().item()) < 1:
                zero = torch.tensor(0.0, device=self.device)
                for cname in self.pair_constraint_keys[pair]:
                    constraint_values[cname] = zero
                    constraint_scalars[cname] = 0.0
                pair_r2_log[pair_key_str] = 0.0
                continue

            if pair in self.high_k_pairs:
                r2_per_k = self.verifier.forward_per_class(z, attr)
                names = self.pair_constraint_keys[pair]
                K_total = len(names)
                if r2_per_k.numel() < K_total:
                    pad = torch.zeros(K_total - r2_per_k.numel(), device=self.device)
                    r2_per_k = torch.cat([r2_per_k, pad])
                pair_max = float("-inf")
                for k, cname in enumerate(names):
                    rk = r2_per_k[k]
                    constraint_values[cname] = rk
                    val = float(rk.detach().item())
                    constraint_scalars[cname] = val
                    pair_max = max(pair_max, val)
                L_verify = L_verify + r2_per_k.mean()
                pair_r2_log[pair_key_str] = pair_max
            else:
                r2_val = self.verifier(z, attr)
                cname = self.pair_constraint_keys[pair][0]
                constraint_values[cname] = r2_val
                constraint_scalars[cname] = float(r2_val.detach().item())
                L_verify = L_verify + r2_val
                pair_r2_log[pair_key_str] = constraint_scalars[cname]

        # ── Per-dim variance Lagrangian: K=repr_dim entries per purpose ──────
        var_slack_per_purpose: dict[str, dict[int, float]] = {}
        var_sigma_per_purpose: dict[str, list[float]] = {}
        for purpose_name, z in reprs.items():
            sigma = per_dim_std_vector(z)  # (D,)
            slacks = torch.relu(self.config.var_gamma - sigma)  # (D,)
            var_sigma_per_purpose[purpose_name] = sigma.detach().tolist()
            d_slacks: dict[int, float] = {}
            for k in range(self.config.repr_dim):
                cname = _per_dim_constraint_name(purpose_name, k)
                slack_k = slacks[k]
                constraint_values[cname] = slack_k
                slack_scalar = float(slack_k.detach().item())
                constraint_scalars[cname] = slack_scalar
                d_slacks[k] = slack_scalar
            var_slack_per_purpose[purpose_name] = d_slacks

        # ── Eff-rank soft penalty (kept identical to variance retraining) ──
        L_eff_rank = torch.tensor(0.0, device=self.device)
        eff_rank_log: dict[str, float] = {}
        for purpose_name, z in reprs.items():
            er = participation_ratio(z)
            margin = self.config.eff_rank_threshold - er
            L_eff_rank = L_eff_rank + torch.sigmoid(
                self.config.eff_rank_sigmoid_alpha * margin
            )
            eff_rank_log[purpose_name] = float(er.detach().item())

        if apply_constraints:
            base = (
                L_task
                + self.config.lambda_vicreg * L_vicreg
                + self.config.lambda_vclub * L_vclub
                + self.config.lambda_hsic_aux * L_hsic_aux
                + self.config.lambda_verify * L_verify
                + self.config.eff_rank_penalty_weight * L_eff_rank
            )
            primal_loss = self.proxy.lagrangian_loss(base, constraint_values)
        else:
            primal_loss = L_task + self.config.lambda_vicreg * L_vicreg

        self.primal_optimizer.zero_grad()
        primal_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.primal_optimizer.param_groups[0]["params"]],
                max_norm=self.config.grad_clip,
            )
        self.primal_optimizer.step()

        if apply_constraints:
            self.proxy.dual_step(constraint_scalars)

        # Aggregate per-purpose stats for compact logging (don't dump 192 numbers
        # per epoch). Saturation = lambda >= 0.95 * lambda_max.
        n_saturated = 0
        for purpose_name in self.purpose_names:
            for k in range(self.config.repr_dim):
                cname = _per_dim_constraint_name(purpose_name, k)
                if self.proxy.constraints[cname].lambda_value >= 0.95 * self.config.var_lambda_max:
                    n_saturated += 1

        # Compact summary across purposes
        sigma_all = []
        for arr in var_sigma_per_purpose.values():
            sigma_all.extend(arr)
        sigma_arr = torch.tensor(sigma_all)

        return {
            "primal_loss": float(primal_loss.detach().item()),
            "task": float(L_task.detach().item()),
            "vicreg": float(L_vicreg.detach().item()),
            "vclub_primal": float(L_vclub.detach().item()),
            "verify": float(L_verify.detach().item()),
            "eff_rank_pen": float(L_eff_rank.detach().item()),
            "r2_mean": (
                sum(v for k, v in constraint_scalars.items()
                    if not k.startswith("var_dim__"))
                / max(sum(1 for k in constraint_scalars if not k.startswith("var_dim__")), 1)
            ),
            "hsic_mean": (
                sum(hsic_scalars.values()) / max(len(hsic_scalars), 1)
            ),
            "perdim_n_saturated": n_saturated,
            "perdim_n_total": len(self.purpose_names) * self.config.repr_dim,
            "sigma_median": float(sigma_arr.median().item()),
            "sigma_min": float(sigma_arr.min().item()),
            "sigma_p10": float(sigma_arr.quantile(0.10).item()),
            "sigma_p25": float(sigma_arr.quantile(0.25).item()),
            "eff_rank_proxy_mean": (
                sum(eff_rank_log.values()) / max(len(eff_rank_log), 1)
            ),
            **{f"r2[{k}]": v for k, v in pair_r2_log.items()},
            **{f"r2[{k}]": v for k, v in constraint_scalars.items()
               if k not in pair_r2_log and not k.startswith("var_dim__")},
            **{f"hsic[{k}]": v for k, v in hsic_scalars.items()},
        }


# ───────────────────────────────────────────────────────────────────────────
# Warm-start + run
# ───────────────────────────────────────────────────────────────────────────


def _load_warm_start(trainer: PerDimLagrangianTrainer, ckpt_path: Path,
                     n_purposes: int, device: str) -> dict:
    log.info("[PERDIM/WARMSTART] loading %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.encoder.backbone.load_state_dict(ckpt["backbone"])
    trainer.encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    n_buffers_loaded = 0
    for p_idx in range(n_purposes):
        P_key = f"leace_P_p{p_idx}"
        mu_key = f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            trainer.encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
            n_buffers_loaded += 1
    trainer.task_heads.load_state_dict(ckpt["task_heads"])
    log.info(
        "[PERDIM/WARMSTART] loaded backbone + %d LoRA adapters + %d task heads "
        "+ %d leace buffers",
        n_purposes, len(trainer.task_heads), n_buffers_loaded,
    )
    return {
        "warm_start_path": str(ckpt_path),
        "had_encoder_buffers": bool(enc_buf),
        "n_buffers_loaded": n_buffers_loaded,
    }


def run_one_cell(dataset: str, seed: int, warm_start_tag: str,
                 epochs: int, device: str, out_dir: Path,
                 var_gamma: float, var_lambda_init: float,
                 var_lambda_max: float,
                 eff_rank_threshold: float, eff_rank_penalty_weight: float,
                 ) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info("=" * 72)
    log.info("[PERDIM] dataset=%s seed=%d epochs=%d device=%s lambda_max=%.1f",
             dataset, seed, epochs, device, var_lambda_max)
    log.info("=" * 72)

    purposes, train_ds, val_ds, test_ds = build_datasets(dataset)

    n_workers = NUM_WORKERS_BY_DATASET.get(dataset, 0)
    persistent = n_workers > 0
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch,
                              num_workers=n_workers,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate_pcrl_batch,
                            num_workers=n_workers,
                            persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch,
                             num_workers=n_workers,
                             persistent_workers=persistent)

    input_dim = train_ds.info.num_features

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    lora_rank, lora_alpha = LORA_BY_DATASET.get(dataset, (8, 16.0))
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=REPR_DIM, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes),
        rank=lora_rank, alpha=lora_alpha, dropout=0.0,
    )

    task_heads: dict = {}
    vclubs: dict = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        out_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=out_dim)
        for attr in p.disallowed_attrs:
            n_classes = p.disallowed_attr_dims.get(attr, 2)
            key = f"{p.name}__{attr}"
            vclubs[key] = VCLUB(
                x_dim=REPR_DIM, z_dim=n_classes,
                hidden_dim=128, z_categorical=True, l2=1e-1,
            )

    cell_dir = out_dir / f"{dataset}_s{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cell_dir / "checkpoints"

    config = PerDimLagrangianConfig(
        lambda_vicreg=1.0,
        vicreg_lambda_var=0.0,        # variance migrates to per-dim Lagrangian
        vicreg_lambda_cov=0.04,
        vicreg_gamma=1.0,
        lambda_vclub=1.0, lambda_verify=0.0,
        r2_threshold=0.05,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        batch_size=256, epochs=epochs,
        weight_decay=1e-4, grad_clip=1.0, vclub_steps=1,
        lambda_min=5.0,  # R² constraint floor (kept identical to varconstraint)
        per_class_constraint_threshold=6,
        checkpoint_dir=str(ckpt_dir),
        warmup_when_leace_init=True,
        warmup_epochs=0,
        leace_init=False,
        var_gamma=var_gamma,
        var_lambda_init=var_lambda_init,
        var_lambda_max=var_lambda_max,
        var_lambda_min=VAR_LAMBDA_MIN_DEFAULT,
        var_lr_lambda=VAR_LR_LAMBDA_DEFAULT,
        eff_rank_threshold=eff_rank_threshold,
        eff_rank_penalty_weight=eff_rank_penalty_weight,
        eff_rank_sigmoid_alpha=EFF_RANK_SIGMOID_ALPHA,
        repr_dim=REPR_DIM,
    )

    trainer = PerDimLagrangianTrainer(
        encoder=encoder, task_heads=task_heads, vclubs=vclubs,
        purpose_registry=registry, config=config, device=device,
    )

    warm_dir = ROOT / "checkpoints" / f"v2_{dataset}{warm_start_tag}_s{seed}"
    candidates = [warm_dir / "final.pt", warm_dir / "best.pt",
                  warm_dir / "canonical_iterate.pt"]
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        raise FileNotFoundError(
            f"No warm-start in {warm_dir}; tried {[p.name for p in candidates]}"
        )
    warm_meta = _load_warm_start(trainer, chosen, len(purposes), device)

    encoder.eval()
    pre_health: dict[str, dict] = {}
    for idx, p in enumerate(purposes):
        reps = reps_for_purpose(encoder, test_loader, idx, device)
        pre_health[p.name] = repr_health(reps)
    log.info("[PERDIM/PRE] per-purpose pre-training health:")
    for pname, h in pre_health.items():
        log.info("  %s: per_dim_std mean=%.3f min=%.3f eff_rank=%.2f",
                 pname, h["per_dim_std_mean"], h["per_dim_std_min"], h["effective_rank"])

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    log.info("[PERDIM] trained in %.1fs (last_epoch=%d, best_epoch=%d)",
             train_time, state.epoch, state.best_epoch)

    canonical = ckpt_dir / "canonical_iterate.pt"
    best = ckpt_dir / "best.pt"
    final = ckpt_dir / "final.pt"
    chosen_post = canonical if canonical.exists() else (best if best.exists() else final)
    if chosen_post.exists():
        ckpt = torch.load(chosen_post, map_location=device, weights_only=False)
        encoder.backbone.load_state_dict(ckpt["backbone"])
        encoder.adapters.load_state_dict(ckpt["lora_adapters"])
        enc_buf = ckpt.get("encoder_buffers", {}) or {}
        for p_idx in range(len(purposes)):
            P_key = f"leace_P_p{p_idx}"
            mu_key = f"leace_mu_p{p_idx}"
            if P_key in enc_buf and mu_key in enc_buf:
                encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])
        trainer.task_heads.load_state_dict(ckpt["task_heads"])
        log.info("[PERDIM/POST] reloaded %s for eval", chosen_post.name)
    encoder.eval()

    reports = generate_report(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    attr_results = []
    pass_count_r2 = 0
    pass_count_combined = 0
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        r2_pass = r.linear_r2 < 0.05
        combined_pass = bool(delta < 0.02 and r2_pass)
        if r2_pass:
            pass_count_r2 += 1
        if combined_pass:
            pass_count_combined += 1
        attr_results.append({
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "linear_r2": round(r.linear_r2, 6),
            "empirical_best_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "r2_pass": r2_pass,
            "adj_pass": combined_pass,
        })

    post_health: dict[str, dict] = {}
    for idx, p in enumerate(purposes):
        reps = reps_for_purpose(encoder, test_loader, idx, device)
        post_health[p.name] = repr_health(reps)

    val_metrics = trainer.evaluate(test_loader)
    task_accs = {k: round(float(v), 6) for k, v in val_metrics.task_accuracy.items()}

    # Per-dim duals: final lambda values per purpose (192 values for HMDA)
    perdim_lambdas: dict[str, list[float]] = {}
    for purpose_name in trainer.purpose_names:
        ls = []
        for k in range(REPR_DIM):
            cname = _per_dim_constraint_name(purpose_name, k)
            ls.append(float(trainer.proxy.constraints[cname].lambda_value))
        perdim_lambdas[purpose_name] = ls

    # Other (R²) duals
    r2_lambdas = {n: float(c.lambda_value)
                  for n, c in trainer.proxy.constraints.items()
                  if not n.startswith("var_dim__")}

    cleanly_compliant_pairs = 0
    for r in attr_results:
        h = post_health[r["purpose"]]
        clean = (
            r["r2_pass"]
            and h["per_dim_std_mean"] >= 0.5
            and h["effective_rank"] >= 2.0
        )
        r["cleanly_compliant"] = clean
        if clean:
            cleanly_compliant_pairs += 1

    return {
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "warm_start": warm_meta,
        "train_time_s": round(train_time, 1),
        "last_epoch": state.epoch,
        "best_epoch": state.best_epoch,
        "task_accuracies": task_accs,
        "attribute_results": attr_results,
        "pass_count_r2": pass_count_r2,
        "pass_count_combined": pass_count_combined,
        "cleanly_compliant_pairs": cleanly_compliant_pairs,
        "total_pairs": len(reports),
        "pre_health": pre_health,
        "post_health": post_health,
        "r2_lambdas_final": r2_lambdas,
        "perdim_lambdas_final": perdim_lambdas,
        "perdim_lambdas_summary": {
            p: {
                "min": min(ls), "max": max(ls), "median": sorted(ls)[len(ls)//2],
                "n_saturated_at_max": sum(1 for x in ls if x >= 0.95 * VAR_LAMBDA_MAX_DEFAULT),
            } for p, ls in perdim_lambdas.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["adult", "hmda", "diabetes"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=75)
    p.add_argument("--warm-start-tag", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="results/v2_perdim_lagrangian")
    p.add_argument("--var-gamma", type=float, default=VAR_GAMMA_DEFAULT)
    p.add_argument("--var-lambda-init", type=float, default=VAR_LAMBDA_INIT_DEFAULT)
    p.add_argument("--var-lambda-max", type=float, default=VAR_LAMBDA_MAX_DEFAULT)
    p.add_argument("--eff-rank-threshold", type=float, default=EFF_RANK_THRESHOLD_DEFAULT)
    p.add_argument("--eff-rank-penalty-weight", type=float,
                   default=EFF_RANK_PENALTY_WEIGHT_DEFAULT)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.warm_start_tag is None:
        args.warm_start_tag = "_ROUND7" if args.dataset == "diabetes" else "_ROUND5"

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_one_cell(
        dataset=args.dataset, seed=args.seed,
        warm_start_tag=args.warm_start_tag,
        epochs=args.epochs, device=device,
        out_dir=out_dir,
        var_gamma=args.var_gamma,
        var_lambda_init=args.var_lambda_init,
        var_lambda_max=args.var_lambda_max,
        eff_rank_threshold=args.eff_rank_threshold,
        eff_rank_penalty_weight=args.eff_rank_penalty_weight,
    )

    cell_dir = out_dir / f"{args.dataset}_s{args.seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cell_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("[PERDIM/DONE] wrote %s", metrics_path)
    log.info(
        "[PERDIM/DONE] R²-pass=%d/%d  combined-pass=%d/%d  cleanly-compliant=%d/%d",
        result["pass_count_r2"], result["total_pairs"],
        result["pass_count_combined"], result["total_pairs"],
        result["cleanly_compliant_pairs"], result["total_pairs"],
    )


if __name__ == "__main__":
    main()

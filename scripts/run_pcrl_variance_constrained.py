"""Variance-constrained PCRL retraining driver.

Tests whether adding a HARD per-dimension std constraint (per_dim_std >= 0.5
enforced via Lagrangian dual, not soft VICReg) plus a soft eff-rank penalty
converts collapse-compliant cells to cleanly compliant.

Spec:
  - Per-purpose variance Lagrangian: lambda_var * max(0, gamma - per_dim_std).mean()
    with gamma = 0.5, lambda_var in Lagrangian variables (separate dual per purpose).
  - Per-purpose eff-rank soft penalty using participation-ratio proxy
    (Sum(sigma))^2 / Sum(sigma^2), passed through sigmoid hinge.
  - Existing R^2 constraint (proxy-Lagrangian per (purpose, attribute)) preserved.
  - VICReg variance term replaced by the hard Lagrangian. VICReg covariance term
    kept (decorrelation is orthogonal and not a collapse mechanism).
  - Warm-start LoRA from existing trained checkpoint; reuse its leace_P/mu buffers.
  - 150 epochs, no extra warmup (model already converged to a feasible-collapsed
    fixed point; warmup-from-LEACE is irrelevant when starting from trained state).

Single (dataset, seed) per invocation. The launcher batches 9 invocations.

Usage:
    python scripts/run_pcrl_variance_constrained.py \\
        --dataset adult --seed 0 \\
        --warm-start-tag _ROUND5 \\
        --epochs 150 \\
        --out-dir results/v2_pcrl_variance_constrained
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, replace
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

# Reuse run_v2_dataset's dataset + health helpers without modifying it.
sys.path.insert(0, str(ROOT / "experiments"))
from run_v2_dataset import (  # noqa: E402
    LORA_BY_DATASET,
    NUM_WORKERS_BY_DATASET,
    build_datasets,
    effective_rank as audit_effective_rank,
    repr_health,
    reps_for_purpose,
)


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
)
log = logging.getLogger("pcrl_varconstraint")

VAR_GAMMA_DEFAULT = 0.5  # per-dim std hard floor (matches audit threshold)
EFF_RANK_THRESHOLD_DEFAULT = 2.0  # soft floor on participation-ratio proxy
EFF_RANK_PENALTY_WEIGHT_DEFAULT = 1.0
EFF_RANK_SIGMOID_ALPHA = 2.0  # steepness of sigmoid hinge
VAR_LAMBDA_INIT_DEFAULT = 1.0
VAR_LAMBDA_MAX_DEFAULT = 100.0
VAR_LAMBDA_MIN_DEFAULT = 0.1
VAR_LR_LAMBDA_DEFAULT = 0.05  # dual ascent rate for the var constraint


def _var_constraint_name(purpose_name: str) -> str:
    return f"var_min__{purpose_name}"


def participation_ratio(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """(sum(sigma))^2 / sum(sigma^2) on centered representation matrix.

    Lower bound is 1 (rank-1); upper bound is rank(z). Differentiable in z.
    Uses eigendecomposition of Z^T Z (more numerically stable than svd for
    near-rank-1 matrices).
    """
    z_c = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0] - 1, 1)
    cov = (z_c.T @ z_c) / n  # (d, d)
    # Symmetrize defensively for numerical noise.
    cov = 0.5 * (cov + cov.T)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=eps)
    sigma = torch.sqrt(eigvals)
    return (sigma.sum() ** 2) / (sigma.pow(2).sum() + eps)


def per_dim_std_slack(z: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
    """Mean over dims of max(0, gamma - sigma_d). Differentiable in z. Zero
    when every dim already has std >= gamma."""
    sigma = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(gamma - sigma).mean()


@dataclass
class VarianceConstrainedConfig(V2TrainerConfig):
    """Extends V2TrainerConfig with the new variance + eff-rank knobs."""

    var_gamma: float = VAR_GAMMA_DEFAULT
    var_lambda_init: float = VAR_LAMBDA_INIT_DEFAULT
    var_lambda_max: float = VAR_LAMBDA_MAX_DEFAULT
    var_lambda_min: float = VAR_LAMBDA_MIN_DEFAULT
    var_lr_lambda: float = VAR_LR_LAMBDA_DEFAULT
    eff_rank_threshold: float = EFF_RANK_THRESHOLD_DEFAULT
    eff_rank_penalty_weight: float = EFF_RANK_PENALTY_WEIGHT_DEFAULT
    eff_rank_sigmoid_alpha: float = EFF_RANK_SIGMOID_ALPHA


class VarianceConstrainedTrainer(V2Trainer):
    """V2Trainer + per-purpose hard variance constraint + soft eff-rank penalty.

    Adds one Lagrangian dual per purpose for the variance constraint.
    Eff-rank stays as a fixed-weight sigmoid hinge (rank is non-differentiable
    as a hard equality; sigmoid hinge gives a smooth lower-bound pressure).
    """

    config: VarianceConstrainedConfig  # for type checkers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        var_constraints: list[Constraint] = []
        for purpose_name in self.purpose_names:
            c = Constraint(
                name=_var_constraint_name(purpose_name),
                threshold=0.0,
                direction="<=",  # mean-slack must be <= 0 (= every dim >= gamma)
                eta_lambda=cfg.var_lr_lambda,
                lambda_init=cfg.var_lambda_init,
                lambda_max=cfg.var_lambda_max,
                lambda_min=cfg.var_lambda_min,
            )
            var_constraints.append(c)
            self.proxy.constraints[c.name] = c
        log.info(
            "[VARCONST/INIT] added %d variance Lagrangian duals (gamma=%.2f, "
            "lam_init=%.2f, lam_min=%.2f, lr_lambda=%.4f)",
            len(var_constraints), cfg.var_gamma, cfg.var_lambda_init,
            cfg.var_lambda_min, cfg.var_lr_lambda,
        )
        log.info(
            "[VARCONST/INIT] eff-rank soft penalty: threshold=%.2f, weight=%.2f, "
            "sigmoid_alpha=%.2f, vicreg_lambda_var=%.2f (should be 0 to avoid "
            "double-counting variance pressure), vicreg_lambda_cov=%.2f",
            cfg.eff_rank_threshold, cfg.eff_rank_penalty_weight,
            cfg.eff_rank_sigmoid_alpha, cfg.vicreg_lambda_var, cfg.vicreg_lambda_cov,
        )

    def _primal_and_dual_step(self, batch, apply_constraints: bool = True):
        # Run the base step but intercept its return to add var/eff-rank logging.
        # The base impl computes primal_loss INCLUDING var constraints (we
        # override by extending self.proxy.constraints in __init__) but does NOT
        # compute the var constraint values or the eff-rank penalty.
        # We re-implement the body so we can fold both in cleanly.
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

        # ── L_task ──────────────────────────────────────────────────────
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

        # ── L_vicreg (covariance term only; variance migrates to Lagrangian) ──
        L_vicreg = torch.tensor(0.0, device=self.device)
        for z in reprs.values():
            L_vicreg = L_vicreg + vicreg_loss(
                z,
                lambda_var=self.config.vicreg_lambda_var,  # set 0 in main()
                lambda_cov=self.config.vicreg_lambda_cov,
                gamma=self.config.vicreg_gamma,
            )

        # ── R² constraint values (per-pair, mirrors base impl) ───────────
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
                    pad = torch.zeros(
                        K_total - r2_per_k.numel(), device=self.device,
                    )
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

        # ── Variance hard constraint (per purpose) ──────────────────────
        var_slack_log: dict[str, float] = {}
        for purpose_name, z in reprs.items():
            slack = per_dim_std_slack(z, gamma=self.config.var_gamma)
            cname = _var_constraint_name(purpose_name)
            constraint_values[cname] = slack
            slack_scalar = float(slack.detach().item())
            constraint_scalars[cname] = slack_scalar
            var_slack_log[purpose_name] = slack_scalar

        # ── Eff-rank soft penalty (per purpose, fixed weight) ────────────
        L_eff_rank = torch.tensor(0.0, device=self.device)
        eff_rank_log: dict[str, float] = {}
        for purpose_name, z in reprs.items():
            er = participation_ratio(z)
            # Sigmoid hinge: large when er << threshold, near 0 when er > threshold.
            margin = self.config.eff_rank_threshold - er
            L_eff_rank = L_eff_rank + torch.sigmoid(
                self.config.eff_rank_sigmoid_alpha * margin
            )
            eff_rank_log[purpose_name] = float(er.detach().item())

        # ── Lagrangian + scalarised primal loss ──────────────────────────
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

        # ── Primal step ─────────────────────────────────────────────────
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

        return {
            "primal_loss": float(primal_loss.detach().item()),
            "task": float(L_task.detach().item()),
            "vicreg": float(L_vicreg.detach().item()),
            "vclub_primal": float(L_vclub.detach().item()),
            "verify": float(L_verify.detach().item()),
            "eff_rank_pen": float(L_eff_rank.detach().item()),
            "r2_mean": (
                sum(v for k, v in constraint_scalars.items()
                    if not k.startswith("var_min__"))
                / max(sum(1 for k in constraint_scalars if not k.startswith("var_min__")), 1)
            ),
            "hsic_mean": (
                sum(hsic_scalars.values()) / max(len(hsic_scalars), 1)
            ),
            "var_slack_mean": (
                sum(var_slack_log.values()) / max(len(var_slack_log), 1)
            ),
            "eff_rank_proxy_mean": (
                sum(eff_rank_log.values()) / max(len(eff_rank_log), 1)
            ),
            **{f"r2[{k}]": v for k, v in pair_r2_log.items()},
            **{f"r2[{k}]": v for k, v in constraint_scalars.items()
               if k not in pair_r2_log and not k.startswith("var_min__")},
            **{f"hsic[{k}]": v for k, v in hsic_scalars.items()},
            **{f"var_slack[{k}]": v for k, v in var_slack_log.items()},
            **{f"eff_rank_proxy[{k}]": v for k, v in eff_rank_log.items()},
        }


# ───────────────────────────────────────────────────────────────────────────
# Warm-start + run
# ───────────────────────────────────────────────────────────────────────────


def _load_warm_start(trainer: VarianceConstrainedTrainer, ckpt_path: Path,
                     n_purposes: int, device: str) -> dict:
    """Load backbone, lora_adapters, task_heads, and leace_P/mu buffers from
    ``ckpt_path`` into ``trainer``. Mirrors the reload path in
    ``run_v2_dataset.run_seed`` lines 320-339."""
    log.info("[VARCONST/WARMSTART] loading %s", ckpt_path)
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
        "[VARCONST/WARMSTART] loaded backbone + %d LoRA adapters + %d task heads "
        "+ %d leace_P/mu buffer pairs",
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
                 eff_rank_threshold: float, eff_rank_penalty_weight: float,
                 ) -> dict:
    """Train one (dataset, seed) cell with the variance-constrained trainer
    and return a per-cell summary dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info("=" * 72)
    log.info("[VARCONST] dataset=%s seed=%d epochs=%d device=%s",
             dataset, seed, epochs, device)
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
    repr_dim = 64

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    lora_rank, lora_alpha = LORA_BY_DATASET.get(dataset, (8, 16.0))
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=repr_dim, dropout=0.3,
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
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=out_dim)
        for attr in p.disallowed_attrs:
            n_classes = p.disallowed_attr_dims.get(attr, 2)
            key = f"{p.name}__{attr}"
            vclubs[key] = VCLUB(
                x_dim=repr_dim, z_dim=n_classes,
                hidden_dim=128, z_categorical=True, l2=1e-1,
            )

    cell_dir = out_dir / f"{dataset}_s{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cell_dir / "checkpoints"

    config = VarianceConstrainedConfig(
        # Inherit the canonical run knobs
        lambda_vicreg=1.0,            # multiplier on covariance only (var weight zeroed below)
        vicreg_lambda_var=0.0,        # OFF: variance migrates to Lagrangian
        vicreg_lambda_cov=0.04,       # ON: keeps decorrelation pressure
        vicreg_gamma=1.0,
        lambda_vclub=1.0, lambda_verify=0.0,
        r2_threshold=0.05,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
        batch_size=256, epochs=epochs,
        weight_decay=1e-4, grad_clip=1.0, vclub_steps=1,
        lambda_min=5.0,
        per_class_constraint_threshold=6,
        checkpoint_dir=str(ckpt_dir),
        # Skip warmup — we're warm-starting from a trained checkpoint, not
        # a zero-init LoRA. The warmup-from-LEACE phase only makes sense
        # the first time the LoRA sees task gradient.
        warmup_when_leace_init=True,  # forces respect of warmup_epochs=0 below
        warmup_epochs=0,
        leace_init=False,             # buffers come from ckpt, not refit
        # New variance + eff-rank knobs
        var_gamma=var_gamma,
        var_lambda_init=var_lambda_init,
        var_lambda_max=VAR_LAMBDA_MAX_DEFAULT,
        var_lambda_min=VAR_LAMBDA_MIN_DEFAULT,
        var_lr_lambda=VAR_LR_LAMBDA_DEFAULT,
        eff_rank_threshold=eff_rank_threshold,
        eff_rank_penalty_weight=eff_rank_penalty_weight,
        eff_rank_sigmoid_alpha=EFF_RANK_SIGMOID_ALPHA,
    )

    trainer = VarianceConstrainedTrainer(
        encoder=encoder, task_heads=task_heads, vclubs=vclubs,
        purpose_registry=registry, config=config, device=device,
    )

    # Warm-start: prefer canonical_iterate.pt > best.pt > final.pt, but the
    # spec calls for "the existing collapsed checkpoint" which IS final.pt
    # for Round 5/7 (paper's strict-pass numbers come from final).
    warm_dir = ROOT / "checkpoints" / f"v2_{dataset}{warm_start_tag}_s{seed}"
    candidates = [warm_dir / "final.pt", warm_dir / "best.pt",
                  warm_dir / "canonical_iterate.pt"]
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        raise FileNotFoundError(
            f"No warm-start checkpoint found in {warm_dir}; "
            f"tried {[p.name for p in candidates]}"
        )
    warm_meta = _load_warm_start(trainer, chosen, len(purposes), device)

    # Pre-training audit: confirm the warm-started rep is collapse-compliant
    # as expected.
    encoder.eval()
    pre_health: dict[str, dict] = {}
    for idx, p in enumerate(purposes):
        reps = reps_for_purpose(encoder, test_loader, idx, device)
        pre_health[p.name] = repr_health(reps)
    log.info("[VARCONST/PRE] per-purpose pre-training health:")
    for pname, h in pre_health.items():
        log.info(
            "  %s: per_dim_std_mean=%.3f min=%.3f eff_rank=%.2f",
            pname, h["per_dim_std_mean"], h["per_dim_std_min"], h["effective_rank"],
        )

    # Train
    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    log.info("[VARCONST] trained in %.1fs (last_epoch=%d, best_epoch=%d)",
             train_time, state.epoch, state.best_epoch)

    # Reload Cotter best (mirrors run_v2_dataset's behavior).
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
        log.info("[VARCONST/POST] reloaded %s for eval", chosen_post.name)
    encoder.eval()

    # Compliance via paper's adjusted criterion (R² + post-hoc auditor delta).
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

    # Post-training health
    post_health: dict[str, dict] = {}
    for idx, p in enumerate(purposes):
        reps = reps_for_purpose(encoder, test_loader, idx, device)
        post_health[p.name] = repr_health(reps)

    # Task accuracies via the trained heads
    val_metrics = trainer.evaluate(test_loader)
    task_accs = {k: round(float(v), 6) for k, v in val_metrics.task_accuracy.items()}

    # Final lambda values
    lambdas_final = {n: float(c.lambda_value) for n, c in trainer.proxy.constraints.items()}

    # Cleanly-compliant per pair: r2_pass AND purpose health passes
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
        "lambdas_final": lambdas_final,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["adult", "hmda", "diabetes"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warm-start-tag", default=None,
                   help="Default: _ROUND5 for adult/hmda, _ROUND7 for diabetes")
    p.add_argument("--device", default=None,
                   help="Default: cuda if available else cpu")
    p.add_argument("--out-dir", default="results/v2_pcrl_variance_constrained")
    p.add_argument("--var-gamma", type=float, default=VAR_GAMMA_DEFAULT)
    p.add_argument("--var-lambda-init", type=float, default=VAR_LAMBDA_INIT_DEFAULT)
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
        eff_rank_threshold=args.eff_rank_threshold,
        eff_rank_penalty_weight=args.eff_rank_penalty_weight,
    )

    cell_dir = out_dir / f"{args.dataset}_s{args.seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cell_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("[VARCONST/DONE] wrote %s", metrics_path)
    log.info(
        "[VARCONST/DONE] R²-pass=%d/%d  combined-pass=%d/%d  cleanly-compliant=%d/%d  "
        "(total pairs %d)",
        result["pass_count_r2"], result["total_pairs"],
        result["pass_count_combined"], result["total_pairs"],
        result["cleanly_compliant_pairs"], result["total_pairs"],
        result["total_pairs"],
    )


if __name__ == "__main__":
    main()

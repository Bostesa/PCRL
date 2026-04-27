"""V2 trainer: LoRA + linear-R² constraint + HSIC aux + vCLUB + VICReg + proxy-Lagrangian.

Replaces the adversarial / minimax loop in ``trainer.py`` with a
non-adversarial constrained optimisation loop. The primary independence
constraint is the auditor's metric — linear R² of the optimal predictor
of attr from z — so the optimiser and the audit cannot disagree.

    Primal step (per batch):
        z_p   = encoder(x, p)               # frozen backbone + per-purpose LoRA
        L_task     = sum_p CE(task_head_p(z_p), y_p)
        L_vicreg   = sum_p vicreg_loss(z_p)
        L_vclub    = sum_{p,a} vclub_{p,a}.mi_upper_bound(z_p, attr_a)
        L_hsic_aux = sum_{p,a} HSIC(z_p, attr_a)        # fixed-weight nonlinear aux
        primal_loss = L_task
                    + lambda_vicreg   * L_vicreg
                    + lambda_vclub    * L_vclub
                    + lambda_hsic_aux * L_hsic_aux
                    + sum_{p,a} lambda_{p,a} * (R²(z_p, attr_a) - tau)
        backward; step (LoRA adapters + task head params)

    vCLUB q-net step (per batch, before primal):
        for each (p, a): minimise q-net's learning_loss(z_p.detach(), attr_a)
        (separate optimiser; X is detached so encoder isn't updated)

    Dual step (per batch, after primal):
        for each (p, a): lambda_{p,a} <- proj([0, lam_max],
                                              lambda + eta * (R² - tau))

Early stopping: best val_task_loss only (not composite). Composite includes
the constraint violations and would reward collapse, exactly as in v1.

The backbone is frozen; only LoRA adapters and task heads are trained on
the primal optimiser. vCLUB q-nets get their own optimiser. The
``VerificationRegularizer`` is parameter-less (just an R² computation),
so no separate optimiser is needed for it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.independence.hsic import hsic
from pcrl.training.independence.vclub import VCLUB
from pcrl.training.independence.vicreg import vicreg_loss
from pcrl.training.losses import VerificationRegularizer, task_loss
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer

logger = logging.getLogger(__name__)


@dataclass
class V2TrainerConfig:
    """Hyperparameters for the v2 trainer."""

    # Optimiser learning rates
    lr_primal: float = 1e-3
    lr_vclub: float = 1e-3
    lr_lambda: float = 0.05

    # Loss weights (fixed scalarisation for terms that aren't constraints)
    lambda_vicreg: float = 1.0
    lambda_vclub: float = 1.0
    lambda_verify: float = 0.0  # Legacy; R² is now the constraint, not a fixed-weight term.
    lambda_hsic_aux: float = 0.1  # Fixed weight for HSIC as a nonlinear auxiliary.

    # Linear-R² constraint (proxy-Lagrangian primary; matches auditor's metric).
    lambda_hsic_init: float = 1.0  # Reused as initial dual variable for the R² constraint.
    r2_threshold: float = 0.05
    r2_lambda_max: float = 100.0

    # VICReg knobs
    vicreg_gamma: float = 1.0
    vicreg_lambda_var: float = 1.0
    vicreg_lambda_cov: float = 0.04

    # LoRA architecture
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # vCLUB q-network
    vclub_hidden: int = 128
    vclub_l2: float = 1e-1
    vclub_steps: int = 1  # q-net updates per encoder step

    # Training schedule
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int | None = 20
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    log_interval: int = 50
    checkpoint_dir: str = "checkpoints/v2"


@dataclass
class V2EpochMetrics:
    """Snapshot of training/eval state after one epoch."""

    loss: float = 0.0
    task_loss: float = 0.0
    vclub_q_loss: float = 0.0
    vicreg_loss: float = 0.0
    verify_loss: float = 0.0
    hsic_mean: float = 0.0
    r2_mean: float = 0.0
    task_accuracy: dict[str, float] = field(default_factory=dict)
    hsic_per_pair: dict[str, float] = field(default_factory=dict)
    r2_per_pair: dict[str, float] = field(default_factory=dict)
    epoch_time: float = 0.0


@dataclass
class V2State:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)


def _pair_key(purpose_name: str, attr_name: str) -> str:
    return f"{purpose_name}__{attr_name}"


class V2Trainer:
    """Constrained-optimisation trainer for v2 PCRL."""

    def __init__(
        self,
        encoder: PerPurposeLoRAEncoder,
        task_heads: dict[str, nn.Module],
        vclubs: dict[str, VCLUB],
        purpose_registry: PurposeRegistry,
        config: V2TrainerConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)
        self.vclubs = nn.ModuleDict(vclubs)
        self.verifier = VerificationRegularizer()
        self.config = config
        self.device = torch.device(device)
        self.purpose_registry = purpose_registry

        # Freeze the backbone explicitly (LoRA wrapper does this in __init__,
        # but redoing it here is harmless and self-documenting).
        for p in self.encoder.backbone.parameters():
            p.requires_grad_(False)

        self.encoder.to(self.device)
        self.task_heads.to(self.device)
        self.vclubs.to(self.device)
        self.verifier.to(self.device)

        # Build per-purpose config from registry order: purpose_idx == registration order.
        self.purpose_configs: dict[str, dict[str, Any]] = {}
        for idx, purpose in enumerate(purpose_registry.purposes):
            self.purpose_configs[purpose.name] = {
                "purpose_idx": idx,
                "task_type": purpose.task_type,
                "allowed_tasks": purpose.allowed_tasks,
                "disallowed_attrs": purpose.disallowed_attrs,
            }

        self.purpose_names: list[str] = list(self.purpose_configs.keys())

        # Build linear-R² constraints (one per (purpose, attr) pair).
        # The auditor measures linear R²; constraining the same quantity removes
        # the metric mismatch that made HSIC-only training pass-without-passing.
        constraints: list[Constraint] = []
        self.pair_keys: list[tuple[str, str]] = []
        for purpose_name in self.purpose_names:
            for attr_name in self.purpose_configs[purpose_name]["disallowed_attrs"]:
                name = _pair_key(purpose_name, attr_name)
                constraints.append(
                    Constraint(
                        name=name,
                        threshold=config.r2_threshold,
                        direction="<=",
                        eta_lambda=config.lr_lambda,
                        lambda_init=config.lambda_hsic_init,
                        lambda_max=config.r2_lambda_max,
                    )
                )
                self.pair_keys.append((purpose_name, attr_name))

        # Optimisers
        primal_params: list[nn.Parameter] = list(self.encoder.trainable_parameters())
        primal_params += list(self.task_heads.parameters())
        # NB: VerificationRegularizer is parameter-less, nothing to add.
        self.primal_optimizer = AdamW(
            primal_params, lr=config.lr_primal, weight_decay=config.weight_decay,
        )
        self.proxy = ProxyLagrangianOptimizer(self.primal_optimizer, constraints)

        # vCLUB q-network optimiser (separate; trains on detached representations).
        self.vclub_optimizer = Adam(
            list(self.vclubs.parameters()), lr=config.lr_vclub,
        )

        self.state = V2State()

    # ── helpers ─────────────────────────────────────────────────────────

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "features": batch["features"].to(self.device),
            "task_labels": {k: v.to(self.device) for k, v in batch["task_labels"].items()},
            "sensitive_attrs": {k: v.to(self.device) for k, v in batch["sensitive_attrs"].items()},
        }

    def _primary_task(self, purpose_name: str) -> str:
        return self.purpose_configs[purpose_name]["allowed_tasks"][0]

    def _task_type(self, purpose_name: str) -> str:
        return self.purpose_configs[purpose_name]["task_type"]

    def _purpose_idx(self, purpose_name: str) -> int:
        return self.purpose_configs[purpose_name]["purpose_idx"]

    # ── training step ───────────────────────────────────────────────────

    def _vclub_q_step(self, batch: dict[str, Any]) -> float:
        """Train every (p, a) vCLUB q-network on detached representations."""
        total = torch.tensor(0.0, device=self.device)
        # We need representations without tracking encoder grads:
        with torch.no_grad():
            reprs = {
                purpose_name: self.encoder(batch["features"], self._purpose_idx(purpose_name))
                for purpose_name in self.purpose_names
            }
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            vclub = self.vclubs[_pair_key(purpose_name, attr_name)]
            total = total + vclub.learning_loss(z, attr)

        self.vclub_optimizer.zero_grad()
        total.backward()
        self.vclub_optimizer.step()
        return float(total.detach().item())

    def _primal_and_dual_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """One primal step (LoRA + heads) + one dual ascent on lambdas."""
        self.encoder.train()
        self.task_heads.train()

        # Forward all purposes (each requires a separate hook-injected pass).
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

        # ── L_vicreg ────────────────────────────────────────────────────
        L_vicreg = torch.tensor(0.0, device=self.device)
        for z in reprs.values():
            L_vicreg = L_vicreg + vicreg_loss(
                z,
                lambda_var=self.config.vicreg_lambda_var,
                lambda_cov=self.config.vicreg_lambda_cov,
                gamma=self.config.vicreg_gamma,
            )

        # ── L_vclub, L_hsic_aux, L_verify (logging), R² constraint values ──
        L_vclub = torch.tensor(0.0, device=self.device)
        L_hsic_aux = torch.tensor(0.0, device=self.device)
        L_verify = torch.tensor(0.0, device=self.device)
        constraint_values: dict[str, torch.Tensor] = {}
        constraint_scalars: dict[str, float] = {}
        hsic_scalars: dict[str, float] = {}
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            key = _pair_key(purpose_name, attr_name)

            L_vclub = L_vclub + self.vclubs[key].mi_upper_bound(z, attr)

            # HSIC kept as differentiable fixed-weight auxiliary (nonlinear cover).
            hsic_val = hsic(z, attr)
            L_hsic_aux = L_hsic_aux + hsic_val
            hsic_scalars[key] = float(hsic_val.detach().item())

            # Linear R² is the proxy-Lagrangian constraint value (matches auditor).
            if int(attr.max().item()) >= 1:  # verifier only valid for >= 2 classes
                r2_val = self.verifier(z, attr)
            else:
                r2_val = torch.tensor(0.0, device=self.device)
            L_verify = L_verify + r2_val
            constraint_values[key] = r2_val
            constraint_scalars[key] = float(r2_val.detach().item())

        # ── Lagrangian + scalarised primal loss ─────────────────────────
        base = (
            L_task
            + self.config.lambda_vicreg * L_vicreg
            + self.config.lambda_vclub * L_vclub
            + self.config.lambda_hsic_aux * L_hsic_aux
            + self.config.lambda_verify * L_verify  # legacy; default 0
        )
        primal_loss = self.proxy.lagrangian_loss(base, constraint_values)

        # ── Primal step ─────────────────────────────────────────────────
        self.primal_optimizer.zero_grad()
        primal_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.primal_optimizer.param_groups[0]["params"]],
                max_norm=self.config.grad_clip,
            )
        self.primal_optimizer.step()

        # ── Dual step ───────────────────────────────────────────────────
        self.proxy.dual_step(constraint_scalars)

        return {
            "primal_loss": float(primal_loss.detach().item()),
            "task": float(L_task.detach().item()),
            "vicreg": float(L_vicreg.detach().item()),
            "vclub_primal": float(L_vclub.detach().item()),
            "verify": float(L_verify.detach().item()),
            "r2_mean": (
                sum(constraint_scalars.values()) / max(len(constraint_scalars), 1)
            ),
            "hsic_mean": (
                sum(hsic_scalars.values()) / max(len(hsic_scalars), 1)
            ),
            **{f"r2[{k}]": v for k, v in constraint_scalars.items()},
            **{f"hsic[{k}]": v for k, v in hsic_scalars.items()},
        }

    # ── epoch + eval ────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> V2EpochMetrics:
        self.encoder.train()
        self.task_heads.train()
        self.vclubs.train()

        t0 = time.time()
        sums: dict[str, float] = {}
        n = 0

        for batch in loader:
            batch = self._to_device(batch)

            # vCLUB q-network step(s)
            q_loss = 0.0
            for _ in range(self.config.vclub_steps):
                q_loss = self._vclub_q_step(batch)

            stats = self._primal_and_dual_step(batch)
            stats["vclub_q_loss"] = q_loss
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            self.state.global_step += 1

        m = V2EpochMetrics(
            loss=sums.get("primal_loss", 0.0) / max(n, 1),
            task_loss=sums.get("task", 0.0) / max(n, 1),
            vclub_q_loss=sums.get("vclub_q_loss", 0.0) / max(n, 1),
            vicreg_loss=sums.get("vicreg", 0.0) / max(n, 1),
            verify_loss=sums.get("verify", 0.0) / max(n, 1),
            hsic_mean=sums.get("hsic_mean", 0.0) / max(n, 1),
            r2_mean=sums.get("r2_mean", 0.0) / max(n, 1),
            epoch_time=time.time() - t0,
        )
        m.hsic_per_pair = {
            k[len("hsic["):-1]: sums[k] / max(n, 1)
            for k in sums if k.startswith("hsic[")
        }
        m.r2_per_pair = {
            k[len("r2["):-1]: sums[k] / max(n, 1)
            for k in sums if k.startswith("r2[")
        }
        return m

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> V2EpochMetrics:
        self.encoder.eval()
        self.task_heads.eval()
        self.vclubs.eval()

        t0 = time.time()
        total_task = 0.0
        total_hsic = 0.0
        total_r2 = 0.0
        per_pair_hsic_sum: dict[str, float] = {}
        per_pair_r2_sum: dict[str, float] = {}
        task_correct: dict[str, int] = {}
        task_total: dict[str, int] = {}
        n = 0
        n_pair_batches: dict[str, int] = {}

        for batch in loader:
            batch = self._to_device(batch)
            reprs: dict[str, torch.Tensor] = {}
            for purpose_name in self.purpose_names:
                reprs[purpose_name] = self.encoder(
                    batch["features"], self._purpose_idx(purpose_name),
                )
            # Task loss + accuracy
            L_task = 0.0
            for purpose_name, z in reprs.items():
                task_name = self._primary_task(purpose_name)
                if task_name not in batch["task_labels"]:
                    continue
                head = self.task_heads[purpose_name]
                preds = head(z)
                if isinstance(preds, dict):
                    preds = preds[task_name]
                targets = batch["task_labels"][task_name]
                L_task += float(task_loss(preds, targets, self._task_type(purpose_name)).item())
                pred_classes = preds.argmax(dim=-1)
                task_correct[task_name] = task_correct.get(task_name, 0) + int((pred_classes == targets).sum().item())
                task_total[task_name] = task_total.get(task_name, 0) + targets.numel()
            total_task += L_task

            # HSIC + R² values per (p, a)
            for purpose_name, attr_name in self.pair_keys:
                z = reprs[purpose_name]
                attr = batch["sensitive_attrs"][attr_name].long()
                key = _pair_key(purpose_name, attr_name)
                hv = float(hsic(z, attr).item())
                if int(attr.max().item()) >= 1:
                    rv = float(self.verifier(z, attr).item())
                else:
                    rv = 0.0
                per_pair_hsic_sum[key] = per_pair_hsic_sum.get(key, 0.0) + hv
                per_pair_r2_sum[key] = per_pair_r2_sum.get(key, 0.0) + rv
                n_pair_batches[key] = n_pair_batches.get(key, 0) + 1
                total_hsic += hv
                total_r2 += rv

            n += 1

        m = V2EpochMetrics(
            loss=total_task / max(n, 1),  # task loss alone for early stopping
            task_loss=total_task / max(n, 1),
            hsic_mean=total_hsic / max(n * len(self.pair_keys), 1),
            r2_mean=total_r2 / max(n * len(self.pair_keys), 1),
            task_accuracy={k: task_correct[k] / task_total[k] for k in task_correct if task_total[k] > 0},
            hsic_per_pair={k: v / max(n_pair_batches[k], 1) for k, v in per_pair_hsic_sum.items()},
            r2_per_pair={k: v / max(n_pair_batches[k], 1) for k, v in per_pair_r2_sum.items()},
            epoch_time=time.time() - t0,
        )
        return m

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> V2State:
        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            tr = self.train_epoch(train_loader)
            self._update_history("train", tr)
            logger.info(
                f"[V2/TRAIN] epoch={epoch} loss={tr.loss:.4f} task={tr.task_loss:.4f} "
                f"vicreg={tr.vicreg_loss:.4f} r2_mean={tr.r2_mean:.4f} "
                f"hsic_mean={tr.hsic_mean:.4f} q_loss={tr.vclub_q_loss:.4f} time={tr.epoch_time:.1f}s"
            )

            if val_loader is not None:
                val = self.evaluate(val_loader)
                self._update_history("val", val)
                logger.info(
                    f"[V2/VAL]   epoch={epoch} task={val.task_loss:.4f} "
                    f"r2_mean={val.r2_mean:.4f} hsic_mean={val.hsic_mean:.4f} "
                    f"time={val.epoch_time:.1f}s"
                )
                # Early stop on val task loss alone (NOT composite — composite would
                # reward collapse via the HSIC + vCLUB terms).
                if val.task_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val.task_loss
                    self.state.patience_counter = 0
                    self.save_checkpoint("best")
                else:
                    self.state.patience_counter += 1

                if (
                    self.config.early_stopping_patience is not None
                    and self.state.patience_counter >= self.config.early_stopping_patience
                ):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self.save_checkpoint("final")
        return self.state

    # ── checkpointing + history ─────────────────────────────────────────

    def _update_history(self, prefix: str, metrics: V2EpochMetrics) -> None:
        for key, val in (
            ("loss", metrics.loss),
            ("task_loss", metrics.task_loss),
            ("vicreg_loss", metrics.vicreg_loss),
            ("verify_loss", metrics.verify_loss),
            ("hsic_mean", metrics.hsic_mean),
            ("r2_mean", metrics.r2_mean),
            ("vclub_q_loss", metrics.vclub_q_loss),
        ):
            self.state.history.setdefault(f"{prefix}_{key}", []).append(val)

    def save_checkpoint(self, name: str) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "backbone": self.encoder.backbone.state_dict(),
            "lora_adapters": self.encoder.adapters.state_dict(),
            "task_heads": self.task_heads.state_dict(),
            "vclubs": self.vclubs.state_dict(),
            "lambdas": {n: c.lambda_value for n, c in self.proxy.constraints.items()},
            "config": vars(self.config),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
            },
            "history": self.state.history,
        }
        path = ckpt_dir / f"{name}.pt"
        torch.save(ckpt, path)
        return path

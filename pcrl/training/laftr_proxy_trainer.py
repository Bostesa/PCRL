"""LAFTR + proxy-Lagrangian trainer.

Replaces LAFTR's fixed-weight adversarial loss with a hard linear-R²
constraint enforced via proxy-Lagrangian dual variables — the same
mechanism PCRL's v2 trainer uses. LAFTR's adversarial discriminators
survive as a fixed-weight regulariser at ``lambda_adv=1.0`` (Appendix Q's
published weight) so the encoder still gets a nonlinear "fool the
adversary" gradient on top of the linear-R² constraint.

Primal step (per batch):
    z_p   = encoder(x, p)               # frozen backbone + per-purpose LoRA
    L_task     = sum_p CE(task_head_p(z_p), y_p)
    L_vicreg   = sum_p vicreg_loss(z_p)
    L_adv      = sum_{p,a} CE(disc_{p,a}(z_p), attr_a)  # detach=False
    primal_loss = L_task
                + lambda_vicreg * L_vicreg
                - lambda_adv    * L_adv                            # NEW vs v2
                + sum_{p,a} lambda_{p,a} * (R²(z_p, attr_a) - tau)
    backward; step (LoRA adapters + task head params)

Discriminator step (per batch, BEFORE primal):
    for each (p, a): minimise CE(disc_{p,a}(z_p.detach()), attr_a)
    (separate optimiser; X is detached so encoder isn't updated)

Dual step (per batch, AFTER primal):
    for each (p, a): lambda_{p,a} <- proj([lambda_min, lambda_max],
                                          lambda + lr_lambda * (R² - tau))

All other machinery (LEACE warm-start, Cotter best-iterate, per-class
OvR for high-K attrs, frozen backbone BN, checkpointing) is copied
unmodified from V2Trainer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from pcrl.baselines.laftr import LAFTRDiscriminator
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.independence.vicreg import vicreg_loss
from pcrl.training.losses import VerificationRegularizer, task_loss
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer

logger = logging.getLogger(__name__)


def _pair_key(purpose_name: str, attr_name: str) -> str:
    return f"{purpose_name}__{attr_name}"


def _per_class_key(purpose_name: str, attr_name: str, k: int) -> str:
    return f"{purpose_name}__{attr_name}__class_{k}"


@dataclass
class LAFTRProxyTrainerConfig:
    """LAFTR + proxy-Lagrangian hyperparameters.

    Defaults intentionally mirror ``V2TrainerConfig`` so the only knob that
    differs vs PCRL is ``lambda_adv`` (the LAFTR discriminator weight).
    """

    lr_primal: float = 1e-3
    lr_disc: float = 1e-3
    lr_lambda: float = 0.02

    lambda_vicreg: float = 1.0
    lambda_adv: float = 1.0   # Headline: apples-to-apples with Appendix Q's LAFTR.
                              # Fallback to 0.1 only if 1.0 is empirically unstable
                              # (adversary fights dual → oscillation); document
                              # the instability in SUMMARY.md if invoked.
                              # Ablation at 0.0 (pure proxy-Lagrangian, no adversary)
                              # runs after headline if compute permits.
    lambda_hsic_init: float = 1.0
    r2_threshold: float = 0.05
    r2_lambda_max: float = 1000.0
    lambda_min: float = 5.0   # Round 5 fix (memory: project_v2_round5)
    per_class_constraint_threshold: int = 6

    cotter_fallback_task_slack: float = 0.10

    leace_init: bool = True
    warmup_when_leace_init: bool = False
    warmup_epochs: int = 5

    vicreg_gamma: float = 1.0
    vicreg_lambda_var: float = 1.0
    vicreg_lambda_cov: float = 0.04

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    disc_steps: int = 1

    batch_size: int = 256
    epochs: int = 200
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    log_interval: int = 50
    checkpoint_dir: str = "checkpoints/laftr_hard_r2"

    report_best_iterate: bool = False


class LAFTRProxyTrainer:
    """LAFTR with hard linear-R² constraint via proxy-Lagrangian dual.

    Composes the same primitives as ``V2Trainer`` but swaps the vCLUB block
    for LAFTR discriminators and adds a ``− lambda_adv · Σ CE(disc, attr)``
    term to the encoder's primal loss.
    """

    def __init__(
        self,
        encoder: PerPurposeLoRAEncoder,
        task_heads: dict[str, nn.Module],
        purpose_registry: PurposeRegistry,
        config: LAFTRProxyTrainerConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)
        self.verifier = VerificationRegularizer()
        self.config = config
        self.device = torch.device(device)
        self.purpose_registry = purpose_registry

        # Freeze backbone (params + BN buffers).
        for p in self.encoder.backbone.parameters():
            p.requires_grad_(False)
        self._freeze_backbone_bn()
        self.encoder.to(self.device)
        self.task_heads.to(self.device)
        self.verifier.to(self.device)

        self.purpose_configs: dict[str, dict[str, Any]] = {}
        for idx, purpose in enumerate(purpose_registry.purposes):
            self.purpose_configs[purpose.name] = {
                "purpose_idx": idx,
                "task_type": purpose.task_type,
                "allowed_tasks": purpose.allowed_tasks,
                "disallowed_attrs": purpose.disallowed_attrs,
                "disallowed_attr_dims": dict(purpose.disallowed_attr_dims),
            }
        self.purpose_names: list[str] = list(self.purpose_configs.keys())

        # ── Constraint construction (mirrors V2Trainer.__init__ l.362–476) ──
        constraints: list[Constraint] = []
        self.pair_keys: list[tuple[str, str]] = []
        self.pair_constraint_keys: dict[tuple[str, str], list[str]] = {}
        self.high_k_pairs: dict[tuple[str, str], int] = {}
        per_class_threshold = config.per_class_constraint_threshold

        # ── Discriminator construction (one per (purpose, attribute)) ──
        self.discriminators = nn.ModuleDict()

        for purpose_name in self.purpose_names:
            attr_dims = self.purpose_configs[purpose_name]["disallowed_attr_dims"]
            for attr_name in self.purpose_configs[purpose_name]["disallowed_attrs"]:
                pair = (purpose_name, attr_name)
                self.pair_keys.append(pair)
                K = int(attr_dims.get(attr_name, 2))
                pkey = _pair_key(purpose_name, attr_name)
                # One discriminator per pair (predicts attr from z_p)
                self.discriminators[pkey] = LAFTRDiscriminator(
                    repr_dim=64, num_classes=K,
                ).to(self.device)
                if K >= per_class_threshold:
                    self.high_k_pairs[pair] = K
                    names = [_per_class_key(purpose_name, attr_name, k) for k in range(K)]
                    for n in names:
                        constraints.append(
                            Constraint(
                                name=n, threshold=config.r2_threshold,
                                direction="<=", eta_lambda=config.lr_lambda,
                                lambda_init=config.lambda_hsic_init,
                                lambda_max=config.r2_lambda_max,
                                lambda_min=config.lambda_min,
                            )
                        )
                    self.pair_constraint_keys[pair] = names
                else:
                    constraints.append(
                        Constraint(
                            name=pkey, threshold=config.r2_threshold,
                            direction="<=", eta_lambda=config.lr_lambda,
                            lambda_init=config.lambda_hsic_init,
                            lambda_max=config.r2_lambda_max,
                            lambda_min=config.lambda_min,
                        )
                    )
                    self.pair_constraint_keys[pair] = [pkey]

        primal_params = list(self.encoder.trainable_parameters()) + list(self.task_heads.parameters())
        self.primal_optimizer = AdamW(
            primal_params, lr=config.lr_primal, weight_decay=config.weight_decay,
        )
        self.proxy = ProxyLagrangianOptimizer(self.primal_optimizer, constraints)

        self.disc_optimizer = Adam(
            list(self.discriminators.parameters()), lr=config.lr_disc,
        )

        # Minimal in-memory state; full V2State copy comes in Task 6.
        self.state: dict[str, Any] = {"epoch": 0, "global_step": 0, "history": {}}

    def _freeze_backbone_bn(self) -> None:
        for m in self.encoder.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def _purpose_idx(self, purpose_name: str) -> int:
        return self.purpose_configs[purpose_name]["purpose_idx"]

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "features": batch["features"].to(self.device),
            "task_labels": {k: v.to(self.device) for k, v in batch["task_labels"].items()},
            "sensitive_attrs": {k: v.to(self.device) for k, v in batch["sensitive_attrs"].items()},
        }

    def _primal_step_components(
        self, batch: dict[str, Any], apply_constraints: bool = True,
    ) -> dict[str, Any]:
        """One primal+dual step. Returns scalar stats including a verifiable
        decomposition of the primal loss (used by tests + logs).
        """
        self.encoder.train()
        self._freeze_backbone_bn()
        self.task_heads.train()
        for d in self.discriminators.values():
            d.train()

        reprs: dict[str, torch.Tensor] = {}
        for purpose_name in self.purpose_names:
            reprs[purpose_name] = self.encoder(
                batch["features"], self._purpose_idx(purpose_name),
            )

        # L_task
        L_task = torch.tensor(0.0, device=self.device)
        for purpose_name, z in reprs.items():
            task_name = self.purpose_configs[purpose_name]["allowed_tasks"][0]
            if task_name not in batch["task_labels"]:
                continue
            head = self.task_heads[purpose_name]
            preds = head(z)
            if isinstance(preds, dict):
                preds = preds[task_name]
            targets = batch["task_labels"][task_name]
            L_task = L_task + task_loss(
                preds, targets, self.purpose_configs[purpose_name]["task_type"],
            )

        # L_vicreg
        L_vicreg = torch.tensor(0.0, device=self.device)
        for z in reprs.values():
            L_vicreg = L_vicreg + vicreg_loss(
                z, lambda_var=self.config.vicreg_lambda_var,
                lambda_cov=self.config.vicreg_lambda_cov,
                gamma=self.config.vicreg_gamma,
            )

        # L_adv (discriminator CE, gradients flow into encoder via z)
        # Freeze discriminator params for this backward so disc weights don't
        # accumulate phantom gradients.
        for p_ in self.discriminators.parameters():
            p_.requires_grad_(False)
        L_adv = torch.tensor(0.0, device=self.device)
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pkey = _pair_key(purpose_name, attr_name)
            disc = self.discriminators[pkey]
            L_adv = L_adv + F.cross_entropy(disc(z), attr)
        for p_ in self.discriminators.parameters():
            p_.requires_grad_(True)

        # Per-pair linear-R² constraint values (mirrors V2Trainer l.716–770)
        constraint_values: dict[str, torch.Tensor] = {}
        constraint_scalars: dict[str, float] = {}
        pair_r2_log: dict[str, float] = {}
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pair = (purpose_name, attr_name)
            pkey = _pair_key(purpose_name, attr_name)
            if int(attr.max().item()) < 1:
                zero = torch.tensor(0.0, device=self.device)
                for cname in self.pair_constraint_keys[pair]:
                    constraint_values[cname] = zero
                    constraint_scalars[cname] = 0.0
                pair_r2_log[pkey] = 0.0
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
                pair_r2_log[pkey] = pair_max
            else:
                r2_val = self.verifier(z, attr)
                cname = self.pair_constraint_keys[pair][0]
                constraint_values[cname] = r2_val
                constraint_scalars[cname] = float(r2_val.detach().item())
                pair_r2_log[pkey] = constraint_scalars[cname]

        # Snapshot λ values BEFORE the dual_step mutates them so the returned
        # stats let a caller reconstruct the exact primal_loss they got back.
        pre_dual_lambdas = {n: c.lambda_value for n, c in self.proxy.constraints.items()}

        if apply_constraints:
            base = (
                L_task
                + self.config.lambda_vicreg * L_vicreg
                - self.config.lambda_adv * L_adv
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

        return {
            "primal_loss": float(primal_loss.detach().item()),
            "L_task": float(L_task.detach().item()),
            "L_vicreg": float(L_vicreg.detach().item()),
            "L_adv": float(L_adv.detach().item()),
            "constraint_scalars": constraint_scalars,
            "pre_dual_lambdas": pre_dual_lambdas,
            "pair_r2": pair_r2_log,
        }

    def _discriminator_step(self, batch: dict[str, Any]) -> float:
        """Phase 1: minimise Σ_{p,a} CE(disc_{p,a}(z_p.detach()), attr_a).

        Detaching the representation means gradients do NOT propagate into
        the encoder; only discriminator parameters move.
        """
        self.encoder.eval()  # no BN stat drift in this forward
        with torch.no_grad():
            reprs = {
                p: self.encoder(batch["features"], self._purpose_idx(p))
                for p in self.purpose_names
            }
        self.encoder.train()
        self._freeze_backbone_bn()

        self.disc_optimizer.zero_grad(set_to_none=True)
        total = batch["features"].new_zeros(())
        for purpose_name, attr_name in self.pair_keys:
            z = reprs[purpose_name]
            attr = batch["sensitive_attrs"][attr_name].long()
            pkey = _pair_key(purpose_name, attr_name)
            disc = self.discriminators[pkey]
            total = total + F.cross_entropy(disc(z), attr)
        total.backward()
        self.disc_optimizer.step()
        return float(total.detach().item())

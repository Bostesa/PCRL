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

import numpy as np
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


def _linear_r2_train(H, Z, reg: float = 1e-6) -> float:
    """Train-set linear R² of optimal Tikhonov-regularised one-hot predictor.

    Mirrors ``rlace_diagnostic._linear_r2`` and the auditor's metric
    (LinearComplianceCertificate). Used for LEACE warm-start diagnostics.
    """
    Z = Z.astype("int64") if hasattr(Z, "astype") else Z
    n_classes = int(Z.max()) + 1
    Z_oh = np.eye(n_classes)[Z]
    n, d = H.shape
    H_c = H - H.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + reg * np.eye(d)
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))


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
    freeze_leace_projection: bool = False


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

    # ── training infrastructure (ported from V2Trainer) ──────────────────

    def leace_warm_start(self, train_loader: DataLoader) -> dict[str, dict[str, float]]:
        """Initialise each purpose's last-Linear LoRA from a closed-form LEACE
        eraser on (backbone_features, concatenated-one-hot disallowed_attrs).

        Per purpose:
          1. Run the frozen backbone on the full train loader → features Z (N, d).
             (No LoRA; the frozen-backbone output is the same for every purpose
             at this stage since adapters are still zero.)
          2. Stack disallowed_attrs as a concatenated one-hot concept matrix
             A_oh (N, sum_a c_a). LEACE Theorem 4.2 — the affine eraser
             driving Σ_{Pz, Z} = 0 covers all linearly independent directions
             of Z simultaneously.
          3. Fit ``LeaceEraser`` to (Z, A_oh) and extract the affine map
             e(z) = Q z + (I − Q) μ.
          4. Call ``encoder.init_last_layer_from_affine(p, Q, (I-Q) μ)`` —
             rank-r SVD of (Q − I) W_repr factors into the LoRA's A, B; bias
             absorbs the translation.

        Returns per-purpose diagnostics: pre-init train R² and post-init
        train R² for each (p, a) pair.
        """
        from concept_erasure import LeaceEraser

        # Move backbone to eval mode for clean BN statistics.
        was_training = self.encoder.training
        self.encoder.eval()
        # Pre-init: zero the last-layer LoRA so backbone(x) is unaffected.
        # (At construction the LoRAs are zero-init; this is just defensive.)
        last_idx = len(self.encoder._linear_modules) - 1
        for p_idx in range(self.encoder.n_purposes):
            adapter = self.encoder.adapters[p_idx][last_idx]
            adapter.B.weight.zero_()
            adapter.bias.zero_()

        # Collect features + every disallowed attr across the loader.
        # Use purpose 0's path; with zeroed adapters every purpose gives the
        # same features (= raw backbone).
        feats: list[torch.Tensor] = []
        attrs_collect: dict[str, list[torch.Tensor]] = {}
        for batch in train_loader:
            batch = self._to_device(batch)
            z = self.encoder(batch["features"], 0)
            feats.append(z.detach().cpu())
            for k, v in batch["sensitive_attrs"].items():
                attrs_collect.setdefault(k, []).append(v.detach().cpu().long())

        Z = torch.cat(feats, dim=0)  # (N, d)
        attrs = {k: torch.cat(v, dim=0) for k, v in attrs_collect.items()}

        diagnostics: dict[str, dict[str, float]] = {}
        eye_d = torch.eye(Z.shape[1])

        for purpose_name in self.purpose_names:
            p_idx = self._purpose_idx(purpose_name)
            disallowed = self.purpose_configs[purpose_name]["disallowed_attrs"]

            # Build concatenated one-hot concept matrix.
            oh_blocks: list[torch.Tensor] = []
            for a_name in disallowed:
                a_int = attrs[a_name]
                n_classes = int(a_int.max().item()) + 1
                oh = torch.eye(n_classes)[a_int]  # (N, c_a)
                oh_blocks.append(oh)
            A_oh = torch.cat(oh_blocks, dim=1).float()

            eraser = LeaceEraser.fit(Z.float(), A_oh)
            Q = eraser.P.detach()  # (d, d)
            mu = (
                eraser.bias.detach()
                if eraser.bias is not None
                else torch.zeros(Z.shape[1])
            )
            c = (eye_d - Q) @ mu

            # Pre-init R² per pair (on train set, current backbone).
            pre = {a: _linear_r2_train(Z.numpy(), attrs[a].numpy()) for a in disallowed}

            # Apply LEACE-erased Z and report linear R² post-erasure (sanity).
            Z_erased = eraser(Z.float())
            post_closed = {
                a: _linear_r2_train(Z_erased.numpy(), attrs[a].numpy())
                for a in disallowed
            }

            self.encoder.init_last_layer_from_affine(p_idx, Q, c)

            # Frozen LEACE projection (opt-in). Register the concept-subspace
            # projection ``P_sub = I − Q`` and the LEACE shift ``μ`` as
            # non-trainable buffers on the encoder for this purpose. Once set,
            # ``encoder.forward`` applies ``h_proj = h − (h − μ) @ P_sub.T`` at
            # output, which is mathematically identical to ``eraser(h)`` and
            # therefore idempotent w.r.t. the LoRA-side warm-start above
            # (P_sub² = P_sub, so re-projecting an already-erased h is a
            # no-op). When LoRA drifts during training the projection catches
            # the drift on the next forward pass.
            if self.config.freeze_leace_projection:
                P_sub = (eye_d - Q).to(Q.dtype)
                self.encoder.set_leace_projection(p_idx, P_sub, mu)
                logger.info(
                    f"[LAFTR/LEACE] purpose={purpose_name}: registered frozen "
                    f"projection buffer (P_sub shape={tuple(P_sub.shape)}, "
                    f"mu shape={tuple(mu.shape)})"
                )

            diagnostics[purpose_name] = {
                **{f"r2_pre[{a}]": pre[a] for a in disallowed},
                **{f"r2_post_closed[{a}]": post_closed[a] for a in disallowed},
            }
            logger.info(
                f"[LAFTR/LEACE] purpose={purpose_name} attrs={disallowed} "
                f"pre={pre} post_closed={post_closed}"
            )

        # After init, switch back to original mode.
        if was_training:
            self.encoder.train()
            self._freeze_backbone_bn()
        return diagnostics

    def train_epoch(self, loader: DataLoader, apply_constraints: bool = True) -> dict[str, float]:
        self.encoder.train()
        self._freeze_backbone_bn()
        self.task_heads.train()
        for d in self.discriminators.values():
            d.train()

        t0 = time.time()
        sums: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            d_loss = 0.0
            for _ in range(self.config.disc_steps):
                d_loss = self._discriminator_step(batch)
            stats = self._primal_step_components(batch, apply_constraints=apply_constraints)
            stats["disc_loss"] = d_loss
            for k, v in stats.items():
                if isinstance(v, dict):
                    continue
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            self.state["global_step"] = self.state.get("global_step", 0) + 1
        return {
            "primal_loss": sums.get("primal_loss", 0.0) / max(n, 1),
            "task_loss": sums.get("L_task", 0.0) / max(n, 1),
            "adv_loss": sums.get("L_adv", 0.0) / max(n, 1),
            "vicreg_loss": sums.get("L_vicreg", 0.0) / max(n, 1),
            "disc_loss": sums.get("disc_loss", 0.0) / max(n, 1),
            "epoch_time": time.time() - t0,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.encoder.eval()
        self.task_heads.eval()

        t0 = time.time()
        total_task = 0.0
        total_r2 = 0.0
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
                task_name = self.purpose_configs[purpose_name]["allowed_tasks"][0]
                if task_name not in batch["task_labels"]:
                    continue
                head = self.task_heads[purpose_name]
                preds = head(z)
                if isinstance(preds, dict):
                    preds = preds[task_name]
                targets = batch["task_labels"][task_name]
                L_task += float(
                    task_loss(preds, targets, self.purpose_configs[purpose_name]["task_type"]).item()
                )
                pred_classes = preds.argmax(dim=-1)
                task_correct[task_name] = task_correct.get(task_name, 0) + int(
                    (pred_classes == targets).sum().item()
                )
                task_total[task_name] = task_total.get(task_name, 0) + targets.numel()
            total_task += L_task

            # R² values per (p, a). For high-K pairs the per-pair R² reported
            # is max_k per-class OvR R². Low-K pairs report joint multi-output R².
            for purpose_name, attr_name in self.pair_keys:
                z = reprs[purpose_name]
                attr = batch["sensitive_attrs"][attr_name].long()
                pair = (purpose_name, attr_name)
                key = _pair_key(purpose_name, attr_name)
                if int(attr.max().item()) >= 1:
                    if pair in self.high_k_pairs:
                        r2_per_k = self.verifier.forward_per_class(z, attr)
                        K_total = self.high_k_pairs[pair]
                        if r2_per_k.numel() < K_total:
                            pad = torch.zeros(
                                K_total - r2_per_k.numel(), device=self.device,
                            )
                            r2_per_k = torch.cat([r2_per_k, pad])
                        rv = float(r2_per_k.max().item())
                    else:
                        rv = float(self.verifier(z, attr).item())
                else:
                    rv = 0.0
                per_pair_r2_sum[key] = per_pair_r2_sum.get(key, 0.0) + rv
                n_pair_batches[key] = n_pair_batches.get(key, 0) + 1
                total_r2 += rv

            n += 1

        return {
            "task_loss": total_task / max(n, 1),
            "r2_mean": total_r2 / max(n * len(self.pair_keys), 1),
            "task_accuracy": {
                k: task_correct[k] / task_total[k]
                for k in task_correct if task_total.get(k, 0) > 0
            },
            "r2_per_pair": {
                k: v / max(n_pair_batches[k], 1) for k, v in per_pair_r2_sum.items()
            },
            "epoch_time": time.time() - t0,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        """Training loop: warmup → constrained → true Cotter best-iterate.

        ``warmup_epochs`` task-only epochs (constraint Lagrangian + dual ascent
        skipped), then ``epochs`` constrained epochs. After training, the best
        iterate is selected post-hoc by the true Cotter rule (see
        ``_select_cotter_best``) and snapshotted as ``best.pt``.
        """
        threshold = self.config.r2_threshold
        # Fix R2: with LEACE warm-start the LoRA already starts inside the
        # feasible set; the legacy task-only warmup phase erases that
        # initialisation before constraints engage. Skip warmup when
        # leace_init=True unless caller forces warmup_when_leace_init=True.
        if self.config.leace_init and not self.config.warmup_when_leace_init:
            warmup_n = 0
        else:
            warmup_n = self.config.warmup_epochs
        total_epochs = warmup_n + self.config.epochs
        logger.info(
            f"[LAFTR/TRAIN] schedule: warmup_epochs={warmup_n} "
            f"(config.warmup_epochs={self.config.warmup_epochs}, "
            f"leace_init={self.config.leace_init}, "
            f"warmup_when_leace_init={self.config.warmup_when_leace_init}), "
            f"constrained_epochs={self.config.epochs}, "
            f"lambda_min={self.config.lambda_min}"
        )

        if self.config.leace_init:
            logger.info("[LAFTR/TRAIN] running LEACE warm-start")
            self.leace_warm_start(train_loader)

        # Per-epoch records: (epoch, snapshot_state_dicts, val_task_loss,
        #                     r2_per_pair, violation_sum, in_warmup).
        epoch_records: list[
            tuple[int, dict[str, dict[str, torch.Tensor]], float, dict[str, float], float, bool]
        ] = []

        for epoch in range(total_epochs):
            self.state["epoch"] = epoch
            in_warmup = epoch < warmup_n
            apply_constraints = not in_warmup

            tr = self.train_epoch(train_loader, apply_constraints=apply_constraints)
            self._update_history("train", tr)
            logger.info(
                f"[LAFTR/TRAIN] epoch={epoch} {'(warmup)' if in_warmup else ''} "
                f"loss={tr['primal_loss']:.4f} task={tr['task_loss']:.4f} "
                f"vicreg={tr['vicreg_loss']:.4f} r2_mean=n/a "
                f"disc_loss={tr['disc_loss']:.4f} time={tr['epoch_time']:.1f}s"
            )

            if val_loader is not None:
                val = self.evaluate(val_loader)
                self._update_history("val", val)

                r2_per_pair = {k: float(v) for k, v in val["r2_per_pair"].items()}
                violation_sum = sum(max(0.0, v - threshold) for v in r2_per_pair.values())
                self.state["history"].setdefault("val_violation_sum", []).append(violation_sum)
                self.state["history"].setdefault("r2_per_pair_per_epoch", []).append(r2_per_pair)
                self.state["history"].setdefault("warmup_flag_per_epoch", []).append(bool(in_warmup))

                logger.info(
                    f"[LAFTR/VAL]   epoch={epoch} task={val['task_loss']:.4f} "
                    f"viol_sum={violation_sum:.4f} r2_mean={val['r2_mean']:.4f} "
                    f"time={val['epoch_time']:.1f}s"
                )

                snapshot = {
                    "backbone": {k: v.detach().cpu().clone() for k, v in self.encoder.backbone.state_dict().items()},
                    "lora_adapters": {k: v.detach().cpu().clone() for k, v in self.encoder.adapters.state_dict().items()},
                    "task_heads": {k: v.detach().cpu().clone() for k, v in self.task_heads.state_dict().items()},
                }
                epoch_records.append(
                    (epoch, snapshot, float(val["task_loss"]), r2_per_pair, violation_sum, in_warmup)
                )

        # Always save the final iterate.
        self.save_checkpoint("final")

        # ── True Cotter best-iterate selection ──────────────────────────
        sel = self._select_cotter_best(epoch_records, threshold)
        if sel is not None:
            sel_epoch, snap, sel_loss, sel_r2, sel_viol, sel_kind = sel
            # Restore the selected weights into the live model and save as best.pt.
            self.encoder.backbone.load_state_dict(snap["backbone"])
            self.encoder.adapters.load_state_dict(snap["lora_adapters"])
            self.task_heads.load_state_dict(snap["task_heads"])
            self.state["best_epoch"] = sel_epoch
            self.state["best_val_loss"] = sel_loss  # task_loss at best, not composite
            self.save_checkpoint("best")

            n_feasible = sum(
                1 for r in epoch_records
                if not r[5] and all(v < threshold for v in r[3].values())
            )
            n_eligible = sum(1 for r in epoch_records if not r[5])
            logger.info(
                f"Cotter selection [{sel_kind}]: epoch={sel_epoch} "
                f"task_loss={sel_loss:.4f} viol_sum={sel_viol:.4f} "
                f"(n_feasible={n_feasible}/{n_eligible} post-warmup iterates)"
            )
            self.state["history"].setdefault("cotter_selection", []).append(
                {
                    "epoch": sel_epoch,
                    "task_loss": sel_loss,
                    "violation_sum": sel_viol,
                    "kind": sel_kind,
                    "n_feasible_post_warmup": n_feasible,
                    "n_eligible_post_warmup": n_eligible,
                }
            )
        else:
            logger.warning("Cotter selection: no records to select from")

        # Second-pass selector (opt-in via ``report_best_iterate``). Writes
        # ``canonical_iterate.pt`` containing whichever of (best.pt, final.pt)
        # has the lower mean R² across all (purpose, attribute) pairs on the
        # validation set. ``best.pt`` is the Cotter-selected snapshot
        # currently loaded into the live model; ``final.pt`` is the last
        # iterate. Adult/HMDA/Diabetes runs leave this flag False so no
        # canonical_iterate.pt is created, preserving existing behavior.
        if self.config.report_best_iterate and epoch_records and sel is not None:
            final_r2 = epoch_records[-1][3]
            best_r2 = sel[3]
            mean_final = (
                sum(final_r2.values()) / max(len(final_r2), 1) if final_r2 else float("inf")
            )
            mean_best = (
                sum(best_r2.values()) / max(len(best_r2), 1) if best_r2 else float("inf")
            )
            if mean_final < mean_best:
                # Restore final snapshot, save as canonical_iterate.pt, then
                # restore best snapshot back into the live model so downstream
                # callers see the Cotter-selected weights as before.
                final_snap = epoch_records[-1][1]
                self.encoder.backbone.load_state_dict(final_snap["backbone"])
                self.encoder.adapters.load_state_dict(final_snap["lora_adapters"])
                self.task_heads.load_state_dict(final_snap["task_heads"])
                self.save_checkpoint("canonical_iterate")
                # Restore best snapshot.
                best_snap = sel[1]
                self.encoder.backbone.load_state_dict(best_snap["backbone"])
                self.encoder.adapters.load_state_dict(best_snap["lora_adapters"])
                self.task_heads.load_state_dict(best_snap["task_heads"])
                logger.info(
                    f"[LAFTR/CANONICAL] selected final.pt "
                    f"(mean_R²_final={mean_final:.4f} < mean_R²_best={mean_best:.4f})"
                )
            else:
                # Live model is already best; just save under the new name.
                self.save_checkpoint("canonical_iterate")
                logger.info(
                    f"[LAFTR/CANONICAL] selected best.pt "
                    f"(mean_R²_best={mean_best:.4f} <= mean_R²_final={mean_final:.4f})"
                )

        return self.state

    # ── Cotter best-iterate selection ───────────────────────────────────

    def _select_cotter_best(
        self, records, threshold: float,
    ):
        """Cotter et al. JMLR 2019 §4.6 best-iterate selection.

        Among post-warmup iterates:
          1. If any iterate is fully feasible (every R² < threshold), return
             the one with min val_task_loss.
          2. Otherwise, among iterates whose task_loss is within
             ``cotter_fallback_task_slack`` (default 10%) of the best task_loss,
             return the one with smallest sum-of-violations.
        Returns ``(epoch, snapshot, task_loss, r2_per_pair, viol_sum, kind)``
        with ``kind`` in {"feasible", "fallback"}, or None if no records exist.
        """
        if not records:
            return None
        eligible = [r for r in records if not r[5]]  # exclude warmup epochs
        if not eligible:
            eligible = list(records)

        feasible = [r for r in eligible if all(v < threshold for v in r[3].values())]
        if feasible:
            b = min(feasible, key=lambda r: r[2])
            # records are (epoch, snap, loss, r2, viol, in_warmup); drop in_warmup
            return (b[0], b[1], b[2], b[3], b[4], "feasible")

        slack = self.config.cotter_fallback_task_slack
        best_task = min(r[2] for r in eligible)
        # Treat best_task >= 0 (cross-entropy is non-negative); 1+slack scaling.
        near_optimal = [r for r in eligible if r[2] <= best_task * (1.0 + slack)]
        if near_optimal:
            b = min(near_optimal, key=lambda r: r[4])
            return (b[0], b[1], b[2], b[3], b[4], "fallback")
        return None

    # ── checkpointing + history ─────────────────────────────────────────

    def _update_history(self, prefix: str, metrics: dict[str, Any]) -> None:
        for key, val in (
            ("primal_loss", metrics.get("primal_loss", 0.0)),
            ("task_loss", metrics.get("task_loss", 0.0)),
            ("vicreg_loss", metrics.get("vicreg_loss", 0.0)),
            ("r2_mean", metrics.get("r2_mean", 0.0)),
            ("disc_loss", metrics.get("disc_loss", 0.0)),
            ("adv_loss", metrics.get("adv_loss", 0.0)),
        ):
            self.state["history"].setdefault(f"{prefix}_{key}", []).append(val)

    def save_checkpoint(self, name: str) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Encoder top-level buffers that don't live under ``backbone`` or
        # ``adapters`` (e.g. ``leace_P_p{p}``, ``leace_mu_p{p}`` registered by
        # ``set_leace_projection``). For checkpoints written before the
        # frozen-projection feature this dict is empty and the key is harmless
        # backward-compat.
        encoder_full = self.encoder.state_dict()
        backbone_keys = set(self.encoder.backbone.state_dict().keys())
        adapter_keys = set(self.encoder.adapters.state_dict().keys())
        encoder_buffers = {
            k: v for k, v in encoder_full.items()
            if k.split(".", 1)[0] not in {"backbone", "adapters"}
            and k not in backbone_keys
            and k not in adapter_keys
        }
        ckpt = {
            "backbone": self.encoder.backbone.state_dict(),
            "lora_adapters": self.encoder.adapters.state_dict(),
            "encoder_buffers": encoder_buffers,
            "task_heads": self.task_heads.state_dict(),
            "lambdas": {n: c.lambda_value for n, c in self.proxy.constraints.items()},
            "config": vars(self.config),
            "state": {
                "epoch": self.state["epoch"],
                "global_step": self.state.get("global_step", 0),
                "best_val_loss": self.state.get("best_val_loss", float("inf")),
                "best_epoch": self.state.get("best_epoch", -1),
            },
            "history": self.state["history"],
        }
        path = ckpt_dir / f"{name}.pt"
        torch.save(ckpt, path)
        return path

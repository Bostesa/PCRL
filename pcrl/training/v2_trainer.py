"""V2 trainer: LoRA + linear-R² constraint + vCLUB + VICReg + proxy-Lagrangian.

Replaces the adversarial / minimax loop in ``trainer.py`` with a
non-adversarial constrained optimisation loop. The primary independence
constraint is the auditor's metric — linear R² of the optimal predictor
of attr from z — so the optimiser and the audit cannot disagree.

    Primal step (per batch):
        z_p   = encoder(x, p)               # frozen backbone + per-purpose LoRA
        L_task     = sum_p CE(task_head_p(z_p), y_p)
        L_vicreg   = sum_p vicreg_loss(z_p)
        L_vclub    = sum_{p,a} vclub_{p,a}.mi_upper_bound(z_p, attr_a)
        primal_loss = L_task
                    + lambda_vicreg   * L_vicreg
                    + lambda_vclub    * L_vclub
                    + sum_{p,a} lambda_{p,a} * (R²(z_p, attr_a) - tau)
        backward; step (LoRA adapters + task head params)

    vCLUB q-net step (per batch, before primal):
        for each (p, a): minimise q-net's learning_loss(z_p.detach(), attr_a)
        (separate optimiser; X is detached so encoder isn't updated)

    Dual step (per batch, after primal):
        for each (p, a): lambda_{p,a} <- proj([0, lam_max],
                                              lambda + eta * (R² - tau))

Best-iterate selection (true Cotter, Cotter et al. JMLR 2019 §4.6):
during training, snapshot every epoch's state in memory along with its
val_task_loss and per-pair R²s. After training:

    1. Among iterates after warmup that are FULLY FEASIBLE (every R² < tau),
       return the one with min val_task_loss.
    2. If no iterate is feasible, among iterates with val_task_loss within
       10% of the best val_task_loss, return the one with smallest
       sum_{p,a} max(0, R² - tau).

The Round 1 composite `task + 0.5 * Σ violation` rewarded epochs 3-5
where task was low but violation was still ~2.5 — exactly the wrong
regime (lambdas hadn't ramped yet).

Round 2 schedule: ``warmup_epochs`` task-only epochs (constraint
Lagrangian and dual updates skipped) followed by ``epochs`` constrained
epochs. Lambda damping (``lr_lambda=0.005``, cap 1000) prevents the
saturate-and-oscillate failure mode that Round 1 exhibited.

HSIC was previously kept as a fixed-weight differentiable auxiliary
(``lambda_hsic_aux=0.1``). The R-LACE/LEACE diagnostic (commit f8d1966)
showed the linear-R² constraint is feasible on the frozen backbone with
LEACE driving R² to 0.005 on 7/8 pairs. Under median bandwidth HSIC adds
a competing nonlinear gradient that crowds out the R² signal; default is
now ``lambda_hsic_aux=0.0``. The HSIC value is still computed for logging.

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
    lr_lambda: float = 0.02  # Round 4: intermediate between R1's 0.05
                             # (saturated at cap) and R2's 0.005 (under-engaged).
                             # Round 3 probe + Fix 1 (LEACE init) needed
                             # enough dual authority to keep R² descending.

    # Loss weights (fixed scalarisation for terms that aren't constraints)
    lambda_vicreg: float = 1.0  # Kept at full weight: VICReg variance term
                                # protects per-dim std from collapsing under
                                # sustained constraint pressure.
    lambda_vclub: float = 0.0   # Round 4: vCLUB disabled in constrained phase.
                                # Fix 2 probe showed vCLUB MI bound did not
                                # contribute usefully to R² descent and
                                # competed with the proxy-Lagrangian term.
                                # vCLUB q-net still trains; estimate is logged
                                # but contributes 0 to the primal loss.
    lambda_verify: float = 0.0  # Legacy; R² is now the constraint, not a fixed-weight term.
    lambda_hsic_aux: float = 0.0  # Round 1 fix: HSIC under median bandwidth was
                                  # redundant with the linear-R² constraint and
                                  # fought the dual. Set to 0.0; HSIC still logged.

    # Linear-R² constraint (proxy-Lagrangian primary; matches auditor's metric).
    lambda_hsic_init: float = 1.0  # Round 4: start duals with some authority.
                                   # Round 2's 0.0 + low lr_lambda left duals
                                   # stuck at single-digit values throughout.
    r2_threshold: float = 0.05
    r2_lambda_max: float = 1000.0  # Round 2: raised from 100; gives the dual room
                                   # to apply more pressure before saturating.

    # Cotter best-iterate selection (Cotter et al. JMLR 2019 §4.6).
    # Round 2 uses true Cotter: best-feasible by min task_loss; fallback by
    # min violation among iterates within 10% of best task_loss.
    cotter_fallback_task_slack: float = 0.10  # 10% slack on task_loss for fallback
    composite_violation_weight: float = 0.5  # legacy (Round 1); unused by true Cotter

    # Fix 1: closed-form LEACE warm-start of the per-purpose LoRA adapters
    # (Belrose et al. 2023). For each purpose, fits a LEACE eraser on
    # (backbone_features, concatenated-one-hot disallowed_attrs) and
    # initialises the last-Linear LoRA so that the encoder output at
    # epoch 0 already approximately satisfies R²(z, A) ≈ 0. Closed-form
    # diagnostic showed LEACE drives R² to ~0.005 on 7/8 Adult pairs, and
    # the gradient optimiser couldn't reach this region from a zero-init
    # LoRA. Starting inside the feasible set (or close to it) means the
    # dual variables stay small and don't have to fight task gradient.
    leace_init: bool = True

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
    epochs: int = 100  # Constrained epochs (warmup_epochs are added on top).
    warmup_epochs: int = 5  # Round 4: was 20; Round 2 showed K=20 lets task
                            # overfit to ~100% during warmup, locking the rep
                            # into task-optimal before constraints engage.
    early_stopping_patience: int | None = None  # Deprecated; Cotter runs full schedule.
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
    # best_val_loss now tracks the Cotter best-iterate composite
    # (val_task_loss + w * Σ max(0, R² - τ)), not raw val_task_loss.
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    patience_counter: int = 0  # Deprecated; retained for checkpoint backward compat.
    history: dict[str, list[float]] = field(default_factory=dict)


def _linear_r2_train(H, Z, reg: float = 1e-6) -> float:
    """Train-set linear R² of optimal Tikhonov-regularised one-hot predictor.

    Mirrors ``rlace_diagnostic._linear_r2`` and the auditor's metric
    (LinearComplianceCertificate). Used for LEACE warm-start diagnostics.
    """
    import numpy as np

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
        # Also freeze BN running statistics. ``requires_grad_(False)`` only
        # locks Parameters; BatchNorm's ``running_mean`` / ``running_var``
        # are Buffers and still update in ``train()`` mode, drifting the
        # backbone away from its initialised distribution and silently
        # invalidating any closed-form structure (e.g. a LEACE warm-start)
        # applied at construction time. We force every BN module in the
        # backbone to ``eval()`` permanently — the encoder's ``train()``
        # toggle no longer flips them back. LoRA adapters and task heads
        # still respond to ``train()`` for their own dropout layers.
        self._freeze_backbone_bn()

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

    def _freeze_backbone_bn(self) -> None:
        """Force every BatchNorm in the backbone into ``eval()``. Called once
        at construction and re-asserted after every ``encoder.train()``."""
        for m in self.encoder.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    # ── LEACE warm-start ────────────────────────────────────────────────

    @torch.no_grad()
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

            diagnostics[purpose_name] = {
                **{f"r2_pre[{a}]": pre[a] for a in disallowed},
                **{f"r2_post_closed[{a}]": post_closed[a] for a in disallowed},
            }
            logger.info(
                f"[V2/LEACE] purpose={purpose_name} attrs={disallowed} "
                f"pre={pre} post_closed={post_closed}"
            )

        # After init, switch back to original mode.
        if was_training:
            self.encoder.train()
            self._freeze_backbone_bn()
        return diagnostics

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

    def _primal_and_dual_step(
        self, batch: dict[str, Any], apply_constraints: bool = True,
    ) -> dict[str, float]:
        """One primal step (LoRA + heads) + one dual ascent on lambdas.

        If apply_constraints is False (warmup), the Lagrangian term and the
        dual-ascent step are skipped — the primal loss is task + VICReg only.
        Lambdas remain at their initial values.
        """
        self.encoder.train()
        self._freeze_backbone_bn()
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
        if apply_constraints:
            base = (
                L_task
                + self.config.lambda_vicreg * L_vicreg
                + self.config.lambda_vclub * L_vclub
                + self.config.lambda_hsic_aux * L_hsic_aux
                + self.config.lambda_verify * L_verify  # legacy; default 0
            )
            primal_loss = self.proxy.lagrangian_loss(base, constraint_values)
        else:
            # Warmup: task + VICReg only. vCLUB / HSIC / Lagrangian skipped so
            # the LoRA can fit the task before constraint pressure starts.
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

        # ── Dual step ───────────────────────────────────────────────────
        if apply_constraints:
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

    def train_epoch(self, loader: DataLoader, apply_constraints: bool = True) -> V2EpochMetrics:
        self.encoder.train()
        self._freeze_backbone_bn()
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

            stats = self._primal_and_dual_step(batch, apply_constraints=apply_constraints)
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
        """Round 2 training loop: warmup → constrained → true Cotter best-iterate.

        ``warmup_epochs`` task-only epochs (constraint Lagrangian + dual ascent
        skipped), then ``epochs`` constrained epochs. After training, the best
        iterate is selected post-hoc by the true Cotter rule (see
        ``_select_cotter_best``) and snapshotted as ``best.pt``.
        """
        threshold = self.config.r2_threshold
        warmup_n = self.config.warmup_epochs
        total_epochs = warmup_n + self.config.epochs

        # Per-epoch records: (epoch, snapshot_state_dicts, val_task_loss,
        #                     r2_per_pair, violation_sum, in_warmup).
        # Snapshots live in CPU memory; one snapshot ≈ 1 MB on Adult, so 220
        # snapshots is ~220 MB per seed (well under instance RAM).
        epoch_records: list[
            tuple[int, dict[str, dict[str, torch.Tensor]], float, dict[str, float], float, bool]
        ] = []

        for epoch in range(total_epochs):
            self.state.epoch = epoch
            in_warmup = epoch < warmup_n
            apply_constraints = not in_warmup

            tr = self.train_epoch(train_loader, apply_constraints=apply_constraints)
            self._update_history("train", tr)
            logger.info(
                f"[V2/TRAIN] epoch={epoch} {'(warmup)' if in_warmup else ''} "
                f"loss={tr.loss:.4f} task={tr.task_loss:.4f} "
                f"vicreg={tr.vicreg_loss:.4f} r2_mean={tr.r2_mean:.4f} "
                f"hsic_mean={tr.hsic_mean:.4f} q_loss={tr.vclub_q_loss:.4f} time={tr.epoch_time:.1f}s"
            )

            if val_loader is not None:
                val = self.evaluate(val_loader)
                self._update_history("val", val)

                r2_per_pair = {k: float(v) for k, v in val.r2_per_pair.items()}
                violation_sum = sum(max(0.0, v - threshold) for v in r2_per_pair.values())
                self.state.history.setdefault("val_violation_sum", []).append(violation_sum)
                self.state.history.setdefault("r2_per_pair_per_epoch", []).append(r2_per_pair)
                self.state.history.setdefault("warmup_flag_per_epoch", []).append(bool(in_warmup))

                logger.info(
                    f"[V2/VAL]   epoch={epoch} task={val.task_loss:.4f} "
                    f"viol_sum={violation_sum:.4f} r2_mean={val.r2_mean:.4f} "
                    f"hsic_mean={val.hsic_mean:.4f} time={val.epoch_time:.1f}s"
                )

                snapshot = {
                    "backbone": {k: v.detach().cpu().clone() for k, v in self.encoder.backbone.state_dict().items()},
                    "lora_adapters": {k: v.detach().cpu().clone() for k, v in self.encoder.adapters.state_dict().items()},
                    "task_heads": {k: v.detach().cpu().clone() for k, v in self.task_heads.state_dict().items()},
                }
                epoch_records.append(
                    (epoch, snapshot, float(val.task_loss), r2_per_pair, violation_sum, in_warmup)
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
            self.state.best_epoch = sel_epoch
            self.state.best_val_loss = sel_loss  # task_loss at best, not composite
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
            self.state.history.setdefault("cotter_selection", []).append(
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
                "best_epoch": self.state.best_epoch,
            },
            "history": self.state.history,
        }
        path = ckpt_dir / f"{name}.pt"
        torch.save(ckpt, path)
        return path

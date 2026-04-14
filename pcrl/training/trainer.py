"""Main training loop for PCRL."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.losses import (
    LossBreakdown,
    VerificationRegularizer,
    adversarial_loss,
    compute_multi_auditor_loss,
    compute_multi_task_loss,
    task_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for PCRLTrainer.

    Attributes:
        batch_size: Number of samples per batch.
        lr_encoder: Learning rate for encoder and task heads.
        lr_auditor: Learning rate for auditors.
        lambda_adv: Weight for adversarial loss.
        auditor_steps: Number of auditor updates per encoder update (K).
        epochs: Total number of training epochs.
        weight_decay: L2 regularization weight.
        grad_clip: Maximum gradient norm (None to disable).
        early_stopping_patience: Epochs without improvement before stopping.
        checkpoint_dir: Directory for saving checkpoints.
        log_interval: Steps between logging.
        purpose_sampling: How to sample purposes ("all", "alternating", "random").
        confusion_type: Type of confusion loss ("negative_ce", "entropy", "uniform_kl").
    """
    batch_size: int = 256
    lr_encoder: float = 1e-3
    lr_auditor: float = 1e-3
    lambda_adv: float = 1.0
    auditor_steps: int = 5
    epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    early_stopping_patience: int | None = 10
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    lambda_verify: float = 0.0
    lambda_comp: float = 0.0
    purpose_sampling: Literal["all", "alternating", "random"] = "all"
    confusion_type: Literal["negative_ce", "entropy", "uniform_kl"] = "negative_ce"
    gradient_reversal: bool = False
    lambda_adv_per_attr: dict[str, float] | None = None
    lambda_verify_per_attr: dict[str, float] | None = None
    lambda_adv_per_purpose: dict[str, float] | None = None
    sequential_purposes: bool = False
    warmup_epochs: int = 0


@dataclass
class TrainingState:
    """State of training progress.

    Attributes:
        epoch: Current epoch.
        global_step: Total training steps.
        best_val_loss: Best validation loss seen.
        patience_counter: Epochs since last improvement.
        history: Training history metrics.
    """
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    """Metrics from a training/evaluation epoch.

    Attributes:
        loss: Total loss.
        task_loss: Task prediction loss.
        adversarial_loss: Adversarial loss.
        task_accuracy: Per-task accuracy.
        auditor_accuracy: Per-attribute auditor accuracy.
        epoch_time: Time taken for epoch in seconds.
    """
    loss: float
    task_loss: float
    adversarial_loss: float
    task_accuracy: dict[str, float] = field(default_factory=dict)
    auditor_accuracy: dict[str, float] = field(default_factory=dict)
    verification_loss: float = 0.0
    epoch_time: float = 0.0


class PCRLTrainer:
    """Trainer for Purpose-Conditioned Representation Learning.

    Implements the adversarial training procedure:
    1. Get representations: h_p = encoder(x, purpose_idx)
    2. Update auditors (K steps): minimize CE(auditor(h_p.detach()), true_attr)
    3. Update encoder + task heads (1 step):
       - L_task = task_loss(task_head(h_p), y)
       - L_adv = confusion_loss(auditor(h_p), true_attr)  # maximize auditor error
       - L_total = L_task + lambda * L_adv
    """

    def __init__(
        self,
        encoder: nn.Module,
        task_heads: dict[str, nn.Module],
        auditors: dict[str, nn.Module],
        config: TrainerConfig,
        purpose_configs: dict[str, dict[str, Any]] | None = None,
        purpose_registry: PurposeRegistry | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the trainer.

        Args:
            encoder: Purpose-conditioned encoder. Should have forward(x, purpose_idx).
            task_heads: Dictionary mapping purpose_name to task head module.
            auditors: Dictionary mapping purpose_name to auditor module.
            config: Training configuration.
            purpose_configs: Optional dict mapping purpose_name to config with:
                - "purpose_idx": int
                - "task_type": str ("classification" or "regression")
                - "disallowed_attrs": list[str]
            purpose_registry: Optional PurposeRegistry. If provided, purpose_configs
                will be auto-built from the registry.
            device: Device to train on.
        """
        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)
        self.auditors = nn.ModuleDict(auditors)
        self.config = config
        self.purpose_configs = purpose_configs or {}
        self.device = torch.device(device)

        # Build purpose_configs from registry if provided
        if purpose_registry is not None:
            for idx, purpose in enumerate(purpose_registry.purposes):
                self.purpose_configs[purpose.name] = {
                    "purpose_idx": idx,
                    "task_type": purpose.task_type,
                    "allowed_tasks": purpose.allowed_tasks,
                    "disallowed_attrs": purpose.disallowed_attrs,
                }

        # Verification regularizer
        self.verify_reg: VerificationRegularizer | None = None
        if self.config.lambda_verify > 0 or self.config.lambda_verify_per_attr:
            self.verify_reg = VerificationRegularizer()

        # Move to device
        self.encoder.to(self.device)
        self.task_heads.to(self.device)
        self.auditors.to(self.device)

        # Optimizers
        self.encoder_optimizer = self._create_encoder_optimizer()
        self.auditor_optimizer = self._create_auditor_optimizer()

        # Schedulers (will be set when train_loader is available)
        self.encoder_scheduler: LRScheduler | None = None
        self.auditor_scheduler: LRScheduler | None = None

        # Training state
        self.state = TrainingState()

        # Purpose indices for iteration
        self.purpose_names = list(task_heads.keys())

    def _create_encoder_optimizer(self) -> Optimizer:
        """Create optimizer for encoder and task heads."""
        params = list(self.encoder.parameters())
        for head in self.task_heads.values():
            params.extend(head.parameters())

        return AdamW(
            params,
            lr=self.config.lr_encoder,
            weight_decay=self.config.weight_decay,
        )

    def _create_auditor_optimizer(self) -> Optimizer:
        """Create optimizer for auditors."""
        params = []
        for auditor in self.auditors.values():
            params.extend(auditor.parameters())

        return AdamW(
            params,
            lr=self.config.lr_auditor,
            weight_decay=self.config.weight_decay,
        )

    def _setup_schedulers(self, total_steps: int, steps_per_epoch: int = 0) -> None:
        """Set up learning rate schedulers with optional warmup."""
        if self.config.warmup_epochs > 0 and steps_per_epoch > 0:
            warmup_steps = self.config.warmup_epochs * steps_per_epoch
            from torch.optim.lr_scheduler import LambdaLR, SequentialLR

            def warmup_fn(step):
                return min(1.0, (step + 1) / max(warmup_steps, 1))

            enc_warmup = LambdaLR(self.encoder_optimizer, warmup_fn)
            enc_cosine = CosineAnnealingLR(
                self.encoder_optimizer, T_max=max(total_steps - warmup_steps, 1),
            )
            self.encoder_scheduler = SequentialLR(
                self.encoder_optimizer,
                schedulers=[enc_warmup, enc_cosine],
                milestones=[warmup_steps],
            )

            aud_warmup = LambdaLR(self.auditor_optimizer, warmup_fn)
            aud_cosine = CosineAnnealingLR(
                self.auditor_optimizer, T_max=max(total_steps - warmup_steps, 1),
            )
            self.auditor_scheduler = SequentialLR(
                self.auditor_optimizer,
                schedulers=[aud_warmup, aud_cosine],
                milestones=[warmup_steps],
            )
        else:
            self.encoder_scheduler = CosineAnnealingLR(
                self.encoder_optimizer,
                T_max=total_steps,
            )
            self.auditor_scheduler = CosineAnnealingLR(
                self.auditor_optimizer,
                T_max=total_steps,
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> TrainingState:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            Final training state.
        """
        steps_per_epoch = len(train_loader)
        if self.config.sequential_purposes:
            steps_per_epoch = len(train_loader) * len(self.purpose_names)
        total_steps = steps_per_epoch * self.config.epochs
        self._setup_schedulers(total_steps, steps_per_epoch)

        logger.info("=" * 60)
        logger.info("Starting PCRL Training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Lambda adversarial: {self.config.lambda_adv}")
        logger.info(f"Auditor steps per encoder step: {self.config.auditor_steps}")
        logger.info(f"Purposes: {self.purpose_names}")
        logger.info("=" * 60)

        epoch_pbar = tqdm(range(self.config.epochs), desc="Training")
        for epoch in epoch_pbar:
            self.state.epoch = epoch

            # Training epoch
            if self.config.sequential_purposes:
                train_metrics = self.train_epoch_sequential(train_loader)
            else:
                train_metrics = self.train_epoch(train_loader)
            self._log_epoch_metrics("TRAIN", train_metrics)
            self._update_history("train", train_metrics)

            # Update progress bar
            epoch_pbar.set_postfix(
                loss=f"{train_metrics.loss:.4f}",
                task=f"{train_metrics.task_loss:.4f}",
                adv=f"{train_metrics.adversarial_loss:.4f}",
            )

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self._log_epoch_metrics("VAL", val_metrics)
                self._update_history("val", val_metrics)

                # Early stopping check
                if val_metrics.loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics.loss
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

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        self.save_checkpoint("final")
        logger.info("Training complete!")

        return self.state

    def train_epoch(self, train_loader: DataLoader) -> EpochMetrics:
        """Run one training epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Aggregated training metrics.
        """
        self.encoder.train()
        self.task_heads.train()
        self.auditors.train()

        start_time = time.time()

        total_loss = 0.0
        total_task_loss = 0.0
        total_adv_loss = 0.0
        total_verify_loss = 0.0
        num_batches = 0

        task_correct: dict[str, int] = {}
        task_total: dict[str, int] = {}
        auditor_correct: dict[str, int] = {}
        auditor_total: dict[str, int] = {}

        batch_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {self.state.epoch}",
            leave=False,
        )
        for batch_idx, batch in batch_pbar:
            batch = self._to_device(batch)

            # Select purposes for this batch
            purposes = self._select_purposes(batch_idx)

            if self.config.gradient_reversal:
                # GRL mode: joint training step
                loss, task_l, adv_l, verify_l = self._train_grl_step(
                    batch, purposes
                )
                # Get representations for accuracy tracking
                with torch.no_grad():
                    representations = {}
                    for purpose_name in purposes:
                        purpose_idx = self._get_purpose_idx(purpose_name)
                        representations[purpose_name] = self.encoder(
                            batch["features"], purpose_idx
                        )
            else:
                # Minimax mode: separate auditor and encoder steps
                # Compute representations once; auditor uses detached copies,
                # encoder step reuses the same tensors (with live gradients).
                representations = {}
                for purpose_name in purposes:
                    purpose_idx = self._get_purpose_idx(purpose_name)
                    h_p = self.encoder(batch["features"], purpose_idx)
                    representations[purpose_name] = h_p

                for _ in range(self.config.auditor_steps):
                    self._train_auditor_step(representations, batch)

                loss, task_l, adv_l, verify_l = self._train_encoder_step(
                    batch, purposes
                )

            total_loss += loss
            total_task_loss += task_l
            total_adv_loss += adv_l
            total_verify_loss += verify_l
            num_batches += 1

            batch_pbar.set_postfix(
                loss=f"{loss:.4f}", task=f"{task_l:.4f}", adv=f"{adv_l:.4f}"
            )

            # Update accuracy stats
            with torch.no_grad():
                self._update_accuracy_stats(
                    representations, batch, purposes,
                    task_correct, task_total, auditor_correct, auditor_total
                )

            self.state.global_step += 1

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {self.state.epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss:.4f} Task: {task_l:.4f} Adv: {adv_l:.4f} Verify: {verify_l:.4f}"
                )

        epoch_time = time.time() - start_time

        return EpochMetrics(
            loss=total_loss / num_batches,
            task_loss=total_task_loss / num_batches,
            adversarial_loss=total_adv_loss / num_batches,
            task_accuracy={k: task_correct[k] / task_total[k] for k in task_correct if task_total[k] > 0},
            auditor_accuracy={k: auditor_correct[k] / auditor_total[k] for k in auditor_correct if auditor_total[k] > 0},
            verification_loss=total_verify_loss / num_batches if num_batches > 0 else 0.0,
            epoch_time=epoch_time,
        )

    def train_epoch_sequential(self, train_loader: DataLoader) -> EpochMetrics:
        """Run one training epoch with sequential purpose training.

        For each epoch:
        1. Shuffle purpose order
        2. For each purpose, do a full pass over all batches
        3. Each purpose updates only its own task head, auditors, and the shared encoder

        This avoids conflicting gradients from different purposes fighting
        within a single batch.
        """
        import random

        self.encoder.train()
        self.task_heads.train()
        self.auditors.train()

        start_time = time.time()

        total_loss = 0.0
        total_task_loss = 0.0
        total_adv_loss = 0.0
        total_verify_loss = 0.0
        num_batches = 0

        task_correct: dict[str, int] = {}
        task_total: dict[str, int] = {}
        auditor_correct: dict[str, int] = {}
        auditor_total: dict[str, int] = {}

        # Shuffle purpose order each epoch
        purpose_order = list(self.purpose_names)
        random.shuffle(purpose_order)

        for purpose_name in purpose_order:
            purpose_idx = self._get_purpose_idx(purpose_name)
            purpose_lambda = self._get_lambda_adv_for_purpose(purpose_name)

            for batch in train_loader:
                batch = self._to_device(batch)

                if self.config.gradient_reversal:
                    # GRL mode for this single purpose
                    loss, task_l, adv_l, verify_l = self._train_grl_step_single(
                        batch, purpose_name, purpose_idx, purpose_lambda
                    )
                else:
                    # Minimax for single purpose
                    h_p = self.encoder(batch["features"], purpose_idx)
                    reps = {purpose_name: h_p}

                    for _ in range(self.config.auditor_steps):
                        self._train_auditor_step(reps, batch)

                    loss, task_l, adv_l, verify_l = self._train_encoder_step_single(
                        batch, purpose_name, purpose_idx, purpose_lambda
                    )

                total_loss += loss
                total_task_loss += task_l
                total_adv_loss += adv_l
                total_verify_loss += verify_l
                num_batches += 1

                # Accuracy tracking
                with torch.no_grad():
                    h_p = self.encoder(batch["features"], purpose_idx)
                    self._update_accuracy_stats(
                        {purpose_name: h_p}, batch, [purpose_name],
                        task_correct, task_total, auditor_correct, auditor_total,
                    )

                self.state.global_step += 1

        epoch_time = time.time() - start_time

        return EpochMetrics(
            loss=total_loss / max(num_batches, 1),
            task_loss=total_task_loss / max(num_batches, 1),
            adversarial_loss=total_adv_loss / max(num_batches, 1),
            task_accuracy={k: task_correct[k] / task_total[k] for k in task_correct if task_total[k] > 0},
            auditor_accuracy={k: auditor_correct[k] / auditor_total[k] for k in auditor_correct if auditor_total[k] > 0},
            verification_loss=total_verify_loss / max(num_batches, 1),
            epoch_time=epoch_time,
        )

    def _train_grl_step_single(
        self,
        batch: dict[str, Any],
        purpose_name: str,
        purpose_idx: int,
        purpose_lambda: float,
    ) -> tuple[float, float, float, float]:
        """Single GRL training step for one purpose."""
        self.encoder_optimizer.zero_grad()
        self.auditor_optimizer.zero_grad()

        h_p = self.encoder(batch["features"], purpose_idx)

        # Task loss
        total_task_loss = torch.tensor(0.0, device=self.device)
        if purpose_name in self.task_heads:
            task_head = self.task_heads[purpose_name]
            task_preds = task_head(h_p)
            task_type = self._get_task_type(purpose_name)

            if isinstance(task_preds, dict):
                for task_name, preds in task_preds.items():
                    if task_name in batch["task_labels"]:
                        targets = batch["task_labels"][task_name]
                        total_task_loss = total_task_loss + task_loss(preds, targets, task_type)
            else:
                task_name = self._get_primary_task(purpose_name)
                if task_name and task_name in batch["task_labels"]:
                    targets = batch["task_labels"][task_name]
                    total_task_loss = total_task_loss + task_loss(task_preds, targets, task_type)

        # Auditor loss with GRL
        total_adv_loss = torch.tensor(0.0, device=self.device)
        if purpose_name in self.auditors:
            auditor = self.auditors[purpose_name]
            auditor_preds = auditor(h_p, lambda_=purpose_lambda)

            if isinstance(auditor_preds, dict):
                for attr_name, preds in auditor_preds.items():
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        total_adv_loss = total_adv_loss + adversarial_loss(
                            preds, targets, maximize=False
                        )
            else:
                attr_name = self._get_primary_attr(purpose_name)
                if attr_name and attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    total_adv_loss = total_adv_loss + adversarial_loss(
                        auditor_preds, targets, maximize=False
                    )

        # Verification loss
        total_verify_loss = torch.tensor(0.0, device=self.device)
        if self.config.lambda_verify > 0 and self.verify_reg is not None:
            disallowed = self._get_disallowed_attrs(purpose_name)
            for attr_name in disallowed:
                if attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    r_sq = self.verify_reg(h_p, targets)
                    total_verify_loss = total_verify_loss + r_sq

        total_loss = (
            total_task_loss
            + total_adv_loss
            + self.config.lambda_verify * total_verify_loss
        )

        total_loss.backward()
        if self.config.grad_clip is not None:
            all_params = list(self.encoder.parameters())
            for head in self.task_heads.values():
                all_params.extend(head.parameters())
            for aud in self.auditors.values():
                all_params.extend(aud.parameters())
            nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)
        self.encoder_optimizer.step()
        self.auditor_optimizer.step()

        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.auditor_scheduler is not None:
            self.auditor_scheduler.step()

        return (
            total_loss.item(),
            total_task_loss.item(),
            total_adv_loss.item(),
            total_verify_loss.item(),
        )

    def _train_encoder_step_single(
        self,
        batch: dict[str, Any],
        purpose_name: str,
        purpose_idx: int,
        purpose_lambda: float,
    ) -> tuple[float, float, float, float]:
        """Single encoder step for one purpose (minimax mode)."""
        self.encoder_optimizer.zero_grad()

        h_p = self.encoder(batch["features"], purpose_idx)

        # Task loss
        total_task_loss = torch.tensor(0.0, device=self.device)
        if purpose_name in self.task_heads:
            task_head = self.task_heads[purpose_name]
            task_preds = task_head(h_p)
            task_type = self._get_task_type(purpose_name)

            if isinstance(task_preds, dict):
                for task_name, preds in task_preds.items():
                    if task_name in batch["task_labels"]:
                        targets = batch["task_labels"][task_name]
                        total_task_loss = total_task_loss + task_loss(preds, targets, task_type)
            else:
                task_name = self._get_primary_task(purpose_name)
                if task_name and task_name in batch["task_labels"]:
                    targets = batch["task_labels"][task_name]
                    total_task_loss = total_task_loss + task_loss(task_preds, targets, task_type)

        # Adversarial confusion loss
        total_adv_loss = torch.tensor(0.0, device=self.device)
        if purpose_name in self.auditors:
            auditor = self.auditors[purpose_name]
            auditor_preds = auditor(h_p)

            if isinstance(auditor_preds, dict):
                for attr_name, preds in auditor_preds.items():
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        total_adv_loss = total_adv_loss + adversarial_loss(
                            preds, targets, maximize=True,
                            confusion_type=self.config.confusion_type,
                        )
            else:
                attr_name = self._get_primary_attr(purpose_name)
                if attr_name and attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    total_adv_loss = total_adv_loss + adversarial_loss(
                        auditor_preds, targets, maximize=True,
                        confusion_type=self.config.confusion_type,
                    )

        # Verification loss
        total_verify_loss = torch.tensor(0.0, device=self.device)
        if self.config.lambda_verify > 0 and self.verify_reg is not None:
            disallowed = self._get_disallowed_attrs(purpose_name)
            for attr_name in disallowed:
                if attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    r_sq = self.verify_reg(h_p, targets)
                    total_verify_loss = total_verify_loss + r_sq

        total_loss = (
            total_task_loss
            + purpose_lambda * total_adv_loss
            + self.config.lambda_verify * total_verify_loss
        )

        total_loss.backward()
        if self.config.grad_clip is not None:
            params = list(self.encoder.parameters())
            for head in self.task_heads.values():
                params.extend(head.parameters())
            nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        self.encoder_optimizer.step()

        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()

        return (
            total_loss.item(),
            total_task_loss.item(),
            total_adv_loss.item(),
            total_verify_loss.item(),
        )

    def _train_auditor_step(
        self,
        representations: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> float:
        """Single auditor training step.

        Auditors try to predict sensitive attributes from detached representations.

        Args:
            representations: Dictionary mapping purpose_name to representation.
            batch: Batch of data.

        Returns:
            Auditor loss value.
        """
        self.auditor_optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)

        for purpose_name, h_p in representations.items():
            if purpose_name not in self.auditors:
                continue

            auditor = self.auditors[purpose_name]

            # Detach representations to not update encoder
            h_p_detached = h_p.detach()

            # Get auditor predictions
            auditor_preds = auditor(h_p_detached)

            # Get targets for this purpose's disallowed attributes
            if isinstance(auditor_preds, dict):
                # Multi-attribute auditor
                for attr_name, preds in auditor_preds.items():
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        loss = adversarial_loss(preds, targets, maximize=False)
                        total_loss = total_loss + loss
            else:
                # Single auditor - need to get the attribute from config
                attr_name = self._get_primary_attr(purpose_name)
                if attr_name and attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    loss = adversarial_loss(auditor_preds, targets, maximize=False)
                    total_loss = total_loss + loss

        if total_loss.requires_grad:
            total_loss.backward()
            if self.config.grad_clip is not None:
                params = []
                for auditor in self.auditors.values():
                    params.extend(auditor.parameters())
                nn.utils.clip_grad_norm_(params, self.config.grad_clip)
            self.auditor_optimizer.step()

        if self.auditor_scheduler is not None:
            self.auditor_scheduler.step()

        return total_loss.item()

    def _train_encoder_step(
        self,
        batch: dict[str, Any],
        purposes: list[str],
    ) -> tuple[float, float, float, float]:
        """Single encoder + task heads training step.

        Args:
            batch: Batch of data.
            purposes: List of purpose names to train on.

        Returns:
            Tuple of (total_loss, task_loss, adv_loss, verify_loss).
        """
        self.encoder_optimizer.zero_grad()

        # Compute representations with gradients for encoder update
        representations = {}
        for purpose_name in purposes:
            purpose_idx = self._get_purpose_idx(purpose_name)
            h_p = self.encoder(batch["features"], purpose_idx)
            representations[purpose_name] = h_p

        # Compute task loss
        total_task_loss = torch.tensor(0.0, device=self.device)
        for purpose_name, h_p in representations.items():
            if purpose_name not in self.task_heads:
                continue

            task_head = self.task_heads[purpose_name]
            task_preds = task_head(h_p)

            # Get task type from config
            task_type = self._get_task_type(purpose_name)

            # Get appropriate targets
            if isinstance(task_preds, dict):
                for task_name, preds in task_preds.items():
                    if task_name in batch["task_labels"]:
                        targets = batch["task_labels"][task_name]
                        loss = task_loss(preds, targets, task_type)
                        total_task_loss = total_task_loss + loss
            else:
                # Single task head - use first available task label
                task_name = self._get_primary_task(purpose_name)
                if task_name and task_name in batch["task_labels"]:
                    targets = batch["task_labels"][task_name]
                    loss = task_loss(task_preds, targets, task_type)
                    total_task_loss = total_task_loss + loss

        # Compute adversarial loss (maximize confusion)
        # Use per-attribute lambdas when available, otherwise global lambda
        use_per_attr = (self.config.lambda_adv_per_attr is not None
                        or self.config.lambda_verify_per_attr is not None)
        total_adv_loss = torch.tensor(0.0, device=self.device)
        weighted_adv_loss = torch.tensor(0.0, device=self.device)
        for purpose_name, h_p in representations.items():
            if purpose_name not in self.auditors:
                continue

            auditor = self.auditors[purpose_name]
            auditor_preds = auditor(h_p)

            if isinstance(auditor_preds, dict):
                for attr_name, preds in auditor_preds.items():
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        loss = adversarial_loss(
                            preds, targets,
                            maximize=True,
                            confusion_type=self.config.confusion_type,
                        )
                        total_adv_loss = total_adv_loss + loss
                        if use_per_attr:
                            weighted_adv_loss = weighted_adv_loss + self._get_lambda_adv(attr_name) * loss
            else:
                attr_name = self._get_primary_attr(purpose_name)
                if attr_name and attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    loss = adversarial_loss(
                        auditor_preds, targets,
                        maximize=True,
                        confusion_type=self.config.confusion_type,
                    )
                    total_adv_loss = total_adv_loss + loss
                    if use_per_attr:
                        weighted_adv_loss = weighted_adv_loss + self._get_lambda_adv(attr_name) * loss

        # Verification regularizer: penalize high linear R² of disallowed attrs
        total_verify_loss = torch.tensor(0.0, device=self.device)
        weighted_verify_loss = torch.tensor(0.0, device=self.device)
        if (self.config.lambda_verify > 0 or self.config.lambda_verify_per_attr) and self.verify_reg is not None:
            for purpose_name, h_p in representations.items():
                disallowed = self._get_disallowed_attrs(purpose_name)
                for attr_name in disallowed:
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        r_sq = self.verify_reg(h_p, targets)
                        total_verify_loss = total_verify_loss + r_sq
                        if use_per_attr:
                            weighted_verify_loss = weighted_verify_loss + self._get_lambda_verify(attr_name) * r_sq

        # Combined loss: L_task + lambda_adv * L_adv + lambda_verify * L_verify
        if use_per_attr:
            total_loss = total_task_loss + weighted_adv_loss + weighted_verify_loss
        else:
            total_loss = (
                total_task_loss
                + self.config.lambda_adv * total_adv_loss
                + self.config.lambda_verify * total_verify_loss
            )

        # Backward and update
        total_loss.backward()
        if self.config.grad_clip is not None:
            params = list(self.encoder.parameters())
            for head in self.task_heads.values():
                params.extend(head.parameters())
            nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        self.encoder_optimizer.step()

        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()

        return (
            total_loss.item(),
            total_task_loss.item(),
            total_adv_loss.item(),
            total_verify_loss.item(),
        )

    def _train_grl_step(
        self,
        batch: dict[str, Any],
        purposes: list[str],
    ) -> tuple[float, float, float, float]:
        """Single training step using gradient reversal layer.

        In GRL mode, encoder and auditors are trained jointly. The GRL inside
        each auditor reverses gradients flowing to the encoder, so we use
        standard CE loss (not confusion loss). lambda_adv is passed to the
        auditor's GRL layer as the gradient scaling factor.

        Args:
            batch: Batch of data.
            purposes: List of purpose names to train on.

        Returns:
            Tuple of (total_loss, task_loss, adv_loss, verify_loss).
        """
        self.encoder_optimizer.zero_grad()
        self.auditor_optimizer.zero_grad()

        # Forward through encoder
        representations = {}
        for purpose_name in purposes:
            purpose_idx = self._get_purpose_idx(purpose_name)
            h_p = self.encoder(batch["features"], purpose_idx)
            representations[purpose_name] = h_p

        # Task loss
        total_task_loss = torch.tensor(0.0, device=self.device)
        for purpose_name, h_p in representations.items():
            if purpose_name not in self.task_heads:
                continue
            task_head = self.task_heads[purpose_name]
            task_preds = task_head(h_p)
            task_type = self._get_task_type(purpose_name)

            if isinstance(task_preds, dict):
                for task_name, preds in task_preds.items():
                    if task_name in batch["task_labels"]:
                        targets = batch["task_labels"][task_name]
                        loss = task_loss(preds, targets, task_type)
                        total_task_loss = total_task_loss + loss
            else:
                task_name = self._get_primary_task(purpose_name)
                if task_name and task_name in batch["task_labels"]:
                    targets = batch["task_labels"][task_name]
                    loss = task_loss(task_preds, targets, task_type)
                    total_task_loss = total_task_loss + loss

        # Auditor loss with GRL (standard CE; GRL handles gradient reversal)
        total_adv_loss = torch.tensor(0.0, device=self.device)
        for purpose_name, h_p in representations.items():
            if purpose_name not in self.auditors:
                continue
            auditor = self.auditors[purpose_name]
            # Pass lambda_adv to auditor — GRL inside reverses/scales encoder gradients
            auditor_preds = auditor(h_p, lambda_=self.config.lambda_adv)

            if isinstance(auditor_preds, dict):
                for attr_name, preds in auditor_preds.items():
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        loss = adversarial_loss(preds, targets, maximize=False)
                        total_adv_loss = total_adv_loss + loss
            else:
                attr_name = self._get_primary_attr(purpose_name)
                if attr_name and attr_name in batch["sensitive_attrs"]:
                    targets = batch["sensitive_attrs"][attr_name]
                    loss = adversarial_loss(auditor_preds, targets, maximize=False)
                    total_adv_loss = total_adv_loss + loss

        # Verification loss
        total_verify_loss = torch.tensor(0.0, device=self.device)
        if self.config.lambda_verify > 0 and self.verify_reg is not None:
            for purpose_name, h_p in representations.items():
                disallowed = self._get_disallowed_attrs(purpose_name)
                for attr_name in disallowed:
                    if attr_name in batch["sensitive_attrs"]:
                        targets = batch["sensitive_attrs"][attr_name]
                        r_sq = self.verify_reg(h_p, targets)
                        total_verify_loss = total_verify_loss + r_sq

        # Combined: task + adv + verify (no lambda_adv multiplier — GRL handles it)
        total_loss = (
            total_task_loss
            + total_adv_loss
            + self.config.lambda_verify * total_verify_loss
        )

        # Backward and update both optimizers
        total_loss.backward()
        if self.config.grad_clip is not None:
            all_params = list(self.encoder.parameters())
            for head in self.task_heads.values():
                all_params.extend(head.parameters())
            for aud in self.auditors.values():
                all_params.extend(aud.parameters())
            nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)
        self.encoder_optimizer.step()
        self.auditor_optimizer.step()

        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.auditor_scheduler is not None:
            self.auditor_scheduler.step()

        return (
            total_loss.item(),
            total_task_loss.item(),
            total_adv_loss.item(),
            total_verify_loss.item(),
        )

    def _select_purposes(self, batch_idx: int) -> list[str]:
        """Select purposes to train on for this batch."""
        if self.config.purpose_sampling == "all":
            return self.purpose_names
        elif self.config.purpose_sampling == "alternating":
            idx = batch_idx % len(self.purpose_names)
            return [self.purpose_names[idx]]
        elif self.config.purpose_sampling == "random":
            idx = torch.randint(len(self.purpose_names), (1,)).item()
            return [self.purpose_names[idx]]
        else:
            return self.purpose_names

    def _get_purpose_idx(self, purpose_name: str) -> int:
        """Get purpose index from name."""
        if purpose_name in self.purpose_configs:
            return self.purpose_configs[purpose_name].get("purpose_idx", 0)
        return self.purpose_names.index(purpose_name)

    def _get_task_type(self, purpose_name: str) -> str:
        """Get task type for purpose."""
        if purpose_name in self.purpose_configs:
            return self.purpose_configs[purpose_name].get("task_type", "classification")
        return "classification"

    def _get_primary_task(self, purpose_name: str) -> str | None:
        """Get primary task name for purpose."""
        if purpose_name in self.purpose_configs:
            tasks = self.purpose_configs[purpose_name].get("allowed_tasks", [])
            return tasks[0] if tasks else None
        return None

    def _get_primary_attr(self, purpose_name: str) -> str | None:
        """Get primary sensitive attribute for purpose."""
        if purpose_name in self.purpose_configs:
            attrs = self.purpose_configs[purpose_name].get("disallowed_attrs", [])
            return attrs[0] if attrs else None
        return None

    def _get_disallowed_attrs(self, purpose_name: str) -> list[str]:
        """Get all disallowed attributes for a purpose."""
        if purpose_name in self.purpose_configs:
            return self.purpose_configs[purpose_name].get("disallowed_attrs", [])
        return []

    def _get_lambda_adv_for_purpose(self, purpose_name: str) -> float:
        """Get adversarial lambda for a specific purpose."""
        if self.config.lambda_adv_per_purpose and purpose_name in self.config.lambda_adv_per_purpose:
            return self.config.lambda_adv_per_purpose[purpose_name]
        return self.config.lambda_adv

    def _get_lambda_adv(self, attr_name: str) -> float:
        """Get adversarial lambda for a specific attribute."""
        if self.config.lambda_adv_per_attr and attr_name in self.config.lambda_adv_per_attr:
            return self.config.lambda_adv_per_attr[attr_name]
        return self.config.lambda_adv

    def _get_lambda_verify(self, attr_name: str) -> float:
        """Get verification lambda for a specific attribute."""
        if self.config.lambda_verify_per_attr and attr_name in self.config.lambda_verify_per_attr:
            return self.config.lambda_verify_per_attr[attr_name]
        return self.config.lambda_verify

    def _update_accuracy_stats(
        self,
        representations: dict[str, torch.Tensor],
        batch: dict[str, Any],
        purposes: list[str],
        task_correct: dict[str, int],
        task_total: dict[str, int],
        auditor_correct: dict[str, int],
        auditor_total: dict[str, int],
    ) -> None:
        """Update accuracy statistics."""
        for purpose_name in purposes:
            if purpose_name not in representations:
                continue

            h_p = representations[purpose_name]

            # Task accuracy
            if purpose_name in self.task_heads:
                task_head = self.task_heads[purpose_name]
                task_preds = task_head(h_p)

                if isinstance(task_preds, dict):
                    for task_name, logits in task_preds.items():
                        if task_name in batch["task_labels"]:
                            preds = logits.argmax(dim=-1)
                            targets = batch["task_labels"][task_name]
                            correct = (preds == targets).sum().item()
                            task_correct[task_name] = task_correct.get(task_name, 0) + correct
                            task_total[task_name] = task_total.get(task_name, 0) + len(targets)
                else:
                    task_name = self._get_primary_task(purpose_name)
                    if task_name and task_name in batch["task_labels"]:
                        preds = task_preds.argmax(dim=-1)
                        targets = batch["task_labels"][task_name]
                        correct = (preds == targets).sum().item()
                        task_correct[task_name] = task_correct.get(task_name, 0) + correct
                        task_total[task_name] = task_total.get(task_name, 0) + len(targets)

            # Auditor accuracy
            if purpose_name in self.auditors:
                auditor = self.auditors[purpose_name]
                auditor_preds = auditor(h_p)

                if isinstance(auditor_preds, dict):
                    for attr_name, logits in auditor_preds.items():
                        if attr_name in batch["sensitive_attrs"]:
                            preds = logits.argmax(dim=-1)
                            targets = batch["sensitive_attrs"][attr_name]
                            correct = (preds == targets).sum().item()
                            auditor_correct[attr_name] = auditor_correct.get(attr_name, 0) + correct
                            auditor_total[attr_name] = auditor_total.get(attr_name, 0) + len(targets)
                else:
                    attr_name = self._get_primary_attr(purpose_name)
                    if attr_name and attr_name in batch["sensitive_attrs"]:
                        preds = auditor_preds.argmax(dim=-1)
                        targets = batch["sensitive_attrs"][attr_name]
                        correct = (preds == targets).sum().item()
                        auditor_correct[attr_name] = auditor_correct.get(attr_name, 0) + correct
                        auditor_total[attr_name] = auditor_total.get(attr_name, 0) + len(targets)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochMetrics:
        """Evaluate on a data loader.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Evaluation metrics.
        """
        self.encoder.eval()
        self.task_heads.eval()
        self.auditors.eval()

        start_time = time.time()

        total_loss = 0.0
        total_task_loss = 0.0
        total_adv_loss = 0.0
        num_batches = 0

        task_correct: dict[str, int] = {}
        task_total: dict[str, int] = {}
        auditor_correct: dict[str, int] = {}
        auditor_total: dict[str, int] = {}

        for batch in loader:
            batch = self._to_device(batch)

            # Get representations for all purposes
            representations = {}
            for purpose_name in self.purpose_names:
                purpose_idx = self._get_purpose_idx(purpose_name)
                h_p = self.encoder(batch["features"], purpose_idx)
                representations[purpose_name] = h_p

            # Compute losses
            task_l = torch.tensor(0.0, device=self.device)
            adv_l = torch.tensor(0.0, device=self.device)

            for purpose_name, h_p in representations.items():
                # Task loss
                if purpose_name in self.task_heads:
                    task_head = self.task_heads[purpose_name]
                    task_preds = task_head(h_p)
                    task_type = self._get_task_type(purpose_name)

                    if isinstance(task_preds, dict):
                        for task_name, preds in task_preds.items():
                            if task_name in batch["task_labels"]:
                                targets = batch["task_labels"][task_name]
                                task_l = task_l + task_loss(preds, targets, task_type)
                    else:
                        task_name = self._get_primary_task(purpose_name)
                        if task_name and task_name in batch["task_labels"]:
                            targets = batch["task_labels"][task_name]
                            task_l = task_l + task_loss(task_preds, targets, task_type)

                # Adversarial loss
                if purpose_name in self.auditors:
                    auditor = self.auditors[purpose_name]
                    auditor_preds = auditor(h_p)

                    if isinstance(auditor_preds, dict):
                        for attr_name, preds in auditor_preds.items():
                            if attr_name in batch["sensitive_attrs"]:
                                targets = batch["sensitive_attrs"][attr_name]
                                adv_l = adv_l + adversarial_loss(preds, targets, maximize=False)
                    else:
                        attr_name = self._get_primary_attr(purpose_name)
                        if attr_name and attr_name in batch["sensitive_attrs"]:
                            targets = batch["sensitive_attrs"][attr_name]
                            adv_l = adv_l + adversarial_loss(auditor_preds, targets, maximize=False)

            loss = task_l + self.config.lambda_adv * adv_l

            total_loss += loss.item()
            total_task_loss += task_l.item()
            total_adv_loss += adv_l.item()
            num_batches += 1

            # Update accuracy stats
            self._update_accuracy_stats(
                representations, batch, self.purpose_names,
                task_correct, task_total, auditor_correct, auditor_total
            )

        epoch_time = time.time() - start_time

        return EpochMetrics(
            loss=total_loss / num_batches if num_batches > 0 else 0.0,
            task_loss=total_task_loss / num_batches if num_batches > 0 else 0.0,
            adversarial_loss=total_adv_loss / num_batches if num_batches > 0 else 0.0,
            task_accuracy={k: task_correct[k] / task_total[k] for k in task_correct if task_total[k] > 0},
            auditor_accuracy={k: auditor_correct[k] / auditor_total[k] for k in auditor_correct if auditor_total[k] > 0},
            epoch_time=epoch_time,
        )

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move batch to device."""
        return {
            "features": batch["features"].to(self.device),
            "task_labels": {k: v.to(self.device) for k, v in batch["task_labels"].items()},
            "sensitive_attrs": {k: v.to(self.device) for k, v in batch["sensitive_attrs"].items()},
        }

    def _log_epoch_metrics(self, prefix: str, metrics: EpochMetrics) -> None:
        """Log epoch metrics."""
        logger.info(
            f"[{prefix}] Epoch {self.state.epoch} - "
            f"Loss: {metrics.loss:.4f}, Task: {metrics.task_loss:.4f}, "
            f"Adv: {metrics.adversarial_loss:.4f}, Verify: {metrics.verification_loss:.4f}, "
            f"Time: {metrics.epoch_time:.1f}s"
        )
        for task, acc in metrics.task_accuracy.items():
            logger.info(f"  Task '{task}' accuracy: {acc:.4f}")
        for attr, acc in metrics.auditor_accuracy.items():
            logger.info(f"  Auditor '{attr}' accuracy: {acc:.4f}")

    def _update_history(self, prefix: str, metrics: EpochMetrics) -> None:
        """Update training history."""
        for key in ["loss", "task_loss", "adversarial_loss", "verification_loss"]:
            history_key = f"{prefix}_{key}"
            if history_key not in self.state.history:
                self.state.history[history_key] = []
            self.state.history[history_key].append(getattr(metrics, key))

        for task, acc in metrics.task_accuracy.items():
            key = f"{prefix}_task_acc_{task}"
            if key not in self.state.history:
                self.state.history[key] = []
            self.state.history[key].append(acc)

        for attr, acc in metrics.auditor_accuracy.items():
            key = f"{prefix}_auditor_acc_{attr}"
            if key not in self.state.history:
                self.state.history[key] = []
            self.state.history[key].append(acc)

    def save_checkpoint(self, name: str) -> Path:
        """Save a checkpoint.

        Args:
            name: Name for the checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "task_heads": self.task_heads.state_dict(),
            "auditors": self.auditors.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "auditor_optimizer": self.auditor_optimizer.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
                "patience_counter": self.state.patience_counter,
            },
            "history": self.state.history,
            "config": {
                "lambda_adv": self.config.lambda_adv,
                "auditor_steps": self.config.auditor_steps,
            },
        }

        path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.task_heads.load_state_dict(checkpoint["task_heads"])
        self.auditors.load_state_dict(checkpoint["auditors"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        self.auditor_optimizer.load_state_dict(checkpoint["auditor_optimizer"])

        self.state.epoch = checkpoint["state"]["epoch"]
        self.state.global_step = checkpoint["state"]["global_step"]
        self.state.best_val_loss = checkpoint["state"]["best_val_loss"]
        self.state.patience_counter = checkpoint["state"].get("patience_counter", 0)
        self.state.history = checkpoint.get("history", {})

        logger.info(f"Loaded checkpoint from {path}")

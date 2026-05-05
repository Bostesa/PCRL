"""LAFTR-style adversarial fair representation baseline.

Single-purpose: one task, one set of disallowed attributes. The encoder is
PCRL's :class:`StandardEncoder` (MLP[128,128]→64) trained jointly with a task
head and one MLP discriminator per disallowed attribute. We optimize the
encoder to minimize task loss while *maximizing* discriminator cross-entropy
(i.e., remove attribute-predictive information) via an alternating two-phase
update; discriminators are themselves trained to predict the attribute on a
detached representation each step.

Hyperparameters per the design plan: lambda_adv=1.0, batch=256, 200 epochs,
Adam lr=1e-3, discriminator MLP [64, 32]→K with ReLU, no GRL (alternating
optimizers instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LAFTRDiscriminator(nn.Module):
    """2-layer MLP per the spec: 64 → 32 → num_classes."""

    def __init__(self, repr_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


@dataclass
class LAFTRHistory:
    train_task: list[float] = field(default_factory=list)
    train_adv: list[float] = field(default_factory=list)
    train_disc: list[float] = field(default_factory=list)
    val_task: list[float] = field(default_factory=list)
    val_disc_acc: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_task: float = float("inf")


def _disc_accuracy(
    discriminators: dict[str, LAFTRDiscriminator],
    encoder: nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    device: torch.device,
) -> dict[str, float]:
    """Per-attribute discriminator accuracy on `loader` (no grad)."""
    encoder.eval()
    for d in discriminators.values():
        d.eval()
    correct = {a: 0 for a in discriminators}
    total = {a: 0 for a in discriminators}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            for attr, disc in discriminators.items():
                if attr not in batch["sensitive_attrs"]:
                    continue
                y = batch["sensitive_attrs"][attr].to(device)
                preds = disc(h).argmax(-1)
                correct[attr] += int((preds == y).sum().item())
                total[attr] += int(y.numel())
    return {a: (correct[a] / total[a]) if total[a] else 0.0 for a in discriminators}


def train_laftr(
    encoder: nn.Module,
    task_head: nn.Module,
    discriminators: dict[str, LAFTRDiscriminator],
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    purpose_idx: int,
    task_name: str,
    disallowed_attrs: list[str],
    lambda_adv: float = 1.0,
    epochs: int = 200,
    lr: float = 1e-3,
    disc_steps: int = 1,
    device: str | torch.device = "cpu",
    log_every: int = 10,
    patience: int = 20,
    log_fn=print,
) -> LAFTRHistory:
    """Alternating LAFTR training loop.

    Phase 1 (per batch, ``disc_steps`` times): freeze encoder, update
    discriminators on detached representations to *minimize* CE.
    Phase 2: update encoder + task head with
    ``L = task_loss − λ_adv · Σ_a CE(disc_a(h), y_a)``.

    Returns the training history; best-validation encoder/task-head weights
    are restored into the modules in-place before return.
    """
    device = torch.device(device)
    encoder = encoder.to(device)
    task_head = task_head.to(device)
    for d in discriminators.values():
        d.to(device)

    enc_params = list(encoder.parameters()) + list(task_head.parameters())
    enc_opt = torch.optim.Adam(enc_params, lr=lr)
    disc_params: list[nn.Parameter] = []
    for d in discriminators.values():
        disc_params.extend(d.parameters())
    disc_opt = torch.optim.Adam(disc_params, lr=lr)

    history = LAFTRHistory()
    best_state: dict | None = None
    bad = 0

    for epoch in range(epochs):
        encoder.train()
        task_head.train()
        for d in discriminators.values():
            d.train()

        running_task = running_adv = running_disc = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["features"].to(device)
            y_task = batch["task_labels"][task_name].to(device)
            y_attrs = {
                a: batch["sensitive_attrs"][a].to(device) for a in disallowed_attrs
            }

            # ----- Phase 1: discriminator update on detached repr -----
            for _ in range(disc_steps):
                disc_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    h_det = encoder(x, purpose_idx)
                d_loss = x.new_zeros(())
                for attr, disc in discriminators.items():
                    d_loss = d_loss + F.cross_entropy(disc(h_det), y_attrs[attr])
                d_loss.backward()
                disc_opt.step()

            # ----- Phase 2: encoder + task head update -----
            enc_opt.zero_grad(set_to_none=True)
            h = encoder(x, purpose_idx)
            t_logits = task_head(h)
            task_loss = F.cross_entropy(t_logits, y_task)

            # Adversarial term: encoder wants to MAXIMIZE disc CE, so we
            # subtract λ * disc_CE from the encoder loss. Discriminator
            # weights are frozen for this phase via requires_grad toggle so
            # the disc params don't move under enc_opt's step (they aren't
            # in enc_opt anyway, but we also want backward through them
            # without populating their .grad).
            adv_loss = x.new_zeros(())
            for disc in discriminators.values():
                for p in disc.parameters():
                    p.requires_grad_(False)
            for attr, disc in discriminators.items():
                adv_loss = adv_loss + F.cross_entropy(disc(h), y_attrs[attr])
            for disc in discriminators.values():
                for p in disc.parameters():
                    p.requires_grad_(True)

            enc_loss = task_loss - lambda_adv * adv_loss
            enc_loss.backward()
            enc_opt.step()

            running_task += float(task_loss.item())
            running_adv += float(adv_loss.item())
            running_disc += float(d_loss.item())
            n_batches += 1

        history.train_task.append(running_task / max(n_batches, 1))
        history.train_adv.append(running_adv / max(n_batches, 1))
        history.train_disc.append(running_disc / max(n_batches, 1))

        # ----- Validation -----
        encoder.eval()
        task_head.eval()
        val_task = 0.0
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y_task = batch["task_labels"][task_name].to(device)
                h = encoder(x, purpose_idx)
                val_task += float(F.cross_entropy(task_head(h), y_task).item())
                n_v += 1
        avg_val = val_task / max(n_v, 1)
        history.val_task.append(avg_val)

        disc_acc = _disc_accuracy(
            discriminators, encoder, val_loader, purpose_idx, device,
        )
        history.val_disc_acc.append(disc_acc)

        if avg_val < history.best_val_task:
            history.best_val_task = avg_val
            history.best_epoch = epoch
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "task_head": {k: v.detach().cpu().clone() for k, v in task_head.state_dict().items()},
                "discriminators": {
                    a: {k: v.detach().cpu().clone() for k, v in d.state_dict().items()}
                    for a, d in discriminators.items()
                },
            }
            bad = 0
        else:
            bad += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            disc_str = " ".join(f"{a}={v:.3f}" for a, v in disc_acc.items())
            log_fn(
                f"  [laftr ep {epoch + 1:3d}/{epochs}] "
                f"task={history.train_task[-1]:.4f} "
                f"adv={history.train_adv[-1]:.4f} "
                f"disc={history.train_disc[-1]:.4f} "
                f"val_task={avg_val:.4f} "
                f"disc_acc(val) {disc_str}"
            )

        if bad >= patience:
            log_fn(f"  [laftr] early stop at epoch {epoch + 1} (no val improvement for {patience})")
            break

    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
        task_head.load_state_dict({k: v.to(device) for k, v in best_state["task_head"].items()})
        for a, d in discriminators.items():
            d.load_state_dict({k: v.to(device) for k, v in best_state["discriminators"][a].items()})

    return history

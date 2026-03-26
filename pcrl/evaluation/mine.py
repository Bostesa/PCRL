"""Mutual Information Neural Estimation (MINE) — Belghazi et al. 2018.

Estimates mutual information I(H; Z) between representations H and
disallowed attribute labels Z using a learned statistics network T:

    I(H; Z) >= E_joint[T(h, z)] - log(E_marginal[exp(T(h, z'))])

where z' is drawn from the marginal distribution of Z (by shuffling).
The bound is tight when T is optimal.

This gives a single scalar measuring total information leakage — not
just linear (R²), not just what specific auditors find, but an estimate
of ALL recoverable information between the representation and the
attribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class MINEResult:
    """Result of a MINE estimation.

    Attributes:
        mi_estimate: Estimated mutual information in nats.
        mi_bits: Estimated mutual information in bits.
        training_losses: Per-step loss values (negative MI lower bound).
    """

    mi_estimate: float
    mi_bits: float
    training_losses: list[float]


class StatisticsNetwork(nn.Module):
    """Small MLP that maps (h, z_onehot) -> scalar."""

    def __init__(self, repr_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = repr_dim + num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, z_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def _compute_dv_bound(
    net: StatisticsNetwork,
    H: torch.Tensor,
    Z_onehot: torch.Tensor,
    batch_size: int,
    n: int,
    device: str | torch.device,
) -> float:
    """Compute the Donsker-Varadhan MI lower bound on held-out data."""
    mi_vals = []
    with torch.no_grad():
        for _ in range(20):
            idx = torch.randint(0, n, (min(batch_size, n),), device=device)
            h_b = H[idx]
            z_b = Z_onehot[idx]
            marg_idx = torch.randint(0, n, (min(batch_size, n),), device=device)
            z_m = Z_onehot[marg_idx]

            t_joint = net(h_b, z_b)
            t_marginal = net(h_b, z_m)
            mi = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)
            mi_vals.append(mi.item())
    return float(np.mean(mi_vals))


class MINEstimator:
    """MINE mutual information estimator.

    Trains a statistics network to maximize the Donsker-Varadhan
    lower bound on mutual information between representations and
    attribute labels.

    Uses a train/val split to prevent overfitting: the network is
    trained on the train portion and MI is evaluated on the val portion.
    Weight decay provides additional regularization.

    Args:
        hidden_dim: Hidden layer size for the statistics network.
        num_steps: Number of training steps.
        batch_size: Batch size for training.
        lr: Learning rate.
        weight_decay: L2 regularization on network weights.
        ema_decay: Exponential moving average decay for the log-sum-exp
            term (bias-corrected gradient estimator from Belghazi et al.).
        val_fraction: Fraction of data held out for evaluation.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_steps: int = 500,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.99,
        val_fraction: float = 0.3,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.val_fraction = val_fraction

    def estimate(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
        device: str = "cpu",
    ) -> MINEResult:
        """Estimate mutual information I(H; Z).

        Args:
            representations: (n, d) array of representations.
            labels: (n,) integer labels for the disallowed attribute.
            device: Device for training.

        Returns:
            MINEResult with estimated MI in nats and bits.
        """
        n, d = representations.shape
        num_classes = int(labels.max()) + 1

        # Train/val split
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        n_val = max(int(n * self.val_fraction), 100)
        n_train = n - n_val
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        H_train = torch.tensor(
            representations[train_idx], dtype=torch.float32, device=device,
        )
        Z_train_int = torch.tensor(
            labels[train_idx].astype(int), dtype=torch.long, device=device,
        )
        Z_train = torch.zeros(n_train, num_classes, device=device)
        Z_train.scatter_(1, Z_train_int.unsqueeze(1), 1.0)

        H_val = torch.tensor(
            representations[val_idx], dtype=torch.float32, device=device,
        )
        Z_val_int = torch.tensor(
            labels[val_idx].astype(int), dtype=torch.long, device=device,
        )
        Z_val = torch.zeros(n_val, num_classes, device=device)
        Z_val.scatter_(1, Z_val_int.unsqueeze(1), 1.0)

        # Initialize statistics network
        net = StatisticsNetwork(d, num_classes, self.hidden_dim).to(device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        # EMA for bias-corrected gradient
        ema_running = 1.0
        losses = []
        best_val_mi = -float("inf")
        best_state = None
        patience_counter = 0

        net.train()
        for step in range(self.num_steps):
            # Sample batch indices from train set
            idx = torch.randint(0, n_train, (self.batch_size,), device=device)
            h_batch = H_train[idx]
            z_batch = Z_train[idx]

            # Marginal samples: independent label indices
            marg_idx = torch.randint(0, n_train, (self.batch_size,), device=device)
            z_marginal = Z_train[marg_idx]

            # Forward pass
            t_joint = net(h_batch, z_batch)
            t_marginal = net(h_batch, z_marginal)

            # DV bound components
            joint_mean = t_joint.mean()
            exp_marginal = torch.exp(t_marginal)
            exp_mean = exp_marginal.mean()

            # Update EMA of E_Q[exp(T)]
            ema_running = (
                self.ema_decay * ema_running
                + (1 - self.ema_decay) * exp_mean.item()
            )

            # Bias-corrected gradient (Belghazi et al. 2018):
            # weights = exp(T) / EMA (detached so gradient only flows through T)
            weights = (exp_marginal / (ema_running + 1e-8)).detach()
            loss = -(joint_mean - (weights * t_marginal).mean())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            # Monitor DV bound on train
            with torch.no_grad():
                mi_lb = joint_mean - torch.log(exp_mean + 1e-8)
            losses.append(mi_lb.item())

            # Early stopping on validation MI (check every 50 steps)
            if (step + 1) % 50 == 0:
                net.eval()
                val_mi = _compute_dv_bound(
                    net, H_val, Z_val, self.batch_size, n_val, device,
                )
                net.train()

                if val_mi > best_val_mi:
                    best_val_mi = val_mi
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 6:  # 300 steps without improvement
                        break

        # Restore best model and evaluate on validation set
        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        final_mi = _compute_dv_bound(
            net, H_val, Z_val, self.batch_size, n_val, device,
        )

        mi_nats = max(0.0, final_mi)
        mi_bits = mi_nats / np.log(2)

        return MINEResult(
            mi_estimate=mi_nats,
            mi_bits=mi_bits,
            training_losses=losses,
        )

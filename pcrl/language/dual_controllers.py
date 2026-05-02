"""Dual-update controllers for Phase-1 BIOS-medium.

Two controllers are provided. Both update a single ``marginal_gender``
constraint's ``lambda_value`` on the existing
``ProxyLagrangianOptimizer.constraints`` dict — neither modifies the proxy
class itself.

  * ``HeldoutR2Controller`` (Option D, default): maintains a periodically-
    refreshed scalar from a fixed held-out forward pass, hands it to
    ``proxy.dual_step``. The proxy's ``Constraint.update_lambda`` does
    its standard gradient-ascent update.

  * ``EmaCrossCovPIController`` (Plan B, swap target): tracks EMA of
    ``E[xx^T]``, ``E[zz^T]``, ``E[xz^T]``, ``E[x]``, ``E[z]`` from each
    primal mini-batch (no extra forward), reconstructs the closed-form
    ridge OLS R² from those moments, and PI-controls ``lambda_value``
    on ``r2_ema - threshold``.

The Plan-B swap is triggered by ``FiveSignalMonitor`` when, in epoch 1,
any of the following fire:
    (a) ``lambda_max`` saturation streak > 100 consecutive primal steps,
    (b) sign-flip rate (sign(holdout_r2 - threshold)) > 0.4 over last 20
        dual updates,
    (c) mid-window jump in holdout_r2 > 0.20 between consecutive refreshes.

Pre-committed by the user 2026-05-02 after the drift diagnostic showed the
post-projection alone is insufficient (Δ dev R² = +0.85 in 1 epoch on 5K).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------
# 5-signal monitor
# --------------------------------------------------------------------------

class FiveSignalMonitor:
    """Tracks the five Phase-1 progress signals.

    Per-primal-step input: in-batch R² scalar (from the differentiable
    verifier). Per-dual-refresh input: current ``lambda_value`` and the
    held-out R² scalar. Outputs a snapshot dict consumed by the run loop
    for periodic logging and Plan-B trip detection.
    """

    def __init__(
        self,
        *,
        threshold: float,
        ema_decay: float = 0.95,
        flip_window: int = 20,
        lambda_saturation_streak_trip: int = 100,
        flip_rate_trip: float = 0.4,
        mid_window_jump_trip: float = 0.20,
        epsilon_lambda_max: float = 1e-3,
    ):
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.flip_window = flip_window
        self.lambda_saturation_streak_trip = lambda_saturation_streak_trip
        self.flip_rate_trip = flip_rate_trip
        self.mid_window_jump_trip = mid_window_jump_trip
        self.epsilon_lambda_max = epsilon_lambda_max

        self.delta_lambda_ema: float = 0.0
        self.prev_lambda: Optional[float] = None
        self.sign_buffer: deque[int] = deque(maxlen=flip_window)
        self.inbatch_r2_window: list[float] = []
        self.last_holdout_r2: Optional[float] = None
        self.lambda_saturation_streak: int = 0
        # The most recent mid-window jump observed (for trip + reporting).
        self.last_mid_window_jump: float = 0.0

    def on_primal_step(self, *, lambda_now: float, lambda_max: float,
                        inbatch_r2: float) -> None:
        self.inbatch_r2_window.append(float(inbatch_r2))
        if lambda_now >= lambda_max - self.epsilon_lambda_max:
            self.lambda_saturation_streak += 1
        else:
            self.lambda_saturation_streak = 0

    def on_dual_refresh(self, *, lambda_now: float, holdout_r2: float) -> dict:
        # |Δλ| EMA
        if self.prev_lambda is not None:
            delta_l = abs(lambda_now - self.prev_lambda)
            self.delta_lambda_ema = (
                self.ema_decay * self.delta_lambda_ema
                + (1.0 - self.ema_decay) * delta_l
            )
        self.prev_lambda = lambda_now

        # sign(holdout_r2 - threshold) FIFO
        sign = 1 if holdout_r2 > self.threshold else 0
        self.sign_buffer.append(sign)

        # mid-window jump
        if self.last_holdout_r2 is not None:
            self.last_mid_window_jump = float(
                holdout_r2 - self.last_holdout_r2
            )
        self.last_holdout_r2 = float(holdout_r2)

        # in-batch R² window mean (resets each refresh)
        if self.inbatch_r2_window:
            inbatch_window_mean = float(np.mean(self.inbatch_r2_window))
        else:
            inbatch_window_mean = 0.0
        self.inbatch_r2_window = []

        return {
            "lambda": lambda_now,
            "delta_lambda_ema": self.delta_lambda_ema,
            "holdout_r2": float(holdout_r2),
            "inbatch_r2_window_mean": inbatch_window_mean,
            "sign": sign,
            "flip_rate": self.flip_rate(),
            "saturation_streak": self.lambda_saturation_streak,
            "mid_window_jump": self.last_mid_window_jump,
        }

    def flip_rate(self) -> float:
        if len(self.sign_buffer) < 2:
            return 0.0
        b = list(self.sign_buffer)
        flips = sum(1 for a, c in zip(b[:-1], b[1:]) if a != c)
        return flips / (len(b) - 1)

    def trip_check(self) -> Optional[str]:
        """Return a trip reason string if any Plan-B condition fires, else None."""
        if self.lambda_saturation_streak > self.lambda_saturation_streak_trip:
            return (
                f"PLAN_B_TRIP[lambda_saturation]: lambda saturated at "
                f"lambda_max for {self.lambda_saturation_streak} > "
                f"{self.lambda_saturation_streak_trip} consecutive primal steps"
            )
        if (
            len(self.sign_buffer) >= self.flip_window
            and self.flip_rate() > self.flip_rate_trip
        ):
            return (
                f"PLAN_B_TRIP[sign_flip]: flip-rate {self.flip_rate():.2f} > "
                f"{self.flip_rate_trip} over last {self.flip_window} dual updates"
            )
        if abs(self.last_mid_window_jump) > self.mid_window_jump_trip:
            return (
                f"PLAN_B_TRIP[mid_window_jump]: |Δholdout_r2| = "
                f"{abs(self.last_mid_window_jump):.4f} > "
                f"{self.mid_window_jump_trip} between consecutive dual refreshes"
            )
        return None


# --------------------------------------------------------------------------
# Plan B: EMA-of-cross-covariance + νPI controller
# --------------------------------------------------------------------------

@dataclass
class EmaState:
    mean_x: Optional[torch.Tensor] = None
    mean_z: Optional[torch.Tensor] = None
    M_xx: Optional[torch.Tensor] = None  # E[x x^T] (unsubtracted second moment)
    M_zz: Optional[torch.Tensor] = None  # E[z z^T]
    M_xz: Optional[torch.Tensor] = None  # E[x z^T]


class EmaCrossCovPIController:
    """EMA-of-second-moments → closed-form OLS R² → PI controller on λ.

    Reads ``(x, z_int)`` from each primal mini-batch and updates EMAs of
    the five second moments needed to reconstruct closed-form ridge OLS
    R². The controller's λ then tracks
        λ_{t+1} = clip(λ_t + kp · (r2_ema - τ) + ki · ∫(r2_ema - τ) dt)
    with anti-windup on the integral.

    Activated only when the held-out controller trips one of the three
    early-failure conditions in epoch 1. After activation, the run loop
    bypasses ``proxy.dual_step`` and instead overwrites
    ``proxy.constraints['marginal_gender'].lambda_value = controller.lam``
    before each ``proxy.lagrangian_loss`` call.

    Compute cost: O(d²) per primal step on CPU (d=768 → 768×768 = 590K
    multiplications, about 1ms on a modern CPU). No extra forward pass.
    """

    def __init__(
        self,
        *,
        d: int,
        n_classes: int = 2,
        threshold: float = 0.05,
        ema_decay: float = 0.99,
        kp: float = 50.0,
        ki: float = 5.0,
        integral_clip: float = 5.0,
        lambda_min: float = 5.0,
        lambda_init: float = 1.0,
        lambda_max: float = 100.0,
        ridge: float = 1e-6,
    ):
        self.d = d
        self.n_classes = n_classes
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.kp = kp
        self.ki = ki
        self.integral_clip = integral_clip
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.ridge = ridge
        self.lam: float = max(lambda_min, lambda_init)
        self.integral: float = 0.0
        self.state = EmaState()
        self.last_r2_ema: float = 0.0
        self.n_observations: int = 0

    @torch.no_grad()
    def observe(self, x: torch.Tensor, z_int: torch.Tensor) -> None:
        """Update EMA stats from one mini-batch.

        ``x`` may be on any device; we move to CPU+float64 for numerical
        stability of the gram solve in ``r2_ema``.
        """
        x_d = x.detach().to(dtype=torch.float64, device="cpu")
        z_d = z_int.detach().to(device="cpu").long()
        z_oh = torch.zeros(
            (x_d.shape[0], self.n_classes), dtype=torch.float64,
        )
        z_oh.scatter_(1, z_d.view(-1, 1), 1.0)

        n = x_d.shape[0]
        mu_x = x_d.mean(dim=0)
        mu_z = z_oh.mean(dim=0)
        Mxx = (x_d.t() @ x_d) / n
        Mzz = (z_oh.t() @ z_oh) / n
        Mxz = (x_d.t() @ z_oh) / n

        s = self.state
        if s.mean_x is None:
            s.mean_x, s.mean_z = mu_x, mu_z
            s.M_xx, s.M_zz, s.M_xz = Mxx, Mzz, Mxz
        else:
            d = self.ema_decay
            s.mean_x.mul_(d).add_(mu_x, alpha=1.0 - d)
            s.mean_z.mul_(d).add_(mu_z, alpha=1.0 - d)
            s.M_xx.mul_(d).add_(Mxx, alpha=1.0 - d)
            s.M_zz.mul_(d).add_(Mzz, alpha=1.0 - d)
            s.M_xz.mul_(d).add_(Mxz, alpha=1.0 - d)
        self.n_observations += n

    def r2_ema(self) -> float:
        """Closed-form ridge OLS R² on EMA-tracked centred moments."""
        s = self.state
        if s.mean_x is None:
            return 0.0
        cov_xx = s.M_xx - torch.outer(s.mean_x, s.mean_x)
        cov_zz = s.M_zz - torch.outer(s.mean_z, s.mean_z)
        cov_xz = s.M_xz - torch.outer(s.mean_x, s.mean_z)
        gram = cov_xx + self.ridge * torch.eye(
            cov_xx.shape[0], dtype=cov_xx.dtype,
        )
        try:
            sol = torch.linalg.solve(gram, cov_xz)
        except RuntimeError:
            return 0.0
        ss_explained = float(torch.trace(cov_xz.t() @ sol).item())
        ss_tot = float(torch.trace(cov_zz).item())
        if ss_tot <= 1e-12:
            return 0.0
        return float(max(0.0, min(1.0, ss_explained / ss_tot)))

    def step(self) -> tuple[float, float]:
        """Recompute λ from current EMA stats. Returns ``(lambda, r2_ema)``."""
        r2 = self.r2_ema()
        self.last_r2_ema = r2
        err = r2 - self.threshold
        self.integral = max(
            -self.integral_clip,
            min(self.integral_clip, self.integral + err),
        )
        self.lam = self.lam + self.kp * err + self.ki * self.integral
        self.lam = max(self.lambda_min, min(self.lambda_max, self.lam))
        return self.lam, r2

"""EMA-streaming LEACE fitter (PRO-LoRA Component 1).

Subclasses ``concept_erasure.LeaceFitter`` and replaces its 1/n cumulative
moment averages with bias-corrected exponential moving averages. The Round-2
``OnlineLeaceRefit`` wraps a fresh ``LeaceFitter`` over a sliding deque every
``refit_every`` steps; that re-pays the SVD/eigendecomp cost from scratch and
makes the eraser oscillate as the deque churns. EMA-Σ tracks the LoRA-shifted
[CLS] distribution continuously: every batch updates Σ_xx and Σ_xz cheaply,
and we recompute the eraser only every K=5 steps.

Bias correction follows Adam: divide by ``1 - (1-α)^t`` so the cold start does
not dominate. With α=0.02 the half-life is ~34 batches (spec said "~50",
within rounding).

The eraser property is inherited unchanged — we override ``sigma_xx`` and
``sigma_xz`` so the parent's eigendecomp + SVD path consumes EMA-tracked
covariances directly. Shrinkage and ``constrain_cov_trace`` still apply.
"""
from __future__ import annotations

import torch
from concept_erasure import LeaceFitter
from concept_erasure.caching import invalidates_cache


def _stable_shrinkage(S: torch.Tensor, n_eff: float) -> torch.Tensor:
    """Linear shrinkage toward an isotropic target, clamped to a convex combo.

    ``concept_erasure.shrinkage.optimal_linear_shrinkage`` is an *asymptotic*
    Ledoit-Wolf-style estimator that becomes unstable at small effective
    sample size. With our EMA effective N ≈ 2/α = 100 and p = 768 (BERT
    [CLS]), the closed-form α can land outside [0, 1] — i.e. it extrapolates
    *away* from the isotropic prior, producing negative eigenvalues that
    destroy the LEACE eigendecomp.

    This replacement uses the same (1-ρ)·S + ρ·F structure with target
    ``F = tr(S)/p · I`` and shrinkage intensity ρ chosen by the standard
    p/n regularization rule, then clamped so the result is always a convex
    combination. The rank-1 concept signal is preserved (target is isotropic,
    not low-rank).
    """
    p = S.shape[-1]
    trace_S = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(p, dtype=S.dtype, device=S.device).expand_as(S)
    target = eye * (trace_S / p)
    # Standard p/n shrinkage intensity. ρ → 1 (full shrinkage) as p/n → ∞;
    # ρ → 0 (no shrinkage) as n → ∞. Always in [0, 1).
    rho = float(p) / (float(p) + float(n_eff))
    return (1.0 - rho) * S + rho * target


class EmaLeaceFitter(LeaceFitter):
    """LEACE fitter with bias-corrected EMA moment tracking.

    Args:
        x_dim: Representation dim (768 for BERT [CLS]).
        z_dim: Concept dim (1 for scalar gender).
        alpha: EMA step size. α=0.02 → half-life ≈ 34 batches.
        device, dtype, shrinkage, constrain_cov_trace, svd_tol, affine,
        method: passed to ``LeaceFitter.__init__``.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        *,
        alpha: float = 0.02,
        device=None,
        dtype=None,
        shrinkage: bool = True,
        constrain_cov_trace: bool = True,
        svd_tol: float = 0.01,
        affine: bool = True,
        method: str = "leace",
    ):
        super().__init__(
            x_dim=x_dim,
            z_dim=z_dim,
            method=method,
            affine=affine,
            constrain_cov_trace=constrain_cov_trace,
            device=device,
            dtype=dtype,
            shrinkage=shrinkage,
            svd_tol=svd_tol,
        )
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.alpha = float(alpha)
        self.t = 0  # EMA step counter for bias correction.
        self._last_batch_n = 0  # Cached for shrinkage n_eff computation.

        # EMA accumulators (raw, not bias-corrected). Shadow the parent's
        # cumulative buffers; the parent's `n`, `sigma_xx_`, `sigma_xz_`,
        # `mean_x`, `mean_z` are unused on this subclass.
        self._mean_x_ema = torch.zeros(x_dim, device=device, dtype=dtype)
        self._mean_z_ema = torch.zeros(z_dim, device=device, dtype=dtype)
        self._sigma_xx_ema = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)
        self._sigma_xz_ema = torch.zeros(x_dim, z_dim, device=device, dtype=dtype)

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: torch.Tensor, z: torch.Tensor) -> "EmaLeaceFitter":
        """Update EMA covariance statistics with one batch."""
        d, c = self._sigma_xz_ema.shape
        x = x.reshape(-1, d).type_as(self._mean_x_ema)
        n = x.shape[0]
        z = z.reshape(n, -1).type_as(x)
        if z.shape[-1] != c:
            raise ValueError(f"Expected z dim {c}, got {z.shape[-1]}")
        if n < 2:
            return self  # need ≥2 samples to estimate a covariance.

        a = self.alpha
        self._last_batch_n = n

        # 1. EMA the raw batch means.
        mu_x_b = x.mean(dim=0)
        mu_z_b = z.mean(dim=0)
        self._mean_x_ema.mul_(1.0 - a).add_(mu_x_b, alpha=a)
        self._mean_z_ema.mul_(1.0 - a).add_(mu_z_b, alpha=a)

        # 2. Increment t before computing bias-correction so we use the
        # post-update step count, matching Adam's m_hat = m_t / (1 - β^t).
        self.t += 1
        bc = 1.0 - (1.0 - a) ** self.t

        # 3. Center the batch around the *bias-corrected* running mean.
        mu_x_hat = self._mean_x_ema / bc
        mu_z_hat = self._mean_z_ema / bc
        dx = x - mu_x_hat
        dz = z - mu_z_hat

        # 4. Per-batch centered second moments (1/B normalization — we do not
        # apply Bessel's correction at the batch level; the EMA over many
        # batches dominates the bias).
        b_xx = (dx.mH @ dx) / float(n)
        b_xz = (dx.mH @ dz) / float(n)

        # 5. EMA the second moments.
        self._sigma_xx_ema.mul_(1.0 - a).add_(b_xx, alpha=a)
        self._sigma_xz_ema.mul_(1.0 - a).add_(b_xz, alpha=a)
        return self

    def _bc(self) -> float:
        """Bias-correction factor for EMA at the current step."""
        if self.t == 0:
            return 1.0
        return 1.0 - (1.0 - self.alpha) ** self.t

    @property
    def mean_x(self) -> torch.Tensor:  # type: ignore[override]
        return self._mean_x_ema / self._bc()

    @mean_x.setter
    def mean_x(self, value: torch.Tensor) -> None:
        # LeaceFitter.__init__ writes to mean_x; absorb that into the EMA buf.
        self._mean_x_ema = value

    @property
    def mean_z(self) -> torch.Tensor:  # type: ignore[override]
        return self._mean_z_ema / self._bc()

    @mean_z.setter
    def mean_z(self, value: torch.Tensor) -> None:
        self._mean_z_ema = value

    @property
    def sigma_xx(self) -> torch.Tensor:  # type: ignore[override]
        if self.t < 1:
            raise RuntimeError("Call update() before accessing sigma_xx")
        S_raw = self._sigma_xx_ema / self._bc()
        # Symmetrize against accumulated numerical drift.
        S_hat = (S_raw + S_raw.mH) * 0.5
        if self.shrinkage:
            # We average per-batch covariances (each computed from B samples),
            # so the effective independent-sample count is batches × B. With
            # an EMA over ~2/α batches, that is `B / α` total samples — the
            # right scale for shrinkage. Using just `2/α` (treating each
            # batch as one sample) over-shrinks: at p=64, B=32, α=0.02 the
            # right n_eff is 3200, not 100.
            n_eff = float(max(self._last_batch_n, 1)) * 2.0 / self.alpha
            return _stable_shrinkage(S_hat, n_eff)
        return S_hat

    @property
    def sigma_xz(self) -> torch.Tensor:  # type: ignore[override]
        if self.t < 1:
            raise RuntimeError("Call update() before accessing sigma_xz")
        return self._sigma_xz_ema / self._bc()


class EmaLeaceRefit:
    """Streaming wrapper: update EMA every primal step, recompute eraser every K.

    Drop-in replacement for ``OnlineLeaceRefit`` (sliding-buffer LeaceFitter).
    Same external interface — ``observe(x, z)`` per step, ``should_refit(step)``
    + ``refit(step)`` to get a fresh eraser.
    """

    def __init__(
        self,
        *,
        d_x: int = 768,
        d_z: int = 1,
        alpha: float = 0.02,
        refit_every: int = 5,
        device="cpu",
        shrinkage: bool = True,
        constrain_cov_trace: bool = True,
    ):
        self.d_x = d_x
        self.d_z = d_z
        self.refit_every = max(int(refit_every), 1)
        self.device = torch.device(device)
        self.fitter = EmaLeaceFitter(
            x_dim=d_x,
            z_dim=d_z,
            alpha=alpha,
            device=self.device,
            dtype=torch.float32,
            shrinkage=shrinkage,
            constrain_cov_trace=constrain_cov_trace,
        )
        self.last_refit_step: int | None = None
        self.refit_count: int = 0

    @torch.no_grad()
    def observe(self, x: torch.Tensor, z: torch.Tensor) -> None:
        x_d = x.detach().to(dtype=torch.float32, device=self.device)
        z_d = z.detach().to(dtype=torch.float32, device=self.device).view(-1, 1)
        self.fitter.update(x_d, z_d)

    def should_refit(self, global_step: int) -> bool:
        if self.fitter.t < 2:
            return False
        if self.last_refit_step is None:
            return True
        return (global_step - self.last_refit_step) >= self.refit_every

    @torch.no_grad()
    def refit(self, global_step: int):
        eraser = self.fitter.eraser
        self.last_refit_step = global_step
        self.refit_count += 1
        return eraser

"""Online LEACE refit with sliding buffer + shrinkage.

Replaces the **static** post-projection (fit once at construction, registered
as frozen buffers) with a **dynamic** post-projection that tracks LoRA-shifted
[CLS]. The static post-projection failed in Round 1 because LoRA fine-tuning
shifts the [CLS] distribution off the LEACE-fit centroid within ~20 primal
steps (drift diagnostic confirmed Δ R² = +0.85 in 1 epoch on 5K samples).

Pipeline:
  1. Each primal step: append new (x, z) batch to a sliding buffer of size
     ``buffer_size`` (default 512) — FIFO eviction.
  2. Every ``refit_every`` primal steps: instantiate a fresh ``LeaceFitter``
     with shrinkage + cov-trace constraint, feed it the entire buffer, query
     its ``eraser`` property, register the new ``(P, mu)`` on the model via
     ``BertWithLoRA.set_leace_projection``.

Shrinkage (Ledoit-Wolf style, per concept_erasure default) is essential at
``d=768`` with ``n=512``: the empirical covariance is rank-deficient and the
standard LEACE solve would be ill-conditioned.

``d_z=1`` (scalar gender ∈ {0, 1} cast to float) per the Round-2 plan; the
shrinkage estimator handles low-cardinality concepts well in this regime.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import torch


class OnlineLeaceRefit:
    """Sliding-buffer online LEACE refit.

    Args:
        d_x: Feature dimension (768 for BERT [CLS]).
        d_z: Concept dimension (1 for scalar gender).
        buffer_size: Sliding-window size for refit data (default 512).
        refit_every: Refit cadence in primal steps (default 10 — matches the
            held-out R² refresh cadence).
        device: Device to keep the LeaceFitter on (matches model.device).
        shrinkage: Use Ledoit-Wolf-style shrinkage for cov estimation.
        constrain_cov_trace: Constrain post-erasure cov trace ≤ pre-erasure
            (prevents [CLS] norm explosion when LoRA shifts the distribution).
    """

    def __init__(
        self,
        *,
        d_x: int = 768,
        d_z: int = 1,
        buffer_size: int = 512,
        refit_every: int = 10,
        device: torch.device | str = "cpu",
        shrinkage: bool = True,
        constrain_cov_trace: bool = True,
    ):
        self.d_x = d_x
        self.d_z = d_z
        self.buffer_size = buffer_size
        self.refit_every = max(int(refit_every), 1)
        self.device = torch.device(device)
        self.shrinkage = shrinkage
        self.constrain_cov_trace = constrain_cov_trace
        self.x_buf: deque[torch.Tensor] = deque(maxlen=buffer_size)
        self.z_buf: deque[torch.Tensor] = deque(maxlen=buffer_size)
        self.last_refit_step: Optional[int] = None
        self.refit_count: int = 0

    @torch.no_grad()
    def observe(self, x: torch.Tensor, z: torch.Tensor) -> None:
        """Append ``(x, z)`` to the sliding buffer.

        ``x`` is ``(B, d_x)``; ``z`` is ``(B,)`` integer (gender label).
        Stored on CPU as fp32 for portability — refit moves to ``device``.
        """
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu")
        z_cpu = z.detach().to(dtype=torch.float32, device="cpu").view(-1, 1)
        for i in range(x_cpu.shape[0]):
            self.x_buf.append(x_cpu[i])
            self.z_buf.append(z_cpu[i])

    def should_refit(self, global_step: int) -> bool:
        """``True`` if a refit is due at this primal step."""
        if len(self.x_buf) < max(2 * self.d_z + 4, 16):
            return False
        if self.last_refit_step is None:
            return True
        return (global_step - self.last_refit_step) >= self.refit_every

    @torch.no_grad()
    def refit(self, global_step: int):
        """Fit a fresh ``LeaceFitter`` on the current buffer; return its eraser.

        Returns the ``LeaceEraser`` (with ``.P`` and ``.bias`` accessible)
        ready to register on the model.
        """
        from concept_erasure import LeaceFitter

        x = torch.stack(list(self.x_buf), dim=0).to(self.device)
        z = torch.stack(list(self.z_buf), dim=0).to(self.device)

        fitter = LeaceFitter(
            x_dim=self.d_x,
            z_dim=self.d_z,
            shrinkage=self.shrinkage,
            constrain_cov_trace=self.constrain_cov_trace,
            device=self.device,
            dtype=torch.float32,
        )
        # update() expects matched batch dimension on x and z.
        fitter.update(x, z)
        eraser = fitter.eraser
        self.last_refit_step = global_step
        self.refit_count += 1
        return eraser

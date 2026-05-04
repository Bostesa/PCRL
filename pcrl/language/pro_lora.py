"""PRO-LoRA: Pre-projected residual + LoRA encoder (Component 2).

Hooks the residual stream at intermediate BERT layers and applies an EMA-tracked
LEACE projection to the [CLS] token before the next layer reads it. This is the
linear analogue of CAFT (arXiv:2507.16795) — concept ablation at multiple sites
inside the transformer, not just at the final [CLS].

Architecture
------------
- Inherits from ``BertWithLoRA``: same frozen-base + Q/V LoRA + last-layer
  ``output.dense`` LoRA.
- Adds forward hooks on ``encoder.layer[i]`` for ``i in hook_layers`` (default
  ``[0, 6, 11]``).
- Each hook owns an ``EmaLeaceFitter`` and (lazily) an active ``LeaceEraser``.
- Per primal step:
    forward → hooks capture pre-projection [CLS] → hooks apply current eraser
    (if set) → modified hidden_states feed next layer → final [CLS] returned.
- Training loop calls ``observe_residuals(z)`` after the primal forward to
  push captured (h, z) pairs into each layer's EMA fitter.
- Training loop calls ``refit_erasers(global_step)`` at cadence to swap each
  layer's active eraser to the freshly-fit one.

The post-hoc [CLS] LEACE (``set_leace_projection`` on the parent) is preserved
unchanged — applied AFTER layer 11 hook + final encoder output. With the
default ``hook_layers=[0, 6, 11]`` the layer-11 hook already covers the
post-encoder [CLS], so the post-hoc LEACE acts as a final guard for whatever
slipped through.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from .bert_encoder import BertWithLoRA, CLS_DIM
from .ema_leace import EmaLeaceFitter


class _ProjectionHookHandle:
    """Per-layer state: EMA fitter + current eraser + capture stash."""

    def __init__(self, *, d_x: int, d_z: int, alpha: float, device, refit_every: int):
        self.fitter = EmaLeaceFitter(
            x_dim=d_x, z_dim=d_z, alpha=alpha,
            device=device, dtype=torch.float32,
            shrinkage=True, constrain_cov_trace=True,
        )
        self.refit_every = max(int(refit_every), 1)
        self.eraser = None  # set by refit_eraser()
        self.last_refit_step: Optional[int] = None
        self.refit_count = 0
        self.capture: Optional[torch.Tensor] = None  # (B, d) pre-projection [CLS]

    def reset_capture(self) -> None:
        self.capture = None

    def should_refit(self, global_step: int) -> bool:
        if self.fitter.t < 2:
            return False
        if self.last_refit_step is None:
            return True
        return (global_step - self.last_refit_step) >= self.refit_every

    @torch.no_grad()
    def refit_eraser(self, global_step: int) -> bool:
        """Recompute the active eraser from the EMA fitter; return True if updated."""
        if self.fitter.t < 2:
            return False
        self.eraser = self.fitter.eraser  # cached_property — invalidated by update()
        self.last_refit_step = global_step
        self.refit_count += 1
        return True


class ProLoRAEncoder(BertWithLoRA):
    """BERT + LoRA + EMA-tracked pre-projection at intermediate layers."""

    def __init__(
        self,
        *,
        model_name: str = "bert-base-uncased",
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
        hook_layers: Iterable[int] = (0, 6, 11),
        ema_alpha: float = 0.02,
        refit_every: int = 5,
        z_dim: int = 1,
    ) -> None:
        super().__init__(
            model_name=model_name, rank=rank, alpha=alpha, dropout=dropout,
        )
        self.hook_layers = tuple(int(i) for i in hook_layers)
        self.ema_alpha = float(ema_alpha)
        self.z_dim = int(z_dim)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self._hook_handles: dict[int, _ProjectionHookHandle] = {}
        for layer_idx in self.hook_layers:
            self._hook_handles[layer_idx] = _ProjectionHookHandle(
                d_x=CLS_DIM, d_z=z_dim, alpha=ema_alpha,
                device=device, refit_every=refit_every,
            )
        self._registered_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

    def _bert_layers(self) -> nn.ModuleList:
        """Return the ``BertLayer`` ModuleList from the PEFT-wrapped model."""
        # PEFT wraps: ProLoRAEncoder.peft_model.base_model.model is the BertModel.
        # base_model is a LoraModel; .model is the actual BertModel.
        bert = self.peft_model.base_model.model
        return bert.encoder.layer

    def _install_hooks(self) -> None:
        layers = self._bert_layers()
        for layer_idx in self.hook_layers:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(
                    f"hook_layers contains {layer_idx} but encoder has {len(layers)} layers"
                )
            handle = self._make_hook(layer_idx)
            h = layers[layer_idx].register_forward_hook(handle)
            self._registered_hooks.append(h)

    def _make_hook(self, layer_idx: int):
        """Build a forward hook for ``layers[layer_idx]``.

        ``BertLayer`` returns a tuple ``(hidden_states, *attns)``. We capture
        ``hidden_states[:, 0, :]`` (the [CLS] token) for EMA observation, then
        apply the current eraser to **every token position** before passing
        to the next layer.

        Why all-token projection: the layer-stratified probe (results/
        layer_stratified_probe.json) showed that CLS-only projection at
        layer 0 has Δ=0.000 effect on layer-1 [CLS] R² because attention
        in layer 1 reconstructs gender from non-CLS token positions
        (pronouns, names, gendered nouns). Projecting all positions closes
        that leakage path. The eraser P is fit on [CLS]↔gender pairs (still
        captured CLS-only) but is applied uniformly across positions.
        """
        state = self._hook_handles[layer_idx]

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                tail = output[1:]
            else:
                hidden_states = output
                tail = ()
            # Capture pre-projection [CLS] for the EMA fitter (CLS-only,
            # because the EMA fits the [CLS]↔gender covariance).
            cls_pre = hidden_states[:, 0, :].detach()
            state.capture = cls_pre.float()
            eraser = state.eraser
            if eraser is not None:
                P = eraser.P.to(dtype=hidden_states.dtype, device=hidden_states.device)
                bias = eraser.bias
                if bias is not None:
                    mu = bias.to(dtype=hidden_states.dtype, device=hidden_states.device)
                else:
                    mu = torch.zeros(P.shape[0], dtype=hidden_states.dtype, device=hidden_states.device)
                # All-token projection: (B, T, D) @ (D, D) = (B, T, D).
                # mu broadcasts over batch and sequence dims.
                new_hidden = mu + (hidden_states - mu) @ P.T
                if tail:
                    return (new_hidden,) + tail
                return new_hidden
            return output

        return hook

    @torch.no_grad()
    def observe_residuals(self, z: torch.Tensor) -> None:
        """Feed the captured pre-projection [CLS]s + concept ``z`` into each
        layer's EMA fitter. Call after a forward pass; clears captures.

        ``z`` is ``(B,)`` integer or float. Cast to fp32 before update().
        """
        z_f = z.detach().to(torch.float32).view(-1, self.z_dim)
        for layer_idx, state in self._hook_handles.items():
            cls = state.capture
            if cls is None:
                continue
            target_device = state.fitter._mean_x_ema.device
            x = cls.to(device=target_device, dtype=torch.float32)
            zd = z_f.to(device=target_device)
            state.fitter.update(x, zd)
            state.reset_capture()

    @torch.no_grad()
    def refit_erasers(self, global_step: int) -> dict[int, bool]:
        """Refresh each layer's active eraser if its cadence is due.

        Returns a map ``{layer_idx: refit_happened}`` for monitoring.
        """
        out: dict[int, bool] = {}
        for layer_idx, state in self._hook_handles.items():
            if state.should_refit(global_step):
                out[layer_idx] = state.refit_eraser(global_step)
            else:
                out[layer_idx] = False
        return out

    def hook_status(self) -> dict[int, dict]:
        """Diagnostic snapshot of each hook's EMA state + active eraser."""
        out = {}
        for layer_idx, state in self._hook_handles.items():
            out[layer_idx] = {
                "ema_steps": state.fitter.t,
                "refit_count": state.refit_count,
                "last_refit_step": state.last_refit_step,
                "has_eraser": state.eraser is not None,
            }
        return out

    def remove_hooks(self) -> None:
        for h in self._registered_hooks:
            h.remove()
        self._registered_hooks.clear()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

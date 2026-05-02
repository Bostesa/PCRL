"""Frozen bert-base-uncased + PEFT LoRA encoder for BIOS-medium.

Architecture
------------
- Backbone: ``AutoModel.from_pretrained("bert-base-uncased")`` with all base
  parameters frozen (``requires_grad=False``).
- LoRA via PEFT (rank=32, alpha=64, dropout=0.05, bias="none"). Targets
  (PEFT 0.19+ ``endswith`` matching):
    * ``"query"``                          — Q on every layer (12 adapters)
    * ``"value"``                          — V on every layer (12 adapters)
    * ``"encoder.layer.11.output.dense"``  — last-layer FFN output projection
                                             only (1 adapter)
  PEFT's ``layers_to_transform`` kwarg can't apply different layer ranges per
  module, so the explicit suffix list is the cleanest way to get Q/V on all
  layers + ``output.dense`` on layer 11 only. Total: 25 LoRA adapters.
- Representation: ``last_hidden_state[:, 0, :]`` (raw [CLS] of dimension 768).
  The BERT pooler is skipped — its tanh projection is learned from the NSP
  objective and can leak protected info that LEACE warm-start cannot reach.

The post-FFN LayerNorm sits AFTER ``layer.11.output.dense``. LEACE warm-start
applied at the LoRA injects a pre-LN affine; the LoRA adapters can compensate
for any LN-induced drift during fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

CLS_DIM: int = 768
LAYER11_OUTPUT_DENSE_NAME: str = "encoder.layer.11.output.dense"


def build_lora_config(
    *, rank: int = 32, alpha: int = 64, dropout: float = 0.05,
):
    """Build the canonical LoRA config for BIOS-medium (Phase 1 + Phase 2)."""
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["query", "value", LAYER11_OUTPUT_DENSE_NAME],
    )


@dataclass
class ParamBreakdown:
    trainable: int
    frozen: int
    ratio: float
    lora_q_v_count: int
    lora_output_dense_count: int

    def as_dict(self) -> dict:
        return {
            "trainable": self.trainable,
            "frozen": self.frozen,
            "ratio_trainable": self.ratio,
            "lora_q_v_adapters": self.lora_q_v_count,
            "lora_output_dense_adapters": self.lora_output_dense_count,
        }


def count_param_breakdown(model: nn.Module) -> ParamBreakdown:
    """Sum trainable / frozen params and count LoRA-A weights by location.

    LoRA-A weights are counted (not LoRA-B) because each adapter has exactly
    one ``lora_A.default.weight``, giving a clean adapter count.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    qv = od = 0
    for n, p in model.named_parameters():
        if not p.requires_grad or "lora_A" not in n or "weight" not in n:
            continue
        if ".query." in n or ".value." in n:
            qv += 1
        elif "output.dense" in n:
            od += 1
    total = trainable + frozen
    return ParamBreakdown(
        trainable=trainable,
        frozen=frozen,
        ratio=trainable / total if total else 0.0,
        lora_q_v_count=qv,
        lora_output_dense_count=od,
    )


class BertWithLoRA(nn.Module):
    """Frozen ``bert-base-uncased`` + LoRA on Q/V (all layers) + last-layer
    ``output.dense``. Forward returns raw [CLS] of shape ``(B, 768)``.

    Optional frozen LEACE post-projection: ``set_leace_projection(P, mu)``
    registers ``leace_P`` and ``leace_mu`` buffers. After ``set_leace_projection``,
    every ``forward`` returns ``mu + ([CLS] - mu) @ P.T`` instead of raw
    ``[CLS]``. This mirrors ``pcrl/models/lora.py:246`` and is the only
    LEACE-warm-start mechanism that survives BERT's post-FFN LayerNorm —
    a pre-LN LoRA-side warm-start gets re-centred by LayerNorm and does not
    drive [CLS] R² to zero. The buffer is non-trainable but gradients still
    flow through it during backprop, so the LoRA learns inside the null
    space defined by ``P``.
    """

    def __init__(
        self,
        *,
        model_name: str = "bert-base-uncased",
        rank: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        from peft import get_peft_model
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(model_name)
        # Freeze defensively even though PEFT also freezes base params; this
        # guards against a future PEFT regression silently un-freezing BERT.
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.peft_model = get_peft_model(
            backbone,
            build_lora_config(rank=rank, alpha=alpha, dropout=dropout),
        )
        self.repr_dim = CLS_DIM

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        if hasattr(self, "leace_P"):
            P = self.leace_P
            mu = self.leace_mu
            cls = mu + (cls - mu) @ P.T
        return cls

    @torch.no_grad()
    def set_leace_projection(
        self, P: torch.Tensor, mu: torch.Tensor,
    ) -> None:
        """Register a frozen LEACE post-projection.

        After this call, every ``forward`` applies
        ``cls' = mu + (cls - mu) @ P.T`` to the raw [CLS] before returning.
        ``P`` must be ``(d, d)`` (= ``eraser.P`` from concept_erasure) and
        ``mu`` must be ``(d,)`` (= ``eraser.bias``).

        The buffers are persistent so they survive ``state_dict``
        round-trips. They are stored on the module's current device.
        """
        if P.dim() != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(
                f"P must be a square matrix; got {tuple(P.shape)}"
            )
        if mu.dim() != 1 or mu.shape[0] != P.shape[0]:
            raise ValueError(
                f"mu must be (d,) matching P; got mu={tuple(mu.shape)} "
                f"P={tuple(P.shape)}"
            )
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = P.device
        self.register_buffer(
            "leace_P", P.detach().to(target_device).clone(), persistent=True,
        )
        self.register_buffer(
            "leace_mu", mu.detach().to(target_device).clone(), persistent=True,
        )

    def has_leace_projection(self) -> bool:
        return hasattr(self, "leace_P")

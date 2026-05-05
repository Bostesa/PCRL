"""ResNet18 + frozen-BN backbone with PEFT LoRA on a penultimate projection.

Architecture (option b' from the design discussion):

    image -> ResNet18 (frozen, BN frozen) -> avgpool -> 512-d
          -> penultimate_proj: nn.Linear(512, 512, bias=False)   weight = I (frozen)
                + PEFT LoRA(rank=8, alpha=16, dropout=0.05, bias='none')
          -> 512-d projected
          -> task head (separate nn.Linear(512, 2), trainable)

LEACE is fit on the post-avgpool 512-d, so the LoRA acts in the same space
the eraser was fit on -- the rank-r SVD of (Q-I) plugged into the Linear LoRA
yields an exact penultimate-space projection at construction time.

Trainable parameters after construction: only the LoRA A/B for ``penultimate_proj``
(plus the task head, which is constructed externally).
"""
from __future__ import annotations

from typing import Iterable

import timm.layers
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18EraseTaskLoRA(nn.Module):
    """ResNet-18 (frozen) -> ERASE layer (frozen Linear, LEACE-fit) -> TASK_PROJ
    (Linear w/ LoRA, identity init).

    Novel decoupled architecture: the erase layer is structurally frozen and
    contains the closed-form LEACE projection (Q, mu). The downstream task
    projection carries a LoRA adapter that trains on task signal. Because
    the erase layer is *outside* the trainable path, task gradients cannot
    rotate the projection itself; they can only re-encode concept signal
    into the residual that LEACE has already minimised.

    Forward: x -> backbone -> erase(.) -> task_proj(.) -> 512-d.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone = timm.layers.freeze_batch_norm_2d(backbone)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone

        # Erase layer: weight=Q (=I-pl@pr), bias=mu adjustment.
        # Filled in by leace_warmstart.fit_and_set_erase_layer; placeholder=identity.
        self.erase = nn.Linear(512, 512, bias=True)
        with torch.no_grad():
            self.erase.weight.copy_(torch.eye(512))
            self.erase.bias.zero_()
        for p in self.erase.parameters():
            p.requires_grad_(False)

        # Task projection: identity-init Linear, target of PEFT LoRA.
        self.task_proj = nn.Linear(512, 512, bias=False)
        with torch.no_grad():
            self.task_proj.weight.copy_(torch.eye(512))
        for p in self.task_proj.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            cls_name = type(m).__name__
            if "BatchNorm" in cls_name or "FrozenBatchNorm" in cls_name:
                m.eval()
        return self

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_erase(self, x: torch.Tensor) -> torch.Tensor:
        return self.erase(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_proj(self.erase(self.backbone(x)))


class ResNet18Penultimate(nn.Module):
    """ResNet18 + post-avgpool 512-d output + identity-init projection layer.

    ``forward(x)`` returns ``penultimate_proj(backbone(x))``, a 512-d tensor.
    BN is frozen via ``timm.layers.freeze_batch_norm_2d`` and the .train()
    override pins every BN-like submodule to eval to prevent running-stat
    drift. The penultimate_proj base weight is initialised to ``I`` and
    frozen; PEFT then attaches LoRA(A, B) to it for the LEACE warm-start.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone = timm.layers.freeze_batch_norm_2d(backbone)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone

        self.penultimate_proj = nn.Linear(512, 512, bias=False)
        with torch.no_grad():
            self.penultimate_proj.weight.copy_(torch.eye(512))
        for p in self.penultimate_proj.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D401 - intentional override
        super().train(mode)
        for m in self.modules():
            cls_name = type(m).__name__
            if "BatchNorm" in cls_name or "FrozenBatchNorm" in cls_name:
                m.eval()
        return self

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone-only forward (pre-projection): the 512-d post-avgpool features."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.penultimate_proj(h)


def build_resnet18_lora(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> tuple[nn.Module, "PeftModel"]:
    """Legacy single-Linear (penultimate_proj) architecture with LoRA + LEACE init."""
    if target_modules is None:
        target_modules = ["penultimate_proj"]
    wrapper = ResNet18Penultimate()
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    peft_model = get_peft_model(wrapper, config)
    return peft_model, peft_model


def build_resnet18_erase_task(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> tuple[nn.Module, "PeftModel"]:
    """Novel decoupled architecture: frozen erase layer + LoRA-on-task_proj."""
    wrapper = ResNet18EraseTaskLoRA()
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["task_proj"],
    )
    peft_model = get_peft_model(wrapper, config)
    return peft_model, peft_model


def get_erase_layer(peft_model: nn.Module) -> nn.Linear:
    """Locate the frozen erase nn.Linear inside the PEFT-wrapped model."""
    for name, m in peft_model.named_modules():
        if name.endswith(".erase") or name == "erase":
            if isinstance(m, nn.Linear):
                return m
    raise RuntimeError("erase Linear not found in peft_model")


def get_penultimate_proj_lora(peft_model: nn.Module) -> nn.Module:
    """Locate the PEFT LoRA wrapper around ``penultimate_proj``."""
    for name, m in peft_model.named_modules():
        if name.endswith("penultimate_proj"):
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                return m
    raise RuntimeError("penultimate_proj LoRA module not found")


def get_backbone_only(peft_model: nn.Module) -> nn.Module:
    """Locate the inner ResNet18 module for backbone-only forwards."""
    for name, m in peft_model.named_modules():
        if name.endswith(".backbone") or name == "backbone":
            return m
    # Fall back to the wrapper's backbone attribute path.
    return peft_model.base_model.model.backbone


def trainable_param_summary(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (name, tuple(p.shape))
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def bn_state_summary(model: nn.Module) -> list[tuple[str, str, bool]]:
    out: list[tuple[str, str, bool]] = []
    for name, m in model.named_modules():
        cls = type(m).__name__
        if "BatchNorm" in cls or "FrozenBatchNorm" in cls:
            out.append((name, cls, bool(m.training)))
    return out


def iter_lora_params(model: nn.Module) -> Iterable[nn.Parameter]:
    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            yield p

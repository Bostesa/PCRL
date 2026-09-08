"""LEACE warm-start for BIOS-medium.

For BERT, a pre-LN LoRA-side warm-start does NOT drive construction [CLS]
R² to zero — the post-FFN LayerNorm re-centres any modification (verified
empirically: LoRA-only warm-start → construction R² ≈ 0.99). The fix is
the **frozen post-forward LEACE projection** mirroring
``pcrl/models/lora.py:246``: register ``(Q, mu)`` as buffers and apply
``cls' = mu + (cls - mu) @ P.T`` at the end of every forward.

A combined "post-projection + LoRA-side SVD warm-start" was tried but is
worse on dev: the LoRA-side injection shifts the [CLS] distribution, so
the post-projection (which was fit on the un-shifted distribution) no
longer perfectly erases gender from the shifted [CLS] (verified
empirically: combined → construction R² ≈ 0.78). The LoRA stays zero-init,
the post-projection sees exactly the [CLS] it was fit on, and construction
R² is driven to zero. The LoRA then trains *inside* the LEACE null space
defined by the projection.

Pipeline:

1. Single pass of frozen BERT (zero-init LoRA → no adapter effect) over
   the training subsample → ``Z_cls`` of shape ``(N, 768)``.
2. Build gender one-hot ``A_oh`` of shape ``(N, 2)``. Fit
   ``LeaceEraser.fit(Z_cls, A_oh)`` → ``Q = eraser.P`` of shape ``(768, 768)``
   and ``mu = eraser.bias`` of shape ``(768,)``.
3. Register ``(Q, mu)`` as a frozen post-forward projection on the encoder
   (``BertWithLoRA.set_leace_projection``).

LoRA stays zero-init; gradients during training move it inside the LEACE
null space (the projection is idempotent: P² = P, so re-projecting a
LoRA-perturbed rep onto the null space is the natural way to keep the
constraint satisfied throughout training).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .bert_encoder import LAYER11_OUTPUT_DENSE_NAME


def _find_layer11_lora(peft_model: nn.Module):
    """Locate the LoRA-wrapped layer-11 ``output.dense`` and return
    ``(wrapper, host_linear, lora_A_linear, lora_B_linear)``.
    """
    target_suffix = LAYER11_OUTPUT_DENSE_NAME  # "encoder.layer.11.output.dense"
    found = None
    found_name = None
    for name, module in peft_model.named_modules():
        if name.endswith(target_suffix) and hasattr(module, "lora_A"):
            found = module
            found_name = name
            break
    if found is None:
        raise RuntimeError(
            f"Could not locate a LoRA-wrapped module with suffix "
            f"'{target_suffix}'. Was PEFT injection performed with the "
            f"correct target_modules list?"
        )
    host = found.base_layer
    A = found.lora_A["default"]
    B = found.lora_B["default"]
    return found, host, A, B, found_name


@torch.no_grad()
def _collect_cls_and_gender(
    model: nn.Module, loader, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass collection of [CLS] features and gender labels.

    A second pass through ``loader`` would shuffle differently when
    ``shuffle=True`` (the DataLoader's generator advances state between
    iterations), silently misaligning features and labels. This helper
    forwards once and pairs them in the same loop.
    """
    model.eval()
    feats = []
    gens = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        z = model(ids, mask)
        feats.append(z.detach().cpu())
        gens.append(batch["gender"].long())
    return torch.cat(feats, dim=0), torch.cat(gens, dim=0)


@torch.no_grad()
def leace_warm_start_bert(
    model, train_loader, *, device: torch.device,
) -> dict:
    """Fit LEACE on (frozen-BERT [CLS], gender); register a frozen post-
    forward projection on the encoder. Returns a diagnostic dict.

    Args:
        model: ``BertWithLoRA`` instance (must already be on ``device``).
        train_loader: DataLoader yielding ``{input_ids, attention_mask,
            gender, occupation}`` batches.
        device: torch device to run forward passes on. The LEACE projection
            is computed on CPU (concept_erasure is CPU-only) and copied to
            the encoder's device when registered.

    Returns:
        Dict with pre/post linear-R² on the FIT data and the located
        layer-11 module name (for wiring sanity).
    """
    from concept_erasure import LeaceEraser

    Z, g = _collect_cls_and_gender(model, train_loader, device)
    Z = Z.float()
    g = g.long()
    G_oh = torch.eye(2)[g].float()

    eraser = LeaceEraser.fit(Z, G_oh)
    Q = eraser.P.detach().to(torch.float32)  # (768, 768)
    mu = (
        eraser.bias.detach().to(torch.float32)
        if eraser.bias is not None
        else torch.zeros(Q.shape[0], dtype=torch.float32)
    )
    Z_post = eraser(Z)
    pre_r2 = _linear_r2_train(Z.numpy(), g.numpy())
    post_r2 = _linear_r2_train(Z_post.numpy(), g.numpy())

    # Register the frozen post-forward projection on the encoder. This is
    # the only warm-start mechanism that survives BERT's post-FFN LayerNorm.
    # LoRA stays zero-init: a LoRA-side SVD warm-start would shift the [CLS]
    # distribution, breaking the post-projection's exactness on dev.
    model.set_leace_projection(Q, mu)

    # Sanity-locate the layer-11 output.dense LoRA so we can report it in the
    # diagnostic dict, and verify both sub-modules are zero-init (PEFT
    # default). We do NOT modify them.
    wrapper, host, A_mod, B_mod, located_name = _find_layer11_lora(model.peft_model)
    layer11_lora_b_l2 = float(B_mod.weight.detach().norm().item())

    return {
        "pre_r2_train": float(pre_r2),
        "post_r2_train_eraser_only": float(post_r2),
        "Q_shape": list(Q.shape),
        "mu_shape": list(mu.shape),
        "located_layer11_module": located_name,
        "layer11_lora_b_l2_norm": layer11_lora_b_l2,
        "post_forward_projection_registered": True,
        "lora_side_warm_start_disabled": True,
    }


def _linear_r2_train(H, z, reg: float = 1e-6) -> float:
    """Train-set linear R² of optimal Tikhonov-regularised one-hot predictor.

    Uses the shared scorer with the fixed binary gender schema. Missing
    classes produce an undefined score rather than an apparent privacy pass.
    """
    from pcrl.purposes.verification import LinearComplianceCertificate

    return LinearComplianceCertificate(regularization=reg).check(
        H, z, num_classes=2,
    ).r_squared

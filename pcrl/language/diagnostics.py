"""Pre-launch diagnostics for BIOS-medium.

The four print-outs the spec gates Phase-1 launch on:

1. :func:`describe_modules` — named modules with ``requires_grad`` after PEFT
   injection, plus total trainable / frozen / ratio counts. Should be ~1-2%.
2. :func:`cls_shape_trace` — input_ids → last_hidden_state → ``[:, 0, :]`` →
   768 shape trace, including a forward through the model on a small batch.
3. :func:`bio_length_by_gender` — token-count median, p90, p99 stratified by
   gender; per-gender truncation rate at ``max_length``; gap-between-genders
   in pp; warning + suggested ``seq_len=256`` alternative if female - male
   truncation rate > 2 pp.
4. :func:`construction_r2` — held-out 1024-sample-batch linear R² of [CLS]
   on gender after LEACE warm-start. Aborts via ``RuntimeError`` if R² >
   ``abort_threshold`` (default 0.05).

All four are inference-only and Mac/MPS-compatible.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .bert_encoder import count_param_breakdown


def describe_modules(model: nn.Module, *, max_examples: int = 6) -> dict:
    """Summarise PEFT injection: trainable param count, frozen count, ratio,
    plus a sampling of LoRA module names so the structure is human-checkable.
    """
    breakdown = count_param_breakdown(model)
    lora_names: list[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_A" in name and "weight" in name:
            lora_names.append(name)
    sample = lora_names[:max_examples]
    if len(lora_names) > max_examples:
        sample.append(f"... ({len(lora_names) - max_examples} more)")
    return {
        **breakdown.as_dict(),
        "lora_module_names_sample": sample,
        "lora_module_names_total": len(lora_names),
    }


@torch.no_grad()
def cls_shape_trace(
    model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device,
) -> dict:
    """Trace input_ids → last_hidden_state → [CLS] → repr shape on one batch.

    Args:
        model: ``BertWithLoRA`` instance.
        batch: dict with ``input_ids`` and ``attention_mask`` (each shape
            ``(B, L)``). The function uses up to the first 4 rows.
        device: torch device.

    Returns:
        Dict with shapes at each step.
    """
    model.eval()
    ids = batch["input_ids"][:4].to(device)
    mask = batch["attention_mask"][:4].to(device)
    out = model.peft_model(input_ids=ids, attention_mask=mask, return_dict=True)
    lhs = out.last_hidden_state
    cls = lhs[:, 0, :]
    z_via_forward = model(ids, mask)
    return {
        "input_ids_shape": list(ids.shape),
        "attention_mask_shape": list(mask.shape),
        "last_hidden_state_shape": list(lhs.shape),
        "cls_after_indexing_shape": list(cls.shape),
        "model_forward_repr_shape": list(z_via_forward.shape),
        "matches": bool(torch.allclose(cls, z_via_forward, atol=1e-6)),
    }


def bio_length_by_gender(
    texts: Iterable[str],
    genders: Iterable[int],
    tokenizer,
    *,
    max_length: int = 128,
) -> dict:
    """Tokenise without truncation to get true lengths; compute per-gender
    median, p90, p99, and truncation rate at ``max_length``.

    Returns a dict including a ``warning`` field when
    ``female_truncation_rate − male_truncation_rate > 2 pp``.
    """
    encoded = tokenizer(
        list(texts),
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    lengths = np.array(
        [len(ids) for ids in encoded["input_ids"]], dtype=np.int64,
    )
    g = np.asarray(list(genders), dtype=np.int64)
    out: dict = {"max_length": max_length, "n_total": int(len(lengths))}
    for gid, name in [(0, "male"), (1, "female")]:
        L = lengths[g == gid]
        if len(L) == 0:
            out[name] = {
                "n": 0, "median": None, "p90": None, "p99": None,
                "truncation_rate": None,
            }
            continue
        out[name] = {
            "n": int(len(L)),
            "median": int(np.median(L)),
            "p90": int(np.percentile(L, 90)),
            "p99": int(np.percentile(L, 99)),
            "truncation_rate": float((L > max_length).mean()),
        }
    f_rate = out["female"]["truncation_rate"]
    m_rate = out["male"]["truncation_rate"]
    if f_rate is not None and m_rate is not None:
        gap_pp = (f_rate - m_rate) * 100.0
        out["truncation_gap_pp_female_minus_male"] = gap_pp
        if gap_pp > 2.0:
            out["warning"] = (
                f"Female truncation rate exceeds male by "
                f"{gap_pp:.2f} pp (>2 pp threshold). Consider running with "
                f"seq_len=256 to reduce gender-asymmetric information loss "
                f"(potential confound for gender-leakage estimates). "
                f"This change requires user approval — defaulting to "
                f"max_length={max_length} for now."
            )
        else:
            out["warning"] = None
    return out


@torch.no_grad()
def construction_r2(
    model: nn.Module,
    held_out_loader,
    device: torch.device,
    *,
    max_samples: int | None = None,
    abort_threshold: float = 0.05,
    decimal_places: int = 4,
) -> dict:
    """Linear R² of held-out [CLS] on gender after LEACE warm-start.

    Pulls up to ``max_samples`` (default: all available) from
    ``held_out_loader`` (a DataLoader over the *dev* split — disjoint from
    the LEACE-fitting subsample), runs the model in eval mode, and computes
    closed-form OLS R² of one-hot(gender) on centered [CLS] with ridge ``1e-6``.

    If ``r2 > abort_threshold``, raises ``RuntimeError``: the LEACE warm-start
    failed and Phase 1 must not be launched.

    Note on sample size: at d=768 the unregularised in-sample OLS R² has a
    known overfitting bias of approximately ``d/N`` (verified empirically:
    pure-noise R² ≈ 0.73 at N=1024, ≈ 0.024 at N=31_764). The default
    ``max_samples=None`` uses the entire held-out loader so the threshold
    ≤ 0.05 is actually reachable. Setting a small ``max_samples`` makes the
    threshold structurally unreachable.
    """
    model.eval()
    cls_blocks: list[torch.Tensor] = []
    g_blocks: list[torch.Tensor] = []
    n_collected = 0
    cap = max_samples if max_samples is not None else float("inf")
    for batch in held_out_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        z = model(ids, mask).detach().cpu()
        cls_blocks.append(z)
        g_blocks.append(batch["gender"].long())
        n_collected += z.shape[0]
        if n_collected >= cap:
            break
    Z_full = torch.cat(cls_blocks, dim=0)
    g_full = torch.cat(g_blocks, dim=0)
    if max_samples is not None:
        Z_full = Z_full[:max_samples]
        g_full = g_full[:max_samples]
    Z = Z_full.float().numpy()
    g = g_full.numpy().astype(np.int64)

    # Closed-form ridge R² on one-hot(gender). Identical formula to
    # pcrl.training.v2_trainer._linear_r2_train.
    n_classes = 2
    Z_oh = np.eye(n_classes)[g].astype(np.float64)
    H = Z.astype(np.float64)
    H_c = H - H.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    gram = H_c.T @ H_c + 1e-6 * np.eye(H_c.shape[1])
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    r2 = float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))
    r2_rounded = round(r2, decimal_places)

    n = int(Z.shape[0])
    ols_overfit_bias_estimate = Z.shape[1] / n  # ≈ pure-noise in-sample R²
    out = {
        "r2": r2,
        "r2_rounded": r2_rounded,
        "n_held_out": n,
        "d": int(Z.shape[1]),
        "ols_overfit_bias_estimate": float(ols_overfit_bias_estimate),
        "abort_threshold": abort_threshold,
        "passed": r2 <= abort_threshold,
    }
    if r2 > abort_threshold:
        raise RuntimeError(
            f"Construction-time linear-R²([CLS], gender) = "
            f"{r2_rounded:.{decimal_places}f} exceeds abort threshold "
            f"{abort_threshold} (n_held_out={n}, d={Z.shape[1]}, "
            f"OLS noise bias ≈ d/N = {ols_overfit_bias_estimate:.4f}). "
            f"The LEACE warm-start did not take effect — the wrapper has a "
            f"bug. Aborting before Phase 1 launch."
        )
    return out

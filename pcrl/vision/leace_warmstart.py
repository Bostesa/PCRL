"""LEACE warm-start for ResNet18 + Linear LoRA on penultimate_proj.

Pipeline:
    1. With LoRA at zero-init (delta = 0, base weight = I), collect penultimate
       features ``H in R^{N,512}`` from ``backbone(x)`` and concept matrix
       ``A_oh in R^{N,4}`` (one-hot Male and Young concatenated).
    2. Fit ``LeaceEraser.fit(H, A_oh)``. Pull ``proj_left, proj_right``.
    3. Compute multiplicative correction ``M = -proj_left @ proj_right``,
       so that ``Q = I + M`` is the LEACE projection.
    4. SVD of M; truncate to rank-r; absorb into Linear-LoRA A,B on
       penultimate_proj. With base weight = I and LoRA delta = M, the
       effective transform is ``W + delta = Q``.
    5. Verify by re-collecting features through the now-LoRA'd path and
       computing per-attribute linear-R^2.

Linear LoRA injection math:
    PEFT Linear LoRA forward (bias='none'):
        y = base.weight @ x + scaling * (lora_B.weight @ lora_A.weight) @ x
    With base.weight = I and scaling = alpha / r, we want:
        scaling * lora_B.weight @ lora_A.weight = M
    Setting:
        lora_A.weight[k, :]  = V[k, :]                    # shape (r, 512)
        lora_B.weight[:, k]  = U[:, k] * S[k] / scaling   # shape (512, r)
    yields  scaling * (U S / scaling) @ V = U S V = M.    (Bias dropped:
    linear-R^2 is shift-invariant.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from concept_erasure import LeaceEraser
from torch.utils.data import DataLoader

from pcrl.vision.backbone import (
    get_backbone_only,
    get_erase_layer,
    get_penultimate_proj_lora,
)
from pcrl.vision.r2_helper import linear_r2


@dataclass
class LeaceDiagnostics:
    n_samples: int
    rank_M: int
    singular_values_full: np.ndarray
    truncation_residual: float
    pre_r2: dict[str, float]
    post_r2_closed: dict[str, float]
    construction_r2: dict[str, float]
    leace_proj_left_shape: tuple[int, int]
    leace_proj_right_shape: tuple[int, int]


def _make_concept_matrix(
    male: torch.Tensor, young: torch.Tensor
) -> torch.Tensor:
    """One-hot concat: (N, 4). Columns: [Male=0, Male=1, Young=0, Young=1]."""
    N = male.shape[0]
    oh = torch.zeros(N, 4, dtype=torch.float32)
    oh[torch.arange(N), male.long()] = 1.0
    oh[torch.arange(N), 2 + young.long()] = 1.0
    return oh


@torch.no_grad()
def collect_penultimate_pre(
    peft_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward through the **backbone only** (no penultimate_proj) and return
    ``(H, male, young)`` on CPU. ``H`` shape (N, 512).

    Used for fitting LEACE in the same space the eraser will be evaluated.
    """
    peft_model.eval()
    backbone = get_backbone_only(peft_model)
    H_list: list[torch.Tensor] = []
    male_list: list[torch.Tensor] = []
    young_list: list[torch.Tensor] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        h = backbone(x).detach().cpu()
        H_list.append(h)
        male_list.append(batch["male"].cpu())
        young_list.append(batch["young"].cpu())
    return torch.cat(H_list, 0), torch.cat(male_list, 0), torch.cat(young_list, 0)


@torch.no_grad()
def collect_penultimate_post(
    peft_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward through the **full pipeline** (backbone + penultimate_proj)
    and return projected features."""
    peft_model.eval()
    H_list: list[torch.Tensor] = []
    male_list: list[torch.Tensor] = []
    young_list: list[torch.Tensor] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        h = peft_model(x).detach().cpu()
        H_list.append(h)
        male_list.append(batch["male"].cpu())
        young_list.append(batch["young"].cpu())
    return torch.cat(H_list, 0), torch.cat(male_list, 0), torch.cat(young_list, 0)


def fit_leace(
    H: torch.Tensor, male: torch.Tensor, young: torch.Tensor
) -> tuple[LeaceEraser, torch.Tensor]:
    """Fit ``LeaceEraser`` and return ``(eraser, M)`` where ``M = Q - I``."""
    A_oh = _make_concept_matrix(male, young)
    eraser = LeaceEraser.fit(H.float(), A_oh)
    pl = eraser.proj_left.detach()    # (d, k)
    pr = eraser.proj_right.detach()   # (k, d)
    M = -(pl @ pr)
    return eraser, M


@torch.no_grad()
def init_penultimate_proj_lora(
    peft_model: nn.Module, M: torch.Tensor, rank: int, alpha: int,
) -> tuple[int, np.ndarray, float]:
    """SVD-truncate M and write into the penultimate_proj Linear LoRA.

    PEFT Linear LoRA stores:
        lora_A[adapter].weight  shape (r, in=512)
        lora_B[adapter].weight  shape (out=512, r)
    Forward: y = base(x) + scaling * lora_B(lora_A(x)),
             scaling = alpha / r.
    """
    target = get_penultimate_proj_lora(peft_model)
    adapter = next(iter(target.lora_A.keys()))
    A_layer = target.lora_A[adapter]
    B_layer = target.lora_B[adapter]

    U, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    if S.numel():
        s_max = float(S[0])
        tol = max(1e-10, 1e-8 * s_max)
        rank_M = int((S > tol).sum().item())
    else:
        rank_M = 0
    r_eff = min(rank, rank_M, S.numel())

    U_r = U[:, :r_eff].float()      # (512, r_eff)
    S_r = S[:r_eff].float()         # (r_eff,)
    V_r = Vh[:r_eff, :].float()     # (r_eff, 512)
    M_recon = (U_r * S_r) @ V_r
    M_norm = float(torch.linalg.norm(M.float())) + 1e-12
    residual = float(torch.linalg.norm(M.float() - M_recon)) / M_norm

    scaling = alpha / rank

    A_w = torch.zeros_like(A_layer.weight.data)   # (r, 512)
    if r_eff > 0:
        A_w[:r_eff, :] = V_r
    A_layer.weight.data.copy_(A_w)

    B_w = torch.zeros_like(B_layer.weight.data)   # (512, r)
    if r_eff > 0:
        B_w[:, :r_eff] = U_r * (S_r / scaling)
    B_layer.weight.data.copy_(B_w)

    top_k = min(8, S.numel())
    return rank_M, S[:top_k].cpu().numpy(), residual


@torch.no_grad()
def refit_leace_lora(
    peft_model: nn.Module,
    refit_loader: DataLoader,
    device: torch.device,
    rank: int,
    alpha: int,
) -> tuple[int, np.ndarray, float]:
    """C3: re-fit LEACE on CURRENT post-projection features and compose into LoRA.

    Forward pass over ``refit_loader`` collects ``h_proj = W_old @ x`` (current
    LoRA-applied features). Fit LEACE on those: gives ``Q_proj = I + M_proj``
    that erases concept from h_proj. Compose: ``W_new = Q_proj @ W_old``, so
    ``M_new = W_new - I``. SVD-truncate and overwrite the LoRA.
    """
    h_proj, male, young = collect_penultimate_post(peft_model, refit_loader, device)
    _, M_proj = fit_leace(h_proj, male, young)  # (d, d), M_proj = Q_proj - I

    target = get_penultimate_proj_lora(peft_model)
    adapter = next(iter(target.lora_A.keys()))
    A_layer = target.lora_A[adapter]
    B_layer = target.lora_B[adapter]
    d = A_layer.weight.shape[1]
    scaling = alpha / rank

    A_old = A_layer.weight.data.detach().cpu().float()  # (r, d)
    B_old = B_layer.weight.data.detach().cpu().float()  # (d, r)
    delta_old = scaling * (B_old @ A_old)               # (d, d)
    I_d = torch.eye(d, dtype=torch.float32)
    W_old = I_d + delta_old
    M_proj_f = M_proj.float()
    W_new = (I_d + M_proj_f) @ W_old
    M_new = W_new - I_d

    return init_penultimate_proj_lora(peft_model, M_new, rank=rank, alpha=alpha)


@torch.no_grad()
def fit_and_set_erase_layer(
    peft_model: nn.Module,
    fit_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
) -> LeaceDiagnostics:
    """Novel architecture warmstart: fit LeaceEraser on backbone features and
    write the projection into the frozen erase Linear layer.

    Math: LeaceEraser computes ``y = (x - bias) @ W_r + bias`` where
    ``W_r = I - (proj_left @ proj_right).T``. Linear with parameters
    ``W, b`` computes ``y = x @ W.T + b``. To match the eraser:
        W = W_r.T = I - proj_left @ proj_right
        b = bias - bias @ W_r.T = bias @ (proj_left @ proj_right).T
    """
    H, male, young = collect_penultimate_pre(peft_model, fit_loader, device)
    H_np = H.numpy()
    pre = {"male": linear_r2(H_np, male.numpy()), "young": linear_r2(H_np, young.numpy())}

    A_oh = _make_concept_matrix(male, young)
    eraser = LeaceEraser.fit(H.float(), A_oh)

    pl = eraser.proj_left.detach()           # (d, k)
    pr = eraser.proj_right.detach()          # (k, d)
    d = pl.shape[0]
    Q = torch.eye(d, dtype=pl.dtype) - pl @ pr  # (d, d)
    if hasattr(eraser, "bias") and eraser.bias is not None:
        center = eraser.bias.detach()
    else:
        center = torch.zeros(d, dtype=pl.dtype)
    b = (center @ pr.T) @ pl.T               # mu @ (pl @ pr).T

    erase = get_erase_layer(peft_model)
    erase.weight.data.copy_(Q.float().to(erase.weight.device))
    erase.bias.data.copy_(b.float().to(erase.bias.device))

    H_erased = eraser(H.float())
    post_closed = {
        "male": linear_r2(H_erased.numpy(), male.numpy()),
        "young": linear_r2(H_erased.numpy(), young.numpy()),
    }

    H2, m2, y2 = collect_penultimate_post(peft_model, probe_loader, device)
    constr = {
        "male": linear_r2(H2.numpy(), m2.numpy()),
        "young": linear_r2(H2.numpy(), y2.numpy()),
    }

    M = -(pl @ pr)
    sing = torch.linalg.svdvals(M.float()).cpu().numpy()
    rank_M = int((torch.linalg.svdvals(M.float()) > 1e-8 * sing[0]).sum().item()) if sing.size else 0

    return LeaceDiagnostics(
        n_samples=H.shape[0],
        rank_M=rank_M,
        singular_values_full=sing,
        truncation_residual=0.0,  # exact projection (no LoRA truncation)
        pre_r2=pre,
        post_r2_closed=post_closed,
        construction_r2=constr,
        leace_proj_left_shape=tuple(pl.shape),
        leace_proj_right_shape=tuple(pr.shape),
    )


def warm_start(
    peft_model: nn.Module,
    fit_loader: DataLoader,
    probe_loader: DataLoader,
    device: torch.device,
    rank: int,
    alpha: int,
) -> LeaceDiagnostics:
    """Full LEACE warm-start: collect (backbone-only) -> fit -> SVD-init
    penultimate_proj LoRA -> re-probe linear-R^2 (full pipeline).
    """
    H, male, young = collect_penultimate_pre(peft_model, fit_loader, device)
    H_np = H.numpy()
    pre = {
        "male": linear_r2(H_np, male.numpy()),
        "young": linear_r2(H_np, young.numpy()),
    }
    eraser, M = fit_leace(H, male, young)
    H_erased = eraser(H.float())
    H_erased_np = H_erased.numpy()
    post_closed = {
        "male": linear_r2(H_erased_np, male.numpy()),
        "young": linear_r2(H_erased_np, young.numpy()),
    }
    rank_M, top_s, residual = init_penultimate_proj_lora(
        peft_model, M, rank=rank, alpha=alpha,
    )

    # Re-probe through the **full pipeline** (backbone + penultimate_proj).
    H2, male2, young2 = collect_penultimate_post(peft_model, probe_loader, device)
    H2_np = H2.numpy()
    constr = {
        "male": linear_r2(H2_np, male2.numpy()),
        "young": linear_r2(H2_np, young2.numpy()),
    }

    return LeaceDiagnostics(
        n_samples=H.shape[0],
        rank_M=rank_M,
        singular_values_full=torch.linalg.svdvals(M.float()).cpu().numpy(),
        truncation_residual=residual,
        pre_r2=pre,
        post_r2_closed=post_closed,
        construction_r2=constr,
        leace_proj_left_shape=tuple(eraser.proj_left.shape),
        leace_proj_right_shape=tuple(eraser.proj_right.shape),
    )

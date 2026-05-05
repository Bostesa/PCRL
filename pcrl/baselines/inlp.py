"""Iterative null-space projection (INLP) baseline.

Standard INLP recipe:
1. Fit a vanilla encoder on the task (purely supervised; no fairness term).
2. Extract train+test representations h.
3. Iterate: train multinomial Logistic Regression on (h, attribute);
   project h onto the null-space of the LR weight matrix; repeat until
   the LR can no longer separate the attribute (within ``tol_pp`` of
   the test-split majority baseline) or ``max_iters``.
4. For multi-attribute purposes (e.g., race + sex), iterate over attributes
   in order of decreasing cardinality.
5. Train a fresh task head on the projected reprs to recover task accuracy.

The eval pipeline (``compute_dominant_axis_r2``, ``compute_mlp_ovr_delta``,
``LinearComplianceCertificate``) consumes ``encoder(x, purpose_idx)`` →
representation. We expose an :class:`INLPEncoder` wrapper that applies the
accumulated projection ``P`` after the underlying StandardEncoder forward,
so the existing eval helpers work unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.linear_model import LogisticRegression


class INLPEncoder(nn.Module):
    """Wraps a frozen :class:`StandardEncoder` with a fixed projection matrix.

    The projection ``P`` (d×d) is computed offline and stored as a buffer.
    Forward applies ``h @ P`` after the underlying encoder so downstream eval
    sees a ``(B, repr_dim)`` representation just like PCRL.
    """

    def __init__(self, backbone: nn.Module, projection: np.ndarray | torch.Tensor):
        super().__init__()
        self.backbone = backbone
        if isinstance(projection, np.ndarray):
            P = torch.from_numpy(projection.astype(np.float32))
        else:
            P = projection.float()
        self.register_buffer("projection", P)

    def forward(
        self,
        x: torch.Tensor,
        purpose_idx: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        h = self.backbone(x, purpose_idx)
        return h @ self.projection


@dataclass
class INLPDiagnostics:
    per_attr_iterations: dict[str, int] = field(default_factory=dict)
    per_attr_history: dict[str, list[dict]] = field(default_factory=dict)
    per_attr_majority: dict[str, float] = field(default_factory=dict)
    final_lr_acc: dict[str, float] = field(default_factory=dict)


def _null_space_projector(W: np.ndarray, sv_tol: float = 1e-10) -> np.ndarray:
    """Projector onto the null-space of ``W`` (rows span the to-remove subspace).

    For multinomial LR ``W`` is (K, d); for binary LR (1, d). We SVD W and
    take the right-singular vectors corresponding to non-trivial singular
    values; those are the directions we want to remove.
    """
    if W.size == 0:
        return np.eye(W.shape[1])
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = int((S > sv_tol).sum())
    if rank == 0:
        return np.eye(W.shape[1])
    V = Vt[:rank].T  # (d, rank)
    return np.eye(W.shape[1]) - V @ V.T


def _train_lr_one_iter(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Train multinomial LR on (h_train, y_train); return (W, test_accuracy)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
        )
        clf.fit(h_train, y_train)
        acc = float(clf.score(h_test, y_test))
    # clf.coef_ has shape (1, d) for binary, (K, d) for multiclass. Either is
    # the right thing to feed into _null_space_projector.
    return clf.coef_.copy(), acc


def inlp_iterate_one_attr(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_iters: int = 32,
    tol_pp: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[dict], float, float]:
    """Iteratively project out attribute-predictive directions.

    Stops when test accuracy is within ``tol_pp`` percentage points of the
    test-split majority baseline, or after ``max_iters``. Returns:
      • P: cumulative (d, d) projection in the *current* feature space
      • history: list of {iter, lr_test_acc} dicts
      • majority: test-split majority baseline (float in [0, 1])
      • final_acc: LR test accuracy after the last iteration
    """
    d = h_train.shape[1]
    majority = float(np.bincount(y_test).max() / max(len(y_test), 1))

    P = np.eye(d, dtype=np.float64)
    h_tr = h_train.astype(np.float64).copy()
    h_te = h_test.astype(np.float64).copy()

    history: list[dict] = []
    final_acc = majority
    for it in range(max_iters):
        W, acc = _train_lr_one_iter(h_tr, y_train, h_te, y_test, seed=seed + it)
        history.append({"iter": it, "lr_test_acc": float(acc)})
        final_acc = float(acc)
        # Stop *before* projecting if we're already within tolerance.
        if (acc - majority) * 100.0 < tol_pp:
            break
        P_step = _null_space_projector(W)
        h_tr = h_tr @ P_step
        h_te = h_te @ P_step
        P = P @ P_step

    return P.astype(np.float32), history, majority, final_acc


def _extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    device: torch.device,
    task_name: str,
    sensitive_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Single-pass extract of (h, y_task, sensitive_dict)."""
    encoder.eval()
    H, T = [], []
    S: dict[str, list[np.ndarray]] = {a: [] for a in sensitive_names}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx).detach().cpu().numpy()
            H.append(h)
            T.append(batch["task_labels"][task_name].numpy())
            for a in sensitive_names:
                S[a].append(batch["sensitive_attrs"][a].numpy())
    H = np.concatenate(H, axis=0)
    T = np.concatenate(T, axis=0)
    S = {a: np.concatenate(v, axis=0) for a, v in S.items()}
    return H, T, S


def _train_vanilla_encoder(
    encoder: nn.Module,
    task_head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    purpose_idx: int,
    task_name: str,
    epochs: int,
    lr: float,
    device: torch.device,
    log_every: int = 25,
    log_fn=print,
    patience: int = 20,
) -> dict:
    """Train encoder + task head end-to-end on task loss only (no fairness)."""
    enc_params = list(encoder.parameters()) + list(task_head.parameters())
    opt = torch.optim.Adam(enc_params, lr=lr)

    best_val = float("inf")
    best_state: dict | None = None
    bad = 0
    history = {"train_loss": [], "val_loss": [], "best_epoch": -1}

    for epoch in range(epochs):
        encoder.train(); task_head.train()
        running = 0.0; n = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["task_labels"][task_name].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(task_head(encoder(x, purpose_idx)), y)
            loss.backward()
            opt.step()
            running += float(loss.item()); n += 1
        train_avg = running / max(n, 1)
        history["train_loss"].append(train_avg)

        encoder.eval(); task_head.eval()
        v_running = 0.0; v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y = batch["task_labels"][task_name].to(device)
                v_running += float(F.cross_entropy(task_head(encoder(x, purpose_idx)), y).item())
                v_n += 1
        val_avg = v_running / max(v_n, 1)
        history["val_loss"].append(val_avg)

        if val_avg < best_val:
            best_val = val_avg
            history["best_epoch"] = epoch
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "task_head": {k: v.detach().cpu().clone() for k, v in task_head.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1

        if (epoch + 1) % log_every == 0 or epoch == 0:
            log_fn(f"  [inlp pretrain ep {epoch + 1:3d}/{epochs}] train={train_avg:.4f} val={val_avg:.4f}")

        if bad >= patience:
            log_fn(f"  [inlp pretrain] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
        task_head.load_state_dict({k: v.to(device) for k, v in best_state["task_head"].items()})

    return history


def _train_post_proj_task_head(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_val: np.ndarray,
    y_val: np.ndarray,
    *,
    repr_dim: int,
    num_classes: int,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
    log_fn=print,
    patience: int = 20,
) -> tuple[nn.Module, dict]:
    """Fit a fresh TaskHead on already-projected reprs (numpy in, module out).

    Returns (task_head_module_with_best_weights, history).
    """
    from pcrl.models.task_head import TaskHead

    torch.manual_seed(seed)
    task_head = TaskHead(repr_dim=repr_dim, output_dim=num_classes).to(device)
    opt = torch.optim.Adam(task_head.parameters(), lr=lr)

    Xtr = torch.from_numpy(h_train.astype(np.float32)).to(device)
    Ytr = torch.from_numpy(y_train.astype(np.int64)).to(device)
    Xv = torch.from_numpy(h_val.astype(np.float32)).to(device)
    Yv = torch.from_numpy(y_val.astype(np.int64)).to(device)

    n = Xtr.shape[0]
    bs = 256
    history = {"train_loss": [], "val_loss": [], "best_epoch": -1}
    best_val = float("inf"); bad = 0
    best_state: dict | None = None

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        task_head.train()
        idx = rng.permutation(n)
        running = 0.0; nb = 0
        for s in range(0, n, bs):
            batch_idx = idx[s:s + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(task_head(Xtr[batch_idx]), Ytr[batch_idx])
            loss.backward(); opt.step()
            running += float(loss.item()); nb += 1
        history["train_loss"].append(running / max(nb, 1))

        task_head.eval()
        with torch.no_grad():
            v_loss = float(F.cross_entropy(task_head(Xv), Yv).item())
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            history["best_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in task_head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        task_head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return task_head, history


def run_inlp(
    encoder: nn.Module,
    task_head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    purpose_idx: int,
    task_name: str,
    num_task_classes: int,
    disallowed_attrs: list[str],
    attr_cardinalities: dict[str, int],
    repr_dim: int,
    pretrain_epochs: int = 200,
    pretrain_lr: float = 1e-3,
    inlp_max_iters: int = 32,
    inlp_tol_pp: float = 1.0,
    head_epochs: int = 200,
    head_lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    log_fn=print,
) -> tuple[INLPEncoder, nn.Module, INLPDiagnostics, dict]:
    """End-to-end INLP run.

    Steps:
      1. Pretrain encoder + task head on task loss (no fairness).
      2. Extract train + val features.
      3. For each disallowed attr in order of decreasing cardinality, run
         iterative null-space projection on the *current* features.
      4. Train a fresh task head on the projected features.

    Returns (inlp_encoder, post_proj_task_head, diagnostics, training_history).
    """
    device = torch.device(device)
    encoder = encoder.to(device)
    task_head = task_head.to(device)

    # ------ 1. Pretrain encoder + task head ------
    log_fn("  [inlp] pretraining encoder + task head (no fairness term)…")
    pre_hist = _train_vanilla_encoder(
        encoder, task_head, train_loader, val_loader,
        purpose_idx=purpose_idx, task_name=task_name,
        epochs=pretrain_epochs, lr=pretrain_lr, device=device, log_fn=log_fn,
    )

    # ------ 2. Extract features ------
    log_fn("  [inlp] extracting train / val features for projection…")
    h_tr, ytask_tr, attrs_tr = _extract_features(
        encoder, train_loader, purpose_idx, device, task_name, disallowed_attrs,
    )
    h_va, ytask_va, attrs_va = _extract_features(
        encoder, val_loader, purpose_idx, device, task_name, disallowed_attrs,
    )

    # ------ 3. INLP per attribute, largest-cardinality first ------
    diagnostics = INLPDiagnostics()
    P_total = np.eye(repr_dim, dtype=np.float64)
    h_tr_proj = h_tr.astype(np.float64).copy()
    h_va_proj = h_va.astype(np.float64).copy()

    order = sorted(disallowed_attrs, key=lambda a: -attr_cardinalities[a])
    log_fn(f"  [inlp] attribute order (largest cardinality first): {order}")
    for attr in order:
        log_fn(f"  [inlp] projecting out {attr} (K={attr_cardinalities[attr]})…")
        P_attr, history, majority, final_acc = inlp_iterate_one_attr(
            h_tr_proj, attrs_tr[attr], h_va_proj, attrs_va[attr],
            max_iters=inlp_max_iters, tol_pp=inlp_tol_pp, seed=seed,
        )
        diagnostics.per_attr_iterations[attr] = len(history)
        diagnostics.per_attr_history[attr] = history
        diagnostics.per_attr_majority[attr] = majority
        diagnostics.final_lr_acc[attr] = final_acc
        log_fn(
            f"  [inlp]   {attr}: iters={len(history)} "
            f"final_lr_acc={final_acc:.4f} majority={majority:.4f} "
            f"gap_pp={(final_acc - majority) * 100:.2f}"
        )
        h_tr_proj = h_tr_proj @ P_attr
        h_va_proj = h_va_proj @ P_attr
        P_total = P_total @ P_attr

    # ------ 4. Fresh task head on projected reprs ------
    log_fn("  [inlp] training post-projection task head…")
    post_head, head_hist = _train_post_proj_task_head(
        h_tr_proj, ytask_tr, h_va_proj, ytask_va,
        repr_dim=repr_dim, num_classes=num_task_classes,
        epochs=head_epochs, lr=head_lr, seed=seed, device=device, log_fn=log_fn,
    )

    inlp_encoder = INLPEncoder(encoder, P_total).to(device)
    history = {"pretrain": pre_hist, "post_proj_head": head_hist}
    return inlp_encoder, post_head, diagnostics, history

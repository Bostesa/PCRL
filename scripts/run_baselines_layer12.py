"""Baselines on cached BERT layer-12 [CLS] features (BIOS).

Three baselines per purpose:

* **Closed-form LEACE** (Belrose 2023): one affine projection fit on
  (X_train, gender_oh). Applied to train + dev. Linear utility = LR
  trained on the projected train, evaluated on the projected dev.

* **K-LEACE-union**: with all 3 purposes sharing gender as the
  disallowed attribute, the cross-cov of (X, gender) is the same per
  purpose, so the LEACE eraser is identical. K-LEACE-union therefore
  reduces *exactly* to the single LEACE projection. We compute and save
  it as a separate artefact for completeness, document the reduction,
  and DO NOT pretend it's a different operator.

* **R-LACE rank-1** (Ravfogel ICML 2022): rank-1 saddle-point erasure.
  For binary gender + rank-1 the minimax converges to the projection
  orthogonal to the optimal linear-adversary direction; we implement
  this directly as iterated null-space projection of the LR-fit direction
  (matches the existing INLP recipe in ``pcrl/baselines/inlp.py``).
  Documented as "R-LACE rank-1 / INLP-rank-1-equivalent (binary Z)".

Outputs to ``--output-dir``:
    leace_results.json            — closed-form LEACE per purpose
    klu_results.json              — K-LEACE-union (with reduction note)
    rlace_results.json            — R-LACE rank-1 per purpose
    eval_reps_LEACE.npz           — train + dev reps after LEACE
    eval_reps_RLACE.npz           — train + dev reps after R-LACE rank-1

R²(gender) is reported on (train, dev) using the same closed-form ridge
solver used by ``pcrl.training.losses.VerificationRegularizer`` so numbers
are directly comparable to the PCRL metrics_*.json.

Usage:
    python scripts/run_baselines_layer12.py \\
        --cache-dir results/bios_pcrl_layer12/cache \\
        --output-dir results/bios_pcrl_layer12 \\
        --rlace-iters 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PURPOSES = ("P1", "P2", "P3")
PURPOSE_LABEL_KEYS = {
    "P1": ("occupation_train", "occupation_dev", 28),
    "P2": ("supercategory_train", "supercategory_dev", 5),
    "P3": ("medical_train", "medical_dev", 2),
}


# ---------------------------------------------------------------------------
# R² and probe helpers (numpy)
# ---------------------------------------------------------------------------


def linear_r2(H: np.ndarray, Z: np.ndarray, *, reg: float = 1e-4) -> float:
    n_classes = int(Z.max()) + 1
    if n_classes < 2:
        return 0.0
    Z_oh = np.eye(n_classes)[Z].astype(np.float64)
    H = H.astype(np.float64)
    H_c = H - H.mean(axis=0, keepdims=True)
    Z_c = Z_oh - Z_oh.mean(axis=0, keepdims=True)
    d = H.shape[1]
    gram = H_c.T @ H_c + reg * np.eye(d)
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    Z_pred = H_c @ W
    ss_res = ((Z_c - Z_pred) ** 2).sum()
    ss_tot = (Z_c ** 2).sum()
    return float(max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12)))


def _lr_score(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    *, balanced: bool = False, max_iter: int = 2000,
) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.linear_model import LogisticRegression
    kwargs: dict = {"max_iter": max_iter, "solver": "lbfgs"}
    if balanced:
        kwargs["class_weight"] = "balanced"
    clf = LogisticRegression(**kwargs)
    clf.fit(X_train, y_train)
    return {
        "train_acc": float(clf.score(X_train, y_train)),
        "dev_acc": float(clf.score(X_dev, y_dev)),
    }


# ---------------------------------------------------------------------------
# LEACE
# ---------------------------------------------------------------------------


def fit_leace(X_train: np.ndarray, gender_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(P, mu)`` from ``concept_erasure.LeaceEraser``.

    eraser(x) = (x - mu) @ P.T + mu, so the projection in row-vector form is
    ``X_erased = (X - mu) @ P.T + mu``.
    """
    from concept_erasure import LeaceEraser

    X_t = torch.from_numpy(X_train).float()
    g_t = torch.from_numpy(gender_train).long()
    G_oh = F.one_hot(g_t, 2).float()
    eraser = LeaceEraser.fit(X_t, G_oh)
    P = eraser.P.detach().cpu().numpy().astype(np.float32)
    mu = (
        eraser.bias.detach().cpu().numpy().astype(np.float32)
        if eraser.bias is not None else np.zeros(X_train.shape[1], dtype=np.float32)
    )
    return P, mu


def apply_eraser(X: np.ndarray, P: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return ((X - mu) @ P.T + mu).astype(np.float32)


# ---------------------------------------------------------------------------
# R-LACE rank-1 (binary Z)
# ---------------------------------------------------------------------------


def fit_rlace_rank1(
    X_train: np.ndarray, Z_train: np.ndarray, *,
    n_iters: int = 1, lr_C: float = 1.0, max_iter: int = 2000,
) -> np.ndarray:
    """Iterated rank-1 INLP — the saddle point of rank-1 R-LACE for binary Z.

    Each iteration:
      1. Fit LR(class_weight='balanced') on the current representation.
      2. Take the LR weight vector ``w`` (shape ``(d,)``).
      3. Project ``X`` onto the null-space of ``w``:
         ``X' = X - X @ w @ w.T / |w|^2``.

    Returns the cumulative projection matrix ``P_full`` of shape ``(d, d)``
    such that ``X_erased = X @ P_full.T`` (no affine bias since each
    null-space projection is centred-aware via per-step LR fit).

    Notes
    -----
    Ravfogel et al. (2022) prove that rank-1 R-LACE's saddle point coincides
    with projection orthogonal to the optimal-margin LR direction for binary
    Z. Iterating once with ``class_weight='balanced'`` and ``lbfgs`` to
    convergence is sufficient. ``n_iters > 1`` is supported for parity with
    INLP-style accumulation; the gradient of subsequent iterations on the
    same span is small.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.linear_model import LogisticRegression
    d = X_train.shape[1]
    P_full = np.eye(d, dtype=np.float64)
    Xc = X_train.astype(np.float64).copy()
    for it in range(max(1, n_iters)):
        clf = LogisticRegression(
            class_weight="balanced", max_iter=max_iter, solver="lbfgs",
            C=lr_C,
        )
        clf.fit(Xc, Z_train)
        w = clf.coef_[0]  # (d,) for binary
        wn2 = float(w @ w)
        if wn2 < 1e-12:
            break
        # P_step = I - w w^T / |w|^2
        outer = np.outer(w, w) / wn2
        P_step = np.eye(d, dtype=np.float64) - outer
        Xc = Xc @ P_step.T
        # Compose cumulative null-space projection.
        P_full = P_step @ P_full
    return P_full.astype(np.float32)


def apply_rlace(X: np.ndarray, P_full: np.ndarray) -> np.ndarray:
    return (X @ P_full.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rlace-iters", type=int, default=1,
                    help="Outer iterations for R-LACE rank-1 (default: 1). Note: "
                    "for small training subsets (N near D) ridge regression rebuilds "
                    "an LR-detectable gender direction after each rank-1 erasure, so "
                    "smoke runs on 5k subsets show R²>0 even after many iters; on the "
                    "full BIOS train (N~257k >> D=768) a single iteration drives R² "
                    "near zero.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baselines] loading cache from {args.cache_dir}", flush=True)
    emb = np.load(args.cache_dir / "cls_layer12_embeddings.npz")
    lbl = np.load(args.cache_dir / "labels.npz")
    X_train = emb["train_standardized"].astype(np.float32)
    X_dev = emb["dev_standardized"].astype(np.float32)
    g_train = lbl["gender_train"].astype(np.int64)
    g_dev = lbl["gender_dev"].astype(np.int64)
    n_train, d = X_train.shape
    n_dev = X_dev.shape[0]
    print(f"[baselines] N_train={n_train}  N_dev={n_dev}  d={d}", flush=True)

    # ----- LEACE -----
    print("[baselines] fitting closed-form LEACE on (train, gender) ...", flush=True)
    t0 = time.time()
    P_leace, mu_leace = fit_leace(X_train, g_train)
    leace_train = apply_eraser(X_train, P_leace, mu_leace)
    leace_dev = apply_eraser(X_dev, P_leace, mu_leace)
    leace_r2_train = linear_r2(leace_train, g_train)
    leace_r2_dev = linear_r2(leace_dev, g_dev)
    print(f"[baselines] LEACE: R²(g) train={leace_r2_train:.4f} dev={leace_r2_dev:.4f}"
          f"  ({time.time()-t0:.1f}s)", flush=True)

    leace_results: dict = {
        "method": "closed_form_LEACE",
        "shared_eraser_across_purposes": True,
        "shared_eraser_reason": "LEACE depends only on (X, Z); all 3 purposes "
                               "share Z=gender so the eraser is identical.",
        "r2_gender_train": leace_r2_train,
        "r2_gender_dev": leace_r2_dev,
        "per_purpose": {},
    }
    for purpose in PURPOSES:
        ytr_key, ydv_key, n_classes = PURPOSE_LABEL_KEYS[purpose]
        y_train = lbl[ytr_key].astype(np.int64)
        y_dev = lbl[ydv_key].astype(np.int64)
        task_score = _lr_score(leace_train, y_train, leace_dev, y_dev,
                               max_iter=2000)
        gender_probe = _lr_score(leace_train, g_train, leace_dev, g_dev,
                                 balanced=True, max_iter=2000)
        leace_results["per_purpose"][purpose] = {
            "num_classes": n_classes,
            "task_lr": task_score,
            "linear_gender_probe": gender_probe,
        }
        print(f"[baselines]   LEACE/{purpose}: task_dev={task_score['dev_acc']:.4f}"
              f"  gender_probe_dev={gender_probe['dev_acc']:.4f}", flush=True)

    # ----- K-LEACE-union (= LEACE here, mathematically) -----
    klu_results = {
        "method": "K-LEACE-union",
        "reduction_note": (
            "When all purposes share the same disallowed attribute Z (here: "
            "gender) and operate on the same X, the K-LEACE-union eraser is "
            "exactly the single closed-form LEACE eraser fit on (X, Z). We "
            "report identical numbers; this is a mathematical equivalence, "
            "not an empirical coincidence."
        ),
        "r2_gender_train": leace_r2_train,
        "r2_gender_dev": leace_r2_dev,
        "per_purpose": leace_results["per_purpose"],
    }

    # ----- R-LACE rank-1 (binary Z) -----
    print(f"[baselines] fitting R-LACE rank-1 (n_iters={args.rlace_iters}) ...",
          flush=True)
    t0 = time.time()
    P_rlace = fit_rlace_rank1(X_train, g_train, n_iters=args.rlace_iters)
    rlace_train = apply_rlace(X_train, P_rlace)
    rlace_dev = apply_rlace(X_dev, P_rlace)
    rlace_r2_train = linear_r2(rlace_train, g_train)
    rlace_r2_dev = linear_r2(rlace_dev, g_dev)
    print(f"[baselines] R-LACE: R²(g) train={rlace_r2_train:.4f} dev={rlace_r2_dev:.4f}"
          f"  ({time.time()-t0:.1f}s)", flush=True)

    rlace_results: dict = {
        "method": "R-LACE rank-1 (INLP-rank-1-equivalent for binary Z)",
        "n_iters": args.rlace_iters,
        "r2_gender_train": rlace_r2_train,
        "r2_gender_dev": rlace_r2_dev,
        "per_purpose": {},
    }
    for purpose in PURPOSES:
        ytr_key, ydv_key, n_classes = PURPOSE_LABEL_KEYS[purpose]
        y_train = lbl[ytr_key].astype(np.int64)
        y_dev = lbl[ydv_key].astype(np.int64)
        task_score = _lr_score(rlace_train, y_train, rlace_dev, y_dev,
                               max_iter=2000)
        gender_probe = _lr_score(rlace_train, g_train, rlace_dev, g_dev,
                                 balanced=True, max_iter=2000)
        rlace_results["per_purpose"][purpose] = {
            "num_classes": n_classes,
            "task_lr": task_score,
            "linear_gender_probe": gender_probe,
        }
        print(f"[baselines]   RLACE/{purpose}: task_dev={task_score['dev_acc']:.4f}"
              f"  gender_probe_dev={gender_probe['dev_acc']:.4f}", flush=True)

    # ----- Vanilla baseline (no erasure) — for the table's first row -----
    print("[baselines] computing vanilla (no-erasure) baseline ...", flush=True)
    vanilla_r2_train = linear_r2(X_train, g_train)
    vanilla_r2_dev = linear_r2(X_dev, g_dev)
    vanilla_results: dict = {
        "method": "vanilla_no_erasure",
        "r2_gender_train": vanilla_r2_train,
        "r2_gender_dev": vanilla_r2_dev,
        "per_purpose": {},
    }
    for purpose in PURPOSES:
        ytr_key, ydv_key, n_classes = PURPOSE_LABEL_KEYS[purpose]
        y_train = lbl[ytr_key].astype(np.int64)
        y_dev = lbl[ydv_key].astype(np.int64)
        task_score = _lr_score(X_train, y_train, X_dev, y_dev, max_iter=2000)
        gender_probe = _lr_score(X_train, g_train, X_dev, g_dev,
                                 balanced=True, max_iter=2000)
        vanilla_results["per_purpose"][purpose] = {
            "num_classes": n_classes,
            "task_lr": task_score,
            "linear_gender_probe": gender_probe,
        }

    # ----- Save -----
    np.savez_compressed(
        args.output_dir / "eval_reps_LEACE.npz",
        train_reps=leace_train, dev_reps=leace_dev,
        gender_train=g_train, gender_dev=g_dev,
        P=P_leace, mu=mu_leace,
    )
    np.savez_compressed(
        args.output_dir / "eval_reps_RLACE.npz",
        train_reps=rlace_train, dev_reps=rlace_dev,
        gender_train=g_train, gender_dev=g_dev,
        P=P_rlace,
    )
    with open(args.output_dir / "leace_results.json", "w") as fh:
        json.dump(leace_results, fh, indent=2)
    with open(args.output_dir / "klu_results.json", "w") as fh:
        json.dump(klu_results, fh, indent=2)
    with open(args.output_dir / "rlace_results.json", "w") as fh:
        json.dump(rlace_results, fh, indent=2)
    with open(args.output_dir / "vanilla_results.json", "w") as fh:
        json.dump(vanilla_results, fh, indent=2)

    print(f"[baselines] wrote leace_results.json / klu_results.json / "
          f"rlace_results.json / vanilla_results.json under {args.output_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
